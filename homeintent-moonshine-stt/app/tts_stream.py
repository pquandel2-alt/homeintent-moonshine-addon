"""Streaming-TTS state machine for Wyoming's synthesize-start/-chunk/-stop
protocol (see app/handler.py's module docstring for the full picture).

Home Assistant's real Wyoming TTS client (verified directly against
``homeassistant/components/wyoming/tts.py`` upstream, not assumed) sends,
for a streaming-capable server::

    synthesize-start
    synthesize-chunk (one or more)
    synthesize            <- the ENTIRE accumulated message again, for
                              backwards compatibility with servers that
                              only understand the old single-shot event
    synthesize-stop

For a normal HomeIntent/Assist intent response (the common case: the full
reply text is already known before TTS starts, not token-streamed from an
LLM), HA's own code wraps the message in a single-chunk generator, so in
practice the sequence is exactly one ``synthesize-chunk`` carrying the
whole text, then the backwards-compatible ``synthesize`` with that same
text again.

The critical correctness requirement this class exists for: that
backwards-compatible ``synthesize`` event must NEVER cause the
already-known text to be synthesized (and therefore spoken) a second
time. This class tracks exactly one thing per Wyoming connection -- how
far a single streaming request has progressed -- so app/handler.py can
tell "the text is now fully known, start synthesizing" apart from "we
already started; ignore this".

Pocket TTS itself has no incremental-input API (verified against
``pocket_tts.models.tts_model.TTSModel.generate_audio_stream()`` --
it takes one complete ``text_to_generate`` string per call and only
streams *output* incrementally). This class therefore does not attempt
to synthesize partial text as chunks arrive; it only accumulates them as
a fallback for a purely spec-following client that never sends the
backwards-compatible ``synthesize`` event, and hands the complete text to
the caller in one piece once it is known.
"""

from dataclasses import dataclass, field
from enum import Enum


class TtsStreamPhase(Enum):
    """Lifecycle of one streaming Wyoming TTS request on one connection."""

    IDLE = "idle"
    COLLECTING = "collecting"
    SYNTHESIZING = "synthesizing"
    FINISHED = "finished"
    CANCELLED = "cancelled"


@dataclass
class TtsStreamState:
    """Tracks one streaming TTS request across synthesize-start/-chunk/-stop.

    Pure bookkeeping, no I/O -- fully unit-testable without a real Wyoming
    connection or Pocket TTS model. One instance lives for the lifetime of
    a Wyoming connection (see app/handler.py) and is reused across
    multiple requests on that connection: :meth:`start` unconditionally
    resets it for a new request.
    """

    phase: TtsStreamPhase = TtsStreamPhase.IDLE
    voice_name: str | None = None
    _text_parts: list[str] = field(default_factory=list)

    def start(self, voice_name: str | None) -> None:
        """Handle synthesize-start: begin a fresh streaming request.

        Always resets to COLLECTING, even if a previous request was left
        unfinished (a client sending a second synthesize-start without
        properly ending the first is a protocol violation on its part,
        not ours to reject -- we just start clean rather than getting
        stuck).
        """
        self.phase = TtsStreamPhase.COLLECTING
        self.voice_name = voice_name
        self._text_parts = []

    def add_chunk(self, text: str) -> bool:
        """Handle synthesize-chunk: accumulate one text chunk.

        Returns True if accepted (state was COLLECTING), False if this
        chunk arrived outside of an active streaming request (the caller
        should log and ignore it -- see app/handler.py).
        """
        if self.phase is not TtsStreamPhase.COLLECTING:
            return False
        self._text_parts.append(text)
        return True

    @property
    def collected_text(self) -> str:
        """The text accumulated so far from synthesize-chunk events."""
        return "".join(self._text_parts)

    def begin_synthesis(self, full_text: str | None = None) -> str | None:
        """Transition COLLECTING -> SYNTHESIZING and return the text to
        synthesize, or None if synthesis was already triggered (or never
        started) for this request.

        Called from two possible triggers, whichever arrives first:
        - The backwards-compatible ``synthesize`` event, which carries the
          complete message explicitly (``full_text``) -- the common case
          with real Home Assistant.
        - ``synthesize-stop``, if that backwards-compatible event never
          arrives (a purely spec-following streaming client) -- in that
          case ``full_text`` is omitted and the text accumulated via
          :meth:`add_chunk` is used instead.

        Idempotent by design: once the phase leaves COLLECTING, a second
        call (e.g. synthesize-stop arriving after the backwards-compatible
        synthesize already triggered synthesis) returns None instead of
        re-triggering -- this is the mechanism that prevents the
        backwards-compatible duplicate text from being spoken twice.
        """
        if self.phase is not TtsStreamPhase.COLLECTING:
            return None
        text = full_text if full_text is not None else self.collected_text
        self.phase = TtsStreamPhase.SYNTHESIZING
        return text

    def finish(self) -> None:
        """Handle successful completion of the synthesis for this request."""
        self.phase = TtsStreamPhase.FINISHED

    def cancel(self) -> None:
        """Handle an error or disconnect during collection or synthesis."""
        self.phase = TtsStreamPhase.CANCELLED

    def reset(self) -> None:
        """Return to IDLE (e.g. on disconnect, or between requests)."""
        self.phase = TtsStreamPhase.IDLE
        self.voice_name = None
        self._text_parts = []

    @property
    def is_streaming_request(self) -> bool:
        """True once synthesize-start has been seen for the current/most
        recent request (COLLECTING or SYNTHESIZING) -- used by the handler
        to distinguish a streaming request in progress from a legacy,
        single-shot ``synthesize`` event with no preceding
        synthesize-start."""
        return self.phase in (TtsStreamPhase.COLLECTING, TtsStreamPhase.SYNTHESIZING)
