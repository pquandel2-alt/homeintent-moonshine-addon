"""Generic STT engine abstraction shared by every recognizer backend.

Mirrors app/tts_engine.py's design philosophy on the STT side: app/handler.py
depends only on this structural interface, never on a concrete engine, so
adding a new STT backend (Kroko today; Speechcatcher/Vosk later) is a matter
of writing one more SttEngine/SttSession implementation, not adding another
``if stt_engine == ...`` branch to the Wyoming handler.

Moonshine (app/moonshine_engine.py) and Kroko (app/kroko_engine.py) both
implement :class:`SttEngine`/:class:`SttSession`. Moonshine's own
MoonshineStreamingSession (app/streaming.py) already exposes exactly this
session shape unchanged -- it is used directly as the SttSession
implementation, not reimplemented, so Moonshine's existing, production
behavior (VAD, decode_incomplete_lines, per-chunk performance counters, ...)
is preserved byte-for-byte.

Capability flags exist so app/handler.py and app/__main__.py can make
generic decisions (e.g. "does this engine support keyterm/HA-vocabulary
biasing?") without knowing which concrete engine is loaded:

- ``supports_streaming``: audio is fed incrementally via add_audio() and
  transcribed as it arrives, not buffered then batch-transcribed. True for
  every engine in this add-on (Moonshine and Kroko both stream natively).
- ``supports_hotwords``: the engine has *some* mechanism to bias
  recognition towards a supplied word/phrase list (Moonshine's
  set_keyterms(), Kroko's sherpa-onnx hotwords).
- ``supports_dynamic_vocabulary``: the hotword mechanism can be updated at
  runtime (HA vocabulary refresh, see app/__main__.py's periodic refresh)
  without reloading the whole model. True for both Moonshine (set_keyterms()
  is a cheap native call) and Kroko (a new per-utterance hotwords string is
  applied to every newly created OnlineStream, no model reload needed).
- ``supports_partial_results``: the engine can emit an intermediate
  transcript before the utterance ends. Both engines are architecturally
  capable of this, but neither is currently wired to a Wyoming
  partial-transcript event (app/handler.py only ever sends one final
  ``Transcript`` per utterance) -- this is reported as False for both,
  honestly reflecting what this add-on's Wyoming surface actually does
  today, not what the underlying library could theoretically do.
"""

from dataclasses import dataclass
from typing import Protocol


@dataclass(frozen=True)
class SttCapabilities:
    """What one loaded STT engine can actually do, for generic dispatch."""

    supports_streaming: bool
    supports_hotwords: bool
    supports_dynamic_vocabulary: bool
    supports_partial_results: bool = False


class SttSession(Protocol):
    """One isolated transcription request's session state.

    Structurally identical to app/streaming.py's MoonshineStreamingSession,
    which already implements every member below without any changes -- see
    this module's docstring.
    """

    audio_duration_seconds: float

    async def start(self) -> None: ...

    async def add_audio(self, samples: list[float], sample_rate: int = 16000) -> None: ...

    async def finalize(self) -> str: ...

    def close(self) -> None: ...

    @property
    def inference_time_seconds(self) -> float: ...

    @property
    def finalize_time_seconds(self) -> float: ...

    @property
    def add_audio_chunk_count(self) -> int: ...

    @property
    def add_audio_compute_total_seconds(self) -> float: ...

    @property
    def add_audio_compute_max_seconds(self) -> float: ...

    @property
    def average_chunk_compute_seconds(self) -> float: ...


class SttEngine(Protocol):
    """Structural interface app/handler.py depends on for STT.

    Implemented identically by every engine (MoonshineSttEngine,
    KrokoSttEngine, ...) so the handler and Wyoming service discovery never
    need an ``if engine == ...`` branch: they read whichever engine was
    loaded at startup through this interface alone.
    """

    @property
    def engine_id(self) -> str:
        """Stable machine identifier, matches the add-on's own
        ``stt_engine`` option values (e.g. "moonshine"/"kroko")."""
        ...

    @property
    def capabilities(self) -> SttCapabilities: ...

    @property
    def program_name(self) -> str:
        """Wyoming ``AsrProgram.name`` advertised in ``describe``."""
        ...

    @property
    def model_display_name(self) -> str:
        """Short model identifier for Wyoming discovery/logging."""
        ...

    @property
    def description(self) -> str:
        """Human-readable description for Wyoming ``describe``."""
        ...

    @property
    def attribution_name(self) -> str: ...

    @property
    def attribution_url(self) -> str: ...

    def create_session(self, lock: object | None = None) -> SttSession:
        """Create one new, isolated transcription session.

        ``lock`` is an ``asyncio.Lock | None`` shared across all sessions of
        this engine (serializing native calls); typed as ``object`` here
        purely to keep this module free of an ``asyncio`` import it
        otherwise would not need.
        """
        ...

    def set_keyterms(
        self, terms: list[str], fallback_terms: list[str] | None = None
    ) -> tuple[list[str], bool]:
        """Apply a hotword/keyterm list.

        Returns ``(accepted_terms, applied_succeeded)``. ``applied_succeeded``
        distinguishes "the apply itself completed" (even if some candidate
        terms were individually rejected -- normal, expected) from "an
        unexpected, structural failure occurred and ``fallback_terms`` (the
        caller's last known-good list) was restored instead" -- see
        app/keyterms.py's ``SafeKeytermResult`` for the concept this
        mirrors. Never raises. An engine without
        ``capabilities.supports_hotwords`` logs a clear message and returns
        ``([], True)`` rather than pretending to apply anything.
        """
        ...
