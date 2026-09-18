"""Wyoming ASR/TTS event handler for Moonshine STT + Pocket TTS."""

import asyncio
import logging
import time
from dataclasses import dataclass
from pathlib import Path

from moonshine_voice import Transcriber
from wyoming.asr import Transcript
from wyoming.audio import AudioChunk, AudioStart, AudioStop
from wyoming.error import Error as WyomingError
from wyoming.event import Event
from wyoming.info import (
    AsrModel,
    AsrProgram,
    Attribution,
    Info,
    TtsProgram,
    TtsVoice,
)
from wyoming.server import AsyncEventHandler
from wyoming.tts import (
    Synthesize,
    SynthesizeChunk,
    SynthesizeStart,
    SynthesizeStop,
    SynthesizeStopped,
)

from app.audio import float32_to_pcm_int16, pcm_int16_to_float32, validate_audio_format
from app.debug_audio import DEFAULT_DEBUG_AUDIO_DIR, save_debug_audio
from app.models import get_model_info
from app.streaming import MoonshineStreamingSession
from app.tts_engine import TtsSynthesisStats, TtsSynthesizer
from app.tts_stream import TtsStreamPhase, TtsStreamState

_LOGGER = logging.getLogger(__name__)

_ATTRIBUTION_PROGRAM = Attribution(
    name="HomeIntent Contributors",
    url="https://github.com/pquandel2-alt/homeintent-moonshine-addon",
)
_ATTRIBUTION_MODEL = Attribution(
    name="Moonshine AI",
    url="https://github.com/moonshine-ai/moonshine",
)


@dataclass
class _SynthesisStats:
    """Accumulator for one _handle_synthesize() call's performance data.

    ``wyoming_send_seconds`` is the cumulative time spent inside
    write_event() for audio-start/audio-chunk events -- real socket I/O
    (and any client backpressure), never Pocket TTS's own compute. Kept
    strictly separate from ``tts_stats.model_generation_seconds`` (see
    app/tts_session.py's TtsSynthesisStats) so a slow/backpressured client
    can never be misread as slow model inference.
    """

    audio_started: bool = False
    total_samples: int = 0
    first_chunk_generated_at: float | None = None
    first_chunk_sent_at: float | None = None
    wyoming_send_seconds: float = 0.0


class MoonshineAsrHandler(AsyncEventHandler):
    """Handles Wyoming STT (Moonshine) and TTS (Pocket TTS) events.

    Per Wyoming connection, one handler instance sharing the single
    pre-loaded Transcriber and/or PocketTtsSynthesizer created once at
    add-on startup. Either can be disabled (``transcriber``/
    ``tts_synthesizer`` left ``None``) per the ``stt_enabled``/
    ``tts_enabled`` add-on options -- service discovery then only
    advertises whichever is actually active.

    Key STT design (unchanged from the STT-only releases): add_audio() is
    called PER Wyoming audio-chunk (streaming), not after all audio is
    buffered, via one isolated MoonshineStreamingSession per transcription
    request (Transcriber.create_stream()).

    Key TTS design: audio-chunks are forwarded to the client as soon as
    Pocket TTS's own generate_audio_stream() yields them (see
    app/tts_session.py) -- not after synthesizing the whole utterance.

    Wyoming streaming-TTS protocol (synthesize-start/-chunk/-stop): Home
    Assistant's real Wyoming TTS client only uses this if the server
    advertises ``supports_synthesize_streaming=True`` in its `describe`
    response; otherwise it falls back to the old single-shot ``synthesize``
    event and BUFFERS THE ENTIRE RESPONSE ITSELF before returning any audio
    at all -- even though this add-on already streams internally, Home
    Assistant would never see that until now (verified directly against
    ``homeassistant/components/wyoming/tts.py`` upstream: its
    ``async_get_tts_audio()``, used for the non-streaming case, loops until
    ``audio-stop`` and only returns the fully-collected WAV afterward). This
    was the actual, verified root cause of the perceived TTS latency, not
    Pocket TTS itself.

    With streaming enabled, real Home Assistant sends
    ``synthesize-start``, one or more ``synthesize-chunk``, then the
    ENTIRE message again via a backwards-compatible ``synthesize`` event,
    then ``synthesize-stop`` -- see app/tts_stream.py's module docstring
    for the full protocol analysis and why that backwards-compatible event
    must never cause the text to be synthesized (and spoken) twice.
    ``self._tts_stream_state`` (a :class:`TtsStreamState`) is what
    prevents that. Home Assistant's streaming reader loop
    (``_read_tts_audio()``) only terminates on a ``synthesize-stopped``
    event (verified from the same upstream source -- a plain
    ``audio-stop`` is not enough to end that particular loop), so every
    streaming request must end with one.
    """

    def __init__(
        self,
        reader: asyncio.StreamReader,
        writer: asyncio.StreamWriter,
        transcriber: Transcriber | None,
        model_name: str = "",
        language: str = "de",
        log_transcripts: bool = False,
        log_performance: bool = True,
        save_debug_audio_enabled: bool = False,
        debug_audio_max_files: int = 100,
        debug_audio_dir: Path = DEFAULT_DEBUG_AUDIO_DIR,
        moonshine_lock: asyncio.Lock | None = None,
        tts_synthesizer: TtsSynthesizer | None = None,
        tts_model_name: str = "",
        tts_log_performance: bool = True,
    ):
        """Initialize handler.

        Args:
            reader: Wyoming protocol reader
            writer: Wyoming protocol writer
            transcriber: The single Transcriber loaded once at add-on
                startup, or None if ``stt_enabled`` is false.
            model_name: STT model in use ("tiny" or "small").
            language: Language code ("de" for German)
            log_transcripts: If True, log the recognized text at INFO. If
                False (default), only log its length -- recognized speech is
                privacy-sensitive and must not end up in logs unasked.
            log_performance: If True, log a compact per-utterance timing
                line (audio duration, finalize time, RTF). Never includes
                transcript text.
            save_debug_audio_enabled: If True, persist received audio (and a
                metadata sidecar) to disk for debugging. Off by default --
                see the README privacy note.
            debug_audio_max_files: Retention limit enforced when debug audio
                is enabled; oldest recordings are deleted beyond this count.
            debug_audio_dir: Directory debug recordings are written to.
            moonshine_lock: Lock shared across all handlers of the same
                Transcriber, serializing native calls (see
                app/streaming.py's MoonshineStreamingSession docstring for
                why). Pass the same lock instance to every handler.
            tts_synthesizer: The single PocketTtsSynthesizer created once at
                add-on startup, or None if ``tts_enabled`` is false.
            tts_model_name: TTS model in use ("german" or "german_24l"),
                for logging/discovery only.
            tts_log_performance: If True, log a compact per-request TTFA/
                RTF line (see _handle_synthesize). Never includes the
                synthesized text.
        """
        super().__init__(reader, writer)
        self._transcriber = transcriber
        self._model_name = model_name
        self._language = language
        self._log_transcripts = log_transcripts
        self._log_performance = log_performance
        self._save_debug_audio_enabled = save_debug_audio_enabled
        self._debug_audio_max_files = debug_audio_max_files
        self._debug_audio_dir = debug_audio_dir
        self._moonshine_lock = moonshine_lock
        self._tts_synthesizer = tts_synthesizer
        self._tts_model_name = tts_model_name
        self._tts_log_performance = tts_log_performance

        self._session: MoonshineStreamingSession | None = None
        self._audio_rejected = False
        self._raw_audio_buffer: bytearray | None = None

        self._tts_stream_state = TtsStreamState()
        self._tts_stream_requested_at: float | None = None

    async def handle_event(self, event: Event) -> bool:
        """Main event handler for the Wyoming STT and TTS lifecycles.

        Handles:
        - describe: Service discovery (ASR and/or TTS, whichever enabled)
        - transcribe: Start a new STT transcription
        - audio-start/chunk/stop: STT audio stream lifecycle
        - synthesize(-start/-chunk/-stop): TTS requests, legacy single-shot
          or streaming (see _dispatch_tts_event())
        """
        try:
            match event.type:
                case "describe":
                    await self._handle_describe()

                case "transcribe":
                    self._reset_session()

                case "audio-start":
                    await self._handle_audio_start(event)

                case "audio-chunk":
                    await self._handle_audio_chunk(event)

                case "audio-stop":
                    await self._handle_audio_stop()

                case "synthesize" | "synthesize-start" | "synthesize-chunk" | "synthesize-stop":
                    await self._dispatch_tts_event(event)

                case _:
                    _LOGGER.debug(f"Ignoring unsupported event type: {event.type}")

            return True

        except Exception as e:
            _LOGGER.error(f"Error handling event {event.type}: {e}", exc_info=True)
            self._close_session()
            return False

    async def disconnect(self) -> None:
        """Called by AsyncEventHandler.run() when the client disconnects.

        Guarantees the native Moonshine stream is released even if the
        client vanishes mid-utterance (no audio-stop ever arrives). Any
        in-flight TTS synthesis (see _handle_synthesize) is stopped by its
        own async-generator cleanup when the client disconnect surfaces as
        a write failure or a cancelled task -- there is no separate
        long-lived TTS session object to release here. Also cancels any
        in-progress streaming-TTS state so a disconnect mid-collection
        (e.g. between synthesize-start and synthesize-stop) can never be
        mistaken for a still-active request.
        """
        self._close_session()
        self._tts_stream_state.cancel()

    async def _handle_describe(self) -> None:
        """Respond to service discovery request with whichever of ASR/TTS
        is actually enabled (transcriber/tts_synthesizer not None)."""
        asr_programs = []
        if self._transcriber is not None:
            model_info = get_model_info(self._model_name)
            asr_model = AsrModel(
                name=model_info.get("name", "unknown"),
                attribution=_ATTRIBUTION_MODEL,
                installed=True,
                description=model_info.get("description", ""),
                version=model_info.get("name", "0.1.0"),
                languages=[self._language],
            )
            asr_programs.append(
                AsrProgram(
                    name="homeintent-moonshine",
                    attribution=_ATTRIBUTION_PROGRAM,
                    installed=True,
                    description="HomeIntent Moonshine STT - German streaming ASR",
                    version=None,
                    models=[asr_model],
                )
            )

        tts_programs = []
        if self._tts_synthesizer is not None:
            synth = self._tts_synthesizer
            tts_voice = TtsVoice(
                name=synth.default_voice,
                attribution=synth.attribution,
                installed=True,
                description=synth.description,
                version=None,
                languages=[self._language],
            )
            tts_programs.append(
                TtsProgram(
                    name=synth.program_name,
                    attribution=synth.attribution,
                    installed=True,
                    description=synth.description,
                    version=None,
                    voices=[tts_voice],
                    # True: synthesize-start/-chunk/-stop are fully
                    # implemented (see TtsStreamState/_handle_synthesize_*
                    # below) and end every streaming request with
                    # synthesize-stopped, which real Home Assistant's
                    # streaming Wyoming TTS client requires to end its own
                    # read loop (verified against
                    # homeassistant/components/wyoming/tts.py upstream).
                    # This is what makes Home Assistant use the streaming
                    # client path at all instead of buffering the whole
                    # response before returning any audio.
                    supports_synthesize_streaming=True,
                )
            )

        info = Info(asr=asr_programs, tts=tts_programs)
        await self.write_event(info.event())

    def _close_session(self) -> None:
        """Release the active session's native resources, if any. Idempotent."""
        if self._session is not None:
            self._session.close()
        self._session = None
        self._raw_audio_buffer = None

    def _reset_session(self) -> None:
        """Reset for a new transcription request."""
        self._close_session()
        self._audio_rejected = False
        _LOGGER.debug("Reset for new transcription")

    async def _handle_audio_start(self, event: Event) -> None:
        """Handle audio stream start."""
        if self._transcriber is None:
            _LOGGER.warning("Received audio-start but STT is disabled (stt_enabled: false)")
            return

        audio_start = AudioStart.from_event(event)

        error = validate_audio_format(
            audio_start.rate or 16000,
            audio_start.width or 2,
            audio_start.channels or 1,
            context="audio-start",
        )
        if error is not None:
            self._audio_rejected = True
            await self.write_event(
                WyomingError(text=error, code="unsupported_audio_format").event()
            )
            _LOGGER.error("Rejecting audio stream: %s", error)
            return

        if self._session is not None:
            _LOGGER.warning(
                "Received audio-start while a session was already active; "
                "closing the previous session first"
            )
            self._close_session()

        self._session = MoonshineStreamingSession(self._transcriber, lock=self._moonshine_lock)
        self._raw_audio_buffer = bytearray() if self._save_debug_audio_enabled else None
        await self._session.start()

        _LOGGER.debug(
            f"Audio stream started: {audio_start.rate}Hz, "
            f"{audio_start.width}B width, {audio_start.channels} channel(s)"
        )

    async def _handle_audio_chunk(self, event: Event) -> None:
        """Handle audio chunk. CRITICAL: feed to Moonshine immediately."""
        if self._audio_rejected:
            return

        if self._session is None:
            _LOGGER.warning("audio-chunk received before audio-start")
            return

        chunk = AudioChunk.from_event(event)

        error = validate_audio_format(
            chunk.rate or 16000,
            chunk.width or 2,
            chunk.channels or 1,
            context="audio-chunk",
        )
        if error is not None:
            self._audio_rejected = True
            self._close_session()
            await self.write_event(
                WyomingError(text=error, code="unsupported_audio_format").event()
            )
            _LOGGER.error("Rejecting audio stream mid-stream: %s", error)
            return

        if self._raw_audio_buffer is not None:
            self._raw_audio_buffer.extend(chunk.audio)

        float32_audio = pcm_int16_to_float32(chunk.audio)

        # STREAMING: Feed immediately to Moonshine (not buffered)
        try:
            await self._session.add_audio(list(float32_audio))
        except Exception:
            self._close_session()
            raise

    async def _handle_audio_stop(self) -> None:
        """Handle audio stream end and send transcript."""
        if self._session is None:
            # Empty, invalid, or rejected stream
            await self.write_event(Transcript(text="", language=self._language).event())
            _LOGGER.debug("Empty transcription")
            return

        session = self._session
        raw_audio = self._raw_audio_buffer
        try:
            text = await session.finalize()
        finally:
            self._close_session()

        await self.write_event(Transcript(text=text, language=self._language).event())

        if self._log_transcripts:
            _LOGGER.info("Transcript sent: '%s'", text)
        else:
            _LOGGER.info("Transcript sent (%d chars)", len(text))

        if self._log_performance:
            audio_duration = session.audio_duration_seconds
            # Real STT compute time: cumulative Moonshine processing during
            # add_audio() (streaming ASR already transcribes incrementally
            # as audio arrives) plus the final stop() pass -- NOT the wall
            # time the Wyoming client spent streaming audio in, which is
            # bounded by how long the user talked, not by model speed.
            inference_time = session.inference_time_seconds
            finalize_time = session.finalize_time_seconds
            rtf = inference_time / audio_duration if audio_duration > 0 else 0.0
            # v0.2.4: chunks/add_audio_total/add_audio_max/avg_chunk answer
            # WHERE the time went -- if add_audio_total alone (excluding
            # finalize) already exceeds audio_duration, the model is
            # processing slower than real-time on this CPU (a genuine
            # throughput problem, not a one-off stall); a max much larger
            # than avg_chunk instead points at an isolated stall/backlog.
            _LOGGER.info(
                "STT completed: model=%s audio=%.2fs inference=%.2fs finalize=%.2fs rtf=%.2f "
                "chunks=%d add_audio_total=%.2fs add_audio_max=%.3fs avg_chunk=%.3fs",
                self._model_name,
                audio_duration,
                inference_time,
                finalize_time,
                rtf,
                session.add_audio_chunk_count,
                session.add_audio_compute_total_seconds,
                session.add_audio_compute_max_seconds,
                session.average_chunk_compute_seconds,
            )

        if self._save_debug_audio_enabled and raw_audio:
            try:
                save_debug_audio(
                    bytes(raw_audio),
                    model=self._model_name,
                    language=self._language,
                    transcript=text,
                    directory=self._debug_audio_dir,
                    max_files=self._debug_audio_max_files,
                )
            except OSError as err:
                _LOGGER.error("Failed to save debug audio: %s", err)

    async def _stream_synthesis_chunks(
        self,
        text: str,
        voice_name: str | None,
        sample_rate: int,
        stats: "_SynthesisStats",
        tts_stats: TtsSynthesisStats,
    ) -> None:
        """Consume the TTS stream, forwarding audio-start/chunk as it goes.

        Split out of _handle_synthesize() to keep that method's own
        complexity manageable; raises on synthesis failure (caller decides
        how to answer the client), always sends audio-start at most once,
        and always releases the underlying async generator via aclose(),
        including on early client disconnect (B26).

        ``stats`` is mutated in place rather than returned: if synthesis
        fails partway through, the exception propagates out of this
        coroutine and a `return stats` would never run -- the caller needs
        to know whether audio-start was already sent (``stats.audio_started``)
        even in that case, so it never sends a second one (see
        _handle_synthesize's except branch).
        """
        assert self._tts_synthesizer is not None  # only called when TTS is enabled

        agen = self._tts_synthesizer.synthesize_stream(text, voice_name, tts_stats)
        try:
            async for chunk in agen:
                if stats.first_chunk_generated_at is None:
                    stats.first_chunk_generated_at = time.monotonic()
                if not stats.audio_started:
                    send_started = time.monotonic()
                    await self.write_event(
                        AudioStart(rate=sample_rate, width=2, channels=1).event()
                    )
                    stats.wyoming_send_seconds += time.monotonic() - send_started
                    stats.audio_started = True

                pcm = float32_to_pcm_int16(chunk)
                if not pcm:
                    continue
                send_started = time.monotonic()
                await self.write_event(
                    AudioChunk(rate=sample_rate, width=2, channels=1, audio=pcm).event()
                )
                stats.wyoming_send_seconds += time.monotonic() - send_started
                if stats.first_chunk_sent_at is None:
                    stats.first_chunk_sent_at = time.monotonic()
                stats.total_samples += len(chunk)
        finally:
            # Runs on normal completion, a synthesis error, AND on the
            # caller cancelling this coroutine (client disconnect
            # mid-stream) -- always signals PocketTtsSynthesizer's producer
            # thread to stop and releases its lock (see app/tts_session.py).
            await agen.aclose()

    async def _dispatch_tts_event(self, event: Event) -> None:
        """Route one of the four Wyoming TTS event types to its handler.

        Split out of handle_event() purely to keep that method's own
        branching simple -- the actual legacy-vs-streaming decision lives
        in _handle_synthesize()/TtsStreamState, not here.
        """
        match event.type:
            case "synthesize":
                await self._handle_synthesize(event)
            case "synthesize-start":
                await self._handle_synthesize_start(event)
            case "synthesize-chunk":
                await self._handle_synthesize_chunk(event)
            case "synthesize-stop":
                await self._handle_synthesize_stop(event)

    async def _handle_synthesize(self, event: Event) -> None:
        """Handle a Wyoming ``synthesize`` event.

        This event has two distinct meanings depending on connection
        state (see TtsStreamState/app/tts_stream.py):

        - If a streaming request is being collected (synthesize-start
          already seen), this is Home Assistant's backwards-compatible
          "entire message again" event that always follows the last
          synthesize-chunk. The text is already fully known at this
          point -- ``begin_synthesis()`` triggers the actual synthesis
          exactly once and returns None on any later, redundant call
          (there is at most one legitimate call per streaming request).
        - Otherwise, this is a legacy, single-shot request with no
          preceding synthesize-start -- handled exactly as in prior
          releases.
        """
        synthesize = Synthesize.from_event(event)

        if self._tts_stream_state.phase is TtsStreamPhase.COLLECTING:
            text = self._tts_stream_state.begin_synthesis(full_text=synthesize.text)
            if text is None:
                return  # already triggered; never synthesize/speak twice
            voice_name = (
                synthesize.voice.name if synthesize.voice else self._tts_stream_state.voice_name
            )
            requested_at = self._tts_stream_requested_at or time.monotonic()
            await self._run_tts_synthesis(text, voice_name, "streaming", requested_at)
            return

        # Legacy, single-shot request: no synthesize-start preceded this.
        voice_name = synthesize.voice.name if synthesize.voice else None
        await self._run_tts_synthesis(synthesize.text, voice_name, "legacy", time.monotonic())

    async def _handle_synthesize_start(self, event: Event) -> None:
        """Handle synthesize-start: begin collecting a streaming request.

        If TTS is disabled, answers immediately with an error and leaves
        the stream state at IDLE (so the inevitable synthesize-chunk/
        synthesize/synthesize-stop that follow are simply ignored/routed
        to the ordinary "TTS disabled" error path instead of being
        collected pointlessly).
        """
        if self._tts_synthesizer is None:
            _LOGGER.warning("Received synthesize-start but TTS is disabled (tts_enabled: false)")
            await self.write_event(
                WyomingError(text="TTS is disabled on this add-on", code="tts_disabled").event()
            )
            return

        start = SynthesizeStart.from_event(event)
        voice_name = start.voice.name if start.voice else None
        self._tts_stream_state.start(voice_name)
        self._tts_stream_requested_at = time.monotonic()

    async def _handle_synthesize_chunk(self, event: Event) -> None:
        """Handle synthesize-chunk: accumulate one piece of streamed text.

        Only ever used as a fallback text source for a purely
        spec-following client that never sends the backwards-compatible
        whole-message ``synthesize`` event (see _handle_synthesize) --
        Pocket TTS itself has no incremental-text API to feed these to
        as they arrive (verified against
        ``pocket_tts.models.tts_model.TTSModel.generate_audio_stream()``;
        see app/tts_stream.py's module docstring).
        """
        chunk = SynthesizeChunk.from_event(event)
        if not self._tts_stream_state.add_chunk(chunk.text):
            _LOGGER.warning(
                "Received synthesize-chunk without an active synthesize-start; ignoring"
            )

    async def _handle_synthesize_stop(self, event: Event) -> None:
        """Handle synthesize-stop: end of the streaming text input.

        Only actually triggers synthesis if it hasn't already happened --
        the normal Home Assistant case is that the backwards-compatible
        ``synthesize`` event (handled in _handle_synthesize) already
        triggered it by the time this arrives, in which case
        ``begin_synthesis()`` returns None and this is a no-op. A purely
        spec-following streaming client that never sends that
        backwards-compatible event relies on synthesize-stop as the only
        signal that the text is complete.
        """
        SynthesizeStop.from_event(event)  # no fields; parsed for validation/consistency

        if self._tts_stream_state.phase is TtsStreamPhase.IDLE:
            _LOGGER.warning("Received synthesize-stop without an active synthesize-start; ignoring")
            return

        text = self._tts_stream_state.begin_synthesis()
        if text is None:
            return  # already triggered by the backwards-compatible `synthesize` event

        requested_at = self._tts_stream_requested_at or time.monotonic()
        await self._run_tts_synthesis(
            text, self._tts_stream_state.voice_name, "streaming", requested_at
        )

    async def _run_tts_synthesis(
        self,
        text: str,
        voice_name: str | None,
        protocol_mode: str,
        requested_at: float,
    ) -> None:
        """Shared implementation for both the legacy single-shot
        ``synthesize`` path and the streaming synthesize-start/-chunk/-stop
        path: once the complete text to speak is known, run Pocket TTS and
        forward audio-start/-chunk/-stop -- plus, for the streaming path,
        a final synthesize-stopped, which real Home Assistant's streaming
        Wyoming TTS client requires to end its own read loop (a plain
        audio-stop is not enough there -- see this module's docstring).

        ``protocol_mode`` is ``"legacy"`` or ``"streaming"`` -- purely for
        ending the response correctly and for the performance log; the
        actual synthesis path is identical either way.
        """
        is_streaming = protocol_mode == "streaming"

        if self._tts_synthesizer is None:
            _LOGGER.warning("Received synthesize event but TTS is disabled (tts_enabled: false)")
            await self.write_event(
                WyomingError(text="TTS is disabled on this add-on", code="tts_disabled").event()
            )
            if is_streaming:
                self._tts_stream_state.cancel()
            return

        sample_rate = self._tts_synthesizer.sample_rate

        if not text or not text.strip():
            # Empty text: answer with a well-formed, empty audio response
            # rather than erroring -- Wyoming clients (and HA's Assist
            # pipeline) can legitimately send this, e.g. for a no-op reply.
            await self.write_event(AudioStart(rate=sample_rate, width=2, channels=1).event())
            await self.write_event(AudioStop().event())
            if is_streaming:
                await self.write_event(SynthesizeStopped().event())
                self._tts_stream_state.finish()
            return

        stats = _SynthesisStats()
        tts_stats = TtsSynthesisStats()
        synthesis_started_at = time.monotonic()
        try:
            await self._stream_synthesis_chunks(text, voice_name, sample_rate, stats, tts_stats)
        except Exception as err:
            await self._handle_synthesis_error(err, sample_rate, stats, is_streaming)
            return

        if not stats.audio_started:
            # Non-empty text produced no audio at all -- shouldn't normally
            # happen, but still answer rather than leaving the client
            # waiting forever for audio-stop.
            await self.write_event(AudioStart(rate=sample_rate, width=2, channels=1).event())
        await self.write_event(AudioStop().event())
        if is_streaming:
            await self.write_event(SynthesizeStopped().event())
            self._tts_stream_state.finish()

        if self._tts_log_performance:
            self._log_tts_performance(
                text,
                sample_rate,
                requested_at,
                synthesis_started_at,
                protocol_mode,
                stats,
                tts_stats,
            )

    async def _handle_synthesis_error(
        self,
        err: Exception,
        sample_rate: int,
        stats: "_SynthesisStats",
        is_streaming: bool,
    ) -> None:
        """Answer a mid-synthesis failure: error, but always a well-formed
        end to the response (audio-stop, plus synthesize-stopped for a
        streaming request) so the client is never left waiting forever.

        Split out of _run_tts_synthesis() purely to keep that method's own
        branching simple.
        """
        _LOGGER.error("TTS synthesis failed: %s", err, exc_info=True)
        # Only one audio-start per synthesize request: _stream_synthesis
        # _chunks() may have already sent it (and possibly some chunks)
        # before failing mid-stream -- a second one here would be a
        # protocol violation, not just a cosmetic duplicate.
        if not stats.audio_started:
            await self.write_event(AudioStart(rate=sample_rate, width=2, channels=1).event())
        await self.write_event(AudioStop().event())
        await self.write_event(
            WyomingError(text="TTS synthesis failed", code="tts_synthesis_error").event()
        )
        if is_streaming:
            await self.write_event(SynthesizeStopped().event())
            self._tts_stream_state.cancel()

    def _log_tts_performance(
        self,
        text: str,
        sample_rate: int,
        requested_at: float,
        synthesis_started_at: float,
        protocol_mode: str,
        stats: "_SynthesisStats",
        tts_stats: TtsSynthesisStats,
    ) -> None:
        """Log a compact timing breakdown, never the synthesized text.

        Distinguishes real Pocket TTS model compute
        (``tts_stats.model_generation_seconds``) from time this request
        spent waiting for the shared TTS lock behind another request
        (``tts_stats.lock_wait_seconds``) and from real Wyoming socket I/O/
        client backpressure (``stats.wyoming_send_seconds``) -- none of
        those must be misread as one another. ``model_rtf`` reflects only
        the model's own compute; ``wall_rtf`` is the full, real
        client-observed request cost including all of the above.

        ``requested_at`` is anchored at synthesize-start for a streaming
        request (i.e. it includes the round trip of Home Assistant
        actually sending its synthesize-chunk/synthesize/synthesize-stop
        events), or at the single ``synthesize`` event for a legacy
        request. ``first_audio_to_ha`` is anchored at
        ``synthesis_started_at`` instead -- the moment the complete text
        was known and real synthesis actually began -- isolating pure
        model+send latency from that protocol/collection overhead. For a
        legacy request the two anchors are identical, so
        ``first_audio_to_ha`` intentionally equals ``ttfa_sent`` there.
        """
        wall_time = time.monotonic() - requested_at
        audio_duration = stats.total_samples / sample_rate if sample_rate else 0.0
        ttfa_generated = (
            stats.first_chunk_generated_at - requested_at if stats.first_chunk_generated_at else 0.0
        )
        ttfa_sent = stats.first_chunk_sent_at - requested_at if stats.first_chunk_sent_at else 0.0
        first_audio_to_ha = (
            stats.first_chunk_sent_at - synthesis_started_at if stats.first_chunk_sent_at else 0.0
        )
        model_rtf = (
            tts_stats.model_generation_seconds / audio_duration if audio_duration > 0 else 0.0
        )
        wall_rtf = wall_time / audio_duration if audio_duration > 0 else 0.0
        engine_id = (
            self._tts_synthesizer.engine_id if self._tts_synthesizer is not None else "unknown"
        )
        _LOGGER.info(
            "TTS completed: engine=%s model=%s chars=%d protocol_mode=%s ttfa_generated=%.3fs "
            "ttfa_sent=%.3fs first_audio_to_ha=%.3fs lock_wait=%.3fs model_compute=%.2fs "
            "wyoming_send=%.3fs wall=%.2fs audio=%.2fs model_rtf=%.2f wall_rtf=%.2f",
            engine_id,
            self._tts_model_name,
            len(text),
            protocol_mode,
            ttfa_generated,
            ttfa_sent,
            first_audio_to_ha,
            tts_stats.lock_wait_seconds,
            tts_stats.model_generation_seconds,
            stats.wyoming_send_seconds,
            wall_time,
            audio_duration,
            model_rtf,
            wall_rtf,
        )
