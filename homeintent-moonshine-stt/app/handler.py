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
from wyoming.tts import Synthesize

from app.audio import float32_to_pcm_int16, pcm_int16_to_float32, validate_audio_format
from app.debug_audio import DEFAULT_DEBUG_AUDIO_DIR, save_debug_audio
from app.models import get_model_info
from app.streaming import MoonshineStreamingSession
from app.tts import get_tts_model_info
from app.tts_session import TtsSynthesisStats, TtsSynthesizer

_LOGGER = logging.getLogger(__name__)

_ATTRIBUTION_PROGRAM = Attribution(
    name="HomeIntent Contributors",
    url="https://github.com/pquandel2-alt/homeintent-moonshine-addon",
)
_ATTRIBUTION_MODEL = Attribution(
    name="Moonshine AI",
    url="https://github.com/moonshine-ai/moonshine",
)
_ATTRIBUTION_TTS_MODEL = Attribution(
    name="Kyutai",
    url="https://github.com/kyutai-labs/pocket-tts",
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

    async def handle_event(self, event: Event) -> bool:
        """Main event handler for the Wyoming STT and TTS lifecycles.

        Handles:
        - describe: Service discovery (ASR and/or TTS, whichever enabled)
        - transcribe: Start a new STT transcription
        - audio-start/chunk/stop: STT audio stream lifecycle
        - synthesize: A TTS request (single-shot text -> audio-chunk stream)
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

                case "synthesize":
                    await self._handle_synthesize(event)

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
        long-lived TTS session object to release here.
        """
        self._close_session()

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
            tts_model_info = get_tts_model_info(self._tts_model_name)
            tts_voice = TtsVoice(
                name=self._tts_synthesizer.default_voice,
                attribution=_ATTRIBUTION_TTS_MODEL,
                installed=True,
                description=tts_model_info.get("description", ""),
                version=None,
                languages=[self._language],
            )
            tts_programs.append(
                TtsProgram(
                    name="homeintent-pocket-tts",
                    attribution=_ATTRIBUTION_TTS_MODEL,
                    installed=True,
                    description="HomeIntent Pocket TTS - German streaming TTS",
                    version=None,
                    voices=[tts_voice],
                    # Only audio *output* is chunk-streamed (see
                    # _handle_synthesize) -- incremental *text* input via
                    # Wyoming's synthesize-start/-chunk/-stop protocol is
                    # not implemented, so this stays false.
                    supports_synthesize_streaming=False,
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

    async def _handle_synthesize(self, event: Event) -> None:
        """Handle a Wyoming TTS request: text in, streamed audio-chunks out.

        Streams audio to the client as soon as Pocket TTS yields each chunk
        (see app/tts_session.py) rather than buffering the whole utterance
        first -- this is what keeps time-to-first-audio low.
        """
        if self._tts_synthesizer is None:
            _LOGGER.warning("Received synthesize event but TTS is disabled (tts_enabled: false)")
            await self.write_event(
                WyomingError(text="TTS is disabled on this add-on", code="tts_disabled").event()
            )
            return

        synthesize = Synthesize.from_event(event)
        text = synthesize.text
        voice_name = synthesize.voice.name if synthesize.voice else None
        sample_rate = self._tts_synthesizer.sample_rate

        if not text or not text.strip():
            # Empty text: answer with a well-formed, empty audio response
            # rather than erroring -- Wyoming clients (and HA's Assist
            # pipeline) can legitimately send this, e.g. for a no-op reply.
            await self.write_event(AudioStart(rate=sample_rate, width=2, channels=1).event())
            await self.write_event(AudioStop().event())
            return

        requested_at = time.monotonic()
        stats = _SynthesisStats()
        tts_stats = TtsSynthesisStats()
        try:
            await self._stream_synthesis_chunks(text, voice_name, sample_rate, stats, tts_stats)
        except Exception as err:
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
            return

        if not stats.audio_started:
            # Non-empty text produced no audio at all -- shouldn't normally
            # happen, but still answer rather than leaving the client
            # waiting forever for audio-stop.
            await self.write_event(AudioStart(rate=sample_rate, width=2, channels=1).event())
        await self.write_event(AudioStop().event())

        if self._tts_log_performance:
            self._log_tts_performance(text, sample_rate, requested_at, stats, tts_stats)

    def _log_tts_performance(
        self,
        text: str,
        sample_rate: int,
        requested_at: float,
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
        """
        wall_time = time.monotonic() - requested_at
        audio_duration = stats.total_samples / sample_rate if sample_rate else 0.0
        ttfa_generated = (
            stats.first_chunk_generated_at - requested_at if stats.first_chunk_generated_at else 0.0
        )
        ttfa_sent = stats.first_chunk_sent_at - requested_at if stats.first_chunk_sent_at else 0.0
        model_rtf = (
            tts_stats.model_generation_seconds / audio_duration if audio_duration > 0 else 0.0
        )
        wall_rtf = wall_time / audio_duration if audio_duration > 0 else 0.0
        _LOGGER.info(
            "TTS completed: model=%s chars=%d ttfa_generated=%.3fs ttfa_sent=%.3fs "
            "lock_wait=%.3fs model_compute=%.2fs wyoming_send=%.3fs wall=%.2fs audio=%.2fs "
            "model_rtf=%.2f wall_rtf=%.2f",
            self._tts_model_name,
            len(text),
            ttfa_generated,
            ttfa_sent,
            tts_stats.lock_wait_seconds,
            tts_stats.model_generation_seconds,
            stats.wyoming_send_seconds,
            wall_time,
            audio_duration,
            model_rtf,
            wall_rtf,
        )
