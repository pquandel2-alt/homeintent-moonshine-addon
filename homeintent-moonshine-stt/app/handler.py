"""Wyoming ASR event handler for Moonshine streaming."""

import asyncio
import logging
import time
from pathlib import Path

from moonshine_voice import Transcriber
from wyoming.asr import Transcript
from wyoming.audio import AudioChunk, AudioStart
from wyoming.error import Error as WyomingError
from wyoming.event import Event
from wyoming.info import AsrModel, AsrProgram, Attribution, Info
from wyoming.server import AsyncEventHandler

from app.audio import pcm_int16_to_float32, validate_audio_format
from app.debug_audio import DEFAULT_DEBUG_AUDIO_DIR, save_debug_audio
from app.models import get_model_info
from app.streaming import MoonshineStreamingSession

_LOGGER = logging.getLogger(__name__)

_ATTRIBUTION_PROGRAM = Attribution(
    name="HomeIntent Contributors",
    url="https://github.com/pquandel2-alt/homeintent-moonshine-addon",
)
_ATTRIBUTION_MODEL = Attribution(
    name="Moonshine AI",
    url="https://github.com/moonshine-ai/moonshine",
)


class MoonshineAsrHandler(AsyncEventHandler):
    """Handles Wyoming STT events and runs Moonshine for transcription.

    Per Wyoming connection, one handler instance sharing the single
    pre-loaded Transcriber; per transcription request, one isolated
    MoonshineStreamingSession created via Transcriber.create_stream().

    Key design: add_audio() is called PER Wyoming audio-chunk (streaming),
    not after all audio is buffered.
    """

    def __init__(
        self,
        reader: asyncio.StreamReader,
        writer: asyncio.StreamWriter,
        transcriber: Transcriber,
        model_name: str,
        language: str = "de",
        log_transcripts: bool = False,
        log_performance: bool = True,
        save_debug_audio_enabled: bool = False,
        debug_audio_max_files: int = 100,
        debug_audio_dir: Path = DEFAULT_DEBUG_AUDIO_DIR,
        moonshine_lock: asyncio.Lock | None = None,
    ):
        """Initialize handler.

        Args:
            reader: Wyoming protocol reader
            writer: Wyoming protocol writer
            transcriber: The single Transcriber loaded once at add-on startup
            model_name: Model to use ("tiny" or "small")
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

        self._session: MoonshineStreamingSession | None = None
        self._audio_rejected = False
        self._raw_audio_buffer: bytearray | None = None

    async def handle_event(self, event: Event) -> bool:
        """Main event handler for Wyoming STT lifecycle.

        Handles:
        - describe: Service discovery
        - transcribe: Start a new transcription
        - audio-start/chunk/stop: Audio stream lifecycle
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
        client vanishes mid-utterance (no audio-stop ever arrives).
        """
        self._close_session()

    async def _handle_describe(self) -> None:
        """Respond to service discovery request."""
        model_info = get_model_info(self._model_name)

        asr_model = AsrModel(
            name=model_info.get("name", "unknown"),
            attribution=_ATTRIBUTION_MODEL,
            installed=True,
            description=model_info.get("description", ""),
            version=model_info.get("name", "0.1.0"),
            languages=[self._language],
        )
        asr_program = AsrProgram(
            name="homeintent-moonshine",
            attribution=_ATTRIBUTION_PROGRAM,
            installed=True,
            description="HomeIntent Moonshine STT - German streaming ASR",
            version=None,
            models=[asr_model],
        )
        info = Info(asr=[asr_program])
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
        started = time.monotonic()
        try:
            text = await session.finalize()
        finally:
            self._close_session()
        finalize_time = time.monotonic() - started

        await self.write_event(Transcript(text=text, language=self._language).event())

        if self._log_transcripts:
            _LOGGER.info("Transcript sent: '%s'", text)
        else:
            _LOGGER.info("Transcript sent (%d chars)", len(text))

        if self._log_performance:
            audio_duration = session.audio_duration_seconds
            rtf = finalize_time / audio_duration if audio_duration > 0 else 0.0
            _LOGGER.info(
                "STT completed: model=%s audio=%.2fs finalize=%.2fs rtf=%.2f",
                self._model_name,
                audio_duration,
                finalize_time,
                rtf,
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
