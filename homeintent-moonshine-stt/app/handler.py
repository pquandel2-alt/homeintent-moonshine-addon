"""Wyoming ASR event handler for Moonshine streaming."""

import asyncio
import logging

from moonshine_voice import Transcriber
from wyoming.asr import Transcript
from wyoming.audio import AudioChunk, AudioStart
from wyoming.error import Error as WyomingError
from wyoming.event import Event
from wyoming.info import AsrModel, AsrProgram, Attribution, Info
from wyoming.server import AsyncEventHandler

from app.audio import pcm_int16_to_float32, validate_audio_format
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
    ):
        """Initialize handler.

        Args:
            reader: Wyoming protocol reader
            writer: Wyoming protocol writer
            transcriber: The single Transcriber loaded once at add-on startup
            model_name: Model to use ("tiny" or "small")
            language: Language code ("de" for German)
        """
        super().__init__(reader, writer)
        self._transcriber = transcriber
        self._model_name = model_name
        self._language = language

        self._session: MoonshineStreamingSession | None = None
        self._audio_rejected = False

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
            return False

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

    def _reset_session(self) -> None:
        """Reset for a new transcription request."""
        if self._session is not None:
            self._session.close()
        self._session = None
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

        self._session = MoonshineStreamingSession(self._transcriber)
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
            self._session.close()
            self._session = None
            await self.write_event(
                WyomingError(text=error, code="unsupported_audio_format").event()
            )
            _LOGGER.error("Rejecting audio stream mid-stream: %s", error)
            return

        float32_audio = pcm_int16_to_float32(chunk.audio)

        # STREAMING: Feed immediately to Moonshine (not buffered)
        await self._session.add_audio(list(float32_audio))

    async def _handle_audio_stop(self) -> None:
        """Handle audio stream end and send transcript."""
        if self._session is None:
            # Empty, invalid, or rejected stream
            await self.write_event(Transcript(text="", language=self._language).event())
            _LOGGER.debug("Empty transcription")
            return

        try:
            text = await self._session.finalize()
        finally:
            self._session.close()
            self._session = None

        await self.write_event(Transcript(text=text, language=self._language).event())
        _LOGGER.info(f"Transcript sent: '{text}'")
