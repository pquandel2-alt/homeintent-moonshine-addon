"""Wyoming ASR event handler for Moonshine streaming."""

import logging
from typing import Optional

from wyoming.asr import Transcript
from wyoming.audio import AudioChunk, AudioStart
from wyoming.event import Event
from wyoming.server import AsyncEventHandler

from app.audio import pcm_int16_to_float32, validate_audio_format
from app.models import get_model_info
from app.streaming import MoonshineStreamingSession

_LOGGER = logging.getLogger(__name__)


class MoonshineAsrHandler(AsyncEventHandler):
    """Handles Wyoming STT events and runs Moonshine for transcription.

    Per Wyoming connection, one handler instance.
    Per transcribe request, one Moonshine session.

    Key design: add_audio() is called PER Wyoming audio-chunk (streaming),
    not after all audio is buffered.
    """

    def __init__(
        self,
        reader,
        writer,
        transcriber_factory,
        model_name: str,
        language: str = "de",
    ):
        """Initialize handler.

        Args:
            reader: Wyoming protocol reader
            writer: Wyoming protocol writer
            transcriber_factory: Callable that returns a loaded Moonshine Transcriber
            model_name: Model to use ("tiny" or "small")
            language: Language code ("de" for German)
        """
        super().__init__(reader, writer)
        self._transcriber_factory = transcriber_factory
        self._model_name = model_name
        self._language = language

        self._session: Optional[MoonshineStreamingSession] = None
        self._audio_format: Optional[AudioStart] = None

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
                    self._handle_audio_start(event)

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

        asr_program = {
            "name": "homeintent-moonshine",
            "attribution": {
                "name": "HomeIntent Contributors",
                "url": "https://github.com/pquandel/homeintent-moonshine-addon",
            },
            "installed": True,
            "description": "HomeIntent Moonshine STT - German streaming ASR",
            "models": [
                {
                    "name": model_info.get("name", "unknown"),
                    "attribution": {
                        "name": "Moonshine AI",
                        "url": "https://github.com/moonshine-ai/moonshine",
                    },
                    "installed": True,
                    "description": model_info.get("description", ""),
                    "languages": [self._language],
                    "version": model_info.get("name", "0.1.0"),
                }
            ],
        }

        info_event = Event(type="info", data={"asr": [asr_program]})
        _LOGGER.debug(f"Sending service info: {info_event.data}")
        await self.write_event(info_event)

    def _reset_session(self) -> None:
        """Reset for a new transcription request."""
        self._session = None
        self._audio_format = None
        _LOGGER.debug("Reset for new transcription")

    def _handle_audio_start(self, event: Event) -> None:
        """Handle audio stream start."""
        audio_start = AudioStart.from_event(event)

        # Validate audio format
        error = validate_audio_format(
            audio_start.rate or 16000,
            audio_start.width or 2,
            audio_start.channels or 1,
            context="audio-start",
        )

        self._audio_format = audio_start

        # Create new Moonshine session
        transcriber = self._transcriber_factory()
        self._session = MoonshineStreamingSession(transcriber)
        self._session.start()

        _LOGGER.debug(
            f"Audio stream started: {audio_start.rate}Hz, "
            f"{audio_start.width}B width, {audio_start.channels} channel(s)"
        )

    async def _handle_audio_chunk(self, event: Event) -> None:
        """Handle audio chunk. CRITICAL: feed to Moonshine immediately."""
        if self._session is None:
            _LOGGER.warning("audio-chunk received before audio-start")
            return

        chunk = AudioChunk.from_event(event)

        # Validate format
        validate_audio_format(
            chunk.rate or 16000,
            chunk.width or 2,
            chunk.channels or 1,
            context="audio-chunk",
        )

        # Convert: Wyoming int16 PCM → Moonshine float32
        try:
            float32_audio = pcm_int16_to_float32(chunk.audio)
        except ValueError as e:
            _LOGGER.error(f"Audio conversion failed: {e}")
            return

        # STREAMING: Feed immediately to Moonshine (not buffered)
        self._session.add_audio(float32_audio)

    async def _handle_audio_stop(self) -> None:
        """Handle audio stream end and send transcript."""
        if self._session is None:
            # Empty or invalid stream
            await self.write_event(Transcript(text="", language=self._language).event())
            _LOGGER.debug("Empty transcription")
            return

        # Finalize and get result
        text = await self._session.finalize()
        self._session = None

        await self.write_event(Transcript(text=text, language=self._language).event())
        _LOGGER.info(f"Transcript sent: '{text}'")
