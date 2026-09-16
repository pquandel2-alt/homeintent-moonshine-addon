"""Moonshine streaming session management."""

import asyncio
import logging
from typing import Callable, Optional

import numpy as np
from moonshine_voice import Transcriber, TranscriptEventListener

_LOGGER = logging.getLogger(__name__)


class TranscriptLine:
    """Represents a completed transcript line."""

    def __init__(self, text: str):
        self.text = text


class MoonshineListener(TranscriptEventListener):
    """Listener for Moonshine transcription events.

    Bridges Moonshine's callback-based event system with asyncio.
    """

    def __init__(self, loop: asyncio.AbstractEventLoop):
        self._loop = loop
        self._done = asyncio.Event()
        self.final_text: str = ""
        self._partial_text: str = ""

    def on_line_started(self) -> None:
        """Called when a new line (utterance) starts."""
        _LOGGER.debug("Transcript line started")
        self._partial_text = ""

    def on_line_updated(self) -> None:
        """Called when a line is updated."""
        _LOGGER.debug("Transcript line updated")

    def on_line_text_changed(self, line_text: str) -> None:
        """Called when the recognized text changes (partial result).

        This is called multiple times as new words are recognized during
        the audio stream. Useful for real-time UI updates.
        """
        self._partial_text = line_text
        _LOGGER.debug(f"Partial transcript: {line_text}")

    def on_line_completed(self, line_text: str) -> None:
        """Called when a line (utterance) completes.

        This is the final, highest-confidence transcription.
        """
        self.final_text = line_text
        _LOGGER.info(f"Transcript completed: {line_text}")
        # Signal that we're done (via thread-safe call)
        self._loop.call_soon_threadsafe(self._done.set)

    def on_error(self, error: str) -> None:
        """Called if an error occurs during transcription."""
        _LOGGER.error(f"Transcription error: {error}")
        self._loop.call_soon_threadsafe(self._done.set)

    def wait_for_completion(self) -> asyncio.Event:
        """Return the event that signals completion."""
        return self._done

    def reset(self) -> None:
        """Reset for a new utterance."""
        self._done.clear()
        self.final_text = ""
        self._partial_text = ""


class MoonshineStreamingSession:
    """Manages a single Moonshine transcription session.

    This wraps the Moonshine Transcriber API to support streaming:
    - start(): Begin a new transcription session
    - add_audio(float32_pcm): Feed audio data immediately (non-blocking)
    - finalize(): Wait for transcription to complete and get the result

    One session per Wyoming transcription request.
    Per-connection: new session for each transcribe request.
    """

    def __init__(self, transcriber: Transcriber):
        """Initialize session with a Transcriber instance.

        Args:
            transcriber: Already-loaded Moonshine Transcriber
        """
        self._transcriber = transcriber
        loop = asyncio.get_event_loop()
        self._listener = MoonshineListener(loop)
        self._transcriber.add_listener(self._listener)
        self._started = False

    def start(self) -> None:
        """Start a new transcription session.

        Must be called before feeding audio.
        """
        _LOGGER.debug("Starting Moonshine session")
        self._listener.reset()
        self._transcriber.start()
        self._started = True

    def add_audio(self, float32_pcm: np.ndarray) -> None:
        """Feed audio to the streaming decoder.

        CRITICAL: This is called PER Wyoming audio-chunk, not after all audio is received.
        This enables real-time streaming.

        Args:
            float32_pcm: Audio samples as float32 array, normalized to [-1.0, 1.0]
        """
        if not self._started:
            _LOGGER.warning("add_audio called before session started")
            return

        if len(float32_pcm) == 0:
            _LOGGER.warning("Empty audio chunk")
            return

        # Feed to Moonshine (non-blocking, queued internally)
        try:
            self._transcriber.add_audio(float32_pcm, 16000)
            _LOGGER.debug(f"Fed {len(float32_pcm)} audio samples to Moonshine")
        except Exception as e:
            _LOGGER.error(f"Failed to add audio: {e}")
            raise

    async def finalize(self) -> str:
        """Stop the session and wait for final transcript.

        This:
        1. Tells Moonshine there's no more audio
        2. Waits for the listener to receive the final result
        3. Returns the transcript text

        Returns:
            Final transcribed text
        """
        if not self._started:
            return ""

        _LOGGER.debug("Finalizing Moonshine session")

        # stop() may block, so run in thread
        try:
            await asyncio.to_thread(self._transcriber.stop)
        except Exception as e:
            _LOGGER.error(f"Error stopping transcriber: {e}")
            return self._listener.final_text

        # Wait for listener to receive completion event (with timeout)
        try:
            await asyncio.wait_for(self._listener.wait_for_completion().wait(), timeout=10.0)
        except asyncio.TimeoutError:
            _LOGGER.warning("Timeout waiting for transcript completion")

        self._started = False
        return self._listener.final_text
