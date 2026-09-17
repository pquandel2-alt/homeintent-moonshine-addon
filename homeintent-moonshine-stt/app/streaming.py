"""Moonshine streaming session management.

Bridges Wyoming audio chunks to a Moonshine Transcriber stream. Each Wyoming
connection gets its own isolated Stream via Transcriber.create_stream(), so
concurrent/sequential connections never share transcription state while all
of them reuse the single model already loaded into the shared Transcriber.
"""

import asyncio
import logging

from moonshine_voice import (
    Error,
    LineCompleted,
    Stream,
    Transcriber,
    TranscriptEventListener,
)

_LOGGER = logging.getLogger(__name__)


class _CompletedLineCollector(TranscriptEventListener):  # type: ignore[misc]
    """Accumulates ordered LineCompleted text.

    A single spoken utterance can complete across several transcript lines
    (e.g. on pauses), so completed texts must be appended in order rather
    than overwritten.
    """

    def __init__(self) -> None:
        self._completed_texts: list[str] = []
        self.error: Exception | None = None

    def on_line_completed(self, event: LineCompleted) -> None:
        if event.line.text:
            self._completed_texts.append(event.line.text)

    def on_error(self, event: Error) -> None:
        self.error = event.error
        _LOGGER.error("Moonshine stream error: %s", event.error)

    @property
    def text(self) -> str:
        return " ".join(self._completed_texts)


class MoonshineStreamingSession:
    """One Wyoming connection's isolated Moonshine streaming session."""

    def __init__(self, transcriber: Transcriber, update_interval: float = 0.5) -> None:
        self._stream: Stream = transcriber.create_stream(update_interval=update_interval)
        self._collector = _CompletedLineCollector()
        self._stream.add_listener(self._collector)

    async def start(self) -> None:
        """Start the underlying stream."""
        await asyncio.to_thread(self._stream.start)

    async def add_audio(self, samples: list[float], sample_rate: int = 16000) -> None:
        """Feed one chunk of float32 PCM audio into the stream immediately."""
        await asyncio.to_thread(self._stream.add_audio, samples, sample_rate)

    async def finalize(self) -> str:
        """Stop the stream and return the accumulated final transcript text.

        Stream.stop() runs a final synchronous update_transcription() pass
        before returning, so every LineCompleted event has already fired by
        the time this coroutine resumes.
        """
        await asyncio.to_thread(self._stream.stop)
        if self._collector.error is not None:
            raise self._collector.error
        return self._collector.text

    def close(self) -> None:
        """Release native stream resources. Call once the session is done."""
        self._stream.close()
