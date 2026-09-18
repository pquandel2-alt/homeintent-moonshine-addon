"""Moonshine streaming session management.

Bridges Wyoming audio chunks to a Moonshine Transcriber stream. Each Wyoming
connection gets its own isolated Stream via Transcriber.create_stream(), so
concurrent/sequential connections never share transcription state while all
of them reuse the single model already loaded into the shared Transcriber.
"""

import asyncio
import logging
import time

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
    """One Wyoming connection's isolated Moonshine streaming session.

    All Streams created on the same Transcriber share its native handle
    (every ``moonshine_transcribe_*`` call takes both the transcriber and
    stream handles), and moonshine-voice 0.1.5 does not document that this
    is safe to call concurrently from multiple threads. Rather than assume
    it is, every native call here is serialized through an ``asyncio.Lock``
    shared across all sessions of the same Transcriber (passed in by the
    caller, see app/__main__.py) -- concurrent Wyoming connections are
    correct, if not concurrent, until upstream documents otherwise.
    """

    def __init__(
        self,
        transcriber: Transcriber,
        update_interval: float | None = None,
        lock: asyncio.Lock | None = None,
    ) -> None:
        """Create an isolated stream on ``transcriber``.

        ``update_interval=None`` (the default) defers to the Transcriber's
        own configured update_interval (see Transcriber.create_stream()),
        so the configured transcription_interval option is honored without
        having to be threaded through twice.

        ``lock=None`` creates a session-local lock, which is fine in
        isolation (e.g. unit tests) but does NOT serialize across sibling
        sessions of the same Transcriber -- production code must pass one
        shared lock per Transcriber.
        """
        self._stream: Stream = transcriber.create_stream(update_interval=update_interval)
        self._collector = _CompletedLineCollector()
        self._stream.add_listener(self._collector)
        self._lock = lock if lock is not None else asyncio.Lock()
        self.audio_duration_seconds = 0.0
        # Cumulative wall time actually spent inside the native
        # add_audio()/stop() calls (see add_audio()/finalize() below) --
        # this is real Moonshine/CPU processing time, not the time the
        # Wyoming client spent streaming audio in (which is bounded by how
        # long the user was talking, not by how long the model took).
        self._add_audio_time_seconds = 0.0
        self.finalize_time_seconds = 0.0
        # Per-chunk breakdown (v0.2.4): a real production install showed
        # audio=4.6s but inference=7+s (RTF > 1) -- these let a performance
        # log line show WHERE that time went: is compute spread evenly
        # across chunks (the model is just slower than real-time on this
        # CPU), or is one or a few chunks pathologically slow (a backlog/
        # stall, not a steady-state throughput problem)?
        self._add_audio_chunk_count = 0
        self._add_audio_compute_max_seconds = 0.0

    @property
    def inference_time_seconds(self) -> float:
        """Total native Moonshine compute time: add_audio() calls + stop().

        Moonshine already does most of its transcription work incrementally
        during add_audio() (that's the point of streaming ASR), so this is
        NOT just how long the final stop()/finalize pass took -- it is the
        sum of every add_audio() call's own processing time plus stop()'s.
        """
        return self._add_audio_time_seconds + self.finalize_time_seconds

    @property
    def add_audio_chunk_count(self) -> int:
        """Number of add_audio() calls (Wyoming AudioChunk events) received."""
        return self._add_audio_chunk_count

    @property
    def add_audio_compute_total_seconds(self) -> float:
        """Cumulative native compute time across all add_audio() calls
        (excludes finalize/stop) -- if this alone exceeds
        ``audio_duration_seconds``, the model is processing slower than
        real-time on this CPU (a genuine throughput problem), independent
        of any single slow chunk."""
        return self._add_audio_time_seconds

    @property
    def add_audio_compute_max_seconds(self) -> float:
        """The single slowest add_audio() call's compute time -- a value
        much larger than the average points at a stall/backlog rather than
        uniformly-slow-but-steady processing."""
        return self._add_audio_compute_max_seconds

    @property
    def average_chunk_compute_seconds(self) -> float:
        if self._add_audio_chunk_count == 0:
            return 0.0
        return self._add_audio_time_seconds / self._add_audio_chunk_count

    async def start(self) -> None:
        """Start the underlying stream."""
        async with self._lock:
            await asyncio.to_thread(self._stream.start)

    async def add_audio(self, samples: list[float], sample_rate: int = 16000) -> None:
        """Feed one chunk of float32 PCM audio into the stream immediately."""
        async with self._lock:
            started = time.monotonic()
            await asyncio.to_thread(self._stream.add_audio, samples, sample_rate)
            elapsed = time.monotonic() - started
            self._add_audio_time_seconds += elapsed
            self._add_audio_chunk_count += 1
            if elapsed > self._add_audio_compute_max_seconds:
                self._add_audio_compute_max_seconds = elapsed
        self.audio_duration_seconds += len(samples) / sample_rate

    async def finalize(self) -> str:
        """Stop the stream and return the accumulated final transcript text.

        Stream.stop() runs a final synchronous update_transcription() pass
        before returning, so every LineCompleted event has already fired by
        the time this coroutine resumes. The time that pass itself took is
        recorded in ``finalize_time_seconds`` (also folded into
        ``inference_time_seconds``) for performance logging.
        """
        async with self._lock:
            started = time.monotonic()
            await asyncio.to_thread(self._stream.stop)
            self.finalize_time_seconds = time.monotonic() - started
        if self._collector.error is not None:
            raise self._collector.error
        return self._collector.text

    def close(self) -> None:
        """Release native stream resources. Call once the session is done.

        Safe to call more than once (e.g. from both a normal finalize path
        and an exception handler's cleanup) -- Stream.close() below is a
        thin wrapper that itself tolerates being called on an
        already-closed handle via moonshine_voice's own bookkeeping.
        """
        self._stream.close()
