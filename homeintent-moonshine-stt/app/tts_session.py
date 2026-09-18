"""Pocket TTS synthesis session management.

Bridges Wyoming synthesize requests to Pocket TTS's own streaming
generation API. See app/tts.py's module docstring for the verified facts
this design relies on (thread-safety, streaming semantics, voice
resolution).
"""

import asyncio
import logging
import queue
import threading
import time
from collections.abc import AsyncGenerator, Callable
from dataclasses import dataclass
from typing import Protocol

import numpy as np
from pocket_tts import TTSModel

_LOGGER = logging.getLogger(__name__)


@dataclass
class TtsSynthesisStats:
    """Per-request timing breakdown, populated by synthesize_stream() as it
    runs (see its docstring). A fresh instance per request -- not shared or
    reused across calls, and safe to read only after the request has
    finished (normally or with an error), since the values are written
    incrementally from a background thread while the request is in flight.

    Only the synthesizer itself can distinguish real model compute time
    from time spent waiting for the shared lock: both are opaque to
    app/handler.py, which measures the outer request wall-time and its own
    Wyoming I/O time instead (see app/handler.py's _SynthesisStats).
    """

    lock_wait_seconds: float = 0.0
    model_generation_seconds: float = 0.0


class TtsSynthesizer(Protocol):
    """Structural interface app/handler.py depends on for TTS.

    Lets tests substitute a lightweight fake without subclassing the real
    PocketTtsSynthesizer (which requires a real/mocked TTSModel).
    """

    @property
    def sample_rate(self) -> int: ...

    @property
    def default_voice(self) -> str: ...

    def synthesize_stream(
        self, text: str, voice: str | None = None, stats: TtsSynthesisStats | None = None
    ) -> AsyncGenerator[np.ndarray, None]: ...


# How long to wait for the producer thread to notice cancellation and exit
# after a consumer stops iterating early (client disconnect mid-synthesis,
# see B26). Best-effort: Pocket TTS's own internal generation/decode
# threads are not directly cancellable from here (no public API for that),
# so this only stops *our* forwarding thread; the model's own threads will
# be cleaned up by the generator's own GeneratorExit handling once garbage
# collected.
_PRODUCER_JOIN_TIMEOUT_SECONDS = 5.0


class PocketTtsSynthesizer:
    """Wraps one shared Pocket TTS model instance for reuse across requests.

    ``generate_audio_stream()``/``generate_audio()``/
    ``get_state_for_audio_prompt()`` are all methods on the same
    ``TTSModel`` instance, and upstream explicitly documents generation as
    NOT thread-safe ("separate model instances should be used for
    concurrent generation" -- see app/tts.py's module docstring). Every
    call into ``self._model`` is therefore serialized through
    ``self._lock``, for the full duration of one synthesis request --
    concurrent Wyoming TTS requests are correct, if not concurrent, rather
    than racing on shared native/torch state.

    This lock is independent of Moonshine's STT lock (app/streaming.py):
    Moonshine and Pocket TTS are unrelated runtimes with no shared state,
    so an STT request never waits on a TTS request or vice versa.
    """

    def __init__(
        self,
        model: TTSModel,
        default_voice: str,
        lock: asyncio.Lock | None = None,
    ) -> None:
        self._model = model
        self._default_voice = default_voice
        self._lock = lock if lock is not None else asyncio.Lock()
        # get_state_for_audio_prompt() is documented upstream as "relatively
        # slow" -- cache resolved voice states by name so repeated requests
        # for the same voice (the common case: one configured default voice)
        # don't pay that cost every time.
        self._voice_state_cache: dict[str, object] = {}

    @property
    def sample_rate(self) -> int:
        return int(self._model.sample_rate)

    @property
    def default_voice(self) -> str:
        return self._default_voice

    async def _resolve_voice_state(self, voice: str) -> object:
        cached = self._voice_state_cache.get(voice)
        if cached is not None:
            return cached
        async with self._lock:
            cached = self._voice_state_cache.get(voice)  # re-check after awaiting the lock
            if cached is not None:
                return cached
            state = await asyncio.to_thread(self._model.get_state_for_audio_prompt, voice)
        self._voice_state_cache[voice] = state
        return state

    async def preload_default_voice(self) -> None:
        """Resolve and cache the configured default voice's state.

        Called once at add-on startup (see app/__main__.py's
        _load_tts_synthesizer()) so an invalid ``tts_voice`` name is caught
        immediately, with a clear startup error, instead of surfacing only
        on the first real synthesize request. Also means the first real
        request never pays get_state_for_audio_prompt()'s own documented
        "relatively slow" cost (see app/tts.py's module docstring).
        """
        await self._resolve_voice_state(self._default_voice)

    async def warmup(self, text: str = "Eins, zwei, drei.") -> float:
        """Run one full, discarded synthesis of ``text`` and return its
        wall-clock duration in seconds.

        Rationale (see app/__main__.py's ``tts_warmup`` option): Pocket TTS
        itself performs no explicit JIT compilation or quantization at load
        time (verified directly against upstream source -- no
        ``torch.compile``/``torch.jit`` call sites in the model's own
        forward path), so there is no framework-level "first call recompiles
        the graph" effect to warm away. What this warms up is the more
        general, well-documented PyTorch-on-CPU cost of a process's very
        first forward pass: the caching memory allocator and OpenMP/MKL
        thread pools initialize lazily on first use and are otherwise paid
        by whichever request happens to be first. Must be called only after
        ``preload_default_voice()`` -- it reuses the now-cached voice state
        rather than resolving it again.
        """
        started = time.monotonic()
        async for _ in self.synthesize_stream(text, self._default_voice):
            pass  # discarded: this call exists purely to warm up torch, no audio output
        return time.monotonic() - started

    def _make_producer(
        self,
        voice_state: object,
        text: str,
        chunk_queue: "queue.Queue[np.ndarray | BaseException | None]",
        stop_event: threading.Event,
        stats: TtsSynthesisStats | None,
    ) -> "Callable[[], None]":
        """Build the background-thread function that drives
        generate_audio_stream() and forwards its chunks (and timing, if
        ``stats`` is given) to synthesize_stream()'s consumer loop."""

        def _produce() -> None:
            try:
                generation_started = time.monotonic()
                for chunk in self._model.generate_audio_stream(voice_state, text):
                    if stats is not None:
                        stats.model_generation_seconds += time.monotonic() - generation_started
                    if stop_event.is_set():
                        return
                    chunk_queue.put(chunk.detach().cpu().numpy())
                    generation_started = time.monotonic()
            except BaseException as err:  # noqa: BLE001 - forwarded to the consumer, not swallowed
                chunk_queue.put(err)
                return
            chunk_queue.put(None)

        return _produce

    async def synthesize_stream(
        self, text: str, voice: str | None = None, stats: TtsSynthesisStats | None = None
    ) -> AsyncGenerator[np.ndarray, None]:
        """Yield mono float32 numpy chunks as Pocket TTS generates them.

        Real incremental streaming: each chunk is forwarded to the caller
        as soon as Pocket TTS's own generate_audio_stream() yields it, not
        after the full utterance has been synthesized. Holds the shared
        lock for the entire request (start to finish) so concurrent TTS
        requests are serialized rather than interleaved on the same model.

        If ``stats`` is given, it is populated with a timing breakdown
        (see TtsSynthesisStats): ``lock_wait_seconds`` is how long this
        call waited to acquire the shared lock (time some *other* request
        was still using the model, not this request's own work);
        ``model_generation_seconds`` is the cumulative real time spent
        inside Pocket TTS's own generate_audio_stream() -- i.e. actual
        model compute, excluding time this method spends waiting for the
        consumer to pull a chunk off the queue (client backpressure/Wyoming
        I/O time must never be counted as model time, see B-item "Client
        backpressure").
        """
        lock_wait_started = time.monotonic()
        voice_state = await self._resolve_voice_state(voice or self._default_voice)

        chunk_queue: queue.Queue[np.ndarray | BaseException | None] = queue.Queue()
        stop_event = threading.Event()
        _produce = self._make_producer(voice_state, text, chunk_queue, stop_event, stats)

        async with self._lock:
            if stats is not None:
                stats.lock_wait_seconds = time.monotonic() - lock_wait_started
            thread = threading.Thread(target=_produce, daemon=True)
            thread.start()
            try:
                while True:
                    item = await asyncio.to_thread(chunk_queue.get)
                    if item is None:
                        break
                    if isinstance(item, BaseException):
                        raise item
                    yield item
            finally:
                # Runs on normal completion, an exception, AND on the
                # consumer stopping early (aclose()/GeneratorExit on client
                # disconnect) -- always signal the producer to stop pulling
                # further chunks and wait for it to actually exit before
                # releasing the lock for the next request.
                stop_event.set()
                await asyncio.to_thread(thread.join, _PRODUCER_JOIN_TIMEOUT_SECONDS)
                if thread.is_alive():
                    _LOGGER.warning(
                        "Pocket TTS producer thread did not exit within %.1fs after "
                        "cancellation; continuing without waiting further",
                        _PRODUCER_JOIN_TIMEOUT_SECONDS,
                    )
