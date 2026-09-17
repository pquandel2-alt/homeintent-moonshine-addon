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
from collections.abc import AsyncGenerator
from typing import Protocol

import numpy as np
from pocket_tts import TTSModel

_LOGGER = logging.getLogger(__name__)


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
        self, text: str, voice: str | None = None
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

    async def synthesize_stream(
        self, text: str, voice: str | None = None
    ) -> AsyncGenerator[np.ndarray, None]:
        """Yield mono float32 numpy chunks as Pocket TTS generates them.

        Real incremental streaming: each chunk is forwarded to the caller
        as soon as Pocket TTS's own generate_audio_stream() yields it, not
        after the full utterance has been synthesized. Holds the shared
        lock for the entire request (start to finish) so concurrent TTS
        requests are serialized rather than interleaved on the same model.
        """
        voice_state = await self._resolve_voice_state(voice or self._default_voice)

        chunk_queue: queue.Queue[np.ndarray | BaseException | None] = queue.Queue()
        stop_event = threading.Event()

        def _produce() -> None:
            try:
                for chunk in self._model.generate_audio_stream(voice_state, text):
                    if stop_event.is_set():
                        return
                    chunk_queue.put(chunk.detach().cpu().numpy())
            except BaseException as err:  # noqa: BLE001 - forwarded to the consumer, not swallowed
                chunk_queue.put(err)
                return
            chunk_queue.put(None)

        async with self._lock:
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
