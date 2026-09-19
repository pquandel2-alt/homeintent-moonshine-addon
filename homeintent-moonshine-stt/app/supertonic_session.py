"""Supertonic 3 synthesis session management.

Bridges Wyoming synthesize requests to sherpa-onnx's real, native
callback-streaming API for ``OfflineTts`` -- verified directly from
sherpa-onnx's own pybind11 sources (``sherpa-onnx/python/csrc/offline-tts.cc``):

    tts.generate(text, config, callback)

where ``callback(samples: np.ndarray, progress: float) -> int`` is invoked
by the native C++ implementation as audio is generated, and the whole
``generate()`` call releases the GIL (``py::call_guard<py::gil_scoped_release>``)
so it is safe to drive from a background thread exactly like Pocket TTS's
own ``generate_audio_stream()`` (see app/tts_session.py) -- this is real
incremental generation, not a WAV-file round trip: each chunk the callback
receives is forwarded to app/handler.py's Wyoming AudioChunk stream as soon
as it exists, before the rest of the utterance has even been synthesized.

Threading/producer-consumer design deliberately mirrors
PocketTtsSynthesizer (app/tts_session.py) as closely as possible: a
background thread runs the blocking, GIL-released ``generate()`` call and
pushes each callback-delivered chunk onto a ``queue.Queue``; the async
generator consumes that queue via ``asyncio.to_thread(queue.get)``. Returning
a non-zero value from the callback is sherpa-onnx's own documented way to
stop generation early (verified from the same pybind source above), used
here exactly like Pocket TTS's ``stop_event`` on client disconnect.
"""

import asyncio
import logging
import queue
import threading
import time
from collections.abc import AsyncGenerator, Callable

import numpy as np
from wyoming.info import Attribution

from app.supertonic_tts import (
    DEFAULT_SUPERTONIC_STEPS,
    DEFAULT_SUPERTONIC_VOICE,
    SUPERTONIC_MODEL_METADATA,
    resolve_supertonic_sid,
)
from app.tts_engine import TtsSynthesisStats

_LOGGER = logging.getLogger(__name__)

ENGINE_ID = "supertonic_3"

ATTRIBUTION = Attribution(
    name="Supertone Inc. (Supertonic) via k2-fsa/sherpa-onnx",
    url="https://github.com/supertone-inc/supertonic",
)

_PRODUCER_JOIN_TIMEOUT_SECONDS = 5.0


class SupertonicSynthesizer:
    """Wraps one shared ``sherpa_onnx.OfflineTts`` Supertonic instance.

    Every call into ``self._tts`` is serialized through ``self._lock`` for
    the full duration of one synthesis request -- same "correctness over
    unproven parallelism" stance already taken for Pocket TTS/Kokoro
    (neither sherpa-onnx's docs nor its source rule out that concurrent
    ``generate()`` calls on one ``OfflineTts`` instance are unsafe).
    """

    def __init__(
        self,
        tts: object,  # sherpa_onnx.OfflineTts; typed as object, see app/supertonic_tts.py
        default_voice: str = DEFAULT_SUPERTONIC_VOICE,
        speed: float = 1.0,
        num_steps: int = DEFAULT_SUPERTONIC_STEPS,
        language: str = "de",
        lock: asyncio.Lock | None = None,
    ) -> None:
        self._tts = tts
        self._default_voice = default_voice
        self._speed = speed
        self._num_steps = num_steps
        self._language = language
        self._lock = lock if lock is not None else asyncio.Lock()

    @property
    def engine_id(self) -> str:
        return ENGINE_ID

    @property
    def sample_rate(self) -> int:
        return int(self._tts.sample_rate)  # type: ignore[attr-defined]

    @property
    def default_voice(self) -> str:
        return self._default_voice

    @property
    def model_name(self) -> str:
        return str(SUPERTONIC_MODEL_METADATA["name"])

    @property
    def program_name(self) -> str:
        return "homeintent-supertonic-3"

    @property
    def attribution(self) -> Attribution:
        return ATTRIBUTION

    @property
    def description(self) -> str:
        return str(SUPERTONIC_MODEL_METADATA["description"])

    async def preload_default_voice(self) -> None:
        """Validate the configured default voice/sid at startup, exactly
        like Pocket TTS's/Kokoro's own preload_default_voice()."""
        resolve_supertonic_sid(self._default_voice)

    async def warmup(self, text: str = "Eins, zwei, drei.") -> float:
        started = time.monotonic()
        async for _ in self.synthesize_stream(text, self._default_voice):
            pass  # discarded: warms up ONNX Runtime's session
        return time.monotonic() - started

    def _make_producer(
        self,
        sid: int,
        text: str,
        chunk_queue: "queue.Queue[np.ndarray | BaseException | None]",
        stop_event: threading.Event,
        stats: TtsSynthesisStats | None,
    ) -> "Callable[[], None]":
        import sherpa_onnx

        def _produce() -> None:
            gen_config = sherpa_onnx.GenerationConfig()
            gen_config.sid = sid
            gen_config.speed = self._speed
            gen_config.num_steps = self._num_steps
            gen_config.extra["lang"] = self._language

            generation_started = time.monotonic()

            def _callback(samples: np.ndarray, _progress: float) -> int:
                nonlocal generation_started
                if stats is not None:
                    stats.model_generation_seconds += time.monotonic() - generation_started
                if stop_event.is_set():
                    return 1  # non-zero: sherpa-onnx stops generation early
                if len(samples) > 0:
                    chunk_queue.put(np.asarray(samples, dtype=np.float32).copy())
                generation_started = time.monotonic()
                return 0

            try:
                self._tts.generate(text, gen_config, _callback)  # type: ignore[attr-defined]
            except BaseException as err:  # noqa: BLE001 - forwarded to the consumer
                chunk_queue.put(err)
                return
            chunk_queue.put(None)

        return _produce

    async def synthesize_stream(
        self, text: str, voice: str | None = None, stats: TtsSynthesisStats | None = None
    ) -> AsyncGenerator[np.ndarray, None]:
        """Yield mono float32 numpy chunks as Supertonic generates them via
        sherpa-onnx's native callback -- see module docstring."""
        lock_wait_started = time.monotonic()
        sid = resolve_supertonic_sid(voice or self._default_voice)

        chunk_queue: queue.Queue[np.ndarray | BaseException | None] = queue.Queue()
        stop_event = threading.Event()
        _produce = self._make_producer(sid, text, chunk_queue, stop_event, stats)

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
                stop_event.set()
                await asyncio.to_thread(thread.join, _PRODUCER_JOIN_TIMEOUT_SECONDS)
                if thread.is_alive():
                    _LOGGER.warning(
                        "Supertonic producer thread did not exit within %.1fs after "
                        "cancellation; continuing without waiting further",
                        _PRODUCER_JOIN_TIMEOUT_SECONDS,
                    )
