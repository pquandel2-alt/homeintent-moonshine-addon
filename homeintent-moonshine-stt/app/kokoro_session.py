"""Kokoro ONNX synthesis session management.

Bridges Wyoming synthesize requests to Kokoro ONNX's own streaming
generation API. See app/kokoro_tts.py's module docstring for the verified
facts this design relies on (API signatures, thread-safety caveats,
sample rate, phonemization).

Thread-/concurrency-safety: kokoro-onnx documents its own espeak-ng calls
as needing a lock (a module-global ``threading.Lock`` inside
``kokoro_onnx.tokenizer``, verified from source) because espeak-ng holds
process-global state, but does NOT document whether concurrently calling
``Kokoro.create_stream()`` for two different requests against the same
loaded model/session is safe end-to-end. Per this project's own
"correctness over parallelism" policy for an undocumented case (see
app/tts_session.py's identical reasoning for Pocket TTS), every request is
serialized through one shared lock here too, exactly like Pocket TTS --
not because a problem was found, but because none could be ruled out from
upstream's own documentation.
"""

import asyncio
import logging
import time
from collections.abc import AsyncGenerator

import numpy as np
from wyoming.info import Attribution

from app.german_text_normalizer import normalize_german_text
from app.kokoro_tts import DEFAULT_KOKORO_VOICE, KOKORO_MODEL_METADATA, KOKORO_SAMPLE_RATE
from app.tts_engine import TtsSynthesisStats

_LOGGER = logging.getLogger(__name__)

ENGINE_ID = "kokoro_onnx"

ATTRIBUTION = Attribution(
    name="Godelaune (German Martin fine-tune) / hexgrad (Kokoro-82M)",
    url="https://huggingface.co/Godelaune/Kokoro-82M-ONNX-German-Martin",
)


class KokoroOnnxSynthesizer:
    """Wraps one shared Kokoro ONNX model instance for reuse across requests.

    ``create_stream()`` is a method on the same loaded ``kokoro_onnx.Kokoro``
    instance for every request; every call is serialized through
    ``self._lock`` for the full duration of one synthesis request (see
    module docstring), independent of Pocket TTS's own lock -- the two
    engines never run at the same time in practice (only one is selected
    via ``tts_engine``), but keeping the locks separate avoids coupling two
    otherwise-unrelated runtimes' concurrency behavior together.
    """

    def __init__(
        self,
        kokoro: object,  # kokoro_onnx.Kokoro; typed as object to avoid a
        # hard import-time dependency on kokoro_onnx for callers that only
        # need the TtsSynthesizer protocol (e.g. tests using a fake).
        default_voice: str = DEFAULT_KOKORO_VOICE,
        speed: float = 1.0,
        sentence_pause: float = 0.25,
        clause_pause: float = 0.1,
        lock: asyncio.Lock | None = None,
    ) -> None:
        self._kokoro = kokoro
        self._default_voice = default_voice
        self._speed = speed
        self._sentence_pause = sentence_pause
        self._clause_pause = clause_pause
        self._lock = lock if lock is not None else asyncio.Lock()

    @property
    def engine_id(self) -> str:
        return ENGINE_ID

    @property
    def sample_rate(self) -> int:
        return KOKORO_SAMPLE_RATE

    @property
    def default_voice(self) -> str:
        return self._default_voice

    @property
    def model_name(self) -> str:
        return KOKORO_MODEL_METADATA["name"]

    @property
    def program_name(self) -> str:
        return "homeintent-kokoro-onnx"

    @property
    def attribution(self) -> Attribution:
        return ATTRIBUTION

    @property
    def description(self) -> str:
        return KOKORO_MODEL_METADATA["description"]

    async def preload_default_voice(self) -> None:
        """Validate the configured default voice against the loaded
        model's real voice list.

        Called once at add-on startup (see app/__main__.py) so an invalid
        ``kokoro_voice`` name is caught immediately with a clear startup
        error, instead of surfacing only on the first real synthesize
        request -- same rationale as
        PocketTtsSynthesizer.preload_default_voice().
        """
        available = self._kokoro.get_voices()  # type: ignore[attr-defined]
        if self._default_voice not in available:
            raise ValueError(
                f"Kokoro voice '{self._default_voice}' not found in this model; "
                f"available voices: {sorted(available)}"
            )

    async def warmup(self, text: str = "Eins, zwei, drei.") -> float:
        """Run one full, discarded synthesis of ``text`` and return its
        wall-clock duration in seconds -- warms up ONNX Runtime's session
        and espeak-ng's phonemizer so the first real request isn't slower
        than later ones. Must be called only after
        ``preload_default_voice()`` succeeded.
        """
        started = time.monotonic()
        async for _ in self.synthesize_stream(text, self._default_voice):
            pass  # discarded: this call exists purely to warm up the runtime
        return time.monotonic() - started

    async def synthesize_stream(
        self, text: str, voice: str | None = None, stats: TtsSynthesisStats | None = None
    ) -> AsyncGenerator[np.ndarray, None]:
        """Yield mono float32 numpy chunks as Kokoro ONNX generates them.

        Real incremental streaming: ``kokoro_onnx.Kokoro.create_stream()``
        is itself a native async generator that yields each phoneme
        batch's audio as soon as it is synthesized, in a background
        asyncio executor thread -- not "generate everything, then chunk
        it" (verified directly against its source, see
        app/kokoro_tts.py's module docstring).

        The text is run through :func:`normalize_german_text` first (times,
        temperatures, units, currency -- see app/german_text_normalizer.py)
        since kokoro-onnx's own tokenizer does no such normalization
        itself (verified: it only trims whitespace before phonemizing).
        This is Kokoro-specific and deliberately never applied to Pocket
        TTS's own pipeline (app/tts_session.py) -- Pocket TTS's real e2e
        test already establishes it handles this class of German
        formatting on its own.

        If ``stats`` is given, populated exactly like
        PocketTtsSynthesizer.synthesize_stream(): ``lock_wait_seconds`` is
        time spent waiting for the shared lock, ``model_generation_seconds``
        is cumulative real time inside Kokoro's own generation, excluding
        time this method spends waiting for the caller to consume a
        yielded chunk.
        """
        lock_wait_started = time.monotonic()
        normalized_text = normalize_german_text(text)
        resolved_voice = voice or self._default_voice

        async with self._lock:
            if stats is not None:
                stats.lock_wait_seconds = time.monotonic() - lock_wait_started

            agen = self._kokoro.create_stream(  # type: ignore[attr-defined]
                normalized_text,
                voice=resolved_voice,
                speed=self._speed,
                lang="de",
                trim=True,
                sentence_pause=self._sentence_pause,
                clause_pause=self._clause_pause,
            )
            generation_started = time.monotonic()
            try:
                async for chunk, _chunk_sample_rate in agen:
                    if stats is not None:
                        stats.model_generation_seconds += time.monotonic() - generation_started
                    yield np.asarray(chunk, dtype=np.float32)
                    generation_started = time.monotonic()
            finally:
                # Runs on normal completion, a synthesis error, AND on the
                # caller cancelling this generator (client disconnect
                # mid-stream) -- always releases Kokoro's own background
                # task before this method's lock is released for the next
                # request.
                await agen.aclose()
