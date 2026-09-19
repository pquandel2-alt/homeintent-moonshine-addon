"""Vosk SttEngine: real, in-process streaming German ASR via Kaldi/Vosk.

Uses ``vosk.KaldiRecognizer`` directly -- no subprocess, no separate
container, no websocket server -- verified against the actual Python API
shipped in the ``vosk`` PyPI wheel (see app/vosk_model.py's module
docstring for how the model itself was verified). Audio is fed
incrementally via ``KaldiRecognizer.AcceptWaveform()`` as Wyoming
``audio-chunk`` events arrive (real streaming, matching every other engine
in this add-on), not buffered and transcribed in one batch at the end.

This engine's whole purpose (an explicit design goal, not an
implementation detail) is to be the lightest-weight, lowest-RAM/CPU
streaming STT baseline this add-on offers -- Kaldi's C++ decoder core is
small, CPU-only (no CUDA/GPU dependency in the plain PyPI wheel; see
app/vosk_model.py's module docstring), and the small German model itself
is only ~45MB, versus Kroko/Speechcatcher's much larger transducer/
transformer models. Its config surface is therefore deliberately minimal:
no CPU-thread option (``vosk.KaldiRecognizer`` exposes no thread-count
knob to tune -- Kaldi's own decoding graph search here is single-threaded
by design) and no beam-size/quality knob (the small model ships one fixed
decoding graph) -- adding tuning options with nothing real behind them
would only complicate the UI for no behavioral benefit.

Hotwords / HA vocabulary (context biasing) -- ``supports_dynamic_vocabulary
= False``, a deliberate design decision, NOT an unverified gap: Vosk's
``KaldiRecognizer`` DOES expose a grammar mechanism
(``KaldiRecognizer(model, sample_rate, grammar_json_string)`` /
``SetGrammar()``, verified directly from the installed wheel's own
``vosk/__init__.py`` source and from upstream's own Ruby binding docs,
which are the clearest primary documentation of this feature's semantics).
However, that grammar is a HARD CLOSED-SET restriction on the decoding
graph, not a soft bias/boost: passing a JSON array of phrases makes the
recognizer capable of recognizing *only* those phrases (plus a literal
``"[unk]"`` catch-all token for anything else, which does not recover the
actual words spoken). Wiring the full, open-ended Home Assistant
vocabulary (area/device/entity names *plus* every other word a user might
say in a natural sentence, e.g. "mach", "bitte", "aus", "wie", "warm",
...) into this mechanism would not bias recognition towards HA terms the
way Moonshine's ``set_keyterms()`` or Kroko's sherpa-onnx ``hotwords`` do
-- it would instead break free-form recognition of anything not in that
exact closed list, which is a strictly worse user experience than no
biasing at all. This is therefore honestly reported as
``supports_hotwords=False`` / ``supports_dynamic_vocabulary=False``, and
:meth:`VoskSttEngine.set_keyterms` logs "HA vocabulary not supported by
engine vosk_german" and returns ``([], True)`` -- exactly like
Speechcatcher -- rather than silently degrading general recognition
quality to approximate a capability this add-on does not actually offer
for this engine.

Endpointing / "final": Wyoming's own audio-start/audio-chunk/audio-stop
sequence already delimits exactly one utterance per session (identical to
every other engine here) -- ``finalize()`` calls ``FinalResult()``, which
flushes any buffered audio and returns the best final hypothesis; the
recognizer's internal state does not need to be reused across utterances,
so a fresh ``KaldiRecognizer`` is created per session (cheap: it is a thin
handle onto the one shared, already-loaded ``vosk.Model``, not a full model
reload -- verified from ``vosk.KaldiRecognizer.__init__``'s own source,
which only calls ``vosk_recognizer_new(model_handle, sample_rate)``).
"""

import asyncio
import json
import logging
import time
from typing import Any

import numpy as np

from app.audio import float32_to_pcm_int16
from app.stt_engine import SttCapabilities, SttSession
from app.vosk_model import VOSK_MODEL_METADATA, VOSK_SAMPLE_RATE, resolve_vosk_model_dir

_LOGGER = logging.getLogger(__name__)

ENGINE_ID = "vosk_german"

_ATTRIBUTION_NAME = "AlphaCephei (Vosk)"
_ATTRIBUTION_URL = "https://alphacephei.com/vosk/"


class VoskSttSession:
    """One isolated Vosk transcription session (one Wyoming connection's
    one utterance).

    Structurally matches app/kroko_engine.py's KrokoSttSession /
    app/speechcatcher_engine.py's SpeechcatcherSttSession (SttSession
    protocol). Unlike those two, a Vosk ``KaldiRecognizer`` is a cheap,
    per-session handle (not a shared per-model stream/decoder state), so
    ``start()`` creates a brand-new recognizer against the one shared,
    already-loaded ``vosk.Model`` -- there is no cross-session state to
    reset or leak. Every native call is still serialized through the
    shared ``lock`` for the same "correctness over unproven parallelism"
    reason as every other engine here: Vosk's own docs do not document
    concurrent native calls FROM DIFFERENT recognizer objects sharing one
    ``Model`` as safe, only that a single ``KaldiRecognizer`` itself is not
    meant to be driven from multiple threads at once.
    """

    def __init__(
        self,
        model: Any,
        lock: asyncio.Lock | None = None,
    ) -> None:
        self._model = model
        self._lock = lock if lock is not None else asyncio.Lock()
        self._recognizer: Any = None
        self.audio_duration_seconds = 0.0
        self._add_audio_time_seconds = 0.0
        self.finalize_time_seconds = 0.0
        self._add_audio_chunk_count = 0
        self._add_audio_compute_max_seconds = 0.0
        self._closed = False

    @property
    def inference_time_seconds(self) -> float:
        return self._add_audio_time_seconds + self.finalize_time_seconds

    @property
    def add_audio_chunk_count(self) -> int:
        return self._add_audio_chunk_count

    @property
    def add_audio_compute_total_seconds(self) -> float:
        return self._add_audio_time_seconds

    @property
    def add_audio_compute_max_seconds(self) -> float:
        return self._add_audio_compute_max_seconds

    @property
    def average_chunk_compute_seconds(self) -> float:
        if self._add_audio_chunk_count == 0:
            return 0.0
        return self._add_audio_time_seconds / self._add_audio_chunk_count

    async def start(self) -> None:
        import vosk

        async with self._lock:
            self._recognizer = await asyncio.to_thread(
                vosk.KaldiRecognizer, self._model, VOSK_SAMPLE_RATE
            )

    async def add_audio(self, samples: list[float], sample_rate: int = 16000) -> None:
        assert self._recognizer is not None, "add_audio() called before start()"
        assert sample_rate == VOSK_SAMPLE_RATE, (
            f"Vosk model expects {VOSK_SAMPLE_RATE}Hz audio, got {sample_rate}Hz"
        )
        pcm_bytes = float32_to_pcm_int16(np.asarray(samples, dtype=np.float32))

        def _work() -> None:
            self._recognizer.AcceptWaveform(pcm_bytes)

        async with self._lock:
            started = time.monotonic()
            await asyncio.to_thread(_work)
            elapsed = time.monotonic() - started
            self._add_audio_time_seconds += elapsed
            self._add_audio_chunk_count += 1
            if elapsed > self._add_audio_compute_max_seconds:
                self._add_audio_compute_max_seconds = elapsed
        self.audio_duration_seconds += len(samples) / sample_rate

    async def finalize(self) -> str:
        assert self._recognizer is not None, "finalize() called before start()"

        def _work() -> str:
            raw = self._recognizer.FinalResult()
            try:
                return str(json.loads(raw).get("text", ""))
            except (json.JSONDecodeError, AttributeError):
                return ""

        async with self._lock:
            started = time.monotonic()
            text = await asyncio.to_thread(_work)
            self.finalize_time_seconds = time.monotonic() - started
        return text.strip()

    def close(self) -> None:
        # vosk.KaldiRecognizer has no explicit native close/free call in
        # its Python API beyond __del__ (which runs vosk_recognizer_free()
        # automatically when garbage collected) -- dropping the reference
        # is sufficient. Idempotent: safe to call more than once.
        self._recognizer = None
        self._closed = True


class VoskSttEngine:
    """Wraps one shared ``vosk.Model``, loaded once at startup."""

    def __init__(self, model: Any) -> None:
        self._model = model

    @property
    def engine_id(self) -> str:
        return ENGINE_ID

    @property
    def capabilities(self) -> SttCapabilities:
        return SttCapabilities(
            supports_streaming=True,
            supports_hotwords=False,
            supports_dynamic_vocabulary=False,
            # Vosk's KaldiRecognizer genuinely exposes PartialResult(), but
            # exactly like every other engine here (see
            # app/stt_engine.py's SttCapabilities docstring),
            # app/handler.py never sends a Wyoming partial-transcript event
            # for ANY engine -- reported as False to honestly reflect what
            # this add-on's Wyoming surface actually does today, not what
            # the underlying library could theoretically support.
            supports_partial_results=False,
        )

    @property
    def program_name(self) -> str:
        return "homeintent-vosk-german"

    @property
    def model_display_name(self) -> str:
        return str(VOSK_MODEL_METADATA["name"])

    @property
    def description(self) -> str:
        return str(VOSK_MODEL_METADATA["description"])

    @property
    def attribution_name(self) -> str:
        return _ATTRIBUTION_NAME

    @property
    def attribution_url(self) -> str:
        return _ATTRIBUTION_URL

    def create_session(self, lock: object | None = None) -> SttSession:
        assert lock is None or isinstance(lock, asyncio.Lock)
        return VoskSttSession(self._model, lock=lock)

    def set_keyterms(
        self, terms: list[str], fallback_terms: list[str] | None = None
    ) -> tuple[list[str], bool]:
        """Vosk's only context-biasing mechanism (grammar) is a hard
        closed-set restriction unsuitable as this add-on's HA-vocabulary
        biasing mechanism (see this module's docstring) -- never pretends
        to apply keyterms. Always succeeds trivially
        (``applied_succeeded=True``): there is nothing that can
        structurally fail here."""
        if terms:
            _LOGGER.info(
                "HA vocabulary not supported by engine %s (%d term(s) ignored)",
                ENGINE_ID,
                len(terms),
            )
        return [], True


def load_vosk_engine(cache_dir: Any = None) -> VoskSttEngine:
    """Resolve/download the Vosk German model and construct the engine.

    A local import (only ever needed when stt_engine=vosk_german, exactly
    like app/kroko_engine.py's local ``import sherpa_onnx``).

    Raises:
        VoskModelDownloadError: see resolve_vosk_model_dir().
        Exception: if vosk itself fails to load the resolved model
            directory (e.g. a corrupted/incompatible model).
    """
    import vosk

    vosk.SetLogLevel(-1)  # silence Kaldi's own verbose native stderr logging

    model_dir = resolve_vosk_model_dir(cache_dir=cache_dir)
    _LOGGER.info("Loading Vosk German ASR model from %s", model_dir)
    model = vosk.Model(str(model_dir))
    _LOGGER.info("Vosk German ASR model ready (sample_rate=%dHz)", VOSK_SAMPLE_RATE)
    return VoskSttEngine(model)
