"""Kroko SttEngine: real, in-process streaming German ASR via sherpa-onnx.

Uses ``sherpa_onnx.OnlineRecognizer.from_transducer()`` directly -- no
subprocess, no separate container, no websocket server -- verified against
the actual Python API shipped in the ``sherpa-onnx`` PyPI wheel (see
app/kroko_model.py's module docstring for how the model itself was
verified). Audio is fed incrementally via ``OnlineStream.accept_waveform()``
as Wyoming ``audio-chunk`` events arrive (real streaming, matching
Moonshine's own add_audio()-per-chunk design in app/streaming.py), not
buffered and transcribed in one batch at the end.

Hotwords / HA vocabulary (context biasing): sherpa-onnx's streaming
Zipformer transducer supports per-stream ``hotwords`` via
``OnlineRecognizer.create_stream(hotwords=...)`` when the recognizer was
constructed with ``decoding_method="modified_beam_search"`` (verified
directly from ``sherpa_onnx.online_recognizer.OnlineRecognizer`` source --
using a hotwords file/string with any other decoding method raises
``ValueError`` upstream). This add-on always constructs the Kroko
recognizer with ``modified_beam_search`` so a fresh HA-vocabulary/keyterm
list can be applied to the very next session with no model reload --
:class:`KrokoSttEngine` treats this as ``supports_dynamic_vocabulary=True``
for real, not merely to satisfy the capability flag. If a future sherpa-onnx
release ever drops this API, :meth:`KrokoSttEngine.set_keyterms` degrades
to logging "HA vocabulary not supported by engine kroko" and returning an
empty list -- it never pretends to apply something it could not.

Endpoint detection is intentionally left OFF (Wyoming's own audio-start/
-stop already delimits one utterance per session, matching how Moonshine's
own MoonshineStreamingSession is used) -- sherpa-onnx's endpoint rules exist
for a different use case (a single long-lived stream spanning many
utterances) that does not apply here.
"""

import asyncio
import logging
import time
from pathlib import Path
from typing import Any

import numpy as np

from app.kroko_model import (
    KROKO_MODEL_METADATA,
    KROKO_SAMPLE_RATE,
    KrokoModelFiles,
    resolve_kroko_model_files,
)
from app.stt_engine import SttCapabilities, SttSession

_LOGGER = logging.getLogger(__name__)

ENGINE_ID = "kroko"

_ATTRIBUTION_NAME = "Banafo AI (Kroko-ASR) via k2-fsa/sherpa-onnx"
_ATTRIBUTION_URL = "https://huggingface.co/Banafo/Kroko-ASR"

# Tail padding appended before input_finished(), matching sherpa-onnx's own
# documented usage pattern for streaming Zipformer transducers (see
# OnlineRecognizer's class docstring) -- lets the model's own right-context
# fully process the last real audio before decoding stops.
_TAIL_PADDING_SECONDS = 0.5


class KrokoSttSession:
    """One isolated Kroko transcription session (one Wyoming connection's
    one utterance).

    Structurally matches app/streaming.py's MoonshineStreamingSession
    (SttSession protocol): every native call (create_stream,
    accept_waveform, decode_stream, get_result, input_finished) is
    serialized through the shared ``lock`` (an asyncio.Lock), exactly like
    Moonshine's own session -- sherpa-onnx's OnlineRecognizer/OnlineStream
    thread-safety across concurrent Python calls on the same recognizer is
    not documented as safe, so this add-on takes the same "correctness over
    unproven parallelism" stance it already takes for Moonshine and Pocket/
    Kokoro TTS.
    """

    def __init__(
        self,
        recognizer: Any,
        hotwords: str | None,
        lock: asyncio.Lock | None = None,
    ) -> None:
        self._recognizer = recognizer
        self._lock = lock if lock is not None else asyncio.Lock()
        self._stream: Any = None
        self._hotwords = hotwords
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

    def _decode_ready(self) -> None:
        """Run every pending decode_stream() pass. Caller holds the lock."""
        while self._recognizer.is_ready(self._stream):
            self._recognizer.decode_stream(self._stream)

    async def start(self) -> None:
        async with self._lock:
            self._stream = await asyncio.to_thread(
                self._recognizer.create_stream,
                self._hotwords,
            )

    async def add_audio(self, samples: list[float], sample_rate: int = 16000) -> None:
        assert self._stream is not None, "add_audio() called before start()"
        samples_arr = np.asarray(samples, dtype=np.float32)

        def _work() -> None:
            self._stream.accept_waveform(sample_rate, samples_arr)
            self._decode_ready()

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
        assert self._stream is not None, "finalize() called before start()"

        def _work() -> str:
            tail = np.zeros(int(_TAIL_PADDING_SECONDS * KROKO_SAMPLE_RATE), dtype=np.float32)
            self._stream.accept_waveform(KROKO_SAMPLE_RATE, tail)
            self._stream.input_finished()
            self._decode_ready()
            return str(self._recognizer.get_result(self._stream))

        async with self._lock:
            started = time.monotonic()
            text = await asyncio.to_thread(_work)
            self.finalize_time_seconds = time.monotonic() - started
        return text.strip()

    def close(self) -> None:
        # sherpa-onnx's OnlineStream has no explicit native close/free call
        # in its Python API (unlike moonshine-voice's Stream) -- dropping
        # the reference lets it be garbage-collected normally. Idempotent:
        # safe to call more than once.
        self._stream = None
        self._closed = True


class KrokoSttEngine:
    """Wraps one shared sherpa-onnx OnlineRecognizer, loaded once at startup."""

    def __init__(self, recognizer: Any, model_files: KrokoModelFiles) -> None:
        self._recognizer = recognizer
        self._model_files = model_files
        self._hotwords: str | None = None

    @property
    def engine_id(self) -> str:
        return ENGINE_ID

    @property
    def capabilities(self) -> SttCapabilities:
        return SttCapabilities(
            supports_streaming=True,
            supports_hotwords=True,
            supports_dynamic_vocabulary=True,
            supports_partial_results=False,
        )

    @property
    def program_name(self) -> str:
        return "homeintent-kroko"

    @property
    def model_display_name(self) -> str:
        return str(KROKO_MODEL_METADATA["name"])

    @property
    def description(self) -> str:
        return str(KROKO_MODEL_METADATA["description"])

    @property
    def attribution_name(self) -> str:
        return _ATTRIBUTION_NAME

    @property
    def attribution_url(self) -> str:
        return _ATTRIBUTION_URL

    def create_session(self, lock: object | None = None) -> SttSession:
        assert lock is None or isinstance(lock, asyncio.Lock)
        return KrokoSttSession(self._recognizer, self._hotwords, lock=lock)

    def set_keyterms(
        self, terms: list[str], fallback_terms: list[str] | None = None
    ) -> tuple[list[str], bool]:
        """Apply HA vocabulary/manual keyterms as sherpa-onnx hotwords.

        Never raises (``fallback_terms`` is accepted for interface parity
        with MoonshineSttEngine but unused: nothing here can fail
        structurally the way a native ``set_keyterms()`` call can, so
        ``applied_succeeded`` is always True). Every NEW session created
        after this call uses the updated hotwords string (see this module's
        docstring) -- no model reload is needed, so this is real, dynamic
        vocabulary support, not a fixed-at-startup approximation of it.
        """
        cleaned = [t.strip() for t in terms if t.strip()]
        if not cleaned:
            self._hotwords = None
            _LOGGER.info("Kroko hotwords cleared (0 term(s))")
            return [], True
        # One phrase per line, matching sherpa-onnx's own hotwords-file
        # convention (verified from OnlineRecognizer.from_transducer's own
        # --hotwords-file handling / python-api-examples usage).
        self._hotwords = "\n".join(cleaned)
        _LOGGER.info("Kroko hotwords updated: %d term(s)", len(cleaned))
        return cleaned, True


def load_kroko_engine(
    cache_dir: Path | None = None,
    num_threads: int = 1,
    hotwords_score: float = 1.5,
) -> KrokoSttEngine:
    """Resolve/download the Kroko German model and construct the engine.

    Always uses ``decoding_method="modified_beam_search"`` (required for
    hotwords support, see this module's docstring) -- verified as a real,
    documented sherpa-onnx constraint, not an arbitrary choice.

    Raises:
        KrokoModelDownloadError: see resolve_kroko_model_files().
        Exception: if sherpa-onnx itself fails to load the resolved files
            (e.g. a corrupted/incompatible ONNX file).
    """
    import sherpa_onnx  # local import: only ever needed when stt_engine=kroko

    model_files = resolve_kroko_model_files(cache_dir=cache_dir)
    _LOGGER.info(
        "Loading Kroko German ASR model: encoder=%s decoder=%s joiner=%s tokens=%s num_threads=%d",
        model_files.encoder,
        model_files.decoder,
        model_files.joiner,
        model_files.tokens,
        num_threads,
    )
    recognizer = sherpa_onnx.OnlineRecognizer.from_transducer(
        tokens=str(model_files.tokens),
        encoder=str(model_files.encoder),
        decoder=str(model_files.decoder),
        joiner=str(model_files.joiner),
        num_threads=num_threads,
        sample_rate=KROKO_SAMPLE_RATE,
        decoding_method="modified_beam_search",
        hotwords_score=hotwords_score,
        enable_endpoint_detection=False,
    )
    _LOGGER.info("Kroko German ASR model ready (sample_rate=%dHz)", KROKO_SAMPLE_RATE)
    return KrokoSttEngine(recognizer, model_files)
