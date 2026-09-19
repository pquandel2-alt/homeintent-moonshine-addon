"""Speechcatcher SttEngine: real, in-process streaming German ASR.

Wraps upstream's own ``Speech2TextStreaming`` object (as constructed by
``speechcatcher.speechcatcher.load_model(..., decoder_impl="espnet")``, see
app/speechcatcher_model.py's module docstring for exactly which decoder
this is and why) directly in-process -- there is no subprocess, no separate
container, and no websocket server involved.

Why not ``speechcatcher_server``: upstream ships a ``speechcatcher_server``
entry point that runs a websocket server for browser/remote clients. Its
own source (``speechcatcher_server.py``) shows it is a thin wrapper: it
calls the exact same ``load_model()`` this module calls, then feeds audio
into the returned object with ``speech2text(speech=data, is_final=...)`` --
i.e. it wraps the same in-process Python object this module uses directly,
adding only a websocket transport and (for browser clients) webm decoding
this add-on has no use for, since Wyoming already delivers raw 16kHz PCM.
Spinning up that server as a second process and talking to it over a local
websocket would add a redundant network hop, a redundant serialization
format, and a second process to supervise for zero behavioral benefit --
so this add-on imports and drives the same underlying object directly,
exactly as the user-facing task for this feature required.

Streaming granularity -- what is real here: ``Speech2TextStreaming.__call__``
performs genuine incremental decoding, not one-shot whole-utterance batch
transcription. Each call feeds newly arrived raw audio samples; internally
it buffers only the small STFT-window remainder needed for frame
continuity (see its own ``apply_frontend()``), runs the contextual-block
streaming encoder incrementally, and advances beam search block by block
(K. Tsunoo et al., "Streaming Transformer ASR with Blockwise Synchronous
Beam Search", https://arxiv.org/abs/2006.14941 -- cited directly in
upstream's own class docstring). This is chunk/block-granular streaming
(the model's own configured block size determines how much new audio is
needed before a block advances), not sample-by-sample, but it genuinely
streams as audio arrives -- audio is never buffered whole and decoded once
at the end, matching this add-on's other engines (Moonshine, Kroko).

Hotwords / HA vocabulary (context biasing): verified directly from
upstream's source (espnet_streaming_decoder's ``Speech2TextStreaming``,
its ``BatchBeamSearchOnline``/scorer classes, and speechcatcher's own
native decoder and encoder modules) that "contextual" in this codebase
refers exclusively to the *contextual block* streaming ENCODER architecture
(chunked attention over blocks of frames), not to keyword/vocabulary
context-biasing. There is no hotword list, bias list, or vocabulary
injection API anywhere in this dependency chain. This is honestly reported
as ``supports_hotwords=False`` / ``supports_dynamic_vocabulary=False`` --
:meth:`SpeechcatcherSttEngine.set_keyterms` logs "HA vocabulary not
supported by engine <engine_id>" and returns ``([], True)`` rather than
pretending to apply anything, exactly as app/stt_engine.py's own protocol
docstring requires of an engine without this capability.

Endpointing / "final": Wyoming's own audio-start/audio-chunk/audio-stop
sequence already delimits exactly one utterance per session (identical to
how Moonshine's and Kroko's sessions are used) -- ``finalize()`` below
calls ``speech2text(..., is_final=True)``, which upstream's own
implementation uses to flush all remaining buffered audio, assemble the
best hypothesis, and internally reset its own decoding state ready for the
next utterance. No artificial endpoint/VAD detection is layered on top.
"""

import asyncio
import logging
import time
from typing import Any

import numpy as np

from app.speechcatcher_model import (
    SPEECHCATCHER_MODEL_METADATA,
    load_speechcatcher_model,
)
from app.stt_engine import SttCapabilities, SttSession

_LOGGER = logging.getLogger(__name__)

_ATTRIBUTION_NAME = "Benjamin Milde / Speechcatcher"
_ATTRIBUTION_URL = "https://github.com/speechcatcher-asr/speechcatcher"


class SpeechcatcherSttSession:
    """One isolated Speechcatcher transcription session (one Wyoming
    connection's one utterance).

    Structurally matches app/kroko_engine.py's KrokoSttSession /
    app/streaming.py's MoonshineStreamingSession (SttSession protocol).
    Speechcatcher's ``Speech2TextStreaming`` keeps ALL of its streaming
    state (frontend buffer, encoder cache, beam search hypotheses) as
    plain instance attributes on one shared object, not on a per-call
    stream handle the way sherpa-onnx's OnlineStream does -- so, exactly
    like Moonshine's and Kroko's own sessions, every native call is
    serialized through the shared ``lock`` and only one session may be
    "in flight" against the shared model at a time. ``start()`` calls
    upstream's own ``reset()`` defensively before the first chunk, so a
    previous connection that dropped mid-utterance (without ever reaching
    ``finalize()``, whose ``is_final=True`` call already resets
    internally) can never leak state into the next session.
    """

    def __init__(
        self,
        speech2text: Any,
        lock: asyncio.Lock | None = None,
    ) -> None:
        self._speech2text = speech2text
        self._lock = lock if lock is not None else asyncio.Lock()
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
        async with self._lock:
            await asyncio.to_thread(self._speech2text.reset)

    async def add_audio(self, samples: list[float], sample_rate: int = 16000) -> None:
        samples_arr = np.asarray(samples, dtype=np.float32)

        def _work() -> None:
            # always_assemble_hyps=False: this add-on never sends a
            # Wyoming partial-transcript event (see app/stt_engine.py's
            # SttCapabilities.supports_partial_results docstring), so
            # assembling text on every intermediate chunk would only waste
            # CPU -- beam search itself still advances on every call
            # regardless of this flag (verified from
            # Speech2TextStreaming.__call__'s own source).
            self._speech2text(samples_arr, is_final=False, always_assemble_hyps=False)

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
        def _work() -> str:
            empty = np.zeros(0, dtype=np.float32)
            hyps = self._speech2text(empty, is_final=True, always_assemble_hyps=True)
            if not hyps:
                return ""
            best_text = hyps[0][0]
            return str(best_text) if best_text is not None else ""

        async with self._lock:
            started = time.monotonic()
            text = await asyncio.to_thread(_work)
            self.finalize_time_seconds = time.monotonic() - started
        return text.strip()

    def close(self) -> None:
        # Speech2TextStreaming has no explicit native close/free call --
        # dropping the reference to the shared model is wrong (it is
        # shared across sessions, see SpeechcatcherSttEngine), so this only
        # marks the session closed. Idempotent: safe to call more than
        # once.
        self._closed = True


class SpeechcatcherSttEngine:
    """Wraps one shared Speechcatcher ``Speech2TextStreaming`` model,
    loaded once at startup.

    ``engine_id`` is "speechcatcher_m" or "speechcatcher_l" -- the two
    distinct ``stt_engine`` dropdown values this add-on exposes. Both share
    this exact same engine/session implementation; only the model
    checkpoint (chosen via app/speechcatcher_model.py's
    ``SPEECHCATCHER_ENGINE_TAGS`` mapping, applied once at
    :func:`load_speechcatcher_engine` call time) differs -- there is no
    behavioral difference in this class between the two.
    """

    def __init__(self, speech2text: Any, engine_id: str) -> None:
        self._speech2text = speech2text
        self._engine_id = engine_id

    @property
    def engine_id(self) -> str:
        return self._engine_id

    @property
    def capabilities(self) -> SttCapabilities:
        return SttCapabilities(
            supports_streaming=True,
            supports_hotwords=False,
            supports_dynamic_vocabulary=False,
            supports_partial_results=False,
        )

    @property
    def program_name(self) -> str:
        return f"homeintent-{self._engine_id.replace('_', '-')}"

    @property
    def model_display_name(self) -> str:
        return str(SPEECHCATCHER_MODEL_METADATA[self._engine_id]["name"])

    @property
    def description(self) -> str:
        return str(SPEECHCATCHER_MODEL_METADATA[self._engine_id]["description"])

    @property
    def attribution_name(self) -> str:
        return _ATTRIBUTION_NAME

    @property
    def attribution_url(self) -> str:
        return _ATTRIBUTION_URL

    def create_session(self, lock: object | None = None) -> SttSession:
        assert lock is None or isinstance(lock, asyncio.Lock)
        return SpeechcatcherSttSession(self._speech2text, lock=lock)

    def set_keyterms(
        self, terms: list[str], fallback_terms: list[str] | None = None
    ) -> tuple[list[str], bool]:
        """Speechcatcher has no hotword/context-biasing API (see this
        module's docstring) -- never pretends to apply keyterms. Always
        succeeds trivially (``applied_succeeded=True``): there is nothing
        that can structurally fail here."""
        if terms:
            _LOGGER.info(
                "HA vocabulary not supported by engine %s (%d term(s) ignored)",
                self._engine_id,
                len(terms),
            )
        return [], True


def load_speechcatcher_engine(
    engine_id: str,
    cache_dir: Any = None,
    beam_size: int = 5,
    num_threads: int = 0,
) -> SpeechcatcherSttEngine:
    """Resolve/download the Speechcatcher model for ``engine_id`` and
    construct the engine.

    ``num_threads``: if > 0, sets Torch's own intra-op thread count before
    loading (see app/tts.py's ``load_tts_model`` for the identical,
    already-shipped pattern for Pocket TTS's own Torch runtime) -- 0 leaves
    Torch's own default untouched. A local import (only when this function
    actually runs, exactly like app/kroko_engine.py's local
    ``import sherpa_onnx``) so importing this module never requires Torch
    thread configuration unless ``stt_engine`` actually selects
    speechcatcher_m/l.

    Raises:
        SpeechcatcherModelDownloadError: see load_speechcatcher_model().
    """
    if num_threads > 0:
        import torch

        torch.set_num_threads(num_threads)
        _LOGGER.info("Speechcatcher: torch intra-op threads set to %d", num_threads)

    speech2text = load_speechcatcher_model(engine_id, cache_dir=cache_dir, beam_size=beam_size)
    return SpeechcatcherSttEngine(speech2text, engine_id)
