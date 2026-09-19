"""Speechcatcher German streaming ASR model resolution and loading.

Speechcatcher (https://github.com/speechcatcher-asr/speechcatcher, MIT
license, verified directly against a clone of the upstream repository at
commit ``1f647a43105b7167b6a47e7b2af2ef3c4ff53a2b`` while building this
add-on) is NOT published on PyPI -- ``pip index versions speechcatcher``
returns no result, confirmed from this add-on's own development sandbox.
It is installed straight from that pinned git commit (see
requirements-runtime.txt / Dockerfile), matching speechcatcher's own
``pyproject.toml``, which itself pulls two more git dependencies:

- ``espnet_streaming_decoder`` (https://github.com/speechcatcher-asr/
  espnet_streaming_decoder, MIT) -- upstream's own words: "Espnet streaming
  ASR decoder with smaller footprint and fewer requirements, extracted from
  ESPnet". Verified from its own requirements.txt that it depends on numpy/
  torch/torchaudio/scipy/librosa/... but NOT on the full ``espnet`` PyPI
  package -- it vendors the small subset of espnet/espnet2 modules it needs
  directly inside its own package tree. This is the "lighter decoder"
  referred to elsewhere in this add-on's docs, relative to a full ESPnet
  install, while still being the CURRENT, STABLE, DEFAULT decoder path
  (``--decoder espnet`` is speechcatcher's own CLI default).
- ``espnet_model_zoo`` (https://github.com/speechcatcher-asr/
  espnet_model_zoo, Apache-2.0, a fork of espnet/espnet_model_zoo) -- used
  only for its ``ModelDownloader``, which resolves/downloads a model tag
  via ``huggingface_hub.snapshot_download()`` under the hood (verified from
  its own downloader.py source) -- the SAME underlying, atomic,
  resume-capable caching mechanism this add-on's Pocket TTS and Kokoro ONNX
  loaders already rely on via huggingface_hub (already pinned in
  requirements-runtime.txt), not a new download mechanism this add-on
  invented.

Decoder choice -- native vs. ESPnet-family: speechcatcher 0.5.0 (10/2025)
added an experimental, dependency-free "native" streaming decoder
(``speechcatcher.speech2text_streaming.Speech2TextStreaming``, pure
PyTorch, no espnet_streaming_decoder needed) alongside the original
ESPnet-family one. Upstream's own README states: '--decoder {native,espnet}
... "espnet" (default) or "native" (experimental)'. Per this add-on's own
policy (prefer a lighter decoder only if it is the officially, stably
supported one), the native decoder is NOT used here -- it is explicitly
documented as experimental by the project that ships it. This module always
loads via ``decoder_impl="espnet"``, i.e. the smaller-footprint
espnet_streaming_decoder fork described above, which IS speechcatcher's own
current, stable default.

Model tags: speechcatcher's own ``speechcatcher.speechcatcher.tags`` dict
(verified from source) maps:

    "de_streaming_transformer_m" ->
        "speechcatcher/speechcatcher_german_espnet_streaming_transformer_13k_train_size_m_raw_de_bpe1024"
    "de_streaming_transformer_l" ->
        "speechcatcher/speechcatcher_german_espnet_streaming_transformer_13k_train_size_l_raw_de_bpe1024"

Both are Hugging Face Hub repo ids under the ``speechcatcher`` org, resolved
by ``ModelDownloader.download_and_unpack()`` (see above). This add-on's own
network-restricted development sandbox could not reach huggingface.co
(confirmed: a plain ``git clone`` of these two HF repos returned a blocked
CONNECT tunnel), so the exact checkpoint file sizes and the model
checkpoints' OWN license (as opposed to speechcatcher's own code license,
MIT, which IS verified) could not be independently confirmed from here --
this is called out explicitly rather than guessed at, see this add-on's own
docs/CHANGELOG.

We deliberately reuse ``speechcatcher.speechcatcher.load_model()`` itself
(rather than re-implementing model download + Speech2TextStreaming
construction ourselves) -- it already does exactly the resolve/download/
build-decoder sequence above, is upstream's own tested code path (the same
one ``speechcatcher_server.py`` calls, see app/speechcatcher_engine.py's
module docstring for why this add-on uses that object directly instead of
speechcatcher_server's websocket layer), and re-implementing it would only
add a second, drifting copy of upstream's own model-loading logic.
"""

import contextlib
import importlib
import io
import logging
import os
from pathlib import Path
from typing import Any

_LOGGER = logging.getLogger(__name__)

DEFAULT_SPEECHCATCHER_CACHE_DIR = Path(
    os.environ.get("SPEECHCATCHER_CACHE", "/data/models/speechcatcher")
)

# Our own short, config.yaml-facing engine ids -> speechcatcher's own short
# tag names (as used by its `tags` dict / --model CLI argument). Only the
# German "m" (medium) and "l" (large) streaming transformer models are
# exposed -- "xl" and the Spanish/English models are out of scope for this
# German-only add-on (see config.yaml's `language: list(de)`).
SPEECHCATCHER_ENGINE_TAGS = {
    "speechcatcher_m": "de_streaming_transformer_m",
    "speechcatcher_l": "de_streaming_transformer_l",
}

SPEECHCATCHER_MODEL_METADATA = {
    "speechcatcher_m": {
        "name": "speechcatcher-de-streaming-transformer-m",
        "description": "German streaming Transformer ASR, medium (Speechcatcher, ESPnet-family decoder)",
    },
    "speechcatcher_l": {
        "name": "speechcatcher-de-streaming-transformer-l",
        "description": "German streaming Transformer ASR, large (Speechcatcher, ESPnet-family decoder)",
    },
}

SPEECHCATCHER_SAMPLE_RATE = 16000


class SpeechcatcherModelDownloadError(Exception):
    """Raised when the Speechcatcher model could not be resolved/downloaded
    or the speechcatcher package itself could not be imported.

    A distinct exception type (mirrors KrokoModelDownloadError /
    KokoroModelDownloadError) so app/__main__.py can log a clear, specific
    startup error instead of a generic failure -- selecting
    ``stt_engine: speechcatcher_m``/``speechcatcher_l`` and having this fail
    must be a visible configuration/network/dependency error, never a
    silent fallback to another engine.
    """


def load_speechcatcher_model(
    engine_id: str,
    cache_dir: Path | None = None,
    beam_size: int = 5,
) -> Any:
    """Resolve/download the Speechcatcher model for ``engine_id`` and
    return upstream's own ready-to-use ``Speech2TextStreaming`` instance.

    ``engine_id`` is "speechcatcher_m" or "speechcatcher_l" (this add-on's
    own ``stt_engine`` option values). The heavy ``speechcatcher`` package
    (and its own git dependencies) is imported lazily, only when this
    function actually runs -- exactly like app/kroko_engine.py's local
    ``import sherpa_onnx`` -- so importing this module, or any other engine
    module, never requires speechcatcher to be installed unless
    ``stt_engine`` actually selects it.

    Uses ``decoder_impl="espnet"`` (speechcatcher's current, stable
    default -- see this module's docstring) and always ``device="cpu"``
    (this add-on's whole runtime is CPU-only; see requirements-runtime.txt/
    Dockerfile -- Torch itself never triggers a CUDA-only code path here
    because no CUDA build of Torch is installed at all).

    Model download progress/informational ``print()`` calls made by
    upstream's own ``load_model()``/``show_model_info()`` (not this add-on's
    ``logging`` module) are redirected away from stdout so they do not
    interleave with this add-on's structured log lines; an equivalent
    ``_LOGGER.info`` line is emitted here instead.

    Raises:
        SpeechcatcherModelDownloadError: if the speechcatcher package
            cannot be imported, or if model resolution/download/loading
            fails for any reason (network, corrupt cache, incompatible
            checkpoint, ...). Never returns a partial/unusable result.
    """
    if engine_id not in SPEECHCATCHER_ENGINE_TAGS:
        raise SpeechcatcherModelDownloadError(
            f"Unknown Speechcatcher engine id {engine_id!r}, expected one of "
            f"{sorted(SPEECHCATCHER_ENGINE_TAGS)}"
        )
    short_tag = SPEECHCATCHER_ENGINE_TAGS[engine_id]
    resolved_cache_dir = cache_dir if cache_dir is not None else DEFAULT_SPEECHCATCHER_CACHE_DIR
    resolved_cache_dir.mkdir(parents=True, exist_ok=True)

    try:
        speechcatcher_cli: Any = importlib.import_module("speechcatcher.speechcatcher")
    except Exception as err:
        raise SpeechcatcherModelDownloadError(
            f"Failed to import the 'speechcatcher' package (is it installed?): "
            f"{type(err).__name__}: {err}"
        ) from err

    full_tag = speechcatcher_cli.tags.get(short_tag)
    if full_tag is None:
        raise SpeechcatcherModelDownloadError(
            f"speechcatcher's own tags table has no entry for {short_tag!r} -- "
            f"upstream may have renamed/removed this model"
        )

    _LOGGER.info(
        "Loading Speechcatcher model: engine=%s tag=%s cache_dir=%s beam_size=%d",
        engine_id,
        full_tag,
        resolved_cache_dir,
        beam_size,
    )
    try:
        with contextlib.redirect_stdout(io.StringIO()):
            model = speechcatcher_cli.load_model(
                tag=full_tag,
                device="cpu",
                beam_size=beam_size,
                quiet=True,
                cache_dir=str(resolved_cache_dir),
                decoder_impl="espnet",
            )
    except Exception as err:
        raise SpeechcatcherModelDownloadError(
            f"Failed to download/load Speechcatcher model {full_tag!r}: {type(err).__name__}: {err}"
        ) from err

    _LOGGER.info("Speechcatcher model ready (sample_rate=%dHz)", SPEECHCATCHER_SAMPLE_RATE)
    return model
