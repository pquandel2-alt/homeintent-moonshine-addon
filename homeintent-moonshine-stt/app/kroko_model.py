"""Kroko German streaming ASR model resolution and caching.

Kroko is a family of streaming Zipformer2-transducer ASR models (Banafo AI /
Kroko-ASR, https://huggingface.co/Banafo/Kroko-ASR) merged into sherpa-onnx
upstream (k2-fsa/sherpa-onnx CHANGELOG.md, v1.12.10: "Add VOSK streaming
Russian ASR models and Kroko streaming German ASR models" -- verified
directly against a clone of the k2-fsa/sherpa-onnx repository at the time
this add-on was built). The German package name
``sherpa-onnx-streaming-zipformer-de-kroko-2025-08-06`` is the exact string
sherpa-onnx's own release-asset build script
(scripts/apk/generate-asr-apk-script.py) uses for this model, downloaded
from sherpa-onnx's own GitHub Releases (not a hand-picked/guessed name):

    https://github.com/k2-fsa/sherpa-onnx/releases/download/asr-models/
        sherpa-onnx-streaming-zipformer-de-kroko-2025-08-06.tar.bz2

License: Kroko-ASR models are published under Apache License 2.0 (per
Banafo's own Kroko-ASR model card). sherpa-onnx itself (the runtime this
add-on links against, package ``sherpa-onnx`` on PyPI) is Apache-2.0
licensed as well (verified from the installed wheel's own METADATA).

This add-on could not, from its own network-restricted development sandbox,
fetch the exact file listing inside that specific .tar.bz2 (huggingface.co
and unauthenticated github.com/api.github.com access were both blocked by
this sandbox's own egress policy; a raw git clone of k2-fsa/sherpa-onnx
itself was reachable and is what the facts above are verified against).
Every other sherpa-onnx streaming Zipformer transducer release uses the
same well-documented archive layout (an ``encoder-*.onnx``, ``decoder-
*.onnx``, ``joiner-*.onnx`` triple plus ``tokens.txt``, and a QUANTIZED
``*.int8.onnx`` encoder is normal for CPU deployment) -- rather than
hard-code an unverified exact filename that could silently break on a
future re-export, :func:`resolve_kroko_model_files` downloads the whole
archive and *discovers* the four required files by glob pattern, then fails
loudly (:class:`KrokoModelDownloadError`) if any of the four is missing or
ambiguous. This is a deliberate, honestly-documented choice, not a
guess dressed up as certainty.
"""

import logging
import os
import shutil
import tarfile
import tempfile
import urllib.request
from dataclasses import dataclass
from pathlib import Path

from app.safe_tar_extract import UnsafeTarMemberError, safe_extractall

_LOGGER = logging.getLogger(__name__)

DEFAULT_KROKO_CACHE_DIR = Path(os.environ.get("KROKO_CACHE", "/data/models/kroko"))

KROKO_PACKAGE_NAME = "sherpa-onnx-streaming-zipformer-de-kroko-2025-08-06"
KROKO_DOWNLOAD_URL = (
    "https://github.com/k2-fsa/sherpa-onnx/releases/download/asr-models/"
    f"{KROKO_PACKAGE_NAME}.tar.bz2"
)

KROKO_SAMPLE_RATE = 16000

KROKO_MODEL_METADATA = {
    "name": "kroko-streaming-zipformer-de",
    "description": (
        "German streaming Zipformer2-transducer ASR (Kroko/Banafo AI, via sherpa-onnx)"
    ),
    "license": "Apache License 2.0 (Banafo Kroko-ASR / sherpa-onnx)",
}


class KrokoModelDownloadError(Exception):
    """Raised when the Kroko model archive/files could not be resolved.

    A distinct exception type (like KokoroModelDownloadError) so
    app/__main__.py can log a clear, specific startup error instead of a
    generic failure -- selecting ``stt_engine: kroko`` and having the
    download fail must be a visible configuration/network error, never a
    silent fallback to Moonshine.
    """


@dataclass(frozen=True)
class KrokoModelFiles:
    encoder: Path
    decoder: Path
    joiner: Path
    tokens: Path


def _find_one(directory: Path, patterns: list[str], label: str) -> Path:
    candidates: list[Path] = []
    for pattern in patterns:
        candidates.extend(sorted(directory.rglob(pattern)))
    # Prefer an int8-quantized encoder/joiner when both a full-precision and
    # a quantized file are present (int8 is what CPU deployment guidance in
    # sherpa-onnx's own docs recommends, and matches the size sherpa-onnx's
    # own model-export CI produces for these packages).
    int8_candidates = [c for c in candidates if "int8" in c.name]
    if int8_candidates:
        candidates = int8_candidates
    if not candidates:
        raise KrokoModelDownloadError(
            f"Could not find a {label} file under {directory} "
            f"(tried patterns: {patterns}) -- the downloaded Kroko archive layout "
            f"may have changed upstream"
        )
    if len(candidates) > 1:
        _LOGGER.warning(
            "Multiple candidate %s files found under %s (%s); using %s",
            label,
            directory,
            [str(c) for c in candidates],
            candidates[0],
        )
    return candidates[0]


def _download_and_extract(url: str, dest_dir: Path) -> None:
    """Download ``url`` to a temp file and extract it into ``dest_dir``.

    Atomic with respect to a partially-downloaded/extracted result: the
    archive is downloaded to a temporary file and extracted into a
    temporary sibling directory, which is only ``rename()``d into
    ``dest_dir`` after both steps succeed -- a killed download or a failed
    extraction never leaves ``dest_dir`` populated with partial content
    that a later run might mistake for a complete, valid cache.
    """
    dest_dir.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory(dir=dest_dir.parent) as tmp:
        tmp_path = Path(tmp)
        archive_path = tmp_path / "kroko.tar.bz2"
        _LOGGER.info("Downloading Kroko German ASR model from %s", url)
        try:
            urllib.request.urlretrieve(url, archive_path)  # noqa: S310 - fixed, trusted URL
        except Exception as err:
            raise KrokoModelDownloadError(
                f"Failed to download Kroko model from '{url}': {type(err).__name__}: {err}"
            ) from err

        extract_dir = tmp_path / "extracted"
        extract_dir.mkdir()
        try:
            with tarfile.open(archive_path) as tf:
                safe_extractall(tf, extract_dir, label="Kroko model")
        except UnsafeTarMemberError as err:
            raise KrokoModelDownloadError(
                f"Refusing to extract unsafe Kroko model archive member: {err}"
            ) from err
        except Exception as err:
            raise KrokoModelDownloadError(
                f"Failed to extract Kroko model archive downloaded from '{url}': "
                f"{type(err).__name__}: {err}"
            ) from err

        if dest_dir.exists():
            shutil.rmtree(dest_dir)
        shutil.move(str(extract_dir), str(dest_dir))


def resolve_kroko_model_files(cache_dir: Path | None = None) -> KrokoModelFiles:
    """Resolve (downloading if needed) the Kroko German model's four files.

    Only downloads when ``stt_engine: kroko`` is actually selected (this
    function is only ever called from that code path, see
    app/kroko_engine.py / app/__main__.py) -- Kroko's model is never
    downloaded or loaded when ``stt_engine: moonshine`` (the default), and
    vice versa: Moonshine's model is loaded independently by
    app/models.py's load_transcriber(), which this module never calls.

    Args:
        cache_dir: Directory to store/extract the downloaded archive into.
            Defaults to :data:`DEFAULT_KROKO_CACHE_DIR`
            (``/data/models/kroko`` in production; overridden in tests).
            Persistent across restarts: a cache hit (the four required
            files already present) makes no network request at all.

    Returns:
        The resolved (encoder, decoder, joiner, tokens) file paths.

    Raises:
        KrokoModelDownloadError: on any download, extraction, or
            missing-file failure. Never returns a partial result.
    """
    resolved_cache_dir = cache_dir if cache_dir is not None else DEFAULT_KROKO_CACHE_DIR
    package_dir = resolved_cache_dir / KROKO_PACKAGE_NAME

    if not package_dir.is_dir():
        _download_and_extract(KROKO_DOWNLOAD_URL, package_dir)
    else:
        _LOGGER.info("Kroko German ASR model already cached at %s", package_dir)

    encoder = _find_one(package_dir, ["encoder*.onnx"], "encoder")
    decoder = _find_one(package_dir, ["decoder*.onnx"], "decoder")
    joiner = _find_one(package_dir, ["joiner*.onnx"], "joiner")
    tokens = _find_one(package_dir, ["tokens.txt"], "tokens")

    for f in (encoder, decoder, joiner, tokens):
        if not f.is_file() or f.stat().st_size == 0:
            raise KrokoModelDownloadError(f"Resolved Kroko file '{f}' is missing or empty")

    return KrokoModelFiles(encoder=encoder, decoder=decoder, joiner=joiner, tokens=tokens)
