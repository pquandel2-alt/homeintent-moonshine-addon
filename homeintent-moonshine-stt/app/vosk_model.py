"""Vosk German ASR model resolution and caching.

Vosk (https://alphacephei.com/vosk/, https://github.com/alphacep/vosk-api)
is a Kaldi-based offline speech recognition toolkit. Its Python package
(``vosk`` on PyPI) ships a self-contained native library (Kaldi's C++ core
via cffi) -- no CUDA/GPU dependency (verified: ``pip show vosk`` lists only
``cffi``, ``requests``, ``srt``, ``tqdm``, ``websockets`` as dependencies,
none of them GPU-related; the ``GpuInit``/``GpuThreadInit`` calls the
Python API exposes are for an optional, separately-built server variant
this add-on never uses -- the plain PyPI wheel is CPU-only).

Model: this add-on uses ``vosk-model-small-de-0.15`` (45MB, Apache License
2.0), verified directly from the official model catalogue
(alphacep/vosk-space, ``models.md``, "German" section -- this repository
mirrors https://alphacephei.com/vosk/models, itself unreachable from this
add-on's own development sandbox, so the catalogue file in that repository
is the closest verifiable primary source). It remains the current,
recommended lightweight German model: no smaller or newer "small" German
model supersedes it in that catalogue as of this add-on's development. A
much larger ``vosk-model-de-0.21`` (1.9GB, "big German model for telephony
and server", also Apache 2.0) exists upstream for higher accuracy, but is
deliberately NOT offered as a second selectable option here -- this
engine's entire purpose (per its own design goal) is to be the lightest,
lowest-RAM/CPU STT baseline this add-on offers; adding a multi-gigabyte
"large" variant would both contradict that purpose and complicate the UI
for a marginal, rarely-needed benefit (Kroko/Speechcatcher-L already cover
"bigger/more accurate"). Download URL:

    https://alphacephei.com/vosk/models/vosk-model-small-de-0.15.zip

matching ``vosk.MODEL_PRE_URL`` ("https://alphacephei.com/vosk/models/"),
the exact base URL the ``vosk`` package's own bundled downloader
(``vosk.Model``'s auto-download path) uses -- this add-on does not use that
auto-download helper (it wants an atomic, verifiable, testable download of
its own, matching every other engine's model-caching module in this
add-on), but the URL it constructs is the identical one that helper would
otherwise fetch.
"""

import logging
import os
import shutil
import tempfile
import urllib.request
import zipfile
from pathlib import Path

_LOGGER = logging.getLogger(__name__)

DEFAULT_VOSK_CACHE_DIR = Path(os.environ.get("VOSK_CACHE", "/data/models/vosk"))

VOSK_MODEL_NAME = "vosk-model-small-de-0.15"
VOSK_DOWNLOAD_URL = f"https://alphacephei.com/vosk/models/{VOSK_MODEL_NAME}.zip"

VOSK_SAMPLE_RATE = 16000

VOSK_MODEL_METADATA = {
    "name": VOSK_MODEL_NAME,
    "description": (
        "Extremely lightweight German Kaldi ASR model (Vosk/AlphaCephei), "
        "the lowest-RAM/CPU streaming STT option this add-on offers"
    ),
    "license": "Apache License 2.0 (AlphaCephei Vosk German small model)",
}


class VoskModelDownloadError(Exception):
    """Raised when the Vosk model archive/files could not be resolved.

    A distinct exception type (like KrokoModelDownloadError/
    SpeechcatcherModelDownloadError) so app/__main__.py can log a clear,
    specific startup error instead of a generic failure -- selecting
    ``stt_engine: vosk_german`` and having the download fail must be a
    visible configuration/network error, never a silent fallback to
    Moonshine.
    """


def _download_and_extract(url: str, dest_dir: Path) -> None:
    """Download ``url`` (a zip archive) to a temp file and extract it into
    ``dest_dir``.

    Atomic with respect to a partially-downloaded/extracted result, exactly
    like app/kroko_model.py's ``_download_and_extract``: the archive is
    downloaded to a temporary file and extracted into a temporary sibling
    directory, which is only ``rename()``d into ``dest_dir`` after both
    steps succeed -- a killed download or a failed extraction never leaves
    ``dest_dir`` populated with partial content a later run might mistake
    for a complete, valid cache.
    """
    dest_dir.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory(dir=dest_dir.parent) as tmp:
        tmp_path = Path(tmp)
        archive_path = tmp_path / "vosk-model.zip"
        _LOGGER.info("Downloading Vosk German model from %s", url)
        try:
            urllib.request.urlretrieve(url, archive_path)  # noqa: S310 - fixed, trusted URL
        except Exception as err:
            raise VoskModelDownloadError(
                f"Failed to download Vosk model from '{url}': {type(err).__name__}: {err}"
            ) from err

        extract_dir = tmp_path / "extracted"
        extract_dir.mkdir()
        try:
            with zipfile.ZipFile(archive_path) as zf:
                zf.extractall(extract_dir)  # noqa: S202 - fixed, trusted URL/archive
        except Exception as err:
            raise VoskModelDownloadError(
                f"Failed to extract Vosk model archive downloaded from '{url}': "
                f"{type(err).__name__}: {err}"
            ) from err

        if dest_dir.exists():
            shutil.rmtree(dest_dir)
        shutil.move(str(extract_dir), str(dest_dir))


def _find_model_dir(package_dir: Path) -> Path:
    """Find the actual Vosk model directory (the one containing ``conf/``,
    ``am/`` etc.) somewhere under ``package_dir``.

    Vosk's zip archives normally contain one top-level directory named
    after the model (e.g. ``vosk-model-small-de-0.15/conf/...``), but this
    does not hard-code that exact nesting -- it searches (like
    app/kroko_model.py's ``_find_one``) for the well-known ``conf``
    subdirectory every Vosk model directory contains, so a future
    re-release with a slightly different top-level layout (or none at all)
    still resolves correctly, and fails loudly and clearly if it does not.
    """
    if (package_dir / "conf").is_dir():
        return package_dir
    candidates = sorted(p.parent for p in package_dir.rglob("conf") if p.is_dir())
    if not candidates:
        raise VoskModelDownloadError(
            f"Could not find a Vosk model directory (one containing 'conf/') under "
            f"{package_dir} -- the downloaded Vosk archive layout may have changed upstream"
        )
    return candidates[0]


def resolve_vosk_model_dir(cache_dir: Path | None = None) -> Path:
    """Resolve (downloading if needed) the Vosk German model directory.

    Only downloads when ``stt_engine: vosk_german`` is actually selected
    (this function is only ever called from that code path, see
    app/vosk_engine.py / app/__main__.py) -- Vosk's model is never
    downloaded or loaded for any other ``stt_engine`` value, and no other
    engine's model is touched by this module.

    Args:
        cache_dir: Directory to store/extract the downloaded archive into.
            Defaults to :data:`DEFAULT_VOSK_CACHE_DIR`
            (``/data/models/vosk`` in production; overridden in tests).
            Persistent across restarts: a cache hit (a valid model
            directory already present) makes no network request at all.

    Returns:
        The resolved model directory path (suitable for ``vosk.Model(path)``).

    Raises:
        VoskModelDownloadError: on any download, extraction, or
            missing-directory failure. Never returns a partial result.
    """
    resolved_cache_dir = cache_dir if cache_dir is not None else DEFAULT_VOSK_CACHE_DIR
    package_dir = resolved_cache_dir / VOSK_MODEL_NAME

    if not package_dir.is_dir() or not any(package_dir.rglob("conf")):
        _download_and_extract(VOSK_DOWNLOAD_URL, package_dir)
    else:
        _LOGGER.info("Vosk German model already cached at %s", package_dir)

    model_dir = _find_model_dir(package_dir)

    if not (model_dir / "conf").is_dir():
        raise VoskModelDownloadError(f"Resolved Vosk model directory '{model_dir}' is invalid")

    return model_dir
