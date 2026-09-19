"""Supertonic 3 (INT8) TTS model resolution, caching, and loading.

Supertonic is Supertone Inc.'s on-device TTS system
(https://github.com/supertone-inc/supertonic, MIT-licensed -- verified
directly from that repository's own ``LICENSE`` file). k2-fsa/sherpa-onnx
exports it to a quantized ONNX Runtime package via its own CI (see
``.github/workflows/export-supertonic.yaml`` in a clone of k2-fsa/sherpa-onnx,
inspected directly while building this add-on) and publishes it as a GitHub
Release asset -- not invented or guessed:

    https://github.com/k2-fsa/sherpa-onnx/releases/download/tts-models/
        sherpa-onnx-supertonic-3-tts-int8-2026-05-11.tar.bz2

The archive contains exactly these files (verified from that same CI
workflow's own "Collect results" step, which names each file explicitly):
``duration_predictor.int8.onnx``, ``text_encoder.int8.onnx``,
``vector_estimator.int8.onnx``, ``vocoder.int8.onnx``, ``tts.json``,
``unicode_indexer.bin``, ``voice.bin``, plus ``LICENSE``/``README.md`` copied
from the upstream supertone-inc/supertonic repository.

Voices/speakers: the bundled ``voice.bin`` is built by sherpa-onnx's own
``scripts/supertonic/generate_voices_bin.py``, which merges
``assets/voice_styles/*.json`` in ``sorted()`` (i.e. plain alphabetical
filename) order into one file, printing "Merged N voice(s) -> ... (sid
0..N-1)" -- verified directly from that script's source. Supertonic's own
upstream README documents exactly 10 named voice styles as of this add-on's
Supertonic 3 integration: ``F1``-``F5`` (female) and ``M1``-``M5`` (male),
each shipped as ``<name>.json`` (verified from supertone-inc/supertonic's
own README, which also confirms ``M1`` is used as its own default/example
male voice). Sorting those 10 filenames alphabetically ("F1".."F5" sort
before "M1".."M5") gives the sid table below. This mapping is a verified
DEDUCTION from two independent, real upstream sources (the merge script's
sort order + the standard voice-style filenames), not a value read directly
out of the shipped ``voice.bin`` itself -- see :data:`SUPERTONIC_VOICES`'s
own comment and this add-on's CHANGELOG for that caveat.

Language: the underlying model is documented upstream as multi-lingual (31
language codes, including "de" for German -- verified from sherpa-onnx's own
``python-api-examples/supertonic-tts.py``, which sets
``gen_config.extra["lang"]`` to select the language for a single shared
checkpoint).

Sample rate: read at runtime from the loaded ``sherpa_onnx.OfflineTts``
instance's own ``sample_rate`` property rather than hard-coded here -- no
sherpa-onnx/Supertonic documentation found during this add-on's development
states a fixed number, and reading it from the real, loaded model is more
reliable than guessing one.
"""

import logging
import os
import shutil
import tarfile
import tempfile
import urllib.request
from pathlib import Path

from app.safe_tar_extract import UnsafeTarMemberError, safe_extractall

_LOGGER = logging.getLogger(__name__)

DEFAULT_SUPERTONIC_CACHE_DIR = Path(os.environ.get("SUPERTONIC_CACHE", "/data/models/supertonic"))

SUPERTONIC_PACKAGE_NAME = "sherpa-onnx-supertonic-3-tts-int8-2026-05-11"
SUPERTONIC_DOWNLOAD_URL = (
    "https://github.com/k2-fsa/sherpa-onnx/releases/download/tts-models/"
    f"{SUPERTONIC_PACKAGE_NAME}.tar.bz2"
)

# sid -> name, per this module's docstring (alphabetical merge order of the
# standard F1-F5/M1-M5 voice-style filenames). Index IS the sid sherpa-onnx's
# GenerationConfig.sid expects.
SUPERTONIC_VOICES: dict[int, str] = {
    0: "F1",
    1: "F2",
    2: "F3",
    3: "F4",
    4: "F5",
    5: "M1",
    6: "M2",
    7: "M3",
    8: "M4",
    9: "M5",
}
SUPERTONIC_VOICE_NAME_TO_SID: dict[str, int] = {
    name: sid for sid, name in SUPERTONIC_VOICES.items()
}

# M1 is the male voice supertone-inc/supertonic's own examples use as their
# default (see module docstring) -- matches this add-on's own requirement of
# defaulting to a male voice.
DEFAULT_SUPERTONIC_VOICE = "M1"

# supertone-inc/supertonic's own Python example documents 8 as the default
# and 10 as a "higher quality, slower" alternative (py/README.md) -- these
# are the only two step counts upstream actually documents; any other
# integer is accepted (sherpa-onnx enforces no fixed enum here) but not
# specially labeled.
DEFAULT_SUPERTONIC_STEPS = 8

SUPERTONIC_MODEL_METADATA = {
    "name": "supertonic-3-int8",
    "description": "Supertonic 3 (INT8, via sherpa-onnx) multilingual streaming TTS",
    "license": "MIT (Supertone Inc., see supertone-inc/supertonic)",
}


class SupertonicModelDownloadError(Exception):
    """Raised when the Supertonic model archive/files could not be resolved.

    Selecting ``tts_engine: supertonic_3`` and having the download fail is a
    visible, fatal startup error -- never a silent fallback to Pocket TTS or
    Kokoro (see app/__main__.py's module docstring / this repo's overall
    fail-loudly philosophy for TTS engine loading).
    """


def _download_and_extract(url: str, dest_dir: Path) -> None:
    """Same atomic download+extract strategy as app/kroko_model.py's own
    helper (download to a temp file, extract to a temp sibling dir, rename
    into place only once both steps succeed), and the same shared
    :func:`app.safe_tar_extract.safe_extractall` for the extraction step
    itself -- see that module's docstring for the real production bug
    (Debian bookworm's apt-installed python3.11 lacking
    `TarFile.extractall()`'s `filter=` keyword argument) this guards against
    for both Kroko and Supertonic."""
    dest_dir.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory(dir=dest_dir.parent) as tmp:
        tmp_path = Path(tmp)
        archive_path = tmp_path / "supertonic.tar.bz2"
        _LOGGER.info("Downloading Supertonic 3 TTS model from %s", url)
        try:
            urllib.request.urlretrieve(url, archive_path)  # noqa: S310 - fixed, trusted URL
        except Exception as err:
            raise SupertonicModelDownloadError(
                f"Failed to download Supertonic model from '{url}': {type(err).__name__}: {err}"
            ) from err

        extract_dir = tmp_path / "extracted"
        extract_dir.mkdir()
        try:
            with tarfile.open(archive_path) as tf:
                safe_extractall(tf, extract_dir, label="Supertonic model")
        except UnsafeTarMemberError as err:
            raise SupertonicModelDownloadError(
                f"Refusing to extract unsafe Supertonic model archive member: {err}"
            ) from err
        except Exception as err:
            raise SupertonicModelDownloadError(
                f"Failed to extract Supertonic model archive downloaded from '{url}': "
                f"{type(err).__name__}: {err}"
            ) from err

        if dest_dir.exists():
            shutil.rmtree(dest_dir)
        shutil.move(str(extract_dir), str(dest_dir))


_REQUIRED_FILES = (
    "duration_predictor.int8.onnx",
    "text_encoder.int8.onnx",
    "vector_estimator.int8.onnx",
    "vocoder.int8.onnx",
    "tts.json",
    "unicode_indexer.bin",
    "voice.bin",
)


def resolve_supertonic_model_files(cache_dir: Path | None = None) -> dict[str, Path]:
    """Resolve (downloading if needed) the Supertonic model's required files.

    Only ever called when ``tts_enabled: true`` AND ``tts_engine:
    supertonic_3`` (see app/__main__.py) -- never downloaded or loaded for
    any other TTS engine selection, and never baked into the Docker image
    (matches config.yaml's ``backup_exclude: ["models/*"]``).
    """
    resolved_cache_dir = cache_dir if cache_dir is not None else DEFAULT_SUPERTONIC_CACHE_DIR
    package_dir = resolved_cache_dir / SUPERTONIC_PACKAGE_NAME

    if not package_dir.is_dir():
        _download_and_extract(SUPERTONIC_DOWNLOAD_URL, package_dir)
    else:
        _LOGGER.info("Supertonic 3 TTS model already cached at %s", package_dir)

    resolved: dict[str, Path] = {}
    for name in _REQUIRED_FILES:
        candidates = list(package_dir.rglob(name))
        if not candidates:
            raise SupertonicModelDownloadError(
                f"Required Supertonic file '{name}' not found under {package_dir} -- "
                f"the downloaded archive layout may have changed upstream"
            )
        path = candidates[0]
        if not path.is_file() or path.stat().st_size == 0:
            raise SupertonicModelDownloadError(f"Resolved Supertonic file '{path}' is empty")
        resolved[name] = path
    return resolved


def resolve_supertonic_sid(voice: str) -> int:
    """Resolve a voice name (case-insensitive, e.g. "M1") or a raw numeric
    sid string (e.g. "6") to its integer sid.

    Raises:
        ValueError: if ``voice`` is neither a known name nor 0-9.
    """
    if voice.strip().isdigit():
        sid = int(voice.strip())
        if sid not in SUPERTONIC_VOICES:
            raise ValueError(
                f"Supertonic sid must be between 0 and {len(SUPERTONIC_VOICES) - 1}, got {sid}"
            )
        return sid
    normalized = voice.strip().upper()
    if normalized not in SUPERTONIC_VOICE_NAME_TO_SID:
        raise ValueError(
            f"Unknown Supertonic voice '{voice}'; available voices: "
            f"{sorted(SUPERTONIC_VOICE_NAME_TO_SID)}"
        )
    return SUPERTONIC_VOICE_NAME_TO_SID[normalized]


def load_supertonic_tts(
    cache_dir: Path | None = None,
    num_threads: int = 1,
) -> object:
    """Resolve/download the Supertonic model and construct a loaded
    ``sherpa_onnx.OfflineTts`` instance.

    Returns ``object`` (not ``sherpa_onnx.OfflineTts``) so this module's
    public type surface does not force a hard, always-imported dependency
    on ``sherpa_onnx`` for callers that only need the small set of methods
    this add-on actually calls -- same rationale as app/kokoro_tts.py's
    ``load_kokoro_model()``.
    """
    import sherpa_onnx  # local import: only needed when tts_engine=supertonic_3

    files = resolve_supertonic_model_files(cache_dir=cache_dir)
    _LOGGER.info(
        "Loading Supertonic 3 TTS model from %s (num_threads=%d)",
        files["tts.json"].parent,
        num_threads,
    )
    tts_config = sherpa_onnx.OfflineTtsConfig(
        model=sherpa_onnx.OfflineTtsModelConfig(
            supertonic=sherpa_onnx.OfflineTtsSupertonicModelConfig(
                duration_predictor=str(files["duration_predictor.int8.onnx"]),
                text_encoder=str(files["text_encoder.int8.onnx"]),
                vector_estimator=str(files["vector_estimator.int8.onnx"]),
                vocoder=str(files["vocoder.int8.onnx"]),
                tts_json=str(files["tts.json"]),
                unicode_indexer=str(files["unicode_indexer.bin"]),
                voice_style=str(files["voice.bin"]),
            ),
            debug=False,
            num_threads=num_threads,
            provider="cpu",
        )
    )
    if not tts_config.validate():
        raise SupertonicModelDownloadError(
            "sherpa_onnx.OfflineTtsConfig.validate() rejected the resolved Supertonic "
            "model files -- see preceding log lines from sherpa-onnx itself for details"
        )
    tts = sherpa_onnx.OfflineTts(tts_config)
    _LOGGER.info("Supertonic 3 TTS model ready (sample_rate=%dHz)", tts.sample_rate)
    return tts
