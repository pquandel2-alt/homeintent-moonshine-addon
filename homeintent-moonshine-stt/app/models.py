"""Moonshine model loading and management."""

import logging
import os
from enum import StrEnum
from pathlib import Path

from moonshine_voice import ModelArch, Transcriber, get_model_for_language

_LOGGER = logging.getLogger(__name__)

# Home Assistant provides a persistent /data/ directory for add-ons. Moonshine
# reads its cache location from the MOONSHINE_VOICE_CACHE env var (see
# moonshine_voice.download_file.get_cache_dir); we pass the same path
# explicitly as cache_root so model resolution does not depend on that env
# var being set by the caller (e.g. under pytest).
DEFAULT_MODEL_CACHE_DIR = Path(os.environ.get("MOONSHINE_VOICE_CACHE", "/data/models"))


class GermanModel(StrEnum):
    """Supported German Moonshine models."""

    TINY = "tiny"
    SMALL = "small"


# Only these two streaming architectures are published for German (see
# moonshine_voice.download._stt_catalog()["de"]["models"]). Non-streaming
# ModelArch.TINY/BASE are not valid for this add-on's use case.
_MODEL_ARCH = {
    GermanModel.TINY: ModelArch.TINY_STREAMING,
    GermanModel.SMALL: ModelArch.SMALL_STREAMING,
}

# Upstream-published model metadata (moonshine-voice 0.1.5, German). No
# latency/RTF numbers here since we have not benchmarked them ourselves.
#
# License: moonshine_voice.download.get_model_for_language() prints, at
# download time for any wanted_language != "en", "Using a model released
# under the non-commercial Moonshine Community License." (confirmed by
# running the real download against language="de"). Only Moonshine's
# English models are MIT; the German models used here are NOT MIT.
_GERMAN_MODEL_LICENSE = (
    "Moonshine Community License (non-commercial, see https://www.moonshine.ai/license)"
)

MODEL_METADATA = {
    GermanModel.TINY: {
        "name": "moonshine-german-tiny-streaming",
        "params": "34M",
        "wer": "12.0%",
        "license": _GERMAN_MODEL_LICENSE,
        "description": "Fast German speech recognition (Tiny, 34M parameters)",
    },
    GermanModel.SMALL: {
        "name": "moonshine-german-small-streaming",
        "params": "123M",
        "wer": "7.5%",
        "license": _GERMAN_MODEL_LICENSE,
        "description": "Accurate German speech recognition (Small, 123M parameters)",
    },
}


def validate_model(model: str) -> bool:
    """Check if model name is valid."""
    try:
        GermanModel(model)
        return True
    except ValueError:
        return False


# Upstream defaults, confirmed from the real moonshine-voice 0.1.5 C++
# source (core/transcriber.h / core/moonshine-c-api.cpp) at GitHub tag
# v0.1.5. Do not change these without re-verifying against upstream: they
# exist so that leaving a config option untouched reproduces upstream's own
# behavior exactly, not an invented "tuned" value.
DEFAULT_TRANSCRIPTION_INTERVAL = 0.5  # Transcriber.__init__(update_interval=...)
DEFAULT_VAD_THRESHOLD = 0.5  # options["vad_threshold"]
DEFAULT_DECODE_INCOMPLETE_LINES = True  # options["decode_incomplete_lines"]
DEFAULT_KEYTERM_BOOST = 2.0  # options["keyterm_boost"], ContextBiaser::kDefaultBoost


def load_transcriber(
    model: str = GermanModel.SMALL.value,
    language: str = "de",
    cache_root: Path | None = None,
    transcription_interval: float = DEFAULT_TRANSCRIPTION_INTERVAL,
    vad_threshold: float = DEFAULT_VAD_THRESHOLD,
    decode_incomplete_lines: bool = DEFAULT_DECODE_INCOMPLETE_LINES,
    keyterm_boost: float = DEFAULT_KEYTERM_BOOST,
) -> Transcriber:
    """Resolve, download (if needed), and load a Moonshine Transcriber.

    Must be called exactly once at add-on startup; the returned Transcriber
    is shared across all Wyoming connections via Transcriber.create_stream()
    for per-session isolation.

    Args:
        model: Model size ("tiny" or "small"). Default: "small"
        language: Language code. Always "de" for this add-on.
        cache_root: Override for the model cache directory (used in tests).
        transcription_interval: Seconds between streaming transcription
            passes (Transcriber's ``update_interval``). Upstream default 0.5.
        vad_threshold: Voice-activity-detection sensitivity, native option
            ``vad_threshold``. Upstream default 0.5.
        decode_incomplete_lines: Native option ``decode_incomplete_lines``.
            Upstream default True.
        keyterm_boost: Strength applied to terms passed to set_keyterms().
            Native option ``keyterm_boost``. Upstream default 2.0.

    Returns:
        Initialized Transcriber instance ready to create streaming sessions.

    Raises:
        ValueError: If model name is invalid.
        Exception: If model download or initialization fails.
    """
    if not validate_model(model):
        raise ValueError(
            f"Invalid model: {model}. Must be one of: {', '.join(m.value for m in GermanModel)}"
        )

    model_enum = GermanModel(model)
    model_arch = _MODEL_ARCH[model_enum]
    resolved_cache_root = cache_root if cache_root is not None else DEFAULT_MODEL_CACHE_DIR
    resolved_cache_root.mkdir(parents=True, exist_ok=True)

    _LOGGER.info(
        "Resolving Moonshine model: language=%s arch=%s cache_root=%s",
        language,
        model_arch.name,
        resolved_cache_root,
    )

    # get_model_for_language()/download_file() (moonshine_voice.download_file)
    # already download to a ".partial" file and atomically rename to the
    # final path only after a size/hash/CRC32C check succeeds, and re-check
    # under a file lock before reusing a cached file -- so first download,
    # restart-with-cache, resumed partial download, and a corrupted cache
    # entry (mismatched hash -> deleted and re-downloaded) are all upstream's
    # own responsibility. We deliberately do not duplicate or second-guess
    # that here with our own deletion logic; we only add clear logging.
    try:
        model_path, resolved_arch = get_model_for_language(
            wanted_language=language,
            wanted_model_arch=model_arch,
            cache_root=resolved_cache_root,
        )
    except Exception as err:
        _LOGGER.error(
            "Failed to resolve/download Moonshine model (language=%s arch=%s "
            "cache_root=%s): %s: %s",
            language,
            model_arch.name,
            resolved_cache_root,
            type(err).__name__,
            err,
        )
        raise

    _LOGGER.info(
        "Loading Moonshine transcriber: model_path=%s arch=%s "
        "update_interval=%s vad_threshold=%s decode_incomplete_lines=%s keyterm_boost=%s",
        model_path,
        resolved_arch.name,
        transcription_interval,
        vad_threshold,
        decode_incomplete_lines,
        keyterm_boost,
    )
    options = {
        "vad_threshold": vad_threshold,
        "decode_incomplete_lines": "true" if decode_incomplete_lines else "false",
        "keyterm_boost": keyterm_boost,
    }
    transcriber = Transcriber(
        model_path,
        resolved_arch,
        update_interval=transcription_interval,
        options=options,
    )
    _LOGGER.info("Moonshine %s model ready", model)

    return transcriber


def get_model_info(model: str) -> dict[str, str]:
    """Get metadata for a model (for Wyoming service discovery)."""
    try:
        model_enum = GermanModel(model)
        return MODEL_METADATA[model_enum]
    except ValueError:
        return {
            "name": "unknown",
            "description": "Unknown model",
        }
