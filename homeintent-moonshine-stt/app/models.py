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
_GERMAN_MODEL_LICENSE = "Moonshine Community License (non-commercial, see https://www.moonshine.ai/license)"

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


def load_transcriber(
    model: str = GermanModel.SMALL.value,
    language: str = "de",
    cache_root: Path | None = None,
) -> Transcriber:
    """Resolve, download (if needed), and load a Moonshine Transcriber.

    Must be called exactly once at add-on startup; the returned Transcriber
    is shared across all Wyoming connections via Transcriber.create_stream()
    for per-session isolation.

    Args:
        model: Model size ("tiny" or "small"). Default: "small"
        language: Language code. Always "de" for this add-on.
        cache_root: Override for the model cache directory (used in tests).

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

    model_path, resolved_arch = get_model_for_language(
        wanted_language=language,
        wanted_model_arch=model_arch,
        cache_root=resolved_cache_root,
    )

    _LOGGER.info(
        "Loading Moonshine transcriber: model_path=%s arch=%s",
        model_path,
        resolved_arch.name,
    )
    transcriber = Transcriber(model_path, resolved_arch)
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
