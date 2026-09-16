"""Moonshine model loading and management."""

import logging
from enum import Enum
from pathlib import Path
from typing import Optional

from moonshine_voice import Transcriber, ModelArch

_LOGGER = logging.getLogger(__name__)

# Persistent storage: Home Assistant provides /data/ directory
MODEL_CACHE_DIR = Path("/data/models")


class GermanModel(str, Enum):
    """Supported German Moonshine models."""

    TINY = "tiny"
    SMALL = "small"


# Model metadata for Wyoming service discovery
MODEL_METADATA = {
    GermanModel.TINY: {
        "name": "moonshine-german-tiny-streaming",
        "params": "34M",
        "wer": "12.0%",
        "description": "Fast German speech recognition (Tiny, 34M parameters)",
    },
    GermanModel.SMALL: {
        "name": "moonshine-german-small-streaming",
        "params": "123M",
        "wer": "7.5%",
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
    model: str = GermanModel.SMALL, language: str = "de"
) -> Transcriber:
    """Load and return a Moonshine Transcriber for German streaming ASR.

    This function:
    1. Validates the model name
    2. Creates a Transcriber instance
    3. Configures it for German language
    4. Sets the model architecture (Tiny or Small)
    5. Sets the cache directory for model files
    6. Loads the model (downloads if not cached)

    On first call, downloads ~200MB (Small) or ~80MB (Tiny) models.
    On subsequent calls, uses cached models from /data/models.

    Args:
        model: Model size ("tiny" or "small"). Default: "small"
        language: Language code. Always "de" for this add-on.

    Returns:
        Initialized Transcriber instance ready for streaming

    Raises:
        ValueError: If model name is invalid
        Exception: If model download or initialization fails (Moonshine runtime error)
    """
    if not validate_model(model):
        raise ValueError(
            f"Invalid model: {model}. Must be one of: {', '.join(m.value for m in GermanModel)}"
        )

    _LOGGER.info(f"Loading Moonshine {model.upper()} model for German...")

    # Create cache directory if it doesn't exist
    MODEL_CACHE_DIR.mkdir(parents=True, exist_ok=True)

    # Initialize Transcriber
    transcriber = Transcriber()

    # Set language (language code "de" for German)
    transcriber.language(language)

    # Set model architecture
    arch = ModelArch.TINY if model == "tiny" else ModelArch.SMALL
    transcriber.model_arch(arch)

    # Set cache directory (via environment variable HF_HOME as fallback,
    # or directly if Transcriber API supports it)
    # Note: Moonshine uses HuggingFace model hub, so HF_HOME env var is respected
    transcriber.model_dir(str(MODEL_CACHE_DIR))

    # Load model (downloads if needed, uses cache otherwise)
    try:
        _LOGGER.info(f"Initializing Moonshine {model} model...")
        transcriber.load()
        _LOGGER.info(f"Moonshine {model} model ready")
    except Exception as e:
        _LOGGER.error(f"Failed to load Moonshine model: {e}")
        raise

    return transcriber


def get_model_info(model: str) -> dict:
    """Get metadata for a model (for Wyoming service discovery)."""
    try:
        model_enum = GermanModel(model)
        return MODEL_METADATA[model_enum]
    except ValueError:
        return {
            "name": "unknown",
            "description": "Unknown model",
        }
