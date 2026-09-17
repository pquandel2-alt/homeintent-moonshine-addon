"""Audio format conversion utilities."""

import logging
from typing import Any

import numpy as np

_LOGGER = logging.getLogger(__name__)

# Wyoming standard: 16-bit signed PCM (2 bytes per sample)
WYOMING_BYTES_PER_SAMPLE = 2
WYOMING_SAMPLE_RATE = 16000

# Moonshine standard: 32-bit float, normalized to [-1.0, 1.0]
MOONSHINE_FLOAT_RANGE = 32768.0


def pcm_int16_to_float32(pcm_bytes: bytes) -> np.ndarray[Any, np.dtype[np.float32]]:
    """Convert 16-bit PCM bytes to 32-bit float array.

    Wyoming sends 16-bit signed PCM (mono, 16kHz).
    Moonshine expects 32-bit float normalized to [-1.0, 1.0].

    Args:
        pcm_bytes: Raw PCM audio bytes (assumed 16-bit signed, little-endian)

    Returns:
        numpy array of float32 samples normalized to [-1.0, 1.0]

    Raises:
        ValueError: If pcm_bytes length is odd (not a multiple of 2)
    """
    if len(pcm_bytes) % 2 != 0:
        raise ValueError(
            f"PCM bytes must be even length (got {len(pcm_bytes)}, expected multiple of 2)"
        )

    # Interpret bytes as int16 (little-endian, signed)
    samples_int16 = np.frombuffer(pcm_bytes, dtype=np.int16)

    # Convert to float32 and normalize
    samples_float32 = samples_int16.astype(np.float32) / MOONSHINE_FLOAT_RANGE

    return samples_float32


def validate_audio_format(
    rate: int, width: int, channels: int, context: str = "audio event"
) -> str | None:
    """Validate Wyoming audio format.

    Wyoming standard: 16kHz sample rate, 2-byte width (16-bit), 1 channel (mono).
    This function checks conformance and logs warnings.

    Args:
        rate: Sample rate in Hz
        width: Bytes per sample
        channels: Number of audio channels
        context: Brief description for logging (e.g. "audio-start", "audio-chunk")

    Returns:
        Error message if validation fails, None if valid
    """
    expected = {
        "rate": WYOMING_SAMPLE_RATE,
        "width": WYOMING_BYTES_PER_SAMPLE,
        "channels": 1,
    }

    if rate != expected["rate"]:
        msg = f"{context}: unexpected sample rate {rate}Hz (expected {expected['rate']}Hz)"
        _LOGGER.warning(msg)
        return msg

    if width != expected["width"]:
        msg = (
            f"{context}: unexpected sample width {width} bytes "
            f"(expected {expected['width']} bytes / 16-bit)"
        )
        _LOGGER.warning(msg)
        return msg

    if channels != expected["channels"]:
        msg = f"{context}: unexpected channels {channels} (expected {expected['channels']} / mono)"
        _LOGGER.warning(msg)
        return msg

    return None
