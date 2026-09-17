"""Optional on-disk recording of received audio, for debugging only.

Disabled by default (``save_debug_audio: false``). When enabled, every
finalized Wyoming audio session is written as a 16kHz mono 16-bit PCM WAV
file under ``debug_audio_dir``, with a JSON metadata sidecar -- and nothing
else: no Home Assistant entity states or other data are ever written here.
This persists microphone audio to disk; see the README privacy note before
enabling it.
"""

import itertools
import json
import logging
import wave
from dataclasses import asdict, dataclass
from datetime import UTC, datetime
from pathlib import Path

_LOGGER = logging.getLogger(__name__)

DEFAULT_DEBUG_AUDIO_DIR = Path("/data/debug_audio")
SAMPLE_RATE = 16000
SAMPLE_WIDTH = 2  # bytes per sample (16-bit signed PCM, matches Wyoming's wire format)
CHANNELS = 1


@dataclass(frozen=True)
class DebugAudioMetadata:
    timestamp: str
    model: str
    language: str
    transcript: str
    duration_ms: float
    sample_rate: int


def _next_available_stem(directory: Path, timestamp: datetime) -> str:
    """Find a filename stem that doesn't collide with an existing file.

    Two sessions finalizing within the same second both want the same
    timestamp-based name, hence the numeric suffix.
    """
    base = timestamp.strftime("%Y-%m-%d_%H-%M-%S")
    for suffix in itertools.count(1):
        candidate = f"{base}_{suffix:03d}"
        if not (directory / f"{candidate}.wav").exists():
            return candidate
    raise AssertionError("unreachable")  # pragma: no cover


def save_debug_audio(
    pcm_bytes: bytes,
    *,
    model: str,
    language: str,
    transcript: str,
    directory: Path = DEFAULT_DEBUG_AUDIO_DIR,
    max_files: int = 100,
) -> Path:
    """Write ``pcm_bytes`` (16kHz mono 16-bit PCM) as a WAV + JSON sidecar.

    Enforces retention: after writing, deletes the oldest WAV/JSON pairs in
    ``directory`` beyond ``max_files`` so debug recording can never grow
    unbounded.

    Returns the path to the written WAV file.
    """
    directory.mkdir(parents=True, exist_ok=True)
    now = datetime.now(UTC)
    stem = _next_available_stem(directory, now)
    wav_path = directory / f"{stem}.wav"
    json_path = directory / f"{stem}.json"

    with wave.open(str(wav_path), "wb") as wav_file:
        wav_file.setnchannels(CHANNELS)
        wav_file.setsampwidth(SAMPLE_WIDTH)
        wav_file.setframerate(SAMPLE_RATE)
        wav_file.writeframes(pcm_bytes)

    duration_ms = (len(pcm_bytes) / SAMPLE_WIDTH / SAMPLE_RATE) * 1000
    metadata = DebugAudioMetadata(
        timestamp=now.isoformat(),
        model=model,
        language=language,
        transcript=transcript,
        duration_ms=round(duration_ms, 1),
        sample_rate=SAMPLE_RATE,
    )
    json_path.write_text(
        json.dumps(asdict(metadata), ensure_ascii=False, indent=2), encoding="utf-8"
    )

    _enforce_retention(directory, max_files)
    return wav_path


def _enforce_retention(directory: Path, max_files: int) -> None:
    """Delete the oldest WAV/JSON pairs beyond ``max_files``, by filename order.

    Filenames are timestamp-prefixed, so lexicographic order is chronological.
    """
    wav_files = sorted(directory.glob("*.wav"), key=lambda p: p.name)
    excess = len(wav_files) - max_files
    if excess <= 0:
        return
    for wav_path in wav_files[:excess]:
        json_path = wav_path.with_suffix(".json")
        wav_path.unlink(missing_ok=True)
        json_path.unlink(missing_ok=True)
        _LOGGER.debug("Deleted old debug audio recording: %s", wav_path.name)
