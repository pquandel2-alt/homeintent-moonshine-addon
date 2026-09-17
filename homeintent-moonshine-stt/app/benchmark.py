"""Local benchmark CLI: measures real-time factor (RTF) on the user's own
hardware against a user-supplied WAV file.

Not run in CI and not a source of any RTF/latency numbers quoted in
README/DOCS -- Moonshine's performance depends heavily on the host CPU, so
the only honest numbers are ones users measure themselves with this tool.

Usage:
    python -m app.benchmark path/to/sample.wav --model small --language de
"""

import argparse
import asyncio
import logging
import sys
import time
import wave
from collections.abc import Iterator
from dataclasses import dataclass
from pathlib import Path

from moonshine_voice import Transcriber

from app.audio import pcm_int16_to_float32, validate_audio_format
from app.models import DEFAULT_MODEL_CACHE_DIR, load_transcriber
from app.streaming import MoonshineStreamingSession

_LOGGER = logging.getLogger(__name__)


@dataclass
class BenchmarkResult:
    audio_duration_s: float
    processing_time_s: float
    transcript: str

    @property
    def rtf(self) -> float:
        """Real-time factor: <1.0 means faster than real time."""
        if self.audio_duration_s == 0:
            return float("inf")
        return self.processing_time_s / self.audio_duration_s


def read_wav_pcm(path: Path) -> tuple[bytes, int, int, int]:
    """Read a WAV file's raw PCM bytes and format (rate, width, channels).

    Raises:
        ValueError: If the file is not 16kHz/16-bit/mono.
    """
    with wave.open(str(path), "rb") as wav_file:
        rate = wav_file.getframerate()
        width = wav_file.getsampwidth()
        channels = wav_file.getnchannels()
        pcm_bytes = wav_file.readframes(wav_file.getnframes())

    error = validate_audio_format(rate=rate, width=width, channels=channels, context=str(path))
    if error is not None:
        raise ValueError(error)

    return pcm_bytes, rate, width, channels


def chunk_pcm(pcm_bytes: bytes, chunk_bytes: int) -> Iterator[bytes]:
    """Split raw PCM bytes into fixed-size chunks (last chunk may be shorter).

    chunk_bytes is rounded down to an even number so every chunk stays valid
    16-bit-sample-aligned input for pcm_int16_to_float32().
    """
    chunk_bytes -= chunk_bytes % 2
    for offset in range(0, len(pcm_bytes), chunk_bytes):
        yield pcm_bytes[offset : offset + chunk_bytes]


async def run_benchmark(
    transcriber: Transcriber, pcm_bytes: bytes, sample_rate: int, chunk_ms: int
) -> BenchmarkResult:
    """Feed a WAV's PCM through a streaming session as fast as possible and
    time it, to measure processing speed independent of real-time playback.
    """
    chunk_bytes = int(sample_rate * (chunk_ms / 1000) * 2)
    session = MoonshineStreamingSession(transcriber, lock=asyncio.Lock())

    start = time.perf_counter()
    await session.start()
    for chunk in chunk_pcm(pcm_bytes, chunk_bytes):
        samples = pcm_int16_to_float32(chunk).tolist()
        await session.add_audio(samples, sample_rate)
    transcript = await session.finalize()
    processing_time_s = time.perf_counter() - start
    session.close()

    audio_duration_s = (len(pcm_bytes) / 2) / sample_rate
    return BenchmarkResult(
        audio_duration_s=audio_duration_s,
        processing_time_s=processing_time_s,
        transcript=transcript,
    )


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Benchmark Moonshine STT streaming performance on this machine "
            "using a local 16kHz/mono/16-bit WAV file."
        )
    )
    parser.add_argument("wav_file", type=Path, help="Path to a 16kHz mono 16-bit WAV file")
    parser.add_argument("--model", choices=["tiny", "small"], default="small")
    parser.add_argument("--language", default="de")
    parser.add_argument(
        "--chunk-ms",
        type=int,
        default=30,
        help="Simulated Wyoming audio-chunk size in milliseconds (default: 30)",
    )
    return parser


def main() -> int:
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")
    parser = build_arg_parser()
    args = parser.parse_args()

    if not args.wav_file.exists():
        _LOGGER.error("File not found: %s", args.wav_file)
        return 1

    try:
        pcm_bytes, rate, _width, _channels = read_wav_pcm(args.wav_file)
    except ValueError as e:
        _LOGGER.error("Invalid WAV file: %s", e)
        return 1

    _LOGGER.info(
        "Loading model=%s language=%s cache=%s (first run downloads the model)",
        args.model,
        args.language,
        DEFAULT_MODEL_CACHE_DIR,
    )
    transcriber = load_transcriber(model=args.model, language=args.language)

    _LOGGER.info("Benchmarking %s (chunk_ms=%d)...", args.wav_file, args.chunk_ms)
    result = asyncio.run(run_benchmark(transcriber, pcm_bytes, rate, args.chunk_ms))

    print(f"Audio duration:   {result.audio_duration_s:.2f}s")
    print(f"Processing time:  {result.processing_time_s:.2f}s")
    print(
        f"Real-time factor: {result.rtf:.3f} ({'faster' if result.rtf < 1 else 'slower'} than real time)"
    )
    print(f"Transcript:       {result.transcript}")

    return 0


if __name__ == "__main__":
    sys.exit(main())
