"""Local benchmark CLI: measures real-time factor (RTF) on the user's own
hardware against a user-supplied WAV file.

Not run in CI and not a source of any RTF/latency numbers quoted in
README/DOCS -- Moonshine's performance depends heavily on the host CPU (see
GitHub Actions runners are NOT representative of real Home Assistant
hardware), so the only honest numbers are ones users measure themselves
with this tool, on their own target machine.

Usage:
    python -m app.benchmark path/to/sample.wav --models small
    python -m app.benchmark path/to/sample.wav --models tiny,small --output json
    python -m app.benchmark path/to/sample.wav --models tiny,small \
        --transcription-interval 0.25 --keyterm-count 100 --output csv
"""

import argparse
import asyncio
import json
import logging
import sys
import time
import wave
from collections.abc import Iterator
from dataclasses import asdict, dataclass
from pathlib import Path

from moonshine_voice import Transcriber

from app.audio import pcm_int16_to_float32, validate_audio_format
from app.models import DEFAULT_MODEL_CACHE_DIR, DEFAULT_TRANSCRIPTION_INTERVAL, load_transcriber
from app.streaming import MoonshineStreamingSession

_LOGGER = logging.getLogger(__name__)


@dataclass
class BenchmarkResult:
    model: str
    transcription_interval: float
    decode_incomplete_lines: bool
    keyterm_count: int
    audio_duration_s: float
    processing_time_s: float
    inference_time_s: float
    finalize_time_s: float
    chunk_count: int
    add_audio_compute_max_s: float
    average_chunk_compute_s: float
    transcript: str

    @property
    def rtf(self) -> float:
        """Real-time factor computed from real Moonshine compute time
        (add_audio + finalize), not wall-clock processing_time_s -- matches
        the same definition app/handler.py's production RTF log uses."""
        if self.audio_duration_s == 0:
            return float("inf")
        return self.inference_time_s / self.audio_duration_s


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


def synthetic_keyterms(count: int) -> list[str]:
    """Generate ``count`` plausible-looking German smart-home keyterms for
    benchmarking keyterm-bias overhead (item 14) without depending on a
    real Home Assistant instance. Not meant to resemble any specific
    installation's real vocabulary -- only its *size*."""
    rooms = ["Wohnzimmer", "Küche", "Schlafzimmer", "Bad", "Büro", "Flur", "Keller", "Garage"]
    things = ["Licht", "Lampe", "Rollladen", "Steckdose", "Heizung", "Sensor", "Schalter", "Tür"]
    return [f"{rooms[i % len(rooms)]} {things[i % len(things)]} {i}" for i in range(count)]


async def run_benchmark(
    transcriber: Transcriber,
    pcm_bytes: bytes,
    sample_rate: int,
    chunk_ms: int,
    model: str,
    transcription_interval: float,
    decode_incomplete_lines: bool,
    keyterm_count: int,
) -> BenchmarkResult:
    """Feed a WAV's PCM through a streaming session as fast as possible and
    time it, to measure processing speed independent of real-time playback.
    """
    if keyterm_count > 0:
        transcriber.set_keyterms(synthetic_keyterms(keyterm_count))

    chunk_bytes = int(sample_rate * (chunk_ms / 1000) * 2)
    session = MoonshineStreamingSession(
        transcriber, update_interval=transcription_interval, lock=asyncio.Lock()
    )

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
        model=model,
        transcription_interval=transcription_interval,
        decode_incomplete_lines=decode_incomplete_lines,
        keyterm_count=keyterm_count,
        audio_duration_s=audio_duration_s,
        processing_time_s=processing_time_s,
        inference_time_s=session.inference_time_seconds,
        finalize_time_s=session.finalize_time_seconds,
        chunk_count=session.add_audio_chunk_count,
        add_audio_compute_max_s=session.add_audio_compute_max_seconds,
        average_chunk_compute_s=session.average_chunk_compute_seconds,
        transcript=transcript,
    )


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Benchmark Moonshine STT streaming performance on this machine "
            "using a local 16kHz/mono/16-bit WAV file. Run this ON the "
            "actual target hardware (e.g. the real Home Assistant host) -- "
            "numbers from a development machine or CI runner are not "
            "representative."
        )
    )
    parser.add_argument("wav_file", type=Path, help="Path to a 16kHz mono 16-bit WAV file")
    parser.add_argument(
        "--models",
        default="small",
        help="Comma-separated list of models to benchmark, e.g. 'tiny,small' (default: small)",
    )
    parser.add_argument("--language", default="de")
    parser.add_argument(
        "--chunk-ms",
        type=int,
        default=30,
        help="Simulated Wyoming audio-chunk size in milliseconds (default: 30)",
    )
    parser.add_argument(
        "--transcription-interval",
        type=float,
        default=DEFAULT_TRANSCRIPTION_INTERVAL,
        help="Same meaning as the add-on's transcription_interval option (item 12)",
    )
    parser.add_argument(
        "--decode-incomplete-lines",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Same meaning as the add-on's decode_incomplete_lines option (item 13)",
    )
    parser.add_argument(
        "--keyterm-count",
        type=int,
        default=0,
        help="Apply this many synthetic keyterms before benchmarking (item 14)",
    )
    parser.add_argument(
        "--output",
        choices=["text", "json", "csv"],
        default="text",
        help="Output format (item 28)",
    )
    return parser


def _print_text(result: BenchmarkResult) -> None:
    print(f"--- model={result.model} ---")
    print(f"Audio duration:        {result.audio_duration_s:.2f}s")
    print(f"Wall processing time:  {result.processing_time_s:.2f}s")
    print(f"Model inference time:  {result.inference_time_s:.2f}s")
    print(f"Finalize time:         {result.finalize_time_s:.2f}s")
    print(f"Chunks:                {result.chunk_count}")
    print(f"Slowest chunk:         {result.add_audio_compute_max_s:.3f}s")
    print(f"Average chunk compute: {result.average_chunk_compute_s:.3f}s")
    print(
        f"Real-time factor:      {result.rtf:.3f} "
        f"({'faster' if result.rtf < 1 else 'slower'} than real time)"
    )
    print(f"Transcript:            {result.transcript}")


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

    models = [m.strip() for m in args.models.split(",") if m.strip()]
    results: list[BenchmarkResult] = []
    for model in models:
        _LOGGER.info(
            "Loading model=%s language=%s cache=%s (first run downloads the model)",
            model,
            args.language,
            DEFAULT_MODEL_CACHE_DIR,
        )
        transcriber = load_transcriber(
            model=model,
            language=args.language,
            transcription_interval=args.transcription_interval,
            decode_incomplete_lines=args.decode_incomplete_lines,
        )

        _LOGGER.info(
            "Benchmarking %s (model=%s chunk_ms=%d transcription_interval=%s "
            "decode_incomplete_lines=%s keyterm_count=%d)...",
            args.wav_file,
            model,
            args.chunk_ms,
            args.transcription_interval,
            args.decode_incomplete_lines,
            args.keyterm_count,
        )
        result = asyncio.run(
            run_benchmark(
                transcriber,
                pcm_bytes,
                rate,
                args.chunk_ms,
                model=model,
                transcription_interval=args.transcription_interval,
                decode_incomplete_lines=args.decode_incomplete_lines,
                keyterm_count=args.keyterm_count,
            )
        )
        results.append(result)

    if args.output == "json":
        print(json.dumps([{**asdict(r), "rtf": r.rtf} for r in results], indent=2))
    elif args.output == "csv":
        fields = [*asdict(results[0]).keys(), "rtf"] if results else []
        print(",".join(fields))
        for r in results:
            row = {**asdict(r), "rtf": r.rtf}
            print(",".join(str(row[f]) for f in fields))
    else:
        for result in results:
            _print_text(result)

    return 0


if __name__ == "__main__":
    sys.exit(main())
