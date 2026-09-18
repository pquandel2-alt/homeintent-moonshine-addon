"""Local benchmark CLI: measures Pocket TTS time-to-first-audio and
real-time factor across different ``torch.set_num_threads()`` values on
the user's own hardware.

Not run in CI and not a source of any TTFA/RTF numbers quoted in
README/DOCS -- Pocket TTS's performance depends heavily on the host CPU
(GitHub Actions runners are NOT representative of real Home Assistant
hardware, e.g. an Intel Core i5-12450H), so the only honest numbers are
ones users measure themselves with this tool, on their own target
machine, before choosing a non-default ``tts_threads`` value.

Each requested thread count is benchmarked in its own, freshly-started
subprocess (never by calling ``torch.set_num_threads()`` repeatedly in one
long-lived process): PyTorch's own docs and this add-on's
``load_tts_model()`` docstring (app/tts.py) both note that the setting is
only reliably honored before any parallel/eager work has run in the
process, so reusing one process across configurations would risk
measuring whichever value happened to be set first, not each one
independently.

Usage:
    python -m app.tts_benchmark
    python -m app.tts_benchmark --threads 1,2,4,6,8,auto --output json
    python -m app.tts_benchmark --threads 4,8 --model german_24l \
        --text "Das Küchenfenster und das Schlafzimmerfenster sind geöffnet."
"""

import argparse
import asyncio
import json
import logging
import subprocess
import sys
import time
from dataclasses import asdict, dataclass
from pathlib import Path

_LOGGER = logging.getLogger(__name__)

DEFAULT_TEXT = "Das Küchenfenster und das Schlafzimmerfenster sind geöffnet."
DEFAULT_THREADS = "1,2,4,6,8,auto"


@dataclass
class TtsThreadBenchmarkResult:
    threads: str  # the requested value ("auto" or a positive integer as str)
    ttfa_s: float
    model_compute_s: float
    audio_duration_s: float
    total_s: float

    @property
    def rtf(self) -> float:
        """Real-time factor from real model compute time, matching the
        same definition app/handler.py's production TTS log uses
        (``model_rtf``)."""
        if self.audio_duration_s <= 0:
            return float("inf")
        return self.model_compute_s / self.audio_duration_s


async def _measure_one(
    threads: str, text: str, model: str, voice: str, cache_dir: Path | None
) -> TtsThreadBenchmarkResult:
    """Load a Pocket TTS model with the given thread setting and measure
    one synthesis of ``text``. Meant to run in its own subprocess -- see
    module docstring."""
    from app.tts import load_tts_model
    from app.tts_session import PocketTtsSynthesizer, TtsSynthesisStats

    num_threads = 0 if threads == "auto" else int(threads)
    tts_model = load_tts_model(model=model, cache_dir=cache_dir, num_threads=num_threads)
    synthesizer = PocketTtsSynthesizer(tts_model, default_voice=voice)
    await synthesizer.preload_default_voice()

    stats = TtsSynthesisStats()
    total_samples = 0
    first_chunk_at: float | None = None
    started = time.monotonic()
    async for chunk in synthesizer.synthesize_stream(text, voice, stats):
        if first_chunk_at is None:
            first_chunk_at = time.monotonic()
        total_samples += len(chunk)
    total_s = time.monotonic() - started

    audio_duration_s = total_samples / synthesizer.sample_rate
    ttfa_s = (first_chunk_at - started) if first_chunk_at is not None else 0.0
    return TtsThreadBenchmarkResult(
        threads=threads,
        ttfa_s=ttfa_s,
        model_compute_s=stats.model_generation_seconds,
        audio_duration_s=audio_duration_s,
        total_s=total_s,
    )


def _run_in_subprocess(
    threads: str, text: str, model: str, voice: str, cache_dir: Path | None
) -> TtsThreadBenchmarkResult:
    """Run _measure_one() for one thread count in a fresh subprocess and
    parse its printed JSON result -- see module docstring for why each
    value needs its own process."""
    argv = [
        sys.executable,
        "-m",
        "app.tts_benchmark",
        "--_single-run",
        threads,
        "--text",
        text,
        "--model",
        model,
        "--voice",
        voice,
    ]
    if cache_dir is not None:
        argv.extend(["--cache-dir", str(cache_dir)])

    proc = subprocess.run(argv, capture_output=True, text=True, check=True)
    last_line = proc.stdout.strip().splitlines()[-1]
    data = json.loads(last_line)
    return TtsThreadBenchmarkResult(**data)


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Benchmark Pocket TTS time-to-first-audio and real-time factor "
            "across different torch thread counts on this machine. Run this "
            "ON the actual target hardware -- numbers from a development "
            "machine or CI runner are not representative."
        )
    )
    parser.add_argument(
        "--threads",
        default=DEFAULT_THREADS,
        help=(
            "Comma-separated thread counts to benchmark, e.g. '1,2,4,6,8,auto' "
            f"(default: {DEFAULT_THREADS}). 'auto' leaves PyTorch's own "
            "default untouched, matching tts_threads: 0."
        ),
    )
    parser.add_argument("--text", default=DEFAULT_TEXT)
    parser.add_argument("--model", default="german", help="'german' or 'german_24l'")
    parser.add_argument("--voice", default="juergen")
    parser.add_argument("--cache-dir", type=Path, default=None)
    parser.add_argument(
        "--output",
        choices=["text", "json", "csv"],
        default="text",
        help="Output format",
    )
    parser.add_argument(
        "--_single-run",
        metavar="THREADS",
        help=argparse.SUPPRESS,  # internal: run exactly one configuration, print JSON, exit
    )
    return parser


def _print_text(result: TtsThreadBenchmarkResult) -> None:
    print(f"--- threads={result.threads} ---")
    print(f"Time to first audio:  {result.ttfa_s:.3f}s")
    print(f"Model compute time:   {result.model_compute_s:.2f}s")
    print(f"Synthesized audio:    {result.audio_duration_s:.2f}s")
    print(f"Total wall time:      {result.total_s:.2f}s")
    print(
        f"Real-time factor:     {result.rtf:.3f} "
        f"({'faster' if result.rtf < 1 else 'slower'} than real time)"
    )


def main(argv: list[str] | None = None) -> int:
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")
    parser = build_arg_parser()
    args = parser.parse_args(argv)

    if args._single_run is not None:
        # Internal entry point: one isolated subprocess, one configuration.
        result = asyncio.run(
            _measure_one(args._single_run, args.text, args.model, args.voice, args.cache_dir)
        )
        print(json.dumps(asdict(result)))
        return 0

    thread_values = [v.strip() for v in args.threads.split(",") if v.strip()]
    results: list[TtsThreadBenchmarkResult] = []
    for value in thread_values:
        _LOGGER.info("Benchmarking tts_threads=%s (model=%s)...", value, args.model)
        try:
            result = _run_in_subprocess(value, args.text, args.model, args.voice, args.cache_dir)
        except subprocess.CalledProcessError as err:
            _LOGGER.error(
                "Benchmark subprocess failed for threads=%s: %s", value, err.stderr.strip()
            )
            return 1
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
