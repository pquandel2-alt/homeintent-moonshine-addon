"""Local benchmark CLI: measures TTS time-to-first-audio and real-time
factor for either engine (Pocket TTS or Kokoro ONNX) across different
thread-count settings, on the user's own hardware.

Not run in CI and not a source of any TTFA/RTF/RAM numbers quoted in
README/DOCS -- both engines' performance depends heavily on the host CPU
(GitHub Actions runners are NOT representative of real Home Assistant
hardware, e.g. an Intel Core i5-12450H), so the only honest numbers are
ones users measure themselves with this tool, on their own target
machine, before choosing a non-default thread setting or switching
``tts_engine``.

Each requested thread count is benchmarked in its own, freshly-started
subprocess (never by changing a thread setting repeatedly in one
long-lived process): both PyTorch's own docs (see app/tts.py's
``load_tts_model()`` docstring) and this add-on's own onnxruntime
``SessionOptions`` usage (app/kokoro_tts.py) only reliably honor a thread
count set before any parallel/eager work has run in the process, so
reusing one process across configurations would risk measuring whichever
value happened to be set first, not each one independently. This also
gives a real, per-configuration model LOAD time and peak RSS measurement,
which a shared process could not.

Usage:
    python -m app.tts_benchmark --engine pocket_tts
    python -m app.tts_benchmark --engine kokoro_onnx --threads 1,2,4,6,8,auto
    python -m app.tts_benchmark --engine kokoro_onnx --output json
    python -m app.tts_benchmark --engine pocket_tts --model german_24l \
        --text "Das Küchenfenster und das Schlafzimmerfenster sind geöffnet."
"""

import argparse
import asyncio
import json
import logging
import resource
import subprocess
import sys
import time
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from app.tts_engine import TtsSynthesizer

_LOGGER = logging.getLogger(__name__)

DEFAULT_THREADS = "1,2,4,6,8,auto"

# Item 19's four required benchmark sentences -- deliberately include
# German-specific formatting (a decimal temperature, a multi-item list, a
# time) that exercises different parts of each engine's own text handling.
GERMAN_TEST_SENTENCES = (
    "Das Küchenfenster ist geöffnet.",
    "Im Wohnzimmer sind 21,5 Grad Celsius.",
    "Das Küchenfenster, das Bürofenster und das Schlafzimmerfenster sind geöffnet.",
    "Die Waschmaschine ist um 18:20 Uhr fertig.",
)


@dataclass
class TtsThreadBenchmarkResult:
    engine: str  # "pocket_tts" or "kokoro_onnx"
    threads: str  # the requested value ("auto" or a positive integer as str)
    sentence: str
    model_load_s: float
    warmup_s: float
    ttfa_s: float
    first_chunk_duration_s: float
    model_compute_s: float
    audio_duration_s: float
    total_s: float
    peak_rss_kb: int

    @property
    def rtf(self) -> float:
        """Real-time factor from real model compute time, matching the
        same definition app/handler.py's production TTS log uses
        (``model_rtf``)."""
        if self.audio_duration_s <= 0:
            return float("inf")
        return self.model_compute_s / self.audio_duration_s


async def _build_pocket_synthesizer(
    threads: str, model: str, voice: str, cache_dir: Path | None
) -> "TtsSynthesizer":
    from app.tts import load_tts_model
    from app.tts_session import PocketTtsSynthesizer

    num_threads = 0 if threads == "auto" else int(threads)
    tts_model = load_tts_model(model=model, cache_dir=cache_dir, num_threads=num_threads)
    synthesizer = PocketTtsSynthesizer(tts_model, default_voice=voice)
    await synthesizer.preload_default_voice()
    return synthesizer


async def _build_kokoro_synthesizer(
    threads: str, voice: str, cache_dir: Path | None
) -> "TtsSynthesizer":
    from app.kokoro_session import KokoroOnnxSynthesizer
    from app.kokoro_tts import load_kokoro_model

    num_threads = 0 if threads == "auto" else int(threads)
    kokoro_model = load_kokoro_model(cache_dir=cache_dir, intra_op_num_threads=num_threads)
    synthesizer = KokoroOnnxSynthesizer(kokoro_model, default_voice=voice)
    await synthesizer.preload_default_voice()
    return synthesizer


async def _measure_all_sentences(
    engine: str,
    threads: str,
    sentences: tuple[str, ...],
    model: str,
    voice: str,
    cache_dir: Path | None,
) -> list[TtsThreadBenchmarkResult]:
    """Load one engine/thread configuration once, then measure every
    sentence against that single loaded instance. Meant to run in its own
    subprocess -- see module docstring."""
    load_started = time.monotonic()
    if engine == "kokoro_onnx":
        synthesizer = await _build_kokoro_synthesizer(threads, voice, cache_dir)
    else:
        synthesizer = await _build_pocket_synthesizer(threads, model, voice, cache_dir)
    model_load_s = time.monotonic() - load_started

    warmup_started = time.monotonic()
    async for _ in synthesizer.synthesize_stream("Eins, zwei, drei.", voice):
        pass
    warmup_s = time.monotonic() - warmup_started

    results = []
    for sentence in sentences:
        total_samples = 0
        first_chunk_at: float | None = None
        first_chunk_samples = 0
        stats_started = time.monotonic()

        from app.tts_engine import TtsSynthesisStats

        stats = TtsSynthesisStats()
        started = time.monotonic()
        async for chunk in synthesizer.synthesize_stream(sentence, voice, stats):
            if first_chunk_at is None:
                first_chunk_at = time.monotonic()
                first_chunk_samples = len(chunk)
            total_samples += len(chunk)
        total_s = time.monotonic() - started
        _ = stats_started  # kept for clarity; not otherwise used

        audio_duration_s = total_samples / synthesizer.sample_rate
        first_chunk_duration_s = first_chunk_samples / synthesizer.sample_rate
        ttfa_s = (first_chunk_at - started) if first_chunk_at is not None else 0.0

        # ru_maxrss is peak resident set size in KB on Linux (KB*1024 on
        # some other platforms -- this add-on only ships/runs on Linux).
        peak_rss_kb = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss

        results.append(
            TtsThreadBenchmarkResult(
                engine=engine,
                threads=threads,
                sentence=sentence,
                model_load_s=model_load_s,
                warmup_s=warmup_s,
                ttfa_s=ttfa_s,
                first_chunk_duration_s=first_chunk_duration_s,
                model_compute_s=stats.model_generation_seconds,
                audio_duration_s=audio_duration_s,
                total_s=total_s,
                peak_rss_kb=peak_rss_kb,
            )
        )
    return results


def _run_in_subprocess(
    engine: str,
    threads: str,
    sentences: tuple[str, ...],
    model: str,
    voice: str,
    cache_dir: Path | None,
) -> list[TtsThreadBenchmarkResult]:
    """Run _measure_all_sentences() for one thread count in a fresh
    subprocess and parse its printed JSON result -- see module docstring
    for why each value needs its own process."""
    argv = [
        sys.executable,
        "-m",
        "app.tts_benchmark",
        "--_single-run",
        threads,
        "--engine",
        engine,
        "--model",
        model,
        "--voice",
        voice,
    ]
    for sentence in sentences:
        argv.extend(["--sentence", sentence])
    if cache_dir is not None:
        argv.extend(["--cache-dir", str(cache_dir)])

    proc = subprocess.run(argv, capture_output=True, text=True, check=True)
    last_line = proc.stdout.strip().splitlines()[-1]
    data = json.loads(last_line)
    return [TtsThreadBenchmarkResult(**row) for row in data]


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Benchmark TTS time-to-first-audio and real-time factor across "
            "different thread counts on this machine, for either engine. Run "
            "this ON the actual target hardware -- numbers from a "
            "development machine or CI runner are not representative."
        )
    )
    parser.add_argument(
        "--engine",
        choices=["pocket_tts", "kokoro_onnx"],
        default="pocket_tts",
        help="Which TTS engine to benchmark (default: pocket_tts)",
    )
    parser.add_argument(
        "--threads",
        default=DEFAULT_THREADS,
        help=(
            "Comma-separated thread counts to benchmark, e.g. '1,2,4,6,8,auto' "
            f"(default: {DEFAULT_THREADS}). 'auto' leaves the engine's own "
            "default untouched, matching tts_threads/kokoro_threads: 0."
        ),
    )
    parser.add_argument(
        "--sentence",
        dest="sentences",
        action="append",
        default=None,
        help="Sentence to synthesize; repeatable. Defaults to the 4 standard "
        "German test sentences (item 19) if omitted.",
    )
    parser.add_argument("--text", dest="sentences_legacy", default=None, help=argparse.SUPPRESS)
    parser.add_argument(
        "--model", default="german", help="Pocket TTS model: 'german' or 'german_24l'"
    )
    parser.add_argument("--voice", default=None, help="Voice name (defaults per engine)")
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
    print(f"--- engine={result.engine} threads={result.threads} ---")
    print(f"Sentence:              {result.sentence!r}")
    print(f"Model load time:       {result.model_load_s:.2f}s")
    print(f"Warmup time:           {result.warmup_s:.2f}s")
    print(f"Time to first audio:   {result.ttfa_s:.3f}s")
    print(f"First chunk duration:  {result.first_chunk_duration_s:.3f}s")
    print(f"Model compute time:    {result.model_compute_s:.2f}s")
    print(f"Synthesized audio:     {result.audio_duration_s:.2f}s")
    print(f"Total wall time:       {result.total_s:.2f}s")
    print(f"Peak RSS:              {result.peak_rss_kb / 1024:.1f} MB")
    print(
        f"Real-time factor:      {result.rtf:.3f} "
        f"({'faster' if result.rtf < 1 else 'slower'} than real time)"
    )


def main(argv: list[str] | None = None) -> int:
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")
    parser = build_arg_parser()
    args = parser.parse_args(argv)

    default_voice = "martin" if args.engine == "kokoro_onnx" else "juergen"
    voice = args.voice or default_voice

    sentences = tuple(args.sentences) if args.sentences else GERMAN_TEST_SENTENCES
    if args.sentences_legacy:
        sentences = (args.sentences_legacy,)

    if args._single_run is not None:
        # Internal entry point: one isolated subprocess, one configuration,
        # every requested sentence.
        results = asyncio.run(
            _measure_all_sentences(
                args.engine, args._single_run, sentences, args.model, voice, args.cache_dir
            )
        )
        print(json.dumps([asdict(r) for r in results]))
        return 0

    thread_values = [v.strip() for v in args.threads.split(",") if v.strip()]
    all_results: list[TtsThreadBenchmarkResult] = []
    for value in thread_values:
        _LOGGER.info("Benchmarking engine=%s threads=%s...", args.engine, value)
        try:
            results = _run_in_subprocess(
                args.engine, value, sentences, args.model, voice, args.cache_dir
            )
        except subprocess.CalledProcessError as err:
            _LOGGER.error(
                "Benchmark subprocess failed for threads=%s: %s", value, err.stderr.strip()
            )
            return 1
        all_results.extend(results)

    if args.output == "json":
        print(json.dumps([{**asdict(r), "rtf": r.rtf} for r in all_results], indent=2))
    elif args.output == "csv":
        fields = [*asdict(all_results[0]).keys(), "rtf"] if all_results else []
        print(",".join(fields))
        for r in all_results:
            row = {**asdict(r), "rtf": r.rtf}
            print(",".join(str(row[f]) for f in fields))
    else:
        for result in all_results:
            _print_text(result)

    return 0


if __name__ == "__main__":
    sys.exit(main())
