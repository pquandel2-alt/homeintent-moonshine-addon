"""Local benchmark CLI: measures real-time factor (RTF) and latency for
ANY of this add-on's five STT engines against a user-supplied WAV file,
driven entirely through the shared app/stt_engine.py ``SttEngine``/
``SttSession`` abstraction -- one code path for all five engines, not five
separate ad-hoc scripts (app/benchmark.py, this module's Moonshine-only
predecessor, is kept unchanged for backward compatibility but is now a
special case of what this module does generically).

Not run in CI and not a source of any RTF/latency numbers quoted in
README/DOCS -- every engine's performance depends heavily on the host CPU
(GitHub Actions runners are NOT representative of real Home Assistant
hardware, e.g. an Intel Core i5-12450H), so the only honest numbers are
ones users measure themselves with this tool, on their own target machine.

What this tool measures, honestly, and what it does NOT:
- Audio duration, wall/model inference time, finalize ("final result")
  latency, and RTF (inference time / audio duration) -- all real,
  measured quantities from the engine's own SttSession counters (the same
  counters app/handler.py's own production performance logging uses).
- Time-to-first-partial-result: only reported for an engine whose
  ``capabilities.supports_partial_results`` is True. As of this add-on's
  own Wyoming surface, NONE of the five engines currently report this
  (see app/stt_engine.py's SttCapabilities docstring for why -- this
  add-on's handler never sends a Wyoming partial-transcript event for any
  engine today, even where the underlying library could support it) -- so
  this column honestly prints "n/a (no partial results)" instead of a
  fabricated number.
- The raw transcript text, for manual inspection. This tool deliberately
  does NOT compute a Word Error Rate: no ground-truth reference transcript
  mechanism exists anywhere in this add-on's test fixtures, and inventing
  a WER number with no real reference corpus would be dishonest. Read the
  printed transcript yourself against what you actually said.

Usage:
    python -m app.stt_benchmark path/to/sample.wav --engine moonshine
    python -m app.stt_benchmark path/to/sample.wav --engine kroko
    python -m app.stt_benchmark path/to/sample.wav --engine vosk_german --output json
    python -m app.stt_benchmark path/to/sample.wav \
        --engine speechcatcher_m --output csv
"""

import argparse
import asyncio
import json
import logging
import sys
import time
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import TYPE_CHECKING

from app.audio import pcm_int16_to_float32
from app.benchmark import chunk_pcm, read_wav_pcm

if TYPE_CHECKING:
    from app.stt_engine import SttEngine

_LOGGER = logging.getLogger(__name__)

ALL_ENGINES = ("moonshine", "kroko", "speechcatcher_m", "speechcatcher_l", "vosk_german")


@dataclass
class SttBenchmarkResult:
    engine: str
    model_load_s: float
    audio_duration_s: float
    processing_time_s: float
    inference_time_s: float
    finalize_time_s: float  # "final result latency after audio end"
    chunk_count: int
    add_audio_compute_max_s: float
    average_chunk_compute_s: float
    supports_partial_results: bool
    time_to_first_partial_s: float | None  # None when not supported
    transcript: str

    @property
    def rtf(self) -> float:
        """Real-time factor from real engine compute time (add_audio +
        finalize), matching the same definition every other benchmark/
        production log in this add-on uses."""
        if self.audio_duration_s == 0:
            return float("inf")
        return self.inference_time_s / self.audio_duration_s


def _build_moonshine_engine(model: str, language: str) -> "SttEngine":
    from app.models import load_transcriber
    from app.moonshine_engine import MoonshineSttEngine

    transcriber = load_transcriber(model=model, language=language)
    return MoonshineSttEngine(transcriber, model, language)


def _build_kroko_engine() -> "SttEngine":
    from app.kroko_engine import load_kroko_engine

    return load_kroko_engine()


def _build_speechcatcher_engine(engine_id: str) -> "SttEngine":
    from app.speechcatcher_engine import load_speechcatcher_engine

    return load_speechcatcher_engine(engine_id)


def _build_vosk_engine() -> "SttEngine":
    from app.vosk_engine import load_vosk_engine

    return load_vosk_engine()


def build_engine(engine_id: str, model: str, language: str) -> "SttEngine":
    """Load ONE engine (downloading its model on first use if needed),
    dispatching through the exact same functions app/__main__.py's own
    startup dispatch uses -- never a separate, potentially-diverging
    loading path."""
    if engine_id == "kroko":
        return _build_kroko_engine()
    if engine_id in ("speechcatcher_m", "speechcatcher_l"):
        return _build_speechcatcher_engine(engine_id)
    if engine_id == "vosk_german":
        return _build_vosk_engine()
    return _build_moonshine_engine(model, language)


async def run_benchmark(
    engine: "SttEngine",
    pcm_bytes: bytes,
    sample_rate: int,
    chunk_ms: int,
) -> SttBenchmarkResult:
    """Feed a WAV's PCM through one engine's SttSession as fast as
    possible and time it -- identical driving logic for every engine,
    since app/handler.py itself never branches on which engine is loaded
    (see app/stt_engine.py's module docstring)."""
    chunk_bytes = int(sample_rate * (chunk_ms / 1000) * 2)
    session = engine.create_session(lock=asyncio.Lock())

    first_partial_at: float | None = None
    start = time.perf_counter()
    await session.start()
    for chunk in chunk_pcm(pcm_bytes, chunk_bytes):
        samples = pcm_int16_to_float32(chunk).tolist()
        await session.add_audio(samples, sample_rate)
        if engine.capabilities.supports_partial_results and first_partial_at is None:
            # No engine in this add-on currently exposes a real
            # incremental "give me your current best guess" call through
            # this shared abstraction (see this module's docstring) -- if
            # a future engine does add supports_partial_results=True, this
            # is where its actual partial-fetch call would go. Left
            # unimplemented (not faked) so it fails loudly instead of
            # reporting a fabricated timestamp.
            raise NotImplementedError(
                f"engine {engine.engine_id} reports supports_partial_results=True but "
                "app/stt_benchmark.py has no partial-result measurement wired up for it yet"
            )
    transcript = await session.finalize()
    processing_time_s = time.perf_counter() - start
    session.close()

    audio_duration_s = (len(pcm_bytes) / 2) / sample_rate
    return SttBenchmarkResult(
        engine=engine.engine_id,
        model_load_s=0.0,  # filled in by the caller, which times the load separately
        audio_duration_s=audio_duration_s,
        processing_time_s=processing_time_s,
        inference_time_s=session.inference_time_seconds,
        finalize_time_s=session.finalize_time_seconds,
        chunk_count=session.add_audio_chunk_count,
        add_audio_compute_max_s=session.add_audio_compute_max_seconds,
        average_chunk_compute_s=session.average_chunk_compute_seconds,
        supports_partial_results=engine.capabilities.supports_partial_results,
        time_to_first_partial_s=first_partial_at,
        transcript=transcript,
    )


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Benchmark STT streaming performance on this machine using a local "
            "16kHz/mono/16-bit WAV file, for any of this add-on's five STT "
            "engines. Run this ON the actual target hardware (e.g. the real "
            "Home Assistant host) -- numbers from a development machine or CI "
            "runner are not representative."
        )
    )
    parser.add_argument("wav_file", type=Path, help="Path to a 16kHz mono 16-bit WAV file")
    parser.add_argument(
        "--engine",
        choices=ALL_ENGINES,
        default="moonshine",
        help=f"Which STT engine to benchmark, one of {ALL_ENGINES} (default: moonshine)",
    )
    parser.add_argument(
        "--model",
        choices=["tiny", "small"],
        default="small",
        help="Moonshine model size (only used when --engine=moonshine)",
    )
    parser.add_argument("--language", default="de")
    parser.add_argument(
        "--chunk-ms",
        type=int,
        default=30,
        help="Simulated Wyoming audio-chunk size in milliseconds (default: 30)",
    )
    parser.add_argument(
        "--output",
        choices=["text", "json", "csv"],
        default="text",
        help="Output format",
    )
    return parser


def _print_text(result: SttBenchmarkResult) -> None:
    print(f"--- engine={result.engine} ---")
    print(f"Model load time:             {result.model_load_s:.2f}s")
    print(f"Audio duration:              {result.audio_duration_s:.2f}s")
    print(f"Wall processing time:        {result.processing_time_s:.2f}s")
    print(f"Model inference time:        {result.inference_time_s:.2f}s")
    print(f"Final result latency:        {result.finalize_time_s:.3f}s")
    print(f"Chunks:                      {result.chunk_count}")
    print(f"Slowest chunk:               {result.add_audio_compute_max_s:.3f}s")
    print(f"Average chunk compute:       {result.average_chunk_compute_s:.3f}s")
    if result.supports_partial_results and result.time_to_first_partial_s is not None:
        print(f"Time to first partial:       {result.time_to_first_partial_s:.3f}s")
    else:
        print("Time to first partial:       n/a (no partial results)")
    print(
        f"Real-time factor:            {result.rtf:.3f} "
        f"({'faster' if result.rtf < 1 else 'slower'} than real time)"
    )
    print(f"Transcript:                  {result.transcript}")


def main(argv: list[str] | None = None) -> int:
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")
    parser = build_arg_parser()
    args = parser.parse_args(argv)

    if not args.wav_file.exists():
        _LOGGER.error("File not found: %s", args.wav_file)
        return 1

    try:
        pcm_bytes, rate, _width, _channels = read_wav_pcm(args.wav_file)
    except ValueError as e:
        _LOGGER.error("Invalid WAV file: %s", e)
        return 1

    _LOGGER.info(
        "Loading engine=%s (first run downloads its model; this can take a while)...",
        args.engine,
    )
    load_started = time.perf_counter()
    try:
        engine = build_engine(args.engine, args.model, args.language)
    except Exception as e:
        _LOGGER.error("Failed to load engine %s: %s", args.engine, e)
        return 1
    model_load_s = time.perf_counter() - load_started

    _LOGGER.info(
        "Benchmarking %s (engine=%s chunk_ms=%d)...", args.wav_file, args.engine, args.chunk_ms
    )
    result = asyncio.run(run_benchmark(engine, pcm_bytes, rate, args.chunk_ms))
    result.model_load_s = model_load_s

    if args.output == "json":
        print(json.dumps({**asdict(result), "rtf": result.rtf}, indent=2))
    elif args.output == "csv":
        fields = [*asdict(result).keys(), "rtf"]
        print(",".join(fields))
        row = {**asdict(result), "rtf": result.rtf}
        print(",".join(str(row[f]) for f in fields))
    else:
        _print_text(result)

    return 0


if __name__ == "__main__":
    sys.exit(main())
