"""Tests for the Pocket TTS thread-tuning benchmark CLI (app/tts_benchmark.py).

Only the hermetic pieces (argument parsing, RTF math) -- actually running a
benchmark loads a real Pocket TTS model in a subprocess and is a local,
manual tool (see its module docstring), not something the default test
suite should depend on.
"""

from app.tts_benchmark import DEFAULT_THREADS, TtsThreadBenchmarkResult, build_arg_parser


class TestArgParsing:
    def test_default_threads_include_the_requested_sweep(self):
        args = build_arg_parser().parse_args([])
        assert args.threads == DEFAULT_THREADS
        assert args.threads.split(",") == ["1", "2", "4", "6", "8", "auto"]

    def test_custom_threads(self):
        args = build_arg_parser().parse_args(["--threads", "2,8"])
        assert args.threads == "2,8"

    def test_single_run_flag_defaults_to_none(self):
        args = build_arg_parser().parse_args([])
        assert args._single_run is None

    def test_single_run_flag_parsed(self):
        args = build_arg_parser().parse_args(["--_single-run", "4"])
        assert args._single_run == "4"

    def test_default_output_is_text(self):
        args = build_arg_parser().parse_args([])
        assert args.output == "text"


class TestRtf:
    def test_rtf_is_model_compute_over_audio_duration(self):
        result = TtsThreadBenchmarkResult(
            threads="4", ttfa_s=0.2, model_compute_s=1.0, audio_duration_s=2.0, total_s=1.3
        )
        assert result.rtf == 0.5

    def test_rtf_infinite_when_no_audio_produced(self):
        result = TtsThreadBenchmarkResult(
            threads="auto", ttfa_s=0.0, model_compute_s=0.0, audio_duration_s=0.0, total_s=0.1
        )
        assert result.rtf == float("inf")
