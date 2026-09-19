"""Tests for the TTS thread-tuning benchmark CLI (app/tts_benchmark.py).

Only the hermetic pieces (argument parsing, RTF math) -- actually running a
benchmark loads a real TTS model in a subprocess and is a local, manual tool
(see its module docstring), not something the default test suite should
depend on.
"""

from app.tts_benchmark import (
    DEFAULT_THREADS,
    GERMAN_TEST_SENTENCES,
    TtsThreadBenchmarkResult,
    build_arg_parser,
)


def _make_result(
    engine: str = "pocket_tts",
    threads: str = "4",
    sentence: str = "Das Küchenfenster ist geöffnet.",
    model_load_s: float = 0.5,
    warmup_s: float = 0.1,
    ttfa_s: float = 0.2,
    first_chunk_duration_s: float = 0.05,
    model_compute_s: float = 1.0,
    audio_duration_s: float = 2.0,
    total_s: float = 1.3,
    peak_rss_kb: int = 123456,
) -> TtsThreadBenchmarkResult:
    return TtsThreadBenchmarkResult(
        engine=engine,
        threads=threads,
        sentence=sentence,
        model_load_s=model_load_s,
        warmup_s=warmup_s,
        ttfa_s=ttfa_s,
        first_chunk_duration_s=first_chunk_duration_s,
        model_compute_s=model_compute_s,
        audio_duration_s=audio_duration_s,
        total_s=total_s,
        peak_rss_kb=peak_rss_kb,
    )


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

    def test_default_engine_is_pocket_tts(self):
        args = build_arg_parser().parse_args([])
        assert args.engine == "pocket_tts"

    def test_engine_can_be_kokoro_onnx(self):
        args = build_arg_parser().parse_args(["--engine", "kokoro_onnx"])
        assert args.engine == "kokoro_onnx"

    def test_engine_can_be_supertonic_3(self):
        args = build_arg_parser().parse_args(["--engine", "supertonic_3"])
        assert args.engine == "supertonic_3"

    def test_invalid_engine_rejected(self):
        import pytest

        with pytest.raises(SystemExit):
            build_arg_parser().parse_args(["--engine", "not_a_real_engine"])

    def test_sentences_default_to_none_meaning_the_standard_set(self):
        args = build_arg_parser().parse_args([])
        assert args.sentences is None

    def test_sentence_flag_repeatable(self):
        args = build_arg_parser().parse_args(
            ["--sentence", "Erster Satz.", "--sentence", "Zweiter Satz."]
        )
        assert args.sentences == ["Erster Satz.", "Zweiter Satz."]

    def test_legacy_text_flag_still_parses(self):
        args = build_arg_parser().parse_args(["--text", "Legacy Satz."])
        assert args.sentences_legacy == "Legacy Satz."


class TestGermanTestSentences:
    def test_exactly_four_required_sentences_present(self):
        assert len(GERMAN_TEST_SENTENCES) == 4

    def test_sentences_match_spec_item_19(self):
        assert GERMAN_TEST_SENTENCES == (
            "Das Küchenfenster ist geöffnet.",
            "Im Wohnzimmer sind 21,5 Grad Celsius.",
            "Das Küchenfenster, das Bürofenster und das Schlafzimmerfenster sind geöffnet.",
            "Die Waschmaschine ist um 18:20 Uhr fertig.",
        )


class TestMeasureAllSentencesDispatch:
    """Only the requested --engine's own builder is ever called -- proves
    Supertonic 3 slots into the same _measure_all_sentences() dispatch as
    Pocket TTS/Kokoro, not a separate code path (this module's own
    docstring)."""

    def test_supertonic_dispatch_calls_only_supertonic_builder(self, monkeypatch) -> None:
        import asyncio

        import app.tts_benchmark as mod

        called = {"pocket": False, "kokoro": False, "supertonic": False}

        class _FakeSynthesizer:
            sample_rate = 24000

            async def synthesize_stream(self, text, voice, stats=None):
                import numpy as np

                yield np.zeros(100, dtype=np.float32)

        async def _fake_supertonic(threads, voice, cache_dir):
            called["supertonic"] = True
            return _FakeSynthesizer()

        async def _fake_kokoro(threads, voice, cache_dir):
            called["kokoro"] = True
            return _FakeSynthesizer()

        async def _fake_pocket(threads, model, voice, cache_dir):
            called["pocket"] = True
            return _FakeSynthesizer()

        monkeypatch.setattr(mod, "_build_supertonic_synthesizer", _fake_supertonic)
        monkeypatch.setattr(mod, "_build_kokoro_synthesizer", _fake_kokoro)
        monkeypatch.setattr(mod, "_build_pocket_synthesizer", _fake_pocket)

        asyncio.run(
            mod._measure_all_sentences("supertonic_3", "1", ("Eins.",), "german", "M1", None)
        )
        assert called == {"pocket": False, "kokoro": False, "supertonic": True}


class TestRtf:
    def test_rtf_is_model_compute_over_audio_duration(self):
        result = _make_result(model_compute_s=1.0, audio_duration_s=2.0)
        assert result.rtf == 0.5

    def test_rtf_infinite_when_no_audio_produced(self):
        result = _make_result(model_compute_s=0.0, audio_duration_s=0.0)
        assert result.rtf == float("inf")

    def test_result_carries_engine_and_sentence(self):
        result = _make_result(engine="kokoro_onnx", sentence="Testsatz.")
        assert result.engine == "kokoro_onnx"
        assert result.sentence == "Testsatz."
