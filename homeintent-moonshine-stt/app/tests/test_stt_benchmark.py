"""Tests for the cross-engine STT benchmark CLI (app/stt_benchmark.py).

Covers argument parsing/RTF math (hermetic) and a full run_benchmark()
pass against a fake, in-memory SttEngine/SttSession pair (no real model
download, no real moonshine-voice/sherpa-onnx/vosk install needed) --
proving the SAME driving code genuinely works against the shared
SttEngine abstraction, not a per-engine special case. Actually loading a
real model (build_engine()) is a local, manual tool step (see the
module's own docstring) and not something the default test suite depends
on; test_build_engine_dispatch below only checks that each --engine value
routes to the correct loader function, using monkeypatched loaders.
"""

import asyncio
import wave
from pathlib import Path

import numpy as np
import pytest

from app.audio import float32_to_pcm_int16
from app.benchmark import read_wav_pcm
from app.stt_benchmark import (
    ALL_ENGINES,
    SttBenchmarkResult,
    build_arg_parser,
    run_benchmark,
)
from app.stt_engine import SttCapabilities


class _FakeSession:
    def __init__(self, final_text: str) -> None:
        self.audio_duration_seconds = 0.0
        self._chunks = 0
        self._final_text = final_text
        self.finalize_time_seconds = 0.02
        self.inference_time_seconds = 0.05

    async def start(self) -> None:
        pass

    async def add_audio(self, samples: list[float], sample_rate: int = 16000) -> None:
        self._chunks += 1
        self.audio_duration_seconds += len(samples) / sample_rate

    async def finalize(self) -> str:
        return self._final_text

    def close(self) -> None:
        pass

    @property
    def add_audio_chunk_count(self) -> int:
        return self._chunks

    @property
    def add_audio_compute_total_seconds(self) -> float:
        return 0.05

    @property
    def add_audio_compute_max_seconds(self) -> float:
        return 0.01

    @property
    def average_chunk_compute_seconds(self) -> float:
        return 0.05 / self._chunks if self._chunks else 0.0


class _FakeEngine:
    def __init__(self, engine_id: str, final_text: str = "licht an") -> None:
        self._engine_id = engine_id
        self._final_text = final_text

    @property
    def engine_id(self) -> str:
        return self._engine_id

    @property
    def capabilities(self) -> SttCapabilities:
        return SttCapabilities(
            supports_streaming=True,
            supports_hotwords=False,
            supports_dynamic_vocabulary=False,
            supports_partial_results=False,
        )

    def create_session(self, lock: object | None = None) -> _FakeSession:
        return _FakeSession(self._final_text)

    @property
    def program_name(self) -> str:
        return f"fake-{self._engine_id}"

    @property
    def model_display_name(self) -> str:
        return f"fake-{self._engine_id}-model"

    @property
    def description(self) -> str:
        return "fake engine for testing"

    @property
    def attribution_name(self) -> str:
        return "Fake"

    @property
    def attribution_url(self) -> str:
        return "https://example.invalid"

    def set_keyterms(
        self, terms: list[str], fallback_terms: list[str] | None = None
    ) -> tuple[list[str], bool]:
        return [], True


def _write_wav(path: Path, seconds: float = 0.5, rate: int = 16000) -> None:
    samples = np.zeros(int(seconds * rate), dtype=np.float32)
    pcm = float32_to_pcm_int16(samples)
    with wave.open(str(path), "wb") as wf:
        wf.setnchannels(1)
        wf.setsampwidth(2)
        wf.setframerate(rate)
        wf.writeframes(pcm)


class TestArgParsing:
    def test_default_engine_is_moonshine(self) -> None:
        args = build_arg_parser().parse_args([Path("x.wav").as_posix()])
        assert args.engine == "moonshine"

    def test_all_five_engines_accepted(self) -> None:
        for engine in ALL_ENGINES:
            args = build_arg_parser().parse_args(["x.wav", "--engine", engine])
            assert args.engine == engine

    def test_invalid_engine_rejected(self) -> None:
        with pytest.raises(SystemExit):
            build_arg_parser().parse_args(["x.wav", "--engine", "not_a_real_engine"])

    def test_default_output_is_text(self) -> None:
        args = build_arg_parser().parse_args(["x.wav"])
        assert args.output == "text"

    def test_all_five_engines_constant_matches_stt_engines(self) -> None:
        assert ALL_ENGINES == (
            "moonshine",
            "kroko",
            "speechcatcher_m",
            "speechcatcher_l",
            "vosk_german",
        )


class TestRtf:
    def test_rtf_is_inference_time_over_audio_duration(self) -> None:
        result = SttBenchmarkResult(
            engine="moonshine",
            model_load_s=0.0,
            audio_duration_s=2.0,
            processing_time_s=1.0,
            inference_time_s=1.0,
            finalize_time_s=0.1,
            chunk_count=5,
            add_audio_compute_max_s=0.1,
            average_chunk_compute_s=0.1,
            supports_partial_results=False,
            time_to_first_partial_s=None,
            transcript="",
        )
        assert result.rtf == 0.5

    def test_rtf_infinite_when_no_audio(self) -> None:
        result = SttBenchmarkResult(
            engine="moonshine",
            model_load_s=0.0,
            audio_duration_s=0.0,
            processing_time_s=0.0,
            inference_time_s=0.0,
            finalize_time_s=0.0,
            chunk_count=0,
            add_audio_compute_max_s=0.0,
            average_chunk_compute_s=0.0,
            supports_partial_results=False,
            time_to_first_partial_s=None,
            transcript="",
        )
        assert result.rtf == float("inf")


class TestRunBenchmarkAgainstFakeEngine:
    """Proves run_benchmark() drives ANY SttEngine identically -- the same
    function, unmodified, is used regardless of which of the five real
    engines is passed in at runtime (app/__main__.py's own dispatch
    picks the concrete engine; this test substitutes a fake one)."""

    @pytest.mark.parametrize("engine_id", list(ALL_ENGINES))
    def test_run_benchmark_works_uniformly_for_every_engine_id(
        self, tmp_path: Path, engine_id: str
    ) -> None:
        wav_path = tmp_path / "sample.wav"
        _write_wav(wav_path, seconds=0.3)

        pcm_bytes, rate, _w, _c = read_wav_pcm(wav_path)

        fake_engine = _FakeEngine(engine_id, final_text="schalte das licht ein")
        result = asyncio.run(run_benchmark(fake_engine, pcm_bytes, rate, chunk_ms=30))

        assert result.engine == engine_id
        assert result.transcript == "schalte das licht ein"
        assert result.audio_duration_s == pytest.approx(0.3, abs=0.02)
        assert result.chunk_count > 0
        assert result.supports_partial_results is False
        assert result.time_to_first_partial_s is None


class TestBuildEngineDispatch:
    """Only the requested --engine's own loader is ever called -- mirrors
    app/tests/test_engine_dispatch.py's guarantee for app/__main__.py's
    startup dispatch, applied to this benchmark tool's own dispatch."""

    def _patch_all_builders(self, monkeypatch, called: dict[str, bool]) -> None:
        import app.stt_benchmark as mod

        def _kroko() -> object:
            called["kroko"] = True
            return object()

        def _moonshine(*a: object, **k: object) -> object:
            called["moonshine"] = True
            return object()

        def _speechcatcher(*a: object, **k: object) -> object:
            called["speechcatcher"] = True
            return object()

        def _vosk() -> object:
            called["vosk"] = True
            return object()

        monkeypatch.setattr(mod, "_build_kroko_engine", _kroko)
        monkeypatch.setattr(mod, "_build_moonshine_engine", _moonshine)
        monkeypatch.setattr(mod, "_build_speechcatcher_engine", _speechcatcher)
        monkeypatch.setattr(mod, "_build_vosk_engine", _vosk)

    def test_kroko_dispatch_calls_only_kroko_loader(self, monkeypatch) -> None:
        import app.stt_benchmark as mod

        called = {"kroko": False, "moonshine": False, "speechcatcher": False, "vosk": False}
        self._patch_all_builders(monkeypatch, called)

        mod.build_engine("kroko", "small", "de")
        assert called == {
            "kroko": True,
            "moonshine": False,
            "speechcatcher": False,
            "vosk": False,
        }

    def test_vosk_dispatch_calls_only_vosk_loader(self, monkeypatch) -> None:
        import app.stt_benchmark as mod

        called = {"kroko": False, "moonshine": False, "speechcatcher": False, "vosk": False}
        self._patch_all_builders(monkeypatch, called)

        mod.build_engine("vosk_german", "small", "de")
        assert called == {
            "kroko": False,
            "moonshine": False,
            "speechcatcher": False,
            "vosk": True,
        }
