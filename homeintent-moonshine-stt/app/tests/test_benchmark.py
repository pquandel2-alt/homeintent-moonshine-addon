"""Tests for the local RTF benchmark CLI helper (app/benchmark.py).

Only exercises the pure logic (WAV reading, chunking, RTF math) against a
mocked streaming session -- this tool measures real hardware performance, so
it is not run against a real model in CI.
"""

import wave
from pathlib import Path

import pytest

from app.benchmark import (
    BenchmarkResult,
    chunk_pcm,
    read_wav_pcm,
    run_benchmark,
    synthetic_keyterms,
)


def _write_wav(path: Path, num_samples: int = 1600, rate: int = 16000) -> None:
    with wave.open(str(path), "wb") as wav_file:
        wav_file.setnchannels(1)
        wav_file.setsampwidth(2)
        wav_file.setframerate(rate)
        wav_file.writeframes(b"\x00\x00" * num_samples)


class TestReadWavPcm:
    def test_reads_valid_wav(self, tmp_path):
        path = tmp_path / "sample.wav"
        _write_wav(path, num_samples=1600)

        pcm_bytes, rate, width, channels = read_wav_pcm(path)

        assert len(pcm_bytes) == 3200
        assert rate == 16000
        assert width == 2
        assert channels == 1

    def test_rejects_wrong_sample_rate(self, tmp_path):
        path = tmp_path / "wrong_rate.wav"
        _write_wav(path, rate=44100)

        with pytest.raises(ValueError, match="44100"):
            read_wav_pcm(path)


class TestChunkPcm:
    def test_splits_into_fixed_size_chunks(self):
        pcm_bytes = b"\x00\x01" * 100  # 200 bytes
        chunks = list(chunk_pcm(pcm_bytes, chunk_bytes=60))

        assert len(chunks) == 4
        assert all(len(c) <= 60 for c in chunks)
        assert b"".join(chunks) == pcm_bytes

    def test_odd_chunk_size_rounded_down_to_even(self):
        pcm_bytes = b"\x00\x01" * 10  # 20 bytes
        chunks = list(chunk_pcm(pcm_bytes, chunk_bytes=7))

        assert all(len(c) % 2 == 0 for c in chunks)

    def test_empty_input_yields_no_chunks(self):
        assert list(chunk_pcm(b"", chunk_bytes=100)) == []


def _result(**overrides: object) -> BenchmarkResult:
    defaults: dict[str, object] = {
        "model": "small",
        "transcription_interval": 0.5,
        "decode_incomplete_lines": True,
        "keyterm_count": 0,
        "audio_duration_s": 2.0,
        "processing_time_s": 1.0,
        "inference_time_s": 1.0,
        "finalize_time_s": 0.1,
        "chunk_count": 5,
        "add_audio_compute_max_s": 0.05,
        "average_chunk_compute_s": 0.02,
        "transcript": "x",
    }
    defaults.update(overrides)
    return BenchmarkResult(**defaults)  # type: ignore[arg-type]


class TestBenchmarkResult:
    def test_rtf_below_one_when_faster_than_real_time(self):
        result = _result(audio_duration_s=2.0, inference_time_s=1.0)
        assert result.rtf == 0.5

    def test_rtf_infinite_for_zero_duration_audio(self):
        result = _result(audio_duration_s=0.0, inference_time_s=1.0)
        assert result.rtf == float("inf")

    def test_rtf_uses_inference_time_not_wall_processing_time(self):
        """The wall clock (processing_time_s) includes Python/asyncio
        overhead the real production RTF metric does not -- rtf must match
        app/handler.py's own definition (inference_time / audio_duration)."""
        result = _result(audio_duration_s=2.0, processing_time_s=5.0, inference_time_s=1.0)
        assert result.rtf == 0.5


class TestSyntheticKeyterms:
    def test_generates_requested_count(self):
        assert len(synthetic_keyterms(50)) == 50

    def test_zero_count_yields_empty_list(self):
        assert synthetic_keyterms(0) == []

    def test_terms_are_unique(self):
        terms = synthetic_keyterms(100)
        assert len(set(terms)) == 100


class TestRunBenchmark:
    @pytest.mark.asyncio
    async def test_computes_duration_and_transcript(self, mock_transcriber, mock_stream):
        def fake_stop():
            for call in mock_stream.add_listener.call_args_list:
                listener = call.args[0]
                from moonshine_voice import LineCompleted, TranscriptLine

                listener.on_line_completed(
                    LineCompleted(
                        line=TranscriptLine(
                            text="hallo welt",
                            start_time=0.0,
                            duration=1.0,
                            line_id=1,
                            is_complete=True,
                            is_updated=True,
                        ),
                        stream_handle=1,
                    )
                )

        mock_stream.stop.side_effect = fake_stop
        pcm_bytes = b"\x00\x00" * 1600  # 0.1s at 16kHz

        result = await run_benchmark(
            mock_transcriber,
            pcm_bytes,
            sample_rate=16000,
            chunk_ms=30,
            model="small",
            transcription_interval=0.5,
            decode_incomplete_lines=True,
            keyterm_count=0,
        )

        assert result.audio_duration_s == pytest.approx(0.1)
        assert result.transcript == "hallo welt"
        assert result.processing_time_s >= 0.0
        assert result.model == "small"
        mock_stream.close.assert_called_once()
        mock_transcriber.set_keyterms.assert_not_called()

    @pytest.mark.asyncio
    async def test_keyterm_count_applies_synthetic_keyterms(self, mock_transcriber, mock_stream):
        pcm_bytes = b"\x00\x00" * 1600

        await run_benchmark(
            mock_transcriber,
            pcm_bytes,
            sample_rate=16000,
            chunk_ms=30,
            model="small",
            transcription_interval=0.5,
            decode_incomplete_lines=True,
            keyterm_count=50,
        )

        mock_transcriber.set_keyterms.assert_called_once()
        applied = mock_transcriber.set_keyterms.call_args.args[0]
        assert len(applied) == 50

    @pytest.mark.asyncio
    async def test_chunk_metrics_are_populated(self, mock_transcriber, mock_stream):
        pcm_bytes = b"\x00\x00" * 1600

        result = await run_benchmark(
            mock_transcriber,
            pcm_bytes,
            sample_rate=16000,
            chunk_ms=30,
            model="tiny",
            transcription_interval=0.5,
            decode_incomplete_lines=True,
            keyterm_count=0,
        )

        assert result.chunk_count > 0
        assert result.average_chunk_compute_s >= 0.0
