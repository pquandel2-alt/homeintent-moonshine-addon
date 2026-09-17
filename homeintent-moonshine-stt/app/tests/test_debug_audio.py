"""Tests for optional on-disk debug audio recording."""

import json
import wave

from app.debug_audio import save_debug_audio


def _pcm(num_samples: int = 1600) -> bytes:
    return b"\x00\x00" * num_samples  # 0.1s of silence at 16kHz


class TestSaveDebugAudio:
    def test_writes_valid_wav_file(self, tmp_path):
        path = save_debug_audio(
            _pcm(), model="small", language="de", transcript="Hallo", directory=tmp_path
        )

        assert path.exists()
        with wave.open(str(path), "rb") as wav_file:
            assert wav_file.getnchannels() == 1
            assert wav_file.getsampwidth() == 2
            assert wav_file.getframerate() == 16000
            assert wav_file.getnframes() == 1600

    def test_writes_metadata_json_sidecar(self, tmp_path):
        path = save_debug_audio(
            _pcm(), model="tiny", language="de", transcript="Test Transkript", directory=tmp_path
        )

        json_path = path.with_suffix(".json")
        assert json_path.exists()
        metadata = json.loads(json_path.read_text(encoding="utf-8"))
        assert metadata["model"] == "tiny"
        assert metadata["language"] == "de"
        assert metadata["transcript"] == "Test Transkript"
        assert metadata["sample_rate"] == 16000
        assert metadata["duration_ms"] > 0
        assert "timestamp" in metadata

    def test_metadata_never_contains_ha_entity_state_keys(self, tmp_path):
        path = save_debug_audio(
            _pcm(), model="small", language="de", transcript="Licht an", directory=tmp_path
        )
        metadata = json.loads(path.with_suffix(".json").read_text(encoding="utf-8"))
        assert set(metadata.keys()) == {
            "timestamp",
            "model",
            "language",
            "transcript",
            "duration_ms",
            "sample_rate",
        }

    def test_creates_directory_if_missing(self, tmp_path):
        directory = tmp_path / "nested" / "debug_audio"
        path = save_debug_audio(
            _pcm(), model="small", language="de", transcript="", directory=directory
        )
        assert path.exists()

    def test_concurrent_saves_within_same_second_do_not_collide(self, tmp_path):
        path1 = save_debug_audio(
            _pcm(), model="small", language="de", transcript="eins", directory=tmp_path
        )
        path2 = save_debug_audio(
            _pcm(), model="small", language="de", transcript="zwei", directory=tmp_path
        )
        assert path1 != path2
        assert path1.exists()
        assert path2.exists()


class TestRetention:
    def test_deletes_oldest_files_beyond_max_files(self, tmp_path):
        for i in range(5):
            save_debug_audio(
                _pcm(),
                model="small",
                language="de",
                transcript=str(i),
                directory=tmp_path,
                max_files=3,
            )

        wav_files = sorted(tmp_path.glob("*.wav"))
        json_files = sorted(tmp_path.glob("*.json"))
        assert len(wav_files) == 3
        assert len(json_files) == 3

    def test_does_not_delete_when_under_limit(self, tmp_path):
        for i in range(2):
            save_debug_audio(
                _pcm(),
                model="small",
                language="de",
                transcript=str(i),
                directory=tmp_path,
                max_files=100,
            )

        assert len(list(tmp_path.glob("*.wav"))) == 2

    def test_deletes_matching_json_sidecar_alongside_wav(self, tmp_path):
        for i in range(4):
            save_debug_audio(
                _pcm(),
                model="small",
                language="de",
                transcript=str(i),
                directory=tmp_path,
                max_files=2,
            )

        wav_stems = {p.stem for p in tmp_path.glob("*.wav")}
        json_stems = {p.stem for p in tmp_path.glob("*.json")}
        assert wav_stems == json_stems
