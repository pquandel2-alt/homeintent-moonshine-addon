"""Tests for model loading and validation."""

from unittest.mock import MagicMock

import pytest

import app.models as models_mod
from app.models import GermanModel, get_model_info, load_transcriber, validate_model


class TestModelValidation:
    """Tests for model name validation."""

    def test_valid_tiny_model(self):
        """Test that 'tiny' is a valid model."""
        assert validate_model("tiny") is True

    def test_valid_small_model(self):
        """Test that 'small' is a valid model."""
        assert validate_model("small") is True

    def test_invalid_model(self):
        """Test that invalid model names are rejected."""
        assert validate_model("large") is False
        assert validate_model("base") is False
        assert validate_model("") is False
        assert validate_model("unknown") is False

    def test_model_enum_values(self):
        """Test that GermanModel enum has expected values."""
        assert GermanModel.TINY.value == "tiny"
        assert GermanModel.SMALL.value == "small"


class TestModelInfo:
    """Tests for model metadata retrieval."""

    def test_tiny_model_info(self):
        """Test metadata for Tiny model."""
        info = get_model_info("tiny")
        assert "tiny" in info["name"].lower()
        assert "34M" in info["params"]
        assert "12.0%" in info["wer"]

    def test_small_model_info(self):
        """Test metadata for Small model."""
        info = get_model_info("small")
        assert "small" in info["name"].lower()
        assert "123M" in info["params"]
        assert "7.5%" in info["wer"]

    def test_invalid_model_info(self):
        """Test metadata for invalid model returns default."""
        info = get_model_info("unknown")
        assert info["name"] == "unknown"
        assert "Unknown" in info["description"]

    def test_model_descriptions(self):
        """Test that model descriptions are present and descriptive."""
        tiny_info = get_model_info("tiny")
        small_info = get_model_info("small")
        assert len(tiny_info["description"]) > 10
        assert len(small_info["description"]) > 10


class TestLoadTranscriber:
    """Tests for load_transcriber(), mocking the real download/native layers.

    get_model_for_language() and Transcriber itself are moonshine-voice's own
    responsibility (atomic/resumable/integrity-checked download, native model
    load) -- these tests only verify our own glue: correct args passed
    through, invalid input rejected, and download failures surfaced clearly
    rather than swallowed.
    """

    def _patch_download(
        self,
        monkeypatch: pytest.MonkeyPatch,
        model_path: str = "/data/models/fake.onnx",
        arch: MagicMock | None = None,
    ) -> tuple[MagicMock, MagicMock]:
        fake_get_model = MagicMock(
            return_value=(model_path, arch or MagicMock(name="TINY_STREAMING"))
        )
        monkeypatch.setattr(models_mod, "get_model_for_language", fake_get_model)
        fake_transcriber_cls = MagicMock()
        monkeypatch.setattr(models_mod, "Transcriber", fake_transcriber_cls)
        return fake_get_model, fake_transcriber_cls

    def test_invalid_model_raises_value_error(self, tmp_path):
        with pytest.raises(ValueError, match="Invalid model"):
            load_transcriber(model="large", cache_root=tmp_path)

    def test_creates_cache_root_directory(self, monkeypatch, tmp_path):
        self._patch_download(monkeypatch)
        cache_root = tmp_path / "models"
        assert not cache_root.exists()

        load_transcriber(model="tiny", cache_root=cache_root)

        assert cache_root.exists()

    def test_default_options_match_verified_upstream_defaults(self, monkeypatch, tmp_path):
        _, fake_transcriber_cls = self._patch_download(monkeypatch)

        load_transcriber(model="small", cache_root=tmp_path)

        _, kwargs = fake_transcriber_cls.call_args
        assert kwargs["update_interval"] == 0.5
        assert kwargs["options"]["vad_threshold"] == 0.5
        assert kwargs["options"]["decode_incomplete_lines"] == "true"
        assert kwargs["options"]["keyterm_boost"] == 2.0

    def test_custom_options_propagate_to_transcriber(self, monkeypatch, tmp_path):
        _, fake_transcriber_cls = self._patch_download(monkeypatch)

        load_transcriber(
            model="small",
            cache_root=tmp_path,
            transcription_interval=1.5,
            vad_threshold=0.7,
            decode_incomplete_lines=False,
            keyterm_boost=4.0,
        )

        _, kwargs = fake_transcriber_cls.call_args
        assert kwargs["update_interval"] == 1.5
        assert kwargs["options"]["vad_threshold"] == 0.7
        assert kwargs["options"]["decode_incomplete_lines"] == "false"
        assert kwargs["options"]["keyterm_boost"] == 4.0

    def test_download_failure_is_logged_and_reraised(self, monkeypatch, tmp_path):
        fake_get_model = MagicMock(side_effect=OSError("disk full during download"))
        monkeypatch.setattr(models_mod, "get_model_for_language", fake_get_model)

        with pytest.raises(OSError, match="disk full"):
            load_transcriber(model="small", cache_root=tmp_path)

    def test_corrupted_cache_error_propagates_not_swallowed(self, monkeypatch, tmp_path):
        """A hash-mismatch/corrupted cache raises inside get_model_for_language
        upstream; we must not swallow it or silently fall back."""
        fake_get_model = MagicMock(side_effect=RuntimeError("checksum mismatch"))
        monkeypatch.setattr(models_mod, "get_model_for_language", fake_get_model)

        with pytest.raises(RuntimeError, match="checksum mismatch"):
            load_transcriber(model="tiny", cache_root=tmp_path)
