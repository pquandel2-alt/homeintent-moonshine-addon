"""Tests for Pocket TTS model loading and management."""

from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest

from app.tts import (
    DEFAULT_TTS_VOICE,
    GermanTtsModel,
    get_tts_model_info,
    load_tts_model,
    validate_tts_model,
)


class TestValidateTtsModel:
    def test_valid_german(self):
        assert validate_tts_model("german") is True

    def test_valid_german_24l(self):
        assert validate_tts_model("german_24l") is True

    def test_invalid_model(self):
        assert validate_tts_model("english") is False

    def test_model_enum_values(self):
        assert {m.value for m in GermanTtsModel} == {"german", "german_24l"}


class TestGetTtsModelInfo:
    def test_german_model_info(self):
        info = get_tts_model_info("german")
        assert info["name"] == "pocket-tts-german"
        assert "description" in info

    def test_german_24l_model_info(self):
        info = get_tts_model_info("german_24l")
        assert info["name"] == "pocket-tts-german-24l"

    def test_invalid_model_info(self):
        info = get_tts_model_info("bogus")
        assert info["name"] == "unknown"


class TestDefaultVoice:
    def test_default_voice_is_juergen(self):
        # "juergen" is the only real German predefined voice Pocket TTS
        # ships (verified against upstream's README/utils.py) -- must never
        # silently change to an invented name.
        assert DEFAULT_TTS_VOICE == "juergen"


class TestLoadTtsModel:
    def test_invalid_model_raises_value_error(self, tmp_path: Path):
        with pytest.raises(ValueError, match="Invalid TTS model"):
            load_tts_model(model="english", cache_dir=tmp_path)

    def test_creates_cache_dir(self, tmp_path: Path):
        cache_dir = tmp_path / "pocket-tts"
        with patch("app.tts.TTSModel") as mock_tts_model_cls:
            mock_tts_model_cls.load_model.return_value = MagicMock(
                has_voice_cloning=True, sample_rate=24000
            )
            load_tts_model(model="german", cache_dir=cache_dir)
        assert cache_dir.exists()

    def test_calls_load_model_with_language(self, tmp_path: Path):
        with patch("app.tts.TTSModel") as mock_tts_model_cls:
            mock_tts_model_cls.load_model.return_value = MagicMock(
                has_voice_cloning=True, sample_rate=24000
            )
            load_tts_model(model="german_24l", cache_dir=tmp_path)
        mock_tts_model_cls.load_model.assert_called_once_with(language="german_24l")

    def test_download_failure_is_logged_and_reraised(self, tmp_path: Path):
        with patch("app.tts.TTSModel") as mock_tts_model_cls:
            mock_tts_model_cls.load_model.side_effect = RuntimeError("network error")
            with pytest.raises(RuntimeError, match="network error"):
                load_tts_model(model="german", cache_dir=tmp_path)

    def test_no_voice_cloning_does_not_raise(self, tmp_path: Path):
        """The gated voice-cloning weights repo being unreachable is a
        normal, non-fatal case (see app/tts.py's module docstring) --
        predefined voices like the default "juergen" still work."""
        with patch("app.tts.TTSModel") as mock_tts_model_cls:
            mock_tts_model_cls.load_model.return_value = MagicMock(
                has_voice_cloning=False, sample_rate=24000
            )
            model = load_tts_model(model="german", cache_dir=tmp_path)
        assert model.has_voice_cloning is False
