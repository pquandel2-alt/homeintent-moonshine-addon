"""Tests for model loading and validation."""

import pytest

from app.models import GermanModel, get_model_info, validate_model


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
