"""Tests for config option bounds validation."""

import pytest

from app.validation import (
    validate_debug_audio_max_files,
    validate_ha_vocabulary_refresh_minutes,
    validate_keyterm_boost,
    validate_transcription_interval,
    validate_vad_threshold,
)


class TestValidateTranscriptionInterval:
    def test_default_accepted(self):
        assert validate_transcription_interval(0.5) == 0.5

    def test_low_valid_boundary(self):
        assert validate_transcription_interval(0.05) == 0.05

    def test_high_valid_boundary(self):
        assert validate_transcription_interval(10.0) == 10.0

    def test_below_minimum_rejected(self):
        with pytest.raises(ValueError, match="transcription_interval"):
            validate_transcription_interval(0.01)

    def test_above_maximum_rejected(self):
        with pytest.raises(ValueError, match="transcription_interval"):
            validate_transcription_interval(10.1)


class TestValidateVadThreshold:
    def test_default_accepted(self):
        assert validate_vad_threshold(0.5) == 0.5

    def test_low_valid_boundary(self):
        assert validate_vad_threshold(0.0) == 0.0

    def test_high_valid_boundary(self):
        assert validate_vad_threshold(1.0) == 1.0

    def test_below_minimum_rejected(self):
        with pytest.raises(ValueError, match="vad_threshold"):
            validate_vad_threshold(-0.1)

    def test_above_maximum_rejected(self):
        with pytest.raises(ValueError, match="vad_threshold"):
            validate_vad_threshold(1.1)


class TestValidateKeytermBoost:
    def test_default_matches_upstream(self):
        # Upstream ContextBiaser::kDefaultBoost == 2.0 (see app/models.py).
        assert validate_keyterm_boost(2.0) == 2.0

    def test_low_valid_boundary(self):
        assert validate_keyterm_boost(0.0) == 0.0

    def test_high_valid_boundary(self):
        assert validate_keyterm_boost(10.0) == 10.0

    def test_below_minimum_rejected(self):
        with pytest.raises(ValueError, match="keyterm_boost"):
            validate_keyterm_boost(-1.0)

    def test_above_maximum_rejected(self):
        with pytest.raises(ValueError, match="keyterm_boost"):
            validate_keyterm_boost(10.5)


class TestValidateDebugAudioMaxFiles:
    def test_default_accepted(self):
        assert validate_debug_audio_max_files(100) == 100

    def test_minimum_boundary(self):
        assert validate_debug_audio_max_files(1) == 1

    def test_below_minimum_rejected(self):
        with pytest.raises(ValueError, match="debug_audio_max_files"):
            validate_debug_audio_max_files(0)

    def test_above_maximum_rejected(self):
        with pytest.raises(ValueError, match="debug_audio_max_files"):
            validate_debug_audio_max_files(10001)


class TestValidateHaVocabularyRefreshMinutes:
    def test_zero_disables_refresh_and_is_valid(self):
        assert validate_ha_vocabulary_refresh_minutes(0) == 0

    def test_default_accepted(self):
        assert validate_ha_vocabulary_refresh_minutes(30) == 30

    def test_above_maximum_rejected(self):
        with pytest.raises(ValueError, match="ha_vocabulary_refresh_minutes"):
            validate_ha_vocabulary_refresh_minutes(1441)

    def test_negative_rejected(self):
        with pytest.raises(ValueError, match="ha_vocabulary_refresh_minutes"):
            validate_ha_vocabulary_refresh_minutes(-1)
