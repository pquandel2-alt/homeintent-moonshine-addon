"""Tests for config option bounds validation."""

import pytest

from app.validation import (
    validate_debug_audio_max_files,
    validate_ha_vocabulary_refresh_minutes,
    validate_keyterm_boost,
    validate_supertonic_steps,
    validate_supertonic_voice,
    validate_transcription_interval,
    validate_tts_threads,
    validate_vad_threshold,
)


class TestValidateSupertonicSteps:
    """v0.6.1: supertonic_steps is now a dropdown of only the two documented
    presets (8, 10) -- unlike most validators here, out-of-set values never
    raise; they gracefully fall back to the default (8) so an old
    installation with a previously-valid (but now unlisted) value from the
    old int(2,32) schema upgrades cleanly instead of crash-looping."""

    def test_default_preset_accepted(self):
        assert validate_supertonic_steps(8) == 8

    def test_other_documented_preset_accepted(self):
        assert validate_supertonic_steps(10) == 10

    def test_out_of_set_value_falls_back_to_default(self):
        assert validate_supertonic_steps(16) == 8

    def test_previously_valid_boundary_falls_back_to_default(self):
        # 2 and 32 were valid under the old int(2,32) schema.
        assert validate_supertonic_steps(2) == 8
        assert validate_supertonic_steps(32) == 8

    def test_never_raises(self):
        # No exception for any int input -- see class docstring.
        validate_supertonic_steps(-5)
        validate_supertonic_steps(0)
        validate_supertonic_steps(999)


class TestValidateSupertonicVoice:
    """v0.6.1: supertonic_voice is now a dropdown of the 10 real voices
    (M1-M5/F1-F5). Never raises: an unrecognized value falls back to the
    default (M1) with a warning log instead of crashing."""

    @pytest.mark.parametrize("voice", ["M1", "M2", "M3", "M4", "M5", "F1", "F2", "F3", "F4", "F5"])
    def test_all_ten_valid_voices_accepted(self, voice):
        assert validate_supertonic_voice(voice) == voice

    def test_case_insensitive(self):
        assert validate_supertonic_voice("m1") == "M1"

    def test_legacy_numeric_sid_still_accepted(self):
        # The previous free-text schema also accepted a raw numeric sid
        # (0-9, see app/supertonic_tts.py's resolve_supertonic_sid) --
        # preserved for backward compatibility.
        assert validate_supertonic_voice("5") == "5"

    def test_default_preserved_on_invalid_value(self):
        assert validate_supertonic_voice("not-a-real-voice") == "M1"

    def test_out_of_range_numeric_sid_falls_back_to_default(self):
        assert validate_supertonic_voice("99") == "M1"

    def test_never_raises(self):
        validate_supertonic_voice("")
        validate_supertonic_voice("  ")


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


class TestValidateTtsThreads:
    def test_zero_means_auto_and_is_valid(self):
        assert validate_tts_threads(0) == 0

    def test_positive_value_accepted(self):
        assert validate_tts_threads(4) == 4

    def test_above_maximum_rejected(self):
        with pytest.raises(ValueError, match="tts_threads"):
            validate_tts_threads(65)

    def test_negative_rejected(self):
        with pytest.raises(ValueError, match="tts_threads"):
            validate_tts_threads(-1)
