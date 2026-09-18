"""Bounds-checking for user-configurable Moonshine/add-on options.

These ranges are our own defensive UI bounds (matched by config.yaml's
schema), not upstream-documented hard limits -- moonshine-voice's C API
does not publish min/max values for these options. Each default equals the
real upstream default (see app/models.py); only the *bounds* around that
default are our own choice, picked to keep the add-on usable rather than to
reflect a verified upstream constraint.
"""

TRANSCRIPTION_INTERVAL_MIN = 0.05
TRANSCRIPTION_INTERVAL_MAX = 10.0

VAD_THRESHOLD_MIN = 0.0
VAD_THRESHOLD_MAX = 1.0

KEYTERM_BOOST_MIN = 0.0
KEYTERM_BOOST_MAX = 10.0

DEBUG_AUDIO_MAX_FILES_MIN = 1
DEBUG_AUDIO_MAX_FILES_MAX = 10000

HA_VOCABULARY_REFRESH_MINUTES_MIN = 0  # 0 disables periodic refresh
HA_VOCABULARY_REFRESH_MINUTES_MAX = 1440

TTS_THREADS_MIN = 0  # 0 = auto (PyTorch's own default, untouched)
TTS_THREADS_MAX = 64  # generous upper bound; no real CPU has more cores


def validate_transcription_interval(value: float) -> float:
    if not (TRANSCRIPTION_INTERVAL_MIN <= value <= TRANSCRIPTION_INTERVAL_MAX):
        raise ValueError(
            f"transcription_interval must be between {TRANSCRIPTION_INTERVAL_MIN} and "
            f"{TRANSCRIPTION_INTERVAL_MAX} seconds, got {value}"
        )
    return value


def validate_vad_threshold(value: float) -> float:
    if not (VAD_THRESHOLD_MIN <= value <= VAD_THRESHOLD_MAX):
        raise ValueError(
            f"vad_threshold must be between {VAD_THRESHOLD_MIN} and {VAD_THRESHOLD_MAX}, "
            f"got {value}"
        )
    return value


def validate_keyterm_boost(value: float) -> float:
    if not (KEYTERM_BOOST_MIN <= value <= KEYTERM_BOOST_MAX):
        raise ValueError(
            f"keyterm_boost must be between {KEYTERM_BOOST_MIN} and {KEYTERM_BOOST_MAX}, "
            f"got {value}"
        )
    return value


def validate_debug_audio_max_files(value: int) -> int:
    if not (DEBUG_AUDIO_MAX_FILES_MIN <= value <= DEBUG_AUDIO_MAX_FILES_MAX):
        raise ValueError(
            f"debug_audio_max_files must be between {DEBUG_AUDIO_MAX_FILES_MIN} and "
            f"{DEBUG_AUDIO_MAX_FILES_MAX}, got {value}"
        )
    return value


def validate_ha_vocabulary_refresh_minutes(value: int) -> int:
    if not (HA_VOCABULARY_REFRESH_MINUTES_MIN <= value <= HA_VOCABULARY_REFRESH_MINUTES_MAX):
        raise ValueError(
            f"ha_vocabulary_refresh_minutes must be between "
            f"{HA_VOCABULARY_REFRESH_MINUTES_MIN} and {HA_VOCABULARY_REFRESH_MINUTES_MAX}, "
            f"got {value}"
        )
    return value


def validate_tts_threads(value: int) -> int:
    if not (TTS_THREADS_MIN <= value <= TTS_THREADS_MAX):
        raise ValueError(
            f"tts_threads must be between {TTS_THREADS_MIN} and {TTS_THREADS_MAX}, got {value}"
        )
    return value
