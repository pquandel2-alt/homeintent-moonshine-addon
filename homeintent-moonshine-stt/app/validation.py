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

# Kokoro-specific bounds. Speed is not our own defensive choice -- it is
# kokoro-onnx's own hard-enforced range, verified directly from source
# (``Kokoro._prepare()`` raises ValueError outside 0.5..2.0).
KOKORO_SPEED_MIN = 0.5
KOKORO_SPEED_MAX = 2.0

KOKORO_THREADS_MIN = 0  # 0 = auto (onnxruntime's own default, untouched)
KOKORO_THREADS_MAX = 64  # same generous bound as tts_threads

# Pause bounds are our own defensive UI bounds around kokoro-onnx's own
# real defaults (sentence_pause=0.25, clause_pause=0.1, verified from
# source) -- not an upstream-documented limit.
KOKORO_SENTENCE_PAUSE_MIN = 0.0
KOKORO_SENTENCE_PAUSE_MAX = 2.0

KOKORO_CLAUSE_PAUSE_MIN = 0.0
KOKORO_CLAUSE_PAUSE_MAX = 2.0

TTS_ENGINES = ("pocket_tts", "kokoro_onnx", "supertonic_3")

STT_ENGINES = ("moonshine", "kroko")

KROKO_THREADS_MIN = 1  # sherpa-onnx's own num_threads has no "0 = auto" mode
KROKO_THREADS_MAX = 64

SUPERTONIC_SPEED_MIN = 0.25
SUPERTONIC_SPEED_MAX = 3.0

# Bounds are our own defensive UI bounds around Supertonic's own documented
# default (8) and higher-quality example (10) -- see
# app/supertonic_tts.py's module docstring; not an upstream hard limit.
SUPERTONIC_STEPS_MIN = 2
SUPERTONIC_STEPS_MAX = 32

SUPERTONIC_THREADS_MIN = 1
SUPERTONIC_THREADS_MAX = 64


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


def validate_tts_engine(value: str) -> str:
    if value not in TTS_ENGINES:
        raise ValueError(f"tts_engine must be one of {TTS_ENGINES}, got {value!r}")
    return value


def validate_kokoro_speed(value: float) -> float:
    if not (KOKORO_SPEED_MIN <= value <= KOKORO_SPEED_MAX):
        raise ValueError(
            f"kokoro_speed must be between {KOKORO_SPEED_MIN} and {KOKORO_SPEED_MAX}, got {value}"
        )
    return value


def validate_kokoro_threads(value: int) -> int:
    if not (KOKORO_THREADS_MIN <= value <= KOKORO_THREADS_MAX):
        raise ValueError(
            f"kokoro_threads must be between {KOKORO_THREADS_MIN} and {KOKORO_THREADS_MAX}, "
            f"got {value}"
        )
    return value


def validate_kokoro_sentence_pause(value: float) -> float:
    if not (KOKORO_SENTENCE_PAUSE_MIN <= value <= KOKORO_SENTENCE_PAUSE_MAX):
        raise ValueError(
            f"kokoro_sentence_pause must be between {KOKORO_SENTENCE_PAUSE_MIN} and "
            f"{KOKORO_SENTENCE_PAUSE_MAX}, got {value}"
        )
    return value


def validate_kokoro_clause_pause(value: float) -> float:
    if not (KOKORO_CLAUSE_PAUSE_MIN <= value <= KOKORO_CLAUSE_PAUSE_MAX):
        raise ValueError(
            f"kokoro_clause_pause must be between {KOKORO_CLAUSE_PAUSE_MIN} and "
            f"{KOKORO_CLAUSE_PAUSE_MAX}, got {value}"
        )
    return value


def validate_stt_engine(value: str) -> str:
    if value not in STT_ENGINES:
        raise ValueError(f"stt_engine must be one of {STT_ENGINES}, got {value!r}")
    return value


def validate_kroko_threads(value: int) -> int:
    if not (KROKO_THREADS_MIN <= value <= KROKO_THREADS_MAX):
        raise ValueError(
            f"kroko_threads must be between {KROKO_THREADS_MIN} and {KROKO_THREADS_MAX}, "
            f"got {value}"
        )
    return value


def validate_supertonic_speed(value: float) -> float:
    if not (SUPERTONIC_SPEED_MIN <= value <= SUPERTONIC_SPEED_MAX):
        raise ValueError(
            f"supertonic_speed must be between {SUPERTONIC_SPEED_MIN} and "
            f"{SUPERTONIC_SPEED_MAX}, got {value}"
        )
    return value


def validate_supertonic_steps(value: int) -> int:
    if not (SUPERTONIC_STEPS_MIN <= value <= SUPERTONIC_STEPS_MAX):
        raise ValueError(
            f"supertonic_steps must be between {SUPERTONIC_STEPS_MIN} and "
            f"{SUPERTONIC_STEPS_MAX}, got {value}"
        )
    return value


def validate_supertonic_threads(value: int) -> int:
    if not (SUPERTONIC_THREADS_MIN <= value <= SUPERTONIC_THREADS_MAX):
        raise ValueError(
            f"supertonic_threads must be between {SUPERTONIC_THREADS_MIN} and "
            f"{SUPERTONIC_THREADS_MAX}, got {value}"
        )
    return value
