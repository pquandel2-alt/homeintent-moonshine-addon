"""Kyutai Pocket TTS model loading and management.

Pocket TTS (package ``pocket-tts`` on PyPI, import name ``pocket_tts``,
https://github.com/kyutai-labs/pocket-tts, MIT-licensed code) is a small
(100M parameter) CPU-first text-to-speech model with a real incremental
streaming API. Verified directly against the upstream source
(``pocket_tts/models/tts_model.py``) and PyPI metadata at package version
3.1.0 -- see ABSCHLUSSBERICHT_V0.2.0.md for the full verification notes.

Key facts this module relies on:
- ``TTSModel.load_model(language=...)`` resolves/downloads weights via
  huggingface_hub (``hf://`` paths), cached under the standard
  ``HF_HOME``/``HUGGINGFACE_HUB_CACHE`` env vars (set by
  rootfs's run script, mirroring ``MOONSHINE_VOICE_CACHE``).
- If the voice-cloning-capable weights repo (gated, requires accepting HF
  terms) can't be downloaded, ``TTSModel`` automatically falls back to the
  ungated ``kyutai/pocket-tts-without-voice-cloning`` repo and sets
  ``has_voice_cloning = False`` -- this add-on never needs an HF login.
- Predefined voice names (e.g. "juergen") resolve through
  ``get_predefined_voice()``, which *always* points at the ungated
  ``pocket-tts-without-voice-cloning`` repo regardless of
  ``has_voice_cloning`` -- so the shipped default voice never requires HF
  auth, confirmed directly from upstream's ``pocket_tts/utils/utils.py``.
- ``generate_audio_stream(voice_state, text) -> Iterator[torch.Tensor]``
  is real incremental streaming (internally runs a generation thread and a
  Mimi-decode thread, yielding a flat 1D mono float32 chunk as soon as it is
  decoded) -- not a fake "generate everything, then chunk it" wrapper.
  ``generate_audio()`` is upstream's own thin wrapper that just drains this
  generator and concatenates; this add-on uses the streaming form directly
  for low time-to-first-audio (see app/tts_session.py).
- Both ``generate_audio()`` and ``generate_audio_stream()`` are explicitly
  documented upstream as NOT thread-safe ("separate model instances should
  be used for concurrent generation") -- this add-on serializes every call
  into the same ``TTSModel`` instance through one lock (see
  app/tts_session.py), independent from Moonshine's own lock, since the two
  are unrelated native/torch runtimes with no shared state.
- Output: mono float32 samples (numpy-convertible via ``.numpy()``) at
  ``TTSModel.sample_rate`` (24000 Hz for both German models, taken from
  ``mimi.sample_rate`` in the model's own config).
"""

import logging
import os
from enum import StrEnum
from pathlib import Path

from pocket_tts import TTSModel

_LOGGER = logging.getLogger(__name__)

# Home Assistant provides a persistent /data/ directory for add-ons. Kept
# separate from Moonshine's own /data/models cache (see app/models.py) so
# STT and TTS model files never collide and can be inspected/cleared
# independently.
DEFAULT_TTS_CACHE_DIR = Path(os.environ.get("POCKET_TTS_CACHE", "/data/models/pocket-tts"))


class GermanTtsModel(StrEnum):
    """Supported German Pocket TTS language configs.

    Both are real, verified ``--language``/``language=`` values shipped in
    upstream's ``pocket_tts/config/`` directory (``german.yaml``,
    ``german_24l.yaml``) -- not invented names.
    """

    GERMAN = "german"  # 6 transformer layers -- faster, default
    GERMAN_24L = "german_24l"  # 24 transformer layers -- higher quality, slower


# Upstream-published facts (pocket-tts 3.1.0, verified from source/PyPI/
# README -- see ABSCHLUSSBERICHT_V0.2.0.md). No RTF/latency numbers here are
# our own measurements unless explicitly marked "gemessen" in that report;
# the README's own numbers (~200ms first-chunk, ~6x realtime on Apple
# Silicon) are upstream's, not verified on this add-on's target hardware.
TTS_MODEL_METADATA = {
    GermanTtsModel.GERMAN: {
        "name": "pocket-tts-german",
        "layers": "6",
        "description": "Fast German text-to-speech (6 transformer layers)",
    },
    GermanTtsModel.GERMAN_24L: {
        "name": "pocket-tts-german-24l",
        "layers": "24",
        "description": "Higher-quality German text-to-speech (24 transformer layers, slower)",
    },
}

# The only real German predefined voice Pocket TTS ships (verified in
# upstream's README voice catalog and utils.py's _ORIGINS_OF_PREDEFINED_VOICES
# dict). Resolving it never requires Hugging Face authentication (see
# module docstring).
DEFAULT_TTS_VOICE = "juergen"

# TTSModel.sample_rate is authoritative at runtime; this constant exists
# only for callers that need it before a model is loaded (e.g. Wyoming
# service discovery when TTS is disabled). Verified: both German configs'
# mimi.sample_rate is 24000.
POCKET_TTS_SAMPLE_RATE = 24000


def validate_tts_model(model: str) -> bool:
    """Check if a TTS model name is valid."""
    try:
        GermanTtsModel(model)
        return True
    except ValueError:
        return False


def get_tts_model_info(model: str) -> dict[str, str]:
    """Get metadata for a TTS model (for Wyoming service discovery)."""
    try:
        model_enum = GermanTtsModel(model)
        return TTS_MODEL_METADATA[model_enum]
    except ValueError:
        return {"name": "unknown", "description": "Unknown model"}


def load_tts_model(
    model: str = GermanTtsModel.GERMAN.value,
    cache_dir: Path | None = None,
) -> TTSModel:
    """Resolve, download (if needed), and load a Pocket TTS model.

    Must be called at most once per add-on process for a given language
    (see app/tts_session.py, which wraps the loaded model plus a lock and a
    per-voice state cache for reuse across Wyoming connections).

    Args:
        model: Language config name ("german" or "german_24l").
        cache_dir: Override for the Hugging Face cache directory (used in
            tests). Sets HF_HOME so huggingface_hub's own hf_hub_download
            (used internally by pocket_tts for every ``hf://`` weight/voice
            path) caches there instead of its default location.

    Returns:
        A loaded TTSModel ready for get_state_for_audio_prompt()/
        generate_audio_stream().

    Raises:
        ValueError: If the model name is invalid.
        Exception: If model download or initialization fails.
    """
    if not validate_tts_model(model):
        raise ValueError(
            f"Invalid TTS model: {model}. Must be one of: "
            f"{', '.join(m.value for m in GermanTtsModel)}"
        )

    resolved_cache_dir = cache_dir if cache_dir is not None else DEFAULT_TTS_CACHE_DIR
    resolved_cache_dir.mkdir(parents=True, exist_ok=True)
    os.environ.setdefault("HF_HOME", str(resolved_cache_dir))

    _LOGGER.info("Loading Pocket TTS model: language=%s cache_dir=%s", model, resolved_cache_dir)
    try:
        tts_model = TTSModel.load_model(language=model)
    except Exception as err:
        _LOGGER.error(
            "Failed to load Pocket TTS model (language=%s cache_dir=%s): %s: %s",
            model,
            resolved_cache_dir,
            type(err).__name__,
            err,
        )
        raise

    if not tts_model.has_voice_cloning:
        _LOGGER.info(
            "Pocket TTS loaded without voice-cloning weights (predefined voices still work "
            "fully -- see module docstring); this is expected without a Hugging Face login."
        )
    _LOGGER.info("Pocket TTS %s model ready (sample_rate=%dHz)", model, tts_model.sample_rate)

    return tts_model
