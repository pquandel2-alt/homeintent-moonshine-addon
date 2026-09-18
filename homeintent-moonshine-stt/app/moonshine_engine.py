"""Moonshine's SttEngine adapter.

Thin wrapper around the existing, unmodified app/models.py (model loading)
and app/streaming.py (MoonshineStreamingSession) so Moonshine sits behind
the app/stt_engine.py abstraction exactly like every other engine, with
ZERO behavioral change: model choice (tiny/small), keyterm_boost, VAD,
decode_incomplete_lines, and performance counters are all still handled by
the same, untouched code paths as before this abstraction existed.

app/streaming.py's MoonshineStreamingSession already implements every
member of the SttSession protocol without any changes (see
app/stt_engine.py's module docstring) -- create_session() below returns one
directly, it is not reimplemented or subclassed.
"""

import asyncio
import logging

from moonshine_voice import Transcriber

from app.keyterms import apply_safe_keyterms
from app.models import get_model_info
from app.streaming import MoonshineStreamingSession
from app.stt_engine import SttCapabilities, SttSession

_LOGGER = logging.getLogger(__name__)

ENGINE_ID = "moonshine"

_ATTRIBUTION_NAME = "Moonshine AI"
_ATTRIBUTION_URL = "https://github.com/moonshine-ai/moonshine"


class MoonshineSttEngine:
    """Wraps one shared Moonshine Transcriber, loaded once at startup.

    ``model_name`` is "tiny" or "small" (the existing, unrenamed ``model``
    add-on option -- see config.yaml's own comment on why it keeps this
    name for upgrade compatibility).
    """

    def __init__(self, transcriber: Transcriber, model_name: str, language: str = "de") -> None:
        self._transcriber = transcriber
        self._model_name = model_name
        self._language = language

    @property
    def engine_id(self) -> str:
        return ENGINE_ID

    @property
    def capabilities(self) -> SttCapabilities:
        return SttCapabilities(
            supports_streaming=True,
            supports_hotwords=True,
            supports_dynamic_vocabulary=True,
            supports_partial_results=False,
        )

    @property
    def program_name(self) -> str:
        return "homeintent-moonshine"

    @property
    def model_display_name(self) -> str:
        return self._model_name

    @property
    def description(self) -> str:
        return "HomeIntent Moonshine STT - German streaming ASR"

    @property
    def attribution_name(self) -> str:
        return _ATTRIBUTION_NAME

    @property
    def attribution_url(self) -> str:
        return _ATTRIBUTION_URL

    def model_info(self) -> dict[str, str]:
        return get_model_info(self._model_name)

    def create_session(self, lock: object | None = None) -> SttSession:
        assert lock is None or isinstance(lock, asyncio.Lock)
        return MoonshineStreamingSession(self._transcriber, lock=lock)

    def set_keyterms(
        self, terms: list[str], fallback_terms: list[str] | None = None
    ) -> tuple[list[str], bool]:
        result = apply_safe_keyterms(self._transcriber, terms, fallback_terms)
        return result.accepted, result.apply_succeeded
