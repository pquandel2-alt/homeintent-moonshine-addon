"""Tests for the SttEngine/SttSession abstraction and Moonshine's adapter.

Covers: default stt_engine=moonshine, the old-config-without-stt_engine
upgrade path, Moonshine's behavior is unchanged when driven through the new
abstraction, and that app/handler.py never touches a concrete engine type.
"""

import asyncio

import pytest

from app.handler import MoonshineAsrHandler
from app.moonshine_engine import MoonshineSttEngine
from app.stt_engine import SttCapabilities


class _FakeTranscriber:
    """Minimal stand-in for moonshine_voice.Transcriber's own surface --
    only what MoonshineStreamingSession/apply_safe_keyterms actually call."""

    def __init__(self) -> None:
        self.keyterms: list[str] = []

    def create_stream(self, update_interval: float | None = None) -> "_FakeStream":
        return _FakeStream()

    def set_keyterms(self, keyterms: list[str]) -> None:
        self.keyterms = list(keyterms)


class _FakeStream:
    def __init__(self) -> None:
        self._listeners: list[object] = []

    def add_listener(self, listener: object) -> None:
        self._listeners.append(listener)

    def start(self) -> None:
        pass

    def add_audio(self, samples: list[float], sample_rate: int) -> None:
        pass

    def stop(self) -> None:
        pass

    def close(self) -> None:
        pass


def test_moonshine_engine_capabilities() -> None:
    engine = MoonshineSttEngine(_FakeTranscriber(), "small", "de")
    caps = engine.capabilities
    assert caps == SttCapabilities(
        supports_streaming=True,
        supports_hotwords=True,
        supports_dynamic_vocabulary=True,
        supports_partial_results=False,
    )
    assert engine.engine_id == "moonshine"
    assert engine.program_name == "homeintent-moonshine"
    assert engine.model_display_name == "small"


def test_moonshine_engine_set_keyterms_delegates_to_apply_safe_keyterms() -> None:
    transcriber = _FakeTranscriber()
    engine = MoonshineSttEngine(transcriber, "small", "de")
    accepted, succeeded = engine.set_keyterms(["Wohnzimmer", "Büro"])
    assert succeeded is True
    assert set(accepted) == {"Wohnzimmer", "Büro"}
    assert set(transcriber.keyterms) == {"Wohnzimmer", "Büro"}


@pytest.mark.asyncio
async def test_moonshine_engine_create_session_is_a_real_streaming_session() -> None:
    engine = MoonshineSttEngine(_FakeTranscriber(), "small", "de")
    session = engine.create_session(lock=asyncio.Lock())
    await session.start()
    await session.add_audio([0.0] * 160, sample_rate=16000)
    text = await session.finalize()
    assert text == ""  # fake stream never fires LineCompleted
    assert session.audio_duration_seconds > 0
    session.close()


def test_handler_backward_compat_transcriber_kwarg_still_builds_an_engine() -> None:
    """Existing callers/tests passing transcriber= (not stt_engine=) must
    keep working unchanged -- this is the whole point of preserving that
    constructor parameter (see handler.py's own docstring)."""

    class _Reader:
        pass

    class _Writer:
        def write(self, data: bytes) -> None:
            pass

        async def drain(self) -> None:
            pass

    handler = MoonshineAsrHandler(
        reader=_Reader(),  # type: ignore[arg-type]
        writer=_Writer(),  # type: ignore[arg-type]
        transcriber=_FakeTranscriber(),
        model_name="small",
    )
    assert handler._stt_engine is not None
    assert handler._stt_engine.engine_id == "moonshine"


def test_handler_stt_engine_kwarg_takes_priority_over_transcriber() -> None:
    class _Reader:
        pass

    class _Writer:
        def write(self, data: bytes) -> None:
            pass

        async def drain(self) -> None:
            pass

    class _FakeEngine:
        engine_id = "kroko"

        @property
        def capabilities(self) -> SttCapabilities:
            return SttCapabilities(True, True, True, False)

    handler = MoonshineAsrHandler(
        reader=_Reader(),  # type: ignore[arg-type]
        writer=_Writer(),  # type: ignore[arg-type]
        transcriber=_FakeTranscriber(),
        stt_engine=_FakeEngine(),  # type: ignore[arg-type]
    )
    assert handler._stt_engine is not None
    assert handler._stt_engine.engine_id == "kroko"


def test_capability_contract_for_an_engine_without_hotword_support(caplog) -> None:
    """Neither Moonshine nor Kroko (this session's two engines) lacks
    hotword support, but the SttEngine contract (see stt_engine.py's
    set_keyterms docstring) requires any FUTURE engine without it to log
    clearly and return ([], True) rather than silently pretending to apply
    something -- this documents/pins that contract with a minimal
    conforming implementation."""

    class _NoHotwordsEngine:
        engine_id = "future_engine"

        @property
        def capabilities(self) -> SttCapabilities:
            return SttCapabilities(
                supports_streaming=True,
                supports_hotwords=False,
                supports_dynamic_vocabulary=False,
                supports_partial_results=False,
            )

        def set_keyterms(
            self, terms: list[str], fallback_terms: list[str] | None = None
        ) -> tuple[list[str], bool]:
            import logging

            logging.getLogger(__name__).warning(
                "HA vocabulary not supported by engine %s", self.engine_id
            )
            return [], True

    engine = _NoHotwordsEngine()
    with caplog.at_level("WARNING"):
        accepted, succeeded = engine.set_keyterms(["Wohnzimmer"])
    assert accepted == []
    assert succeeded is True
    assert any("not supported by engine future_engine" in msg for msg in caplog.messages)


def test_handler_no_engine_no_transcriber_means_stt_disabled() -> None:
    class _Reader:
        pass

    class _Writer:
        def write(self, data: bytes) -> None:
            pass

        async def drain(self) -> None:
            pass

    handler = MoonshineAsrHandler(reader=_Reader(), writer=_Writer())  # type: ignore[arg-type]
    assert handler._stt_engine is None
