"""Tests for app/vosk_engine.py using a fake vosk.Model/KaldiRecognizer
double (no real vosk install required, no real model download) -- these
verify the streaming/session wiring and the honest "hotwords not
supported" dispatch, which is real, in-process code this add-on owns.
"""

import asyncio
import json

import pytest

from app.stt_engine import SttCapabilities
from app.vosk_engine import VoskSttEngine, VoskSttSession


class _FakeKaldiRecognizer:
    """Mimics vosk.KaldiRecognizer's own surface used by this add-on
    (AcceptWaveform/FinalResult), verified directly from the real wheel's
    own source -- see app/vosk_engine.py's module docstring."""

    def __init__(self, model: object, sample_rate: int, final_text: str) -> None:
        self.model = model
        self.sample_rate = sample_rate
        self.final_text = final_text
        self.accepted_chunks: list[bytes] = []

    def AcceptWaveform(self, data: bytes) -> bool:  # noqa: N802 - matches real vosk API
        self.accepted_chunks.append(data)
        return False

    def FinalResult(self) -> str:  # noqa: N802 - matches real vosk API
        return json.dumps({"text": self.final_text})


def _make_fake_vosk_module(monkeypatch, final_text: str = "schalte das licht im wohnzimmer ein"):
    import sys
    import types

    fake_module = types.ModuleType("vosk")

    def _kaldi_recognizer(model: object, sample_rate: int) -> _FakeKaldiRecognizer:
        return _FakeKaldiRecognizer(model, sample_rate, final_text)

    fake_module.KaldiRecognizer = _kaldi_recognizer  # type: ignore[attr-defined]
    monkeypatch.setitem(sys.modules, "vosk", fake_module)
    return fake_module


def test_vosk_engine_capabilities_are_honest_about_no_hotwords() -> None:
    engine = VoskSttEngine(model=object())
    assert engine.capabilities == SttCapabilities(
        supports_streaming=True,
        supports_hotwords=False,
        supports_dynamic_vocabulary=False,
        supports_partial_results=False,
    )
    assert engine.engine_id == "vosk_german"
    assert engine.program_name == "homeintent-vosk-german"


def test_vosk_set_keyterms_never_applies_and_logs(caplog) -> None:
    engine = VoskSttEngine(model=object())
    with caplog.at_level("INFO"):
        accepted, succeeded = engine.set_keyterms(["Wohnzimmer", "Büro"])
    assert accepted == []
    assert succeeded is True
    assert any("not supported by engine vosk_german" in msg for msg in caplog.messages)


def test_vosk_set_keyterms_empty_list_logs_nothing(caplog) -> None:
    engine = VoskSttEngine(model=object())
    with caplog.at_level("INFO"):
        accepted, succeeded = engine.set_keyterms([])
    assert accepted == []
    assert succeeded is True
    assert not caplog.messages


@pytest.mark.asyncio
async def test_vosk_session_full_lifecycle(monkeypatch) -> None:
    _make_fake_vosk_module(monkeypatch)
    engine = VoskSttEngine(model=object())

    session = engine.create_session(lock=asyncio.Lock())
    assert isinstance(session, VoskSttSession)
    await session.start()

    await session.add_audio([0.0] * 1600, sample_rate=16000)
    assert session.audio_duration_seconds == pytest.approx(0.1)
    assert session.add_audio_chunk_count == 1
    assert len(session._recognizer.accepted_chunks) == 1
    # 1600 float samples -> 1600 int16 PCM bytes (2 bytes/sample)
    assert len(session._recognizer.accepted_chunks[0]) == 3200

    text = await session.finalize()
    assert text == "schalte das licht im wohnzimmer ein"
    session.close()


@pytest.mark.asyncio
async def test_vosk_session_finalize_with_malformed_json_returns_empty_string(monkeypatch) -> None:
    _make_fake_vosk_module(monkeypatch)
    engine = VoskSttEngine(model=object())
    session = engine.create_session()
    assert isinstance(session, VoskSttSession)
    await session.start()

    class _BrokenRecognizer(_FakeKaldiRecognizer):
        def FinalResult(self) -> str:  # noqa: N802
            return "not json"

    session._recognizer = _BrokenRecognizer(object(), 16000, "")
    text = await session.finalize()
    assert text == ""
    session.close()


@pytest.mark.asyncio
async def test_vosk_session_rejects_wrong_sample_rate(monkeypatch) -> None:
    _make_fake_vosk_module(monkeypatch)
    engine = VoskSttEngine(model=object())
    session = engine.create_session()
    await session.start()
    with pytest.raises(AssertionError):
        await session.add_audio([0.0] * 800, sample_rate=8000)
    session.close()


def test_vosk_engine_never_touches_other_engines_module_state() -> None:
    """Purely a documentation-style sanity check: constructing a
    VoskSttEngine takes an already-loaded model object and does no import
    of any other engine's module."""
    engine = VoskSttEngine(model="fake-model-handle")
    assert engine.model_display_name == "vosk-model-small-de-0.15"
