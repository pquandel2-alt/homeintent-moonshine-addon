"""Tests for app/kroko_engine.py using a fake sherpa-onnx recognizer double
(no real sherpa_onnx.OnlineRecognizer, no real model download) -- these
verify the streaming/session wiring and hotwords dispatch, which is real,
in-process code this add-on owns. A real end-to-end Kroko test against the
actual downloaded model is gated behind RUN_KROKO_E2E=1 (see
test_e2e_kroko_stt.py).
"""

import asyncio

import numpy as np
import pytest

from app.kroko_engine import KrokoSttEngine, KrokoSttSession
from app.kroko_model import KrokoModelFiles
from app.stt_engine import SttCapabilities


class _FakeOnlineStream:
    def __init__(self, hotwords: str | None) -> None:
        self.hotwords = hotwords
        self.samples: list[np.ndarray] = []
        self.finished = False

    def accept_waveform(self, sample_rate: int, samples: np.ndarray) -> None:
        self.samples.append(np.asarray(samples))

    def input_finished(self) -> None:
        self.finished = True


class _FakeOnlineRecognizer:
    """Mimics sherpa_onnx.OnlineRecognizer's own surface used by this add-on."""

    def __init__(self) -> None:
        self.created_streams: list[_FakeOnlineStream] = []
        self._ready_calls = 0

    def create_stream(self, hotwords: str | None = None) -> _FakeOnlineStream:
        stream = _FakeOnlineStream(hotwords)
        self.created_streams.append(stream)
        return stream

    def is_ready(self, stream: _FakeOnlineStream) -> bool:
        return False  # never any pending decode work in this fake

    def decode_stream(self, stream: _FakeOnlineStream) -> None:
        raise AssertionError("is_ready() returned False; decode_stream() should not be called")

    def get_result(self, stream: _FakeOnlineStream) -> str:
        return "schalte das licht im wohnzimmer ein"


def _fake_model_files(tmp_path) -> KrokoModelFiles:
    return KrokoModelFiles(
        encoder=tmp_path / "encoder.onnx",
        decoder=tmp_path / "decoder.onnx",
        joiner=tmp_path / "joiner.onnx",
        tokens=tmp_path / "tokens.txt",
    )


def test_kroko_engine_capabilities(tmp_path) -> None:
    engine = KrokoSttEngine(_FakeOnlineRecognizer(), _fake_model_files(tmp_path))
    assert engine.capabilities == SttCapabilities(
        supports_streaming=True,
        supports_hotwords=True,
        supports_dynamic_vocabulary=True,
        supports_partial_results=False,
    )
    assert engine.engine_id == "kroko"
    assert engine.program_name == "homeintent-kroko"


def test_kroko_engine_set_keyterms_never_raises_and_joins_with_newlines(tmp_path) -> None:
    engine = KrokoSttEngine(_FakeOnlineRecognizer(), _fake_model_files(tmp_path))
    accepted, succeeded = engine.set_keyterms(["Wohnzimmer", "Büro"])
    assert succeeded is True
    assert accepted == ["Wohnzimmer", "Büro"]
    assert engine._hotwords == "Wohnzimmer\nBüro"


def test_kroko_engine_set_keyterms_empty_list_clears_hotwords(tmp_path) -> None:
    engine = KrokoSttEngine(_FakeOnlineRecognizer(), _fake_model_files(tmp_path))
    engine.set_keyterms(["Wohnzimmer"])
    accepted, succeeded = engine.set_keyterms([])
    assert accepted == []
    assert succeeded is True
    assert engine._hotwords is None


@pytest.mark.asyncio
async def test_kroko_session_full_lifecycle_uses_current_hotwords(tmp_path) -> None:
    recognizer = _FakeOnlineRecognizer()
    engine = KrokoSttEngine(recognizer, _fake_model_files(tmp_path))
    engine.set_keyterms(["Küche"])

    session = engine.create_session(lock=asyncio.Lock())
    assert isinstance(session, KrokoSttSession)
    await session.start()
    assert recognizer.created_streams[-1].hotwords == "Küche"

    await session.add_audio([0.0] * 1600, sample_rate=16000)
    assert session.audio_duration_seconds == pytest.approx(0.1)
    assert session.add_audio_chunk_count == 1

    text = await session.finalize()
    assert text == "schalte das licht im wohnzimmer ein"
    assert recognizer.created_streams[-1].finished is True
    session.close()


@pytest.mark.asyncio
async def test_kroko_session_new_session_after_set_keyterms_picks_up_new_hotwords(
    tmp_path,
) -> None:
    """Dynamic vocabulary: a NEW session created after set_keyterms() uses
    the updated hotwords, with no model reload -- see kroko_engine.py's
    module docstring."""
    recognizer = _FakeOnlineRecognizer()
    engine = KrokoSttEngine(recognizer, _fake_model_files(tmp_path))

    session1 = engine.create_session()
    await session1.start()
    assert recognizer.created_streams[0].hotwords is None

    engine.set_keyterms(["Schlafzimmer"])
    session2 = engine.create_session()
    await session2.start()
    assert recognizer.created_streams[1].hotwords == "Schlafzimmer"
