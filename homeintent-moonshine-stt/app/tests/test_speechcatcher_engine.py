"""Tests for app/speechcatcher_engine.py using a fake Speech2TextStreaming
double (no real speechcatcher/torch/espnet_streaming_decoder install, no
real model download) -- these verify the streaming/session wiring and the
honest "hotwords not supported" dispatch, which is real, in-process code
this add-on owns. A real end-to-end Speechcatcher test against the actual
downloaded model is gated behind RUN_SPEECHCATCHER_E2E=1 (see
test_e2e_speechcatcher_stt.py).
"""

import asyncio

import numpy as np
import pytest

from app.speechcatcher_engine import SpeechcatcherSttEngine, SpeechcatcherSttSession
from app.stt_engine import SttCapabilities


class _FakeSpeech2TextStreaming:
    """Mimics the real, upstream-verified surface this add-on drives:
    ``__call__(speech, is_final, always_assemble_hyps)`` and ``reset()``
    (see app/speechcatcher_engine.py's module docstring)."""

    def __init__(self, final_text: str = "schalte das licht im wohnzimmer ein") -> None:
        self.final_text = final_text
        self.reset_calls = 0
        self.calls: list[tuple[np.ndarray, bool, bool]] = []

    def reset(self) -> None:
        self.reset_calls += 1

    def __call__(
        self, speech: np.ndarray, is_final: bool = False, always_assemble_hyps: bool = True
    ) -> list[tuple[str, list[str], list[int]]]:
        self.calls.append((np.asarray(speech), is_final, always_assemble_hyps))
        if is_final:
            return [(self.final_text, [], [])]
        return []


def test_speechcatcher_engine_capabilities_are_honest_about_no_hotwords() -> None:
    engine = SpeechcatcherSttEngine(_FakeSpeech2TextStreaming(), "speechcatcher_m")
    assert engine.capabilities == SttCapabilities(
        supports_streaming=True,
        supports_hotwords=False,
        supports_dynamic_vocabulary=False,
        supports_partial_results=False,
    )
    assert engine.engine_id == "speechcatcher_m"
    assert engine.program_name == "homeintent-speechcatcher-m"


def test_speechcatcher_engine_l_variant_ids() -> None:
    engine = SpeechcatcherSttEngine(_FakeSpeech2TextStreaming(), "speechcatcher_l")
    assert engine.engine_id == "speechcatcher_l"
    assert engine.program_name == "homeintent-speechcatcher-l"
    assert engine.model_display_name == "speechcatcher-de-streaming-transformer-l"


def test_speechcatcher_set_keyterms_never_applies_and_logs(caplog) -> None:
    engine = SpeechcatcherSttEngine(_FakeSpeech2TextStreaming(), "speechcatcher_m")
    with caplog.at_level("INFO"):
        accepted, succeeded = engine.set_keyterms(["Wohnzimmer", "Büro"])
    assert accepted == []
    assert succeeded is True
    assert any("not supported by engine speechcatcher_m" in msg for msg in caplog.messages)


def test_speechcatcher_set_keyterms_empty_list_logs_nothing(caplog) -> None:
    engine = SpeechcatcherSttEngine(_FakeSpeech2TextStreaming(), "speechcatcher_m")
    with caplog.at_level("INFO"):
        accepted, succeeded = engine.set_keyterms([])
    assert accepted == []
    assert succeeded is True
    assert not caplog.messages


@pytest.mark.asyncio
async def test_speechcatcher_session_full_lifecycle() -> None:
    model = _FakeSpeech2TextStreaming()
    engine = SpeechcatcherSttEngine(model, "speechcatcher_m")

    session = engine.create_session(lock=asyncio.Lock())
    assert isinstance(session, SpeechcatcherSttSession)
    await session.start()
    assert model.reset_calls == 1

    await session.add_audio([0.0] * 1600, sample_rate=16000)
    assert session.audio_duration_seconds == pytest.approx(0.1)
    assert session.add_audio_chunk_count == 1
    assert model.calls[0][1] is False  # is_final=False during add_audio
    assert model.calls[0][2] is False  # always_assemble_hyps=False during add_audio

    text = await session.finalize()
    assert text == "schalte das licht im wohnzimmer ein"
    assert model.calls[-1][1] is True  # is_final=True during finalize
    assert model.calls[-1][2] is True  # always_assemble_hyps=True during finalize
    session.close()


@pytest.mark.asyncio
async def test_speechcatcher_session_finalize_with_no_hyps_returns_empty_string() -> None:
    class _EmptyModel(_FakeSpeech2TextStreaming):
        def __call__(
            self, speech: np.ndarray, is_final: bool = False, always_assemble_hyps: bool = True
        ) -> list[tuple[str, list[str], list[int]]]:
            return []

    engine = SpeechcatcherSttEngine(_EmptyModel(), "speechcatcher_m")
    session = engine.create_session(lock=asyncio.Lock())
    await session.start()
    text = await session.finalize()
    assert text == ""
    session.close()


@pytest.mark.asyncio
async def test_speechcatcher_session_start_resets_state_defensively() -> None:
    """Guards against state leaking between Wyoming connections if a
    previous connection dropped mid-utterance without ever reaching
    finalize() (whose is_final=True call already resets internally)."""
    model = _FakeSpeech2TextStreaming()
    engine = SpeechcatcherSttEngine(model, "speechcatcher_m")

    session1 = engine.create_session()
    await session1.start()
    await session1.add_audio([0.0] * 800)
    # session1 "drops" here without calling finalize()

    session2 = engine.create_session()
    await session2.start()
    assert model.reset_calls == 2
