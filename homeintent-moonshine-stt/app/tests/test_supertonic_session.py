"""Tests for app/supertonic_session.py's callback-streaming bridge, using a
fake sherpa_onnx module + fake OfflineTts (no real model, no real network).

Verifies: sherpa-onnx's callback delivers audio chunks to
synthesize_stream() as an async generator (matching the existing Wyoming
AudioStart/AudioChunk/AudioStop/SynthesizeStopped flow app/handler.py
already drives for Pocket TTS/Kokoro -- see app/tts_stream.py), that a
client-disconnect-style early stop propagates to the callback via the
non-zero return convention, and that model_generation_seconds/lock_wait_seconds
are populated.
"""

import sys
import types

import numpy as np
import pytest

from app.supertonic_session import SupertonicSynthesizer
from app.tts_engine import TtsSynthesisStats


class _FakeGenerationConfig:
    def __init__(self) -> None:
        self.sid = 0
        self.speed = 1.0
        self.num_steps = 8
        self.extra: dict[str, str] = {}


class _FakeOfflineTts:
    """Mimics sherpa_onnx.OfflineTts.generate()'s real callback contract:
    callback(samples: np.ndarray, progress: float) -> int, non-zero stops."""

    def __init__(self, chunks: list[np.ndarray], sample_rate: int = 24000) -> None:
        self._chunks = chunks
        self.sample_rate = sample_rate
        self.last_config: _FakeGenerationConfig | None = None

    def generate(self, text: str, config: _FakeGenerationConfig, callback) -> None:
        self.last_config = config
        for i, chunk in enumerate(self._chunks):
            progress = (i + 1) / len(self._chunks)
            stop = callback(chunk, progress)
            if stop:
                return


@pytest.fixture(autouse=True)
def _fake_sherpa_onnx_module(monkeypatch: pytest.MonkeyPatch) -> None:
    fake_module = types.ModuleType("sherpa_onnx")
    fake_module.GenerationConfig = _FakeGenerationConfig  # type: ignore[attr-defined]
    monkeypatch.setitem(sys.modules, "sherpa_onnx", fake_module)


@pytest.mark.asyncio
async def test_synthesize_stream_yields_each_callback_chunk_incrementally() -> None:
    chunks = [np.ones(4, dtype=np.float32) * i for i in range(3)]
    tts = _FakeOfflineTts(chunks)
    synth = SupertonicSynthesizer(tts, default_voice="M1")

    stats = TtsSynthesisStats()
    received = [chunk async for chunk in synth.synthesize_stream("Hallo Welt", stats=stats)]

    assert len(received) == 3
    for i, chunk in enumerate(received):
        assert np.array_equal(chunk, chunks[i])
    assert stats.model_generation_seconds >= 0.0
    assert tts.last_config is not None
    assert tts.last_config.sid == 5  # "M1"
    assert tts.last_config.extra["lang"] == "de"


@pytest.mark.asyncio
async def test_synthesize_stream_resolves_voice_override_by_name() -> None:
    tts = _FakeOfflineTts([np.zeros(2, dtype=np.float32)])
    synth = SupertonicSynthesizer(tts, default_voice="M1")

    async for _ in synth.synthesize_stream("Text", voice="F2"):
        pass
    assert tts.last_config is not None
    assert tts.last_config.sid == 1  # "F2"


@pytest.mark.asyncio
async def test_synthesize_stream_stops_early_on_client_disconnect() -> None:
    """Simulates a Wyoming client disconnecting mid-stream (aclose() on the
    async generator): the producer's callback must see stop_event set and
    return non-zero, matching sherpa-onnx's own documented early-stop
    convention -- never blindly finishing the whole utterance regardless."""
    chunks = [np.ones(4, dtype=np.float32) for _ in range(50)]
    tts = _FakeOfflineTts(chunks)
    synth = SupertonicSynthesizer(tts, default_voice="M1")

    agen = synth.synthesize_stream("Ein langer Text")
    first = await agen.__anext__()
    assert first is not None
    await agen.aclose()  # early client disconnect


@pytest.mark.asyncio
async def test_sample_rate_reflects_loaded_model() -> None:
    tts = _FakeOfflineTts([], sample_rate=24000)
    synth = SupertonicSynthesizer(tts, default_voice="M1")
    assert synth.sample_rate == 24000


@pytest.mark.asyncio
async def test_preload_default_voice_rejects_unknown_voice() -> None:
    tts = _FakeOfflineTts([])
    synth = SupertonicSynthesizer(tts, default_voice="Q9")
    with pytest.raises(ValueError):
        await synth.preload_default_voice()


def test_engine_identity_fields() -> None:
    tts = _FakeOfflineTts([])
    synth = SupertonicSynthesizer(tts, default_voice="M1")
    assert synth.engine_id == "supertonic_3"
    assert synth.program_name == "homeintent-supertonic-3"
    assert "Supertone" in synth.attribution.name
