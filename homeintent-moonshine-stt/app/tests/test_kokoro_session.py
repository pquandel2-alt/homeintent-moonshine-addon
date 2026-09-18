"""Tests for KokoroOnnxSynthesizer (app/kokoro_session.py).

Uses a fake ``kokoro_onnx.Kokoro``-shaped object (create_stream/get_voices)
-- no real ONNX model, no real espeak-ng phonemization needed to test the
session/streaming/locking logic itself.
"""

import asyncio

import numpy as np
import pytest
from wyoming.info import Attribution

from app.kokoro_session import ENGINE_ID, KokoroOnnxSynthesizer
from app.tts_engine import TtsSynthesisStats


class _FakeKokoro:
    """Stands in for kokoro_onnx.Kokoro."""

    def __init__(
        self,
        chunks: list[list[float]] | None = None,
        voices: list[str] | None = None,
        raise_error: BaseException | None = None,
    ) -> None:
        self._chunks = chunks if chunks is not None else [[0.1, 0.2], [0.3, 0.4]]
        self._voices = voices if voices is not None else ["martin"]
        self.raise_error = raise_error
        self.calls: list[dict[str, object]] = []

    def get_voices(self) -> list[str]:
        return self._voices

    async def create_stream(self, text, voice, speed=1.0, lang="en-us", **kwargs):
        self.calls.append({"text": text, "voice": voice, "speed": speed, "lang": lang, **kwargs})
        if self.raise_error is not None:
            raise self.raise_error
        for chunk in self._chunks:
            await asyncio.sleep(0)  # yield control, like the real async generator
            yield np.array(chunk, dtype=np.float32), 24000


@pytest.mark.asyncio
class TestMetadata:
    async def test_engine_id(self):
        synth = KokoroOnnxSynthesizer(_FakeKokoro())
        assert synth.engine_id == "kokoro_onnx" == ENGINE_ID

    async def test_sample_rate_is_24000(self):
        synth = KokoroOnnxSynthesizer(_FakeKokoro())
        assert synth.sample_rate == 24000

    async def test_default_voice_is_martin(self):
        synth = KokoroOnnxSynthesizer(_FakeKokoro())
        assert synth.default_voice == "martin"

    async def test_program_name(self):
        synth = KokoroOnnxSynthesizer(_FakeKokoro())
        assert synth.program_name == "homeintent-kokoro-onnx"

    async def test_attribution_is_not_kyutai(self):
        """Regression guard: Kokoro must never display Pocket TTS's
        (Kyutai's) attribution."""
        synth = KokoroOnnxSynthesizer(_FakeKokoro())
        assert isinstance(synth.attribution, Attribution)
        assert "kyutai" not in synth.attribution.name.lower()
        assert "kyutai" not in synth.attribution.url.lower()


@pytest.mark.asyncio
class TestPreloadDefaultVoice:
    async def test_valid_voice_passes(self):
        synth = KokoroOnnxSynthesizer(_FakeKokoro(voices=["martin"]), default_voice="martin")
        await synth.preload_default_voice()  # does not raise

    async def test_invalid_voice_raises(self):
        synth = KokoroOnnxSynthesizer(_FakeKokoro(voices=["martin"]), default_voice="juergen")
        with pytest.raises(ValueError, match="juergen"):
            await synth.preload_default_voice()


@pytest.mark.asyncio
class TestSynthesizeStream:
    async def test_yields_all_chunks_as_float32(self):
        fake = _FakeKokoro(chunks=[[0.1, 0.2], [0.3, 0.4, 0.5]])
        synth = KokoroOnnxSynthesizer(fake)
        chunks = [c async for c in synth.synthesize_stream("Hallo", "martin")]
        assert len(chunks) == 2
        assert chunks[0].dtype == np.float32
        assert list(chunks[0]) == [0.1, 0.2]
        assert list(chunks[1]) == [0.3, 0.4, 0.5]

    async def test_default_voice_used_when_none_requested(self):
        fake = _FakeKokoro()
        synth = KokoroOnnxSynthesizer(fake, default_voice="martin")
        async for _ in synth.synthesize_stream("Hallo", None):
            pass
        assert fake.calls[0]["voice"] == "martin"

    async def test_requested_voice_overrides_default(self):
        fake = _FakeKokoro()
        synth = KokoroOnnxSynthesizer(fake, default_voice="martin")
        async for _ in synth.synthesize_stream("Hallo", "other-voice"):
            pass
        assert fake.calls[0]["voice"] == "other-voice"

    async def test_lang_is_always_de(self):
        fake = _FakeKokoro()
        synth = KokoroOnnxSynthesizer(fake)
        async for _ in synth.synthesize_stream("Hallo", "martin"):
            pass
        assert fake.calls[0]["lang"] == "de"

    async def test_speed_and_pauses_forwarded(self):
        fake = _FakeKokoro()
        synth = KokoroOnnxSynthesizer(fake, speed=1.25, sentence_pause=0.3, clause_pause=0.05)
        async for _ in synth.synthesize_stream("Hallo", "martin"):
            pass
        call = fake.calls[0]
        assert call["speed"] == 1.25
        assert call["sentence_pause"] == 0.3
        assert call["clause_pause"] == 0.05

    async def test_text_is_normalized_before_synthesis(self):
        """German text normalization (times/units/currency) must run
        before the text reaches Kokoro -- see
        app/german_text_normalizer.py."""
        fake = _FakeKokoro()
        synth = KokoroOnnxSynthesizer(fake)
        async for _ in synth.synthesize_stream("45 Min.", "martin"):
            pass
        assert fake.calls[0]["text"] == "45 Minuten."

    async def test_stats_populated(self):
        fake = _FakeKokoro(chunks=[[0.1, 0.2]])
        synth = KokoroOnnxSynthesizer(fake)
        stats = TtsSynthesisStats()
        async for _ in synth.synthesize_stream("Hallo", "martin", stats):
            pass
        assert stats.model_generation_seconds >= 0.0
        assert stats.lock_wait_seconds >= 0.0

    async def test_error_during_generation_propagates(self):
        fake = _FakeKokoro(raise_error=RuntimeError("kokoro exploded"))
        synth = KokoroOnnxSynthesizer(fake)
        with pytest.raises(RuntimeError, match="kokoro exploded"):
            async for _ in synth.synthesize_stream("Hallo", "martin"):
                pass

    async def test_early_stop_still_releases_lock_for_next_request(self):
        """Simulates a client disconnect mid-stream (consumer stops
        iterating early) -- the shared lock must not be left held."""
        fake = _FakeKokoro(chunks=[[0.1, 0.2], [0.3, 0.4], [0.5, 0.6]])
        synth = KokoroOnnxSynthesizer(fake)

        agen = synth.synthesize_stream("Hallo", "martin")
        await agen.__anext__()  # consume exactly one chunk
        await agen.aclose()

        # A second, independent request must not deadlock waiting for the
        # lock the first request's generator held.
        async def _second_request() -> int:
            count = 0
            async for _ in synth.synthesize_stream("Zweite Anfrage", "martin"):
                count += 1
            return count

        count = await asyncio.wait_for(_second_request(), timeout=5.0)
        assert count == 3


@pytest.mark.asyncio
class TestLocking:
    async def test_concurrent_requests_are_serialized_not_interleaved(self):
        """Two concurrent synthesize_stream() calls against the same
        synthesizer must not interleave their create_stream() calls --
        matches PocketTtsSynthesizer's own "correctness over parallelism"
        policy for an engine whose concurrent-request safety is not
        documented upstream (see app/kokoro_session.py's module
        docstring)."""
        fake = _FakeKokoro(chunks=[[0.1], [0.2]])
        synth = KokoroOnnxSynthesizer(fake)
        order: list[str] = []

        async def _run(label: str) -> None:
            async for _ in synth.synthesize_stream(label, "martin"):
                order.append(f"{label}:chunk")
            order.append(f"{label}:done")

        await asyncio.gather(_run("A"), _run("B"))

        # Whichever ran first, it must fully complete before the other
        # one's chunks start -- never interleaved.
        first_label = order[0].split(":")[0]
        first_block = [o for o in order if o.startswith(first_label)]
        assert order[: len(first_block)] == first_block


@pytest.mark.asyncio
class TestWarmup:
    async def test_warmup_returns_elapsed_time_and_discards_audio(self):
        fake = _FakeKokoro(chunks=[[0.1, 0.2]])
        synth = KokoroOnnxSynthesizer(fake)
        elapsed = await synth.warmup("Eins, zwei, drei.")
        assert elapsed >= 0.0
        assert fake.calls[0]["text"] == "Eins, zwei, drei."
