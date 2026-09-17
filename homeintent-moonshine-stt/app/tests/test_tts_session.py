"""Tests for Pocket TTS synthesis session management (app/tts_session.py).

Pocket TTS itself is never imported for real here -- generate_audio_stream()
is faked with a small object exposing the same .detach().cpu().numpy()
chain a real torch.Tensor chunk would (see app/tts.py's module docstring:
upstream yields flat 1D float32 tensors).
"""

import asyncio
import time

import numpy as np
import pytest

from app.tts_session import PocketTtsSynthesizer


class _FakeTensor:
    """Stands in for a torch.Tensor chunk yielded by generate_audio_stream()."""

    def __init__(self, samples: list[float]) -> None:
        self._array = np.array(samples, dtype=np.float32)

    def detach(self) -> "_FakeTensor":
        return self

    def cpu(self) -> "_FakeTensor":
        return self

    def numpy(self) -> np.ndarray:
        return self._array


class _FakeTtsModel:
    """Stands in for pocket_tts.TTSModel."""

    def __init__(self, chunks_by_text: dict[str, list[list[float]]] | None = None) -> None:
        self.sample_rate = 24000
        self._chunks_by_text = chunks_by_text or {}
        self.get_state_calls: list[str] = []
        self.generate_calls: list[tuple[object, str]] = []
        self.raise_on_generate: BaseException | None = None
        self._active_generations = 0
        self.max_concurrent_generations = 0

    def get_state_for_audio_prompt(self, voice: str) -> dict[str, str]:
        self.get_state_calls.append(voice)
        return {"voice": voice}

    def generate_audio_stream(self, voice_state: object, text: str):
        self.generate_calls.append((voice_state, text))
        if self.raise_on_generate is not None:
            raise self.raise_on_generate
        self._active_generations += 1
        self.max_concurrent_generations = max(
            self.max_concurrent_generations, self._active_generations
        )
        try:
            for chunk in self._chunks_by_text.get(text, [[0.1, 0.2, 0.3]]):
                yield _FakeTensor(chunk)
        finally:
            self._active_generations -= 1


class TestSampleRate:
    @pytest.mark.asyncio
    async def test_sample_rate_from_model(self):
        model = _FakeTtsModel()
        synth = PocketTtsSynthesizer(model, default_voice="juergen")
        assert synth.sample_rate == 24000


class TestSynthesizeStream:
    @pytest.mark.asyncio
    async def test_yields_chunks_from_model(self):
        model = _FakeTtsModel({"Hallo Welt": [[0.1, 0.2], [0.3, 0.4]]})
        synth = PocketTtsSynthesizer(model, default_voice="juergen")

        chunks = [chunk async for chunk in synth.synthesize_stream("Hallo Welt")]

        assert len(chunks) == 2
        np.testing.assert_allclose(chunks[0], [0.1, 0.2])
        np.testing.assert_allclose(chunks[1], [0.3, 0.4])

    @pytest.mark.asyncio
    async def test_uses_default_voice_when_none_given(self):
        model = _FakeTtsModel()
        synth = PocketTtsSynthesizer(model, default_voice="juergen")

        async for _ in synth.synthesize_stream("Text"):
            pass

        assert model.get_state_calls == ["juergen"]

    @pytest.mark.asyncio
    async def test_uses_requested_voice_override(self):
        model = _FakeTtsModel()
        synth = PocketTtsSynthesizer(model, default_voice="juergen")

        async for _ in synth.synthesize_stream("Text", voice="alba"):
            pass

        assert model.get_state_calls == ["alba"]

    @pytest.mark.asyncio
    async def test_voice_state_is_cached_across_requests(self):
        model = _FakeTtsModel()
        synth = PocketTtsSynthesizer(model, default_voice="juergen")

        async for _ in synth.synthesize_stream("Erster Satz"):
            pass
        async for _ in synth.synthesize_stream("Zweiter Satz"):
            pass

        # get_state_for_audio_prompt() is documented upstream as "relatively
        # slow" -- must only be called once per distinct voice name.
        assert model.get_state_calls == ["juergen"]

    @pytest.mark.asyncio
    async def test_synthesis_error_propagates_to_caller(self):
        model = _FakeTtsModel()
        model.raise_on_generate = RuntimeError("synthesis boom")
        synth = PocketTtsSynthesizer(model, default_voice="juergen")

        with pytest.raises(RuntimeError, match="synthesis boom"):
            async for _ in synth.synthesize_stream("Text"):
                pass

    @pytest.mark.asyncio
    async def test_early_close_stops_iteration_cleanly(self):
        """Simulates a client disconnect mid-stream (B26): aclose() must
        not raise and must release the lock for the next request."""
        model = _FakeTtsModel({"Long text": [[0.1], [0.2], [0.3], [0.4], [0.5]]})
        synth = PocketTtsSynthesizer(model, default_voice="juergen")

        agen = synth.synthesize_stream("Long text")
        first = await agen.__anext__()
        assert first is not None
        await agen.aclose()

        # Lock must be free again for a subsequent request.
        async with asyncio.timeout(1.0):
            async for _ in synth.synthesize_stream("Long text"):
                pass


class TestConcurrencySerialization:
    @pytest.mark.asyncio
    async def test_concurrent_requests_are_serialized_not_interleaved(self):
        """Two simultaneous TTS requests on the same model must never
        interleave their generate_audio_stream() calls (B10): the second
        request's consumption must not start until the first one's has
        fully finished, since both share one lock."""
        model = _FakeTtsModel({"A": [[0.1]] * 3, "B": [[0.2]] * 3})
        synth = PocketTtsSynthesizer(model, default_voice="juergen")
        events: list[str] = []

        async def consume(text: str) -> None:
            events.append(f"{text}:start")
            async for _ in synth.synthesize_stream(text):
                await asyncio.sleep(0)
            events.append(f"{text}:end")

        await asyncio.gather(consume("A"), consume("B"))

        assert sorted(events) == ["A:end", "A:start", "B:end", "B:start"]
        assert len(model.generate_calls) == 2
        # The real assertion: never more than one generate_audio_stream()
        # call actually running at the same time, whatever order they
        # started in.
        assert model.max_concurrent_generations == 1


class TestThreadSafetyDocumentation:
    @pytest.mark.asyncio
    async def test_lock_is_held_for_full_synthesis_not_released_between_chunks(self):
        model = _FakeTtsModel({"Text": [[0.1], [0.2]]})
        synth = PocketTtsSynthesizer(model, default_voice="juergen")
        lock_states: list[bool] = []

        agen = synth.synthesize_stream("Text")
        async for _ in agen:
            lock_states.append(synth._lock.locked())

        assert all(lock_states), "lock must stay held for the whole streamed request"
        assert not synth._lock.locked(), "lock must be released once the request completes"


def test_real_time_elapsed_does_not_block_producer_unreasonably():
    """Smoke test: consuming a short stream completes quickly (the
    thread+queue plumbing in synthesize_stream() must not add significant
    overhead for a handful of instantly-available fake chunks)."""

    async def _run() -> float:
        model = _FakeTtsModel({"Text": [[0.1]] * 5})
        synth = PocketTtsSynthesizer(model, default_voice="juergen")
        started = time.monotonic()
        async for _ in synth.synthesize_stream("Text"):
            pass
        return time.monotonic() - started

    elapsed = asyncio.run(_run())
    assert elapsed < 2.0
