"""Tests for the v0.6.1 Speechcatcher concurrency fix (see
app/speechcatcher_engine.py's ``SpeechcatcherSttSession`` docstring).

Uses the same fake ``Speech2TextStreaming`` double as
test_speechcatcher_engine.py -- no real speechcatcher/torch install needed.
These tests exercise ONE shared ``asyncio.Lock`` across two sessions,
exactly like app/__main__.py wires ``moonshine_lock`` into every Wyoming
connection's handler, to prove the lock is held for a whole session's
lifetime (start() through close()) rather than re-acquired per native call.
"""

import asyncio

import numpy as np
import pytest

from app.speechcatcher_engine import SpeechcatcherSttEngine


class _FakeSpeech2TextStreaming:
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


@pytest.mark.asyncio
async def test_second_session_start_blocks_until_first_session_closes() -> None:
    """Session A starts and holds the shared lock across add_audio(); a
    concurrent Session B's start() must genuinely block (not interleave)
    until A calls close() -- proves the lock is held for the whole session
    lifetime, not released between A's own start()/add_audio() calls."""
    model = _FakeSpeech2TextStreaming()
    engine = SpeechcatcherSttEngine(model, "speechcatcher_m")
    shared_lock = asyncio.Lock()

    session_a = engine.create_session(lock=shared_lock)
    session_b = engine.create_session(lock=shared_lock)

    await session_a.start()
    await session_a.add_audio([0.0] * 1600)

    events: list[str] = []

    async def run_b() -> None:
        await session_b.start()
        events.append("b_started")

    task_b = asyncio.create_task(run_b())
    # Give task_b every chance to run; it must still be blocked on the lock
    # since session_a never released it.
    await asyncio.sleep(0.05)
    assert not task_b.done()
    assert events == []

    # A finishes its lifecycle (finalize() + close(), exactly like
    # app/handler.py's own `finally: self._close_session()`).
    await session_a.finalize()
    session_a.close()

    await asyncio.wait_for(task_b, timeout=1.0)
    assert events == ["b_started"]
    session_b.close()


@pytest.mark.asyncio
async def test_no_interleaving_between_two_concurrent_sessions() -> None:
    """Session B's reset() (called from start()) must never land between
    session A's own add_audio() calls -- the historical state-corruption
    bug this fix addresses."""
    model = _FakeSpeech2TextStreaming()
    engine = SpeechcatcherSttEngine(model, "speechcatcher_m")
    shared_lock = asyncio.Lock()

    session_a = engine.create_session(lock=shared_lock)
    session_b = engine.create_session(lock=shared_lock)

    order: list[str] = []

    async def run_a() -> None:
        await session_a.start()
        order.append("a_start")
        await session_a.add_audio([0.0] * 1600)
        order.append("a_add_audio_1")
        await asyncio.sleep(0.01)
        await session_a.add_audio([0.0] * 1600)
        order.append("a_add_audio_2")
        await session_a.finalize()
        order.append("a_finalize")
        session_a.close()
        order.append("a_close")

    async def run_b() -> None:
        await asyncio.sleep(0.005)  # start slightly after A
        await session_b.start()
        order.append("b_start")
        session_b.close()

    await asyncio.gather(run_a(), run_b())

    # Every one of A's own steps must appear as a contiguous, uninterrupted
    # block before b_start -- b_start can only appear after a_close.
    a_index = order.index("a_close")
    b_index = order.index("b_start")
    assert b_index > a_index, f"Session B interleaved into A's lifecycle: {order}"
    assert model.reset_calls == 2  # one from A.start(), one from B.start()


@pytest.mark.asyncio
async def test_exception_during_add_audio_releases_lock() -> None:
    """An exception raised out of add_audio() must not leak the lock held
    forever -- app/handler.py always calls close() in this path (see its
    own `except Exception: self._close_session(); raise`)."""

    class _RaisingModel(_FakeSpeech2TextStreaming):
        def __call__(
            self, speech: np.ndarray, is_final: bool = False, always_assemble_hyps: bool = True
        ) -> list[tuple[str, list[str], list[int]]]:
            raise RuntimeError("boom")

    engine = SpeechcatcherSttEngine(_RaisingModel(), "speechcatcher_m")
    shared_lock = asyncio.Lock()
    session_a = engine.create_session(lock=shared_lock)
    session_b = engine.create_session(lock=shared_lock)

    await session_a.start()
    with pytest.raises(RuntimeError):
        await session_a.add_audio([0.0] * 1600)
    # app/handler.py's own exception path always calls close() next.
    session_a.close()

    # The lock must be free again -- session_b can acquire it without
    # blocking.
    await asyncio.wait_for(session_b.start(), timeout=1.0)
    session_b.close()


@pytest.mark.asyncio
async def test_close_is_idempotent_and_safe_to_call_multiple_times() -> None:
    model = _FakeSpeech2TextStreaming()
    engine = SpeechcatcherSttEngine(model, "speechcatcher_m")
    shared_lock = asyncio.Lock()
    session_a = engine.create_session(lock=shared_lock)
    session_b = engine.create_session(lock=shared_lock)

    await session_a.start()
    session_a.close()
    session_a.close()  # must not raise / double-release
    session_a.close()

    # Lock genuinely released exactly once -- B can still acquire it.
    await asyncio.wait_for(session_b.start(), timeout=1.0)
    session_b.close()
    session_b.close()  # idempotent here too


@pytest.mark.asyncio
async def test_simulated_mid_stream_disconnect_releases_lock() -> None:
    """Mirrors a client disconnecting mid-utterance: app/handler.py's
    connection_lost()/eof_received() path calls close() directly, without
    ever reaching finalize()."""
    model = _FakeSpeech2TextStreaming()
    engine = SpeechcatcherSttEngine(model, "speechcatcher_m")
    shared_lock = asyncio.Lock()
    session_a = engine.create_session(lock=shared_lock)
    session_b = engine.create_session(lock=shared_lock)

    await session_a.start()
    await session_a.add_audio([0.0] * 1600)
    # Client disconnects here -- no finalize(), straight to close().
    session_a.close()

    await asyncio.wait_for(session_b.start(), timeout=1.0)
    session_b.close()


@pytest.mark.asyncio
async def test_cancelled_task_still_releases_lock_via_close() -> None:
    """A cancelled asyncio task must still release the lock through
    app/handler.py's own finally-block pattern (`_close_session()` runs in
    a finally around finalize())."""
    model = _FakeSpeech2TextStreaming()
    engine = SpeechcatcherSttEngine(model, "speechcatcher_m")
    shared_lock = asyncio.Lock()
    session_a = engine.create_session(lock=shared_lock)
    session_b = engine.create_session(lock=shared_lock)

    started = asyncio.Event()

    async def run_a() -> None:
        await session_a.start()
        started.set()
        try:
            await session_a.add_audio([0.0] * 1600)
            await asyncio.sleep(10)  # will be cancelled here
        finally:
            session_a.close()

    task_a = asyncio.create_task(run_a())
    await started.wait()
    await asyncio.sleep(0.01)
    task_a.cancel()
    with pytest.raises(asyncio.CancelledError):
        await task_a

    await asyncio.wait_for(session_b.start(), timeout=1.0)
    session_b.close()
