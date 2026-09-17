"""Tests that concurrent Wyoming sessions on the same Transcriber are
serialized through a shared asyncio.Lock, since moonshine-voice 0.1.5 does
not document create_stream()/Stream native calls as thread-safe (see
app/streaming.py's MoonshineStreamingSession docstring).
"""

import asyncio
from unittest.mock import MagicMock

import pytest

from app.streaming import MoonshineStreamingSession


def _make_transcriber_with_streams() -> tuple[MagicMock, list[MagicMock]]:
    """A mock Transcriber whose create_stream() returns a fresh mock Stream
    each call (unlike the shared mock_stream fixture), so two sessions can
    be told apart.
    """
    transcriber = MagicMock()
    streams: list[MagicMock] = []

    def _create_stream(update_interval: float | None = None) -> MagicMock:
        stream = MagicMock()
        streams.append(stream)
        return stream

    transcriber.create_stream.side_effect = _create_stream
    return transcriber, streams


class TestSharedLockSerializesConcurrentSessions:
    @pytest.mark.asyncio
    async def test_two_sessions_never_run_native_calls_concurrently(self):
        transcriber, streams = _make_transcriber_with_streams()
        shared_lock = asyncio.Lock()

        in_native_call = 0
        overlap_detected = False

        def blocking_add_audio(samples: list[float], sample_rate: int) -> None:
            nonlocal in_native_call, overlap_detected
            in_native_call += 1
            if in_native_call > 1:
                overlap_detected = True
            import time

            time.sleep(0.05)
            in_native_call -= 1

        for stream in streams:
            stream.add_audio.side_effect = blocking_add_audio

        session_a = MoonshineStreamingSession(transcriber, lock=shared_lock)
        session_b = MoonshineStreamingSession(transcriber, lock=shared_lock)
        for stream in streams:
            stream.add_audio.side_effect = blocking_add_audio

        await asyncio.gather(
            session_a.add_audio([0.0] * 10),
            session_b.add_audio([0.0] * 10),
        )

        assert not overlap_detected

    @pytest.mark.asyncio
    async def test_sessions_without_shared_lock_use_independent_locks(self):
        """Documents current behavior: omitting the shared lock (e.g. a
        single-session unit test) falls back to a session-local lock, which
        is fine in isolation but does not serialize across sibling
        sessions -- production code (app/__main__.py) always passes one
        shared lock per Transcriber.
        """
        transcriber, streams = _make_transcriber_with_streams()

        session_a = MoonshineStreamingSession(transcriber)
        session_b = MoonshineStreamingSession(transcriber)

        assert session_a._lock is not session_b._lock
