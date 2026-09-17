"""Tests for Moonshine streaming session management."""

import time

import pytest
from moonshine_voice import Error, LineCompleted, TranscriptLine

from app.streaming import MoonshineStreamingSession, _CompletedLineCollector


def _line(text: str, *, is_complete: bool = True, is_updated: bool = True) -> TranscriptLine:
    return TranscriptLine(
        text=text,
        start_time=0.0,
        duration=1.0,
        line_id=1,
        is_complete=is_complete,
        is_updated=is_updated,
    )


class TestCompletedLineCollector:
    """Tests for the LineCompleted accumulator."""

    def test_accumulates_ordered_completed_lines(self):
        collector = _CompletedLineCollector()
        collector.on_line_completed(LineCompleted(line=_line("Hallo"), stream_handle=1))
        collector.on_line_completed(LineCompleted(line=_line("Welt"), stream_handle=1))

        assert collector.text == "Hallo Welt"

    def test_ignores_empty_text(self):
        collector = _CompletedLineCollector()
        collector.on_line_completed(LineCompleted(line=_line(""), stream_handle=1))

        assert collector.text == ""

    def test_records_error(self):
        collector = _CompletedLineCollector()
        err = RuntimeError("native failure")
        collector.on_error(Error(line=None, stream_handle=1, error=err))

        assert collector.error is err


class TestMoonshineStreamingSession:
    """Tests for the per-connection streaming session."""

    def test_session_creates_stream_and_registers_listener(self, mock_transcriber, mock_stream):
        session = MoonshineStreamingSession(mock_transcriber)

        mock_transcriber.create_stream.assert_called_once()
        mock_stream.add_listener.assert_called_once_with(session._collector)

    @pytest.mark.asyncio
    async def test_start_calls_stream_start(self, mock_transcriber, mock_stream):
        session = MoonshineStreamingSession(mock_transcriber)
        await session.start()

        mock_stream.start.assert_called_once()

    @pytest.mark.asyncio
    async def test_add_audio_forwards_to_stream(self, mock_transcriber, mock_stream):
        session = MoonshineStreamingSession(mock_transcriber)
        samples = [0.1, 0.2, 0.3]

        await session.add_audio(samples)

        mock_stream.add_audio.assert_called_once_with(samples, 16000)

    @pytest.mark.asyncio
    async def test_finalize_calls_stop_and_returns_accumulated_text(
        self, mock_transcriber, mock_stream
    ):
        session = MoonshineStreamingSession(mock_transcriber)

        def fake_stop():
            session._collector.on_line_completed(
                LineCompleted(line=_line("Hallo Welt"), stream_handle=1)
            )

        mock_stream.stop.side_effect = fake_stop

        result = await session.finalize()

        mock_stream.stop.assert_called_once()
        assert result == "Hallo Welt"

    @pytest.mark.asyncio
    async def test_finalize_multiple_completed_lines(self, mock_transcriber, mock_stream):
        session = MoonshineStreamingSession(mock_transcriber)

        def fake_stop():
            session._collector.on_line_completed(
                LineCompleted(line=_line("erste Zeile"), stream_handle=1)
            )
            session._collector.on_line_completed(
                LineCompleted(line=_line("zweite Zeile"), stream_handle=1)
            )

        mock_stream.stop.side_effect = fake_stop

        result = await session.finalize()

        assert result == "erste Zeile zweite Zeile"

    @pytest.mark.asyncio
    async def test_finalize_raises_on_stream_error(self, mock_transcriber, mock_stream):
        session = MoonshineStreamingSession(mock_transcriber)
        boom = RuntimeError("native failure")

        def fake_stop():
            session._collector.on_error(Error(line=None, stream_handle=1, error=boom))

        mock_stream.stop.side_effect = fake_stop

        with pytest.raises(RuntimeError, match="native failure"):
            await session.finalize()

    def test_close_releases_stream(self, mock_transcriber, mock_stream):
        session = MoonshineStreamingSession(mock_transcriber)
        session.close()

        mock_stream.close.assert_called_once()


class TestRealInferenceTiming:
    """A2 review fix: RTF must reflect real Moonshine compute time, not the
    wall time the Wyoming client spent streaming audio in."""

    @pytest.mark.asyncio
    async def test_inference_time_accumulates_across_add_audio_calls(
        self, mock_transcriber, mock_stream
    ):
        def slow_add_audio(_samples, _rate):
            time.sleep(0.02)

        mock_stream.add_audio.side_effect = slow_add_audio
        session = MoonshineStreamingSession(mock_transcriber)

        await session.add_audio([0.0] * 100)
        await session.add_audio([0.0] * 100)

        # Two ~20ms native calls: cumulative, not just the last one.
        assert session.inference_time_seconds >= 0.03
        # finalize() has not run yet, so it must be 0 -- not folded in early.
        assert session.finalize_time_seconds == 0.0

    @pytest.mark.asyncio
    async def test_finalize_time_is_folded_into_inference_time(self, mock_transcriber, mock_stream):
        def slow_stop():
            time.sleep(0.02)

        mock_stream.stop.side_effect = slow_stop
        session = MoonshineStreamingSession(mock_transcriber)

        await session.finalize()

        assert session.finalize_time_seconds >= 0.015
        assert session.inference_time_seconds == pytest.approx(
            session.finalize_time_seconds, abs=0.005
        )

    @pytest.mark.asyncio
    async def test_inference_time_excludes_wyoming_stream_wait(self, mock_transcriber, mock_stream):
        """Time spent awaiting audio from the Wyoming client between chunks
        must never be counted as inference time -- only add_audio()/stop()
        themselves are timed."""
        session = MoonshineStreamingSession(mock_transcriber)

        await session.add_audio([0.0] * 100)
        # Simulate a long pause where the *user* is talking slowly, i.e.
        # time the session is simply not doing anything -- not measured.
        import asyncio

        await asyncio.sleep(0.05)
        await session.add_audio([0.0] * 100)

        # Both add_audio() calls used an instant mock, so inference time
        # must stay near zero despite the 50ms real-world gap between them.
        assert session.inference_time_seconds < 0.01
