"""Tests for Moonshine streaming session management."""

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
