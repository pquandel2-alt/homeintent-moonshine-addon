"""Tests for Wyoming event handler."""

import pytest
from moonshine_voice import LineCompleted, TranscriptLine
from wyoming.audio import AudioChunk, AudioStart
from wyoming.event import Event

from app.handler import MoonshineAsrHandler


def _line(text: str) -> TranscriptLine:
    return TranscriptLine(
        text=text, start_time=0.0, duration=1.0, line_id=1, is_complete=True, is_updated=True
    )


@pytest.mark.asyncio
async def test_describe_handler(mock_transcriber, mock_reader, mock_writer):
    """Test service discovery describe handler."""
    handler = MoonshineAsrHandler(
        reader=mock_reader,
        writer=mock_writer,
        transcriber=mock_transcriber,
        model_name="small",
        language="de",
    )

    await handler.handle_event(Event(type="describe", data={}))

    assert mock_writer.write.called
    assert mock_writer.drain.await_count == 1


@pytest.mark.asyncio
async def test_transcribe_resets_session(mock_transcriber, mock_stream, mock_reader, mock_writer):
    """Test that transcribe event resets session state."""
    handler = MoonshineAsrHandler(
        reader=mock_reader,
        writer=mock_writer,
        transcriber=mock_transcriber,
        model_name="small",
        language="de",
    )

    await handler.handle_event(AudioStart(rate=16000, width=2, channels=1).event())
    assert handler._session is not None

    await handler.handle_event(Event(type="transcribe", data={}))

    assert handler._session is None
    mock_stream.close.assert_called_once()


@pytest.mark.asyncio
async def test_audio_start_creates_session(mock_transcriber, mock_reader, mock_writer):
    """Test that audio-start creates a new session for valid audio format."""
    handler = MoonshineAsrHandler(
        reader=mock_reader,
        writer=mock_writer,
        transcriber=mock_transcriber,
        model_name="small",
        language="de",
    )

    audio_start = AudioStart(rate=16000, width=2, channels=1)
    await handler.handle_event(audio_start.event())

    assert handler._session is not None
    mock_transcriber.create_stream.assert_called_once()


@pytest.mark.asyncio
async def test_audio_chunk_without_start_ignored(mock_transcriber, mock_reader, mock_writer):
    """Test that audio-chunk before audio-start is safely ignored."""
    handler = MoonshineAsrHandler(
        reader=mock_reader,
        writer=mock_writer,
        transcriber=mock_transcriber,
        model_name="small",
        language="de",
    )

    chunk = AudioChunk(rate=16000, width=2, channels=1, audio=b"\x00\x00" * 100)

    await handler.handle_event(chunk.event())

    assert handler._session is None


@pytest.mark.asyncio
async def test_invalid_audio_format_rejected(
    mock_transcriber, mock_stream, mock_reader, mock_writer
):
    """Test that an invalid audio format is strictly rejected (no session, Error event sent)."""
    handler = MoonshineAsrHandler(
        reader=mock_reader,
        writer=mock_writer,
        transcriber=mock_transcriber,
        model_name="small",
        language="de",
    )

    # Wrong sample rate (44100 instead of 16000) must be rejected, not silently accepted.
    audio_start = AudioStart(rate=44100, width=2, channels=1)
    await handler.handle_event(audio_start.event())

    assert handler._session is None
    mock_transcriber.create_stream.assert_not_called()
    assert mock_writer.write.called


@pytest.mark.asyncio
async def test_audio_stop_sends_transcript(mock_transcriber, mock_stream, mock_reader, mock_writer):
    """Test that audio-stop triggers transcript send with the accumulated text."""

    def fake_stop():
        # Stream.stop() synchronously fires LineCompleted before returning.
        for call in mock_stream.add_listener.call_args_list:
            listener = call.args[0]
            listener.on_line_completed(
                LineCompleted(line=_line("test transcript"), stream_handle=1)
            )

    mock_stream.stop.side_effect = fake_stop

    handler = MoonshineAsrHandler(
        reader=mock_reader,
        writer=mock_writer,
        transcriber=mock_transcriber,
        model_name="small",
        language="de",
    )

    await handler.handle_event(AudioStart(rate=16000, width=2, channels=1).event())
    await handler.handle_event(Event(type="audio-stop", data={}))

    mock_stream.stop.assert_called_once()
    mock_stream.close.assert_called_once()
    assert handler._session is None


@pytest.mark.asyncio
async def test_audio_stop_without_session_sends_empty_transcript(
    mock_transcriber, mock_reader, mock_writer
):
    """Test that audio-stop without a prior session sends an empty transcript."""
    handler = MoonshineAsrHandler(
        reader=mock_reader,
        writer=mock_writer,
        transcriber=mock_transcriber,
        model_name="small",
        language="de",
    )

    await handler.handle_event(Event(type="audio-stop", data={}))

    assert mock_writer.write.called
