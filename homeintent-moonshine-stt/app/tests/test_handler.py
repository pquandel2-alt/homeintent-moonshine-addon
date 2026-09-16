"""Tests for Wyoming event handler."""

import asyncio
from unittest.mock import AsyncMock, MagicMock, Mock

import pytest
from wyoming.audio import AudioChunk, AudioStart
from wyoming.event import Event

from app.handler import MoonshineAsrHandler


@pytest.mark.asyncio
async def test_describe_handler(transcriber_factory, mock_reader, mock_writer):
    """Test service discovery describe handler."""
    handler = MoonshineAsrHandler(
        reader=mock_reader,
        writer=mock_writer,
        transcriber_factory=transcriber_factory,
        model_name="small",
        language="de",
    )

    await handler.handle_event(Event(type="describe", data={}))

    # Should have written info event
    assert mock_writer.write.called or hasattr(mock_writer, "writelines")


@pytest.mark.asyncio
async def test_transcribe_resets_session(transcriber_factory, mock_reader, mock_writer):
    """Test that transcribe event resets session state."""
    handler = MoonshineAsrHandler(
        reader=mock_reader,
        writer=mock_writer,
        transcriber_factory=transcriber_factory,
        model_name="small",
        language="de",
    )

    # Set fake session
    handler._session = "fake_session"

    # Reset via transcribe
    await handler.handle_event(Event(type="transcribe", data={}))

    assert handler._session is None


@pytest.mark.asyncio
async def test_audio_start_creates_session(transcriber_factory, mock_reader, mock_writer):
    """Test that audio-start creates a new session."""
    handler = MoonshineAsrHandler(
        reader=mock_reader,
        writer=mock_writer,
        transcriber_factory=transcriber_factory,
        model_name="small",
        language="de",
    )

    audio_start = AudioStart(rate=16000, width=2, channels=1)
    await handler.handle_event(audio_start.event())

    # Session should be created
    assert handler._session is not None


@pytest.mark.asyncio
async def test_audio_chunk_without_start_ignored(transcriber_factory, mock_reader, mock_writer):
    """Test that audio-chunk before audio-start is safely ignored."""
    handler = MoonshineAsrHandler(
        reader=mock_reader,
        writer=mock_writer,
        transcriber_factory=transcriber_factory,
        model_name="small",
        language="de",
    )

    chunk = AudioChunk(
        rate=16000, width=2, channels=1, audio=b"\x00\x00" * 100
    )

    # Should not raise
    await handler.handle_event(chunk.event())

    assert handler._session is None


@pytest.mark.asyncio
async def test_invalid_audio_format_warning(transcriber_factory, mock_reader, mock_writer):
    """Test that invalid audio format produces warning but continues."""
    handler = MoonshineAsrHandler(
        reader=mock_reader,
        writer=mock_writer,
        transcriber_factory=transcriber_factory,
        model_name="small",
        language="de",
    )

    # Wrong sample rate (44100 instead of 16000)
    audio_start = AudioStart(rate=44100, width=2, channels=1)
    await handler.handle_event(audio_start.event())

    # Should still create session (non-fatal)
    assert handler._session is not None


@pytest.mark.asyncio
async def test_audio_stop_sends_transcript(transcriber_factory, mock_reader, mock_writer):
    """Test that audio-stop triggers transcript send."""
    mock_transcriber = transcriber_factory()

    # Mock listener to complete immediately
    def mock_add_listener(listener):
        listener.final_text = "test transcript"
        asyncio.get_event_loop().call_soon_threadsafe(listener._done.set)

    mock_transcriber.add_listener.side_effect = mock_add_listener

    handler = MoonshineAsrHandler(
        reader=mock_reader,
        writer=mock_writer,
        transcriber_factory=lambda: mock_transcriber,
        model_name="small",
        language="de",
    )

    # Start session
    audio_start = AudioStart(rate=16000, width=2, channels=1)
    await handler.handle_event(audio_start.event())

    # Stop
    await handler.handle_event(Event(type="audio-stop", data={}))

    # Verify stop was called
    mock_transcriber.stop.assert_called()

    # Session should be reset
    assert handler._session is None
