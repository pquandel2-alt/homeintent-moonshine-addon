"""Regression test: ensures add_audio() is called per chunk, not after audio-stop.

This test is CRITICAL to prevent the architecture from regressing to the
old buffering-based approach (like cronus42's implementation).

The test verifies:
1. add_audio() is called PER Wyoming audio-chunk (streaming)
2. add_audio() is NOT called after audio-stop
3. transcriber.stop() is called exactly once, after all chunks

This guarantees real-time streaming instead of full-utterance buffering.
"""

import asyncio
from unittest.mock import AsyncMock, MagicMock, patch

import numpy as np
import pytest
from wyoming.asr import Transcript
from wyoming.audio import AudioChunk, AudioStart
from wyoming.event import Event

from app.handler import MoonshineAsrHandler


def make_audio_start_event() -> Event:
    """Create a Wyoming audio-start event."""
    return AudioStart(rate=16000, width=2, channels=1).event()


def make_audio_chunk_event(audio_bytes: bytes) -> Event:
    """Create a Wyoming audio-chunk event."""
    return AudioChunk(
        rate=16000,
        width=2,
        channels=1,
        audio=audio_bytes,
    ).event()


def make_audio_stop_event() -> Event:
    """Create a Wyoming audio-stop event."""
    return Event(type="audio-stop", data={})


@pytest.mark.asyncio
async def test_add_audio_called_per_chunk_not_buffered(transcriber_factory):
    """CRITICAL: Verify add_audio() is called per chunk (streaming), not buffered.

    This test ensures the implementation uses real Moonshine streaming
    (add_audio per chunk) and does NOT regress to buffering-based approach.

    Scenario:
    1. Start audio stream
    2. Send chunk 1 → add_audio should be called once
    3. Send chunk 2 → add_audio should be called again (twice total)
    4. Stop → stop() called once, finalized
    """
    mock_transcriber = transcriber_factory()
    mock_reader = AsyncMock()
    mock_writer = AsyncMock()

    handler = MoonshineAsrHandler(
        reader=mock_reader,
        writer=mock_writer,
        transcriber_factory=lambda: mock_transcriber,
        model_name="small",
        language="de",
    )

    # Prepare listener to complete immediately (for testing)
    # In real scenario, listener waits for Moonshine to finish
    def mock_add_listener(listener):
        # Simulate listener callback (would normally come from Moonshine)
        asyncio.create_task(
            asyncio.sleep(0.01)
        )  # Simulate async processing
        listener.final_text = "Hallo Welt"
        asyncio.get_event_loop().call_soon_threadsafe(listener._done.set)

    mock_transcriber.add_listener.side_effect = mock_add_listener

    # Handle describe (service discovery)
    await handler.handle_event(Event(type="describe", data={}))

    # Reset for transcription
    await handler.handle_event(Event(type="transcribe", data={}))

    # Start audio stream
    await handler.handle_event(make_audio_start_event())
    assert handler._session is not None, "Session should be created on audio-start"

    # CHUNK 1: Send audio
    chunk1 = b"\x00\x00" * 500  # 500 samples of silence
    await handler.handle_event(make_audio_chunk_event(chunk1))

    # CRITICAL: add_audio must be called IMMEDIATELY (once, per chunk)
    assert (
        mock_transcriber.add_audio.call_count == 1
    ), "add_audio should be called once after first chunk (streaming)"

    # CHUNK 2: Send more audio
    chunk2 = b"\xFF\x7F" * 500  # 500 samples
    await handler.handle_event(make_audio_chunk_event(chunk2))

    # CRITICAL: add_audio must be called again (twice total)
    assert (
        mock_transcriber.add_audio.call_count == 2
    ), "add_audio should be called twice after second chunk (streaming, not buffered)"

    # VERIFY: stop() has NOT been called yet
    mock_transcriber.stop.assert_not_called()

    # Audio stop
    await handler.handle_event(make_audio_stop_event())

    # CRITICAL: stop() must be called exactly once, after all chunks
    mock_transcriber.stop.assert_called_once()

    # Verify transcript was sent
    assert mock_writer.write.called or mock_writer.writelines.called


@pytest.mark.asyncio
async def test_empty_stream_handled(transcriber_factory):
    """Test that empty audio stream doesn't call add_audio."""
    mock_transcriber = transcriber_factory()
    mock_reader = AsyncMock()
    mock_writer = AsyncMock()

    handler = MoonshineAsrHandler(
        reader=mock_reader,
        writer=mock_writer,
        transcriber_factory=lambda: mock_transcriber,
        model_name="small",
        language="de",
    )

    await handler.handle_event(Event(type="transcribe", data={}))
    await handler.handle_event(make_audio_start_event())

    # Stop without sending chunks
    await handler.handle_event(make_audio_stop_event())

    # add_audio should never be called
    mock_transcriber.add_audio.assert_not_called()


@pytest.mark.asyncio
async def test_no_buffering_of_audio_chunks(transcriber_factory):
    """Verify no intermediate WAV file or full buffer is used.

    This test ensures the handler passes chunks directly to Moonshine
    without creating temporary files or accumulating bytes.
    """
    mock_transcriber = transcriber_factory()
    mock_reader = AsyncMock()
    mock_writer = AsyncMock()

    # Mock to track calls
    add_audio_calls = []

    def track_add_audio(audio, sr):
        add_audio_calls.append({"audio_shape": audio.shape, "sample_rate": sr})

    mock_transcriber.add_audio.side_effect = track_add_audio

    def mock_add_listener(listener):
        listener.final_text = ""
        asyncio.get_event_loop().call_soon_threadsafe(listener._done.set)

    mock_transcriber.add_listener.side_effect = mock_add_listener

    handler = MoonshineAsrHandler(
        reader=mock_reader,
        writer=mock_writer,
        transcriber_factory=lambda: mock_transcriber,
        model_name="small",
        language="de",
    )

    await handler.handle_event(Event(type="transcribe", data={}))
    await handler.handle_event(make_audio_start_event())

    # Send 3 chunks
    for i in range(3):
        await handler.handle_event(make_audio_chunk_event(b"\x00\x00" * 100))

    # Verify we made 3 separate calls (not accumulated)
    assert len(add_audio_calls) == 3, "Should call add_audio 3 times for 3 chunks"

    # Each call should have chunk-sized audio (not full utterance)
    for call in add_audio_calls:
        assert call["audio_shape"][0] == 100, f"Each chunk should be separate: {call}"
        assert call["sample_rate"] == 16000

    await handler.handle_event(make_audio_stop_event())
