"""Regression test: ensures add_audio() is called per chunk, not after audio-stop.

This test is CRITICAL to prevent the architecture from regressing to a
buffering-based approach (like cronus42's implementation, which accumulates
all audio and only calls transcribe() once at audio-stop).

The test verifies:
1. Stream.add_audio() is called PER Wyoming audio-chunk (streaming)
2. Stream.add_audio() is NOT deferred until audio-stop
3. Stream.stop() is called exactly once, after all chunks

This guarantees real-time streaming instead of full-utterance buffering.
"""

import pytest
from wyoming.audio import AudioChunk, AudioStart
from wyoming.event import Event

from app.handler import MoonshineAsrHandler


def make_audio_start_event() -> Event:
    return AudioStart(rate=16000, width=2, channels=1).event()


def make_audio_chunk_event(audio_bytes: bytes) -> Event:
    return AudioChunk(rate=16000, width=2, channels=1, audio=audio_bytes).event()


def make_audio_stop_event() -> Event:
    return Event(type="audio-stop", data={})


@pytest.mark.asyncio
async def test_add_audio_called_per_chunk_not_buffered(
    mock_transcriber, mock_stream, mock_reader, mock_writer
):
    """CRITICAL: Verify Stream.add_audio() is called per chunk, not buffered."""
    handler = MoonshineAsrHandler(
        reader=mock_reader,
        writer=mock_writer,
        transcriber=mock_transcriber,
        model_name="small",
        language="de",
    )

    await handler.handle_event(Event(type="describe", data={}))
    await handler.handle_event(Event(type="transcribe", data={}))

    await handler.handle_event(make_audio_start_event())
    assert handler._session is not None, "Session should be created on audio-start"

    # CHUNK 1
    chunk1 = b"\x00\x00" * 500  # 500 samples of silence
    await handler.handle_event(make_audio_chunk_event(chunk1))

    assert mock_stream.add_audio.call_count == 1, (
        "add_audio should be called once after first chunk (streaming)"
    )

    # CHUNK 2
    chunk2 = b"\xff\x7f" * 500
    await handler.handle_event(make_audio_chunk_event(chunk2))

    assert mock_stream.add_audio.call_count == 2, (
        "add_audio should be called twice after second chunk (streaming, not buffered)"
    )

    # stop() must not have fired yet
    mock_stream.stop.assert_not_called()

    await handler.handle_event(make_audio_stop_event())

    mock_stream.stop.assert_called_once()
    assert mock_writer.write.called


@pytest.mark.asyncio
async def test_empty_stream_handled(mock_transcriber, mock_stream, mock_reader, mock_writer):
    """Test that an empty audio stream never calls add_audio."""
    handler = MoonshineAsrHandler(
        reader=mock_reader,
        writer=mock_writer,
        transcriber=mock_transcriber,
        model_name="small",
        language="de",
    )

    await handler.handle_event(Event(type="transcribe", data={}))
    await handler.handle_event(make_audio_start_event())

    await handler.handle_event(make_audio_stop_event())

    mock_stream.add_audio.assert_not_called()
    mock_stream.stop.assert_called_once()


@pytest.mark.asyncio
async def test_no_buffering_of_audio_chunks(
    mock_transcriber, mock_stream, mock_reader, mock_writer
):
    """Verify each chunk is forwarded on its own, without accumulation."""
    add_audio_calls = []

    def track_add_audio(samples, sample_rate):
        add_audio_calls.append({"num_samples": len(samples), "sample_rate": sample_rate})

    mock_stream.add_audio.side_effect = track_add_audio

    handler = MoonshineAsrHandler(
        reader=mock_reader,
        writer=mock_writer,
        transcriber=mock_transcriber,
        model_name="small",
        language="de",
    )

    await handler.handle_event(Event(type="transcribe", data={}))
    await handler.handle_event(make_audio_start_event())

    for _ in range(3):
        await handler.handle_event(make_audio_chunk_event(b"\x00\x00" * 100))

    assert len(add_audio_calls) == 3, "Should call add_audio 3 times for 3 chunks"

    for call in add_audio_calls:
        assert call["num_samples"] == 100, f"Each chunk should be separate: {call}"
        assert call["sample_rate"] == 16000

    await handler.handle_event(make_audio_stop_event())
