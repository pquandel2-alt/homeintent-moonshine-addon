"""Tests for the Wyoming streaming-TTS protocol (synthesize-start/-chunk/
-stop/-stopped) in MoonshineAsrHandler.

Home Assistant's real Wyoming TTS client (verified against
homeassistant/components/wyoming/tts.py upstream) sends, for a streaming-
capable server: synthesize-start, one or more synthesize-chunk, the ENTIRE
message again via a backwards-compatible synthesize event, then
synthesize-stop -- and its reader loop only ends on synthesize-stopped.
These tests exercise exactly that sequence (and edge cases around it)
against the real handler and a real byte-level Wyoming connection (see
test_handler_tts.py's module docstring for why a Mock writer won't do).
"""

import asyncio

import numpy as np
import pytest
from wyoming.error import Error as WyomingError
from wyoming.event import Event, async_read_event
from wyoming.info import Attribution, Info
from wyoming.tts import Synthesize, SynthesizeChunk, SynthesizeStart, SynthesizeStop

from app.handler import MoonshineAsrHandler


class _FakeSynthesizer:
    """Stands in for app.tts_session.PocketTtsSynthesizer.

    Records every ``text`` it was asked to synthesize (in call order) so
    tests can assert synthesis happened exactly once per logical request,
    even though Home Assistant sends the same full text twice on the wire
    (once as a synthesize-chunk, once as the backwards-compatible
    synthesize event).
    """

    engine_id = "pocket_tts"
    model_name = "german"
    program_name = "homeintent-pocket-tts"
    description = "HomeIntent Pocket TTS - German streaming TTS"
    attribution = Attribution(name="Kyutai", url="https://github.com/kyutai-labs/pocket-tts")

    def __init__(self, chunks: list[list[float]] | None = None, default_voice: str = "juergen"):
        self.sample_rate = 24000
        self.default_voice = default_voice
        self._chunks = chunks if chunks is not None else [[0.1, 0.2], [0.3, 0.4]]
        self.requested_texts: list[str] = []
        self.requested_voices: list[str | None] = []
        self.raise_error: BaseException | None = None

    async def synthesize_stream(self, text: str, voice: str | None = None, stats=None):
        self.requested_texts.append(text)
        self.requested_voices.append(voice)
        if self.raise_error is not None:
            raise self.raise_error
        for chunk in self._chunks:
            yield np.array(chunk, dtype=np.float32)


class _CollectingWriter:
    """A minimal real asyncio.StreamWriter stand-in that actually records
    the wire bytes in call order, so they can be decoded back into events."""

    def __init__(self) -> None:
        self.buffer = bytearray()

    def writelines(self, chunks) -> None:
        for chunk in chunks:
            self.buffer.extend(chunk)

    def write(self, data: bytes) -> None:
        self.buffer.extend(data)

    async def drain(self) -> None:
        pass

    def close(self) -> None:
        pass

    async def wait_closed(self) -> None:
        pass

    def get_extra_info(self, name: str, default=None):
        return ("127.0.0.1", 12345)

    def is_closing(self) -> bool:
        return False


async def _read_all_events(writer: _CollectingWriter) -> list[Event]:
    reader = asyncio.StreamReader()
    reader.feed_data(bytes(writer.buffer))
    reader.feed_eof()
    events = []
    while True:
        event = await async_read_event(reader)
        if event is None:
            break
        events.append(event)
    return events


@pytest.fixture
def collecting_writer() -> _CollectingWriter:
    return _CollectingWriter()


def _make_handler(mock_reader, collecting_writer, synth) -> MoonshineAsrHandler:
    return MoonshineAsrHandler(
        reader=mock_reader,
        writer=collecting_writer,
        transcriber=None,
        tts_synthesizer=synth,
    )


@pytest.mark.asyncio
class TestDescribeAdvertisesStreaming:
    async def test_supports_synthesize_streaming_is_true(self, mock_reader, collecting_writer):
        synth = _FakeSynthesizer()
        handler = _make_handler(mock_reader, collecting_writer, synth)
        await handler.handle_event(Event(type="describe", data={}))

        events = await _read_all_events(collecting_writer)
        info = Info.from_event(events[0])
        assert info.tts[0].supports_synthesize_streaming is True


@pytest.mark.asyncio
class TestStreamingSingleChunk:
    """The common real-world case: one synthesize-chunk carrying the whole
    HomeIntent response, then the backwards-compatible synthesize event
    with the same text, then synthesize-stop."""

    async def test_full_sequence_produces_audio_and_exactly_one_synthesize_stopped(
        self, mock_reader, collecting_writer
    ):
        synth = _FakeSynthesizer(chunks=[[0.1, 0.2], [0.3, 0.4]])
        handler = _make_handler(mock_reader, collecting_writer, synth)
        text = "Das Küchenfenster und das Schlafzimmerfenster sind geöffnet."

        await handler.handle_event(SynthesizeStart().event())
        await handler.handle_event(SynthesizeChunk(text=text).event())
        await handler.handle_event(Synthesize(text=text).event())
        await handler.handle_event(SynthesizeStop().event())

        events = await _read_all_events(collecting_writer)
        types = [e.type for e in events]
        assert types == [
            "audio-start",
            "audio-chunk",
            "audio-chunk",
            "audio-stop",
            "synthesize-stopped",
        ]
        assert types.count("audio-start") == 1
        assert types.count("synthesize-stopped") == 1

    async def test_synthesizes_exactly_once_not_twice(self, mock_reader, collecting_writer):
        """Core correctness requirement: the backwards-compatible
        synthesize event (carrying the identical full text again) must
        never cause a second call into Pocket TTS."""
        synth = _FakeSynthesizer()
        handler = _make_handler(mock_reader, collecting_writer, synth)
        text = "Hallo Welt."

        await handler.handle_event(SynthesizeStart().event())
        await handler.handle_event(SynthesizeChunk(text=text).event())
        await handler.handle_event(Synthesize(text=text).event())
        await handler.handle_event(SynthesizeStop().event())

        assert synth.requested_texts == [text]


@pytest.mark.asyncio
class TestStreamingMultipleChunks:
    async def test_multiple_chunks_are_concatenated_for_the_fallback_path(
        self, mock_reader, collecting_writer
    ):
        """If a client never sends the backwards-compatible synthesize
        event, synthesize-stop must trigger synthesis using the
        concatenated chunk text."""
        synth = _FakeSynthesizer()
        handler = _make_handler(mock_reader, collecting_writer, synth)

        await handler.handle_event(SynthesizeStart().event())
        await handler.handle_event(SynthesizeChunk(text="Das Küchenfenster ").event())
        await handler.handle_event(SynthesizeChunk(text="ist geöffnet.").event())
        await handler.handle_event(SynthesizeStop().event())

        assert synth.requested_texts == ["Das Küchenfenster ist geöffnet."]

        events = await _read_all_events(collecting_writer)
        types = [e.type for e in events]
        assert types.count("audio-start") == 1
        assert types.count("synthesize-stopped") == 1

    async def test_multiple_chunks_plus_compat_synthesize_still_synthesizes_once(
        self, mock_reader, collecting_writer
    ):
        synth = _FakeSynthesizer()
        handler = _make_handler(mock_reader, collecting_writer, synth)
        full_text = "Das Küchenfenster ist geöffnet."

        await handler.handle_event(SynthesizeStart().event())
        await handler.handle_event(SynthesizeChunk(text="Das Küchenfenster ").event())
        await handler.handle_event(SynthesizeChunk(text="ist geöffnet.").event())
        await handler.handle_event(Synthesize(text=full_text).event())
        await handler.handle_event(SynthesizeStop().event())

        assert synth.requested_texts == [full_text]


@pytest.mark.asyncio
class TestStreamingEdgeCases:
    async def test_synthesize_stop_without_start_is_ignored(self, mock_reader, collecting_writer):
        synth = _FakeSynthesizer()
        handler = _make_handler(mock_reader, collecting_writer, synth)

        await handler.handle_event(SynthesizeStop().event())

        events = await _read_all_events(collecting_writer)
        assert events == []
        assert synth.requested_texts == []

    async def test_chunk_without_start_is_ignored(self, mock_reader, collecting_writer):
        synth = _FakeSynthesizer()
        handler = _make_handler(mock_reader, collecting_writer, synth)

        await handler.handle_event(SynthesizeChunk(text="orphan").event())

        events = await _read_all_events(collecting_writer)
        assert events == []
        assert synth.requested_texts == []

    async def test_empty_text_streaming_request_sends_stopped_event(
        self, mock_reader, collecting_writer
    ):
        synth = _FakeSynthesizer()
        handler = _make_handler(mock_reader, collecting_writer, synth)

        await handler.handle_event(SynthesizeStart().event())
        await handler.handle_event(SynthesizeChunk(text="   ").event())
        await handler.handle_event(Synthesize(text="   ").event())
        await handler.handle_event(SynthesizeStop().event())

        events = await _read_all_events(collecting_writer)
        types = [e.type for e in events]
        assert types == ["audio-start", "audio-stop", "synthesize-stopped"]
        assert synth.requested_texts == []

    async def test_tts_disabled_answers_synthesize_start_with_error(
        self, mock_reader, collecting_writer
    ):
        handler = _make_handler(mock_reader, collecting_writer, None)

        await handler.handle_event(SynthesizeStart().event())

        events = await _read_all_events(collecting_writer)
        assert any(WyomingError.is_type(e.type) for e in events)

    async def test_error_during_streaming_synthesis_sends_error_and_stopped(
        self, mock_reader, collecting_writer
    ):
        synth = _FakeSynthesizer()
        synth.raise_error = RuntimeError("model exploded")
        handler = _make_handler(mock_reader, collecting_writer, synth)
        text = "Hallo Welt."

        await handler.handle_event(SynthesizeStart().event())
        await handler.handle_event(SynthesizeChunk(text=text).event())
        await handler.handle_event(Synthesize(text=text).event())
        await handler.handle_event(SynthesizeStop().event())

        events = await _read_all_events(collecting_writer)
        types = [e.type for e in events]
        assert types.count("audio-start") == 1
        assert "error" in types
        assert types.count("synthesize-stopped") == 1

    async def test_disconnect_during_collection_does_not_deadlock_next_request(
        self, mock_reader, collecting_writer
    ):
        """A client disconnect between synthesize-start and synthesize-stop
        must not leave the handler stuck thinking a request is still being
        collected."""
        synth = _FakeSynthesizer()
        handler = _make_handler(mock_reader, collecting_writer, synth)

        await handler.handle_event(SynthesizeStart().event())
        await handler.handle_event(SynthesizeChunk(text="unfinished").event())
        await handler.disconnect()

        assert synth.requested_texts == []  # never triggered

    async def test_two_consecutive_streaming_requests_on_same_connection(
        self, mock_reader, collecting_writer
    ):
        synth = _FakeSynthesizer()
        handler = _make_handler(mock_reader, collecting_writer, synth)

        await handler.handle_event(SynthesizeStart().event())
        await handler.handle_event(SynthesizeChunk(text="Erste Anfrage.").event())
        await handler.handle_event(Synthesize(text="Erste Anfrage.").event())
        await handler.handle_event(SynthesizeStop().event())

        await handler.handle_event(SynthesizeStart().event())
        await handler.handle_event(SynthesizeChunk(text="Zweite Anfrage.").event())
        await handler.handle_event(Synthesize(text="Zweite Anfrage.").event())
        await handler.handle_event(SynthesizeStop().event())

        assert synth.requested_texts == ["Erste Anfrage.", "Zweite Anfrage."]

        events = await _read_all_events(collecting_writer)
        types = [e.type for e in events]
        assert types.count("synthesize-stopped") == 2
        assert types.count("audio-start") == 2

    async def test_legacy_synthesize_after_a_finished_streaming_request_still_works(
        self, mock_reader, collecting_writer
    ):
        """A bare, non-streaming synthesize request on the same connection
        after a completed streaming one must still take the legacy path
        (no synthesize-stopped) rather than getting stuck in a stale
        streaming phase."""
        synth = _FakeSynthesizer()
        handler = _make_handler(mock_reader, collecting_writer, synth)

        await handler.handle_event(SynthesizeStart().event())
        await handler.handle_event(SynthesizeChunk(text="Streaming-Anfrage.").event())
        await handler.handle_event(Synthesize(text="Streaming-Anfrage.").event())
        await handler.handle_event(SynthesizeStop().event())

        await handler.handle_event(Synthesize(text="Legacy-Anfrage.").event())

        assert synth.requested_texts == ["Streaming-Anfrage.", "Legacy-Anfrage."]
        events = await _read_all_events(collecting_writer)
        types = [e.type for e in events]
        assert types.count("synthesize-stopped") == 1  # only for the streaming request


@pytest.mark.asyncio
class TestNoDeadlockUnderTimeout:
    async def test_full_streaming_sequence_completes_within_timeout(
        self, mock_reader, collecting_writer
    ):
        """Guards against a regression where the handler could hang
        waiting on an event that never triggers synthesis or a response."""
        synth = _FakeSynthesizer()
        handler = _make_handler(mock_reader, collecting_writer, synth)
        text = "Hallo Welt."

        async def _run() -> None:
            await handler.handle_event(SynthesizeStart().event())
            await handler.handle_event(SynthesizeChunk(text=text).event())
            await handler.handle_event(Synthesize(text=text).event())
            await handler.handle_event(SynthesizeStop().event())

        await asyncio.wait_for(_run(), timeout=5.0)

        events = await _read_all_events(collecting_writer)
        assert events[-1].type == "synthesize-stopped"
