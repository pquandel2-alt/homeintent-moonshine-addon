"""Tests for Wyoming TTS handling in MoonshineAsrHandler (Pocket TTS).

Uses a real byte-accumulating fake StreamWriter (not a bare Mock) because
wyoming's async_write_event() splits one event across separate
writer.writelines()/writer.write() calls (header line, then a JSON data
payload, then a binary audio payload) -- a Mock can't be decoded back into
events without actually capturing that exact wire sequence.
"""

import asyncio

import numpy as np
import pytest
from wyoming.audio import AudioChunk, AudioStart, AudioStop
from wyoming.error import Error as WyomingError
from wyoming.event import Event, async_read_event
from wyoming.info import Info
from wyoming.tts import Synthesize, SynthesizeVoice

from app.handler import MoonshineAsrHandler


class _FakeSynthesizer:
    """Stands in for app.tts_session.PocketTtsSynthesizer."""

    def __init__(self, chunks: list[list[float]] | None = None, default_voice: str = "juergen"):
        self.sample_rate = 24000
        self.default_voice = default_voice
        self._chunks = chunks if chunks is not None else [[0.1, 0.2], [0.3, 0.4]]
        self.requested_voices: list[str | None] = []
        self.raise_error: BaseException | None = None

    async def synthesize_stream(self, text: str, voice: str | None = None):
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


@pytest.mark.asyncio
class TestDescribeDiscovery:
    async def test_asr_only_when_tts_disabled(
        self, mock_transcriber, mock_reader, collecting_writer
    ):
        handler = MoonshineAsrHandler(
            reader=mock_reader,
            writer=collecting_writer,
            transcriber=mock_transcriber,
            model_name="small",
            language="de",
            tts_synthesizer=None,
        )
        await handler.handle_event(Event(type="describe", data={}))

        events = await _read_all_events(collecting_writer)
        info = Info.from_event(events[0])
        assert len(info.asr) == 1
        assert len(info.tts) == 0

    async def test_tts_only_when_stt_disabled(self, mock_reader, collecting_writer):
        synth = _FakeSynthesizer()
        handler = MoonshineAsrHandler(
            reader=mock_reader,
            writer=collecting_writer,
            transcriber=None,
            tts_synthesizer=synth,
            tts_model_name="german",
        )
        await handler.handle_event(Event(type="describe", data={}))

        events = await _read_all_events(collecting_writer)
        info = Info.from_event(events[0])
        assert len(info.asr) == 0
        assert len(info.tts) == 1
        assert info.tts[0].voices[0].name == "juergen"

    async def test_both_asr_and_tts_when_both_enabled(
        self, mock_transcriber, mock_reader, collecting_writer
    ):
        synth = _FakeSynthesizer()
        handler = MoonshineAsrHandler(
            reader=mock_reader,
            writer=collecting_writer,
            transcriber=mock_transcriber,
            model_name="small",
            language="de",
            tts_synthesizer=synth,
            tts_model_name="german",
        )
        await handler.handle_event(Event(type="describe", data={}))

        events = await _read_all_events(collecting_writer)
        info = Info.from_event(events[0])
        assert len(info.asr) == 1
        assert len(info.tts) == 1

    async def test_neither_when_both_disabled(self, mock_reader, collecting_writer):
        handler = MoonshineAsrHandler(
            reader=mock_reader,
            writer=collecting_writer,
            transcriber=None,
            tts_synthesizer=None,
        )
        await handler.handle_event(Event(type="describe", data={}))

        events = await _read_all_events(collecting_writer)
        info = Info.from_event(events[0])
        assert len(info.asr) == 0
        assert len(info.tts) == 0


@pytest.mark.asyncio
class TestHandleSynthesize:
    async def test_tts_disabled_returns_error(self, mock_reader, collecting_writer):
        handler = MoonshineAsrHandler(
            reader=mock_reader, writer=collecting_writer, transcriber=None, tts_synthesizer=None
        )
        await handler.handle_event(Synthesize(text="Hallo").event())

        events = await _read_all_events(collecting_writer)
        assert any(WyomingError.is_type(e.type) for e in events)

    async def test_empty_text_sends_audio_start_and_stop_only(self, mock_reader, collecting_writer):
        synth = _FakeSynthesizer()
        handler = MoonshineAsrHandler(
            reader=mock_reader, writer=collecting_writer, transcriber=None, tts_synthesizer=synth
        )
        await handler.handle_event(Synthesize(text="   ").event())

        events = await _read_all_events(collecting_writer)
        assert [e.type for e in events] == ["audio-start", "audio-stop"]
        assert synth.requested_voices == []  # never even called into the model

    async def test_synthesize_sends_audio_start_chunks_and_stop(
        self, mock_reader, collecting_writer
    ):
        synth = _FakeSynthesizer(chunks=[[0.1, 0.2], [0.3, 0.4, 0.5]])
        handler = MoonshineAsrHandler(
            reader=mock_reader, writer=collecting_writer, transcriber=None, tts_synthesizer=synth
        )
        await handler.handle_event(Synthesize(text="Hallo Welt").event())

        events = await _read_all_events(collecting_writer)
        assert [e.type for e in events] == [
            "audio-start",
            "audio-chunk",
            "audio-chunk",
            "audio-stop",
        ]

        start = AudioStart.from_event(events[0])
        assert start.rate == 24000
        assert start.width == 2
        assert start.channels == 1

        chunk1 = AudioChunk.from_event(events[1])
        assert len(chunk1.audio) == 2 * 2  # 2 samples * 2 bytes
        chunk2 = AudioChunk.from_event(events[2])
        assert len(chunk2.audio) == 3 * 2

        AudioStop.from_event(events[3])  # does not raise

    async def test_synthesize_uses_default_voice_when_none_requested(
        self, mock_reader, collecting_writer
    ):
        synth = _FakeSynthesizer(default_voice="juergen")
        handler = MoonshineAsrHandler(
            reader=mock_reader, writer=collecting_writer, transcriber=None, tts_synthesizer=synth
        )
        await handler.handle_event(Synthesize(text="Hallo").event())
        assert synth.requested_voices == [None]

    async def test_synthesize_forwards_requested_voice(self, mock_reader, collecting_writer):
        synth = _FakeSynthesizer()
        handler = MoonshineAsrHandler(
            reader=mock_reader, writer=collecting_writer, transcriber=None, tts_synthesizer=synth
        )
        await handler.handle_event(
            Synthesize(text="Hallo", voice=SynthesizeVoice(name="alba")).event()
        )
        assert synth.requested_voices == ["alba"]

    async def test_synthesis_error_sends_wyoming_error_and_audio_stop(
        self, mock_reader, collecting_writer
    ):
        synth = _FakeSynthesizer()
        synth.raise_error = RuntimeError("model exploded")
        handler = MoonshineAsrHandler(
            reader=mock_reader, writer=collecting_writer, transcriber=None, tts_synthesizer=synth
        )
        await handler.handle_event(Synthesize(text="Hallo").event())

        events = await _read_all_events(collecting_writer)
        types = [e.type for e in events]
        assert "audio-stop" in types
        assert any(WyomingError.is_type(t) for t in types)

    async def test_no_performance_log_when_disabled(self, mock_reader, collecting_writer, caplog):
        synth = _FakeSynthesizer()
        handler = MoonshineAsrHandler(
            reader=mock_reader,
            writer=collecting_writer,
            transcriber=None,
            tts_synthesizer=synth,
            tts_log_performance=False,
        )
        with caplog.at_level("INFO"):
            await handler.handle_event(Synthesize(text="Hallo Welt").event())
        assert "TTS completed" not in caplog.text

    async def test_performance_log_never_contains_synthesized_text(
        self, mock_reader, collecting_writer, caplog
    ):
        synth = _FakeSynthesizer()
        handler = MoonshineAsrHandler(
            reader=mock_reader,
            writer=collecting_writer,
            transcriber=None,
            tts_synthesizer=synth,
            tts_log_performance=True,
        )
        secret_text = "Das geheime Passwort ist Schmetterling"
        with caplog.at_level("INFO"):
            await handler.handle_event(Synthesize(text=secret_text).event())

        perf_lines = [r.message for r in caplog.records if "TTS completed" in r.message]
        assert len(perf_lines) == 1
        assert secret_text not in perf_lines[0]
        assert "ttfa_generated=" in perf_lines[0]
        assert "ttfa_sent=" in perf_lines[0]
        assert "rtf=" in perf_lines[0]
