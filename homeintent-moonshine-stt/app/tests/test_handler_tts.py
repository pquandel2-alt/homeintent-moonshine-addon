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
from wyoming.info import Attribution, Info
from wyoming.tts import Synthesize, SynthesizeVoice

from app.handler import MoonshineAsrHandler


class _FakeSynthesizer:
    """Stands in for app.tts_session.PocketTtsSynthesizer."""

    engine_id = "pocket_tts"
    model_name = "german"
    program_name = "homeintent-pocket-tts"
    description = "HomeIntent Pocket TTS - German streaming TTS"
    attribution = Attribution(name="Kyutai", url="https://github.com/kyutai-labs/pocket-tts")

    def __init__(self, chunks: list[list[float]] | None = None, default_voice: str = "juergen"):
        self.sample_rate = 24000
        self.default_voice = default_voice
        self._chunks = chunks if chunks is not None else [[0.1, 0.2], [0.3, 0.4]]
        self.requested_voices: list[str | None] = []
        self.raise_error: BaseException | None = None
        # If set, raise_error is raised only after this many chunks have
        # already been yielded (0 = before any chunk, i.e. the previous
        # behavior) -- lets tests simulate a mid-stream failure.
        self.raise_after_chunks = 0

    async def synthesize_stream(self, text: str, voice: str | None = None, stats=None):
        self.requested_voices.append(voice)
        if self.raise_error is not None and self.raise_after_chunks == 0:
            raise self.raise_error
        for i, chunk in enumerate(self._chunks, start=1):
            yield np.array(chunk, dtype=np.float32)
            if self.raise_error is not None and i == self.raise_after_chunks:
                raise self.raise_error


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

    async def test_kokoro_engine_describe_uses_kokoro_metadata_not_kyutai(
        self, mock_reader, collecting_writer
    ):
        """Wyoming discovery must be fully engine-agnostic: describing a
        Kokoro-backed handler must never show Pocket TTS's own program
        name/attribution -- see app/tts_engine.py's TtsSynthesizer
        protocol and app/handler.py's generalized _handle_describe()."""

        class _FakeKokoroSynthesizer:
            engine_id = "kokoro_onnx"
            sample_rate = 24000
            default_voice = "martin"
            model_name = "kokoro-82m-german-martin"
            program_name = "homeintent-kokoro-onnx"
            description = "German Kokoro-82M ONNX text-to-speech (voice: Martin)"
            attribution = Attribution(
                name="Godelaune (German Martin fine-tune) / hexgrad (Kokoro-82M)",
                url="https://huggingface.co/Godelaune/Kokoro-82M-ONNX-German-Martin",
            )

            async def synthesize_stream(self, text, voice=None, stats=None):
                yield None  # pragma: no cover - never called in this test

        handler = MoonshineAsrHandler(
            reader=mock_reader,
            writer=collecting_writer,
            transcriber=None,
            tts_synthesizer=_FakeKokoroSynthesizer(),
        )
        await handler.handle_event(Event(type="describe", data={}))

        events = await _read_all_events(collecting_writer)
        info = Info.from_event(events[0])
        assert len(info.tts) == 1
        tts_program = info.tts[0]
        assert tts_program.name == "homeintent-kokoro-onnx"
        assert tts_program.voices[0].name == "martin"
        assert "kyutai" not in tts_program.attribution.name.lower()
        assert "kyutai" not in tts_program.attribution.url.lower()
        assert tts_program.supports_synthesize_streaming is True

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

    async def test_performance_log_reports_separate_timing_breakdown(
        self, mock_reader, collecting_writer, caplog
    ):
        """Item 9-11: model compute, lock-wait, and Wyoming-send time must
        each appear as their own field, distinct from the overall wall
        time/RTF -- not folded into one opaque "synthesis" number."""
        synth = _FakeSynthesizer(chunks=[[0.1, 0.2], [0.3, 0.4]])
        handler = MoonshineAsrHandler(
            reader=mock_reader,
            writer=collecting_writer,
            transcriber=None,
            tts_synthesizer=synth,
            tts_log_performance=True,
        )
        with caplog.at_level("INFO"):
            await handler.handle_event(Synthesize(text="Hallo Welt").event())

        perf_lines = [r.message for r in caplog.records if "TTS completed" in r.message]
        assert len(perf_lines) == 1
        line = perf_lines[0]
        for field in (
            "engine=",
            "lock_wait=",
            "model_compute=",
            "wyoming_send=",
            "wall=",
            "audio=",
            "model_rtf=",
            "wall_rtf=",
        ):
            assert field in line, f"missing {field!r} in: {line}"

    async def test_performance_log_reports_engine_id(self, mock_reader, collecting_writer, caplog):
        """Item 18: the engine that actually served the request must be
        identifiable in the log, so Pocket and Kokoro can be compared."""
        synth = _FakeSynthesizer()
        handler = MoonshineAsrHandler(
            reader=mock_reader,
            writer=collecting_writer,
            transcriber=None,
            tts_synthesizer=synth,
            tts_log_performance=True,
        )
        with caplog.at_level("INFO"):
            await handler.handle_event(Synthesize(text="Hallo").event())

        perf_lines = [r.message for r in caplog.records if "TTS completed" in r.message]
        assert "engine=pocket_tts" in perf_lines[0]


@pytest.mark.asyncio
class TestSynthesizeAudioStartStateMachine:
    """A mid-stream Pocket TTS failure must never cause a second audio-start
    within the same synthesize request -- see app/handler.py's
    _stream_synthesis_chunks()/_handle_synthesize() docstrings. Exactly one
    audio-start per request, whatever else happens."""

    async def test_error_before_first_chunk_sends_single_audio_start_then_stop_and_error(
        self, mock_reader, collecting_writer
    ):
        synth = _FakeSynthesizer()
        synth.raise_error = RuntimeError("boom before any audio")
        handler = MoonshineAsrHandler(
            reader=mock_reader, writer=collecting_writer, transcriber=None, tts_synthesizer=synth
        )
        await handler.handle_event(Synthesize(text="Hallo").event())

        events = await _read_all_events(collecting_writer)
        types = [e.type for e in events]
        assert types.count("audio-start") == 1
        assert types == ["audio-start", "audio-stop", "error"]

    async def test_error_after_audio_start_does_not_resend_audio_start(
        self, mock_reader, collecting_writer
    ):
        """Failure occurs after the first chunk (and therefore after
        audio-start was already sent) -- the except branch must not send a
        second one."""
        synth = _FakeSynthesizer(chunks=[[0.1, 0.2]])
        synth.raise_error = RuntimeError("boom after first chunk")
        synth.raise_after_chunks = 1
        handler = MoonshineAsrHandler(
            reader=mock_reader, writer=collecting_writer, transcriber=None, tts_synthesizer=synth
        )
        await handler.handle_event(Synthesize(text="Hallo").event())

        events = await _read_all_events(collecting_writer)
        types = [e.type for e in events]
        assert types.count("audio-start") == 1
        assert types == ["audio-start", "audio-chunk", "audio-stop", "error"]

    async def test_error_after_multiple_chunks_still_sends_only_one_audio_start(
        self, mock_reader, collecting_writer
    ):
        synth = _FakeSynthesizer(chunks=[[0.1, 0.2], [0.3, 0.4], [0.5, 0.6]])
        synth.raise_error = RuntimeError("boom after third chunk")
        synth.raise_after_chunks = 3
        handler = MoonshineAsrHandler(
            reader=mock_reader, writer=collecting_writer, transcriber=None, tts_synthesizer=synth
        )
        await handler.handle_event(Synthesize(text="Hallo Welt").event())

        events = await _read_all_events(collecting_writer)
        types = [e.type for e in events]
        assert types.count("audio-start") == 1
        assert types == [
            "audio-start",
            "audio-chunk",
            "audio-chunk",
            "audio-chunk",
            "audio-stop",
            "error",
        ]

    async def test_normal_successful_stream_sends_exactly_one_audio_start(
        self, mock_reader, collecting_writer
    ):
        synth = _FakeSynthesizer(chunks=[[0.1, 0.2], [0.3, 0.4]])
        handler = MoonshineAsrHandler(
            reader=mock_reader, writer=collecting_writer, transcriber=None, tts_synthesizer=synth
        )
        await handler.handle_event(Synthesize(text="Hallo Welt").event())

        events = await _read_all_events(collecting_writer)
        types = [e.type for e in events]
        assert types.count("audio-start") == 1
        assert "error" not in types

    async def test_empty_text_sends_exactly_one_audio_start(self, mock_reader, collecting_writer):
        synth = _FakeSynthesizer()
        handler = MoonshineAsrHandler(
            reader=mock_reader, writer=collecting_writer, transcriber=None, tts_synthesizer=synth
        )
        await handler.handle_event(Synthesize(text="").event())

        events = await _read_all_events(collecting_writer)
        types = [e.type for e in events]
        assert types.count("audio-start") == 1
        assert types == ["audio-start", "audio-stop"]

    async def test_client_disconnect_mid_stream_sends_at_most_one_audio_start(
        self, mock_reader, collecting_writer
    ):
        """Simulates a client disconnect mid-stream: the write_event() call
        for a later chunk fails, propagating out of _stream_synthesis_chunks
        via its finally/aclose() path -- must not double-send audio-start
        either, matching the mid-stream-failure cases above."""
        synth = _FakeSynthesizer(chunks=[[0.1, 0.2], [0.3, 0.4], [0.5, 0.6]])
        handler = MoonshineAsrHandler(
            reader=mock_reader, writer=collecting_writer, transcriber=None, tts_synthesizer=synth
        )

        original_write_event = handler.write_event
        call_count = 0

        async def flaky_write_event(event):
            nonlocal call_count
            call_count += 1
            if call_count == 3:  # audio-start, then 1st audio-chunk, then fail on the 2nd
                raise ConnectionResetError("client disconnected")
            await original_write_event(event)

        handler.write_event = flaky_write_event  # type: ignore[method-assign]

        await handler.handle_event(Synthesize(text="Hallo Welt").event())

        events = await _read_all_events(collecting_writer)
        types = [e.type for e in events]
        assert types.count("audio-start") <= 1
