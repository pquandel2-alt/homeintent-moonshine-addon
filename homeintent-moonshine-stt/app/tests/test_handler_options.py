"""Tests for handler-level v0.1.2 options: transcript/performance logging,
debug audio persistence, and session lifecycle hardening.
"""

import logging
from unittest.mock import MagicMock

import pytest
from moonshine_voice import LineCompleted, TranscriptLine
from wyoming.audio import AudioChunk, AudioStart
from wyoming.event import Event

from app.handler import MoonshineAsrHandler


def _line(text: str) -> TranscriptLine:
    return TranscriptLine(
        text=text, start_time=0.0, duration=1.0, line_id=1, is_complete=True, is_updated=True
    )


def _make_stop_completing(mock_stream: MagicMock, text: str) -> None:
    def fake_stop() -> None:
        for call in mock_stream.add_listener.call_args_list:
            listener = call.args[0]
            listener.on_line_completed(LineCompleted(line=_line(text), stream_handle=1))

    mock_stream.stop.side_effect = fake_stop


async def _run_one_utterance(
    handler: MoonshineAsrHandler, mock_stream: MagicMock, chunk: bytes = b"\x00\x00" * 800
) -> None:
    await handler.handle_event(AudioStart(rate=16000, width=2, channels=1).event())
    await handler.handle_event(AudioChunk(rate=16000, width=2, channels=1, audio=chunk).event())
    await handler.handle_event(Event(type="audio-stop", data={}))


class TestLogTranscripts:
    @pytest.mark.asyncio
    async def test_log_transcripts_true_logs_full_text(
        self, mock_transcriber, mock_stream, mock_reader, mock_writer, caplog
    ):
        _make_stop_completing(mock_stream, "schalte das licht im wohnzimmer ein")
        handler = MoonshineAsrHandler(
            reader=mock_reader,
            writer=mock_writer,
            transcriber=mock_transcriber,
            model_name="small",
            log_transcripts=True,
        )
        with caplog.at_level(logging.INFO, logger="app.handler"):
            await _run_one_utterance(handler, mock_stream)

        assert "schalte das licht im wohnzimmer ein" in caplog.text

    @pytest.mark.asyncio
    async def test_log_transcripts_false_never_logs_text(
        self, mock_transcriber, mock_stream, mock_reader, mock_writer, caplog
    ):
        _make_stop_completing(mock_stream, "geheimer satz")
        handler = MoonshineAsrHandler(
            reader=mock_reader,
            writer=mock_writer,
            transcriber=mock_transcriber,
            model_name="small",
            log_transcripts=False,
        )
        with caplog.at_level(logging.INFO, logger="app.handler"):
            await _run_one_utterance(handler, mock_stream)

        assert "geheimer satz" not in caplog.text
        assert "chars)" in caplog.text


class TestLogPerformance:
    @pytest.mark.asyncio
    async def test_log_performance_true_logs_rtf_line(
        self, mock_transcriber, mock_stream, mock_reader, mock_writer, caplog
    ):
        _make_stop_completing(mock_stream, "Test")
        handler = MoonshineAsrHandler(
            reader=mock_reader,
            writer=mock_writer,
            transcriber=mock_transcriber,
            model_name="small",
            log_performance=True,
        )
        with caplog.at_level(logging.INFO, logger="app.handler"):
            await _run_one_utterance(handler, mock_stream)

        assert "STT completed" in caplog.text
        assert "rtf=" in caplog.text

    @pytest.mark.asyncio
    async def test_log_performance_line_never_contains_transcript_text(
        self, mock_transcriber, mock_stream, mock_reader, mock_writer, caplog
    ):
        _make_stop_completing(mock_stream, "vertraulicher inhalt")
        handler = MoonshineAsrHandler(
            reader=mock_reader,
            writer=mock_writer,
            transcriber=mock_transcriber,
            model_name="small",
            log_transcripts=False,
            log_performance=True,
        )
        with caplog.at_level(logging.INFO, logger="app.handler"):
            await _run_one_utterance(handler, mock_stream)

        perf_lines = [r.message for r in caplog.records if "STT completed" in r.message]
        assert perf_lines
        assert "vertraulicher inhalt" not in perf_lines[0]

    @pytest.mark.asyncio
    async def test_log_performance_false_suppresses_rtf_line(
        self, mock_transcriber, mock_stream, mock_reader, mock_writer, caplog
    ):
        _make_stop_completing(mock_stream, "Test")
        handler = MoonshineAsrHandler(
            reader=mock_reader,
            writer=mock_writer,
            transcriber=mock_transcriber,
            model_name="small",
            log_performance=False,
        )
        with caplog.at_level(logging.INFO, logger="app.handler"):
            await _run_one_utterance(handler, mock_stream)

        assert "STT completed" not in caplog.text


class TestSttPerformanceLogNamesEngine:
    """v0.6.1: the STT performance line must name the active engine
    (`engine=<engine_id>`), consistently across every engine via the shared
    SttEngine/SttSession abstraction -- not just for the Moonshine
    transcriber-path handler construction the other tests above use."""

    @pytest.mark.asyncio
    async def test_moonshine_transcriber_path_logs_engine_moonshine(
        self, mock_transcriber, mock_stream, mock_reader, mock_writer, caplog
    ):
        _make_stop_completing(mock_stream, "Test")
        handler = MoonshineAsrHandler(
            reader=mock_reader,
            writer=mock_writer,
            transcriber=mock_transcriber,
            model_name="small",
            log_performance=True,
        )
        with caplog.at_level(logging.INFO, logger="app.handler"):
            await _run_one_utterance(handler, mock_stream)

        perf_lines = [r.message for r in caplog.records if "STT completed" in r.message]
        assert perf_lines
        assert "engine=moonshine" in perf_lines[0]

    @pytest.mark.asyncio
    async def test_speechcatcher_engine_path_logs_engine_speechcatcher_m(
        self, mock_reader, mock_writer, caplog
    ):
        from app.speechcatcher_engine import SpeechcatcherSttEngine

        class _FakeSpeech2TextStreaming:
            def reset(self) -> None:
                pass

            def __call__(self, speech, is_final=False, always_assemble_hyps=True):
                if is_final:
                    return [("schalte das licht ein", [], [])]
                return []

        stt_engine = SpeechcatcherSttEngine(_FakeSpeech2TextStreaming(), "speechcatcher_m")
        handler = MoonshineAsrHandler(
            reader=mock_reader,
            writer=mock_writer,
            stt_engine=stt_engine,
            model_name=stt_engine.model_display_name,
            log_performance=True,
        )
        with caplog.at_level(logging.INFO, logger="app.handler"):
            await _run_one_utterance(handler, mock_stream=MagicMock())

        perf_lines = [r.message for r in caplog.records if "STT completed" in r.message]
        assert perf_lines
        line = perf_lines[0]
        assert "engine=speechcatcher_m" in line
        for field in (
            "model=",
            "audio=",
            "inference=",
            "finalize=",
            "rtf=",
            "chunks=",
            "add_audio_total=",
            "add_audio_max=",
            "avg_chunk=",
        ):
            assert field in line, f"missing {field!r} in {line!r}"


class TestDebugAudio:
    @pytest.mark.asyncio
    async def test_save_debug_audio_true_writes_wav_and_json(
        self, mock_transcriber, mock_stream, mock_reader, mock_writer, tmp_path
    ):
        _make_stop_completing(mock_stream, "Debug Test")
        handler = MoonshineAsrHandler(
            reader=mock_reader,
            writer=mock_writer,
            transcriber=mock_transcriber,
            model_name="small",
            save_debug_audio_enabled=True,
            debug_audio_dir=tmp_path,
        )
        await _run_one_utterance(handler, mock_stream)

        assert len(list(tmp_path.glob("*.wav"))) == 1
        assert len(list(tmp_path.glob("*.json"))) == 1

    @pytest.mark.asyncio
    async def test_save_debug_audio_false_writes_nothing(
        self, mock_transcriber, mock_stream, mock_reader, mock_writer, tmp_path
    ):
        _make_stop_completing(mock_stream, "Kein Debug")
        handler = MoonshineAsrHandler(
            reader=mock_reader,
            writer=mock_writer,
            transcriber=mock_transcriber,
            model_name="small",
            save_debug_audio_enabled=False,
            debug_audio_dir=tmp_path,
        )
        await _run_one_utterance(handler, mock_stream)

        assert not tmp_path.exists() or not list(tmp_path.glob("*.wav"))

    @pytest.mark.asyncio
    async def test_debug_audio_retention_enforced(
        self, mock_transcriber, mock_stream, mock_reader, mock_writer, tmp_path
    ):
        _make_stop_completing(mock_stream, "x")
        handler = MoonshineAsrHandler(
            reader=mock_reader,
            writer=mock_writer,
            transcriber=mock_transcriber,
            model_name="small",
            save_debug_audio_enabled=True,
            debug_audio_max_files=2,
            debug_audio_dir=tmp_path,
        )
        for _ in range(4):
            await _run_one_utterance(handler, mock_stream)

        assert len(list(tmp_path.glob("*.wav"))) == 2


class TestSessionLifecycleHardening:
    @pytest.mark.asyncio
    async def test_duplicate_audio_start_closes_previous_session(
        self, mock_transcriber, mock_stream, mock_reader, mock_writer
    ):
        handler = MoonshineAsrHandler(
            reader=mock_reader, writer=mock_writer, transcriber=mock_transcriber, model_name="small"
        )
        await handler.handle_event(AudioStart(rate=16000, width=2, channels=1).event())
        first_session = handler._session
        await handler.handle_event(AudioStart(rate=16000, width=2, channels=1).event())

        assert handler._session is not None
        assert handler._session is not first_session
        mock_stream.close.assert_called()

    @pytest.mark.asyncio
    async def test_transcribe_event_while_active_closes_session(
        self, mock_transcriber, mock_stream, mock_reader, mock_writer
    ):
        handler = MoonshineAsrHandler(
            reader=mock_reader, writer=mock_writer, transcriber=mock_transcriber, model_name="small"
        )
        await handler.handle_event(AudioStart(rate=16000, width=2, channels=1).event())
        await handler.handle_event(Event(type="transcribe", data={}))

        assert handler._session is None
        mock_stream.close.assert_called_once()

    @pytest.mark.asyncio
    async def test_disconnect_closes_active_session(
        self, mock_transcriber, mock_stream, mock_reader, mock_writer
    ):
        handler = MoonshineAsrHandler(
            reader=mock_reader, writer=mock_writer, transcriber=mock_transcriber, model_name="small"
        )
        await handler.handle_event(AudioStart(rate=16000, width=2, channels=1).event())

        await handler.disconnect()

        assert handler._session is None
        mock_stream.close.assert_called_once()

    @pytest.mark.asyncio
    async def test_disconnect_without_session_does_not_raise(
        self, mock_transcriber, mock_reader, mock_writer
    ):
        handler = MoonshineAsrHandler(
            reader=mock_reader, writer=mock_writer, transcriber=mock_transcriber, model_name="small"
        )
        await handler.disconnect()
        assert handler._session is None

    @pytest.mark.asyncio
    async def test_exception_during_add_audio_closes_session_and_returns_false(
        self, mock_transcriber, mock_stream, mock_reader, mock_writer
    ):
        mock_stream.add_audio.side_effect = RuntimeError("native failure")
        handler = MoonshineAsrHandler(
            reader=mock_reader, writer=mock_writer, transcriber=mock_transcriber, model_name="small"
        )
        await handler.handle_event(AudioStart(rate=16000, width=2, channels=1).event())

        result = await handler.handle_event(
            AudioChunk(rate=16000, width=2, channels=1, audio=b"\x00\x00" * 100).event()
        )

        assert result is False
        assert handler._session is None
        mock_stream.close.assert_called_once()

    @pytest.mark.asyncio
    async def test_exception_during_finalize_still_closes_session(
        self, mock_transcriber, mock_stream, mock_reader, mock_writer
    ):
        mock_stream.stop.side_effect = RuntimeError("finalize blew up")
        handler = MoonshineAsrHandler(
            reader=mock_reader, writer=mock_writer, transcriber=mock_transcriber, model_name="small"
        )
        await handler.handle_event(AudioStart(rate=16000, width=2, channels=1).event())

        result = await handler.handle_event(Event(type="audio-stop", data={}))

        assert result is False
        assert handler._session is None
        mock_stream.close.assert_called_once()

    @pytest.mark.asyncio
    async def test_empty_audio_chunk_does_not_crash(
        self, mock_transcriber, mock_stream, mock_reader, mock_writer
    ):
        handler = MoonshineAsrHandler(
            reader=mock_reader, writer=mock_writer, transcriber=mock_transcriber, model_name="small"
        )
        await handler.handle_event(AudioStart(rate=16000, width=2, channels=1).event())

        result = await handler.handle_event(
            AudioChunk(rate=16000, width=2, channels=1, audio=b"").event()
        )

        assert result is True
        assert handler._session is not None

    @pytest.mark.asyncio
    async def test_odd_length_chunk_bytes_closes_session_and_returns_false(
        self, mock_transcriber, mock_stream, mock_reader, mock_writer
    ):
        handler = MoonshineAsrHandler(
            reader=mock_reader, writer=mock_writer, transcriber=mock_transcriber, model_name="small"
        )
        await handler.handle_event(AudioStart(rate=16000, width=2, channels=1).event())

        result = await handler.handle_event(
            AudioChunk(rate=16000, width=2, channels=1, audio=b"\x00").event()
        )

        assert result is False
        assert handler._session is None
