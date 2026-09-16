"""Tests for Moonshine streaming session management."""

import asyncio
from unittest.mock import MagicMock, patch

import numpy as np
import pytest

from app.streaming import MoonshineListener, MoonshineStreamingSession


class TestMoonshineListener:
    """Tests for event listener."""

    @pytest.mark.asyncio
    async def test_line_completed_sets_event(self):
        """Test that on_line_completed sets the done event."""
        loop = asyncio.get_event_loop()
        listener = MoonshineListener(loop)

        # Event should not be set initially
        assert not listener._done.is_set()

        # Trigger completion
        listener.on_line_completed("test transcript")

        # Give the thread-safe call time to propagate
        await asyncio.sleep(0.1)

        # Event should now be set
        assert listener._done.is_set()
        assert listener.final_text == "test transcript"

    @pytest.mark.asyncio
    async def test_partial_text_tracking(self):
        """Test that partial results are tracked."""
        loop = asyncio.get_event_loop()
        listener = MoonshineListener(loop)

        listener.on_line_text_changed("Hallo")
        assert listener._partial_text == "Hallo"

        listener.on_line_text_changed("Hallo Welt")
        assert listener._partial_text == "Hallo Welt"

    def test_listener_reset(self):
        """Test that listener can be reset for new utterance."""
        loop = asyncio.get_event_loop()
        listener = MoonshineListener(loop)

        listener.on_line_completed("first")
        assert listener.final_text == "first"

        listener.reset()
        assert listener.final_text == ""
        assert not listener._done.is_set()


class TestMoonshineStreamingSession:
    """Tests for streaming session."""

    def test_session_initialization(self):
        """Test session creation with transcriber."""
        mock_transcriber = MagicMock()
        session = MoonshineStreamingSession(mock_transcriber)

        # Should register listener
        mock_transcriber.add_listener.assert_called_once()
        assert session._transcriber is mock_transcriber

    def test_session_start(self):
        """Test session start calls transcriber.start()."""
        mock_transcriber = MagicMock()
        session = MoonshineStreamingSession(mock_transcriber)

        session.start()

        mock_transcriber.start.assert_called_once()
        assert session._started is True

    def test_add_audio_not_started(self):
        """Test that add_audio before start() is ignored."""
        mock_transcriber = MagicMock()
        session = MoonshineStreamingSession(mock_transcriber)

        # Don't call start()
        float32_audio = np.array([0.1, 0.2, 0.3], dtype=np.float32)
        session.add_audio(float32_audio)

        # add_audio should not have been called
        mock_transcriber.add_audio.assert_not_called()

    def test_add_audio_empty(self):
        """Test that empty audio is rejected."""
        mock_transcriber = MagicMock()
        session = MoonshineStreamingSession(mock_transcriber)

        session.start()
        session.add_audio(np.array([], dtype=np.float32))

        # add_audio should not have been called
        mock_transcriber.add_audio.assert_not_called()

    def test_add_audio_success(self):
        """Test that audio is fed to transcriber."""
        mock_transcriber = MagicMock()
        session = MoonshineStreamingSession(mock_transcriber)

        session.start()

        float32_audio = np.array([0.1, 0.2, 0.3], dtype=np.float32)
        session.add_audio(float32_audio)

        # Verify add_audio was called with correct params
        mock_transcriber.add_audio.assert_called_once()
        call_args = mock_transcriber.add_audio.call_args
        np.testing.assert_array_equal(call_args[0][0], float32_audio)
        assert call_args[0][1] == 16000

    @pytest.mark.asyncio
    async def test_finalize_before_start(self):
        """Test finalize before start returns empty."""
        mock_transcriber = MagicMock()
        session = MoonshineStreamingSession(mock_transcriber)

        result = await session.finalize()

        assert result == ""
        mock_transcriber.stop.assert_not_called()

    @pytest.mark.asyncio
    async def test_finalize_calls_stop(self):
        """Test that finalize calls transcriber.stop()."""
        mock_transcriber = MagicMock()
        session = MoonshineStreamingSession(mock_transcriber)

        # Get listener and set it to complete immediately
        listener = session._listener
        listener.final_text = "test result"
        loop = asyncio.get_event_loop()
        loop.call_soon_threadsafe(listener._done.set)

        session.start()

        result = await session.finalize()

        mock_transcriber.stop.assert_called_once()
        assert result == "test result"
        assert session._started is False
