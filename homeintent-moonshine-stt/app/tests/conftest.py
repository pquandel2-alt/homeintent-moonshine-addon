"""Pytest fixtures and mocks for Moonshine tests."""

from unittest.mock import AsyncMock, MagicMock

import pytest


@pytest.fixture
def mock_stream():
    """Mock moonshine_voice.Stream (as returned by Transcriber.create_stream())."""
    stream = MagicMock()
    stream.start = MagicMock()
    stream.stop = MagicMock()
    stream.add_audio = MagicMock()
    stream.add_listener = MagicMock()
    stream.close = MagicMock()
    return stream


@pytest.fixture
def mock_transcriber(mock_stream):
    """Mock Moonshine Transcriber for testing.

    create_stream() always returns the same mock_stream so tests can assert
    on it directly.
    """
    transcriber = MagicMock()
    transcriber.create_stream = MagicMock(return_value=mock_stream)
    return transcriber


@pytest.fixture
def mock_reader():
    """Mock asyncio.StreamReader matching wyoming.event.async_read_event's usage."""
    reader = MagicMock()
    reader.readline = AsyncMock(return_value=b"")
    reader.readexactly = AsyncMock(return_value=b"")
    return reader


@pytest.fixture
def mock_writer():
    """Mock asyncio.StreamWriter matching wyoming.event.async_write_event's usage.

    writelines()/write() are synchronous calls on a real StreamWriter; only
    drain() is awaited. Mixing this up (e.g. a blanket AsyncMock()) produces
    "coroutine was never awaited" warnings because the code never awaits
    writelines()/write().
    """
    writer = MagicMock()
    writer.writelines = MagicMock()
    writer.write = MagicMock()
    writer.drain = AsyncMock()
    writer.close = MagicMock()
    writer.wait_closed = AsyncMock()
    writer.get_extra_info = MagicMock(return_value=("127.0.0.1", 54321))
    writer.is_closing = MagicMock(return_value=False)
    return writer
