"""Pytest fixtures and mocks for Moonshine tests."""

import asyncio
from unittest.mock import AsyncMock, MagicMock, Mock

import pytest


@pytest.fixture
def event_loop():
    """Create event loop for async tests."""
    loop = asyncio.new_event_loop()
    yield loop
    loop.close()


@pytest.fixture
def mock_transcriber():
    """Mock Moonshine Transcriber for testing."""
    mock = MagicMock()
    mock.add_listener = Mock()
    mock.start = Mock()
    mock.stop = Mock()
    mock.add_audio = Mock()
    mock.language = Mock(return_value=mock)
    mock.model_arch = Mock(return_value=mock)
    mock.model_dir = Mock(return_value=mock)
    mock.load = Mock()
    return mock


@pytest.fixture
def mock_reader():
    """Mock Wyoming protocol reader."""
    reader = AsyncMock()
    return reader


@pytest.fixture
def mock_writer():
    """Mock Wyoming protocol writer."""
    writer = AsyncMock()
    writer.get_extra_info = Mock(return_value=("127.0.0.1", 54321))
    writer.close = Mock()
    writer.wait_closed = AsyncMock()
    return writer


@pytest.fixture
def transcriber_factory(mock_transcriber):
    """Factory that returns mock transcriber."""
    def factory():
        return mock_transcriber
    return factory
