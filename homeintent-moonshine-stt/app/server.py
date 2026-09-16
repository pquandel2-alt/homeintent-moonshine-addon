"""Wyoming STT server for Moonshine."""

import asyncio
import logging
from typing import Callable

from wyoming.server import AsyncServer

from app.handler import MoonshineAsrHandler

_LOGGER = logging.getLogger(__name__)


class MoonshineServer(AsyncServer):
    """Wyoming STT server that uses MoonshineAsrHandler."""

    def __init__(
        self,
        host: str,
        port: int,
        transcriber_factory: Callable,
        model_name: str,
        language: str = "de",
    ):
        """Initialize server.

        Args:
            host: Bind hostname/IP
            port: Bind port
            transcriber_factory: Callable that returns a loaded Moonshine Transcriber
            model_name: Model to use ("tiny" or "small")
            language: Language code ("de")
        """
        super().__init__()
        self.host = host
        self.port = port
        self._transcriber_factory = transcriber_factory
        self._model_name = model_name
        self._language = language

    async def run(self) -> None:
        """Start the server and handle connections."""
        _LOGGER.info(f"Starting Moonshine server on {self.host}:{self.port}")

        # Start listening
        server = await asyncio.start_server(
            self._handle_connection,
            self.host,
            self.port,
        )

        async with server:
            _LOGGER.info(f"Moonshine server listening on {self.host}:{self.port}")
            await server.serve_forever()

    async def _handle_connection(self, reader, writer) -> None:
        """Handle a new Wyoming client connection."""
        addr = writer.get_extra_info("peername")
        _LOGGER.debug(f"New connection from {addr}")

        try:
            handler = MoonshineAsrHandler(
                reader=reader,
                writer=writer,
                transcriber_factory=self._transcriber_factory,
                model_name=self._model_name,
                language=self._language,
            )

            await handler.process_events()

        except Exception as e:
            _LOGGER.error(f"Error handling connection from {addr}: {e}", exc_info=True)
        finally:
            writer.close()
            await writer.wait_closed()
            _LOGGER.debug(f"Connection closed from {addr}")
