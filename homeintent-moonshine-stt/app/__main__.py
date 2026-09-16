"""Main entry point for HomeIntent Moonshine STT."""

import argparse
import asyncio
import json
import logging
import os
import sys
from functools import partial
from pathlib import Path

from app.models import load_transcriber
from app.server import MoonshineServer

# Setup logging
logging.basicConfig(
    format="%(asctime)s %(levelname)s [%(name)s] %(message)s",
    level=logging.INFO,
    handlers=[logging.StreamHandler(sys.stdout)],
)
_LOGGER = logging.getLogger(__name__)


def main() -> int:
    """Main entry point."""
    parser = argparse.ArgumentParser(description="HomeIntent Moonshine STT Server")

    parser.add_argument(
        "--port",
        type=int,
        default=10300,
        help="Port to listen on (default: 10300)",
    )
    parser.add_argument(
        "--host",
        default="0.0.0.0",
        help="Host to bind to (default: 0.0.0.0)",
    )
    parser.add_argument(
        "--model",
        choices=["tiny", "small"],
        default="small",
        help="Model size (default: small)",
    )
    parser.add_argument(
        "--language",
        default="de",
        help="Language code (default: de for German)",
    )
    parser.add_argument(
        "--debug",
        action="store_true",
        help="Enable debug logging",
    )
    parser.add_argument(
        "--config",
        type=Path,
        help="Path to JSON config file",
    )

    args = parser.parse_args()

    # Load from config file if provided
    if args.config and args.config.exists():
        try:
            with open(args.config) as f:
                config = json.load(f)
                args.model = config.get("model", args.model)
                args.language = config.get("language", args.language)
                args.debug = config.get("debug", args.debug) or args.debug
        except Exception as e:
            _LOGGER.error(f"Failed to load config: {e}")
            return 1

    # Set log level
    if args.debug:
        logging.getLogger().setLevel(logging.DEBUG)
        _LOGGER.info("Debug logging enabled")

    _LOGGER.info(f"HomeIntent Moonshine STT v0.1.0")
    _LOGGER.info(f"Model: {args.model}")
    _LOGGER.info(f"Language: {args.language}")

    # Load model once
    try:
        _LOGGER.info("Loading Moonshine model...")
        transcriber = load_transcriber(model=args.model, language=args.language)
        _LOGGER.info("Model loaded successfully")
    except Exception as e:
        _LOGGER.error(f"Failed to load model: {e}")
        return 1

    # Create server with factory that returns the same transcriber
    # (shared across connections, but each session gets its own)
    transcriber_factory = partial(load_transcriber, model=args.model, language=args.language)

    server = MoonshineServer(
        host=args.host,
        port=args.port,
        transcriber_factory=transcriber_factory,
        model_name=args.model,
        language=args.language,
    )

    try:
        asyncio.run(server.run())
    except KeyboardInterrupt:
        _LOGGER.info("Interrupted")
        return 0
    except Exception as e:
        _LOGGER.error(f"Server error: {e}", exc_info=True)
        return 1

    return 0


if __name__ == "__main__":
    sys.exit(main())
