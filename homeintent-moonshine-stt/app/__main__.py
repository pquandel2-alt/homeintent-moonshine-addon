"""Main entry point for HomeIntent Moonshine STT."""

import argparse
import asyncio
import json
import logging
import sys
from functools import partial
from pathlib import Path

from wyoming.server import AsyncTcpServer

from app.debug_audio import DEFAULT_DEBUG_AUDIO_DIR
from app.ha_vocabulary import fetch_ha_vocabulary
from app.handler import MoonshineAsrHandler
from app.keyterms import merge_keyterms, parse_extra_keyterms
from app.models import (
    DEFAULT_DECODE_INCOMPLETE_LINES,
    DEFAULT_KEYTERM_BOOST,
    DEFAULT_MODEL_CACHE_DIR,
    DEFAULT_TRANSCRIPTION_INTERVAL,
    DEFAULT_VAD_THRESHOLD,
    load_transcriber,
)
from app.validation import (
    validate_debug_audio_max_files,
    validate_ha_vocabulary_refresh_minutes,
    validate_keyterm_boost,
    validate_transcription_interval,
    validate_vad_threshold,
)

# Setup logging
logging.basicConfig(
    format="%(asctime)s %(levelname)s [%(name)s] %(message)s",
    level=logging.INFO,
    handlers=[logging.StreamHandler(sys.stdout)],
)
_LOGGER = logging.getLogger(__name__)

VERSION = "0.1.2"


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="HomeIntent Moonshine STT Server")

    parser.add_argument("--port", type=int, default=10300, help="Port to listen on")
    parser.add_argument("--host", default="0.0.0.0", help="Host to bind to")
    parser.add_argument("--model", choices=["tiny", "small"], default="small")
    parser.add_argument("--language", default="de")
    parser.add_argument("--debug", action="store_true", help="Enable debug logging")
    parser.add_argument("--config", type=Path, help="Path to JSON config file")

    parser.add_argument(
        "--log-transcripts",
        action="store_true",
        default=False,
        help="Log recognized text at INFO (off by default; privacy-sensitive)",
    )
    parser.add_argument(
        "--log-performance",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Log a compact per-utterance timing/RTF line",
    )
    parser.add_argument("--extra-keyterms", default="", help="Comma-separated manual keyterms")
    parser.add_argument("--keyterm-boost", type=float, default=DEFAULT_KEYTERM_BOOST)
    parser.add_argument(
        "--use-ha-vocabulary",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Read Areas/Devices/Entities from Home Assistant as keyterms (read-only)",
    )
    parser.add_argument("--ha-vocabulary-refresh-minutes", type=int, default=30)
    parser.add_argument(
        "--transcription-interval", type=float, default=DEFAULT_TRANSCRIPTION_INTERVAL
    )
    parser.add_argument("--vad-threshold", type=float, default=DEFAULT_VAD_THRESHOLD)
    parser.add_argument(
        "--decode-incomplete-lines",
        action=argparse.BooleanOptionalAction,
        default=DEFAULT_DECODE_INCOMPLETE_LINES,
    )
    parser.add_argument("--save-debug-audio", action="store_true", default=False)
    parser.add_argument("--debug-audio-max-files", type=int, default=100)

    return parser


def _load_json_config_overrides(args: argparse.Namespace) -> bool:
    """Merge a JSON --config file's values into args. Returns False on failure."""
    if not args.config or not args.config.exists():
        return True
    try:
        with open(args.config) as f:
            config = json.load(f)
        for key in (
            "model",
            "language",
            "debug",
            "log_transcripts",
            "log_performance",
            "extra_keyterms",
            "keyterm_boost",
            "use_ha_vocabulary",
            "ha_vocabulary_refresh_minutes",
            "transcription_interval",
            "vad_threshold",
            "decode_incomplete_lines",
            "save_debug_audio",
            "debug_audio_max_files",
        ):
            if key in config:
                setattr(args, key.replace("-", "_"), config[key])
    except Exception as e:
        _LOGGER.error(f"Failed to load config: {e}")
        return False
    return True


async def _refresh_ha_vocabulary_periodically(
    transcriber: object,
    manual_keyterms: list[str],
    interval_minutes: int,
    last_known_good_terms: list[str],
    moonshine_lock: asyncio.Lock,
) -> None:
    """Reload HA vocabulary every ``interval_minutes`` and re-apply keyterms.

    Runs only while use_ha_vocabulary is enabled and interval_minutes > 0.

    Last-known-good handling: a *failed* refresh (unreachable HA, timeout,
    auth error, ...) must not wipe out a previously working HA vocabulary --
    it keeps using ``last_known_good_terms`` unchanged and does not touch
    the transcriber at all. Only a *successful* fetch (which may legitimately
    return an empty term list, e.g. no areas/devices/entities configured)
    replaces it. See HaVocabularyResult's docstring in app/ha_vocabulary.py.

    ``last_known_good_terms`` is mutated in place (list passed by the
    caller) so the caller's copy always reflects the latest successful
    fetch, matching how it was seeded from the initial startup fetch.

    Every native call into the shared Moonshine transcriber -- including
    set_keyterms(), not just the streaming Start/AddAudio/Stop calls in
    app/streaming.py -- is serialized through the same ``moonshine_lock``
    passed in from app/__main__.py's main(), since moonshine-voice 0.1.5
    does not document set_keyterms() as safe to call concurrently with
    other native calls on the same Transcriber.
    """
    while True:
        await asyncio.sleep(interval_minutes * 60)
        result = await fetch_ha_vocabulary()
        if not result.success:
            _LOGGER.warning(
                "HA vocabulary refresh failed; keeping last-known-good vocabulary (%d term(s))",
                len(last_known_good_terms),
            )
            continue

        last_known_good_terms[:] = result.terms
        effective = merge_keyterms(last_known_good_terms, manual_keyterms)
        async with moonshine_lock:
            await asyncio.to_thread(transcriber.set_keyterms, effective)  # type: ignore[attr-defined]
        _LOGGER.info(
            "Refreshed Home Assistant vocabulary: ha_terms=%d effective keyterms=%d",
            len(last_known_good_terms),
            len(effective),
        )


def _validate_args(args: argparse.Namespace) -> bool:
    """Validate numeric option bounds. Returns False (and logs) on failure."""
    try:
        validate_transcription_interval(args.transcription_interval)
        validate_vad_threshold(args.vad_threshold)
        validate_keyterm_boost(args.keyterm_boost)
        validate_debug_audio_max_files(args.debug_audio_max_files)
        validate_ha_vocabulary_refresh_minutes(args.ha_vocabulary_refresh_minutes)
    except ValueError as e:
        _LOGGER.error(f"Invalid configuration: {e}")
        return False
    return True


def _resolve_keyterms(args: argparse.Namespace) -> tuple[list[str], list[str]]:
    """Parse manual keyterms and fetch HA vocabulary (if enabled).

    Returns (manual_keyterms, ha_terms). Never raises: an unreachable HA
    instance just means an empty ha_terms list -- there is no previous
    last-known-good vocabulary yet at startup, so a failed initial fetch
    can only fall back to "no HA terms" (see fetch_ha_vocabulary() /
    HaVocabularyResult).
    """
    manual_keyterms = parse_extra_keyterms(args.extra_keyterms)
    ha_terms: list[str] = []
    if args.use_ha_vocabulary:
        try:
            result = asyncio.run(fetch_ha_vocabulary())
            if result.success:
                ha_terms = result.terms
        except Exception as e:
            _LOGGER.warning("HA vocabulary unavailable (%s), continuing without HA vocabulary", e)
    return manual_keyterms, ha_terms


def _log_startup_banner(
    args: argparse.Namespace, manual_keyterms: list[str], effective_keyterms: list[str]
) -> None:
    _LOGGER.info("HomeIntent Moonshine STT v%s", VERSION)
    _LOGGER.info("model=%s", args.model)
    _LOGGER.info("language=%s", args.language)
    _LOGGER.info("HA vocabulary=%s", "enabled" if args.use_ha_vocabulary else "disabled")
    _LOGGER.info("manual keyterms=%d", len(manual_keyterms))
    _LOGGER.info("effective keyterms=%d", len(effective_keyterms))
    _LOGGER.info("transcription logging=%s", "enabled" if args.log_transcripts else "disabled")
    _LOGGER.info("debug audio=%s", "enabled" if args.save_debug_audio else "disabled")
    _LOGGER.info("model cache=%s", DEFAULT_MODEL_CACHE_DIR)


def _build_handler_factory(
    args: argparse.Namespace, transcriber: object, moonshine_lock: asyncio.Lock
) -> partial:  # type: ignore[type-arg]
    return partial(
        MoonshineAsrHandler,
        transcriber=transcriber,
        model_name=args.model,
        language=args.language,
        log_transcripts=args.log_transcripts,
        log_performance=args.log_performance,
        save_debug_audio_enabled=args.save_debug_audio,
        debug_audio_max_files=args.debug_audio_max_files,
        debug_audio_dir=DEFAULT_DEBUG_AUDIO_DIR,
        moonshine_lock=moonshine_lock,
    )


def _load_and_bias_transcriber(
    args: argparse.Namespace,
) -> tuple[object, list[str], list[str]] | None:
    """Load the transcriber and apply keyterm biasing. Returns None on failure.

    Returns (transcriber, manual_keyterms, ha_terms) so the caller can seed
    the periodic refresh loop's last-known-good HA vocabulary with exactly
    what was actually applied at startup.
    """
    try:
        transcriber = load_transcriber(
            model=args.model,
            language=args.language,
            transcription_interval=args.transcription_interval,
            vad_threshold=args.vad_threshold,
            decode_incomplete_lines=args.decode_incomplete_lines,
            keyterm_boost=args.keyterm_boost,
        )
    except Exception as e:
        _LOGGER.error(f"Failed to load model: {e}")
        return None

    manual_keyterms, ha_terms = _resolve_keyterms(args)
    effective_keyterms = merge_keyterms(ha_terms, manual_keyterms)
    if effective_keyterms:
        transcriber.set_keyterms(effective_keyterms)

    _log_startup_banner(args, manual_keyterms, effective_keyterms)
    return transcriber, manual_keyterms, ha_terms


def main() -> int:
    """Main entry point."""
    parser = build_arg_parser()
    args = parser.parse_args()

    if not _load_json_config_overrides(args):
        return 1

    if args.debug:
        logging.getLogger().setLevel(logging.DEBUG)
        _LOGGER.info("Debug logging enabled")

    if not _validate_args(args):
        return 1

    loaded = _load_and_bias_transcriber(args)
    if loaded is None:
        return 1
    transcriber, manual_keyterms, initial_ha_terms = loaded

    # One lock shared by every Wyoming connection's handler (native
    # start/add_audio/stop calls, see app/streaming.py) AND the periodic HA
    # vocabulary refresh's set_keyterms() call -- all native calls into the
    # same Moonshine Transcriber must be serialized consistently.
    moonshine_lock = asyncio.Lock()
    handler_factory = _build_handler_factory(args, transcriber, moonshine_lock)
    server = AsyncTcpServer(args.host, args.port)
    _LOGGER.info(f"Starting Wyoming server on {args.host}:{args.port}")

    async def run_server() -> None:
        refresh_task: asyncio.Task[None] | None = None
        if args.use_ha_vocabulary and args.ha_vocabulary_refresh_minutes > 0:
            last_known_good_terms = list(initial_ha_terms)
            refresh_task = asyncio.create_task(
                _refresh_ha_vocabulary_periodically(
                    transcriber,
                    manual_keyterms,
                    args.ha_vocabulary_refresh_minutes,
                    last_known_good_terms,
                    moonshine_lock,
                )
            )
        try:
            await server.run(handler_factory)
        finally:
            if refresh_task is not None:
                refresh_task.cancel()

    try:
        asyncio.run(run_server())
    except KeyboardInterrupt:
        _LOGGER.info("Interrupted")
        return 0
    except Exception as e:
        _LOGGER.error(f"Server error: {e}", exc_info=True)
        return 1

    return 0


if __name__ == "__main__":
    sys.exit(main())
