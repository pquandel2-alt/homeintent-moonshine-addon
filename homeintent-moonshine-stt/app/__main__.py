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
from app.keyterms import apply_safe_keyterms, merge_keyterms, parse_extra_keyterms
from app.models import (
    DEFAULT_DECODE_INCOMPLETE_LINES,
    DEFAULT_KEYTERM_BOOST,
    DEFAULT_MODEL_CACHE_DIR,
    DEFAULT_TRANSCRIPTION_INTERVAL,
    DEFAULT_VAD_THRESHOLD,
    load_transcriber,
)
from app.tts import DEFAULT_TTS_VOICE, GermanTtsModel, load_tts_model
from app.tts_session import PocketTtsSynthesizer
from app.validation import (
    validate_debug_audio_max_files,
    validate_ha_vocabulary_refresh_minutes,
    validate_keyterm_boost,
    validate_transcription_interval,
    validate_tts_threads,
    validate_vad_threshold,
)

# Setup logging
logging.basicConfig(
    format="%(asctime)s %(levelname)s [%(name)s] %(message)s",
    level=logging.INFO,
    handlers=[logging.StreamHandler(sys.stdout)],
)
_LOGGER = logging.getLogger(__name__)

VERSION = "0.2.6"


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="HomeIntent Moonshine Voice Server")

    parser.add_argument("--port", type=int, default=10300, help="Port to listen on")
    parser.add_argument("--host", default="0.0.0.0", help="Host to bind to")
    parser.add_argument("--debug", action="store_true", help="Enable debug logging")
    parser.add_argument("--config", type=Path, help="Path to JSON config file")

    parser.add_argument(
        "--stt-enabled",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Enable Moonshine speech-to-text",
    )
    parser.add_argument("--model", choices=["tiny", "small"], default="small")
    parser.add_argument("--language", default="de")
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

    parser.add_argument(
        "--tts-enabled",
        action=argparse.BooleanOptionalAction,
        # Off by default so an add-on *upgrade* never suddenly downloads a
        # PyTorch runtime + Pocket TTS model (several hundred MB) for a
        # previously STT-only installation -- see
        # ABSCHLUSSBERICHT_V0.2.0.md, "Backward compatibility".
        default=False,
        help="Enable Kyutai Pocket TTS text-to-speech",
    )
    parser.add_argument(
        "--tts-model",
        choices=[m.value for m in GermanTtsModel],
        default=GermanTtsModel.GERMAN.value,
    )
    parser.add_argument("--tts-voice", default=DEFAULT_TTS_VOICE)
    parser.add_argument(
        "--tts-log-performance",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Log a compact per-request TTFA/RTF line",
    )
    parser.add_argument(
        "--tts-warmup",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Run one discarded synthesis at startup to avoid a slow first real request",
    )
    parser.add_argument(
        "--tts-threads",
        type=int,
        default=0,
        help="torch intra-op CPU threads for Pocket TTS (0 = PyTorch's own default)",
    )

    return parser


def _load_json_config_overrides(args: argparse.Namespace) -> bool:
    """Merge a JSON --config file's values into args. Returns False on failure."""
    if not args.config or not args.config.exists():
        return True
    try:
        with open(args.config) as f:
            config = json.load(f)
        for key in (
            "stt_enabled",
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
            "tts_enabled",
            "tts_model",
            "tts_voice",
            "tts_log_performance",
            "tts_warmup",
            "tts_threads",
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
    is even considered for applying. See HaVocabularyResult's docstring in
    app/ha_vocabulary.py.

    Applying the candidate terms always goes through apply_safe_keyterms()
    (app/keyterms.py), which never raises: an individual HA name the
    loaded model's tokenizer cannot represent (e.g. a newly added or
    renamed entity/alias/area, possibly only incompatible in an
    automatically generated Area+Entity combination) is skipped, not fatal
    -- the refresh loop and STT both keep running.

    ``last_known_good_terms`` is only updated (mutated in place, so the
    caller's copy always reflects the latest successful fetch) AFTER
    apply_safe_keyterms() confirms the final authoritative apply itself
    succeeded (``apply_succeeded``) -- never before, and never merely
    because the HA fetch succeeded. This keeps the transition atomic: if
    the final apply call itself fails unexpectedly (a genuinely structural
    error, since per-term content issues are already filtered out by that
    point), apply_safe_keyterms() restores the previous effective keyterms
    itself and ``last_known_good_terms`` is left completely untouched, so a
    later refresh attempt starts from the same known-good baseline again.

    Every native call into the shared Moonshine transcriber -- including
    set_keyterms(), not just the streaming Start/AddAudio/Stop calls in
    app/streaming.py -- is serialized through the same ``moonshine_lock``
    passed in from app/__main__.py's main(), since moonshine-voice 0.1.5
    does not document set_keyterms() as safe to call concurrently with
    other native calls on the same Transcriber. apply_safe_keyterms() may
    issue several native calls while isolating an incompatible term; the
    entire operation runs inside one lock acquisition (via a single
    to_thread call) so it stays atomic with respect to concurrent
    streaming sessions.
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

        candidate_ha_terms = result.terms
        candidate_effective = merge_keyterms(candidate_ha_terms, manual_keyterms)
        previous_effective = merge_keyterms(last_known_good_terms, manual_keyterms)
        async with moonshine_lock:
            apply_result = await asyncio.to_thread(
                apply_safe_keyterms,
                transcriber,  # type: ignore[arg-type]
                candidate_effective,
                previous_effective,
            )

        if not apply_result.apply_succeeded:
            _LOGGER.error(
                "HA vocabulary refresh: applying the new keyterm list failed unexpectedly; "
                "keeping last-known-good vocabulary (%d term(s))",
                len(last_known_good_terms),
            )
            continue

        last_known_good_terms[:] = candidate_ha_terms
        _LOGGER.info(
            "Refreshed Home Assistant vocabulary: ha_terms=%d effective keyterms=%d",
            len(last_known_good_terms),
            len(apply_result.accepted),
        )


def _validate_args(args: argparse.Namespace) -> bool:
    """Validate numeric option bounds. Returns False (and logs) on failure."""
    try:
        validate_transcription_interval(args.transcription_interval)
        validate_vad_threshold(args.vad_threshold)
        validate_keyterm_boost(args.keyterm_boost)
        validate_debug_audio_max_files(args.debug_audio_max_files)
        validate_ha_vocabulary_refresh_minutes(args.ha_vocabulary_refresh_minutes)
        validate_tts_threads(args.tts_threads)
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
    _LOGGER.info("HomeIntent Moonshine Voice v%s", VERSION)
    _LOGGER.info("STT: enabled=%s", args.stt_enabled)
    if args.stt_enabled:
        _LOGGER.info("STT: engine=Moonshine model=%s language=%s", args.model, args.language)
        _LOGGER.info("STT: HA vocabulary=%s", "enabled" if args.use_ha_vocabulary else "disabled")
        _LOGGER.info("STT: manual keyterms=%d", len(manual_keyterms))
        _LOGGER.info("STT: effective keyterms=%d", len(effective_keyterms))
        _LOGGER.info(
            "STT: transcription logging=%s", "enabled" if args.log_transcripts else "disabled"
        )
        _LOGGER.info("STT: debug audio=%s", "enabled" if args.save_debug_audio else "disabled")
        _LOGGER.info("STT: model cache=%s", DEFAULT_MODEL_CACHE_DIR)
    _LOGGER.info("TTS: enabled=%s", args.tts_enabled)
    if args.tts_enabled:
        _LOGGER.info("TTS: engine=Pocket TTS model=%s voice=%s", args.tts_model, args.tts_voice)
        _LOGGER.info(
            "TTS: threads=%s",
            "auto (PyTorch default)" if args.tts_threads == 0 else str(args.tts_threads),
        )


def _build_handler_factory(
    args: argparse.Namespace,
    transcriber: object | None,
    moonshine_lock: asyncio.Lock,
    tts_synthesizer: PocketTtsSynthesizer | None,
) -> partial:  # type: ignore[type-arg]
    return partial(
        MoonshineAsrHandler,
        transcriber=transcriber,
        model_name=args.model if args.stt_enabled else "",
        language=args.language,
        log_transcripts=args.log_transcripts,
        log_performance=args.log_performance,
        save_debug_audio_enabled=args.save_debug_audio,
        debug_audio_max_files=args.debug_audio_max_files,
        debug_audio_dir=DEFAULT_DEBUG_AUDIO_DIR,
        moonshine_lock=moonshine_lock,
        tts_synthesizer=tts_synthesizer,
        tts_model_name=args.tts_model if args.tts_enabled else "",
        tts_log_performance=args.tts_log_performance,
    )


def _load_tts_synthesizer(args: argparse.Namespace) -> PocketTtsSynthesizer | None:
    """Load the Pocket TTS model, validate+cache the configured default
    voice, and optionally warm it up. Returns None on failure.

    A load or voice-resolution failure here is treated as fatal (like a
    Moonshine load failure in _load_and_bias_transcriber()) rather than
    silently falling back to "no TTS": the user explicitly opted into
    tts_enabled with a specific tts_voice, so failing loudly at startup is
    more useful than a working add-on that only discovers an invalid voice
    name on its first real synthesize request. Warmup failure, in contrast,
    is not fatal -- it is a pure performance optimization (see
    PocketTtsSynthesizer.warmup()'s docstring), not a correctness signal.
    """
    try:
        tts_model = load_tts_model(model=args.tts_model, num_threads=args.tts_threads)
    except Exception as e:
        _LOGGER.error(f"Failed to load Pocket TTS model: {e}")
        return None

    synthesizer = PocketTtsSynthesizer(tts_model, default_voice=args.tts_voice)
    try:
        asyncio.run(synthesizer.preload_default_voice())
    except Exception as e:
        _LOGGER.error(f"Failed to resolve TTS voice '{args.tts_voice}': {e}")
        return None

    if args.tts_warmup:
        try:
            elapsed = asyncio.run(synthesizer.warmup())
            _LOGGER.info("Pocket TTS warmup completed in %.2fs", elapsed)
        except Exception as e:
            _LOGGER.warning("Pocket TTS warmup failed (continuing without it): %s", e)

    return synthesizer


def _load_and_bias_transcriber(
    args: argparse.Namespace,
) -> tuple[object, list[str], list[str], list[str]] | None:
    """Load the transcriber and apply keyterm biasing. Returns None on failure.

    Returns (transcriber, manual_keyterms, ha_terms, accepted_keyterms) so
    the caller can seed the periodic refresh loop's last-known-good HA
    vocabulary with exactly what was actually fetched at startup, and log
    the real, post-validation effective keyterm count.

    Keyterm application always goes through apply_safe_keyterms() (see
    app/keyterms.py): a single Home Assistant entity/alias/area name the
    loaded model's tokenizer cannot represent (a real production incident:
    the entity name "/Büro" made set_keyterms() raise MoonshineError, which
    was uncaught here and crashed the whole add-on into a restart loop)
    must never prevent the add-on from starting -- STT without keyterm
    biasing is strictly better than no STT at all. No lock is needed here:
    this runs before the Wyoming server starts serving any connection, so
    nothing else can be touching the transcriber concurrently yet.
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
    result = apply_safe_keyterms(transcriber, effective_keyterms)

    return transcriber, manual_keyterms, ha_terms, result.accepted


class _LoadedEngines:
    """Everything main() needs after loading the enabled STT/TTS engines."""

    def __init__(
        self,
        transcriber: object | None,
        manual_keyterms: list[str],
        initial_ha_terms: list[str],
        accepted_keyterms: list[str],
        tts_synthesizer: PocketTtsSynthesizer | None,
    ) -> None:
        self.transcriber = transcriber
        self.manual_keyterms = manual_keyterms
        self.initial_ha_terms = initial_ha_terms
        self.accepted_keyterms = accepted_keyterms
        self.tts_synthesizer = tts_synthesizer


def _load_engines(args: argparse.Namespace) -> _LoadedEngines | None:
    """Load whichever of STT/TTS is enabled. Returns None on any failure."""
    if not args.stt_enabled and not args.tts_enabled:
        _LOGGER.error("Both stt_enabled and tts_enabled are false; nothing to serve")
        return None

    transcriber: object | None = None
    manual_keyterms: list[str] = []
    initial_ha_terms: list[str] = []
    accepted_keyterms: list[str] = []
    if args.stt_enabled:
        loaded = _load_and_bias_transcriber(args)
        if loaded is None:
            return None
        transcriber, manual_keyterms, initial_ha_terms, accepted_keyterms = loaded

    tts_synthesizer: PocketTtsSynthesizer | None = None
    if args.tts_enabled:
        tts_synthesizer = _load_tts_synthesizer(args)
        if tts_synthesizer is None:
            return None

    return _LoadedEngines(
        transcriber, manual_keyterms, initial_ha_terms, accepted_keyterms, tts_synthesizer
    )


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

    engines = _load_engines(args)
    if engines is None:
        return 1
    transcriber = engines.transcriber
    manual_keyterms = engines.manual_keyterms
    initial_ha_terms = engines.initial_ha_terms
    tts_synthesizer = engines.tts_synthesizer

    _log_startup_banner(args, manual_keyterms, engines.accepted_keyterms)

    # One lock shared by every Wyoming connection's handler (native
    # start/add_audio/stop calls, see app/streaming.py) AND the periodic HA
    # vocabulary refresh's set_keyterms() call -- all native calls into the
    # same Moonshine Transcriber must be serialized consistently. Pocket
    # TTS gets its own, independent lock inside PocketTtsSynthesizer (see
    # app/tts_session.py) since it is an unrelated runtime with no shared
    # state -- an STT request never blocks on a TTS request or vice versa.
    moonshine_lock = asyncio.Lock()
    handler_factory = _build_handler_factory(args, transcriber, moonshine_lock, tts_synthesizer)
    server = AsyncTcpServer(args.host, args.port)
    _LOGGER.info(f"Starting Wyoming server on {args.host}:{args.port}")

    async def run_server() -> None:
        refresh_task: asyncio.Task[None] | None = None
        if args.stt_enabled and args.use_ha_vocabulary and args.ha_vocabulary_refresh_minutes > 0:
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
