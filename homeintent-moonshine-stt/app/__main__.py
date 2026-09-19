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
from app.kokoro_session import KokoroOnnxSynthesizer
from app.kokoro_tts import (
    DEFAULT_KOKORO_REVISION,
    DEFAULT_KOKORO_VOICE,
    KokoroModelDownloadError,
    load_kokoro_model,
)
from app.kroko_engine import KrokoSttEngine, load_kroko_engine
from app.kroko_model import KrokoModelDownloadError
from app.models import (
    DEFAULT_DECODE_INCOMPLETE_LINES,
    DEFAULT_KEYTERM_BOOST,
    DEFAULT_MODEL_CACHE_DIR,
    DEFAULT_TRANSCRIPTION_INTERVAL,
    DEFAULT_VAD_THRESHOLD,
    load_transcriber,
)
from app.moonshine_engine import MoonshineSttEngine
from app.speechcatcher_engine import SpeechcatcherSttEngine, load_speechcatcher_engine
from app.speechcatcher_model import SpeechcatcherModelDownloadError
from app.stt_engine import SttEngine
from app.supertonic_session import SupertonicSynthesizer
from app.supertonic_tts import (
    DEFAULT_SUPERTONIC_STEPS,
    DEFAULT_SUPERTONIC_VOICE,
    SupertonicModelDownloadError,
    load_supertonic_tts,
)
from app.tts import DEFAULT_TTS_VOICE, GermanTtsModel, load_tts_model
from app.tts_engine import TtsSynthesizer
from app.tts_session import PocketTtsSynthesizer
from app.validation import (
    validate_debug_audio_max_files,
    validate_ha_vocabulary_refresh_minutes,
    validate_keyterm_boost,
    validate_kokoro_clause_pause,
    validate_kokoro_sentence_pause,
    validate_kokoro_speed,
    validate_kokoro_threads,
    validate_kroko_threads,
    validate_speechcatcher_beam_size,
    validate_speechcatcher_threads,
    validate_stt_engine,
    validate_supertonic_speed,
    validate_supertonic_steps,
    validate_supertonic_threads,
    validate_supertonic_voice,
    validate_transcription_interval,
    validate_tts_engine,
    validate_tts_threads,
    validate_vad_threshold,
)
from app.vosk_engine import VoskSttEngine, load_vosk_engine
from app.vosk_model import VoskModelDownloadError

# Setup logging
logging.basicConfig(
    format="%(asctime)s %(levelname)s [%(name)s] %(message)s",
    level=logging.INFO,
    handlers=[logging.StreamHandler(sys.stdout)],
)
_LOGGER = logging.getLogger(__name__)

VERSION = "0.6.1"


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
        help="Enable speech-to-text (engine selected via --stt-engine)",
    )
    parser.add_argument(
        "--stt-engine",
        choices=["moonshine", "kroko", "speechcatcher_m", "speechcatcher_l", "vosk_german"],
        # MUST default to moonshine: an existing installation's persisted
        # options.json predates this option entirely, and Supervisor's own
        # schema-default resolution is not something this add-on can
        # inspect from here -- rootfs's run script has its own identical
        # config_or_default() fallback (see that script), so a missing key
        # resolves to "moonshine" through two independent paths, not just
        # one. Upgrading must never silently switch a working Moonshine
        # setup to a different, undownloaded engine.
        default="moonshine",
        help="Which STT engine to use when stt_enabled (default: moonshine)",
    )
    parser.add_argument(
        "--model",
        choices=["tiny", "small"],
        default="small",
        help="Moonshine model size (ignored for stt_engine=kroko)",
    )
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
        "--kroko-threads",
        type=int,
        default=1,
        help="sherpa-onnx CPU threads for the Kroko recognizer (ignored for stt_engine=moonshine)",
    )
    parser.add_argument(
        "--kroko-hotwords-score",
        type=float,
        default=1.5,
        help="sherpa-onnx hotwords_score for Kroko keyterm/HA-vocabulary biasing",
    )
    parser.add_argument(
        "--speechcatcher-threads",
        type=int,
        default=0,
        help="torch intra-op CPU threads for Speechcatcher (0 = PyTorch's own default; "
        "ignored unless stt_engine is speechcatcher_m/speechcatcher_l)",
    )
    parser.add_argument(
        "--speechcatcher-beam-size",
        type=int,
        default=5,
        help="Beam search width for Speechcatcher's streaming decoder (upstream's own "
        "--beamsize default is 5; ignored unless stt_engine is speechcatcher_m/speechcatcher_l)",
    )

    parser.add_argument(
        "--tts-enabled",
        action=argparse.BooleanOptionalAction,
        # Off by default so an add-on *upgrade* never suddenly downloads a
        # PyTorch runtime + Pocket TTS model (several hundred MB) for a
        # previously STT-only installation -- see
        # ABSCHLUSSBERICHT_V0.2.0.md, "Backward compatibility".
        default=False,
        help="Enable text-to-speech (engine selected via --tts-engine)",
    )
    parser.add_argument(
        "--tts-engine",
        choices=["pocket_tts", "kokoro_onnx", "supertonic_3"],
        # MUST default to pocket_tts: an existing v0.2.7 installation's
        # persisted options.json predates this option entirely. This add-on
        # cannot inspect Supervisor's own closed-source schema-upgrade code
        # from here, so it does not rely on any assumption about what
        # Supervisor resolves a missing key to -- rootfs's run script reads
        # every option through its own config_or_default() helper (falling
        # back to this exact same default whenever bashio::config returns
        # empty/null, e.g. an absent key), the identical defensive pattern
        # already exercised end to end when tts_enabled/tts_model/etc. were
        # first added on top of a pure-STT v0.1.x install for v0.2.0 (see
        # ABSCHLUSSBERICHT_V0.2.0.md's "Backward compatibility" section).
        # Upgrading must never silently switch a working Pocket TTS setup to
        # a completely different, undownloaded engine.
        default="pocket_tts",
        help="Which TTS engine to use when tts_enabled (default: pocket_tts)",
    )
    parser.add_argument(
        "--tts-model",
        choices=[m.value for m in GermanTtsModel],
        default=GermanTtsModel.GERMAN.value,
        help="Pocket TTS model config (ignored for tts_engine=kokoro_onnx)",
    )
    parser.add_argument(
        "--tts-voice",
        default=DEFAULT_TTS_VOICE,
        help="Pocket TTS voice (ignored for tts_engine=kokoro_onnx)",
    )
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
        help="torch intra-op CPU threads for Pocket TTS (0 = PyTorch's own default; "
        "ignored for tts_engine=kokoro_onnx, see --kokoro-threads)",
    )

    parser.add_argument(
        "--kokoro-voice",
        default=DEFAULT_KOKORO_VOICE,
        help="Kokoro ONNX voice (ignored for tts_engine=pocket_tts)",
    )
    parser.add_argument(
        "--kokoro-speed",
        type=float,
        default=1.0,
        help="Kokoro ONNX speech speed, 0.5-2.0 (kokoro-onnx's own valid range)",
    )
    parser.add_argument(
        "--kokoro-threads",
        type=int,
        default=0,
        help="ONNX Runtime intra-op CPU threads for Kokoro (0 = onnxruntime's own default)",
    )
    parser.add_argument(
        "--kokoro-sentence-pause",
        type=float,
        default=0.25,
        help="Pause (seconds) after a sentence, Kokoro ONNX's own default",
    )
    parser.add_argument(
        "--kokoro-clause-pause",
        type=float,
        default=0.1,
        help="Pause (seconds) after a clause, Kokoro ONNX's own default",
    )
    parser.add_argument(
        "--kokoro-model-revision",
        default=DEFAULT_KOKORO_REVISION,
        help="Git revision of the Kokoro German model repo to download (advanced)",
    )

    parser.add_argument(
        "--supertonic-voice",
        default=DEFAULT_SUPERTONIC_VOICE,
        help="Supertonic 3 voice name (M1-M5/F1-F5) or numeric sid 0-9 "
        "(ignored for other tts_engine values)",
    )
    parser.add_argument(
        "--supertonic-speed",
        type=float,
        default=1.0,
        help="Supertonic 3 speech speed (larger = faster)",
    )
    parser.add_argument(
        "--supertonic-steps",
        type=int,
        default=DEFAULT_SUPERTONIC_STEPS,
        help="Supertonic 3 denoising steps (8=default/balanced, 10=higher quality per "
        "upstream's own documented example; higher = slower)",
    )
    parser.add_argument(
        "--supertonic-threads",
        type=int,
        default=1,
        help="sherpa-onnx CPU threads for Supertonic 3",
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
            "stt_engine",
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
            "kroko_threads",
            "kroko_hotwords_score",
            "speechcatcher_threads",
            "speechcatcher_beam_size",
            "tts_enabled",
            "tts_engine",
            "tts_model",
            "tts_voice",
            "tts_log_performance",
            "tts_warmup",
            "tts_threads",
            "kokoro_voice",
            "kokoro_speed",
            "kokoro_threads",
            "kokoro_sentence_pause",
            "kokoro_clause_pause",
            "supertonic_voice",
            "supertonic_speed",
            "supertonic_steps",
            "supertonic_threads",
        ):
            if key in config:
                setattr(args, key.replace("-", "_"), config[key])
    except Exception as e:
        _LOGGER.error(f"Failed to load config: {e}")
        return False
    return True


def _should_start_ha_vocabulary_refresh(
    args: argparse.Namespace, stt_engine: SttEngine | None
) -> bool:
    """Whether the periodic HA-vocabulary refresh loop should run.

    v0.6.1: capability-gated -- previously this only checked
    ``use_ha_vocabulary``/``ha_vocabulary_refresh_minutes``, so Speechcatcher
    and Vosk (neither of which can apply a refreshed vocabulary at all, see
    their own ``capabilities.supports_dynamic_vocabulary=False``) still ran
    a periodic Home Assistant API poll for no possible benefit. Now also
    requires ``stt_engine.capabilities.supports_dynamic_vocabulary`` --
    Moonshine and Kroko (both ``True``) keep refreshing exactly as before.
    """
    return (
        args.stt_enabled
        and stt_engine is not None
        and args.use_ha_vocabulary
        and args.ha_vocabulary_refresh_minutes > 0
        and stt_engine.capabilities.supports_dynamic_vocabulary
    )


def _log_ha_vocabulary_refresh_not_started(
    args: argparse.Namespace, stt_engine: SttEngine | None
) -> None:
    """One-time startup log line for the specific, actionable case: HA
    vocabulary is wanted (use_ha_vocabulary + a nonzero refresh interval)
    but the active engine cannot use it. Silent for every other reason the
    refresh loop did not start (STT disabled, use_ha_vocabulary off, ...) --
    those are already self-explanatory from the rest of the startup banner.
    """
    if (
        args.stt_enabled
        and stt_engine is not None
        and args.use_ha_vocabulary
        and args.ha_vocabulary_refresh_minutes > 0
        and not stt_engine.capabilities.supports_dynamic_vocabulary
    ):
        _LOGGER.info(
            "HA vocabulary refresh not supported by engine %s; periodic "
            "refresh disabled (no periodic Home Assistant API polling "
            "will occur)",
            stt_engine.engine_id,
        )


async def _refresh_ha_vocabulary_periodically(
    stt_engine: SttEngine,
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
            accepted, apply_succeeded = await asyncio.to_thread(
                stt_engine.set_keyterms,
                candidate_effective,
                previous_effective,
            )

        if not apply_succeeded:
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
            len(accepted),
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
        validate_stt_engine(args.stt_engine)
        validate_tts_engine(args.tts_engine)
        validate_kokoro_speed(args.kokoro_speed)
        validate_kokoro_threads(args.kokoro_threads)
        validate_kokoro_sentence_pause(args.kokoro_sentence_pause)
        validate_kokoro_clause_pause(args.kokoro_clause_pause)
        validate_kroko_threads(args.kroko_threads)
        validate_speechcatcher_threads(args.speechcatcher_threads)
        validate_speechcatcher_beam_size(args.speechcatcher_beam_size)
        validate_supertonic_speed(args.supertonic_speed)
        # These two never raise -- they gracefully fall back to the
        # documented default (with a warning log) for a value that is no
        # longer valid under the current dropdown schema, e.g. after an
        # upgrade from a version where these fields were free text/a wider
        # numeric range (see app/validation.py's docstrings).
        args.supertonic_steps = validate_supertonic_steps(args.supertonic_steps)
        args.supertonic_voice = validate_supertonic_voice(args.supertonic_voice)
        validate_supertonic_threads(args.supertonic_threads)
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
        _LOGGER.info("STT: engine=%s", args.stt_engine)
        if args.stt_engine == "kroko":
            _LOGGER.info("STT: language=%s threads=%d", args.language, args.kroko_threads)
        elif args.stt_engine in ("speechcatcher_m", "speechcatcher_l"):
            _LOGGER.info(
                "STT: language=%s beam_size=%d threads=%s",
                args.language,
                args.speechcatcher_beam_size,
                "auto (PyTorch default)"
                if args.speechcatcher_threads == 0
                else str(args.speechcatcher_threads),
            )
        elif args.stt_engine == "vosk_german":
            _LOGGER.info("STT: language=%s (lightweight Kaldi/Vosk model)", args.language)
        else:
            _LOGGER.info("STT: model=%s language=%s", args.model, args.language)
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
        _LOGGER.info("TTS: engine=%s", args.tts_engine)
        if args.tts_engine == "kokoro_onnx":
            _LOGGER.info(
                "TTS: model=german-martin voice=%s runtime=ONNX Runtime speed=%s",
                args.kokoro_voice,
                args.kokoro_speed,
            )
            _LOGGER.info(
                "TTS: threads=%s",
                "auto (onnxruntime default)"
                if args.kokoro_threads == 0
                else str(args.kokoro_threads),
            )
            _LOGGER.info("TTS: sample rate=24000 Hz")
        elif args.tts_engine == "supertonic_3":
            _LOGGER.info(
                "TTS: model=supertonic-3-int8 voice=%s runtime=sherpa-onnx (ONNX Runtime) "
                "speed=%s steps=%d threads=%d",
                args.supertonic_voice,
                args.supertonic_speed,
                args.supertonic_steps,
                args.supertonic_threads,
            )
        else:
            _LOGGER.info("TTS: model=%s voice=%s runtime=PyTorch", args.tts_model, args.tts_voice)
            _LOGGER.info(
                "TTS: threads=%s",
                "auto (PyTorch default)" if args.tts_threads == 0 else str(args.tts_threads),
            )
            _LOGGER.info("TTS: sample rate=24000 Hz")


def _build_handler_factory(
    args: argparse.Namespace,
    stt_engine: SttEngine | None,
    moonshine_lock: asyncio.Lock,
    tts_synthesizer: TtsSynthesizer | None,
) -> partial:  # type: ignore[type-arg]
    return partial(
        MoonshineAsrHandler,
        stt_engine=stt_engine,
        model_name=stt_engine.model_display_name if stt_engine is not None else "",
        language=args.language,
        log_transcripts=args.log_transcripts,
        log_performance=args.log_performance,
        save_debug_audio_enabled=args.save_debug_audio,
        debug_audio_max_files=args.debug_audio_max_files,
        debug_audio_dir=DEFAULT_DEBUG_AUDIO_DIR,
        moonshine_lock=moonshine_lock,
        tts_synthesizer=tts_synthesizer,
        tts_model_name=tts_synthesizer.model_name if tts_synthesizer is not None else "",
        tts_log_performance=args.tts_log_performance,
    )


def _load_pocket_tts_synthesizer(args: argparse.Namespace) -> PocketTtsSynthesizer | None:
    """Load the Pocket TTS model, validate+cache the configured default
    voice, and optionally warm it up. Returns None on failure.

    A load or voice-resolution failure here is treated as fatal (like a
    Moonshine load failure in _load_and_bias_stt_engine()) rather than
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

    synthesizer = PocketTtsSynthesizer(
        tts_model, default_voice=args.tts_voice, model_name=args.tts_model
    )
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


def _load_kokoro_synthesizer(args: argparse.Namespace) -> KokoroOnnxSynthesizer | None:
    """Resolve/download the Kokoro German model, validate+cache the
    configured default voice, and optionally warm it up. Returns None on
    failure.

    Exactly the same fail-loudly philosophy as
    _load_pocket_tts_synthesizer(): a download failure (network, repo
    unreachable, corrupt/partial file) or an invalid ``kokoro_voice`` is a
    fatal, clearly logged startup error -- the user explicitly chose
    ``tts_engine: kokoro_onnx``, so silently falling back to Pocket TTS (a
    different engine, different voice, possibly not even downloaded)
    would hide a real configuration problem instead of surfacing it (see
    this module's own docstring / ABSCHLUSSBERICHT_V0.3.0.md).
    """
    try:
        kokoro_model = load_kokoro_model(
            revision=args.kokoro_model_revision,
            intra_op_num_threads=args.kokoro_threads,
        )
    except KokoroModelDownloadError as e:
        _LOGGER.error(f"Failed to download/load Kokoro ONNX model: {e}")
        return None
    except Exception as e:
        _LOGGER.error(f"Failed to load Kokoro ONNX model: {e}")
        return None

    synthesizer = KokoroOnnxSynthesizer(
        kokoro_model,
        default_voice=args.kokoro_voice,
        speed=args.kokoro_speed,
        sentence_pause=args.kokoro_sentence_pause,
        clause_pause=args.kokoro_clause_pause,
    )
    try:
        asyncio.run(synthesizer.preload_default_voice())
    except Exception as e:
        _LOGGER.error(f"Failed to resolve Kokoro voice '{args.kokoro_voice}': {e}")
        return None

    if args.tts_warmup:
        try:
            elapsed = asyncio.run(synthesizer.warmup())
            _LOGGER.info("Kokoro ONNX warmup completed in %.2fs", elapsed)
        except Exception as e:
            _LOGGER.warning("Kokoro ONNX warmup failed (continuing without it): %s", e)

    return synthesizer


def _load_supertonic_synthesizer(args: argparse.Namespace) -> SupertonicSynthesizer | None:
    """Resolve/download the Supertonic 3 model, validate the configured
    voice, and optionally warm it up. Returns None on failure.

    Same fail-loudly philosophy as the other two TTS loaders: a download
    failure or an invalid ``supertonic_voice`` is a fatal, clearly logged
    startup error -- never a silent fallback to Pocket TTS/Kokoro.
    """
    try:
        tts = load_supertonic_tts(num_threads=args.supertonic_threads)
    except SupertonicModelDownloadError as e:
        _LOGGER.error(f"Failed to download/load Supertonic 3 model: {e}")
        return None
    except Exception as e:
        _LOGGER.error(f"Failed to load Supertonic 3 model: {e}")
        return None

    synthesizer = SupertonicSynthesizer(
        tts,
        default_voice=args.supertonic_voice,
        speed=args.supertonic_speed,
        num_steps=args.supertonic_steps,
        language=args.language,
    )
    try:
        asyncio.run(synthesizer.preload_default_voice())
    except Exception as e:
        _LOGGER.error(f"Failed to resolve Supertonic voice '{args.supertonic_voice}': {e}")
        return None

    if args.tts_warmup:
        try:
            elapsed = asyncio.run(synthesizer.warmup())
            _LOGGER.info("Supertonic 3 warmup completed in %.2fs", elapsed)
        except Exception as e:
            _LOGGER.warning("Supertonic 3 warmup failed (continuing without it): %s", e)

    return synthesizer


def _load_tts_synthesizer(args: argparse.Namespace) -> TtsSynthesizer | None:
    """Dispatch to the configured engine's own loader. Returns None on
    failure (see each loader's own docstring -- never a silent fallback to
    another engine)."""
    if args.tts_engine == "kokoro_onnx":
        return _load_kokoro_synthesizer(args)
    if args.tts_engine == "supertonic_3":
        return _load_supertonic_synthesizer(args)
    return _load_pocket_tts_synthesizer(args)


def _load_moonshine_engine(
    args: argparse.Namespace,
) -> tuple[MoonshineSttEngine, list[str], list[str], list[str]] | None:
    """Load the Moonshine transcriber and apply keyterm biasing. Returns
    None on failure.

    Returns (engine, manual_keyterms, ha_terms, accepted_keyterms) so the
    caller can seed the periodic refresh loop's last-known-good HA
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

    engine = MoonshineSttEngine(transcriber, args.model, args.language)
    manual_keyterms, ha_terms = _resolve_keyterms(args)
    effective_keyterms = merge_keyterms(ha_terms, manual_keyterms)
    accepted, _ = engine.set_keyterms(effective_keyterms)

    return engine, manual_keyterms, ha_terms, accepted


def _load_kroko_engine(
    args: argparse.Namespace,
) -> tuple[KrokoSttEngine, list[str], list[str], list[str]] | None:
    """Resolve/download the Kroko German model and apply keyterm/HA-vocabulary
    biasing via sherpa-onnx hotwords. Returns None on failure.

    Same fail-loudly philosophy as the TTS engine loaders: a download or
    load failure is a fatal, clearly logged startup error -- the user
    explicitly chose ``stt_engine: kroko``, so silently falling back to
    Moonshine would hide a real configuration/network problem.
    """
    try:
        engine = load_kroko_engine(
            num_threads=args.kroko_threads,
            hotwords_score=args.kroko_hotwords_score,
        )
    except KrokoModelDownloadError as e:
        _LOGGER.error(f"Failed to download/load Kroko model: {e}")
        return None
    except Exception as e:
        _LOGGER.error(f"Failed to load Kroko model: {e}")
        return None

    manual_keyterms, ha_terms = _resolve_keyterms(args)
    effective_keyterms = merge_keyterms(ha_terms, manual_keyterms)
    accepted, _ = engine.set_keyterms(effective_keyterms)

    return engine, manual_keyterms, ha_terms, accepted


def _load_speechcatcher_engine(
    args: argparse.Namespace,
) -> tuple[SpeechcatcherSttEngine, list[str], list[str], list[str]] | None:
    """Resolve/download the selected Speechcatcher model
    (speechcatcher_m/speechcatcher_l) and construct the engine. Returns
    None on failure.

    Same fail-loudly philosophy as the other STT/TTS engine loaders (see
    _load_kroko_engine's docstring). Speechcatcher has no hotword/HA-
    vocabulary mechanism (see app/speechcatcher_engine.py's module
    docstring) -- ``set_keyterms()`` is still called for interface
    consistency and to log a clear "not supported" message once at
    startup, but it never actually biases recognition.
    """
    try:
        engine = load_speechcatcher_engine(
            args.stt_engine,
            beam_size=args.speechcatcher_beam_size,
            num_threads=args.speechcatcher_threads,
        )
    except SpeechcatcherModelDownloadError as e:
        _LOGGER.error(f"Failed to download/load Speechcatcher model: {e}")
        return None
    except Exception as e:
        _LOGGER.error(f"Failed to load Speechcatcher model: {e}")
        return None

    manual_keyterms, ha_terms = _resolve_keyterms(args)
    effective_keyterms = merge_keyterms(ha_terms, manual_keyterms)
    accepted, _ = engine.set_keyterms(effective_keyterms)

    return engine, manual_keyterms, ha_terms, accepted


def _load_vosk_engine(
    args: argparse.Namespace,
) -> tuple[VoskSttEngine, list[str], list[str], list[str]] | None:
    """Resolve/download the Vosk German model and construct the engine.
    Returns None on failure.

    Same fail-loudly philosophy as the other STT engine loaders (see
    _load_kroko_engine's docstring). Vosk has no usable HA-vocabulary
    biasing mechanism (see app/vosk_engine.py's module docstring) --
    ``set_keyterms()`` is still called for interface consistency and to
    log a clear "not supported" message once at startup, but it never
    actually biases recognition.
    """
    try:
        engine = load_vosk_engine()
    except VoskModelDownloadError as e:
        _LOGGER.error(f"Failed to download/load Vosk model: {e}")
        return None
    except Exception as e:
        _LOGGER.error(f"Failed to load Vosk model: {e}")
        return None

    manual_keyterms, ha_terms = _resolve_keyterms(args)
    effective_keyterms = merge_keyterms(ha_terms, manual_keyterms)
    accepted, _ = engine.set_keyterms(effective_keyterms)

    return engine, manual_keyterms, ha_terms, accepted


def _load_and_bias_stt_engine(
    args: argparse.Namespace,
) -> tuple[SttEngine, list[str], list[str], list[str]] | None:
    """Dispatch to the configured STT engine's own loader. Returns None on
    failure (see each loader's own docstring)."""
    if args.stt_engine == "kroko":
        return _load_kroko_engine(args)
    if args.stt_engine in ("speechcatcher_m", "speechcatcher_l"):
        return _load_speechcatcher_engine(args)
    if args.stt_engine == "vosk_german":
        return _load_vosk_engine(args)
    return _load_moonshine_engine(args)


class _LoadedEngines:
    """Everything main() needs after loading the enabled STT/TTS engines."""

    def __init__(
        self,
        stt_engine: SttEngine | None,
        manual_keyterms: list[str],
        initial_ha_terms: list[str],
        accepted_keyterms: list[str],
        tts_synthesizer: TtsSynthesizer | None,
    ) -> None:
        self.stt_engine = stt_engine
        self.manual_keyterms = manual_keyterms
        self.initial_ha_terms = initial_ha_terms
        self.accepted_keyterms = accepted_keyterms
        self.tts_synthesizer = tts_synthesizer


def _load_engines(args: argparse.Namespace) -> _LoadedEngines | None:
    """Load whichever of STT/TTS is enabled. Returns None on any failure."""
    if not args.stt_enabled and not args.tts_enabled:
        _LOGGER.error("Both stt_enabled and tts_enabled are false; nothing to serve")
        return None

    stt_engine: SttEngine | None = None
    manual_keyterms: list[str] = []
    initial_ha_terms: list[str] = []
    accepted_keyterms: list[str] = []
    if args.stt_enabled:
        loaded = _load_and_bias_stt_engine(args)
        if loaded is None:
            return None
        stt_engine, manual_keyterms, initial_ha_terms, accepted_keyterms = loaded

    tts_synthesizer: TtsSynthesizer | None = None
    if args.tts_enabled:
        tts_synthesizer = _load_tts_synthesizer(args)
        if tts_synthesizer is None:
            return None

    return _LoadedEngines(
        stt_engine, manual_keyterms, initial_ha_terms, accepted_keyterms, tts_synthesizer
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
    stt_engine = engines.stt_engine
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
    handler_factory = _build_handler_factory(args, stt_engine, moonshine_lock, tts_synthesizer)
    server = AsyncTcpServer(args.host, args.port)
    _LOGGER.info(f"Starting Wyoming server on {args.host}:{args.port}")

    async def run_server() -> None:
        refresh_task: asyncio.Task[None] | None = None
        if _should_start_ha_vocabulary_refresh(args, stt_engine):
            assert stt_engine is not None  # narrowed by the check above
            last_known_good_terms = list(initial_ha_terms)
            refresh_task = asyncio.create_task(
                _refresh_ha_vocabulary_periodically(
                    stt_engine,
                    manual_keyterms,
                    args.ha_vocabulary_refresh_minutes,
                    last_known_good_terms,
                    moonshine_lock,
                )
            )
        else:
            _log_ha_vocabulary_refresh_not_started(args, stt_engine)
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
