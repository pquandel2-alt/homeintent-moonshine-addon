"""Config-level tests for app/__main__.py (B12/B13/B27 scenarios).

Covers: old (pre-TTS) vs new config files, stt_enabled/tts_enabled
combinations, and startup failing cleanly (not crashing with a traceback
into the s6 supervision tree) for an invalid TTS model.
"""

import json
import sys
from pathlib import Path
from unittest.mock import MagicMock, patch

from app.__main__ import (
    _load_engines,
    _load_json_config_overrides,
    build_arg_parser,
    main,
)


def _parse(argv: list[str]):
    return build_arg_parser().parse_args(argv)


class TestDefaults:
    def test_stt_enabled_by_default(self):
        args = _parse([])
        assert args.stt_enabled is True

    def test_tts_disabled_by_default(self):
        """Backward-compat default: an upgraded install must not suddenly
        start downloading a TTS model -- see ABSCHLUSSBERICHT_V0.2.0.md."""
        args = _parse([])
        assert args.tts_enabled is False

    def test_default_tts_model_is_the_faster_one(self):
        args = _parse([])
        assert args.tts_model == "german"

    def test_default_tts_voice_is_juergen(self):
        args = _parse([])
        assert args.tts_voice == "juergen"


class TestJsonConfigOverrides:
    def test_old_pre_tts_config_loads_without_error(self, tmp_path: Path):
        """A config file from before TTS existed (only the original v0.1.0
        keys) must still load cleanly, with new options at their CLI
        defaults."""
        config_path = tmp_path / "config.json"
        config_path.write_text(
            json.dumps({"model": "small", "language": "de", "log_level": "INFO"})
        )
        args = _parse(["--config", str(config_path)])

        assert _load_json_config_overrides(args) is True
        assert args.model == "small"
        assert args.tts_enabled is False  # untouched CLI default
        assert args.stt_enabled is True  # untouched CLI default

    def test_new_config_with_tts_enabled(self, tmp_path: Path):
        config_path = tmp_path / "config.json"
        config_path.write_text(
            json.dumps(
                {
                    "stt_enabled": True,
                    "tts_enabled": True,
                    "tts_model": "german_24l",
                    "tts_voice": "alba",
                }
            )
        )
        args = _parse(["--config", str(config_path)])

        assert _load_json_config_overrides(args) is True
        assert args.tts_enabled is True
        assert args.tts_model == "german_24l"
        assert args.tts_voice == "alba"

    def test_stt_disabled_tts_enabled_config(self, tmp_path: Path):
        config_path = tmp_path / "config.json"
        config_path.write_text(json.dumps({"stt_enabled": False, "tts_enabled": True}))
        args = _parse(["--config", str(config_path)])

        assert _load_json_config_overrides(args) is True
        assert args.stt_enabled is False
        assert args.tts_enabled is True

    def test_malformed_json_fails_cleanly(self, tmp_path: Path):
        config_path = tmp_path / "config.json"
        config_path.write_text("{not valid json")
        args = _parse(["--config", str(config_path)])

        assert _load_json_config_overrides(args) is False


class TestLoadEngines:
    def test_both_disabled_returns_none(self):
        args = _parse(["--no-stt-enabled", "--no-tts-enabled"])
        assert _load_engines(args) is None

    def test_stt_only_loads_transcriber_not_tts(self):
        args = _parse(["--no-tts-enabled"])
        with (
            patch("app.__main__.load_transcriber") as mock_load,
            patch("app.__main__.fetch_ha_vocabulary"),
        ):
            mock_load.return_value = MagicMock()
            args.use_ha_vocabulary = False
            engines = _load_engines(args)

        assert engines is not None
        assert engines.stt_engine is not None
        assert engines.tts_synthesizer is None

    def test_tts_only_loads_synthesizer_not_transcriber(self):
        args = _parse(["--no-stt-enabled", "--tts-enabled"])
        with patch("app.__main__.load_tts_model") as mock_load_tts:
            mock_load_tts.return_value = MagicMock(has_voice_cloning=True, sample_rate=24000)
            # Warmup (item 8) actually consumes generate_audio_stream()'s
            # result via a real for-loop -- a bare MagicMock() would iterate
            # forever (its __next__ never raises StopIteration).
            mock_load_tts.return_value.generate_audio_stream.return_value = iter([])
            engines = _load_engines(args)

        assert engines is not None
        assert engines.stt_engine is None
        assert engines.tts_synthesizer is not None

    def test_both_enabled_loads_both(self):
        args = _parse(["--tts-enabled"])
        args.use_ha_vocabulary = False
        with (
            patch("app.__main__.load_transcriber") as mock_load_stt,
            patch("app.__main__.load_tts_model") as mock_load_tts,
        ):
            mock_load_stt.return_value = MagicMock()
            mock_load_tts.return_value = MagicMock(has_voice_cloning=True, sample_rate=24000)
            mock_load_tts.return_value.generate_audio_stream.return_value = iter([])
            engines = _load_engines(args)

        assert engines is not None
        assert engines.stt_engine is not None
        assert engines.tts_synthesizer is not None

    def test_invalid_tts_model_fails_cleanly_not_crashing(self, tmp_path: Path):
        """An invalid tts_model reaching startup (e.g. via a malformed
        options.json bypassing config.yaml's own schema check) must fail
        _load_engines() cleanly, not raise out of it."""
        config_path = tmp_path / "config.json"
        config_path.write_text(json.dumps({"tts_enabled": True, "tts_model": "not-a-real-model"}))
        args = _parse(["--config", str(config_path)])
        assert _load_json_config_overrides(args) is True

        engines = _load_engines(args)  # real load_tts_model() call, not mocked
        assert engines is None


class TestTtsVoiceValidationAndWarmup:
    """Item 7/8: the configured default TTS voice must be resolved (and
    optionally warmed up) once at startup, not lazily on the first real
    synthesize request -- see app/__main__.py's _load_tts_synthesizer()."""

    def _args_with_mock_model(self, mock_load_tts: MagicMock, extra_argv: list[str] | None = None):
        args = _parse(["--no-stt-enabled", "--tts-enabled", *(extra_argv or [])])
        mock_load_tts.return_value = MagicMock(has_voice_cloning=True, sample_rate=24000)
        mock_load_tts.return_value.generate_audio_stream.return_value = iter([])
        return args

    def test_invalid_voice_fails_startup_not_first_request(self):
        """get_state_for_audio_prompt() raising (invalid voice name) must
        fail _load_engines() at startup, before any Wyoming connection."""
        with patch("app.__main__.load_tts_model") as mock_load_tts:
            args = self._args_with_mock_model(mock_load_tts, ["--tts-voice", "not-a-real-voice"])
            mock_load_tts.return_value.get_state_for_audio_prompt.side_effect = ValueError(
                "unknown voice"
            )
            engines = _load_engines(args)

        assert engines is None

    def test_valid_voice_resolved_and_cached_at_startup(self):
        with patch("app.__main__.load_tts_model") as mock_load_tts:
            args = self._args_with_mock_model(mock_load_tts)
            engines = _load_engines(args)

        assert engines is not None
        assert engines.tts_synthesizer is not None
        # Cached during startup preload -- a real synthesize request must
        # not call get_state_for_audio_prompt() again for the same voice.
        assert mock_load_tts.return_value.get_state_for_audio_prompt.call_count == 1

    def test_warmup_runs_by_default(self):
        with patch("app.__main__.load_tts_model") as mock_load_tts:
            args = self._args_with_mock_model(mock_load_tts)
            engines = _load_engines(args)

        assert engines is not None
        mock_load_tts.return_value.generate_audio_stream.assert_called_once()

    def test_warmup_disabled_skips_generate_call(self):
        with patch("app.__main__.load_tts_model") as mock_load_tts:
            args = self._args_with_mock_model(mock_load_tts, ["--no-tts-warmup"])
            engines = _load_engines(args)

        assert engines is not None
        mock_load_tts.return_value.generate_audio_stream.assert_not_called()

    def test_warmup_failure_is_not_fatal(self):
        """Warmup is a pure optimization -- a failure there must not take
        down an otherwise successfully validated TTS engine."""
        with patch("app.__main__.load_tts_model") as mock_load_tts:
            args = self._args_with_mock_model(mock_load_tts)
            mock_load_tts.return_value.generate_audio_stream.side_effect = RuntimeError(
                "warmup boom"
            )
            engines = _load_engines(args)

        assert engines is not None
        assert engines.tts_synthesizer is not None


class TestStartupBannerLoggedOnce:
    """Item 24: a real production log showed the full startup banner
    (version, STT/TTS config summary) printed TWICE -- once from inside
    _load_and_bias_stt_engine() (before Pocket TTS even started loading)
    and once again from main() after everything finished loading. Only the
    second, fully-informed call (with the real tts_enabled/tts_synthesizer
    state) should ever run; loading itself should still emit its own
    incremental "Loading X..."/"X ready" lines (already logged by
    app/models.py and app/tts.py), just not a second full banner."""

    def test_load_and_bias_stt_engine_does_not_log_the_banner(self):
        from app.__main__ import _load_and_bias_stt_engine

        args = _parse([])
        args.use_ha_vocabulary = False
        with (
            patch("app.__main__.load_transcriber") as mock_load,
            patch("app.__main__._log_startup_banner") as mock_banner,
        ):
            mock_load.return_value = MagicMock()
            result = _load_and_bias_stt_engine(args)

        assert result is not None
        mock_banner.assert_not_called()

    def test_main_logs_startup_banner_exactly_once(self):
        argv = ["prog", "--no-tts-enabled"]
        with (
            patch.object(sys, "argv", argv),
            patch("app.__main__.load_transcriber") as mock_load,
            patch("app.__main__.fetch_ha_vocabulary"),
            patch("app.__main__._log_startup_banner") as mock_banner,
            patch("app.__main__.AsyncTcpServer") as mock_server_cls,
        ):
            mock_load.return_value = MagicMock()
            mock_server = MagicMock()

            async def _fake_run(handler_factory):
                return None

            mock_server.run.side_effect = _fake_run
            mock_server_cls.return_value = mock_server

            exit_code = main()

        assert exit_code == 0
        mock_banner.assert_called_once()


class _FakeMoonshineTranscriberForMain:
    """A minimal fake matching real moonshine_voice.Transcriber.set_keyterms()
    semantics: raises MoonshineError for the whole call if any term in the
    attempted list is incompatible -- exactly how the real production
    incident ("/Büro") crashed the add-on."""

    def __init__(self, incompatible_terms: set[str]) -> None:
        self.incompatible_terms = incompatible_terms
        self.set_keyterms_calls: list[list[str]] = []

    def set_keyterms(self, terms: list[str]) -> None:
        self.set_keyterms_calls.append(list(terms))
        for term in terms:
            if term in self.incompatible_terms:
                from moonshine_voice import MoonshineError

                raise MoonshineError(f"Failed to set key terms: No match found for {term!r}")


class TestKeytermCrashRegression:
    """Regression coverage for the exact real production incident: the
    Home-Assistant-derived entity name "/Büro" made
    transcriber.set_keyterms() raise MoonshineError, uncaught, which
    crashed the whole add-on (exit code 1) into a permanent restart loop.
    Startup must now survive an incompatible keyterm from ANY source
    (manual extra_keyterms here; HA-derived terms go through the identical
    apply_safe_keyterms() path, see app/tests/test_keyterms.py)."""

    def test_startup_does_not_crash_on_incompatible_manual_keyterm(self):
        from app.__main__ import _load_and_bias_stt_engine

        args = _parse([])
        args.use_ha_vocabulary = False
        args.extra_keyterms = "Wohnzimmer,/Büro,Küche"
        transcriber = _FakeMoonshineTranscriberForMain(incompatible_terms={"/Büro"})

        with patch("app.__main__.load_transcriber", return_value=transcriber):
            result = _load_and_bias_stt_engine(args)

        assert result is not None  # startup does not fail
        _, _manual_keyterms, _ha_terms, accepted = result
        assert "/Büro" not in accepted
        assert "Wohnzimmer" in accepted
        assert "Küche" in accepted

    def test_startup_survives_when_every_keyterm_is_incompatible(self):
        """Fall D: STT is more important than keyterm biasing -- the
        service must still start with biasing simply turned off. The
        speech-fallback variants ("Büro", "Foo", with the separator turned
        into a space) are also marked incompatible here so this genuinely
        exercises full rejection rather than fallback recovery."""
        from app.__main__ import _load_and_bias_stt_engine

        args = _parse([])
        args.use_ha_vocabulary = False
        args.extra_keyterms = "/Büro,\\Foo"
        transcriber = _FakeMoonshineTranscriberForMain(
            incompatible_terms={"/Büro", "\\Foo", "Büro", "Foo"}
        )

        with patch("app.__main__.load_transcriber", return_value=transcriber):
            result = _load_and_bias_stt_engine(args)

        assert result is not None
        _, _, _, accepted = result
        assert accepted == []

    def test_main_end_to_end_survives_incompatible_keyterm(self):
        """Definition of done: the full main() flow (including starting the
        Wyoming server) must complete normally even with an incompatible
        keyterm in the mix -- no crash, no non-zero exit code."""
        argv = ["prog", "--no-tts-enabled", "--extra-keyterms", "Wohnzimmer,/Büro"]
        transcriber = _FakeMoonshineTranscriberForMain(incompatible_terms={"/Büro"})

        with (
            patch.object(sys, "argv", argv),
            patch("app.__main__.load_transcriber", return_value=transcriber),
            patch("app.__main__.AsyncTcpServer") as mock_server_cls,
        ):
            args_for_ha = build_arg_parser().parse_args(argv[1:])
            args_for_ha.use_ha_vocabulary = False
            with patch("app.__main__.build_arg_parser") as mock_parser:
                mock_parser.return_value.parse_args.return_value = args_for_ha

                mock_server = MagicMock()

                async def _fake_run(handler_factory):
                    return None

                mock_server.run.side_effect = _fake_run
                mock_server_cls.return_value = mock_server

                exit_code = main()

        assert exit_code == 0
        assert "/Büro" not in transcriber.set_keyterms_calls[-1]
        assert "Wohnzimmer" in transcriber.set_keyterms_calls[-1]
