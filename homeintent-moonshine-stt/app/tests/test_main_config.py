"""Config-level tests for app/__main__.py (B12/B13/B27 scenarios).

Covers: old (pre-TTS) vs new config files, stt_enabled/tts_enabled
combinations, and startup failing cleanly (not crashing with a traceback
into the s6 supervision tree) for an invalid TTS model.
"""

import json
from pathlib import Path
from unittest.mock import MagicMock, patch

from app.__main__ import (
    _load_engines,
    _load_json_config_overrides,
    build_arg_parser,
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
        assert engines.transcriber is not None
        assert engines.tts_synthesizer is None

    def test_tts_only_loads_synthesizer_not_transcriber(self):
        args = _parse(["--no-stt-enabled", "--tts-enabled"])
        with patch("app.__main__.load_tts_model") as mock_load_tts:
            mock_load_tts.return_value = MagicMock(has_voice_cloning=True, sample_rate=24000)
            engines = _load_engines(args)

        assert engines is not None
        assert engines.transcriber is None
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
            engines = _load_engines(args)

        assert engines is not None
        assert engines.transcriber is not None
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
