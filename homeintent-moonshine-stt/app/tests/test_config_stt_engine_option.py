"""Config-level tests for the new stt_engine/tts_engine=supertonic_3/kroko_*/
supertonic_* options: default value, the old-config upgrade path, and CLI
argument parsing/validation wiring."""

from pathlib import Path
from typing import Any

import pytest
import yaml

from app.__main__ import _load_json_config_overrides, _validate_args, build_arg_parser

REPO_ROOT = Path(__file__).resolve().parents[3]
CONFIG_YAML_PATH = REPO_ROOT / "homeintent-moonshine-stt" / "config.yaml"


def _load_config_yaml() -> dict[str, Any]:
    with open(CONFIG_YAML_PATH, encoding="utf-8") as f:
        result: dict[str, Any] = yaml.safe_load(f)
        return result


def test_stt_engine_default_is_moonshine_in_config_yaml() -> None:
    config = _load_config_yaml()
    assert config["options"]["stt_engine"] == "moonshine"
    assert config["schema"]["stt_engine"] == "list(moonshine|kroko|speechcatcher_m|speechcatcher_l)"


def test_tts_engine_schema_includes_supertonic_3() -> None:
    config = _load_config_yaml()
    assert config["schema"]["tts_engine"] == "list(pocket_tts|kokoro_onnx|supertonic_3)"
    # Default must stay pocket_tts -- upgrade compatibility for existing
    # Pocket TTS installations.
    assert config["options"]["tts_engine"] == "pocket_tts"


def test_cli_default_stt_engine_is_moonshine() -> None:
    parser = build_arg_parser()
    args = parser.parse_args([])
    assert args.stt_engine == "moonshine"


def test_old_config_without_stt_engine_key_falls_back_to_moonshine(tmp_path: Path) -> None:
    """Simulates an add-on upgrade: a persisted config JSON that predates
    the stt_engine option entirely must still resolve to "moonshine",
    exactly like rootfs's run script's own config_or_default() fallback."""
    import json

    config_file = tmp_path / "options.json"
    config_file.write_text(json.dumps({"stt_enabled": True, "model": "small"}))

    parser = build_arg_parser()
    args = parser.parse_args(["--config", str(config_file)])
    assert _load_json_config_overrides(args) is True
    assert args.stt_engine == "moonshine"  # never overwritten by the missing key


def test_old_config_without_supertonic_options_falls_back_to_defaults(tmp_path: Path) -> None:
    import json

    config_file = tmp_path / "options.json"
    config_file.write_text(json.dumps({"tts_enabled": True, "tts_engine": "pocket_tts"}))

    parser = build_arg_parser()
    args = parser.parse_args(["--config", str(config_file)])
    assert _load_json_config_overrides(args) is True
    assert args.supertonic_voice == "M1"
    assert args.supertonic_steps == 8


def test_validate_args_rejects_unknown_stt_engine() -> None:
    parser = build_arg_parser()
    args = parser.parse_args(["--stt-engine", "moonshine"])
    args.stt_engine = "not_a_real_engine"  # bypass argparse's own choices=
    assert _validate_args(args) is False


@pytest.mark.parametrize("engine", ["moonshine", "kroko", "speechcatcher_m", "speechcatcher_l"])
def test_validate_args_accepts_known_stt_engines(engine: str) -> None:
    parser = build_arg_parser()
    args = parser.parse_args(["--stt-engine", engine])
    assert _validate_args(args) is True


def test_old_config_without_speechcatcher_options_falls_back_to_defaults(tmp_path: Path) -> None:
    """Simulates an add-on upgrade: a persisted config JSON that predates
    the speechcatcher_threads/speechcatcher_beam_size options entirely must
    still resolve to their config.yaml defaults."""
    import json

    config_file = tmp_path / "options.json"
    config_file.write_text(json.dumps({"stt_enabled": True, "stt_engine": "moonshine"}))

    parser = build_arg_parser()
    args = parser.parse_args(["--config", str(config_file)])
    assert _load_json_config_overrides(args) is True
    assert args.speechcatcher_threads == 0
    assert args.speechcatcher_beam_size == 5
