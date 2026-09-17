"""Guards against the CI fake Supervisor stub drifting from config.yaml.

The stub at .github/ci/fake_supervisor.py parses config.yaml's own
``options:`` block directly (see its module docstring) rather than
duplicating a hard-coded options dict, specifically so it cannot go stale
the way it did before v0.1.2 (some newer config.yaml options were missing
from the stub, so bashio::config resolved them to empty/null and the run
script forwarded that straight to argparse). This test still exists as a
second, independent safety net: it fails loudly if that parser ever stops
producing the exact same key set (and equivalent scalar types) config.yaml
itself declares under ``options:``.
"""

import importlib.util
import os
import sys
import types
from pathlib import Path

import pytest
import yaml

REPO_ROOT = Path(__file__).resolve().parents[3]
CONFIG_YAML_PATH = REPO_ROOT / "homeintent-moonshine-stt" / "config.yaml"
FAKE_SUPERVISOR_PATH = REPO_ROOT / ".github" / "ci" / "fake_supervisor.py"


def _load_fake_supervisor_module() -> types.ModuleType:
    spec = importlib.util.spec_from_file_location("fake_supervisor", FAKE_SUPERVISOR_PATH)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    # Loading the module at import time calls _load_options(), which reads
    # CONFIG_YAML_PATH (default /config.yaml) -- point it at the real file
    # instead before exec'ing the module body.
    sys.modules["fake_supervisor"] = module
    os.environ["CONFIG_YAML_PATH"] = str(CONFIG_YAML_PATH)
    spec.loader.exec_module(module)
    return module


@pytest.fixture(scope="module")
def real_options() -> dict[str, object]:
    with open(CONFIG_YAML_PATH, encoding="utf-8") as f:
        config: dict[str, object] = yaml.safe_load(f)
    options = config["options"]
    assert isinstance(options, dict)
    return options


@pytest.fixture(scope="module")
def parsed_options(real_options: dict[str, object]) -> dict[str, object]:
    module = _load_fake_supervisor_module()
    result: dict[str, object] = module.parse_options_block(
        CONFIG_YAML_PATH.read_text(encoding="utf-8")
    )
    return result


def test_parses_same_keys_as_config_yaml(real_options, parsed_options):
    assert set(parsed_options.keys()) == set(real_options.keys())


@pytest.mark.parametrize(
    "key",
    [
        "stt_enabled",
        "model",
        "language",
        "log_level",
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
    ],
)
def test_parses_same_value_as_config_yaml(key, real_options, parsed_options):
    assert parsed_options[key] == real_options[key]


def test_config_yaml_keys_match_schema_keys(real_options):
    with open(CONFIG_YAML_PATH, encoding="utf-8") as f:
        config = yaml.safe_load(f)
    assert set(config["options"].keys()) == set(config["schema"].keys())
