"""translations/de.yaml and translations/en.yaml must parse as valid YAML,
follow the real Home Assistant add-on translation structure
(``configuration: <option>: {name, description}``, verified against a real
published add-on's own translations/en.yaml while building this feature),
and cover every option in config.yaml's own ``schema`` block."""

from pathlib import Path

import pytest
import yaml

REPO_ROOT = Path(__file__).resolve().parents[3]
ADDON_DIR = REPO_ROOT / "homeintent-moonshine-stt"
CONFIG_YAML_PATH = ADDON_DIR / "config.yaml"
TRANSLATIONS_DIR = ADDON_DIR / "translations"


def _load_config_schema_keys() -> set[str]:
    with open(CONFIG_YAML_PATH, encoding="utf-8") as f:
        config = yaml.safe_load(f)
    return set(config["schema"].keys())


@pytest.mark.parametrize("filename", ["en.yaml", "de.yaml"])
def test_translation_file_parses_as_valid_yaml(filename: str) -> None:
    path = TRANSLATIONS_DIR / filename
    assert path.is_file(), f"missing {path}"
    with open(path, encoding="utf-8") as f:
        data = yaml.safe_load(f)
    assert isinstance(data, dict)
    assert "configuration" in data
    assert isinstance(data["configuration"], dict)


@pytest.mark.parametrize("filename", ["en.yaml", "de.yaml"])
def test_translation_covers_every_schema_option(filename: str) -> None:
    schema_keys = _load_config_schema_keys()
    with open(TRANSLATIONS_DIR / filename, encoding="utf-8") as f:
        data = yaml.safe_load(f)
    translated_keys = set(data["configuration"].keys())
    missing = schema_keys - translated_keys
    assert not missing, f"{filename} is missing translations for: {sorted(missing)}"


@pytest.mark.parametrize("filename", ["en.yaml", "de.yaml"])
def test_every_translated_option_has_name_and_description(filename: str) -> None:
    with open(TRANSLATIONS_DIR / filename, encoding="utf-8") as f:
        data = yaml.safe_load(f)
    for key, entry in data["configuration"].items():
        assert isinstance(entry, dict), f"{filename}:{key} is not a mapping"
        assert entry.get("name"), f"{filename}:{key} missing a non-empty 'name'"
        assert entry.get("description"), f"{filename}:{key} missing a non-empty 'description'"


def test_translation_does_not_translate_unknown_options() -> None:
    """Every translated key must correspond to a real schema option --
    otherwise a renamed/removed option would leave stale translations."""
    schema_keys = _load_config_schema_keys()
    for filename in ("en.yaml", "de.yaml"):
        with open(TRANSLATIONS_DIR / filename, encoding="utf-8") as f:
            data = yaml.safe_load(f)
        extra = set(data["configuration"].keys()) - schema_keys
        assert not extra, f"{filename} translates unknown option(s): {sorted(extra)}"
