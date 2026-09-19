"""Tests for app/supertonic_tts.py: voice/sid mapping table and model file
resolution/download-failure handling (no real network access)."""

from pathlib import Path

import pytest

from app import supertonic_tts


def test_voice_table_has_10_names_matching_upstream_f_and_m_series() -> None:
    assert supertonic_tts.SUPERTONIC_VOICES == {
        0: "F1",
        1: "F2",
        2: "F3",
        3: "F4",
        4: "F5",
        5: "M1",
        6: "M2",
        7: "M3",
        8: "M4",
        9: "M5",
    }


def test_default_voice_is_m1_a_male_voice() -> None:
    assert supertonic_tts.DEFAULT_SUPERTONIC_VOICE == "M1"
    assert supertonic_tts.resolve_supertonic_sid(supertonic_tts.DEFAULT_SUPERTONIC_VOICE) == 5


@pytest.mark.parametrize(
    "name,expected_sid",
    [("F1", 0), ("f1", 0), ("M5", 9), ("m3", 7)],
)
def test_resolve_supertonic_sid_by_name_case_insensitive(name: str, expected_sid: int) -> None:
    assert supertonic_tts.resolve_supertonic_sid(name) == expected_sid


def test_resolve_supertonic_sid_accepts_raw_numeric_sid() -> None:
    assert supertonic_tts.resolve_supertonic_sid("6") == 6


def test_resolve_supertonic_sid_rejects_out_of_range_numeric_sid() -> None:
    with pytest.raises(ValueError):
        supertonic_tts.resolve_supertonic_sid("10")


def test_resolve_supertonic_sid_rejects_unknown_name() -> None:
    with pytest.raises(ValueError):
        supertonic_tts.resolve_supertonic_sid("Q9")


def _make_fake_package(base: Path) -> Path:
    package_dir = base / supertonic_tts.SUPERTONIC_PACKAGE_NAME
    package_dir.mkdir(parents=True)
    for name in supertonic_tts._REQUIRED_FILES:
        (package_dir / name).write_bytes(b"fake")
    return package_dir


def test_resolve_from_pre_populated_cache_never_downloads(tmp_path: Path, monkeypatch) -> None:
    _make_fake_package(tmp_path)

    def _fail(*args: object, **kwargs: object) -> None:
        raise AssertionError("must not attempt a network download when the cache is populated")

    monkeypatch.setattr(supertonic_tts, "_download_and_extract", _fail)

    files = supertonic_tts.resolve_supertonic_model_files(cache_dir=tmp_path)
    assert set(files.keys()) == set(supertonic_tts._REQUIRED_FILES)
    for path in files.values():
        assert path.is_file()


def test_missing_required_file_raises_download_error(tmp_path: Path) -> None:
    package_dir = tmp_path / supertonic_tts.SUPERTONIC_PACKAGE_NAME
    package_dir.mkdir(parents=True)
    (package_dir / "tts.json").write_bytes(b"{}")
    # deliberately omit the other required files

    with pytest.raises(supertonic_tts.SupertonicModelDownloadError):
        supertonic_tts.resolve_supertonic_model_files(cache_dir=tmp_path)


def test_download_failure_raises_supertonic_model_download_error(
    tmp_path: Path, monkeypatch
) -> None:
    def _raise(*args: object, **kwargs: object) -> None:
        raise OSError("network unreachable")

    import urllib.request

    monkeypatch.setattr(urllib.request, "urlretrieve", _raise)

    with pytest.raises(supertonic_tts.SupertonicModelDownloadError):
        supertonic_tts.resolve_supertonic_model_files(cache_dir=tmp_path / "empty")
