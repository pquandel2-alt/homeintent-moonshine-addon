"""Tests for app/vosk_model.py: cached-directory discovery and download
failure handling. Never downloads a real model -- either pre-populates a
fake cache directory, or monkeypatches urlretrieve to fail fast."""

from pathlib import Path

import pytest

from app import vosk_model


def _make_fake_model_dir(base: Path, *, nested: bool = True) -> Path:
    package_dir = base / vosk_model.VOSK_MODEL_NAME
    target = package_dir / vosk_model.VOSK_MODEL_NAME if nested else package_dir
    target.mkdir(parents=True)
    (target / "conf").mkdir()
    (target / "conf" / "mfcc.conf").write_text("--num-mel-bins=40\n")
    (target / "am").mkdir()
    (target / "graph").mkdir()
    return package_dir


def test_resolve_from_pre_populated_flat_cache_never_downloads(tmp_path: Path, monkeypatch) -> None:
    package_dir = _make_fake_model_dir(tmp_path, nested=False)

    def _fail(*args: object, **kwargs: object) -> None:
        raise AssertionError("must not attempt a network download when the cache is populated")

    monkeypatch.setattr(vosk_model, "_download_and_extract", _fail)

    resolved = vosk_model.resolve_vosk_model_dir(cache_dir=tmp_path)
    assert resolved == package_dir
    assert (resolved / "conf").is_dir()


def test_resolve_from_pre_populated_nested_cache_never_downloads(
    tmp_path: Path, monkeypatch
) -> None:
    package_dir = _make_fake_model_dir(tmp_path, nested=True)

    def _fail(*args: object, **kwargs: object) -> None:
        raise AssertionError("must not attempt a network download when the cache is populated")

    monkeypatch.setattr(vosk_model, "_download_and_extract", _fail)

    resolved = vosk_model.resolve_vosk_model_dir(cache_dir=tmp_path)
    assert resolved == package_dir / vosk_model.VOSK_MODEL_NAME
    assert (resolved / "conf").is_dir()


def test_missing_conf_dir_raises_download_error_after_download(tmp_path: Path, monkeypatch) -> None:
    def _fake_download(url: str, dest_dir: Path) -> None:
        dest_dir.mkdir(parents=True)
        (dest_dir / "some_other_file.txt").write_text("not a model")

    monkeypatch.setattr(vosk_model, "_download_and_extract", _fake_download)

    with pytest.raises(vosk_model.VoskModelDownloadError):
        vosk_model.resolve_vosk_model_dir(cache_dir=tmp_path)


def test_download_failure_raises_vosk_model_download_error(tmp_path: Path, monkeypatch) -> None:
    def _raise(*args: object, **kwargs: object) -> None:
        raise OSError("network unreachable")

    import urllib.request

    monkeypatch.setattr(urllib.request, "urlretrieve", _raise)

    with pytest.raises(vosk_model.VoskModelDownloadError):
        vosk_model.resolve_vosk_model_dir(cache_dir=tmp_path / "empty")


def test_download_url_matches_vosk_own_pre_url_convention() -> None:
    assert vosk_model.VOSK_DOWNLOAD_URL == (
        "https://alphacephei.com/vosk/models/vosk-model-small-de-0.15.zip"
    )
