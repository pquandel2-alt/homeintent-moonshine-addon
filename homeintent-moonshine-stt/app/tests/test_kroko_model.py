"""Tests for app/kroko_model.py: cached-file discovery and download failure
handling. Never downloads a real model -- either pre-populates a fake cache
directory, or monkeypatches urlretrieve to fail fast."""

from pathlib import Path

import pytest

from app import kroko_model


def _make_fake_package(base: Path, *, with_int8: bool = True) -> Path:
    package_dir = base / kroko_model.KROKO_PACKAGE_NAME
    package_dir.mkdir(parents=True)
    (package_dir / "tokens.txt").write_text("<blk> 0\n")
    if with_int8:
        (package_dir / "encoder-epoch-99-avg-1.int8.onnx").write_bytes(b"fake")
        (package_dir / "encoder-epoch-99-avg-1.onnx").write_bytes(b"fake-fp32")
    else:
        (package_dir / "encoder-epoch-99-avg-1.onnx").write_bytes(b"fake")
    (package_dir / "decoder-epoch-99-avg-1.onnx").write_bytes(b"fake")
    (package_dir / "joiner-epoch-99-avg-1.onnx").write_bytes(b"fake")
    return package_dir


def test_resolve_from_pre_populated_cache_never_downloads(tmp_path: Path, monkeypatch) -> None:
    _make_fake_package(tmp_path)

    def _fail(*args: object, **kwargs: object) -> None:
        raise AssertionError("must not attempt a network download when the cache is populated")

    monkeypatch.setattr(kroko_model, "_download_and_extract", _fail)

    files = kroko_model.resolve_kroko_model_files(cache_dir=tmp_path)
    assert files.tokens.name == "tokens.txt"
    assert files.encoder.name == "encoder-epoch-99-avg-1.int8.onnx"  # int8 preferred
    assert files.decoder.is_file()
    assert files.joiner.is_file()


def test_resolve_prefers_fp32_when_no_int8_present(tmp_path: Path, monkeypatch) -> None:
    _make_fake_package(tmp_path, with_int8=False)
    monkeypatch.setattr(
        kroko_model,
        "_download_and_extract",
        lambda *a, **k: (_ for _ in ()).throw(AssertionError("should not download")),
    )
    files = kroko_model.resolve_kroko_model_files(cache_dir=tmp_path)
    assert files.encoder.name == "encoder-epoch-99-avg-1.onnx"


def test_missing_encoder_raises_download_error(tmp_path: Path, monkeypatch) -> None:
    package_dir = tmp_path / kroko_model.KROKO_PACKAGE_NAME
    package_dir.mkdir(parents=True)
    (package_dir / "tokens.txt").write_text("<blk> 0\n")
    (package_dir / "decoder-epoch-99-avg-1.onnx").write_bytes(b"fake")
    (package_dir / "joiner-epoch-99-avg-1.onnx").write_bytes(b"fake")

    with pytest.raises(kroko_model.KrokoModelDownloadError):
        kroko_model.resolve_kroko_model_files(cache_dir=tmp_path)


def test_download_failure_raises_kroko_model_download_error(tmp_path: Path, monkeypatch) -> None:
    def _raise(*args: object, **kwargs: object) -> None:
        raise OSError("network unreachable")

    import urllib.request

    monkeypatch.setattr(urllib.request, "urlretrieve", _raise)

    with pytest.raises(kroko_model.KrokoModelDownloadError):
        kroko_model.resolve_kroko_model_files(cache_dir=tmp_path / "empty")
