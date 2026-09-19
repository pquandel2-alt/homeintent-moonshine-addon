"""Tests for app/kroko_model.py: cached-file discovery and download failure
handling. Never downloads a real model -- either pre-populates a fake cache
directory, or monkeypatches urlretrieve to fail fast.

The actual tar-extraction safety logic (modern `filter="data"` path, legacy
safe-extraction fallback, and all the security rejection tests) lives in
app/safe_tar_extract.py and is tested once, in test_safe_tar_extract.py --
see that module's docstring for the v0.6.2 root-cause writeup (Debian
bookworm's apt-installed python3.11 lacking `TarFile.extractall()`'s
`filter=` keyword argument). This file only carries a thin integration test
confirming `_download_and_extract` actually goes through that shared
extractor.
"""

import io
import tarfile
import urllib.request
from pathlib import Path

import pytest

from app import kroko_model, safe_tar_extract


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

    monkeypatch.setattr(urllib.request, "urlretrieve", _raise)

    with pytest.raises(kroko_model.KrokoModelDownloadError):
        kroko_model.resolve_kroko_model_files(cache_dir=tmp_path / "empty")


def _build_valid_kroko_tar(archive_path: Path) -> None:
    """Build a small, realistic-shaped, valid Kroko archive fixture."""
    with tarfile.open(archive_path, "w:bz2") as tf:
        files = {
            "model/encoder.int8.onnx": b"fake-encoder",
            "model/decoder.onnx": b"fake-decoder",
            "model/joiner.int8.onnx": b"fake-joiner",
            "model/tokens.txt": b"<blk> 0\n",
        }
        for name, content in files.items():
            info = tarfile.TarInfo(name=name)
            info.size = len(content)
            info.type = tarfile.REGTYPE
            tf.addfile(info, fileobj=io.BytesIO(content))


def test_download_and_extract_uses_the_shared_safe_extractor(tmp_path: Path, monkeypatch) -> None:
    """Integration check that `_download_and_extract` actually calls
    through to `app.safe_tar_extract.safe_extractall` (the security-critical
    validation logic itself -- modern path, legacy fallback, and all the
    rejection cases -- is tested once in test_safe_tar_extract.py, not
    re-tested here)."""
    archive_path = tmp_path / "server" / "kroko.tar.bz2"
    archive_path.parent.mkdir()
    _build_valid_kroko_tar(archive_path)

    def _fake_urlretrieve(url: str, filename: Path, *a: object, **k: object) -> None:
        Path(filename).write_bytes(archive_path.read_bytes())

    monkeypatch.setattr(urllib.request, "urlretrieve", _fake_urlretrieve)

    calls: list[str] = []
    real_safe_extractall = safe_tar_extract.safe_extractall

    def _tracking_safe_extractall(tf: tarfile.TarFile, destination: Path, *, label: str) -> None:
        calls.append(label)
        real_safe_extractall(tf, destination, label=label)

    monkeypatch.setattr(kroko_model, "safe_extractall", _tracking_safe_extractall)

    dest_dir = tmp_path / "cache" / "pkg"
    kroko_model._download_and_extract("https://example.invalid/kroko.tar.bz2", dest_dir)

    assert calls == ["Kroko model"]
    assert (dest_dir / "model" / "tokens.txt").is_file()
    assert (dest_dir / "model" / "encoder.int8.onnx").is_file()


def test_download_and_extract_wraps_unsafe_member_as_kroko_download_error(
    tmp_path: Path, monkeypatch
) -> None:
    """Confirms the shared extractor's `UnsafeTarMemberError` is translated
    into Kroko's own domain exception, not left as a generic error."""
    archive_path = tmp_path / "server" / "evil.tar.bz2"
    archive_path.parent.mkdir()
    with tarfile.open(archive_path, "w:bz2") as tf:
        member = tarfile.TarInfo(name="/tmp/evil.txt")
        member.size = 4
        member.type = tarfile.REGTYPE
        tf.addfile(member, fileobj=io.BytesIO(b"evil"))

    def _fake_urlretrieve(url: str, filename: Path, *a: object, **k: object) -> None:
        Path(filename).write_bytes(archive_path.read_bytes())

    monkeypatch.setattr(urllib.request, "urlretrieve", _fake_urlretrieve)
    # Force the legacy path: the modern filter="data" path already rejects
    # this member itself (via a different exception type), which this test
    # isn't about -- see test_safe_tar_extract.py for coverage of both.
    monkeypatch.setattr(safe_tar_extract, "EXTRACTALL_SUPPORTS_FILTER", False)

    dest_dir = tmp_path / "cache" / "pkg"
    with pytest.raises(kroko_model.KrokoModelDownloadError, match="unsafe Kroko model archive"):
        kroko_model._download_and_extract("https://example.invalid/evil.tar.bz2", dest_dir)
