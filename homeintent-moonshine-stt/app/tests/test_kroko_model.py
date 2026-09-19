"""Tests for app/kroko_model.py: cached-file discovery and download failure
handling. Never downloads a real model -- either pre-populates a fake cache
directory, or monkeypatches urlretrieve to fail fast.

Also covers (v0.6.2) the extraction-path fix for the real production bug
where Debian bookworm's apt-installed python3.11 (what the real add-on
Docker image runs on) lacks the `filter=` keyword argument to
`TarFile.extractall()` that this repo's own CI Python has -- see
kroko_model.py's module-level `_EXTRACTALL_SUPPORTS_FILTER` docstring for
the full root-cause writeup. These tests exercise both the modern
`filter="data"` path and the safe legacy fallback directly (via
`_safe_extractall_legacy`/monkeypatching the feature-detection flag), so
they pass regardless of which Python this test suite itself happens to run
under.
"""

import io
import tarfile
import urllib.request
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


def test_modern_path_used_when_extractall_supports_filter(tmp_path: Path) -> None:
    """Confirms `filter="data"` is actually used when the runtime supports
    it (this sandbox's own Python does -- see _EXTRACTALL_SUPPORTS_FILTER's
    docstring for exactly which real production Python does not)."""
    archive_path = tmp_path / "kroko.tar.bz2"
    _build_valid_kroko_tar(archive_path)
    dest = tmp_path / "extracted"
    dest.mkdir()

    with tarfile.open(archive_path) as tf:
        assert kroko_model._EXTRACTALL_SUPPORTS_FILTER, (
            "this test's own Python is expected to support filter=; "
            "if this fails, the sandbox Python changed -- see module docstring"
        )
        tf.extractall(dest, filter="data")

    assert (dest / "model" / "tokens.txt").read_bytes() == b"<blk> 0\n"
    assert (dest / "model" / "encoder.int8.onnx").is_file()


def test_safe_extractall_legacy_extracts_valid_archive(tmp_path: Path) -> None:
    archive_path = tmp_path / "kroko.tar.bz2"
    _build_valid_kroko_tar(archive_path)
    dest = tmp_path / "extracted"

    with tarfile.open(archive_path) as tf:
        kroko_model._safe_extractall_legacy(tf, dest)

    assert (dest / "model" / "tokens.txt").read_bytes() == b"<blk> 0\n"
    assert (dest / "model" / "encoder.int8.onnx").read_bytes() == b"fake-encoder"
    assert (dest / "model" / "decoder.onnx").is_file()
    assert (dest / "model" / "joiner.int8.onnx").is_file()


def test_download_and_extract_uses_legacy_path_when_filter_unsupported(
    tmp_path: Path, monkeypatch
) -> None:
    """Forces the legacy code path via the feature-detection flag
    (dependency injection), independent of this sandbox's real Python
    version -- exercises the full _download_and_extract flow, not just
    _safe_extractall_legacy in isolation."""
    archive_path = tmp_path / "server" / "kroko.tar.bz2"
    archive_path.parent.mkdir()
    _build_valid_kroko_tar(archive_path)

    def _fake_urlretrieve(url: str, filename: Path, *a: object, **k: object) -> None:
        Path(filename).write_bytes(archive_path.read_bytes())

    monkeypatch.setattr(kroko_model, "_EXTRACTALL_SUPPORTS_FILTER", False)
    monkeypatch.setattr(urllib.request, "urlretrieve", _fake_urlretrieve)

    dest_dir = tmp_path / "cache" / "pkg"
    kroko_model._download_and_extract("https://example.invalid/kroko.tar.bz2", dest_dir)

    assert (dest_dir / "model" / "tokens.txt").is_file()
    assert (dest_dir / "model" / "encoder.int8.onnx").is_file()


def _archive_with_single_bad_member(archive_path: Path, member: tarfile.TarInfo) -> None:
    with tarfile.open(archive_path, "w:bz2") as tf:
        good = tarfile.TarInfo(name="model/tokens.txt")
        content = b"<blk> 0\n"
        good.size = len(content)
        good.type = tarfile.REGTYPE
        tf.addfile(good, fileobj=io.BytesIO(content))
        tf.addfile(member)


@pytest.mark.parametrize(
    "make_member",
    [
        pytest.param(lambda: tarfile.TarInfo(name="../../evil.txt"), id="path-traversal-relative"),
        pytest.param(lambda: tarfile.TarInfo(name="/tmp/evil.txt"), id="absolute-path"),
    ],
)
def test_safe_extractall_legacy_rejects_path_escapes(tmp_path: Path, make_member) -> None:
    member = make_member()
    member.size = 4
    member.type = tarfile.REGTYPE
    archive_path = tmp_path / "evil.tar.bz2"
    with tarfile.open(archive_path, "w:bz2") as tf:
        tf.addfile(member, fileobj=io.BytesIO(b"evil"))

    dest = tmp_path / "dest"
    with tarfile.open(archive_path) as tf, pytest.raises(kroko_model.KrokoModelDownloadError):
        kroko_model._safe_extractall_legacy(tf, dest)

    # Nothing must be written outside (or, given nothing valid preceded the
    # bad member, inside) the intended destination.
    assert not (tmp_path / "evil.txt").exists()
    assert not (tmp_path.parent / "evil.txt").exists()
    assert list(dest.rglob("*")) == [] if dest.exists() else True


def test_safe_extractall_legacy_rejects_symlink_escape(tmp_path: Path) -> None:
    archive_path = tmp_path / "evil.tar.bz2"
    link = tarfile.TarInfo(name="escape-link")
    link.type = tarfile.SYMTYPE
    link.linkname = "../../../etc/passwd"
    _archive_with_single_bad_member(archive_path, link)

    dest = tmp_path / "dest"
    with tarfile.open(archive_path) as tf, pytest.raises(kroko_model.KrokoModelDownloadError):
        kroko_model._safe_extractall_legacy(tf, dest)

    assert not (dest / "model").exists()


def test_safe_extractall_legacy_rejects_hardlink_escape(tmp_path: Path) -> None:
    outside = tmp_path / "outside.txt"
    outside.write_text("secret")
    archive_path = tmp_path / "evil.tar.bz2"
    link = tarfile.TarInfo(name="hardlink-escape")
    link.type = tarfile.LNKTYPE
    link.linkname = str(outside)
    _archive_with_single_bad_member(archive_path, link)

    dest = tmp_path / "dest"
    with tarfile.open(archive_path) as tf, pytest.raises(kroko_model.KrokoModelDownloadError):
        kroko_model._safe_extractall_legacy(tf, dest)

    assert not (dest / "model").exists()


def test_safe_extractall_legacy_rejects_fifo(tmp_path: Path) -> None:
    archive_path = tmp_path / "evil.tar.bz2"
    fifo = tarfile.TarInfo(name="evil-fifo")
    fifo.type = tarfile.FIFOTYPE
    _archive_with_single_bad_member(archive_path, fifo)

    dest = tmp_path / "dest"
    with tarfile.open(archive_path) as tf, pytest.raises(kroko_model.KrokoModelDownloadError):
        kroko_model._safe_extractall_legacy(tf, dest)

    assert not (dest / "model").exists()


@pytest.mark.parametrize(
    "member_type",
    [
        pytest.param(tarfile.CHRTYPE, id="char-device"),
        pytest.param(tarfile.BLKTYPE, id="block-device"),
    ],
)
def test_safe_extractall_legacy_rejects_device_files(tmp_path: Path, member_type: bytes) -> None:
    archive_path = tmp_path / "evil.tar.bz2"
    dev = tarfile.TarInfo(name="evil-dev")
    dev.type = member_type
    dev.devmajor = 1
    dev.devminor = 1
    _archive_with_single_bad_member(archive_path, dev)

    dest = tmp_path / "dest"
    with tarfile.open(archive_path) as tf, pytest.raises(kroko_model.KrokoModelDownloadError):
        kroko_model._safe_extractall_legacy(tf, dest)

    assert not (dest / "model").exists()
