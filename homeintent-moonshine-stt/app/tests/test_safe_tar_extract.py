"""Tests for app/safe_tar_extract.py: the shared safe tar-archive extraction
logic used by both app/kroko_model.py (Kroko ASR) and app/supertonic_tts.py
(Supertonic TTS).

Covers (v0.6.2) the extraction-path fix for the real production bug where
Debian bookworm's apt-installed python3.11 (what the real add-on Docker
image runs on) lacks the `filter=` keyword argument to
`TarFile.extractall()` that this repo's own CI Python has -- see this
module's `EXTRACTALL_SUPPORTS_FILTER` docstring for the full root-cause
writeup. These tests exercise both the modern `filter="data"` path and the
safe legacy fallback directly (via `_safe_extractall_legacy`/monkeypatching
the feature-detection flag), so they pass regardless of which Python this
test suite itself happens to run under.

This security-sensitive validation logic is tested here ONCE; per-engine
test files (test_kroko_model.py, test_supertonic_tts.py) only carry a thin
integration test confirming their own download/extract helper actually
calls into this shared module, not a re-test of the rejection logic itself.
"""

import io
import tarfile

import pytest

from app import safe_tar_extract


def _build_valid_tar(archive_path) -> None:
    """Build a small, realistic-shaped, valid model archive fixture."""
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


def test_modern_path_used_when_extractall_supports_filter(tmp_path) -> None:
    """Confirms `filter="data"` is actually used when the runtime supports
    it (this sandbox's own Python does -- see EXTRACTALL_SUPPORTS_FILTER's
    docstring for exactly which real production Python does not)."""
    archive_path = tmp_path / "archive.tar.bz2"
    _build_valid_tar(archive_path)
    dest = tmp_path / "extracted"
    dest.mkdir()

    assert safe_tar_extract.EXTRACTALL_SUPPORTS_FILTER, (
        "this test's own Python is expected to support filter=; "
        "if this fails, the sandbox Python changed -- see module docstring"
    )
    with tarfile.open(archive_path) as tf:
        safe_tar_extract.safe_extractall(tf, dest, label="Test model")

    assert (dest / "model" / "tokens.txt").read_bytes() == b"<blk> 0\n"
    assert (dest / "model" / "encoder.int8.onnx").is_file()


def test_safe_extractall_legacy_extracts_valid_archive(tmp_path) -> None:
    archive_path = tmp_path / "archive.tar.bz2"
    _build_valid_tar(archive_path)
    dest = tmp_path / "extracted"

    with tarfile.open(archive_path) as tf:
        safe_tar_extract._safe_extractall_legacy(tf, dest)

    assert (dest / "model" / "tokens.txt").read_bytes() == b"<blk> 0\n"
    assert (dest / "model" / "encoder.int8.onnx").read_bytes() == b"fake-encoder"
    assert (dest / "model" / "decoder.onnx").is_file()
    assert (dest / "model" / "joiner.int8.onnx").is_file()


def test_safe_extractall_uses_legacy_path_when_filter_unsupported(tmp_path, monkeypatch) -> None:
    """Forces the legacy code path via the feature-detection flag
    (dependency injection), independent of this sandbox's real Python
    version -- exercises the full safe_extractall() entry point, not just
    _safe_extractall_legacy in isolation."""
    archive_path = tmp_path / "archive.tar.bz2"
    _build_valid_tar(archive_path)
    dest = tmp_path / "extracted"

    monkeypatch.setattr(safe_tar_extract, "EXTRACTALL_SUPPORTS_FILTER", False)

    with tarfile.open(archive_path) as tf:
        safe_tar_extract.safe_extractall(tf, dest, label="Test model")

    assert (dest / "model" / "tokens.txt").is_file()
    assert (dest / "model" / "encoder.int8.onnx").is_file()


def _archive_with_single_bad_member(archive_path, member: tarfile.TarInfo) -> None:
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
def test_safe_extractall_legacy_rejects_path_escapes(tmp_path, make_member) -> None:
    member = make_member()
    member.size = 4
    member.type = tarfile.REGTYPE
    archive_path = tmp_path / "evil.tar.bz2"
    with tarfile.open(archive_path, "w:bz2") as tf:
        tf.addfile(member, fileobj=io.BytesIO(b"evil"))

    dest = tmp_path / "dest"
    with (
        tarfile.open(archive_path) as tf,
        pytest.raises(safe_tar_extract.UnsafeTarMemberError),
    ):
        safe_tar_extract._safe_extractall_legacy(tf, dest)

    # Nothing must be written outside (or, given nothing valid preceded the
    # bad member, inside) the intended destination.
    assert not (tmp_path / "evil.txt").exists()
    assert not (tmp_path.parent / "evil.txt").exists()
    assert list(dest.rglob("*")) == [] if dest.exists() else True


def test_safe_extractall_legacy_rejects_symlink_escape(tmp_path) -> None:
    archive_path = tmp_path / "evil.tar.bz2"
    link = tarfile.TarInfo(name="escape-link")
    link.type = tarfile.SYMTYPE
    link.linkname = "../../../etc/passwd"
    _archive_with_single_bad_member(archive_path, link)

    dest = tmp_path / "dest"
    with (
        tarfile.open(archive_path) as tf,
        pytest.raises(safe_tar_extract.UnsafeTarMemberError),
    ):
        safe_tar_extract._safe_extractall_legacy(tf, dest)

    assert not (dest / "model").exists()


def test_safe_extractall_legacy_rejects_hardlink_escape(tmp_path) -> None:
    outside = tmp_path / "outside.txt"
    outside.write_text("secret")
    archive_path = tmp_path / "evil.tar.bz2"
    link = tarfile.TarInfo(name="hardlink-escape")
    link.type = tarfile.LNKTYPE
    link.linkname = str(outside)
    _archive_with_single_bad_member(archive_path, link)

    dest = tmp_path / "dest"
    with (
        tarfile.open(archive_path) as tf,
        pytest.raises(safe_tar_extract.UnsafeTarMemberError),
    ):
        safe_tar_extract._safe_extractall_legacy(tf, dest)

    assert not (dest / "model").exists()


def test_safe_extractall_legacy_rejects_fifo(tmp_path) -> None:
    archive_path = tmp_path / "evil.tar.bz2"
    fifo = tarfile.TarInfo(name="evil-fifo")
    fifo.type = tarfile.FIFOTYPE
    _archive_with_single_bad_member(archive_path, fifo)

    dest = tmp_path / "dest"
    with (
        tarfile.open(archive_path) as tf,
        pytest.raises(safe_tar_extract.UnsafeTarMemberError),
    ):
        safe_tar_extract._safe_extractall_legacy(tf, dest)

    assert not (dest / "model").exists()


@pytest.mark.parametrize(
    "member_type",
    [
        pytest.param(tarfile.CHRTYPE, id="char-device"),
        pytest.param(tarfile.BLKTYPE, id="block-device"),
    ],
)
def test_safe_extractall_legacy_rejects_device_files(tmp_path, member_type: bytes) -> None:
    archive_path = tmp_path / "evil.tar.bz2"
    dev = tarfile.TarInfo(name="evil-dev")
    dev.type = member_type
    dev.devmajor = 1
    dev.devminor = 1
    _archive_with_single_bad_member(archive_path, dev)

    dest = tmp_path / "dest"
    with (
        tarfile.open(archive_path) as tf,
        pytest.raises(safe_tar_extract.UnsafeTarMemberError),
    ):
        safe_tar_extract._safe_extractall_legacy(tf, dest)

    assert not (dest / "model").exists()
