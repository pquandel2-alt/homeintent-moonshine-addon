"""Tests for app/supertonic_tts.py: voice/sid mapping table and model file
resolution/download-failure handling (no real network access).

The tar-extraction safety logic itself (modern `filter="data"` path, legacy
safe-extraction fallback, and all security rejection tests) is shared with
app/kroko_model.py via app/safe_tar_extract.py and tested once in
test_safe_tar_extract.py -- this file only carries a thin integration test
confirming Supertonic's own `_download_and_extract` goes through that
shared extractor.
"""

import io
import tarfile
import urllib.request
from pathlib import Path

import pytest

from app import safe_tar_extract, supertonic_tts


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

    monkeypatch.setattr(urllib.request, "urlretrieve", _raise)

    with pytest.raises(supertonic_tts.SupertonicModelDownloadError):
        supertonic_tts.resolve_supertonic_model_files(cache_dir=tmp_path / "empty")


def _build_valid_supertonic_tar(archive_path: Path) -> None:
    with tarfile.open(archive_path, "w:bz2") as tf:
        for name in supertonic_tts._REQUIRED_FILES:
            content = b"fake"
            info = tarfile.TarInfo(name=name)
            info.size = len(content)
            info.type = tarfile.REGTYPE
            tf.addfile(info, fileobj=io.BytesIO(content))


def test_download_and_extract_uses_the_shared_safe_extractor(tmp_path: Path, monkeypatch) -> None:
    """Integration check that Supertonic's `_download_and_extract` actually
    calls through to `app.safe_tar_extract.safe_extractall` (the
    security-critical validation logic itself is tested once, against the
    shared module, in test_safe_tar_extract.py)."""
    archive_path = tmp_path / "server" / "supertonic.tar.bz2"
    archive_path.parent.mkdir()
    _build_valid_supertonic_tar(archive_path)

    def _fake_urlretrieve(url: str, filename: Path, *a: object, **k: object) -> None:
        Path(filename).write_bytes(archive_path.read_bytes())

    monkeypatch.setattr(urllib.request, "urlretrieve", _fake_urlretrieve)

    calls: list[str] = []
    real_safe_extractall = safe_tar_extract.safe_extractall

    def _tracking_safe_extractall(tf: tarfile.TarFile, destination: Path, *, label: str) -> None:
        calls.append(label)
        real_safe_extractall(tf, destination, label=label)

    monkeypatch.setattr(supertonic_tts, "safe_extractall", _tracking_safe_extractall)

    dest_dir = tmp_path / "cache" / "pkg"
    supertonic_tts._download_and_extract("https://example.invalid/supertonic.tar.bz2", dest_dir)

    assert calls == ["Supertonic model"]
    for name in supertonic_tts._REQUIRED_FILES:
        assert (dest_dir / name).is_file()


def test_download_and_extract_wraps_unsafe_member_as_supertonic_download_error(
    tmp_path: Path, monkeypatch
) -> None:
    """Confirms the shared extractor's `UnsafeTarMemberError` is translated
    into Supertonic's own domain exception, not left as a generic error."""
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
    with pytest.raises(
        supertonic_tts.SupertonicModelDownloadError, match="unsafe Supertonic model archive"
    ):
        supertonic_tts._download_and_extract("https://example.invalid/evil.tar.bz2", dest_dir)
