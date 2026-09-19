"""Container-level safe-tar-extraction smoke test, CI-only.

Run *inside the real built add-on image* (via `docker cp` + `docker exec`,
see build.yml's build-amd64 job) against the container's own venv Python --
NOT this repo's CI Python (installed separately via actions/setup-python@v4,
which is a different Python build/patch level than what the real add-on
image actually runs on).

This is the exact gap that let the v0.6.1 production bug ship: CI's own
Kroko E2E test (test_e2e_kroko_stt.py) only ever ran under CI's Python,
never inside the actual built Docker image, so it never caught that the
image's own python3.11 (Debian bookworm's apt-installed package, built from
upstream 3.11.2, predating the upstream 3.11.4 backport of PEP 706's
`filter=` keyword argument to `TarFile.extractall()`) lacks that keyword
argument entirely -- while CI's own `actions/setup-python@v4` Python (a
much newer 3.11.x patch release) has it, so CI never saw the real
`TypeError: TarFile.extractall() got an unexpected keyword argument
'filter'` failure a real installation hit in production.

v0.6.2 factored the actual extraction logic out of app/kroko_model.py into
the shared app/safe_tar_extract.py module (also used by
app/supertonic_tts.py for the Supertonic 3 TTS model archive) -- so this
single smoke test now exercises that shared module through BOTH real
callers' own, unmodified download-and-extract functions:
`app.kroko_model._download_and_extract` and
`app.supertonic_tts._download_and_extract`. Since both callers now funnel
through the exact same `app.safe_tar_extract.safe_extractall`, one
container-level run proves the real image's Python/tarfile behavior for
both -- a second, separate smoke test per engine would just repeat the same
check against the same shared code path.

Builds tiny synthetic .tar.bz2 archives (dummy content, no real model
download) matching each real archive's shape, then calls each engine's own
`_download_and_extract` -- the SAME functions production uses -- with
`urllib.request.urlretrieve` monkeypatched to copy the local fixture instead
of hitting the network. Exits non-zero (failing the CI step/job) on any
exception or on missing/empty extracted files.
"""

import inspect
import io
import sys
import tarfile
import tempfile
import urllib.request
from pathlib import Path

print(f"python3 version: {sys.version}")
print(
    f"tarfile.TarFile.extractall signature: {inspect.signature(tarfile.TarFile.extractall)}"
)

sys.path.insert(0, "/app")
from app import kroko_model, safe_tar_extract, supertonic_tts

print(
    f"app.safe_tar_extract.EXTRACTALL_SUPPORTS_FILTER: {safe_tar_extract.EXTRACTALL_SUPPORTS_FILTER}"
)


def _build_fixture_archive(archive_path: Path, files: dict[str, bytes]) -> None:
    with tarfile.open(archive_path, "w:bz2") as tf:
        for name, content in files.items():
            info = tarfile.TarInfo(name=name)
            info.size = len(content)
            info.type = tarfile.REGTYPE
            tf.addfile(info, fileobj=io.BytesIO(content))


def _check_kroko() -> int:
    with tempfile.TemporaryDirectory() as tmp:
        tmp_path = Path(tmp)
        fixture_archive = tmp_path / "fixture-kroko.tar.bz2"
        # Members are relative to the archive root (no package-name prefix)
        # -- matches how _download_and_extract's caller
        # (resolve_kroko_model_files) extracts straight into the
        # package_dir itself, exactly as the real sherpa-onnx release
        # archive does.
        _build_fixture_archive(
            fixture_archive,
            {
                "encoder-epoch-99-avg-1.int8.onnx": b"fake-encoder",
                "decoder-epoch-99-avg-1.onnx": b"fake-decoder",
                "joiner-epoch-99-avg-1.onnx": b"fake-joiner",
                "tokens.txt": b"<blk> 0\n",
            },
        )

        def _fake_urlretrieve(url: str, filename: str, *a: object, **k: object) -> None:
            Path(filename).write_bytes(fixture_archive.read_bytes())

        urllib.request.urlretrieve = _fake_urlretrieve  # type: ignore[assignment]

        dest_dir = tmp_path / "cache" / kroko_model.KROKO_PACKAGE_NAME
        # The exact, real production function -- not a reimplementation.
        kroko_model._download_and_extract(
            "https://example.invalid/fixture.tar.bz2", dest_dir
        )

        tokens = dest_dir / "tokens.txt"
        encoder = dest_dir / "encoder-epoch-99-avg-1.int8.onnx"
        if not tokens.is_file() or tokens.stat().st_size == 0:
            print(f"FAIL: {tokens} missing or empty after extraction", file=sys.stderr)
            return 1
        if not encoder.is_file() or encoder.stat().st_size == 0:
            print(f"FAIL: {encoder} missing or empty after extraction", file=sys.stderr)
            return 1

    print("OK: Kroko model archive extraction succeeded inside the real add-on image")
    return 0


def _check_supertonic() -> int:
    with tempfile.TemporaryDirectory() as tmp:
        tmp_path = Path(tmp)
        fixture_archive = tmp_path / "fixture-supertonic.tar.bz2"
        files = {name: b"fake" for name in supertonic_tts._REQUIRED_FILES}
        _build_fixture_archive(fixture_archive, files)

        def _fake_urlretrieve(url: str, filename: str, *a: object, **k: object) -> None:
            Path(filename).write_bytes(fixture_archive.read_bytes())

        urllib.request.urlretrieve = _fake_urlretrieve  # type: ignore[assignment]

        dest_dir = tmp_path / "cache" / supertonic_tts.SUPERTONIC_PACKAGE_NAME
        # The exact, real production function -- not a reimplementation.
        supertonic_tts._download_and_extract(
            "https://example.invalid/fixture.tar.bz2", dest_dir
        )

        for name in supertonic_tts._REQUIRED_FILES:
            path = dest_dir / name
            if not path.is_file() or path.stat().st_size == 0:
                print(
                    f"FAIL: {path} missing or empty after extraction", file=sys.stderr
                )
                return 1

    print(
        "OK: Supertonic model archive extraction succeeded inside the real add-on image"
    )
    return 0


def main() -> int:
    return _check_kroko() or _check_supertonic()


if __name__ == "__main__":
    sys.exit(main())
