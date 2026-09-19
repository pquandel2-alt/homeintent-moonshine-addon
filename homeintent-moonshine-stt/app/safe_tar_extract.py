"""Shared safe tar-archive extraction for downloaded model archives.

Both ``app/kroko_model.py`` (the Kroko German streaming ASR model archive)
and ``app/supertonic_tts.py`` (the Supertonic 3 TTS model archive) download a
third-party ``.tar.bz2`` release asset and extract it into a local cache
directory. Both need the exact same security-sensitive extraction logic, so
it lives here once instead of being duplicated per engine.

v0.6.2 background: this module's feature-detection flag
(:data:`EXTRACTALL_SUPPORTS_FILTER`) is computed ONCE at import time (cheap,
one-time -- not per-call/per-request overhead), not a `sys.version_info`
guess. This addresses a real production bug this add-on shipped in v0.6.1:
CPython's own `filter=` keyword argument to `TarFile.extractall()` (PEP 706,
CVE-2007-4559 hardening) was added in 3.12 and backported upstream to
3.10.12/3.11.4 -- but Debian bookworm's `apt install python3.11` (what this
add-on's own Dockerfile installs the real image runs on, via
`ghcr.io/home-assistant/{amd64,aarch64}-base-debian:bookworm`) ships
python3.11 packaged from upstream 3.11.2 (verified: Debian's own package page
lists the bookworm python3.11 package version as based on 3.11.2, predating
the 3.11.4 backport) and Debian's own security-patch releases (the
`+deb12uN` suffix) did not backport this specific upstream feature backport
into that package -- so the real add-on image's `python3.11` has no
`filter=` parameter on `TarFile.extractall()` at all, while this repo's own
CI (`actions/setup-python@v4` with `python-version: "3.11"`, which resolves
to a recent python.org-built 3.11.x release, well past 3.11.4) does have it
-- which is exactly why CI's own Kroko E2E test never caught this before it
shipped in a real production add-on install (see the v0.6.2 CHANGELOG entry,
and the container-level smoke test in .github/workflows/build.yml's
build-amd64 job, added specifically to close this CI-vs-real-image gap).
"""

import inspect
import logging
import tarfile
from pathlib import Path

_LOGGER = logging.getLogger(__name__)

EXTRACTALL_SUPPORTS_FILTER = "filter" in inspect.signature(tarfile.TarFile.extractall).parameters


class UnsafeTarMemberError(Exception):
    """Raised when a tar member fails the legacy safety checks.

    Callers of :func:`safe_extractall` should catch this and re-raise it as
    their own domain-specific error (e.g. ``KrokoModelDownloadError``,
    ``SupertonicModelDownloadError``) so a rejected archive is still
    reported as a clear, engine-specific startup failure.
    """


def _reject_unsafe_member(member: tarfile.TarInfo, destination: Path) -> None:
    """Validate a single tar member against ``destination``, raising on any
    unsafe/unsupported member. Only plain regular files and directories are
    ever allowed through -- these model archives only ever need those, so
    the legacy fallback is deliberately conservative and rejects every
    other member type outright (symlinks, hardlinks, device files, FIFOs,
    or anything tarfile doesn't already recognize as a plain type).
    """
    name = member.name

    if Path(name).is_absolute():
        raise UnsafeTarMemberError(f"tar member has an absolute path: {name!r}")

    # Resolve the member's real destination path and verify it stays inside
    # ``destination`` -- via Path.is_relative_to() against *resolved* paths,
    # not a naive string-prefix check (which has known bypass edge cases,
    # e.g. a sibling directory that merely shares a prefix like
    # "dest-evil" vs "dest").
    resolved_destination = destination.resolve()
    member_path = (destination / name).resolve()
    if not member_path.is_relative_to(resolved_destination):
        raise UnsafeTarMemberError(
            f"tar member resolves outside the extraction destination: {name!r}"
        )

    if member.issym() or member.islnk():
        raise UnsafeTarMemberError(f"tar member is a symlink/hardlink, rejected: {name!r}")
    if member.ischr() or member.isblk():
        raise UnsafeTarMemberError(f"tar member is a device file, rejected: {name!r}")
    if member.isfifo():
        raise UnsafeTarMemberError(f"tar member is a FIFO, rejected: {name!r}")
    if not (member.isreg() or member.isdir()):
        raise UnsafeTarMemberError(
            f"tar member is not a regular file or directory, rejected: {name!r} "
            f"(type={member.type!r})"
        )


def _safe_extractall_legacy(tf: tarfile.TarFile, destination: Path) -> None:
    """Safe stand-in for ``TarFile.extractall(destination, filter="data")``
    on Python runtimes whose ``tarfile.TarFile.extractall`` doesn't accept a
    ``filter=`` keyword argument at all (see ``EXTRACTALL_SUPPORTS_FILTER``'s
    module docstring above for exactly which runtime that is and why).

    Every member is validated FIRST, before anything is extracted: this
    ensures a rejected archive never leaves partial content behind, on top
    of (not instead of) the atomic download-to-temp-dir-then-move pattern
    each caller's own download helper already uses.

    Rejects: absolute paths, path traversal outside ``destination``,
    symlinks, hardlinks, device files (character/block), FIFOs, and any
    other non-regular-file/non-directory member. This is deliberately
    conservative -- these model archives only ever contain plain files and
    directories, so nothing else needs to be allowed through.

    Raises:
        UnsafeTarMemberError: on the first unsafe member found.
    """
    destination.mkdir(parents=True, exist_ok=True)
    for member in tf.getmembers():
        _reject_unsafe_member(member, destination)

    # All members validated -- safe to extract the whole archive now.
    tf.extractall(destination)  # noqa: S202 - every member pre-validated above


def safe_extractall(tf: tarfile.TarFile, destination: Path, *, label: str) -> None:
    """Extract ``tf`` into ``destination``, using ``filter="data"`` when the
    running Python's ``tarfile.TarFile.extractall`` supports it, and falling
    back to :func:`_safe_extractall_legacy` otherwise.

    Args:
        tf: An already-open ``tarfile.TarFile``.
        destination: Directory to extract into (created if missing).
        label: Human-readable name of what's being extracted, used only for
            the startup log line (e.g. ``"Kroko model"``,
            ``"Supertonic model"``) -- matches this add-on's existing
            per-engine startup log conventions.

    Raises:
        UnsafeTarMemberError: if the legacy fallback rejects an unsafe
            member. Callers should catch this and re-raise their own
            domain-specific download/extraction error.
    """
    if EXTRACTALL_SUPPORTS_FILTER:
        tf.extractall(destination, filter="data")  # noqa: S202
        _LOGGER.info("%s extracted using: python-tar-filter", label)
    else:
        _safe_extractall_legacy(tf, destination)
        _LOGGER.info("%s extracted using: safe-legacy-extractor", label)
