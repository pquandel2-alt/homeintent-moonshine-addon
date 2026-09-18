"""Guards against runtime dependency versions drifting apart again.

requirements-runtime.txt is the single source of truth for pinned runtime
dependency versions (see its own header comment): the Dockerfile installs
directly from it, and .github/workflows/build.yml's e2e jobs (STT and TTS)
install the same file, so real e2e tests run against the identical
dependency stack the add-on actually ships -- see ABSCHLUSSBERICHT_V0.2.3.md,
which found the e2e jobs had drifted to unpinned installs of
moonshine-voice/wyoming/pocket-tts.

app/pyproject.toml's own ``[project.dependencies]`` list is a second,
separate piece of metadata (not consumed by any install step in this
repository) that could still silently drift from requirements-runtime.txt
if edited independently -- this test is the simple, robust guard against
that: no parser test is needed for the Dockerfile itself, since it now
installs directly from requirements-runtime.txt (structurally cannot
drift).
"""

import re
import tomllib
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[3]
REQUIREMENTS_PATH = REPO_ROOT / "homeintent-moonshine-stt" / "requirements-runtime.txt"
PYPROJECT_PATH = REPO_ROOT / "homeintent-moonshine-stt" / "app" / "pyproject.toml"
DOCKERFILE_PATH = REPO_ROOT / "homeintent-moonshine-stt" / "Dockerfile"


def _parse_requirements(path: Path) -> dict[str, str]:
    pins: dict[str, str] = {}
    for line in path.read_text().splitlines():
        line = line.strip()
        if not line or line.startswith("#"):
            continue
        name, _, version = line.partition("==")
        pins[name.strip()] = version.strip()
    return pins


def test_pyproject_dependencies_match_requirements_runtime() -> None:
    requirements = _parse_requirements(REQUIREMENTS_PATH)

    with open(PYPROJECT_PATH, "rb") as f:
        pyproject = tomllib.load(f)
    pyproject_deps = pyproject["project"]["dependencies"]

    pyproject_pins: dict[str, str] = {}
    for dep in pyproject_deps:
        name, _, version = dep.partition("==")
        pyproject_pins[name.strip()] = version.strip()

    assert pyproject_pins == requirements, (
        "app/pyproject.toml's [project.dependencies] has drifted from "
        "requirements-runtime.txt -- keep both in sync (requirements-runtime.txt "
        "is authoritative; the Dockerfile and CI e2e jobs install from it directly)."
    )


def test_dockerfile_installs_from_requirements_runtime() -> None:
    dockerfile_text = DOCKERFILE_PATH.read_text()
    assert re.search(r"pip install .*-r\s+/tmp/requirements-runtime\.txt", dockerfile_text), (
        "Dockerfile must install runtime dependencies from requirements-runtime.txt "
        "(via COPY + pip install -r), not a separately hand-maintained version list."
    )
    assert "COPY requirements-runtime.txt" in dockerfile_text
