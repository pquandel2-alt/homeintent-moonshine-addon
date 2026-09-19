"""CI-level dependency/import guard for Speechcatcher's git-only dependency
chain (speechcatcher + its espnet_streaming_decoder/espnet_model_zoo forks,
all installed with ``--no-deps``, see requirements-runtime.txt/Dockerfile).

Purpose: catch a missing transitive dependency -- exactly the class of bug
found in v0.6.1 (``pandas`` and the full ``espnet`` package were both
missing, see CHANGELOG.md) -- in NORMAL, non-gated CI (every push/PR), not
only in the slow, release-tag-gated
``e2e-speechcatcher-stt-transcribe``/``test_e2e_speechcatcher_stt.py`` job
that also downloads several-hundred-MB real model weights. This test is
import-only: it never downloads any model weights and never constructs a
``Speech2TextStreaming`` instance.

Runs in two different CI contexts (see .github/workflows/test.yml):
- The main ``test`` job does NOT install the three pinned git packages (to
  keep the normal, fast test job's install step light) -- this test SKIPS
  there, since ``speechcatcher`` genuinely is not installed and that is not
  a bug.
- The dedicated ``speechcatcher-import-guard`` job DOES install them (the
  same pinned commits + ``espnet``, mirroring build.yml's
  ``e2e-speechcatcher-stt-transcribe`` job) and then actually exercises this
  same test file -- so a missing transitive dependency fails loudly on
  every normal push/PR, not just on a release tag.
"""

import importlib.util

import pytest


def test_speechcatcher_dependency_chain_imports_cleanly() -> None:
    if importlib.util.find_spec("speechcatcher") is None:
        pytest.skip(
            "speechcatcher not installed in this job -- see the dedicated "
            "speechcatcher-import-guard job in .github/workflows/test.yml, "
            "which installs it and runs this same test for real"
        )

    import espnet_model_zoo  # noqa: F401
    import espnet_streaming_decoder  # noqa: F401
    import speechcatcher.speechcatcher  # noqa: F401
