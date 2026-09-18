"""Kokoro ONNX (German "Martin" voice) model resolution and loading.

Kokoro ONNX runtime: package ``kokoro-onnx`` on PyPI (import name
``kokoro_onnx``, https://github.com/thewh1teagle/kokoro-onnx, MIT-licensed
code -- verified directly against the installed 0.6.1 wheel's own
``dist-info/licenses/LICENSE`` and ``kokoro_onnx/__init__.py`` source, not
assumed). Verified facts this module relies on:

- ``Kokoro(model_path, voices_path, espeak_config=None, vocab_config=None)``
  loads a local ``.onnx`` model file and a local voices file directly --
  there is no repo-id-based ``load_model()`` helper like Pocket TTS's, so
  this module resolves/downloads the two files itself (see
  :func:`resolve_kokoro_model_files`).
- ``Kokoro.from_session(session, voices_path, ...)`` accepts an
  already-constructed ``onnxruntime.InferenceSession`` -- this is how
  ONNX Runtime thread/execution settings are actually applied (see
  :func:`load_kokoro_model`'s ``num_threads`` handling and
  app/kokoro_session.py), since ``kokoro_onnx.session.create_session()``
  itself builds a session with onnxruntime's own defaults and exposes no
  threading parameters at all (verified against its source).
- ``Kokoro.create_stream(text, voice, speed=1.0, lang="en-us",
  is_phonemes=False, trim=True, sentence_pause=0.25, clause_pause=0.1) ->
  AsyncGenerator[tuple[NDArray[float32], int], None]`` is real incremental
  streaming: each ``(audio_chunk, sample_rate)`` tuple is yielded from a
  background asyncio task as soon as one phoneme batch is synthesized, not
  after the whole text -- verified directly against the installed
  package's source (``kokoro_onnx/__init__.py``). ``0.25``/``0.1`` are
  upstream's own real defaults for ``sentence_pause``/``clause_pause``,
  not the 0.15-0.25/0.05-0.1 range once assumed before verification.
- German phonemization goes through ``phonemizer`` + ``espeakng-loader``
  (both real, pinned PyPI packages), NOT a system ``espeak-ng`` apt
  package: ``espeakng-loader`` ships a self-contained, platform-specific
  wheel with a prebuilt ``libespeak-ng`` shared library and its full
  ``espeak-ng-data`` directory (including a German ``de_dict``) for both
  ``manylinux_2_17_x86_64`` (amd64) and ``manylinux_2_28_aarch64``
  (aarch64) -- verified directly from PyPI's file listing for
  espeakng-loader==0.2.4. ``kokoro_onnx.tokenizer.Tokenizer`` uses it
  automatically whenever ``espeak_config`` is left at its default
  (``None``), confirmed by reading its source -- no manual wiring, no
  system package, needed for either architecture.
- Output: mono float32 samples at 24000 Hz for the German model
  (``kokoro_onnx.config.SAMPLE_RATE`` -- the same constant regardless of
  which language/voice is loaded), matching Pocket TTS's own sample rate.
- The German "Martin" model (fine-tuned by Godelaune on top of
  hexgrad's Kokoro-82M, StyleTTS2 Stage 2) ships as two files on Hugging
  Face, repo ``Godelaune/Kokoro-82M-ONNX-German-Martin``:
  ``kokoro-martin.onnx`` (the ~326MB acoustic model) and
  ``voices-martin.npz`` (the "martin" voice's style vectors) -- verified
  against that repository's own README (fetched via
  raw.githubusercontent.com, since huggingface.co itself is not reachable
  from this development sandbox's network policy -- the real download at
  add-on runtime goes through huggingface_hub directly, unaffected by
  that sandbox-only restriction, exactly as already established for
  Pocket TTS's own model in app/tts.py). Both files are published under
  the Apache License 2.0, the same license as upstream Kokoro-82M itself.
  A separate ``lexicon.txt.gz`` (CC BY 4.0, built from Tatoeba/Leipzig
  Corpora text) also exists in that repository as an optional
  pronunciation-correction lexicon for the reference FastAPI service; it
  is NOT used here -- this add-on relies on kokoro-onnx's own built-in
  espeak-ng phonemization instead (see module docstring above and
  ABSCHLUSSBERICHT_V0.3.0.md for why it was not vendored).
- No immutable Git revision/commit for that Hugging Face repository could
  be found in its own README or via search from this environment, so
  :data:`DEFAULT_KOKORO_REVISION` defaults to the mutable ``"main"`` ref,
  overridable via the ``kokoro_model_revision`` add-on option once a
  stable tag is published upstream -- documented honestly as a known
  limitation rather than a fabricated pinned commit hash.
"""

import logging
import os
from pathlib import Path

from huggingface_hub import hf_hub_download
from huggingface_hub.utils import (  # type: ignore[attr-defined]
    HfHubHTTPError,
    LocalEntryNotFoundError,
)

_LOGGER = logging.getLogger(__name__)

# Kept separate from Moonshine's (/data/models) and Pocket TTS's
# (/data/models/pocket-tts) own caches so all three model families can be
# inspected/cleared independently; matches config.yaml's
# ``backup_exclude: ["models/*"]``, which already covers this path.
DEFAULT_KOKORO_CACHE_DIR = Path(os.environ.get("KOKORO_ONNX_CACHE", "/data/models/kokoro-onnx"))

KOKORO_REPO_ID = "Godelaune/Kokoro-82M-ONNX-German-Martin"
KOKORO_MODEL_FILENAME = "kokoro-martin.onnx"
KOKORO_VOICES_FILENAME = "voices-martin.npz"
DEFAULT_KOKORO_REVISION = "main"

DEFAULT_KOKORO_VOICE = "martin"

# kokoro_onnx.config.SAMPLE_RATE, verified identical for every Kokoro
# language/voice combination (including German) directly from source.
KOKORO_SAMPLE_RATE = 24000

# Upstream-published facts (kokoro-onnx==0.6.1, Godelaune/Kokoro-82M-ONNX-
# German-Martin's own README, verified via raw.githubusercontent.com --
# see module docstring). No RTF/latency numbers here are our own
# measurements unless explicitly marked "gemessen" in
# ABSCHLUSSBERICHT_V0.3.0.md.
KOKORO_MODEL_METADATA = {
    "name": "kokoro-82m-german-martin",
    "description": "German Kokoro-82M ONNX text-to-speech (voice: Martin)",
}


class KokoroModelDownloadError(Exception):
    """Raised when the Kokoro model/voices files could not be resolved.

    Deliberately its own exception type (not a bare RuntimeError) so
    app/__main__.py can log a clear, specific startup error instead of a
    generic failure -- selecting ``tts_engine: kokoro_onnx`` and having the
    download fail must be a visible configuration error, never a silent
    fallback to Pocket TTS (see app/__main__.py's module docstring).
    """


def resolve_kokoro_model_files(
    cache_dir: Path | None = None,
    revision: str = DEFAULT_KOKORO_REVISION,
) -> tuple[Path, Path]:
    """Resolve (downloading if needed) the Kokoro German model + voices files.

    Uses ``huggingface_hub.hf_hub_download()`` rather than a hand-rolled
    HTTP downloader: it already downloads to a temporary path and only
    ``rename()``s it into the final cache location once the transfer
    completes successfully (verified against huggingface_hub's own
    documented behavior), so a killed/interrupted download can never leave
    a partial file that a later run would mistake for a valid, complete
    one -- this add-on gets atomic-download-or-nothing for free instead of
    reimplementing it. A cache hit (files already downloaded, matching the
    resolved revision) makes no network request at all beyond a single
    metadata check, so repeated starts do not re-download several hundred
    MB every time.

    Args:
        cache_dir: Directory to store the downloaded files under. Defaults
            to :data:`DEFAULT_KOKORO_CACHE_DIR` (``/data/models/kokoro-onnx``
            in production; overridden in tests).
        revision: Git revision on the Hugging Face repo to pin. Defaults to
            :data:`DEFAULT_KOKORO_REVISION` (``"main"`` -- see module
            docstring for why no immutable commit could be pinned instead).

    Returns:
        ``(model_path, voices_path)`` -- both real, existing local files.

    Raises:
        KokoroModelDownloadError: If either file could not be downloaded
            (network failure, repo/file not found, or any other
            huggingface_hub error). Never returns a partially-downloaded
            or missing file silently.
    """
    resolved_cache_dir = cache_dir if cache_dir is not None else DEFAULT_KOKORO_CACHE_DIR
    resolved_cache_dir.mkdir(parents=True, exist_ok=True)

    try:
        model_path = hf_hub_download(
            repo_id=KOKORO_REPO_ID,
            filename=KOKORO_MODEL_FILENAME,
            revision=revision,
            cache_dir=resolved_cache_dir,
        )
        voices_path = hf_hub_download(
            repo_id=KOKORO_REPO_ID,
            filename=KOKORO_VOICES_FILENAME,
            revision=revision,
            cache_dir=resolved_cache_dir,
        )
    except (HfHubHTTPError, LocalEntryNotFoundError, OSError) as err:
        raise KokoroModelDownloadError(
            f"Failed to download Kokoro German model from "
            f"'{KOKORO_REPO_ID}' (revision={revision}): {type(err).__name__}: {err}"
        ) from err

    model_file = Path(model_path)
    voices_file = Path(voices_path)
    if not model_file.is_file() or model_file.stat().st_size == 0:
        raise KokoroModelDownloadError(
            f"Kokoro model file resolved to '{model_file}' but it is missing or empty"
        )
    if not voices_file.is_file() or voices_file.stat().st_size == 0:
        raise KokoroModelDownloadError(
            f"Kokoro voices file resolved to '{voices_file}' but it is missing or empty"
        )

    return model_file, voices_file


def load_kokoro_model(
    cache_dir: Path | None = None,
    revision: str = DEFAULT_KOKORO_REVISION,
    intra_op_num_threads: int = 0,
    inter_op_num_threads: int = 0,
) -> object:
    """Resolve/download the Kokoro German model, then load it with a
    deliberately-configured ONNX Runtime session.

    Returns ``object`` rather than ``kokoro_onnx.Kokoro`` so this module's
    public type surface does not force a hard, always-imported dependency
    on ``kokoro_onnx`` for callers (e.g. app/kokoro_session.py's
    ``KokoroOnnxSynthesizer``) that only need the small, structural set of
    methods this add-on actually calls -- consistent with how that class
    itself types its own ``kokoro`` constructor parameter.

    ``kokoro_onnx.session.create_session()`` (used internally by
    ``Kokoro(model_path, ...)``) builds an ``onnxruntime.InferenceSession``
    with onnxruntime's own defaults and exposes no way to configure
    threading -- verified directly against its source. To actually control
    ``intra_op_num_threads``/``inter_op_num_threads`` (see
    app/kokoro_benchmark usage via app/tts_benchmark.py and
    ABSCHLUSSBERICHT_V0.3.0.md's thread-tuning section), this function
    builds the ``onnxruntime.InferenceSession`` itself with an explicit
    ``SessionOptions`` and hands it to ``Kokoro.from_session()`` --
    both verified, real, public APIs.

    Args:
        cache_dir: See :func:`resolve_kokoro_model_files`.
        revision: See :func:`resolve_kokoro_model_files`.
        intra_op_num_threads: If > 0, sets
            ``SessionOptions.intra_op_num_threads`` (parallelism within one
            operator). 0 (default) leaves onnxruntime's own default
            untouched -- consistent ``0 = auto`` semantics with Pocket
            TTS's ``tts_threads`` option.
        inter_op_num_threads: If > 0, sets
            ``SessionOptions.inter_op_num_threads`` (parallelism across
            independent operators/graph nodes). Only meaningful together
            with ``execution_mode=ORT_PARALLEL``; kept at onnxruntime's
            own sequential default when 0, matching upstream's own
            typical CPU deployment.

    Returns:
        A loaded ``kokoro_onnx.Kokoro`` instance ready for
        ``create_stream()``.

    Raises:
        KokoroModelDownloadError: See :func:`resolve_kokoro_model_files`.
    """
    import onnxruntime as ort
    from kokoro_onnx import Kokoro

    model_path, voices_path = resolve_kokoro_model_files(cache_dir=cache_dir, revision=revision)

    session_options = ort.SessionOptions()
    if intra_op_num_threads > 0:
        session_options.intra_op_num_threads = intra_op_num_threads
    if inter_op_num_threads > 0:
        session_options.inter_op_num_threads = inter_op_num_threads
        session_options.execution_mode = ort.ExecutionMode.ORT_PARALLEL

    _LOGGER.info(
        "Loading Kokoro ONNX model: model=%s voices=%s intra_op_threads=%s inter_op_threads=%s",
        model_path,
        voices_path,
        intra_op_num_threads or "auto",
        inter_op_num_threads or "auto",
    )
    try:
        session = ort.InferenceSession(
            str(model_path), sess_options=session_options, providers=["CPUExecutionProvider"]
        )
        kokoro = Kokoro.from_session(session, str(voices_path))
    except Exception as err:
        _LOGGER.error("Failed to load Kokoro ONNX model: %s: %s", type(err).__name__, err)
        raise

    _LOGGER.info("Kokoro ONNX German model ready (sample_rate=%dHz)", KOKORO_SAMPLE_RATE)
    return kokoro
