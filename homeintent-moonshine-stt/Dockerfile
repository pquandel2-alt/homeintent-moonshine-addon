ARG BUILD_FROM
FROM ${BUILD_FROM}

ARG PYTHON_VERSION=3.11

SHELL ["/bin/bash", "-o", "pipefail", "-c"]

# Install system dependencies
#
# portaudio19-dev: needed to build the `pyaudio` wheel (no prebuilt wheels
# exist for it on PyPI, verified). pyaudio is a transitive dependency of
# the `speechcatcher` package (see requirements-runtime.txt) -- its own CLI
# module imports pyaudio unconditionally for a live-microphone feature this
# add-on's Wyoming-driven code path never uses, but the import must still
# succeed. Only pulled in when stt_engine=speechcatcher_m/speechcatcher_l;
# harmless (a few hundred KB) for every other configuration.
RUN apt-get update \
    && apt-get install -y --no-install-recommends \
        python${PYTHON_VERSION} \
        python${PYTHON_VERSION}-venv \
        python${PYTHON_VERSION}-dev \
        build-essential \
        portaudio19-dev \
        git \
    && rm -rf /var/lib/apt/lists/*

# Create virtual environment
RUN python${PYTHON_VERSION} -m venv /opt/venv
ENV PATH="/opt/venv/bin:$PATH"

# Install Python packages.
#
# torch is installed FIRST and explicitly from PyTorch's own CPU-only wheel
# index, given as --index-url (primary source) PLUS --extra-index-url
# pointing back at PyPI (fallback for anything not hosted on PyTorch's
# index, e.g. typing-extensions/filelock). A bare --extra-index-url-only
# install was found, while developing this add-on, to let pip's resolver
# pick PyPI's default CUDA build instead, pulling ~1GB of unused nvidia-*
# packages -- but --index-url ALONE (no PyPI fallback) broke entirely in
# real CI: PyTorch's index doesn't host plain-Python packages like
# typing-extensions as wheels, so pip fell back to building it from an
# sdist, whose build backend (flit_core) isn't on that index either,
# failing the whole build (see ABSCHLUSSBERICHT_V0.2.1.md). This
# index-url-primary + extra-index-url-fallback combination is what
# actually keeps both properties: CPU-only torch, and every other
# dependency still resolvable. Installing pocket-tts afterwards then finds
# its own "torch>=2.5.0" requirement already satisfied and never touches
# it again.
COPY requirements-runtime.txt /tmp/requirements-runtime.txt
RUN pip install --no-cache-dir \
    torch torchaudio \
    --index-url https://download.pytorch.org/whl/cpu \
    --extra-index-url https://pypi.org/simple \
    && pip install --no-cache-dir -r /tmp/requirements-runtime.txt \
    && rm /tmp/requirements-runtime.txt

# Speechcatcher (stt_engine=speechcatcher_m/speechcatcher_l) and its own
# two forked dependencies are not published on PyPI (verified) and are
# installed straight from pinned git commits, with --no-deps: every one of
# their own transitive dependencies is already pinned and installed above
# (requirements-runtime.txt), so pip never has to resolve (or trust the
# unpinned refs inside) speechcatcher's/espnet_model_zoo's own
# requirements files at build time. Order matters: espnet_streaming_decoder
# and espnet_model_zoo first (speechcatcher imports both at runtime),
# speechcatcher itself last. Commits pinned to the exact revisions this
# add-on was verified against (see app/speechcatcher_model.py's module
# docstring for what was verified from each).
RUN pip install --no-cache-dir --no-deps \
    "git+https://github.com/speechcatcher-asr/espnet_streaming_decoder.git@76752e2eb37b68e4466fa65bae3a662ef1043f4c" \
    "git+https://github.com/speechcatcher-asr/espnet_model_zoo.git@e3f3c8f69794e3cacf0a88900f2c3d5f17360fae" \
    "git+https://github.com/speechcatcher-asr/speechcatcher.git@1f647a43105b7167b6a47e7b2af2ef3c4ff53a2b"

# Copy application
WORKDIR /app
COPY app/ /app/

# Copy rootfs (s6-overlay services, scripts)
WORKDIR /
COPY rootfs /

# Set environment for model caching. MOONSHINE_VOICE_CACHE: see
# moonshine_voice.download_file.get_cache_dir. HF_HOME: pocket_tts's
# download_if_necessary() delegates hf:// weight/voice paths to
# huggingface_hub's hf_hub_download(), which honors this standard env var
# for its cache location (see app/tts.py's module docstring). Kept as a
# separate subdirectory from Moonshine's own model cache so STT and TTS
# model files never collide and can be inspected/cleared independently.
# KOKORO_ONNX_CACHE: app/kokoro_tts.py passes this as hf_hub_download()'s
# own explicit cache_dir= (NOT via HF_HOME, which stays pointed at Pocket
# TTS's cache above) so the Kokoro German model -- several hundred MB,
# only ever downloaded when tts_engine=kokoro_onnx -- lives in its own
# inspectable/clearable directory too, never baked into the image itself
# (see config.yaml's `backup_exclude: ["models/*"]`, which already covers
# this path).
ENV MOONSHINE_VOICE_CACHE=/data/models
ENV HF_HOME=/data/models/pocket-tts
ENV KOKORO_ONNX_CACHE=/data/models/kokoro-onnx
# KROKO_CACHE: app/kroko_model.py's own model cache, only ever populated
# when stt_engine=kroko. SUPERTONIC_CACHE: app/supertonic_tts.py's own
# model cache, only ever populated when tts_enabled=true and
# tts_engine=supertonic_3. Both are separate subdirectories under /data/
# models for the same reasons as Pocket TTS/Kokoro above (independent
# inspection/clearing; already covered by config.yaml's
# backup_exclude: ["models/*"]).
ENV KROKO_CACHE=/data/models/kroko
ENV SUPERTONIC_CACHE=/data/models/supertonic
# SPEECHCATCHER_CACHE: app/speechcatcher_model.py's own model cache, only
# ever populated when stt_engine is speechcatcher_m or speechcatcher_l.
# Same independent-subdirectory reasoning as the other engines above.
ENV SPEECHCATCHER_CACHE=/data/models/speechcatcher
# VOSK_CACHE: app/vosk_model.py's own model cache, only ever populated
# when stt_engine=vosk_german. Same independent-subdirectory reasoning as
# the other engines above.
ENV VOSK_CACHE=/data/models/vosk

# Health check: real Wyoming describe round-trip, not a raw-byte grep (see
# app/healthcheck.py). start-period raised from 60s: a first-run Pocket TTS
# model download can take longer than plain Moonshine STT startup did.
HEALTHCHECK --start-period=180s --interval=30s --timeout=20s \
    CMD python3 -m app.healthcheck || exit 1

ENTRYPOINT ["/init"]
