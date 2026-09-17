ARG BUILD_FROM
FROM ${BUILD_FROM}

ARG PYTHON_VERSION=3.11

SHELL ["/bin/bash", "-o", "pipefail", "-c"]

# Install system dependencies
RUN apt-get update \
    && apt-get install -y --no-install-recommends \
        python${PYTHON_VERSION} \
        python${PYTHON_VERSION}-venv \
        python${PYTHON_VERSION}-dev \
        build-essential \
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
RUN pip install --no-cache-dir \
    torch \
    --index-url https://download.pytorch.org/whl/cpu \
    --extra-index-url https://pypi.org/simple \
    && pip install --no-cache-dir \
    moonshine-voice==0.1.5 \
    wyoming==1.10.2 \
    numpy==2.4.6 \
    websockets==17.1 \
    pocket-tts==3.1.0

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
ENV MOONSHINE_VOICE_CACHE=/data/models
ENV HF_HOME=/data/models/pocket-tts

# Health check: real Wyoming describe round-trip, not a raw-byte grep (see
# app/healthcheck.py). start-period raised from 60s: a first-run Pocket TTS
# model download can take longer than plain Moonshine STT startup did.
HEALTHCHECK --start-period=180s --interval=30s --timeout=20s \
    CMD python3 -m app.healthcheck || exit 1

ENTRYPOINT ["/init"]
