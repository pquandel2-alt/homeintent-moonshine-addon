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

# Install Python packages
RUN pip install --no-cache-dir \
    moonshine-voice==0.1.5 \
    wyoming==1.10.2 \
    numpy==1.24.3

# Copy application
WORKDIR /app
COPY app/ /app/

# Copy rootfs (s6-overlay services, scripts)
WORKDIR /
COPY rootfs /

# Set environment for model caching (see moonshine_voice.download_file.get_cache_dir)
ENV MOONSHINE_VOICE_CACHE=/data/models

# Health check: real Wyoming describe round-trip, not a raw-byte grep (see app/healthcheck.py)
HEALTHCHECK --start-period=60s --interval=30s --timeout=20s \
    CMD python3 -m app.healthcheck || exit 1

ENTRYPOINT ["/init"]
