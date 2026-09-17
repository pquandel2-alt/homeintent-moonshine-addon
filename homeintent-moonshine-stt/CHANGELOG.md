# Changelog - HomeIntent Moonshine STT Add-on

## [0.1.1] - 2026-09-17

### Fixed
- Base image switched from Debian `trixie` to `bookworm`: trixie's apt archive no longer
  ships `python3.11`/`python3.11-venv`/`python3.11-dev`, which would have broken the
  container build outright
- Fixed `cd /app` → `cd /` in the s6 run script: `COPY app/ /app/` places the `app` package
  directly at `/app`, so `python3 -m app` needs `/` (the parent) as cwd, not `/app` — this
  bug would have crashed the container immediately on every real start
- Replaced the invalid `echo | nc | grep` HEALTHCHECK (not valid Wyoming wire framing) with
  `app/healthcheck.py`, a real Wyoming describe/info round trip using the same
  `wyoming.client.AsyncTcpClient` the server itself runs on

### Added
- `.github/workflows/build.yml`: real multi-arch Docker build in CI (amd64 native,
  aarch64 via QEMU), with an end-to-end smoke test that boots the actual container via its
  real `/init` entrypoint and verifies a real Wyoming describe round trip against the
  running server

### Note
The previous `v0.1.0` tag predates the full Moonshine/Wyoming API remediation and was never
actually verified to build or run — treat it as superseded. This is the first release with a
verified, real container build and startup. See `ABSCHLUSSBERICHT_V0.1.md` for the full
verification report.

## [0.1.0] - 2026-09-16

### Added
- Initial release with Moonshine v0.1.5 streaming ASR
- German Tiny (34M) and Small (123M) model support
- Wyoming STT protocol on port 10300
- Real-time streaming inference
- Model caching to `/data/models`
- Configurable log levels
- Multi-architecture Docker support (amd64, aarch64)
- Home Assistant automatic discovery
- Comprehensive test suite with streaming regression test
- GitHub Actions CI/CD (lint, type-check, test)
