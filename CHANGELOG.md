# Changelog

## [0.1.1] - 2026-09-17

### Fixed
- Docker build was never actually verified until now; fixed a broken base image
  (trixie → bookworm, trixie lacks python3.11 in apt) and a startup bug (`cd /app` → `cd /`
  in the s6 run script) that would have crashed the container on every real start
- Replaced an invalid HEALTHCHECK with a real Wyoming describe/info round trip

### Added
- `.github/workflows/build.yml`: real multi-arch Docker build + container smoke test in CI

See `homeintent-moonshine-stt/CHANGELOG.md` and `ABSCHLUSSBERICHT_V0.1.md` for full details.
The previous `v0.1.0` release predates these fixes and was never verified to actually run.

## [0.1.0] - 2026-09-16

### Added
- Initial release: HomeIntent Moonshine STT Add-on for Home Assistant
- Real-time streaming ASR using Moonshine German models (Tiny, Small)
- Wyoming protocol integration (Port 10300)
- CPU-only inference support
- Model caching to `/data/models`
- Multi-architecture support (amd64, aarch64)
- Auto-discovery for Home Assistant Assist integration
- Configuration options for model selection and logging
- Comprehensive unit tests with streaming regression test
- CI/CD with lint, type-check, and test workflows
