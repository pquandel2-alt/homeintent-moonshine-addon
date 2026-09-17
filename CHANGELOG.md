# Changelog

## [0.2.0] - 2026-09-17

Adds local German text-to-speech (Kyutai Pocket TTS, off by default on upgrade) and
fixes the v0.1.2 review findings (CI config sync, upgrade-safe option handling, HA
vocabulary last-known-good, `set_keyterms()` locking, real STT RTF, Area+Entity
contextual keyterms). Add-on display name is now "HomeIntent Moonshine Voice"; the
Supervisor slug is unchanged so existing installs update in place.

See `homeintent-moonshine-stt/CHANGELOG.md` and `ABSCHLUSSBERICHT_V0.2.0.md` for full details.

## [0.1.2] - 2026-09-17

Home Assistant / smart-home optimization pass, ahead of the first real-hardware
practice run.

### Added
- Home Assistant vocabulary biasing (`use_ha_vocabulary`, read-only, degrades
  gracefully), manual `extra_keyterms`, and a verified `keyterm_boost`
- Real Moonshine streaming parameters exposed as options (`vad_threshold`,
  `decode_incomplete_lines`, `transcription_interval`)
- Optional transcript logging (`log_transcripts`), performance logging
  (`log_performance`), and debug audio recording (`save_debug_audio`) with
  strict metadata allowlists and retention limits — all off/limited by default
- Local RTF benchmark CLI (`app/benchmark.py`) and a real end-to-end Wyoming
  round-trip test using self-synthesized German audio (no third-party audio
  committed), run as its own CI job

### Changed
- Discovery readiness now uses s6's own service-readiness polling
- Session lifecycle hardened against duplicate/overlapping events, client
  disconnects, and mid-processing exceptions
- Concurrent sessions on the same Transcriber are now serialized, since
  moonshine-voice 0.1.5 does not document concurrent-call safety

See `homeintent-moonshine-stt/CHANGELOG.md` and `ABSCHLUSSBERICHT_V0.1.2.md` for full details.

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
