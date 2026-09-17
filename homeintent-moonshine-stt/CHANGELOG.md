# Changelog - HomeIntent Moonshine STT Add-on

## [0.1.2] - 2026-09-17

Home Assistant / smart-home optimization pass, ahead of the first real-hardware
practice run. See `ABSCHLUSSBERICHT_V0.1.2.md` for the full verification report.

### Added
- `use_ha_vocabulary` (default `true`): reads Areas/Devices/Entities/Floors from
  Home Assistant's registries over the Supervisor-proxied Core WebSocket API
  (`app/ha_vocabulary.py`) and biases Moonshine's recognition towards the user's
  actual room/device names. Strictly read-only — only `config/*_registry/list`
  commands are issued; no service is ever called and no state is ever changed.
  Degrades gracefully (a WARNING log, not a crash) if Home Assistant is
  unreachable. Optional periodic refresh via `ha_vocabulary_refresh_minutes`.
- `extra_keyterms`: comma-separated manual keyterms, merged with the Home
  Assistant vocabulary and deduplicated.
- `keyterm_boost` (default `2.0`, matching Moonshine's own
  `ContextBiaser::kDefaultBoost`): strength applied to keyterms.
- Real Moonshine streaming parameters now exposed as add-on options, at their
  verified upstream defaults: `transcription_interval` (`update_interval`,
  default `0.5`), `vad_threshold` (default `0.5`), `decode_incomplete_lines`
  (default `true`).
- `log_transcripts` (default `false`): recognized text is never logged unless
  explicitly enabled.
- `log_performance` (default `true`): a compact per-utterance
  `model/audio-duration/finalize-time/rtf` line, never including transcript text.
- `save_debug_audio` (default `false`) + `debug_audio_max_files`: optional
  on-disk WAV + JSON-metadata recording of received audio for debugging, with
  automatic oldest-first retention. Metadata is limited to an explicit allowlist
  (timestamp, model, language, transcript, duration, sample rate) — never a Home
  Assistant entity state.
- `app/benchmark.py` (`python -m app.benchmark <file.wav>`): a local CLI to
  measure real-time factor (RTF) on the user's own hardware. Not run in CI and
  not a source of any RTF numbers quoted in this README — Moonshine's speed
  depends heavily on the host CPU, so the only honest numbers are ones users
  measure themselves.
- Real end-to-end Wyoming round-trip test (`app/tests/test_e2e_transcribe.py`,
  `pytest -m e2e`): synthesizes a German sentence with Moonshine's own
  TextToSpeech (no third-party audio is committed to this repository), feeds it
  through a real Wyoming TCP server backed by a real Transcriber, and asserts
  the transcript contains the expected words. Skipped (not faked) if the model
  or TTS voice can't be downloaded. Runs as its own CI job in `build.yml`.

### Changed
- Discovery readiness (`rootfs/etc/s6-overlay/s6-rc.d/discovery/run`) now polls
  the Wyoming server with s6's own service-readiness mechanism instead of a raw
  `/dev/tcp` check.
- Session lifecycle hardened: duplicate `audio-start`, a `transcribe` event
  while a session is already active, client disconnects, and exceptions during
  `add_audio()`/finalize now all close the session and release native
  resources instead of leaking them.
- Concurrent Wyoming sessions on the same Transcriber are now serialized
  through a shared `asyncio.Lock`, since moonshine-voice 0.1.5 does not
  document `create_stream()`/`Stream` native calls as safe to call
  concurrently from multiple threads.
- Startup now prints a compact, single-block summary (model, language, HA
  vocabulary status, keyterm counts, logging settings, model cache path).

### Fixed
- `type-check.yml`/`test.yml` were missing the `websockets` dependency that
  `app/ha_vocabulary.py` requires; both now install it explicitly.

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
