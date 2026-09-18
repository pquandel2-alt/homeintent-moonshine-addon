# Changelog

## [0.2.7] - 2026-09-18

Real Wyoming TTS audio streaming to Home Assistant: the add-on now
advertises and correctly implements `synthesize-start`/`-chunk`/`-stop`/
`-stopped`, so Home Assistant plays audio as soon as Pocket TTS produces
its first chunk instead of waiting for the entire response to buffer
first (its legacy client behavior when streaming wasn't advertised, which
was the actual, verified cause of the perceived TTS latency). A new
`TtsStreamState` state machine guarantees the backwards-compatible
duplicate full-text event Home Assistant also sends never causes text to
be synthesized twice. Adds a new local `python -m app.tts_benchmark` CLI
for tuning `tts_threads` on real hardware. No other behavior changes.

See `homeintent-moonshine-stt/CHANGELOG.md` and `ABSCHLUSSBERICHT_V0.2.7.md` for full details.

## [0.2.6] - 2026-09-18

Follow-up to v0.2.5: keyterms that were still being discarded entirely
because of an invisible formatting character (soft hyphen) or a
decorative symbol/separator (emoji, `/`) are now recovered via a two-stage
normalize-then-speech-fallback pipeline instead of being skipped -- e.g.
`"Wasch\xadmaschine"` -> `"Waschmaschine"`, `"Treppe/Büro"` ->
`"Treppe Büro"`, `"Familie ⚠️"` -> `"Familie"`. Every fallback is still
validated against the actually loaded Moonshine model before use. No
other behavior changes.

See `homeintent-moonshine-stt/CHANGELOG.md` and `ABSCHLUSSBERICHT_V0.2.6.md` for full details.

## [0.2.5] - 2026-09-18

Critical production bug fix: a single Home-Assistant-derived keyterm the
loaded Moonshine model's tokenizer could not represent (real incident: the
entity name "/Büro") crashed the whole add-on into a permanent restart loop.
Introduces a reusable, safe keyterm-application layer used by startup,
manual `extra_keyterms`, and the periodic HA vocabulary refresh alike --
an incompatible term is now skipped and logged instead of taking down the
service. No other behavior changes.

See `homeintent-moonshine-stt/CHANGELOG.md` and `ABSCHLUSSBERICHT_V0.2.5.md` for full details.

## [0.2.4] - 2026-09-18

Real-hardware performance + HA vocabulary fix: fixes a real bug found during
the first real Home Assistant installation (HA vocabulary silently produced
0 keyterms on a larger installation, due to a 1 MiB WebSocket message-size
limit) and adds performance instrumentation/tuning based on real production
STT/TTS timing numbers. No new STT engine, TTS model, fine-tuning, or
architecture changes; Pocket TTS is kept as-is per this round's explicit
scope.

See `homeintent-moonshine-stt/CHANGELOG.md` and `ABSCHLUSSBERICHT_V0.2.4.md` for full details.

## [0.2.3] - 2026-09-18

Verification/stability release: fixes the actually-red v0.2.2 release-tag CI
(a flaky STT e2e test that synthesized its own test audio, now replaced with a
fixed deterministic fixture), corrects an over-eager legacy Home Assistant
entity exposure fallback to a conservative, positive-signal-only strategy,
unifies runtime and e2e dependency versions, and makes the real Pocket TTS
e2e test run automatically as a release gate on version tags. No new STT
engine, TTS model, fine-tuning, or architecture changes.

See `homeintent-moonshine-stt/CHANGELOG.md` and `ABSCHLUSSBERICHT_V0.2.3.md` for full details.

## [0.2.2] - 2026-09-18

Targeted quality/stability pass: legacy (non-registry) Home Assistant entities
now contribute to the automatic STT vocabulary, Pocket TTS's default voice is
validated (and optionally warmed up) at startup instead of lazily on first use,
a mid-stream TTS failure can no longer send a duplicate `audio-start`, TTS
performance logging now separates model compute time from lock-wait and
Wyoming-send time, and both real e2e tests now fail loudly on an actual code/API
regression instead of silently skipping. No new STT engine, no new TTS model, no
architecture changes.

See `homeintent-moonshine-stt/CHANGELOG.md` and `ABSCHLUSSBERICHT_V0.2.2.md` for full details.

## [0.2.1] - 2026-09-17

Makes the automatic Moonshine STT vocabulary Assist-aware: only entities actually
exposed to Home Assistant's built-in Assist/Conversation pipeline (verified against
Home Assistant Core's own exposure logic) now contribute automatic vocabulary
terms, instead of the entire registry. Also fixes a real GitHub Actions CI failure
in the published v0.2.0 Docker build and e2e job (broken torch CPU wheel
resolution, missing pocket-tts dependency).

See `homeintent-moonshine-stt/CHANGELOG.md` and `ABSCHLUSSBERICHT_V0.2.1.md` for full details.

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
