# Changelog - HomeIntent Moonshine Voice Add-on

## [0.2.1] - 2026-09-17

Makes the automatic Moonshine STT vocabulary Assist-aware: it now reflects the
entities actually exposed to Home Assistant's built-in Assist/Conversation
pipeline instead of the entire registry indiscriminately. See
`ABSCHLUSSBERICHT_V0.2.1.md` for the full verification report. Pocket TTS and the
core Moonshine streaming/locking implementation are unchanged.

### Changed
- **HA vocabulary now sourced from Assist exposure** (`app/ha_vocabulary.py`): only
  entities *effectively exposed* to Assist (the `"conversation"` assistant)
  contribute automatic vocabulary terms. Effective exposure is computed the same
  way Home Assistant Core itself does -- an explicit, cached
  `should_expose` override always wins; otherwise the same default-exposure rule
  Home Assistant Core uses (`DEFAULT_EXPOSED_DOMAINS`, plus device-class allowlists
  for `binary_sensor`/`sensor`), gated on the assistant's "expose new entities
  automatically" setting -- reimplemented and verified against the `home-assistant/
  core` `dev` branch (see the module docstring for the exact source references).
  A previously-unexposed sensor such as `sensor.router_cpu_temperature` no longer
  pollutes the STT vocabulary; an exposed `cover.wohnzimmer_rolllade` with an alias
  now contributes its name, alias, and area combinations (`Rolllade`, `Rollo`,
  `Wohnzimmer Rolllade`, `Wohnzimmer Rollo`).
  - Area+Entity and Area+Alias combination terms are still generated (see v0.2.0),
    now scoped to Assist-exposed entities only.
  - Entity aliases are fetched via `config/entity_registry/get_entries`
    (`config/entity_registry/list` alone doesn't include them), for exposed
    entities only.
  - An entity_id (e.g. `cover.wohnzimmer_rolllade` -> "Wohnzimmer Rolllade") is
    used as a last-resort vocabulary term only when an exposed entity has no
    name, no original name, and no alias at all -- the domain itself
    (`cover`, `sensor`, ...) is never added as a keyword.
  - A device's default name is skipped as a vocabulary term if it looks like a
    MAC address, a UUID, or a bare hex serial (e.g. "Shelly Plus 2PM 84FCE6");
    a user-customized device name (`name_by_user`) is always trusted.
  - A failed refresh (Supervisor/HA unreachable, timeout, auth error, ...) keeps
    the previous last-known-good Assist vocabulary unchanged (see v0.2.0); a
    *successful* refresh that no longer sees a previously-exposed entity removes
    it from the vocabulary, since that's not an API failure.
  - Manual `extra_keyterms` are unaffected and merged in exactly as before.
  - Startup/refresh logging stays compact (`Assist-exposed entities: N, HA
    vocabulary terms: M`), never dumping full entity/alias lists at INFO level.

### Fixed
- Real GitHub Actions CI for v0.2.0 was red on Docker build (amd64+aarch64) and
  `e2e-audio-transcribe`: `pip install torch --index-url
  https://download.pytorch.org/whl/cpu` broke resolution of `typing-extensions`
  (no PyPI fallback), and `e2e-audio-transcribe` was missing `pocket-tts`. Fixed by
  adding `--extra-index-url https://pypi.org/simple` everywhere torch is installed
  and adding the missing TTS dependency install.

## [0.2.0] - 2026-09-17

Adds local German text-to-speech (Kyutai Pocket TTS) alongside the existing Moonshine
STT, and fixes the technical review findings from v0.1.2. See
`ABSCHLUSSBERICHT_V0.2.0.md` for the full verification report. Add-on display name
changed to "HomeIntent Moonshine Voice"; the Supervisor slug
(`homeintent-moonshine-stt`) is deliberately unchanged so existing installs update in
place.

### Added
- **Kyutai Pocket TTS** (`pocket-tts` PyPI package, MIT-licensed code, verified
  against upstream source): local, CPU-first, streaming German text-to-speech.
  - `tts_enabled` (default `false` -- see "Changed" below for why), `tts_model`
    (`german` default / `german_24l`), `tts_voice` (default `juergen`, the only real
    German preset voice), `tts_log_performance`.
  - Real incremental audio streaming via Pocket TTS's own
    `generate_audio_stream()` generator API -- chunks are forwarded to the Wyoming
    client as soon as they are decoded, not after the whole reply is synthesized.
  - Combined Wyoming discovery: `Info(asr=[...], tts=[...])` reflects whichever of
    `stt_enabled`/`tts_enabled` is actually on, supporting STT-only, TTS-only, and
    both-enabled configurations.
  - TTFA (generated/sent) + synthesis time + RTF performance logging, symmetric with
    STT's own performance line, never logging synthesized text.
  - Voice-state caching (`get_state_for_audio_prompt()` is "relatively slow" per
    upstream) and a model-specific lock (independent of Moonshine's own STT lock)
    serializing concurrent TTS requests without blocking STT.
  - Graceful handling of empty text, synthesis errors, and client disconnect
    mid-stream (the async generator's own `aclose()` releases the lock and signals
    the producer thread to stop).
- `stt_enabled` (default `true`): lets STT be disabled entirely for a TTS-only
  deployment.
- `float32_to_pcm_int16()` (`app/audio.py`): clips before converting Pocket TTS's
  float32 output to the 16-bit PCM Wyoming's audio-chunk events carry.
- CI: an upgrade-safety smoke test job that starts the add-on against a fake
  Supervisor missing several newer config.yaml options (and two explicitly
  null-valued), to catch a re-introduction of the v0.1.2 startup crash.
- CI: `test_e2e_tts.py`, a real Pocket TTS Wyoming round-trip test (`pytest -m e2e`),
  wired to a manually-triggered (`workflow_dispatch`) CI job rather than every
  push/PR, to avoid a real model download on every normal change (see B30 in the
  review prompt / `ABSCHLUSSBERICHT_V0.2.0.md`).

### Changed
- `numpy` pin bumped `1.24.3` → `2.4.6`: Pocket TTS requires `numpy>=2`; verified
  Moonshine's own moonshine-voice 0.1.5 still imports and works correctly against
  numpy 2.x (it declares no numpy version constraint of its own).
- `tts_enabled` defaults to `false`: an add-on *upgrade* must not suddenly download a
  PyTorch runtime and a Pocket TTS model (several hundred MB) for a previously
  STT-only installation. New installations are documented to turn it on explicitly.
- Dockerfile installs `torch` first, explicitly from PyTorch's own CPU wheel index
  (`--index-url`, which *replaces* rather than merely extends the default index) --
  an `--extra-index-url`-only install was found, during development, to still let
  pip's resolver pick PyPI's default CUDA build, pulling ~1GB of unused `nvidia-*`
  packages.
- Docker `HEALTHCHECK`/discovery-service startup grace period increased (60s → 180s
  for discovery's own poll loop) since a first-run Pocket TTS download can take
  longer than plain Moonshine STT startup did.

### Fixed (review findings from v0.1.2, see `ABSCHLUSSBERICHT_V0.2.0.md` for detail)
- **CI Supervisor config sync**: the fake Supervisor stub used in CI now parses
  `config.yaml`'s own `options:` block directly instead of a hand-duplicated dict,
  so it cannot drift out of sync with the real add-on config the way it previously
  did.
- **Upgrade-safe option handling**: the production run script now falls back to
  `config.yaml`'s own default whenever a Supervisor-provided option value is absent,
  empty, or the literal string `"null"` -- previously, an upgrade with a
  partially-migrated `options.json` could pass e.g. `--keyterm-boost null` straight
  to argparse and crash the add-on on every start.
- **HA vocabulary last-known-good**: `fetch_ha_vocabulary()` now returns a typed
  `HaVocabularyResult(success, terms)` distinguishing a failed refresh from a
  legitimately empty one; the periodic refresh loop keeps the previous vocabulary on
  failure instead of silently wiping it out.
- **Moonshine `set_keyterms()` locking**: the periodic HA vocabulary refresh's
  `set_keyterms()` call is now serialized through the same lock used by streaming
  STT sessions, closing a race between vocabulary refresh and concurrent audio
  streaming.
- **Real STT RTF**: performance logging now reports cumulative Moonshine inference
  time (summed across `add_audio()` calls plus the final `stop()` pass), not
  finalize-wall-time-over-audio-duration, which undercounted work already done
  during streaming.
- **Area+Entity contextual keyterms**: HA vocabulary now also generates "Area
  Entity" combination terms (e.g. "Wohnzimmer Rolllade") from real
  Entity→Area/Entity→Device→Area registry relationships, deterministically sorted
  and capped to avoid a combinatorial explosion on large installs.

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
