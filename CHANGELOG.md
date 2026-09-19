# Changelog

## [0.6.1] - 2026-09-19

Stability/bugfix-only release. See `homeintent-moonshine-stt/CHANGELOG.md`
for the full technical writeup. Summary: fixed an incomplete Speechcatcher
dependency chain (missing `pandas`/`espnet`, verified by reproducing the
real `ModuleNotFoundError` in a clean install), fixed a Speechcatcher
concurrency bug where two simultaneous connections could corrupt each
other's in-flight transcription state, made the periodic Home Assistant
vocabulary refresh only run for engines that actually support it
(Moonshine/Kroko, not Speechcatcher/Vosk), added the engine name to the STT
performance log line, turned Supertonic's voice and quality-steps options
into real dropdowns instead of free text, added a real end-to-end test for
Supertonic's modern Wyoming streaming path, and added a fast CI job that
catches a missing Speechcatcher dependency on every push instead of only in
the slow release-gated end-to-end job. No new engines, no config renames,
no breaking changes.

## [0.6.0] - 2026-09-19

Adds a fifth and final user-selectable STT engine, **Vosk German**
(`vosk_german`), the lightest-weight, lowest-CPU/RAM option this add-on
offers -- a Kaldi-based recognizer (via the `vosk` PyPI package) driven
in-process through `vosk.KaldiRecognizer.AcceptWaveform()` for real,
incremental streaming. Uses the current, officially recommended small
German model (`vosk-model-small-de-0.15`, ~45MB, Apache 2.0), downloaded
on demand into `/data/models/vosk/` only when selected. Vosk's own
grammar/context-biasing API was investigated and deliberately NOT wired
up as HA-vocabulary biasing: it is a hard closed-set recognition
restriction, not a soft bias, and would break free-form recognition of
anything outside the given phrase list -- honestly reported as
`supports_hotwords=False` via the existing `SttCapabilities` abstraction,
matching Speechcatcher's own precedent. This completes the originally
planned 5 STT x 3 TTS engine matrix; no further STT engines are currently
planned.

Also adds `app/stt_benchmark.py`, a cross-engine STT benchmark CLI driven
entirely through the shared `SttEngine` abstraction (`--engine
moonshine|kroko|speechcatcher_m|speechcatcher_l|vosk_german`), and extends
`app/tts_benchmark.py` (previously Pocket TTS/Kokoro ONNX only) to also
cover **Supertonic 3** -- an explicitly-reported gap from an earlier
phase. Both tools are local, manual, on-your-own-hardware benchmarking
tools (like the pre-existing `app/benchmark.py`), not something CI runs or
a source of any numbers quoted in README/DOCS.

Moonshine, Kroko, Speechcatcher M/L, Pocket TTS, Kokoro ONNX, and
Supertonic 3 are unchanged; the defaults (`stt_engine: moonshine`,
`tts_engine: pocket_tts`) are unchanged for full backward compatibility.

See `homeintent-moonshine-stt/CHANGELOG.md` for full details.

## [0.5.0] - 2026-09-19

Adds two more user-selectable STT engines, **Speechcatcher M** and
**Speechcatcher L** (`speechcatcher_m`/`speechcatcher_l`), a streaming
Transformer ASR toolbox driven in-process via its own `Speech2TextStreaming`
object (real, block-granular incremental streaming decode -- not
`speechcatcher_server`'s websocket layer, which is itself only a thin
wrapper around that same object). Uses Speechcatcher's own current, stable
ESPnet-family decoder (`espnet_streaming_decoder`, a lighter, from-scratch
extraction of ESPnet's streaming code, not the full `espnet` package); its
newer "native" decoder was deliberately NOT used since upstream's own
README still labels it experimental. Speechcatcher has no hotword/HA-
vocabulary mechanism -- this is honestly reported via the `SttCapabilities`
abstraction, not faked. Models are downloaded on demand, only when
explicitly selected, into `/data/models/speechcatcher/`. Moonshine, Kroko,
Pocket TTS, Kokoro ONNX, and Supertonic 3 are unchanged; the default
(`stt_engine: moonshine`) is unchanged for full backward compatibility.

Speechcatcher and its own two git-only forked dependencies
(`espnet_streaming_decoder`, `espnet_model_zoo`) are not published on PyPI
and pull in a meaningfully larger dependency set than Kroko's single
`sherpa-onnx` wheel (Torch/torchaudio, librosa, numba, scikit-learn, h5py,
kaldiio, and others) -- see `homeintent-moonshine-stt/CHANGELOG.md` for the
full dependency/build-time breakdown.

See `homeintent-moonshine-stt/CHANGELOG.md` for full details.

## [0.4.0] - 2026-09-18

Adds a second, user-selectable STT engine (**Kroko**, via sherpa-onnx,
real streaming, German, Apache-2.0 model) and a third TTS engine
(**Supertonic 3**, via sherpa-onnx, real callback streaming, 10 voices,
MIT-licensed model), both run in-process behind a new generic
`SttEngine`/`SttSession` abstraction. Moonshine, Pocket TTS, and Kokoro
ONNX are unchanged; defaults (`stt_engine: moonshine`, `tts_engine:
pocket_tts`) are unchanged for full backward compatibility. New models are
downloaded on demand only when explicitly selected.

See `homeintent-moonshine-stt/CHANGELOG.md` for full details.

## [0.3.0] - 2026-09-18

Adds Kokoro German ONNX as a second, user-selectable local TTS engine
(voice "Martin", ONNX Runtime, CPU-only) alongside the existing Pocket
TTS. Integrated directly in-process (no second server/container). Default
engine remains `pocket_tts` for full backward compatibility with existing
installs; select `tts_engine: kokoro_onnx` to try the new engine. The
Kokoro model (~326MB, Apache-2.0) is downloaded on demand into persistent
storage only when selected, never baked into the image.

See `homeintent-moonshine-stt/CHANGELOG.md` and `ABSCHLUSSBERICHT_V0.3.0.md` for full details.

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
