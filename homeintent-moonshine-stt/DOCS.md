# HomeIntent Moonshine Voice

Local German speech-to-text (Moonshine) and text-to-speech (Kyutai Pocket TTS), both
over one Wyoming service.

## Features

- **Streaming STT**: audio is transcribed in real-time as it arrives
- **Streaming TTS**: audio is sent back as it is generated, not after the whole reply
- **German language**: both STT and TTS
- **CPU-only**: no GPU required for either engine
- **Model caching**: STT and TTS models are cached separately, no repeated downloads
- **Home Assistant integration**: automatic discovery, advertises ASR and/or TTS
  depending on what's enabled

## Installation

1. Open Home Assistant **Settings** → **Add-ons** → **Add-on Store**
2. Click the menu button (⋮) in the top-right
3. Click **Repositories**
4. Add this repository URL: `https://github.com/pquandel2-alt/homeintent-moonshine-addon`
5. Click **Create** and wait for it to load
6. Go back to the Add-on Store
7. Find **HomeIntent Moonshine Voice** and click it
8. Click **Install**

## Configuration

### STT

Five STT engines are available, selected via **stt_engine** (this is the
final, complete STT lineup this add-on plans to ship):

| Engine | Value | Streaming | Hotwords/HA vocabulary |
|--------|-------|-----------|--------------------------|
| Moonshine (default) | `moonshine` | yes | yes |
| Kroko | `kroko` | yes | yes (sherpa-onnx hotwords) |
| Speechcatcher M | `speechcatcher_m` | yes, block-granular | no |
| Speechcatcher L | `speechcatcher_l` | yes, block-granular | no |
| Vosk German | `vosk_german` | yes (`AcceptWaveform`) | no (see below) |

- **stt_enabled** (default `true`): enable speech-to-text.
- **stt_engine** (default `moonshine`): `moonshine`, `kroko`, `speechcatcher_m`,
  `speechcatcher_l`, or `vosk_german`. **Must stay `moonshine` on upgrade**
  unless you explicitly opt in.
- **model**: `tiny` (34M, faster, ~12% WER) or `small` (123M, ~7.5% WER, recommended).
  Only relevant for `stt_engine: moonshine`.
- **language**: `de` (German only).
- **log_level**: `DEBUG`/`INFO`/`WARNING`/`ERROR`.
- **kroko_threads** (default `1`), **kroko_hotwords_score** (default `1.5`):
  sherpa-onnx tuning for the Kroko engine, ignored unless `stt_engine: kroko`.
- **speechcatcher_threads** (default `0` = PyTorch's own default),
  **speechcatcher_beam_size** (default `5`): tuning for Speechcatcher, ignored
  unless `stt_engine` is `speechcatcher_m`/`speechcatcher_l`. Speechcatcher has
  no hotword/HA-vocabulary mechanism -- `extra_keyterms`/HA vocabulary are
  silently ignored (logged once at startup) when it is selected.
- **vosk_german** has no tuning options of its own -- it is deliberately
  kept as simple as possible, matching its whole purpose as the lightest-
  weight, lowest-CPU/RAM STT option this add-on offers (~45MB model). Like
  Speechcatcher, it has no hotword/HA-vocabulary mechanism suitable for
  this add-on's use: Vosk's own grammar API is a hard closed-set
  restriction on recognition, not a soft bias, so wiring it in would break
  free-form recognition of anything outside the given phrase list --
  `extra_keyterms`/HA vocabulary are silently ignored (logged once at
  startup) when it is selected.

### TTS

Three TTS engines are available, selected via **tts_engine**:

| Engine | Value | Voices | Streaming |
|--------|-------|--------|-----------|
| Pocket TTS (default) | `pocket_tts` | 1 (`juergen`) | yes |
| Kokoro ONNX | `kokoro_onnx` | 1 (`martin`) | yes |
| Supertonic 3 | `supertonic_3` | 10 (`M1`-`M5`, `F1`-`F5`) | yes (native callback, via sherpa-onnx) |

- **tts_enabled** (default `false`): enable text-to-speech. Off by default on
  upgrade so an existing STT-only install doesn't suddenly download an extra runtime
  and model — see the README's Backward Compatibility section.
- **tts_engine** (default `pocket_tts`): `pocket_tts` (Kyutai Pocket TTS,
  PyTorch, voice `juergen`) or `kokoro_onnx` (German Kokoro-82M fine-tune,
  ONNX Runtime, voice `martin`, added in v0.3.0 to try for lower CPU
  time-to-first-audio -- benchmark both on your own hardware before
  switching a production setup, see Performance Notes). **Must stay
  `pocket_tts` on upgrade** unless you explicitly opt in -- an existing
  installation's persisted config predates this option entirely and must
  keep behaving exactly as before.
- **tts_model**: `german` (6 transformer layers, faster, default) or `german_24l`
  (24 layers, higher quality, slower). Pocket TTS only; ignored for `kokoro_onnx`.
- **tts_voice** (default `juergen`): the German preset voice. `juergen` is the only
  real German preset Pocket TTS ships. Validated (and cached) once at add-on
  startup -- an invalid voice name fails startup immediately with a clear error,
  instead of only on the first real synthesize request. Pocket TTS only.
- **kokoro_voice** (default `martin`): Kokoro ONNX voice. `martin` is the only
  voice this add-on's German model ships. Validated the same way as `tts_voice`.
  Ignored for `tts_engine: pocket_tts`.
- **kokoro_speed** (default `1.0`, range `0.5`-`2.0`): Kokoro's own hard-enforced
  speed range (verified from its source).
- **kokoro_sentence_pause** / **kokoro_clause_pause** (defaults `0.25`/`0.1`
  seconds): Kokoro's own real upstream defaults for the pause inserted after a
  sentence/clause.
- **supertonic_voice** (default `M1`): one of 10 built-in voices, `M1`-`M5`
  (male) or `F1`-`F5` (female); a numeric sid `0`-`9` also works. Ignored
  unless `tts_engine: supertonic_3`.
- **supertonic_speed** (default `1.0`, range `0.25`-`3.0`).
- **supertonic_steps** (default `8`): a dropdown of the only two values
  Supertonic's own docs document -- `8` (default, balanced) or `10`
  (higher quality, slower).
- **supertonic_threads** (default `1`): sherpa-onnx CPU threads.
- **tts_warmup** (default `true`): runs one discarded synthesis at startup so the
  first real request isn't slower than later ones. Pure performance optimization
  (a failure here is logged but never fails startup). Applies to whichever engine
  is selected.
- **tts_log_performance** (default `true`): logs a compact per-request timing
  breakdown (`engine=`, time-to-first-audio, `first_audio_to_ha`, lock-wait, model
  compute, Wyoming-send, wall time, model RTF, wall RTF). Never logs the
  synthesized text.
- **tts_threads** (default `0` = auto): CPU threads Pocket TTS's PyTorch backend
  uses (`torch.set_num_threads()`). Pocket TTS's own library forces this to `1`
  internally at import time (`torch.set_num_threads(1)` in its `tts_model`
  module) -- `0` leaves that untouched; a positive value overrides it. This is
  a plausible real lever for real-time-factor on a multi-core host, since
  Pocket TTS otherwise never uses more than one core by default. Ignored for
  `tts_engine: kokoro_onnx` (see `kokoro_threads`).
- **kokoro_threads** (default `0` = auto): ONNX Runtime `intra_op_num_threads`
  for Kokoro. `kokoro_onnx` itself builds its ONNX Runtime session with no
  threading configuration at all (verified from its source) -- this add-on
  builds the session itself with an explicit `SessionOptions` to make this
  configurable. Ignored for `tts_engine: pocket_tts`.
- Benchmark either engine's thread setting on your own hardware with
  `python -m app.tts_benchmark --engine pocket_tts` /
  `--engine kokoro_onnx` (sweeps `1,2,4,6,8,auto` by default -- see
  Performance Notes) before changing it -- too high a value on a shared
  host can slow STT and TTS down together instead of speeding TTS up (CPU
  oversubscription).

### Recognition tuning (STT keyterms and vocabulary)

- **use_ha_vocabulary** (default `true`): automatically takes over the vocabulary of
  entities you've exposed to Home Assistant **Assist** (Settings → Voice assistants
  → Expose). For every effectively Assist-exposed entity, it biases recognition
  towards its friendly name, aliases, and area, including "Area Entity"/"Area Alias"
  combinations (e.g. `cover.wohnzimmer_rolllade` with alias "Rollo" in area
  "Wohnzimmer" → `Rolllade`, `Rollo`, `Wohnzimmer Rolllade`, `Wohnzimmer Rollo`).
  Entities not exposed to Assist (e.g. an unexposed `sensor.router_cpu_temperature`)
  contribute nothing. Newly exposed entities are picked up at the next refresh;
  removing an exposure makes its terms disappear after the next successful refresh.
  Effective exposure is computed the same way Home Assistant Core itself does (an
  explicit override always wins, otherwise Core's own default-exposure rule).
  Strictly read-only. If Home Assistant can't be reached on a periodic refresh, the
  add-on keeps the last successfully loaded vocabulary rather than dropping it.
  On larger installations, this used to fail entirely with a WebSocket
  "message too big" error (fixed in v0.2.4 -- see CHANGELOG); a real,
  larger installation should now get its vocabulary correctly.
- **Legacy (non-registry) entities**: entities with no Home Assistant entity
  registry entry at all are handled conservatively. Home Assistant provides no
  read-only API that can tell an explicit "hide from Assist" apart from "never
  evaluated" for such an entity, so it is only ever added to the automatic
  vocabulary (using its live friendly name, never an Area combination since it
  has no registry entry to resolve an area from) when its Assist exposure can be
  positively confirmed. An ambiguous legacy entity is simply left out — this
  add-on never guesses either way, to avoid silently re-including an entity you
  explicitly removed from Assist.
- **ha_vocabulary_refresh_minutes** (default `30`): how often the vocabulary is
  re-read. `0` disables periodic refresh.
- **extra_keyterms** (default empty): your own comma-separated words/phrases, merged
  with the Home Assistant vocabulary and deduplicated.
- **keyterm_boost** (default `2.0`): strength applied to keyterms.
- **Keyterm safety**: every keyterm (from Home Assistant or `extra_keyterms`)
  is validated against the loaded Moonshine model before use. If the model
  rejects a name outright (e.g. because of an invisible formatting
  character, an emoji, or a separator like `/`), the add-on first tries a
  cleaned-up, still-German variant (e.g. `Treppe/Büro` -> `Treppe Büro`)
  and re-validates that against the model too -- only a name that fails
  both attempts is skipped and logged as a warning. STT always comes up,
  with or without full keyterm biasing.
- **transcription_interval** (default `0.5`): seconds between partial transcript
  updates.
- **vad_threshold** (default `0.5`): voice-activity-detection sensitivity.
- **decode_incomplete_lines** (default `true`): decode/emit lines still in progress
  for lower-latency partial results.

### Logging and debugging

- **log_transcripts** (default `false`): recognized text is never logged unless you
  turn this on.
- **log_performance** (default `true`): a compact per-utterance STT line (model, audio
  duration, real inference time, finalize time, RTF) — never includes transcript text.
- **save_debug_audio** (default `false`): saves received STT audio as WAV + JSON
  metadata to `/data/debug_audio` for troubleshooting. Excluded from backups.

## First Start

- STT: downloads the selected Moonshine model (~200MB for Small) if enabled.
- TTS: downloads the Pocket TTS model + configured voice's embedding, if enabled.

Both are cached under `/data/` — subsequent starts are fast.

## Using with Home Assistant Assist

1. Open **Settings** → **Voice assistants**
2. Click **Create assistant** or edit your existing one
3. Select **HomeIntent Moonshine** as speech-to-text (if `stt_enabled`)
4. Select **HomeIntent Pocket TTS** as text-to-speech (if `tts_enabled`)
5. Ensure **German** is selected
6. Save and test by voice

## Troubleshooting

### Add-on Fails to Start

- **Network**: Model download failed (STT or TTS). Check your internet and restart.
- **Storage**: Not enough space in `/data/models`. Free up space and restart.
- **Memory**: Moonshine needs ~500MB RAM; Pocket TTS is a ~100M-parameter model on
  top of that if TTS is enabled. On low-RAM hardware, use the Tiny STT model and the
  `german` (6-layer, not `german_24l`) TTS model.
- **"Failed to load Pocket TTS model"**: a TTS load failure is treated as fatal (not
  a silent fallback) so you notice it immediately — check the exact error in the
  logs, or set `tts_enabled: false` temporarily to keep STT working.

### Recognition/Synthesis Not Working

- Ensure **German** is selected in your Assist pipeline
- Check that your microphone is working (test with another app first) for STT
- Check `tts_enabled: true` is actually set if you expect a TTS voice to appear

## Performance Notes

- **STT Tiny model** (34M params): faster inference, ~12% WER (upstream-published, German)
- **STT Small model** (123M params): more compute per chunk, ~7.5% WER (upstream-published,
  German) — recommended for accuracy
- **TTS german model**: faster, recommended default
- **TTS german_24l model**: higher quality, slower
- Real-time-factor and end-to-end latency for both STT and TTS depend heavily on your
  own CPU. Use `python -m app.benchmark <file.wav> --models tiny,small` for STT RTF
  (also supports `--transcription-interval`, `--decode-incomplete-lines`,
  `--keyterm-count`, and `--output json`/`csv` for comparing configurations), and
  `tts_log_performance: true`'s log line for TTS TTFA/RTF, to measure your own
  hardware — no number in this document has been measured on this add-on's own
  target hardware; see `ABSCHLUSSBERICHT_V0.2.0.md`/`ABSCHLUSSBERICHT_V0.2.4.md`
  for exactly what has and hasn't been measured.
- **STT thread tuning**: investigated (v0.2.4) and confirmed NOT available —
  moonshine-voice==0.1.5's native transcriber only recognizes `vad_threshold`,
  `decode_incomplete_lines`, `keyterm_boost`, and `spelling_model_path` as
  options (verified by inspecting its compiled library); there is no
  `num_threads`/intra-op/inter-op equivalent, so this add-on does not expose
  an `stt_threads` option or set `OMP_NUM_THREADS`-style environment
  variables that would have no documented effect.
- **TTS thread tuning**: Pocket TTS's own library forces single-threaded
  PyTorch CPU execution (`torch.set_num_threads(1)`) by default, which is a
  real, plausible reason for its real-time-factor being consistently above
  1.0 on production hardware. `tts_threads` (see Configuration above) is the
  only way to override this. Benchmark different values on your own target
  hardware with `python -m app.tts_benchmark --threads 1,2,4,6,8,auto`
  before changing the default -- each value is measured in its own
  subprocess (a clean `torch.set_num_threads()` per run), reporting
  time-to-first-audio, model compute time, and real-time factor.
- **Streaming to Home Assistant**: since v0.2.7, the add-on advertises and
  implements Wyoming's `synthesize-start`/`-chunk`/`-stop`/`-stopped`
  streaming protocol, so Home Assistant plays audio as soon as Pocket TTS
  produces its first chunk instead of buffering the entire response first.
  This was the actual, verified cause of most of the previously perceived
  TTS latency -- not Pocket TTS's own compute speed.

## Privacy

- Transcripts, synthesized text, and debug audio are **off by default** — nothing
  beyond compact performance-metrics lines is logged unless explicitly enabled.
- `use_ha_vocabulary` only ever *reads* Home Assistant's registries (entity names,
  aliases, device names, area names, and Assist exposure) — it never calls a
  service, changes a state, reads sensor values, stores states, or edits an
  entity/automation/Assist exposure setting.

## License

This add-on's code, and Pocket TTS's own code, are licensed under the MIT License.

**Model weights are not all MIT and differ between STT and TTS:**
- The German Moonshine STT models are published under Moonshine's non-commercial
  Community License — free for researchers, developers, small businesses, and
  creators with less than $1M in annual revenue. See
  [moonshine.ai/license](https://www.moonshine.ai/license).
- Pocket TTS's German model weights carry Kyutai's own usage restrictions (no voice
  impersonation/cloning without consent, no misinformation/deception uses, see the
  README) — the exact Hugging Face license field could not be verified during this
  add-on's development (huggingface.co was unreachable from the development
  environment); review it yourself before commercial use.
- Kroko German STT model weights: Apache License 2.0 (Banafo AI's
  Kroko-ASR, via sherpa-onnx).
- Speechcatcher's own code (`speechcatcher_m`/`speechcatcher_l`): MIT License.
  The model checkpoints' own Hugging Face Hub license field could not be
  verified during this add-on's development (huggingface.co was unreachable
  from the development environment); review it yourself before commercial use.
- Supertonic 3 TTS model weights: MIT License (Supertone Inc.).
- `sherpa-onnx` itself (runtime for Kroko + Supertonic 3): Apache-2.0.
- Vosk German STT model weights (`vosk_german`): Apache License 2.0
  (AlphaCephei's `vosk-model-small-de-0.15`).
