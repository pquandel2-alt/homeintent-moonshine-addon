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

- **stt_enabled** (default `true`): enable Moonshine speech-to-text.
- **model**: `tiny` (34M, faster, ~12% WER) or `small` (123M, ~7.5% WER, recommended).
- **language**: `de` (German only).
- **log_level**: `DEBUG`/`INFO`/`WARNING`/`ERROR`.

### TTS

- **tts_enabled** (default `false`): enable Kyutai Pocket TTS. Off by default on
  upgrade so an existing STT-only install doesn't suddenly download an extra runtime
  and model — see the README's Backward Compatibility section.
- **tts_model**: `german` (6 transformer layers, faster, default) or `german_24l`
  (24 layers, higher quality, slower).
- **tts_voice** (default `juergen`): the German preset voice. `juergen` is the only
  real German preset Pocket TTS ships.
- **tts_log_performance** (default `true`): logs a compact TTFA/synthesis-time/RTF
  line per request. Never logs the synthesized text.

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
- **ha_vocabulary_refresh_minutes** (default `30`): how often the vocabulary is
  re-read. `0` disables periodic refresh.
- **extra_keyterms** (default empty): your own comma-separated words/phrases, merged
  with the Home Assistant vocabulary and deduplicated.
- **keyterm_boost** (default `2.0`): strength applied to keyterms.
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
  own CPU. Use `python -m app.benchmark <file.wav>` for STT RTF, and
  `tts_log_performance: true`'s log line for TTS TTFA/RTF, to measure your own
  hardware — no number in this document has been measured on this add-on's own
  target hardware; see `ABSCHLUSSBERICHT_V0.2.0.md` for exactly what has and hasn't
  been measured.

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
