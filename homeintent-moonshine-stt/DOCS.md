# HomeIntent Moonshine STT

Real-time German speech-to-text using the Moonshine streaming ASR model.

## Features

- **Streaming ASR**: Audio is processed in real-time as it arrives, not after recording ends
- **German Language**: Optimized for German speech recognition
- **Tiny & Small Models**: Choose between speed (Tiny, 34M params) or accuracy (Small, 123M params)
- **CPU-only**: Works on any hardware without GPU
- **Model Caching**: Downloaded models are cached to avoid re-downloads
- **Home Assistant Integration**: Automatic discovery for Assist pipelines

## Installation

1. Open Home Assistant **Settings** → **Add-ons** → **Add-on Store**
2. Click the menu button (⋮) in the top-right
3. Click **Repositories**
4. Add this repository URL: `https://github.com/pquandel2-alt/homeintent-moonshine-addon`
5. Click **Create** and wait for it to load
6. Go back to the Add-on Store
7. Find **HomeIntent Moonshine STT** and click it
8. Click **Install**

## Configuration

### Model Selection

- **Tiny** (34M): Faster inference, lower accuracy (~12% WER)
- **Small** (123M): Better accuracy (~7.5% WER) — Recommended

### Log Level

- **DEBUG**: Detailed logging (helpful for troubleshooting)
- **INFO**: Normal operation
- **WARNING**: Only important messages
- **ERROR**: Only errors

### Recognition tuning (keyterms and vocabulary)

- **use_ha_vocabulary** (default `true`): reads your Home Assistant Areas, Devices,
  Entities, and Floors (via the Supervisor-proxied Core API) and biases recognition
  towards your actual room/device names. Strictly read-only — never calls a service or
  changes a state. If Home Assistant can't be reached, the add-on logs a warning and
  still starts, using whatever keyterms it has.
- **ha_vocabulary_refresh_minutes** (default `30`): how often the vocabulary is
  re-read. `0` disables periodic refresh.
- **extra_keyterms** (default empty): your own comma-separated words/phrases, merged
  with the Home Assistant vocabulary and deduplicated, e.g. `Wohnzimmer, Rolladen`.
- **keyterm_boost** (default `2.0`): strength applied to keyterms.
- **transcription_interval** (default `0.5`): seconds between partial transcript
  updates.
- **vad_threshold** (default `0.5`): voice-activity-detection sensitivity.
- **decode_incomplete_lines** (default `true`): decode/emit lines still in progress for
  lower-latency partial results.

### Logging and debugging

- **log_transcripts** (default `false`): recognized text is never logged unless you
  turn this on.
- **log_performance** (default `true`): a compact per-utterance line (model, audio
  duration, processing time, real-time factor) — never includes transcript text.
- **save_debug_audio** (default `false`): saves received audio as WAV + JSON metadata
  to `/data/debug_audio` for troubleshooting. Metadata is limited to an explicit
  allowlist (timestamp, model, language, transcript, duration, sample rate) — never a
  Home Assistant entity state. Excluded from backups. Turn on only while debugging.
- **debug_audio_max_files** (default `100`): oldest-first retention limit for saved
  debug audio.

## First Start

The first time you start the add-on, it will download the selected model (~200MB for Small).
This may take a few minutes depending on your internet speed.

On subsequent starts, the cached model is used — no download needed.

## Using with Home Assistant Assist

1. Open **Settings** → **Voice assistants**
2. Click **Create assistant** or edit your existing one
3. In the STT (Speech-to-Text) section, look for **HomeIntent Moonshine STT**
4. Select it and ensure **German** is selected
5. Save and test by voice

## Troubleshooting

### No Model Available

If the add-on fails to start, check the logs. Common issues:

- **Network**: Model download failed. Check your internet and try restarting.
- **Storage**: Not enough space in `/data/models`. Free up space and restart.
- **Memory**: Moonshine needs ~500MB RAM. On Raspberry Pi, close other add-ons.

### Recognition Not Working

- Ensure **German** is selected in your Assist pipeline
- Check that your microphone is working (test with another app first)
- Switch to **Small** model if using **Tiny** (faster but less accurate)

## Performance Notes

- **Tiny model** (34M params): faster inference, ~12% WER (upstream-published, German)
- **Small model** (123M params): more compute per chunk, ~7.5% WER (upstream-published, German) — recommended for accuracy
- Real-time-factor and end-to-end latency depend heavily on your own CPU. Use
  `python -m app.benchmark <file.wav>` (in the add-on's source repository) to measure
  RTF on your own hardware — this is not run in CI and not the source of any number
  quoted here.

## Privacy

- Transcripts and debug audio are **off by default** — nothing beyond a compact
  performance-metrics line is logged unless you explicitly enable `log_transcripts` or
  `save_debug_audio`.
- `use_ha_vocabulary` only ever *reads* Home Assistant's registries — it never calls a
  service, changes a state, or edits an entity/automation.

## License

This add-on's code is licensed under the MIT License.

**The German Moonshine models are NOT MIT licensed.** They are published under
Moonshine's non-commercial Community License — free for researchers, developers, small
businesses, and creators with less than $1M in annual revenue. See
[moonshine.ai/license](https://www.moonshine.ai/license) for full terms before
commercial use.
