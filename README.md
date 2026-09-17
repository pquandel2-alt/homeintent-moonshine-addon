# HomeIntent Moonshine STT

Real-time German speech-to-text for Home Assistant using Moonshine streaming ASR.

![Code License](https://img.shields.io/badge/code%20license-MIT-green)
![Model License](https://img.shields.io/badge/model%20license-non--commercial-orange)
![Python](https://img.shields.io/badge/python-3.11+-blue)
![Home Assistant](https://img.shields.io/badge/home%20assistant-2023.11+-brightgreen)

## What is This?

This is a **Home Assistant Add-on** that provides offline, real-time German speech recognition for your Home Assistant instance.

**Key Features:**
- ✅ **Real-time Streaming**: Audio is processed as it arrives, not after recording ends
- ✅ **German Language**: Optimized for German speech (Deutsch)
- ✅ **Tiny & Small Models**: Choose between speed or accuracy
- ✅ **CPU-only**: Works on any hardware (Raspberry Pi, NAS, etc.)
- ✅ **Offline**: No cloud API, everything runs locally
- ✅ **Open Source**: Add-on code is MIT licensed (see [Model License](#license) for the German model weights)

## Why This Add-on?

There are other Moonshine add-ons for Home Assistant, but this one has a critical difference:

**Streaming Architecture:**

Old approach (cronus42):
```
audio-chunk → buffer audio
audio-chunk → buffer audio
audio-chunk → buffer audio
audio-stop  → transcribe entire buffer at once
```

Our approach (HomeIntent Moonshine STT):
```
audio-chunk → process immediately
audio-chunk → process immediately  
audio-chunk → process immediately
audio-stop  → finalize and return
```

The model starts processing while you're still speaking, instead of waiting until you stop.

## Installation

### Via Home Assistant Add-on Store (Recommended)

1. Open Home Assistant **Settings** → **Add-ons** → **Add-on Store**
2. Click the menu (⋮) in the top-right corner
3. Select **Repositories**
4. Add this repository URL:
   ```
   https://github.com/pquandel2-alt/homeintent-moonshine-addon
   ```
5. Click **Create** and wait for it to load
6. Go back to the Add-on Store tab
7. Look for **HomeIntent Moonshine STT**
8. Click it and select **Install**
9. Wait for the installation to complete
10. Click **Start** to run the add-on

### First Start

The first time you start the add-on, it will download the Moonshine model (~200MB for Small, ~80MB for Tiny).
This may take a few minutes depending on your internet speed.

**Subsequent starts are fast** — the model is cached locally.

## Configuration

The add-on is configured via Home Assistant's UI:

**Settings → Add-ons → HomeIntent Moonshine STT → Configuration**

### Options

| Option | Values | Default | Description |
|--------|--------|---------|-------------|
| **model** | `tiny`, `small` | `small` | Model size. Tiny = faster but less accurate, Small = better accuracy (recommended) |
| **language** | `de` | `de` | Language (German only in v0.1) |
| **log_level** | `DEBUG`, `INFO`, `WARNING`, `ERROR` | `INFO` | Logging verbosity |
| **log_transcripts** | `true`/`false` | `false` | Log recognized text. **Off by default** — enable only if you need to debug what was actually recognized |
| **log_performance** | `true`/`false` | `true` | Log a compact per-utterance line (model, audio duration, processing time, real-time factor). Never includes transcript text |
| **use_ha_vocabulary** | `true`/`false` | `true` | Read your Home Assistant Areas/Devices/Entities/Floors (via the Supervisor-proxied Core API) and bias recognition towards your actual room and device names. **Strictly read-only** — no service is ever called, no state is ever changed. If Home Assistant is unreachable, the add-on logs a warning and starts anyway with whatever keyterms it has |
| **ha_vocabulary_refresh_minutes** | `0`–`1440` | `30` | How often to re-read the HA registries and refresh keyterms. `0` disables periodic refresh (read once at startup) |
| **extra_keyterms** | comma-separated text | `""` | Additional words/phrases to bias recognition towards, merged with the Home Assistant vocabulary and deduplicated. Example: `Wohnzimmer, Kaffeemaschine, Rolladen` |
| **keyterm_boost** | `0.0`–`10.0` | `2.0` | Strength applied to keyterms (matches Moonshine's own default) |
| **transcription_interval** | `0.05`–`10.0` | `0.5` | How often (in seconds) partial transcription updates are computed |
| **vad_threshold** | `0.0`–`1.0` | `0.5` | Voice-activity-detection sensitivity (Moonshine's own default) |
| **decode_incomplete_lines** | `true`/`false` | `true` | Whether to decode and emit lines that haven't finished yet, for lower-latency partial results |
| **save_debug_audio** | `true`/`false` | `false` | Save received audio to `/data/debug_audio` as WAV + JSON metadata for troubleshooting. **Off by default** — see [Privacy](#privacy) below before enabling |
| **debug_audio_max_files** | `1`–`10000` | `100` | Oldest-first retention limit for saved debug audio files |

### Example Configuration

For best accuracy, using your Home Assistant vocabulary:
```yaml
model: small
language: de
log_level: INFO
use_ha_vocabulary: true
keyterm_boost: 2.0
```

For faster inference on slower hardware:
```yaml
model: tiny
language: de
log_level: INFO
```

Adding your own keyterms on top of (or instead of) the Home Assistant vocabulary:
```yaml
model: small
language: de
use_ha_vocabulary: true
extra_keyterms: "Wohnzimmer, Kaffeemaschine, Rolladen"
keyterm_boost: 3.0
```

Troubleshooting a misrecognition (temporarily):
```yaml
model: small
language: de
log_level: DEBUG
log_transcripts: true
save_debug_audio: true
debug_audio_max_files: 20
```

## Using with Home Assistant Assist

Once the add-on is running, integrate it with Home Assistant Assist:

1. Open **Settings** → **Voice assistants**
2. Create a new assistant or edit your existing one
3. In the **Speech-to-text** section, select **HomeIntent Moonshine STT**
4. Ensure the language is set to **German** (de)
5. Save

Now use voice commands with the Home Assistant voice assistant!

### Testing

To test microphone input:
1. Create a test script or automation
2. Use the voice assistant to trigger it
3. Check the add-on logs for transcript output

## Performance

### Model Sizes

| Model | Parameters | File Size | WER | Use Case |
|-------|-----------|-----------|-----|----------|
| **Tiny** | 34M | ~80MB | 12.0% | Low-end hardware, fast feedback |
| **Small** | 123M | ~200MB | 7.5% | Recommended for most setups |

WER (Word Error Rate) figures are upstream-published values for German. Latency and
real-time-factor depend heavily on your own CPU, so no numbers are quoted here. If you
want to measure real-time factor on your own hardware, use the bundled benchmark CLI:

```bash
python -m app.benchmark path/to/your-sample.wav --model small --language de
```

This runs locally against a real model and prints your own RTF — it is not run in CI and
is not the source of any number in this README.

## Privacy

- **Transcripts are never logged by default** (`log_transcripts: false`). Enable it only
  when you need to debug what was actually recognized.
- **Performance logging never includes transcript text** — only a compact
  model/duration/processing-time/RTF line.
- **`use_ha_vocabulary` is strictly read-only.** It only issues
  `config/*_registry/list` calls against the Home Assistant Core API to read Area,
  Device, Entity, and Floor names for keyterm biasing. It never calls a service, changes
  a state, or edits an entity/automation. If Home Assistant is unreachable, the add-on
  logs a warning and keeps running with whatever keyterms it already has.
- **`save_debug_audio` is off by default.** When enabled, it writes raw audio (WAV) plus
  a JSON metadata file to `/data/debug_audio`. Metadata is limited to an explicit
  allowlist (timestamp, model, language, transcript, duration, sample rate) — it never
  contains a Home Assistant entity state. Files are excluded from HA backups
  (`backup_exclude`) and pruned oldest-first once `debug_audio_max_files` is reached.
  Only turn this on temporarily while troubleshooting.

## Troubleshooting

### Add-on Won't Start

Check the logs (**Add-ons → HomeIntent Moonshine STT → Logs**):

**"Model download failed"**
- Check your internet connection
- Ensure `/data/` has at least 500MB free space
- Try restarting the add-on

**"Memory error"**
- Close other add-ons to free RAM
- Try the Tiny model (uses less memory)
- If on Raspberry Pi 3 or earlier: Tiny model is required

**Native library / model load errors**
- This is a container build issue. Try stopping and restarting the add-on
- If persistent, report an issue

### Recognition Not Working

**"No service available"** in Assist pipeline:
- Wait 30-60 seconds after starting the add-on (model loading)
- Restart the add-on
- Check that the port (10300) isn't blocked

**Inaccurate transcripts:**
- Try the **Small** model (better accuracy)
- Speak clearly in German
- Check microphone is working (test with another app first)

**Only partial transcripts shown:**
- Some text may come as partial results — this is normal
- Final result is always sent when audio stops

### Performance Issues

If transcription is slow:

1. **Check system load** — close other processes
2. **Use Tiny model** — faster inference
3. **Check logs for errors** — Look for CPU usage spikes
4. **Reduce quality** — Use Tiny model instead of Small

## Roadmap

### v0.1 / v0.1.1
- ✅ Real-time streaming ASR
- ✅ German language support
- ✅ Tiny + Small models
- ✅ CPU-only inference
- ✅ Model caching
- ✅ Home Assistant Assist integration
- ✅ Verified Docker build + container smoke test

### v0.1.2 (Current)
- ✅ Automatic Home Assistant vocabulary import (Areas/Devices/Entities/Floors,
  read-only, graceful degradation)
- ✅ Manual `extra_keyterms` + verified `keyterm_boost`
- ✅ Real Moonshine streaming parameters exposed as options (VAD threshold,
  transcription interval, incomplete-line decoding)
- ✅ Optional transcript logging, performance metrics logging, debug audio recording
  (all privacy-conscious, off/limited by default)
- ✅ Local RTF benchmark CLI
- ✅ Real end-to-end Wyoming round-trip test with synthesized German audio in CI
- ✅ Hardened session lifecycle (duplicate events, disconnects, concurrent sessions)

### v0.2 (Future)
- [ ] Fine-tuning / adaptation on real usage data (explicitly out of scope for v0.1.2)
- [ ] Device-specific adaptation

### v0.3 (Future)
- [ ] Integration with homeintent-stt dataset
- [ ] Custom model benchmark comparisons

### v0.4 (Future)
- [ ] Home Assistant domain-specific fine-tuned model

## Architecture

```
                    Home Assistant
                          ↓
                   Assist Pipeline
                          ↓
                   Wyoming Protocol
                    (TCP Port 10300)
                          ↓
              HomeIntent Moonshine STT
                  (This Add-on)
                          ↓
               Moonshine Streaming ASR
                          ↓
              Local CPU Inference (native)
                          ↓
                  Transcript Result
```

## Development

### Local Testing

Clone the repository:
```bash
git clone https://github.com/pquandel2-alt/homeintent-moonshine-addon.git
cd homeintent-moonshine-addon
```

Install dependencies:
```bash
cd homeintent-moonshine-stt/app
pip install -r ../requirements.txt -r requirements-dev.txt
```

Run tests:
```bash
pytest tests/ -v
```

Run linting:
```bash
ruff check .
mypy . --ignore-missing-imports
```

## Documentation

- **[DOCS.md](homeintent-moonshine-stt/DOCS.md)** — In-app documentation
- **[CHANGELOG.md](homeintent-moonshine-stt/CHANGELOG.md)** — Version history
- **[Moonshine Repo](https://github.com/moonshine-ai/moonshine)** — Model documentation
- **[Wyoming Protocol](https://github.com/rhasspy/wyoming)** — Integration protocol

## License

This add-on's code is licensed under the MIT License — see [LICENSE](LICENSE).

**The German Moonshine models used by this add-on are NOT MIT licensed.** Moonshine
publishes German (and other non-English) models under its non-commercial **Community
License** — free for researchers, developers, small businesses, and creators with less
than $1M in annual revenue; commercial use beyond that requires Moonshine's Enterprise
license. Only Moonshine's English models are MIT. See
[moonshine.ai/license](https://www.moonshine.ai/license) for the full terms before using
this add-on in a commercial context.

## Contributing

Issues and pull requests are welcome!

Please ensure:
- Code passes `ruff check` and `mypy --ignore-missing-imports`
- Tests pass: `pytest tests/ -v`
- Commit messages are clear and descriptive

## Credits

- **Moonshine ASR**: https://github.com/moonshine-ai/moonshine
- **Wyoming Protocol**: https://github.com/rhasspy/wyoming
- **Home Assistant**: https://www.home-assistant.io

## Support

If you encounter issues:

1. **Check the logs** — Add-ons → HomeIntent Moonshine STT → Logs
2. **Review the [Troubleshooting](#troubleshooting) section**
3. **Open an issue** — https://github.com/pquandel2-alt/homeintent-moonshine-addon/issues

---

**Made with ❤️ for the Home Assistant community**
