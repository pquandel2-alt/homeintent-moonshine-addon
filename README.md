# HomeIntent Moonshine STT

Real-time German speech-to-text for Home Assistant using Moonshine streaming ASR.

![License](https://img.shields.io/badge/license-MIT-green)
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
- ✅ **Open Source**: MIT License

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

This means **latency is reduced dramatically** — the model starts processing while you're still speaking.

## Installation

### Via Home Assistant Add-on Store (Recommended)

1. Open Home Assistant **Settings** → **Add-ons** → **Add-on Store**
2. Click the menu (⋮) in the top-right corner
3. Select **Repositories**
4. Add this repository URL:
   ```
   https://github.com/pquandel/homeintent-moonshine-addon
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

### Example Configuration

For best accuracy:
```yaml
model: small
language: de
log_level: INFO
```

For faster inference on slower hardware:
```yaml
model: tiny
language: de
log_level: INFO
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

| Model | Parameters | File Size | WER | Latency | Use Case |
|-------|-----------|-----------|-----|---------|----------|
| **Tiny** | 34M | ~80MB | 12.0% | Very fast | Low-end hardware, fast feedback |
| **Small** | 123M | ~200MB | 7.5% | Fast | Recommended for most setups |

### Latency Expectations

- **Model Load**: ~5-10 seconds (first start only)
- **Audio Stream**: ~200-500ms from end of speech to transcript

This is the inherent latency of speech recognition. The streaming architecture minimizes buffering overhead.

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

**"ONNX Runtime not found"**
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

### v0.1 (Current)
- ✅ Real-time streaming ASR
- ✅ German language support
- ✅ Tiny + Small models
- ✅ CPU-only inference
- ✅ Model caching
- ✅ Home Assistant Assist integration

### v0.2 (Planned)
- [ ] Keyterms/context biasing
- [ ] Partial transcript streaming
- [ ] Performance metrics (latency, RTF)
- [ ] Better logging

### v0.3 (Future)
- [ ] Automatic Home Assistant vocabulary import
- [ ] Custom entity/area name recognition
- [ ] Device-specific adaptation

### v0.4 (Future)
- [ ] Integration with homeintent-stt dataset
- [ ] Custom model benchmark

### v0.5 (Future)
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
            Local CPU Inference (ONNX)
                          ↓
                  Transcript Result
```

## Development

### Local Testing

Clone the repository:
```bash
git clone https://github.com/pquandel/homeintent-moonshine-addon.git
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

This add-on is licensed under the MIT License — see [LICENSE](LICENSE).

Moonshine models are also MIT licensed.

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
3. **Open an issue** — https://github.com/pquandel/homeintent-moonshine-addon/issues

---

**Made with ❤️ for the Home Assistant community**
