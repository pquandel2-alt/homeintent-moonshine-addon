# HomeIntent Moonshine Voice

Local, offline German voice for Home Assistant: real-time streaming speech-to-text
(Moonshine) and streaming text-to-speech (Kyutai Pocket TTS), both served over one
Wyoming TCP service.

![Code License](https://img.shields.io/badge/code%20license-MIT-green)
![STT Model License](https://img.shields.io/badge/STT%20model-non--commercial-orange)
![TTS Code License](https://img.shields.io/badge/TTS%20code-MIT-green)
![Python](https://img.shields.io/badge/python-3.11+-blue)
![Home Assistant](https://img.shields.io/badge/home%20assistant-2023.11+-brightgreen)

## What is This?

This is a **Home Assistant Add-on** that provides offline German speech recognition
*and* offline German speech synthesis, entirely on your own hardware.

**Key Features:**
- ✅ **Real-time STT streaming**: audio is transcribed as it arrives, not after recording ends
- ✅ **Real-time TTS streaming**: audio is sent back as it is generated, not after the whole reply is synthesized
- ✅ **German language**: both STT and TTS are German-first
- ✅ **CPU-only**: no GPU required for either engine (Raspberry Pi, NAS, etc.)
- ✅ **Offline**: no cloud API, no API keys, everything runs locally
- ✅ **Not Piper**: TTS is [Kyutai's Pocket TTS](https://github.com/kyutai-labs/pocket-tts), a modern, more natural-sounding local voice — not one of Home Assistant's default Piper voices
- ✅ **Home Assistant Assist**: works as both the STT and TTS engine in an Assist pipeline
- ✅ Add-on code is MIT licensed (see [License](#license) for the STT/TTS *model* licenses, which differ from the code)

TTS is **off by default** on upgrade so existing STT-only installations don't suddenly
download an extra runtime — see [Configuration](#configuration) and
[Backward compatibility](#backward-compatibility).

## Why This Add-on?

**Streaming STT**, unchanged since v0.1:
```
audio-chunk → process immediately
audio-chunk → process immediately
audio-chunk → process immediately
audio-stop  → finalize and return
```
The model starts transcribing while you're still speaking, instead of waiting until you stop.

**Streaming TTS**, new in v0.2.0:
```
synthesize (text) → Pocket TTS starts generating
                  → first audio-chunk sent as soon as it's ready
                  → more audio-chunks sent while synthesis continues
                  → audio-stop once done
```
Home Assistant's voice satellite can start playing your reply before the whole
sentence has finished synthesizing — see [Performance](#performance) for what
"as soon as it's ready" actually means for Pocket TTS.

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
7. Look for **HomeIntent Moonshine Voice**
8. Click it and select **Install**
9. Wait for the installation to complete
10. Click **Start** to run the add-on

If you already have the add-on installed (previously "HomeIntent Moonshine STT"),
this is a normal update — the add-on's internal slug did not change, so your
existing configuration carries over. See [Manual steps after updating](#manual-steps-after-updating).

### First Start

- **STT only** (the default): downloads the Moonshine model (~200MB for Small,
  ~80MB for Tiny) on first start.
- **STT + TTS**: also downloads the Pocket TTS model (~100M-parameter model,
  see [Performance](#performance) for what's actually been measured) and the
  configured voice's embedding, the first time TTS is used/started.

**Subsequent starts are fast** — both model caches are persisted under `/data/`.

## Configuration

The add-on is configured via Home Assistant's UI:

**Settings → Add-ons → HomeIntent Moonshine Voice → Configuration**

### STT Options

| Option | Values | Default | Description |
|--------|--------|---------|-------------|
| **stt_enabled** | `true`/`false` | `true` | Enable Moonshine speech-to-text |
| **model** | `tiny`, `small` | `small` | STT model size. Tiny = faster but less accurate, Small = better accuracy (recommended) |
| **language** | `de` | `de` | Language (German only) |
| **log_level** | `DEBUG`, `INFO`, `WARNING`, `ERROR` | `INFO` | Logging verbosity |
| **log_transcripts** | `true`/`false` | `false` | Log recognized text. **Off by default** — enable only if you need to debug what was actually recognized |
| **log_performance** | `true`/`false` | `true` | Log a compact per-utterance STT timing line (model, audio duration, inference time, RTF). Never includes transcript text |
| **use_ha_vocabulary** | `true`/`false` | `true` | Automatically bias recognition towards the entities you've actually exposed to Home Assistant Assist — names, aliases, and "Area Entity"/"Area Alias" combinations like "Wohnzimmer Rolllade" or "Wohnzimmer Rollo" where a real relationship is known. Entities not exposed to Assist (e.g. an unexposed `sensor.router_cpu_temperature`) contribute no vocabulary. Entities with no Home Assistant entity registry entry at all ("legacy" entities) are included only when their Assist exposure can be positively confirmed — see [Privacy](#privacy) for why this is intentionally conservative. **Strictly read-only.** If Home Assistant is unreachable, the add-on keeps using the last successfully loaded vocabulary rather than dropping it |
| **ha_vocabulary_refresh_minutes** | `0`–`1440` | `30` | How often to re-read the HA registries and refresh keyterms. `0` disables periodic refresh (read once at startup) |
| **extra_keyterms** | comma-separated text | `""` | Additional words/phrases to bias recognition towards, merged with the Home Assistant vocabulary and deduplicated |
| **keyterm_boost** | `0.0`–`10.0` | `2.0` | Strength applied to keyterms (matches Moonshine's own default) |
| **transcription_interval** | `0.05`–`10.0` | `0.5` | How often (in seconds) partial transcription updates are computed |
| **vad_threshold** | `0.0`–`1.0` | `0.5` | Voice-activity-detection sensitivity (Moonshine's own default) |
| **decode_incomplete_lines** | `true`/`false` | `true` | Whether to decode and emit lines that haven't finished yet, for lower-latency partial results |
| **save_debug_audio** | `true`/`false` | `false` | Save received audio to `/data/debug_audio` as WAV + JSON metadata for troubleshooting. **Off by default** — see [Privacy](#privacy) |
| **debug_audio_max_files** | `1`–`10000` | `100` | Oldest-first retention limit for saved debug audio files |

### TTS Options

Two TTS engines are available, selected with **tts_engine**:

- **Pocket TTS** (`pocket_tts`, the default): [Kyutai's Pocket TTS](https://github.com/kyutai-labs/pocket-tts),
  PyTorch-based, voice **juergen**. Kept as the default for full backward
  compatibility — an existing installation's behavior never changes on
  upgrade.
- **Kokoro ONNX** (`kokoro_onnx`): a German fine-tune of
  [Kokoro-82M](https://huggingface.co/hexgrad/Kokoro-82M) running on
  [ONNX Runtime](https://onnxruntime.ai/) instead of PyTorch, voice
  **Martin**. No GPU needed (same as Pocket TTS). Added in v0.3.0
  specifically to try for lower CPU time-to-first-audio than Pocket TTS —
  see [Performance](#performance) for what has actually been measured on
  this add-on's own target hardware, and benchmark both yourself with
  `python -m app.tts_benchmark --engine pocket_tts` /
  `--engine kokoro_onnx` before switching a production setup.

| Option | Values | Default | Description |
|--------|--------|---------|-------------|
| **tts_enabled** | `true`/`false` | `false` | Enable text-to-speech (engine selected via `tts_engine`). Off by default on upgrade — see [Backward compatibility](#backward-compatibility) |
| **tts_engine** | `pocket_tts`, `kokoro_onnx` | `pocket_tts` | Which TTS engine to use. **Must stay `pocket_tts` on upgrade** unless you explicitly opt in — see above |
| **tts_model** | `german`, `german_24l` | `german` | Pocket TTS model: `german` (6 transformer layers) is faster and is the default; `german_24l` (24 layers) is higher quality but slower. Ignored for `tts_engine: kokoro_onnx` |
| **tts_voice** | voice name or `hf://...` path | `juergen` | Pocket TTS voice. `juergen` is the only real German preset voice Pocket TTS ships. Ignored for `tts_engine: kokoro_onnx` (see `kokoro_voice`) |
| **tts_log_performance** | `true`/`false` | `true` | Log a compact per-request timing breakdown (`engine=`, TTFA, lock-wait, model compute, Wyoming-send, wall time, model RTF, wall RTF). Never includes the synthesized text |
| **tts_warmup** | `true`/`false` | `true` | Run one discarded synthesis at startup so the first real TTS request isn't slower than later ones. Also validates the configured voice at startup — an invalid voice fails the add-on immediately with a clear error, instead of only on the first real request |
| **tts_threads** | `0`–`64` | `0` | Pocket TTS: PyTorch intra-op CPU threads (`0` = PyTorch's own default). Ignored for `tts_engine: kokoro_onnx` (see `kokoro_threads`) |
| **kokoro_voice** | voice name | `martin` | Kokoro ONNX voice. `martin` is the only voice this add-on's German model ships. Ignored for `tts_engine: pocket_tts` |
| **kokoro_speed** | `0.5`–`2.0` | `1.0` | Kokoro ONNX speech speed (kokoro-onnx's own valid range) |
| **kokoro_threads** | `0`–`64` | `0` | ONNX Runtime intra-op CPU threads (`0` = onnxruntime's own default) |
| **kokoro_sentence_pause** | seconds | `0.25` | Pause after a sentence (kokoro-onnx's own default) |
| **kokoro_clause_pause** | seconds | `0.1` | Pause after a clause (kokoro-onnx's own default) |

Picking an unsupported combination (e.g. a Pocket voice name while
`tts_engine: kokoro_onnx` is selected) can't happen: each engine only ever
reads its own voice/model/thread options, and a genuinely invalid voice
for whichever engine is active fails startup with a clear error instead of
silently falling back to the other engine.

### Example Configuration

STT + TTS, both enabled with recommended defaults:
```yaml
stt_enabled: true
model: small
language: de
use_ha_vocabulary: true
tts_enabled: true
tts_model: german
tts_voice: juergen
```

STT only (previous default, still fully supported):
```yaml
stt_enabled: true
model: small
language: de
tts_enabled: false
```

TTS only (no STT):
```yaml
stt_enabled: false
tts_enabled: true
tts_model: german
tts_voice: juergen
```

Higher-quality (slower) TTS:
```yaml
tts_enabled: true
tts_model: german_24l
```

## Using with Home Assistant Assist

1. Open **Settings** → **Voice assistants**
2. Create a new assistant or edit your existing one
3. In the **Speech-to-text** section, select **HomeIntent Moonshine** (if `stt_enabled`)
4. In the **Text-to-speech** section, select **HomeIntent Pocket TTS** (if `tts_enabled`)
5. Ensure the language is set to **German** (de)
6. Save

Expected flow with a voice satellite:
```
"Hey Nabu"
      ↓
Voice Satellite
      ↓
Home Assistant Assist
      ↓
HomeIntent Moonshine (STT)
      ↓
HomeIntent conversation agent
      ↓
Text response
      ↓
HomeIntent Pocket TTS (TTS)
      ↓
Wyoming audio
      ↓
Voice Satellite
```

Home Assistant orchestrates this pipeline; Pocket TTS never talks to the conversation
agent directly (see [Architecture](#architecture)).

## Performance

### STT Model Sizes

| Model | Parameters | File Size | WER | Use Case |
|-------|-----------|-----------|-----|----------|
| **Tiny** | 34M | ~80MB | 12.0% | Low-end hardware, fast feedback |
| **Small** | 123M | ~200MB | 7.5% | Recommended for most setups |

WER figures are upstream-published values for German. STT latency/RTF depends heavily
on your CPU — measure it yourself:
```bash
python -m app.benchmark path/to/your-sample.wav --model small --language de
```

### TTS Models

| Model | Layers | Speed | Quality | Notes |
|-------|--------|-------|---------|-------|
| **german** (default) | 6 | Faster | Good | Recommended default for Home Assistant |
| **german_24l** | 24 | Slower | Higher | Use if you have CPU headroom to spare |

Both are ~100M-parameter-class models per Kyutai's own README; neither add-on file size
nor RAM has been independently measured here (see the note below).

**Upstream-published numbers (Kyutai's own README, not independently verified on this
add-on's target hardware):** ~200ms to first audio chunk, roughly 2–2.5x real-time
throughput on CPU, "faster than real-time" (~6x) measured by Kyutai on Apple Silicon
(a MacBook Air M4) — not representative of a Raspberry Pi or NAS. **No numbers in this
README have been measured by this add-on's own maintainers** — see
`ABSCHLUSSBERICHT_V0.2.0.md`'s Performance section for exactly what was and wasn't
measured during development, and use the add-on's own TTFA/RTF log line
(`tts_log_performance: true`) to see real numbers on your own hardware.

### Real streaming, not fake streaming

Pocket TTS exposes a genuine incremental generator API
(`generate_audio_stream()`) that yields audio as it's decoded, using its own
internal generation and decode threads — this add-on forwards each chunk to
Home Assistant via Wyoming `audio-chunk` events as soon as it arrives, not
after buffering the full reply. There is no artificial/fixed-size chunking
layered on top.

## Privacy

- **Transcripts are never logged by default** (`log_transcripts: false`).
- **Synthesized text is never logged** by the TTS performance line — only timing/RTF numbers.
- **Performance logging never includes transcript or synthesis text.**
- **`use_ha_vocabulary` is strictly read-only** — only `config/*_registry/list`,
  `config/entity_registry/get_entries` (aliases), and `get_states` (device class)
  calls against the Home Assistant Core API; it never calls a service, changes a
  state, and never changes an entity's Assist exposure, automation, or any other HA
  config. Only entities you've actually exposed to Assist contribute vocabulary,
  computed the same way Home Assistant Core itself determines effective exposure.
  A failed refresh keeps the last successfully loaded vocabulary rather than losing
  it; a *successful* refresh that no longer sees a previously-exposed entity does
  remove it from the vocabulary.
- **Legacy (non-registry) entities are handled conservatively.** Home Assistant
  provides no read-only API that can tell an explicit "hide from Assist" apart
  from "never evaluated" for an entity with no entity registry entry — so this
  add-on never guesses. A legacy entity is only ever added to the automatic
  vocabulary when its Assist exposure can be positively confirmed; an ambiguous
  one is simply left out, on the principle that silently re-including an entity
  you removed from Assist would be worse than occasionally missing an obscure
  one. This only affects STT keyterm biasing — it never changes Home Assistant's
  own Assist exposure or any other state.
- **`save_debug_audio` is off by default**, and applies only to received STT audio
  (never TTS output). Metadata is an explicit allowlist (timestamp, model, language,
  transcript, duration, sample rate) — never an entity state.

## Model Cache

STT and TTS models are cached separately under `/data/`:
```
/data/models/              (Moonshine STT models)
/data/models/pocket-tts/   (Pocket TTS models + voice embeddings, via Hugging Face's
                             own cache under HF_HOME)
```
Both are excluded from Home Assistant backups (`backup_exclude` in config.yaml) since
they can always be re-downloaded. A cached model is loaded directly on subsequent
starts — no repeated downloads.

## Troubleshooting

### Add-on Won't Start

Check the logs (**Add-ons → HomeIntent Moonshine Voice → Logs**):

**"Model download failed"** (STT or TTS)
- Check your internet connection
- Ensure `/data/` has enough free space (STT: ~500MB; TTS adds more — see Performance)
- Try restarting the add-on

**"Failed to load Pocket TTS model"**
- TTS load failures are fatal on purpose (if you enabled `tts_enabled`, a silent
  fallback to "no TTS" would be more confusing than a clear startup error) —
  check the exact error in the logs
- If it's a download error, verify network access and retry
- As a temporary workaround, set `tts_enabled: false` to keep STT working while you investigate

**"Memory error"**
- Close other add-ons to free RAM
- Try the Tiny STT model / the `german` (6-layer) TTS model, both use less memory than their larger counterparts

### Recognition/Synthesis Not Working

**"No service available"** in Assist pipeline:
- Wait for the add-on to finish loading models (see the startup log banner)
- Restart the add-on
- Check that port 10300 isn't blocked

**No TTS voice appears in Assist:**
- Check `tts_enabled: true` is actually set
- Check the add-on logs for a Pocket TTS load failure

**Inaccurate transcripts:**
- Try the **Small** STT model
- Speak clearly in German

### Performance Issues

1. **Check system load** — close other processes
2. **STT**: use the Tiny model
3. **TTS**: use the `german` (6-layer) model instead of `german_24l`
4. **Check logs** for errors/CPU spikes

## Backward Compatibility

Upgrading from a pre-0.2.0 (STT-only) installation:

- Your existing `model`, `language`, `log_level`, and all other v0.1.x options
  **keep working unchanged** — nothing was renamed.
- `stt_enabled` defaults to `true`, so STT keeps working exactly as before.
- `tts_enabled` defaults to **`false`** on purpose: enabling TTS downloads a
  PyTorch runtime and a Pocket TTS model (see [Performance](#performance)) that a
  pure-STT user never asked for and shouldn't have to pay for on an update. Turn
  it on explicitly once you're ready.
- The add-on's Supervisor **slug** (`homeintent-moonshine-stt`) was deliberately
  **not** changed even though its display name changed to "HomeIntent Moonshine
  Voice" — changing the slug would make Home Assistant treat this as a brand new
  add-on instead of an upgrade of your existing one.

### Manual steps after updating

1. Update the add-on in Home Assistant
2. Start it and check the logs for the startup banner (STT/TTS enabled status, models, voice)
3. If you want TTS: set `tts_enabled: true` (and optionally `tts_model`/`tts_voice`), save, restart
4. Open **Settings → Voice assistants** and (re)select HomeIntent Moonshine as STT and/or
   HomeIntent Pocket TTS as TTS in your Assist pipeline
5. Test with a voice command

## Architecture

```
Voice Satellite
      │
      ▼
Home Assistant Assist
      │
      ▼
   Wyoming
      │
      ├───────────────┐
      ▼               ▼
Moonshine STT     Pocket TTS
      │               ▲
      ▼               │
HomeIntent Conversation Agent
      │               │
      └── Text reply ─┘
```

Both STT and TTS are served by the same Wyoming TCP service (port 10300); Home
Assistant's own Assist pipeline is what routes a transcript to the conversation
agent and its text reply on to TTS — Pocket TTS never talks to the conversation
agent directly.

## Development

### Local Testing

```bash
git clone https://github.com/pquandel2-alt/homeintent-moonshine-addon.git
cd homeintent-moonshine-addon/homeintent-moonshine-stt
# CPU-only torch first (avoids downloading unused CUDA packages on Linux):
pip install torch --index-url https://download.pytorch.org/whl/cpu
pip install -r requirements.txt pytest pytest-asyncio pyyaml
```

Run tests (excludes real-model e2e tests by default):
```bash
pytest app/tests/ -v
```

Run the real e2e tests (downloads real models):
```bash
pytest app/tests/ -v -m e2e
```

Run linting/type-checking:
```bash
ruff check app/
mypy --config-file app/pyproject.toml app/
```

## Documentation

- **[DOCS.md](homeintent-moonshine-stt/DOCS.md)** — In-app documentation
- **[CHANGELOG.md](homeintent-moonshine-stt/CHANGELOG.md)** — Version history
- **[ABSCHLUSSBERICHT_V0.2.0.md](ABSCHLUSSBERICHT_V0.2.0.md)** — Full verification report for this release
- **[Moonshine Repo](https://github.com/moonshine-ai/moonshine)** — STT model documentation
- **[Pocket TTS Repo](https://github.com/kyutai-labs/pocket-tts)** — TTS model documentation
- **[Wyoming Protocol](https://github.com/rhasspy/wyoming)** — Integration protocol

## License

This add-on's code is licensed under the MIT License — see [LICENSE](LICENSE). Pocket
TTS's own code is also MIT licensed. Kokoro ONNX's own runtime code
(`kokoro-onnx`) is also MIT licensed.

**Model weights are NOT all MIT licensed, and differ between STT and TTS:**

- **Moonshine German STT models**: Moonshine publishes German (and other non-English)
  models under its non-commercial **Community License** — free for researchers,
  developers, small businesses, and creators with less than $1M in annual revenue;
  commercial use beyond that requires Moonshine's Enterprise license. Only Moonshine's
  English models are MIT. See [moonshine.ai/license](https://www.moonshine.ai/license).
- **Pocket TTS German model weights**: Kyutai's README states usage of their released
  model weights must not "result in, involve, or facilitate any illegal, harmful,
  deceptive, fraudulent, or unauthorized activity," explicitly prohibiting voice
  impersonation/cloning without consent and misinformation/deception uses. The exact
  license field on Hugging Face's `kyutai/pocket-tts` model card could not be verified
  during this add-on's development (see `ABSCHLUSSBERICHT_V0.2.0.md`) — review it
  yourself at https://huggingface.co/kyutai/pocket-tts before commercial use.
- **Kokoro ONNX German "Martin" model weights**: Apache License 2.0 (per
  [Godelaune/Kokoro-82M-ONNX-German-Martin](https://huggingface.co/Godelaune/Kokoro-82M-ONNX-German-Martin)'s
  own README — a fine-tune of hexgrad's Kokoro-82M, also Apache 2.0). No
  immutable Git revision could be found published for that model
  repository at the time this add-on was built, so `main` is used by
  default; see `ABSCHLUSSBERICHT_V0.3.0.md` and the `kokoro_model_revision`
  advanced setting if a stable tag is published later.

## Contributing

Issues and pull requests are welcome! Please ensure:
- Code passes `ruff check` and `mypy --config-file homeintent-moonshine-stt/app/pyproject.toml`
- Tests pass: `pytest homeintent-moonshine-stt/app/tests/ -v`
- Commit messages are clear and descriptive

## Credits

- **Moonshine ASR**: https://github.com/moonshine-ai/moonshine
- **Pocket TTS**: https://github.com/kyutai-labs/pocket-tts
- **Kokoro ONNX runtime**: https://github.com/thewh1teagle/kokoro-onnx
- **Kokoro-82M / German "Martin" fine-tune**: https://huggingface.co/hexgrad/Kokoro-82M,
  https://huggingface.co/Godelaune/Kokoro-82M-ONNX-German-Martin
- **Wyoming Protocol**: https://github.com/rhasspy/wyoming
- **Home Assistant**: https://www.home-assistant.io

## Support

If you encounter issues:

1. **Check the logs** — Add-ons → HomeIntent Moonshine Voice → Logs
2. **Review the [Troubleshooting](#troubleshooting) section**
3. **Open an issue** — https://github.com/pquandel2-alt/homeintent-moonshine-addon/issues

---

**Made with ❤️ for the Home Assistant community**
