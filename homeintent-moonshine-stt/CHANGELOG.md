# Changelog - HomeIntent Moonshine Voice Add-on

## [0.6.0] - 2026-09-19

Adds a fifth and final user-selectable STT engine, **Vosk German**
(`vosk_german`) -- the lightest-weight, lowest-CPU/RAM STT option this
add-on offers -- completing the planned 5 STT x 3 TTS engine matrix. Also
adds `app/stt_benchmark.py` (a cross-engine STT benchmark CLI) and extends
`app/tts_benchmark.py` to cover Supertonic 3. Moonshine, Kroko,
Speechcatcher M/L, Pocket TTS, Kokoro ONNX, and Supertonic 3 are unchanged
and remain their previous defaults; upgrading requires no action.

### Added
- **New STT engine: Vosk German** (`app/vosk_engine.py`,
  `app/vosk_model.py`), a Kaldi-based recognizer via the
  [`vosk`](https://alphacephei.com/vosk/) PyPI package (Apache 2.0),
  driven in-process through `vosk.KaldiRecognizer` -- no subprocess, no
  separate container.
  - **Streaming**: real, sample-incremental streaming -- every Wyoming
    audio chunk is fed straight into `AcceptWaveform()` as it arrives
    (verified directly from the installed wheel's own
    `vosk/__init__.py` source), not buffered and decoded once at the end.
    `FinalResult()` is called at `finalize()`.
  - **Model**: `vosk-model-small-de-0.15` (~45MB, Apache License 2.0),
    verified as the current recommended small German model against the
    official model catalogue (alphacep/vosk-space's own `models.md`,
    since alphacephei.com itself was unreachable from this add-on's
    development sandbox) -- no smaller or newer small German model
    supersedes it. Downloaded from
    `https://alphacephei.com/vosk/models/vosk-model-small-de-0.15.zip`
    (matching `vosk.MODEL_PRE_URL`'s own base URL) to
    `/data/models/vosk/` only when `vosk_german` is selected, via an
    atomic download-then-rename pattern identical to
    `app/kroko_model.py`'s. A much larger `vosk-model-de-0.21` (1.9GB)
    exists upstream but is deliberately NOT offered as a second option --
    it would contradict this engine's entire reason for existing.
  - **Hotwords / HA vocabulary -- investigated and deliberately NOT
    wired up, not an unimplemented gap**: Vosk's `KaldiRecognizer` does
    expose a grammar mechanism (a constructor grammar argument /
    `SetGrammar()`, verified directly from the installed wheel's source
    and from upstream's own Ruby binding docs, the clearest primary
    documentation of its semantics), but it is a **hard closed-set
    restriction** on the decoding graph (recognizes ONLY the given
    phrases, plus a literal `"[unk]"` catch-all), not a soft bias/boost
    like Moonshine's `set_keyterms()` or Kroko's sherpa-onnx hotwords.
    Wiring the full, open-ended Home Assistant vocabulary into it would
    break free-form recognition of everything else a user might say --
    a strictly worse UX than no biasing. Honestly reported as
    `supports_hotwords=False`/`supports_dynamic_vocabulary=False`
    (`set_keyterms()` logs "HA vocabulary not supported by engine
    vosk_german" and returns `([], True)`), matching Speechcatcher's own
    precedent.
  - **No CPU/GPU dependency of its own**: verified via a real install
    (`pip show vosk`) that Vosk's only dependencies are `cffi`,
    `requests`, `srt`, and `tqdm` -- no numpy/torch, and no CUDA/GPU
    dependency (Kaldi's C++ decoder core is CPU-only in the plain PyPI
    wheel; the `GpuInit`/`GpuThreadInit` calls its Python API exposes are
    for a separately-built server variant this add-on never uses).
  - **Deliberately minimal config surface**: no CPU-thread or beam-size
    option -- `vosk.KaldiRecognizer` exposes neither for tuning, and this
    engine's whole purpose is to be the simplest, lightest option.
  - Only the selected engine is ever loaded (covered by
    `app/tests/test_engine_dispatch.py`).
- **New runtime dependencies**: `vosk==0.3.45`, plus its own real,
  verified dependency footprint `cffi==2.1.1`, `requests==2.34.2`,
  `srt==3.5.3` (`tqdm`/`websockets` were already pinned at compatible
  versions). Verified manylinux/aarch64 wheels for cp311 exist for
  `vosk`/`cffi`; `srt` is a pure-Python package with no compiled
  extension (verified from its own `setup.py`), so it installs
  identically on both architectures despite shipping only an sdist. A
  full `pip install -r requirements-runtime.txt` (minus `pyaudio`, which
  needs system PortAudio headers already handled by the Dockerfile) was
  verified to resolve with no conflicts (`pip check`) against the entire
  existing dependency stack (sherpa-onnx, onnxruntime, torch/torchaudio +
  Speechcatcher's ESPnet-family chain, numpy, kokoro-onnx).
- **`app/stt_benchmark.py`**: a new cross-engine STT benchmark CLI,
  driven entirely through the shared `SttEngine`/`SttSession`
  abstraction (`--engine moonshine|kroko|speechcatcher_m|speechcatcher_l|
  vosk_german`) -- one code path for all five engines, not five separate
  scripts. Reports audio duration, wall/model inference time, final-
  result latency, RTF, and the raw transcript for manual inspection.
  Does NOT compute a Word Error Rate (no ground-truth reference
  transcript mechanism exists in this add-on's fixtures -- inventing one
  would be dishonest). Time-to-first-partial-result is only reported for
  an engine with `supports_partial_results=True`; none of the five
  currently is, so it honestly prints "n/a" rather than a fabricated
  number. Like `app/benchmark.py`/`app/tts_benchmark.py`, this is a
  local, manual, run-on-your-own-hardware tool, not something CI runs or
  a source of any numbers quoted in README/DOCS.
- **`app/tts_benchmark.py` extended to cover Supertonic 3**
  (`--engine supertonic_3`) -- an explicitly-reported gap from v0.4.0's
  own development, now closed. Uses the exact same
  `TtsSynthesizer`-interface measurement code as Pocket TTS/Kokoro ONNX,
  not a separate code path.
- Config schema: `stt_engine` enum extended to
  `list(moonshine|kroko|speechcatcher_m|speechcatcher_l|vosk_german)`;
  default remains `moonshine` (verified: an old persisted config with no
  `stt_engine` key still resolves to `moonshine` through both the CLI
  default and rootfs's `config_or_default()` fallback).
- Dockerfile/rootfs run script: new `VOSK_CACHE=/data/models/vosk` env
  var, matching the existing per-engine cache-directory pattern. No new
  CLI flags needed (Vosk has no tunable options).
- New CI e2e job `e2e-vosk-stt-transcribe` and
  `app/tests/test_e2e_vosk_stt.py` (gated behind `RUN_VOSK_E2E=1`,
  mirrors `test_e2e_kroko_stt.py`), plus new
  `app/tests/test_vosk_engine.py`, `app/tests/test_vosk_model.py`,
  `app/tests/test_stt_benchmark.py`, and extended
  `test_engine_dispatch.py`/`test_config_stt_engine_option.py`/
  `test_tts_benchmark.py` coverage.

### Known limitations
- The real end-to-end Vosk test (`app/tests/test_e2e_vosk_stt.py`) was
  not run during development -- alphacephei.com was unreachable from this
  add-on's development sandbox (only a git clone of alphacep/vosk-api and
  alphacep/vosk-space was reachable, which is how the model name/API/
  download-URL convention were verified). Run it explicitly
  (`RUN_VOSK_E2E=1 pytest -m e2e`) before relying on it.
- Real multi-arch (amd64 + aarch64) Docker builds and real hardware
  benchmarks (both `app/stt_benchmark.py` and `app/tts_benchmark.py`) were
  not performed in this add-on's development sandbox -- wheel
  availability and dependency resolution were verified, but an actual
  build and on-target-hardware benchmark run are needed before treating
  this release as production-validated.

## [0.5.0] - 2026-09-19

Adds two more user-selectable STT engines, **Speechcatcher M** and
**Speechcatcher L** (`speechcatcher_m`/`speechcatcher_l`), behind the same
`SttEngine`/`SttSession` abstraction introduced in v0.4.0. Moonshine,
Kroko, Pocket TTS, Kokoro ONNX, and Supertonic 3 are unchanged and remain
their previous defaults; upgrading requires no action.

### Added
- **New STT engines: Speechcatcher M and L** (`app/speechcatcher_engine.py`,
  `app/speechcatcher_model.py`), a streaming Transformer ASR toolbox
  ([Speechcatcher](https://github.com/speechcatcher-asr/speechcatcher),
  MIT). Both variants share the exact same engine/session implementation;
  only the model checkpoint differs (constructor parameter, not
  duplicated code). Run in-process by driving Speechcatcher's own
  `Speech2TextStreaming` object directly -- NOT `speechcatcher_server`'s
  websocket layer, which is itself only a thin wrapper around that same
  object (verified directly from its source). Real, incremental,
  block-granular streaming decode (Tsunoo et al.'s "Streaming Transformer
  ASR with Blockwise Synchronous Beam Search"), never buffer-whole-then-
  decode-once.
  - **Decoder choice**: uses Speechcatcher's own `espnet_streaming_decoder`
    package (a from-scratch, "smaller footprint" extraction of ESPnet's
    streaming code -- NOT the full `espnet` PyPI package) via
    `decoder_impl="espnet"`. Speechcatcher 0.5.0 added an experimental,
    dependency-free "native" decoder, but its own README still documents
    `--decoder espnet` as the default/stable choice and labels `native`
    explicitly experimental -- this add-on's own policy is to only use a
    lighter decoder when it is the officially, stably supported one, so
    `native` is deliberately not used here.
  - **Models**: `speechcatcher_m` ->
    `speechcatcher/speechcatcher_german_espnet_streaming_transformer_13k_train_size_m_raw_de_bpe1024`,
    `speechcatcher_l` -> the `..._l_raw_de_bpe1024` variant (exact tags
    from Speechcatcher's own `tags` table). Downloaded to
    `/data/models/speechcatcher/` only when selected, via
    `huggingface_hub.snapshot_download()` (through Speechcatcher's own
    `espnet_model_zoo` fork's `ModelDownloader`) -- the same atomic,
    resumable caching mechanism already used by Pocket TTS/Kokoro ONNX,
    not a new download mechanism.
  - **No hotword/HA-vocabulary support**: verified directly from source
    that nothing in Speechcatcher's dependency chain exposes a keyword/
    context-biasing API ("contextual" there refers only to the contextual
    *block* streaming encoder architecture). `set_keyterms()` logs "HA
    vocabulary not supported by engine speechcatcher_m"/`_l` once and
    never pretends to apply anything -- `supports_hotwords=False`,
    `supports_dynamic_vocabulary=False`.
  - Only the selected engine is ever loaded (covered by
    `app/tests/test_engine_dispatch.py`).
- **New Speechcatcher options**: `speechcatcher_threads` (default `0` =
  PyTorch's own default), `speechcatcher_beam_size` (default `5`).
- **New runtime dependencies**: Speechcatcher and its own two git-only
  forks (`espnet_streaming_decoder`, `espnet_model_zoo`) are not on PyPI
  and are installed from pinned git commits (see `requirements-runtime.txt`
  for the exact commits and full reasoning). Their own PyPI-installable
  transitive dependencies are pinned in the same file: `torch`/`torchaudio`
  (from PyTorch's CPU wheel index, alongside the pre-existing `torch`
  install for Pocket TTS), `scipy`, `soundfile`, `six`, `tqdm`, `somajo`,
  `sentencepiece`, `pyyaml`, `ffmpeg-python`, `pyaudio`,
  `python_speech_features`, `typeguard`, `einops`, `hydra-core`,
  `opt-einsum`, `humanfriendly`, `torch-complex`, `librosa`, `h5py`,
  `kaldiio`, `jaconv`, `pytorch-wpe`, and `ctc-segmentation`. Verified
  manylinux2014 wheels exist for cp311 on both x86_64 and aarch64 for every
  one of these except `ctc-segmentation` (no prebuilt wheels on any
  platform; compiles from its Cython/C sources at install time, using the
  `build-essential`/`python3-dev` packages already present) and `pyaudio`
  (needs PortAudio's headers; `portaudio19-dev` added to the Dockerfile).
  This is a meaningfully larger dependency footprint than Kroko's single
  `sherpa-onnx` wheel; flagged here explicitly rather than silently grown.
- **Dockerfile**: installs `git` and `portaudio19-dev` (new system
  packages), installs `torchaudio` alongside the existing `torch` CPU-only
  install, and installs the three pinned Speechcatcher git packages with
  `--no-deps` after every other pinned dependency (deterministic
  resolution, no reliance on those repos' own unpinned transitive git
  refs). New `SPEECHCATCHER_CACHE=/data/models/speechcatcher` env var,
  matching the existing per-engine cache-directory pattern.
- Config schema: `stt_engine` enum extended to
  `list(moonshine|kroko|speechcatcher_m|speechcatcher_l)`; default remains
  `moonshine` (verified: an old persisted config with no `stt_engine` key
  still resolves to `moonshine` through both the CLI default and rootfs's
  `config_or_default()` fallback).
- Tests: `app/tests/test_speechcatcher_engine.py`,
  `app/tests/test_speechcatcher_model.py` (fake-double unit tests, no real
  model download), `app/tests/test_e2e_speechcatcher_stt.py` (gated behind
  `RUN_SPEECHCATCHER_E2E=1`, mirrors `test_e2e_kroko_stt.py`), plus
  extended `test_engine_dispatch.py`/`test_config_stt_engine_option.py`
  coverage for the two new engine ids.

### Known limitations
- Speechcatcher's own model checkpoints' Hugging Face Hub license field
  could not be independently verified (huggingface.co was unreachable from
  this add-on's development sandbox) -- Speechcatcher's own code is MIT,
  but review the model cards yourself before commercial use.
- The real end-to-end Speechcatcher test
  (`app/tests/test_e2e_speechcatcher_stt.py`) was not run during
  development -- it needs the full, heavy dependency stack installed and
  network access to Hugging Face Hub, neither of which was available in
  this add-on's development sandbox. Run it explicitly
  (`RUN_SPEECHCATCHER_E2E=1 pytest -m e2e`) before relying on it.

## [0.4.0] - 2026-09-18

Adds a second, user-selectable STT engine (**Kroko**, via sherpa-onnx) and a
third TTS engine (**Supertonic 3**, via sherpa-onnx), both real, in-process,
streaming implementations behind a new generic `SttEngine`/`SttSession`
abstraction that mirrors the existing `TtsSynthesizer` abstraction's design.
Moonshine, Pocket TTS, and Kokoro ONNX are unchanged and remain the
defaults; upgrading requires no action.

### Added
- **STT engine abstraction** (`app/stt_engine.py`): a generic
  `SttEngine`/`SttSession` interface (`engine_id`, `capabilities`,
  `program_name`, `create_session()`, `set_keyterms()`) that
  `app/handler.py` depends on instead of a concrete Moonshine
  `Transcriber` -- adding a future engine (Speechcatcher, Vosk) is now
  one more implementation of this interface, not a new branch in the
  handler.
- **Moonshine adapter** (`app/moonshine_engine.py`): wraps the existing,
  unmodified `app/models.py`/`app/streaming.py`/`app/keyterms.py` behind
  the new interface with zero behavioral change -- model choice, keyterm
  boosting, VAD, `decode_incomplete_lines`, and performance counters all
  work exactly as before. `app/handler.py`'s `transcriber=` constructor
  parameter is kept for backward compatibility with existing callers/tests.
- **New STT engine: Kroko** (`app/kroko_engine.py`, `app/kroko_model.py`),
  a German streaming Zipformer2-transducer model (Banafo AI's
  [Kroko-ASR](https://huggingface.co/Banafo/Kroko-ASR), Apache-2.0,
  merged into sherpa-onnx upstream) run in-process via
  `sherpa_onnx.OnlineRecognizer.from_transducer()` -- real incremental
  streaming (`accept_waveform()` per Wyoming audio-chunk), with dynamic
  HA-vocabulary/keyterm biasing via sherpa-onnx's own per-stream hotwords
  (`modified_beam_search` decoding). Selected via the new `stt_engine:
  kroko` option; default remains `moonshine`. Model downloaded to
  `/data/models/kroko/` only when selected, from sherpa-onnx's own GitHub
  Release (`sherpa-onnx-streaming-zipformer-de-kroko-2025-08-06.tar.bz2`).
- **New Kroko options**: `kroko_threads` (default `1`),
  `kroko_hotwords_score` (default `1.5`).
- **New TTS engine: Supertonic 3** (`app/supertonic_tts.py`,
  `app/supertonic_session.py`), Supertone Inc.'s multilingual
  ([supertone-inc/supertonic](https://github.com/supertone-inc/supertonic),
  MIT-licensed) TTS system, run in-process via sherpa-onnx's INT8-quantized
  export and its real native callback-streaming `generate()` API --
  chunks are forwarded to the existing Wyoming AudioChunk pipeline exactly
  like Pocket TTS/Kokoro, with no WAV round-trip. 10 built-in voices
  (`M1`-`M5`/`F1`-`F5`); default `M1`. Selected via the new `tts_engine:
  supertonic_3` option; default remains `pocket_tts`. Model downloaded to
  `/data/models/supertonic/` only when `tts_enabled: true` AND
  `tts_engine: supertonic_3`.
- **New Supertonic options**: `supertonic_voice` (default `M1`),
  `supertonic_speed`, `supertonic_steps` (default `8`),
  `supertonic_threads` (default `1`).
- **New runtime dependency**: `sherpa-onnx==1.13.8` (Apache-2.0), verified
  to ship manylinux (amd64) and aarch64 wheels for Python 3.11, and to
  bundle its own statically-linked ONNX Runtime build with no Python-level
  `onnxruntime` dependency -- does not conflict with the
  `onnxruntime==1.30.0` pin already used by Kokoro ONNX.
- **Native Home Assistant translations** (`translations/en.yaml`,
  `translations/de.yaml`): every visible config option now has a proper
  display name and description in both languages, following the real
  add-on translation structure (`configuration: <key>: {name,
  description}`), instead of relying on `config.yaml`'s own raw option
  names. The flat `config.yaml` schema itself is unchanged (no new,
  currently-undocumented schema keywords were introduced) for full
  upgrade compatibility.
- Both `app/kroko_model.py`/`app/supertonic_tts.py` only ever load their
  respective model when their engine is actually selected -- selecting
  `moonshine`/`pocket_tts` (the defaults) never touches Kroko or
  Supertonic code at all, verified by dedicated dispatch tests.

### Changed
- `app/handler.py` now takes an optional `stt_engine=` constructor
  parameter (used for Kroko); the previous `transcriber=` parameter still
  works unchanged for Moonshine and every existing caller/test.
- Version bumped to 0.4.0 across `config.yaml`, `app/pyproject.toml`, and
  `app/__main__.py`'s `VERSION`.

## [0.3.0] - 2026-09-18

Adds **Kokoro German ONNX** as a second, user-selectable local TTS engine
alongside the existing Pocket TTS, integrated directly in-process (no
second server, no HTTP hop). See `ABSCHLUSSBERICHT_V0.3.0.md` for the
full analysis, verification, and remaining risks.

### Added
- **New TTS engine: Kokoro German ONNX** (`app/kokoro_tts.py`,
  `app/kokoro_session.py`), using the `kokoro-onnx` package
  (`Godelaune/Kokoro-82M-ONNX-German-Martin`, Apache-2.0 licensed model,
  male voice "Martin", 24kHz output, CPU-only via ONNX Runtime). Selected
  via the new `tts_engine: kokoro_onnx` option; default remains
  `pocket_tts` for full backward compatibility with existing installs.
- **Generalized TTS synthesizer interface** (`app/tts_engine.py`): a
  shared `TtsSynthesizer` protocol (sample_rate, default_voice,
  engine_id, model_name, program_name, attribution, description,
  `synthesize_stream()`) used identically by `handler.py`/`__main__.py`
  for both Pocket TTS and Kokoro ONNX -- no per-engine branching in the
  Wyoming request/response path.
- **New Kokoro options**: `kokoro_voice` (default `martin`),
  `kokoro_speed`, `kokoro_threads` (0 = auto, same semantics as
  `tts_threads`), `kokoro_sentence_pause`, `kokoro_clause_pause`.
  Existing Pocket TTS options are unchanged.
- **Persistent, on-demand model download**: the ~326MB Kokoro model is
  never baked into the Docker image. It is downloaded via
  `huggingface_hub.hf_hub_download` (atomic, resumable, cache-aware) into
  `/data/models/kokoro-onnx/` only when `tts_enabled: true` AND
  `tts_engine: kokoro_onnx`; selecting Pocket TTS never triggers a Kokoro
  download. Already-covered by the existing `backup_exclude:
  ["models/*"]`.
- **Original German text normalizer** (`app/german_text_normalizer.py`):
  expands times, temperatures, units, currency, and abbreviations (e.g.
  "18:20 Uhr", "21,5 °C", "500 g", "49,99 EUR") before Kokoro
  synthesis, so common HomeIntent sensor output is spoken correctly.
- **Streaming synthesis**: Kokoro's real async `create_stream()`
  generator feeds raw float32 chunks directly into the existing Wyoming
  streaming pipeline, so Home Assistant receives the first audio chunk
  while the rest is still being generated -- matching Pocket TTS's own
  behavior, never assembling a full WAV first.
- **Extended benchmark** (`app/tts_benchmark.py`): `python -m
  app.tts_benchmark --engine pocket_tts|kokoro_onnx` now measures both
  engines against the same 4 German test sentences (model load, warmup,
  TTFA, first-chunk duration, model compute time, RTF, peak RSS) across a
  thread-count sweep, run on the user's own target hardware.
- **Dynamic Wyoming discovery**: `describe` now reports correct,
  per-engine metadata -- Pocket TTS still advertises "HomeIntent Pocket
  TTS" with Kyutai attribution and voice "juergen"; Kokoro advertises
  "HomeIntent Kokoro ONNX" with Godelaune/hexgrad attribution and voice
  "martin". No cross-engine attribution leakage.
- **Performance log** (`_log_tts_performance`) now includes `engine=...`
  so Pocket TTS and Kokoro ONNX runs can be directly compared.

### Dependencies
- Added `huggingface_hub`, `kokoro-onnx`, `onnxruntime`, `espeakng-loader`,
  `phonemizer` to `requirements-runtime.txt` / `app/pyproject.toml`. No
  system `espeak-ng` apt package is required -- `espeakng-loader` ships
  self-contained prebuilt libraries and data for both amd64 and aarch64.
  No web server packages (FastAPI/uvicorn) were added; Kokoro is
  integrated fully in-process.

### Unchanged
- Pocket TTS (models, voices, `tts_threads`, warmup, streaming protocol)
  continues to work exactly as before; it remains the default engine.
  STT (Moonshine), keyterms, HA vocabulary, and the wake word/audio input
  pipeline were not touched.

## [0.2.7] - 2026-09-18

Real Wyoming TTS audio streaming to Home Assistant. See
`ABSCHLUSSBERICHT_V0.2.7.md` for the full end-to-end analysis and
verification against upstream Home Assistant Core source.

### Fixed
- **Home Assistant waited for the entire Pocket TTS response before
  playing any audio, even though this add-on already streamed audio
  internally.** Root cause (verified directly against
  `homeassistant/components/wyoming/tts.py` upstream, not assumed): the
  add-on's `describe` response advertised
  `supports_synthesize_streaming=False`, so Home Assistant used its
  legacy `async_get_tts_audio()` path, which loops reading `audio-chunk`
  events into an in-memory `BytesIO` WAV file until `audio-stop`, and only
  returns the complete file afterward -- Home Assistant's own streaming
  audio player was never even reached. This add-on's internal
  `generate_audio_stream()` use was real, but invisible to Home Assistant
  under the old protocol.

### Added
- **Full Wyoming streaming-TTS protocol**: `synthesize-start`,
  `synthesize-chunk`, `synthesize-stop`, and `synthesize-stopped` are now
  implemented (`app/tts_stream.py`, `app/handler.py`), verified against
  the installed `wyoming==1.10.2` package's real event classes.
  `supports_synthesize_streaming` is now `True`, which makes Home
  Assistant use its streaming Wyoming TTS client
  (`async_stream_tts_audio()`) instead of the legacy buffering path --
  audio now reaches Home Assistant as soon as Pocket TTS produces its
  first chunk.
- **`TtsStreamState`** (`app/tts_stream.py`): a small state machine
  (idle/collecting/synthesizing/finished/cancelled) that prevents Home
  Assistant's real client behavior -- sending the entire message a SECOND
  time via a backwards-compatible `synthesize` event, right after the
  streamed `synthesize-chunk`(s) -- from causing the text to be
  synthesized (and spoken) twice. Verified this exact behavior directly
  from `homeassistant/components/wyoming/tts.py`'s `_write_tts_message()`.
- Every streaming request now ends with `synthesize-stopped`, which Home
  Assistant's streaming reader (`_read_tts_audio()`) requires to end its
  own read loop -- a plain `audio-stop` is not enough there (verified
  from the same upstream source); the legacy single-shot path is
  unchanged (no `synthesize-stopped`, since that client never reads for
  one).
- New performance-log fields: `protocol_mode=legacy|streaming` and
  `first_audio_to_ha=`, isolating pure model+send latency (from the
  moment the complete text was known) from the protocol/collection
  overhead already included in `ttfa_generated`/`ttfa_sent` for a
  streaming request.
- **`python -m app.tts_benchmark`**: a new local CLI benchmarking Pocket
  TTS time-to-first-audio and real-time factor across different
  `tts_threads` values (default sweep: `1,2,4,6,8,auto`), each measured in
  its own subprocess for a clean `torch.set_num_threads()` setting per
  run. Not run in CI; a tool for tuning `tts_threads` on real target
  hardware (see DOCS.md).

### Investigated, not changed
- Pocket TTS 3.1.0's `generate_audio_stream()` takes one complete
  `text_to_generate` string per call (verified directly against
  `pocket_tts.models.tts_model.TTSModel.generate_audio_stream()`'s
  source) -- there is no incremental/appendable text API to feed
  synthesize-chunk text into as it arrives. For the common case (a
  HomeIntent/Assist intent response, where the full reply text is already
  known before TTS starts), this is a non-issue: Home Assistant itself
  sends the whole text as a single synthesize-chunk anyway (verified from
  `homeassistant/components/tts/__init__.py`'s `_async_generate_tts_audio()`,
  which wraps a plain string response in a single-item async generator).
  No sentence/phrase buffering was added, since it isn't needed for that
  case and would add real risk to word boundaries/ordering for no
  measured benefit; `TtsStreamState` still correctly accumulates multiple
  chunks as a fallback for a genuinely token-streamed LLM response or a
  spec-only client.

## [0.2.6] - 2026-09-18

Follow-up to v0.2.5's safe-keyterm fix, driven by real production log
output: several real HA-derived keyterms were still being discarded
entirely even though they were fully recoverable -- an invisible
formatting character or a decorative symbol/separator, not the underlying
German word, was what the tokenizer actually rejected. See
`ABSCHLUSSBERICHT_V0.2.6.md` for the full report.

### Added
- **Two-stage keyterm normalization** in `apply_safe_keyterms()`
  (`app/keyterms.py`):
  - Stage 1 (`normalize_keyterm()`, lossless, applied to every candidate
    before any native call): in addition to NFC composition and
    whitespace trimming, invisible Unicode *format* characters (category
    `Cf` -- soft hyphen U+00AD, zero-width space/joiners, word joiner,
    byte-order-mark) are now removed and internal whitespace is
    collapsed. Real production case: `"Wasch\xadmaschine"` (a soft hyphen
    between "h" and "m", invisible in the HA UI) now normalizes to
    `"Waschmaschine"` before ever reaching the tokenizer.
  - Stage 2 (`_generate_speech_fallback()`, only tried after the loaded
    model's tokenizer has actually rejected the stage-1 term): visual
    separators (`/`, `\`, `|`) become a space, and emoji/decorative
    symbols plus variation selectors are dropped. The result is
    re-validated against the real model with one more native call --
    never assumed valid. Real production cases:
    `"Treppe/Büro"` -> `"Treppe Büro"`, `"Familie ⚠️"` -> `"Familie"`.
  - A term where nothing speech-relevant remains after stage 2 (e.g. an
    emoji-only entry) is rejected rather than sent to the model as an
    empty string.
  - Applies uniformly to HA vocabulary (including Area+Entity composite
    terms), manual `extra_keyterms`, and the periodic refresh, since all
    three already funnel through this one function.
- Startup/refresh logging now distinguishes unchanged, normalized, and
  speech-fallback-recovered keyterms (each capped and summarized the same
  way rejections already were), plus a one-line summary:
  `Moonshine keyterms: candidate=N unchanged=N normalized=N
  speech-fallback=N rejected=N applied=N`.

### Investigated, not changed
- The native Moonshine C library's own stderr diagnostics during
  bisection (`No match found for remaining bytes ...`,
  `moonshine_transcriber_set_keyterms(): Failed to set key terms ...`)
  come directly from the compiled library, not from Python logging, and
  could not be suppressed without risking hiding a genuine unrelated
  native error; the additional per-term fallback validation call adds at
  most one more such native attempt per already-failing term (not per
  otherwise-valid term), so the extra noise is bounded by the number of
  actually incompatible terms.

## [0.2.5] - 2026-09-18

Critical production bug fix. See `ABSCHLUSSBERICHT_V0.2.5.md` for the full
root-cause analysis and architecture description.

### Fixed
- **A single Home-Assistant-derived keyterm the loaded Moonshine model's
  tokenizer could not represent crashed the whole add-on into a permanent
  restart loop.** Real production log:
  `moonshine_voice.errors.MoonshineError: Failed to set key terms: Unknown
  error` from `transcriber.set_keyterms(effective_keyterms)` in
  `_load_and_bias_transcriber()`, raised for the entity name "/Büro" among
  249 otherwise valid HA vocabulary terms, uncaught, exit code 1.
  `Transcriber.set_keyterms()` joins the whole list into ONE comma-delimited
  string for its native C API in a single call (verified against the
  installed moonshine-voice==0.1.5 source directly) -- if the native
  tokenizer rejects any single term, the entire call fails with a generic,
  per-call (not per-term) error, and moonshine-voice's C API exposes no
  separate "validate without activating" function to call instead.

### Added
- **`apply_safe_keyterms()`** (`app/keyterms.py`): the one, reusable, safe
  path every caller now uses instead of calling
  `transcriber.set_keyterms()` directly -- startup, manual
  `extra_keyterms`, and the periodic HA vocabulary refresh all go through
  it. Cheap syntactic pre-validation (empty/whitespace-only entries,
  control characters, the comma wire-delimiter, unencodable Unicode;
  German orthography -- umlauts, ß, spaces, hyphens, slashes -- is never
  touched) is followed by validation against the ACTUAL loaded model via
  `set_keyterms()` itself, using a divide-and-conquer search to isolate
  incompatible terms without necessarily needing one native call per term.
  Always ends with exactly one final, explicit, authoritative
  `set_keyterms()` call for the accepted list -- diagnostic calls made
  while isolating a bad term are never mistaken for the final state. Never
  raises: a genuinely unexpected, non-per-term native error restores a
  caller-supplied fallback (or disables biasing) instead of propagating.
- Startup now logs a real, post-validation effective keyterm count and a
  clear warning (with reason: empty/control_character/delimiter/
  invalid_unicode/tokenizer_rejected/native_error) for every skipped term,
  capped at 20 individual lines with a summary for larger rejections; full
  detail remains available at DEBUG. The full HA vocabulary is never
  logged at INFO.
- The periodic refresh's last-known-good handling is now properly atomic:
  a newly fetched HA vocabulary is only committed as the new
  last-known-good state after `apply_safe_keyterms()` confirms its final,
  authoritative apply actually succeeded -- not merely because the HA
  fetch itself succeeded. If that final apply fails unexpectedly (a
  genuinely structural error, since per-term issues are already filtered
  out by that point), the previous known-good keyterms are restored and
  `last_known_good_terms` is left untouched, so the next refresh attempt
  starts from the same known-good baseline.

### Investigated, not changed
- `moonshine-voice` is already pinned at its latest published version
  (0.1.5, confirmed against PyPI) -- no upgrade is available that would
  fix this upstream. The defensive application layer above is necessary
  regardless of any future upstream fix.

## [0.2.4] - 2026-09-18

Real-hardware performance + HA vocabulary fix, driven entirely by real
production numbers from the first real Home Assistant installation and
Assist test run. See `ABSCHLUSSBERICHT_V0.2.4.md` for the full report,
including everything that was investigated but NOT changed (and why).

### Fixed
- **HA vocabulary silently produced 0 keyterms on a real, larger Home
  Assistant installation.** The real log showed `ConnectionClosedError:
  sent 1009 (message too big); frame exceeds limit of 1048576 bytes` --
  websockets==17.1's own default `max_size` for `websockets.connect()` is
  1 MiB, and a real `get_states`/`config/entity_registry/list` response on
  a larger installation exceeds that. Fixed by passing an explicit,
  generous `max_size` (32 MiB) to `websockets.connect()` -- chosen over
  `max_size=None` (unlimited) even though the connection is strictly to
  the local, trusted `ws://supervisor/core/websocket`, so a pathological
  or malformed response still can't cause unbounded memory use. New tests
  deliberately build a `get_states` response and a
  `config/entity_registry/list` response that each exceed 1 MiB (verified
  via `json.dumps()` length) and confirm both are handled correctly.
- **The full startup banner was logged twice** -- once from inside
  transcriber loading (before Pocket TTS even started loading, so its
  info was necessarily stale) and once again after everything finished.
  Removed the premature, duplicate call.

### Performance
- **STT**: real production numbers showed RTF 1.6-1.95 (audio ~4.5-4.6s,
  inference 7.5-8.7s) -- slower than real-time on that hardware. Added
  per-chunk instrumentation (`chunks`, `add_audio_total`, `add_audio_max`,
  `avg_chunk` in the "STT completed" log line) so a real deployment can
  tell whether that's steady-state throughput (every chunk takes about
  the same, elevated time) or an isolated stall. Investigated
  moonshine-voice==0.1.5's native C API for a thread-count option
  (inspected its compiled library directly): confirmed none exists --
  only `vad_threshold`, `decode_incomplete_lines`, `keyterm_boost`, and
  `spelling_model_path` are recognized transcriber options. Per explicit
  instruction not to blindly set `OMP_NUM_THREADS`-style environment
  variables without a real, documented, effective API behind them, **no
  `stt_threads` option was added** -- this is a checked-and-rejected
  item, not something silently skipped.
- **TTS**: real production numbers showed TTFA 1.2-2.5s and model RTF
  1.5-2.0. Found that Pocket TTS's own library unconditionally forces
  single-threaded PyTorch CPU execution (`torch.set_num_threads(1)` at
  import time in its `tts_model` module) -- a real, plausible explanation
  for RTF consistently above 1.0. Added a new `tts_threads` option
  (default `0` = leave Pocket TTS's forced default alone; a positive
  value overrides it via the same real, documented
  `torch.set_num_threads()` API), applied once before the model loads.
  Voice state was already cached (confirmed, not new in this release);
  the "Prompting text took ..." cost visible in real logs comes from
  Pocket TTS's own flow-LM text-conditioning pass inside its `_generate()`
  method, which necessarily runs per request on the request's own text --
  it is not a cache that can be shared across different synthesize
  requests without changing what gets synthesized.
- **Benchmark tooling** (`app/benchmark.py`): now supports comparing
  multiple STT models in one run (`--models tiny,small`), the same tuning
  knobs the add-on itself has (`--transcription-interval`,
  `--decode-incomplete-lines`), synthetic keyterm-count benchmarking
  (`--keyterm-count`), and machine-readable output (`--output json/csv`).
  Still not run in CI and not a source of any number quoted in
  README/DOCS -- GitHub Actions runners are not representative of real
  Home Assistant hardware.

### Investigated, not changed (see ABSCHLUSSBERICHT_V0.2.4.md for detail)
- GPU device discovery warning (harmless on CPU-only hosts): no officially
  documented suppression API found in the native library; left as-is
  rather than risk fragile log manipulation.
- The `kyutai/pocket-tts` 401 fallback to the public
  `pocket-tts-without-voice-cloning` repo is entirely internal to
  upstream's own `TTSModel.load_model()` (a `try/except` around the gated
  download, with no parameter to skip straight to the fallback) -- not
  something this add-on can configure without monkey-patching upstream
  internals.
- Sentence-by-sentence TTS streaming (to reduce perceived TTFA for long
  replies): not implemented -- this development environment cannot run
  real Pocket TTS inference to actually measure whether it would help,
  and implementing unmeasured segmentation risked either no real benefit
  or audible seams for no verified gain.
- Real, verified tiny-vs-small and german-vs-german_24l hardware
  comparisons: not performed in this environment (no network access to
  download the models here) -- the extended benchmark tooling above lets
  the actual target hardware produce these numbers.

## [0.2.3] - 2026-09-18

Verification/stability release. No new STT engine, TTS model, fine-tuning, or
architecture changes. See `ABSCHLUSSBERICHT_V0.2.3.md` for the full report.

### Fixed
- **Stabilized the real German Moonshine STT end-to-end test.** The published
  v0.2.2 release tag's CI was actually red: the test used to synthesize its own
  German test audio with a Piper voice on every run, and one such run was
  mis-transcribed by Moonshine as "Schalte das Little Moons im Ei." instead of
  "Schalte das Licht im Wohnzimmer ein." -- an unnecessary source of flakiness,
  since the test's real purpose is to verify Moonshine, not Piper. It now uses
  a fixed, committed, deterministic audio fixture (`app/tests/fixtures/
  schalte_lampe_wohnzimmer.wav`, MIT/CC0-licensed Piper `thorsten-medium`
  voice output -- see `app/tests/fixtures/README.md` for full provenance) and
  no longer runs any TTS at all; Pocket TTS keeps its own separate e2e test.
  The fixture's own generation is also empirically self-verifying now: while
  preparing it, "Licht" turned out to be systematically misheard by Moonshine
  with this Piper voice across several real CI runs (differently each time,
  but reproducibly within one run), so the sentence was changed to "Schalte
  die Lampe im Wohnzimmer ein." and the generator script
  (`scripts/generate_stt_fixture.py`) now only accepts a candidate recording
  that a real Moonshine transcription round trip actually recognizes
  correctly, instead of trusting a single synthesis call to be usable.
- **Corrected an over-eager legacy/non-registry Home Assistant Assist exposure
  fallback.** Re-verified against the current home-assistant/core source: no
  read-only API can tell an explicit "hidden from Assist" apart from "never
  evaluated" for an entity with no entity registry entry. The previous
  implementation fell back to the same default-exposure rule registry entities
  use, which risked silently re-including, in the automatic STT vocabulary, a
  legacy entity a user had explicitly removed from Assist. Legacy entities are
  now only included when their exposure can be positively confirmed; an
  ambiguous one is simply left out.
- **Runtime and CI/e2e dependency versions no longer drift.** The e2e test jobs
  installed `moonshine-voice`/`wyoming`/`pocket-tts` unpinned, while the shipped
  Dockerfile pins exact versions -- real e2e tests could silently run against a
  different dependency stack than the add-on actually ships. Both now install
  from a single `requirements-runtime.txt`, which the Dockerfile also installs
  from directly; a new test guards `app/pyproject.toml`'s dependency metadata
  against drifting from it again.
- **Repository metadata**: `repository.yaml`'s `name` now reads "HomeIntent
  Moonshine Voice" (matching the add-on's own display name since v0.2.0,
  previously still "HomeIntent Moonshine STT"). The add-on slug is unchanged.

### CI
- The real Pocket TTS Wyoming end-to-end test now also runs automatically on
  every version-tag (`v*`) push, not just on manual `workflow_dispatch` --
  a release must never ship without this having actually run and passed.
- New strict release-gate mode (`E2E_REQUIRE_ONLINE=1`, set automatically for
  both real e2e jobs on a version-tag push): a genuine network/infrastructure
  failure fails the test outright instead of skipping it, so a tag's CI run
  can never look green purely because its e2e tests were silently skipped.
  Local/manual runs without this variable set may still skip on a real,
  classified infra failure, as before.
- A new `generate-stt-fixture` `workflow_dispatch`-only job can regenerate the
  STT audio fixture (uploaded as a build artifact for manual review) --
  needed because this add-on's own real e2e model downloads require network
  access a local development sandbox may not have.

## [0.2.2] - 2026-09-18

Targeted quality/stability pass on top of v0.2.1's Assist-aware vocabulary: no
new STT engine, no new TTS model, no architecture changes. See
`ABSCHLUSSBERICHT_V0.2.2.md` for the full verification report.

### Added
- **Legacy (non-registry) entities now contribute to the HA vocabulary**
  (`app/ha_vocabulary.py`): entities with no Home Assistant entity registry
  entry at all (state-machine-only) are identified from `get_states` minus
  `config/entity_registry/list`, and their effective Assist exposure is
  computed the same way Home Assistant Core's own
  `_async_should_expose_legacy_entity()` does -- reimplemented and verified
  against the real `home-assistant/core` `dev` branch source. An explicit
  override is read (read-only) via `homeassistant/expose_entity/list`, the
  only WS surface that reaches into legacy exposure settings at all; falling
  back to the same default-exposure rule as registry entities otherwise.
  Legacy entities use their live state's `attributes.friendly_name` (with
  the same entity-id fallback as registry entities) and never contribute
  Area combination terms, since they have no registry entry to resolve an
  area from.
- **`tts_warmup`** (default `true`): runs one discarded synthesis at startup
  so the first real TTS request isn't slower than later ones (general
  PyTorch-on-CPU first-forward-pass cost, not a Pocket-TTS-specific JIT
  effect -- verified against upstream source, no `torch.compile`/`torch.jit`
  call sites exist in its model path).
- **TTS voice validated at startup**: the configured `tts_voice` is now
  resolved (`get_state_for_audio_prompt()`) and cached once during add-on
  startup rather than lazily on the first real synthesize request. An
  invalid voice name now fails the add-on at startup with a clear error
  message instead of surfacing only when a user first tries to use it.
- **Detailed TTS performance breakdown** (`tts_log_performance`): the
  per-request log line now separately reports `lock_wait`, `model_compute`,
  `wyoming_send`, `wall`, `model_rtf`, and `wall_rtf`, instead of one opaque
  `synthesis`/`rtf` figure that could include lock-wait time or client
  backpressure. `model_rtf` reflects only Pocket TTS's own real compute
  time; `wall_rtf` is the full, real client-observed request cost.

### Fixed
- **Pocket TTS could send two `audio-start` events for one synthesize
  request**: if synthesis failed after already sending `audio-start` (and
  possibly some `audio-chunk`s), the error handler unconditionally sent a
  second `audio-start` before `audio-stop`/the error event -- a Wyoming
  protocol violation. Now tracked with an explicit per-request state: at
  most one `audio-start` per request, in every failure path (before the
  first chunk, mid-stream, or on a client disconnect).
- **Real GitHub Actions CI failure for the published Pocket TTS e2e test**:
  a bare `except Exception: pytest.skip(...)` around the real model
  download/round-trip would have silently hidden a genuine code or upstream
  API regression as a "skip" rather than failing the build. Both real e2e
  test files now distinguish an actual network/infrastructure failure
  (connection/timeout/DNS/proxy errors) from anything else, which is left
  to fail loudly.

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
