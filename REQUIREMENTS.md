# Anforderungen: HomeIntent Moonshine STT v0.1

## Überblick
Neues, eigenständiges Home-Assistant-Add-on für echtes Moonshine German Streaming ASR über Wyoming.

**NICHT:** bestehendes `/home/philipp/homeintent-stt/` verändern.

## Kritische Entscheidung: Echtes Streaming

### Problem mit bestehender Lösung
```
cronus42/homeassistant-moonshine-addons:
  audio-start
  audio-chunk
  ...
  audio-stop
  → komplettes Audio gepuffert
  → transcribe() aufgerufen
```

### Unsere Lösung
```
Wyoming audio-chunk
  → sofort an Moonshine Streaming API
  → Streaming Decoder verarbeitet WÄHREND der Sprache
  → nicht: erst nach audio-stop
```

## Research-Abhängigkeiten
- [ ] Moonshine aktuelle Streaming API (Python Package, Klassen, Methods)
- [ ] Exakte deutsche Modellnamen (Tiny, Small)
- [ ] HA Add-on Repository Spezifikation (aktuell)
- [ ] Wyoming STT Protocol Details

## Architektur
```
ReSpeaker / Voice Satellite
  ↓
Home Assistant Assist
  ↓
Wyoming Audio Chunks (16kHz mono 16-bit PCM)
  ↓
HomeIntent Moonshine STT Add-on
  ├── Wyoming Server (Port 10300)
  ├── Moonshine Streaming Session Manager
  ├── Audio Chunk Handler
  └── German Streaming Decoder
  ↓
Transcript
  ↓
HomeIntent / Conversation Agent
```

## v0.1 Scope

### Must-Have
- [ ] Wyoming STT Service (Port 10300)
- [ ] Moonshine German Streaming ASR
- [ ] Tiny + Small Modelle auswählbar
- [ ] German language support verified
- [ ] CPU-only inference
- [ ] Echtes Streaming (nicht buffering)
- [ ] HA Add-on Store Installation
- [ ] config.yaml (model, language, log_level)
- [ ] Multi-arch: amd64, aarch64
- [ ] Model Cache (persistent)
- [ ] Tests
- [ ] README + DOCS.md

### Nice-to-Have (später)
- [ ] Keyterms/Context API
- [ ] Partial Transcripts
- [ ] HA Entity Integration
- [ ] Performance Metrics

### Not Planned
- [ ] Model Training
- [ ] Fine-Tuning
- [ ] GPU Support
- [ ] HA Token/API Access

## Repository-Struktur

```
homeintent-moonshine-addon/
├── repository.yaml              # Add-on Store Registration
├── README.md                    # Benutzer Dokumentation
├── LICENSE                      # MIT oder ähnlich
├── CHANGELOG.md                 # Versions Log
├── .github/
│   └── workflows/
│       ├── lint.yml            # ruff
│       ├── type-check.yml      # mypy
│       ├── test.yml            # pytest
│       └── build.yml           # Docker Multi-Arch
│
└── homeintent-moonshine-stt/    # Add-on Package
    ├── config.yaml              # Add-on Konfiguration
    ├── build.yaml               # Docker Build Multi-Arch
    ├── Dockerfile               # Docker Image
    ├── DOCS.md                  # In-App Dokumentation
    ├── CHANGELOG.md             # Add-on spezifisch
    ├── run.sh                   # Entrypoint
    │
    └── app/
        ├── __init__.py
        ├── __main__.py          # Entry Point
        ├── server.py            # Wyoming Server (asyncio)
        ├── handler.py           # Request Handler
        ├── streaming.py         # Moonshine Streaming Manager
        ├── models.py            # Model Loading + Registry
        ├── audio.py             # Audio Format Handling
        ├── logger.py            # Structured Logging
        │
        └── tests/
            ├── __init__.py
            ├── conftest.py
            ├── test_server.py
            ├── test_handler.py
            ├── test_streaming.py
            ├── test_models.py
            ├── test_audio.py
            └── test_streaming_e2e.py
```

## Modelle (zu verifizieren)
Exakte Namen TBD nach Research:
- German Tiny Streaming
- German Small Streaming (Standard)

## Konfiguration
```yaml
model: "moonshine-german-small-streaming"  # TBD
language: "de"
log_level: "INFO"
```

## Testing Strategy
1. **Unit Tests**: Wyoming Lifecycle, Audio Format, Model Config
2. **Streaming Regression Test**: Beweist, dass Chunks sofort verarbeitet werden
3. **Integration Test**: Full lifecycle mit Mock Moonshine Backend
4. **CI**: lint (ruff), typecheck (mypy), tests (pytest)

## Known Unknowns
- [ ] Moonshine Streaming API Details
- [ ] German Model Naming Convention
- [ ] Moonshine Concurrency Safety (Serial vs Parallel Streams)
- [ ] Wyoming Discovery/Metadata Format (aktuell in HA)
- [ ] HA Add-on Build Best Practices (aktuell)
- [ ] cronus42 Implementation Details (für Referenz)

## Ressourcen
- Bestehend (NICHT kopieren): https://github.com/cronus42/homeassistant-moonshine-addons
- Bestehend (NICHT verändern): /home/philipp/homeintent-stt/
- Moonshine: github.com/moonshine-ai/moonshine
- Wyoming: github.com/rhasspy/wyoming
- HA Docs: developers.home-assistant.io

## Abnahmekriterien v0.1
- HA Add-on Store Installation funktioniert
- Wyoming läuft
- German Tiny/Small auswählbar
- Echtes Streaming funktioniert (Test verbietet Regression)
- Alle Tests grün
- README + DOCS.md
- Keine manuellen /addons Schritte
- Abschlussbericht mit Architektur + Streaming-Nachweis
