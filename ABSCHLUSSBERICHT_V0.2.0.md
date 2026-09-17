# Abschlussbericht: HomeIntent Moonshine Voice v0.2.0

**Datum**: 17. September 2026
**Autor**: Claude Sonnet 5
**Anlass dieser Fassung**: Zwei-Phasen-Auftrag. Phase A behebt die beim Review von
v0.1.2 gefundenen technischen Probleme. Phase B erweitert das Add-on um lokales
deutsches TTS mit Kyutai Pocket TTS über Wyoming, ohne den bestehenden
Moonshine-STT-Teil zu verschlechtern.

---

## Executive Summary

**Phase A (Review-Fixes): CODE FERTIG, LOKAL VOLLSTÄNDIG VERIFIZIERT.**
**Phase B (Pocket TTS): CODE FERTIG, LOKAL VOLLSTÄNDIG VERIFIZIERT (bis auf reale
Modell-Downloads).**
**GESAMT: NICHT als vollständig CI-grün bestätigt** — aus einem Grund, der außerhalb
des Codes liegt: Docker-Image-Pulls von `ghcr.io` und alle Kyutai/Hugging-Face-Domains
sind aus dieser Entwicklungsumgebung heraus durch die Netzwerk-Policy blockiert (siehe
unten). Der tatsächliche Docker-Build, der reale Container-/s6-/bashio-Smoke-Test und
jeder echte Modell-Download (Pocket TTS, Moonshine im Container) müssen im GitHub
-Actions-CI-Lauf (voller Internetzugang) verifiziert werden, nicht in dieser Sitzung.

Alles, was in dieser Umgebung tatsächlich ausführbar war, wurde ausgeführt und ist
grün: 238 Unit-/Integrationstests (Mocks/Fakes, kein echter Modell-Download), ruff
clean, mypy --strict clean, `pocket-tts` real installiert und seine API real gegen den
in dieser Sitzung geschriebenen Code getestet.

---

## Ausgangsstand

- **Vorherige Version**: 0.1.2
- **HEAD vorher**: `e2d51e5` ("Add ABSCHLUSSBERICHT_V0.1.2.md") auf
  `master`/`claude/moonshine-voice-stt-tts-73sjml`
- **Branch**: `claude/moonshine-voice-stt-tts-73sjml`

---

## Phase A: Review-Fixes

### A1 — CI Fake Supervisor / Config-Robustheit

**Problem**: Der CI-Fake-Supervisor-Stub (`.github/ci/fake_supervisor.py`) duplizierte
die `options`-Defaults aus `config.yaml` manuell und war bereits einmal (v0.1.2) hinter
der echten Konfiguration zurückgefallen. Das Startskript reichte
Supervisor-Config-Werte unverändert an argparse durch — ein leerer/`null`-Wert (z. B.
bei einer teilmigrierten `options.json` nach einem Upgrade) hätte z. B.
`--keyterm-boost null` erzeugt und argparse zum Absturz gebracht.

**Fix**:
- `fake_supervisor.py` parst jetzt `config.yaml`s `options:`-Block direkt (ein
  abhängigkeitsfreier Mini-Parser, da der Stub in einem nackten
  `python:3.11-slim`-Container ohne installierte Pakete läuft) statt eine eigene
  Kopie zu pflegen — kann strukturell nicht mehr divergieren.
- Das produktive Startskript (`rootfs/.../homeintent-moonshine/run`) liest jede Option
  über eine neue `config_or_default()`-Hilfsfunktion: fällt auf `config.yaml`s eigenen
  Default zurück, wenn der Supervisor-Wert fehlt, leer ist oder das literale
  `"null"` ist.
- Neuer CI-Job-Schritt in `build.yml`: startet das Add-on gegen einen zweiten
  Fake-Supervisor, dem mehrere neuere Optionen komplett fehlen (`CI_OMIT_OPTIONS`) und
  zwei weitere explizit `null` sind (`CI_NULL_OPTIONS`) — simuliert exakt das
  Upgrade-Szenario aus dem Review.
- Test `test_fake_supervisor_config_sync.py`: verifiziert, dass der Parser dieselben
  Keys/Werte wie `config.yaml` liefert (zweites, unabhängiges Sicherheitsnetz zusätzlich
  zur strukturellen Lösung).

**Verifiziert durch**: 21 neue/aktualisierte Tests (siehe unten), lokal grün. Der
reale Docker-Build + s6/bashio-Smoke-Test (inkl. des neuen Upgrade-Jobs) konnte in
dieser Sitzung **nicht** ausgeführt werden — `ghcr.io`-Image-Pulls sind aus dieser
Sandbox heraus von der Netzwerk-Policy blockiert (`403 Forbidden` beim `docker build`,
bestätigt per `curl .../__agentproxy/status`). Muss im echten CI-Lauf grün laufen.

### A2 — HA-Vokabular: Last-Known-Good

**Problem**: `fetch_ha_vocabulary()` gab bei jedem Fehlschlag (Timeout, Auth-Fehler,
unerreichbares HA) dieselbe leere Liste zurück wie bei einem echten, aber leeren
Registry-Ergebnis — der Aufrufer konnte beide Fälle nicht unterscheiden. Ein
periodischer Refresh, der während eines HA-Ausfalls lief, hätte das zuletzt erfolgreich
geladene Vokabular durch nichts ersetzt.

**Fix**: Neuer Typ `HaVocabularyResult(success: bool, terms: list[str])`.
`fetch_ha_vocabulary()` gibt jetzt explizit `success=False` bei jedem Fehler zurück,
`success=True` (auch mit leerer `terms`-Liste) bei einer echten, aber leeren Abfrage.
Der periodische Refresh-Loop in `__main__.py` behält bei `success=False` das zuletzt
erfolgreich geladene Vokabular unverändert bei und rührt den Transcriber gar nicht an;
nur ein `success=True`-Ergebnis ersetzt es (auch wenn das Ergebnis leer ist — das ist
ein legitimer, echter Zustand, kein Fehler).

**Verifiziert durch**: `test_ha_vocabulary_refresh.py` (7 Tests: erfolgreicher Refresh,
fehlgeschlagener Refresh behält alten Zustand, manuelle Keyterms überleben einen
Fehlschlag, echtes leeres Ergebnis ersetzt den alten Zustand, HA-Erholung nach Ausfall)
plus aktualisierte `test_ha_vocabulary.py`-Tests für den neuen Rückgabetyp.

### A3 — Moonshine `set_keyterms()`-Locking

**Problem**: `stream.start()`/`add_audio()`/`stop()` liefen bereits unter einem
gemeinsamen `asyncio.Lock`, aber der periodische HA-Vokabular-Refresh rief
`transcriber.set_keyterms()` außerhalb dieses Locks auf — eine potenzielle Race
Condition zwischen Vokabular-Refresh und laufendem Audio-Streaming auf demselben
nativen Transcriber-Handle. moonshine-voice 0.1.5 dokumentiert `set_keyterms()` nicht
als nebenläufigkeitssicher, also wurde konservativ vorgegangen (siehe Prompt-Vorgabe).

**Fix**: `set_keyterms()` läuft jetzt (per `asyncio.to_thread`) innerhalb desselben
`moonshine_lock`, der auch von jeder Wyoming-Session verwendet wird — ein einziger
Lock pro Transcriber, konsequent für alle nativen Aufrufe.

**Verifiziert durch**: `test_ha_vocabulary_refresh.py::TestSetKeytermsLocking` — ein
gezielter Test lässt eine simulierte Streaming-Session den Lock halten, während der
Refresh versucht `set_keyterms()` aufzurufen, und prüft per Ausführungsreihenfolge,
dass beide sich nie überlappen.

### A4 — Echter STT-RTF

**Problem**: Die geloggte "RTF" war `finalize_time / audio_duration` — aber Moonshine
transkribiert als Streaming-ASR bereits während `add_audio()`, sodass diese Zahl fast
nur die letzte `stop()`-Pass-Dauer maß, nicht die tatsächliche Modell-Rechenzeit.

**Fix**: `MoonshineStreamingSession` misst jetzt kumulativ die reale Wanduhrzeit jedes
`add_audio()`-Aufrufs (`inference_time_seconds`, akkumuliert) plus die
`stop()`-Pass-Dauer (`finalize_time_seconds`, separat exponiert). Das
Performance-Log zeigt beide getrennt:
```
STT completed: model=small audio=3.82s inference=0.61s finalize=0.08s rtf=0.16
```
`rtf = inference_time / audio_duration`. Die Zeit, die der Wyoming-Client zwischen
Chunks "wartet" (z. B. weil der Nutzer langsam spricht), zählt explizit **nicht** mit.

**Verifiziert durch**: `test_streaming.py::TestRealInferenceTiming` — drei Tests mit
künstlich verzögerten Mock-`add_audio()`/`stop()`-Aufrufen, die kumulative Messung,
korrekte Trennung Finalize/Inference, und dass reine Wartezeit zwischen Chunks nicht
mitgezählt wird.

### A5 — Area+Entity-Kombinationsvokabular

**Problem**: Das HA-Vokabular enthielt nur einzelne Namen (Areas, Devices, Entities),
keine kontextuellen Kombinationen wie "Wohnzimmer Rolllade".

**Fix**: Neue Funktion `_area_entity_combination_terms()` in `ha_vocabulary.py`,
die für jede Entity ihre **echte** Area ermittelt — entweder direkt (`entity.area_id`)
oder vererbt über `entity.device_id` → `device.area_id` — und bei Erfolg
`"{Area} {Entity}"` als zusätzlichen Term erzeugt. Keine geratenen Kombinationen: eine
Entity ohne ermittelbare Area erzeugt keinen Kombinationsterm. Entity-Domains
(`light`, `binary_sensor`, …) werden nie als Wörter verwendet — nur die bereits
bestehende Namensauflösung (echter `name`/`original_name`) fließt ein. Ergebnisse
werden dedupliziert, deterministisch sortiert (`casefold`) und bei
`MAX_COMBINATION_TERMS = 500` gedeckelt, um eine kombinatorische Explosion auf sehr
großen Installationen zu verhindern.

**Verifiziert durch**: `test_ha_vocabulary.py::TestAreaEntityCombinationTerms` — 9
Tests mit realistischen Registry-Fixtures (Wohnzimmer/Küche/Schlafzimmer/Garage mit
mehreren Entities, direkte und vererbte Area-Zuordnung, keine Kombination ohne
Area-Bezug, keine Domain-Wörter, keine redundante Verdopplung, deterministische
Sortierung, Deckelung bei Überschreitung).

---

## Phase B: Kyutai Pocket TTS

### B1 — Verifizierte Fakten zu Pocket TTS (nicht aus altem Wissen übernommen)

Alle folgenden Fakten wurden **während dieser Sitzung** gegen den echten
GitHub-Quellcode (`raw.githubusercontent.com/kyutai-labs/pocket-tts`, nicht
paywalled/blockiert), die echten PyPI-Metadaten, und das tatsächlich installierte
`pocket-tts==3.1.0`-Paket verifiziert — huggingface.co und kyutai.org waren aus dieser
Umgebung heraus nicht erreichbar (Netzwerk-Policy blockiert beide Domains), daher sind
Aussagen zu Modell-Lizenzfeldern auf Hugging Face explizit als unverifiziert markiert.

- **Paket**: `pocket-tts` auf PyPI (Import-Name `pocket_tts`), aktuelle Version zum
  Zeitpunkt dieser Sitzung **3.1.0**. Code-Lizenz: **MIT** (echte `LICENSE`-Datei
  gelesen).
- **Python/Runtime**: Python 3.10–3.14, `torch>=2.5.0` (keine GPU-Version nötig).
  Tatsächlich installiert und getestet: `torch==2.14.0` (zunächst versehentlich die
  CUDA-Variante von PyPI, siehe unten).
- **Deutsche Modelle, real verifiziert** (Config-Dateien im Repo geprüft, nicht nur
  Doku): `"german"` (6 Transformer-Layer, schneller) und `"german_24l"` (24 Layer,
  höhere Qualität, langsamer). Beide existieren real als eigene Config-YAMLs im
  Upstream-Repo. `"german_24l"` aus dem ursprünglichen Prompt war korrekt; `"german"`
  wurde zusätzlich real bestätigt (im offiziellen Docstring nicht explizit gelistet,
  aber im Code lädt `load_model()` einfach `{language}.yaml` ohne Allowlist-Prüfung —
  die Datei existiert und lädt).
- **Deutsche Preset-Stimme**: **`juergen`** ist die einzige echte deutsche Preset-Stimme
  (README-Stimmenkatalog + `_ORIGINS_OF_PREDEFINED_VOICES`-Dict im Quellcode
  bestätigt). Kein erfundener Name.
- **Streaming-API, real, kein Fake-Streaming**: `TTSModel.generate_audio_stream(voice_state,
  text) -> Iterator[torch.Tensor]` — verifiziert im Quellcode: startet intern einen
  Generierungs- und einen Mimi-Decode-Thread und liefert flache 1D-Mono-Chunks, sobald
  sie dekodiert sind. `generate_audio()` ist upstream selbst nur ein dünner Wrapper,
  der diesen Generator vollständig durchläuft und konkateniert — dieses Add-on nutzt
  bewusst die Streaming-Form direkt.
- **Audioformat**: mono, **24000 Hz**, float32-Samples (bestätigt in beiden
  Config-YAMLs: `mimi.sample_rate: 24000`, `mimi.channels: 1`).
- **Thread-Sicherheit**: **explizit NICHT thread-safe** dokumentiert (Docstring-Zitat:
  "separate model instances should be used for concurrent generation") — dieses
  Add-on serialisiert deshalb jeden Aufruf in dasselbe `TTSModel` über einen
  eigenen, von Moonshines STT-Lock unabhängigen Lock.
- **Voice-Cloning/Gating**: Die stimmklon-fähigen Gewichte (`kyutai/pocket-tts` auf
  Hugging Face) sind gated (Nutzungsbedingungen + `hf auth login` nötig). Der
  Quellcode fängt einen Download-Fehler dort automatisch ab und fällt auf die
  **ungated** Gewichte (`kyutai/pocket-tts-without-voice-cloning`) zurück
  (`has_voice_cloning = False`). **Wichtiger, in dieser Sitzung verifizierter Fakt**:
  `get_predefined_voice()` (für benannte Preset-Stimmen wie `"juergen"`) zeigt **immer**
  auf das ungated Repository, unabhängig von `has_voice_cloning` — die Default-Stimme
  dieses Add-ons benötigt also **nie** einen Hugging-Face-Login, auch nicht beim
  allerersten Start.
- **Modell-Cache**: `download_if_necessary()` delegiert `hf://`-Pfade an
  `huggingface_hub.hf_hub_download()`, das den Standard-Umgebungsvariablen
  `HF_HOME`/`HUGGINGFACE_HUB_CACHE` folgt — dieses Add-on setzt `HF_HOME=/data/models/pocket-tts`.
- **aarch64**: Der Quellcode hat einen dedizierten Pfad für ARM
  (`torch.backends.quantized.engine = "qnnpack"` für `arm64`/`aarch64` in
  `pocket_tts/quantization.py`) — echte Unterstützung im Code, aber keine
  Raspberry-Pi-spezifischen Performance-Zahlen von Kyutai selbst gefunden.
- **Bereits existierende Referenzimplementierung**: Das offizielle README listet
  `github.com/ikidd/pocket-tts-wyoming` ("Docker container for pocket-tts using
  Wyoming protocol, ready for Home Assistant Voice use") unter "Projects using Pocket
  TTS". Dieser Code wurde **nicht** eingesehen oder kopiert (aus Zeit-/Scope-Gründen
  in dieser Sitzung) — die hier gebaute Implementierung basiert eigenständig auf der
  verifizierten Kyutai-API und der bestehenden Add-on-Architektur. Wer diese
  Implementierung weiterentwickelt, sollte das Repo als Referenz prüfen.

### B2 — Wichtiger, während der Entwicklung entdeckter Bug: CUDA-Wheel-Fallback

Der von Kyutais eigenem README empfohlene Befehl
```
pip install pocket-tts --extra-index-url https://download.pytorch.org/whl/cpu
```
wurde in dieser Sitzung tatsächlich ausgeführt und **installierte trotzdem die
CUDA-Variante von PyTorch** (`torch==2.14.0+cu130` plus ~1 GB `nvidia-*`-Pakete) — pip
bevorzugte offenbar PyPIs Standard-Index gegenüber dem `--extra-index-url`. Das ist
exakt das Problem, das das README selbst beschreibt ("PyPI serves the CUDA build of
PyTorch by default … roughly 3 GB instead of 200 MB"), nur dass der dort empfohlene
Fix es in diesem konkreten pip/Umgebungs-Setup nicht vollständig löste.

**Fix im Dockerfile**: `torch` wird jetzt in einem **eigenen, vorgezogenen**
`pip install`-Schritt mit `--index-url` (ersetzt den Index, statt ihn nur zu
erweitern) installiert; `pocket-tts` wird danach separat installiert und findet seine
`torch>=2.5.0`-Abhängigkeit bereits erfüllt vor. Dieser zweistufige Ansatz konnte in
dieser Sandbox **nicht Ende-zu-Ende verifiziert** werden — `download.pytorch.org` ist
hier ebenfalls von der Netzwerk-Policy blockiert (`403` auch bei direktem `pip
install torch --index-url https://download.pytorch.org/whl/cpu`). Er beruht auf
dokumentiertem, verbreitetem pip-Verhalten (garantierte Auflösung einer bereits
erfüllten Abhängigkeit ohne erneuten Indexzugriff), muss aber im echten CI-Docker-Build
(voller Internetzugang) bestätigt werden.

**Nebenwirkung entdeckt und behoben**: `pocket-tts` verlangt `numpy>=2`, das Add-on
hatte bisher `numpy==1.24.3` fest gepinnt (ein eigener, nicht von moonshine-voice
erzwungener Pin). `moonshine-voice==0.1.5` deklariert selbst **keine**
numpy-Versionsbindung und wurde in dieser Sitzung real gegen `numpy==2.4.6` importiert
und getestet — funktioniert einwandfrei. Der Pin wurde projektweit (Dockerfile,
`requirements.txt`, `pyproject.toml`, alle drei CI-Workflows) auf `numpy==2.4.6`
angehoben.

### B3–B37 — Implementierung

- **Architektur** (siehe README): Ein Wyoming-TCP-Service (Port 10300) meldet ASR
  und/oder TTS abhängig von `stt_enabled`/`tts_enabled` (`Info(asr=[...], tts=[...])`,
  reale Wyoming-1.10.2-API, kein altes Beispiel übernommen — inklusive der neueren
  `SynthesizeStart`/`SynthesizeChunk`/`SynthesizeStop`/`SynthesizeStopped`-Klassen für
  inkrementelles Text-Streaming, die bewusst **nicht** implementiert wurden — siehe
  "Einschränkungen").
- **`app/tts.py`**: Modell-Laden (`load_tts_model()`), Modell-Metadaten,
  `DEFAULT_TTS_VOICE = "juergen"`.
- **`app/tts_session.py`**: `PocketTtsSynthesizer` — ein Lock pro geladenem Modell
  (unabhängig vom Moonshine-Lock), Voice-State-Cache (da `get_state_for_audio_prompt()`
  laut Upstream "relatively slow" ist), Thread+Queue-Brücke zwischen Pocket TTS'
  synchronem Generator und dem asyncio-Event-Loop — derselbe Musterentwurf, den
  Kyutais eigener `serve`-Befehl intern verwendet (`write_to_queue`/
  `generate_data_with_state` in `pocket_tts/main.py`, verifiziert, nicht kopiert).
- **`app/handler.py`**: `MoonshineAsrHandler` behandelt jetzt zusätzlich
  `synthesize`-Events; `_handle_describe()` baut ASR-/TTS-Programme unabhängig
  voneinander abhängig davon, ob `transcriber`/`tts_synthesizer` gesetzt sind.
  Audio-Chunks werden gestreamt, sobald Pocket TTS sie liefert (kein
  Buffer-dann-Senden). Leerer Text → leere, aber wohlgeformte
  audio-start/audio-stop-Antwort. Syntheses-Fehler → `WyomingError` +
  audio-stop, kein Absturz der ganzen Verbindung. Client-Disconnect
  mid-stream → `agen.aclose()` im `finally`-Block gibt Lock und Producer-Thread
  garantiert frei.
- **Performance-Logging**: `TTS completed: model=... chars=... ttfa_generated=...
  ttfa_sent=... synthesis=... audio=... rtf=...` — nie der synthetisierte Text.
- **`app/audio.py`**: `float32_to_pcm_int16()` mit Clipping vor der Konvertierung
  (kein Wraparound bei Überschwingern).
- **Konfiguration** (`config.yaml`): flach gehalten wie gewünscht;
  `stt_enabled`/`tts_enabled`/`tts_model`/`tts_voice`/`tts_log_performance` neu, alle
  bestehenden v0.1.x-Optionen unverändert (keine Umbenennung). `tts_enabled` defaultet
  auf `false` — Begründung siehe unten.
- **Name/Slug**: sichtbarer Name → "HomeIntent Moonshine Voice", Slug bewusst
  **unverändert** (`homeintent-moonshine-stt`), damit bestehende Installationen ein
  normales Update statt eines neuen Add-ons sehen.
- **Docker/aarch64**: `Dockerfile` installiert `torch` (CPU-Index) +
  `pocket-tts==3.1.0` zusätzlich zu den bisherigen Paketen; `HF_HOME`-Umgebungsvariable
  gesetzt; Healthcheck-/Discovery-Startzeit von 60s auf 180s erhöht (Pocket-TTS-Download
  beim ersten Start kann länger dauern). Der bestehende `build-aarch64`-CI-Job
  (QEMU, build-only) deckt jetzt automatisch auch die neuen Pakete ab — ob Pocket TTS
  auf echter aarch64-Hardware tatsächlich läuft (nicht nur baut), ist **nicht**
  verifiziert.

---

## Backward Compatibility — Entscheidung und Begründung

`tts_enabled` defaultet auf **`false`**. Ein bestehender reiner-STT-Nutzer, der
aktualisiert, bekommt beim nächsten Start **keinen** zusätzlichen Download (kein
PyTorch-Runtime-Download zur Laufzeit — der ist bereits im Docker-Image gebacken —
aber kein Pocket-TTS-Modell-Download, keine zusätzliche Startup-Zeit für TTS-Laden).
Alle bisherigen Optionsnamen (`model`, `language`, `log_level`, …) bleiben unverändert
— eine alte `options.json` mit nur diesen Feldern lädt unverändert (siehe A1 und
`test_main_config.py::TestJsonConfigOverrides::test_old_pre_tts_config_loads_without_error`).

---

## Architektur

Siehe README.md, Abschnitt "Architecture" — Home Assistant orchestriert die
Assist-Pipeline; Pocket TTS kommuniziert nie direkt mit dem HomeIntent
Conversation Agent, sondern nur über Wyoming mit Home Assistant.

## Home Assistant

Nach einem `describe`-Request meldet der Wyoming-Service (abhängig von
`stt_enabled`/`tts_enabled`):
- **Speech-to-Text**: "homeintent-moonshine" (Programmname im `Info.asr`-Eintrag)
- **Text-to-Speech**: "homeintent-pocket-tts" (Programmname im `Info.tts`-Eintrag),
  Stimme `juergen`

Getestet (mit Mocks, siehe unten) für alle vier Kombinationen: nur ASR, nur TTS, beide,
keins.

---

## Performance

### STT
Keine neuen Zahlen in dieser Iteration gemessen (Moonshine-Teil unverändert an der
Inferenz-Logik, nur an der Messung selbst — siehe A4). Frühere Werte bleiben
Herstellerangaben (WER) bzw. nutzerabhängig (RTF, siehe `app/benchmark.py`).

### TTS
**Nicht gemessen** (keine reale Modell-Inferenz in dieser Sandbox möglich —
huggingface.co blockiert). Alle in README/DOCS genannten Zahlen (~200ms bis zum ersten
Chunk, ~2–2.5x Echtzeit auf CPU, ~6x auf Apple-Silicon-MacBook) sind explizit als
**Herstellerangaben aus dem Kyutai-README** gekennzeichnet, nicht als eigene Messung.

```
model load:      nicht gemessen
warmup:           nicht gemessen
TTFA generated:   nicht gemessen
TTFA sent:        nicht gemessen
synthesis:        nicht gemessen
audio:            nicht gemessen
RTF:              nicht gemessen
RAM:              nicht gemessen
```

---

## Tests

```
pytest (app/tests/, -m "not e2e"):  238 passed, 5 deselected (e2e-Marker)
ruff check:                          All checks passed
ruff format --check:                 36 files already formatted
mypy --strict:                       Success: no issues found in 36 source files
docker build:                        NICHT ausführbar in dieser Umgebung (ghcr.io blockiert)
startup smoke test:                  NICHT ausführbar in dieser Umgebung (s. o.)
STT E2E (test_e2e_transcribe.py):    NICHT ausgeführt in dieser Sitzung (unverändert seit v0.1.2, real in CI)
TTS E2E (test_e2e_tts.py):           Geschrieben, lokal ausgeführt → sauber übersprungen
                                      ("Pocket TTS model/voice unavailable") mangels
                                      Netzwerkzugriff auf huggingface.co; nie ein falsches
                                      Grün vorgetäuscht
```

Neu hinzugekommene Testdateien: `test_fake_supervisor_config_sync.py` (16),
`test_ha_vocabulary_refresh.py` (7), `test_tts.py` (13), `test_tts_session.py` (10),
`test_handler_tts.py` (12), `test_main_config.py` (13), `test_e2e_tts.py` (3,
`e2e`-markiert). Erweiterte Dateien: `test_ha_vocabulary.py` (+9 Area/Entity-Tests, +1
Last-known-good-Test), `test_streaming.py` (+3 RTF-Tests), `test_audio.py` (+7
float32→PCM16-Tests).

Alle Unit-/Integrationstests laufen gegen Mocks/Fakes bzw. das echte,
lokal installierte `pocket-tts`-Python-API (Modell selbst gemockt) — kein Download
großer Gewichte in der normalen Testsuite, wie gefordert.

---

## CI

Vorhandene GitHub-Checks: `Lint`, `Type Check`, `Tests`, `Build` (mit den Jobs
`build-amd64`, `e2e-audio-transcribe`, `e2e-tts-synthesize` [neu,
`workflow_dispatch`-only], `build-aarch64`).

**Kein Check wurde in dieser Sitzung tatsächlich in GitHub Actions ausgeführt** — die
Änderungen liegen als lokaler, noch ungepushter Commit vor (siehe "Manuelle Schritte").
Alles, was *lokal* reproduzierbar war (pytest, ruff, mypy, YAML-Validierung der
Workflow-Dateien), ist grün; der reale Docker-Build und beide echten E2E-Jobs sind
**nicht** als grün bestätigt, bis der CI-Lauf das zeigt.

---

## Architektur-Support

- **amd64**: Docker-Build-Definition vorhanden und lokal syntaktisch/strukturell
  geprüft (Dockerfile parst, Layer-Reihenfolge korrekt); realer Build nicht
  durchführbar in dieser Sandbox.
- **aarch64**: Build-Job vorhanden (QEMU, build-only, wie zuvor) — deckt jetzt auch
  Pocket TTS ab. Pocket TTS hat einen echten ARM-Codepfad (QNNPACK-Quantisierung),
  aber ob das Modell auf echter aarch64-Hardware tatsächlich läuft, wurde **nicht**
  verifiziert (weder von diesem Projekt noch mit konkreten Zahlen von Kyutai selbst).

---

## Einschränkungen (real, nicht beschönigt)

- **Docker-Build und Container-Smoke-Test wurden in dieser Sitzung nicht real
  ausgeführt** — `ghcr.io`-Pulls sind in dieser Entwicklungsumgebung durch die
  Netzwerk-Policy blockiert. Muss im echten CI verifiziert werden, bevor diese
  Version als tatsächlich installationsbereit gilt.
- **Kein echter Pocket-TTS-Modell-Download/keine echte Inferenz wurde in dieser
  Sitzung durchgeführt** — huggingface.co ist blockiert. Der neue `test_e2e_tts.py`
  ist geschrieben und läuft sauber in den Skip-Pfad, wurde aber nie tatsächlich gegen
  ein echtes Modell verifiziert. Keine Performance-Zahl in diesem Bericht oder den
  Docs ist eine eigene Messung.
- **Der zweistufige CPU-Torch-Install (Dockerfile) ist nicht Ende-zu-Ende
  verifiziert** — `download.pytorch.org` ist ebenfalls blockiert. Basiert auf
  dokumentiertem pip-Verhalten, nicht auf einem beobachteten erfolgreichen Docker-Build
  in dieser Sitzung.
- **Kein Voice Cloning implementiert** (bewusst, wie gefordert) — nur Preset-Stimmen
  (Default `juergen`). Die Architektur (`PocketTtsSynthesizer.synthesize_stream(text,
  voice=...)`) verhindert das spätere Hinzufügen nicht.
- **Kein inkrementelles Text-Streaming** (`SynthesizeStart`/`-Chunk`/`-Stop`) — nur
  klassisches `Synthesize` (ganzer Text auf einmal rein), dafür mit echtem
  Audio-Output-Streaming. `supports_synthesize_streaming=False` wird entsprechend
  wahrheitsgemäß gemeldet.
- **Keine eigene deutsche Textnormalisierung** implementiert — Pocket TTS' eigenes
  Verhalten bei Zahlen/Uhrzeiten/Einheiten wurde mangels Modellzugriff nicht getestet;
  siehe `test_e2e_tts.py`s Testsätze (Uhrzeit, Temperatur), die genau das prüfen
  würden, sobald ein echter Lauf möglich ist.
- **`ikidd/pocket-tts-wyoming`** (vom offiziellen README als Referenzimplementierung
  gelistet) wurde nicht eingesehen — die hier gebaute Lösung ist eigenständig
  entwickelt, könnte aber von einem Abgleich profitieren.
- **Lizenzfeld der Pocket-TTS-Modellgewichte auf Hugging Face** konnte nicht
  verifiziert werden (huggingface.co blockiert) — nur die im GitHub-README zitierte
  Nutzungsbeschränkung ("Prohibited use"-Klausel) ist bestätigt.
- **Push zu GitHub war während dieser Sitzung nicht möglich** (Claude GitHub App nicht
  für dieses Repository autorisiert, `403` bei `git push`) — der Commit liegt lokal
  vor.

---

## Manuelle Schritte

Bevor diese Version als fertig gilt:

1. Diese Änderungen zu GitHub pushen (aktuell durch fehlenden App-Zugriff blockiert —
   siehe Einschränkungen) und den echten CI-Lauf abwarten.
2. Prüfen, dass `build-amd64` (inkl. des neuen Upgrade-Safety-Schritts),
   `e2e-audio-transcribe`, `build-aarch64`, `Lint`, `Type Check`, `Tests` alle grün
   sind.
3. Optional: `e2e-tts-synthesize` manuell über die Actions-UI (`workflow_dispatch`)
   auslösen, um Pocket TTS real gegen Netzwerk zu verifizieren.
4. Nach grünem CI: Add-on in Home Assistant aktualisieren, starten, Logs prüfen
   (Startup-Banner zeigt STT-/TTS-Status, Modelle, Stimme).
5. `tts_enabled: true` setzen, falls gewünscht, neu starten.
6. Assist-Pipeline öffnen, HomeIntent Moonshine als STT und/oder HomeIntent Pocket TTS
   als TTS auswählen.
7. Mit einem Sprachbefehl testen.
