# Abschlussbericht v0.2.4 – Real-Hardware Performance + HA Vocabulary Fix

Datum: 2026-09-18

## Real Hardware Ausgangswerte

Diese Zahlen stammen aus dem echten Home-Assistant-Assist-Test des Nutzers
(nicht aus dieser Entwicklungsumgebung) und sind der Ausgangspunkt dieser
Runde:

**STT (Moonshine `small`)**

| Beispiel | audio | inference | finalize | RTF |
|---|---|---|---|---|
| 1 | 4.49s | 8.74s | 1.79s | 1.95 |
| 2 | 4.62s | 7.46s | 1.59s | 1.61 |

**Pocket TTS**

| Beispiel | chars | ttfa_generated | model_compute | audio | model_rtf |
|---|---|---|---|---|---|
| 1 | 79 | 2.471s | 7.23s | 3.68s | 1.97 |
| 2 | 89 | 1.265s | 5.16s | 3.36s | 1.53 |

**HA Vocabulary**: fehlgeschlagen mit `ConnectionClosedError: sent 1009
(message too big); frame exceeds limit of 1048576 bytes` → `effective
keyterms=0` trotz `use_ha_vocabulary=true`.

## HA Vocabulary Fix

- **Ursache**: `websockets==17.1`s eigener Default für `max_size` in
  `websockets.connect()` ist 1 MiB (1.048.576 Bytes). Eine reale, größere
  Home-Assistant-Installation liefert bei `get_states` oder
  `config/entity_registry/list` leicht mehr als das.
- **Lösung**: `websockets.connect(..., max_size=MAX_WS_MESSAGE_SIZE)` mit
  `MAX_WS_MESSAGE_SIZE = 32 * 1024 * 1024` (32 MiB).
- **max_size-Entscheidung**: `max_size=None` (unlimitiert) wurde bewusst
  NICHT gewählt, obwohl die Verbindung ausschließlich zur lokalen,
  vertrauenswürdigen `ws://supervisor/core/websocket` geht. Ein großes,
  aber endliches Limit schützt weiterhin vor unbegrenztem Speicherverbrauch
  bei einer fehlerhaften/pathologischen Antwort, deckt aber auch sehr große
  reale Installationen (zehntausende Entities) komfortabel ab.
- **Tests**: Drei neue Tests in `test_ha_vocabulary.py`:
  1. `websockets.connect()` wird tatsächlich mit dem großen `max_size`
     aufgerufen (nicht nur behauptet).
  2. Eine `get_states`-Antwort, die nachweislich (per `json.dumps()`-Länge
     geprüft) über 1 MiB liegt, wird korrekt verarbeitet — Exposure und
     Vokabular bleiben korrekt.
  3. Dieselbe Prüfung für eine über 1 MiB liegende
     `config/entity_registry/list`-Antwort.

## STT Performance

- **Bottleneck-Sichtbarkeit**: Die "STT completed"-Logzeile enthält jetzt
  zusätzlich `chunks`, `add_audio_total`, `add_audio_max`, `avg_chunk` —
  damit lässt sich auf echter Hardware unterscheiden, ob die Verarbeitung
  gleichmäßig langsamer als Echtzeit ist (Backlog durch generelle
  CPU-Langsamkeit) oder ob ein einzelner Ausreißer-Chunk das Problem ist.
- **Tiny vs. Small**: NICHT auf echter Hardware verglichen — diese
  Sandbox hat keinen Netzwerkzugriff auf `download.moonshine.ai`, um die
  Modelle herunterzuladen. `app/benchmark.py` unterstützt jetzt
  `--models tiny,small` in einem Lauf, damit der Nutzer dies auf der
  echten Home-Assistant-Maschine selbst durchführen kann.
- **Threads**: Das native `moonshine-voice==0.1.5`-C-API (per `strings`
  auf der kompilierten `libmoonshine.so` direkt geprüft) erkennt nur
  `vad_threshold`, `decode_incomplete_lines`, `keyterm_boost` und
  `spelling_model_path` als Optionen — **keine** Thread-Konfiguration
  existiert. Es wurde bewusst **keine** `stt_threads`-Option und **kein**
  `OMP_NUM_THREADS` o.ä. gesetzt, da die Aufgabenstellung explizit
  verlangt, nur tatsächlich wirksame APIs zu verwenden.
- **Chunking/Interval/Incomplete-Decoding**: NICHT auf echter Hardware
  benchmarkt (gleicher Netzwerk-Grund). `app/benchmark.py` unterstützt
  jetzt `--transcription-interval` und `--decode-incomplete-lines` als
  Parameter für genau diesen Vergleich durch den Nutzer.
- **Keyterm-Overhead**: NICHT auf echter Hardware gemessen.
  `app/benchmark.py --keyterm-count N` generiert N synthetische Keyterms
  zum Testen, ohne eine echte Home-Assistant-Instanz zu benötigen.

## TTS Performance

- **Wichtigster Fund**: Pocket TTS erzwingt in seiner eigenen Bibliothek
  beim Modul-Import **unbedingt** `torch.set_num_threads(1)`
  (`pocket_tts/models/tts_model.py`, Zeile 57) — verifiziert durch direktes
  Lesen des installierten Pakets, nicht angenommen. Das ist eine plausible,
  konkrete Erklärung für die durchgehend über 1.0 liegende model_rtf: Pocket
  TTS nutzt standardmäßig nur einen CPU-Kern.
- **Neue Option `tts_threads`** (Default `0` = unverändert lassen): ruft
  die echte, dokumentierte `torch.set_num_threads()`-API auf, bevor das
  Modell geladen wird (torch verlangt dies laut eigener Doku, um
  zuverlässig zu wirken). Nur Intra-Op-Threads werden gesetzt;
  `set_num_interop_threads()` wird bewusst NICHT verwendet, da diese API
  laut PyTorch nur einmal, vor jeglicher paralleler Arbeit im gesamten
  Prozess, sicher gesetzt werden kann — das kann dieses Add-on nicht
  garantieren.
- **Voice-State-Caching**: bereits vorhanden (nicht neu in dieser Version,
  verifiziert in `app/tts_session.py`s `_voice_state_cache`).
- **"Prompting text took ..." (im echten Log sichtbar)**: stammt aus Pocket
  TTS' eigenem `_generate()` (`pocket_tts/models/tts_model.py`), das dort
  den Flow-LM-Forward-Pass über den jeweiligen Anfragetext misst. Das ist
  echte, pro Anfrage unvermeidbare Modellarbeit — der Text unterscheidet
  sich pro Anfrage, daher ist dieser Anteil grundsätzlich nicht über
  Anfragen hinweg cachebar, ohne die Semantik zu verändern.
- **german vs. german_24l**: NICHT auf echter Hardware verglichen (gleicher
  Netzwerk-Grund). Die Modellwahl bleibt unverändert konfigurierbar.
- **Satzweise TTS-Synthese** (Item 22): NICHT implementiert. Ohne echte
  Pocket-TTS-Inferenz in dieser Umgebung lässt sich ein tatsächlicher
  TTFA-Vorteil nicht nachweisen; eine ungemessene Segmentierung hätte das
  Risiko hörbarer Brüche ohne verifizierten Nutzen bedeutet — laut Vorgabe
  nur bei nachgewiesenem Nutzen implementieren.
- **GPU-Warnung** (`GPU device discovery failed`): stammt aus der in
  `libmoonshine.so` statisch eingebundenen ONNX-Runtime-GPU-Execution-
  Provider-Erkennung; keine von Python aus ansprechbare, offizielle
  Unterdrückungs-API gefunden. Auf CPU-only-Systemen harmlos. Belassen wie
  sie ist, um keine fragile Log-Manipulation einzuführen.
- **HuggingFace-401-Log beim `kyutai/pocket-tts`-Fallback**: vollständig
  intern in Pocket TTS' eigenem `TTSModel.load_model()` (`try/except` um
  den gated Download, kein Parameter, um direkt das öffentliche
  `pocket-tts-without-voice-cloning`-Repo zu wählen) — nicht ohne
  Monkey-Patching upstream-interner Logik veränderbar. Wie in der Vorgabe
  gefordert dokumentiert, nicht gepatcht.

## Empfehlungen

**Für maximale Geschwindigkeit** (unverifiziert auf Zielhardware, da hier
nicht messbar):
- STT: `model: tiny` ausprobieren und mit `app/benchmark.py --models
  tiny,small` gegen echte Sätze vergleichen, bevor der Default geändert
  wird.
- TTS: `tts_threads` schrittweise erhöhen (z.B. 2, 4) und dabei
  `tts_log_performance`s `model_rtf`/`ttfa_generated` beobachten — Vorsicht
  vor CPU-Oversubscription, falls STT und TTS gleichzeitig aktiv sind.

**Für maximale Qualität** (Status quo, unverändert):
- STT: `model: small` (Default) bleibt empfohlen, bis ein echter Benchmark
  zeigt, dass `tiny` auf der Zielhardware ausreichend genau ist.
- TTS: `tts_model: german_24l` für höhere Qualität, falls das
  Geschwindigkeitsbudget das zulässt (nicht real verglichen).

Keine dieser Empfehlungen ersetzt einen echten Benchmark auf der
Ziel-Hardware — das ist genau der Zweck der erweiterten
`app/benchmark.py`.

## Tests

- ruff check: **GRÜN**
- ruff format --check: **GRÜN**
- mypy (`--config-file app/pyproject.toml`, korrekter `--python-executable`
  wegen mehrerer Interpreter in dieser Sandbox): **GRÜN**, 0 Fehler
- pytest (lokal, `-m "not e2e"`): **349 bestanden**, 4 e2e-Tests korrekt
  deselektiert
- amd64/aarch64 Docker-Build: siehe finaler Status weiter unten (CI-Ergebnis
  wird nach Push dokumentiert)
- Real STT E2E: siehe finaler Status weiter unten
- Real Pocket TTS E2E: siehe finaler Status weiter unten

## Offene Limits (ehrlich)

- Keine STT-Thread-Option, da keine wirksame API existiert (siehe oben) —
  das ist eine bestätigte Einschränkung von moonshine-voice==0.1.5, keine
  Unterlassung dieser Runde.
- Kein echter Tiny-vs-Small- oder German-vs-German_24l-Hardwarevergleich in
  dieser Umgebung durchgeführt — nur die Tooling-Grundlage dafür wurde
  geschaffen.
- Keine Aussage, ob `tts_threads > 0` die reale TTFA/RTF auf der
  Zielhardware tatsächlich verbessert — das muss der Nutzer selbst mit
  `tts_log_performance` verifizieren.
- Satzweise TTS-Synthese nicht implementiert (siehe oben).
- Pocket TTS bleibt, wie explizit gefordert, unverändert als Engine
  bestehen. Falls echte Messungen auf der Zielhardware nach sauberem
  Tuning weiterhin dauerhaft RTF > 1 und hohe TTFA zeigen: **Pocket TTS is
  functionally correct but not fast enough for the desired low-latency
  CPU-only Assist experience on the tested hardware** wäre die ehrliche
  Schlussfolgerung — das wurde in dieser Runde nicht abschließend
  verifiziert, da keine echte Inferenz in dieser Sandbox möglich ist.
