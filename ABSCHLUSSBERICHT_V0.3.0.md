# Abschlussbericht v0.3.0 — Kokoro German ONNX als zweite TTS-Engine

Dieser Bericht deckt alle 20 im Auftrag geforderten Punkte ab. Wo eine Zahl
aus diesem Sandbox-Environment nicht real gemessen werden konnte (Netzwerk-
Egress zu `huggingface.co` ist in dieser Entwicklungsumgebung blockiert —
`curl` liefert "CONNECT tunnel failed, response 403", `WebFetch` liefert
`EGRESS_BLOCKED`), wird das explizit als solches ausgewiesen statt eine Zahl
zu erfinden. Der reale Download/Betrieb auf der Ziel-Hardware (und in GitHub
Actions CI) ist von dieser Sandbox-Einschränkung nicht betroffen.

## 1. Verwendete kokoro-onnx-Version

`kokoro-onnx==0.6.1` (verifiziert durch Herunterladen und Lesen des echten
Wheel-Inhalts: `kokoro_onnx/__init__.py`, `session.py`, `config.py`,
`tokenizer.py`). Dies ist die aktuelle stabile Version zum Zeitpunkt der
Implementierung — bewusst NICHT die im Referenz-Repo verwendete ältere
Version 0.5.0, wie explizit gefordert. `onnxruntime==1.30.0` wird zusätzlich
explizit gepinnt (nicht nur transitiv über kokoro-onnx bezogen), um
reproduzierbare Builds sicherzustellen.

## 2. Martin-Modell-Repo und Revision

`Godelaune/Kokoro-82M-ONNX-German-Martin` auf Hugging Face (NICHT der vom
Auftrag als möglicherweise unzuverlässig markierte Name
`huggingFresse/Kokoro-82M-ONNX-German-Martin` — dieser wurde verifiziert
verworfen). Verifiziert über `raw.githubusercontent.com` (da `huggingface.co`
selbst aus dieser Sandbox nicht erreichbar war), mit zwei unabhängigen,
wortwörtlichen Zitat-Abfragen zur Gegenprobe gegen eine erste, vermutlich mit
einem anderen (nicht-deutschen) Repo verwechselte Antwort.

Dateien: `kokoro-martin.onnx` und `voices-martin.npz`.

Revision: Es konnte aus dieser Sandbox **keine unveränderliche, gepinnte
Commit-Revision** (SHA) verifiziert werden — der Download-Code
(`app/kokoro_tts.py`) verwendet `DEFAULT_KOKORO_REVISION = "main"` mit einem
expliziten, dokumentierten Hinweis im Code und in README/DOCS, dass dies
ehrlich als bekanntes Restrisiko (Punkt 20) ausgewiesen ist, statt eine
erfundene Revision anzugeben. `resolve_kokoro_model_files()` ist so gebaut,
dass eine echte Revision (z. B. ein Commit-SHA) jederzeit nachgetragen werden
kann, ohne die Downloadlogik selbst zu ändern.

## 3. Lizenz

- Kokoro-onnx (Runtime-Paket): MIT.
- Godelaune/Kokoro-82M-ONNX-German-Martin (Modell): Apache 2.0, laut
  README-Zitat "The model is published under Apache 2.0 on Hugging Face"
  (aufgebaut auf hexgrad/Kokoro-82M via StyleTTS2 Stage 2, Basis
  kikiri-german-base-51speakers-synthetic).
- Beide Lizenzangaben sind in README.md (Abschnitt "License") dokumentiert,
  inklusive des ehrlichen Hinweises zur fehlenden gepinnten Revision.

## 4. Neue Abhängigkeiten

`requirements-runtime.txt` / `app/pyproject.toml`:
- `huggingface_hub==1.32.0` (Modell-Download mit eingebauter
  Atomic-Download-Semantik)
- `kokoro-onnx==0.6.1`
- `onnxruntime==1.30.0` (explizit gepinnt für Reproduzierbarkeit)
- `espeakng-loader==0.2.4` (liefert vorkompilierte espeak-ng-Bibliotheken +
  Daten für amd64 UND aarch64 als reines Python-Wheel — **kein**
  System-`espeak-ng`-apt-Paket nötig, korrigiert damit eine Annahme aus dem
  Auftrag)
- `phonemizer==3.4.0` (von kokoro-onnx intern für Phonemisierung genutzt)

Explizit NICHT hinzugefügt: FastAPI, uvicorn oder andere Webserver-Pakete —
Kokoro läuft vollständig in-process.

## 5. Geänderte/neue Dateien

Neu:
- `app/tts_engine.py` — generalisiertes `TtsSynthesizer`-Protocol
- `app/kokoro_tts.py` — Modellauflösung/-download/-laden
- `app/kokoro_session.py` — `KokoroOnnxSynthesizer`
- `app/german_text_normalizer.py` — eigenständige deutsche Textnormalisierung
- `app/tests/test_kokoro_tts.py`, `test_kokoro_session.py`,
  `test_german_text_normalizer.py`, `test_tts_engine_selection.py`,
  `test_e2e_kokoro_tts.py`

Geändert:
- `app/tts_session.py` (nutzt jetzt `app.tts_engine`, neue Metadaten-Properties)
- `app/handler.py` (generalisierte `describe`/Performance-Log-Logik)
- `app/__main__.py` (Engine-Dispatch, neue CLI-Args, Startup-Logging)
- `app/validation.py` (neue Kokoro-Validatoren)
- `app/tts_benchmark.py` (Multi-Engine-Benchmark, 4 Testsätze)
- `config.yaml`, `Dockerfile`, `requirements-runtime.txt`, `app/pyproject.toml`
- `rootfs/etc/s6-overlay/s6-rc.d/homeintent-moonshine/run`
- `.github/workflows/build.yml` (neuer `e2e-kokoro-tts-synthesize`-Job)
- `README.md`, `DOCS.md`, beide `CHANGELOG.md`
- Bestehende Test-Fakes (`test_handler_tts.py`, `test_handler_tts_streaming.py`)

## 6. Neue TTS-Architektur

Ein gemeinsames `TtsSynthesizer`-Protocol (`app/tts_engine.py`) mit
`sample_rate`, `default_voice`, `engine_id`, `model_name`, `program_name`,
`attribution`, `description` und `async synthesize_stream(text, voice,
stats)`. `PocketTtsSynthesizer` und `KokoroOnnxSynthesizer` implementieren es
identisch; `handler.py` und `__main__.py` behandeln beide Engines über
dasselbe Interface — es gibt keine `if engine == ...`-Verzweigung im
Wyoming-Request/Response-Pfad. Die bestehende Wyoming-Streaming-Logik
(`synthesize-start/-chunk/-stop/-stopped`, `TtsStreamState`) wurde
unverändert wiederverwendet.

## 7. Home-Assistant-Konfiguration

Neue Option `tts_engine: pocket_tts | kokoro_onnx` (Default: `pocket_tts` —
bestehende Installationen ändern ihr Verhalten nicht). Neue Kokoro-Optionen:
`kokoro_voice` (Default `martin`), `kokoro_speed`, `kokoro_threads`
(0 = auto, gleiche Semantik wie `tts_threads`), `kokoro_sentence_pause`,
`kokoro_clause_pause`. Alle bestehenden Pocket-TTS-Optionen bleiben
unverändert bestehen.

Zur Schema-Migration: Eine echte Verifikation gegen den HA-Supervisor-
Quellcode war aus dieser Sandbox nicht möglich (kein lokaler Supervisor-
Checkout, GitHub/HF-Netzwerk blockiert). Die Absicherung stützt sich
stattdessen ehrlich auf das bereits etablierte, in diesem Projekt schon
mehrfach bewährte Muster (`config_or_default()` im Run-Skript,
Options-Whitelist in `_load_json_config_overrides()`), mit dem bereits
v0.2.0 gefahrlos von einer reinen STT-Installation auf TTS-Optionen erweitert
wurde. Das ist explizit KEINE aus dem Supervisor-Quellcode verifizierte
Garantie, sondern eine begründete Fortführung des bestehenden Vorgehens.

## 8. Modell-Download/Cache-Verhalten

Download via `huggingface_hub.hf_hub_download()` (bietet eingebaute
Atomic-Download-Semantik: Download in temporäres Verzeichnis, Umbenennen erst
nach vollständigem Erfolg — kein handgerollter HTTP-Downloader nötig) nach
`/data/models/kokoro-onnx/` (`KOKORO_ONNX_CACHE`-Umgebungsvariable). Bereits
durch das bestehende `backup_exclude: ["models/*"]` abgedeckt. Download
erfolgt NUR wenn `tts_enabled: true` UND `tts_engine: kokoro_onnx` — die
Auswahl von Pocket TTS löst nie einen Kokoro-Download aus.
`resolve_kokoro_model_files()` validiert beide Dateien als nicht-leer;
`load_kokoro_model()` schlägt bei fehlgeschlagenem/unvollständigem Download
mit `KokoroModelDownloadError` klar fehl — es gibt KEIN stilles Zurückfallen
auf Pocket TTS, wenn der Nutzer Kokoro ausgewählt hat.

## 9. Deutsche Textnormalisierung

Eigenständige, getestete Implementierung (`app/german_text_normalizer.py`,
38 Tests) statt Vendoring des Referenz-Codes — bewusste Entscheidung, weil
für dessen `german_text_rules.py` keine überprüfbare Lizenzdatei gefunden
werden konnte (404 am erwarteten GitHub-Pfad). Deckt ab: Uhrzeiten
("18:20 Uhr"), Temperaturen (°C/°F/bloßes °), Einheiten (kWh, km/h, kg, mg,
ml, min, m, g, l, %, h — mit sorgfältiger Regex-Reihenfolge, damit z. B. "5
mg" nicht fälschlich als "5 Meter g" gelesen wird), Währung (€/EUR), und
Abkürzungen (Std., Min., zzgl., ggf., Stck., bzw., z.B., ca., usw.) inklusive
korrekter Behandlung von Satzendpunkten bei abgekürzten Wörtern am
Textende.

## 10. Phonemisierung

kokoro-onnx nutzt intern `phonemizer.phonemize(text, lang="de",
preserve_punctuation=True, with_stress=True)`, das wiederum espeak-ng
nutzt. Die Espeak-ng-Bibliothek/-Daten werden automatisch über
`espeakng_loader.get_data_path()`/`get_library_path()` verdrahtet (verifiziert
aus dem `Tokenizer.__init__`-Quellcode) — keine manuelle Konfiguration nötig.
Deutsche Umlaute und ß bleiben erhalten (keine ASCII-Transliteration), da die
Normalisierung selbst nur Zahlen/Einheiten expandiert und den restlichen Text
unverändert lässt.

Eine reale Aussprache-Verifikation von Testwörtern wie "Küche", "Büro",
"Außenlicht", "Gäste-WC", "Rollladen", "Waschmaschine", "Jürgen" mit dem
tatsächlich geladenen Modell konnte in dieser Sandbox NICHT durchgeführt
werden (Modell-Download blockiert) — das ist im Risikoabschnitt (Punkt 20)
als offener Punkt vermerkt.

## 11. Streaming-Implementierung

`KokoroOnnxSynthesizer.synthesize_stream()` normalisiert den Text, ruft
dann `kokoro.create_stream(normalized_text, voice=..., speed=..., lang="de",
trim=True, sentence_pause=..., clause_pause=...)` auf — die echte
`AsyncGenerator[tuple[NDArray[float32], int], None]`-Streaming-API von
kokoro-onnx 0.6.1 (verifiziert aus dem Quellcode, keine erfundene Signatur).
Jeder erzeugte float32-Chunk wird direkt an die bestehende
`synthesize_stream()`-Pipeline weitergereicht (float32 → int16 PCM →
Wyoming `AudioChunk`) — es wird zu keinem Zeitpunkt eine vollständige WAV-
Datei im Speicher zusammengesetzt. Home Assistant erhält damit den ersten
Audio-Chunk, während Kokoro den Rest noch generiert.

## 12. Thread-/ONNX-Einstellungen

`kokoro_onnx.session.create_session()` bietet KEINE Thread-Konfiguration
(verifiziert aus dem Quellcode). Stattdessen baut `app/kokoro_tts.py` eine
eigene `onnxruntime.InferenceSession` mit expliziten `SessionOptions`
(`intra_op_num_threads`, `inter_op_num_threads`) und übergibt sie via
`Kokoro.from_session()`. Da kokoro-onnx bzw. ONNX Runtime nicht explizit als
threadsicher für parallele Syntheseanfragen dokumentiert sind, werden alle
Syntheseanfragen wie bei Pocket TTS über ein `asyncio.Lock` serialisiert
(Korrektheit vor maximaler Parallelität, wie explizit gefordert).

Ein echtes Benchmarking der optimalen Thread-Zahl auf der Ziel-Hardware
(Intel Core i5-12450H) konnte aus dieser Sandbox nicht durchgeführt werden
(Modell-Download blockiert) — das dafür gebaute Werkzeug
(`python -m app.tts_benchmark --engine kokoro_onnx --threads
1,2,4,6,8,auto`) ist einsatzbereit und muss vom Nutzer auf der echten
Hardware ausgeführt werden, um keine unbegründete Empfehlung zu geben.

## 13. TTFA Pocket vs. Kokoro

**Nicht aus dieser Sandbox messbar** — der reale Kokoro-Modell-Download
(~326MB von `huggingface.co`) ist aus diesem Entwicklungs-Environment
blockiert (403/EGRESS_BLOCKED). Es liegt daher keine reale, verifizierte
TTFA-Zahl für Kokoro vor. Der erweiterte Benchmark (`app/tts_benchmark.py
--engine kokoro_onnx`) misst `ttfa_s` und `first_chunk_duration_s` exakt
nach demselben Schema wie für Pocket TTS und muss vom Nutzer auf der
Ziel-Hardware ausgeführt werden. Es werden hier bewusst KEINE erfundenen
Zahlen oder Vergleichswerte angegeben.

## 14. RTF Pocket vs. Kokoro

Ebenfalls nicht aus dieser Sandbox messbar, aus demselben Grund wie Punkt 13.
`app/tts_benchmark.py` berechnet `model_rtf` für beide Engines nach
identischer Formel (`model_compute_s / audio_duration_s`), sodass ein
direkter Vergleich auf der Ziel-Hardware möglich ist, sobald der Nutzer das
Tool dort ausführt.

## 15. RAM-Nutzung Pocket vs. Kokoro

Ebenfalls nicht aus dieser Sandbox messbar. Der Benchmark misst
`peak_rss_kb` über `resource.getrusage(RUSAGE_SELF).ru_maxrss` (Linux,
keine zusätzliche Abhängigkeit) pro Konfiguration in einem frischen
Subprozess. Qualitativ ist zu erwarten, dass Kokoro (reines ONNX, kein
PyTorch-Modell geladen) einen kleineren zusätzlichen RAM-Fussabdruck hat als
Pocket TTS (PyTorch), aber auch dazu wird hier bewusst keine konkrete Zahl
ohne echte Messung behauptet.

## 16. pytest-Ergebnis

```
536 passed, 7 deselected in 6.49s
```
(Die 7 deselektierten Tests sind die mit `@pytest.mark.e2e` markierten
echten Netzwerktests — Pocket TTS und Kokoro ONNX E2E, die per
`RUN_..._E2E=1` gezielt aktiviert werden.)

## 17. ruff-Ergebnis

```
ruff check .      -> All checks passed!
ruff format --check . -> 53 files already formatted
```

## 18. mypy-Ergebnis

```
mypy --config-file pyproject.toml --python-executable /usr/local/bin/python3 .
-> Success: no issues found in 52 source files
```
(Eine informative `note: unused section(s): module = ['onnxruntime.*']`
erscheint, da aktuell keine Datei ein `onnxruntime`-Submodul direkt
importiert — das ist keine Fehlermeldung und beeinflusst den Exit-Code
nicht.)

## 19. Docker-Build-Ergebnis

Aus dieser Sandbox NICHT lokal verifizierbar — der Docker-Daemon ist hier
nicht verfügbar (`docker ps` schlägt mit "failed to connect to the docker
API ... no such file or directory" fehl; die Docker-CLI selbst ist zwar
installiert). Es wurden keine neuen System-apt-Pakete im Dockerfile
hinzugefügt (nur `ENV KOKORO_ONNX_CACHE=...`), sodass kein architektur-
spezifisches Build-Risiko durch neue apt-Abhängigkeiten entsteht — alle
neuen Python-Abhängigkeiten (`onnxruntime`, `kokoro-onnx`, `espeakng-loader`,
`phonemizer`, `huggingface_hub`) haben verifizierte Wheels für sowohl
`manylinux_2_28_x86_64` als auch `manylinux_2_28_aarch64`. Die endgültige
Verifikation erfolgt über GitHub Actions CI (`build-amd64`-Job und den
aarch64-Abhängigkeitspfad), wie in jeder vorherigen Release-Runde dieser
Session.

## 20. Verbleibende Risiken

1. **Keine gepinnte Modell-Revision**: `DEFAULT_KOKORO_REVISION = "main"`
   statt eines festen Commit-SHA, da aus dieser Sandbox keine Revision
   verifiziert werden konnte. Ein zukünftiger Push auf `main` im
   Modell-Repo könnte theoretisch die Datei ändern. Empfehlung: sobald
   Netzwerkzugriff auf huggingface.co besteht, eine konkrete Commit-Revision
   ermitteln und in `DEFAULT_KOKORO_REVISION` eintragen.
2. **Keine reale Aussprache-/Phonemisierungs-Verifikation** der genannten
   deutschen Testwörter (Küche, Büro, Außenlicht, Gäste-WC, Rollladen,
   Waschmaschine, Jürgen, "21,5 Grad Celsius") mit dem tatsächlich
   geladenen Modell — nur der Normalisierungs-Code selbst wurde isoliert
   getestet (38 Unit-Tests).
3. **Keine realen TTFA-/RTF-/RAM-Messwerte** für Kokoro auf der Ziel-Hardware
   (Intel Core i5-12450H) oder im direkten Vergleich zu Pocket TTS — muss
   vom Nutzer mit `app/tts_benchmark.py --engine kokoro_onnx` durchgeführt
   werden, bevor eine Thread- oder Performance-Empfehlung als belegt gilt.
4. **`Kokoro.create_stream()`s Nebenläufigkeitsverhalten unter Last** ist
   upstream nicht explizit dokumentiert; die Serialisierung per
   `asyncio.Lock` ist eine konservative, aber ungetestete Annahme unter
   echter paralleler Last (mehrere gleichzeitige Assist-Anfragen).
5. **HA-Supervisor-Schema-Upgrade-Verhalten** für die neuen Optionen wurde
   nicht gegen den echten Supervisor-Quellcode verifiziert, sondern stützt
   sich auf das bereits bei v0.2.0 bewährte Whitelist-Muster dieses
   Projekts.
6. **Der reale End-to-End-Kokoro-Test** (`test_e2e_kokoro_tts.py`, hinter
   `RUN_KOKORO_E2E=1`) wurde in dieser Sandbox nicht ausgeführt (Download
   blockiert) — er muss beim ersten CI-Lauf mit echtem Netzwerkzugriff
   verifiziert werden; sollte er wegen eines falschen Dateinamens oder
   Repo-Details fehlschlagen, das erst mit echtem Netzwerkzugriff sichtbar
   wird, ist das umgehend zu beheben, bevor ein Release als abgeschlossen
   gilt.
