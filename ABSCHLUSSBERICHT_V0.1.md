# Abschlussbericht: HomeIntent Moonshine STT v0.1.0

**Datum**: 17. September 2026
**Autor**: Claude Sonnet 4.6
**Anlass**: Ein unabhängiger 21-Punkte-Review der vorherigen Fassung stellte fest, dass die
Implementierung durchgängig eine **erfundene, nicht existierende Moonshine/Wyoming-API**
verwendete (CI rot, 32/39 Tests grün, mypy fehlgeschlagen). Dieser Bericht ersetzt den
vorherigen, der fälschlich "✅ ABGESCHLOSSEN" und "100% erfüllt" behauptete, ohne dass der
Code tatsächlich gegen echte Quellen geprüft worden war.

---

## Executive Summary

Alle 21 Review-Punkte wurden gegen die tatsächlich installierten Paketquellen
(`moonshine-voice==0.1.5`, `wyoming==1.10.2`) nachgearbeitet — nicht geraten, sondern jede
API-Signatur direkt aus dem Quellcode der Pakete gelesen und verifiziert. Zusätzlich wurden
bei der Verifikation zwei weitere, bis dahin unentdeckte Fehler gefunden und behoben (siehe
"Zusätzliche Funde" unten).

**V0.1 INSTALLATIONSBEREIT: NEIN (mit einer offenen, klar benannten Lücke)**

Alles, was in dieser Umgebung tatsächlich geprüft werden konnte, ist grün. Der eine Punkt,
der **nicht** verifiziert werden konnte, ist der containerisierte Docker-Build — dafür stand
in dieser Entwicklungsumgebung kein Container-Runtime (Docker/Podman) zur Verfügung, und das
Repository besitzt bislang **keinen** CI-Workflow, der den Docker-Build ausführt (nur Lint,
Type-Check, Tests). Das heißt: Der Docker-Build wurde bisher weder lokal noch in CI von
irgendjemandem tatsächlich verifiziert. Bevor "installationsbereit" behauptet werden kann,
muss das nachgeholt werden (siehe "Offene Punkte").

---

## Was echt verifiziert wurde

### 1. Code gegen echte API-Quellen korrigiert

Alle API-Aufrufe in `models.py`, `streaming.py`, `handler.py`, `__main__.py` wurden direkt
gegen den installierten Paketquellcode gelesen und korrigiert:

- `Transcriber(model_path, model_arch)` — echter Konstruktor, nicht der erfundene
  `Transcriber().language().model_arch().load()`-Builder aus v0.1
- `get_model_for_language(wanted_language="de", wanted_model_arch=...)` — echte
  Modellauflösung; bestätigt, dass Deutsch nur `TINY_STREAMING`/`SMALL_STREAMING` als
  Architekturen hat
- `MOONSHINE_VOICE_CACHE`-Umgebungsvariable für Modell-Caching (nicht `HF_HOME`)
- `Transcriber.create_stream()` pro Wyoming-Verbindung — isolierte Session pro Verbindung,
  Modell wird genau einmal beim Start geladen
- `TranscriptEventListener`-basierte Events (`on_line_completed`, `on_error`, …) statt
  erfundener Callback-Signaturen
- Mehrere `LineCompleted`-Events pro Utterance werden korrekt akkumuliert (nicht
  überschrieben)
- `asyncio.to_thread()` für alle blockierenden ctypes-Aufrufe — behebt den früheren
  `asyncio.get_event_loop()`-Bug grundsätzlich, nicht nur symptomatisch
- `wyoming.server.AsyncTcpServer(host, port).run(handler_factory)` direkt verwendet;
  `app/server.py` (eigene, fehlerhafte Server-Implementierung) entfernt
- Strikte Ablehnung nicht-konformer Audioformate (Option A): Bei falscher Sample-Rate/Breite/
  Kanalzahl wird ein `Error`-Event gesendet und keine Session erstellt — vorher wurde
  stillschweigend weitergemacht

### 2. s6-Overlay repariert

Der Service wurde vorher **nie tatsächlich gestartet**, weil die Registrierung in
`user/contents.d/` fehlte. Gegen die offizielle `home-assistant/addons`-Whisper-Referenz
(via `gh api` abgerufen) korrigiert:

- `user/contents.d/{homeintent-moonshine,discovery}` Markierungsdateien ergänzt
- `discovery/run` von einem festen `sleep` auf eine Polling-Schleife umgestellt (wartet
  tatsächlich, bis der Wyoming-Port bereit ist, statt blind zu raten)
- `dependencies.d`-Verkettung (`discovery` wartet auf `homeintent-moonshine`, dieser auf
  `base`), `down-signal` (`SIGINT`), `finish`-Skript ergänzt

### 3. Testsuite komplett neu geschrieben

40/40 Tests grün gegen die korrigierten APIs, inklusive eines bewusst umgedrehten Tests
(`test_invalid_audio_format_rejected`), der jetzt die strikte Ablehnung statt der früheren
stillschweigenden Annahme prüft. Mock-Fixtures für `reader`/`writer` wurden so gebaut, dass
sync- vs. async-Methoden korrekt getrennt sind (behebt die vom Review bemängelten "coroutine
was never awaited"-Warnungen).

### 4. Lokale Quality Gates — alle grün, mit CI-identischem Kommando geprüft

```
ruff check homeintent-moonshine-stt/app/          → All checks passed
ruff format --check homeintent-moonshine-stt/app/ → 13 files already formatted
mypy --config-file homeintent-moonshine-stt/app/pyproject.toml \
     homeintent-moonshine-stt/app/                → Success: no issues found (strict mode)
pytest homeintent-moonshine-stt/app/tests/ -v     → 40 passed
```

Strict-Mode wurde aktiv verifiziert (nicht nur "keine Fehler gemeldet"): Ein absichtlich
untypisierter Testcode wurde eingefügt und von mypy korrekt beanstandet, dann wieder
entfernt.

### 5. Echter Smoke-Test (kein Mock)

Da kein Docker verfügbar war, wurde stattdessen der reale Stack direkt getestet:

1. Echtes deutsches `tiny-streaming`-Modell von `download.moonshine.ai` heruntergeladen
   (über `load_transcriber()`, den echten Produktionscode-Pfad)
2. Reale `Transcriber`/`Stream`-Objekte durch `start()` → `add_audio()` ×5 → `stop()` →
   `close()` getrieben — kein Fehler
3. Den echten Wyoming-TCP-Server (`python -m app --model tiny --language de --port 12300`)
   gestartet und über einen echten TCP-Socket mit dem echten Wyoming-Wire-Format
   (`data_length`/`payload_length`-Framing, nicht das vereinfachte Lehrbuch-JSON) angesprochen:
   `describe` → `transcribe` → `audio-start` → 10× `audio-chunk` → `audio-stop` →
   `transcript`-Antwort erhalten. Alles lief fehlerfrei durch.

### 6. CI grün (verifiziert nach Push)

```
Lint         success
Type Check   success
Tests        success
```
(Commit `22fbc67`, https://github.com/pquandel2-alt/homeintent-moonshine-addon/actions)

---

## Zusätzliche Funde (nicht in den ursprünglichen 21 Punkten, aber bei der Verifikation entdeckt)

1. **Lizenz-Fehlinformation korrigiert**: README/DOCS/Code behaupteten durchgängig, die
   Moonshine-Modelle seien MIT-lizenziert. Der echte Modell-Download gibt jedoch beim Laden
   jedes nicht-englischen Modells folgende Meldung aus: *"Using a model released under the
   non-commercial Moonshine Community License."* Nur Moonshines englische Modelle sind MIT;
   die hier verwendeten deutschen Modelle sind es **nicht** — sie sind kostenlos für
   Forschende, Entwickler, Kleinunternehmen und Creator mit weniger als 1 Mio. USD
   Jahresumsatz, kommerzielle Nutzung darüber hinaus erfordert eine Moonshine-Enterprise-
   Lizenz. README.md, DOCS.md und `models.py` wurden entsprechend korrigiert; der Add-on-Code
   selbst bleibt MIT.
2. **CI Type-Check lief nie im Strict-Mode**: Der Workflow rief `mypy ... --ignore-missing-
   imports` ohne `--config-file` auf, wodurch die im Projekt definierte `strict = true`-
   Konfiguration nie geladen wurde. Verifiziert durch absichtlich eingefügten untypisierten
   Code, der durchrutschte. Behoben durch explizites `--config-file`.
3. **`pyproject.toml` war für `ruff check` komplett kaputt**: Ein ungültiger Isort-Schlüssel
   (`profile = "black"`, gehört zu reinem `isort`, nicht zu Ruffs Isort-Implementierung) ließ
   `ruff check` mit einem TOML-Parse-Fehler abbrechen, bevor überhaupt eine Datei geprüft
   wurde — d.h. Lint lief nie wirklich.

---

## Offene Punkte (bewusst nicht als erledigt behauptet)

1. **Docker-Build nicht verifiziert.** Kein Container-Runtime in dieser Umgebung verfügbar;
   das Repository hat aktuell auch keinen CI-Workflow, der den Image-Build ausführt. Dies
   muss nachgeholt werden (z. B. lokal mit Docker/Podman, oder ein neuer GitHub-Actions-Job
   analog zu `home-assistant/addons`' `builder`-Action), bevor v0.1 als installationsbereit
   gelten kann.
2. **aarch64 ungeprüft am realen Gerät.** Es wurde verifiziert, dass für `moonshine-voice`,
   `numpy` und `wyoming` echte aarch64-Wheels für Python 3.11 existieren und dass Debian
   Trixies glibc die `manylinux_2_34`-Anforderung erfüllt — das rechtfertigt den `arch:`-
   Eintrag in `config.yaml`. Ein tatsächlicher Lauf auf echter aarch64-Hardware (z. B.
   Raspberry Pi) steht aber aus.
3. **Kein Test unter realer Sprache.** Der Smoke-Test hat den Server mit zufälligem Rauschen
   gefüttert (kein echtes Deutsch), daher kam erwartungsgemäß ein leeres Transkript zurück.
   Das beweist, dass die Pipeline mechanisch korrekt läuft — nicht, dass die
   Erkennungsqualität stimmt.

---

## Fazit

Die im Review kritisierte fundamentale Ursache — eine komplett erfundene API — ist behoben
und jede Korrektur ist gegen den echten Paketquellcode nachvollziehbar. Lint, Typecheck und
Tests sind sowohl lokal als auch in CI grün. Der reale End-to-End-Smoke-Test (echtes Modell,
echter Server, echtes Wire-Protokoll) lief fehlerfrei durch. Was fehlt, ist ausschließlich
die Container-Build-Verifikation — dafür braucht es entweder eine Umgebung mit Docker/Podman
oder einen entsprechenden CI-Job. Bis das nachgeholt ist, bleibt die ehrliche Antwort auf
"installationsbereit?" ein **Nein**, auch wenn der Code-Stand deutlich weiter ist als alles,
was vorher in diesem Repository existierte.
