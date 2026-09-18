# Abschlussbericht v0.2.7 — Echtes Wyoming-TTS-Audio-Streaming zu Home Assistant

## 1. Warum Home Assistant trotz internem Audio-Streaming bisher gewartet hat

Root Cause verifiziert direkt gegen den lokal vorhandenen Home-Assistant-Core-
Checkout (`homeassistant/components/wyoming/tts.py`,
`homeassistant/components/tts/__init__.py`,
`homeassistant/components/tts/entity.py`), nicht angenommen:

Home Assistants `WyomingTtsProvider` entscheidet anhand von
`tts_service.supports_synthesize_streaming` (aus der `describe`-Antwort des
Wyoming-Servers), welchen von zwei komplett unterschiedlichen Code-Pfaden es
nutzt:

- **`supports_synthesize_streaming=False`** (unser bisheriger Zustand) →
  `async_get_tts_audio()`: sendet `Synthesize(text)`, liest dann in einer
  Schleife Events, bis `AudioStop` empfangen wird, sammelt dabei jeden
  `AudioChunk` in ein `wave.Wave_write`-Objekt über einem `io.BytesIO`, und
  gibt **erst danach** die komplette WAV-Datei zurück. Home Assistant selbst
  wartet also auf die komplette Antwort, bevor überhaupt etwas an die
  Medienwiedergabe geht — völlig unabhängig davon, dass unser Add-on intern
  bereits `generate_audio_stream()` nutzt und Audio-Chunks sofort sendet.
  Das interne Streaming war real, aber für Home Assistant unter dem alten
  Protokoll unsichtbar.
- **`supports_synthesize_streaming=True`** → `async_stream_tts_audio()`:
  öffnet eine eigene TCP-Verbindung, schreibt `synthesize-start`, dann
  `synthesize-chunk`(s), dann (aus Kompatibilitätsgründen) noch einmal die
  komplette Nachricht als `synthesize`, dann `synthesize-stop` — und liest
  parallel bereits eintreffende `audio-start`/`audio-chunk`-Events, die es
  sofort als WAV-Stream an Home Assistants eigene Medienwiedergabe
  weiterreicht (`_read_tts_audio()` in `tts.py`).

## 2. Jetzt unterstützte Wyoming-Events

Verifiziert gegen die tatsächlich installierte Paketversion `wyoming==1.10.2`
(`wyoming/tts.py`, echte Dataclasses/Signaturen, keine erfundenen Events):

- `synthesize-start` (`SynthesizeStart`) — neu implementiert
  (`_handle_synthesize_start`)
- `synthesize-chunk` (`SynthesizeChunk`) — neu implementiert
  (`_handle_synthesize_chunk`)
- `synthesize` (`Synthesize`) — bestehend, jetzt mit doppelter Bedeutung
  (legacy Single-Shot ODER das Kompatibilitäts-Event einer laufenden
  Streaming-Anfrage), siehe Punkt 7
- `synthesize-stop` (`SynthesizeStop`) — neu implementiert
  (`_handle_synthesize_stop`)
- `synthesize-stopped` (`SynthesizeStopped`) — neu: wird nach jeder
  Streaming-Antwort gesendet (zusätzlich zu `audio-stop`), da Home
  Assistants `_read_tts_audio()`-Lese-Loop **ausschließlich** darauf endet,
  nicht auf `audio-stop` (verifiziert aus derselben Upstream-Quelle — ein
  reines `audio-stop` würde die Schleife nicht beenden).

Die neue zentrale State-Machine (`app/tts_stream.py::TtsStreamState`) verwaltet
genau die geforderten Zustände: `idle` → `collecting` → `synthesizing` →
`finished`/`cancelled`.

## 3. Kann `supports_synthesize_streaming` jetzt sicher `True` sein?

Ja. Alle vier Streaming-Events sind korrekt implementiert, das
Kompatibilitäts-Duplikat wird sicher abgefangen (Punkt 7), jede
Streaming-Antwort endet garantiert mit `synthesize-stopped`, und der Flag ist
jetzt in `_handle_describe()` auf `True` gesetzt. Die vollständige lokale
Test-Suite (siehe Punkt 8) deckt Normalfall, Mehrfach-Chunks, Fehler,
Disconnect, leeren Text und zwei aufeinanderfolgende Anfragen ab.

## 4. TTFA vorher/nachher

**Architektonisch, nicht als konkrete Millisekundenzahl behauptet** (dieses
Projekt fabriziert grundsätzlich keine auf dieser Sandbox nicht gemessenen
Hardware-Zahlen — siehe frühere Abschlussberichte):

- **Vorher**: Home Assistant musste auf `audio-stop`, also auf die
  **komplette** Synthese der gesamten Antwort, warten, bevor überhaupt ein
  Ton abgespielt werden konnte. Die wahrgenommene Latenz entsprach damit
  praktisch der vollen Synthesedauer der Antwort (`wall` im bisherigen
  Performance-Log), nicht der Zeit bis zum ersten Chunk.
- **Nachher**: Home Assistant beginnt die Wiedergabe, sobald der erste
  `audio-chunk` eintrifft — das ist jetzt exakt das neue
  `first_audio_to_ha`-Feld im Performance-Log (Zeit von "vollständiger Text
  bekannt" bis "erster Chunk beim Client"), strukturell unabhängig von der
  Gesamtlänge der Antwort.
- **Reale Zahlen für die eigene i5-12450H-Hardware**: mit
  `python -m app.tts_benchmark` sowie den neuen Log-Feldern
  (`ttfa_generated`, `ttfa_sent`, `first_audio_to_ha`, `protocol_mode`) im
  laufenden Add-on selbst messen — diese Werte hängen stark vom Modell
  (`german`/`german_24l`) und `tts_threads` ab und würden hier nur geraten,
  nicht gemessen.

## 5. RTF vorher/nachher

**Unverändert.** Diese Änderung betrifft ausschließlich das
Wyoming-Übertragungsprotokoll (wann/wie Audio an Home Assistant geht), nicht
Pocket TTS selbst — `model_rtf` (reine Modell-Rechenzeit ÷ Audiodauer) wird
exakt wie zuvor berechnet und geloggt. Es wurden bewusst **keine**
Modellparameter verändert (wie in der Aufgabenstellung gefordert). Ein
tatsächlicher RTF-Effekt aus dem in diesem Report ebenfalls ausgelieferten
Thread-Benchmark-Tool (`python -m app.tts_benchmark`) ist eine separate,
optionale Optimierung, die der Nutzer selbst auf der Zielhardware messen und
danach `tts_threads` entsprechend setzen kann.

## 6. Anzahl nativer Pocket-TTS-Generierungen pro Antwort

**Exakt eine** — sowohl im alten Legacy-Pfad als auch im neuen
Streaming-Pfad. Das ist die zentrale Garantie von `TtsStreamState`:
`begin_synthesis()` liefert nur beim ersten gültigen Aufruf einen Text
zurück, jeder weitere (z. B. durch das Kompatibilitäts-`synthesize`-Event
nach bereits verarbeiteten Chunks, oder ein nachfolgendes `synthesize-stop`)
liefert `None` und wird ignoriert. Verifiziert per Test
(`test_synthesizes_exactly_once_not_twice`,
`test_multiple_chunks_plus_compat_synthesize_still_synthesizes_once`): der
Fake-Synthesizer zeichnet jeden tatsächlichen Aufruf auf, und es ist in
keinem Szenario mehr als einer pro logischer Antwort.

## 7. Kann Text doppelt erzeugt werden — und wie wurde das verhindert?

Ohne Schutzmaßnahme: ja, potenziell. Der reale Home-Assistant-Client sendet
für eine normale HomeIntent-Antwort exakt diese Sequenz (verifiziert aus
`_write_tts_message()` in `homeassistant/components/wyoming/tts.py`):

```
synthesize-start
synthesize-chunk(text=vollständige Nachricht)   # HA verpackt eine fertige
                                                  # Antwort in EINEN Chunk
synthesize(text=vollständige Nachricht)          # Kompatibilitäts-Duplikat
synthesize-stop
```

Ohne State-Machine hätte ein naiver Handler den Text zweimal an Pocket TTS
geschickt — einmal ausgelöst durch den (fiktiven) Chunk-Trigger, einmal durch
das bestehende `synthesize`-Handling. **Verhindert durch `TtsStreamState`**:

- `add_chunk()` sammelt Text nur, löst selbst **nie** eine Synthese aus.
- `begin_synthesis()` ist die einzige Stelle, die den Zustand von
  `COLLECTING` zu `SYNTHESIZING` überführt — und das nur **einmal** pro
  Anfrage. Der erste Aufruf (durch das Kompatibilitäts-`synthesize`-Event,
  das den vollständigen Text explizit mitliefert) gewinnt; ein späterer
  Aufruf (z. B. durch `synthesize-stop`, das denselben Text aus dem Puffer
  verwenden würde) liefert `None` zurück und wird vom Handler ignoriert.
- Für einen rein spezifikationstreuen Client, der das Kompatibilitäts-Event
  NIE sendet, übernimmt `synthesize-stop` stattdessen genau einmal diese
  Rolle (Fallback auf den aus den Chunks zusammengesetzten Text).

## 8. Testergebnisse

```
pytest tests/ -q  → 436 passed, 4 deselected, 1 warning in 6.33s
ruff check .      → All checks passed!
ruff format --check .  → 43 files already formatted
mypy --config-file pyproject.toml --python-executable /usr/local/bin/python3 .
                  → Success: no issues found in 43 source files
```

Neue Tests (27 zusätzlich zu v0.2.6, alle bestanden):
- `app/tests/test_tts_stream.py` (17 Tests): reine State-Machine-Logik —
  idle/collecting/synthesizing/finished/cancelled, Mehrfach-Chunks,
  Kompatibilitäts-Text gewinnt über Puffer, zweiter `begin_synthesis()`-Call
  liefert `None`, Reset, zwei aufeinanderfolgende Anfragen.
- `app/tests/test_handler_tts_streaming.py` (14 Tests, echte
  Wyoming-Byte-Ebene über einen echten `AsyncEventHandler`): `describe`
  meldet `supports_synthesize_streaming=True`; normaler Ein-Chunk-Ablauf
  inkl. Kompatibilitäts-`synthesize` und `synthesize-stop`; kein doppeltes
  Audio (Synthesizer wird nur einmal aufgerufen); mehrere Text-Chunks
  (mit und ohne Kompatibilitäts-Event); `synthesize-stop`/`synthesize-chunk`
  ohne vorheriges `synthesize-start` werden ignoriert statt zu crashen;
  leerer Text im Streaming-Modus sendet trotzdem `synthesize-stopped`;
  TTS deaktiviert → Fehler auf `synthesize-start`; Fehler während der
  Synthese sendet `error` **und** `synthesize-stopped` (kein hängender
  Client); Disconnect zwischen `synthesize-start` und `synthesize-stop`
  löst keine Synthese aus und blockiert keine Folge-Anfrage; zwei
  aufeinanderfolgende Streaming-Anfragen auf derselben Verbindung; ein
  Legacy-`synthesize` nach einer abgeschlossenen Streaming-Anfrage
  funktioniert weiterhin korrekt; die komplette Sequenz läuft nachweislich
  ohne Deadlock innerhalb eines 5-Sekunden-Timeouts durch.
- `app/tests/test_tts_benchmark.py` (7 Tests): Argument-Parsing und
  RTF-Berechnung des neuen Benchmark-CLIs (hermetisch, kein echtes Modell).

Bestehende STT-, Keyterm- und Legacy-TTS-Tests (inkl. des echten
Model-E2E-Tests `test_e2e_tts.py`) sind **unverändert** und weiterhin grün —
der alte Single-Shot-`synthesize`-Pfad (kein vorheriges `synthesize-start`)
verhält sich exakt wie zuvor, inklusive exakt eines `audio-start` pro
Anfrage.

## Geänderte/neue Dateien

- `homeintent-moonshine-stt/app/tts_stream.py` (neu): `TtsStreamPhase`,
  `TtsStreamState`.
- `homeintent-moonshine-stt/app/handler.py`: `synthesize-start/-chunk/-stop`-
  Handler, gemeinsamer `_run_tts_synthesis()`/`_handle_synthesis_error()`,
  `_dispatch_tts_event()`, `supports_synthesize_streaming=True`, erweitertes
  Performance-Logging (`protocol_mode`, `first_audio_to_ha`).
- `homeintent-moonshine-stt/app/tts_benchmark.py` (neu): Thread-Tuning-CLI.
- `homeintent-moonshine-stt/app/tests/test_tts_stream.py`,
  `test_handler_tts_streaming.py`, `test_tts_benchmark.py` (neu).
- Version 0.2.6 → 0.2.7 in `config.yaml`, `app/pyproject.toml`,
  `app/__main__.py`; beide `CHANGELOG.md`; `DOCS.md`.

**Nicht verändert** (wie gefordert): `app/keyterms.py`, `app/ha_vocabulary.py`,
STT-Pfad, Pocket-TTS-Voice `juergen`, keine Modellparameter.
