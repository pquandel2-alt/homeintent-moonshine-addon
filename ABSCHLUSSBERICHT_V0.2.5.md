# Abschlussbericht v0.2.5 – Kritischer Keyterm-Crash-Fix

Datum: 2026-09-18

## 1. Genaue Root Cause

`Transcriber.set_keyterms()` (moonshine-voice==0.1.5, verifiziert am tatsächlich
installierten Quellcode, nicht angenommen) verkettet die gesamte übergebene
Liste zu EINEM komma-separierten String und übergibt diesen in einem
einzigen nativen Aufruf an die C-API:

```python
result = self._lib.moonshine_transcriber_set_keyterms(
    self._handle, ",".join(terms).encode("utf-8")
)
if result != 0:
    raise MoonshineError(f"Failed to set key terms: {error_str}")
```

Lehnt der native Tokenizer auch nur einen einzelnen Begriff im String ab,
schlägt der **gesamte** Aufruf fehl – mit einer generischen, nicht auf den
konkreten Begriff bezogenen Fehlermeldung. Für den Entity-Namen `/Büro`
ergab das: `MoonshineError: Failed to set key terms: Unknown error`. Da
`app/__main__.py`s `_load_and_bias_transcriber()` diesen Aufruf
ungeschützt (`transcriber.set_keyterms(effective_keyterms)`) direkt beim
Start ausführte, propagierte die Exception ungefangen bis `main()`, der
Prozess beendete sich mit Exit-Code 1, und Home Assistant startete das
Add-on in einer Endlosschleife neu.

**Verifiziert, nicht angenommen:** Das native C-API von
`moonshine-voice==0.1.5` (per `strings` auf die kompilierte `libmoonshine.so`
sowie per Introspektion der ctypes-Bindings direkt geprüft) bietet **keine**
separate Funktion, um einen Begriff gegen den geladenen Tokenizer zu
validieren, ohne ihn aktiv zu setzen – nur `moonshine_transcriber_set_keyterms`
und `moonshine_transcriber_set_context` existieren.

## 2. Geänderte Dateien

- `homeintent-moonshine-stt/app/keyterms.py` — komplett neu aufgebaut um die
  zentrale Safe-Keyterm-Schicht (`apply_safe_keyterms()`, `normalize_keyterm()`,
  `_basic_validate()`, `_bisect_valid_keyterms()`, `RejectedKeyterm`,
  `SafeKeytermResult`). `parse_extra_keyterms()`/`merge_keyterms()` unverändert.
- `homeintent-moonshine-stt/app/__main__.py` — `_load_and_bias_transcriber()`
  und `_refresh_ha_vocabulary_periodically()` nutzen jetzt ausschließlich
  `apply_safe_keyterms()` statt direktem `transcriber.set_keyterms()`;
  `_LoadedEngines`/`_load_engines()`/`main()` reichen die tatsächlich
  akzeptierte Keyterm-Zahl für das Startup-Logging durch.
- `homeintent-moonshine-stt/app/tests/test_keyterms.py` — 20 neue Tests für
  `apply_safe_keyterms()`, `_basic_validate()`, `normalize_keyterm()`.
- `homeintent-moonshine-stt/app/tests/test_main_config.py` — 3 neue Tests
  (`TestKeytermCrashRegression`), die exakt das produktive Absturzszenario
  nachstellen (inkl. eines vollständigen `main()`-End-to-End-Laufs).
- `homeintent-moonshine-stt/app/tests/test_ha_vocabulary_refresh.py` —
  bestehende Last-known-good-Tests an die neue (zweifache) `set_keyterms()`-
  Aufrufsemantik angepasst; 2 neue Tests für Fall H (Refresh mit neu
  auftretendem inkompatiblem Begriff).
- `homeintent-moonshine-stt/config.yaml`, `app/pyproject.toml`, `app/__main__.py`
  — Version 0.2.4 → 0.2.5.
- `CHANGELOG.md`, `homeintent-moonshine-stt/CHANGELOG.md`,
  `homeintent-moonshine-stt/DOCS.md` — dokumentiert.
- `ha_vocabulary.py` — **unverändert** (wie gefordert; die Logik zur
  Vokabular-Erzeugung selbst hat keinen Fehler, das Problem lag ausschließlich
  in der ungeschützten Anwendung des Ergebnisses).

## 3. Neue Safe-Keyterm-Architektur

```
candidate_terms (HA-Vokabular + manuelle extra_keyterms, bereits gemerged)
        │
        ▼
normalize_keyterm()  — NFC-Normalisierung, Trim (keine Transliteration)
        │
        ▼
_basic_validate()    — leer/Whitespace, Steuerzeichen, Komma-Delimiter,
        │               nicht-kodierbares Unicode (ohne jeden nativen Aufruf)
        ▼
_bisect_valid_keyterms() — Divide-and-Conquer gegen das ECHTE geladene
        │                  Modell via set_keyterms() selbst (da keine
        │                  separate Validierungs-API existiert)
        ▼
   akzeptierte Liste
        │
        ▼
EIN finaler, expliziter, autoritativer set_keyterms(akzeptiert)-Aufruf
        │
        ├── Erfolg → SafeKeytermResult(accepted, rejected, apply_succeeded=True)
        └── Unerwarteter Fehler → Fallback/Wiederherstellung der vorherigen
                                    bekannt-guten Liste,
                                    apply_succeeded=False
```

Alle Aufrufer (Startup, manuelle `extra_keyterms`, periodischer Refresh)
verwenden ausschließlich diesen einen Pfad — keine verstreuten
Sonderlösungen.

Die Bisektion probiert zunächst die gesamte Liste; schlägt sie fehl, wird
sie geteilt und rekursiv erneut versucht, bis jeder einzelne unverträgliche
Begriff isoliert ist. Bei 249 Begriffen mit genau einem defekten Begriff
sind das typischerweise nur wenige native Aufrufe (≈ log₂(249) ≈ 8), nicht
249 Einzelaufrufe. Nach der Diagnose erfolgt **immer** ein abschließender,
expliziter `set_keyterms()`-Aufruf mit exakt der akzeptierten Liste — ein
Test-Zwischenzustand aus der Bisektion wird niemals versehentlich zum
aktiven Endzustand.

## 4. Wie `/Büro` jetzt behandelt wird

`/Büro` besteht die Basisvalidierung (kein verbotenes Zeichen — `/` ist NICHT
pauschal verboten, wie gefordert). Bei der Validierung gegen das echte
Modell schlägt `set_keyterms()` für einen Batch fehl, der `/Büro` enthält;
die Bisektion isoliert `/Büro` als einzelnen fehlerhaften Begriff
(`reason=tokenizer_rejected`). Er wird aus der finalen Liste entfernt, alle
anderen 248 Begriffe werden angewendet, eine WARNING wird geloggt, der
Dienst startet normal.

## 5. Verhalten beim Startup

- 249 gewünschte Begriffe, 248 gültig, 1 ungültig (`/Büro`):
  ```
  WARNING [app.keyterms] Skipping Moonshine-incompatible keyterm: '/Büro' (reason=tokenizer_rejected)
  WARNING [app.keyterms] 1 of 249 candidate keyterm(s) were incompatible with the loaded Moonshine model and were skipped
  INFO [app.keyterms] Applied 248 effective Moonshine keyterm(s)
  INFO [app.__main__] HomeIntent Moonshine Voice v0.2.5
  ...
  ```
  Der Wyoming-Server startet normal, kein Crash, keine Restart-Schleife.
- Sind ALLE Begriffe ungültig, startet der Dienst trotzdem — Moonshine läuft
  dann ohne Keyterm-Biasing weiter (STT ist wichtiger als Biasing).
- Die komplette HA-Vokabularliste wird nie bei INFO geloggt; nur bei DEBUG
  ist die volle Rejected-Liste sichtbar.

## 6. Verhalten beim periodischen Refresh

- Ein neu auftretender inkompatibler Begriff (z. B. eine neu angelegte
  Entity `/Büro Licht`) lässt den Refresh-Task **nicht** sterben: der
  Begriff wird übersprungen, die restlichen Begriffe werden angewendet,
  der Task läuft beim nächsten Intervall normal weiter.
- **Atomare Last-known-good-Behandlung**: `last_known_good_terms` wird erst
  aktualisiert, NACHDEM `apply_safe_keyterms()` bestätigt, dass der finale,
  autoritative Apply-Aufruf selbst erfolgreich war (`apply_succeeded=True`)
  — nicht schon, weil der HA-Fetch erfolgreich war. Schlägt der finale
  Apply-Aufruf unerwartet strukturell fehl, werden die vorherigen bekannt
  guten Keyterms wiederhergestellt, und `last_known_good_terms` bleibt
  komplett unverändert, sodass der nächste Refresh-Versuch von derselben
  bekannten guten Basis ausgeht.

## 7. Verhalten bei manuellen Keyterms

`extra_keyterms` durchläuft exakt denselben `apply_safe_keyterms()`-Pfad wie
automatisch aus Home Assistant geladene Begriffe — keine separate
Sonderlösung. Ein inkompatibler manueller Begriff wird übersprungen, die
übrigen (z. B. `Jarvis`, `Küche`, `HomeIntent` aus
`extra_keyterms: Jarvis,/Büro,Küche,HomeIntent`) werden angewendet.

## 8. Neue Tests

- `test_keyterms.py`: Fall A (exakter Produktionsfall `/Büro`), Fall B
  (deutsche Sonderzeichen bleiben unverändert), Fall C (Mischung
  gültig/ungültig), Fall D (alle ungültig → Start ohne Biasing), Fall I
  (unerwarteter nativer Fehler → Fallback-Wiederherstellung), plus Tests für
  Basisvalidierung, NFC-Normalisierung, Deduplizierung, Bisektion mit
  mehreren verteilten Fehlern, Whitespace/leere Einträge, „nie werfen auch
  wenn Wiederherstellung fehlschlägt".
- `test_main_config.py`: `TestKeytermCrashRegression` mit 3 Tests, darunter
  ein vollständiger `main()`-End-to-End-Lauf (Definition-of-Done-Fall).
- `test_ha_vocabulary_refresh.py`: Fall H (2 Tests) — Refresh-Task und
  STT laufen nach einem neu auftretenden inkompatiblen Begriff weiter;
  bestehende Last-known-good-Tests an die neue (zweifache)
  `set_keyterms()`-Aufrufsemantik angepasst.

Fall E (ungültiger manueller Term), Fall F (ungültiger HA-Alias) und Fall G
(ungültige Area+Entity-Kombination) sind strukturell durch Fall A/C
abgedeckt: `apply_safe_keyterms()` operiert auf der bereits vollständig
zusammengeführten flachen Begriffsliste und unterscheidet nicht nach
Herkunft (Entity-Name, Alias, Area+Entity-Kombination oder manueller
Begriff) — genau das war die Vorgabe ("denselben zentralen
Validierungspfad").

## 9. Ergebnis pytest

```
376 passed, 4 deselected, 1 warning in 5.79s
```

(4 deselektierte e2e-Tests: unverändert, benötigen echten Netzwerkzugriff,
in dieser Sandbox blockiert — wie in allen vorherigen Runden.)

## 10. Ergebnis ruff

```
ruff check .      → All checks passed!
ruff format --check . → 39 files already formatted
```

## 11. Ergebnis mypy

```
mypy --config-file app/pyproject.toml --python-executable /usr/local/bin/python3 app
→ Success: no issues found in 38 source files
```

## 12. Verbleibende Risiken

- Die Bisektionsannahme („ein Begriff ist unabhängig von anderen Begriffen
  im selben Aufruf gültig/ungültig") basiert auf Moonshines eigenem
  komma-getrenntem Tokenisierungsverhalten und wurde nicht gegen das echte
  Modell in dieser Sandbox verifiziert (kein Netzwerkzugriff auf
  `download.moonshine.ai`). Sollte diese Annahme in einem Randfall falsch
  sein, terminiert die Bisektion dennoch korrekt (der jeweils kleinste
  fehlschlagende Batch wird als Übeltäter behandelt) — im schlimmsten Fall
  werden dann mehr Begriffe als nötig verworfen, nie weniger, und der
  Dienst bleibt funktionsfähig.
- Der native Fehlertext von Moonshine ("Unknown error") lässt sich nicht
  weiter differenzieren, als moonshine-voice selbst zurückgibt — das wird
  ehrlich als `detail`-Feld durchgereicht, nicht verschleiert.
- Diese Änderung wurde nicht gegen das echte deutsche Moonshine-Modell auf
  echter Hardware getestet (nur gegen Fakes, die das dokumentierte
  Fehlerverhalten exakt nachbilden) — die reale Verifikation erfolgt über
  die produktive Installation des Nutzers.

## 13. Empfehlung zu moonshine-voice-Upgrade

**Kein Upgrade verfügbar.** `moonshine-voice==0.1.5` (aktuell gepinnt) ist
laut PyPI (per WebFetch direkt abgefragt, Stand 2026-09-18) bereits die
neueste veröffentlichte Version — es existiert keine neuere Version, auf
die aktualisiert werden könnte, geschweige denn eine, die dieses Verhalten
upstream behebt. Die in dieser Version implementierte defensive
Anwendungsschicht ist daher unabhängig von einem zukünftigen Upstream-Fix
notwendig und bleibt auch nach einem eventuellen späteren Upgrade sinnvoll
(defense in depth).
