# Abschlussbericht v0.2.6 — Speech-Normalisierung & Speech-Fallback für Moonshine-Keyterms

## 1. Root Cause von `Wasch\xadmaschine`

Die HA-Entity hieß im produktiven Log `Wasch\xadmaschine`. Die Bytes
(`0xC2 0xAD`) sind die UTF-8-Kodierung von **U+00AD SOFT HYPHEN** — einem
einzelnen unsichtbaren Zeichen zwischen "h" und "m", keiner Escape-Sequenz.

Der bisherige `apply_safe_keyterms()`-Pfad (v0.2.5) normalisierte nur mit
NFC und `strip()` — beides ändert an U+00AD nichts, da es kein
kombinierendes Zeichen und keine Randwhitespace ist. Der Begriff ging
unverändert (inklusive Soft Hyphen) an den nativen Moonshine-Tokenizer,
der ihn ablehnte, und wurde bisher komplett verworfen.

## 2. Unicode-Erklärung von U+00AD

`U+00AD SOFT HYPHEN` ist ein **Format-Zeichen** (Unicode-Kategorie `Cf`,
"Format"). Es signalisiert historisch eine *mögliche* Trennstelle für
Zeilenumbrüche in Textverarbeitungsprogrammen und ist beim normalen
Rendern unsichtbar — für Sprache/Aussprache hat es null Bedeutung. Es kann
unbemerkt beim Kopieren/Einfügen von Namen (z. B. aus Word/Excel/Browser)
in einen Home-Assistant-Entity-Namen landen.

**Behandlung**: In `normalize_keyterm()` (Stufe 1) werden jetzt alle
Zeichen der Kategorie `Cf` entfernt — nicht nur U+00AD, auch:

| Zeichen | Codepoint | Bedeutung |
|---|---|---|
| SOFT HYPHEN | U+00AD | mögliche Trennstelle, unsichtbar |
| ZERO WIDTH SPACE | U+200B | unsichtbarer "Leerraum" ohne Breite |
| ZERO WIDTH NON-JOINER | U+200C | verhindert Ligaturbildung |
| ZERO WIDTH JOINER | U+200D | erzwingt Ligaturbildung/Emoji-Kombination |
| WORD JOINER | U+2060 | verhindert Zeilenumbruch, unsichtbar |
| BOM / ZERO WIDTH NO-BREAK SPACE | U+FEFF | Byte-Order-Mark / Formatierung |

Diese Entfernung ist **verlustfrei bezüglich der Aussprache** — keines
dieser Zeichen wurde je gesprochen. Anschließend wird Whitespace
kollabiert (mehrfache Leerzeichen → eines) und getrimmt. Deutsche
Umlaute/ß/Akzente werden dabei nicht angefasst, da sie nicht in Kategorie
`Cf` fallen.

`Wasch\xadmaschine` → `Waschmaschine` passiert komplett in Python, **vor**
jedem nativen `set_keyterms()`-Aufruf — der Tokenizer sieht den kaputten
Begriff nie.

## 3. Umgang mit Emoji/Variation Selectors

`⚠️` besteht aus zwei Codepoints: U+26A0 (WARNING SIGN, Kategorie `So`,
Symbol) + U+FE0F (VARIATION SELECTOR-16, wählt die "Emoji"-Darstellung des
vorangehenden Zeichens). Variation Selectors sind selbst unsichtbar/
bedeutungslos, wenn ihr Basiszeichen entfernt wird — sie dürfen daher
nicht isoliert übrig bleiben.

`_generate_speech_fallback()` entfernt explizit:
- Unicode-Kategorien `So` (Symbol, sonstige — deckt die meisten Emoji ab)
  und `Sk` (Symbol, Modifikator)
- Variation Selectors im Bereich U+FE00–U+FE0F und dem ergänzenden Bereich
  U+E0100–U+E01EF (per Codepoint-Bereich, nicht per Kategorie, da manche
  Variation Selectors technisch als `Mn` klassifiziert sind — dieselbe
  Kategorie wie normale kombinierende Akzente, die NICHT entfernt werden
  dürfen)

`Familie ⚠️` → `Familie` (beide Codepoints vollständig entfernt, kein
verwaistes U+FE0F).

## 4. Neue Speech-Normalisierung (Architektur)

Zweistufiges Modell, zentral in `app/keyterms.py`:

```
candidate_terms
      │
      ▼
normalize_keyterm()          ← Stufe 1, verlustfrei, IMMER angewendet
  • NFC
  • Cf-Zeichen entfernen (Soft Hyphen, Zero-Width-*, BOM, ...)
  • Whitespace kollabieren + trimmen
      │
      ▼
_basic_validate()             ← wie bisher (leer, Steuerzeichen, Komma, ...)
      │
      ▼
Dedup (nach Stufe-1-Normalisierung)
      │
      ▼
_bisect_valid_keyterms()      ← echte Modellprüfung, Teile-und-herrsche
      │
      ├─ Batch akzeptiert → alle Begriffe: Status "unchanged"/"normalized"
      │
      └─ Einzelner Begriff abgelehnt (Batch-Größe 1)
              │
              ▼
      _generate_speech_fallback()   ← Stufe 2, NUR bei echter Ablehnung
        • "/", "\", "|" → Leerzeichen
        • Emoji/Symbole (So/Sk) + Variation Selectors entfernen
        • Whitespace kollabieren
              │
              ▼
      erneuter echter set_keyterms()-Aufruf mit der Fallback-Variante
              │
              ├─ akzeptiert → Status "speech_fallback"
              └─ abgelehnt / leer → REJECTED (reason=tokenizer_rejected)
      │
      ▼
EIN finaler, expliziter, autoritativer set_keyterms(accepted)-Aufruf
```

Jeder Schritt bleibt vollständig zentral in `app/keyterms.py` und wird
identisch von Startup, `extra_keyterms` und dem periodischen HA-Refresh
genutzt — `ha_vocabulary.py` liefert weiterhin nur die rohen Namen
(inklusive Area+Entity-Kombinationen wie `EG/Büro Licht`), die Bereinigung
passiert ausschließlich in der Keyterm-Schicht.

**Wichtig**: Keine der beiden Stufen behauptet jemals eigenständig, ein
Begriff sei tokenisierbar — das entscheidet ausschließlich der echte
`set_keyterms()`-Aufruf gegen das geladene Modell. Stufe 2 wird nur nach
tatsächlicher Ablehnung von Stufe 1 versucht.

## 5. Verhalten bei `/`

`/` wird **nicht** vorab entfernt. `_basic_validate()` lässt `/` weiterhin
uneingeschränkt durch (siehe bestehender Test
`test_slash_is_not_rejected_by_basic_validation`). Der Ablauf ist:

1. `Treppe/Büro` wird unverändert an das Modell getestet.
2. Akzeptiert das Modell es (z. B. ein zukünftiges Modell mit `/`-Support)
   → Original bleibt exakt erhalten, Status `unchanged`.
3. Lehnt das Modell es ab → Fallback `Treppe Büro` wird generiert und
   erneut real getestet.
4. Akzeptiert das Modell die Fallback-Variante → diese wird verwendet,
   Status `speech_fallback`.
5. Lehnt das Modell auch die Fallback-Variante ab → Begriff wird
   verworfen (`reason=tokenizer_rejected`), Ursprungsverhalten aus v0.2.5
   bleibt für diesen Fall erhalten.

Das gilt identisch für `\` und `|` sowie für zusammengesetzte
Area+Entity-Begriffe wie `EG/Büro Licht` → `EG Büro Licht` (siehe neuer
Test `test_separator_composite_term_recovers_via_speech_fallback` in
`test_ha_vocabulary_refresh.py`).

## 6. Anzahl zusätzlicher nativer `set_keyterms()`-Calls

Ein zusätzlicher nativer Aufruf **nur** für Begriffe, die bereits einzeln
(Batch-Größe 1) vom Modell abgelehnt wurden — also exakt für die
tatsächlich inkompatiblen Begriffe, nicht für die validen. Im produktiven
Beispiel (4 von 249 inkompatibel) bedeutet das 4 zusätzliche native Calls
(einen pro Fallback-Versuch), unabhängig von der Gesamtgröße der Liste.
Das bestehende Teile-und-herrsche-Verhalten für die Isolierung selbst
(mehrere Aufrufe bei mehreren fehlerhaften Begriffen) ist unverändert und
notwendig für Korrektheit.

## 7. Konnte die native Fehlermeldungsflut reduziert werden?

Teilweise, aber nicht vollständig, und bewusst ohne Abstriche bei der
Korrektheit:

- Untersucht: Die nativen Meldungen (`No match found for remaining
  bytes ...`, `moonshine_transcriber_set_keyterms(): Failed to set key
  terms ...`) kommen direkt aus der kompilierten C-Bibliothek auf stderr,
  nicht aus Python-Logging — es gibt keine dokumentierte Möglichkeit, sie
  selektiv zu unterdrücken, ohne stderr global umzuleiten. Ein globales
  Umleiten von stderr wurde **bewusst nicht** umgesetzt, da dadurch auch
  echte, unerwartete native Fehler unsichtbar würden (Sicherheitsrisiko
  laut Vorgabe #10).
- Die Cf-Entfernung in Stufe 1 verhindert, dass Begriffe wie
  `Wasch\xadmaschine` überhaupt einen fehlschlagenden nativen Aufruf
  auslösen — sie werden bereits vor der ersten Modellprüfung korrigiert.
  Das reduziert die Anzahl der nativen Fehlversuche für diese
  Fehlerklasse auf null.
- Für Begriffe, die tatsächlich abgelehnt werden (Separatoren, Emoji),
  kommt zwangsläufig ein zusätzlicher Validierungsaufruf hinzu (Punkt 6)
  — das ist notwendig, um die Fallback-Variante laut Vorgabe #11 real zu
  bestätigen, statt sie anzunehmen.
- Keine Caching-Heuristik über mehrere `apply_safe_keyterms()`-Aufrufe
  hinweg wurde eingebaut (z. B. "bekannte kaputte Muster" persistent
  merken) — das hätte das Risiko einer falschen Annahme über zukünftige
  Modell-/Tokenizer-Versionen eingeführt und wurde als nicht
  sicherheitsneutral verworfen.

## 8. Geänderte Dateien

- `homeintent-moonshine-stt/app/keyterms.py` — Kernänderung: Stufe-1-
  Normalisierung (Cf-Entfernung, Whitespace-Kollabierung),
  `_generate_speech_fallback()`, erweiterte Bisection mit Fallback-Retry,
  neue `AppliedKeyterm`-Dataclass, erweiterte `SafeKeytermResult.applied`,
  neues Logging (`_log_summary()`).
- `homeintent-moonshine-stt/app/tests/test_keyterms.py` — neue/angepasste
  Tests (siehe Abschnitt 9).
- `homeintent-moonshine-stt/app/tests/test_main_config.py` — eine
  bestehende Fall-D-Testerwartung angepasst (Fallback-Varianten müssen im
  Fake ebenfalls als inkompatibel markiert werden, um "alles abgelehnt"
  weiterhin echt zu testen).
- `homeintent-moonshine-stt/app/tests/test_ha_vocabulary_refresh.py` —
  neuer Test für zusammengesetzte Area+Entity-Separator-Fallback-Wiederherstellung.
- `homeintent-moonshine-stt/config.yaml`, `app/pyproject.toml`,
  `app/__main__.py` — Version 0.2.5 → 0.2.6.
- `homeintent-moonshine-stt/CHANGELOG.md`, `CHANGELOG.md` (Repo-Root),
  `homeintent-moonshine-stt/DOCS.md` — dokumentiert.
- `ABSCHLUSSBERICHT_V0.2.6.md` (diese Datei) — neu.

`app/__main__.py` (Aufrufer von `apply_safe_keyterms()`) und
`app/ha_vocabulary.py` wurden **nicht** verändert — die gesamte
Verbesserung ist zentral in `app/keyterms.py` gekapselt, wie gefordert.

## 9. Neue Tests

In `test_keyterms.py`:
- `test_fall_a_exact_production_incident_recovers_via_speech_fallback` —
  `/Büro` wird über Fallback zu `Büro` gerettet.
- `test_fall_a_variant_rejected_when_speech_fallback_also_fails` —
  Fallback selbst inkompatibel → Begriff bleibt verworfen.
- `test_separator_speech_fallback_recovers_slash_term` — `Treppe/Büro` →
  `Treppe Büro`.
- `test_original_kept_when_model_accepts_it_despite_separator` — `/` wird
  nicht präventiv entfernt, wenn das Modell es akzeptiert.
- `test_soft_hyphen_is_normalized_before_any_native_call_and_deduped` —
  `Wasch\xadmaschine` → `Waschmaschine`, dedupe mit sauberem Duplikat, kein
  nativer Call sieht je den Soft Hyphen.
- `test_emoji_with_variation_selector_recovers_via_speech_fallback` —
  `Familie ⚠️` → `Familie`.
- `test_emoji_only_term_with_nothing_left_is_rejected_not_sent_empty` —
  reiner Emoji-Begriff → verworfen, nie als leerer String gesendet.
- `TestNormalizeKeytermFormatCharacters` (8 Tests) — Soft Hyphen,
  Zero-Width-Space/Joiner/Non-Joiner, Word Joiner, BOM, Whitespace-
  Kollabierung, deutsche Begriffe unverändert.
- `TestGenerateSpeechFallback` (7 Tests) — Separatoren, Emoji,
  Emoji+Variation-Selector, reiner Emoji-Input, deutsche Zeichen/Ziffern/
  Bindestrich unverändert.
- Bestehende Fall-C/-D/Bisection-Tests angepasst, um echte
  Vollablehnung von echter Fallback-Wiederherstellung zu unterscheiden.

In `test_ha_vocabulary_refresh.py`:
- `test_separator_composite_term_recovers_via_speech_fallback` — Fall H
  erweitert um eine zusammengesetzte Area+Entity-Kombination
  (`EG/Büro Licht` → `EG Büro Licht`) über den Refresh-Pfad.

In `test_main_config.py`:
- Bestehender Fall-D-Test (`test_startup_survives_when_every_keyterm_is_incompatible`)
  angepasst, damit er weiterhin echte Vollablehnung testet statt versehentlich
  durch die neue Fallback-Logik zu grün zu werden.

## 10. Ergebnis pytest

```
398 passed, 4 deselected, 1 warning in 5.28s
```

(`python -m pytest tests/ -q`, volle Suite, keine Regression gegenüber v0.2.5's 376/4 — 22 neue Tests hinzugekommen)

## 11. Ergebnis ruff

```
ruff check .        → All checks passed!
ruff format --check . → 38 files already formatted
```

## 12. Ergebnis mypy

```
mypy --config-file pyproject.toml --python-executable /usr/local/bin/python3 .
Success: no issues found in 38 source files
```

## 13. Produktives erwartetes Verhalten für die vier konkreten Problembegriffe

Unter der Annahme, dass das tatsächlich geladene deutsche Moonshine-Modell
die bereinigten Varianten akzeptiert (das entscheidet weiterhin
ausschließlich das Modell selbst zur Laufzeit):

| Original | Stufe | Ergebnis |
|---|---|---|
| `Treppe/Büro` | Speech-Fallback (`/` → Leerzeichen) | `Treppe Büro` |
| `Wasch\xadmaschine` | Stufe-1-Normalisierung (Cf entfernt) | `Waschmaschine` |
| `Familie ⚠️` | Speech-Fallback (Emoji+VS entfernt) | `Familie` |
| `Erinnerungen ⚠️` | Speech-Fallback (Emoji+VS entfernt) | `Erinnerungen` |

Erwarteter Log (ungefähr, abhängig vom tatsächlichen Modellverhalten):

```
INFO  Normalized keyterm: 'Wasch\xadmaschine' -> 'Waschmaschine'
INFO  Replaced tokenizer-incompatible keyterm with speech-safe variant: 'Treppe/Büro' -> 'Treppe Büro'
INFO  Replaced tokenizer-incompatible keyterm with speech-safe variant: 'Familie ⚠️' -> 'Familie'
INFO  Replaced tokenizer-incompatible keyterm with speech-safe variant: 'Erinnerungen ⚠️' -> 'Erinnerungen'
INFO  Moonshine keyterms: candidate=249 unchanged=245 normalized=1 speech-fallback=3 rejected=0 applied=249
```

Sollte das Modell auch nur eine dieser Fallback-Varianten tatsächlich
ablehnen, wird genau dieser eine Begriff (nicht mehr) verworfen und wie
gehabt geloggt — der Dienst startet in jedem Fall, mit oder ohne
vollständiges Keyterm-Biasing.
