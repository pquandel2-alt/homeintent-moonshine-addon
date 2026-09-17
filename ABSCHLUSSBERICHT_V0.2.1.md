# Abschlussbericht: HomeIntent Moonshine Voice v0.2.1

**Datum**: 17. September 2026
**Autor**: Claude Sonnet 5
**Anlass dieser Fassung**: Gezieltes Follow-up auf v0.2.0. Ziel: Die automatische
Home-Assistant-Vokabularerzeugung soll primär genau die Entitäten berücksichtigen,
die für Home Assistant Assist/Conversation tatsächlich freigegeben ("exposed") sind
— statt (wie bisher) die gesamte Registry unabhängig von Assist-Freigabe zu
verwenden. Keine neue Architektur, kein Pocket-TTS-Redesign, kein unnötiges
Moonshine-STT-Refactoring.

---

## Executive Summary

**Assist-aware HA-Vokabular: CODE FERTIG, LOKAL VOLLSTÄNDIG VERIFIZIERT.**
Alle 26 vom Auftrag geforderten Testszenarien sind abgedeckt. Ruff, `ruff format
--check`, mypy (Projekt-Config, kein `--strict`-Flag separat nötig, siehe
`app/pyproject.toml`) und die vollständige lokale Testsuite (280 Tests) sind grün.

Nebenbei wurde eine tatsächlich rote GitHub-Actions-CI für den bereits
veröffentlichten v0.2.0-Tag entdeckt und behoben (Commit `30fac45`, bereits vor
Beginn dieses Auftrags auf `master` und diesem Branch gepusht und real grün
verifiziert) — siehe "Vorab-Fix" unten.

---

## Ausgangsstand

- **Vorherige veröffentlichte Version**: 0.2.0 (Tag `v0.2.0`, GitHub Release
  "STT & TTS", bereits vom Nutzer angelegt)
- **HEAD vorher**: `30fac45` auf `master`/`claude/moonshine-voice-stt-tts-73sjml`
- **Branch**: `claude/moonshine-voice-stt-tts-73sjml`
- **Bereits vorhandene Tags**: `v0.1.0`, `v0.1.1`, `v0.1.2`, `v0.2.0` → neue Version
  ist der nächste Patch-Schritt: **0.2.1**

### Vorab-Fix (vor diesem Auftrag, aber Teil desselben Release-Zyklus)

Der reale GitHub-Actions-Lauf für `eaa3a64` (v0.2.0) war rot:

- Docker-Build (amd64 + aarch64): `pip install torch --index-url
  https://download.pytorch.org/whl/cpu` ohne PyPI-Fallback scheiterte, weil
  `typing-extensions` dort nicht als Wheel liegt und dessen Build-Backend
  (`flit_core`) ebenfalls nicht auf diesem Index verfügbar ist.
- `e2e-audio-transcribe`: fehlender `pocket-tts`-Install, obwohl `app/handler.py`
  `app.tts` (und damit `pocket_tts`) jetzt unbedingt importiert.

Fix (Commit `30fac45`): `--extra-index-url https://pypi.org/simple` überall, wo
Torch installiert wird, plus der fehlende TTS-Dependency-Install in
`e2e-audio-transcribe`. Real per GitHub Actions verifiziert grün (Build-Workflow
`conclusion: "success"`).

---

## Recherche: Home Assistant Assist-Exposure-API (verifiziert gegen aktuellen Upstream)

Gegen den `dev`-Branch von `home-assistant/core` (Commit
`6c2d4140cc9b9b926a0df96c6eaa87b249b893c1`, 2026-09-17) verifiziert, nicht aus dem
Gedächtnis rekonstruiert:

- **Assistant-ID**: das eingebaute Assist/Conversation-Backend heißt in der
  Exposure-Logik **`"conversation"`** (`homeassistant/components/conversation/
  const.py`s `DOMAIN`), **nicht** `"conversation.home_assistant"` (das ist eine
  Conversation-Agent-Entity-ID, ein anderes Konzept). Bestätigt an jeder
  `async_should_expose()`-Aufrufstelle in dieser Komponente.
- **Default-Exposure-Regel** (`homeassistant/components/homeassistant/
  exposed_entities.py`, `ExposedEntities._is_default_exposed()`): `entity_category`
  gesetzt oder `hidden_by` gesetzt → nicht exponiert; sonst Domain in
  `DEFAULT_EXPOSED_DOMAINS` (`climate`, `cover`, `fan`, `humidifier`, `light`,
  `media_player`, `scene`, `switch`, `todo`, `vacuum`, `water_heater`) → exponiert;
  `binary_sensor` mit `device_class` in einer festen Allowlist (`door`,
  `garage_door`, `lock`, `motion`, `opening`, `presence`, `window`) → exponiert;
  `sensor` mit `device_class` in einer festen Allowlist (`aqi`,
  `carbon_monoxide`, `carbon_dioxide`, `humidity`, `pm10`, `pm25`, `temperature`,
  `volatile_organic_compounds`) → exponiert; sonst nicht.
- **Effektive Exposure** (`ExposedEntities.async_should_expose()`): ein expliziter,
  gecachter `registry_entry.options["conversation"]["should_expose"]`-Wert
  gewinnt immer über die Default-Regel; ohne Override greift die Default-Regel nur,
  wenn "expose new entities automatically" (`homeassistant/expose_new_entities/get`,
  Feld `expose_new`) für `conversation` aktiv ist; sonst nicht exponiert.
- **Warum nicht `homeassistant/expose_entity/list`**: dieser Bulk-Read gibt nur
  Entitäten zurück, deren `should_expose` bereits mindestens einmal (lazy, beim
  ersten echten Assist-Zugriff) berechnet und gecacht wurde — unzuverlässig als
  Quelle der Wahrheit, weil er sonst tatsächlich exponierte, aber noch nie
  berechnete Entitäten stillschweigend ausließe. Stattdessen liest dieses Add-on
  `config/entity_registry/list` (inkl. `options`/`entity_category`/`hidden_by`/
  `disabled_by`) und `get_states` (für `device_class` bei `sensor`/
  `binary_sensor`) und berechnet die effektive Exposure selbst.
- **Aliase**: `config/entity_registry/list` liefert keine Aliase; sie kommen aus
  dem separaten Bulk-Kommando `config/entity_registry/get_entries` (nimmt eine
  Liste von `entity_id`s, hier bewusst nur die bereits als exponiert ermittelten),
  um nicht die komplette Registry-Aliasliste unnötig zu übertragen.

Alle diese Fakten sind im Modul-Docstring von `app/ha_vocabulary.py` mit exakter
Quellenangabe dokumentiert (Anforderung: "im Code dokumentieren, auf welcher
HA-Core-Logik die Implementierung basiert").

---

## Implementierung (`app/ha_vocabulary.py`)

### Neue/geänderte Bausteine

- `is_default_exposed(entity, device_class)` — exakte Reimplementierung von
  `_is_default_exposed()` (siehe oben).
- `is_effectively_exposed(entity, device_class, expose_new_default)` —
  Override-Vorrang, sonst Default-Regel nur bei aktivem `expose_new`; zusätzlich
  `disabled_by`-Entitäten werden praktisch nie exponiert (Upstream prüft das an
  dieser Stelle nicht explizit, aber eine deaktivierte Entität soll nie in der
  Sprachvokabular-Biasing landen).
- `compute_exposed_entity_ids(entities, states, expose_new_default)` — verknüpft
  `get_states`-`device_class` mit der Registry und liefert die Menge der
  effektiv exponierten `entity_id`s.
- `extract_vocabulary_terms(...)` — neuer optionaler Parameter
  `exposed_entity_ids: set[str] | None`. `None` = unverändertes,
  rückwärtskompatibles "ungefiltertes" Verhalten (nur intern für Tests/Altcode
  relevant); ein konkretes Set filtert Areas/Devices/Floors/Entities auf genau die
  von exponierten Entitäten erreichbare Teilmenge (`_compute_relevance()`), bevor
  Namen/Alias/Area-Kombinationsterme gebaut werden.
- Entity-ID-Fallback (`_fallback_term_from_entity_id`): wird **nur** verwendet,
  wenn eine exponierte Entität weder `name` noch `original_name` noch Aliase hat —
  z. B. `cover.wohnzimmer_rolllade` → "Wohnzimmer Rolllade". Die Domain
  (`cover`, `sensor`, `binary_sensor`, ...) wird dabei **nie** als Wort verwendet.
- Geräte-Technik-Filter (`_has_technical_token` / `_device_term`): ein Default-
  Gerätename mit MAC-Adresse, UUID oder nacktem Hex-Token (z. B. "Shelly Plus
  2PM 84FCE6") wird verworfen; ein vom Nutzer gesetzter `name_by_user` wird
  immer als sinnvoll vertraut.
- Area+Entity- **und** Area+Alias-Kombinationsterme (`_area_entity_combination_
  terms`): iteriert jetzt über alle Namens-Kandidaten einer Entität (Name **und**
  jeden Alias), nicht mehr nur über den Primärnamen — genau das vom Auftrag
  geforderte "Wohnzimmer Rollo" aus dem Alias "Rollo".
- `fetch_ha_vocabulary()`: zusätzliche WS-Kommandos `get_states`,
  `homeassistant/expose_new_entities/get` und (nur falls mindestens eine Entität
  exponiert ist) `config/entity_registry/get_entries`; alle drei degradieren bei
  Nichtunterstützung durch ältere HA-Core-Versionen graceful (kein Absturz, kein
  Abbruch des gesamten Fetches).
- `HaVocabularyResult` hat ein neues Feld `exposed_entity_count: int` für kompaktes
  Logging.

### Unverändert (bewusst, laut Auftrag)

- Last-known-good-Verhalten bei fehlgeschlagenem Refresh (`app/__main__.py`s
  `_refresh_ha_vocabulary_periodically()`): unverändert, da `HaVocabularyResult.
  success` dieselbe Bedeutung wie zuvor hat. Ein *erfolgreicher* Refresh, der eine
  vormals exponierte Entität nicht mehr sieht, ersetzt jedoch (wie gefordert) die
  gesamte `terms`-Liste — das ist kein Last-known-good-Fall, sondern eine normale,
  korrekte Aktualisierung.
- Moonshine-Locking: `set_keyterms()` läuft weiterhin ausschließlich innerhalb des
  bereits vorhandenen `moonshine_lock`, unverändert.
- `extra_keyterms`/`merge_keyterms()`: unverändert, unabhängig von Assist-Exposure.
- Polling-basierter Refresh (`ha_vocabulary_refresh_minutes`): beibehalten. Es
  existiert in aktuellem Home Assistant Core kein dediziertes, stabil
  dokumentiertes "Exposure geändert"-Event — `entity_registry_updated` ist ein
  genereller, sehr rauschender Registry-Event ohne Exposure-Semantik. Polling
  bleibt daher die angemessene Lösung; kein aggressiveres Polling eingeführt.
- Pocket TTS, Wyoming-TTS-Lifecycle, TTS-Streaming: keine Änderungen.

### Logging

`fetch_ha_vocabulary()` loggt bei jedem (Re-)Fetch eine kompakte Zeile:
`"Assist-exposed entities: %d, HA vocabulary terms: %d"`. `app/__main__.py`s
bestehende Startbanner-Zeilen (`STT: manual keyterms=%d`, `STT: effective
keyterms=%d`) blieben unverändert bestehen. Keine vollständige Entitäts-/
Alias-/Vokabularliste wird bei Standard-Loglevel ausgegeben.

---

## Tests

Neue/erweiterte Testdatei: `app/tests/test_ha_vocabulary.py`. Deckt alle 26 vom
Auftrag geforderten Szenarien ab, u. a.:

1. Explizit exponierte Entität (`TestIsEffectivelyExposed`)
2. Explizit versteckte Entität (`should_expose: false` gewinnt über Default)
3. Fehlender Override → Default-Regel
4. Effektive Exposure exakt nach realer HA-Core-Logik
   (`TestIsDefaultExposed`, `TestIsEffectivelyExposed`, `TestComputeExposedEntityIds`)
5. Entity-Alias (`test_entity_alias_included_alongside_name`)
6. Mehrere Aliase (`test_multiple_entity_aliases_all_included`)
7. Entity-eigene Area (`test_entity_direct_area_id_produces_combination`)
8. Device-Area-Fallback (`test_entity_inherits_area_from_device`)
9. Area+Entity-Keyterm (`test_realistic_multi_room_registry`, u. a.)
10. Area+Alias-Keyterm (`test_area_alias_combination_terms_from_entity_alias`)
11. Sinnvoll behandelter technischer Gerätename (`test_device_user_customized_
    name_always_trusted`)
12. MAC/Hex-artiger Gerätename NICHT verwendet (`test_device_technical_default_
    name_with_mac_address_excluded`, `..._with_hex_serial_excluded`)
13. Entity-ID-Fallback (`test_entity_id_used_as_last_resort_fallback_when_no_
    name_or_alias`)
14. Domain nie als Keyterm (`test_entity_id_fallback_never_includes_the_domain`,
    `test_entity_domain_never_used_as_a_word`)
15. `extra_keyterms` zusätzlich (unverändertes bestehendes Verhalten in
    `app/tests/test_keyterms.py`, hier nicht angetastet)
16. Dedupe (`TestDedupePreserveOrder`)
17. Umlaute (`test_dedupes_hyphen_and_umlaut_variants`, `TestCleanTerm`)
18. API-Fehler (`test_auth_failure_returns_failed_result`,
    `test_connection_error_returns_failed_result_not_raises`)
19. Last-known-good bleibt erhalten (unverändertes `app/tests/test_main.py`-
    Verhalten für `_refresh_ha_vocabulary_periodically`, nicht angetastet)
20. Erfolgreicher Snapshot entfernt nicht mehr exponierte Entität
    (`test_non_exposed_entity_produces_no_vocabulary`,
    `test_realistic_wohnzimmer_rolllade_scenario`)
21. Refresh (`_full_script`-basierte `TestFetchHaVocabulary`-Tests)
22. Moonshine-Lock (unverändert, bereits in `app/tests/test_main.py` abgedeckt)
23. Leere Assist-Liste (`test_empty_assist_exposure_list_is_success_with_no_terms`)
24. Sehr viele Entitäten (`test_many_exposed_entities_all_produce_vocabulary`,
    200 Entitäten)
25. Bestehende STT-Tests: alle unverändert grün
26. Bestehende TTS-Tests: alle unverändert grün

Realistisches Kern-Szenario aus dem Auftrag exakt nachgebaut
(`test_realistic_wohnzimmer_rolllade_scenario`): Area "Wohnzimmer", Device
"Rollladenaktor", Entity `cover.wohnzimmer_rolllade` mit Name "Rolllade" und Alias
"Rollo", exponiert → erwartete Terme `Wohnzimmer`, `Rolllade`, `Rollo`,
`Wohnzimmer Rolllade`, `Wohnzimmer Rollo`; eine zweite, nicht exponierte
`sensor.router_cpu_temperature` erzeugt nachweislich kein Vokabular.

Keine bestehenden Tests wurden gelöscht oder abgeschwächt, um grün zu werden;
`test_never_derives_term_from_raw_entity_id` wurde durch mehrere Tests ersetzt, die
das absichtlich geänderte (und vom Auftrag explizit geforderte) Fallback-Verhalten
korrekt abbilden.

**Ergebnis**: `pytest app/tests/test_ha_vocabulary.py` → 79 Tests grün.
`pytest` (gesamte Suite) → **280 Tests grün, 5 übersprungen** (die 5 sind die
bereits vorher als `@pytest.mark.e2e`/`workflow_dispatch`-only markierten realen
Modell-Roundtrips, die einen echten Modell-Download brauchen — unverändert seit
v0.2.0).

---

## Quality Gates (lokal ausgeführt)

| Check | Ergebnis |
|---|---|
| `ruff check .` | ✅ All checks passed |
| `ruff format --check .` | ✅ 36 files already formatted |
| `mypy --config-file app/pyproject.toml app` | ✅ Success: no issues found in 36 source files |
| `pytest` (volle Suite) | ✅ 280 passed, 5 skipped |

Docker-Build, Add-on-Startup-Smoke-Test, Wyoming-Describe, echte STT-/TTS-E2E-Tests:
wie schon bei v0.2.0 in dieser Sandbox nicht ausführbar (Netzwerk-Policy blockiert
Docker-Image-Pulls und externe Modell-Downloads) — werden im echten
GitHub-Actions-Lauf verifiziert (siehe unten).

---

## Git / GitHub

- **Commit**: `19223c7` ("feat: sync STT vocabulary with Assist-exposed entities"),
  gepusht auf `claude/moonshine-voice-stt-tts-73sjml` und direkt auf `master`
  (bisheriges Vorgehen dieses Repos, wie bei v0.1.x/v0.2.0 beibehalten).
- **CI (realer GitHub-Actions-Lauf für `19223c7`)**: **GRÜN** — alle Pflicht-Checks:
  - `Lint` ✅ success
  - `Type Check` ✅ success
  - `Tests` ✅ success
  - `Build` ✅ success, mit allen vier Jobs grün: `e2e-audio-transcribe`
    (echtes Wyoming-STT-Roundtrip), `Build amd64 image + smoke test` (echter
    Docker-Build + realer Container-Start + echtes Wyoming-Describe, inkl. des
    Upgrade-Simulations-Jobs aus v0.2.0), `Build aarch64 image (QEMU,
    build-only)`. Der TTS-E2E-Job (`workflow_dispatch`-only) wurde erwartungsgemäß
    übersprungen (`skipped`), wie bei jedem normalen Push seit v0.2.0.
- **Tag `v0.2.1`**: **FEHLGESCHLAGEN.** `git tag -a v0.2.1 19223c7 -m "..."` gelang
  lokal; `git push origin v0.2.1` scheiterte mit:
  ```
  error: RPC failed; HTTP 403 curl 22 The requested URL returned error: 403
  send-pack: unexpected disconnect while reading sideband packet
  fatal: the remote end hung up unexpectedly
  ```
  Das ist dieselbe Einschränkung wie beim v0.2.0-Release: normale Branch-Pushes
  (`git push origin <branch>`) funktionieren mit der aktuell autorisierten
  Claude-GitHub-App, ein `git push` eines Tag-Refs wird jedoch von GitHub selbst
  mit 403 abgelehnt (kein Netzwerk-Policy-Block dieser Sandbox — der zugrunde
  liegende `CONNECT` zu github.com gelingt; der 403 kommt von GitHub). Vermutlich
  fehlt der App die Berechtigung, Tag-Refs zu erstellen. Der lokale Tag wurde
  wieder gelöscht, um keinen falschen Eindruck zu hinterlassen.
- **GitHub Release**: **NICHT VERÖFFENTLICHT**, da kein Tag existiert und keines
  der verfügbaren GitHub-MCP-Tools eine Release- oder Tag-Ref-Erstellung anbietet
  (nur Lese-Tools: `get_latest_release`, `get_release_by_tag`, `list_releases`,
  `list_tags`, `get_tag`).

**Nächster Schritt für den Nutzer** (wie schon bei v0.2.0 erfolgreich genutzt):
GitHub erlaubt, beim Anlegen eines Release direkt einen neuen Tag auf einem
bestimmten Commit zu erzeugen. Dafür genügt es, diese vorausgefüllte URL zu öffnen
und **Ziel-Commit `19223c7`** auszuwählen (Tag `v0.2.1` existiert noch nicht und
wird beim Veröffentlichen automatisch erstellt):

`https://github.com/pquandel2-alt/homeintent-moonshine-addon/releases/new?tag=v0.2.1&target=19223c7&title=HomeIntent+Moonshine+Voice+v0.2.1`

Vorgeschlagene Release-Notes:

> **What's changed**
> - Assist-aware STT vocabulary: automatic keyterm generation now reflects
>   exactly the entities exposed to Home Assistant Assist, computed the same
>   way Home Assistant Core itself determines effective exposure — including
>   entity aliases and Area+Entity/Area+Alias combinations (e.g. "Wohnzimmer
>   Rolllade", "Wohnzimmer Rollo").
> - Reliability: a successful refresh that no longer sees a previously-exposed
>   entity removes it; a failed refresh (HA unreachable) still keeps the
>   last-known-good vocabulary, as before.
> - Existing features unchanged: Moonshine STT streaming/locking and Kyutai
>   Pocket TTS are untouched by this release.
>
> **Upgrade**: normal update through the Home Assistant Add-on Store — no
> manual installation steps required.
