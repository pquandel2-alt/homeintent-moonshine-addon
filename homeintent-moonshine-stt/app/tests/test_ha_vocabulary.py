"""Tests for read-only Home Assistant vocabulary collection.

Covers the pure normalization/exposure helpers directly, and
fetch_ha_vocabulary() against a fake WebSocket connection standing in for
Supervisor's proxied Home Assistant Core WebSocket API -- no real
network/Supervisor is reached.
"""

import itertools
import json
from typing import Any

import pytest

from app.ha_vocabulary import (
    CONVERSATION_ASSISTANT,
    MAX_WS_MESSAGE_SIZE,
    HaVocabularyResult,
    build_legacy_pseudo_entities,
    clean_term,
    compute_exposed_entity_ids,
    compute_legacy_exposed_entity_ids,
    dedupe_preserve_order,
    extract_vocabulary_terms,
    fetch_ha_vocabulary,
    is_default_exposed,
    is_effectively_exposed,
    legacy_entity_ids,
)


class TestCleanTerm:
    def test_trims_leading_trailing_whitespace(self):
        assert clean_term("  Wohnzimmer  ") == "Wohnzimmer"

    def test_collapses_internal_whitespace(self):
        assert clean_term("Wohn   zimmer") == "Wohn zimmer"

    def test_preserves_umlauts(self):
        assert clean_term("Küche") == "Küche"

    def test_preserves_hyphens(self):
        assert clean_term("Büro-Schreibtischlampe") == "Büro-Schreibtischlampe"

    def test_none_input_returns_none(self):
        assert clean_term(None) is None

    def test_empty_string_returns_none(self):
        assert clean_term("") is None

    def test_whitespace_only_returns_none(self):
        assert clean_term("   ") is None


class TestDedupePreserveOrder:
    def test_dedupes_case_insensitively(self):
        assert dedupe_preserve_order(["Küche", "küche", "Bad"]) == ["Küche", "Bad"]

    def test_preserves_first_seen_casing(self):
        assert dedupe_preserve_order(["wohnzimmer", "Wohnzimmer"]) == ["wohnzimmer"]

    def test_empty_list(self):
        assert dedupe_preserve_order([]) == []

    def test_dedupes_hyphen_and_umlaut_variants(self):
        assert dedupe_preserve_order(["Büro-Lampe", "büro-lampe"]) == ["Büro-Lampe"]


class TestIsDefaultExposed:
    """Reimplementation of ExposedEntities._is_default_exposed() (see
    module docstring for the exact verified upstream source)."""

    def test_domain_in_default_exposed_domains_is_true(self):
        entity = {"entity_id": "light.wohnzimmer", "entity_category": None, "hidden_by": None}
        assert is_default_exposed(entity, device_class=None) is True

    def test_domain_not_in_default_exposed_domains_is_false(self):
        entity = {
            "entity_id": "sensor.router_cpu_temperature",
            "entity_category": None,
            "hidden_by": None,
        }
        assert is_default_exposed(entity, device_class="voltage") is False

    def test_entity_category_excludes_from_default_exposure(self):
        entity = {"entity_id": "light.wohnzimmer", "entity_category": "config", "hidden_by": None}
        assert is_default_exposed(entity, device_class=None) is False

    def test_hidden_by_excludes_from_default_exposure(self):
        entity = {"entity_id": "light.wohnzimmer", "entity_category": None, "hidden_by": "user"}
        assert is_default_exposed(entity, device_class=None) is False

    def test_binary_sensor_with_allowlisted_device_class_is_true(self):
        entity = {
            "entity_id": "binary_sensor.garage_tor",
            "entity_category": None,
            "hidden_by": None,
        }
        assert is_default_exposed(entity, device_class="garage_door") is True

    def test_binary_sensor_with_non_allowlisted_device_class_is_false(self):
        entity = {
            "entity_id": "binary_sensor.something",
            "entity_category": None,
            "hidden_by": None,
        }
        assert is_default_exposed(entity, device_class="battery") is False

    def test_sensor_with_allowlisted_device_class_is_true(self):
        entity = {
            "entity_id": "sensor.wohnzimmer_temperatur",
            "entity_category": None,
            "hidden_by": None,
        }
        assert is_default_exposed(entity, device_class="temperature") is True

    def test_sensor_with_non_allowlisted_device_class_is_false(self):
        entity = {
            "entity_id": "sensor.router_cpu_temperature",
            "entity_category": None,
            "hidden_by": None,
        }
        assert is_default_exposed(entity, device_class="voltage") is False


class TestIsEffectivelyExposed:
    """Reimplementation of ExposedEntities.async_should_expose(): an
    explicit cached should_expose value always wins over the default rule;
    disabled entities are never exposed."""

    def test_explicit_should_expose_true_wins(self):
        entity = {
            "entity_id": "sensor.router_cpu_temperature",
            "entity_category": None,
            "hidden_by": None,
            "options": {CONVERSATION_ASSISTANT: {"should_expose": True}},
        }
        assert (
            is_effectively_exposed(entity, device_class="voltage", expose_new_default=True) is True
        )

    def test_explicit_should_expose_false_wins_over_default_true(self):
        entity = {
            "entity_id": "light.wohnzimmer",
            "entity_category": None,
            "hidden_by": None,
            "options": {CONVERSATION_ASSISTANT: {"should_expose": False}},
        }
        assert is_effectively_exposed(entity, device_class=None, expose_new_default=True) is False

    def test_no_override_falls_back_to_default_rule_when_expose_new_enabled(self):
        entity = {"entity_id": "light.wohnzimmer", "entity_category": None, "hidden_by": None}
        assert is_effectively_exposed(entity, device_class=None, expose_new_default=True) is True

    def test_no_override_and_expose_new_disabled_is_false(self):
        entity = {"entity_id": "light.wohnzimmer", "entity_category": None, "hidden_by": None}
        assert is_effectively_exposed(entity, device_class=None, expose_new_default=False) is False

    def test_disabled_entity_is_never_exposed_even_with_explicit_override(self):
        entity = {
            "entity_id": "light.wohnzimmer",
            "entity_category": None,
            "hidden_by": None,
            "disabled_by": "user",
            "options": {CONVERSATION_ASSISTANT: {"should_expose": True}},
        }
        assert is_effectively_exposed(entity, device_class=None, expose_new_default=True) is False

    def test_override_for_a_different_assistant_is_ignored(self):
        entity = {
            "entity_id": "light.wohnzimmer",
            "entity_category": None,
            "hidden_by": None,
            "options": {"cloud.alexa": {"should_expose": True}},
        }
        # No conversation-assistant override present -> falls through to
        # the default rule, which is True for a light domain anyway; use a
        # non-default-exposed domain to actually distinguish the two paths.
        entity["entity_id"] = "sensor.router_cpu_temperature"
        assert (
            is_effectively_exposed(entity, device_class="voltage", expose_new_default=True) is False
        )


class TestComputeExposedEntityIds:
    def test_empty_list_produces_empty_set(self):
        assert compute_exposed_entity_ids([], [], expose_new_default=True) == set()

    def test_default_exposed_entity_included(self):
        entities = [{"entity_id": "light.wohnzimmer", "entity_category": None, "hidden_by": None}]
        assert compute_exposed_entity_ids(entities, [], expose_new_default=True) == {
            "light.wohnzimmer"
        }

    def test_non_exposed_sensor_excluded_without_override(self):
        entities = [
            {
                "entity_id": "sensor.router_cpu_temperature",
                "entity_category": None,
                "hidden_by": None,
            }
        ]
        states = [
            {
                "entity_id": "sensor.router_cpu_temperature",
                "attributes": {"device_class": "temperature"},
            }
        ]
        # temperature device_class *is* allowlisted for sensors, so this
        # entity is actually exposed by default; use a non-allowlisted
        # device_class to prove exclusion.
        states = [
            {
                "entity_id": "sensor.router_cpu_temperature",
                "attributes": {"device_class": "voltage"},
            }
        ]
        assert compute_exposed_entity_ids(entities, states, expose_new_default=True) == set()

    def test_device_class_resolved_from_live_state(self):
        entities = [
            {
                "entity_id": "sensor.wohnzimmer_temperatur",
                "entity_category": None,
                "hidden_by": None,
            }
        ]
        states = [
            {
                "entity_id": "sensor.wohnzimmer_temperatur",
                "attributes": {"device_class": "temperature"},
            }
        ]
        assert compute_exposed_entity_ids(entities, states, expose_new_default=True) == {
            "sensor.wohnzimmer_temperatur"
        }

    def test_entity_without_entity_id_is_skipped(self):
        entities = [{"entity_category": None, "hidden_by": None}]
        assert compute_exposed_entity_ids(entities, [], expose_new_default=True) == set()

    def test_mixed_exposed_and_non_exposed_entities(self):
        entities: list[dict[str, Any]] = [
            {"entity_id": "light.wohnzimmer", "entity_category": None, "hidden_by": None},
            {
                "entity_id": "sensor.router_cpu_temperature",
                "entity_category": None,
                "hidden_by": None,
            },
            {
                "entity_id": "cover.garage_tor",
                "entity_category": None,
                "hidden_by": None,
                "options": {CONVERSATION_ASSISTANT: {"should_expose": False}},
            },
        ]
        states = [
            {
                "entity_id": "sensor.router_cpu_temperature",
                "attributes": {"device_class": "voltage"},
            }
        ]
        assert compute_exposed_entity_ids(entities, states, expose_new_default=True) == {
            "light.wohnzimmer"
        }


class TestExtractVocabularyTerms:
    def test_area_name_included(self):
        areas = [{"name": "Wohnzimmer", "aliases": []}]
        assert extract_vocabulary_terms(areas, [], []) == ["Wohnzimmer"]

    def test_area_aliases_included(self):
        areas = [{"name": "Wohnzimmer", "aliases": ["Salon", "Living Room"]}]
        terms = extract_vocabulary_terms(areas, [], [])
        assert terms == ["Wohnzimmer", "Salon", "Living Room"]

    def test_floor_name_included(self):
        floors = [{"name": "Erdgeschoss"}]
        assert extract_vocabulary_terms([], [], [], floors) == ["Erdgeschoss"]

    def test_device_prefers_name_by_user_over_name(self):
        devices = [{"name": "Shelly 1PM", "name_by_user": "Deckenlampe Küche"}]
        assert extract_vocabulary_terms([], devices, []) == ["Deckenlampe Küche"]

    def test_device_falls_back_to_name_when_no_name_by_user(self):
        devices = [{"name": "Shelly 1PM", "name_by_user": None}]
        assert extract_vocabulary_terms([], devices, []) == ["Shelly 1PM"]

    def test_device_technical_default_name_with_mac_address_excluded(self):
        devices = [{"name": "Shelly Plus 2PM 84:FC:E6:12:34:56", "name_by_user": None}]
        assert extract_vocabulary_terms([], devices, []) == []

    def test_device_technical_default_name_with_hex_serial_excluded(self):
        devices = [{"name": "Shelly Plus 2PM 84FCE612", "name_by_user": None}]
        assert extract_vocabulary_terms([], devices, []) == []

    def test_device_user_customized_name_always_trusted(self):
        devices = [{"name": "Shelly Plus 2PM 84FCE612", "name_by_user": "Rolllade Aktor"}]
        assert extract_vocabulary_terms([], devices, []) == ["Rolllade Aktor"]

    def test_entity_prefers_name_over_original_name(self):
        entities = [{"name": "Deckenlampe", "original_name": "light.wohnzimmer_2"}]
        assert extract_vocabulary_terms([], [], entities) == ["Deckenlampe"]

    def test_entity_falls_back_to_original_name(self):
        entities = [{"name": None, "original_name": "Deckenlampe Flur"}]
        assert extract_vocabulary_terms([], [], entities) == ["Deckenlampe Flur"]

    def test_entity_id_used_as_last_resort_fallback_when_no_name_or_alias(self):
        entities: list[dict[str, Any]] = [
            {
                "entity_id": "cover.wohnzimmer_rolllade",
                "name": None,
                "original_name": None,
                "aliases": [],
            }
        ]
        assert extract_vocabulary_terms([], [], entities) == ["Wohnzimmer Rolllade"]

    def test_entity_id_fallback_never_includes_the_domain(self):
        entities: list[dict[str, Any]] = [
            {
                "entity_id": "binary_sensor.wohnzimmer_fenster",
                "name": None,
                "original_name": None,
                "aliases": [],
            }
        ]
        terms = extract_vocabulary_terms([], [], entities)
        assert terms == ["Wohnzimmer Fenster"]
        assert not any("sensor" in t.casefold() for t in terms)

    def test_entity_id_fallback_not_used_when_name_is_present(self):
        entities = [
            {
                "entity_id": "light.wohnzimmer_deckenlampe",
                "name": "Deckenlampe",
                "original_name": None,
            }
        ]
        assert extract_vocabulary_terms([], [], entities) == ["Deckenlampe"]

    def test_entity_id_fallback_not_used_when_only_alias_is_present(self):
        entities = [
            {
                "entity_id": "light.wohnzimmer_deckenlampe",
                "name": None,
                "original_name": None,
                "aliases": ["Lampe"],
            }
        ]
        assert extract_vocabulary_terms([], [], entities) == ["Lampe"]

    def test_entity_alias_included_alongside_name(self):
        entities = [
            {
                "entity_id": "cover.wohnzimmer_rolllade",
                "name": "Rolllade",
                "original_name": None,
                "aliases": ["Rollo"],
            }
        ]
        terms = extract_vocabulary_terms([], [], entities)
        assert terms == ["Rolllade", "Rollo"]

    def test_multiple_entity_aliases_all_included(self):
        entities = [
            {
                "entity_id": "cover.wohnzimmer_rolllade",
                "name": "Rolllade",
                "original_name": None,
                "aliases": ["Rollo", "Jalousie"],
            }
        ]
        terms = extract_vocabulary_terms([], [], entities)
        assert terms == ["Rolllade", "Rollo", "Jalousie"]

    def test_dedupes_across_categories(self):
        areas = [{"name": "Küche", "aliases": []}]
        devices = [{"name": "Küche", "name_by_user": None}]
        assert extract_vocabulary_terms(areas, devices, []) == ["Küche"]

    def test_empty_registries_produce_empty_list(self):
        assert extract_vocabulary_terms([], [], []) == []

    def test_skips_entries_without_usable_name(self):
        areas: list[dict[str, Any]] = [
            {"name": None, "aliases": []},
            {"name": "Bad", "aliases": []},
        ]
        assert extract_vocabulary_terms(areas, [], []) == ["Bad"]

    def test_exposed_entity_ids_filters_out_non_exposed_entities(self):
        entities = [
            {"entity_id": "light.wohnzimmer", "name": "Deckenlampe", "original_name": None},
            {
                "entity_id": "sensor.router_cpu_temperature",
                "name": "CPU Temperatur",
                "original_name": None,
            },
        ]
        terms = extract_vocabulary_terms([], [], entities, exposed_entity_ids={"light.wohnzimmer"})
        assert terms == ["Deckenlampe"]

    def test_exposed_entity_ids_empty_set_produces_no_entity_terms(self):
        entities = [{"entity_id": "light.wohnzimmer", "name": "Deckenlampe", "original_name": None}]
        terms = extract_vocabulary_terms([], [], entities, exposed_entity_ids=set())
        assert terms == []

    def test_exposed_entity_ids_none_means_unfiltered_backward_compatible_mode(self):
        entities = [
            {
                "entity_id": "sensor.router_cpu_temperature",
                "name": "CPU Temperatur",
                "original_name": None,
            }
        ]
        terms = extract_vocabulary_terms([], [], entities, exposed_entity_ids=None)
        assert terms == ["CPU Temperatur"]

    def test_areas_devices_floors_filtered_to_only_those_reachable_from_exposed_entities(self):
        areas = [
            {"area_id": "living_room", "name": "Wohnzimmer", "aliases": [], "floor_id": "ground"},
            {"area_id": "office", "name": "Büro", "aliases": [], "floor_id": "upper"},
        ]
        devices = [
            {
                "id": "dev_lr",
                "area_id": "living_room",
                "name": "Aktor Wohnzimmer",
                "name_by_user": None,
            },
            {"id": "dev_office", "area_id": "office", "name": "Aktor Büro", "name_by_user": None},
        ]
        entities = [
            {
                "entity_id": "cover.wohnzimmer_rolllade",
                "area_id": None,
                "device_id": "dev_lr",
                "name": "Rolllade",
                "original_name": None,
            },
            {
                "entity_id": "sensor.buero_cpu_temperature",
                "area_id": None,
                "device_id": "dev_office",
                "name": "CPU Temperatur",
                "original_name": None,
            },
        ]
        floors = [
            {"floor_id": "ground", "name": "Erdgeschoss"},
            {"floor_id": "upper", "name": "Obergeschoss"},
        ]

        terms = extract_vocabulary_terms(
            areas, devices, entities, floors, exposed_entity_ids={"cover.wohnzimmer_rolllade"}
        )

        assert "Wohnzimmer" in terms
        assert "Erdgeschoss" in terms
        assert "Aktor Wohnzimmer" in terms
        assert "Büro" not in terms
        assert "Obergeschoss" not in terms
        assert "Aktor Büro" not in terms
        assert "CPU Temperatur" not in terms


class TestAreaEntityCombinationTerms:
    """A5 review fix: "Area Entity" combination terms from real, known HA
    Entity->Area / Entity->Device->Area relationships only."""

    def test_entity_direct_area_id_produces_combination(self):
        areas = [{"area_id": "living_room", "name": "Wohnzimmer", "aliases": []}]
        entities = [
            {
                "entity_id": "cover.wohnzimmer_rolllade",
                "area_id": "living_room",
                "device_id": None,
                "name": "Rolllade",
                "original_name": None,
            }
        ]
        terms = extract_vocabulary_terms(areas, [], entities)
        assert "Wohnzimmer Rolllade" in terms

    def test_entity_inherits_area_from_device(self):
        areas = [{"area_id": "kitchen", "name": "Küche", "aliases": []}]
        devices = [{"id": "dev1", "area_id": "kitchen", "name": "Shelly", "name_by_user": None}]
        entities = [
            {
                "entity_id": "binary_sensor.kueche_fenster",
                "area_id": None,
                "device_id": "dev1",
                "name": "Fenster",
                "original_name": None,
            }
        ]
        terms = extract_vocabulary_terms(areas, devices, entities)
        assert "Küche Fenster" in terms

    def test_area_alias_combination_terms_from_entity_alias(self):
        """Wohnzimmer + alias "Rollo" -> "Wohnzimmer Rollo", per the user's
        cover.wohnzimmer_rolllade example."""
        areas = [{"area_id": "living_room", "name": "Wohnzimmer", "aliases": []}]
        entities = [
            {
                "entity_id": "cover.wohnzimmer_rolllade",
                "area_id": "living_room",
                "device_id": None,
                "name": "Rolllade",
                "original_name": None,
                "aliases": ["Rollo"],
            }
        ]
        terms = extract_vocabulary_terms(areas, [], entities)
        for expected in (
            "Wohnzimmer",
            "Rolllade",
            "Rollo",
            "Wohnzimmer Rolllade",
            "Wohnzimmer Rollo",
        ):
            assert expected in terms

    def test_realistic_multi_room_registry(self):
        """Wohnzimmer: Rolllade, Deckenlampe, Fenster / Küche: Fenster,
        Deckenlampe / Schlafzimmer: Heizung / Garage: Licht."""
        areas = [
            {"area_id": "living_room", "name": "Wohnzimmer", "aliases": []},
            {"area_id": "kitchen", "name": "Küche", "aliases": []},
            {"area_id": "bedroom", "name": "Schlafzimmer", "aliases": []},
            {"area_id": "garage", "name": "Garage", "aliases": []},
            {"area_id": "bathroom", "name": "Badezimmer", "aliases": []},
        ]
        devices = [
            {"id": "dev_lr", "area_id": "living_room", "name": "Dev", "name_by_user": None},
            {"id": "dev_kitchen", "area_id": "kitchen", "name": "Dev", "name_by_user": None},
        ]
        entities = [
            {
                "entity_id": "cover.rolllade",
                "area_id": "living_room",
                "device_id": None,
                "name": "Rolllade",
                "original_name": None,
            },
            {
                "entity_id": "light.deckenlampe_wz",
                "area_id": None,
                "device_id": "dev_lr",
                "name": "Deckenlampe",
                "original_name": None,
            },
            {
                "entity_id": "binary_sensor.fenster_wz",
                "area_id": "living_room",
                "device_id": None,
                "name": "Fenster",
                "original_name": None,
            },
            {
                "entity_id": "binary_sensor.fenster_kueche",
                "area_id": None,
                "device_id": "dev_kitchen",
                "name": "Fenster",
                "original_name": None,
            },
            {
                "entity_id": "light.deckenlampe_kueche",
                "area_id": "kitchen",
                "device_id": None,
                "name": "Deckenlampe",
                "original_name": None,
            },
            {
                "entity_id": "climate.heizung_sz",
                "area_id": "bedroom",
                "device_id": None,
                "name": "Heizung",
                "original_name": None,
            },
            {
                "entity_id": "light.garage",
                "area_id": "garage",
                "device_id": None,
                "name": "Licht",
                "original_name": None,
            },
            {
                "entity_id": "fan.badezimmer_lueftung",
                "area_id": "bathroom",
                "device_id": None,
                "name": "Lüfter",
                "original_name": None,
            },
        ]

        terms = extract_vocabulary_terms(areas, devices, entities)

        for expected in (
            "Wohnzimmer Rolllade",
            "Wohnzimmer Deckenlampe",
            "Wohnzimmer Fenster",
            "Küche Fenster",
            "Küche Deckenlampe",
            "Schlafzimmer Heizung",
            "Garage Licht",
            "Badezimmer Lüfter",
        ):
            assert expected in terms

    def test_entity_without_any_area_produces_no_combination(self):
        areas = [{"area_id": "living_room", "name": "Wohnzimmer", "aliases": []}]
        entities = [
            {
                "entity_id": "sensor.unassigned",
                "area_id": None,
                "device_id": None,
                "name": "Unbekannt",
                "original_name": None,
            }
        ]
        terms = extract_vocabulary_terms(areas, [], entities)
        assert not any(t.startswith("Wohnzimmer ") for t in terms)

    def test_entity_domain_never_used_as_a_word(self):
        areas = [{"area_id": "living_room", "name": "Wohnzimmer", "aliases": []}]
        entities = [
            {
                "entity_id": "binary_sensor.wohnzimmer_deckenlampe",
                "area_id": "living_room",
                "device_id": None,
                "name": None,
                "original_name": None,
            }
        ]
        terms = extract_vocabulary_terms(areas, [], entities)
        assert not any("binary_sensor" in t.casefold() for t in terms)
        assert not any("sensor" in t.casefold() for t in terms)

    def test_no_redundant_duplicate_when_entity_name_already_has_area_prefix(self):
        areas = [{"area_id": "living_room", "name": "Wohnzimmer", "aliases": []}]
        entities = [
            {
                "entity_id": "light.wz_lampe",
                "area_id": "living_room",
                "device_id": None,
                "name": "Wohnzimmer Lampe",
                "original_name": None,
            }
        ]
        terms = extract_vocabulary_terms(areas, [], entities)
        assert "Wohnzimmer Wohnzimmer Lampe" not in terms

    def test_combination_terms_are_deterministically_sorted(self):
        areas = [{"area_id": "a", "name": "Zimmer", "aliases": []}]
        entities = [
            {"entity_id": f"light.{i}", "area_id": "a", "device_id": None, "name": name}
            for i, name in enumerate(["Zebra", "Anna", "Mitte"])
        ]
        from app.ha_vocabulary import _area_entity_combination_terms

        combos = _area_entity_combination_terms(areas, [], entities)
        assert combos == sorted(combos, key=str.casefold)

    def test_combination_terms_capped_at_maximum(self):
        from app.ha_vocabulary import MAX_COMBINATION_TERMS, _area_entity_combination_terms

        areas = [{"area_id": "a", "name": "Zimmer", "aliases": []}]
        entities = [
            {"entity_id": f"light.e{i}", "area_id": "a", "device_id": None, "name": f"Ding{i}"}
            for i in range(MAX_COMBINATION_TERMS + 50)
        ]
        combos = _area_entity_combination_terms(areas, [], entities)
        assert len(combos) == MAX_COMBINATION_TERMS

    def test_no_area_relationship_means_no_combinatorial_guessing(self):
        """Multiple areas and multiple unrelated entities must never be
        cross-combined -- only real, known relationships produce terms."""
        areas = [
            {"area_id": "a", "name": "Wohnzimmer", "aliases": []},
            {"area_id": "b", "name": "Küche", "aliases": []},
        ]
        entities = [
            {
                "entity_id": "light.x",
                "area_id": None,
                "device_id": None,
                "name": "Deckenlampe",
                "original_name": None,
            }
        ]
        terms = extract_vocabulary_terms(areas, [], entities)
        assert "Wohnzimmer Deckenlampe" not in terms
        assert "Küche Deckenlampe" not in terms


class _FakeWebSocket:
    """Stands in for a websockets ClientConnection in tests."""

    def __init__(self, script: list[dict[str, Any]]) -> None:
        self._incoming = list(script)
        self.sent: list[dict[str, Any]] = []

    async def send(self, message: str) -> None:
        self.sent.append(json.loads(message))

    async def recv(self) -> str:
        return json.dumps(self._incoming.pop(0))

    async def __aenter__(self) -> "_FakeWebSocket":
        return self

    async def __aexit__(self, *exc: object) -> bool:
        return False


def _command_response(command_id: int, result: Any) -> dict[str, Any]:
    return {"id": command_id, "type": "result", "success": True, "result": result}


def _error_response(command_id: int) -> dict[str, Any]:
    return {
        "id": command_id,
        "type": "result",
        "success": False,
        "error": {"code": "unknown_command", "message": "Unknown command"},
    }


def _full_script(
    areas: list[dict[str, Any]] | None = None,
    devices: list[dict[str, Any]] | None = None,
    entities: list[dict[str, Any]] | None = None,
    floors: list[dict[str, Any]] | None = None,
    states: list[dict[str, Any]] | None = None,
    expose_new: bool = True,
    floors_supported: bool = True,
    states_supported: bool = True,
    expose_new_supported: bool = True,
    aliases_by_entity_id: dict[str, list[str]] | None = None,
    get_entries_supported: bool = True,
    explicitly_exposed_legacy_ids: set[str] | None = None,
    expose_entity_list_supported: bool = True,
) -> list[dict[str, Any]]:
    """Build the full WS script fetch_ha_vocabulary() sends: areas,
    devices, entities, floors, get_states, expose_new_entities/get, (only
    if legacy/non-registry entities exist) homeassistant/expose_entity/list,
    and (only if at least one registry entity is exposed)
    entity_registry/get_entries for those exposed entities' aliases --
    mirroring the exact command order in fetch_ha_vocabulary()."""
    areas = areas or []
    devices = devices or []
    entities = entities or []
    floors = floors if floors is not None else []
    states = states if states is not None else []
    aliases_by_entity_id = aliases_by_entity_id or {}
    explicitly_exposed_legacy_ids = explicitly_exposed_legacy_ids or set()

    script: list[dict[str, Any]] = [{"type": "auth_required"}, {"type": "auth_ok"}]
    ids = itertools.count(1)

    script.append(_command_response(next(ids), areas))
    script.append(_command_response(next(ids), devices))
    script.append(_command_response(next(ids), entities))

    script.append(
        _command_response(next(ids), floors) if floors_supported else _error_response(next(ids))
    )
    script.append(
        _command_response(next(ids), states) if states_supported else _error_response(next(ids))
    )
    effective_expose_new = expose_new if expose_new_supported else True
    script.append(
        _command_response(next(ids), {"expose_new": expose_new})
        if expose_new_supported
        else _error_response(next(ids))
    )

    legacy_ids = legacy_entity_ids(entities, states)
    if legacy_ids:
        if expose_entity_list_supported:
            exposed_entities = {
                entity_id: {"conversation": True} for entity_id in explicitly_exposed_legacy_ids
            }
            script.append(_command_response(next(ids), {"exposed_entities": exposed_entities}))
        else:
            script.append(_error_response(next(ids)))

    registry_exposed_ids = compute_exposed_entity_ids(entities, states, effective_expose_new)

    if registry_exposed_ids:
        if get_entries_supported:
            result = {
                entity_id: {"aliases": aliases_by_entity_id.get(entity_id, [])}
                for entity_id in sorted(registry_exposed_ids)
            }
            script.append(_command_response(next(ids), result))
        else:
            script.append(_error_response(next(ids)))

    return script


def _fake_websockets_module(
    fake_ws: _FakeWebSocket, connect_calls: list[dict[str, Any]] | None = None
) -> Any:
    class _FakeWebsocketsModule:
        def connect(
            self,
            url: str,
            open_timeout: float | None = None,
            max_size: int | None = None,
        ) -> _FakeWebSocket:
            if connect_calls is not None:
                connect_calls.append(
                    {"url": url, "open_timeout": open_timeout, "max_size": max_size}
                )
            return fake_ws

    return _FakeWebsocketsModule()


class TestFetchHaVocabulary:
    @pytest.mark.asyncio
    async def test_no_token_returns_failed_result(self, monkeypatch):
        monkeypatch.delenv("SUPERVISOR_TOKEN", raising=False)
        result = await fetch_ha_vocabulary(token=None)
        assert result == HaVocabularyResult(success=False, terms=[])

    @pytest.mark.asyncio
    async def test_websockets_missing_returns_failed_result(self, monkeypatch):
        import app.ha_vocabulary as mod

        monkeypatch.setattr(mod, "websockets", None)
        result = await fetch_ha_vocabulary(token="abc")
        assert result == HaVocabularyResult(success=False, terms=[])

    @pytest.mark.asyncio
    async def test_successful_fetch_returns_merged_terms_for_exposed_entities(self, monkeypatch):
        import app.ha_vocabulary as mod

        areas = [
            {"area_id": "living_room", "name": "Wohnzimmer", "aliases": [], "floor_id": "ground"}
        ]
        devices: list[dict[str, Any]] = []
        entities = [
            {
                "entity_id": "light.wohnzimmer",
                "area_id": "living_room",
                "device_id": None,
                "name": "Deckenlampe",
                "original_name": None,
                "entity_category": None,
                "hidden_by": None,
            }
        ]
        floors = [{"floor_id": "ground", "name": "Erdgeschoss"}]
        fake_ws = _FakeWebSocket(_full_script(areas, devices, entities, floors))
        monkeypatch.setattr(mod, "websockets", _fake_websockets_module(fake_ws))

        result = await fetch_ha_vocabulary(token="abc")

        assert result.success is True
        assert result.exposed_entity_count == 1
        for expected in ("Wohnzimmer", "Erdgeschoss", "Deckenlampe", "Wohnzimmer Deckenlampe"):
            assert expected in result.terms

    @pytest.mark.asyncio
    async def test_non_exposed_entity_produces_no_vocabulary(self, monkeypatch):
        import app.ha_vocabulary as mod

        entities = [
            {
                "entity_id": "sensor.router_cpu_temperature",
                "area_id": None,
                "device_id": None,
                "name": "CPU Temperatur",
                "original_name": None,
                "entity_category": None,
                "hidden_by": None,
            }
        ]
        states = [
            {
                "entity_id": "sensor.router_cpu_temperature",
                "attributes": {"device_class": "voltage"},
            }
        ]
        fake_ws = _FakeWebSocket(_full_script(entities=entities, states=states))
        monkeypatch.setattr(mod, "websockets", _fake_websockets_module(fake_ws))

        result = await fetch_ha_vocabulary(token="abc")

        assert result.success is True
        assert result.exposed_entity_count == 0
        assert result.terms == []

    @pytest.mark.asyncio
    async def test_realistic_wohnzimmer_rolllade_scenario(self, monkeypatch):
        """The exact scenario from the task spec: Area "Wohnzimmer", Device
        "Rollladenaktor", Entity cover.wohnzimmer_rolllade with friendly
        name "Rolllade" and alias "Rollo", exposed to Assist -> expects
        Wohnzimmer, Rolllade, Rollo, Wohnzimmer Rolllade, Wohnzimmer Rollo.
        A second, non-exposed sensor.router_cpu_temperature must produce
        no vocabulary at all."""
        import app.ha_vocabulary as mod

        areas = [{"area_id": "living_room", "name": "Wohnzimmer", "aliases": []}]
        devices = [
            {
                "id": "dev_rolllade",
                "area_id": "living_room",
                "name": "Rollladenaktor",
                "name_by_user": None,
            }
        ]
        entities = [
            {
                "entity_id": "cover.wohnzimmer_rolllade",
                "area_id": None,
                "device_id": "dev_rolllade",
                "name": "Rolllade",
                "original_name": None,
                "entity_category": None,
                "hidden_by": None,
            },
            {
                "entity_id": "sensor.router_cpu_temperature",
                "area_id": None,
                "device_id": None,
                "name": "CPU Temperatur",
                "original_name": None,
                "entity_category": None,
                "hidden_by": None,
            },
        ]
        states = [
            {
                "entity_id": "sensor.router_cpu_temperature",
                "attributes": {"device_class": "voltage"},
            }
        ]
        aliases_by_entity_id = {"cover.wohnzimmer_rolllade": ["Rollo"]}
        fake_ws = _FakeWebSocket(
            _full_script(
                areas, devices, entities, states=states, aliases_by_entity_id=aliases_by_entity_id
            )
        )
        monkeypatch.setattr(mod, "websockets", _fake_websockets_module(fake_ws))

        result = await fetch_ha_vocabulary(token="abc")

        assert result.success is True
        assert result.exposed_entity_count == 1
        for expected in (
            "Wohnzimmer",
            "Rolllade",
            "Rollo",
            "Wohnzimmer Rolllade",
            "Wohnzimmer Rollo",
        ):
            assert expected in result.terms
        assert "CPU Temperatur" not in result.terms
        assert not any("router" in t.casefold() for t in result.terms)

    @pytest.mark.asyncio
    async def test_successful_fetch_with_no_registry_entries_is_success_not_failure(
        self, monkeypatch
    ):
        """A legitimately empty HA (no areas/devices/entities) must be
        distinguishable from a failed fetch -- see HaVocabularyResult."""
        import app.ha_vocabulary as mod

        fake_ws = _FakeWebSocket(_full_script())
        monkeypatch.setattr(mod, "websockets", _fake_websockets_module(fake_ws))

        result = await fetch_ha_vocabulary(token="abc")

        assert result == HaVocabularyResult(success=True, terms=[], exposed_entity_count=0)

    @pytest.mark.asyncio
    async def test_auth_failure_returns_failed_result(self, monkeypatch):
        import app.ha_vocabulary as mod

        script = [{"type": "auth_required"}, {"type": "auth_invalid"}]
        fake_ws = _FakeWebSocket(script)
        monkeypatch.setattr(mod, "websockets", _fake_websockets_module(fake_ws))

        result = await fetch_ha_vocabulary(token="abc")
        assert result == HaVocabularyResult(success=False, terms=[])

    @pytest.mark.asyncio
    async def test_connection_error_returns_failed_result_not_raises(self, monkeypatch):
        import app.ha_vocabulary as mod

        class _FailingWebsocketsModule:
            def connect(self, url: str, open_timeout: float | None = None) -> _FakeWebSocket:
                raise OSError("connection refused")

        monkeypatch.setattr(mod, "websockets", _FailingWebsocketsModule())

        result = await fetch_ha_vocabulary(token="abc")
        assert result == HaVocabularyResult(success=False, terms=[])

    @pytest.mark.asyncio
    async def test_missing_floor_registry_command_is_tolerated(self, monkeypatch):
        """Older HA cores may not support config/floor_registry/list."""
        import app.ha_vocabulary as mod

        areas = [{"area_id": "living_room", "name": "Wohnzimmer", "aliases": []}]
        entities = [
            {
                "entity_id": "light.wohnzimmer",
                "area_id": "living_room",
                "device_id": None,
                "name": "Deckenlampe",
                "original_name": None,
                "entity_category": None,
                "hidden_by": None,
            }
        ]
        fake_ws = _FakeWebSocket(
            _full_script(areas=areas, entities=entities, floors_supported=False)
        )
        monkeypatch.setattr(mod, "websockets", _fake_websockets_module(fake_ws))

        result = await fetch_ha_vocabulary(token="abc")

        assert result.success is True
        assert "Wohnzimmer" in result.terms

    @pytest.mark.asyncio
    async def test_missing_get_states_command_is_tolerated(self, monkeypatch):
        """Without get_states, sensor/binary_sensor default-exposure device
        classes can't be resolved, but domain-only entities (e.g. light)
        are unaffected."""
        import app.ha_vocabulary as mod

        entities = [
            {
                "entity_id": "light.wohnzimmer",
                "area_id": None,
                "device_id": None,
                "name": "Deckenlampe",
                "original_name": None,
                "entity_category": None,
                "hidden_by": None,
            }
        ]
        fake_ws = _FakeWebSocket(_full_script(entities=entities, states_supported=False))
        monkeypatch.setattr(mod, "websockets", _fake_websockets_module(fake_ws))

        result = await fetch_ha_vocabulary(token="abc")

        assert result.success is True
        assert result.terms == ["Deckenlampe"]

    @pytest.mark.asyncio
    async def test_missing_expose_new_entities_command_defaults_to_enabled(self, monkeypatch):
        import app.ha_vocabulary as mod

        entities = [
            {
                "entity_id": "light.wohnzimmer",
                "area_id": None,
                "device_id": None,
                "name": "Deckenlampe",
                "original_name": None,
                "entity_category": None,
                "hidden_by": None,
            }
        ]
        fake_ws = _FakeWebSocket(_full_script(entities=entities, expose_new_supported=False))
        monkeypatch.setattr(mod, "websockets", _fake_websockets_module(fake_ws))

        result = await fetch_ha_vocabulary(token="abc")

        assert result.success is True
        assert result.terms == ["Deckenlampe"]

    @pytest.mark.asyncio
    async def test_expose_new_disabled_excludes_default_exposed_entities(self, monkeypatch):
        import app.ha_vocabulary as mod

        entities = [
            {
                "entity_id": "light.wohnzimmer",
                "area_id": None,
                "device_id": None,
                "name": "Deckenlampe",
                "original_name": None,
                "entity_category": None,
                "hidden_by": None,
            }
        ]
        fake_ws = _FakeWebSocket(_full_script(entities=entities, expose_new=False))
        monkeypatch.setattr(mod, "websockets", _fake_websockets_module(fake_ws))

        result = await fetch_ha_vocabulary(token="abc")

        assert result.success is True
        assert result.terms == []

    @pytest.mark.asyncio
    async def test_missing_get_entries_command_degrades_to_no_aliases(self, monkeypatch):
        """Aliases are a nice-to-have, fetched in a second round trip --
        their absence must not fail the whole vocabulary fetch."""
        import app.ha_vocabulary as mod

        entities = [
            {
                "entity_id": "cover.wohnzimmer_rolllade",
                "area_id": None,
                "device_id": None,
                "name": "Rolllade",
                "original_name": None,
                "entity_category": None,
                "hidden_by": None,
            }
        ]
        fake_ws = _FakeWebSocket(_full_script(entities=entities, get_entries_supported=False))
        monkeypatch.setattr(mod, "websockets", _fake_websockets_module(fake_ws))

        result = await fetch_ha_vocabulary(token="abc")

        assert result.success is True
        assert result.terms == ["Rolllade"]

    @pytest.mark.asyncio
    async def test_empty_assist_exposure_list_is_success_with_no_terms(self, monkeypatch):
        """No entity is exposed at all (e.g. expose_new disabled and no
        explicit overrides) -- a legitimate, successful empty result."""
        import app.ha_vocabulary as mod

        entities = [
            {
                "entity_id": "sensor.router_cpu_temperature",
                "area_id": None,
                "device_id": None,
                "name": "CPU Temperatur",
                "original_name": None,
                "entity_category": None,
                "hidden_by": None,
            }
        ]
        fake_ws = _FakeWebSocket(_full_script(entities=entities, expose_new=False))
        monkeypatch.setattr(mod, "websockets", _fake_websockets_module(fake_ws))

        result = await fetch_ha_vocabulary(token="abc")

        assert result == HaVocabularyResult(success=True, terms=[], exposed_entity_count=0)

    @pytest.mark.asyncio
    async def test_many_exposed_entities_all_produce_vocabulary(self, monkeypatch):
        """A large number of Assist-exposed entities must all be reflected
        -- no arbitrary invented cap on the underlying entity vocabulary
        itself (only the area+entity combination list is capped, see
        TestAreaEntityCombinationTerms.test_combination_terms_capped_at_maximum)."""
        import app.ha_vocabulary as mod

        entities = [
            {
                "entity_id": f"light.lampe_{i}",
                "area_id": None,
                "device_id": None,
                "name": f"Lampe {i}",
                "original_name": None,
                "entity_category": None,
                "hidden_by": None,
            }
            for i in range(200)
        ]
        fake_ws = _FakeWebSocket(_full_script(entities=entities))
        monkeypatch.setattr(mod, "websockets", _fake_websockets_module(fake_ws))

        result = await fetch_ha_vocabulary(token="abc")

        assert result.success is True
        assert result.exposed_entity_count == 200
        assert "Lampe 0" in result.terms
        assert "Lampe 199" in result.terms


class TestLegacyEntityIds:
    """Entities with no entity registry entry (get_states-only) -- see
    ha_vocabulary.py's module docstring, "Legacy (non-registry) entities"."""

    def test_state_entity_absent_from_registry_is_legacy(self):
        entities = [{"entity_id": "light.registered"}]
        states = [{"entity_id": "light.registered"}, {"entity_id": "switch.legacy_only"}]
        assert legacy_entity_ids(entities, states) == {"switch.legacy_only"}

    def test_state_entity_present_in_registry_is_not_legacy(self):
        entities = [{"entity_id": "light.registered"}]
        states = [{"entity_id": "light.registered"}]
        assert legacy_entity_ids(entities, states) == set()

    def test_no_states_means_no_legacy_entities(self):
        entities = [{"entity_id": "light.registered"}]
        assert legacy_entity_ids(entities, None) == set()

    def test_empty_registry_all_states_are_legacy(self):
        states = [{"entity_id": "switch.old_kaffeemaschine"}]
        assert legacy_entity_ids([], states) == {"switch.old_kaffeemaschine"}


class TestComputeLegacyExposedEntityIds:
    """Deliberately CONSERVATIVE (see ha_vocabulary.py's module docstring
    and this function's own docstring, re-verified against the current
    home-assistant/core dev branch source on 2026-09-18): a legacy entity
    is included ONLY when a positive, observed exposure signal exists.
    There is no read-only API that can tell an explicit ``should_expose=
    False`` apart from "never evaluated" for a legacy entity, so this
    module never guesses -- unlike registry entities, it never falls back
    to the default-exposure rule for them.
    """

    def test_positive_signal_included(self):
        legacy_ids = {"switch.alte_kaffeemaschine"}
        result = compute_legacy_exposed_entity_ids(legacy_ids, legacy_ids)
        assert result == legacy_ids

    def test_no_signal_at_all_is_excluded(self):
        """switch is a DEFAULT_EXPOSED_DOMAINS domain -- a registry entity
        in this domain would default-expose, but a legacy entity with no
        observed positive signal must NOT, since that would risk
        re-including an entity explicitly hidden from Assist."""
        legacy_ids = {"switch.alte_kaffeemaschine"}
        result = compute_legacy_exposed_entity_ids(legacy_ids, set())
        assert result == set()

    def test_sensor_domain_with_positive_signal_included(self):
        legacy_ids = {"sensor.legacy_temp"}
        result = compute_legacy_exposed_entity_ids(legacy_ids, legacy_ids)
        assert result == legacy_ids

    def test_sensor_domain_without_signal_excluded(self):
        legacy_ids = {"sensor.legacy_router_cpu"}
        result = compute_legacy_exposed_entity_ids(legacy_ids, set())
        assert result == set()

    def test_positive_signal_outside_legacy_ids_is_ignored(self):
        """explicitly_exposed_legacy_ids may (structurally) contain ids
        outside legacy_ids -- only the intersection counts."""
        legacy_ids = {"switch.a"}
        result = compute_legacy_exposed_entity_ids(legacy_ids, {"switch.b"})
        assert result == set()

    def test_empty_legacy_ids_produces_empty_result(self):
        assert compute_legacy_exposed_entity_ids(set(), {"switch.a"}) == set()


class TestBuildLegacyPseudoEntities:
    def test_uses_friendly_name_from_state(self):
        states = [
            {
                "entity_id": "switch.alte_kaffeemaschine",
                "attributes": {"friendly_name": "Kaffeemaschine"},
            }
        ]
        pseudo = build_legacy_pseudo_entities({"switch.alte_kaffeemaschine"}, states)
        assert pseudo == [
            {
                "entity_id": "switch.alte_kaffeemaschine",
                "name": "Kaffeemaschine",
                "original_name": None,
                "aliases": [],
                "area_id": None,
                "device_id": None,
            }
        ]

    def test_no_area_or_device_ever_assigned(self):
        states = [{"entity_id": "switch.x", "attributes": {"friendly_name": "X"}}]
        pseudo = build_legacy_pseudo_entities({"switch.x"}, states)
        assert pseudo[0]["area_id"] is None
        assert pseudo[0]["device_id"] is None

    def test_missing_friendly_name_produces_none_for_entity_id_fallback(self):
        states = [{"entity_id": "switch.x", "attributes": {}}]
        pseudo = build_legacy_pseudo_entities({"switch.x"}, states)
        assert pseudo[0]["name"] is None


class TestLegacyEntityEndToEndVocabulary:
    """Legacy entities flow through extract_vocabulary_terms() exactly like
    registry entities (name/alias/entity-id-fallback resolution), but never
    contribute Area combination terms."""

    def test_legacy_entity_with_friendly_name_produces_term(self):
        entities: list[dict[str, Any]] = []
        pseudo = build_legacy_pseudo_entities(
            {"switch.alte_kaffeemaschine"},
            [
                {
                    "entity_id": "switch.alte_kaffeemaschine",
                    "attributes": {"friendly_name": "Kaffeemaschine"},
                }
            ],
        )
        terms = extract_vocabulary_terms(
            [], [], entities + pseudo, exposed_entity_ids={"switch.alte_kaffeemaschine"}
        )
        assert terms == ["Kaffeemaschine"]

    def test_legacy_entity_without_friendly_name_falls_back_to_entity_id(self):
        pseudo = build_legacy_pseudo_entities(
            {"switch.alte_kaffeemaschine"},
            [{"entity_id": "switch.alte_kaffeemaschine", "attributes": {}}],
        )
        terms = extract_vocabulary_terms(
            [], [], pseudo, exposed_entity_ids={"switch.alte_kaffeemaschine"}
        )
        assert terms == ["Alte Kaffeemaschine"]
        assert not any("switch" in t.casefold() for t in terms)

    def test_legacy_entity_produces_no_area_combination(self):
        areas = [{"area_id": "kitchen", "name": "Küche", "aliases": []}]
        pseudo = build_legacy_pseudo_entities(
            {"switch.alte_kaffeemaschine"},
            [
                {
                    "entity_id": "switch.alte_kaffeemaschine",
                    "attributes": {"friendly_name": "Kaffeemaschine"},
                }
            ],
        )
        terms = extract_vocabulary_terms(
            areas, [], pseudo, exposed_entity_ids={"switch.alte_kaffeemaschine"}
        )
        assert "Küche Kaffeemaschine" not in terms
        assert "Küche" not in terms  # area unreachable from any exposed entity


class TestFetchHaVocabularyLegacyEntities:
    """Scenarios A-F from the v0.2.3 task spec, reflecting the real
    information the read-only API can actually provide (see
    compute_legacy_exposed_entity_ids()'s docstring) -- never a simulated
    signal the real API cannot give."""

    @pytest.mark.asyncio
    async def test_a_registry_entity_explicit_false_excluded(self, monkeypatch):
        """A: registry entity switch.kaffeemaschine, explicit false ->
        NOT in vocabulary. Registry entities are unaffected by the legacy
        conservatism change -- their own cached options are authoritative."""
        import app.ha_vocabulary as mod

        entities = [
            {
                "entity_id": "switch.kaffeemaschine",
                "area_id": None,
                "device_id": None,
                "name": "Kaffeemaschine",
                "original_name": None,
                "entity_category": None,
                "hidden_by": None,
                "options": {CONVERSATION_ASSISTANT: {"should_expose": False}},
            }
        ]
        fake_ws = _FakeWebSocket(_full_script(entities=entities))
        monkeypatch.setattr(mod, "websockets", _fake_websockets_module(fake_ws))

        result = await fetch_ha_vocabulary(token="abc")

        assert result.success is True
        assert result.terms == []

    @pytest.mark.asyncio
    async def test_b_legacy_entity_positive_exposure_included(self, monkeypatch):
        """B: legacy entity switch.alte_kaffeemaschine with a positive
        explicit conversation exposure signal -> IN vocabulary."""
        import app.ha_vocabulary as mod

        entities: list[dict[str, Any]] = []
        states = [
            {
                "entity_id": "switch.alte_kaffeemaschine",
                "attributes": {"friendly_name": "Kaffeemaschine"},
            }
        ]
        fake_ws = _FakeWebSocket(
            _full_script(
                entities=entities,
                states=states,
                explicitly_exposed_legacy_ids={"switch.alte_kaffeemaschine"},
            )
        )
        monkeypatch.setattr(mod, "websockets", _fake_websockets_module(fake_ws))

        result = await fetch_ha_vocabulary(token="abc")

        assert result.success is True
        assert result.exposed_entity_count == 1
        assert "Kaffeemaschine" in result.terms

    @pytest.mark.asyncio
    async def test_c_legacy_entity_no_positive_proof_excluded(self, monkeypatch):
        """C: legacy entity switch.alte_kaffeemaschine, no positive
        exposure proof at all -> NOT in vocabulary under the conservative
        strategy (even though switch is a default-exposed domain for
        registry entities)."""
        import app.ha_vocabulary as mod

        entities: list[dict[str, Any]] = []
        states = [
            {
                "entity_id": "switch.alte_kaffeemaschine",
                "attributes": {"friendly_name": "Kaffeemaschine"},
            }
        ]
        fake_ws = _FakeWebSocket(_full_script(entities=entities, states=states))
        monkeypatch.setattr(mod, "websockets", _fake_websockets_module(fake_ws))

        result = await fetch_ha_vocabulary(token="abc")

        assert result.success is True
        assert result.exposed_entity_count == 0
        assert result.terms == []

    @pytest.mark.asyncio
    async def test_d_legacy_sensor_positive_exposure_included(self, monkeypatch):
        """D: legacy entity sensor.temp with a positive explicit exposure
        -> IN vocabulary."""
        import app.ha_vocabulary as mod

        entities: list[dict[str, Any]] = []
        states = [{"entity_id": "sensor.temp", "attributes": {"friendly_name": "Temperatur"}}]
        fake_ws = _FakeWebSocket(
            _full_script(
                entities=entities, states=states, explicitly_exposed_legacy_ids={"sensor.temp"}
            )
        )
        monkeypatch.setattr(mod, "websockets", _fake_websockets_module(fake_ws))

        result = await fetch_ha_vocabulary(token="abc")

        assert result.success is True
        assert "Temperatur" in result.terms

    @pytest.mark.asyncio
    async def test_e_legacy_entity_uses_friendly_name(self, monkeypatch):
        """E: no registry entry, friendly_name present -> friendly_name used."""
        import app.ha_vocabulary as mod

        entities: list[dict[str, Any]] = []
        states = [
            {
                "entity_id": "switch.alte_kaffeemaschine",
                "attributes": {"friendly_name": "Kaffeemaschine"},
            }
        ]
        fake_ws = _FakeWebSocket(
            _full_script(
                entities=entities,
                states=states,
                explicitly_exposed_legacy_ids={"switch.alte_kaffeemaschine"},
            )
        )
        monkeypatch.setattr(mod, "websockets", _fake_websockets_module(fake_ws))

        result = await fetch_ha_vocabulary(token="abc")

        assert result.terms == ["Kaffeemaschine"]

    @pytest.mark.asyncio
    async def test_legacy_entity_without_friendly_name_falls_back_to_entity_id(self, monkeypatch):
        import app.ha_vocabulary as mod

        entities: list[dict[str, Any]] = []
        states = [{"entity_id": "switch.alte_kaffeemaschine", "attributes": {}}]
        fake_ws = _FakeWebSocket(
            _full_script(
                entities=entities,
                states=states,
                explicitly_exposed_legacy_ids={"switch.alte_kaffeemaschine"},
            )
        )
        monkeypatch.setattr(mod, "websockets", _fake_websockets_module(fake_ws))

        result = await fetch_ha_vocabulary(token="abc")

        assert result.terms == ["Alte Kaffeemaschine"]

    @pytest.mark.asyncio
    async def test_no_duplicate_term_when_entity_id_appears_in_both_registry_and_states(
        self, monkeypatch
    ):
        """A registry entity's entity_id also appears in get_states (as it
        always does for a live entity) -- it must be treated purely as a
        registry entity, never double-counted as legacy too."""
        import app.ha_vocabulary as mod

        entities = [
            {
                "entity_id": "light.wohnzimmer",
                "area_id": None,
                "device_id": None,
                "name": "Deckenlampe",
                "original_name": None,
                "entity_category": None,
                "hidden_by": None,
            }
        ]
        states = [
            {
                "entity_id": "light.wohnzimmer",
                "attributes": {"friendly_name": "Deckenlampe (state)"},
            }
        ]
        fake_ws = _FakeWebSocket(_full_script(entities=entities, states=states))
        monkeypatch.setattr(mod, "websockets", _fake_websockets_module(fake_ws))

        result = await fetch_ha_vocabulary(token="abc")

        assert result.success is True
        assert result.exposed_entity_count == 1
        assert result.terms.count("Deckenlampe") == 1
        assert "Deckenlampe (state)" not in result.terms

    @pytest.mark.asyncio
    async def test_missing_expose_entity_list_command_degrades_gracefully(self, monkeypatch):
        """Older Home Assistant cores without homeassistant/expose_entity/
        list must not fail the whole vocabulary fetch -- with no way at all
        to observe a positive legacy exposure signal, legacy entities are
        conservatively excluded (never guessed via the default rule)."""
        import app.ha_vocabulary as mod

        entities: list[dict[str, Any]] = []
        states = [
            {
                "entity_id": "switch.alte_kaffeemaschine",
                "attributes": {"friendly_name": "Kaffeemaschine"},
            }
        ]
        fake_ws = _FakeWebSocket(
            _full_script(entities=entities, states=states, expose_entity_list_supported=False)
        )
        monkeypatch.setattr(mod, "websockets", _fake_websockets_module(fake_ws))

        result = await fetch_ha_vocabulary(token="abc")

        assert result.success is True
        assert result.terms == []

    @pytest.mark.asyncio
    async def test_no_legacy_entities_never_sends_expose_entity_list_command(self, monkeypatch):
        """When every get_states entity_id is already in the registry,
        homeassistant/expose_entity/list must never be sent at all."""
        import app.ha_vocabulary as mod

        entities = [
            {
                "entity_id": "light.wohnzimmer",
                "area_id": None,
                "device_id": None,
                "name": "Deckenlampe",
                "original_name": None,
                "entity_category": None,
                "hidden_by": None,
            }
        ]
        states = [{"entity_id": "light.wohnzimmer", "attributes": {}}]
        fake_ws = _FakeWebSocket(_full_script(entities=entities, states=states))
        monkeypatch.setattr(mod, "websockets", _fake_websockets_module(fake_ws))

        await fetch_ha_vocabulary(token="abc")

        sent_types = [msg.get("type") for msg in fake_ws.sent]
        assert "homeassistant/expose_entity/list" not in sent_types


class TestLargeWebSocketPayloads:
    """v0.2.4: a real, larger Home Assistant installation hit
    ``ConnectionClosedError: sent 1009 (message too big); frame exceeds
    limit of 1048576 bytes`` -- websockets==17.1's own default `max_size`
    for `websockets.connect()` is 1 MiB, and a real `get_states`/
    `config/entity_registry/list` response can exceed that. This produced
    "effective keyterms=0" despite `use_ha_vocabulary=true`. See
    MAX_WS_MESSAGE_SIZE's own docstring in app/ha_vocabulary.py for why a
    large, explicit limit (not max_size=None) was chosen."""

    def test_connect_uses_a_large_explicit_max_size(self, monkeypatch):
        """The real fix: websockets.connect() must be called with a
        max_size well above the library's 1 MiB default."""
        import asyncio

        import app.ha_vocabulary as mod

        fake_ws = _FakeWebSocket(_full_script())
        connect_calls: list[dict[str, Any]] = []
        monkeypatch.setattr(mod, "websockets", _fake_websockets_module(fake_ws, connect_calls))

        asyncio.run(fetch_ha_vocabulary(token="abc"))

        assert len(connect_calls) == 1
        assert connect_calls[0]["max_size"] == MAX_WS_MESSAGE_SIZE
        # websockets' own default is 1 MiB (1_048_576) -- the whole point
        # of this fix is that our configured limit is well above it.
        assert MAX_WS_MESSAGE_SIZE > 1_048_576

    @pytest.mark.asyncio
    async def test_get_states_response_larger_than_1mib_is_handled(self, monkeypatch):
        """Simulates a real large installation: a get_states response
        alone exceeds 1 MiB. The connection must not break, and the
        exposed entity's vocabulary/exposure must still come out correct
        even with thousands of unrelated large states mixed in."""
        import app.ha_vocabulary as mod

        entities = [
            {
                "entity_id": "light.wohnzimmer",
                "area_id": None,
                "device_id": None,
                "name": "Deckenlampe",
                "original_name": None,
                "entity_category": None,
                "hidden_by": None,
                "options": {CONVERSATION_ASSISTANT: {"should_expose": True}},
            }
        ]
        # Pad with enough large sensor states to exceed 1 MiB once
        # JSON-serialized -- a realistic shape for a large real installation
        # (long attribute strings, e.g. weather/history/forecast entities).
        states = [{"entity_id": "light.wohnzimmer", "attributes": {}}]
        padding_attributes = {"forecast": "x" * 2000, "history": list(range(200))}
        for i in range(1000):
            states.append(
                {
                    "entity_id": f"sensor.padding_{i}",
                    "attributes": dict(padding_attributes),
                }
            )
        serialized_size = len(json.dumps(states).encode("utf-8"))
        assert serialized_size > 1_048_576, "test fixture must actually exceed 1 MiB"

        fake_ws = _FakeWebSocket(_full_script(entities=entities, states=states))
        monkeypatch.setattr(mod, "websockets", _fake_websockets_module(fake_ws))

        result = await fetch_ha_vocabulary(token="abc")

        assert result.success is True
        assert result.exposed_entity_count == 1
        assert "Deckenlampe" in result.terms

    @pytest.mark.asyncio
    async def test_large_entity_registry_response_is_handled(self, monkeypatch):
        """Same scenario but for a large config/entity_registry/list
        response (many registry entities), as real large installations
        also reported for registry responses, not just get_states."""
        import app.ha_vocabulary as mod

        entities = [
            {
                "entity_id": "light.wohnzimmer",
                "area_id": None,
                "device_id": None,
                "name": "Deckenlampe",
                "original_name": None,
                "entity_category": None,
                "hidden_by": None,
                "options": {CONVERSATION_ASSISTANT: {"should_expose": True}},
            }
        ]
        long_name_suffix = "x" * 500
        for i in range(3000):
            entities.append(
                {
                    "entity_id": f"sensor.padding_{i}",
                    "area_id": None,
                    "device_id": None,
                    "name": f"Padding Sensor {i} {long_name_suffix}",
                    "original_name": None,
                    "entity_category": None,
                    "hidden_by": "user",
                    "options": {},
                }
            )
        serialized_size = len(json.dumps(entities).encode("utf-8"))
        assert serialized_size > 1_048_576, "test fixture must actually exceed 1 MiB"

        fake_ws = _FakeWebSocket(_full_script(entities=entities))
        monkeypatch.setattr(mod, "websockets", _fake_websockets_module(fake_ws))

        result = await fetch_ha_vocabulary(token="abc")

        assert result.success is True
        assert result.exposed_entity_count == 1
        assert "Deckenlampe" in result.terms
