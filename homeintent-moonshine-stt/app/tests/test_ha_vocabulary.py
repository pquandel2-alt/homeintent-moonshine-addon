"""Tests for read-only Home Assistant vocabulary collection.

Covers the pure normalization helpers directly, and fetch_ha_vocabulary()
against a fake WebSocket connection standing in for Supervisor's proxied
Home Assistant Core WebSocket API -- no real network/Supervisor is reached.
"""

import json
from typing import Any

import pytest

from app.ha_vocabulary import (
    HaVocabularyResult,
    clean_term,
    dedupe_preserve_order,
    extract_vocabulary_terms,
    fetch_ha_vocabulary,
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

    def test_entity_prefers_name_over_original_name(self):
        entities = [{"name": "Deckenlampe", "original_name": "light.wohnzimmer_2"}]
        assert extract_vocabulary_terms([], [], entities) == ["Deckenlampe"]

    def test_entity_falls_back_to_original_name(self):
        entities = [{"name": None, "original_name": "Deckenlampe Flur"}]
        assert extract_vocabulary_terms([], [], entities) == ["Deckenlampe Flur"]

    def test_never_derives_term_from_raw_entity_id(self):
        entities = [
            {"entity_id": "light.wohnzimmer_deckenlampe", "name": None, "original_name": None}
        ]
        assert extract_vocabulary_terms([], [], entities) == []

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

    def test_realistic_multi_room_registry(self):
        """Wohnzimmer: Rolllade, Deckenlampe, Fenster / Küche: Fenster,
        Deckenlampe / Schlafzimmer: Heizung / Garage: Licht."""
        areas = [
            {"area_id": "living_room", "name": "Wohnzimmer", "aliases": []},
            {"area_id": "kitchen", "name": "Küche", "aliases": []},
            {"area_id": "bedroom", "name": "Schlafzimmer", "aliases": []},
            {"area_id": "garage", "name": "Garage", "aliases": []},
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


def _command_response(command_id: int, result: list[dict[str, Any]]) -> dict[str, Any]:
    return {"id": command_id, "type": "result", "success": True, "result": result}


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
    async def test_successful_fetch_returns_merged_terms(self, monkeypatch):
        import app.ha_vocabulary as mod

        script = [
            {"type": "auth_required"},
            {"type": "auth_ok"},
            _command_response(1, [{"name": "Wohnzimmer", "aliases": []}]),
            _command_response(2, [{"name": "Küche", "name_by_user": None}]),
            _command_response(3, [{"name": "Deckenlampe", "original_name": None}]),
            _command_response(4, [{"name": "Erdgeschoss"}]),
        ]
        fake_ws = _FakeWebSocket(script)

        class _FakeWebsocketsModule:
            def connect(self, url: str, open_timeout: float | None = None) -> _FakeWebSocket:
                return fake_ws

        monkeypatch.setattr(mod, "websockets", _FakeWebsocketsModule())

        result = await fetch_ha_vocabulary(token="abc")

        assert result.success is True
        assert result.terms == ["Wohnzimmer", "Erdgeschoss", "Küche", "Deckenlampe"]

    @pytest.mark.asyncio
    async def test_successful_fetch_with_no_registry_entries_is_success_not_failure(
        self, monkeypatch
    ):
        """A legitimately empty HA (no areas/devices/entities) must be
        distinguishable from a failed fetch -- see HaVocabularyResult."""
        import app.ha_vocabulary as mod

        script = [
            {"type": "auth_required"},
            {"type": "auth_ok"},
            _command_response(1, []),
            _command_response(2, []),
            _command_response(3, []),
            _command_response(4, []),
        ]
        fake_ws = _FakeWebSocket(script)

        class _FakeWebsocketsModule:
            def connect(self, url: str, open_timeout: float | None = None) -> _FakeWebSocket:
                return fake_ws

        monkeypatch.setattr(mod, "websockets", _FakeWebsocketsModule())

        result = await fetch_ha_vocabulary(token="abc")

        assert result == HaVocabularyResult(success=True, terms=[])

    @pytest.mark.asyncio
    async def test_auth_failure_returns_failed_result(self, monkeypatch):
        import app.ha_vocabulary as mod

        script = [{"type": "auth_required"}, {"type": "auth_invalid"}]
        fake_ws = _FakeWebSocket(script)

        class _FakeWebsocketsModule:
            def connect(self, url: str, open_timeout: float | None = None) -> _FakeWebSocket:
                return fake_ws

        monkeypatch.setattr(mod, "websockets", _FakeWebsocketsModule())

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

        script = [
            {"type": "auth_required"},
            {"type": "auth_ok"},
            _command_response(1, [{"name": "Wohnzimmer", "aliases": []}]),
            _command_response(2, []),
            _command_response(3, []),
            {
                "id": 4,
                "type": "result",
                "success": False,
                "error": {"code": "unknown_command", "message": "Unknown command"},
            },
        ]
        fake_ws = _FakeWebSocket(script)

        class _FakeWebsocketsModule:
            def connect(self, url: str, open_timeout: float | None = None) -> _FakeWebSocket:
                return fake_ws

        monkeypatch.setattr(mod, "websockets", _FakeWebsocketsModule())

        result = await fetch_ha_vocabulary(token="abc")

        assert result == HaVocabularyResult(success=True, terms=["Wohnzimmer"])
