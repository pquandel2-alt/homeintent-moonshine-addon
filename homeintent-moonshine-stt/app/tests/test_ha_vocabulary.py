"""Tests for read-only Home Assistant vocabulary collection.

Covers the pure normalization helpers directly, and fetch_ha_vocabulary()
against a fake WebSocket connection standing in for Supervisor's proxied
Home Assistant Core WebSocket API -- no real network/Supervisor is reached.
"""

import json
from typing import Any

import pytest

from app.ha_vocabulary import (
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
    async def test_no_token_returns_empty_list(self, monkeypatch):
        monkeypatch.delenv("SUPERVISOR_TOKEN", raising=False)
        assert await fetch_ha_vocabulary(token=None) == []

    @pytest.mark.asyncio
    async def test_websockets_missing_returns_empty_list(self, monkeypatch):
        import app.ha_vocabulary as mod

        monkeypatch.setattr(mod, "websockets", None)
        assert await fetch_ha_vocabulary(token="abc") == []

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

        terms = await fetch_ha_vocabulary(token="abc")

        assert terms == ["Wohnzimmer", "Erdgeschoss", "Küche", "Deckenlampe"]

    @pytest.mark.asyncio
    async def test_auth_failure_returns_empty_list(self, monkeypatch):
        import app.ha_vocabulary as mod

        script = [{"type": "auth_required"}, {"type": "auth_invalid"}]
        fake_ws = _FakeWebSocket(script)

        class _FakeWebsocketsModule:
            def connect(self, url: str, open_timeout: float | None = None) -> _FakeWebSocket:
                return fake_ws

        monkeypatch.setattr(mod, "websockets", _FakeWebsocketsModule())

        assert await fetch_ha_vocabulary(token="abc") == []

    @pytest.mark.asyncio
    async def test_connection_error_returns_empty_list_not_raises(self, monkeypatch):
        import app.ha_vocabulary as mod

        class _FailingWebsocketsModule:
            def connect(self, url: str, open_timeout: float | None = None) -> _FakeWebSocket:
                raise OSError("connection refused")

        monkeypatch.setattr(mod, "websockets", _FailingWebsocketsModule())

        assert await fetch_ha_vocabulary(token="abc") == []

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

        terms = await fetch_ha_vocabulary(token="abc")

        assert terms == ["Wohnzimmer"]
