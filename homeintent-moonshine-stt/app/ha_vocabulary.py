"""Read-only Home Assistant vocabulary collection for Moonshine keyterm biasing.

Reads Areas, Devices, Entities, and Floors from Home Assistant's registries
via the Supervisor-proxied Core WebSocket API
(``ws://supervisor/core/websocket``, see developers.home-assistant.io) to
build a list of natural-language terms (room names, device names, entity
names) that bias Moonshine's recognition towards the user's actual smart
home vocabulary.

Strictly read-only: only ``config/*_registry/list`` commands are issued.
This module never calls a service, changes a state, or edits an
automation/entity. Requires ``homeassistant_api: true`` in config.yaml,
which grants the add-on the SUPERVISOR_TOKEN used to authenticate here.
"""

import asyncio
import itertools
import json
import logging
import os
from typing import Any

try:
    import websockets
except ImportError:  # pragma: no cover - exercised only if dependency missing
    websockets = None  # type: ignore[assignment]

_LOGGER = logging.getLogger(__name__)

DEFAULT_WS_URL = "ws://supervisor/core/websocket"
CONNECT_TIMEOUT = 10.0
COMMAND_TIMEOUT = 10.0


def clean_term(raw: str | None) -> str | None:
    """Normalize a single vocabulary term.

    Trims leading/trailing whitespace and collapses internal whitespace runs,
    but otherwise preserves casing, umlauts, and hyphens exactly as the user
    named the area/device/entity in Home Assistant -- Moonshine's
    set_keyterms() docstring asks callers to "match the capitalization and
    spelling you want to see in the output".
    """
    if not raw:
        return None
    term = " ".join(raw.split())
    return term or None


def dedupe_preserve_order(terms: list[str]) -> list[str]:
    """Deduplicate case-insensitively while keeping first-seen order/casing."""
    seen: set[str] = set()
    result: list[str] = []
    for term in terms:
        key = term.casefold()
        if key in seen:
            continue
        seen.add(key)
        result.append(term)
    return result


def _area_terms(area: dict[str, Any]) -> list[str]:
    terms = [t for t in (clean_term(area.get("name")),) if t]
    for alias in area.get("aliases") or []:
        alias_term = clean_term(alias)
        if alias_term:
            terms.append(alias_term)
    return terms


def extract_vocabulary_terms(
    areas: list[dict[str, Any]],
    devices: list[dict[str, Any]],
    entities: list[dict[str, Any]],
    floors: list[dict[str, Any]] | None = None,
) -> list[str]:
    """Build a deduplicated term list from raw HA registry list responses.

    Prefers human-assigned names over anything derived from technical IDs:
    area/floor ``name`` (+ area ``aliases``), device ``name_by_user`` (falls
    back to ``name``), and entity ``name`` (falls back to ``original_name``).
    Entities/devices/areas without any usable name are skipped outright --
    this deliberately never falls back to splitting a raw entity_id like
    ``light.wohnzimmer_deckenlampe`` into technical tokens.
    """
    terms: list[str] = []

    for area in areas:
        terms.extend(_area_terms(area))

    for floor in floors or []:
        term = clean_term(floor.get("name"))
        if term:
            terms.append(term)

    for device in devices:
        term = clean_term(device.get("name_by_user") or device.get("name"))
        if term:
            terms.append(term)

    for entity in entities:
        term = clean_term(entity.get("name") or entity.get("original_name"))
        if term:
            terms.append(term)

    return dedupe_preserve_order(terms)


async def _send_command(ws: Any, command_id: int, command_type: str) -> list[dict[str, Any]]:
    await ws.send(json.dumps({"id": command_id, "type": command_type}))
    async with asyncio.timeout(COMMAND_TIMEOUT):
        while True:
            message = json.loads(await ws.recv())
            if message.get("id") != command_id:
                # Not our response (e.g. a stray event); keep waiting.
                continue
            if not message.get("success", False):
                error = message.get("error", {})
                raise RuntimeError(f"{command_type} failed: {error.get('message', error)}")
            result = message.get("result")
            return result if isinstance(result, list) else []


async def fetch_ha_vocabulary(
    token: str | None = None,
    ws_url: str = DEFAULT_WS_URL,
) -> list[str]:
    """Collect vocabulary terms from HA's Area/Device/Entity/Floor registries.

    Never raises: on any failure (missing token, unreachable Supervisor,
    auth failure, HA core down, etc.) this logs a WARNING and returns an
    empty list, so the add-on can start and keep serving plain STT without
    HA vocabulary rather than treating this as a hard dependency.
    """
    token = token or os.environ.get("SUPERVISOR_TOKEN")
    if not token:
        _LOGGER.warning(
            "HA vocabulary unavailable: SUPERVISOR_TOKEN not set, continuing without HA vocabulary"
        )
        return []
    if websockets is None:  # pragma: no cover - dependency is always installed in CI
        _LOGGER.warning(
            "HA vocabulary unavailable: websockets package not installed, "
            "continuing without HA vocabulary"
        )
        return []

    try:
        async with websockets.connect(ws_url, open_timeout=CONNECT_TIMEOUT) as ws:
            hello = json.loads(await ws.recv())
            if hello.get("type") != "auth_required":
                raise RuntimeError(f"Unexpected Home Assistant WS handshake: {hello.get('type')}")

            await ws.send(json.dumps({"type": "auth", "access_token": token}))
            auth_result = json.loads(await ws.recv())
            if auth_result.get("type") != "auth_ok":
                raise RuntimeError("Home Assistant WebSocket authentication failed")

            command_id = itertools.count(1)
            areas = await _send_command(ws, next(command_id), "config/area_registry/list")
            devices = await _send_command(ws, next(command_id), "config/device_registry/list")
            entities = await _send_command(ws, next(command_id), "config/entity_registry/list")
            try:
                floors = await _send_command(ws, next(command_id), "config/floor_registry/list")
            except RuntimeError:
                # Floor registry is a newer HA feature; older cores may
                # not support the command. Optional, not fatal.
                floors = []
    except Exception as err:
        _LOGGER.warning(
            "HA vocabulary unavailable (%s: %s), continuing without HA vocabulary",
            type(err).__name__,
            err,
        )
        return []

    terms = extract_vocabulary_terms(areas, devices, entities, floors)
    _LOGGER.info("Loaded %d Home Assistant vocabulary terms", len(terms))
    return terms
