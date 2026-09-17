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
from dataclasses import dataclass, field
from typing import Any

try:
    import websockets
except ImportError:  # pragma: no cover - exercised only if dependency missing
    websockets = None  # type: ignore[assignment]

_LOGGER = logging.getLogger(__name__)


@dataclass(frozen=True)
class HaVocabularyResult:
    """Outcome of one fetch_ha_vocabulary() call.

    ``success`` distinguishes a fetch that actually talked to Home Assistant
    (even if the registries turned out to be legitimately empty) from one
    that failed before getting a real answer (unreachable Supervisor, auth
    failure, timeout, HA core down, unexpected response, ...). Callers that
    keep a "last-known-good" vocabulary across periodic refreshes need this
    distinction: a real empty result should replace the previous vocabulary,
    but a failed fetch must not -- otherwise a transient HA outage would
    silently wipe out previously working keyterm biasing.
    """

    success: bool
    terms: list[str] = field(default_factory=list)


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


# Upper bound on generated "Area Entity" combination terms (see
# _area_entity_combination_terms()). Purely a defensive cap against
# unbounded growth on very large installs -- not a documented HA/Moonshine
# limit -- so a single misconfigured huge instance cannot blow up keyterm
# biasing with thousands of terms.
MAX_COMBINATION_TERMS = 500


def _area_id_to_name(areas: list[dict[str, Any]]) -> dict[str, str]:
    """Map area_id -> area's primary display name (not aliases).

    Only the primary name is used for combination terms (not each alias)
    to avoid a combinatorial explosion of near-duplicate "Area Entity"
    phrases for areas with several aliases.
    """
    mapping: dict[str, str] = {}
    for area in areas:
        area_id = area.get("area_id")
        name = clean_term(area.get("name"))
        if area_id and name:
            mapping[area_id] = name
    return mapping


def _device_id_to_area_id(devices: list[dict[str, Any]]) -> dict[str, str]:
    mapping: dict[str, str] = {}
    for device in devices:
        device_id = device.get("id")
        area_id = device.get("area_id")
        if device_id and area_id:
            mapping[device_id] = area_id
    return mapping


def _area_entity_combination_terms(
    areas: list[dict[str, Any]],
    devices: list[dict[str, Any]],
    entities: list[dict[str, Any]],
) -> list[str]:
    """Build "Area Entity" phrases (e.g. "Wohnzimmer Rolllade") from real,
    known HA registry relationships only -- never guessed or derived from
    a raw entity_id.

    An entity's area comes from its own ``area_id`` if set (an explicit
    per-entity override in HA), otherwise inherited from its device's
    ``area_id`` (Entity -> Device -> Area). Entities/devices with no
    resolvable area, or without a usable human-assigned name, are skipped.
    Entity *domains* (``light``, ``switch``, ``binary_sensor``, ...) are
    never used as vocabulary words -- only ``extract_vocabulary_terms()``'s
    existing name resolution (entity ``name``/``original_name``) feeds
    this, so a domain can only ever appear here if a user literally named
    an entity that.
    """
    area_by_id = _area_id_to_name(areas)
    device_area_by_id = _device_id_to_area_id(devices)

    combos: list[str] = []
    for entity in entities:
        area_id = entity.get("area_id") or device_area_by_id.get(entity.get("device_id") or "")
        if not area_id:
            continue
        area_name = area_by_id.get(area_id)
        if not area_name:
            continue

        entity_term = clean_term(entity.get("name") or entity.get("original_name"))
        if not entity_term:
            continue

        # Skip if the entity's own name already starts with the area name
        # (e.g. entity named "Wohnzimmer Lampe" in area "Wohnzimmer") --
        # "Wohnzimmer Wohnzimmer Lampe" is a redundant, awkward duplicate.
        if entity_term.casefold().startswith(area_name.casefold()):
            continue

        combos.append(f"{area_name} {entity_term}")

    combos = dedupe_preserve_order(combos)
    combos.sort(key=str.casefold)  # deterministic order, independent of registry list order

    if len(combos) > MAX_COMBINATION_TERMS:
        _LOGGER.info(
            "Capping area+entity combination vocabulary at %d terms (had %d)",
            MAX_COMBINATION_TERMS,
            len(combos),
        )
        combos = combos[:MAX_COMBINATION_TERMS]

    return combos


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

    Additionally generates "Area Entity" combination terms (e.g. "Wohnzimmer
    Rolllade") wherever a real Entity -> Area or Entity -> Device -> Area
    relationship is known -- see _area_entity_combination_terms().
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

    terms.extend(_area_entity_combination_terms(areas, devices, entities))

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
) -> HaVocabularyResult:
    """Collect vocabulary terms from HA's Area/Device/Entity/Floor registries.

    Never raises: on any failure (missing token, unreachable Supervisor,
    auth failure, HA core down, etc.) this logs a WARNING and returns
    ``HaVocabularyResult(success=False, terms=[])``, so the add-on can start
    and keep serving plain STT without HA vocabulary rather than treating
    this as a hard dependency. A *successful* fetch that legitimately finds
    no areas/devices/entities returns ``HaVocabularyResult(success=True,
    terms=[])`` -- callers must not treat that the same as a failure (see
    HaVocabularyResult's docstring: this is what lets a caller keep the
    previous, last-known-good vocabulary across a failed periodic refresh).
    """
    token = token or os.environ.get("SUPERVISOR_TOKEN")
    if not token:
        _LOGGER.warning(
            "HA vocabulary unavailable: SUPERVISOR_TOKEN not set, continuing without HA vocabulary"
        )
        return HaVocabularyResult(success=False)
    if websockets is None:  # pragma: no cover - dependency is always installed in CI
        _LOGGER.warning(
            "HA vocabulary unavailable: websockets package not installed, "
            "continuing without HA vocabulary"
        )
        return HaVocabularyResult(success=False)

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
        return HaVocabularyResult(success=False)

    terms = extract_vocabulary_terms(areas, devices, entities, floors)
    _LOGGER.info("Loaded %d Home Assistant vocabulary terms", len(terms))
    return HaVocabularyResult(success=True, terms=terms)
