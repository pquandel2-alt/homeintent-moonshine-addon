"""Read-only Home Assistant vocabulary collection for Moonshine keyterm biasing.

Reads Areas, Devices, Entities, and Floors from Home Assistant's registries
via the Supervisor-proxied Core WebSocket API
(``ws://supervisor/core/websocket``, see developers.home-assistant.io) to
build a list of natural-language terms (room names, device names, entity
names) that bias Moonshine's recognition towards the user's actual smart
home vocabulary.

The vocabulary is now primarily built from entities that are *effectively
exposed to Home Assistant Assist* (the ``"conversation"`` assistant) --
not from the full registry indiscriminately. This mirrors Home Assistant
Core's own exposure logic exactly, verified against the ``dev`` branch of
home-assistant/core (commit ``6c2d4140cc9b9b926a0df96c6eaa87b249b893c1``,
2026-09-17):

- ``homeassistant/components/homeassistant/exposed_entities.py``,
  ``ExposedEntities._is_default_exposed()`` -- the exact default-exposure
  rule reimplemented in :func:`is_default_exposed` below (entity_category/
  hidden_by, ``DEFAULT_EXPOSED_DOMAINS``, and device-class allowlists for
  ``binary_sensor``/``sensor``).
- ``ExposedEntities.async_should_expose()`` -- an explicit, cached
  ``registry_entry.options["conversation"]["should_expose"]`` value always
  wins over the default rule; reimplemented in :func:`is_effectively_exposed`.
- The assistant identifier for Home Assistant's built-in Assist pipeline is
  the literal string ``"conversation"`` (``homeassistant/components/
  conversation/const.py``'s ``DOMAIN``, confirmed at every
  ``async_should_expose()`` call site in that component) -- NOT
  ``"conversation.home_assistant"`` (that is a specific conversation
  *agent entity id*, a different concept).
- The bulk WS command ``homeassistant/expose_entity/list`` is deliberately
  NOT used as the source of truth here: it only returns entities whose
  ``should_expose`` has already been computed-and-cached at least once
  (lazily, the first time Assist actually looked at that entity), silently
  omitting anything else even if it would default to exposed. Instead this
  module reads ``config/entity_registry/list`` (which includes the same
  cached ``options`` field, plus ``entity_category``/``hidden_by``/
  ``disabled_by``) and ``get_states`` (for ``binary_sensor``/``sensor``
  ``device_class``, needed by the default-exposure rule) and computes the
  effective exposure itself -- reliable for every entity, not just
  previously-cached ones.

Strictly read-only: only ``config/*_registry/list``, ``config/entity_registry/
get_entries`` (for exposed entities' aliases only, never the full registry),
``get_states``, and ``homeassistant/expose_new_entities/get`` commands are
issued. This module never calls a service, changes a state, or edits an
automation/entity/exposure setting. Requires ``homeassistant_api: true`` in
config.yaml, which grants the add-on the SUPERVISOR_TOKEN used to
authenticate here.
"""

import asyncio
import itertools
import json
import logging
import os
import re
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
    exposed_entity_count: int = 0


DEFAULT_WS_URL = "ws://supervisor/core/websocket"
CONNECT_TIMEOUT = 10.0
COMMAND_TIMEOUT = 10.0

# The identifier Home Assistant's built-in Assist pipeline uses when calling
# async_should_expose()/the expose_entity WS commands -- see module
# docstring for the exact source verification.
CONVERSATION_ASSISTANT = "conversation"

# Verified against ExposedEntities._is_default_exposed() /
# DEFAULT_EXPOSED_DOMAINS / DEFAULT_EXPOSED_*_DEVICE_CLASSES in
# homeassistant/components/homeassistant/exposed_entities.py (dev branch,
# see module docstring for exact commit). Not invented here -- do not
# change without re-verifying against upstream.
DEFAULT_EXPOSED_DOMAINS = {
    "climate",
    "cover",
    "fan",
    "humidifier",
    "light",
    "media_player",
    "scene",
    "switch",
    "todo",
    "vacuum",
    "water_heater",
}
DEFAULT_EXPOSED_BINARY_SENSOR_DEVICE_CLASSES = {
    "door",
    "garage_door",
    "lock",
    "motion",
    "opening",
    "presence",
    "window",
}
DEFAULT_EXPOSED_SENSOR_DEVICE_CLASSES = {
    "aqi",
    "carbon_monoxide",
    "carbon_dioxide",
    "humidity",
    "pm10",
    "pm25",
    "temperature",
    "volatile_organic_compounds",
}


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


def _entity_domain(entity_id: str) -> str:
    return entity_id.split(".", 1)[0] if "." in entity_id else ""


def is_default_exposed(entity: dict[str, Any], device_class: str | None) -> bool:
    """Reimplements Home Assistant Core's
    ``ExposedEntities._is_default_exposed()`` exactly (see module docstring
    for the verified source). Only used when an entity has no explicit
    cached ``should_expose`` value yet (see :func:`is_effectively_exposed`).
    """
    if entity.get("entity_category") is not None or entity.get("hidden_by") is not None:
        return False

    domain = _entity_domain(entity.get("entity_id") or "")
    if domain in DEFAULT_EXPOSED_DOMAINS:
        return True
    if domain == "binary_sensor":
        return device_class in DEFAULT_EXPOSED_BINARY_SENSOR_DEVICE_CLASSES
    if domain == "sensor":
        return device_class in DEFAULT_EXPOSED_SENSOR_DEVICE_CLASSES
    return False


def is_effectively_exposed(
    entity: dict[str, Any],
    device_class: str | None,
    expose_new_default: bool,
) -> bool:
    """Return whether ``entity`` is effectively exposed to Assist right now.

    Mirrors ``ExposedEntities.async_should_expose()``: an explicit (or
    previously cached-as-default) ``should_expose`` value in
    ``entity["options"]["conversation"]`` always wins; otherwise falls back
    to the default-exposure rule (:func:`is_default_exposed`) if
    ``expose_new_default`` (Home Assistant's per-assistant "expose new
    entities automatically" setting) is enabled, else False.

    Disabled entities (``disabled_by is not None``) are always excluded as
    a practical matter -- they have no live state and are never usable via
    voice -- even though upstream's own ``_is_default_exposed()`` does not
    explicitly test ``disabled_by`` itself (see module docstring).
    """
    if entity.get("disabled_by") is not None:
        return False

    options = entity.get("options") or {}
    assistant_options = options.get(CONVERSATION_ASSISTANT) or {}
    if "should_expose" in assistant_options:
        return bool(assistant_options["should_expose"])

    if not expose_new_default:
        return False

    return is_default_exposed(entity, device_class)


def compute_exposed_entity_ids(
    entities: list[dict[str, Any]],
    states: list[dict[str, Any]] | None,
    expose_new_default: bool = True,
) -> set[str]:
    """Compute the set of entity_ids effectively exposed to Assist.

    ``states`` (as returned by the ``get_states`` WS command) supplies each
    entity's live ``device_class`` attribute, needed by the default-exposure
    rule for ``binary_sensor``/``sensor`` domains -- entity registry list
    responses do not carry ``device_class`` themselves.
    """
    device_class_by_entity_id: dict[str, str | None] = {}
    for state in states or []:
        entity_id = state.get("entity_id")
        if entity_id:
            device_class_by_entity_id[entity_id] = (state.get("attributes") or {}).get(
                "device_class"
            )

    exposed: set[str] = set()
    for entity in entities:
        entity_id = entity.get("entity_id")
        if not entity_id:
            continue
        device_class = device_class_by_entity_id.get(entity_id)
        if is_effectively_exposed(entity, device_class, expose_new_default):
            exposed.add(entity_id)
    return exposed


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


def _entity_area_id(entity: dict[str, Any], device_area_by_id: dict[str, str]) -> str | None:
    """An entity's area: its own ``area_id`` if set, else inherited via its
    device's ``area_id`` (Entity -> Device -> Area). Never guessed."""
    return entity.get("area_id") or device_area_by_id.get(entity.get("device_id") or "")


# A device's default name (no name_by_user) is only used as a vocabulary
# term if it doesn't look machine-generated. These patterns are deliberately
# conservative heuristics (not an HA-documented rule) -- see
# _has_technical_token()'s docstring.
_HEX_TOKEN_RE = re.compile(r"^[0-9a-fA-F]{6,}$")
_MAC_ADDRESS_RE = re.compile(r"^([0-9a-fA-F]{2}[:-]){5}[0-9a-fA-F]{2}$")
_UUID_RE = re.compile(
    r"^[0-9a-fA-F]{8}-[0-9a-fA-F]{4}-[0-9a-fA-F]{4}-[0-9a-fA-F]{4}-[0-9a-fA-F]{12}$"
)


def _looks_like_technical_id(word: str) -> bool:
    return bool(_MAC_ADDRESS_RE.match(word) or _UUID_RE.match(word) or _HEX_TOKEN_RE.match(word))


def _has_technical_token(name: str) -> bool:
    """True if any whitespace-separated word in ``name`` looks like a MAC
    address, a UUID, or a bare hex ID (e.g. "84FCE6") -- the kind of
    manufacturer-default suffix seen in names like "Shelly Plus 2PM
    84FCE6". Only applied to a device's *default* ``name``; a user-set
    ``name_by_user`` is always trusted as-is (see ``_device_term``)."""
    return any(_looks_like_technical_id(word) for word in name.split())


def _device_term(device: dict[str, Any]) -> str | None:
    """A device's vocabulary term: name_by_user (user-customized, always
    trusted) if set, else the default name -- but only if that default
    doesn't contain a technical-looking token (see _has_technical_token)."""
    user_term = clean_term(device.get("name_by_user"))
    if user_term:
        return user_term
    default_term = clean_term(device.get("name"))
    if default_term and not _has_technical_token(default_term):
        return default_term
    return None


def _fallback_term_from_entity_id(entity_id: str) -> str | None:
    """Last-resort term derived from an entity_id's object-id part only
    (never the domain) -- e.g. ``cover.wohnzimmer_rolllade`` ->
    "Wohnzimmer Rolllade". Only used when an entity has no name, no
    original_name, and no aliases at all."""
    if "." not in entity_id:
        return None
    _, object_id = entity_id.split(".", 1)
    words = [w for w in object_id.split("_") if w]
    if not words:
        return None
    return " ".join(w.capitalize() for w in words)


def _entity_name_candidates(entity: dict[str, Any]) -> list[str]:
    """All usable name-like terms for one entity: aliases plus its primary
    name (``name``/``original_name``), falling back to an entity_id-derived
    term only if nothing else is usable at all (see
    _fallback_term_from_entity_id)."""
    aliases = [t for a in (entity.get("aliases") or []) if (t := clean_term(a))]
    primary = clean_term(entity.get("name") or entity.get("original_name"))
    if not primary and not aliases:
        primary = _fallback_term_from_entity_id(entity.get("entity_id") or "")
    return ([primary] if primary else []) + aliases


def _area_entity_combination_terms(
    areas: list[dict[str, Any]],
    devices: list[dict[str, Any]],
    entities: list[dict[str, Any]],
) -> list[str]:
    """Build "Area Entity" (and "Area Alias") phrases (e.g. "Wohnzimmer
    Rolllade", "Wohnzimmer Rollo") from real, known HA registry
    relationships only -- never guessed or derived from a raw entity_id
    beyond the same last-resort fallback _entity_name_candidates() itself
    uses.

    Entities/devices with no resolvable area are skipped. Entity *domains*
    (``light``, ``switch``, ``binary_sensor``, ...) are never used as
    vocabulary words -- only _entity_name_candidates()'s own name
    resolution feeds this, so a domain can only ever appear here if a user
    literally named an entity that.
    """
    area_by_id = _area_id_to_name(areas)
    device_area_by_id = _device_id_to_area_id(devices)

    combos: list[str] = []
    for entity in entities:
        area_id = _entity_area_id(entity, device_area_by_id)
        if not area_id:
            continue
        area_name = area_by_id.get(area_id)
        if not area_name:
            continue

        for entity_term in _entity_name_candidates(entity):
            # Skip if the entity's own name already starts with the area
            # name (e.g. entity named "Wohnzimmer Lampe" in area
            # "Wohnzimmer") -- "Wohnzimmer Wohnzimmer Lampe" is a
            # redundant, awkward duplicate.
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


@dataclass
class _RelevanceFilter:
    """Which entities/areas/devices/floors are "relevant" for term
    generation. All *_ids fields are ``None`` in unfiltered mode (accept
    everything), else the concrete set reachable from the exposed entities.
    """

    entities: list[dict[str, Any]]
    area_ids: set[str] | None
    device_ids: set[str] | None
    floor_ids: set[str] | None


def _compute_relevance(
    areas: list[dict[str, Any]],
    devices: list[dict[str, Any]],
    entities: list[dict[str, Any]],
    exposed_entity_ids: set[str] | None,
) -> _RelevanceFilter:
    if exposed_entity_ids is None:
        return _RelevanceFilter(entities, None, None, None)

    relevant_entities = [e for e in entities if e.get("entity_id") in exposed_entity_ids]

    device_area_by_id = _device_id_to_area_id(devices)
    area_ids: set[str] = set()
    device_ids: set[str] = set()
    for entity in relevant_entities:
        area_id = _entity_area_id(entity, device_area_by_id)
        if area_id:
            area_ids.add(area_id)
        device_id = entity.get("device_id")
        if device_id:
            device_ids.add(device_id)

    floor_ids: set[str] = {
        floor_id
        for area in areas
        if area.get("area_id") in area_ids and (floor_id := area.get("floor_id"))
    }

    return _RelevanceFilter(relevant_entities, area_ids, device_ids, floor_ids)


def extract_vocabulary_terms(
    areas: list[dict[str, Any]],
    devices: list[dict[str, Any]],
    entities: list[dict[str, Any]],
    floors: list[dict[str, Any]] | None = None,
    exposed_entity_ids: set[str] | None = None,
) -> list[str]:
    """Build a deduplicated term list from raw HA registry list responses.

    If ``exposed_entity_ids`` is given (the normal production path, see
    fetch_ha_vocabulary()), only entities in that set -- and the
    areas/floors/devices actually reachable from them -- contribute terms:
    the vocabulary is now driven by what's effectively exposed to Assist,
    not the whole registry indiscriminately. If ``exposed_entity_ids`` is
    ``None`` (used by callers/tests that don't need exposure filtering),
    every area/device/entity contributes, matching this function's
    pre-exposure-filtering behavior exactly.

    Prefers human-assigned names over anything derived from technical IDs:
    area/floor ``name`` (+ area ``aliases``), device ``name_by_user`` (falls
    back to a non-technical-looking default ``name``, see _device_term()),
    and entity ``name``/``original_name``/``aliases`` (falls back to an
    entity_id-derived term only if none of those exist at all, see
    _fallback_term_from_entity_id()).

    Additionally generates "Area Entity"/"Area Alias" combination terms
    (e.g. "Wohnzimmer Rolllade") wherever a real Entity -> Area or
    Entity -> Device -> Area relationship is known -- see
    _area_entity_combination_terms().
    """
    relevance = _compute_relevance(areas, devices, entities, exposed_entity_ids)

    terms: list[str] = []

    for area in areas:
        if relevance.area_ids is not None and area.get("area_id") not in relevance.area_ids:
            continue
        terms.extend(_area_terms(area))

    for floor in floors or []:
        if relevance.floor_ids is not None and floor.get("floor_id") not in relevance.floor_ids:
            continue
        term = clean_term(floor.get("name"))
        if term:
            terms.append(term)

    for device in devices:
        if relevance.device_ids is not None and device.get("id") not in relevance.device_ids:
            continue
        term = _device_term(device)
        if term:
            terms.append(term)

    for entity in relevance.entities:
        terms.extend(_entity_name_candidates(entity))

    terms.extend(_area_entity_combination_terms(areas, devices, relevance.entities))

    return dedupe_preserve_order(terms)


async def _send_command(
    ws: Any, command_id: int, command_type: str, **params: Any
) -> list[dict[str, Any]]:
    """Send a command and return its ``result`` if it's a list, else []."""
    result = await _send_command_raw(ws, command_id, command_type, **params)
    return result if isinstance(result, list) else []


async def _send_command_raw(ws: Any, command_id: int, command_type: str, **params: Any) -> Any:
    """Send a command and return its raw ``result`` (any JSON value)."""
    payload = {"id": command_id, "type": command_type, **params}
    await ws.send(json.dumps(payload))
    async with asyncio.timeout(COMMAND_TIMEOUT):
        while True:
            message = json.loads(await ws.recv())
            if message.get("id") != command_id:
                # Not our response (e.g. a stray event); keep waiting.
                continue
            if not message.get("success", False):
                error = message.get("error", {})
                raise RuntimeError(f"{command_type} failed: {error.get('message', error)}")
            return message.get("result")


async def _fetch_entity_aliases(
    ws: Any, command_id: int, entity_ids: list[str]
) -> dict[str, list[str]]:
    """Fetch aliases for ``entity_ids`` in one round trip via
    ``config/entity_registry/get_entries`` (verified: takes a bulk
    ``entity_ids`` list and returns each entity's ``extended_dict``, which
    -- unlike the bulk ``config/entity_registry/list`` used elsewhere in
    this module -- includes ``aliases``). Only called for entities already
    known to be Assist-exposed, so this never touches the full registry.

    Never raises: degrades to no aliases (not a fetch failure) if the
    command is unsupported (e.g. an older Home Assistant Core) or returns
    something unexpected.
    """
    if not entity_ids:
        return {}
    try:
        result = await _send_command_raw(
            ws, command_id, "config/entity_registry/get_entries", entity_ids=entity_ids
        )
    except RuntimeError:
        return {}
    if not isinstance(result, dict):
        return {}

    aliases_by_entity_id: dict[str, list[str]] = {}
    for entity_id, entry in result.items():
        if not isinstance(entry, dict):
            continue
        # aliases is list[str | None] upstream: a null entry is the
        # "computed full entity name" sentinel, never a real alias string.
        aliases_by_entity_id[entity_id] = [a for a in (entry.get("aliases") or []) if a]
    return aliases_by_entity_id


async def _fetch_expose_new_default(ws: Any, command_id: int) -> bool:
    """Home Assistant's per-assistant "expose new entities automatically"
    setting for the conversation/Assist assistant (``homeassistant/
    expose_new_entities/get``). Defaults to True (Home Assistant Core's own
    default for the "conversation" assistant, see
    ``DEFAULT_EXPOSED_ASSISTANT`` in exposed_entities.py) if the command
    fails or is unsupported (e.g. an older Home Assistant Core)."""
    try:
        result = await _send_command_raw(
            ws,
            command_id,
            "homeassistant/expose_new_entities/get",
            assistant=CONVERSATION_ASSISTANT,
        )
    except RuntimeError:
        return True
    if isinstance(result, dict) and "expose_new" in result:
        return bool(result["expose_new"])
    return True


async def fetch_ha_vocabulary(
    token: str | None = None,
    ws_url: str = DEFAULT_WS_URL,
) -> HaVocabularyResult:
    """Collect vocabulary terms from entities effectively exposed to Home
    Assistant Assist (see module docstring for the exact exposure logic).

    Never raises: on any failure (missing token, unreachable Supervisor,
    auth failure, HA core down, etc.) this logs a WARNING and returns
    ``HaVocabularyResult(success=False, terms=[])``, so the add-on can start
    and keep serving plain STT without HA vocabulary rather than treating
    this as a hard dependency. A *successful* fetch that legitimately finds
    no exposed entities returns ``HaVocabularyResult(success=True,
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
            try:
                states = await _send_command(ws, next(command_id), "get_states")
            except RuntimeError:
                # Needed only for binary_sensor/sensor default-exposure
                # device_class checks; degrade gracefully if unavailable.
                states = []
            expose_new_default = await _fetch_expose_new_default(ws, next(command_id))

            exposed_entity_ids = compute_exposed_entity_ids(entities, states, expose_new_default)

            # Aliases (config/entity_registry/list doesn't return them, see
            # module docstring) are fetched only for exposed entities --
            # never the full registry -- in one bulk round trip.
            aliases_by_entity_id = await _fetch_entity_aliases(
                ws, next(command_id), sorted(exposed_entity_ids)
            )
            for entity in entities:
                entity_id = entity.get("entity_id")
                if entity_id in aliases_by_entity_id:
                    entity["aliases"] = aliases_by_entity_id[entity_id]
    except Exception as err:
        _LOGGER.warning(
            "HA vocabulary unavailable (%s: %s), continuing without HA vocabulary",
            type(err).__name__,
            err,
        )
        return HaVocabularyResult(success=False)

    terms = extract_vocabulary_terms(
        areas, devices, entities, floors, exposed_entity_ids=exposed_entity_ids
    )
    _LOGGER.info(
        "Assist-exposed entities: %d, HA vocabulary terms: %d",
        len(exposed_entity_ids),
        len(terms),
    )
    return HaVocabularyResult(
        success=True, terms=terms, exposed_entity_count=len(exposed_entity_ids)
    )
