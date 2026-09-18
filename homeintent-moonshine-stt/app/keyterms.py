"""Parsing, merging, and safe application of Moonshine keyterm biasing lists.

Moonshine's Transcriber.set_keyterms() takes a flat list of strings and
joins it into a single comma-delimited string for its native C API in ONE
call (see moonshine_voice.transcriber.Transcriber.set_keyterms() -- verified
against the installed moonshine-voice==0.1.5 source directly, not assumed).
If the native tokenizer rejects ANY single term in that combined string, the
WHOLE call raises ``MoonshineError`` with a generic, per-call (not per-term)
message -- a real production incident hit exactly this with the entity name
"/Büro", crashing the whole add-on (``set_keyterms()`` raised out of
startup, uncaught, exit code 1, restart loop). moonshine-voice's C API
exposes only ``moonshine_transcriber_set_keyterms`` and
``moonshine_transcriber_set_context`` (confirmed by inspecting its ctypes
bindings) -- there is no separate "validate a term against the loaded
model's tokenizer without activating it" API to call instead.

:func:`apply_safe_keyterms` is the one, reusable, safe path every caller
(startup, manual extra_keyterms, the periodic HA vocabulary refresh) must
use instead of calling ``transcriber.set_keyterms()`` directly: it
normalizes, does cheap syntactic pre-validation (empty strings, control
characters, the comma delimiter itself, unencodable Unicode), then
validates the survivors against the ACTUAL loaded model via
``set_keyterms()`` itself, isolating incompatible terms with a
divide-and-conquer search (not necessarily one native call per term) so a
single bad term is skipped rather than taking down the whole list. It
never raises: an unexpected, non-per-term native error degrades to
restoring a caller-supplied fallback list (or disabling biasing entirely)
rather than propagating. It always ends with exactly one authoritative,
explicit ``set_keyterms()`` call for the final accepted list -- the
diagnostic calls made while isolating a bad term are side-effecting but
never authoritative on their own.

This module deliberately does NOT strip diacritics/umlauts or otherwise
transliterate terms (e.g. "Büro" -> "Buro") -- German names must survive
unless the model's own tokenizer genuinely cannot represent them, which is
what the native validation step (not a hand-rolled character blocklist)
determines.
"""

import logging
import unicodedata
from dataclasses import dataclass, field
from typing import Protocol

from moonshine_voice import MoonshineError

_LOGGER = logging.getLogger(__name__)

# Only per-term-content rejection reasons are ever attributed to a specific
# term; "native_error" means something other than this term's own content
# caused the failure (transcriber closed, non-streaming architecture, an
# unexpected exception type, ...) -- see apply_safe_keyterms()'s docstring.
REASON_EMPTY = "empty"
REASON_CONTROL_CHARACTER = "control_character"
REASON_DELIMITER = "delimiter"
REASON_INVALID_UNICODE = "invalid_unicode"
REASON_TOKENIZER_REJECTED = "tokenizer_rejected"
REASON_NATIVE_ERROR = "native_error"

# Individual rejected-term warnings are capped to keep a large, mostly-valid
# HA vocabulary from flooding the log; the summary line always reports the
# true total regardless of this cap (full detail remains available at DEBUG).
_MAX_LOGGED_REJECTIONS = 20


class _KeytermTranscriber(Protocol):
    def set_keyterms(self, keyterms: list[str]) -> None: ...


@dataclass(frozen=True)
class RejectedKeyterm:
    """One keyterm that did not make it into the final applied list."""

    term: str
    reason: str
    detail: str = ""


@dataclass(frozen=True)
class SafeKeytermResult:
    """Outcome of one :func:`apply_safe_keyterms` call.

    ``apply_succeeded`` distinguishes "the final authoritative apply call
    itself completed" (even if some candidate terms were rejected along the
    way -- that is normal, expected behavior, not a failure) from "the
    final apply itself raised an unexpected, non-per-term error and had to
    fall back". Callers that track a last-known-good state (see
    app/__main__.py's periodic refresh) must only commit a new
    last-known-good snapshot when this is True.
    """

    accepted: list[str] = field(default_factory=list)
    rejected: list[RejectedKeyterm] = field(default_factory=list)
    apply_succeeded: bool = True


def parse_extra_keyterms(raw: str) -> list[str]:
    """Parse the user-entered ``extra_keyterms`` config value.

    Trims whitespace around each entry, drops empty entries (e.g. from
    trailing commas or double commas), and removes duplicates while
    preserving first-seen order.
    """
    seen: set[str] = set()
    result: list[str] = []
    for part in raw.split(","):
        term = part.strip()
        if not term or term in seen:
            continue
        seen.add(term)
        result.append(term)
    return result


def merge_keyterms(*term_lists: list[str]) -> list[str]:
    """Merge multiple keyterm lists, deduping while keeping first-seen order."""
    seen: set[str] = set()
    result: list[str] = []
    for terms in term_lists:
        for term in terms:
            if term not in seen:
                seen.add(term)
                result.append(term)
    return result


def normalize_keyterm(term: str) -> str:
    """Trim surrounding whitespace and apply NFC Unicode normalization.

    NFC (canonical composition) is lossless and does not change how a term
    reads or sounds -- e.g. an "u" + combining diaeresis and a precomposed
    "ü" normalize to the same single codepoint. This is normalization, not
    transliteration: real diacritics/umlauts/ß are never removed or
    replaced here.
    """
    return unicodedata.normalize("NFC", term.strip())


def _basic_validate(term: str) -> str | None:
    """Cheap, model-independent syntactic pre-check.

    Returns a rejection reason (one of the ``REASON_*`` constants), or
    ``None`` if the term passes. This intentionally rejects only what is
    unambiguously never valid (empty content, control characters, the
    comma wire-delimiter, unencodable Unicode) -- ordinary German
    orthography (umlauts, ß, spaces, hyphens, slashes) always passes here;
    whether a *specific* model's tokenizer can actually represent the term
    is decided later, against the real loaded model.
    """
    if not term:
        return REASON_EMPTY
    if "," in term:
        # Moonshine's own wire delimiter (Transcriber.set_keyterms() joins
        # the list with "," for its native C API) -- never valid regardless
        # of model.
        return REASON_DELIMITER
    if any(unicodedata.category(ch) == "Cc" for ch in term):
        # Covers NUL, newlines, tabs, and other C0/C1 control characters
        # under one consistent rule.
        return REASON_CONTROL_CHARACTER
    try:
        term.encode("utf-8")
    except UnicodeEncodeError:
        return REASON_INVALID_UNICODE
    return None


def _bisect_valid_keyterms(
    transcriber: _KeytermTranscriber, terms: list[str]
) -> tuple[list[str], list[RejectedKeyterm]]:
    """Divide-and-conquer search for which of ``terms`` the loaded model's
    tokenizer accepts, without necessarily needing one native call per term.

    Tries the whole batch first; only recurses into halves when the batch
    as a whole fails. A batch of size 1 that fails identifies exactly one
    bad term. This assumes (consistent with Moonshine's own C API, which
    tokenizes each comma-separated piece independently) that a term's
    validity does not depend on which other terms share the call -- if that
    assumption were ever wrong, the search still terminates and simply
    treats the smallest failing batch as the culprit.

    Every call here is a REAL, side-effecting ``set_keyterms()`` call and
    therefore diagnostic, not authoritative -- callers must always follow
    up with one final, explicit ``set_keyterms(accepted)`` call (see
    :func:`apply_safe_keyterms`).

    Raises whatever non-``MoonshineError`` exception a native call raises;
    such an exception is not a per-term content problem and the caller
    handles it as a structural/native failure instead.
    """
    if not terms:
        return [], []
    try:
        transcriber.set_keyterms(terms)
        return list(terms), []
    except MoonshineError as e:
        if len(terms) == 1:
            return [], [
                RejectedKeyterm(term=terms[0], reason=REASON_TOKENIZER_REJECTED, detail=str(e))
            ]
        mid = len(terms) // 2
        left_ok, left_bad = _bisect_valid_keyterms(transcriber, terms[:mid])
        right_ok, right_bad = _bisect_valid_keyterms(transcriber, terms[mid:])
        return left_ok + right_ok, left_bad + right_bad


def _log_summary(
    candidate_count: int, accepted: list[str], rejected: list[RejectedKeyterm]
) -> None:
    if rejected:
        for bad in rejected[:_MAX_LOGGED_REJECTIONS]:
            _LOGGER.warning(
                "Skipping Moonshine-incompatible keyterm: %r (reason=%s)", bad.term, bad.reason
            )
        if len(rejected) > _MAX_LOGGED_REJECTIONS:
            _LOGGER.warning(
                "... and %d more incompatible keyterm(s) skipped (see DEBUG for full list)",
                len(rejected) - _MAX_LOGGED_REJECTIONS,
            )
        _LOGGER.debug("All rejected keyterms: %s", [(r.term, r.reason, r.detail) for r in rejected])
        _LOGGER.warning(
            "%d of %d candidate keyterm(s) were incompatible with the loaded Moonshine model "
            "and were skipped",
            len(rejected),
            candidate_count,
        )
    _LOGGER.info("Applied %d effective Moonshine keyterm(s)", len(accepted))


def apply_safe_keyterms(
    transcriber: _KeytermTranscriber,
    candidate_terms: list[str],
    fallback_terms: list[str] | None = None,
) -> SafeKeytermResult:
    """Safely apply ``candidate_terms`` to ``transcriber``, never raising.

    This is the single, reusable path for every caller that wants to set
    Moonshine keyterms -- automatic HA vocabulary (entities, aliases,
    areas, floors, area+entity combinations -- all of it flows through
    here uniformly, since it all arrives as one flat string list by the
    time it reaches this function), manual ``extra_keyterms``, and the
    periodic refresh all use this instead of calling
    ``transcriber.set_keyterms()`` directly.

    Steps: normalize + cheap syntactic pre-validation (rejects empty/
    control-character/delimiter/unencodable terms without needing a native
    call at all) -> bisection-based validation of the survivors against the
    actually loaded model -> exactly one final, explicit, authoritative
    ``set_keyterms()`` call for the accepted list (even if empty -- turning
    biasing off is itself a valid, explicit outcome, never an accident of a
    half-finished diagnostic call).

    If that final call itself raises an exception that is not attributable
    to a specific term (the bisection search already isolated all
    per-term-content failures before reaching this point, so this is a
    genuinely unexpected/structural error), ``fallback_terms`` -- the
    caller's last known-good list, or ``None``/empty to disable biasing
    entirely -- is applied instead, and ``apply_succeeded=False`` is
    returned so the caller knows NOT to commit ``candidate_terms`` as a new
    known-good state.

    Returns:
        A :class:`SafeKeytermResult`. Never raises.
    """
    rejected: list[RejectedKeyterm] = []
    survivors: list[str] = []
    seen: set[str] = set()
    for raw in candidate_terms:
        term = normalize_keyterm(raw)
        reason = _basic_validate(term)
        if reason is not None:
            rejected.append(RejectedKeyterm(term=raw, reason=reason))
            continue
        if term in seen:
            continue
        seen.add(term)
        survivors.append(term)

    try:
        accepted, tokenizer_rejected = _bisect_valid_keyterms(transcriber, survivors)
    except Exception as e:
        _LOGGER.error(
            "Unexpected error while validating Moonshine keyterms (%s: %s) -- "
            "treating this as a structural failure, not a per-term one",
            type(e).__name__,
            e,
        )
        accepted = []
        tokenizer_rejected = [
            RejectedKeyterm(term=t, reason=REASON_NATIVE_ERROR, detail=str(e)) for t in survivors
        ]
    rejected.extend(tokenizer_rejected)

    try:
        transcriber.set_keyterms(accepted)
    except Exception as e:
        _LOGGER.error(
            "Unexpected error applying the final Moonshine keyterm list (%s: %s) -- "
            "restoring the previous known-good keyterms instead",
            type(e).__name__,
            e,
        )
        restore = list(fallback_terms) if fallback_terms else []
        try:
            transcriber.set_keyterms(restore)
        except Exception as restore_err:
            _LOGGER.error(
                "Failed to restore previous Moonshine keyterms (%s: %s); STT continues with "
                "whatever keyterm state is currently active",
                type(restore_err).__name__,
                restore_err,
            )
        return SafeKeytermResult(
            accepted=restore,
            rejected=rejected
            + [
                RejectedKeyterm(term=t, reason=REASON_NATIVE_ERROR, detail=str(e)) for t in accepted
            ],
            apply_succeeded=False,
        )

    _log_summary(len(candidate_terms), accepted, rejected)
    return SafeKeytermResult(accepted=accepted, rejected=rejected, apply_succeeded=True)
