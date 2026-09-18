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

Two-stage normalization (v0.2.6): a term that the model's tokenizer
initially rejects is not simply discarded. It goes through:

1. Lossless normalization (:func:`normalize_keyterm`) -- NFC composition,
   whitespace trimming/collapsing, and removal of invisible Unicode
   *format* characters (category ``Cf``: soft hyphen, zero-width
   space/joiners, word joiner, BOM/zero-width no-break space, ...). These
   carry no speech meaning and their presence is often invisible copy-paste
   noise from Home Assistant's own UI (a real production example: the
   entity name "Wasch\xadmaschine" contains U+00AD SOFT HYPHEN between "h"
   and "m"). This stage never changes how a term reads or sounds and is
   applied to EVERY candidate up front, before any native call.
2. Speech fallback (:func:`_generate_speech_fallback`) -- tried only after
   the loaded model's tokenizer has actually rejected the stage-1 term on
   its own. Replaces purely visual separators ("/", "\\", "|") with a
   space, and drops emoji/decorative symbols (Unicode category ``So``/
   ``Sk``) and variation selectors (U+FE00-U+FE0F, U+E0100-U+E01EF, which
   attach to a preceding symbol and carry no meaning of their own). This
   stage CAN change what the term looks like, so it is only ever attempted
   as a fallback, and the result is always re-validated against the real
   model before use -- it is never assumed to be valid.

Neither stage is a substitute for the real model check: acceptance is
always decided by an actual ``set_keyterms()`` call against the loaded
Moonshine model, never by this module's own heuristics alone.
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

# Outcome statuses for an accepted term (AppliedKeyterm.status).
STATUS_UNCHANGED = "unchanged"
STATUS_NORMALIZED = "normalized"
STATUS_SPEECH_FALLBACK = "speech_fallback"

# Individual rejected-term/normalized-term log lines are capped to keep a
# large, mostly-valid HA vocabulary from flooding the log; the summary line
# always reports the true totals regardless of this cap (full per-term
# detail remains available at DEBUG).
_MAX_LOGGED_REJECTIONS = 20
_MAX_LOGGED_NORMALIZATIONS = 20

# Purely visual separators that carry no speech meaning of their own -- safe
# to turn into a space for a *fallback* variant (never applied blindly to
# the original candidate; only after the model has rejected it as-is).
_SPEECH_FALLBACK_SEPARATORS = ("/", "\\", "|")


class _KeytermTranscriber(Protocol):
    def set_keyterms(self, keyterms: list[str]) -> None: ...


@dataclass(frozen=True)
class RejectedKeyterm:
    """One keyterm that did not make it into the final applied list."""

    term: str
    reason: str
    detail: str = ""


@dataclass(frozen=True)
class AppliedKeyterm:
    """One keyterm that made it into the final applied list, and how."""

    original: str
    final: str
    status: str  # STATUS_UNCHANGED | STATUS_NORMALIZED | STATUS_SPEECH_FALLBACK


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
    applied: list[AppliedKeyterm] = field(default_factory=list)


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
    """Lossless, speech-preserving normalization (stage 1).

    - NFC (canonical composition): lossless and does not change how a term
      reads or sounds -- e.g. an "u" + combining diaeresis and a precomposed
      "ü" normalize to the same single codepoint.
    - Removes Unicode *format* characters (category ``Cf``): these are
      invisible characters that exist purely for text layout/formatting and
      carry no speech information at all -- soft hyphen (U+00AD, a hint for
      where a word MAY be broken across lines, invisible otherwise -- the
      real production case "Wasch\xadmaschine"), zero-width space/
      non-joiner/joiner (U+200B/U+200C/U+200D), word joiner (U+2060), and
      byte-order-mark / zero-width no-break space (U+FEFF). Removing them
      cannot change how the term is pronounced, since they were never
      pronounced to begin with.
    - Collapses any whitespace left behind (and pre-existing multiple
      spaces) to single spaces, and trims the result.

    This is normalization, not transliteration: real diacritics/umlauts/ß
    and every other visible character are never removed or replaced here.
    """
    nfc = unicodedata.normalize("NFC", term)
    without_format_chars = "".join(ch for ch in nfc if unicodedata.category(ch) != "Cf")
    return " ".join(without_format_chars.split())


def _generate_speech_fallback(term: str) -> str:
    """Speech-safe fallback variant (stage 2) -- only ever tried after the
    real model's tokenizer has already rejected ``term`` (stage 1) as-is.

    - Purely visual separators ("/", "\\", "|") become a space -- e.g.
      "Treppe/Büro" -> "Treppe Büro". These are never touched in stage 1
      because a different/future Moonshine model might tokenize them fine,
      and rewriting them unconditionally would prevent that.
    - Emoji/decorative symbols (Unicode category ``So``/``Sk``, e.g.
      "⚠" WARNING SIGN) and variation selectors (U+FE00-U+FE0F,
      U+E0100-U+E01EF -- these attach to a preceding symbol, such as
      "️" VARIATION SELECTOR-16 turning a warning sign into its emoji
      presentation, and carry no meaning on their own) are dropped.
    - Whitespace is then collapsed and trimmed again.

    Ordinary letters (including German umlauts/ß and accented Latin
    letters), digits, hyphens, and spaces are never touched here -- this
    only ever removes characters that are not letters/digits/ordinary
    punctuation a German speaker would actually say out loud.

    Returns the fallback candidate, which may equal ``term`` unchanged (no
    fallback opportunity found) or be empty (nothing speech-relevant was
    left). The caller decides what to do in either case -- this function
    never validates against the model itself.
    """
    replaced = term
    for separator in _SPEECH_FALLBACK_SEPARATORS:
        replaced = replaced.replace(separator, " ")

    kept_chars = []
    for ch in replaced:
        codepoint = ord(ch)
        if 0xFE00 <= codepoint <= 0xFE0F or 0xE0100 <= codepoint <= 0xE01EF:
            continue  # variation selectors
        if unicodedata.category(ch) in ("So", "Sk"):
            continue  # emoji / other symbols, modifier symbols
        kept_chars.append(ch)

    return " ".join("".join(kept_chars).split())


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


@dataclass(frozen=True)
class _AcceptedTerm:
    """One stage-1 term the model accepted, and what was actually applied."""

    stage1: str
    final: str
    used_fallback: bool


def _bisect_valid_keyterms(
    transcriber: _KeytermTranscriber, terms: list[str]
) -> tuple[list[_AcceptedTerm], list[RejectedKeyterm]]:
    """Divide-and-conquer search for which of ``terms`` the loaded model's
    tokenizer accepts, without necessarily needing one native call per term.

    Tries the whole batch first; only recurses into halves when the batch
    as a whole fails. A batch of size 1 that fails identifies exactly one
    bad term -- at that point, and ONLY at that point, a speech-fallback
    variant (:func:`_generate_speech_fallback`) is generated and tested
    with one more real, single-item ``set_keyterms()`` call; if the model
    accepts the fallback, that becomes the term's final applied form.
    Otherwise the term is rejected. This assumes (consistent with
    Moonshine's own C API, which tokenizes each comma-separated piece
    independently) that a term's validity does not depend on which other
    terms share the call -- if that assumption were ever wrong, the search
    still terminates and simply treats the smallest failing batch as the
    culprit.

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
        return [_AcceptedTerm(stage1=t, final=t, used_fallback=False) for t in terms], []
    except MoonshineError as e:
        if len(terms) == 1:
            term = terms[0]
            fallback = _generate_speech_fallback(term)
            if fallback and fallback != term:
                try:
                    transcriber.set_keyterms([fallback])
                    return [_AcceptedTerm(stage1=term, final=fallback, used_fallback=True)], []
                except MoonshineError as fallback_error:
                    return [], [
                        RejectedKeyterm(
                            term=term,
                            reason=REASON_TOKENIZER_REJECTED,
                            detail=(
                                f"original rejected ({e}); speech-fallback {fallback!r} "
                                f"also rejected ({fallback_error})"
                            ),
                        )
                    ]
            return [], [RejectedKeyterm(term=term, reason=REASON_TOKENIZER_REJECTED, detail=str(e))]
        mid = len(terms) // 2
        left_ok, left_bad = _bisect_valid_keyterms(transcriber, terms[:mid])
        right_ok, right_bad = _bisect_valid_keyterms(transcriber, terms[mid:])
        return left_ok + right_ok, left_bad + right_bad


def _log_summary(
    candidate_count: int,
    applied: list[AppliedKeyterm],
    rejected: list[RejectedKeyterm],
) -> None:
    normalized = [a for a in applied if a.status == STATUS_NORMALIZED]
    speech_fallback = [a for a in applied if a.status == STATUS_SPEECH_FALLBACK]
    unchanged = [a for a in applied if a.status == STATUS_UNCHANGED]

    for a in normalized[:_MAX_LOGGED_NORMALIZATIONS]:
        _LOGGER.info("Normalized keyterm: %r -> %r", a.original, a.final)
    if len(normalized) > _MAX_LOGGED_NORMALIZATIONS:
        _LOGGER.info(
            "... and %d more keyterm(s) normalized (see DEBUG for full list)",
            len(normalized) - _MAX_LOGGED_NORMALIZATIONS,
        )

    for a in speech_fallback[:_MAX_LOGGED_NORMALIZATIONS]:
        _LOGGER.info(
            "Replaced tokenizer-incompatible keyterm with speech-safe variant: %r -> %r",
            a.original,
            a.final,
        )
    if len(speech_fallback) > _MAX_LOGGED_NORMALIZATIONS:
        _LOGGER.info(
            "... and %d more keyterm(s) replaced with a speech-safe variant "
            "(see DEBUG for full list)",
            len(speech_fallback) - _MAX_LOGGED_NORMALIZATIONS,
        )

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

    _LOGGER.debug("All normalized keyterms: %s", [(a.original, a.final) for a in normalized])
    _LOGGER.debug(
        "All speech-fallback keyterms: %s", [(a.original, a.final) for a in speech_fallback]
    )

    _LOGGER.info(
        "Moonshine keyterms: candidate=%d unchanged=%d normalized=%d speech-fallback=%d "
        "rejected=%d applied=%d",
        candidate_count,
        len(unchanged),
        len(normalized),
        len(speech_fallback),
        len(rejected),
        len(applied),
    )


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

    Steps: lossless stage-1 normalization (NFC, whitespace, invisible
    format-character removal -- see :func:`normalize_keyterm`) + cheap
    syntactic pre-validation (rejects empty/control-character/delimiter/
    unencodable terms without needing a native call at all) -> bisection-
    based validation of the survivors against the actually loaded model,
    with a stage-2 speech-fallback retry
    (:func:`_generate_speech_fallback`) for any term the model rejects
    individually -> exactly one final, explicit, authoritative
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
    origin_of: dict[str, str] = {}
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
        origin_of[term] = raw

    try:
        accepted_terms, tokenizer_rejected_stage1 = _bisect_valid_keyterms(transcriber, survivors)
        tokenizer_rejected = [
            RejectedKeyterm(term=origin_of[r.term], reason=r.reason, detail=r.detail)
            for r in tokenizer_rejected_stage1
        ]
    except Exception as e:
        _LOGGER.error(
            "Unexpected error while validating Moonshine keyterms (%s: %s) -- "
            "treating this as a structural failure, not a per-term one",
            type(e).__name__,
            e,
        )
        accepted_terms = []
        tokenizer_rejected = [
            RejectedKeyterm(term=origin_of[t], reason=REASON_NATIVE_ERROR, detail=str(e))
            for t in survivors
        ]
    rejected.extend(tokenizer_rejected)

    applied: list[AppliedKeyterm] = []
    for a in accepted_terms:
        original = origin_of[a.stage1]
        if a.used_fallback:
            status = STATUS_SPEECH_FALLBACK
        elif original == a.stage1:
            status = STATUS_UNCHANGED
        else:
            status = STATUS_NORMALIZED
        applied.append(AppliedKeyterm(original=original, final=a.final, status=status))
    accepted = [a.final for a in accepted_terms]

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

    _log_summary(len(candidate_terms), applied, rejected)
    return SafeKeytermResult(
        accepted=accepted, rejected=rejected, apply_succeeded=True, applied=applied
    )
