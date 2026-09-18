"""Original German text normalization for TTS (Kokoro ONNX and, if ever
needed, Pocket TTS).

Written from scratch for this add-on -- NOT vendored from any third-party
"german_text_rules.py"/"tts_normalizer.py" reference implementation. Those
reference projects (e.g. the Kokoro German "Martin" community integration)
could not be reviewed closely enough from this development environment to
vendor their code responsibly (no verifiable top-level LICENSE file was
found at the expected path, so their exact terms could not be confirmed --
see ABSCHLUSSBERICHT_V0.3.0.md), so per this project's own instructions an
original, independently tested implementation was written instead of
guessing at a license.

Why this exists at all (verified, not assumed): kokoro-onnx's own
``Tokenizer.phonemize()`` does essentially no normalization beyond
``text.strip()`` before handing text to espeak-ng's German voice.
Directly phonemizing raw HomeIntent-style strings with espeak-ng's German
backend (verified against phonemizer==3.4.0 + espeakng-loader==0.2.4)
produces mispronunciations for exactly the patterns Home Assistant emits
constantly:

- "18:20 Uhr" -> the colon is read as a literal pause/symbol, not as
  "achtzehn Uhr zwanzig"
- "2,5 kWh" -> "eitscht" is exactly the *English* letter name for "h"
  inserted mid-word; abbreviated units are not expanded into spoken German
  words at all
- "21,5 °C" -> "Grad" is produced but the unit letter "C" is spoken as the
  bare letter, not the word "Celsius"
- "500 g" / "1 l" -> unit abbreviations are read as bare letters ("Ge",
  "El"), not as "Gramm"/"Liter"

This module fixes exactly that class of problem: units, temperatures,
times, and currency amounts that a smart-home assistant's automations and
sensor read-outs produce constantly. It does NOT attempt full German
ordinal declension (grammatical case-correct forms like "dritten"/
"drittem") -- that needs real sentence-level grammar analysis and getting
it wrong is worse than leaving a bare number for espeak's own (already
reasonable) cardinal-number reading to handle. See
ABSCHLUSSBERICHT_V0.3.0.md's "verbleibende Risiken" for what is
deliberately out of scope.
"""

import re

# Ordered abbreviation -> full word expansions. Order matters: longer/more
# specific keys are matched first so e.g. "kWh" is not partially eaten by a
# "W" (watt) rule. Applied only as a *whole unit* right after a number (see
# _expand_units_after_numbers), never to the word standing alone in running
# text, so real abbreviations that happen to also be normal words are not
# broken.
#
# Regex alternation matches the FIRST alternative that fits at the current
# position, not the longest one -- so wherever one unit's spelling is a
# strict prefix of another's (e.g. bare "m" is a prefix of "mg"/"ml"/"mm"/
# "m²"/"mA"/"mAh"/"min", and "kW" is a prefix of "kWh"), the longer,
# more-specific spelling MUST be listed first or it would never be
# reached. Verified by test_german_text_normalizer.py's ordering-conflict
# cases (e.g. "5 mg" must not become "5 Meter g").
_UNIT_EXPANSIONS: list[tuple[str, str]] = [
    ("kWh", "Kilowattstunden"),
    ("Wh", "Wattstunden"),
    ("kW", "Kilowatt"),
    ("mAh", "Milliamperestunden"),
    ("mA", "Milliampere"),
    ("km/h", "Stundenkilometer"),
    ("km", "Kilometer"),
    ("cm", "Zentimeter"),
    ("mm", "Millimeter"),
    ("m²", "Quadratmeter"),
    ("kg", "Kilogramm"),
    ("mg", "Milligramm"),
    ("ml", "Milliliter"),
    ("min", "Minuten"),
    ("m", "Meter"),  # bare "m" last: a strict prefix of several units above
    ("g", "Gramm"),
    ("l", "Liter"),
    ("%", "Prozent"),
    ("h", "Stunden"),
]

# Dotted/plain German abbreviations expanded regardless of a preceding
# number (word-boundary matched, case-sensitive to avoid clobbering
# unrelated lowercase words).
_WORD_ABBREVIATIONS: dict[str, str] = {
    "Std.": "Stunden",
    "Min.": "Minuten",
    "Std": "Stunden",
    "zzgl.": "zuzüglich",
    "ggf.": "gegebenenfalls",
    "Stck.": "Stück",
    "bzw.": "beziehungsweise",
    "z.B.": "zum Beispiel",
    "ca.": "zirka",
    "usw.": "und so weiter",
}

_CURRENCY_SYMBOLS = {
    "€": "Euro",
    "EUR": "Euro",
}

_DEGREE_CELSIUS_RE = re.compile(r"°\s?C\b")
_DEGREE_FAHRENHEIT_RE = re.compile(r"°\s?F\b")
_DEGREE_BARE_RE = re.compile(r"°")

# "18:20 Uhr" / "18:20" (only HH:MM, 0-23:0-59, so it never eats an
# unrelated ratio or timestamp-like string that isn't a real time).
_TIME_RE = re.compile(r"\b([01]?\d|2[0-3]):([0-5]\d)(\s?Uhr)?\b")

# A number (possibly with a German decimal comma) directly followed by one
# of the known unit abbreviations, with an optional space in between, e.g.
# "2,5 kWh", "500g", "21,5 °C" (handled separately), "45 Min.".
# A trailing negative lookahead for a following letter/digit (rather than
# ``\b``) is used deliberately: ``\b`` only fires between a word and a
# non-word character, so it silently fails to match after a symbol like
# "%"/"€"/"°" at the end of a string or before punctuation (both sides
# would count as "non-word") -- verified empirically while writing this
# module's tests (a bare "49,99 €" was NOT expanded with ``\b``).
_NOT_FOLLOWED_BY_LETTER_OR_DIGIT = r"(?![a-zA-ZäöüÄÖÜß0-9])"

_NUMBER_UNIT_RE = re.compile(
    r"(?P<number>\d+(?:,\d+)?)\s?(?P<unit>"
    + "|".join(re.escape(u) for u, _ in _UNIT_EXPANSIONS)
    + r")"
    + _NOT_FOLLOWED_BY_LETTER_OR_DIGIT
)

_CURRENCY_RE = re.compile(
    r"(?P<number>\d+(?:,\d+)?)\s?(?P<symbol>"
    + "|".join(re.escape(c) for c in _CURRENCY_SYMBOLS)
    + r")"
    + _NOT_FOLLOWED_BY_LETTER_OR_DIGIT
)


def _expand_time(match: re.Match[str]) -> str:
    hour, minute = match.group(1), match.group(2)
    hour_i, minute_i = int(hour), int(minute)
    if minute_i == 0:
        return f"{hour_i} Uhr"
    return f"{hour_i} Uhr {minute_i}"


def _expand_number_unit(match: re.Match[str]) -> str:
    number = match.group("number")
    unit = match.group("unit")
    expansion = dict(_UNIT_EXPANSIONS)[unit]
    return f"{number} {expansion}"


def _expand_currency(match: re.Match[str]) -> str:
    number = match.group("number")
    symbol = match.group("symbol")
    return f"{number} {_CURRENCY_SYMBOLS[symbol]}"


def normalize_german_text(text: str) -> str:
    """Expand times, temperatures, units, and currency amounts into words a
    German TTS voice reads correctly.

    Deliberately conservative: only patterns unambiguously identified as a
    time, a number+unit, or a currency amount are touched. Ordinary prose,
    German umlauts/ß, and words that merely *contain* a unit abbreviation
    as a substring (there is no free-standing "l"/"g"/"m" to match inside a
    normal word, since matching requires a preceding number) are never
    altered.

    Idempotent: normalizing already-normalized text is a no-op, so this is
    always safe to run even if called more than once on the same string.
    """
    if not text:
        return text

    result = text

    # Times first: "18:20 Uhr" must not be mistaken for a ratio/number
    # afterward, and doesn't overlap with the unit patterns below.
    result = _TIME_RE.sub(_expand_time, result)

    # Temperatures: normalize the degree symbol before generic unit
    # expansion, since "°C"/"°F" are not plain suffix units of a number
    # (the degree sign sits between the number and the letter).
    result = _DEGREE_CELSIUS_RE.sub(" Grad Celsius", result)
    result = _DEGREE_FAHRENHEIT_RE.sub(" Grad Fahrenheit", result)
    result = _DEGREE_BARE_RE.sub(" Grad", result)

    # Currency amounts before generic units, since "EUR"/"€" are not in
    # _UNIT_EXPANSIONS.
    result = _CURRENCY_RE.sub(_expand_currency, result)

    # Number + unit abbreviation, e.g. "2,5 kWh" -> "2,5 Kilowattstunden".
    result = _NUMBER_UNIT_RE.sub(_expand_number_unit, result)

    # Dotted/plain word abbreviations that appear regardless of a
    # preceding number. A dotted abbreviation at the very end of the
    # string is a real sentence-final period, not just an abbreviation
    # marker -- restored after expansion so "... 2 Std." still ends the
    # utterance with a pause instead of silently losing it. Mid-sentence
    # occurrences ("45 Min. später") deliberately do NOT gain a period:
    # the dot there is only an abbreviation marker, and inserting one
    # would introduce a false sentence break.
    ends_with_dotted_abbreviation = any(
        abbreviation.endswith(".") and result.rstrip().endswith(abbreviation)
        for abbreviation in _WORD_ABBREVIATIONS
    )
    for abbreviation, expansion in _WORD_ABBREVIATIONS.items():
        result = re.sub(re.escape(abbreviation) + r"(?![a-zA-ZäöüÄÖÜß])", expansion, result)
    if ends_with_dotted_abbreviation and not result.rstrip().endswith("."):
        result = result.rstrip() + "."

    # Collapse whitespace left behind by the substitutions above.
    result = " ".join(result.split())
    return result
