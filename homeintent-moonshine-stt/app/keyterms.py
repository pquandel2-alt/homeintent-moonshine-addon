"""Parsing and merging of Moonshine keyterm biasing lists.

Moonshine's Transcriber.set_keyterms() takes a flat list of strings and
raises MoonshineError if any term contains a comma (it is the wire
delimiter for the C API), so a comma-separated user field is exactly the
right shape for app config -- as long as it is parsed defensively.
"""


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
