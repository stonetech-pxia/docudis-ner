# Copyright 2026 the Docudis contributors. Licensed under Apache-2.0.
"""Reference for the host's post-processing of a model intent: adds the regions
and verticals that keywords.json finds in the instruction.

The model leaves out regions and verticals more often than anything else, and
they are lookups (an acronym's country, a document kind's domain) rather than
understanding. The host unions the keyword hits into the model's lists; it
never removes what the model set. Hosts port `keyword_hits` as is: lowercase,
strip accents, apply `except`, then whole-word matching for Latin keywords and
substring matching for CJK ones.
"""

import json
import re
import unicodedata
from functools import cache
from pathlib import Path

KEYWORDS = Path(__file__).parent / "keywords.json"


def _fold(text, lower=True):
    text = text.replace("’", "'").replace("ß", "ss")
    if lower:
        text = text.lower()
    text = unicodedata.normalize("NFKD", text)
    return "".join(c for c in text if not unicodedata.combining(c))


def _pattern(keyword, case_sensitive):
    word = _fold(keyword, lower=not case_sensitive)
    if re.fullmatch(r"[\x00-\x7f]+", word):
        # Whole word: not preceded or followed by a letter or digit.
        return re.compile(r"(?<![0-9A-Za-z])" + re.escape(word) + r"(?![0-9A-Za-z])")
    return re.compile(re.escape(word))


@cache
def _table():
    data = json.loads(KEYWORDS.read_text("utf-8"))
    sensitive = set(data["case_sensitive"])
    groups = {}
    for field in ("regions", "verticals"):
        for value, words in data[field].items():
            groups[(field, value)] = [(_pattern(w, w in sensitive), w in sensitive) for w in words]
    excepts = {key: [_fold(p) for p in phrases] for key, phrases in data["except"].items()}
    return groups, excepts


def keyword_hits(instruction):
    """{"regions": [...], "verticals": [...]} found in the instruction."""
    groups, excepts = _table()
    hits = {"regions": [], "verticals": []}
    for (field, value), patterns in groups.items():
        lower, original = _fold(instruction), _fold(instruction, lower=False)
        for phrase in excepts.get(field, []) + excepts.get(value, []):
            lower = lower.replace(phrase, " ")
        if any(p.search(original if sensitive else lower) for p, sensitive in patterns):
            hits[field].append(value)
    return hits


def complete(instruction, intent):
    """The intent with keyword regions and verticals added to the model's."""
    out = dict(intent)
    for field, values in keyword_hits(instruction).items():
        merged = list(intent.get(field, [])) + [v for v in values if v not in intent.get(field, [])]
        if merged:
            out[field] = merged
    return out
