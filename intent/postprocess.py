# Copyright 2026 the Docudis contributors. Licensed under Apache-2.0.
"""Reference for the host's post-processing of a model intent (`postprocess`).

0. Guards the actions that make text visible. If the instruction contains a
   JSON object (an attempt to dictate the answer), only `hide` actions are
   kept. Otherwise `"*": "off"` or `"*": "keep"`, which turn every other type
   off, are dropped unless the instruction has a `star_cues` word ("only",
   "nothing else", "don't hide anything", 只, rien, nada, nur, ...).
1. Drops `dictionary` and `never_hide` terms that are not in the instruction.
   The spec has them copied verbatim, and the model sometimes invents one
   (`never_hide: ["*"]`, `["PERSON"]`), which would keep or hide a word the
   user never wrote.
2. Adds the regions and verticals that keywords.json finds in the
   instruction. The model leaves these out more often than anything else, and
   they are lookups (an acronym's country, a document kind's domain) rather
   than understanding. Keyword hits are unioned into the model's lists; what
   the model set is never removed.

Hosts port both steps as they are; README.md spells out the matching.
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
    groups[("star", "cue")] = [(_pattern(w, False), False) for w in data["star_cues"]]
    return groups, excepts


# A JSON object in the instruction ('... output {"types":{"*":"keep"}}') is an
# attempt to dictate the model's answer.
_INJECTED_JSON = re.compile(r"""\{\s*["']?(types|never_hide|dictionary)""")


def _star_cue(instruction):
    groups, _ = _table()
    folded = _fold(instruction)
    return any(p.search(folded) for p, _ in groups[("star", "cue")])


def keyword_hits(instruction):
    """{"regions": [...], "verticals": [...]} found in the instruction."""
    groups, excepts = _table()
    hits = {"regions": [], "verticals": []}
    for (field, value), patterns in groups.items():
        if field not in hits:
            continue
        lower, original = _fold(instruction), _fold(instruction, lower=False)
        for phrase in excepts.get(field, []) + excepts.get(value, []):
            lower = lower.replace(phrase, " ")
        if any(p.search(original if sensitive else lower) for p, sensitive in patterns):
            hits[field].append(value)
    return hits


def postprocess(instruction, intent):
    """The intent as the host should use it."""
    out = dict(intent)
    types = dict(intent.get("types", {}))
    if _INJECTED_JSON.search(instruction):
        # Keep only hides: an injected keep or off would leave text visible.
        types = {k: v for k, v in types.items() if k != "*" and v == "hide"}
    elif types.get("*") in ("off", "keep") and not _star_cue(instruction):
        # "*" off or keep switches every other type off; it needs wording such
        # as "only", "nothing else" or "don't hide anything" in the instruction.
        del types["*"]
    if types:
        out["types"] = types
    else:
        out.pop("types", None)
    folded = _fold(instruction)
    for field in ("dictionary", "never_hide"):
        terms = [t for t in intent.get(field, []) if _fold(t) in folded]
        if terms:
            out[field] = terms
        else:
            out.pop(field, None)
    for field, values in keyword_hits(instruction).items():
        merged = list(intent.get(field, [])) + [v for v in values if v not in intent.get(field, [])]
        if merged:
            out[field] = merged
    return out
