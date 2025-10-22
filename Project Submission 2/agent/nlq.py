from __future__ import annotations
from dataclasses import dataclass
import re
from typing import List

from .utils import normalize_text

# Simple keyword triggers
EMBED_TRIGGERS = ["embedding", "i think", "maybe", "suggest", "according to embedding"]
FACT_TRIGGERS = [
    "who", "what", "when", "genre", "director", "writer", "screenwriter",
    "duration", "runtime", "language", "company", "mpaa",
]

@dataclass
class NLQ:
    raw: str
    text_norm: str
    intent: str                 # "factual" | "embedding" | "both"
    entity_strings: List[str]
    relation_triggers: List[str]

def _extract_quoted_titles(s: str) -> List[str]:
    out: List[str] = []
    for m in re.finditer(r'"([^"]+)"', s):
        out.append(m.group(1))
    return out

def parse_nlq(q: str) -> NLQ:
    s = q.strip()
    n = normalize_text(s)

    # intent
    intent = "both"
    if any(t in n for t in EMBED_TRIGGERS):
        intent = "embedding"
    if any(t in n for t in FACT_TRIGGERS):
        intent = "factual" if intent != "embedding" else "both"

    # entities: quoted titles first; fallback to capitalized spans
    titles = _extract_quoted_titles(q)
    if not titles:
        caps = re.findall(r"([A-Z][A-Za-z0-9:.'\-]*(?:\s+[A-Z][A-Za-z0-9:.'\-]*)+)", q)
        titles = caps

    # relation triggers: tokenize to simple lower tokens
    tokens = re.split(r"[^a-z]+", n)
    rel_trigs = [t for t in tokens if t]

    return NLQ(raw=s, text_norm=n, intent=intent, entity_strings=titles, relation_triggers=rel_trigs)