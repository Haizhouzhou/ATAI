from __future__ import annotations
from dataclasses import dataclass
import re
from typing import List

from .utils import normalize_text

# Simple keyword triggers
EMBED_TRIGGERS = ["embedding"]
FACT_TRIGGERS = ["factual"]

@dataclass
class NLQ:
    raw: str
    text_norm: str
    intent: str                 # "factual" | "embedding" | "both"
    entity_strings: List[str]
    relation_triggers: List[str]

FANCY = {"“":"\"", "”":"\"", "‘":"'", "’":"'", "«":"\"", "»":"\""}
def _normalize_quotes(s: str) -> str:
    return "".join(FANCY.get(ch, ch) for ch in s)

def _extract_quoted_titles(s: str) -> List[str]:
    # support "..."、'...'、‘…’
    s = _normalize_quotes(s)
    out: List[str] = []
    for m in re.finditer(r"['\"]([^'\"]+)['\"]", s):
        out.append(m.group(1).strip())
    return out

def parse_nlq(q: str) -> NLQ:
    s = q.strip()
    # n = normalize_text(s)
    s_norm_quotes = _normalize_quotes(s)
    n = normalize_text(s_norm_quotes)

    # intent
    intent = "both"
    if any(t in n for t in EMBED_TRIGGERS):
        intent = "embedding"
    if any(t in n for t in FACT_TRIGGERS):
        intent = "factual" 


    # entities: quoted titles first; fallback to capitalized spans
    titles = _extract_quoted_titles(s)
    if not titles:
        caps = re.findall(r"([A-Z][A-Za-z0-9:.'\-]*(?:\s+[A-Z][A-Za-z0-9:.'\-]*)+)", q)
        titles = caps

    # relation triggers: tokenize to simple lower tokens
    tokens = re.split(r"[^a-z]+", n.lower())
    rel_trigs = [t for t in tokens if t]

    return NLQ(raw=s, text_norm=n, intent=intent, entity_strings=titles, relation_triggers=rel_trigs)