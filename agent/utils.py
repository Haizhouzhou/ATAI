from __future__ import annotations

"""
Miscellaneous utility functions used across the agent.

The most important one is `normalize_title`, which provides a stable,
lowercased representation of movie / person titles for indexing.
"""

import re
import unicodedata
from typing import Iterable, List, TypeVar

T = TypeVar("T")


def normalize_title(title: str) -> str:
    """
    Normalize a movie or person title for indexing.

    Operations:
        - convert to string and strip
        - normalize unicode (NFKC)
        - lowercase
        - remove leading articles (the/a/an)
        - replace punctuation with spaces
        - collapse multiple spaces

    This function is intentionally simple and deterministic; it is used
    both when building the index and when doing lookups.
    """
    if title is None:
        return ""

    s = str(title)
    s = s.strip()
    if not s:
        return ""

    # Unicode normalization
    s = unicodedata.normalize("NFKC", s)

    # unify quotes
    s = (
        s.replace("’", "'")
        .replace("‘", "'")
        .replace("“", '"')
        .replace("”", '"')
    )

    # strip outer quotes
    if (s.startswith("'") and s.endswith("'")) or (s.startswith('"') and s.endswith('"')):
        s = s[1:-1].strip()

    s = s.lower()

    # remove leading English articles
    for pref in ("the ", "a ", "an "):
        if s.startswith(pref):
            s = s[len(pref) :]

    # replace punctuation with spaces
    s = re.sub(r"[^\w\s]", " ", s)

    # collapse whitespace
    s = re.sub(r"\s+", " ", s).strip()

    return s


def unique_preserve_order(items: Iterable[T]) -> List[T]:
    """
    Return a new list with duplicates removed, preserving the original order.
    """
    seen = set()
    result: List[T] = []
    for item in items:
        if item in seen:
            continue
        seen.add(item)
        result.append(item)
    return result
