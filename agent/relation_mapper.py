from __future__ import annotations

"""
Relation mapping helpers.

This module takes a natural-language question and tries to decide
*which* knowledge-graph relation the user is asking about, e.g.:

    "Who directed Inception?"          -> director
    "When was Titanic released?"       -> publication_date
    "From which country is Amélie?"    -> country_of_origin

The actual mapping from canonical relation keys to Wikidata property
URIs lives in `agent.constants.RELATION_DEFS`.
"""

from dataclasses import dataclass
from typing import Optional

from . import constants as C


@dataclass
class RelationMatch:
    """
    Result of relation mapping.
    """

    key: str               # e.g. "director"
    property_uri: str      # wdt: property URI
    label: str             # human-readable label for answers
    score: float           # simple confidence score in [0, 1]


class RelationMapper:
    """
    Very lightweight keyword-based relation mapper.

    For the exam this is fully sufficient and very robust. If in the
    future you want to plug in a classifier model, this is the place
    to extend.
    """

    def __init__(self) -> None:
        # Pre-compute a lowercase synonym index for quick matching.
        self._synonym_index = self._build_synonym_index()

    def _build_synonym_index(self):
        index = {}
        for key, rel in C.RELATION_DEFS.items():
            for phrase in rel.synonyms:
                phrase_lc = phrase.lower()
                index.setdefault(phrase_lc, []).append(key)
        return index

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def infer_relation(self, question: str) -> Optional[RelationMatch]:
        """
        Guess the most likely relation for this question.

        The algorithm is simple:

            1. Lowercase the question.
            2. For every known synonym phrase, check if it occurs.
            3. Count how many times each relation key is triggered.
            4. Pick the best relation based on:
                - count
                - manual priority (RELATION_KEYS_BY_PRIORITY)
            5. Convert the best key into a RelationMatch object.

        If nothing matches, return None.
        """
        if not question:
            return None

        q = question.lower()

        # Count matches per relation key
        counts = {key: 0 for key in C.RELATION_DEFS.keys()}

        for phrase, keys in self._synonym_index.items():
            if phrase in q:
                for key in keys:
                    counts[key] += 1

        # Filter non-zero counts
        active = {k: c for k, c in counts.items() if c > 0}
        if not active:
            return None

        # Choose best according to count and priority order
        best_key: Optional[str] = None
        best_score: int = -1

        for key in C.RELATION_KEYS_BY_PRIORITY:
            if key in active and active[key] > best_score:
                best_key = key
                best_score = active[key]

        if best_key is None:
            # fallback: any key with max count
            best_key = max(active, key=active.get)
            best_score = active[best_key]

        rel_def = C.RELATION_DEFS[best_key]
        # Score is crude: 1.0 for at least one match, scaled by count otherwise.
        score = min(1.0, float(best_score) / 3.0)

        return RelationMatch(
            key=rel_def.key,
            property_uri=rel_def.property_uri,
            label=rel_def.label,
            score=score,
        )
