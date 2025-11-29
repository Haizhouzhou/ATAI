from __future__ import annotations

"""
Very lightweight natural-language query classification.

We only distinguish between a few coarse types:

    - FACTUAL: one-hop graph question ("Who directed Titanic?")
    - RECOMMENDATION: preference / embedding-based ("I like Inception...")
    - MULTIMEDIA: explicitly asking for images/posters
    - OTHER: anything else

This is deliberately rule-based and deterministic.
"""

from dataclasses import dataclass
from enum import Enum, auto
from typing import Optional

from . import constants as C


class QueryType(Enum):
    FACTUAL = auto()
    RECOMMENDATION = auto()
    MULTIMEDIA = auto()
    OTHER = auto()


@dataclass
class NLQAnalysis:
    """
    Result of the query classification.
    """

    query_type: QueryType
    original_text: str


class NLQClassifier:
    """
    Simple keyword-based classifier.

    If you later want to plug in an ML model, you can keep this class
    as thin wrapper and swap out the implementation.
    """

    def classify(self, question: str) -> NLQAnalysis:
        if not question:
            return NLQAnalysis(QueryType.OTHER, question or "")

        q = question.lower()

        # 1) Multimedia questions have highest priority
        if any(kw in q for kw in C.MULTIMEDIA_KEYWORDS):
            return NLQAnalysis(QueryType.MULTIMEDIA, question)

        # 2) Recommendations (watchlist / "I like ..." style)
        if any(kw in q for kw in C.RECOMMENDATION_KEYWORDS):
            return NLQAnalysis(QueryType.RECOMMENDATION, question)

        # 3) Factual questions: WH-word or explicit relation keywords
        wh_words = ("who", "when", "where", "which", "what")
        if any(w + " " in q for w in wh_words) or any(
            kw in q for kw in C.FACTUAL_KEYWORDS
        ):
            return NLQAnalysis(QueryType.FACTUAL, question)

        # 4) Fallback: treat as OTHER
        return NLQAnalysis(QueryType.OTHER, question)
