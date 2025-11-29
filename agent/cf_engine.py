from __future__ import annotations

"""
Very small "collaborative filtering" engine.

For this project we do not have real user rating data, so this module
acts as a thin alias for the embedding-based RecommendationEngine.
Keeping it separate makes it easy to plug in a real CF model later.
"""

from typing import Iterable, List, Optional, Dict

from .recommendation_engine import RecommendationEngine
from .embedding_executor import EmbeddingConfig, MovieEmbeddingIndex


class CollaborativeFilteringEngine:
    """
    Wrapper around RecommendationEngine with a CF-flavoured name.
    """

    def __init__(
        self,
        rec_engine: Optional[RecommendationEngine] = None,
        index: Optional[MovieEmbeddingIndex] = None,
        config: Optional[EmbeddingConfig] = None,
    ) -> None:
        self.rec_engine: RecommendationEngine = rec_engine or RecommendationEngine(
            index=index, config=config
        )

    def recommend(
        self,
        liked_uris: Iterable[str],
        final_top_k: int = 10,
    ) -> List[Dict[str, object]]:
        """
        Return recommendations for a watchlist of liked URIs.

        The result structure matches RecommendationEngine.recommend_from_watchlist.
        """
        return self.rec_engine.recommend_from_watchlist(
            liked_uris=liked_uris,
            top_k_per_seed=20,
            final_top_k=final_top_k,
            min_similarity=0.0,
        )
