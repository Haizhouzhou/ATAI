from __future__ import annotations

"""
High-level recommendation engine.

This module uses MovieEmbeddingIndex (from embedding_executor.py) to provide
movie recommendations based on:

    - a single seed film URI
    - multiple liked films (watchlist)

It does not know anything about the chatbot UI or natural language parsing.
It only deals with URIs and returns plain Python data structures.
"""

from typing import Dict, Iterable, List, Optional

from .embedding_executor import (
    EmbeddingConfig,
    MovieEmbeddingIndex,
    get_global_movie_index,
)
from .graph_executor import get_global_graph_executor, GraphExecutor


class RecommendationEngine:
    """
    Embedding-based movie recommendation engine.

    The engine itself is stateless apart from holding references to:
      - MovieEmbeddingIndex (embeddings + descriptions)
      - GraphExecutor        (labels / titles)
    """

    def __init__(
        self,
        index: Optional[MovieEmbeddingIndex] = None,
        config: Optional[EmbeddingConfig] = None,
        graph: Optional[GraphExecutor] = None,
    ) -> None:
        # Use the provided index if given; otherwise, obtain the global singleton.
        self.index: MovieEmbeddingIndex = index or get_global_movie_index(config=config)
        # We use GraphExecutor only for pretty labels. It is safe and cheap to
        # call get_global_graph_executor() because it is a singleton.
        self.graph: GraphExecutor = graph or get_global_graph_executor()

    # ------------------------------------------------------------------
    # 1. Single-seed recommendations
    # ------------------------------------------------------------------

    def recommend_similar_films(
        self,
        seed_uri: str,
        top_k: int = 10,
        min_similarity: float = 0.0,
        include_seed: bool = False,
    ) -> List[Dict[str, object]]:
        """
        Recommend films similar to a single seed film.

        Parameters
        ----------
        seed_uri:
            URI of the film that the user likes.
        top_k:
            Number of neighbours to return.
        min_similarity:
            Optional minimum cosine similarity threshold.
        include_seed:
            If True, the seed film can appear in the result list.

        Returns
        -------
        List[Dict[str, object]]
            List of candidate films with keys:
                - 'uri'
                - 'index'
                - 'similarity'
                - 'description' (from movie_plots.tsv, if available)
                - 'title'       (from entity_titles, if available)
        """
        neighbours = self.index.neighbors_for_film(
            uri=seed_uri,
            top_k=top_k,
            min_similarity=min_similarity,
            include_seed=include_seed,
        )

        for item in neighbours:
            uri = item.get("uri")
            if isinstance(uri, str) and uri in self.graph.entity_uri_to_idx:
                title = self.graph.get_label(self.graph.entity_uri_to_idx[uri])
            else:
                title = None
            item["title"] = title

        return neighbours

    # ------------------------------------------------------------------
    # 2. Multi-seed recommendations (watchlist / liked films)
    # ------------------------------------------------------------------

    def recommend_from_watchlist(
        self,
        liked_uris: Iterable[str],
        top_k_per_seed: int = 20,
        final_top_k: int = 10,
        min_similarity: float = 0.0,
    ) -> List[Dict[str, object]]:
        """
        Recommend films based on multiple liked films.

        Strategy (simple but effective):
            1. For each liked film:
                - retrieve top_k_per_seed neighbours (excluding the seed)
            2. Aggregate all candidates into a dictionary keyed by URI
               and sum their similarity scores.
            3. Remove any films that are in the liked_uris set.
            4. Sort by aggregated similarity and return the top final_top_k.

        Parameters
        ----------
        liked_uris:
            Iterable of film URIs that the user considers positive examples.
        top_k_per_seed:
            Number of neighbours to consider for each individual seed.
        final_top_k:
            Number of final recommendations to return.
        min_similarity:
            Minimum cosine similarity threshold for each neighbour.

        Returns
        -------
        List[Dict[str, object]]
            Each entry has fields:
                - 'uri'
                - 'score'        (aggregated similarity)
                - 'support'      (how many seeds contributed)
                - 'description'  (from movie_plots.tsv, if available)
                - 'title'        (from entity_titles, if available)
        """
        # Keep only URIs that are actually present in the embedding index.
        liked_set = {uri for uri in liked_uris if self.index.has_uri(uri)}

        if not liked_set:
            return []

        candidate_scores: Dict[str, float] = {}
        candidate_support: Dict[str, int] = {}

        for seed_uri in liked_set:
            neighbours = self.index.neighbors_for_film(
                uri=seed_uri,
                top_k=top_k_per_seed,
                min_similarity=min_similarity,
                include_seed=False,
            )
            for nb in neighbours:
                uri = nb["uri"]
                sim = float(nb["similarity"])

                # Skip films the user already liked
                if uri in liked_set:
                    continue

                candidate_scores[uri] = candidate_scores.get(uri, 0.0) + sim
                candidate_support[uri] = candidate_support.get(uri, 0) + 1

        if not candidate_scores:
            return []

        # Sort candidates by aggregated similarity descending
        sorted_candidates = sorted(
            candidate_scores.items(),
            key=lambda kv: -kv[1],
        )

        results: List[Dict[str, object]] = []
        for uri, score in sorted_candidates[:final_top_k]:
            # We can reuse the description logic from MovieEmbeddingIndex
            description = self.index.get_description(uri)

            if uri in self.graph.entity_uri_to_idx:
                title = self.graph.get_label(self.graph.entity_uri_to_idx[uri])
            else:
                title = None

            results.append(
                {
                    "uri": uri,
                    "score": float(score),
                    "support": candidate_support.get(uri, 0),
                    "description": description,
                    "title": title,
                }
            )

        return results
