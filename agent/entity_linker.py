from __future__ import annotations

"""
Entity linking utilities.

This module provides a small wrapper around GraphExecutor that can map
movie / person names in natural language questions to entity indices
and URIs in our knowledge graph.

Design goals:
    - keep it lightweight (no model loading here),
    - rely on the robust title / label handling in GraphExecutor,
    - provide simple Python objects that other modules can use.
"""

from dataclasses import dataclass
from typing import List, Optional

from .graph_executor import GraphExecutor, get_global_graph_executor


@dataclass
class LinkedEntity:
    """
    Representation of an entity found in the user question.
    """

    idx: int
    uri: str
    label: str
    type: str  # "film", "person", or "other"


class EntityLinker:
    """
    High-level helper for entity linking.

    Internally this uses GraphExecutor.find_entities_by_normalized_title,
    which already performs robust matching over labels and titles loaded
    from entity_titles.tsv, movie_plots.tsv, and images.json.
    """

    def __init__(self, graph: Optional[GraphExecutor] = None) -> None:
        self.graph: GraphExecutor = graph or get_global_graph_executor()

    # ------------------------------------------------------------------
    # Core linking
    # ------------------------------------------------------------------

    def link_entities(self, text: str, max_candidates: int = 10) -> List[LinkedEntity]:
        """
        Try to find entities mentioned in `text`.

        For now this is implemented as a single-title lookup: we assume
        the question contains one main movie or person name and run the
        title matcher once. In practice this already works quite well
        for the exam queries.

        Parameters
        ----------
        text:
            User question in plain English.
        max_candidates:
            Maximum number of entities to return.

        Returns
        -------
        List[LinkedEntity]
            Sorted by a simple heuristic: films first, then persons,
            then others. Within each group the original order from the
            matcher is preserved.
        """
        idxs = self.graph.find_entities_by_normalized_title(text)
        if not idxs:
            return []

        seen: set[int] = set()
        films: List[LinkedEntity] = []
        persons: List[LinkedEntity] = []
        others: List[LinkedEntity] = []

        for idx in idxs:
            if idx in seen:
                continue
            seen.add(idx)

            if not (0 <= idx < len(self.graph.entity_idx_to_uri)):
                continue

            uri = self.graph.entity_idx_to_uri[idx]
            label = self.graph.get_label(idx)

            if idx in self.graph.film_idxs:
                target = films
                ent_type = "film"
            elif idx in self.graph.person_candidate_idxs:
                target = persons
                ent_type = "person"
            else:
                target = others
                ent_type = "other"

            target.append(
                LinkedEntity(
                    idx=idx,
                    uri=uri,
                    label=label,
                    type=ent_type,
                )
            )

            if len(films) + len(persons) + len(others) >= max_candidates:
                break

        # final ordering: films -> persons -> others
        return films + persons + others

    # Convenience helpers used by the composer / preference parser
    # ------------------------------------------------------------------

    def find_best_film(self, text: str) -> Optional[LinkedEntity]:
        """
        Return the single best film match, or None.
        """
        for ent in self.link_entities(text, max_candidates=20):
            if ent.type == "film":
                return ent
        return None

    def find_all_films(self, text: str, max_candidates: int = 10) -> List[LinkedEntity]:
        """
        Return all film matches in the text (up to max_candidates).
        """
        return [
            ent
            for ent in self.link_entities(text, max_candidates=max_candidates)
            if ent.type == "film"
        ]


# Module-level singleton
_GLOBAL_LINKER: Optional[EntityLinker] = None


def get_global_entity_linker() -> EntityLinker:
    """
    Return a process-wide singleton EntityLinker.
    """
    global _GLOBAL_LINKER
    if _GLOBAL_LINKER is None:
        _GLOBAL_LINKER = EntityLinker()
    return _GLOBAL_LINKER
