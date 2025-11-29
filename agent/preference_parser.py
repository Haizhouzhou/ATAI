from __future__ import annotations

"""
Preference parser for recommendation queries.

Given a natural-language description like:

    "I loved Inception and Interstellar, recommend similar movies,
     preferably in German or French."

we try to extract:

    - liked film URIs (based on entity linking)
    - genre hints (keywords)
    - language hints (keywords)
"""

from dataclasses import dataclass
from typing import List, Optional

from . import constants as C
from .entity_linker import EntityLinker, LinkedEntity, get_global_entity_linker
from .utils import unique_preserve_order


@dataclass
class UserPreferences:
    liked_film_uris: List[str]
    liked_film_titles: List[str]
    genre_keywords: List[str]
    language_keywords: List[str]


class PreferenceParser:
    """
    Lightweight, rule-based parser for recommendation preferences.
    """

    def __init__(self, linker: Optional[EntityLinker] = None) -> None:
        self.linker: EntityLinker = linker or get_global_entity_linker()

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def parse(self, text: str, max_films: int = 5) -> UserPreferences:
        """
        Parse user preferences from text.

        Returns a UserPreferences object where lists may be empty but
        are never None.
        """
        films = self._extract_films(text, max_films=max_films)
        film_uris = [f.uri for f in films]
        film_titles = [f.label for f in films]

        genres = self._extract_genre_keywords(text)
        languages = self._extract_language_keywords(text)

        return UserPreferences(
            liked_film_uris=film_uris,
            liked_film_titles=film_titles,
            genre_keywords=genres,
            language_keywords=languages,
        )

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _extract_films(self, text: str, max_films: int) -> List[LinkedEntity]:
        # Use the entity linker to find movies mentioned in the text.
        all_ents = self.linker.link_entities(text, max_candidates=20)
        films = [e for e in all_ents if e.type == "film"]
        # preserve order, truncate
        seen = set()
        result: List[LinkedEntity] = []
        for f in films:
            if f.uri in seen:
                continue
            seen.add(f.uri)
            result.append(f)
            if len(result) >= max_films:
                break
        return result

    def _extract_genre_keywords(self, text: str) -> List[str]:
        q = text.lower()
        found = []
        for g in C.POPULAR_GENRE_KEYWORDS:
            if g in q:
                found.append(g)
        return unique_preserve_order(found)

    def _extract_language_keywords(self, text: str) -> List[str]:
        q = text.lower()
        found = []
        for lang in C.POPULAR_LANGUAGE_KEYWORDS:
            if lang in q:
                found.append(lang)
        return unique_preserve_order(found)
