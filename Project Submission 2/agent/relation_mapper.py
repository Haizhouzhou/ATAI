from __future__ import annotations
from typing import Optional, Iterable
from .constants import RELATIONS, RelationSpec

# keyword buckets → relation key
KEYWORDS = {
    "director": {
        "director", "directed", "filmmaker", "made", "who directed", "who is the director",
    },
    "screenwriter": {
        "screenwriter", "writer", "wrote", "written", "who wrote", "who is the screenwriter",
    },
    "publication_date": {
        "release", "when did it come out", "publication", "premiere", "come out",
        "release date", "when was released", "what is the release date of",
    },
    "genre": {
        "genre", "category", "type", "kind of film", "type of movie", "what genre",
    },
    "duration": {
        "duration", "runtime", "length", "how long",
    },
    "cast_member": {
        "actor", "actress", "cast", "starring", "played", "who is in the cast", "who played",
    },
    "original_language": {
        "language", "what language",
    },
    "production_company": {
        "production", "studio", "company", "made by", "produced",
    },
    "mpaa_rating": {
        "mpaa", "rating", "age rating",
    },
    "country": {  # adding country 
        "from which country", "country", "origin", "where is it from", "what country","produced in", "filmed in", "where was it filmed", "production location"
    },
}


class RelationMapper:
    def __init__(self):
        # Build reverse index from surface forms → relation key (exact phrase)
        self.surface_to_key = {}
        for key, spec in RELATIONS.items():
            for s in spec.surface_forms:
                self.surface_to_key[s.lower()] = key

    def _token_set(self, tokens: Iterable[str]) -> set[str]:
        return {t for t in tokens if t}

    def map_relation(self, tokens) -> Optional[RelationSpec]:
        # 1) Exact phrase (legacy)
        for t in tokens:
            key = self.surface_to_key.get(t)
            if key:
                return RELATIONS[key]

        # 2) Substring of joined tokens (legacy)
        joined = " ".join(tokens)
        for key, spec in RELATIONS.items():
            for s in spec.surface_forms:
                if s in joined:
                    return spec

        # 3) Keyword bucket match (new, robust)
        T = self._token_set(tokens)
        # Normalize simple bigrams like ("come","out") → treat as release
        if {"come", "out"} <= T or {"came", "out"} <= T:
            return RELATIONS["publication_date"]
        # Heuristic: if "when" appears alongside release/premiere words → date
        if "when" in T and ({"release", "released", "premiere", "premiered"} & T):
            return RELATIONS["publication_date"]

        for key, words in KEYWORDS.items():
            if T & words:
                return RELATIONS[key]

        return None