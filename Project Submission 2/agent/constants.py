from __future__ import annotations
from dataclasses import dataclass
from typing import Dict, List

@dataclass
class RelationSpec:
    key: str               # internal key
    predicate: str         # full IRI (wdt:Pxx)
    surface_forms: List[str]

RELATIONS: Dict[str, RelationSpec] = {
    "director": RelationSpec(
        key="director",
        predicate="http://www.wikidata.org/prop/direct/P57",
        surface_forms=["director", "directed by", "who directed", "filmmaker", "made by"],
    ),
    "screenwriter": RelationSpec(
        key="screenwriter",
        predicate="http://www.wikidata.org/prop/direct/P58",
        surface_forms=["screenwriter", "written by", "who wrote", "writer"],
    ),
    "publication_date": RelationSpec(
        key="publication_date",
        predicate="http://www.wikidata.org/prop/direct/P577",
        surface_forms=["release date", "publication date", "when released", "come out", "premiere"],
    ),
    "genre": RelationSpec(
        key="genre",
        predicate="http://www.wikidata.org/prop/direct/P136",
        surface_forms=["genre", "category", "type of film"],
    ),
    "duration": RelationSpec(
        key="duration",
        predicate="http://www.wikidata.org/prop/direct/P2047",
        surface_forms=["duration", "runtime", "length"],
    ),
    "cast_member": RelationSpec(
        key="cast_member",
        predicate="http://www.wikidata.org/prop/direct/P161",
        surface_forms=["actor", "actress", "cast", "starring", "who played"],
    ),
    "original_language": RelationSpec(
        key="original_language",
        predicate="http://www.wikidata.org/prop/direct/P364",
        surface_forms=["language", "original language"],
    ),
    "production_company": RelationSpec(
        key="production_company",
        predicate="http://www.wikidata.org/prop/direct/P272",
        surface_forms=["production company", "studio"],
    ),
    "mpaa_rating": RelationSpec(
        key="mpaa_rating",
        predicate="http://www.wikidata.org/prop/direct/P1657",
        surface_forms=["mpaa", "mpaa rating", "rating"],
    ),
}