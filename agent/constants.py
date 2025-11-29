from __future__ import annotations

"""
Global constants for the movie agent.

This file has three responsibilities:

1. File-system paths to the ATAI dataset (metadata, embeddings, additional).
2. Wikidata property URIs we care about (director, genre, etc.).
3. Small tables for relation mapping and question classification.

We keep everything here light-weight: only stdlib imports and no I/O.
"""

from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List

# ---------------------------------------------------------------------
# 1. Paths
# ---------------------------------------------------------------------

# Resolve project root:   <...>/Final Project Submission/
PROJECT_ROOT: Path = Path(__file__).resolve().parents[1]

METADATA_DIR: Path = PROJECT_ROOT / "metadata"
EMBEDDINGS_DIR: Path = PROJECT_ROOT / "embeddings"
ADDITIONAL_DIR: Path = PROJECT_ROOT / "additional"

# Metadata files
ENTITY_IDS_TSV: Path = METADATA_DIR / "entity_ids_ordered.tsv"
RELATION_IDS_TSV: Path = METADATA_DIR / "relation_ids_ordered.tsv"
FILM_ENTITIES_TSV: Path = METADATA_DIR / "film_entities.tsv"
ENTITY_TITLES_TSV: Path = METADATA_DIR / "entity_titles.tsv"
MOVIE_PLOTS_TSV: Path = METADATA_DIR / "movie_plots.tsv"

# Triple IDs (h_id, r_id, t_id)
ID_TRIPLES_TSV: Path = METADATA_DIR / "rfc_triples_ids.tsv"

# Optional original RDF graph and additional metadata (if present)
GRAPH_TSV_PATH: Path = METADATA_DIR / "graph.tsv"
PLOTS_CSV_PATH: Path = ADDITIONAL_DIR / "plots.csv"
IMAGES_JSON_PATH: Path = ADDITIONAL_DIR / "images.json"

# Embedding file
RFC_ENTITY_EMBEDS_NPY: Path = EMBEDDINGS_DIR / "RFC_entity_embeds.npy"

# Wikidata entity URI prefix
WD_ENTITY: str = "http://www.wikidata.org/entity/"

# ---------------------------------------------------------------------
# 2. Wikidata property URIs (wdt: namespace)
# ---------------------------------------------------------------------

# These must match relation_ids_ordered.tsv.

P_DIRECTOR: str = "http://www.wikidata.org/prop/direct/P57"
P_SCREENWRITER: str = "http://www.wikidata.org/prop/direct/P58"
P_COMPOSER: str = "http://www.wikidata.org/prop/direct/P86"
P_GENRE: str = "http://www.wikidata.org/prop/direct/P136"
P_COUNTRY_OF_ORIGIN: str = "http://www.wikidata.org/prop/direct/P495"
P_RELEASE_DATE: str = "http://www.wikidata.org/prop/direct/P577"
P_NOMINATED_FOR: str = "http://www.wikidata.org/prop/direct/P1411"
P_INSTANCE_OF: str = "http://www.wikidata.org/prop/direct/P31"
P_ORIGINAL_LANGUAGE: str = "http://www.wikidata.org/prop/direct/P364"
P_CAST_MEMBER: str = "http://www.wikidata.org/prop/direct/P161"

# ---------------------------------------------------------------------
# 3. Basic responses / messages
# ---------------------------------------------------------------------

RESPONSE_NO_KNOWLEDGE: str = (
    "I am not fully sure based on the data I have. "
    "Please be a bit more specific or mention a concrete movie or person."
)

RESPONSE_ERROR: str = (
    "I am sorry, something went wrong while processing your request. "
    "Please try again in a moment."
)

# ---------------------------------------------------------------------
# 4. Relation definitions for natural-language mapping
# ---------------------------------------------------------------------


@dataclass(frozen=True)
class RelationDef:
    """
    Canonical definition of a KG relation we support.

    Attributes
    ----------
    key:
        Short canonical key (used inside the agent), e.g. "director".
    label:
        Human-readable label for answer formatting, e.g. "director".
    property_uri:
        Wikidata direct-property URI used in the ID triples.
    answer_type:
        Rough type of the answer: "entity", "date", "number", "literal".
    synonyms:
        Lowercase phrases that, if found in the user question, indicate
        this relation.
    """

    key: str
    label: str
    property_uri: str
    answer_type: str
    synonyms: List[str]


RELATION_DEFS: Dict[str, RelationDef] = {
    "director": RelationDef(
        key="director",
        label="director",
        property_uri=P_DIRECTOR,
        answer_type="entity",
        synonyms=[
            "director",
            "directed by",
            "who directed",
            "who is the director",
            "film director",
        ],
    ),
    "publication_date": RelationDef(
        key="publication_date",
        label="release date",
        property_uri=P_RELEASE_DATE,
        answer_type="date",
        synonyms=[
            "release date",
            "when was",
            "what year",
            "year was it released",
            "released in",
        ],
    ),
    "country_of_origin": RelationDef(
        key="country_of_origin",
        label="country of origin",
        property_uri=P_COUNTRY_OF_ORIGIN,
        answer_type="entity",
        synonyms=[
            "from what country",
            "which country",
            "country of origin",
            "country is the movie from",
            "origin country",
        ],
    ),
    "language": RelationDef(
        key="language",
        label="language",
        property_uri=P_ORIGINAL_LANGUAGE,
        answer_type="entity",
        synonyms=[
            "language",
            "original language",
            "in what language",
            "spoken language",
        ],
    ),
    "award": RelationDef(
        key="award",
        label="award received",
        property_uri=P_NOMINATED_FOR,
        answer_type="entity",
        synonyms=[
            "award",
            "awards",
            "oscar",
            "prize",
            "won any awards",
            "nominated for",
        ],
    ),
    "production_company": RelationDef(
        key="production_company",
        label="production company",
        # there is a dedicated property for production company, but if your
        # relation_ids table does not contain it, this entry will simply never
        # be used by the graph executor.
        property_uri="http://www.wikidata.org/prop/direct/P272",
        answer_type="entity",
        synonyms=[
            "production company",
            "producer",
            "produced by",
            "studio",
        ],
    ),
    "screenwriter": RelationDef(
        key="screenwriter",
        label="screenwriter",
        property_uri=P_SCREENWRITER,
        answer_type="entity",
        synonyms=[
            "screenwriter",
            "screenplay",
            "writer",
            "written by",
        ],
    ),
    "composer": RelationDef(
        key="composer",
        label="composer",
        property_uri=P_COMPOSER,
        answer_type="entity",
        synonyms=[
            "composer",
            "music by",
            "score by",
            "who composed",
        ],
    ),
    "cast_member": RelationDef(
        key="cast_member",
        label="cast member",
        property_uri=P_CAST_MEMBER,
        answer_type="entity",
        synonyms=[
            "starring",
            "cast",
            "actor",
            "actress",
            "who plays",
            "who stars",
        ],
    ),
    "genre": RelationDef(
        key="genre",
        label="genre",
        property_uri=P_GENRE,
        answer_type="entity",
        synonyms=[
            "genre",
            "kind of movie",
            "type of film",
            "what kind of film",
        ],
    ),
}

# Priority list: more specific / common relations first
RELATION_KEYS_BY_PRIORITY: List[str] = [
    "director",
    "screenwriter",
    "composer",
    "cast_member",
    "publication_date",
    "country_of_origin",
    "language",
    "production_company",
    "award",
    "genre",
]

# ---------------------------------------------------------------------
# 5. Keyword sets for question classification
# ---------------------------------------------------------------------

RECOMMENDATION_KEYWORDS: List[str] = [
    "recommend",
    "similar movies",
    "similar films",
    "something like",
    "if i like",
    "i like",
    "i really enjoyed",
    "what else should i watch",
    "suggest some movies",
]

MULTIMEDIA_KEYWORDS: List[str] = [
    "show me a picture",
    "show me the poster",
    "image of",
    "picture of",
    "photo of",
    "poster of",
    "cover of",
]

FACTUAL_KEYWORDS: List[str] = [
    "who ",
    "when ",
    "where ",
    "which ",
    "what ",
    "from what country",
    "in what language",
    "release date",
    "director",
    "language",
    "award",
    "production company",
]

# Popular genre / language hints for the preference parser
POPULAR_GENRE_KEYWORDS: List[str] = [
    "comedy",
    "drama",
    "romance",
    "romantic",
    "thriller",
    "horror",
    "musical",
    "action",
    "science fiction",
    "sci-fi",
    "war",
    "western",
    "animation",
    "animated",
    "documentary",
]

POPULAR_LANGUAGE_KEYWORDS: List[str] = [
    "english",
    "french",
    "german",
    "italian",
    "spanish",
    "japanese",
    "korean",
    "chinese",
    "mandarin",
    "hindi",
]
