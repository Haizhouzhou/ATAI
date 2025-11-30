import os

# --- File Paths ---
DATA_DIR = os.getenv('DATA_DIR', '/space_mounts/atai-hs25/dataset')
CACHE_DIR = os.getenv('CACHE_DIR', '.cache')

KG_PATH = os.path.join(DATA_DIR, 'graph.nt')
ENTITY_INDEX_PATH = os.path.join(DATA_DIR, 'embeddings', 'entity_ids.del')
RELATION_INDEX_PATH = os.path.join(DATA_DIR, 'embeddings', 'relation_ids.del')
LABEL_INDEX_PATH = os.path.join(CACHE_DIR, 'label_index.pkl')

# --- SPARQL Prefixes ---
PREFIXES = """
    PREFIX ddis: <http://ddis.ch/atai/>
    PREFIX wd: <http://www.wikidata.org/entity/>
    PREFIX wdt: <http://www.wikidata.org/prop/direct/>
    PREFIX schema: <http://schema.org/>
    PREFIX rdfs: <http://www.w3.org/2000/01/rdf-schema#>
    PREFIX p: <http://www.wikidata.org/prop/>
    PREFIX ps: <http://www.wikidata.org/prop/statement/>
    PREFIX psv: <http://www.wikidata.org/prop/statement/value/>
"""

# --- PREDICATE MAPPING (FULL URIs for Embedding Compatibility) ---
PREDICATE_MAP = {
    # Facts
    "director": "http://www.wikidata.org/prop/direct/P57",
    "directed by": "http://www.wikidata.org/prop/direct/P57",
    "directed": "http://www.wikidata.org/prop/direct/P57",
    "screenwriter": "http://www.wikidata.org/prop/direct/P58",
    "written by": "http://www.wikidata.org/prop/direct/P58",
    "writer": "http://www.wikidata.org/prop/direct/P58",
    "cast member": "http://www.wikidata.org/prop/direct/P161",
    "actor": "http://www.wikidata.org/prop/direct/P161",
    "actress": "http://www.wikidata.org/prop/direct/P161",
    "starring": "http://www.wikidata.org/prop/direct/P161",
    "cast": "http://www.wikidata.org/prop/direct/P161",
    "producer": "http://www.wikidata.org/prop/direct/P162",
    "produced by": "http://www.wikidata.org/prop/direct/P162",
    "composer": "http://www.wikidata.org/prop/direct/P86",
    "music by": "http://www.wikidata.org/prop/direct/P86",
    "country of origin": "http://www.wikidata.org/prop/direct/P495",
    "country": "http://www.wikidata.org/prop/direct/P495",
    "publication date": "http://www.wikidata.org/prop/direct/P577",
    "release date": "http://www.wikidata.org/prop/direct/P577",
    "released": "http://www.wikidata.org/prop/direct/P577",
    "come out": "http://www.wikidata.org/prop/direct/P577",
    "year": "http://www.wikidata.org/prop/direct/P577",
    "genre": "http://www.wikidata.org/prop/direct/P136",
    "genres": "http://www.wikidata.org/prop/direct/P136",
    "nominated for": "http://www.wikidata.org/prop/direct/P1411",
    "nomination": "http://www.wikidata.org/prop/direct/P1411",
    "award": "http://www.wikidata.org/prop/direct/P166",
    "won": "http://www.wikidata.org/prop/direct/P166",
    "rating": "http://ddis.ch/atai/rating",
    "score": "http://ddis.ch/atai/rating",
    "mpaa": "http://www.wikidata.org/prop/direct/P1657",
    "mpaa rating": "http://www.wikidata.org/prop/direct/P1657",
    "content rating": "http://www.wikidata.org/prop/direct/P1657",
    "language": "http://www.wikidata.org/prop/direct/P364",
    "languages": "http://www.wikidata.org/prop/direct/P364",
    "spoken in": "http://www.wikidata.org/prop/direct/P364",
    
    # Multimedia
    "image": "http://www.wikidata.org/prop/direct/P18",
    "picture": "http://www.wikidata.org/prop/direct/P18",
    "poster": "http://www.wikidata.org/prop/direct/P18",
    "photo": "http://www.wikidata.org/prop/direct/P18",
    
    # Recs
    "part of series": "http://www.wikidata.org/prop/direct/P179",
}

# --- INTENT PARSING ---
RECOMMENDATION_KEYWORDS = [
    "recommend", "suggest", "give me", "find me", "looking for",
    "like", "similar to", "another", "more movies", "movies like"
]

QA_KEYWORDS = [
    "who", "what", "when", "where", "which", "how many", "list", "is", "did", "does"
]

FOLLOW_UP_KEYWORDS = ["another", "more", "like that", "what else"]
NEGATION_KEYWORDS = ["not", "don't", "without", "except", "avoid", "no"]

PREFERENCE_KEYWORDS = {
    "genre": ["genre", "type of movie", "about", "is a"],
    "actor": ["actor", "starring", "with", "features", "cast"],
    "director": ["director", "directed by", "by"],
    "language": ["language", "speak", "spoken"], 
}

SUPPORTED_LANGUAGES_REGEX = r'\b(French|German|Spanish|English|Italian|Japanese|Korean|Chinese)\b'

MULTIMEDIA_KEYWORDS = [
    "show", "picture", "image", "photo", "look like", "poster", "cover", "see"
]