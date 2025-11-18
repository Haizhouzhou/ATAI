import os

# --- File Paths ---
DATA_DIR = os.getenv('DATA_DIR', '/data')
CACHE_DIR = os.getenv('CACHE_DIR', '.cache')

KG_PATH = os.path.join(DATA_DIR, 'ddis-dataset-2024.ttl')
ENTITY_EMBEDDINGS_PATH = os.path.join(DATA_DIR, 'entity_embeddings.npy')
ENTITY_INDEX_PATH = os.path.join(DATA_DIR, 'entity_index.idx')
RELATION_EMBEDDINGS_PATH = os.path.join(DATA_DIR, 'relation_embeddings.npy')
RELATION_INDEX_PATH = os.path.join(DATA_DIR, 'relation_index.idx')
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

# --- PREDICATE MAPPING ---
PREDICATE_MAP = {
    # QA Properties
    "nominated for": "wdt:P1411",
    "was nominated": "wdt:P1411",
    "nomination": "wdt:P1411",
    "award": "wdt:P166",
    "composer": "wdt:P86",
    "music by": "wdt:P86",
    "director": "wdt:P57",
    "directed by": "wdt:P57",
    "screenwriter": "wdt:P58",
    "written by": "wdt:P58",
    "cast member": "wdt:P161",
    "actor": "wdt:P161",
    "starring": "wdt:P161",
    "publication date": "wdt:P577",
    "release date": "wdt:P577",
    "country of origin": "wdt:P495",
    "country": "wdt:P495",
    "genre": "wdt:P136",
    "producer": "wdt:P162",
    "produced by": "wdt:P162",
    "rating": "ddis:rating",
    "user rating": "ddis:rating",
    
    # Recommendation Specific
    "language": "wdt:P407",
    "year": "wdt:P577",
    "part of series": "wdt:P179",
    "based on": "wdt:P4969",
    "image": "wdt:P18"
}

# --- GENRE MAPPING (Rule: Hardcode common genres to avoid linking errors) ---
GENRE_MAPPING = {
    "action": "wd:Q188473",
    "horror": "wd:Q200092",
    "comedy": "wd:Q157443",
    "romance": "wd:Q1054574",
    "drama": "wd:Q130232",
    "sci-fi": "wd:Q471839",
    "science fiction": "wd:Q471839",
    "adventure": "wd:Q319221",
    "thriller": "wd:Q182015",
    "animation": "wd:Q581714",
    "animated": "wd:Q581714",
    "fantasy": "wd:Q157394",
    "crime": "wd:Q959790",
    "documentary": "wd:Q93204"
}

# --- INTENT PARSING ---
RECOMMENDATION_KEYWORDS = [
    "recommend", "suggest", "give me", "find me", "looking for",
    "like", "similar to", "another", "more movies", "movies like",
    "movie with", "film with"
]

QA_KEYWORDS = [
    "who", "what", "when", "where", "which", "how many", "list", "is", "did"
]

FOLLOW_UP_KEYWORDS = [
    "another", "more", "like that", "how about", "what else"
]

NEGATION_KEYWORDS = [
    "not", "don't", "without", "except", "avoid", "no", "less"
]

PREFERENCE_KEYWORDS = {
    "genre": ["genre", "type of movie", "about", "is a"],
    "actor": ["actor", "starring", "with", "features"],
    "director": ["director", "directed by"],
    "language": ["language", "speak"],
}

SUPPORTED_LANGUAGES_REGEX = r'\b(French|German|Spanish|English|Italian|Japanese|Korean)\b'