import os

# --- File Paths ---
# Use environment variables for paths, falling back to defaults
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
# Maps natural language terms to Wikidata predicate IDs
# IMPORTANT: This dictionary is processed by RelationMapper,
# which sorts keys by length (desc) to match longest phrases first.
PREDICATE_MAP = {
    # Properties for QA (Eval 2 Fixes Added)
    "nominated for": "wdt:P1411",
    "was nominated": "wdt:P1411",
    "nomination": "wdt:P1411",
    "award": "wdt:P166", # Shorter key 'award' will be matched after 'nominated for'
    
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
    
    # Properties for Recommendation Preferences
    "language": "wdt:P407",
    "year": "wdt:P577", # Simplified mapping
    "part of series": "wdt:P179",
    "based on": "wdt:P4969", # "derivative work"
}

# --- INTENT PARSING (EVAL 3) ---

# Keywords to detect recommendation intent
RECOMMENDATION_KEYWORDS = [
    "recommend", "suggest", "give me", "find me", "looking for",
    "like", "similar to", "another", "more movies"
]

# Keywords to detect QA intent
QA_KEYWORDS = [
    "who", "what", "when", "where", "which", "how many", "list", "is", "did"
]

# Keywords for follow-up questions
FOLLOW_UP_KEYWORDS = [
    "another", "more", "like that", "how about", "what else"
]

# Keywords to detect negations/exclusions
NEGATION_KEYWORDS = [
    "not", "don't", "without", "except", "avoid", "no", "less"
]

# Keywords to map to preference types (Rule 2.3 fix: "in" removed from language)
PREFERENCE_KEYWORDS = {
    "genre": ["genre", "type of movie", "about", "is a"],
    "actor": ["actor", "starring", "with", "features"],
    "director": ["director", "directed by"],
    "language": ["language", "speak"], # "in [language]" is handled by regex
}

# Supported languages for specific regex matching
SUPPORTED_LANGUAGES_REGEX = r'\b(French|German|Spanish|English|Italian|Japanese|Korean)\b'