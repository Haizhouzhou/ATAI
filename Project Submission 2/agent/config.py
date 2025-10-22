# Configuration placeholder
from __future__ import annotations
from pathlib import Path
from typing import List


# Root where the read-only dataset is mounted
DATA_ROOT = Path("/space_mounts/atai-hs25/dataset").resolve()


# Where we can write caches on Nuvolos
FILES_ROOT = Path("/files").resolve()
CACHE_DIR = FILES_ROOT / "Project Submission 2" / ".cache"
CACHE_DIR.mkdir(parents=True, exist_ok=True)


# Auto-discovery patterns for KG and embeddings
KG_FILE_PATTERNS: List[str] = [
"*.nt", "*.ttl", "*graph*.nt", "*graph*.ttl", "14_graph.nt",
]


EMBED_PATTERNS = {
"entity_ids": ["entity_ids.*", "*entity*ids*"],
"relation_ids": ["relation_ids.*", "*relation*ids*"],
"entity_embeds": ["entity_embeds.npy", "*entity*embed*.npy"],
"relation_embeds": ["relation_embeds.npy", "*relation*embed*.npy"],
}


# Heuristic: search common subfolders
EMBED_SUBDIR_CANDIDATES = [
DATA_ROOT,
DATA_ROOT / "embeddings",
DATA_ROOT / "Embeddings",
DATA_ROOT / "kg_embeddings",
]


# Limits / knobs
MAX_LABELS_TO_INDEX = 5_000_000 # safety cap
ENTITY_TOPK = 3
MIN_FUZZY_SCORE = 70


# Relations (predicates) we support in IE2
SUPPORTED_PREDICATES = {
"director": "http://www.wikidata.org/prop/direct/P57",
"screenwriter": "http://www.wikidata.org/prop/direct/P58",
"publication_date": "http://www.wikidata.org/prop/direct/P577",
"genre": "http://www.wikidata.org/prop/direct/P136",
"duration": "http://www.wikidata.org/prop/direct/P2047",
"cast_member": "http://www.wikidata.org/prop/direct/P161",
"original_language": "http://www.wikidata.org/prop/direct/P364",
"production_company": "http://www.wikidata.org/prop/direct/P272",
"mpaa_rating": "http://www.wikidata.org/prop/direct/P1657",
}


# For type filtering during embedding retrieval, we learn candidate sets from the KG at runtime