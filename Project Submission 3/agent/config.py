import logging
from pathlib import Path
from typing import List, Dict, Optional

from agent.utils import find_file_in_dirs

logger = logging.getLogger(__name__)

# --- Constants ---
DATA_ROOT = Path("/space_mounts/atai-hs25/dataset").resolve()
FILES_ROOT = Path("/files").resolve()
CACHE_DIR = FILES_ROOT / "Project Submission 3" / ".cache"
CACHE_DIR.mkdir(parents=True, exist_ok=True)

KG_FILE_PATTERNS: List[str] = [
    "*.nt", "*.ttl", "*graph*.nt", "*graph*.ttl", "14_graph.nt", "ddis-dataset-2024.ttl"
]

# --- THIS IS THE FIX ---
# The patterns now look for .tsv or .txt files, which is what
# your original Submission 2 code expected to load with pandas.
EMBED_PATTERNS = {
    "entity_ids": ["entity_ids.tsv", "*entity*ids*.tsv", "entity_ids.txt"],
    "relation_ids": ["relation_ids.tsv", "*relation*ids*.tsv", "relation_ids.txt"],
    "entity_embeds": ["entity_embeds.npy", "*entity*embed*.npy"],
    "relation_embeds": ["relation_embeds.npy", "*relation*embed*.npy"],
}

EMBED_SUBDIR_CANDIDATES = [
    DATA_ROOT,
    DATA_ROOT / "embeddings",
    DATA_ROOT / "Embeddings",
    DATA_ROOT / "kg_embeddings",
]

LABEL_INDEX_PATH = CACHE_DIR / "label_index.pkl"

# --- Limits / knobs (from Submission 2) ---
MAX_LABELS_TO_INDEX = 5_000_000
ENTITY_TOPK = 3
MIN_FUZZY_SCORE = 70
# --- End Constants ---


class Config:
    def __init__(self):
        logger.info("Loading configuration...")
        
        self.graph_db = self._find_graph_db()
        
        embed_files = self._find_embedding_files()
        self.entity_embeddings = embed_files.get("entity_embeds")
        self.entity_index = embed_files.get("entity_ids") # This is the entity_ids.tsv
        self.relation_embeddings = embed_files.get("relation_embeds")
        self.relation_index = embed_files.get("relation_ids") # This is the relation_ids.tsv
        
        self.label_index = LABEL_INDEX_PATH
        
        if not self.graph_db:
            raise FileNotFoundError(f"Could not find Knowledge Graph file in {DATA_ROOT} using patterns {KG_FILE_PATTERNS}")
        if not all([self.entity_embeddings, self.entity_index, self.relation_embeddings, self.relation_index]):
            logger.warning(f"Could not find all embedding files. Found: {embed_files}")
            raise FileNotFoundError("Could not find all required embedding files.")
            
        logger.info(f"Graph DB found: {self.graph_db}")
        logger.info(f"Entity Embeddings found: {self.entity_embeddings}")
        logger.info(f"Entity ID map (tsv) found: {self.entity_index}")
        logger.info(f"Relation ID map (tsv) found: {self.relation_index}")
        logger.info(f"Label Index path: {self.label_index}")

    def _find_graph_db(self) -> Optional[Path]:
        logger.info(f"Searching for KG in {DATA_ROOT}...")
        return find_file_in_dirs([DATA_ROOT], KG_FILE_PATTERNS)

    def _find_embedding_files(self) -> Dict[str, Path]:
        logger.info(f"Searching for embedding files in {EMBED_SUBDIR_CANDIDATES}...")
        found_files = {}
        for key, patterns in EMBED_PATTERNS.items():
            found_path = find_file_in_dirs(EMBED_SUBDIR_CANDIDATES, patterns)
            if found_path:
                found_files[key] = found_path
            else:
                logger.warning(f"Could not find file for embedding key: {key}")
        return found_files