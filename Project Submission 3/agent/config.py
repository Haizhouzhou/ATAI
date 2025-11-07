import logging
from pathlib import Path
from typing import List, Dict, Optional

# This helper function should be in agent/utils.py
# (which was [UNCHANGED] from Submission 2)
from agent.utils import find_file_in_dirs, find_files_in_dirs

logger = logging.getLogger(__name__)

# --- Constants from your old config.py ---
DATA_ROOT = Path("/space_mounts/atai-hs25/dataset").resolve()
FILES_ROOT = Path("/files").resolve()
# Updated to Submission 3
CACHE_DIR = FILES_ROOT / "Project Submission 3" / ".cache"
CACHE_DIR.mkdir(parents=True, exist_ok=True)

KG_FILE_PATTERNS: List[str] = [
    "*.nt", "*.ttl", "*graph*.nt", "*graph*.ttl", "14_graph.nt", "ddis-dataset-2024.ttl"
]

EMBED_PATTERNS = {
    "entity_ids": ["entity_ids.*", "*entity*ids*"],
    "relation_ids": ["relation_ids.*", "*relation*ids*"],
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
# --- End Constants ---


class Config:
    """
    Configuration class to load and hold all necessary file paths.
    This provides the class that app/main.py needs.
    """
    def __init__(self):
        logger.info("Loading configuration...")
        
        # 1. Load KG Path
        self.graph_db = self._find_graph_db()
        
        # 2. Load Embedding Paths
        embed_files = self._find_embedding_files()
        self.entity_embeddings = embed_files.get("entity_embeds")
        self.entity_index = embed_files.get("entity_ids")
        self.relation_embeddings = embed_files.get("relation_embeds")
        self.relation_index = embed_files.get("relation_ids")
        
        # 3. Set Label Index Path
        self.label_index = LABEL_INDEX_PATH
        
        # 4. Validate Paths
        if not self.graph_db:
            raise FileNotFoundError(f"Could not find Knowledge Graph file in {DATA_ROOT} using patterns {KG_FILE_PATTERNS}")
        if not all([self.entity_embeddings, self.entity_index, self.relation_embeddings, self.relation_index]):
            logger.warning(f"Could not find all embedding files. Found: {embed_files}")
            raise FileNotFoundError("Could not find all required embedding files.")
            
        logger.info(f"Graph DB found: {self.graph_db}")
        logger.info(f"Entity Embeddings found: {self.entity_embeddings}")
        logger.info(f"Label Index path: {self.label_index}")

    def _find_graph_db(self) -> Optional[Path]:
        """Find the first KG file matching patterns in DATA_ROOT."""
        logger.info(f"Searching for KG in {DATA_ROOT}...")
        return find_file_in_dirs([DATA_ROOT], KG_FILE_PATTERNS)

    def _find_embedding_files(self) -> Dict[str, Path]:
        """Find embedding files based on patterns."""
        logger.info(f"Searching for embedding files in {EMBED_SUBDIR_CANDIDATES}...")
        found_files = {}
        for key, patterns in EMBED_PATTERNS.items():
            found_path = find_file_in_dirs(EMBED_SUBDIR_CANDIDATES, patterns)
            if found_path:
                found_files[key] = found_path
            else:
                logger.warning(f"Could not find file for embedding key: {key}")
        return found_files