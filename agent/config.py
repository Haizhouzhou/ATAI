import logging
import os
from pathlib import Path

logger = logging.getLogger(__name__)

class Config:
    def __init__(self):
        logger.info("Loading configuration...")
        
        self.code_root = Path(__file__).resolve().parents[1] 
        self.data_root = Path(os.getenv("DATA_DIR", "/space_mounts/atai-hs25/dataset"))
        
        self.graph_path = self.data_root / "graph.nt"
        if not self.graph_path.exists(): self.graph_path = self.data_root / "graph.tsv"

        self.entity_index_path = self.data_root / "embeddings" / "entity_ids.del"
        self.relation_index_path = self.data_root / "embeddings" / "relation_ids.del"
        
        # RFC Embeddings
        self.entity_embeds_path = self.code_root / "embeddings" / "RFC_entity_embeds.npy"
        self.relation_embeds_path = self.code_root / "embeddings" / "RFC_relation_embeds.npy"

        self.images_json_path = self.data_root / "additional" / "images.json"
        
        # Metadata Directory (Crucial for labels)
        self.metadata_dir = self.code_root / "metadata"
        
        if not self.entity_embeds_path.exists():
            logger.warning(f"RFC Embeddings not found at {self.entity_embeds_path}")

        logger.info("Config loaded.")