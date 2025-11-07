import logging
import faiss
import numpy as np
import pickle
import pandas as pd # <-- Import pandas
from pathlib import Path
from typing import List, Tuple, Dict, Optional

logger = logging.getLogger(__name__)

# --- THIS IS THE FIX ---
# This helper function is from your original Submission 2 code.
# It correctly loads the ID map file as a TSV.
def load_id_map_from_tsv(id_map_file: Path) -> Dict[str, int]:
    logger.info(f"Loading ID map from TSV: {id_map_file}")
    df = pd.read_csv(id_map_file, sep="\t", header=None, names=["id", "label"])
    # This maps {label: id}
    return {row.label: row.id for row in df.itertuples()}
# --- END FIX ---

class EmbeddingExecutor:
    """
    Manages loading and querying of entity and relation embeddings.
    """
    
    def __init__(self,
                 entity_embed_path: Path,
                 entity_index_path: Path, # This is the entity_ids.tsv
                 relation_embed_path: Path,
                 relation_index_path: Path): # This is the relation_ids.tsv
        """
        Initializes the executor by loading all embedding files.
        """
        try:
            logger.info("Loading embedding models...")
            
            # --- Load Entity Embeddings ---
            logger.info(f"Loading entity embeddings from {entity_embed_path}")
            self.entity_embeds = np.load(entity_embed_path)
            
            # --- THIS IS THE FIX ---
            # Replaced pickle.load() with the correct TSV loader
            logger.info(f"Loading entity ID map from {entity_index_path}")
            self.entity_id_map: Dict[str, int] = load_id_map_from_tsv(entity_index_path)
            self.entity_idx_to_id: List[str] = ["" for _ in range(len(self.entity_id_map))]
            for label, idx in self.entity_id_map.items():
                self.entity_idx_to_id[idx] = label
            # --- END FIX ---
            
            # --- Load Relation Embeddings ---
            logger.info(f"Loading relation embeddings from {relation_embed_path}")
            self.relation_embeds = np.load(relation_embed_path)
            
            # --- THIS IS THE FIX ---
            logger.info(f"Loading relation ID map from {relation_index_path}")
            self.relation_id_map: Dict[str, int] = load_id_map_from_tsv(relation_index_path)
            self.relation_idx_to_id: List[str] = ["" for _ in range(len(self.relation_id_map))]
            for label, idx in self.relation_id_map.items():
                self.relation_idx_to_id[idx] = label
            # --- END FIX ---

            # --- Build FAISS Index for Entities ---
            logger.info("Building FAISS index for entity embeddings...")
            self.dimension = self.entity_embeds.shape[1]
            self.entity_faiss_index = faiss.IndexFlatL2(self.dimension)
            
            # Prepare embeddings for FAISS (must be C-contiguous)
            entity_embeds_contiguous = np.ascontiguousarray(self.entity_embeds, dtype=np.float32)
            faiss.normalize_L2(entity_embeds_contiguous)
            self.entity_faiss_index.add(entity_embeds_contiguous)
            
            # Store the normalized embeds for dot product
            self.normalized_entity_embeds = entity_embeds_contiguous
            
            logger.info("FAISS index built successfully.")

        except Exception as e:
            logger.error(f"Failed to initialize EmbeddingExecutor: {e}", exc_info=True)
            raise

    def get_embedding(self, entity_id: str, embedding_type: str = 'entity') -> Optional[np.ndarray]:
        if embedding_type == 'entity':
            if entity_id in self.entity_id_map:
                idx = self.entity_id_map[entity_id]
                return self.normalized_entity_embeds[idx] # Return normalized embed
        elif embedding_type == 'relation':
            if entity_id in self.relation_id_map:
                idx = self.relation_id_map[entity_id]
                return self.relation_embeds[idx]
        
        logger.warning(f"Could not find embedding for ID: {entity_id} (type: {embedding_type})")
        return None

    def get_embeddings(self, entity_ids: List[str]) -> List[Optional[np.ndarray]]:
        return [self.get_embedding(eid, 'entity') for eid in entity_ids]

    def get_nearest_neighbors(self, 
                              entity_id: str, 
                              k: int = 10, 
                              embedding_type: str = 'entity') -> List[Tuple[str, float]]:
        if embedding_type != 'entity':
            logger.warning("Nearest neighbor search is only implemented for entities.")
            return []
            
        embed_vector = self.get_embedding(entity_id, 'entity')
        if embed_vector is None:
            return []
            
        query_vector = embed_vector.reshape(1, -1)
        
        try:
            distances, indices = self.entity_faiss_index.search(query_vector, k + 1)
            
            results = []
            for i, idx in enumerate(indices[0]):
                neighbor_id = self.entity_idx_to_id[idx]
                
                if neighbor_id == entity_id:
                    continue
                    
                similarity = float(distances[0][i]) 
                similarity_score = 1.0 / (1.0 + similarity)
                
                results.append((neighbor_id, similarity_score))
            
            return results[:k]
            
        except Exception as e:
            logger.error(f"Error during FAISS search for {entity_id}: {e}", exc_info=True)
            return []

    def cosine_similarity(self, vec1: np.ndarray, vec2: np.ndarray) -> float:
        # Assumes vectors are already L2-normalized
        return np.dot(vec1, vec2)