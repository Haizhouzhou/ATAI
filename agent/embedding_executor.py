import logging
import numpy as np
import faiss
from pathlib import Path
from typing import List, Tuple, Dict, Optional

logger = logging.getLogger(__name__)

class EmbeddingExecutor:
    def __init__(self, ent_emb_path, ent_id_path, rel_emb_path, rel_id_path):
        self.entity_embeds = None
        self.relation_embeds = None
        self.entity_id_map = {} 
        self.index_to_uri = []
        self.relation_id_map = {}
        self.faiss_index = None
        
        self._load_data(ent_emb_path, ent_id_path, rel_emb_path, rel_id_path)

    def _load_data(self, ent_emb_path, ent_id_path, rel_emb_path, rel_id_path):
        try:
            # Load Entity Map
            with open(ent_id_path, "r", encoding="utf-8") as f:
                for line in f:
                    p = line.strip().split("\t")
                    if len(p) >= 2:
                        idx = int(p[0])
                        uri = p[1]
                        self.entity_id_map[uri] = idx
                        if len(self.index_to_uri) <= idx:
                            self.index_to_uri.extend([""] * (idx + 1 - len(self.index_to_uri)))
                        self.index_to_uri[idx] = uri
            
            # Load Relation Map
            with open(rel_id_path, "r", encoding="utf-8") as f:
                for line in f:
                    p = line.strip().split("\t")
                    if len(p) >= 2:
                        self.relation_id_map[p[1]] = int(p[0])

            # Load Embeddings
            if ent_emb_path.exists():
                self.entity_embeds = np.load(ent_emb_path)
                self.relation_embeds = np.load(rel_emb_path)
                
                # Build FAISS
                d = self.entity_embeds.shape[1]
                self.faiss_index = faiss.IndexFlatIP(d) # Cosine if normalized
                faiss.normalize_L2(self.entity_embeds)
                self.faiss_index.add(self.entity_embeds)
                logger.info("Embeddings loaded and Index built.")
            else:
                logger.error("Embedding file missing!")

        except Exception as e:
            logger.error(f"Embedding Load Error: {e}")

    def get_nearest_neighbors(self, entity_uri: str, k: int = 10) -> List[Tuple[str, float]]:
        if not self.faiss_index or entity_uri not in self.entity_id_map: return []
        idx = self.entity_id_map[entity_uri]
        vec = self.entity_embeds[idx].reshape(1, -1)
        D, I = self.faiss_index.search(vec, k + 1)
        
        res = []
        for i in range(1, k + 1):
            n_idx = I[0][i]
            if n_idx < len(self.index_to_uri):
                res.append((self.index_to_uri[n_idx], float(D[0][i])))
        return res

    def predict_tail(self, head_uri, rel_uri, k=1):
        if head_uri not in self.entity_id_map or rel_uri not in self.relation_id_map:
            return []
        
        h = self.entity_embeds[self.entity_id_map[head_uri]]
        r = self.relation_embeds[self.relation_id_map[rel_uri]]
        target = (h + r).reshape(1, -1)
        faiss.normalize_L2(target)
        
        D, I = self.faiss_index.search(target, k)
        res = []
        for i in range(k):
            idx = I[0][i]
            if idx < len(self.index_to_uri):
                res.append((self.index_to_uri[idx], float(D[0][i])))
        return res