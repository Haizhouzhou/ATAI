from __future__ import annotations
from dataclasses import dataclass
from typing import List, Dict, Tuple
import pickle
import logging
import os 

from rdflib import Graph, Namespace
from rapidfuzz import process, fuzz

from agent.constants import DATA_DIR, CACHE_DIR, KG_PATH, LABEL_INDEX_PATH

log = logging.getLogger(__name__)
RDFS = Namespace("http://www.w3.org/2000/01/rdf-schema#")

ENTITY_TOPK = 5
MIN_FUZZY_SCORE = 80 

@dataclass
class EntityCandidate:
    label: str
    iri: str
    score: float
    type_hint: str | None = None

class EntityLinker:
    def __init__(self):
        self.index_path = LABEL_INDEX_PATH 
        self.label_to_iri: Dict[str, str] = {}
        self.iri_to_label: Dict[str, str] = {}
        self.lower_label_to_iri: Dict[str, str] = {}
        self._ensure_index()

    def _ensure_index(self):
        if os.path.exists(self.index_path):
            try:
                with open(self.index_path, "rb") as f:
                    self.label_to_iri = pickle.load(f)
                
                self.iri_to_label = {v: k for k, v in self.label_to_iri.items()}
                self.lower_label_to_iri = {k.lower(): v for k, v in self.label_to_iri.items()}
                
                log.info("Loaded label index from cache: %s", self.index_path)
                return
            except Exception as e:
                log.warning("Reloading label index failed, rebuilding: %s", e)

        kg = self._load_graph()
        log.info("Building label index from KG (one-time)...")
        label_to_iri: Dict[str, str] = {}
        iri_to_label: Dict[str, str] = {} 
        lower_label_to_iri: Dict[str, str] = {}
        
        for s, p, o in kg.triples((None, RDFS.label, None)):
            label_str = str(o)
            iri_str = str(s)
            
            label_to_iri[label_str] = iri_str
            lower_label_to_iri[label_str.lower()] = iri_str
            
            if iri_str not in iri_to_label:
                iri_to_label[iri_str] = label_str 

        self.label_to_iri = label_to_iri
        self.iri_to_label = iri_to_label 
        self.lower_label_to_iri = lower_label_to_iri
        
        os.makedirs(os.path.dirname(self.index_path), exist_ok=True)
        with open(self.index_path, "wb") as f:
            pickle.dump(self.label_to_iri, f)
        log.info("Label index built: %d entries", len(self.label_to_iri))

    def _load_graph(self) -> Graph:
        if not os.path.exists(KG_PATH):
            raise FileNotFoundError(f"No KG file found at {KG_PATH}")
        g = Graph()
        g.parse(KG_PATH, format="turtle")
        return g

    def get_label(self, iri: str) -> str | None:
        return self.iri_to_label.get(iri)

    def link(self, raw_strings: List[str]) -> List[EntityCandidate]:
        if not raw_strings:
            return []
        
        labels = list(self.label_to_iri.keys())
        scored: List[Tuple[str, str, float]] = [] 
        
        for s in raw_strings:
            s_lower = s.lower()
            
            # Exact match (case-insensitive) priority
            if s_lower in self.lower_label_to_iri:
                iri = self.lower_label_to_iri[s_lower]
                label = self.iri_to_label.get(iri, s)
                scored.append((label, iri, 101.0)) 
                continue 
            
            # Fuzzy matches
            matches = process.extract(
                s, labels, scorer=fuzz.WRatio, limit=ENTITY_TOPK * 3
            )
            for lbl, score, _ in matches:
                if score >= MIN_FUZZY_SCORE:
                    iri = self.label_to_iri[lbl]
                    scored.append((lbl, iri, float(score)))
        
        best: Dict[str, Tuple[str, float]] = {}
        for lbl, iri, sc in scored:
            if iri not in best or sc > best[iri][1]:
                best[iri] = (lbl, sc)
        
        ranked = sorted(
            [(lbl, iri, sc) for iri, (lbl, sc) in best.items()],
            key=lambda x: -x[2]
        )
        
        return [
            EntityCandidate(label=lbl, iri=iri, score=sc)
            for lbl, iri, sc in ranked[:ENTITY_TOPK]
        ]

    def link_entities(self, text: str) -> List[Tuple[str, str, int]]:
        """
        Legacy function from Project 2 for compatibility.
        """
        candidates = self.link([text])
        return [(c.label, c.iri, int(c.score)) for c in candidates]