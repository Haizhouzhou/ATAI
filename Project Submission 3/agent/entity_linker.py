from __future__ import annotations
from dataclasses import dataclass
from typing import List, Dict, Set
import pickle
import logging
import os 
import re

from rdflib import Graph, Namespace, URIRef
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
    def __init__(self, kg_path: str = None):
        self.index_path = LABEL_INDEX_PATH 
        self.label_to_iri: Dict[str, str] = {}
        self.iri_to_label: Dict[str, str] = {}
        self.lower_label_to_iri: Dict[str, str] = {}
        
        self.kg_path = kg_path if kg_path else KG_PATH
        
        # Always rebuild to ensure the new logic applies
        self._build_index_from_scratch()

    def _build_index_from_scratch(self):
        """
        Builds the index with smart collision handling:
        Prioritizes entities that look like MOVIES (have director, cast, genre, etc.)
        """
        kg = self._load_graph()
        log.info(f"Building label index from KG at {self.kg_path}...")
        
        # 1. Identify "Movie-like" entities based on properties
        # We check if an entity has properties that only movies typically have.
        # P57 (director), P161 (cast), P136 (genre), P577 (date), ddis:rating
        movie_indicators = {
            URIRef("http://www.wikidata.org/prop/direct/P57"),   # director
            URIRef("http://www.wikidata.org/prop/direct/P161"),  # cast member
            URIRef("http://www.wikidata.org/prop/direct/P577"),  # publication date
            URIRef("http://www.wikidata.org/prop/direct/P136"),  # genre
            URIRef("http://ddis.ch/atai/rating")                 # rating
        }
        
        movie_entity_set: Set[str] = set()
        
        log.info("Scanning for movie entities...")
        for s, p, o in kg:
            if p in movie_indicators:
                movie_entity_set.add(str(s))
                
        log.info(f"Identified {len(movie_entity_set)} movie-like entities.")

        # 2. Build Index with Priority Logic
        label_to_iri: Dict[str, str] = {}
        iri_to_label: Dict[str, str] = {} 
        lower_label_to_iri: Dict[str, str] = {}
        
        def get_id_num(iri):
            match = re.search(r'Q(\d+)', iri)
            return int(match.group(1)) if match else float('inf')

        for s, p, o in kg.triples((None, RDFS.label, None)):
            label_str = str(o)
            iri_str = str(s)
            label_lower = label_str.lower()
            
            is_movie = iri_str in movie_entity_set
            
            # --- SMART COLLISION HANDLING ---
            if label_lower in lower_label_to_iri:
                existing_iri = lower_label_to_iri[label_lower]
                existing_is_movie = existing_iri in movie_entity_set
                
                should_replace = False
                
                # Rule 1: Movie > Non-Movie (e.g. Pocahontas Movie > Pocahontas Person)
                if is_movie and not existing_is_movie:
                    should_replace = True
                # Rule 2: If both are Movies (or both not), prefer smaller ID
                elif (is_movie == existing_is_movie):
                    if get_id_num(iri_str) < get_id_num(existing_iri):
                        should_replace = True
                
                if should_replace:
                    label_to_iri[label_str] = iri_str
                    lower_label_to_iri[label_lower] = iri_str
                    iri_to_label[iri_str] = label_str
            else:
                # New label
                label_to_iri[label_str] = iri_str
                lower_label_to_iri[label_lower] = iri_str
                
            # Always update IRI->Label map if missing
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
        if not os.path.exists(self.kg_path):
            raise FileNotFoundError(f"No KG file found at {self.kg_path}")
        
        g = Graph()
        file_ext = os.path.splitext(self.kg_path)[1].lower()
        fmt = "nt" if file_ext == ".nt" else "turtle"
        
        log.info(f"Parsing KG from {self.kg_path} with format {fmt}...")
        g.parse(self.kg_path, format=fmt)
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
            
            if s_lower in self.lower_label_to_iri:
                iri = self.lower_label_to_iri[s_lower]
                label = self.iri_to_label.get(iri, s)
                scored.append((label, iri, 101.0)) 
                continue 
            
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
        candidates = self.link([text])
        return [(c.label, c.iri, int(c.score)) for c in candidates]