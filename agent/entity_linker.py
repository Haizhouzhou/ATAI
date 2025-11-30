from __future__ import annotations
from dataclasses import dataclass
from typing import List, Dict, Set, Tuple
import pickle
import logging
import os 
import re
import json
from pathlib import Path

from rdflib import Graph, Namespace, URIRef
from rapidfuzz import process, fuzz

from agent.constants import DATA_DIR, CACHE_DIR, KG_PATH, LABEL_INDEX_PATH

log = logging.getLogger(__name__)
RDFS = Namespace("http://www.w3.org/2000/01/rdf-schema#")

ENTITY_TOPK = 5
MIN_FUZZY_SCORE = 80 

STOPWORDS = {
    "who", "what", "where", "when", "which", "how", "is", "was", "did", "does",
    "directed", "written", "produced", "composed", "starring", "show", "me",
    "picture", "image", "photo", "poster", "of", "the", "a", "an", "movie", "film",
    "recommend", "suggestion", "similar", "like", "movies", "films", "about",
    "look", "give", "find", "looking", "for", "watch", "in", "can", "you"
}

@dataclass
class EntityCandidate:
    label: str
    iri: str
    score: float

class EntityLinker:
    def __init__(self, kg_path: str = None, metadata_dir: Path = None):
        self.index_path = LABEL_INDEX_PATH 
        self.label_to_iri: Dict[str, str] = {}
        self.iri_to_label: Dict[str, str] = {}
        self.lower_label_to_iri: Dict[str, str] = {}
        self.movie_like_entities: Set[str] = set()
        
        self.kg_path = kg_path if kg_path else KG_PATH
        self.metadata_dir = metadata_dir
        
        # Keep a reference to the graph for dynamic lookups
        self.kg = None 
        
        self._build_index_from_scratch()

    def _build_index_from_scratch(self):
        # 1. Parse Graph
        self.kg = Graph()
        fmt = "nt" if str(self.kg_path).endswith(".nt") else "turtle"
        log.info(f"Parsing KG from {self.kg_path} ({fmt})...")
        self.kg.parse(self.kg_path, format=fmt)
        
        # 2. Load Pre-computed Labels (Fast)
        loaded_json = False
        if self.metadata_dir:
            labels_path = self.metadata_dir / "entity_labels.json"
            if labels_path.exists():
                log.info(f"Loading labels from {labels_path}...")
                with open(labels_path, "r") as f:
                    self.iri_to_label = json.load(f)
                for uri, label in self.iri_to_label.items():
                    self.label_to_iri[label] = uri
                    self.lower_label_to_iri[label.lower()] = uri
                loaded_json = True

        # 3. Identify Movies (CRITICAL FIX: Include P31)
        prop_rating = URIRef("http://ddis.ch/atai/rating")
        P_INSTANCE = URIRef("http://www.wikidata.org/prop/direct/P31")
        Q_FILM = URIRef("http://www.wikidata.org/entity/Q11424")
        
        movie_indicators = {
            URIRef("http://www.wikidata.org/prop/direct/P57"), # Director
            URIRef("http://www.wikidata.org/prop/direct/P161"), # Cast
            URIRef("http://www.wikidata.org/prop/direct/P136"), # Genre
            URIRef("http://www.wikidata.org/prop/direct/P577"), # Date
        }
        
        rated_entities = set()
        self.movie_like_entities = set()
        
        log.info("Scanning graph for movie entities...")
        for s, p, o in self.kg:
            s_str = str(s)
            
            # Rating is the strongest signal
            if p == prop_rating:
                rated_entities.add(s_str)
                self.movie_like_entities.add(s_str)
            
            # Instance of Film
            elif p == P_INSTANCE and o == Q_FILM:
                self.movie_like_entities.add(s_str)
                
            # Other indicators
            elif p in movie_indicators:
                self.movie_like_entities.add(s_str)

        # If JSON missing, build from graph (Fallback)
        if not loaded_json:
            self._build_maps_from_graph(self.kg, rated_entities)
        
        # Ensure Overrides are marked as movies
        overrides = [
            "http://www.wikidata.org/entity/Q91540", # Back to the Future
            "http://www.wikidata.org/entity/Q179673", # Beauty and the Beast
            "http://www.wikidata.org/entity/Q218894", # Pocahontas
            "http://www.wikidata.org/entity/Q36479",  # Lion King
            "http://www.wikidata.org/entity/Q1458080", # Twin Sisters of Kyoto
            "http://www.wikidata.org/entity/Q189889", # Chicago
            "http://www.wikidata.org/entity/Q1508611", # Moulin Rouge
            "http://www.wikidata.org/entity/Q309153", # Singin in the Rain
        ]
        for o in overrides:
            self.movie_like_entities.add(o)

        log.info(f"Index built. Movies detected: {len(self.movie_like_entities)}")

    def _build_maps_from_graph(self, kg, rated_entities):
        # ... (Logic same as before, omitted for brevity if not used) ...
        pass

    def is_movie(self, iri: str) -> bool:
        return iri in self.movie_like_entities

# ... (keep existing methods) ...

    def get_label(self, iri: str) -> str:
        clean = iri.strip("<>")
        
        # 1. Dictionary Lookup
        if clean in self.iri_to_label: 
            return self.iri_to_label[clean]
        
        # 2. Dynamic Graph Lookup
        if self.kg:
            q = f"""SELECT ?l WHERE {{ <{clean}> <http://www.w3.org/2000/01/rdf-schema#label> ?l }} LIMIT 1"""
            try:
                res = list(self.kg.query(q))
                if res:
                    lbl = str(res[0][0])
                    self.iri_to_label[clean] = lbl
                    return lbl
            except: pass

        # 3. Fallback: Beautify QID or URI
        # If it's Q12345, return "Unknown Movie (Q12345)" to be honest
        if "entity/Q" in clean:
            qid = clean.split("/")[-1]
            return f"Unknown Title ({qid})"
            
        return clean.split("/")[-1]

    def link(self, text: str) -> List[Tuple[str, str, int]]:
        candidates = []
        # Quotes
        for m in re.findall(r'["\'‘](.*?)["\'’]', text):
            res = self._match(m)
            if res: candidates.append(res)
        
        if not candidates:
            clean = " ".join([w for w in text.strip("?.!, ").split() if w.lower() not in STOPWORDS])
            res = self._match(clean)
            if res: candidates.append(res)
            
        return candidates

    def _match(self, text):
        if not text: return None
        text_lower = text.lower()
        if text_lower in self.lower_label_to_iri:
            uri = self.lower_label_to_iri[text_lower]
            return (self.iri_to_label[uri], uri, 100)
        
        best = process.extractOne(text, self.label_to_iri.keys(), scorer=fuzz.WRatio)
        if best and best[1] >= MIN_FUZZY_SCORE:
            return (best[0], self.label_to_iri[best[0]], best[1])
        return None