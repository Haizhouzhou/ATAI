from __future__ import annotations
from dataclasses import dataclass
from typing import List, Dict, Tuple
import pickle
import logging

from rdflib import Graph, Namespace
from rapidfuzz import process, fuzz

from .config import DATA_ROOT, CACHE_DIR, KG_FILE_PATTERNS, ENTITY_TOPK, MIN_FUZZY_SCORE
from .utils import pick_first_file

log = logging.getLogger(__name__)
RDFS = Namespace("http://www.w3.org/2000/01/rdf-schema#")

@dataclass
class EntityCandidate:
    label: str
    iri: str
    score: float
    type_hint: str | None = None

class EntityLinker:
    def __init__(self):
        self.index_path = CACHE_DIR / "label_index.pkl"
        self.label_to_iri: Dict[str, str] = {}
        self.iri_to_label: Dict[str, str] = {} # <-- FIX 1: ADD THIS
        self._ensure_index()

    def _ensure_index(self):
        if self.index_path.exists():
            try:
                # Load the label_to_iri map from the cache
                self.label_to_iri = pickle.load(open(self.index_path, "rb"))
                # <-- FIX 2: CREATE THE REVERSE MAP -->
                self.iri_to_label = {v: k for k, v in self.label_to_iri.items()}
                log.info("Loaded label index from cache: %s", self.index_path)
                return
            except Exception as e:
                log.warning("Reloading label index failed, rebuilding: %s", e)

        kg = self._load_graph()
        log.info("Building label index from KG (one-time)...")
        label_to_iri: Dict[str, str] = {}
        iri_to_label: Dict[str, str] = {} # <-- FIX 3: ADD THIS -->
        
        for s, p, o in kg.triples((None, RDFS.label, None)):
            label_str = str(o)
            iri_str = str(s)
            label_to_iri[label_str] = iri_str
            iri_to_label[iri_str] = label_str # <-- FIX 4: POPULATE IT -->

        self.label_to_iri = label_to_iri
        self.iri_to_label = iri_to_label # <-- FIX 5: SAVE IT -->
        
        # Save the label_to_iri map to cache (as before)
        pickle.dump(self.label_to_iri, open(self.index_path, "wb"))
        log.info("Label index built: %d entries", len(self.label_to_iri))

    def _load_graph(self) -> Graph:
        kg_file = None
        for pat in KG_FILE_PATTERNS:
            kg_file = pick_first_file(DATA_ROOT, [pat])
            if kg_file:
                break
        if not kg_file:
            raise FileNotFoundError(f"No KG file found under {DATA_ROOT}")
        g = Graph()
        g.parse(kg_file)
        return g

    # <-- FIX 6: ADD THIS ENTIRE FUNCTION -->
    def get_label(self, iri: str) -> str | None:
        """
        Gets the label for a given IRI.
        """
        return self.iri_to_label.get(iri)

    def link(self, raw_strings: List[str]) -> List[EntityCandidate]:
        if not raw_strings:
            return []
        labels = list(self.label_to_iri.keys())
        scored: List[Tuple[str, str, float]] = []  # (label, iri, score)
        for s in raw_strings:
            # exact match (case-sensitive & insensitive variants)
            for lbl in (s, s.lower(), s.upper()):
                if lbl in self.label_to_iri:
                    iri = self.label_to_iri[lbl]
                    scored.append((lbl, iri, 100.0))
            # fuzzy matches
            matches = process.extract(
                s, labels, scorer=fuzz.WRatio, limit=ENTITY_TOPK * 3
            )
            for lbl, score, _ in matches:
                if score >= MIN_FUZZY_SCORE:
                    iri = self.label_to_iri[lbl]
                    scored.append((lbl, iri, float(score)))
        # deduplicate by iri, keep best label/score
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