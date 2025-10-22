from __future__ import annotations
from dataclasses import dataclass
from typing import List, Optional, Dict, Set
import logging

from rdflib import Graph, Namespace, URIRef, Literal

from .config import DATA_ROOT, KG_FILE_PATTERNS
from .utils import pick_first_file

log = logging.getLogger(__name__)
RDFS = Namespace("http://www.w3.org/2000/01/rdf-schema#")

@dataclass
class FactualResult:
    values: List[str]
    meta: Dict

class GraphExecutor:
    def __init__(self):
        self.g: Graph = self._load_graph()

    def _load_graph(self) -> Graph:
        kg_file = None
        for pat in KG_FILE_PATTERNS:
            kg_file = pick_first_file(DATA_ROOT, [pat])
            if kg_file:
                break
        if not kg_file:
            raise FileNotFoundError(f"No KG file found under {DATA_ROOT}")
        log.info("Parsing KG: %s", kg_file)
        g = Graph()
        g.parse(kg_file)
        return g

    def _labels(self, iri: str) -> List[str]:
        out: Set[str] = set()
        s = URIRef(iri)
        for _, _, o in self.g.triples((s, RDFS.label, None)):
            if isinstance(o, Literal):
                out.add(str(o))
            else:
                out.add(str(o))
        return sorted(out)

    def query_factual(self, candidates, relation_spec) -> Optional[FactualResult]:
        if not candidates or not relation_spec:
            return None
        pred = URIRef(relation_spec.predicate)
        # try each candidate entity until we get values
        for cand in candidates:
            s = URIRef(cand.iri)
            values = []
            for _, _, o in self.g.triples((s, pred, None)):
                if isinstance(o, Literal):
                    values.append(str(o))
                else:
                    # object is an entity → resolve labels
                    labs = self._labels(str(o))
                    if labs:
                        values.extend(labs)
            values = sorted(list(dict.fromkeys(values)))  # unique, stable
            if values:
                # For dates, also extract year view if yyyy-mm-dd present
                display = []
                for v in values:
                    if len(v) >= 10 and v[4] == "-" and v[7] == "-":
                        display.append(v)
                        display.append(v[:4])
                    else:
                        display.append(v)
                display = list(dict.fromkeys(display))
                return FactualResult(values=display, meta={
                    "source": "KG",
                    "subject_label": cand.label,
                    "subject_iri": cand.iri,
                    "predicate": relation_spec.predicate,
                })
        return FactualResult(values=[], meta={"source": "KG"})