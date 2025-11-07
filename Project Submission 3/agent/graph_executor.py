from __future__ import annotations
from dataclasses import dataclass
from typing import List, Optional, Dict, Set
import logging

from rdflib import Graph, Namespace, URIRef, Literal
from rdflib.namespace import RDF, RDFS, SKOS, XSD

from .config import DATA_ROOT, KG_FILE_PATTERNS
from .utils import pick_first_file

log = logging.getLogger(__name__)
RDFS = Namespace("http://www.w3.org/2000/01/rdf-schema#")

SCHEMA = Namespace("http://schema.org/")
WDT = Namespace("http://www.wikidata.org/prop/direct/")

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

    def _types(self, iri: str) -> List[str]:
        s = URIRef(iri)
        types: Set[str] = set()
        for _, _, t in self.g.triples((s, RDF.type, None)):
            types.add(str(t))
        for _, _, t in self.g.triples((s, WDT.P31, None)):  # Wikidata
            types.add(str(t))
        return sorted(types)

    def _labels(self, iri: str) -> List[str]:
        out: Set[str] = set()
        s = URIRef(iri)
        label_props = [RDFS.label, SKOS.prefLabel, SKOS.altLabel, SCHEMA.name, WDT.P1476]
        for lp in label_props:
            for _, _, o in self.g.triples((s, lp, None)):
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
                return FactualResult(values=values, meta={
                    "source": "KG",
                    "subject_label": cand.label,
                    "subject_iri": cand.iri,
                    "predicate": relation_spec.predicate,
                })               
        return FactualResult(values=[], meta={"source": "KG"})