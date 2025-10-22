from __future__ import annotations
from dataclasses import dataclass
from typing import List, Dict, Optional
import logging
import numpy as np
from sklearn.metrics import pairwise_distances

from .config import DATA_ROOT, EMBED_SUBDIR_CANDIDATES, EMBED_PATTERNS
from .graph_executor import GraphExecutor
from .utils import pick_first_file

log = logging.getLogger(__name__)

@dataclass
class EmbeddingHit:
    label: str
    iri: str
    score: float

@dataclass
class EmbeddingResult:
    topk: List[EmbeddingHit]
    meta: Dict

class EmbeddingExecutor:
    def __init__(self, graph_exec: GraphExecutor):
        self.ge = graph_exec
        self.entity_vecs = None
        self.relation_vecs = None
        self.ent2id: Dict[str,int] = {}
        self.id2ent: Dict[int,str] = {}
        self.rel2id: Dict[str,int] = {}
        self.id2rel: Dict[int,str] = {}
        self._predicate_tail_set: Dict[str, List[str]] = {}
        self._load_embeddings()

    def _load_embeddings(self):
        ent_ids = None; rel_ids = None; ent_vec = None; rel_vec = None
        for root in EMBED_SUBDIR_CANDIDATES:
            if not root.exists():
                continue
            if ent_ids is None:
                ent_ids = pick_first_file(root, EMBED_PATTERNS["entity_ids"])
            if rel_ids is None:
                rel_ids = pick_first_file(root, EMBED_PATTERNS["relation_ids"])
            if ent_vec is None:
                ent_vec = pick_first_file(root, EMBED_PATTERNS["entity_embeds"])
            if rel_vec is None:
                rel_vec = pick_first_file(root, EMBED_PATTERNS["relation_embeds"])
        if not all([ent_ids, rel_ids, ent_vec, rel_vec]):
            raise FileNotFoundError("Embedding files not found under /space_mounts/atai-hs25/dataset")

        def load_map(p) -> Dict[str,int]:
            m: Dict[str,int] = {}
            with open(p, "r", encoding="utf-8", errors="ignore") as f:
                for line in f:
                    line = line.strip()
                    if not line:
                        continue
                    parts = [x.strip() for x in line.split("\t")]
                    if len(parts) == 2:
                        a, b = parts
                    else:
                        parts = [x.strip() for x in line.split(",")]
                        if len(parts) != 2:
                            continue
                        a, b = parts
                    if a.isdigit() and not b.isdigit():
                        m[b] = int(a)
                    elif b.isdigit() and not a.isdigit():
                        m[a] = int(b)
            return m

        self.ent2id = load_map(ent_ids)
        self.rel2id = load_map(rel_ids)
        self.id2ent = {v:k for k,v in self.ent2id.items()}
        self.id2rel = {v:k for k,v in self.rel2id.items()}
        self.entity_vecs = np.load(ent_vec)
        self.relation_vecs = np.load(rel_vec)
        log.info("Embeddings loaded: entities %s, relations %s",
                 self.entity_vecs.shape, self.relation_vecs.shape)

    def _predicate_tails(self, predicate_iri: str) -> List[str]:
        if predicate_iri in self._predicate_tail_set:
            return self._predicate_tail_set[predicate_iri]
        tails = set()
        pred = predicate_iri
        for s, p, o in self.ge.g.triples((None, None, None)):
            if str(p) == pred:
                val = str(o)
                if val.startswith("http://") or val.startswith("https://"):
                    tails.add(val)
        self._predicate_tail_set[predicate_iri] = list(tails)
        return self._predicate_tail_set[predicate_iri]

    def _entity_vec(self, iri: str):
        eid = self.ent2id.get(iri)
        if eid is None:
            return None
        return self.entity_vecs[eid]

    def _relation_vec(self, predicate_iri: str):
        rid = self.rel2id.get(predicate_iri)
        if rid is None:
            return None
        return self.relation_vecs[rid]

    def _pretty_label(self, iri: str) -> str:
        labs = self.ge._labels(iri)
        if labs:
            return labs[0]
        # fallback to Q-id if no label present in this dataset
        return iri.rsplit("/", 1)[-1]

    def _round_score(self, x: float) -> float:
        # clamp to [-1, 1], round to 4 decimals, avoid perfect 1.0 saturation
        x = max(-1.0, min(1.0, x))
        r = round(x, 4)
        if r >= 0.99995:
            return 0.9999
        return r

    def query_embedding(self, candidates, relation_spec) -> Optional[EmbeddingResult]:
        if not candidates or not relation_spec:
            return None
        rvec = self._relation_vec(relation_spec.predicate)
        if rvec is None:
            return None

        # pick the first subject candidate that has an embedding
        subj_vec = None
        subj_iri = None
        subj_label = None
        for c in candidates:
            v = self._entity_vec(c.iri)
            if v is not None:
                subj_vec = v; subj_iri = c.iri; subj_label = c.label
                break
        if subj_vec is None:
            return None

        # gather predicate-tail candidates (type-filtered via KG)
        tail_iris = self._predicate_tails(relation_spec.predicate)
        tail_ids = [self.ent2id[i] for i in tail_iris if i in self.ent2id]
        if not tail_ids:
            return None
        tail_mat = self.entity_vecs[tail_ids]

        # Simple TransE-like composition: h + r, then cosine similarity
        target = subj_vec + rvec
        dists = pairwise_distances(target.reshape(1, -1), tail_mat, metric="cosine")[0]
        order = np.argsort(dists)  # smaller is closer

        top_hits: List[EmbeddingHit] = []
        for idx in order[:3]:
            ent_id = tail_ids[idx]
            iri = self.id2ent[ent_id]
            label = self._pretty_label(iri)
            sim = 1.0 - float(dists[idx])
            score = self._round_score(sim)
            top_hits.append(EmbeddingHit(label=label, iri=iri, score=score))

        return EmbeddingResult(
            topk=top_hits,
            meta={
                "source": "KG-Embeddings",
                "subject_label": subj_label,
                "subject_iri": subj_iri,
                "predicate": relation_spec.predicate,
            },
        )