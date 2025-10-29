from __future__ import annotations
from dataclasses import dataclass
from typing import List, Dict, Optional
import logging
from collections import Counter

import numpy as np
from rdflib import URIRef

from .config import EMBED_SUBDIR_CANDIDATES, EMBED_PATTERNS
from .graph_executor import GraphExecutor
from .utils import pick_first_file

log = logging.getLogger(__name__)

MAX_TAILS = 20000
TOPK = 3


@dataclass
class EmbeddingHit:
    """Single embedding retrieval candidate."""
    label: str
    iri: str
    score: float
    type: Optional[str] = None  # short type identifier for display (e.g., Q5, Person)


@dataclass
class EmbeddingResult:
    """Top-k embedding retrieval result and auxiliary metadata."""
    topk: List[EmbeddingHit]
    meta: Dict


class EmbeddingExecutor:
    """
    Embedding-based query executor.

    Strategy:
    1) Prefer candidate tails from the 1-hop neighborhood: (subject, predicate, ?o).
       This keeps candidates semantically close to the asked relation.
    2) Fallback to all tails of the same predicate across the graph.
    3) Score candidates with a TransE-style target: target = subj_vec + rel_vec.
    4) Attach readable labels and a short type for the top hit.
    5) Provide an 'expected_type' in meta as a fallback, derived from the predicate's
       majority object type across the graph.
    """

    def __init__(self, graph_exec: GraphExecutor):
        self.ge = graph_exec

        # Embedding matrices and normalized copies
        self.entity_vecs: Optional[np.ndarray] = None
        self.entity_vecs_norm: Optional[np.ndarray] = None
        self.relation_vecs: Optional[np.ndarray] = None
        self.relation_vecs_norm: Optional[np.ndarray] = None

        # ID mappings
        self.ent2id: Dict[str, int] = {}
        self.id2ent: Dict[int, str] = {}
        self.rel2id: Dict[str, int] = {}
        self.id2rel: Dict[int, str] = {}

        # Caches
        self._predicate_tail_set: Dict[str, List[str]] = {}
        self._pred2_major_type: Dict[str, str] = {}

        self._load_embeddings()

    # -------------------------
    # Loading & utilities
    # -------------------------

    def _load_embeddings(self):
        """Load embedding files (IDs and vectors), build mappings, and L2-normalize."""
        ent_ids = None
        rel_ids = None
        ent_vec = None
        rel_vec = None

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
            raise FileNotFoundError(
                "Embedding files not found under /space_mounts/atai-hs25/dataset"
            )

        def load_map(p) -> Dict[str, int]:
            """Load a mapping file with two columns (ID<->IRI), tab or comma separated."""
            m: Dict[str, int] = {}
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
        self.id2ent = {v: k for k, v in self.ent2id.items()}
        self.id2rel = {v: k for k, v in self.rel2id.items()}

        self.entity_vecs = np.load(ent_vec).astype(np.float32)
        self.relation_vecs = np.load(rel_vec).astype(np.float32)

        def l2norm(X: np.ndarray) -> np.ndarray:
            eps = 1e-12
            n = np.linalg.norm(X, axis=1, keepdims=True)
            n = np.maximum(n, eps)
            return X / n

        self.entity_vecs_norm = l2norm(self.entity_vecs)
        self.relation_vecs_norm = l2norm(self.relation_vecs)

        log.info(
            "Embeddings loaded & normalized: entities %s, relations %s",
            self.entity_vecs.shape,
            self.relation_vecs.shape,
        )

    # --- graph helpers ---

    def _predicate_tails_for_subject(self, subject_iri: str, predicate_iri: str) -> List[str]:
        """
        Collect tails restricted to a given (subject, predicate, ?o) pattern,
        and keep only entities that have embeddings.
        """
        pred_ref = URIRef(predicate_iri)
        subj_ref = URIRef(subject_iri)
        tails: List[str] = []
        for _, _, o in self.ge.g.triples((subj_ref, pred_ref, None)):
            val = str(o)
            if (val.startswith("http://") or val.startswith("https://")) and (val in self.ent2id):
                tails.append(val)
        return tails

    def _predicate_tails(self, predicate_iri: str) -> List[str]:
        """
        Collect all tails for a predicate across the graph, keep only those
        that have embeddings, and optionally downsample to MAX_TAILS.
        """
        if predicate_iri in self._predicate_tail_set:
            return self._predicate_tail_set[predicate_iri]

        pred_ref = URIRef(predicate_iri)
        tails = set()
        for _, _, o in self.ge.g.triples((None, pred_ref, None)):
            val = str(o)
            if val.startswith("http://") or val.startswith("https://"):
                tails.add(val)

        filtered = [iri for iri in tails if iri in self.ent2id]

        if len(filtered) > MAX_TAILS:
            step = max(1, len(filtered) // MAX_TAILS)
            filtered = filtered[::step][:MAX_TAILS]

        self._predicate_tail_set[predicate_iri] = filtered
        return filtered

    # --- NEW: Head prediction helpers ---
    
    def _predicate_heads_for_object(self, predicate_iri: str, object_iri: str) -> List[str]:
        """Collect heads restricted to a given (?s, p, object) pattern."""
        pred_ref = URIRef(predicate_iri)
        obj_ref = URIRef(object_iri)
        heads: List[str] = []
        for s, _, _ in self.ge.g.triples((None, pred_ref, obj_ref)):
            val = str(s)
            if (val.startswith("http://") or val.startswith("https://")) and (val in self.ent2id):
                heads.append(val)
        return heads

    def _predicate_heads(self, predicate_iri: str) -> List[str]:
        """Collect all heads for a predicate across the graph (global fallback)."""
        cache_key = f"__HEADS__::{predicate_iri}"
        if cache_key in self._predicate_tail_set:
            return self._predicate_tail_set[cache_key]

        pred_ref = URIRef(predicate_iri)
        heads = set()
        for s, _, _ in self.ge.g.triples((None, pred_ref, None)):
            val = str(s)
            if val.startswith("http://") or val.startswith("https://"):
                heads.add(val)

        filtered = [iri for iri in heads if iri in self.ent2id]
        if len(filtered) > MAX_TAILS:
            step = max(1, len(filtered) // MAX_TAILS)
            filtered = filtered[::step][:MAX_TAILS]

        self._predicate_tail_set[cache_key] = filtered
        return filtered

    # --- End new helpers ---

    def _entity_vec_norm(self, iri: str) -> Optional[np.ndarray]:
        """Return the normalized embedding vector of an entity, if available."""
        eid = self.ent2id.get(iri)
        if eid is None:
            return None
        return self.entity_vecs_norm[eid]

    def _relation_vec_norm(self, predicate_iri: str) -> Optional[np.ndarray]:
        """Return the normalized embedding vector of a predicate (relation), if available."""
        rid = self.rel2id.get(predicate_iri)
        if rid is None:
            return None
        return self.relation_vecs_norm[rid]

    def _pretty_label(self, iri: str) -> str:
        """Prefer readable labels from the graph; fallback to the IRI tail."""
        labs = self.ge._labels(iri)
        if labs:
            return labs[0]
        return iri.rsplit("/", 1)[-1]

    def _short_tail(self, iri: str) -> str:
        """Return the tail of an IRI for concise display (supports both '#' and '/')."""
        if "#" in iri:
            return iri.rsplit("#", 1)[-1]
        return iri.rsplit("/", 1)[-1]

    def _major_object_type_for_predicate(self, predicate_iri: str) -> Optional[str]:
        """
        Compute the majority object type (short tail string) for a predicate by scanning
        its observed tails and their rdf:type / instance-of types.
        """
        if predicate_iri in self._pred2_major_type:
            return self._pred2_major_type[predicate_iri]

        pred_ref = URIRef(predicate_iri)
        types: List[str] = []
        for _, _, o in self.ge.g.triples((None, pred_ref, None)):
            o_iri = str(o)
            if not o_iri.startswith("http"):
                continue
            for t in self.ge._types(o_iri):  # requires GraphExecutor._types
                types.append(self._short_tail(t))

        if not types:
            return None

        maj = Counter(types).most_common(1)[0][0]
        self._pred2_major_type[predicate_iri] = maj
        return maj

    def _round_score(self, x: float) -> float:
        """Clamp and round cosine similarity for stable logging / inspection."""
        x = float(np.clip(x, -1.0, 1.0))
        r = round(x, 4)
        if r >= 0.99995:
            return 0.9999
        return r

    # -------------------------
    # Query
    # -------------------------

    def query_embedding(self, candidates, relation_spec) -> Optional[EmbeddingResult]:
        """
        Produce an embedding-based answer given linked subject candidates and a mapped predicate.
        Returns top-k hits with labels and (short) type; meta includes an expected_type fallback.
        (Tail prediction: h + r = ?t)
        """
        if not candidates or not relation_spec:
            return None

        # Relation vector
        rvec = self._relation_vec_norm(relation_spec.predicate)
        if rvec is None:
            return None

        # Pick the first subject that has an embedding
        subj_vec = None
        subj_iri = None
        subj_label = None
        for c in candidates:
            v = self._entity_vec_norm(c.iri)
            if v is not None:
                subj_vec = v
                subj_iri = c.iri
                subj_label = c.label
                break
        if subj_vec is None:
            return None

        # Candidate tails: subject-specific first, then global fallback
        tail_iris = self._predicate_tails_for_subject(subj_iri, relation_spec.predicate)
        if not tail_iris:
            tail_iris = self._predicate_tails(relation_spec.predicate)
        if not tail_iris:
            return None

        tail_ids = [self.ent2id[i] for i in tail_iris]
        tail_mat = self.entity_vecs_norm[tail_ids]

        # TransE-style target: target = subj + rel
        target = subj_vec + rvec
        norm = np.linalg.norm(target)
        if norm > 1e-12:
            target = target / norm

        sims = tail_mat @ target

        k = min(TOPK, sims.shape[0])
        idx_part = np.argpartition(-sims, k - 1)[:k]
        idx_sorted = idx_part[np.argsort(-sims[idx_part])]

        top_hits: List[EmbeddingHit] = []
        for j in idx_sorted:
            ent_id = tail_ids[j]
            iri = self.id2ent[ent_id]
            label = self._pretty_label(iri)
            score = self._round_score(sims[j])
            # Attach a short type for display; fallback handled in composer via meta
            types = self.ge._types(iri)  # requires GraphExecutor._types
            t_short = self._short_tail(types[0]) if types else None
            top_hits.append(EmbeddingHit(label=label, iri=iri, score=score, type=t_short))

        # Expected (fallback) type: majority object type for this predicate
        expected_type = self._major_object_type_for_predicate(relation_spec.predicate)

        return EmbeddingResult(
            topk=top_hits,
            meta={
                "source": "KG-Embeddings",
                "subject_label": subj_label,
                "subject_iri": subj_iri,
                "predicate": relation_spec.predicate,
                "expected_type": expected_type,
            },
        )

    # --- NEW: Head prediction query ---
    
    def query_embedding_head(self, tail_candidates, relation_spec) -> Optional[EmbeddingResult]:
        """
        Head prediction: given (?h, predicate, tail) return top-k head entities.
        TransE target: target = tail_vec - rel_vec
        """
        if not tail_candidates or not relation_spec:
            return None

        # relation vector
        rvec = self._relation_vec_norm(relation_spec.predicate)
        if rvec is None:
            return None

        # pick first tail that has an embedding
        tail_vec = None
        tail_iri = None
        tail_label = None
        for c in tail_candidates:
            v = self._entity_vec_norm(c.iri)
            if v is not None:
                tail_vec = v
                tail_iri = c.iri
                tail_label = c.label
                break
        if tail_vec is None:
            return None

        # candidate heads: object-specific first, then global fallback
        head_iris = self._predicate_heads_for_object(relation_spec.predicate, tail_iri)
        if not head_iris:
            head_iris = self._predicate_heads(relation_spec.predicate)
        if not head_iris:
            return None

        head_ids = [self.ent2id[i] for i in head_iris]
        head_mat = self.entity_vecs_norm[head_ids]

        # TransE-style target for head: target = tail - rel
        target = tail_vec - rvec
        norm = np.linalg.norm(target)
        if norm > 1e-12:
            target = target / norm

        sims = head_mat @ target

        k = min(TOPK, sims.shape[0])
        idx_part = np.argpartition(-sims, k - 1)[:k]
        idx_sorted = idx_part[np.argsort(-sims[idx_part])]

        top_hits: List[EmbeddingHit] = []
        for j in idx_sorted:
            ent_id = head_ids[j]
            iri = self.id2ent[ent_id]
            label = self._pretty_label(iri)
            score = self._round_score(sims[j])
            types = self.ge._types(iri)
            t_short = self._short_tail(types[0]) if types else None
            top_hits.append(EmbeddingHit(label=label, iri=iri, score=score, type=t_short))

        # Optional: implement a _major_subject_type_for_predicate
        expected_type = None

        return EmbeddingResult(
            topk=top_hits,
            meta={
                "source": "KG-Embeddings",
                "object_label": tail_label,
                "object_iri": tail_iri,
                "predicate": relation_spec.predicate,
                "expected_type": expected_type,
            },
        )