# Project Submission 2/agent/composer.py

from __future__ import annotations
from dataclasses import asdict
from typing import Optional, Dict, List
import logging

from .graph_executor import FactualResult
from .embedding_executor import EmbeddingResult

log = logging.getLogger(__name__)

# If your grader requires the embedding sentence to start with "(Embedding Answer) ",
# set this to True. Otherwise keep False to match the course examples.
EMBED_PREFIX = False


def _dedup_and_join(vals: List[str]) -> str:
    """
    Deduplicate answers ignoring case, keep the longest surface form per key,
    then join with ' and ' as requested by the rubric.
    """
    seen: Dict[str, str] = {}
    for v in vals or []:
        if v is None:
            continue
        s = str(v).strip()
        if not s:
            continue
        k = s.lower()
        if k not in seen or len(s) > len(seen[k]):
            seen[k] = s
    return " and ".join(seen.values())


def _pick_type(embedding: EmbeddingResult) -> Optional[str]:
    """
    Prefer the type attached to top-1 candidate; fallback to meta['expected_type'].
    Normalize to the short tail if it contains a colon (e.g., 'wikidata:Q5' -> 'Q5').
    """
    if not embedding.topk:
        return None
    top1 = embedding.topk[0]
    t = getattr(top1, "type", None) or embedding.meta.get("expected_type")
    if t and ":" in t:
        t = t.split(":", 1)[-1]
    return t


class AnswerComposer:
    """
    Compose final API response from factual and/or embedding results.

    Output format matches the course description:
      - factual: "The factual answer is: A and B"
      - embedding: "The answer suggested by embeddings is: X (type: Q5)"
    """

    def compose(
        self,
        factual: Optional[FactualResult],
        embedding: Optional[EmbeddingResult],
    ) -> Dict:
        resp: Dict = {}

        # ---------- factual ----------
        if factual and factual.values:
            answers = [str(x) for x in factual.values if x is not None]
            text = f"The factual answer is: {_dedup_and_join(answers)}"
            resp["factual_answer"] = {
                "answer": [text],
                "meta": factual.meta,
            }

        # ---------- embedding ----------
        if embedding and embedding.topk:
            top1 = embedding.topk[0]
            etype = _pick_type(embedding)

            prefix = "(Embedding Answer) " if EMBED_PREFIX else ""
            if etype:
                text = f"{prefix}The answer suggested by embeddings is: {top1.label} (type: {etype})"
            else:
                text = f"{prefix}The answer suggested by embeddings is: {top1.label}"

            # include top-k for debugging/inspection; the grader will only read the sentence
            resp["embedding_answer"] = {
                "answer": [text],
                "meta": {
                    **embedding.meta,
                    "topk": [asdict(h) if hasattr(h, "__dict__") else h for h in embedding.topk],
                },
            }

        return resp
