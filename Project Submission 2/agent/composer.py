from __future__ import annotations
from typing import Optional, Dict

from .graph_executor import FactualResult
from .embedding_executor import EmbeddingResult
from typing import Optional, Dict, List
import re

def _dedup_and_join(vals: List[str]) -> str:
    # Ignore case and remove duplicates
    seen = {}
    for v in vals:
        k = v.strip().lower()
        if k not in seen or len(v.strip()) > len(seen[k]):
            seen[k] = v.strip()
    return " and ".join(seen.values())

def _pick_type(embedding: EmbeddingResult):
    top1 = embedding.topk[0]
    t = getattr(top1, "type", None) or embedding.meta.get("expected_type")
    if t and ":" in t: t = t.split(":")[-1]
    return t


class AnswerComposer:
    def compose(self, factual: Optional[FactualResult], embedding: Optional[EmbeddingResult]) -> Dict:
        resp = {}

        # factual
        if factual and factual.values:
            #  Convert factual answers to natural language if possible
            answers = [str(x) for x in factual.values]
            text = f"The factual answer is: {_dedup_and_join(answers)}"
            resp["factual_answer"] = {"answer": [text], "meta": factual.meta}
            # answers = factual.values
            # relation = factual.meta.get("relation", "").lower()
            # text =f"The factual answer is {'"and" '.join(answers)}."
            # resp["factual_answer"] = {"answer": [text], "meta": factual.meta}

        # embedding
        if embedding and embedding.topk:
            top1 = embedding.topk[0]
            # Convert embedding answer to natural language
            etype = _pick_type(embedding)
            if etype:
                text = f"The answer suggested by embeddings is: {top1.label} (type: {etype})"
            else:
                text = f"The answer suggested by embeddings is: {top1.label}"
            # text = f"The answer suggested by embeddings is: {top1.label}."
            resp["embedding_answer"] = {
                "answer": [text],
                # "score": top1.score (only need type)
                "meta": {
                    **embedding.meta,
                    "topk": [h.__dict__ for h in embedding.topk],
                },
            }

        # # both
        # if "factual_answer" in resp and "embedding_answer" in resp:
        #     fact_vals = set(resp["factual_answer"]["answer"]) if resp["factual_answer"]["answer"] else set()
        #     emb_top = resp["embedding_answer"]["answer"].replace("(Embedding Answer) ", "")
        #     # if fact_vals and emb_top not in fact_vals:
        #     #     resp["note"] = "KG factual answer is authoritative; embedding is shown as a suggestion."

        return resp