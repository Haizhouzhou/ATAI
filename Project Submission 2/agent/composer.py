from __future__ import annotations
from typing import Optional, Dict

from .graph_executor import FactualResult
from .embedding_executor import EmbeddingResult

class AnswerComposer:
    def compose(self, factual: Optional[FactualResult], embedding: Optional[EmbeddingResult]) -> Dict:
        resp = {}

        # factual
        if factual and factual.values:
            resp["factual_answer"] = {"answer": factual.values, "meta": factual.meta}
        elif factual and not factual.values:
            resp["factual_answer"] = {
                "answer": [],
                "meta": factual.meta | {"note": "Not found in KG"},
            }

        # embedding
        if embedding and embedding.topk:
            top1 = embedding.topk[0]
            resp["embedding_answer"] = {
                "answer": f"(Embedding Answer) {top1.label}",
                "score": top1.score,
                "meta": {
                    **embedding.meta,
                    "topk": [h.__dict__ for h in embedding.topk],
                },
            }

        # conflict note: only add a clean, single policy line
        if "factual_answer" in resp and "embedding_answer" in resp:
            fact_vals = set(resp["factual_answer"]["answer"]) if resp["factual_answer"]["answer"] else set()
            emb_top = resp["embedding_answer"]["answer"].replace("(Embedding Answer) ", "")
            if fact_vals and emb_top not in fact_vals:
                resp["note"] = "KG factual answer is authoritative; embedding is shown as a suggestion."

        return resp