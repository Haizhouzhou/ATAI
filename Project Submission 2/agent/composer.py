from __future__ import annotations
from typing import Optional, Dict

from .graph_executor import FactualResult
from .embedding_executor import EmbeddingResult

class AnswerComposer:
    def compose(self, factual: Optional[FactualResult], embedding: Optional[EmbeddingResult]) -> Dict:
        resp = {}

        # factual
        if factual and factual.values:
            # Convert factual answers to natural language if possible
            answers = factual.values
            relation = factual.meta.get("relation", "").lower()

            # Simple natural language templates
            if "director" in relation:
                text = f"I think it is {', '.join(answers)}."
            elif "screenwriter" in relation or "writer" in relation:
                text = f"I think it was written by {', '.join(answers)}."
            elif "publication_date" in relation or "release" in relation:
                text = f"It was released in {', '.join(answers)}."
            elif "genre" in relation:
                text = f"It belongs to the genre of {', '.join(answers)}."
            elif "production_company" in relation:
                text = f"It was produced by {', '.join(answers)}."
            else:
                text = f"I think it is {', '.join(answers)}."

            resp["factual_answer"] = {"answer": [text], "meta": factual.meta}

        # embedding
        if embedding and embedding.topk:
            top1 = embedding.topk[0]
            # Convert embedding answer to natural language
            text = f"The answer suggested by embeddings: {top1.label}."
            resp["embedding_answer"] = {
                "answer": text,
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