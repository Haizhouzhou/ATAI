import logging
from agent.constants import PREDICATE_MAP

logger = logging.getLogger(__name__)

class QAEngine:
    def __init__(self, graph, linker, emb, mm, composer):
        self.graph = graph
        self.linker = linker
        self.emb = emb
        self.mm = mm
        self.composer = composer

    def answer_question(self, query: str) -> str:
        q_lower = query.lower()
        
        target_pred = None
        for k in sorted(PREDICATE_MAP.keys(), key=len, reverse=True):
            if k in q_lower:
                target_pred = PREDICATE_MAP[k]
                break
        
        if not target_pred:
            if "who" in q_lower: target_pred = PREDICATE_MAP["director"]
            elif "when" in q_lower: target_pred = PREDICATE_MAP["publication date"]
            else: return None

        linked = self.linker.link(query)
        if not linked: return "I couldn't identify the subject."
        linked.sort(key=lambda x: x[2], reverse=True)
        lbl, uri, _ = linked[0]

        # 1. Graph Lookup
        for direct in ["forward", "backward"]:
            q = self.composer.compose_one_hop_qa(uri, target_pred, direct)
            res = self.graph.execute_query(q)
            answers = []
            for r in res:
                # Try Label from SPARQL
                ans = str(r.get('oLabel', r.get('sLabel', '')))
                # If empty, get ID and lookup in Linker dict
                if not ans:
                    val = str(r.get('o', r.get('s', '')))
                    if val: ans = self.linker.get_label(val)
                if ans: answers.append(ans)
            
            if answers:
                return f"The answer is: {', '.join(list(set(answers))[:10])}"

        # 2. Embedding Fallback
        preds = self.emb.predict_tail(uri, target_pred, k=1)
        if preds:
            p_uri, _ = preds[0]
            p_lbl = self.linker.get_label(p_uri)
            return f"I think it is {p_lbl}."
            
        return f"I found {lbl}, but couldn't answer that."