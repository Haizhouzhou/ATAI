import json
import sys
from agent.nlq import parse_nlq
from agent.entity_linker import EntityLinker
from agent.relation_mapper import RelationMapper
from agent.graph_executor import GraphExecutor
from agent.embedding_executor import EmbeddingExecutor
from agent.composer import AnswerComposer


if __name__ == "__main__":
    el = EntityLinker()
    rm = RelationMapper()
    ge = GraphExecutor()
    ee = EmbeddingExecutor(ge)
    ac = AnswerComposer()


    path = "tests/e2e_samples.jsonl"
    if len(sys.argv) > 1:
        path = sys.argv[1]


    for line in open(path, "r", encoding="utf-8"):
        q = json.loads(line)["q"]
        print("\nQ:", q)


        nlq = parse_nlq(q)
        cands = el.link(nlq.entity_strings)
        rel = rm.map_relation(nlq.relation_triggers)


        factual = ge.query_factual(cands, rel) if (cands and rel) else None
        embedding = None
        if (not factual or not factual.values):
            if cands and rel:
                embedding = ee.query_embedding(cands, rel)


        resp = ac.compose(factual, embedding)


        print(json.dumps(resp, ensure_ascii=False, indent=2))