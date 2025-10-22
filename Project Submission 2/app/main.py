from fastapi import FastAPI
from pydantic import BaseModel

from agent.nlq import parse_nlq
from agent.entity_linker import EntityLinker
from agent.relation_mapper import RelationMapper
from agent.graph_executor import GraphExecutor
from agent.embedding_executor import EmbeddingExecutor
from agent.composer import AnswerComposer
from agent.logging_config import configure_logging

configure_logging()
app = FastAPI(title="ATAI IE2 Agent", version="0.1.0")

# Initialize singletons (heavy loads are lazy inside classes)
el = EntityLinker()
rm = RelationMapper()
ge = GraphExecutor()
ee = EmbeddingExecutor(ge)
ac = AnswerComposer()

class AskRequest(BaseModel):
    query: str

@app.get("/health")
def health():
    return {"status": "ok"}

@app.post("/ask")
def ask(req: AskRequest):
    nlq = parse_nlq(req.query)

    # 1) entity linking (top-k)
    candidates = el.link(nlq.entity_strings)

    # 2) relation mapping
    rel = rm.map_relation(nlq.relation_triggers)

    factual = None
    embedding = None

    # Factual first (if we have entity + relation)
    if candidates and rel:
        factual = ge.query_factual(candidates, rel)

    # Embedding next if requested/needed (Both default or explicit embedding intent)
    if nlq.intent in ("embedding", "both") and candidates and rel:
        embedding = ee.query_embedding(candidates, rel)

    # Fallback: if intent was factual but KG has no answer, still try embedding as backup
    if nlq.intent == "factual" and (not factual or not factual.values):
        if candidates and rel:
            embedding = ee.query_embedding(candidates, rel)

    return ac.compose(factual, embedding)