from fastapi import FastAPI
from pydantic import BaseModel
from typing import Optional
import logging

from agent.config import Config
from agent.graph_executor import GraphExecutor
from agent.embedding_executor import EmbeddingExecutor
from agent.entity_linker import EntityLinker
from agent.multimedia_index import MultimediaIndex
from agent.composer import Composer
from agent.recommendation_engine import RecommendationEngine
from agent.nlq import QAEngine
from agent.constants import RECOMMENDATION_KEYWORDS, MULTIMEDIA_KEYWORDS

logging.basicConfig(level=logging.INFO)
app = FastAPI()
comps = {}

class QueryRequest(BaseModel):
    query: str
    user_id: Optional[str] = "guest"

@app.on_event("startup")
async def startup():
    cfg = Config()
    graph = GraphExecutor(cfg.graph_path)
    emb = EmbeddingExecutor(cfg.entity_embeds_path, cfg.entity_index_path, 
                            cfg.relation_embeds_path, cfg.relation_index_path)
    linker = EntityLinker(cfg.graph_path, cfg.metadata_dir)
    mm = MultimediaIndex(cfg.images_json_path, cfg.metadata_dir)
    composer = Composer()
    
    comps['qa'] = QAEngine(graph, linker, emb, mm, composer)
    comps['rec'] = RecommendationEngine(graph, emb, linker, mm, composer)
    comps['linker'] = linker
    comps['mm'] = mm

@app.post("/ask")
async def ask(req: QueryRequest):
    q = req.query.lower()
    
    # Multimedia
    if any(k in q for k in MULTIMEDIA_KEYWORDS):
        linked = comps['linker'].link(req.query)
        if linked:
            img = comps['mm'].get_image(linked[0][1])
            if img: return {"answer": f"Image for {linked[0][0]}", "image": img}
            return {"answer": f"No image found for {linked[0][0]}."}

    # Recs
    if any(k in q for k in RECOMMENDATION_KEYWORDS):
        linked = comps['linker'].link(req.query)
        if linked:
            seeds = [x[1] for x in linked]
            recs = comps['rec'].get_recommendations(seeds, {})
            lines = ["Recommendations:"] + [f"- {r['label']}" for r in recs]
            return {"answer": "\n".join(lines), "recommendations": recs}
    
    # QA
    return {"answer": comps['qa'].answer_question(req.query)}