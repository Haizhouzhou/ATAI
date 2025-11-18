import logging
from typing import Dict, Any, Optional
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel

# --- Chatbot Imports ---
from agent.config import Config
from agent.graph_executor import GraphExecutor
from agent.embedding_executor import EmbeddingExecutor
from agent.entity_linker import EntityLinker  # <--- Ensuring this is imported
from agent.relation_mapper import RelationMapper
from agent.composer import Composer
from agent.nlq import NLQ
from agent.session_manager import Session, SessionManager
from agent.preference_parser import PreferenceParser
from agent.recommendation_engine import RecommendationEngine
from agent.constants import PREDICATE_MAP

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# ######################################################################
# --- CHATBOT LOGIC CLASS ---
# ######################################################################

class Chatbot:
    def __init__(self):
        try:
            logger.info("Initializing Chatbot...")
            self.config = Config()
            
            self.graph_executor = GraphExecutor(self.config.graph_db)
            
            self.embedding_executor = EmbeddingExecutor(
                self.config.entity_embeddings,
                self.config.entity_index,
                self.config.relation_embeddings,
                self.config.relation_index
            )
            
            # Initialize Entity Linker (This matches the class name in agent/entity_linker.py)
            self.entity_linker = EntityLinker()
            
            self.relation_mapper = RelationMapper()
            self.composer = Composer()

            self.nlq_processor = NLQ(
                graph_executor=self.graph_executor,
                embedding_executor=self.embedding_executor,
                entity_linker=self.entity_linker,
                relation_mapper=self.relation_mapper,
                composer=self.composer
            )
            
            self.preference_parser = PreferenceParser(
                entity_linker=self.entity_linker,
                relation_mapper=self.relation_mapper
            )
            
            self.recommendation_engine = RecommendationEngine(
                graph_executor=self.graph_executor,
                embedding_executor=self.embedding_executor,
                composer=self.composer,
                entity_linker=self.entity_linker
            )
            
            logger.info("Chatbot initialized successfully.")
            
        except Exception as e:
            logger.error(f"Failed to initialize Chatbot components: {e}", exc_info=True)
            raise

    def process_nl_query(self, query: str, session: Session) -> Dict[str, Any]:
        logger.info(f"Processing query for session {session.user_id}: '{query}'")
        query_lower = query.lower()

        if query.strip().lower() in ["clear", "reset", "start over"]:
            session.clear()
            return {"note": "Okay, let's start fresh. What can I help you with?"}

        if query.strip().lower() in ["help", "info"]:
            return {"note": self.get_help_message()}

        try:
            # 1. Check Intent FIRST (Priority to Recommendations)
            parsed_intent = self.preference_parser.parse(query, session)
            
            if parsed_intent['intent'] == 'recommendation':
                logger.info(f"Intent detected as 'recommendation'. Updating session and routing.")
                session.update(parsed_intent)
                return self.handle_recommendation(session)

            # 2. Attempt Relation Mapping (QA Intent)
            relation = self.relation_mapper.map_relation(query)
            if relation == PREDICATE_MAP['award'] and 'nominat' in query_lower:
                relation = PREDICATE_MAP['nominated for']

            if relation:
                logger.info(f"Relation '{relation}' found. Routing to QA Engine.")
                ans_str = self.nlq_processor.process_query(query, pre_mapped_relation=relation)
                return {"note": ans_str}

            # 3. Fallback: Ambiguous Recs
            if parsed_intent['seed_movies'] or parsed_intent['preferences'] or parsed_intent['constraints']:
                 logger.info("No relation found, but parsed potential recommendation entities. Routing to Rec Engine.")
                 session.update(parsed_intent)
                 return self.handle_recommendation(session)

            # 4. Ultimate Fallback to QA
            logger.warning(f"No clear intent. Defaulting to QA fallback.")
            ans_str = self.nlq_processor.process_query(query)
            return {"note": ans_str}

        except Exception as e:
            logger.error(f"Error during query processing: {e}", exc_info=True)
            return {"note": "I'm sorry, I had trouble understanding that. Could you try rephrasing?"}

    def handle_recommendation(self, session: Session) -> Dict[str, Any]:
        try:
            recommendations = self.recommendation_engine.get_recommendations(session, top_k=5)
            
            if not recommendations:
                msg = "I searched based on your preferences but couldn't find any matching movies."
                if not session.seed_movies and not session.preferences:
                    msg = "I can give you recommendations if you tell me a movie you like!"
                return {"note": msg}

            return {
                "recommendations": recommendations, 
                "note": "Here are a few recommendations:"
            }

        except Exception as e:
            logger.error(f"Error in recommendation handling: {e}", exc_info=True)
            return {"note": "I'm sorry, I had trouble finding recommendations. Please try again."}

    def get_help_message(self) -> str:
        return (
            "I can help you in two ways:\n"
            "1. Answer questions about movies (e.g., 'Who directed Fargo?').\n"
            "2. Give recommendations (e.g., 'Recommend a movie like The Lion King').\n"
            "You can type 'clear' to reset our conversation."
        )

# ######################################################################
# --- FastAPI WEB SERVER ---
# ######################################################################

app = FastAPI()

chatbot_instance: Chatbot = None
session_manager: SessionManager = None

class NLQRequest(BaseModel):
    query: str
    user_id: Optional[str] = "guest" 

@app.on_event("startup")
async def startup_event():
    global chatbot_instance, session_manager
    logger.info("Starting up... initializing Chatbot and SessionManager.")
    try:
        chatbot_instance = Chatbot()
        session_manager = SessionManager()
        logger.info("Startup complete. Server is ready.")
    except Exception as e:
        logger.error(f"Startup failed: {e}", exc_info=True)

@app.get("/health")
async def health_check():
    return {"status": "ok"}

@app.post("/ask")
async def handle_ask(request: NLQRequest):
    if not chatbot_instance or not session_manager:
        raise HTTPException(status_code=503, detail="Server is not ready.")
    
    try:
        user_session = session_manager.get_session(request.user_id)
        response_data = chatbot_instance.process_nl_query(request.query, user_session)
        return response_data
    except Exception as e:
        logger.error(f"Error handling request: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail="Internal error.")