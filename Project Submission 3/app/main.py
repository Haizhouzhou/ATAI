import logging
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel

# --- All Chatbot Imports ---
from agent.config import Config
from agent.graph_executor import GraphExecutor
from agent.embedding_executor import EmbeddingExecutor
from agent.entity_linker import EntityLinker
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
# --- CHATBOT LOGIC CLASS (from previous steps) ---
# This class contains all the Eval 2 and Eval 3 logic
# ######################################################################

class Chatbot:
    """
    The main Chatbot class, orchestrating all components.
    This class is NOT the web app itself.
    """
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
            self.entity_linker = EntityLinker(self.config.label_index)
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

    def process_nl_query(self, query: str, session: Session) -> str:
        logger.info(f"Processing query for session {session.user_id}: '{query}'")
        query_lower = query.lower()

        if query.strip().lower() in ["clear", "reset", "start over"]:
            session.clear()
            return "Okay, let's start fresh. What can I help you with?"

        if query.strip().lower() in ["help", "info"]:
            return self.get_help_message()

        try:
            relation = self.relation_mapper.map_relation(query)
            
            if relation == PREDICATE_MAP['award'] and 'nominat' in query_lower:
                relation = PREDICATE_MAP['nominated for']

            if relation:
                logger.info(f"Relation '{relation}' found. Routing to QA Engine.")
                return self.nlq_processor.process_query(query, pre_mapped_relation=relation)

            logger.debug("No relation found. Parsing for recommendation intent.")
            parsed_intent = self.preference_parser.parse(query, session)
            session.update(parsed_intent)
            
            if parsed_intent['intent'] == 'recommendation':
                logger.info("Routing to Recommendation Engine.")
                return self.handle_recommendation(session)

            logger.warning(f"No clear intent. Defaulting to QA.")
            return self.nlq_processor.process_query(query)

        except Exception as e:
            logger.error(f"Error during query processing: {e}", exc_info=True)
            return "I'm sorry, I had trouble understanding that. Could you try rephrasing?"

    def handle_recommendation(self, session: Session) -> str:
        try:
            recommendations = self.recommendation_engine.get_recommendations(session, top_k=5)
            
            if not recommendations:
                if session.seed_movies or session.preferences:
                    return "I searched based on your preferences but couldn't find any matching movies."
                else:
                    return "I can give you recommendations if you tell me a movie you like!"

            response_parts = ["Here are a few recommendations:"]
            rec_movies = []
            for rec in recommendations:
                movie_label = rec['label']
                reason = rec['reason']
                response_parts.append(f"\n- **{movie_label}**: {reason}")
                rec_movies.append(rec['id'])
            
            session.add_recommendations(rec_movies)
            return " ".join(response_parts)

        except Exception as e:
            logger.error(f"Error in recommendation handling: {e}", exc_info=True)
            return "I'm sorry, I had trouble finding recommendations. Please try again."

    def get_help_message(self) -> str:
        return (
            "I can help you in two ways:\n"
            "1. **Answer questions** about movies (e.g., 'Who directed Fargo?').\n"
            "2. **Give recommendations** (e.g., 'Recommend a movie like The Lion King').\n"
            "You can type 'clear' to reset our conversation."
        )

# ######################################################################
# --- FastAPI WEB SERVER (from Submission 2) ---
# This is the runnable part that uvicorn needs.
# ######################################################################

# THIS IS THE FIX: Define the "app" object
app = FastAPI()

# --- Global Instances ---
chatbot_instance: Chatbot = None
session_manager: SessionManager = None

# --- Request Models (from Submission 2) ---
class NLQRequest(BaseModel):
    query: str
    user_id: str # user_id will be the session key

class NLQResponse(BaseModel):
    answer: str

# --- Startup Event (Modified) ---
@app.on_event("startup")
async def startup_event():
    global chatbot_instance, session_manager
    logger.info("Starting up... initializing Chatbot and SessionManager.")
    try:
        # Initialize the chatbot logic and session manager when the server starts
        chatbot_instance = Chatbot()
        session_manager = SessionManager()
        logger.info("Startup complete. Server is ready.")
    except Exception as e:
        logger.error(f"Startup failed: {e}", exc_info=True)
        # This will still let the server start, but endpoints will fail
        
# --- Endpoint (Modified) ---
@app.post("/nlq", response_model=NLQResponse)
async def handle_nlq(request: NLQRequest):
    if not chatbot_instance or not session_manager:
        logger.error("Server is not fully initialized.")
        raise HTTPException(status_code=503, detail="Server is not ready. Please try again in a moment.")
    
    try:
        # Get the session for this user
        user_session = session_manager.get_session(request.user_id)
        
        # Process the query using the chatbot logic
        answer = chatbot_instance.process_nl_query(request.query, user_session)
        
        return NLQResponse(answer=answer)
    except Exception as e:
        logger.error(f"Error handling request: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail="I'm sorry, I encountered an internal error.")