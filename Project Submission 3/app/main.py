import logging
from agent.config import Config
from agent.graph_executor import GraphExecutor
from agent.embedding_executor import EmbeddingExecutor
from agent.entity_linker import EntityLinker
from agent.relation_mapper import RelationMapper
from agent.composer import Composer
from agent.nlq import NLQ
from agent.session_manager import Session
from agent.preference_parser import PreferenceParser
from agent.recommendation_engine import RecommendationEngine
from agent.constants import PREDICATE_MAP

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

class Chatbot:
    """
    The main Chatbot class, orchestrating all components.
    """
    def __init__(self):
        try:
            logger.info("Initializing Chatbot...")
            self.config = Config()
            
            # Initialize core components
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

            # --- Eval 2 Component ---
            self.nlq_processor = NLQ(
                graph_executor=self.graph_executor,
                embedding_executor=self.embedding_executor,
                entity_linker=self.entity_linker,
                relation_mapper=self.relation_mapper,
                composer=self.composer
            )
            
            # --- Eval 3 Components ---
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
        """
        Processes a natural language query, routes it to the correct engine (QA or Rec),
        and returns a natural language response.
        """
        logger.info(f"Processing query for session {session.user_id}: '{query}'")
        query_lower = query.lower()

        # --- Simple Commands ---
        if query.strip().lower() in ["clear", "reset", "start over"]:
            session.clear()
            return "Okay, let's start fresh. What can I help you with?"

        if query.strip().lower() in ["help", "info"]:
            return self.get_help_message()

        # --- Intent-based Routing (Eval 3 Update) ---
        try:
            # 1. Check for factual relation first (Rule 2.3)
            relation = self.relation_mapper.map_relation(query)
            
            # Handle P1411 (nomination) vs P166 (award) ambiguity
            if relation == PREDICATE_MAP['award'] and 'nominat' in query_lower:
                relation = PREDICATE_MAP['nominated for']
                logger.debug("Ambiguity resolved: 'award' -> 'nominated for'")

            if relation:
                logger.info(f"Relation '{relation}' found. Routing to QA Engine.")
                # Pass the pre-mapped relation to save processing time
                return self.nlq_processor.process_query(query, pre_mapped_relation=relation)

            # 2. No relation found, try parsing for recommendation
            logger.debug("No relation found. Parsing for recommendation intent.")
            parsed_intent = self.preference_parser.parse(query, session)
            session.update(parsed_intent)
            
            if parsed_intent['intent'] == 'recommendation':
                logger.info("Routing to Recommendation Engine.")
                return self.handle_recommendation(session)

            # 3. Fallback: No clear relation and no clear rec intent.
            logger.warning(f"No clear intent. Defaulting to QA.")
            return self.nlq_processor.process_query(query)

        except Exception as e:
            logger.error(f"Error during query processing: {e}", exc_info=True)
            return "I'm sorry, I had trouble understanding that. Could you try rephrasing?"

    def handle_recommendation(self, session: Session) -> str:
        """
        Calls the recommendation engine and formats the output.
        """
        try:
            # Get top_k * 2 candidates for diversification, then select top_k
            recommendations = self.recommendation_engine.get_recommendations(session, top_k=5)
            
            if not recommendations:
                if session.seed_movies or session.preferences:
                    return "I searched based on your preferences but couldn't find any matching movies. You could try broadening your search."
                else:
                    return "I can give you recommendations if you tell me a movie you like!"

            # Format the response
            response_parts = ["Here are a few recommendations:"]
            rec_movies = []
            for rec in recommendations:
                movie_label = rec['label']
                reason = rec['reason']
                response_parts.append(f"\n- **{movie_label}**: {reason}")
                rec_movies.append(rec['id'])
            
            # Update session with movies we just recommended
            session.add_recommendations(rec_movies)

            return " ".join(response_parts)

        except Exception as e:
            logger.error(f"Error in recommendation handling: {e}", exc_info=True)
            return "I'm sorry, I had trouble finding recommendations. Please try again."

    def get_help_message(self) -> str:
        return (
            "I can help you in two ways:\n"
            "1. **Answer questions** about movies (e.g., 'Who directed Fargo?').\n"
            "2. **Give recommendations** (e.g., 'Recommend a movie like The Lion King' or 'I want a comedy movie from the 90s').\n"
            "You can type 'clear' to reset our conversation."
        )