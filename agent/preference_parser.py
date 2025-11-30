import re
import logging
from typing import Dict, Any, List, Tuple
from agent.entity_linker import EntityLinker
from agent.relation_mapper import RelationMapper
from agent.constants import (
    RECOMMENDATION_KEYWORDS, QA_KEYWORDS, NEGATION_KEYWORDS,
    PREFERENCE_KEYWORDS, FOLLOW_UP_KEYWORDS, SUPPORTED_LANGUAGES_REGEX
)
from agent.session_manager import Session

logger = logging.getLogger(__name__)

class PreferenceParser:
    """
    Parses natural language queries to detect intent, extract preferences,
    seed movies, and constraints for the recommendation engine.
    """

    def __init__(self, entity_linker: EntityLinker, relation_mapper: RelationMapper):
        self.entity_linker = entity_linker
        self.relation_mapper = relation_mapper
        
        # Regex for years (e.g., 1990, 1990s, after 2000)
        self.year_regex = re.compile(r'(\b(after|before|since|from|in)\s+)?(\d{4})s?\b', re.IGNORECASE)
        
        # Regex for explicit languages (French, Japanese, etc.)
        # We map the name to the Wikidata Q-ID directly here for robustness
        self.lang_map = {
            "french": "http://www.wikidata.org/entity/Q150",
            "german": "http://www.wikidata.org/entity/Q188",
            "spanish": "http://www.wikidata.org/entity/Q1321",
            "english": "http://www.wikidata.org/entity/Q1860",
            "italian": "http://www.wikidata.org/entity/Q652",
            "japanese": "http://www.wikidata.org/entity/Q5287",
            "korean": "http://www.wikidata.org/entity/Q9176",
            "chinese": "http://www.wikidata.org/entity/Q7850",
        }

        logger.info("PreferenceParser initialized.")

    def parse(self, query: str, session: Session) -> Dict[str, Any]:
        logger.debug(f"Parsing query: {query}")
        query_lower = query.lower()

        intent = self.detect_intent(query_lower)
        seed_movies = self.extract_seed_movies(query)
        preferences, constraints, negations = self.extract_preferences_and_constraints(query)
        is_follow_up = any(keyword in query_lower for keyword in FOLLOW_UP_KEYWORDS)

        if seed_movies or preferences or constraints or negations:
            intent = 'recommendation'
            
        return {
            'intent': intent,
            'seed_movies': seed_movies,
            'preferences': preferences,
            'constraints': constraints, # This passes the language constraint
            'negations': negations,
            'is_follow_up': is_follow_up
        }

    def detect_intent(self, query_lower: str) -> str:
        if any(keyword in query_lower for keyword in RECOMMENDATION_KEYWORDS):
            return 'recommendation'
        if '?' not in query_lower and len(query_lower.split()) < 10:
             return 'recommendation'
        return 'qa'

    def extract_seed_movies(self, query: str) -> List[str]:
        # Use Linker's smart link method which handles quotes and lists
        candidates = self.entity_linker.link(query)
        # Candidates is list of (label, uri, score)
        # We return just the URIs
        return [uri for label, uri, score in candidates]

    def extract_preferences_and_constraints(self, query: str) -> Tuple[Dict, Dict, Dict]:
        preferences = {}
        constraints = {}
        negations = {}
        
        query_lower = query.lower()
        
        # 1. Extract Explicit Languages
        for lang_name, lang_uri in self.lang_map.items():
            # Check for "in Japanese", "Japanese movie", etc.
            if lang_name in query_lower:
                logger.info(f"Detected language constraint: {lang_name} -> {lang_uri}")
                if "language" not in constraints:
                    constraints["language"] = []
                constraints["language"].append(lang_uri)

        # 2. Extract Genre/Actor/Director keywords
        # (Simple heuristic)
        for pref_type, keywords in PREFERENCE_KEYWORDS.items():
            for keyword in keywords:
                if keyword in query_lower:
                    # Try to find entity after keyword? 
                    # Ideally this is handled by entity linker seeds, but sometimes
                    # we want to catch "Action movies" where "Action" is a genre constraint, not a seed movie.
                    pass

        return preferences, constraints, negations