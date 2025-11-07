import re
import logging
from typing import Dict, Any, List
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
    
    (Eval 3 Fix - Rule 2.3)
    """

    def __init__(self, entity_linker: EntityLinker, relation_mapper: RelationMapper):
        self.entity_linker = entity_linker
        self.relation_mapper = relation_mapper
        # Simple regex for years and decades (e.g., 1990, 1990s, after 2000)
        self.year_regex = re.compile(r'(\b(after|before|since|from|in)\s+)?(\d{4})s?\b', re.IGNORECASE)
        # Specific language regex to avoid "in 1990s" conflict
        self.lang_regex = re.compile(r'\b(in|spoken in)\s+(' + SUPPORTED_LANGUAGES_REGEX + r')\b', re.IGNORECASE)
        logger.info("PreferenceParser initialized.")

    def parse(self, query: str, session: Session) -> Dict[str, Any]:
        """
        Main parsing method.
        """
        logger.debug(f"Parsing query: {query}")
        query_lower = query.lower()

        # Intent detection is now simpler, as app/main.py does the QA routing.
        # We just check if this looks like a recommendation query.
        intent = self.detect_intent(query_lower)
        
        seed_movies = self.extract_seed_movies(query)
        preferences, constraints, negations = self.extract_preferences_and_constraints(query)
        is_follow_up = any(keyword in query_lower for keyword in FOLLOW_UP_KEYWORDS)

        # If we found seeds or preferences, it's definitely a rec intent
        if seed_movies or preferences or constraints or negations:
            intent = 'recommendation'
            
        return {
            'intent': intent,
            'seed_movies': seed_movies,
            'preferences': preferences,
            'constraints': constraints,
            'negations': negations,
            'is_follow_up': is_follow_up
        }

    def detect_intent(self, query_lower: str) -> str:
        """
        Detects if the user's intent is likely recommendation.
        The QA intent is already handled by app/main.py checking for relations.
        """
        if any(keyword in query_lower for keyword in RECOMMENDATION_KEYWORDS):
            return 'recommendation'
        
        # If user just types a movie name (and it's not a question)
        if '?' not in query_lower and len(query_lower.split()) < 10:
             return 'recommendation'

        return 'qa' # Default fallback if no rec keywords

    def extract_seed_movies(self, query: str) -> List[str]:
        """
        Extracts movie titles from the query and links them to entity IDs.
        """
        quoted_titles = re.findall(r'["\'](.*?)["\']', query)
        
        # Heuristic: Link entities from the whole query
        linked_entities = self.entity_linker.link_entities(query)
        
        # Add quoted titles
        for title in quoted_titles:
            linked_entities.extend(self.entity_linker.link_entities(title))
        
        seed_movie_ids = []
        for entity in linked_entities:
            # We trust the linker, which prioritizes movies.
            seed_movie_ids.append(entity[1]) # entity[1] is the ID
        
        return list(set(seed_movie_ids))

    def extract_preferences_and_constraints(self, query: str) -> (Dict, Dict, Dict):
        """
        Extracts preferences (e.g., genre, actor) and constraints (e.g., year).
        """
        preferences = {}
        constraints = {}
        negations = {}
        
        query_lower = query.lower()
        is_negated = any(neg in query_lower for neg in NEGATION_KEYWORDS)
        
        # --- 1. Map keyword-based preferences (genre, actor, director) ---
        for pref_type, keywords in PREFERENCE_KEYWORDS.items():
            target_dict = negations if is_negated and pref_type in ['genre', 'actor', 'director'] else preferences
            
            for keyword in keywords:
                if keyword in query_lower:
                    match = re.search(f'{keyword}(\s+(?:by|is|as|a|an|the))?(\s+[\w\s]+)', query_lower)
                    if match and match.group(2):
                        value = match.group(2).strip().split(" from ")[0].split(" in ")[0].split(" with ")[0]
                        
                        if pref_type in ['actor', 'director']:
                            linked_val = self.entity_linker.link_entities(value)
                            if linked_val:
                                target_dict[pref_type] = linked_val[0][1] # Get ID
                        elif pref_type == 'genre':
                            linked_val = self.entity_linker.link_entities(f"{value} film")
                            if not linked_val:
                                linked_val = self.entity_linker.link_entities(value)
                            if linked_val:
                                target_dict[pref_type] = linked_val[0][1] # Get ID
                    break

        # --- 2. Extract year constraints ---
        year_matches = self.year_regex.finditer(query)
        for match in year_matches:
            prefix = (match.group(2) or '').lower()
            year_str = match.group(3)
            
            if 's' in match.group(0): # '1990s'
                year = int(year_str)
                (negations if is_negated else constraints)['year_range'] = (year, year + 9)
                continue
            
            year = int(year_str)
            op = '='
            if prefix == 'after' or prefix == 'since':
                op = '>'
            elif prefix == 'before':
                op = '<'
            elif prefix == 'in' or prefix == 'from':
                op = '='
                
            (negations if is_negated else constraints)['year'] = (op, year)
            
        # --- 3. Extract language constraint (Rule 2.3) ---
        lang_match = self.lang_regex.search(query)
        if lang_match:
            lang_name = lang_match.group(2).capitalize()
            lang_entities = self.entity_linker.link_entities(f"{lang_name} language")
            if lang_entities:
                lang_id = lang_entities[0][1]
                (negations if is_negated else constraints)['language'] = lang_id

        return preferences, constraints, negations