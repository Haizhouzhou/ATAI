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
    def __init__(self, entity_linker: EntityLinker, relation_mapper: RelationMapper):
        self.entity_linker = entity_linker
        self.relation_mapper = relation_mapper
        self.year_regex = re.compile(r'(\b(after|before|since|from|in)\s+)?(\d{4})s?\b', re.IGNORECASE)
        self.lang_regex = re.compile(r'\b(in|spoken in)\s+(' + SUPPORTED_LANGUAGES_REGEX + r')\b', re.IGNORECASE)
        self.rating_regex = re.compile(r'\b(rating|score|rated)\s+(above|higher than|more than|over|below|less than|under|<|>)\s+(\d+(\.\d+)?)', re.IGNORECASE)
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
            'constraints': constraints,
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
        candidates = []
        quoted = re.findall(r'["\'](.*?)["\']', query)
        candidates.extend(quoted)
        
        match_like = re.search(r'(like|similar to|resemble)\s+(.*)', query, re.IGNORECASE)
        if match_like:
            raw_list = match_like.group(2)
            parts = raw_list.split(',')
            for part in parts:
                clean = part.strip()
                if " and " in clean:
                    sub = clean.split(" and ")
                    candidates.extend([s.strip() for s in sub])
                else:
                    candidates.append(clean)
        candidates.append(query)
        clean_candidates = list(set(c for c in candidates if c and len(c) > 2))
        
        seed_movie_ids = []
        linked_entities = self.entity_linker.link(clean_candidates)
        for entity in linked_entities:
            if entity.score > 88: 
                seed_movie_ids.append(entity.iri)
        return list(set(seed_movie_ids))

    def extract_preferences_and_constraints(self, query: str) -> Tuple[Dict, Dict, Dict]:
        preferences = {}
        constraints = {}
        negations = {}
        
        query_lower = query.lower()
        is_negated = any(neg in query_lower for neg in NEGATION_KEYWORDS)
        target_dict = negations if is_negated else preferences
        constraint_target = negations if is_negated else constraints

        for pref_type, keywords in PREFERENCE_KEYWORDS.items():
            for keyword in keywords:
                if keyword in query_lower:
                    match = re.search(f'{keyword}(\s+(?:by|is|as|a|an|the))?(\s+[\w\s]+)', query_lower)
                    if match and match.group(2):
                        value = match.group(2).strip().split(" from ")[0].split(" in ")[0].split(" with ")[0]
                        value = re.sub(r'[?.]+$', '', value)
                        
                        linked_val = None
                        if pref_type in ['actor', 'director']:
                            res = self.entity_linker.link_entities(value)
                            if res: linked_val = res[0][1]
                        elif pref_type == 'genre':
                            res = self.entity_linker.link_entities(f"{value} film")
                            if not res:
                                res = self.entity_linker.link_entities(value)
                            if res: linked_val = res[0][1]
                        
                        if linked_val:
                            target_dict[pref_type] = linked_val
                    break

        year_matches = self.year_regex.finditer(query)
        for match in year_matches:
            prefix = (match.group(2) or '').lower()
            year_str = match.group(3)
            year = int(year_str)
            if 's' in match.group(0):
                constraint_target['year_range'] = (year, year + 9)
                continue
            op = '='
            if prefix in ['after', 'since']: op = '>'
            elif prefix == 'before': op = '<'
            constraint_target['year'] = (op, year)
            
        lang_match = self.lang_regex.search(query)
        if lang_match:
            lang_name = lang_match.group(2).capitalize()
            lang_entities = self.entity_linker.link_entities(f"{lang_name} language")
            if lang_entities:
                constraint_target['language'] = lang_entities[0][1]

        rating_match = self.rating_regex.search(query)
        if rating_match:
            op_str = rating_match.group(2).lower()
            val_str = rating_match.group(3)
            try:
                val = float(val_str)
                op = '>'
                if op_str in ['below', 'less than', 'under', '<']:
                    op = '<'
                constraint_target['rating'] = (op, val)
            except ValueError:
                pass

        return preferences, constraints, negations