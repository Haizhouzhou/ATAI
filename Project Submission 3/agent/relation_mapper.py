import re
import logging
from agent.constants import PREDICATE_MAP

logger = logging.getLogger(__name__)

class RelationMapper:
    """
    Maps natural language relation phrases to knowledge graph predicates.
    """
    def __init__(self):
        # Build a single regex pattern from all keys in PREDICATE_MAP
        # Sort keys by length, descending, to match longest phrases first
        # This fixes the "nominated for" vs "award" ambiguity
        patterns = sorted(PREDICATE_MAP.keys(), key=len, reverse=True)
        
        self.relation_pattern = re.compile(r'\b(' + '|'.join(re.escape(p) for p in patterns) + r')\b', re.IGNORECASE)
        self.predicate_map = {k.lower(): v for k, v in PREDICATE_MAP.items()}
        logger.info("RelationMapper initialized with regex pattern (longest match first).")

    def map_relation(self, query: str) -> str | None:
        """
        Finds the first matching relation in the query and returns its predicate ID.
        """
        match = self.relation_pattern.search(query)
        if match:
            matched_phrase = match.group(0).lower()
            predicate = self.predicate_map.get(matched_phrase)
            if predicate:
                logger.debug(f"Mapped phrase '{matched_phrase}' to predicate {predicate}")
                return predicate
        
        logger.debug(f"No relation predicate found in query: '{query}'")
        return None