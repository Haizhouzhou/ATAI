import logging
from typing import List, Dict, Tuple, Any
from agent.constants import PREFIXES, PREDICATE_MAP

logger = logging.getLogger(__name__)

class Composer:
    """
    Composes SPARQL queries based on parsed query plans.
    """

    def __init__(self):
        self.prefixes = PREFIXES
        logger.info("Composer initialized.")

    def build_query(self, head: str, relation: str, tail: str, limit: int = 10, type_constraint: str = None) -> str:
        """
        Builds a SPARQL query for a simple triple pattern (head, relation, tail).
        (Eval 2 Fix - Rule 1.3)
        """
        
        # Determine which part is the variable
        if head.startswith('?'):
            select_var = head
            triple = f"{head} {relation} {tail} ."
        elif relation.startswith('?'):
            select_var = relation
            triple = f"{head} {relation} {tail} ."
        elif tail.startswith('?'):
            select_var = tail
            triple = f"{head} {relation} {tail} ."
        else:
            logger.error(f"No variable found in triple: {head}, {relation}, {tail}")
            raise ValueError("No variable found in triple.")

        # Try to get a label for the selected variable
        select_clause = f"SELECT DISTINCT {select_var} ?{select_var[1:]}Label"
        label_service = f"""
            OPTIONAL {{
                {select_var} rdfs:label ?{select_var[1:]}Label .
                FILTER(LANG(?{select_var[1:]}Label) = "en")
            }}
        """
        
        type_constraint_str = type_constraint if type_constraint else ""

        query = f"""
            {self.prefixes}
            {select_clause}
            WHERE {{
                {triple}
                {type_constraint_str}
                {label_service}
            }}
            LIMIT {limit}
        """
        logger.debug(f"Composed factual query:\n{query}")
        return query

    # --- Start Helper for Recommendation Filters ---
    
    def _build_filter_block(self,
                             constraints: Dict[str, Tuple[str, Any]] = None,
                             negations: Dict[str, Any] = None,
                             movie_var: str = "?movie"
                             ) -> Tuple[str, str]:
        """
        Helper to build SPARQL FILTER, MINUS, and OPTIONAL blocks for rec queries.
        (Eval 3 Fix - Rule 2.1)
        """
        triples = []
        filters = []
        
        # --- Handle Negations (MINUS) ---
        if negations:
            for pref_type, value_id in negations.items():
                if pref_type in PREDICATE_MAP:
                    pid = PREDICATE_MAP[pref_type]
                    triples.append(f"MINUS {{ {movie_var} {pid} wd:{value_id} . }}")

        # --- Handle Constraints (FILTER) ---
        if constraints:
            if 'year' in constraints:
                op, year = constraints['year']
                triples.append(f"OPTIONAL {{ {movie_var} wdt:P577 ?publicationDate . }}")
                filters.append(f"FILTER(BOUND(?publicationDate) && (YEAR(?publicationDate) {op} {year}))")
            if 'year_range' in constraints:
                start, end = constraints['year_range']
                triples.append(f"OPTIONAL {{ {movie_var} wdt:P577 ?publicationDate . }}")
                filters.append(f"FILTER(BOUND(?publicationDate) && (YEAR(?publicationDate) >= {start} && YEAR(?publicationDate) <= {end}))")
            if 'language' in constraints:
                 # Assumes language is a QID
                 triples.append(f"{movie_var} wdt:P407 wd:{constraints['language']} .")

        triples_block = "\n".join(triples)
        filters_block = "\n".join(filters)
        
        return triples_block, filters_block

    # --- EVAL 3: Recommendation Query Templates ---

    def get_recommendation_by_shared_property_query(
        self,
        seed_uris: List[str],
        property_pid: str,
        constraints: Dict[str, Tuple[str, Any]] = None,
        negations: Dict[str, Any] = None,
        limit: int = 20
    ) -> str:
        """
        Finds movies that share a property with seeds, applying hard filters.
        (Eval 3 Fix - Rule 2.1, 2.5)
        """
        if not seed_uris:
            return ""
            
        seed_values = " ".join(seed_uris)
        
        # Build filter blocks
        filter_triples, filter_clauses = self._build_filter_block(constraints, negations, "?movie")

        query = f"""
            {self.prefixes}
            SELECT ?movie ?movieLabel ?propLabel ?rating (COUNT(DISTINCT ?seed) AS ?sharedCount)
            WHERE {{
                VALUES ?seed {{ {seed_values} }}
                
                ?seed {property_pid} ?property .
                ?movie {property_pid} ?property .
                ?movie wdt:P31 wd:Q11424 .
                
                FILTER(?movie NOT IN ( {seed_values} ))
                
                # --- Apply Hard Filters (Rule 2.1) ---
                {filter_triples}
                
                ?movie rdfs:label ?movieLabel .
                FILTER(LANG(?movieLabel) = "en")
                OPTIONAL {{
                    ?property rdfs:label ?propLabel .
                    FILTER(LANG(?propLabel) = "en")
                }}
                
                # --- Get Rating for Scoring (Rule 2.5) ---
                OPTIONAL {{ ?movie ddis:rating ?rating . }}
                
                {filter_clauses}
            }}
            GROUP BY ?movie ?movieLabel ?propLabel ?rating
            ORDER BY DESC(?sharedCount)
            LIMIT {limit}
        """
        logger.debug(f"Composed recommendation query (shared property):\n{query}")
        return query

    def get_recommendation_by_property_query(
        self,
        preferences: Dict[str, str],
        constraints: Dict[str, Tuple[str, Any]] = None,
        negations: Dict[str, Any] = None,
        limit: int = 20
    ) -> str:
        """
        Finds movies matching specific properties, applying filters.
        (Eval 3 Fix - Rule 2.1, 2.5)
        """
        triples = ["?movie wdt:P31 wd:Q11424 ."] # Start with 'is a movie'
        
        # Add preference triples
        for pid, value_uri in preferences.items():
            triples.append(f"?movie {pid} {value_uri} .")
            
        # Build filter blocks
        filter_triples, filter_clauses = self._build_filter_block(constraints, negations, "?movie")
        
        triples.append(filter_triples)
        triples_block = "\n".join(triples)

        query = f"""
            {self.prefixes}
            SELECT ?movie ?movieLabel ?rating
            WHERE {{
                {triples_block}
                
                ?movie rdfs:label ?movieLabel .
                FILTER(LANG(?movieLabel) = "en")
                
                # --- Get Rating for Scoring (Rule 2.5) ---
                OPTIONAL {{ ?movie ddis:rating ?rating . }}
                
                # --- Apply Hard Filters (Rule 2.1) ---
                {filter_clauses}
            }}
            LIMIT {limit}
        """
        logger.debug(f"Composed recommendation query (by property):\n{query}")
        return query