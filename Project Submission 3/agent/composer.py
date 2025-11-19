import logging
from typing import List, Dict, Tuple, Any
from agent.constants import PREFIXES, PREDICATE_MAP

logger = logging.getLogger(__name__)

class Composer:
    def __init__(self):
        self.prefixes = PREFIXES
        logger.info("Composer initialized.")

    def build_query(self, head: str, relation: str, tail: str, limit: int = 10, type_constraint: str = None) -> str:
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
            raise ValueError("No variable found in triple.")

        select_clause = f"SELECT DISTINCT {select_var} ?{select_var[1:]}Label"
        
        # Relaxed label fetch for QA
        label_service = f"""
            OPTIONAL {{
                {select_var} rdfs:label ?{select_var[1:]}Label .
                FILTER(LANGMATCHES(LANG(?{select_var[1:]}Label), "en") || LANG(?{select_var[1:]}Label) = "")
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
        return query

    def _build_filter_block(self,
                            constraints: Dict[str, Tuple[str, Any]] = None,
                            negations: Dict[str, Any] = None,
                            movie_var: str = "?movie"
                            ) -> Tuple[str, str]:
        triples = []
        filters = []
        
        if negations:
            for pref_type, value_id in negations.items():
                if pref_type in PREDICATE_MAP:
                    pid = PREDICATE_MAP[pref_type]
                    triples.append(f"MINUS {{ {movie_var} <{pid}> <{value_id}> . }}")

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
                triples.append(f"{movie_var} wdt:P407 <{constraints['language']}> .")
                
            if 'rating' in constraints:
                op, val = constraints['rating']
                triples.append(f"OPTIONAL {{ {movie_var} ddis:rating ?ratingVal . }}")
                filters.append(f"FILTER(BOUND(?ratingVal) && (xsd:float(?ratingVal) {op} {val}))")

        triples_block = "\n".join(triples)
        filters_block = "\n".join(filters)
        return triples_block, filters_block

    def get_recommendation_by_shared_property_query(
        self,
        seed_uris: List[str],
        property_pid: str,
        constraints: Dict[str, Tuple[str, Any]] = None,
        negations: Dict[str, Any] = None,
        limit: int = 20
    ) -> str:
        if not seed_uris: return ""
        seed_values = " ".join(seed_uris)
        filter_values = ", ".join(seed_uris) 
        filter_triples, filter_clauses = self._build_filter_block(constraints, negations, "?movie")

        query = f"""
            {self.prefixes}
            SELECT ?movie ?movieLabel ?propLabel ?rating (COUNT(DISTINCT ?seed) AS ?sharedCount)
            WHERE {{
                VALUES ?seed {{ {seed_values} }}
                ?seed <{property_pid}> ?property .
                ?movie <{property_pid}> ?property .
                ?movie wdt:P31 wd:Q11424 .
                
                FILTER(?movie NOT IN ( {filter_values} ))
                
                {filter_triples}
                
                OPTIONAL {{
                    ?movie rdfs:label ?movieLabel .
                    FILTER(LANGMATCHES(LANG(?movieLabel), "en") || LANG(?movieLabel) = "")
                }}
                OPTIONAL {{
                    ?property rdfs:label ?propLabel .
                    FILTER(LANGMATCHES(LANG(?propLabel), "en") || LANG(?propLabel) = "")
                }}
                OPTIONAL {{ ?movie ddis:rating ?rating . }}
                
                {filter_clauses}
            }}
            GROUP BY ?movie ?movieLabel ?propLabel ?rating
            ORDER BY DESC(?sharedCount) DESC(xsd:float(?rating))
            LIMIT {limit}
        """
        return query

    def get_recommendation_by_property_query(
        self,
        preferences: Dict[str, str],
        constraints: Dict[str, Tuple[str, Any]] = None,
        negations: Dict[str, Any] = None,
        limit: int = 20
    ) -> str:
        triples = ["?movie wdt:P31 wd:Q11424 ."]
        for pid, value_uri in preferences.items():
            triples.append(f"?movie <{pid}> {value_uri} .")
        
        filter_triples, filter_clauses = self._build_filter_block(constraints, negations, "?movie")
        triples.append(filter_triples)
        triples_block = "\n".join(triples)

        query = f"""
            {self.prefixes}
            SELECT ?movie ?movieLabel ?rating
            WHERE {{
                {triples_block}
                
                OPTIONAL {{
                    ?movie rdfs:label ?movieLabel .
                    FILTER(LANGMATCHES(LANG(?movieLabel), "en") || LANG(?movieLabel) = "")
                }}
                OPTIONAL {{ ?movie ddis:rating ?rating . }}
                {filter_clauses}
            }}
            ORDER BY DESC(xsd:float(?rating))
            LIMIT {limit}
        """
        return query

    def get_image_query(self, movie_uris: List[str]) -> str:
        """
        Fetches images. Removed type constraint to be safer.
        """
        if not movie_uris: return ""
        
        uri_list = []
        for uri in movie_uris:
            clean = uri.strip()
            if not clean.startswith("<"):
                clean = f"<{clean}>"
            uri_list.append(clean)
            
        filter_str = ", ".join(uri_list)
        
        # REMOVED: ?movie wdt:P31 wd:Q11424 . (Make it optional or just trust the ID)
        query = f"""
            {self.prefixes}
            SELECT ?movie ?imageUrl
            WHERE {{
                ?movie wdt:P18 ?imageUrl .
                FILTER (?movie IN ({filter_str}))
            }}
        """
        return query

    def get_labels_query(self, movie_uris: List[str]) -> str:
        """
        Fetches labels with relaxed language filter.
        """
        if not movie_uris: return ""
        
        uri_list = []
        for uri in movie_uris:
            clean = uri.strip()
            if not clean.startswith("<"):
                clean = f"<{clean}>"
            uri_list.append(clean)
        
        filter_str = ", ".join(uri_list)
        
        query = f"""
            {self.prefixes}
            SELECT ?movie ?movieLabel
            WHERE {{
                ?movie rdfs:label ?movieLabel .
                FILTER(LANGMATCHES(LANG(?movieLabel), "en") || LANG(?movieLabel) = "")
                FILTER (?movie IN ({filter_str}))
            }}
        """
        return query