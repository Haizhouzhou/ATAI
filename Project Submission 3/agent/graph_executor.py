import logging
from typing import List, Dict, Any
from rdflib import Graph

logger = logging.getLogger(__name__)

class GraphExecutor:
    """
    Executes SPARQL queries against the RDF graph.
    """
    def __init__(self, graph_path: str):
        self.graph_path = graph_path
        self.graph = self._load_graph()

    def _load_graph(self) -> Graph:
        logger.info(f"Loading graph from: {self.graph_path}")
        g = Graph()
        try:
            g.parse(self.graph_path)
            logger.info(f"Graph loaded successfully. Contains {len(g)} triples.")
        except Exception as e:
            logger.error(f"Failed to load graph: {e}", exc_info=True)
            raise
        return g

    def execute_query(self, query_string: str) -> List[Dict[str, Any]] | bool:
        """
        Executes a SPARQL query and returns the results.
        Handles both SELECT and ASK queries.
        """
        try:
            logger.debug(f"Executing query:\n{query_string}")
            results = self.graph.query(query_string)

            # --- THIS IS THE FIX ---
            # Check the type of query result
            
            if "ask" in query_string.lower():
                # For ASK queries, result is a bool
                return bool(results)
                
            # For SELECT queries, result is an iterable
            output = []
            for row in results:
                # Convert the row to a dictionary
                output.append(row.asdict())
            
            logger.debug(f"Query returned {len(output)} results.")
            return output
            
        except Exception as e:
            logger.error(f"Error executing SPARQL query: {e}\nQuery:\n{query_string}", exc_info=True)
            # Return empty list for SELECT on failure, False for ASK
            if "ask" in query_string.lower():
                return False
            return []
        # --- END FIX ---