import logging
from pathlib import Path
from rdflib import Graph
from rdflib.query import Result
from typing import List, Dict, Any

logger = logging.getLogger(__name__)

class GraphExecutor:
    """
    Executes SPARQL queries against the loaded RDF graph.
    """
    
    def __init__(self, graph_db_path: Path):
        """
        Initializes the executor and loads the graph from the specified path.
        """
        if not graph_db_path or not graph_db_path.exists():
            logger.error(f"Graph DB file not found at: {graph_db_path}")
            raise FileNotFoundError(f"Graph DB file not found at: {graph_db_path}")
        
        logger.info(f"Loading graph from: {graph_db_path}")
        self.graph = Graph()
        try:
            # Parse the graph
            self.graph.parse(str(graph_db_path))
            logger.info(f"Graph loaded successfully. Contains {len(self.graph)} triples.")
        except Exception as e:
            logger.error(f"Failed to parse graph: {e}", exc_info=True)
            raise

    def execute_query(self, query_string: str) -> List[Dict[str, Any]]:
        """
        Executes a SPARQL query and returns a list of dictionaries.
        """
        logger.debug(f"Executing SPARQL query:\n{query_string}")
        try:
            results = self.graph.query(query_string)
            # Convert results to a standard list of dicts
            output = []
            for row in results:
                output.append(row.asdict())
            return output
        except Exception as e:
            logger.error(f"Error executing SPARQL query: {e}\nQuery:\n{query_string}", exc_info=True)
            return []