import logging
from rdflib import Graph

class GraphExecutor:
    def __init__(self, path):
        self.graph = Graph()
        fmt = "nt" if str(path).endswith(".nt") else "turtle"
        logging.info(f"Loading Graph {path}...")
        self.graph.parse(path, format=fmt)

    def execute_query(self, q):
        try:
            return [row.asdict() for row in self.graph.query(q)]
        except: return []