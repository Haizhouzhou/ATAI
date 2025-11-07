import logging
import sys
from pathlib import Path

# --- FIX: Add project root to path ---
project_root = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(project_root))
# --- END FIX ---

from agent.graph_executor import GraphExecutor
from agent.config import Config

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

def explore_entity_properties(executor, entity_iri, entity_label):
    logger.info(f"\n--- Exploring properties for: {entity_label} ({entity_iri}) ---")
    
    # This query selects ALL properties (predicates) and their labels
    # for a given entity (subject).
    query = f"""
        PREFIX rdfs: <http://www.w3.org/2000/01/rdf-schema#>
        
        SELECT ?property ?propertyLabel (SAMPLE(?value) AS ?exampleValue) (SAMPLE(?valueLabel) AS ?exampleValueLabel)
        WHERE {{
            <{entity_iri}> ?property ?value .
            
            # Try to get the label for the property itself
            OPTIONAL {{
                ?property rdfs:label ?propertyLabel .
                FILTER(LANG(?propertyLabel) = "en")
            }}
            
            # Try to get the label for the value
            OPTIONAL {{
                ?value rdfs:label ?valueLabel .
                FILTER(LANG(?valueLabel) = "en")
            }}
        }}
        GROUP BY ?property ?propertyLabel
        ORDER BY ?property
    """
    
    try:
        properties = executor.execute_query(query)
        
        if not properties:
            logger.warning(f"No properties found for {entity_label}.")
            return

        logger.info(f"Found {len(properties)} properties for {entity_label}:")
        
        # --- THIS IS THE FIX ---
        # The 'prop' dictionary contains rdflib objects, not other dicts.
        # We must convert them to strings directly.
        for prop in properties:
            # Get the rdflib object (URIRef or Literal) or None
            prop_ref = prop.get('property')
            pid_label_lit = prop.get('propertyLabel')
            val_label_lit = prop.get('exampleValueLabel')

            # Convert to string, or use a default
            pid = str(prop_ref) if prop_ref else 'N/A'
            pid_label = str(pid_label_lit) if pid_label_lit else 'N/A'
            val_label = str(val_label_lit) if val_label_lit else 'N/A (Literal?)'
            
            logger.info(f"  -> Property: {pid}")
            logger.info(f"     Label: {pid_label}")
            logger.info(f"     Example Value: {val_label}")
        # --- END FIX ---

    except Exception as e:
        logger.error(f"Error exploring {entity_label}: {e}", exc_info=True)

def main():
    logger.info("Initializing Config and GraphExecutor for data check...")
    try:
        config = Config()
        executor = GraphExecutor(config.graph_db)
        
        # Explore Pocahontas (Q152715)
        explore_entity_properties(executor, "http://www.wikidata.org/entity/Q152715", "Pocahontas")
        
        # Explore A Nightmare on Elm Street (Q300508)
        explore_entity_properties(executor, "http://www.wikidata.org/entity/Q300508", "A Nightmare on Elm Street")
        
    except Exception as e:
        logger.error(f"Failed to initialize: {e}")

if __name__ == "__main__":
    main()