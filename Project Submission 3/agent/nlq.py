import logging
from agent.graph_executor import GraphExecutor
from agent.embedding_executor import EmbeddingExecutor
from agent.entity_linker import EntityLinker
from agent.relation_mapper import RelationMapper
from agent.composer import Composer
from agent.constants import PREDICATE_MAP

logger = logging.getLogger(__name__)

class NLQ:
    """
    Handles Natural Language Questions (Factoid QA) for Eval 2.
    """

    def __init__(self,
                 graph_executor: GraphExecutor,
                 embedding_executor: EmbeddingExecutor,
                 entity_linker: EntityLinker,
                 relation_mapper: RelationMapper,
                 composer: Composer):
        
        self.graph_executor = graph_executor
        self.embedding_executor = embedding_executor
        self.entity_linker = entity_linker
        self.relation_mapper = relation_mapper
        self.composer = composer
        logger.info("NLQ (QA Processor) initialized.")

    def process_query(self, query: str, pre_mapped_relation: str = None) -> str:
        """
        Processes a factual query.
        """
        logger.debug(f"NLQ processing query: {query}")

        # 1. Get Relation (passed from main.py or mapped here)
        relation = pre_mapped_relation
        if not relation:
            relation = self.relation_mapper.map_relation(query)
            # Handle ambiguity if not already handled in main
            if relation == PREDICATE_MAP['award'] and 'nominat' in query.lower():
                relation = PREDICATE_MAP['nominated for']
        
        # 2. Factual Approach (if relation is found)
        if relation:
            try:
                # 3. Get Entity
                entities = self.entity_linker.link_entities(query)
                if not entities:
                    logger.warning(f"No entities found in QA query: {query}")
                    # Entity linking fallback (Rule 1.3) is complex to add
                    # without modifying EntityLinker, so we rely on the primary linker.
                    return "I found a topic but no specific item to look up. Could you be more precise?"

                # Assume the first linked entity is the subject
                entity_label, entity_id, _ = entities[0]
                head = f"wd:{entity_id}"
                tail = "?answer"
                var_name = "answer"
                var_name_friendly = relation.split(':')[-1] # e.g., "P57" -> "P57"
                
                # Simple logic to determine if we are looking for head or tail
                # This could be improved, but works for most WH-questions
                if query.lower().startswith("who") or \
                   query.lower().startswith("what") or \
                   query.lower().startswith("when"):
                    head = f"wd:{entity_id}"
                    tail = f"?{var_name}"
                else: # e.g., "Which movie did..."
                    head = f"?{var_name}"
                    tail = f"wd:{entity_id}"

                # 4. Add Type Constraint (Rule 1.3)
                type_constraint = None
                movie_relations = [
                    PREDICATE_MAP['director'], PREDICATE_MAP['screenwriter'],
                    PREDICATE_MAP['actor'], PREDICATE_MAP['publication date'],
                    PREDICATE_MAP['country'], PREDICATE_MAP['genre'],
                    PREDICATE_MAP['producer'], PREDICATE_MAP['award'],
                    PREDICATE_MAP['nominated for'], PREDICATE_MAP['composer'],
                    PREDICATE_MAP['rating']
                ]
                
                # If the known entity is the subject and we're asking about a movie property:
                if head.startswith("wd:") and relation in movie_relations:
                    type_constraint = f"{head} wdt:P31 wd:Q11424 ."
                # If the answer is the subject and the known entity is the object:
                elif tail.startswith("wd:") and relation in movie_relations:
                     type_constraint = f"{head} wdt:P31 wd:Q11424 ."

                # 5. Build and Execute Query
                query_str = self.composer.build_query(head, relation, tail, type_constraint=type_constraint)
                results = self.graph_executor.execute_query(query_str)

                # 6. Format Response
                if not results:
                    return f"I couldn't find any information about the {var_name_friendly} for {entity_label}."
                
                labels = [res[f"{var_name}Label"]['value'] for res in results if f"{var_name}Label" in res]
                if not labels:
                    # Fallback for values without labels (like dates)
                    labels = [res[var_name]['value'] for res in results if var_name in res]
                
                if not labels:
                    return f"I found some entries for {entity_label} but could not get their names."

                # De-duplicate results (Rule 1.3)
                unique_labels = sorted(list(set(labels)))
                return f"The {var_name_friendly} of {entity_label} is: {', and '.join(unique_labels)}."

            except Exception as e:
                logger.error(f"Error in NLQ factual approach: {e}", exc_info=True)
                return "I had trouble processing that question."

        # 7. Embedding Approach (Fallback if no relation is found)
        logger.info("No relation found, trying embedding approach.")
        try:
            # (Your existing Eval 2 embedding logic would go here)
            # ...
            return "I'm not sure how to answer that factually, and the embedding approach is not connected in this path."
        
        except Exception as e:
            logger.error(f"Error in NLQ embedding approach: {e}", exc_info=True)
            return "I'm sorry, I couldn't answer that question."