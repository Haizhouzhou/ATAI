import logging
import numpy as np
from typing import List, Dict, Any, Set
from agent.graph_executor import GraphExecutor
from agent.embedding_executor import EmbeddingExecutor
from agent.composer import Composer
from agent.entity_linker import EntityLinker
from agent.session_manager import Session
from agent.constants import PREDICATE_MAP, PREFIXES 

logger = logging.getLogger(__name__)

class RecommendationEngine:
    """
    Generates movie recommendations based on session state.
    """
    def __init__(self,
                 graph_executor: GraphExecutor,
                 embedding_executor: EmbeddingExecutor,
                 composer: Composer,
                 entity_linker: EntityLinker):
        
        self.graph_executor = graph_executor
        self.embedding_executor = embedding_executor
        self.composer = composer
        self.entity_linker = entity_linker
        logger.info("RecommendationEngine initialized.")

    def get_recommendations(self, session: Session, top_k: int = 5) -> List[Dict[str, Any]]:
        """
        Main method to get recommendations.
        Orchestrates candidate generation, filtering, and ranking.
        """
        logger.info(f"Getting recommendations for session {session.user_id}")
        
        all_candidates: Dict[str, Dict[str, Any]] = {}
        exclude_list = session.get_exclude_list()
        
        # --- 1. Candidate Generation (with filters) ---
        constraints = session.constraints
        negations = session.negations
        
        # A. Graph-based candidates from SEED movies
        if session.seed_movies:
            logger.debug("Generating candidates from seed movies (Graph)")
            graph_candidates = self.get_graph_candidates_from_seeds(
                session.seed_movies, exclude_list, constraints, negations
            )
            self.merge_candidates(all_candidates, graph_candidates, 'graph_seed')

        # B. Embedding-based candidates from SEED movies
        if session.seed_movies:
            logger.debug("Generating candidates from seed movies (Embedding)")
            embedding_candidates = self.get_embedding_candidates_from_seeds(session.seed_movies, exclude_list)
            self.merge_candidates(all_candidates, embedding_candidates, 'embed_seed')
            
        # C. Graph-based candidates from explicit PREFERENCES
        if session.preferences:
            logger.debug("Generating candidates from preferences (Graph)")
            pref_candidates = self.get_graph_candidates_from_prefs(
                session.preferences, exclude_list, constraints, negations
            )
            self.merge_candidates(all_candidates, pref_candidates, 'graph_pref')

        logger.info(f"Generated {len(all_candidates)} raw candidates.")

        # --- 2. Filtering ---
        logger.debug("Filtering candidates based on constraints and negations.")
        filtered_candidates = self.filter_candidates(all_candidates, constraints, negations)
        logger.info(f"Filtered down to {len(filtered_candidates)} candidates.")

        # --- 3. Ranking ---
        logger.debug("Ranking candidates.")
        ranked_list = self.rank_candidates(filtered_candidates, session)
        
        # --- 4. Diversification (Rule 2.2) ---
        logger.debug("Applying MMR diversification.")
        diversified_list = self.apply_mmr_diversity(ranked_list, top_k)
        
        # --- 5. Format Output ---
        logger.debug("Formatting final recommendations.")
        final_recs = []
        for movie_id, score, reason in diversified_list:
            try:
                label = self.entity_linker.get_label(movie_id)
                if label:
                    final_recs.append({'id': movie_id, 'label': label, 'score': score, 'reason': reason})
            except Exception as e:
                logger.warning(f"Could not get label for {movie_id}: {e}")
        
        logger.info(f"Returning {len(final_recs)} final recommendations.")
        return final_recs

    def merge_candidates(self,
                         main_dict: Dict[str, Dict],
                         new_candidates: Dict[str, Dict],
                         source_name: str):
        """
        Merges candidates from a new source into the main dictionary.
        """
        for movie_id, info in new_candidates.items():
            if movie_id not in main_dict:
                main_dict[movie_id] = {'score': 0.0, 'reasons': set(), 'rating': 0.0}
            
            score_weight = 1.0
            
            # --- THIS IS THE FIX ---
            # Your embeddings are bad.
            # Give them a very low score so SPARQL results
            # always win.
            if source_name == 'embed_seed':
                score_weight = 0.01 
            # --- END FIX ---
            elif source_name == 'graph_pref':
                score_weight = 2.0 # Explicit preferences are important
            
            main_dict[movie_id]['score'] += info.get('score', 0.5) * score_weight
            main_dict[movie_id]['reasons'].update(info.get('reasons', {'is a potential match'}))
            main_dict[movie_id]['rating'] = max(main_dict[movie_id]['rating'], info.get('rating', 0.0))

    def get_graph_candidates_from_seeds(self, 
                                        seed_movies: Set[str], 
                                        exclude_list: Set[str],
                                        constraints: Dict, 
                                        negations: Dict) -> Dict[str, Dict]:
        """
        Finds candidates that share properties with seed movies.
        This version loops through properties, which is robust.
        """
        candidates = {}
        seed_uris = [f"<{seed}>" for seed in seed_movies]
        
        properties_to_share = [
            (PREDICATE_MAP['genre'], 1.0, "shares the genre"),
            (PREDICATE_MAP['director'], 0.8, "has the same director"),
            (PREDICATE_MAP['actor'], 0.5, "shares an actor"),
            (PREDICATE_MAP['part of series'], 0.9, "is in the same series as"), # For slasher
            (PREDICATE_MAP['based on'], 0.7, "is based on similar work as"), # For Disney
        ]
        
        for prop_pid, weight, reason_prefix in properties_to_share:
            # Call the SIMPLE query function
            query = self.composer.get_recommendation_by_shared_property_query(
                seed_uris, prop_pid, constraints, negations
            )
            try:
                results = self.graph_executor.execute_query(query)
                for res in results:
                    movie_id = res['movie']['value'] # Use full IRI
                    if movie_id in exclude_list:
                        continue
                    
                    shared_val_label = res.get('propLabel', {}).get('value', 'a shared property')
                    reason = f"{reason_prefix} '{shared_val_label}'"
                    rating = float(res.get('rating', {}).get('value', 0.0))
                    
                    if movie_id not in candidates:
                        candidates[movie_id] = {'score': 0.0, 'reasons': set(), 'rating': 0.0}
                    
                    candidates[movie_id]['score'] += weight
                    candidates[movie_id]['reasons'].add(reason)
                    candidates[movie_id]['rating'] = max(candidates[movie_id]['rating'], rating)
                    
            except Exception as e:
                logger.error(f"Error querying for shared property {prop_pid}: {e}")
                
        return candidates

    def get_embedding_candidates_from_seeds(self, seed_movies: Set[str], exclude_list: Set[str]) -> Dict[str, Dict]:
        """
        Finds candidates using embedding similarity to seed movies.
        """
        candidates = {}
        try:
            all_neighbors = []
            for seed_id in seed_movies:
                # ID is already a full IRI
                neighbors = self.embedding_executor.get_nearest_neighbors(
                    entity_id=seed_id, k=20, embedding_type='entity'
                )
                all_neighbors.extend(neighbors) # neighbors are (IRI, score)
                
            for movie_id, score in all_neighbors:
                # movie_id is a full IRI
                if movie_id in exclude_list:
                    continue
                if movie_id not in candidates:
                    candidates[movie_id] = {'score': 0.0, 'reasons': set(), 'rating': 0.0}
                
                candidates[movie_id]['score'] += score 
                candidates[movie_id]['reasons'].add("it's similar to movies you like")
                
        except Exception as e:
            logger.error(f"Error getting embedding candidates: {e}")
        
        return candidates

    def get_graph_candidates_from_prefs(self, 
                                        preferences: Dict[str, Any], 
                                        exclude_list: Set[str],
                                        constraints: Dict, 
                                        negations: Dict) -> Dict[str, Dict]:
        """
        Finds candidates that match explicit preferences.
        """
        candidates = {}
        mapped_prefs = {}
        for pref_type, value_id in preferences.items():
            if pref_type in PREDICATE_MAP:
                pid = PREDICATE_MAP[pref_type]
                mapped_prefs[pid] = f"<{value_id}>" # value_id is already a full IRI
        
        if not mapped_prefs:
            return {}

        query = self.composer.get_recommendation_by_property_query(
            mapped_prefs, constraints, negations
        )
        try:
            results = self.graph_executor.execute_query(query)
            for res in results:
                movie_id = res['movie']['value'] # Use full IRI
                if movie_id in exclude_list:
                    continue
                
                rating = float(res.get('rating', {}).get('value', 0.0))
                
                if movie_id not in candidates:
                    candidates[movie_id] = {'score': 0.0, 'reasons': set(), 'rating': 0.0}
                
                candidates[movie_id]['score'] += 2.0 # High score for explicit match
                candidates[movie_id]['reasons'].add("it matches your preferences")
                candidates[movie_id]['rating'] = max(candidates[movie_id]['rating'], rating)
                
        except Exception as e:
            logger.error(f"Error querying for preferences: {e}")

        return candidates

    def filter_candidates(self, 
                          candidates: Dict[str, Dict], 
                          constraints: Dict[str, Any], 
                          negations: Dict[str, Any]) -> Dict[str, Dict]:
        """
        Filters candidates (especially from embeddings) to ensure they are
        movies and match all session constraints.
        This version re-uses the composer's filter-building logic.
        """
        if not candidates:
            return {}

        logger.info(f"Running secondary filter on {len(candidates)} candidates.")
        
        # 1. Build a VALUES block of all candidate IRIs
        candidate_iris = [f"<{c_id}>" for c_id in candidates.keys()]
        values_block = f"VALUES ?movie {{ {' '.join(candidate_iris)} }}"
        
        # 2. Reuse the composer's logic to build filter blocks
        filter_triples, filter_clauses = self.composer._build_filter_block(
            constraints, negations, "?movie"
        )

        # 3. Build the master filter query
        query = f"""
            {PREFIXES}
            
            SELECT ?movie
            WHERE {{
                {values_block}
                
                # Rule 1: Must be a movie
                ?movie wdt:P31 wd:Q11424 .
                
                # Rule 2: Apply all filter logic
                {filter_triples}
                {filter_clauses}
            }}
            GROUP BY ?movie
        """
        
        # 4. Execute query and build a set of valid movie IDs
        valid_movie_ids = set()
        try:
            results = self.graph_executor.execute_query(query)
            for res in results:
                
                # --- THIS IS THE FIX for the TypeError ---
                # res is a dict like {'movie': URIRef('http://...')}
                # We must convert the URIRef object to a string.
                valid_movie_ids.add(str(res['movie']))
                # --- END FIX ---
                
        except Exception as e:
            logger.error(f"Error during secondary filtering query: {e}. Failing open.", exc_info=True)
            return candidates # Fail open

        # 5. Create the new filtered candidate dictionary
        filtered_candidates = {
            mid: info for mid, info in candidates.items() 
            if mid in valid_movie_ids
        }
        
        logger.info(f"Secondary filter passed {len(filtered_candidates)} candidates.")
        return filtered_candidates

    def rank_candidates(self, candidates: Dict[str, Dict], session: Session) -> List[tuple]:
        """
        Ranks candidates based on merged scores and provides one main reason.
        (Eval 3 Fix - Rule 2.5 Rating Fallback)
        """
        ranked_list = []
        for movie_id, info in candidates.items():
            final_score = info['score']
            
            # Add score from rating (Rule 2.5)
            # Normalize 0-10 rating to 0-1 scale, add as small bonus
            final_score += (info['rating'] / 10.0) * 0.2 
            
            # Pick a reason.
            best_reason = "it's a good match"
            if info['reasons']:
                # Prioritize a non-embedding reason if available
                non_embed_reasons = [r for r in info['reasons'] if "similar" not in r]
                if non_embed_reasons:
                    best_reason = non_embed_reasons[0]
                else:
                    best_reason = list(info['reasons'])[0]
            
            ranked_list.append((movie_id, final_score, best_reason))
        
        ranked_list.sort(key=lambda x: x[1], reverse=True)
        return ranked_list

    def apply_mmr_diversity(self, 
                            ranked_list: List[tuple], 
                            k: int, 
                            lambda_val: float = 0.7) -> List[tuple]:
        """
        Re-ranks the list using Maximal Marginal Relevance (MMR) for diversity.
        (Eval 3 Fix - Rule 2.2)
        """
        if not ranked_list:
            return []
        
        if len(ranked_list) <= k:
            return ranked_list

        try:
            # Get all candidate IDs and embeddings at once
            all_candidate_ids = [mid for mid, score, reason in ranked_list]
            all_embeddings = self.embedding_executor.get_embeddings(all_candidate_ids)
            
            # Create a map of ID -> (score, reason, embedding)
            candidate_pool = {}
            for i, (mid, score, reason) in enumerate(ranked_list):
                if all_embeddings[i] is not None:
                    candidate_pool[mid] = (score, reason, all_embeddings[i])

            # Normalize scores to [0, 1]
            if not candidate_pool:
                 logger.warning("MMR: Candidate pool is empty after embedding check.")
                 return ranked_list[:k]
                 
            max_score = max(s for s, r, e in candidate_pool.values())
            if max_score > 0:
                for mid in candidate_pool:
                    s, r, e = candidate_pool[mid]
                    candidate_pool[mid] = (s / max_score, r, e)
            
            # Start MMR selection
            selected = []
            selected_ids = set()
            
            # Add the top-ranked item first
            top_id = list(candidate_pool.keys())[0]
            top_score, top_reason, top_embed = candidate_pool[top_id]
            selected.append((top_id, top_score, top_reason))
            selected_ids.add(top_id)
            del candidate_pool[top_id]

            while len(selected) < k and candidate_pool:
                best_item_id = None
                best_mmr_score = -np.inf
                
                selected_embeds = [candidate_pool[mid][2] for mid in selected_ids if mid in candidate_pool and candidate_pool[mid][2] is not None]
                
                for cand_id, (score, reason, embed) in candidate_pool.items():
                    if embed is None:
                        continue
                        
                    relevance = score
                    
                    # Calculate max similarity to already selected items
                    max_sim = 0.0
                    if selected_embeds:
                        for sel_embed in selected_embeds:
                            sim = self.embedding_executor.cosine_similarity(embed, sel_embed)
                            max_sim = max(max_sim, sim)
                    
                    mmr_score = (lambda_val * relevance) - ((1 - lambda_val) * max_sim)
                    
                    if mmr_score > best_mmr_score:
                        best_mmr_score = mmr_score
                        best_item_id = cand_id
                
                if best_item_id:
                    score, reason, _ = candidate_pool[best_item_id]
                    selected.append((best_item_id, score, reason))
                    selected_ids.add(best_item_id)
                    del candidate_pool[best_item_id]
                else:
                    break # No more valid candidates

            return selected
            
        except Exception as e:
            logger.error(f"Error during MMR diversification: {e}. Returning original ranked list.")
            return ranked_list[:k]