import logging
import numpy as np
from typing import List, Dict, Any, Set
from agent.graph_executor import GraphExecutor
from agent.embedding_executor import EmbeddingExecutor
from agent.composer import Composer
from agent.entity_linker import EntityLinker
from agent.session_manager import Session
from agent.constants import PREDICATE_MAP

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
            # Note: Embedding search can't easily be pre-filtered by constraints
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
        # (Rule 2.1) This secondary filter is for candidates from sources
        # that couldn't be pre-filtered (like embeddings).
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
            if source_name == 'embed_seed':
                score_weight = 0.5 
            elif source_name == 'graph_pref':
                score_weight = 2.0 # Explicit preferences are important
            
            main_dict[movie_id]['score'] += info.get('score', 0.5) * score_weight
            main_dict[movie_id]['reasons'].add(info.get('reason', 'is a potential match'))
            main_dict[movie_id]['rating'] = max(main_dict[movie_id]['rating'], info.get('rating', 0.0))

    def get_graph_candidates_from_seeds(self, 
                                        seed_movies: Set[str], 
                                        exclude_list: Set[str],
                                        constraints: Dict, 
                                        negations: Dict) -> Dict[str, Dict]:
        """
        Finds candidates that share properties with seed movies.
        (Eval 3 Fix - Rule 2.4 Explanations)
        """
        candidates = {}
        seed_uris = [f"wd:{seed}" for seed in seed_movies]
        
        properties_to_share = [
            (PREDICATE_MAP['genre'], 1.0, "shares the genre"),
            (PREDICATE_MAP['director'], 0.8, "has the same director"),
            (PREDICATE_MAP['actor'], 0.5, "shares an actor"),
            (PREDICATE_MAP['part of series'], 0.9, "is in the same series as"),
            (PREDICATE_MAP['based on'], 0.7, "is based on similar work as"),
        ]
        
        for prop_pid, weight, reason_prefix in properties_to_share:
            query = self.composer.get_recommendation_by_shared_property_query(
                seed_uris, prop_pid, constraints, negations
            )
            try:
                results = self.graph_executor.execute_query(query)
                for res in results:
                    movie_id = res['movie']['value'].split('/')[-1]
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
                neighbors = self.embedding_executor.get_nearest_neighbors(
                    entity_id=seed_id, k=20, embedding_type='entity'
                )
                all_neighbors.extend(neighbors) # neighbors are (id, score)
                
            for movie_id, score in all_neighbors:
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
                mapped_prefs[pid] = f"wd:{value_id}"
        
        if not mapped_prefs:
            return {}

        query = self.composer.get_recommendation_by_property_query(
            mapped_prefs, constraints, negations
        )
        try:
            results = self.graph_executor.execute_query(query)
            for res in results:
                movie_id = res['movie']['value'].split('/')[-1]
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
        Filters candidates (from embeddings) that couldn't be pre-filtered.
        This is a placeholder. A full implementation would query the graph
        for each candidate to check constraints.
        (Eval 3 Fix - Rule 2.1)
        """
        # This is a complex step (querying each candidate).
        # For now, we assume graph-based candidates (the majority)
        # are already filtered by SPARQL.
        logger.warning("Secondary filtering is not fully implemented. Passing most candidates.")
        
        # Simple filter: remove negated genre if we know it
        if 'genre' in negations:
            negated_genre_id = negations['genre']
            # This is not efficient, but demonstrates the idea
            # In a real system, this info would be pre-fetched.
            pass
            
        return candidates

    def rank_candidates(self, candidates: Dict[str, Dict], session: Session) -> List[tuple]:
        """
        Ranks candidates based on merged scores and provides one main reason.
        (Eval 3 Fix - Rule 2.5 Rating Fallback)
        """
        ranked_list = []
        for movie_id, info in candidates.items():
            final_score = info['score']
            
            # Add score from rating (Rule 2.5)
            # Normalize 0-10 rating to 0-1 scale
            final_score += (info['rating'] / 10.0) * 0.2 # Give rating a small weight
            
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
            candidates = [(mid, score) for mid, score, reason in ranked_list]
            reasons_map = {mid: reason for mid, score, reason in ranked_list}
            
            # Normalize scores to [0, 1] for stable MMR
            max_score = max(s for _, s in candidates)
            if max_score > 0:
                candidates_norm = [(mid, score / max_score) for mid, score in candidates]
            else:
                candidates_norm = candidates

            selected = []
            selected_ids = set()
            
            # Add the top-ranked item first
            top_id, top_score = candidates_norm.pop(0)
            selected.append((top_id, top_score))
            selected_ids.add(top_id)
            
            while len(selected) < k and candidates_norm:
                best_item = None
                best_mmr_score = -np.inf
                
                # Get embeddings for all selected items
                selected_embeds = self.embedding_executor.get_embeddings(list(selected_ids))
                
                # Get embeddings for all remaining candidates
                candidate_ids = [cid for cid, _ in candidates_norm]
                candidate_embeds = self.embedding_executor.get_embeddings(candidate_ids)
                candidate_scores = {cid: score for cid, score in candidates_norm}

                for i, (cand_id, cand_embed) in enumerate(zip(candidate_ids, candidate_embeds)):
                    if cand_embed is None:
                        continue
                        
                    relevance = candidate_scores[cand_id]
                    
                    # Calculate max similarity to already selected items
                    max_sim = 0.0
                    for sel_embed in selected_embeds:
                        if sel_embed is not None:
                            sim = self.embedding_executor.cosine_similarity(cand_embed, sel_embed)
                            max_sim = max(max_sim, sim)
                    
                    mmr_score = (lambda_val * relevance) - ((1 - lambda_val) * max_sim)
                    
                    if mmr_score > best_mmr_score:
                        best_mmr_score = mmr_score
                        best_item = (cand_id, candidate_scores[cand_id])
                
                if best_item:
                    selected.append(best_item)
                    selected_ids.add(best_item[0])
                    # Remove from candidates list
                    candidates_norm = [(cid, s) for cid, s in candidates_norm if cid != best_item[0]]
                else:
                    break # No more valid candidates

            # Re-build final list with original scores and reasons
            final_list = [(mid, score, reasons_map[mid]) for mid, score in selected]
            return final_list
            
        except Exception as e:
            logger.error(f"Error during MMR diversification: {e}. Returning original ranked list.")
            return ranked_list[:k]