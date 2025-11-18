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
        logger.info(f"Getting recommendations for session {session.user_id}")
        
        all_candidates: Dict[str, Dict[str, Any]] = {}
        exclude_list = session.get_exclude_list()
        
        constraints = session.constraints
        negations = session.negations
        
        # --- 1. Candidate Generation ---
        
        # A. SEED MOVIES (Graph & Embedding)
        if session.seed_movies:
            logger.debug("Generating candidates from seed movies.")
            graph_candidates = self.get_graph_candidates_from_seeds(
                session.seed_movies, exclude_list, constraints, negations
            )
            self.merge_candidates(all_candidates, graph_candidates, 'graph_seed')

            embedding_candidates = self.get_embedding_candidates_from_seeds(session.seed_movies, exclude_list)
            self.merge_candidates(all_candidates, embedding_candidates, 'embed_seed')
            
        # B. PREFERENCES (Graph with Embedding Fallback)
        if session.preferences:
            logger.debug("Generating candidates from preferences.")
            pref_candidates = self.get_graph_candidates_from_prefs(
                session.preferences, exclude_list, constraints, negations
            )
            
            # Fallback: If explicit graph query returns nothing, treat the preference entities as seeds for embedding search
            if not pref_candidates:
                logger.info("No graph results for preferences. Trying embedding fallback.")
                pref_seeds = list(session.preferences.values())
                embed_pref_candidates = self.get_embedding_candidates_from_seeds(pref_seeds, exclude_list)
                self.merge_candidates(all_candidates, embed_pref_candidates, 'embed_seed')
            else:
                self.merge_candidates(all_candidates, pref_candidates, 'graph_pref')

        logger.info(f"Generated {len(all_candidates)} raw candidates.")

        # --- 2. Filtering ---
        filtered_candidates = self.filter_candidates(all_candidates, constraints, negations)
        
        # --- 3. Ranking ---
        ranked_list = self.rank_candidates(filtered_candidates, session)
        
        # --- 4. Diversification ---
        diversified_list = self.apply_mmr_diversity(ranked_list, top_k)
        
        # --- 5. Format Output & Fetch Images ---
        final_recs = []
        
        movie_ids_for_images = [mid for mid, _, _ in diversified_list]
        image_map = self.fetch_images(movie_ids_for_images)

        for movie_id, score, reason in diversified_list:
            try:
                label = self.entity_linker.get_label(movie_id)
                if not label: label = "Unknown Title" 

                rec_item = {'id': movie_id, 'label': label, 'score': score, 'reason': reason}
                
                if movie_id in image_map:
                    rec_item['image_id'] = image_map[movie_id]
                    
                final_recs.append(rec_item)
            except Exception as e:
                logger.warning(f"Error formatting recommendation {movie_id}: {e}")
        
        return final_recs

    def fetch_images(self, movie_ids: List[str]) -> Dict[str, str]:
        if not movie_ids: return {}
        image_map = {}
        try:
            query = self.composer.get_image_query(movie_ids)
            if query:
                results = self.graph_executor.execute_query(query)
                if results and isinstance(results, list):
                    for res in results:
                        try:
                            movie_uri = str(res['movie'])
                            image_url = str(res['imageUrl'])
                            # Expected format: https://.../images/0227/rm73963776.jpg
                            if '/' in image_url:
                                filename = image_url.split('/')[-1]
                                img_id = filename.split('.')[0]
                                image_map[movie_uri] = img_id
                        except Exception as e:
                            logger.debug(f"Failed to parse image URL: {e}")
                            continue
        except Exception as e:
            logger.error(f"Error fetching images: {e}")
        return image_map

    def merge_candidates(self, main_dict, new_candidates, source_name):
        for movie_id, info in new_candidates.items():
            if movie_id not in main_dict:
                main_dict[movie_id] = {'score': 0.0, 'reasons': set(), 'rating': 0.0}
            
            score_weight = 1.0
            if source_name == 'embed_seed': score_weight = 0.8 
            elif source_name == 'graph_pref': score_weight = 3.0
            elif source_name == 'graph_seed': score_weight = 1.5 
            
            main_dict[movie_id]['score'] += info.get('score', 0.5) * score_weight
            main_dict[movie_id]['reasons'].update(info.get('reasons', {'is a potential match'}))
            main_dict[movie_id]['rating'] = max(main_dict[movie_id]['rating'], info.get('rating', 0.0))

    def get_graph_candidates_from_seeds(self, seed_movies, exclude_list, constraints, negations):
        candidates = {}
        seed_uris = [f"<{seed}>" for seed in seed_movies]
        
        properties_to_share = [
            (PREDICATE_MAP['genre'], 2.0, "shares the genre"), 
            (PREDICATE_MAP['part of series'], 3.0, "is in the same series as"),
            (PREDICATE_MAP['director'], 1.5, "has the same director"),
            (PREDICATE_MAP['actor'], 1.0, "shares an actor"),
        ]
        
        for prop_pid, weight, reason_prefix in properties_to_share:
            try:
                query = self.composer.get_recommendation_by_shared_property_query(
                    seed_uris, prop_pid, constraints, negations
                )
                results = self.graph_executor.execute_query(query)
                if results and isinstance(results, list):
                    for res in results:
                        try:
                            movie_id = str(res['movie'])
                            if movie_id in exclude_list: continue
                            
                            prop_label_term = res.get('propLabel')
                            shared_val_label = prop_label_term.value if prop_label_term else 'common element'
                            reason = f"{reason_prefix} '{shared_val_label}'"
                            
                            rating_term = res.get('rating')
                            rating = float(rating_term.value) if rating_term else 0.0
                            
                            if movie_id not in candidates:
                                candidates[movie_id] = {'score': 0.0, 'reasons': set(), 'rating': 0.0}
                            
                            candidates[movie_id]['score'] += weight
                            candidates[movie_id]['reasons'].add(reason)
                            candidates[movie_id]['rating'] = max(candidates[movie_id]['rating'], rating)
                        except Exception: continue
            except Exception: continue
        return candidates

    def get_embedding_candidates_from_seeds(self, seed_movies, exclude_list):
        candidates = {}
        try:
            all_neighbors = []
            for seed_id in seed_movies:
                try:
                    neighbors = self.embedding_executor.get_nearest_neighbors(
                        entity_id=seed_id, k=40, embedding_type='entity'
                    )
                    all_neighbors.extend(neighbors) 
                except Exception: continue
                
            for movie_id, score in all_neighbors:
                if movie_id in exclude_list: continue
                if movie_id not in candidates:
                    candidates[movie_id] = {'score': 0.0, 'reasons': set(), 'rating': 0.0}
                
                candidates[movie_id]['score'] += score 
                candidates[movie_id]['reasons'].add("it's similar to movies you like")
        except Exception: pass
        return candidates

    def get_graph_candidates_from_prefs(self, preferences, exclude_list, constraints, negations):
        candidates = {}
        mapped_prefs = {}
        for pref_type, value_id in preferences.items():
            if pref_type in PREDICATE_MAP:
                pid = PREDICATE_MAP[pref_type]
                mapped_prefs[pid] = f"<{value_id}>" 
        
        if not mapped_prefs: return {}

        try:
            query = self.composer.get_recommendation_by_property_query(
                mapped_prefs, constraints, negations
            )
            results = self.graph_executor.execute_query(query)
            
            if results and isinstance(results, list):
                for res in results:
                    try:
                        movie_id = str(res['movie'])
                        if movie_id in exclude_list: continue
                        
                        rating_term = res.get('rating')
                        rating = float(rating_term.value) if rating_term else 0.0
                        
                        if movie_id not in candidates:
                            candidates[movie_id] = {'score': 0.0, 'reasons': set(), 'rating': 0.0}
                        
                        candidates[movie_id]['score'] += 3.0
                        candidates[movie_id]['reasons'].add("it matches your preferences")
                        candidates[movie_id]['rating'] = max(candidates[movie_id]['rating'], rating)
                    except Exception: continue
        except Exception as e:
            logger.error(f"Error querying for preferences: {e}")
        return candidates

    def filter_candidates(self, candidates, constraints, negations):
        if not candidates: return {}
        
        candidate_iris = [f"<{c_id}>" for c_id in candidates.keys()]
        if len(candidate_iris) > 200:
             sorted_keys = sorted(candidates.keys(), key=lambda k: candidates[k]['score'], reverse=True)[:200]
             candidate_iris = [f"<{k}>" for k in sorted_keys]

        if not candidate_iris: return {}

        values_block = f"VALUES ?movie {{ {' '.join(candidate_iris)} }}"
        filter_triples, filter_clauses = self.composer._build_filter_block(constraints, negations, "?movie")

        query = f"""
            {PREFIXES}
            SELECT ?movie
            WHERE {{
                {values_block}
                ?movie wdt:P31 wd:Q11424 .
                {filter_triples}
                {filter_clauses}
            }}
            GROUP BY ?movie
        """
        
        valid_movie_ids = set()
        try:
            results = self.graph_executor.execute_query(query)
            if results and isinstance(results, list):
                for res in results:
                    valid_movie_ids.add(str(res['movie']))
        except Exception:
            return candidates

        return {mid: info for mid, info in candidates.items() if mid in valid_movie_ids}

    def rank_candidates(self, candidates, session):
        ranked_list = []
        for movie_id, info in candidates.items():
            # Rating boost: A rating of 10 adds 5.0 to the score, dominating other factors if needed
            final_score = info['score'] + (info['rating'] * 0.5) 
            
            best_reason = "it's a good match"
            if info['reasons']:
                non_embed_reasons = [r for r in info['reasons'] if "similar" not in r]
                best_reason = non_embed_reasons[0] if non_embed_reasons else list(info['reasons'])[0]
            
            ranked_list.append((movie_id, final_score, best_reason))
        
        ranked_list.sort(key=lambda x: x[1], reverse=True)
        return ranked_list

    def apply_mmr_diversity(self, ranked_list, k, lambda_val=0.7):
        if not ranked_list or len(ranked_list) <= k: return ranked_list[:k]
        try:
            all_candidate_ids = [mid for mid, _, _ in ranked_list]
            all_embeddings = self.embedding_executor.get_embeddings(all_candidate_ids)
            
            candidate_pool = {}
            for i, (mid, score, reason) in enumerate(ranked_list):
                if all_embeddings[i] is not None:
                    candidate_pool[mid] = (score, reason, all_embeddings[i])

            if not candidate_pool: return ranked_list[:k]
                 
            max_score = max(s for s, r, e in candidate_pool.values())
            if max_score > 0:
                for mid in candidate_pool:
                    s, r, e = candidate_pool[mid]
                    candidate_pool[mid] = (s / max_score, r, e)
            
            selected = []
            selected_ids = set()
            
            top_id = list(candidate_pool.keys())[0]
            top_score, top_reason, top_embed = candidate_pool[top_id]
            selected.append((top_id, top_score, top_reason))
            selected_ids.add(top_id)
            del candidate_pool[top_id]

            while len(selected) < k and candidate_pool:
                best_item_id = None
                best_mmr_score = -np.inf
                
                selected_embeds = [candidate_pool[mid][2] for mid in selected_ids if mid in candidate_pool]
                
                for cand_id, (score, reason, embed) in candidate_pool.items():
                    relevance = score
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
                    break 
            return selected
        except Exception:
            return ranked_list[:k]