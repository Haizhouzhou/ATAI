import logging
from typing import Dict, Any, Set, List, Tuple

logger = logging.getLogger(__name__)

class Session:
    """
    Holds the conversational state for a single user.
    """
    def __init__(self, user_id: str):
        self.user_id: str = user_id
        self.history: List[Tuple[str, str]] = []
        self.seed_movies: Set[str] = set()
        self.preferences: Dict[str, Any] = {}
        self.constraints: Dict[str, Any] = {}
        self.recommended_movies: Set[str] = set()
        self.negations: Dict[str, Any] = {}
        logger.info(f"New session created for user {user_id}")

    def update(self, parsed_intent: Dict[str, Any]):
        """
        Updates the session state based on new parsed preferences.
        """
        logger.debug(f"Updating session {self.user_id} with {parsed_intent}")
        
        # Add new seed movies
        new_seeds = parsed_intent.get('seed_movies', [])
        if new_seeds:
            self.seed_movies.update(new_seeds)
            logger.info(f"Session {self.user_id}: Added seeds {new_seeds}")

        # Update preferences (e.g., genre, actor)
        new_prefs = parsed_intent.get('preferences', {})
        if new_prefs:
            self.preferences.update(new_prefs)
            logger.info(f"Session {self.user_id}: Updated prefs {new_prefs}")

        # Update constraints (e.g., year)
        new_constraints = parsed_intent.get('constraints', {})
        if new_constraints:
            self.constraints.update(new_constraints)
            logger.info(f"Session {self.user_id}: Updated constraints {new_constraints}")
            
        # Update negations
        new_negations = parsed_intent.get('negations', {})
        if new_negations:
            self.negations.update(new_negations)
            logger.info(f"Session {self.user_id}: Updated negations {new_negations}")

        # If it's a follow-up ("more like that")
        if parsed_intent.get('is_follow_up') and not new_seeds and self.recommended_movies:
            logger.info(f"Session {self.user_id}: Using last recommendations as new seeds.")
            self.seed_movies.update(self.recommended_movies)
            self.recommended_movies.clear() # Clear old recs so they can be "seeds"

    def add_recommendations(self, movie_ids: List[str]):
        """
        Adds a list of recommended movie IDs to the state to avoid re-recommending.
        """
        self.recommended_movies.update(movie_ids)
        logger.info(f"Session {self.user_id}: Added {len(movie_ids)} movies to recommended list.")

    def get_exclude_list(self) -> Set[str]:
        """
        Returns all movies to be excluded from the next recommendation.
        """
        return self.seed_movies.union(self.recommended_movies)

    def clear(self):
        """
        Resets the session state.
        """
        logger.info(f"Clearing session for user {self.user_id}")
        self.history.clear()
        self.seed_movies.clear()
        self.preferences.clear()
        self.constraints.clear()
        self.recommended_movies.clear()
        self.negations.clear()


class SessionManager:
    """
    Manages all active user sessions.
    """
    def __init__(self):
        self.sessions: Dict[str, Session] = {}
        logger.info("SessionManager initialized.")

    def get_session(self, user_id: str) -> Session:
        """
        Retrieves or creates a session for a given user ID.
        """
        if user_id not in self.sessions:
            logger.info(f"Creating new session for user_id: {user_id}")
            self.sessions[user_id] = Session(user_id)
        
        return self.sessions[user_id]

    def clear_session(self, user_id: str):
        """
        Clears the state of a specific user's session.
        """
        if user_id in self.sessions:
            self.sessions[user_id].clear()
            logger.info(f"Cleared session for user_id: {user_id}")