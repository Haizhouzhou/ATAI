from __future__ import annotations

"""
Very small in-memory session manager.

The final event runs over Speakeasy with multiple chatrooms. This
module keeps a bit of per-session state (e.g. liked movies) so that
recommendations can get better over time.
"""

from dataclasses import dataclass, field
from typing import Dict, List, Optional, Set

from .utils import unique_preserve_order


@dataclass
class SessionState:
    """
    State associated with a single chat session.
    """

    session_id: str
    liked_film_uris: Set[str] = field(default_factory=set)
    last_mentioned_film_uris: List[str] = field(default_factory=list)
    last_question_type: Optional[str] = None


class SessionManager:
    """
    In-memory session store.

    For the evaluation event this is completely sufficient: each
    Speakeasy room / user can be mapped to a session_id and will get
    its own SessionState.
    """

    def __init__(self) -> None:
        self._sessions: Dict[str, SessionState] = {}

    def get_session(self, session_id: Optional[str]) -> SessionState:
        """
        Get or create a session with the given ID.

        If session_id is None, we use a shared 'default' session. This
        is useful for unit tests and local experiments.
        """
        sid = session_id or "default"
        if sid not in self._sessions:
            self._sessions[sid] = SessionState(session_id=sid)
        return self._sessions[sid]

    def remember_liked_films(self, session: SessionState, film_uris: List[str]) -> None:
        """
        Add the given film URIs to the user's liked set.
        """
        for uri in film_uris:
            session.liked_film_uris.add(uri)

        # keep a short "last mentioned" list in insertion order
        merged = list(session.last_mentioned_film_uris) + list(film_uris)
        session.last_mentioned_film_uris = unique_preserve_order(merged)[-20:]

    def reset(self, session_id: Optional[str] = None) -> None:
        """
        Reset a specific session or all sessions (if session_id is None).
        """
        if session_id is None:
            self._sessions.clear()
        else:
            self._sessions.pop(session_id, None)


# Global singleton accessor
_GLOBAL_SESSION_MANAGER: Optional[SessionManager] = None


def get_global_session_manager() -> SessionManager:
    global _GLOBAL_SESSION_MANAGER
    if _GLOBAL_SESSION_MANAGER is None:
        _GLOBAL_SESSION_MANAGER = SessionManager()
    return _GLOBAL_SESSION_MANAGER
