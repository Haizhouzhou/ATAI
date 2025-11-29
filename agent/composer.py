from __future__ import annotations

"""
Composer: central orchestration of all submodules.

High-level flow for one user message:

    1. Classify the query (factual / recommendation / multimedia / other).
    2. Link entities mentioned in the text (films, people).
    3. Depending on the type:
         - FACTUAL        -> use relation_mapper + graph_executor
         - RECOMMENDATION -> use preference_parser + recommendation_engine
         - MULTIMEDIA     -> use multimedia_index
         - OTHER          -> give a short capability explanation
    4. Format a natural-language answer (plus image tokens if needed).
"""

from dataclasses import dataclass
from typing import Dict, Iterable, List, Optional

from .logging_config import get_logger
from . import constants as C
from .nlq import NLQClassifier, NLQAnalysis, QueryType
from .relation_mapper import RelationMapper
from .graph_executor import GraphExecutor, get_global_graph_executor
from .entity_linker import EntityLinker, LinkedEntity, get_global_entity_linker
from .preference_parser import PreferenceParser, UserPreferences
from .recommendation_engine import RecommendationEngine
from .multimedia_index import MultimediaIndex, get_global_multimedia_index
from .session_manager import SessionManager, SessionState, get_global_session_manager
from .utils import unique_preserve_order

logger = get_logger(__name__)


@dataclass
class Answer:
    """
    Internal representation of an answer before rendering to text.
    """

    text: str
    image_tokens: List[str]


class AgentCore:
    """
    High-level agent core that composes all the building blocks.
    """

    def __init__(
        self,
        graph: Optional[GraphExecutor] = None,
        linker: Optional[EntityLinker] = None,
        nlq: Optional[NLQClassifier] = None,
        relation_mapper: Optional[RelationMapper] = None,
        pref_parser: Optional[PreferenceParser] = None,
        rec_engine: Optional[RecommendationEngine] = None,
        mm_index: Optional[MultimediaIndex] = None,
        session_manager: Optional[SessionManager] = None,
    ) -> None:
        self.graph: GraphExecutor = graph or get_global_graph_executor()
        self.linker: EntityLinker = linker or get_global_entity_linker()
        self.nlq: NLQClassifier = nlq or NLQClassifier()
        self.relation_mapper: RelationMapper = relation_mapper or RelationMapper()
        self.pref_parser: PreferenceParser = pref_parser or PreferenceParser(self.linker)
        self.rec_engine: RecommendationEngine = rec_engine or RecommendationEngine()
        self.mm_index: MultimediaIndex = mm_index or get_global_multimedia_index()
        self.sessions: SessionManager = session_manager or get_global_session_manager()

    # ------------------------------------------------------------------
    # Public entrypoint
    # ------------------------------------------------------------------

    def answer(self, message: str, session_id: Optional[str] = None) -> str:
        """
        Main entrypoint: compute a textual answer for the user message.
        """
        session = self.sessions.get_session(session_id)
        analysis = self.nlq.classify(message)
        session.last_question_type = analysis.query_type.name

        try:
            if analysis.query_type is QueryType.MULTIMEDIA:
                answer = self._handle_multimedia(message, session)
            elif analysis.query_type is QueryType.RECOMMENDATION:
                answer = self._handle_recommendation(message, session)
            elif analysis.query_type is QueryType.FACTUAL:
                answer = self._handle_factual(message, session)
            else:
                answer = self._handle_other(message, session)
        except Exception as e:
            logger.exception("Error while generating answer: %s", e)
            return C.RESPONSE_ERROR

        # Merge text and image tokens into a single string. For Speakeasy,
        # it's enough to append tokens on their own lines.
        if not answer.image_tokens:
            return answer.text

        token_block = "\n".join(answer.image_tokens)
        if answer.text.strip():
            return f"{answer.text}\n\n{token_block}"
        else:
            return token_block

    # ------------------------------------------------------------------
    # Handlers for different query types
    # ------------------------------------------------------------------

    def _handle_multimedia(self, message: str, session: SessionState) -> Answer:
        ents = self.linker.link_entities(message, max_candidates=10)

        if not ents:
            text = (
                "I can show posters or images if you mention a concrete "
                "movie or person. For example: 'Show me the poster of Titanic.'"
            )
            return Answer(text=text, image_tokens=[])

        # Prefer films; fallback to persons; else any entity.
        target = None
        for e in ents:
            if e.type == "film":
                target = e
                break
        if target is None:
            target = ents[0]

        image_tokens = self.mm_index.tokens_for_uri(target.uri, max_images=1)
        if not image_tokens:
            text = f"I could not find an image for '{target.label}'."
            return Answer(text=text, image_tokens=[])

        text = f"Here is an image for '{target.label}':"
        return Answer(text=text, image_tokens=image_tokens)

    def _handle_recommendation(self, message: str, session: SessionState) -> Answer:
        prefs: UserPreferences = self.pref_parser.parse(message)

        # Update session with liked films
        if prefs.liked_film_uris:
            self.sessions.remember_liked_films(session, prefs.liked_film_uris)

        # Use session-level liked films as seeds if available; otherwise
        # fall back to just what was mentioned in this message.
        seeds = (
            list(session.liked_film_uris)
            if session.liked_film_uris
            else prefs.liked_film_uris
        )

        if not seeds:
            text = (
                "To recommend movies, please mention at least one film you like. "
                "For example: 'I loved Inception and Interstellar, recommend me similar movies.'"
            )
            return Answer(text=text, image_tokens=[])

        recs = self.rec_engine.recommend_from_watchlist(
            liked_uris=seeds,
            top_k_per_seed=20,
            final_top_k=10,
            min_similarity=0.0,
        )

        if not recs:
            return Answer(
                text=(
                    "I could not find good recommendations based on your preferences. "
                    "Try mentioning a few more movies that you like."
                ),
                image_tokens=[],
            )

        lines: List[str] = []
        lines.append("Here are some movies you might enjoy:")

        uris_for_images: List[str] = []

        for idx, rec in enumerate(recs, start=1):
            uri = rec["uri"]
            title = rec.get("title") or rec.get("description") or uri.rsplit("/", 1)[-1]
            desc = rec.get("description") or ""
            score = rec.get("score")
            support = rec.get("support")

            extra_bits = []
            if isinstance(support, int) and support > 1:
                extra_bits.append(f"similar to {support} of your liked movies")
            # We deliberately do not print numeric similarity to keep the
            # answer natural.

            if extra_bits:
                line = f"{idx}. {title} ({'; '.join(extra_bits)})"
            else:
                line = f"{idx}. {title}"

            if desc and desc != title:
                line += f" – {desc}"

            lines.append(line)
            uris_for_images.append(uri)

        # Attach at most one image per recommended film
        uris_for_images = unique_preserve_order(uris_for_images)
        image_tokens: List[str] = []
        for uri in uris_for_images[:5]:
            image_tokens.extend(self.mm_index.tokens_for_uri(uri, max_images=1))

        return Answer(text="\n".join(lines), image_tokens=image_tokens)

    def _handle_factual(self, message: str, session: SessionState) -> Answer:
        rel_match = self.relation_mapper.infer_relation(message)
        if rel_match is None:
            # Fallback: treat as generic factual question
            return Answer(
                text=(
                    "I can answer factual questions about movies, such as "
                    "directors, release dates, cast, languages, countries of origin "
                    "and awards. Please rephrase your question with a specific movie."
                ),
                image_tokens=[],
            )

        ents = self.linker.link_entities(message, max_candidates=20)
        films = [e for e in ents if e.type == "film"]
        persons = [e for e in ents if e.type == "person"]

        if films:
            subject = films[0]
            session.last_mentioned_film_uris = [subject.uri]
            text = self._answer_relation_for_film(subject, rel_match)
            # show a poster for the film as small bonus
            img_tokens = self.mm_index.tokens_for_uri(subject.uri, max_images=1)
            return Answer(text=text, image_tokens=img_tokens)

        if persons:
            subject = persons[0]
            text = self._answer_relation_for_person(subject, rel_match)
            img_tokens = self.mm_index.tokens_for_uri(subject.uri, max_images=1)
            return Answer(text=text, image_tokens=img_tokens)

        # No entity found
        return Answer(
            text=(
                "I could not identify which movie or person you are asking about. "
                "Please mention the movie title or the person's name explicitly."
            ),
            image_tokens=[],
        )

    def _handle_other(self, message: str, session: SessionState) -> Answer:
        """
        Fallback for small talk / unsupported questions.
        """
        text = (
            "I am a movie assistant. I can:\n"
            "- answer questions like 'Who directed Inception?' or "
            "'From which country is Amélie?'\n"
            "- recommend similar movies based on films you like\n"
            "- show posters or images for movies and actors\n\n"
            "Try asking me about a specific movie!"
        )
        return Answer(text=text, image_tokens=[])

    # ------------------------------------------------------------------
    # Factual helper methods
    # ------------------------------------------------------------------

    def _answer_relation_for_film(self, film: LinkedEntity, rel_match) -> str:
        key = rel_match.key
        prop_uri = rel_match.property_uri

        idx = film.idx
        title = film.label or self.graph.get_label(idx)

        # Some relations have specialised mappings inside GraphExecutor.
        if key == "director":
            targets = self.graph.directors_by_film.get(idx, [])
        elif key == "screenwriter":
            targets = self.graph.screenwriters_by_film.get(idx, [])
        elif key == "composer":
            targets = self.graph.composers_by_film.get(idx, [])
        elif key == "genre":
            targets = self.graph.genres_by_film.get(idx, [])
        elif key == "country_of_origin":
            targets = self.graph.countries_by_film.get(idx, [])
        elif key == "language":
            targets = self.graph.languages_by_film.get(idx, [])
        elif key == "award":
            targets = self.graph.award_nominations_by_film.get(idx, [])
        elif key == "publication_date":
            dates = self.graph.get_release_dates(idx)
            if dates:
                # Choose the earliest date (sorted lexicographically)
                best = sorted(dates)[0]
                return f"The movie '{title}' was released on {best}."
            # Fallback: try generic KG relation
            targets = self.graph.get_out_neighbors(idx, prop_uri)
        else:
            # generic KG lookup for any other relation
            targets = self.graph.get_out_neighbors(idx, prop_uri)

        if not targets:
            return f"I could not find any information about the {rel_match.label} of '{title}'."

        labels = [self.graph.get_label(t) for t in unique_preserve_order(targets)]
        joined = self._join_list(labels)

        if key in ("director", "screenwriter", "composer", "cast_member"):
            return f"The {rel_match.label} of '{title}' is {joined}."
        elif key in ("country_of_origin", "language", "genre"):
            return f"'{title}' is a {joined} {rel_match.label}."
        elif key == "award":
            return f"'{title}' has been nominated for or received {joined}."
        else:
            return f"The {rel_match.label} of '{title}' is {joined}."

    def _answer_relation_for_person(self, person: LinkedEntity, rel_match) -> str:
        key = rel_match.key
        prop_uri = rel_match.property_uri

        idx = person.idx
        name = person.label or self.graph.get_label(idx)

        # For person queries, we usually want movies where the person
        # is connected as object under the given property, e.g.
        #   (film, P57, director_person)
        film_idxs = self.graph.get_in_neighbors(idx, prop_uri)
        film_idxs = [i for i in film_idxs if i in self.graph.film_idxs]

        if not film_idxs:
            return (
                f"I could not find any movies related to {name} for the relation "
                f"'{rel_match.label}'."
            )

        film_labels = [self.graph.get_label(i) for i in unique_preserve_order(film_idxs)]
        joined = self._join_list(film_labels)

        if key == "director":
            return f"{name} directed: {joined}."
        elif key == "screenwriter":
            return f"{name} wrote the screenplay for: {joined}."
        elif key == "composer":
            return f"{name} composed music for: {joined}."
        elif key == "cast_member":
            return f"{name} appeared in: {joined}."
        else:
            return f"Movies related to {name} ({rel_match.label}): {joined}."

    @staticmethod
    def _join_list(items: Iterable[str]) -> str:
        """
        Join a list of labels into a natural-sounding English phrase.
        """
        lst = [s for s in items if s]
        if not lst:
            return ""
        if len(lst) == 1:
            return lst[0]
        if len(lst) == 2:
            return f"{lst[0]} and {lst[1]}"
        return ", ".join(lst[:-1]) + f", and {lst[-1]}"


# Global singleton for convenience
_GLOBAL_COMPOSER: Optional[AgentCore] = None


def get_global_composer() -> AgentCore:
    global _GLOBAL_COMPOSER
    if _GLOBAL_COMPOSER is None:
        _GLOBAL_COMPOSER = AgentCore()
    return _GLOBAL_COMPOSER
