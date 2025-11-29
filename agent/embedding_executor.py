# agent/embedding_executor.py

from __future__ import annotations

"""
EmbeddingExecutor: central place to load and query movie knowledge-graph embeddings.

This module is responsible for:
    - Loading entity embeddings from RFC_entity_embeds.npy
    - Aligning them with entity_ids_ordered.tsv
    - Filtering film entities using film_entities.tsv
    - Attaching human-readable descriptions from movie_plots.tsv
    - Providing convenient search functions over the film embedding space

It has **no** dependency on Speakeasy, FastAPI, etc. so it can be imported from
notebooks, unit tests, or the agent without side effects.
"""

import os
import re
from dataclasses import dataclass
from typing import Dict, Iterable, List, Optional

import numpy as np
import pandas as pd


# =====================================================================
# 1. Configuration
# =====================================================================


@dataclass
class EmbeddingConfig:
    """
    Configuration for loading the embedding and metadata files.

    All paths are given relative to the project root (or to the process
    working directory when this module is imported).
    """

    embedding_path: str = "embeddings/RFC_entity_embeds.npy"
    entity_ids_path: str = "metadata/entity_ids_ordered.tsv"
    film_entities_path: str = "metadata/film_entities.tsv"
    movie_plots_path: str = "metadata/movie_plots.tsv"

    # If you ever need to limit the number of entities (for debugging),
    # you can set this. In production we normally leave it as None.
    max_entities: Optional[int] = None


# =====================================================================
# 2. Internal helper functions for loading and cleaning metadata
# =====================================================================


def _load_embeddings(config: EmbeddingConfig) -> np.ndarray:
    """
    Load the entity embedding matrix from disk.

    Returns
    -------
    np.ndarray
        Shape (num_entities, dim)
    """
    if not os.path.exists(config.embedding_path):
        raise FileNotFoundError(f"Embedding file not found: {config.embedding_path}")

    entity_embeds = np.load(config.embedding_path)
    if entity_embeds.ndim != 2:
        raise ValueError(f"Expected 2D embeddings, got shape {entity_embeds.shape}")

    if config.max_entities is not None:
        entity_embeds = entity_embeds[: config.max_entities]

    return entity_embeds


def _load_entity_ids(path: str, num_entities: int) -> pd.DataFrame:
    """
    Load entity_ids_ordered.tsv and align it with the embedding rows.

    The file contains at least two columns:
        - 'index' (0..N-1, but with a spurious header-like row)
        - 'uri'   (wikidata URI as string, also with a header-like row)

    We must:
        - drop rows where 'index' == 'index' or 'uri' == 'uri'
        - coerce 'index' to int
        - sort by index
        - truncate to the number of embeddings
    """
    if not os.path.exists(path):
        raise FileNotFoundError(f"entity_ids file not found: {path}")

    df = pd.read_csv(path, sep="\t", dtype=str, low_memory=False)

    if "index" not in df.columns or "uri" not in df.columns:
        raise ValueError("entity_ids_ordered.tsv must have at least 'index' and 'uri' columns.")

    # Drop the fake header row that was written as data
    bad_mask = (df["index"] == "index") | (df["uri"] == "uri")
    df = df[~bad_mask].copy()

    # Coerce to integer indices and drop bad rows
    df["index"] = pd.to_numeric(df["index"], errors="coerce")
    df = df.dropna(subset=["index"])
    df["index"] = df["index"].astype(int)

    # Sort and align with embedding count
    df = df.sort_values("index").reset_index(drop=True)

    if len(df) < num_entities:
        raise ValueError(
            f"entity_ids_ordered.tsv has only {len(df)} rows, "
            f"but embeddings have {num_entities} rows."
        )

    if len(df) > num_entities:
        # Truncate silently but explicitly; we log via print so you see it once.
        print(
            f"[EmbeddingExecutor] entity_ids has {len(df)} rows, "
            f"but embeddings have {num_entities}; truncating."
        )
        df = df.iloc[:num_entities].copy()

    return df


def _load_film_entities(path: str, num_entities: int) -> pd.DataFrame:
    """
    Load film_entities.tsv.

    The file contains:
        - 'index'   : embedding row index
        - 'uri'     : same URI as in entity_ids_ordered
        - 'is_film' : 1 if the entity is a film
        - 'has_plot': 1 if we have a plot for this entity

    As with entity_ids_ordered.tsv, the first data row sometimes repeats the header
    and needs to be removed.
    """
    if not os.path.exists(path):
        raise FileNotFoundError(f"film_entities file not found: {path}")

    df = pd.read_csv(path, sep="\t", dtype=str, low_memory=False)

    required_cols = {"index", "uri", "is_film", "has_plot"}
    missing = required_cols.difference(df.columns)
    if missing:
        raise ValueError(f"film_entities.tsv is missing required columns: {missing}")

    bad_mask = (df["index"] == "index") | (df["uri"] == "uri")
    df = df[~bad_mask].copy()

    df["index"] = pd.to_numeric(df["index"], errors="coerce")
    df = df.dropna(subset=["index"])
    df["index"] = df["index"].astype(int)

    for col in ("is_film", "has_plot"):
        df[col] = pd.to_numeric(df[col], errors="coerce").fillna(0).astype(int)

    # Only keep rows whose index is actually present in the embedding matrix
    df = df[df["index"] < num_entities].copy()

    df = df.sort_values("index").reset_index(drop=True)

    return df


def _load_movie_plots(path: str) -> pd.DataFrame:
    """
    Load movie_plots.tsv and return a DataFrame with columns:

        - 'uri'         : wikidata URI
        - 'description' : free text (often title + plot in one field)

    In the provided dataset this file has two columns without a header:
        0: uri
        1: text (title + plot or just plot)

    There is again a header-like row that we drop by checking the 'uri' prefix.
    """
    if not os.path.exists(path):
        raise FileNotFoundError(f"movie_plots file not found: {path}")

    raw = pd.read_csv(path, sep="\t", header=None, dtype=str, low_memory=False)

    if raw.shape[1] < 2:
        raise ValueError(
            "movie_plots.tsv is expected to have at least 2 columns "
            "(uri, description)."
        )

    df = raw.iloc[:, :2].copy()
    df = df.rename(columns={0: "uri", 1: "description"})

    # Drop non-URI rows such as the accidental header row
    df = df[df["uri"].astype(str).str.startswith("http")].copy()

    df["uri"] = df["uri"].astype(str)
    df["description"] = df["description"].astype(str)

    # There can be duplicates for some URIs; keep the first occurrence.
    df = df.drop_duplicates(subset=["uri"], keep="first")

    return df


# =====================================================================
# 3. Core index class
# =====================================================================


class MovieEmbeddingIndex:
    """
    A read-only index over the movie embeddings.

    This class holds:
        - the full entity embedding matrix
        - mappings between indices and URIs
        - which indices correspond to films
        - optional textual descriptions from movie_plots.tsv

    It also precomputes L2-normalized embeddings and provides methods
    to query nearest neighbours for a given film, and to look up films
    by (approximate) title.
    """

    def __init__(self, config: Optional[EmbeddingConfig] = None) -> None:
        self.config: EmbeddingConfig = config or EmbeddingConfig()

        # Raw embeddings for all entities
        self.entity_embeds: np.ndarray = _load_embeddings(self.config)
        self.num_entities: int = self.entity_embeds.shape[0]
        self.embed_dim: int = self.entity_embeds.shape[1]

        # Metadata
        self.entity_ids_df: pd.DataFrame = _load_entity_ids(
            self.config.entity_ids_path,
            self.num_entities,
        )
        self.film_df: pd.DataFrame = _load_film_entities(
            self.config.film_entities_path,
            self.num_entities,
        )
        self.movie_plots_df: pd.DataFrame = _load_movie_plots(
            self.config.movie_plots_path
        )

        # Build mapping dictionaries between indices and URIs
        self.idx_to_uri: Dict[int, str] = dict(
            zip(self.entity_ids_df["index"], self.entity_ids_df["uri"])
        )
        self.uri_to_idx: Dict[str, int] = {uri: idx for idx, uri in self.idx_to_uri.items()}

        # Join film entities with movie plots so that each film can have a description and title.
        self.film_df = self._attach_plots_to_films(self.film_df)

        # Film indices present both in film_df and in embeddings
        self.film_indices: np.ndarray = (
            self.film_df.loc[self.film_df["is_film"] == 1, "index"].astype(int).values
        )

        # Cached normalized embeddings
        self._entity_embeds_normalized: Optional[np.ndarray] = None
        self._film_embeds_normalized: Optional[np.ndarray] = None

        # Convenience mapping from uri to description / title text
        self.uri_to_description: Dict[str, str] = self._build_uri_to_description()
        self.uri_to_title: Dict[str, str] = self._build_uri_to_title()
        self.title_to_uris: Dict[str, List[str]] = self._build_title_to_uris()

        # Precompute normalized embeddings once; this is safe and cheap.
        self._ensure_normalized()

        print(
            f"[EmbeddingExecutor] Loaded {self.num_entities} entities "
            f"({self.embed_dim}-D); {len(self.film_indices)} of them are films; "
            f"{len(self.uri_to_description)} films have descriptions."
        )

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    @staticmethod
    def _guess_title_from_description(text: str) -> str:
        """
        Heuristic to guess the movie title from the description field.

        The movie_plots descriptions in the ATAI dataset typically start with the
        movie title followed by either " is a ...", " - ...", or similar.
        We try several delimiters and fall back to the whole string if needed.
        """
        if not isinstance(text, str):
            return ""

        s = text.strip()
        if not s:
            return ""

        # Try typical delimiters in order of preference
        candidates: List[int] = []

        # "Title - plot..."
        for sep in [" - ", " – ", " — "]:
            idx = s.find(sep)
            if idx >= 3:
                candidates.append(idx)

        # "Title (YEAR) is a ..."
        idx_is_a = s.lower().find(" is a ")
        if 0 < idx_is_a < 80:
            candidates.append(idx_is_a)

        # First newline, if any
        idx_nl = s.find("\n")
        if 0 < idx_nl < 80:
            candidates.append(idx_nl)

        # Opening parenthesis can also mark the end of the bare title
        idx_paren = s.find(" (")
        if 0 < idx_paren < 80:
            candidates.append(idx_paren)

        if candidates:
            cut = min(candidates)
            title = s[:cut]
        else:
            # As a very rough fallback, truncate long strings
            title = s if len(s) <= 120 else s[:120]

        # Strip common surrounding quotes and whitespace
        return title.strip(' "«»„“”\'\t\r\n')

    @staticmethod
    def _normalise_title(text: str) -> str:
        """
        Normalise a title for fuzzy matching:
            - lowercase
            - remove most punctuation
            - collapse whitespace
        """
        if not isinstance(text, str):
            return ""

        s = text.lower()
        # Remove punctuation except word characters and whitespace
        s = re.sub(r"[^\w\s]", " ", s, flags=re.UNICODE)
        # Collapse multiple spaces
        s = re.sub(r"\s+", " ", s, flags=re.UNICODE)
        return s.strip()

    def _attach_plots_to_films(self, film_df: pd.DataFrame) -> pd.DataFrame:
        """
        Merge movie_plots_df into film_df on the 'uri' column in order to
        attach descriptions (and derived titles) to film rows.
        """
        movie_plots_df = self.movie_plots_df[["uri", "description"]].copy()

        merged = film_df.merge(
            movie_plots_df,
            on="uri",
            how="left",
        )

        # Derive a heuristic title from the description if available
        if "description" in merged.columns:
            merged["title"] = merged["description"].apply(self._guess_title_from_description)
        else:
            merged["title"] = ""

        return merged

    def _build_uri_to_description(self) -> Dict[str, str]:
        """
        Build a mapping from film URI to description.

        Priority:
            1. description column on film_df (after merge)
            2. movie_plots_df (fallback, though normally 1 already covers it)
        """
        desc_map: Dict[str, str] = {}

        # From film_df (preferred)
        if "description" in self.film_df.columns:
            for uri, desc in zip(self.film_df["uri"], self.film_df["description"]):
                if isinstance(uri, str) and isinstance(desc, str) and uri not in desc_map:
                    desc_map[uri] = desc

        # Fallback to raw movie_plots_df if any URIs are missing
        for uri, desc in zip(
            self.movie_plots_df["uri"], self.movie_plots_df["description"]
        ):
            if isinstance(uri, str) and isinstance(desc, str) and uri not in desc_map:
                desc_map[uri] = desc

        return desc_map

    def _build_uri_to_title(self) -> Dict[str, str]:
        """
        Build a mapping from URI to heuristic movie title.
        """
        title_map: Dict[str, str] = {}

        if "title" in self.film_df.columns:
            for uri, title in zip(self.film_df["uri"], self.film_df["title"]):
                if not isinstance(uri, str):
                    continue
                if isinstance(title, str) and title.strip():
                    title_map[uri] = title.strip()
                else:
                    # Fallback to description if title is missing
                    desc = self.uri_to_description.get(uri)
                    if isinstance(desc, str):
                        title_map[uri] = self._guess_title_from_description(desc)

        # Ensure that all URIs with descriptions also have a title
        for uri, desc in self.uri_to_description.items():
            if uri not in title_map and isinstance(desc, str):
                title_map[uri] = self._guess_title_from_description(desc)

        return title_map

    def _build_title_to_uris(self) -> Dict[str, List[str]]:
        """
        Build an inverted index from normalised title -> list of URIs.
        """
        mapping: Dict[str, List[str]] = {}

        for uri, title in self.uri_to_title.items():
            norm = self._normalise_title(title)
            if not norm:
                continue
            mapping.setdefault(norm, []).append(uri)

        return mapping

    def _ensure_normalized(self) -> None:
        """
        Lazily compute L2-normalized embeddings for all entities and films.

        Normalization is done in-place for efficiency but guarded so it
        is only executed once.
        """
        if self._entity_embeds_normalized is None:
            emb = self.entity_embeds.astype(np.float32, copy=False)
            norms = np.linalg.norm(emb, axis=1, keepdims=True)
            norms[norms == 0.0] = 1.0
            self._entity_embeds_normalized = emb / norms

        if self._film_embeds_normalized is None:
            film_emb = self._entity_embeds_normalized[self.film_indices]
            self._film_embeds_normalized = film_emb

    # ------------------------------------------------------------------
    # Public API – basic lookups
    # ------------------------------------------------------------------

    def has_uri(self, uri: str) -> bool:
        """
        Return True if the given URI is present in the embedding index.
        """
        return uri in self.uri_to_idx

    def get_index_for_uri(self, uri: str) -> Optional[int]:
        """
        Return the embedding row index for a given URI, or None if unknown.
        """
        return self.uri_to_idx.get(uri)

    def get_uri_for_index(self, idx: int) -> Optional[str]:
        """
        Return the URI for a given embedding row index, or None if unknown.
        """
        return self.idx_to_uri.get(idx)

    def get_description(self, uri: str) -> Optional[str]:
        """
        Return the description for a film URI, or None if we have none.
        """
        return self.uri_to_description.get(uri)

    def get_title(self, uri: str) -> Optional[str]:
        """
        Return the heuristic title for a film URI, or None if unknown.
        """
        return self.uri_to_title.get(uri)

    # ------------------------------------------------------------------
    # Public API – title based search
    # ------------------------------------------------------------------

    def find_films_by_title(self, raw_title: str) -> List[str]:
        """
        Resolve a (possibly noisy) movie title string to one or more URIs.

        Strategy:
            1. Normalise the given title and look for an exact match in the
               title_to_uris index.
            2. If nothing is found, perform a very small fuzzy search where
               we look for titles that contain the query (or vice versa).

        Parameters
        ----------
        raw_title:
            Title string as it appears in the user's question.

        Returns
        -------
        List[str]
            A list of URIs (may be empty).
        """
        norm = self._normalise_title(raw_title)
        if not norm:
            return []

        uris = list(self.title_to_uris.get(norm, []))
        if uris:
            return uris

        # Very lightweight fuzzy fallback: scan keys once.
        # This is acceptable because the evaluation only asks a small number of questions.
        fallback_matches: List[str] = []
        for key, key_uris in self.title_to_uris.items():
            if norm in key or key in norm:
                fallback_matches.extend(key_uris)

        # Deduplicate while preserving order
        seen = set()
        unique_matches: List[str] = []
        for uri in fallback_matches:
            if uri not in seen:
                seen.add(uri)
                unique_matches.append(uri)

        return unique_matches

    # ------------------------------------------------------------------
    # Public API – vector access & neighbours
    # ------------------------------------------------------------------

    def get_film_vector(self, uri: str) -> np.ndarray:
        """
        Return the normalized embedding vector for a given film URI.

        Raises
        ------
        KeyError
            If the URI is not present or not a film.
        """
        if uri not in self.uri_to_idx:
            raise KeyError(f"URI {uri!r} is not present in the embedding index.")

        idx = self.uri_to_idx[uri]
        if idx not in self.film_indices:
            raise KeyError(f"URI {uri!r} is not marked as a film.")

        self._ensure_normalized()
        return self._entity_embeds_normalized[idx]

    def neighbors_for_film(
        self,
        uri: str,
        top_k: int = 10,
        min_similarity: float = 0.0,
        include_seed: bool = False,
    ) -> List[Dict[str, object]]:
        """
        Find top-k nearest neighbour films for a given film URI.

        Parameters
        ----------
        uri:
            Seed film URI.
        top_k:
            Number of neighbours to return (excluding or including seed).
        min_similarity:
            Optional lower bound on cosine similarity for returned neighbours.
        include_seed:
            If True, the seed film can appear in the result list.
            If False, it will be explicitly removed.

        Returns
        -------
        List[Dict[str, object]]
            Each entry is a dictionary with keys:
                - 'uri'
                - 'index'
                - 'similarity'
                - 'title'
                - 'description'
        """
        if uri not in self.uri_to_idx:
            raise KeyError(f"Unknown URI: {uri}")

        seed_idx = self.uri_to_idx[uri]
        if seed_idx not in self.film_indices:
            raise KeyError(f"URI {uri!r} is not a film according to film_entities.tsv.")

        self._ensure_normalized()

        # Vector for the seed film
        seed_vec = self._entity_embeds_normalized[seed_idx]

        # Candidate film indices and their normalized embeddings
        cand_indices = self.film_indices
        cand_vectors = self._film_embeds_normalized

        # Cosine similarities via dot product (embeddings are normalized)
        sims = cand_vectors @ seed_vec

        # Optionally exclude the seed film itself
        if not include_seed:
            # Identify position of the seed in cand_indices, if present
            seed_pos = np.where(cand_indices == seed_idx)[0]
            if seed_pos.size > 0:
                sims = sims.copy()
                sims[seed_pos[0]] = -1.0  # ensure it will not be selected

        if len(cand_indices) == 0:
            return []

        effective_k = min(top_k, len(cand_indices))
        if effective_k <= 0:
            return []

        # Use argpartition for efficiency, then sort the top candidates
        top_idx = np.argpartition(-sims, effective_k - 1)[:effective_k]
        # Sort them by similarity descending
        top_idx = top_idx[np.argsort(-sims[top_idx])]

        results: List[Dict[str, object]] = []
        for local_idx in top_idx:
            sim = float(sims[local_idx])
            if sim < min_similarity:
                continue

            global_idx = int(cand_indices[local_idx])
            cand_uri = self.idx_to_uri.get(global_idx, "")
            title = self.uri_to_title.get(cand_uri)
            description = self.uri_to_description.get(cand_uri)

            results.append(
                {
                    "uri": cand_uri,
                    "index": global_idx,
                    "similarity": sim,
                    "title": title,
                    "description": description,
                }
            )

        return results

    def sample_random_film(self, rng: Optional[np.random.Generator] = None) -> Dict[str, object]:
        """
        Pick a random film from the index and return its basic info.

        This is mostly useful for debugging or demo purposes.
        """
        if rng is None:
            rng = np.random.default_rng()

        if len(self.film_indices) == 0:
            raise RuntimeError("No films available in the embedding index.")

        idx = int(rng.choice(self.film_indices))
        uri = self.idx_to_uri.get(idx, "")
        desc = self.uri_to_description.get(uri)
        title = self.uri_to_title.get(uri)

        return {
            "uri": uri,
            "index": idx,
            "title": title,
            "description": desc,
        }


# =====================================================================
# 4. Global singleton accessor
# =====================================================================


# We use a simple module-level singleton to avoid repeatedly loading the
# embedding files on each request.
_GLOBAL_INDEX: Optional[MovieEmbeddingIndex] = None


def get_global_movie_index(config: Optional[EmbeddingConfig] = None) -> MovieEmbeddingIndex:
    """
    Return a process-wide singleton MovieEmbeddingIndex.

    The first call will construct the index using the provided config (or the
    default config if none is given). Subsequent calls ignore the config
    and always return the already constructed index.

    This is safe for typical single-process deployments; if you use
    multiprocessing or multi-process servers, each process will build its
    own copy on first use.
    """
    global _GLOBAL_INDEX
    if _GLOBAL_INDEX is None:
        _GLOBAL_INDEX = MovieEmbeddingIndex(config=config)
    return _GLOBAL_INDEX
