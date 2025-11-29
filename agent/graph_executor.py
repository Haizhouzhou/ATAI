# agent/graph_executor.py
from __future__ import annotations

import csv
import json
import re
from collections import defaultdict
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Set, Tuple

import numpy as np  # kept for future extensions / compatibility
import pandas as pd

from .logging_config import get_logger
from . import constants as C
from .constants import RelationDef  # for generic one-hop queries
from .utils import normalize_title

logger = get_logger()


@dataclass
class EntityInfo:
    idx: int
    uri: str
    label: Optional[str] = None


class GraphExecutor:
    """
    Lightweight in-memory view over the knowledge graph and metadata.

    Responsibilities:
    - Load entity / relation id mappings.
    - Load film subset.
    - Load id-based triples and build adjacency lists:
        * spo[h][r] -> [t...]
        * pos[r][t] -> [h...]
    - Build specialised mappings for one-hop & recommendation queries:
        * directors_by_film, screenwriters_by_film, composers_by_film,
          genres_by_film, countries_by_film, languages_by_film,
          award_nominations_by_film.
    - Keep a set of person candidates (for people/actor queries).
    - Load release dates from the original graph (P577).
    - Load labels / titles from entity_titles.tsv + movie_plots.tsv.
    - Build a title index for fast lookup by name.
    - Load image ids for multimedia answers from images.json.
    """

    # ------------------------------------------------------------------
    # Constructor
    # ------------------------------------------------------------------

    def __init__(self) -> None:
        logger.info("Initializing GraphExecutor...")

        # --- entity & relation mappings ---
        self.entity_idx_to_uri: List[str] = []
        self.entity_uri_to_idx: Dict[str, int] = {}

        self.relation_idx_to_uri: List[str] = []
        self.relation_uri_to_idx: Dict[str, int] = {}

        self._load_entity_ids()
        self._load_relation_ids()

        # relation index shortcuts
        self.rel_idx: Dict[str, Optional[int]] = {}
        self._init_relation_shortcuts()

        # --- film subset ---
        self.film_idxs: Set[int] = set()
        self._load_film_entities()

        # --- adjacency ---
        self.spo: Dict[int, Dict[int, List[int]]] = defaultdict(
            lambda: defaultdict(list)
        )
        self.pos: Dict[int, Dict[int, List[int]]] = defaultdict(
            lambda: defaultdict(list)
        )

        # specialised mappings
        self.directors_by_film: Dict[int, List[int]] = defaultdict(list)
        self.screenwriters_by_film: Dict[int, List[int]] = defaultdict(list)
        self.composers_by_film: Dict[int, List[int]] = defaultdict(list)
        self.genres_by_film: Dict[int, List[int]] = defaultdict(list)
        self.countries_by_film: Dict[int, List[int]] = defaultdict(list)
        self.languages_by_film: Dict[int, List[int]] = defaultdict(list)
        self.award_nominations_by_film: Dict[int, List[int]] = defaultdict(list)
        # optional extra mapping: cast members per film
        self.cast_by_film: Dict[int, List[int]] = defaultdict(list)

        # people candidates
        self.person_candidate_idxs: Set[int] = set()

        # literal release dates
        self.release_dates_by_entity: Dict[int, List[str]] = defaultdict(list)

        # 1) load triples (id-based), build adjacency + specialised mappings
        self._load_id_triples()

        # 2) optional: scan original graph for P577 literal dates
        self._scan_graph_literals_for_release_dates()

        # --- labels / titles ---
        self.labels_by_entity: Dict[int, str] = {}

        self._merge_titles_from_entity_titles()
        self._merge_titles_from_movie_plots()
        self._merge_titles_from_additional_plots()

        # title index (legacy normalisation via utils.normalize_title)
        self.title_to_entity_idxs: Dict[str, List[int]] = defaultdict(list)
        self._build_title_index()

        # --- images ---
        self.images_by_entity: Dict[int, List[str]] = defaultdict(list)
        self._load_images()

        # --- person fallback ---
        # If we failed to identify person candidates from triples, fall back
        # to "all labelled entities" so people/actor queries still work.
        if not self.person_candidate_idxs:
            self.person_candidate_idxs = set(self.labels_by_entity.keys())
            logger.info(
                "Person candidates fallback: using %d labelled entities as candidates",
                len(self.person_candidate_idxs),
            )

        logger.info("GraphExecutor initialized.")

    # ------------------------------------------------------------------
    # Loading helpers
    # ------------------------------------------------------------------

    def _load_entity_ids(self) -> None:
        """
        Load entity_ids_ordered.tsv and populate:
          - entity_idx_to_uri
          - entity_uri_to_idx

        Expected columns: 'index', 'uri'

        We are robust to accidental header-like rows written as data:
            index  uri
            0      http://...
        """
        path: Path = C.ENTITY_IDS_TSV
        if not path.exists():
            raise FileNotFoundError(f"Missing entity_ids file: {path}")

        # Read as strings to safely filter out non-numeric indices
        df = pd.read_csv(path, sep="\t", dtype=str, low_memory=False)
        cols_lower = {c.lower(): c for c in df.columns}

        idx_col = cols_lower.get("index") or cols_lower.get("idx")
        uri_col = cols_lower.get("uri") or cols_lower.get("entity") or cols_lower.get(
            "entity_uri"
        )

        if idx_col is None or uri_col is None:
            raise ValueError(
                f"Unexpected columns in entity_ids_ordered.tsv: {df.columns.tolist()}"
            )

        # Drop rows where index/uri are literal header strings
        bad_mask = (df[idx_col] == "index") | (df[uri_col] == "uri")
        df = df[~bad_mask].copy()

        # Coerce indices to integer, drop bad rows
        df[idx_col] = pd.to_numeric(df[idx_col], errors="coerce")
        df = df.dropna(subset=[idx_col])
        df[idx_col] = df[idx_col].astype(int)

        if df.empty:
            raise ValueError("entity_ids_ordered.tsv has no valid rows after cleaning.")

        max_idx = int(df[idx_col].max())
        self.entity_idx_to_uri = [""] * (max_idx + 1)
        self.entity_uri_to_idx.clear()

        assigned = 0
        for _, row in df.iterrows():
            idx = int(row[idx_col])
            uri = str(row[uri_col])
            if idx >= len(self.entity_idx_to_uri):
                # Expand if indices are not contiguous
                self.entity_idx_to_uri.extend([""] * (idx + 1 - len(self.entity_idx_to_uri)))
            self.entity_idx_to_uri[idx] = uri
            self.entity_uri_to_idx[uri] = idx
            assigned += 1

        logger.info(
            "Loaded entity_ids_ordered.tsv: %d valid rows, max_idx=%d",
            assigned,
            max_idx,
        )

    def _load_relation_ids(self) -> None:
        """
        Load relation_ids_ordered.tsv and populate:
          - relation_idx_to_uri
          - relation_uri_to_idx

        Expected columns:
            'index', 'relation_uri'

        We also support a headerless variant.
        """
        path: Path = C.RELATION_IDS_TSV
        if not path.exists():
            raise FileNotFoundError(f"Missing relation_ids file: {path}")

        df = pd.read_csv(path, sep="\t")
        cols_lower = {c.lower(): c for c in df.columns}

        idx_col = cols_lower.get("index") or cols_lower.get("idx")
        uri_col = (
            cols_lower.get("relation_uri")
            or cols_lower.get("uri")
            or cols_lower.get("property")
        )

        if idx_col is None or uri_col is None:
            # fallback to headerless
            df = pd.read_csv(path, sep="\t", header=None, names=["index", "relation_uri"])
            idx_col = "index"
            uri_col = "relation_uri"

        valid_rows: List[Tuple[int, str]] = []
        for _, row in df.iterrows():
            try:
                idx = int(row[idx_col])
            except (TypeError, ValueError):
                # skip header or malformed rows
                continue
            uri = str(row[uri_col])
            valid_rows.append((idx, uri))

        if not valid_rows:
            raise ValueError(f"No valid rows found in relation_ids file: {path}")

        max_idx = max(i for i, _ in valid_rows)
        self.relation_idx_to_uri = [""] * (max_idx + 1)
        self.relation_uri_to_idx.clear()

        for idx, uri in valid_rows:
            if idx >= len(self.relation_idx_to_uri):
                self.relation_idx_to_uri.extend(
                    [""] * (idx + 1 - len(self.relation_idx_to_uri))
                )
            self.relation_idx_to_uri[idx] = uri
            self.relation_uri_to_idx[uri] = idx

        logger.info(
            "Loaded relation_ids_ordered.tsv (rows=%d)", len(self.relation_idx_to_uri)
        )

    def _init_relation_shortcuts(self) -> None:
        """
        After relation_uri_to_idx is populated, resolve the specific
        Wikidata properties we care about for specialised mappings.
        """
        self.rel_idx = {
            "director": self.relation_uri_to_idx.get(C.P_DIRECTOR),
            "screenwriter": self.relation_uri_to_idx.get(C.P_SCREENWRITER),
            "composer": self.relation_uri_to_idx.get(C.P_COMPOSER),
            "genre": self.relation_uri_to_idx.get(C.P_GENRE),
            "country": self.relation_uri_to_idx.get(C.P_COUNTRY_OF_ORIGIN),
            "release_date": self.relation_uri_to_idx.get(C.P_RELEASE_DATE),
            "nominated_for": self.relation_uri_to_idx.get(C.P_NOMINATED_FOR),
            "instance_of": self.relation_uri_to_idx.get(C.P_INSTANCE_OF),
            "original_language": self.relation_uri_to_idx.get(C.P_ORIGINAL_LANGUAGE),
            "cast_member": self.relation_uri_to_idx.get(C.P_CAST_MEMBER),
        }

    def _load_film_entities(self) -> None:
        """
        Load film_entities.tsv (index, uri, is_film, has_plot)
        and populate self.film_idxs.
        """
        path: Path = C.FILM_ENTITIES_TSV
        if not path.exists():
            logger.warning("film_entities.tsv not found, film subset will be empty.")
            return

        df = pd.read_csv(path, sep="\t", dtype=str, low_memory=False)
        cols_lower = {c.lower(): c for c in df.columns}
        idx_col = cols_lower.get("index") or cols_lower.get("idx")
        film_col = cols_lower.get("is_film")

        if idx_col is None or film_col is None:
            logger.warning(
                "film_entities.tsv missing expected columns (index / is_film); columns=%s",
                df.columns.tolist(),
            )
            return

        df[idx_col] = pd.to_numeric(df[idx_col], errors="coerce")
        df[film_col] = pd.to_numeric(df[film_col], errors="coerce")
        df = df.dropna(subset=[idx_col, film_col])

        count = 0
        for _, row in df.iterrows():
            try:
                if int(row[film_col]) == 1:
                    self.film_idxs.add(int(row[idx_col]))
                    count += 1
            except (TypeError, ValueError):
                continue

        logger.info("Film subset size: %d", len(self.film_idxs))

    def _load_id_triples(self) -> None:
        """
        Load rfc_triples_ids.tsv and build adjacency structures.

        The file is id-based: each row is h<TAB>r<TAB>t with possible header.
        We'll read as strings, filter out non-numeric rows, then cast to int.
        """
        path: Path = C.ID_TRIPLES_TSV
        if not path.exists():
            raise FileNotFoundError(f"Missing id triples file: {path}")

        logger.info("Loading id triples from %s", path)
        df = pd.read_csv(path, sep="\t", dtype=str, header=None, low_memory=False)
        triples: List[Tuple[int, int, int]] = []
        non_numeric = 0

        for _, row in df.iterrows():
            if len(row) < 3:
                continue
            h_raw, r_raw, t_raw = str(row[0]), str(row[1]), str(row[2])
            try:
                h = int(h_raw)
                r = int(r_raw)
                t = int(t_raw)
            except ValueError:
                non_numeric += 1
                continue
            triples.append((h, r, t))

        if non_numeric > 0:
            logger.info(
                "Filtered out %d non-numeric rows from triples file (likely header).",
                non_numeric,
            )

        # ready relation indices
        rid_director = self.rel_idx.get("director")
        rid_screenwriter = self.rel_idx.get("screenwriter")
        rid_composer = self.rel_idx.get("composer")
        rid_genre = self.rel_idx.get("genre")
        rid_country = self.rel_idx.get("country")
        rid_language = self.rel_idx.get("original_language")
        rid_award = self.rel_idx.get("nominated_for")
        rid_cast = self.rel_idx.get("cast_member")

        for h, r, t in triples:
            # adjacency
            self.spo[h][r].append(t)
            self.pos[r][t].append(h)

            # specialised mappings (only for film subset on the head side)
            if h in self.film_idxs:
                if r == rid_director:
                    self.directors_by_film[h].append(t)
                    self.person_candidate_idxs.add(t)
                elif r == rid_screenwriter:
                    self.screenwriters_by_film[h].append(t)
                    self.person_candidate_idxs.add(t)
                elif r == rid_composer:
                    self.composers_by_film[h].append(t)
                    self.person_candidate_idxs.add(t)
                elif r == rid_genre:
                    self.genres_by_film[h].append(t)
                elif r == rid_country:
                    self.countries_by_film[h].append(t)
                elif r == rid_language:
                    self.languages_by_film[h].append(t)
                elif r == rid_award:
                    self.award_nominations_by_film[h].append(t)
                elif r == rid_cast:
                    self.cast_by_film[h].append(t)
                    # cast members are also typical persons
                    self.person_candidate_idxs.add(t)

        logger.info("Triples loaded: %d", len(triples))

    # ------------------------------------------------------------------
    # Literal dates (P577) from graph.tsv
    # ------------------------------------------------------------------

    def _scan_graph_literals_for_release_dates(self) -> None:
        """
        Single pass over dataset graph.tsv to extract literal publication dates.

        We don't assume an exact URI for P577; instead we treat every triple
        whose predicate contains "P577" as a candidate and try to parse the
        object as a literal.
        """
        path: Path = C.GRAPH_TSV_PATH
        if not path.exists():
            logger.warning("graph.tsv not found at %s, cannot load literal dates.", path)
            return

        count = 0
        try:
            with path.open("r", encoding="utf-8") as f:
                tsv_reader = csv.reader(f, delimiter="\t")
                for row in tsv_reader:
                    if len(row) < 3:
                        continue
                    s, p, o = row[0], row[1], row[2]
                    if "P577" not in p:
                        continue
                    literal = self._extract_literal(o)
                    if not literal:
                        continue
                    idx = self.entity_uri_to_idx.get(s)
                    if idx is None:
                        continue
                    self.release_dates_by_entity[idx].append(literal)
                    count += 1
        except Exception as e:
            logger.warning("Error while scanning graph.tsv for P577: %s", e)
            return

        logger.info("Loaded %d publication dates (P577-like) from graph.tsv", count)

    # ------------------------------------------------------------------
    # Labels / titles
    # ------------------------------------------------------------------

    def _merge_titles_from_entity_titles(self) -> None:
        """
        Use metadata/entity_titles.tsv as the primary source of human-readable
        labels. The file is part of the Final Submission metadata and usually
        contains at least:

            index (or idx),  title / label / name  [optionally uri/qid]
        """
        path: Path = C.ENTITY_TITLES_TSV
        if not path.exists():
            logger.warning("entity_titles.tsv not found, skipping title merge.")
            return

        df = pd.read_csv(path, sep="\t")
        cols_lower = {c.lower(): c for c in df.columns}

        idx_col = cols_lower.get("index") or cols_lower.get("idx")
        uri_col = None
        for cand in ("uri", "entity", "entity_uri"):
            if cand in cols_lower:
                uri_col = cols_lower[cand]
                break

        label_col = None
        for cand in ("title", "label", "name"):
            if cand in cols_lower:
                label_col = cols_lower[cand]
                break

        # fallback: take first non-idx/uri column as label
        if label_col is None:
            for c in df.columns:
                if c not in {idx_col, uri_col}:
                    label_col = c
                    break

        if label_col is None:
            logger.warning(
                "entity_titles.tsv: could not identify label column; columns=%s",
                df.columns.tolist(),
            )
            return

        merged = 0
        for _, row in df.iterrows():
            label = str(row[label_col]).strip()
            if not label:
                continue

            idx: Optional[int] = None
            if idx_col is not None:
                try:
                    idx = int(row[idx_col])
                except (TypeError, ValueError):
                    idx = None
            elif uri_col is not None:
                uri = str(row[uri_col])
                idx = self.entity_uri_to_idx.get(uri)

            if idx is None:
                continue

            if idx not in self.labels_by_entity:
                self.labels_by_entity[idx] = label
                merged += 1

        logger.info(
            "Merged %d labels from entity_titles.tsv into labels_by_entity", merged
        )

    def _merge_titles_from_movie_plots(self) -> None:
        """
        Use metadata/movie_plots.tsv to enrich labels, if it contains titles.

        This is treated as a secondary source relative to entity_titles.tsv.
        """
        path: Path = C.MOVIE_PLOTS_TSV
        if not path.exists():
            logger.warning("movie_plots.tsv not found, skipping.")
            return

        try:
            df = pd.read_csv(path, sep="\t")
        except Exception:
            try:
                df = pd.read_csv(path)
            except Exception as e:
                logger.warning("Failed to load movie_plots.tsv: %s", e)
                return

        cols_lower = {c.lower(): c for c in df.columns}
        idx_col = cols_lower.get("index") or cols_lower.get("idx")
        uri_col = None
        qid_col = None

        for c in df.columns:
            lc = c.lower()
            if "uri" in lc or "entity" in lc:
                uri_col = c
            if lc in ("qid", "wikidata_id", "wikidata_qid"):
                qid_col = c

        label_col = None
        for cand in ("title", "label", "name"):
            if cand in cols_lower:
                label_col = cols_lower[cand]
                break
        if label_col is None:
            # maybe the second column is the title, if the first is index/qid
            if len(df.columns) >= 2:
                label_col = df.columns[1]

        if label_col is None:
            logger.warning(
                "movie_plots.tsv: could not identify title column; columns=%s",
                df.columns.tolist(),
            )
            return

        merged = 0
        for _, row in df.iterrows():
            label = str(row[label_col]).strip()
            if not label:
                continue

            idx: Optional[int] = None

            if idx_col is not None:
                try:
                    idx = int(row[idx_col])
                except (TypeError, ValueError):
                    idx = None

            if idx is None and uri_col is not None:
                uri = str(row[uri_col])
                idx = self.entity_uri_to_idx.get(uri)

            if idx is None and qid_col is not None:
                qid = str(row[qid_col]).strip()
                if qid:
                    uri = C.WD_ENTITY + qid
                    idx = self.entity_uri_to_idx.get(uri)

            if idx is None:
                continue

            if idx not in self.labels_by_entity:
                self.labels_by_entity[idx] = label
                merged += 1

        logger.info(
            "Merged %d additional titles from movie_plots.tsv into labels_by_entity",
            merged,
        )

    def _merge_titles_from_additional_plots(self) -> None:
        """
        Optional: use dataset/additional/plots.csv to enrich labels if it
        actually contains title-like information.

        In your dataset, plots.csv currently has only 'qid' and 'plot',
        so this will almost certainly be a no-op, but we keep it robust.
        """
        path: Path = C.PLOTS_CSV_PATH
        if not path.exists():
            return

        try:
            df = pd.read_csv(path)
        except Exception as e:
            logger.warning("Failed to load additional/plots.csv: %s", e)
            return

        cols_lower = {c.lower(): c for c in df.columns}
        qid_col = cols_lower.get("qid")
        label_col = None

        for cand in ("title", "label", "name"):
            if cand in cols_lower:
                label_col = cols_lower[cand]
                break

        if qid_col is None or label_col is None:
            # e.g. columns=['qid', 'plot']; skip
            logger.debug(
                "additional/plots.csv has no usable title columns; columns=%s",
                df.columns.tolist(),
            )
            return

        merged = 0
        for _, row in df.iterrows():
            label = str(row[label_col]).strip()
            if not label:
                continue
            qid = str(row[qid_col]).strip()
            if not qid:
                continue
            uri = C.WD_ENTITY + qid
            idx = self.entity_uri_to_idx.get(uri)
            if idx is None:
                continue
            if idx not in self.labels_by_entity:
                self.labels_by_entity[idx] = label
                merged += 1

        if merged > 0:
            logger.info(
                "Merged %d titles from additional/plots.csv into labels_by_entity",
                merged,
            )

    def _build_title_index(self) -> None:
        """
        Build an index from normalized title -> list[entity_idx].

        This uses utils.normalize_title (legacy, strict) as the primary
        index key. For queries we first consult this map; if that fails,
        we fall back to a more robust fuzzy matcher.
        """
        for idx, label in self.labels_by_entity.items():
            try:
                norm = normalize_title(label)
            except Exception:
                norm = None
            if not norm:
                continue
            self.title_to_entity_idxs[norm].append(idx)

        logger.info(
            "Title index built with %d distinct normalized titles",
            len(self.title_to_entity_idxs),
        )

    # ------------------------------------------------------------------
    # Images
    # ------------------------------------------------------------------

    def _load_images(self) -> None:
        """
        Parse additional/images.json in a robust way.

        Course note: images.json “already only stores subfolder/imageID
        (many without .jpg)”. Therefore we cannot assume the value ends
        with '.jpg'. We treat as image ids:

            - any field named like '*image*' or '*poster*' which looks
              like '0000/abcdef...'
            - any string that looks like '0000/abcdef...' even if the
              key name is something else

        Entity association is done via:

            - uri starting with C.WD_ENTITY
            - qid like "Q12345"
            - fuzzy matching on labels if uri/qid are missing
        """
        path: Path = C.IMAGES_JSON_PATH
        if not path.exists():
            logger.warning("images.json not found, multimedia answers will be limited.")
            return

        try:
            with path.open("r", encoding="utf-8") as f:
                data = json.load(f)
        except Exception as e:
            logger.warning("Failed to load images.json: %s", e)
            return

        labels_count = 0
        img_count = 0

        def looks_like_image_id(v: str) -> bool:
            # typical format: '0000/1HJA8MiuTYGf40u7x7Bgw3Yv7py' or with .jpg
            if "/" in v and not v.startswith("http"):
                return True
            if v.lower().endswith(".jpg"):
                return True
            return False

        def normalize_image_id(v: str) -> str:
            v = v.strip()
            # Strip URL base if present, keep only last two segments '0000/xxx'
            if "://" in v:
                v = v.split("://", 1)[1]
                v = v.split("/", 1)[1] if "/" in v else v
            if v.lower().endswith(".jpg"):
                v = v[:-4]
            parts = v.split("/")
            if len(parts) >= 2:
                return "/".join(parts[-2:])
            return v

        def handle_record(rec: dict) -> None:
            nonlocal labels_count, img_count
            if not isinstance(rec, dict):
                return

            uri: Optional[str] = None
            qid: Optional[str] = None
            label: Optional[str] = None
            image_id: Optional[str] = None

            for k, v in rec.items():
                if not isinstance(v, str):
                    continue
                kl = k.lower()
                vs = v.strip()

                # entity identifier
                if vs.startswith(C.WD_ENTITY):
                    uri = vs
                elif vs.startswith("Q") and vs[1:].isdigit():
                    qid = vs

                # image id
                if "image" in kl or "poster" in kl:
                    if looks_like_image_id(vs):
                        image_id = normalize_image_id(vs)
                        continue
                if looks_like_image_id(vs) and image_id is None:
                    image_id = normalize_image_id(vs)
                    continue

                # label-ish field
                if kl in ("label", "title", "name"):
                    label = vs

            if uri is None and qid is not None:
                uri = C.WD_ENTITY + qid

            idx: Optional[int] = None
            if uri is not None:
                idx = self.entity_uri_to_idx.get(uri)

            # If we don't have a uri/qid mapping, try fuzzy label match
            if idx is None and label:
                candidates = self._find_by_label_fuzzy(label)
                if candidates:
                    idx = candidates[0]

            if idx is None or image_id is None:
                return

            self.images_by_entity[idx].append(image_id)
            img_count += 1

            # If label is present and not yet stored, also record it
            if label and idx not in self.labels_by_entity:
                self.labels_by_entity[idx] = label
                labels_count += 1

        # Handle several possible JSON shapes
        if isinstance(data, dict):
            if all(isinstance(v, str) for v in data.values()):
                # shape: { "Qxxxx": "0000/abc" }
                for key, v in data.items():
                    rec = {"qid": key, "image": v}
                    handle_record(rec)
            else:
                maybe_list = None
                if "items" in data and isinstance(data["items"], list):
                    maybe_list = data["items"]
                else:
                    maybe_list = [v for v in data.values() if isinstance(v, dict)]
                for rec in maybe_list or []:
                    handle_record(rec)
        elif isinstance(data, list):
            for rec in data:
                if isinstance(rec, dict):
                    handle_record(rec)

        logger.info(
            "Loaded %d extra labels and %d image ids from images.json",
            labels_count,
            img_count,
        )

    # ------------------------------------------------------------------
    # Public query helpers (index-level)
    # ------------------------------------------------------------------

    def get_label(self, idx: int) -> str:
        """
        Best-effort human-readable label for an entity index.
        """
        if idx in self.labels_by_entity:
            return self.labels_by_entity[idx]
        if 0 <= idx < len(self.entity_idx_to_uri):
            uri = self.entity_idx_to_uri[idx]
            return uri.rsplit("/", 1)[-1]
        return str(idx)

    def has_uri(self, uri: str) -> bool:
        """
        Return True if this URI exists in the entity index.
        """
        return uri in self.entity_uri_to_idx

    def get_index_for_uri(self, uri: str) -> Optional[int]:
        """
        Return the integer entity index for a given URI (or None).
        """
        return self.entity_uri_to_idx.get(uri)

    def get_uri_for_index(self, idx: int) -> Optional[str]:
        """
        Return the URI for a given entity index (or None).
        """
        if 0 <= idx < len(self.entity_idx_to_uri):
            return self.entity_idx_to_uri[idx]
        return None

    def get_label_for_uri(self, uri: str) -> Optional[str]:
        """
        Convenience: directly fetch label from URI.
        """
        idx = self.entity_uri_to_idx.get(uri)
        if idx is None:
            return None
        return self.get_label(idx)

    def get_out_neighbors(self, h_idx: int, rel_uri: str) -> List[int]:
        """
        Get all objects t such that (h_idx, rel_uri, t) exists.
        """
        r_idx = self.relation_uri_to_idx.get(rel_uri)
        if r_idx is None:
            return []
        return self.spo.get(h_idx, {}).get(r_idx, [])

    def get_in_neighbors(self, t_idx: int, rel_uri: str) -> List[int]:
        """
        Get all subjects h such that (h, rel_uri, t_idx) exists.
        """
        r_idx = self.relation_uri_to_idx.get(rel_uri)
        if r_idx is None:
            return []
        return self.pos.get(r_idx, {}).get(t_idx, [])

    def get_release_dates(self, h_idx: int) -> List[str]:
        return self.release_dates_by_entity.get(h_idx, [])

    def get_images(self, idx: int) -> List[str]:
        return self.images_by_entity.get(idx, [])

    # === Robust title matching logic ===

    def _normalize_for_lookup(self, s: str) -> str:
        """
        More permissive normalisation used for fuzzy matching.

        Differences from utils.normalize_title:
            - allows both 'query in label' and 'label in query'
            - keeps behaviour symmetrical to handle extra words
              like 'show me an image of Titanic'.
        """
        if not s:
            return ""
        s = str(s)

        # unify quotes
        s = (
            s.replace("’", "'")
            .replace("‘", "'")
            .replace("“", '"')
            .replace("”", '"')
        )

        s = s.strip()

        # strip outer quotes
        if (s.startswith("'") and s.endswith("'")) or (
            s.startswith('"') and s.endswith('"')
        ):
            s = s[1:-1].strip()

        s = s.lower()

        # drop leading articles
        for pref in ("the ", "a ", "an "):
            if s.startswith(pref):
                s = s[len(pref) :]

        # remove most punctuation
        s = re.sub(r"[^\w\s]", " ", s)
        # collapse whitespace
        s = re.sub(r"\s+", " ", s).strip()
        return s

    def _find_by_label_fuzzy(self, label: str) -> List[int]:
        """
        Do an O(N) fuzzy match over all labels as a fallback.

        We normalise both query and label, then consider:
            - exact match
            - label contained in query
            - query contained in label

        This makes questions like 'Show me an image of Titanic'
        correctly match the movie 'Titanic'.
        """
        q_norm = self._normalize_for_lookup(label)
        if not q_norm:
            return []

        exact: List[int] = []
        partial: List[int] = []

        for idx, lab in self.labels_by_entity.items():
            lab_norm = self._normalize_for_lookup(lab)
            if not lab_norm:
                continue
            if lab_norm == q_norm:
                exact.append(idx)
            elif lab_norm in q_norm or q_norm in lab_norm:
                partial.append(idx)

        if exact:
            return exact
        return partial

    def find_entities_by_normalized_title(self, title: str) -> List[int]:
        """
        Main entrypoint for title-based entity lookup.

        Logic:
        1. Try the strict title_to_entity_idxs index (using
           utils.normalize_title) to remain compatible with earlier
           code and to benefit from fast exact hits.
        2. If that fails, fall back to the more permissive
           _find_by_label_fuzzy() scan.
        """
        if not title:
            return []

        # 1) legacy exact index via utils.normalize_title
        try:
            norm = normalize_title(title)
        except Exception:
            norm = None
        if norm:
            hits = self.title_to_entity_idxs.get(norm)
            if hits:
                return hits

        # 2) robust fuzzy search as fallback
        return self._find_by_label_fuzzy(title)

    def get_entity_info_list(self, idxs: Iterable[int]) -> List[EntityInfo]:
        """
        Convert a sequence of indices into a list of EntityInfo objects.
        """
        result: List[EntityInfo] = []
        for idx in idxs:
            if not (0 <= idx < len(self.entity_idx_to_uri)):
                continue
            uri = self.entity_idx_to_uri[idx]
            label = self.labels_by_entity.get(idx)
            result.append(EntityInfo(idx=idx, uri=uri, label=label))
        return result

    # ------------------------------------------------------------------
    # High-level one-hop KG API (RelationDef-aware)
    # ------------------------------------------------------------------

    def _resolve_rel_idx_from_def(self, relation: RelationDef) -> Optional[int]:
        """
        Helper: map a RelationDef (with property_uri) to an integer relation index.
        """
        if relation is None:
            return None
        return self.relation_uri_to_idx.get(relation.property_uri)

    def get_objects_for_subjects(
        self,
        subject_uris: Iterable[str],
        relation: RelationDef,
        max_results_per_subject: int = 5,
    ) -> Dict[str, List[Dict[str, str]]]:
        """
        For each subject URI and a given relation, return a list of objects.

        Parameters
        ----------
        subject_uris:
            URIs of subjects (movies, people, etc.).
        relation:
            RelationDef describing the KG property to follow.
        max_results_per_subject:
            Limit on number of objects per subject (after deduplication).

        Returns
        -------
        Dict[str, List[Dict[str, str]]]
            Mapping subject_uri -> list of object dicts:
                {
                    "uri": object_uri (may be empty string if not resolvable),
                    "label": human-readable title/label if available,
                }
        """
        rel_idx = self._resolve_rel_idx_from_def(relation)
        if rel_idx is None:
            return {}

        results: Dict[str, List[Dict[str, str]]] = {}

        for subj_uri in subject_uris:
            subj_idx = self.entity_uri_to_idx.get(subj_uri)
            if subj_idx is None:
                continue

            tails = self.spo.get(subj_idx, {}).get(rel_idx, [])
            if not tails:
                continue

            # deduplicate while preserving order
            seen: Set[int] = set()
            ordered_tail_idxs: List[int] = []
            for t_idx in tails:
                if t_idx in seen:
                    continue
                seen.add(t_idx)
                ordered_tail_idxs.append(t_idx)
                if len(ordered_tail_idxs) >= max_results_per_subject:
                    break

            objects: List[Dict[str, str]] = []
            for t_idx in ordered_tail_idxs:
                obj_uri = self.get_uri_for_index(t_idx) or ""
                label = self.get_label(t_idx)
                objects.append({"uri": obj_uri, "label": label})

            if objects:
                results[subj_uri] = objects

        return results

    def get_objects_for_subject(
        self,
        subject_uri: str,
        relation: RelationDef,
        max_results: int = 5,
    ) -> List[Dict[str, str]]:
        """
        Convenience wrapper for a single subject URI.

        Returns a list of objects (uri + label) connected by `relation`.
        """
        mapping = self.get_objects_for_subjects(
            subject_uris=[subject_uri],
            relation=relation,
            max_results_per_subject=max_results,
        )
        return mapping.get(subject_uri, [])

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _extract_literal(self, obj_raw: str) -> Optional[str]:
        """
        Extract literal value from an object field in graph.tsv.
        Example:
            "\"1974-07-19\"^^xsd:date"  -> "1974-07-19"
        """
        if not obj_raw:
            return None
        obj_raw = obj_raw.strip()
        if obj_raw.startswith('"'):
            end = obj_raw.rfind('"')
            if end > 0:
                return obj_raw[1:end]
        return obj_raw


# ======================================================================
# Global singleton accessor
# ======================================================================

_GLOBAL_GRAPH_EXECUTOR: Optional[GraphExecutor] = None


def get_global_graph_executor() -> GraphExecutor:
    """
    Return a process-wide singleton GraphExecutor.

    This mirrors the pattern used in embedding_executor.py and avoids
    re-loading all TSV files on every request.
    """
    global _GLOBAL_GRAPH_EXECUTOR
    if _GLOBAL_GRAPH_EXECUTOR is None:
        _GLOBAL_GRAPH_EXECUTOR = GraphExecutor()
    return _GLOBAL_GRAPH_EXECUTOR
