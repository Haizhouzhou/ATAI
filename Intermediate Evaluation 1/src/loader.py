import os
from pathlib import Path
from typing import Dict, Iterable, List, Tuple

from rdflib import Graph
from rdflib.util import guess_format

from .app_config import (
    # only used for type hints / constants; DATA_TARGETS is not required here
    MAX_FILES, MAX_TOTAL_BYTES, MAX_SINGLE_FILE_BYTES,
    CANDIDATE_FORMATS, EXT_TO_FORMAT,
)

def _iter_candidate_files(targets: List[Path]) -> Iterable[Path]:
    """
    Yield files from the given list of files/directories (recursively).
    Skips obvious non-files.
    """
    seen = set()
    for t in targets:
        if not t.exists():
            continue
        if t.is_file():
            if t not in seen:
                seen.add(t)
                yield t
        else:
            for root, _, files in os.walk(t):
                for fn in files:
                    p = Path(root) / fn
                    if p not in seen:
                        seen.add(p)
                        yield p

def _ext_to_format(path: Path):
    ext = path.suffix.lower().lstrip(".")
    if ext in EXT_TO_FORMAT:
        return EXT_TO_FORMAT[ext]
    return guess_format(ext) if ext else None

def _try_parse_with_formats(g: Graph, path: Path, formats: List[str]) -> Tuple[bool, str]:
    """
    Try multiple formats in order until one succeeds.
    Returns (ok, fmt_or_reason)
    """
    for fmt in formats:
        try:
            g.parse(path.as_posix(), format=fmt)
            return True, fmt
        except Exception:
            continue
    return False, f"unrecognized or unsupported serialization (tried: {formats})"

def load_graph(targets: List[Path]) -> Tuple[Graph, Dict]:
    """
    Load RDF data from files/directories with safety caps.

    Parameters
    ----------
    targets : List[Path]
        A list of files or directories to scan.

    Returns
    -------
    (graph, stats) : (rdflib.Graph, dict)
        stats = {
            "files_considered": int,
            "files_loaded": int,
            "triples": int,
            "total_bytes": int,
            "items": [
                {"path": str, "loaded": bool, "bytes": int, "format": str|None, "reason": str|None}
            ]
        }
    """
    stats = {
        "files_considered": 0,
        "files_loaded": 0,
        "triples": 0,
        "total_bytes": 0,
        "items": [],
    }

    g = Graph()
    files_loaded = 0

    for p in _iter_candidate_files(targets):
        stats["files_considered"] += 1
        it = {"path": str(p), "loaded": False, "bytes": 0, "format": None, "reason": None}

        try:
            size = p.stat().st_size
        except Exception as e:
            it["reason"] = f"stat failed: {e}"
            stats["items"].append(it)
            continue

        it["bytes"] = size

        # global cap
        if stats["total_bytes"] + size > MAX_TOTAL_BYTES:
            it["reason"] = f"skipped: total size cap {MAX_TOTAL_BYTES} reached"
            stats["items"].append(it)
            continue

        # per-file cap
        if size > MAX_SINGLE_FILE_BYTES:
            it["reason"] = f"skipped: single file exceeds {MAX_SINGLE_FILE_BYTES} bytes"
            stats["items"].append(it)
            continue

        # parse
        fmt = _ext_to_format(p)
        try:
            if fmt:
                g.parse(p.as_posix(), format=fmt)
                it["format"] = fmt
                it["loaded"] = True
            else:
                ok, info = _try_parse_with_formats(g, p, CANDIDATE_FORMATS)
                it["loaded"] = ok
                if ok:
                    it["format"] = info
                else:
                    it["reason"] = info
        except Exception as e:
            it["loaded"] = False
            it["reason"] = str(e)

        stats["items"].append(it)

        if it["loaded"]:
            files_loaded += 1
            stats["total_bytes"] += size
            if files_loaded >= MAX_FILES:
                break

    stats["files_loaded"] = files_loaded
    stats["triples"] = len(g)
    return g, stats
