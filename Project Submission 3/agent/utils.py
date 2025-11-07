from __future__ import annotations
from pathlib import Path
from typing import Iterable, Optional, List
import re
import logging

logger = logging.getLogger(__name__)

# --- Functions from your Submission 2 (REQUIRED) ---

_normalize_quotes_re = re.compile(r"[\u2018\u2019\u201A\u201B\u2032\u2035\u0060]")
_normalize_dquotes_re = re.compile(r"[\u201C\u201D\u201E\u201F\u2033\u2036]")

ROMAN_MAP = {
    " i ": " 1 ", " ii ": " 2 ", " iii ": " 3 ", " iv ": " 4 ",
    " v ": " 5 ", " vi ": " 6 ", " vii ": " 7 ", " viii ": " 8 ",
    " ix ": " 9 ", " x ": " 10 ",
}

PUNCT_RE = re.compile(r"[\-–—·•]+")
WS_RE = re.compile(r"\s+")

def normalize_text(s: str) -> str:
    s = s.strip()
    s = _normalize_quotes_re.sub("'", s)
    s = _normalize_dquotes_re.sub('"', s)
    s = s.lower()
    s = PUNCT_RE.sub(" ", s)
    s = WS_RE.sub(" ", s)
    s = f" {s} "
    for k, v in ROMAN_MAP.items():
        s = s.replace(k, v)
    return s.strip()

def pick_first_file(root: Path, patterns: Iterable[str]) -> Optional[Path]:
    """
    (From Submission 2) Finds the first file matching a pattern.
    """
    for pat in patterns:
        try:
            matches = list(root.rglob(pat))
            if matches:
                return matches[0].resolve()
        except Exception as e:
            logger.warning(f"Error during rglob for pattern {pat} in {root}: {e}")
            continue
    return None

# --- Functions from Submission 3 (REQUIRED) ---

def find_file_in_dirs(directories: List[Path], patterns: List[str]) -> Optional[Path]:
    """
    Search through a list of directories for the first file matching any of the patterns.
    """
    for directory in directories:
        if not directory.is_dir():
            continue
        for pattern in patterns:
            try:
                found_file = next(directory.rglob(pattern))
                if found_file.is_file():
                    logger.debug(f"Found file '{found_file}' matching pattern '{pattern}' in '{directory}'")
                    return found_file.resolve()
            except StopIteration:
                continue
    logger.warning(f"Could not find any file matching patterns {patterns} in directories {directories}")
    return None

def find_files_in_dirs(directories: List[Path], patterns: List[str]) -> List[Path]:
    """
    Search through a list of directories for all files matching any of the patterns.
    """
    all_files = []
    for directory in directories:
        if not directory.is_dir():
            continue
        for pattern in patterns:
            try:
                for found_file in directory.rglob(pattern):
                    if found_file.is_file() and found_file not in all_files:
                        all_files.append(found_file.resolve())
            except StopIteration:
                continue
    
    if not all_files:
        logger.warning(f"Could not find any files matching patterns {patterns} in directories {directories}")
    else:
        logger.debug(f"Found {len(all_files)} files matching patterns {patterns}")
    return all_files