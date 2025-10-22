from __future__ import annotations
from pathlib import Path
from typing import Iterable, Optional
import re

# Normalizers
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
    for pat in patterns:
        matches = list(root.rglob(pat))
        if matches:
            return matches[0]
    return None