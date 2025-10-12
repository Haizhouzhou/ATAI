from pathlib import Path

# ===== Data loading targets =====
# Use the real NT file you found:
DATA_TARGETS = [
    Path("/space_mounts/atai-hs25/dataset/graph.nt"),
]

# ===== Loader constraints (caps to avoid loading everything at once) =====
MAX_FILES = 1                         # only this one file
MAX_TOTAL_BYTES = 2_000 * 1024 * 1024 # 2 GB cap (adjust if needed)
MAX_SINGLE_FILE_BYTES = 2_000 * 1024 * 1024

# Try these formats for extension-less files or when guessing fails (ordered).
CANDIDATE_FORMATS = [
    "turtle",    # .ttl, .trig (subset)
    "nt",        # N-Triples
    "xml",       # RDF/XML
    "n3",        # Notation3
    "trig",      # TriG
    "nquads",    # N-Quads
    "json-ld",   # JSON-LD (slower, keep last)
]

# Known-friendly extensions (lowercased, without dot) → rdflib format name
EXT_TO_FORMAT = {
    "ttl": "turtle",
    "nt": "nt",
    "rdf": "xml",
    "xml": "xml",
    "n3": "n3",
    "trig": "trig",
    "nq": "nquads",
    "jsonld": "json-ld",
    "json-ld": "json-ld",
}

# ===== Query execution & output =====
MAX_ROWS = 50
TIMEOUT_SEC = 8
WIDE_IRI_TRUNC = 80

# ===== Standardized messages =====
MSG_NON_SPARQL = "Please input SPARQL query."
MSG_INVALID     = "Invalid SPARQL. Please check syntax."
MSG_TIMEOUT     = "Query timed out. Please simplify or add LIMIT."
MSG_NORESULT    = "No results in the provided dataset."

# ===== Optional safety: auto LIMIT for SELECT without LIMIT =====
AUTO_LIMIT = True
DEFAULT_SELECT_LIMIT = 50