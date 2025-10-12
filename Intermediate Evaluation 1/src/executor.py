import re
import logging
from pathlib import Path
from typing import List, Tuple, Optional, Union

from concurrent.futures import ThreadPoolExecutor, TimeoutError as FuturesTimeout
from rdflib import Graph
from rdflib.query import Result, Processor
from rdflib.plugin import get as plugin_get, PluginException
from tabulate import tabulate

from .app_config import (
    MAX_ROWS, TIMEOUT_SEC, WIDE_IRI_TRUNC,
    MSG_NON_SPARQL, MSG_INVALID, MSG_TIMEOUT, MSG_NORESULT,
)

# --------- SPARQL detection / helpers ---------
SPARQL_HEAD_RE = re.compile(
    r"^\s*(PREFIX|SELECT|ASK|CONSTRUCT|DESCRIBE)\b",
    flags=re.IGNORECASE | re.DOTALL
)

_SELECT_RE = re.compile(
    r"^\s*(?:PREFIX\s+[^>]+>\s*)*SELECT\b",
    re.IGNORECASE | re.DOTALL
)

_LIMIT_RE = re.compile(r"\bLIMIT\s+\d+\b", re.IGNORECASE)

# Default graph used when callers pass only a query string to run_query(...)
_DEFAULT_GRAPH: Optional[Graph] = None


def set_default_graph(g: Graph) -> None:
    """
    Set the module-level default Graph. Call this once at startup so that
    run_query(query, ...) can be used without explicitly passing a Graph.
    """
    global _DEFAULT_GRAPH
    _DEFAULT_GRAPH = g


def set_default_graph_from_path(path: Union[str, Path]) -> Graph:
    """
    Convenience: load a TTL/NT/etc. file into a Graph and set it as default.
    Returns the loaded Graph.
    """
    g = Graph()
    g.parse(str(path))  # rdflib infers format from the file extension
    set_default_graph(g)
    return g


def is_ready() -> bool:
    """Return True if a default Graph has been set."""
    return _DEFAULT_GRAPH is not None


def looks_like_sparql(q: str) -> bool:
    return bool(SPARQL_HEAD_RE.match(q or ""))


def _truncate(s: str, width: int) -> str:
    if s is None:
        return ""
    s = str(s)
    if len(s) <= width:
        return s
    return s[: width - 1] + "…"


def _format_select(res: Result) -> str:
    """
    Stream formatting:
      - iterate rows up to MAX_ROWS + 1
      - if 0 rows -> MSG_NORESULT
      - if > MAX_ROWS -> print first MAX_ROWS and show a truncated note
    """
    headers = [str(v) for v in getattr(res, "vars", [])] or []
    rows: List[List[str]] = []
    count = 0
    over = False

    try:
        for row in res:
            count += 1
            if count <= MAX_ROWS:
                out = []
                if headers:
                    for v in headers:
                        out.append(_truncate(row.get(v), WIDE_IRI_TRUNC))
                else:
                    out = [_truncate(cell, WIDE_IRI_TRUNC) for cell in row]
                rows.append(out)
            elif count == MAX_ROWS + 1:
                over = True
                break
    except Exception:
        return MSG_INVALID

    if count == 0:
        return MSG_NORESULT

    if not headers and rows:
        headers = [f"col{i+1}" for i in range(len(rows[0]))]

    table = tabulate(rows, headers=headers, tablefmt="github")
    if over:
        table += f"\n\n[truncated to first {MAX_ROWS} rows] Use LIMIT/OFFSET for more."
    return table


def _format_ask(res: Result) -> str:
    return "true" if bool(res.askAnswer) else "false"


def _format_graph(g: Graph) -> str:
    if len(g) == 0:
        return MSG_NORESULT
    ttl = g.serialize(format="turtle")
    return ttl.decode("utf-8") if isinstance(ttl, bytes) else ttl


def _ensure_limit_if_needed(q: str, *, auto_limit: bool, default_limit: int) -> str:
    """
    If auto_limit is ON and q looks like a SELECT without an explicit LIMIT,
    append 'LIMIT <default_limit>'.
    """
    if not auto_limit:
        return q
    if not _SELECT_RE.match(q or ""):
        return q
    if _LIMIT_RE.search(q):
        return q
    return f"{q.rstrip()}\nLIMIT {int(default_limit)}"


def _query_with_timeout(g: Graph, q: str, seconds: int):
    with ThreadPoolExecutor(max_workers=1) as ex:
        fut = ex.submit(g.query, q)
        return fut.result(timeout=seconds)


def _pick_graph_and_query(
    g_or_query: Union[Graph, str],
    query: Optional[str]
) -> Tuple[Graph, str]:
    """
    Support two calling styles:
      - run_query(graph, query, ...)
      - run_query(query, ...)  # requires a default graph to be set
    """
    if isinstance(g_or_query, Graph):
        if query is None:
            raise TypeError("run_query() missing required argument: 'query'")
        return g_or_query, query
    else:
        if _DEFAULT_GRAPH is None:
            raise TypeError(
                "run_query(query, ...) requires set_default_graph() to be called first"
            )
        return _DEFAULT_GRAPH, str(g_or_query)


def run_query(
    g_or_query: Union[Graph, str],
    query: Optional[str] = None,
    *,
    auto_limit: bool = True,
    default_limit: int = 100
) -> Tuple[str, str]:
    """
    Execute SPARQL with standardized outcomes.

    Calling styles:
      - run_query(graph, query, ...)
      - run_query(query, ...)         # needs set_default_graph(...)

    Returns a tuple (kind, text) where kind is one of:
      {"non-sparql", "invalid", "timeout", "select", "ask", "graph"}

    * "non-sparql": input doesn't look like a SPARQL query
    * "invalid"   : plugin/runtime error
    * "timeout"   : execution exceeded TIMEOUT_SEC
    * "select"    : tabulated SELECT result (GitHub table format)
    * "ask"       : "true" or "false"
    * "graph"     : a Turtle serialization of the resulting Graph
    """
    graph, q = _pick_graph_and_query(g_or_query, query)

    if not looks_like_sparql(q):
        return "non-sparql", MSG_NON_SPARQL

    try:
        # Ensure SPARQL plugin is registered (rdflib[sparql] installed)
        plugin_get('sparql', Processor)
    except Exception:
        logging.exception("RDFlib SPARQL plugin not available. Please install rdflib[sparql].")
        return "invalid", MSG_INVALID

    # Apply optional LIMIT decoration for plain SELECT queries
    q_exec = _ensure_limit_if_needed(q, auto_limit=auto_limit, default_limit=default_limit).strip()

    try:
        res = _query_with_timeout(graph, q_exec, TIMEOUT_SEC)
    except FuturesTimeout:
        return "timeout", MSG_TIMEOUT
    except PluginException:
        logging.exception("SPARQL plugin lookup failed at runtime.")
        return "invalid", MSG_INVALID
    except Exception:
        logging.exception("SPARQL execution failed")
        return "invalid", MSG_INVALID

    if isinstance(res, Graph):
        return "graph", _format_graph(res)

    ask_answer = getattr(res, "askAnswer", None)
    if ask_answer is not None:
        return "ask", _format_ask(res)

    vars_attr = getattr(res, "vars", None)
    if vars_attr is not None:
        return "select", _format_select(res)

    # Fallback: try to materialize triples into a temp graph
    try:
        tempg = Graph()
        for t in res:
            try:
                tempg.add(t)
            except Exception:
                pass
        return "graph", _format_graph(tempg)
    except Exception:
        logging.exception("SPARQL fallback materialization failed")
        return "invalid", MSG_INVALID
