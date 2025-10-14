import sys
import readline  # command history in Linux shells

from .app_config import DEFAULT_SELECT_LIMIT, AUTO_LIMIT
from .executor import run_query, ensure_default_graph_loaded

BANNER = """\
SPARQL Runner — ATAI Intermediate (1)
- Only executes SPARQL over the course dataset.
- Natural language inputs will get: "Please input SPARQL query."
Tips:
  * Enter multi-line mode with ':m', finish with a line ';;'
  * Use LIMIT/OFFSET for large results
"""

HELP = """\
Commands:
  :m                - enter multi-line mode (finish with ';;')
  :q                - quit
  :al on|off        - toggle auto-limit for SELECT
  :al <n>           - set default LIMIT for auto-limit to <n> (integer)
  :help             - show this help
"""


def read_multiline() -> str:
    print("\nEnter SPARQL (end with line ';;'):")
    lines = []
    while True:
        try:
            line = input()
        except EOFError:
            break
        if line.strip() == ";;":
            break
        lines.append(line)
    return "\n".join(lines).strip()


def main():
    print(BANNER)
    print("Loading data (with caps) ...")
    g, stats = ensure_default_graph_loaded()

    # runtime state (start from app_config defaults)
    auto_limit = bool(AUTO_LIMIT)
    default_limit = int(DEFAULT_SELECT_LIMIT)

    print(f"\n[Load Summary]")
    print(f"  Files loaded     : {stats.get('files_loaded', 0)}")
    print(f"  Total triples    : {stats.get('triples', len(g) if g is not None else 0)}")
    if "items" in stats:
        print("  Items (first 12):")
        for it in stats["items"][:12]:
            mark = "OK" if it.get("loaded") else f"SKIP ({it.get('reason','')})"
            fmt = it.get("format", "")
            size = it.get("bytes", 0)
            print(f"   - {it['path']}  [{fmt}]  {size} bytes  => {mark}")
        more = len(stats["items"]) - 12
        if more > 0:
            print(f"   ... ({more} more)")

    print(f"\n[Settings] auto-limit for SELECT: {'ON' if auto_limit else 'OFF'} (default LIMIT {default_limit})\n")
    print("Type ':help' for commands.\n")

    while True:
        try:
            s = input("SPARQL> ").strip()
        except (EOFError, KeyboardInterrupt):
            print("\nBye.")
            break

        if not s:
            continue

        # commands
        low = s.lower()
        if low in (":q",):
            print("Bye.")
            break
        if low in (":help",):
            print(HELP)
            continue
        if low == ":m":
            s = read_multiline()
            if not s:
                continue
        elif low.startswith(":al"):
            parts = s.split()
            if len(parts) == 2 and parts[1].lower() in ("on", "off"):
                auto_limit = (parts[1].lower() == "on")
                print(f"auto-limit: {'ON' if auto_limit else 'OFF'} (LIMIT {default_limit})")
                continue
            elif len(parts) == 2:
                try:
                    val = int(parts[1])
                    if val <= 0:
                        print("Please provide a positive integer for ':al <n>'.")
                    else:
                        default_limit = val
                        print(f"auto-limit default LIMIT set to {default_limit} (auto-limit is {'ON' if auto_limit else 'OFF'})")
                except ValueError:
                    print("Usage: ':al on|off' or ':al <positive-integer>'")
                continue
            else:
                print("Usage: ':al on|off' or ':al <positive-integer>'")
                continue

        # Execute using the default graph (no need to pass g explicitly)
        kind, out = run_query(
            s,
            auto_limit=auto_limit,
            default_limit=default_limit,
        )
        print(out)


if __name__ == "__main__":
    main()
