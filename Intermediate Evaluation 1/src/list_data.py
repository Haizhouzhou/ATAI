"""
List candidate files under DATA_TARGETS with sizes and extension hints.
Use it to decide whether to raise loader caps or point to a smaller subfolder.
"""
import os
from pathlib import Path

from .app_config import DATA_TARGETS, EXT_TO_FORMAT

def main():
    print("Scanning targets:")
    for t in DATA_TARGETS:
        print(" -", t)

    count = 0
    total = 0
    exts = {}

    for t in DATA_TARGETS:
        if not t.exists():
            continue
        if t.is_file():
            files = [t]
        else:
            files = []
            for root, _, names in os.walk(t):
                for n in names:
                    files.append(Path(root) / n)

        for p in files:
            try:
                sz = p.stat().st_size
            except Exception:
                continue
            total += sz
            count += 1
            ext = p.suffix.lower().lstrip(".") or "(no-ext)"
            exts[ext] = exts.get(ext, 0) + 1

    print(f"\nTotal files: {count}")
    print(f"Approx total bytes: {total}")
    print("\nTop extensions (including '(no-ext)'):")
    for ext, n in sorted(exts.items(), key=lambda x: -x[1])[:40]:
        mapped = EXT_TO_FORMAT.get(ext, "")
        print(f"  {ext:>10}  x{n:>7}   format_hint={mapped}")

if __name__ == "__main__":
    main()
