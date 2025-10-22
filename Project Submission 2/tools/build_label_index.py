"""One-off utility to prebuild the label index cache.

Run:
  python tools/build_label_index.py
"""
from agent.entity_linker import EntityLinker

if __name__ == "__main__":
    EntityLinker()
    print("Label index built and cached.")
    