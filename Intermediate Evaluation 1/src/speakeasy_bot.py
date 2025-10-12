from speakeasypy import Speakeasy, EventType
import os
import re
import logging

from .loader import load_graph
from .executor import set_default_graph, run_query  
from .app_config import (
    DATA_TARGETS,
    MSG_NON_SPARQL, MSG_INVALID,
    DEFAULT_SELECT_LIMIT, AUTO_LIMIT,
)

HOST = os.getenv("SPEAKEASY_HOST", "https://speakeasy.ifi.uzh.ch")
USERNAME = os.getenv("SPEAKEASY_USER", "RedFlickeringCandle")
PASSWORD = os.getenv("SPEAKEASY_PASS", "Sv9Kx0sH")

logging.basicConfig(level=logging.INFO)

SPARQL_HEAD_RE = re.compile(
    r"^\s*(PREFIX|SELECT|ASK|CONSTRUCT|DESCRIBE)\b",
    flags=re.IGNORECASE | re.DOTALL,
)

# --- Login ---
speakeasy = Speakeasy(host=HOST, username=USERNAME, password=PASSWORD)
speakeasy.login()

# --- Load and register the default KG once at startup ---
# DATA_TARGETS = [Path("/space_mounts/atai-hs25/dataset/graph.nt")]
g, stats = load_graph(DATA_TARGETS)
set_default_graph(g)  
logging.info("[KG] files_loaded=%s triples=%s", stats.get("files_loaded", 0), stats.get("triples", 0))

# runtime config
AUTO_LIMIT_ENABLED = bool(AUTO_LIMIT)
DEFAULT_LIMIT = int(DEFAULT_SELECT_LIMIT)

def on_new_message(message: str, room):
    text = (message or "").strip()
    print(f"\n[recv][room={getattr(room, 'room_id', None)}]\n{text}\n")

    try:
        if not SPARQL_HEAD_RE.match(text):
            response = MSG_NON_SPARQL
        else:
            kind, out = run_query(
                text,
                auto_limit=AUTO_LIMIT_ENABLED,
                default_limit=DEFAULT_LIMIT,
            )
            response = out

        if not isinstance(response, str) or not response.strip():
            response = "No results found."
    except Exception as e:
        print(f"[error] executing query: {e}")
        response = MSG_INVALID

    room.post_messages(response)
    print("[sent] response posted.")


if __name__ == "__main__":
    speakeasy.register_callback(on_new_message, EventType.MESSAGE)
    print("Bot is now running. Press Ctrl+C to stop.")
    speakeasy.start_listening()