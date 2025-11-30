import os
import json
import logging
import requests
from speakeasypy import Speakeasy, EventType

logging.basicConfig(level=logging.INFO)

SPEAKEASY_HOST = os.getenv("SPEAKEASY_HOST", "https://speakeasy.ifi.uzh.ch")
SPEAKEASY_USER = os.getenv("SPEAKEASY_USER", "RedFlickeringCandle")
SPEAKEASY_PASS = os.getenv("SPEAKEASY_PASS", "Sv9Kx0sH")

BACKEND_ASK_URL = os.getenv("BACKEND_ASK_URL", "http://localhost:8000/ask")
BACKEND_HEALTH_URL = os.getenv("BACKEND_HEALTH_URL", "http://localhost:8000/health")

def render_answer(payload: dict) -> str:
    """Renders the JSON response from /ask into plain text for Speakeasy."""
    
    # 1. Primary Text Response
    # The updated main.py puts the main text in "answer".
    reply = payload.get("answer")
    
    if not reply:
        # Fallback for older keys if "answer" is missing
        if payload.get("note"):
            reply = payload.get("note")
        elif payload.get("factual_answer"):
            reply = f"Factual: {payload['factual_answer']}"
        else:
            return "I couldn't find any answers or recommendations."

    # 2. Append Single Image (Multimedia)
    # main.py returns {"answer": "...", "image": "image:..."}
    if "image" in payload and payload["image"]:
        # Avoid duplicating if it's already in the text
        if payload["image"] not in reply:
            reply += f"\n\n{payload['image']}"

    # 3. Append Recommendation Images
    # main.py returns {"answer": "list of movies...", "recommendations": [{...}, {...}]}
    recs = payload.get("recommendations")
    if recs and isinstance(recs, list):
        # The text list is usually already in 'reply', so we just check for images to append
        for rec in recs:
            if rec.get("image"):
                # Check if this specific image string is already in the text to avoid duplicates
                if rec["image"] not in reply:
                    reply += f"\n{rec['image']}"

    return reply

def on_new_message(message: str, room):
    text = (message or "").strip()
    room_id = getattr(room, 'room_id', 'unknown_room')
    print(f"\n[recv][room={room_id}]\n{text}\n")

    try:
        resp = requests.post(BACKEND_ASK_URL, json={"query": text, "user_id": str(room_id)}, timeout=30)
        
        if resp.status_code != 200:
            error_msg = f"Backend error: HTTP {resp.status_code}"
            print(f"Error: {error_msg}")
            room.post_messages(error_msg)
            return

        data = resp.json()
        
        # Debug Log
        print("-" * 40)
        print("DEBUG: Raw JSON from Backend:")
        print(json.dumps(data, indent=2))
        print("-" * 40)

        out = render_answer(data)
        
        room.post_messages(out)
        print("[sent] response posted.")
        
    except Exception as e:
        logging.exception("Error handling message")
        room.post_messages("Internal error while contacting backend.")

def main():
    try:
        h = requests.get(BACKEND_HEALTH_URL, timeout=5)
        logging.info("Backend health: %s", h.text)
    except Exception as e:
        # Non-critical warning if health check fails
        logging.warning("Backend health check failed (ignoring): %s", e)

    spx = Speakeasy(host=SPEAKEASY_HOST, username=SPEAKEASY_USER, password=SPEAKEASY_PASS)
    spx.login()
    spx.register_callback(on_new_message, EventType.MESSAGE)
    print("Speakeasy bot is now running. Press Ctrl+C to stop.")
    spx.start_listening()

if __name__ == "__main__":
    main()