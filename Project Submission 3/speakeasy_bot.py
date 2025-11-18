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
    parts = []
    
    # Recommendations
    recs = payload.get("recommendations")
    note = payload.get("note")
    
    if note:
        parts.append(note)

    if recs and isinstance(recs, list):
        for i, rec in enumerate(recs, 1):
            label = rec.get('label', 'Unknown')
            reason = rec.get('reason', '')
            image_id = rec.get('image_id')
            
            line = f"{i}. {label}"
            if reason:
                line += f" ({reason})"
            parts.append(line)
            
            if image_id:
                parts.append(f"image:{image_id}")
    
    if not parts:
        # Fallback for pure QA answers that might just return a string in 'note'
        # or legacy format
        if payload.get("answer"):
             return str(payload.get("answer"))
        return "I couldn't find any answers."

    return "\n".join(parts)

def on_new_message(message: str, room):
    text = (message or "").strip()
    room_id = getattr(room, 'room_id', 'default_room')
    print(f"\n[recv][room={room_id}]\n{text}\n")

    try:
        # FIX: Send user_id in the payload
        payload = {
            "query": text,
            "user_id": str(room_id)
        }
        
        resp = requests.post(BACKEND_ASK_URL, json=payload, timeout=30)
        if resp.status_code != 200:
            room.post_messages(f"Backend error: HTTP {resp.status_code}")
            return

        data = resp.json()
        out = render_answer(data)
        room.post_messages(out)
        print("[sent] response posted.")
    except Exception as e:
        logging.exception("Error handling message")
        room.post_messages("Internal error while contacting backend.")

def main():
    # Health Check
    try:
        h = requests.get(BACKEND_HEALTH_URL, timeout=5)
        logging.info("Backend health: %s", h.text)
    except Exception as e:
        logging.warning("Backend health check failed: %s", e)

    spx = Speakeasy(host=SPEAKEASY_HOST, username=SPEAKEASY_USER, password=SPEAKEASY_PASS)
    spx.login()
    spx.register_callback(on_new_message, EventType.MESSAGE)
    print("Speakeasy bot is now running. Press Ctrl+C to stop.")
    spx.start_listening()

if __name__ == "__main__":
    main()