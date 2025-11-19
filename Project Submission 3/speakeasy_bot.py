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
    
    note = payload.get("note")
    if note:
        parts.append(note)

    recs = payload.get("recommendations")
    if recs and isinstance(recs, list):
        for i, rec in enumerate(recs, 1):
            label = rec.get('label', 'Unknown')
            image_id = rec.get('image_id')
            
            # [MODIFIED] Just the label, no reason
            line = f"{i}. {label}"
            parts.append(line)
            
            if image_id:
                parts.append(f"image:{image_id}")
    
    if not parts:
        if payload.get("answer"):
             return str(payload.get("answer"))
        return "I couldn't find any answers or recommendations."

    return "\n".join(parts)

def on_new_message(message: str, room):
    text = (message or "").strip()
    room_id = getattr(room, 'room_id', 'default_room')
    print(f"\n[recv][room={room_id}]\n{text}\n")

    try:
        payload = {
            "query": text,
            "user_id": str(room_id)
        }
        
        resp = requests.post(BACKEND_ASK_URL, json=payload, timeout=30)
        if resp.status_code != 200:
            room.post_messages(f"Backend error: HTTP {resp.status_code}")
            return

        data = resp.json()
        # Keep debug logging for now
        print("\n[DEBUG] Backend Response Payload:")
        print(json.dumps(data, indent=2))
        
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
        logging.warning("Backend health check failed: %s", e)

    spx = Speakeasy(host=SPEAKEASY_HOST, username=SPEAKEASY_USER, password=SPEAKEASY_PASS)
    spx.login()
    spx.register_callback(on_new_message, EventType.MESSAGE)
    print("Speakeasy bot is now running. Press Ctrl+C to stop.")
    spx.start_listening()

if __name__ == "__main__":
    main()