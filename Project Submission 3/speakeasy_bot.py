# /files/Project Submission 2/speakeasy_bot.py
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

# def render_answer(payload: dict) -> str:
#     """把你 /ask 的 JSON 回答，渲染成 Speakeasy 聊天里的纯文本"""
#     parts = []

#     fa = payload.get("factual_answer")
#     if fa and isinstance(fa, dict):
#         ans = fa.get("answer", [])
#         if ans:
#             parts.append(f"Factual: {', '.join(map(str, ans))}")

#     ea = payload.get("embedding_answer")
#     if ea and isinstance(ea, dict):
#         ans = ea.get("answer")
#         score = ea.get("score")
#         if ans:
#             if score is not None:
#                 parts.append(f"Embedding: {ans} (score={round(float(score), 4)})")
#             else:
#                 parts.append(f"Embedding: {ans}")

#     note = payload.get("note")
#     if note:
#         parts.append(f"Note: {note}")

#     if not parts:
#         return "No results found."

#     return "\n".join(parts)

def render_answer(payload: dict) -> str:
    """Render your backend JSON into plain text for Speakeasy"""
    # 如果你的后端返回 {"answer": "..."}，优先显示它
    if "answer" in payload:
        return payload["answer"]

    # 保留原逻辑（兼容以前的格式）
    parts = []
    fa = payload.get("factual_answer")
    if fa and isinstance(fa, dict):
        ans = fa.get("answer", [])
        if ans:
            parts.append(f"Factual: {', '.join(map(str, ans))}")

    ea = payload.get("embedding_answer")
    if ea and isinstance(ea, dict):
        ans = ea.get("answer")
        score = ea.get("score")
        if ans:
            if score is not None:
                parts.append(f"Embedding: {ans} (score={round(float(score), 4)})")
            else:
                parts.append(f"Embedding: {ans}")

    note = payload.get("note")
    if note:
        parts.append(f"Note: {note}")

    if not parts:
        return "No other results found."

    return "\n".join(parts)

def on_new_message(message: str, room):
    text = (message or "").strip()
    print(f"\n[recv][room={getattr(room, 'room_id', None)}]\n{text}\n")

    try:
        resp = requests.post(BACKEND_ASK_URL, json={"query": text}, timeout=30)
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
