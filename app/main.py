from __future__ import annotations

"""
Minimal application entrypoint.

The important function is `answer_question(message, session_id=None)`.
You can call this from your Speakeasy bot, from unit tests, or from
the command line for quick experiments.
"""

from typing import Optional

from agent.composer import get_global_composer


def answer_question(message: str, session_id: Optional[str] = None) -> str:
    """
    High-level API function: given a user message, return the agent's reply.

    Parameters
    ----------
    message:
        The content of the user message (one turn).
    session_id:
        Optional identifier for the conversation (chatroom, user id, etc.).
        If omitted, a shared default session is used.
    """
    composer = get_global_composer()
    return composer.answer(message, session_id=session_id)


if __name__ == "__main__":
    # Very small REPL for local debugging
    print("Simple movie agent REPL. Type 'quit' to exit.")
    while True:
        try:
            user_in = input("You: ").strip()
        except (EOFError, KeyboardInterrupt):
            break
        if not user_in or user_in.lower() in {"quit", "exit"}:
            break
        reply = answer_question(user_in)
        print(f"Agent: {reply}")
