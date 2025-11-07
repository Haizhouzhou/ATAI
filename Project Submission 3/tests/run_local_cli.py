import requests
import uuid
import sys

# Your local FastAPI server should be running on port 8000
SERVER_URL = "http://127.0.0.1:8000/nlq"


def main():
    """
    A simple command-line client for sending requests
    to the locally running FastAPI /nlq endpoint.
    """
    print("--- Local Chatbot CLI ---")
    print(f"Connecting to: {SERVER_URL}")
    print("Make sure the FastAPI server is running in another terminal, e.g.:")
    print("python -m uvicorn app.main:app")
    print("Type 'exit' to quit.")
    print("-" * 20)

    # Create a random user_id for this session
    # This allows the server to maintain your conversation history
    session_user_id = f"local-cli-{uuid.uuid4()}"
    print(f"Session ID created: {session_user_id}")

    try:
        requests.get("http://127.0.0.1:8000", timeout=3)
    except requests.exceptions.ConnectionError:
        print("\n[Error] Could not connect to the local server.")
        print("Make sure your FastAPI server is running with:")
        print("python -m uvicorn app.main:app --host 0.0.0.0 --port 8000")
        sys.exit(1)

    while True:
        try:
            query = input("\nYou: ")
            if query.lower() in ["exit", "quit"]:
                print("Exiting client.")
                break

            if not query.strip():
                continue

            payload = {
                "query": query,
                "user_id": session_user_id
            }

            response = requests.post(SERVER_URL, json=payload)

            if response.status_code == 200:
                response_data = response.json()
                print(f"Agent: {response_data.get('answer', '... (no answer found)')}")
            else:
                print(f"\n[Server Error (HTTP {response.status_code})]")
                print(response.text)

        except KeyboardInterrupt:
            print("\nExiting.")
            break
        except Exception as e:
            print(f"\n[Client Error]: {e}")
            break


if __name__ == "__main__":
    main()
