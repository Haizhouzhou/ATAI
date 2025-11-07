import os
import logging
from dotenv import load_dotenv
from speakeasy.client import Speakeasy
import requests # Uses requests to talk to the FastAPI server

# Load environment variables
load_dotenv()

# Setup logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

class SpeakeasyBot:
    """
    SpeakeasyBot class to interact with the Speakeasy platform
    by making requests to a separate web server (app/main.py).
    """
    def __init__(self, bot_token, api_key, server_url, bot_id=None):
        self.client = Speakeasy(
            bot_token=bot_token,
            api_key=api_key,
            server_url=server_url,
            bot_id=bot_id
        )
        
        # URL of your FastAPI server
        self.nlq_server_url = os.getenv("NLQ_SERVER_URL", "http://127.0.0.1:8000")
        self.timeout = int(os.getenv("REQUEST_TIMEOUT", 10))
        
        logger.info(f"Bot initialized. NLQ Server URL: {self.nlq_server_url}")

        # Register the message handler
        self.client.register_handler("on_new_message", self.on_new_message)
        logger.info("Registered 'on_new_message' handler.")

    def on_new_message(self, message):
        """
        Handles new incoming messages from Speakeasy.
        """
        logger.info(f"Received message: {message.content} from user: {message.user_id} in chat: {message.chat_room_id}")

        if message.user_id == self.client.bot_id:
            logger.info("Ignoring message from self.")
            return

        try:
            # Post the query to the FastAPI server
            # Note: message.chat_room_id is a good unique session key
            response = requests.post(
                f"{self.nlq_server_url}/nlq",
                json={"query": message.content, "user_id": message.chat_room_id},
                timeout=self.timeout,
            )
            response.raise_for_status() # Raise an exception for bad status codes
            
            data = response.json()
            response_text = data.get("answer", "I received an empty response from my brain.")

        except requests.exceptions.Timeout:
            logger.error(f"Request to NLQ server timed out for query: {message.content}")
            response_text = "I'm sorry, I'm taking a bit too long to think. Please try again."
        except requests.exceptions.RequestException as e:
            logger.error(f"Error connecting to NLQ server: {e}", exc_info=True)
            response_text = "I'm sorry, I'm having trouble connecting to my brain. Please try again later."
        except Exception as e:
            logger.error(f"Error processing message: {e}", exc_info=True)
            response_text = "I'm sorry, I encountered an internal error."

        try:
            # Send the response back to the chat room
            self.client.send_message(
                chat_room_id=message.chat_room_id,
                content=response_text
            )
            logger.info(f"Sent response to chat {message.chat_room_id}: {response_text}")
        except Exception as send_e:
            logger.error(f"Failed to send message to chat {message.chat_room_id}: {send_e}")

    def run(self):
        """
        Connects to the Speakeasy server and starts listening for messages.
        """
        logger.info("Connecting to Speakeasy...")
        try:
            self.client.connect()
            logger.info("Bot has disconnected.")
        except Exception as e:
            logger.error(f"Failed to connect or run bot: {e}", exc_info=True)
        finally:
            logger.info("Bot is shutting down.")

if __name__ == "__main__":
    BOT_TOKEN = os.getenv("SPEAKEASY_BOT_TOKEN")
    API_KEY = os.getenv("SPEAKEASY_API_KEY")
    SERVER_URL = os.getenv("SPEAKEASY_SERVER_URL", "https://api.speakeasy.tools")
    BOT_ID = os.getenv("SPEAKEASY_BOT_ID") 

    if not BOT_TOKEN or not API_KEY:
        logger.error("SPEAKEASY_BOT_TOKEN and SPEAKEASY_API_KEY must be set.")
    else:
        bot = SpeakeasyBot(
            bot_token=BOT_TOKEN,
            api_key=API_KEY,
            server_url=SERVER_URL,
            bot_id=BOT_ID
        )
        bot.run()