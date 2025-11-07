import os
import logging
import time
from dotenv import load_dotenv
from speakeasy.client import Speakeasy
from app.main import Chatbot  # Main logic is now in Chatbot
from agent.session_manager import SessionManager # Import session manager

# Load environment variables
load_dotenv()

# Setup logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

class SpeakeasyBot:
    """
    SpeakeasyBot class to interact with the Speakeasy platform.
    """
    def __init__(self, bot_token, api_key, server_url, bot_id=None):
        self.client = Speakeasy(
            bot_token=bot_token,
            api_key=api_key,
            server_url=server_url,
            bot_id=bot_id
        )
        # Initialize the main Chatbot logic
        try:
            self.chatbot = Chatbot()
            logger.info("Chatbot logic initialized successfully.")
        except Exception as e:
            logger.error(f"Failed to initialize Chatbot: {e}", exc_info=True)
            raise
        
        # Initialize a session manager to hold states for different users
        self.session_manager = SessionManager()
        logger.info("Session manager initialized.")

        # Register the message handler
        self.client.register_handler("on_new_message", self.on_new_message)
        logger.info("Registered 'on_new_message' handler.")

    def on_new_message(self, message):
        """
        Handles new incoming messages from Speakeasy.
        """
        logger.info(f"Received message: {message.content} from user: {message.user_id} in chat: {message.chat_room_id}")

        # Ignore messages from the bot itself
        if message.user_id == self.client.bot_id:
            logger.info("Ignoring message from self.")
            return

        try:
            # Get the session for the current user
            # We use chat_room_id as a proxy for a continuous conversation session
            user_session = self.session_manager.get_session(message.chat_room_id)

            # Process the natural language query using the Chatbot class
            # The Chatbot will now handle intent routing (QA vs. Rec)
            # and use the provided session for context
            response_text = self.chatbot.process_nl_query(message.content, user_session)

            if not response_text:
                response_text = "I'm sorry, I couldn't process that request."
                logger.warning(f"Chatbot returned an empty response for query: {message.content}")

            # Send the response back to the chat room
            self.client.send_message(
                chat_room_id=message.chat_room_id,
                content=response_text
            )
            logger.info(f"Sent response to chat {message.chat_room_id}: {response_text}")

        except Exception as e:
            logger.error(f"Error processing message: {e}", exc_info=True)
            try:
                # Send an error message to the user
                self.client.send_message(
                    chat_room_id=message.chat_room_id,
                    content="I'm sorry, I encountered an internal error. Please try again."
                )
            except Exception as send_e:
                logger.error(f"Failed to send error message to chat {message.chat_room_id}: {send_e}")

    def run(self):
        """
        Connects to the Speakeasy server and starts listening for messages.
        """
        logger.info("Connecting to Speakeasy...")
        try:
            self.client.connect()  # This is a blocking call that runs the bot
            logger.info("Bot has disconnected.")
        except Exception as e:
            logger.error(f"Failed to connect or run bot: {e}", exc_info=True)
        finally:
            logger.info("Bot is shutting down.")

if __name__ == "__main__":
    # Load credentials from environment variables
    BOT_TOKEN = os.getenv("SPEAKEASY_BOT_TOKEN")
    API_KEY = os.getenv("SPEAKEASY_API_KEY")
    SERVER_URL = os.getenv("SPEAKEASY_SERVER_URL", "https://api.speakeasy.tools")
    BOT_ID = os.getenv("SPEAKEASY_BOT_ID") # Optional, but good for self-message check

    if not BOT_TOKEN or not API_KEY:
        logger.error("SPEAKEASY_BOT_TOKEN and SPEAKEASY_API_KEY must be set.")
    else:
        # Create and run the bot
        bot = SpeakeasyBot(
            bot_token=BOT_TOKEN,
            api_key=API_KEY,
            server_url=SERVER_URL,
            bot_id=BOT_ID
        )
        bot.run()