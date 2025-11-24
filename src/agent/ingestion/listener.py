# ----------------------------------------------------------
# src/agent/ingestion/listener.py

"""
Slack listener for the GenAI DataOps Agent.

Enhancements included:
- Input sanitization (remove @bot mentions)
- Input validation (length check, empty check)
- Router-level rate limiting support
- Socket Mode operation
- Safe chart uploads
- Clean structured logging
"""

import signal
import sys
import re
from typing import Optional

from slack_bolt import App
from slack_bolt.adapter.socket_mode import SocketModeHandler
from slack_bolt.error import BoltError

from agent.config import Settings
from agent.logging_config import logger
from agent.knowledge.database import Database
from agent.tools.sql_tool import SQLTool
from agent.tools.pandasai_tool import PandasAITool
from agent.cognition.router import Router


class SlackListener:
    """Production-ready Slack listener."""

    def __init__(self, settings: Settings):
        self.settings = settings
        self.app: Optional[App] = None
        self.handler: Optional[SocketModeHandler] = None

        # Initialize tools + router
        db = Database(self.settings)
        self.sql_tool = SQLTool(db=db)
        self.pandas_tool = PandasAITool(settings=settings)
        self.router = Router(self.sql_tool, self.pandas_tool)

        logger.info("slack_listener_tools_initialized")

    # ------------------------------------------------------
    # Helpers: sanitize & validate input
    # ------------------------------------------------------
    def _sanitize_input(self, text: str) -> str:
        """Remove Slack @mentions and normalize whitespace."""
        text = re.sub(r'<@[A-Z0-9]+>', '', text).strip()
        text = " ".join(text.split())
        return text

    def _validate_input(self, text: str):
        """Validate message before routing."""
        if not text or not text.strip():
            return False, "Please provide a query or message."

        if len(text) > 3000:
            return False, f"Message too long ({len(text)} chars). Max allowed is 3000."

        return True, None

    # ------------------------------------------------------
    # Slack App Creation
    # ------------------------------------------------------
    def _create_app(self) -> App:
        app = App(
            token=self.settings.slack_bot_token,
            signing_secret=self.settings.slack_signing_secret,
        )
        logger.info("slack_app_created")
        return app

    # ------------------------------------------------------
    # Handler Registration
    # ------------------------------------------------------
    def _register_handlers(self, app: App) -> None:

        # --------------------------------------------------
        # @mentions in channels
        # --------------------------------------------------
        @app.event("app_mention")
        def handle_mention(body, say, event, logger):
            try:
                user = event.get("user")
                raw_text = event.get("text", "")

                text = self._sanitize_input(raw_text)
                is_valid, err = self._validate_input(text)
                if not is_valid:
                    say(err)
                    logger.warning("invalid_input", user=user, error=err)
                    return

                # Router call
                response = self.router.route(text, user_id=user)

                # Send main response with truncation if needed
                message = response.get("message", "Processing...")
                if len(message) > 4000:
                    message = message[:3900] + "\n... (truncated)"
                    logger.warning("slack_message_truncated", original_length=len(response.get("message", "")))

                say(message)

                # Upload chart if available
                if response.get("chart_path"):
                    try:
                        with open(response["chart_path"], "rb") as f:
                            app.client.files_upload(
                                channels=event["channel"],
                                file=f,
                                title="Analysis Chart"
                            )
                        logger.info("chart_uploaded", chart=response["chart_path"])
                    except Exception as e:
                        logger.error("chart_upload_failed", error=str(e))

            except Exception as e:
                logger.exception("app_mention_error", error=str(e))
                try:
                    say("⚠️ Something went wrong. Try again.")
                except:
                    pass

        # --------------------------------------------------
        # Direct Messages
        # --------------------------------------------------
        @app.event("message")
        def handle_dm(body, event, say, logger):
            try:
                if event.get("channel_type") != "im":
                    return
                if event.get("bot_id"):
                    return

                user = event.get("user")
                raw_text = event.get("text", "")

                text = self._sanitize_input(raw_text)
                is_valid, err = self._validate_input(text)
                if not is_valid:
                    say(err)
                    logger.warning("invalid_input", user=user, error=err)
                    return

                # Router call
                response = self.router.route(text, user_id=user)

                # Send main response with truncation if needed
                message = response.get("message", "Processing...")
                if len(message) > 4000:
                    message = message[:3900] + "\n... (truncated)"
                    logger.warning("slack_message_truncated", original_length=len(response.get("message", "")))

                say(message)

                if response.get("chart_path"):
                    try:
                        with open(response["chart_path"], "rb") as f:
                            app.client.files_upload(
                                channels=event["channel"],
                                file=f,
                                title="Analysis Chart"
                            )
                        logger.info("chart_uploaded", chart=response["chart_path"])
                    except Exception as e:
                        logger.error("chart_upload_failed", error=str(e))

            except Exception as e:
                logger.exception("dm_error", error=str(e))
                try:
                    say("⚠️ I hit an error. Try again.")
                except:
                    pass

        # Reactions (logged only)
        @app.event("reaction_added")
        def handle_reaction_added(event, logger):
            logger.info("reaction_added", reaction=event.get("reaction"))

        @app.event("reaction_removed")
        def handle_reaction_removed(event, logger):
            logger.info("reaction_removed", reaction=event.get("reaction"))

        logger.info("slack_handlers_registered")

    # ------------------------------------------------------
    # Start Listener
    # ------------------------------------------------------
    def start(self):
        try:
            self.app = self._create_app()
            self._register_handlers(self.app)

            if not self.settings.slack_app_token:
                raise RuntimeError("Missing SLACK_APP_TOKEN")

            self.handler = SocketModeHandler(self.app, self.settings.slack_app_token)

            logger.info("slack_listener_starting")
            self.handler.start()

        except Exception as e:
            logger.exception("listener_failed", error=str(e))
            raise

    # ------------------------------------------------------
    # Shutdown
    # ------------------------------------------------------
    def stop(self):
        logger.info("slack_listener_stopping")
        if self.handler:
            try:
                self.handler.close()
            except Exception as e:
                logger.error("slack_listener_stop_error", error=str(e))
        logger.info("slack_listener_stopped")


def main():
    try:
        settings = Settings()
        listener = SlackListener(settings)
        
        # Setup graceful shutdown handlers
        def signal_handler(signum, frame):
            logger.info("slack_shutdown_signal_received", signal=signum)
            listener.stop()
            sys.exit(0)
        
        signal.signal(signal.SIGINT, signal_handler)
        signal.signal(signal.SIGTERM, signal_handler)
        
        listener.start()
    except KeyboardInterrupt:
        logger.info("slack_listener_interrupted")
        sys.exit(0)
    except Exception as e:
        logger.exception("slack_listener_fatal_error", error=str(e))
        sys.exit(1)


if __name__ == "__main__":
    main()