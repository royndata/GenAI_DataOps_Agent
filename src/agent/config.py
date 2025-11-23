# src/agent/config.py

import os
from dataclasses import dataclass
from dotenv import load_dotenv

# Load .env file once globally
load_dotenv()


@dataclass
class Settings:
    """
    Centralized configuration for the GenAI DataOps Agent.
    Every subsystem imports from here.
    """

    openai_api_key: str
    slack_bot_token: str
    slack_signing_secret: str
    db_connection_string: str


def load_settings() -> Settings:
    """
    Validates environment variables and returns a Settings object.
    Called inside main.py before subsystem initialization.
    """

    required = [
        "OPENAI_API_KEY",
        "SLACK_BOT_TOKEN",
        "SLACK_SIGNING_SECRET",
        "DB_CONNECTION_STRING",
    ]

    missing = [var for var in required if not os.getenv(var)]

    if missing:
        raise EnvironmentError(f"Missing environment variables: {missing}")

    return Settings(
        openai_api_key=os.getenv("OPENAI_API_KEY"),
        slack_bot_token=os.getenv("SLACK_BOT_TOKEN"),
        slack_signing_secret=os.getenv("SLACK_SIGNING_SECRET"),
        db_connection_string=os.getenv("DB_CONNECTION_STRING"),
    )
