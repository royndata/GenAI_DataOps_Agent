# src/agent/config.py

import os
from dataclasses import dataclass
from dotenv import load_dotenv

# Load .env globally
load_dotenv()


@dataclass
class Settings:
    """
    Centralized configuration for the GenAI DataOps Agent.
    Only one DB mode is active at runtime:
    - db_connection_string (single string)
    OR
    - modular DB fields (db_user, db_pass, db_host, db_port, db_name)
    """

    openai_api_key: str
    slack_bot_token: str
    slack_signing_secret: str

    # DB Option A
    db_connection_string: str | None = None

    # DB Option B
    db_user: str | None = None
    db_pass: str | None = None
    db_host: str | None = None
    db_port: str | None = None
    db_name: str | None = None


def load_settings() -> Settings:
    """
    Validates environment variables and returns a Settings object.
    """

    # Required non-DB variables
    required = [
        "OPENAI_API_KEY",
        "SLACK_BOT_TOKEN",
        "SLACK_SIGNING_SECRET",
    ]

    missing = [var for var in required if not os.getenv(var)]
    if missing:
        raise EnvironmentError(f"Missing environment variables: {missing}")

    # DB mode detection
    conn_str = os.getenv("DB_CONNECTION_STRING")
    modular_user = os.getenv("DB_USER")
    modular_pass = os.getenv("DB_PASS")
    modular_host = os.getenv("DB_HOST")
    modular_port = os.getenv("DB_PORT")
    modular_name = os.getenv("DB_NAME")

    modular_ok = all([modular_user, modular_pass, modular_host, modular_port, modular_name])

    if not (conn_str or modular_ok):
        raise EnvironmentError(
            "Database config invalid. Provide DB_CONNECTION_STRING OR all modular DB_* variables."
        )

    # Return clean, explicit DB config
    if conn_str:
        return Settings(
            openai_api_key=os.getenv("OPENAI_API_KEY"),
            slack_bot_token=os.getenv("SLACK_BOT_TOKEN"),
            slack_signing_secret=os.getenv("SLACK_SIGNING_SECRET"),
            db_connection_string=conn_str,
            # modular fields remain None
        )

    # Modular mode
    return Settings(
        openai_api_key=os.getenv("OPENAI_API_KEY"),
        slack_bot_token=os.getenv("SLACK_BOT_TOKEN"),
        slack_signing_secret=os.getenv("SLACK_SIGNING_SECRET"),
        db_connection_string=None,
        db_user=modular_user,
        db_pass=modular_pass,
        db_host=modular_host,
        db_port=modular_port,
        db_name=modular_name,
    )
