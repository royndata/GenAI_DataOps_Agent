# src/agent/config.py

import os
from typing import Optional
from dotenv import load_dotenv

from pydantic_settings import BaseSettings, SettingsConfigDict
from pydantic import Field

# Load .env automatically
load_dotenv()


class Settings(BaseSettings):
    """
    Central configuration for the GenAI DataOps Agent.
    Uses pydantic-settings (Pydantic v2) to load environment variables safely.
    """

    # --- Required API keys ---
    openai_api_key: str = Field(..., env="OPENAI_API_KEY")
    slack_bot_token: str = Field(..., env="SLACK_BOT_TOKEN")
    slack_signing_secret: str = Field(..., env="SLACK_SIGNING_SECRET")
    slack_app_token: str = Field(..., env="SLACK_APP_TOKEN")  # Needed for Socket Mode

    # --- DB Option A: full connection string ---
    db_connection_string: Optional[str] = Field(None, env="DB_CONNECTION_STRING")

    # --- DB Option B: modular DB fields ---
    db_user: Optional[str] = Field(None, env="DB_USER")
    db_pass: Optional[str] = Field(None, env="DB_PASS")
    db_host: Optional[str] = Field(None, env="DB_HOST")
    db_port: Optional[str] = Field(None, env="DB_PORT")
    db_name: Optional[str] = Field(None, env="DB_NAME")

    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        extra="ignore"
    )


def load_settings() -> Settings:
    """
    Validates environment variables and returns a Settings object.
    Ensures DB config is valid before returning.
    """

    # Validate required env variables
    required = ["OPENAI_API_KEY", "SLACK_BOT_TOKEN", "SLACK_SIGNING_SECRET", "SLACK_APP_TOKEN"]
    missing = [var for var in required if not os.getenv(var)]

    if missing:
        raise EnvironmentError(f"Missing environment variables: {missing}")

    # Determine DB mode
    conn_str = os.getenv("DB_CONNECTION_STRING")

    modular_ok = all([
        os.getenv("DB_USER"),
        os.getenv("DB_PASS"),
        os.getenv("DB_HOST"),
        os.getenv("DB_PORT"),
        os.getenv("DB_NAME"),
    ])

    if not (conn_str or modular_ok):
        raise EnvironmentError(
            "Database config invalid. Provide DB_CONNECTION_STRING OR all modular DB_* variables."
        )

    # Instantiate Settings
    return Settings()
