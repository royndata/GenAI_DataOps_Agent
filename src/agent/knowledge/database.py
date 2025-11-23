# src/agent/knowledge/database.py

import os
from sqlalchemy import text, create_engine
from sqlalchemy.orm import sessionmaker, Session
from sqlalchemy.exc import OperationalError, SQLAlchemyError
from agent.logging_config import logger
from agent.config import Settings
from tenacity import retry, stop_after_attempt, wait_exponential
from typing import Optional, List, Tuple

# Read secrets manager flag from environment
USE_SECRETS_MANAGER = os.getenv("USE_SECRETS_MANAGER", "false").lower() == "true"


def build_modular_connection_string(env: Settings) -> Optional[str]:
    """
    Build a secure Postgres connection string using modular DB fields.
    Used only when DB_CONNECTION_STRING is not provided and secrets manager is not enabled.
    """
    if USE_SECRETS_MANAGER:
        logger.info("Using secrets manager for DB credentials")
        # TODO: Integrate actual secrets manager logic here
        return None

    if (
        env.db_user
        and env.db_pass
        and env.db_host
        and env.db_port
        and env.db_name
    ):
        return (
            f"postgresql+psycopg2://{env.db_user}:{env.db_pass}"
            f"@{env.db_host}:{env.db_port}/{env.db_name}?sslmode=require"
        )
    return None


class Database:
    """
    Production-grade SQLAlchemy database layer.
    Supports modular or full connection string configurations.
    Includes retry logic, connection pooling, transaction safety, and optional secrets manager integration.
    """

    def __init__(self, env: Settings):
        # Determine which DB mode to use
        if USE_SECRETS_MANAGER:
            connection_string = self._fetch_connection_from_secrets_manager()
        else:
            modular_str = build_modular_connection_string(env)
            if modular_str:
                connection_string = modular_str
                logger.info("Using modular DB connection (from DB_USER/DB_PASS/DB_HOST/etc.)")
            else:
                connection_string = env.db_connection_string
                logger.info("Using DB_CONNECTION_STRING from .env")

        if not connection_string:
            raise RuntimeError("No valid database configuration found.")

        # Create engine with pooling and future flag
        try:
            self.engine = create_engine(
                connection_string,
                future=True,
                pool_size=10,
                max_overflow=20,
                pool_timeout=30,
                pool_recycle=1800,
            )
            self.SessionLocal = sessionmaker(
                bind=self.engine,
                autoflush=False,
                autocommit=False,
                future=True,
            )
            logger.info("database_init_success", message="Database engine initialized.")
        except Exception as e:
            logger.error("database_init_failed", error=str(e))
            raise RuntimeError(f"Failed to initialize database engine: {e}")

    def _fetch_connection_from_secrets_manager(self) -> Optional[str]:
        """
        Placeholder: implement fetching DB credentials from secrets manager.
        Must return a full connection string.
        """
        # TODO: Replace this with actual secrets manager integration
        logger.warning("Secrets manager integration not implemented; returning None")
        return None

    # Context manager support
    def __enter__(self) -> "Database":
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        self.engine.dispose()
        logger.info("database_disposed", message="Database engine disposed.")

    # Retry decorator for transient errors
    @retry(stop=stop_after_attempt(3), wait=wait_exponential(multiplier=1, min=2, max=10))
    def test_connection(self) -> bool:
        """Quick sanity check run at startup with retries."""
        try:
            with self.engine.connect() as conn:
                conn.execute(text("SELECT 1"))
            logger.info("database_connected", message="Postgres connection OK")
            return True
        except OperationalError as e:
            logger.warning("database_connection_retry", error=str(e))
            raise
        except Exception as e:
            logger.error("database_connection_failed", error=str(e))
            return False

    def run_query(self, query: str) -> List[Tuple]:
        """
        Execute a query in a safe transaction.
        Returns list of row tuples.
        """
        with self.SessionLocal() as session:
            try:
                result = session.execute(text(query))
                session.commit()
                return result.fetchall()
            except SQLAlchemyError as e:
                session.rollback()
                logger.error("query_execution_failed", error=str(e), query=query)
                raise

    def run_transaction(self, queries: List[str]) -> None:
        """
        Execute multiple queries in a single transaction.
        Rolls back all if any fail.
        """
        with self.SessionLocal() as session:
            try:
                for q in queries:
                    session.execute(text(q))
                session.commit()
                logger.info("transaction_success", message=f"{len(queries)} queries executed.")
            except SQLAlchemyError as e:
                session.rollback()
                logger.error("transaction_failed", error=str(e), queries=queries)
                raise
