# src/agent/knowledge/database.py

from sqlalchemy import text, create_engine
from sqlalchemy.orm import sessionmaker
from agent.logging_config import logger


class Database:
    """
    Centralized database connector using SQLAlchemy.
    All SQL tool operations call this object.
    """

    def __init__(self, connection_string: str):
        self.engine = create_engine(connection_string, future=True)
        self.SessionLocal = sessionmaker(bind=self.engine, autoflush=False)

    def test_connection(self) -> bool:
        """
        Quick test to validate DB connection.
        Called once at startup.
        """
        try:
            with self.engine.connect() as conn:
                conn.execute(text("SELECT 1"))
            logger.info("database_connected", message="Postgres connection OK")
            return True
        except Exception as e:
            logger.error("database_connection_failed", error=str(e))
            return False

    def run_query(self, query: str):
        """
        Used by SQL Tool to fetch data.
        """
        with self.SessionLocal() as session:
            result = session.execute(text(query))
            rows = result.fetchall()
            return rows
