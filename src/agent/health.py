# src/agent/health.py

from agent.knowledge.database import Database
from agent.logging_config import logger


def run_health_checks(db: Database) -> bool:
    """
    Simple internal health check for startup validation.
    Returns True if DB is reachable.
    """

    logger.info("health_check_start")

    ok = db.test_connection()

    if not ok:
        logger.error("health_check_failed")
        return False

    logger.info("health_check_passed")
    return True
