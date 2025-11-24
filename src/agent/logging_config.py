# src/agent/logging_config.py

import logging
import structlog


def configure_logging() -> None:
    """
    Configure structured logging for the entire agent.
    Called once from main.py before anything else is imported.
    """

    # 1. Configure standard logging (for libraries)
    logging.basicConfig(
        format="%(message)s",
        level=logging.INFO,
    )

    # 2. Configure structlog processors
    structlog.configure(
        processors=[
            structlog.processors.TimeStamper(fmt="iso"),
            structlog.dev.set_exc_info,
            structlog.processors.StackInfoRenderer(),
            structlog.processors.format_exc_info,
            structlog.processors.JSONRenderer(),
        ],
        wrapper_class=structlog.make_filtering_bound_logger(logging.INFO),
        context_class=dict,
        logger_factory=structlog.PrintLoggerFactory(),
        cache_logger_on_first_use=True,
    )


# Module-level logger for imports
logger = structlog.get_logger()
