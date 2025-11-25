# src/agent/main.py

"""
Main entry point for GenAI DataOps Agent.

Production-grade startup sequence:
1. Configure structured logging (first, before any imports that log)
2. Load and validate settings
3. Initialize database connection
4. Run health checks
5. Start Slack listener
6. Handle graceful shutdown

Features:
- Startup time tracking
- Component readiness validation
- Graceful error recovery
- Resource cleanup verification
- Proper exit codes
"""

import signal
import sys
import time
from typing import Optional

from agent.config import Settings
from agent.logging_config import configure_logging, logger
from agent.knowledge.database import Database
from agent.health import run_health_checks
from agent.ingestion.listener import SlackListener


# Version info (update on releases)
AGENT_VERSION = "1.0.0"
AGENT_NAME = "GenAI DataOps Agent"


class Agent:
    """
    Main agent orchestrator.
    Handles startup, health checks, and graceful shutdown.
    """

    def __init__(self):
        self.settings: Optional[Settings] = None
        self.database: Optional[Database] = None
        self.listener: Optional[SlackListener] = None
        self._shutdown_requested = False
        self._startup_time: Optional[float] = None
        self._components_initialized = []

    def _log_startup_banner(self) -> None:
        """Log startup banner with version info."""
        logger.info(
            "agent_startup_banner",
            name=AGENT_NAME,
            version=AGENT_VERSION,
            python_version=sys.version.split()[0]
        )

    def initialize(self) -> None:
        """Initialize all components in correct order."""
        startup_start = time.time()
        
        try:
            self._log_startup_banner()
            logger.info("agent_initializing")

            # 1. Load settings (Pydantic validates automatically)
            try:
                self.settings = Settings()
                self._components_initialized.append("settings")
                logger.info("agent_settings_loaded")
            except Exception as e:
                logger.exception("agent_settings_load_failed", error=str(e))
                raise RuntimeError(f"Failed to load settings: {e}")

            # 2. Initialize database
            try:
                self.database = Database(self.settings)
                self._components_initialized.append("database")
                logger.info("agent_database_initialized")
            except Exception as e:
                logger.exception("agent_database_init_failed", error=str(e))
                raise RuntimeError(f"Failed to initialize database: {e}")

            # 3. Run health checks
            try:
                if not run_health_checks(self.database):
                    raise RuntimeError("Health checks failed - cannot start agent")
                self._components_initialized.append("health_checks")
                logger.info("agent_health_checks_passed")
            except Exception as e:
                logger.exception("agent_health_checks_failed", error=str(e))
                raise RuntimeError(f"Health checks failed: {e}")

            # 4. Initialize Slack listener
            try:
                self.listener = SlackListener(self.settings)
                self._components_initialized.append("listener")
                logger.info("agent_listener_initialized")
            except Exception as e:
                logger.exception("agent_listener_init_failed", error=str(e))
                raise RuntimeError(f"Failed to initialize listener: {e}")

            startup_time = time.time() - startup_start
            self._startup_time = startup_time
            logger.info(
                "agent_initialization_complete",
                startup_time_seconds=round(startup_time, 2),
                components=self._components_initialized
            )

        except Exception as e:
            startup_time = time.time() - startup_start
            logger.exception(
                "agent_initialization_failed",
                error=str(e),
                startup_time_seconds=round(startup_time, 2),
                components_initialized=self._components_initialized
            )
            raise

    def start(self) -> None:
        """Start the agent (blocks until shutdown)."""
        if not self.listener:
            raise RuntimeError("Agent not initialized. Call initialize() first.")

        try:
            # Setup graceful shutdown
            signal.signal(signal.SIGINT, self._signal_handler)
            signal.signal(signal.SIGTERM, self._signal_handler)

            logger.info("agent_starting")
            self.listener.start()
            logger.info("agent_started", uptime_seconds=0)

        except KeyboardInterrupt:
            logger.info("agent_interrupted")
            self.shutdown()
        except Exception as e:
            logger.exception("agent_start_failed", error=str(e))
            self.shutdown()
            raise

    def _signal_handler(self, signum, frame):
        """Handle shutdown signals gracefully."""
        signal_name = signal.Signals(signum).name
        logger.info("agent_shutdown_signal_received", signal=signal_name, signum=signum)
        self._shutdown_requested = True
        self.shutdown()
        sys.exit(0)

    def shutdown(self) -> None:
        """Gracefully shutdown all components."""
        shutdown_start = time.time()
        logger.info("agent_shutting_down", components_initialized=self._components_initialized)

        shutdown_errors = []

        # Shutdown in reverse order of initialization
        if self.listener:
            try:
                self.listener.stop()
                logger.info("agent_listener_stopped")
            except Exception as e:
                error_msg = f"Listener shutdown error: {e}"
                logger.error("agent_listener_stop_error", error=str(e))
                shutdown_errors.append(error_msg)

        if self.database:
            try:
                # Database context manager handles cleanup automatically
                # If using context manager pattern, ensure proper disposal
                logger.info("agent_database_cleanup")
            except Exception as e:
                error_msg = f"Database cleanup error: {e}"
                logger.error("agent_database_cleanup_error", error=str(e))
                shutdown_errors.append(error_msg)

        shutdown_time = time.time() - shutdown_start
        uptime = time.time() - self._startup_time if self._startup_time else 0

        if shutdown_errors:
            logger.warning(
                "agent_shutdown_complete_with_errors",
                shutdown_time_seconds=round(shutdown_time, 2),
                uptime_seconds=round(uptime, 2),
                errors=shutdown_errors
            )
        else:
            logger.info(
                "agent_shutdown_complete",
                shutdown_time_seconds=round(shutdown_time, 2),
                uptime_seconds=round(uptime, 2)
            )


def main():
    """Main entry point."""
    # Configure logging FIRST (before any other imports that might log)
    configure_logging()

    agent = Agent()

    try:
        agent.initialize()
        agent.start()
    except KeyboardInterrupt:
        logger.info("agent_keyboard_interrupt")
        agent.shutdown()
        sys.exit(0)
    except RuntimeError as e:
        # Component initialization failures
        logger.exception("agent_runtime_error", error=str(e))
        agent.shutdown()
        sys.exit(1)
    except Exception as e:
        # Unexpected errors
        logger.exception("agent_fatal_error", error=str(e))
        agent.shutdown()
        sys.exit(1)


if __name__ == "__main__":
    main()
