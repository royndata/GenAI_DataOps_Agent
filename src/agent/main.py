from agent.health import run_health_checks
from agent.logging_config import configure_logging

if __name__ == "__main__":
    configure_logging()

if not run_health_checks(db):
    raise SystemExit("Startup health check failed")
