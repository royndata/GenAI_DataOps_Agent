# ----------------------------------------------------------
# src/agent/cognition/router.py

"""
Routing engine for the GenAI DataOps Agent.

Improvements added:
- Rate limiting (10 req/min per user)
- router.route() now accepts user_id
- SQL + PandasAI + dataset-info routing unchanged
- Structured logging
"""

import re
from time import time
from collections import defaultdict
from typing import Dict, Any, Optional

from agent.tools.sql_tool import SQLTool
from agent.tools.pandasai_tool import PandasAITool
from agent.logging_config import logger


class Router:
    """
    Converts Slack text → correct tool call → structured Slack-safe response.
    """

    def __init__(self, sql_tool: SQLTool, pandas_tool: PandasAITool):
        self.sql_tool = sql_tool
        self.pandas_tool = pandas_tool

        # Rate limit store: user → timestamps
        self._rate_limits = defaultdict(list)
        self._max_requests = 10
        self._window_seconds = 60

        logger.info("router_initialized")

    # ------------------------------------------------------
    # Rate Limiting
    # ------------------------------------------------------
    def _check_rate_limit(self, user_id: str):
        """Return (allowed, message_if_not_allowed)."""
        now = time()

        timestamps = self._rate_limits[user_id]
        timestamps = [t for t in timestamps if now - t < self._window_seconds]
        self._rate_limits[user_id] = timestamps

        if len(timestamps) >= self._max_requests:
            return False, f"⏱️ Too many requests — limit is {self._max_requests}/minute."
        
        self._rate_limits[user_id].append(now)
        return True, None

    # ------------------------------------------------------
    # Intent Detection
    # ------------------------------------------------------
    @staticmethod
    def detect_intent(text: str) -> str:
        text = text.lower().strip()

        if text.startswith("sql:") or text.startswith("run sql"):
            return "sql"

        if "info on" in text or text.startswith("info "):
            return "dataset_info"

        if any(k in text for k in ["chart", "plot", "summarize", "analysis", "show"]):
            return "pandasai"

        return "unknown"

    # ------------------------------------------------------
    # Main Routing
    # ------------------------------------------------------
    def route(self, text: str, user_id: Optional[str] = None) -> Dict[str, Any]:
        """Route user message to the correct tool."""

        if user_id:
            allowed, err = self._check_rate_limit(user_id)
            if not allowed:
                return {"success": False, "message": err, "chart_path": None}

        intent = self.detect_intent(text)
        logger.info("router_detected_intent", intent=intent)

        try:
            if intent == "sql":
                return self._handle_sql(text)

            if intent == "dataset_info":
                return self._handle_dataset_info(text)

            if intent == "pandasai":
                return self._handle_pandasai(text)

            return {
                "success": False,
                "message": "❓ I didn't understand. Try:\n• `sql: select ...`\n• `summarize sales.csv`\n• `info on sales.csv`",
                "chart_path": None,
            }

        except Exception as e:
            logger.exception("router_error", error=str(e), text_preview=text[:50])
            return {
                "success": False,
                "message": f"⚠️ Internal error: {str(e)}",
                "chart_path": None,
            }

    # ------------------------------------------------------
    # SQL Handler
    # ------------------------------------------------------
    def _handle_sql(self, text: str) -> Dict[str, Any]:
        try:
            query = (
                text.split("sql:", 1)[-1].strip()
                if "sql:" in text.lower()
                else text.split("run sql", 1)[-1].strip()
            )

            if not query:
                return {
                    "success": False,
                    "message": "Please provide a SQL query after 'sql:'",
                    "chart_path": None,
                }

            result = self.sql_tool.run_safe_query(query)

            return {
                "success": True,
                "message": (
                    "✅ SQL executed.\n"
                    f"• Rows: {result.get('row_count', 0)}\n"
                    f"• Time: {result.get('execution_time_ms', 0):.1f}ms"
                ),
                "chart_path": None,
                "raw": result,
            }

        except Exception as e:
            logger.exception("router_sql_error", error=str(e))
            return {"success": False, "message": f"⚠️ SQL error: {str(e)}", "chart_path": None}

    # ------------------------------------------------------
    # Dataset Info Handler
    # ------------------------------------------------------
    def _handle_dataset_info(self, text: str) -> Dict[str, Any]:
        try:
            match = re.search(r"info on ([\w\.\-_]+)", text.lower())
            if not match:
                return {"success": False, "message": "Provide dataset: `info on sales.csv`"}

            filename = match.group(1)
            info = self.pandas_tool.get_dataset_info(filename)

            msg = (
                f"*Dataset:* `{filename}`\n"
                f"*Rows:* {info['rows']}\n"
                f"*Columns:* {', '.join(info['columns'][:10])}"
            )

            return {"success": True, "message": msg, "chart_path": None, "raw": info}

        except Exception as e:
            logger.exception("router_dataset_info_error", error=str(e))
            return {
                "success": False,
                "message": f"⚠️ Error getting dataset info: {str(e)}",
                "chart_path": None,
            }

    # ------------------------------------------------------
    # PandasAI Handler
    # ------------------------------------------------------
    def _handle_pandasai(self, text: str) -> Dict[str, Any]:
        try:
            match = re.search(r"([\w\.\-_]+\.(csv|parquet|xlsx|json))", text.lower())
            if not match:
                return {"success": False, "message": "Provide file: `summarize sales.csv`"}

            filename = match.group(1)

            chart_type = None
            for c in ["bar", "line", "scatter", "histogram", "pie"]:
                if c in text.lower():
                    chart_type = c
                    break

            result = self.pandas_tool.analyze(
                query=text,
                dataset_filename=filename,
                chart_type=chart_type,
            )

            if not result.get("success"):
                return {
                    "success": False,
                    "message": f"⚠️ Analysis error: {result.get('error')}",
                    "chart_path": None,
                }

            msg = (
                f"✅ Analysis completed.\n"
                f"• Dataset: `{filename}`\n"
                f"• Rows analyzed: {result['dataset_rows']}\n"
                f"• Time: {result['execution_time_ms']:.1f}ms"
            )

            if result.get("chart_path"):
                msg += "\n• Chart generated: ✅"

            return {
                "success": True,
                "message": msg,
                "chart_path": result.get("chart_path"),
                "raw": result,
            }

        except Exception as e:
            logger.exception("router_pandasai_error", error=str(e))
            return {
                "success": False,
                "message": f"⚠️ Analysis error: {str(e)}",
                "chart_path": None,
            }