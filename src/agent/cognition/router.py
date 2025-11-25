# ----------------------------------------------------------
# src/agent/cognition/router.py

"""
Routing engine for the GenAI DataOps Agent.

Improvements added:
- Rate limiting (10 req/min per user)
- router.route() now accepts user_id
- SQL + PandasAI + dataset-info routing unchanged
- Structured logging
- Semantic layer integration for dynamic metric detection
- LLM reasoning integration for complex queries (optional)
- Memory integration for conversation context (optional)
"""

import re
from time import time
from collections import defaultdict
from typing import Dict, Any, Optional

from agent.tools.sql_tool import SQLTool
from agent.tools.pandasai_tool import PandasAITool
from agent.logging_config import logger

# Optional semantic layer imports (graceful degradation if not available)
try:
    from agent.knowledge.semantic_loader import SemanticLoader
    from agent.knowledge.database import Database
    SEMANTIC_LAYER_AVAILABLE = True
except ImportError:
    SEMANTIC_LAYER_AVAILABLE = False
    Database = None
    SemanticLoader = None

# Optional LLM reasoner and memory imports (graceful degradation if not available)
try:
    from agent.cognition.llm_reasoner import LLMReasoner
    from agent.cognition.memory import Memory
    from agent.config import Settings
    LLM_REASONER_AVAILABLE = True
except ImportError:
    LLM_REASONER_AVAILABLE = False
    LLMReasoner = None
    Memory = None
    Settings = None


class Router:
    """
    Converts Slack text → correct tool call → structured Slack-safe response.
    Supports semantic layer for dynamic metric detection (optional).
    Supports LLM reasoning and memory for complex queries (optional).
    """

    def __init__(
        self,
        sql_tool: SQLTool,
        pandas_tool: PandasAITool,
        database: Optional[Database] = None,
        settings: Optional[Settings] = None
    ):
        self.sql_tool = sql_tool
        self.pandas_tool = pandas_tool

        # Rate limit store: user → timestamps
        self._rate_limits = defaultdict(list)
        self._max_requests = 10
        self._window_seconds = 60

        # Initialize semantic loader (optional - graceful degradation if fails)
        self.semantic_loader = None
        if database and SEMANTIC_LAYER_AVAILABLE:
            try:
                self.semantic_loader = SemanticLoader(database)
                logger.info("router_semantic_loader_initialized")
            except Exception as e:
                logger.warning("router_semantic_loader_init_failed", error=str(e))
                # Router works without semantic layer

        # Initialize LLM reasoner (optional - graceful degradation if fails)
        self.reasoner = None
        if settings and LLM_REASONER_AVAILABLE:
            try:
                self.reasoner = LLMReasoner(settings=settings)
                logger.info("router_llm_reasoner_initialized", enabled=self.reasoner.enabled)
            except Exception as e:
                logger.warning("router_llm_reasoner_init_failed", error=str(e))
                # Router works without LLM reasoner

        # Initialize memory (optional - graceful degradation if fails)
        self.memory = None
        if LLM_REASONER_AVAILABLE:
            try:
                self.memory = Memory()
                logger.info("router_memory_initialized")
            except Exception as e:
                logger.warning("router_memory_init_failed", error=str(e))
                # Router works without memory

        logger.info(
            "router_initialized",
            semantic_layer_enabled=self.semantic_loader is not None,
            llm_reasoner_enabled=self.reasoner is not None and self.reasoner.enabled if self.reasoner else False,
            memory_enabled=self.memory is not None
        )

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
    def detect_intent(self, text: str) -> str:
        """
        Detect intent with semantic layer support.
        Falls back to simple pattern matching if semantic layer unavailable.
        """
        text = text.lower().strip()

        # Check for explicit SQL intent
        if text.startswith("sql:") or text.startswith("run sql"):
            return "sql"

        # Check for metric intent using semantic layer (if available)
        if self.semantic_loader:
            metric = self._detect_metric_intent(text)
            if metric:
                return "sql"  # Metrics route to SQL

        # Check for dataset info
        if "info on" in text or text.startswith("info "):
            return "dataset_info"

        # Check for PandasAI intent
        if self.semantic_loader:
            routing_hints = self.semantic_loader.get_routing_hints()
            pandasai_keywords = routing_hints.get("pandasai_keywords", ["chart", "plot", "summarize", "analysis", "show"])
            if any(k in text for k in pandasai_keywords):
                return "pandasai"
        else:
            # Fallback to simple keyword matching
            if any(k in text for k in ["chart", "plot", "summarize", "analysis", "show"]):
                return "pandasai"

        return "unknown"

    def _detect_metric_intent(self, text: str) -> Optional[str]:
        """
        Detect if query is asking for a known metric using semantic layer.
        
        Args:
            text: User query text
            
        Returns:
            Metric name if detected, None otherwise
        """
        if not self.semantic_loader:
            return None

        try:
            metric_keywords = self.semantic_loader.get_metric_keywords()
            text_lower = text.lower()
            
            for metric_name, keywords in metric_keywords.items():
                if any(keyword in text_lower for keyword in keywords):
                    # Check if metric is available in current schema
                    if self.semantic_loader.is_metric_available(metric_name):
                        logger.info("router_metric_detected", metric=metric_name)
                        return metric_name
        except Exception as e:
            logger.warning("router_metric_detection_failed", error=str(e))
        
        return None

    # ------------------------------------------------------
    # Main Routing
    # ------------------------------------------------------
    def route(self, text: str, user_id: Optional[str] = None) -> Dict[str, Any]:
        """Route user message to the correct tool."""

        if user_id:
            allowed, err = self._check_rate_limit(user_id)
            if not allowed:
                return {"success": False, "message": err, "chart_path": None}

        # Try LLM-based intent detection first (if available)
        intent = None
        llm_suggested_tool = None
        
        if self.reasoner and self.reasoner.enabled and user_id:
            try:
                # Get conversation context from memory
                context = None
                if self.memory:
                    conversation_history = self.memory.get_conversation_history(user_id, max_messages=5)
                    if conversation_history:
                        context = {
                            "recent_queries": [entry["query"] for entry in conversation_history],
                            "user_id": user_id
                        }

                # Use LLM reasoner for intent interpretation
                llm_result = self.reasoner.interpret_intent(
                    query=text,
                    available_tools=["sql", "pandasai", "dataset_info"],
                    context=context
                )

                # Validate LLM response
                suggested_tool = llm_result.get("suggested_tool")
                confidence = llm_result.get("confidence", 0.0)
                
                # Use LLM suggestion if confidence is high enough
                if suggested_tool and confidence >= 0.7:
                    llm_suggested_tool = suggested_tool
                    intent = suggested_tool
                    logger.info(
                        "router_llm_intent_detected",
                        intent=intent,
                        confidence=confidence,
                        reasoning=llm_result.get("reasoning", "")[:100]
                    )
            except Exception as e:
                logger.warning("router_llm_intent_failed", error=str(e))
                # Fallback to pattern matching

        # Fallback to pattern-based intent detection if LLM didn't provide intent
        if not intent:
            intent = self.detect_intent(text)
            logger.info("router_detected_intent", intent=intent)

        try:
            # Check for metric query first (semantic layer)
            metric = self._detect_metric_intent(text)
            if metric and intent == "sql":
                response = self._handle_metric_query(metric, text)
                # Store in memory
                if self.memory and user_id:
                    self.memory.add_conversation(user_id, text, response)
                return response

            if intent == "sql":
                response = self._handle_sql(text)
                # Store in memory
                if self.memory and user_id:
                    self.memory.add_conversation(user_id, text, response)
                return response

            if intent == "dataset_info":
                response = self._handle_dataset_info(text)
                # Store in memory
                if self.memory and user_id:
                    self.memory.add_conversation(user_id, text, response)
                return response

            if intent == "pandasai":
                response = self._handle_pandasai(text)
                # Store in memory
                if self.memory and user_id:
                    self.memory.add_conversation(user_id, text, response)
                return response

            response = {
                "success": False,
                "message": "❓ I didn't understand. Try:\n• `sql: select ...`\n• `summarize sales.csv`\n• `info on sales.csv`",
                "chart_path": None,
            }
            # Store in memory even for unknown intents
            if self.memory and user_id:
                self.memory.add_conversation(user_id, text, response)
            return response

        except Exception as e:
            logger.exception("router_error", error=str(e), text_preview=text[:50])
            error_response = {
                "success": False,
                "message": f"⚠️ Internal error: {str(e)}",
                "chart_path": None,
            }
            # Store error in memory
            if self.memory and user_id:
                self.memory.add_conversation(user_id, text, error_response)
            return error_response

    # ------------------------------------------------------
    # Metric Query Handler (Semantic Layer)
    # ------------------------------------------------------
    def _handle_metric_query(self, metric_name: str, text: str) -> Dict[str, Any]:
        """
        Handle metric query using semantic layer.
        
        Args:
            metric_name: Detected metric name
            text: Original query text (for date extraction)
            
        Returns:
            Router response dict
        """
        if not self.semantic_loader:
            return {
                "success": False,
                "message": "Semantic layer not available",
                "chart_path": None,
            }

        try:
            # Extract date filter from text if present
            date_filter = None
            if "last 30 days" in text.lower():
                date_filter = "created_at >= NOW() - INTERVAL '30 days'"
            elif "this month" in text.lower():
                date_filter = "created_at >= DATE_TRUNC('month', NOW())"
            elif "last 7 days" in text.lower() or "past week" in text.lower():
                date_filter = "created_at >= NOW() - INTERVAL '7 days'"
            
            # Generate SQL from semantic layer
            sql = self.semantic_loader.generate_sql(metric_name, date_filter)
            
            if not sql:
                return {
                    "success": False,
                    "message": f"Could not generate SQL for metric: {metric_name}",
                    "chart_path": None,
                }

            # Execute SQL
            result = self.sql_tool.run_safe_query(sql)

            # Format response
            rows = result.get("rows", [])
            row_count = result.get("row_count", 0)
            execution_time = result.get("execution_time_ms", 0)

            # Extract metric value from result
            metric_value = "N/A"
            if rows and len(rows) > 0:
                if isinstance(rows[0], tuple) and len(rows[0]) > 0:
                    metric_value = rows[0][0]
                elif isinstance(rows[0], dict):
                    metric_value = list(rows[0].values())[0] if rows[0] else "N/A"

            return {
                "success": True,
                "message": (
                    f"✅ {metric_name.replace('_', ' ').title()} calculated.\n"
                    f"• Value: {metric_value}\n"
                    f"• Rows: {row_count}\n"
                    f"• Time: {execution_time:.1f}ms"
                ),
                "chart_path": None,
                "raw": result,
            }

        except Exception as e:
            logger.exception("router_metric_query_error", metric=metric_name, error=str(e))
            return {
                "success": False,
                "message": f"⚠️ Error calculating {metric_name}: {str(e)}",
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