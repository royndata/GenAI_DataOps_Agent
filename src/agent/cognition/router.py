# ----------------------------------------------------------
# src/agent/cognition/router.py
# **Changes:**
# 1. Added `_convert_natural_language_to_sql()` with regex patterns for common queries
# 2. Added `_llm_convert_to_sql()` for LLM-based conversion
# 3. Updated `_handle_sql()` to convert natural language to SQL before execution
# 4. Added logging for conversion steps

# This converts queries like "Show me total rows in sessions" → "SELECT COUNT(*) as total FROM sessions" automatically.

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
- Natural language to SQL conversion
"""

import re
from time import time
from collections import defaultdict
from typing import Dict, Any, Optional, Tuple, List

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

# Optional new module imports (graceful degradation)
try:
    from agent.knowledge.date_filter_builder import DateFilterBuilder
    from agent.knowledge.query_analyzer import QueryAnalyzer
    NEW_MODULES_AVAILABLE = True
except ImportError:
    NEW_MODULES_AVAILABLE = False
    DateFilterBuilder = None
    QueryAnalyzer = None

try:
    from agent.knowledge.prompt_manager import PromptManager
    PROMPT_MANAGER_AVAILABLE = True
except ImportError:
    PROMPT_MANAGER_AVAILABLE = False
    PromptManager = None

class Router:
    """
    Converts Slack text → correct tool call → structured Slack-safe response.
    Supports semantic layer for dynamic metric detection (optional).
    Supports LLM reasoning and memory for complex queries (optional).
    Supports natural language to SQL conversion.
    """

    def __init__(
        self,
        sql_tool: SQLTool,
        pandas_tool: PandasAITool,
        database: Optional[Database] = None,
        settings: Optional[Settings] = None,
        output_formatter: Optional[Any] = None
    ):
        self.sql_tool = sql_tool
        self.pandas_tool = pandas_tool

        # Rate limit store: user → timestamps
        self._rate_limits = defaultdict(list)
        self._max_requests = 10
        self._window_seconds = 60

        # Initialize LLM reasoner (optional - graceful degradation if fails)
        # Initialize LLM reasoner FIRST (needed by semantic loader)
        self.reasoner = None
        if settings and LLM_REASONER_AVAILABLE:
            try:
                self.reasoner = LLMReasoner(settings=settings)
                logger.info("router_llm_reasoner_initialized", enabled=self.reasoner.enabled)
            except Exception as e:
                logger.warning("router_llm_reasoner_init_failed", error=str(e))
                # Router works without LLM reasoner        

        # Initialize semantic loader (optional - graceful degradation if fails)
        # Now reasoner is available to pass to SemanticLoader
        self.semantic_loader = None
        if database and SEMANTIC_LAYER_AVAILABLE:
            try:
                # Pass LLMReasoner to SemanticLoader for dynamic mapping
                reasoner_for_semantic = self.reasoner if (self.reasoner and self.reasoner.enabled) else None
                self.semantic_loader = SemanticLoader(database, reasoner=reasoner_for_semantic)
                logger.info("router_semantic_loader_initialized", llm_enabled=reasoner_for_semantic is not None)
            except Exception as e:
                logger.warning("router_semantic_loader_init_failed", error=str(e))
                # Router works without semantic layer

        # Initialize date filter builder (optional)
        self.date_filter_builder = None
        if database and NEW_MODULES_AVAILABLE and self.semantic_loader:
            try:
                self.date_filter_builder = DateFilterBuilder(self.semantic_loader.schema_discovery)
                logger.info("router_date_filter_builder_initialized")
            except Exception as e:
                logger.warning("router_date_filter_builder_init_failed", error=str(e))
        
        # Initialize query analyzer (optional)
        self.query_analyzer = None
        if database and NEW_MODULES_AVAILABLE and self.semantic_loader:
            try:
                self.query_analyzer = QueryAnalyzer(self.semantic_loader.schema_discovery)
                logger.info("router_query_analyzer_initialized")
            except Exception as e:
                logger.warning("router_query_analyzer_init_failed", error=str(e))
        
        # Initialize prompt manager (optional)
        self.prompt_manager = None
        if PROMPT_MANAGER_AVAILABLE:
            try:
                self.prompt_manager = PromptManager()
                logger.info("router_prompt_manager_initialized")
            except Exception as e:
                logger.warning("router_prompt_manager_init_failed", error=str(e))
        
        # Initialize output formatter (optional)
        self.output_formatter = output_formatter
        if output_formatter:
            logger.info("router_output_formatter_initialized")

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

        # Check for explicit SQL intent (more comprehensive)
        text_upper = text.upper().strip()
        sql_keywords = ["SELECT", "WITH", "INSERT", "UPDATE", "DELETE", "CREATE", "ALTER", "DROP"]
        if (text.startswith("sql:") or 
            text.startswith("run sql") or
            any(text_upper.startswith(kw) for kw in sql_keywords)):
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
        Returns metric name if keyword is found, regardless of schema availability.
        Schema mapping will be attempted later and may trigger confirmation.
        
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
                    # Return metric name if keyword matches
                    # Don't check availability here - let mapping attempt happen later
                    # This allows LLM mapping to work and trigger confirmation if needed
                    logger.info("router_metric_detected", metric=metric_name)
                    return metric_name
        except Exception as e:
            logger.warning("router_metric_detection_failed", error=str(e))
        
        return None
    
    def _detect_query_complexity(self, text: str, metric_name: Optional[str] = None) -> str:
        """
        Detect query complexity type.
        
        Args:
            text: User query text
            metric_name: Optional metric name if detected
            
        Returns:
            Complexity type: "simple", "multi_table", "time_series", "ranking", "conditional"
        """
        text_lower = text.lower()
        
        # Check for multi-table indicators
        if any(keyword in text_lower for keyword in ["join", "with", "across", "between", "compare"]):
            return "multi_table"
        
        # Check for time series indicators
        if any(keyword in text_lower for keyword in ["daily", "weekly", "monthly", "by day", "by week", "over time", "trend"]):
            return "time_series"
        
        # Check for ranking indicators
        if any(keyword in text_lower for keyword in ["top", "bottom", "rank", "highest", "lowest", "best", "worst"]):
            return "ranking"
        
        # Check for conditional indicators
        if any(keyword in text_lower for keyword in ["if", "when", "case", "conditional", "depending"]):
            return "conditional"
        
        return "simple"    

    # ------------------------------------------------------
    # Main Routing
    # ------------------------------------------------------
    # src/agent/cognition/router.py

    def route(self, text: str, user_id: Optional[str] = None) -> Dict[str, Any]:
        """Route user message to the correct tool."""

        if user_id:
            allowed, err = self._check_rate_limit(user_id)
            if not allowed:
                return {"success": False, "message": err, "chart_path": None}

        # Check for clear/reset commands (like "clear" in terminal)
        if user_id and self.memory:
            text_lower = text.lower().strip()
            clear_commands = ["clear", "reset", "forget", "start fresh", "new chat", "clear memory", "reset chat"]
            if any(cmd in text_lower for cmd in clear_commands):
                self.memory.clear_user_memory(user_id)
                logger.info("router_memory_cleared", user_id=user_id, command=text)
                return {
                    "success": True,
                    "message": "✅ Chat history cleared. Starting fresh!",
                    "chart_path": None,
                }
        # Check for pending confirmation response (yes/no)
        if user_id and self.memory:
            pending_confirmation = self.memory.get_pending_confirmation(user_id)
            if pending_confirmation:
                text_lower = text.lower().strip()
                if text_lower in ["yes", "y", "confirm", "proceed"]:
                    # User confirmed, proceed with calculation
                    return self._execute_confirmed_query(pending_confirmation, user_id)
                elif text_lower in ["no", "n", "cancel", "abort"]:
                    # User declined
                    concept = pending_confirmation.get("concept", "this metric")
                    table = pending_confirmation.get("table", "the table")
                    self.memory.clear_pending_confirmation(user_id)
                    return {
                        "success": False,
                        "message": f"❌ Cannot calculate {concept}: not part of {table}",
                        "chart_path": None
                    }
                # If not yes/no, continue with normal routing

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
            # Check for explicit SQL first (should bypass metric detection and validation)
            # More comprehensive SQL detection
            text_upper = text.upper().strip()
            is_explicit_sql = (
                text.strip().lower().startswith("sql:") or 
                text.strip().lower().startswith("select") or
                text.strip().lower().startswith("with") or
                any(text_upper.startswith(kw) for kw in ["SELECT", "WITH", "INSERT", "UPDATE", "DELETE"])
            )
            
            if intent == "sql" and is_explicit_sql:
                # Explicit SQL query - route directly to SQL handler (bypass validation)
                response = self._handle_sql(text)
                if self.memory and user_id:
                    self.memory.add_conversation(user_id, text, response)
                return response
            
            # Check for metric query (semantic layer) - OVERRIDE LLM intent if metric detected
            metric = self._detect_metric_intent(text)
            if metric:
                # Metric detected - force SQL routing (metrics are database queries, not file analysis)
                logger.info("router_metric_overrides_intent", metric=metric, llm_intent=intent)
                response = self._handle_metric_query(metric, text, user_id=user_id)
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

            # Validate query intent for non-SQL routes (dataset_info, pandasai)
            # Skip validation for explicit SQL (already handled above)
            if intent in ["dataset_info", "pandasai"]:
                # Validate that query is relevant to database
                is_valid_intent, intent_error = self._validate_query_intent(text, None)
                if not is_valid_intent:
                    return {
                        "success": False,
                        "message": f"❌ {intent_error}",
                        "chart_path": None,
                    }
            
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

            # Validate query intent for unknown queries too
            if intent == "unknown":
                # Check if it's an unrelated query (weather, personal questions, etc.)
                is_valid_intent, intent_error = self._validate_query_intent(text, None)
                if not is_valid_intent:
                    return {
                        "success": False,
                        "message": f"❌ {intent_error}",
                        "chart_path": None,
                    }
            
            response = {
                "success": False,
                "message": "❓ I didn't understand. Try:\n• `sql: select ...`\n• `summarize sales.csv`\n• `info on sales.csv`",
                "chart_path": None,
            }

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
    def _handle_metric_query(self, metric_name: str, text: str, user_id: Optional[str] = None) -> Dict[str, Any]:
        """
        Handle metric query using semantic layer.
        Now supports confirmation for inferred mappings.
        
        Args:
            metric_name: Detected metric name
            text: Original query text (for date extraction)
            user_id: User ID for confirmation tracking
            
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

            # (date filter extraction and building logic)
            # Extract date filter using DateFilterBuilder
            date_filter = None
            date_filter_result = None
            date_column = None
            if self.date_filter_builder:
                requirements = self.date_filter_builder.extract_date_requirements(text)
                if requirements.has_date_filter:
                    # Will build date filter after we know the table from mapping
                    date_filter_result = requirements
            
            # Diagnostic logging to understand mapping state
            logger.info(
                "router_mapping_attempt",
                metric=metric_name,
                has_semantic_loader=self.semantic_loader is not None,
                has_reasoner=self.semantic_loader.reasoner is not None if self.semantic_loader else False,
                reasoner_enabled=self.semantic_loader.reasoner.enabled if (self.semantic_loader and self.semantic_loader.reasoner) else False
            )

            # Validate query intent
            is_valid_intent, intent_error = self._validate_query_intent(text, metric_name)
            if not is_valid_intent:
                return {
                    "success": False,
                    "message": f"❌ {intent_error}",
                    "chart_path": None,
                }

            # Map metric to schema (may return needs_confirmation flag)
            mapping = self.semantic_loader.map_metric_to_schema(metric_name)
            
            # Check if metric pattern requires date filter and build default if needed
            if mapping and self.date_filter_builder:
                # Extract date requirements from user query (only if explicitly mentioned)
                date_requirements = self.date_filter_builder.extract_date_requirements(text)
                
                # ONLY add date filter if user explicitly requested it
                if date_requirements.has_date_filter:
                    date_filter_result = self.date_filter_builder.build_date_filter(
                        mapping["table"], 
                        date_requirements
                    )
                    logger.info("router_using_explicit_date_filter", metric=metric_name, date_type=date_requirements.date_type)
                # DO NOT add default date filter - respect user intent

            # Detect query complexity BEFORE confirmation
            # If query is complex, skip confirmation and use complex SQL generator
            complexity = None
            is_complex_query = False
            if self.query_analyzer:
                try:
                    # Get metric table from mapping for semantic inference
                    metric_table = mapping.get("table") if mapping else None
                    complexity = self.query_analyzer.analyze_query_complexity(
                        text, 
                        metric_table=metric_table
                    )
                    # Check if query requires complex features
                    is_complex_query = (
                        complexity.complexity in ["medium", "high"] or
                        complexity.requires_joins or
                        complexity.requires_grouping or
                        complexity.requires_having or
                        complexity.requires_window_functions or
                        complexity.requires_subqueries or
                        len(complexity.metrics_mentioned) > 1  # Multiple metrics
                    )
                except Exception as e:
                    logger.warning("router_complexity_detection_failed", error=str(e))
                    # Fall through to simple handling

            # Build date filter using DateFilterBuilder ONLY if user explicitly requested it
            # date_filter_result was already built at line 505 if user requested it
            if date_filter_result and hasattr(date_filter_result, 'date_filter'):
                # date_filter_result is already a DateFilterResult object from line 505
                date_filter = date_filter_result.date_filter
                date_column = date_filter_result.date_column
            else:
                # No date filter requested by user
                date_filter = None
                date_column = None
                    
            if not mapping:
                # Provide more helpful error message
                reasoner_available = self.semantic_loader.reasoner is not None if self.semantic_loader else False
                reasoner_enabled = self.semantic_loader.reasoner.enabled if (self.semantic_loader and self.semantic_loader.reasoner) else False
                
                error_msg = f"Could not map metric '{metric_name}' to database schema."
                if not reasoner_available:
                    error_msg += "\n\n💡 Tip: LLM mapping is not available. Enable LLM reasoner to map metrics dynamically."
                elif not reasoner_enabled:
                    error_msg += "\n\n💡 Tip: LLM reasoner is available but not enabled. Check your configuration."
                else:
                    error_msg += "\n\n💡 Tip: No matching tables/columns found. Check your database schema matches the semantic layer patterns."
                
                return {
                    "success": False,
                    "message": error_msg,
                    "chart_path": None,
                }

            # Check if confirmation is needed (skip for complex queries)
            if mapping.get("needs_confirmation") and user_id and not is_complex_query:
                
                # Store pending confirmation
                if self.memory:
                    self.memory.set_pending_confirmation(
                        user_id,
                        {
                            "metric_name": metric_name,
                            "concept": mapping.get("concept", metric_name),
                            "table": mapping["table"],
                            "column": mapping["column"],
                            "aggregation": mapping.get("aggregation", "SUM"),
                            "date_filter": date_filter,
                            "date_column": date_column,
                            "text": text
                        }
                    )
                
                # Extract variables from mapping for use below
                concept = mapping.get("concept", metric_name)
                table = mapping["table"]
                column = mapping["column"]
                aggregation = mapping.get("aggregation", "SUM")
                
                # Get available columns and LLM-suggested expression
                schema = self.semantic_loader.schema_discovery.discover_full_schema()
                available_columns = []
                if table in schema:
                    available_columns = [col["name"] for col in schema[table]]
                
                # Get LLM-suggested expression
                llm_suggestion = None
                if self.semantic_loader and available_columns:
                    llm_suggestion = self.semantic_loader.suggest_expression_with_llm(
                        concept=concept,
                        table=table,
                        available_columns=available_columns
                    )
                
                # Build enhanced confirmation message
                message_parts = [
                    f"⚠️ *Confirmation Required*\n\n",
                    f"`{concept}` is not a column in `{table}` table.\n\n"
                ]
                
                # Show available columns
                if available_columns:
                    columns_display = "• " + "\n• ".join(available_columns[:10])
                    if len(available_columns) > 10:
                        columns_display += f"\n• ... and {len(available_columns) - 10} more"
                    message_parts.append(f"*Available columns:*\n{columns_display}\n\n")
                
                # Show LLM-suggested expression or fallback
                if llm_suggestion:
                    suggested_expression = llm_suggestion.get("expression", f"{aggregation}({column}) as {concept}")
                    suggested_column = llm_suggestion.get("column", column)
                    
                    # Clean up suggested_expression - remove SELECT, FROM, WHERE clauses if present
                    # The expression should only be: "COUNT(DISTINCT user_id) AS active_users"
                    if suggested_expression.upper().strip().startswith("SELECT"):
                        suggested_expression = suggested_expression.replace("SELECT", "", 1).strip()
                    
                    # Remove FROM clause and everything after it (including WHERE)
                    # Handle both " FROM " and "FROM " patterns
                    if " FROM " in suggested_expression.upper():
                        suggested_expression = suggested_expression.split(" FROM ", 1)[0].strip()
                    elif "FROM " in suggested_expression.upper():
                        suggested_expression = suggested_expression.split("FROM ", 1)[0].strip()
                    
                    # Remove WHERE clause if it somehow got included before FROM
                    if " WHERE " in suggested_expression.upper():
                        suggested_expression = suggested_expression.split(" WHERE ", 1)[0].strip()
                    
                    # Build WHERE clause only if user explicitly requested date filter
                    # date_filter should be None if user didn't request it (checked at line 540)
                    where_clause = ""
                    if date_filter:
                        where_clause = f"WHERE {date_filter}"
                    
                    message_parts.append(
                        f"*Suggested calculation:*\n"
                        f"Do you want me to use this expression:\n"
                        f"```sql\n"
                        f"SELECT {suggested_expression}\n"
                        f"FROM {table}\n"
                        f"{where_clause}\n"
                        f"```\n"
                        f"to calculate `{concept}`?\n\n"
                    )

                    # Update stored confirmation data with LLM suggestion
                    if self.memory:
                        stored_confirmation = self.memory.get_pending_confirmation(user_id)
                        if stored_confirmation:
                            stored_confirmation["column"] = suggested_column
                            stored_confirmation["aggregation"] = llm_suggestion.get("aggregation", aggregation)
                            self.memory.set_pending_confirmation(user_id, stored_confirmation)    

                else:
                    # Build WHERE clause only if user requested date filter
                    where_clause = f"WHERE {date_filter}" if date_filter else ""
                    
                    message_parts.append(
                        f"*Calculation Logic:*\n"
                        f"```sql\n"
                        f"SELECT {aggregation}({column}) as {concept}\n"
                        f"FROM {table}\n"
                        f"{where_clause}\n"
                        f"```\n\n"
                        f"Do you want me to calculate {concept} based on `{column}` column?\n\n"
                    )
                
                message_parts.append("Reply *yes* to proceed or *no* to cancel.")
                message = "".join(message_parts)
                
                return {
                    "success": False,
                    "message": message,
                    "chart_path": None,
                    "requires_confirmation": True
                }

            # Handle complex queries directly (skip confirmation)
            if is_complex_query and mapping:
                # Generate complex SQL directly
                sql = self.semantic_loader.generate_sql(metric_name, date_filter, query_text=text)
                
                if not sql:
                    return {
                        "success": False,
                        "message": f"Could not generate SQL for complex query: {text}",
                        "chart_path": None,
                    }
                
                # Execute SQL
                result = self.sql_tool.run_safe_query(sql)
                
                # Extract result data
                rows = result.get("rows", [])
                row_count = result.get("row_count", 0)
                execution_time = result.get("execution_time_ms", 0)
                
                # Format response with better information
                message = f"✅ Complex query executed.\n"
                message += f"• Rows: {row_count}\n"
                message += f"• Time: {execution_time:.1f}ms\n"
                
                # Analyze empty result if no rows returned
                if row_count == 0:
                    # Try to analyze why no results
                    # Extract table name from SQL for analysis
                    table_for_analysis = None
                    if "FROM" in sql.upper():
                        # Try to extract first table name
                        from_match = re.search(r'FROM\s+(\w+)', sql, re.IGNORECASE)
                        if from_match:
                            table_for_analysis = from_match.group(1)
                    
                    # Analyze empty result
                    if table_for_analysis:
                        analysis = self._analyze_empty_result(sql, table_for_analysis, date_filter)
                        if analysis:
                            message += f"\n⚠️ {analysis}"
                    else:
                        message += "\n⚠️ No results found. The query executed successfully but returned no rows."
                    
                    # Show the SQL for debugging
                    message += f"\n\n*Executed SQL:*\n```sql\n{sql}\n```"
                else:
                    # Always show the executed SQL for complex queries (for transparency)
                    message += f"\n\n*Executed SQL:*\n```sql\n{sql}\n```"
                    
                    # Show table after SQL if available
                    if rows and len(rows) > 0:
                        # Use output formatter for consistent table formatting
                        columns = result.get("columns", [])
                        if self.output_formatter:
                            table_preview = self.output_formatter._format_table(rows[:min(10, len(rows))], columns)
                            if table_preview:
                                message += f"\n\n*Results:*\n```\n{table_preview}\n```"
                                if row_count > min(10, len(rows)):
                                    message += f"\n... and {row_count - min(10, len(rows))} more rows"
                        else:
                            # Fallback to simple preview
                            preview_rows = min(3, len(rows))
                            message += f"\n\n*Results (showing {preview_rows} of {row_count} rows):*\n"
                            for i, row in enumerate(rows[:preview_rows]):
                                if isinstance(row, (tuple, list)):
                                    message += f"Row {i+1}: {', '.join(str(v) for v in row[:5])}\n"
                                elif isinstance(row, dict):
                                    message += f"Row {i+1}: {', '.join(f'{k}={v}' for k, v in list(row.items())[:5])}\n"
                            if row_count > preview_rows:
                                message += f"\n... and {row_count - preview_rows} more rows"
                return {
                    "success": True,
                    "message": message,
                    "chart_path": None,
                    "raw": result,
                }

            # Generate SQL from semantic layer (pass query_text for complexity detection)
            sql = self.semantic_loader.generate_sql(metric_name, date_filter, query_text=text)
            
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

    def _validate_date_range(self, table: str, date_filter: Optional[str] = None, date_column: Optional[str] = None) -> Tuple[bool, Optional[str]]:
        """
        Validate if date filter is within available data range.
        Uses DateFilterBuilder if available.
        
        Args:
            table: Table name
            date_filter: Date filter string
            date_column: Date column name (optional, will be discovered if not provided)
            
        Returns:
            (is_valid, error_message)
        """
        if not date_filter:
            return True, None
        
        # Use DateFilterBuilder if available
        if self.date_filter_builder:
            return self.date_filter_builder.validate_date_range(table, date_filter)
        
        # Fallback to basic validation
        if not self.semantic_loader:
            return True, None
        
        try:
            # Discover date column if not provided
            if not date_column:
                schema = self.semantic_loader.schema_discovery.discover_full_schema()
                # Try to discover date column from schema directly (without date_filter_builder)
                # Look for date columns in schema
                if table in schema:
                    date_columns = [
                        col["name"] for col in schema[table]
                        if any(dt in col.get("type", "").lower() for dt in ["date", "timestamp", "time"])
                    ]
                    if date_columns:
                        discovered_date_col = date_columns[0]
                    else:
                        discovered_date_col = None
                else:
                    discovered_date_col = None
                date_column = discovered_date_col or "created_at"
            
            # Get date range for table
            date_range = self.semantic_loader.schema_discovery.get_table_date_range(table, date_column)
            
            if not date_range:
                return True, None  # Can't validate, allow query
            
            min_date = date_range.get("min_date")
            max_date = date_range.get("max_date")
            
            if not min_date or not max_date:
                return True, None  # Can't validate, allow query
            
            # Extract year from date filter (simple check)
            import re
            year_match = re.search(r'(\d{4})', date_filter)
            if year_match:
                filter_year = int(year_match.group(1))
                max_year = max_date.year if hasattr(max_date, 'year') else None
                min_year = min_date.year if hasattr(min_date, 'year') else None
                
                if max_year and filter_year > max_year + 1:  # Allow 1 year buffer
                    return False, f"Date {filter_year} is outside available data range ({min_year}-{max_year})"
                if min_year and filter_year < min_year - 1:  # Allow 1 year buffer
                    return False, f"Date {filter_year} is outside available data range ({min_year}-{max_year})"
            
            return True, None
            
        except Exception as e:
            logger.warning("router_date_validation_failed", error=str(e))
            return True, None  # On error, allow query
                
    def _analyze_empty_result(self, sql: str, table: str, date_filter: Optional[str] = None) -> Optional[str]:
        """
        Analyze why query returned no results and generate helpful message.
        
        Args:
            sql: SQL query that was executed
            table: Table name
            date_filter: Date filter that was used
            
        Returns:
            Error message explaining why no results, or None if can't determine
        """
        if not self.semantic_loader:
            return None
        
        try:
            # Check if table has any data at all
            count_query = f"SELECT COUNT(*) FROM {table}"
            count_result = self.sql_tool.run_safe_query(count_query)
            
            if not count_result or not count_result.get("rows"):
                return f"Table {table} is empty (no rows)."
            
            row_count = count_result["rows"][0][0] if count_result["rows"] else 0
            if row_count == 0:
                return f"Table {table} is empty (no rows)."
            
            # Check date range if date filter was used
            if date_filter:
                # Discover date column from schema
                schema = self.semantic_loader.schema_discovery.discover_full_schema()
                date_col = None
                
                # Use DateFilterBuilder if available
                if self.date_filter_builder:
                    date_col = self.date_filter_builder.discover_date_column(table, schema)
                else:
                    # Fallback: find first date/timestamp column
                    if table in schema:
                        for col in schema[table]:
                            col_type = col.get("type", "").lower()
                            if any(date_type in col_type for date_type in ["date", "timestamp", "timestamptz"]):
                                date_col = col["name"]
                                break
                
                if date_col:
                    # Get actual date range from table
                    date_range_query = f"SELECT MIN({date_col}) as min_date, MAX({date_col}) as max_date FROM {table}"
                    date_range_result = self.sql_tool.run_safe_query(date_range_query)
                    
                    if date_range_result and date_range_result.get("rows"):
                        min_date = date_range_result["rows"][0][0]
                        max_date = date_range_result["rows"][0][1]
                        
                        if min_date and max_date:
                            return f"No data found for the specified date range. Available data in {table}: {min_date} to {max_date}"
            
            # Table has data but query returned no results - likely a filter/join issue
            return f"No data found in {table} matching the specified criteria. Table has {row_count} total rows."
            
        except Exception as e:
            logger.warning("router_empty_result_analysis_failed", error=str(e))
            return f"Unable to analyze empty result: {str(e)}"
    
    def _validate_query_intent(self, query: str, metric_name: Optional[str] = None) -> Tuple[bool, Optional[str]]:
        """
        Validate if query makes sense for the database using LLM-based validation.
        
        Args:
            query: User query text
            metric_name: Detected metric name (if any)
            
        Returns:
            (is_valid, error_message)
        """
        if not self.semantic_loader:
            return True, None  # Can't validate without semantic loader
        
        # Get database context
        context = self.semantic_loader.get_database_context()
        available_tables = context.get("available_tables", [])
        available_metrics = context.get("available_metrics", [])
        
        # Use LLM to validate query relevance
        if self.reasoner and self.reasoner.enabled:
            try:
                is_valid, error_msg = self._llm_validate_query_relevance(
                    query, available_tables, available_metrics
                )
                return is_valid, error_msg
            except Exception as e:
                logger.warning("router_llm_intent_validation_failed", error=str(e))
                # Fall through to allow query if LLM validation fails
        
        # If no LLM reasoner, allow query (can't validate)
        return True, None
    
    def _llm_validate_query_relevance(
        self, 
        query: str, 
        available_tables: List[str], 
        available_metrics: List[str]
    ) -> Tuple[bool, Optional[str]]:
        """
        Use LLM to validate if query is related to database schema.
        
        Args:
            query: User query
            available_tables: List of available tables
            available_metrics: List of available metrics
            
        Returns:
            (is_valid, error_message)
        """
        if not self.reasoner or not self.reasoner.enabled:
            return True, None
        
        # Hardcoded bypass for common database queries that LLM might incorrectly reject
        query_lower = query.lower()
        database_keywords = ["session duration", "average session", "average duration", "active users", "revenue", "payments", "subscriptions"]
        if any(keyword in query_lower for keyword in database_keywords):
            return True, None
        
        try:
            import json
            
            # Get schema info for context
            schema_info = ""
            if self.semantic_loader:
                schema = self.semantic_loader.schema_discovery.discover_full_schema()
                schema_info = f"Tables: {', '.join(list(schema.keys())[:10])}"
            
            # Use PromptManager if available
            if self.prompt_manager:
                prompt = self.prompt_manager.build_query_relevance_prompt(
                    query=query,
                    available_tables=available_tables,
                    available_metrics=available_metrics,
                    schema_info=schema_info
                )
            else:
                # Fallback to old hardcoded prompt (for backward compatibility)
                prompt = f"""Determine if this query can be answered from the database.

Available tables: {', '.join(available_tables[:10])}
Available metrics: {', '.join(available_metrics[:10])}
{schema_info}

Query: {query}

ANALYSIS PROCESS:
1. Check if query mentions entities/concepts that semantically match available tables/metrics
2. Consider synonyms (e.g., "sales" matches "revenue", "customers" matches "users", "duration" matches "session duration")
3. If query is about unrelated topics (weather, movies, general knowledge, personal questions), return false
4. If query is about database entities or business data, return true

Examples of IRRELEVANT queries:
- "What is your name?" (personal question)
- "What's the weather?" (unrelated domain)
- "Tell me about movies" (unrelated domain)

Examples of RELEVANT queries:
- Questions about data in the database (revenue, users, orders, transactions)
- Questions about metrics, KPIs, analytics
- Questions about business data, trends, aggregations
- Questions about session data, duration, subscriptions, payments
- Questions about averages, counts, sums of database columns
- "What's the average session duration?" (database metric)
- "How many subscriptions are active?" (database query)
- "Show me revenue by country" (database aggregation)

IMPORTANT: 
- "session duration", "average duration", "subscriptions", "active subscriptions" are all database-related
- Be lenient - if query could be answered from database, return true
- Only reject clearly unrelated queries (general knowledge, personal, external domains)

Respond in JSON only:
{{
    "is_relevant": true/false,
    "reason": "brief explanation of why this query is or isn't relevant to the database"
}}"""
            
            response = self.reasoner._call_llm(prompt)
            
            # Parse JSON response
            if "```json" in response:
                json_start = response.find("```json") + 7
                json_end = response.find("```", json_start)
                response = response[json_start:json_end].strip()
            elif "```" in response:
                json_start = response.find("```") + 3
                json_end = response.find("```", json_start)
                response = response[json_start:json_end].strip()
            
            # Extract JSON object
            json_start = response.find("{")
            if json_start != -1:
                brace_count = 0
                json_end = -1
                for i in range(json_start, len(response)):
                    if response[i] == "{":
                        brace_count += 1
                    elif response[i] == "}":
                        brace_count -= 1
                        if brace_count == 0:
                            json_end = i + 1
                            break
                if json_end != -1:
                    response = response[json_start:json_end].strip()
            
            result = json.loads(response)
            
            if not result.get("is_relevant", True):
                return False, "Please reframe your question: it is not related to the database."
            
            return True, None
            
        except Exception as e:
            logger.warning("router_llm_query_validation_failed", error=str(e))
            return True, None  # On error, allow query
    
    def _generate_contextual_error(self, error_type: str, context: Dict[str, Any]) -> str:
        """
        Generate contextual error message based on error type.
        
        Args:
            error_type: Type of error ("no_data", "invalid_date", "invalid_concept", "nonsensical")
            context: Context dict with relevant information
            
        Returns:
            Formatted error message
        """
        db_name = context.get("database_name", "the database")
        
        if error_type == "no_data":
            table = context.get("table", "table")
            date_range = context.get("date_range")
            if date_range:
                return f"Cannot calculate: No data found in {table}. Available date range: {date_range}"
            return f"Cannot calculate: No data found in {table} for the specified criteria."
        
        elif error_type == "invalid_date":
            date = context.get("date", "date")
            date_range = context.get("date_range", "unknown range")
            return f"Cannot calculate: Date {date} is outside available data range ({date_range})"
        
        elif error_type == "invalid_concept":
            concept = context.get("concept", "concept")
            available = context.get("available_metrics", [])
            if available:
                metrics_list = ", ".join(available[:5])
                return f"Concept '{concept}' is not available. Available metrics: {metrics_list}"
            return f"Concept '{concept}' is not available in {db_name}."
        
        elif error_type == "nonsensical":
            available = context.get("available_metrics", [])
            if available:
                metrics_list = ", ".join(available[:5])
                return f"This question cannot be answered from {db_name}. Please ask about: {metrics_list}"
            return f"This question cannot be answered from {db_name}. Please reframe your question based on available data."
        
        return "Cannot calculate: Unknown error occurred."
    def _execute_confirmed_query(self, confirmation_data: Dict[str, Any], user_id: str) -> Dict[str, Any]:
        """
        Execute query after user confirmation. Validates column exists and finds correct column if needed.
        """
        try:
            metric_name = confirmation_data.get("metric_name")
            table = confirmation_data.get("table")
            column = confirmation_data.get("column")
            aggregation = confirmation_data.get("aggregation", "SUM")
            date_filter = confirmation_data.get("date_filter")
            concept = confirmation_data.get("concept", metric_name)

            # Validate query intent first
            query_text = confirmation_data.get("text", "")
            is_valid_intent, intent_error = self._validate_query_intent(query_text, metric_name)
            if not is_valid_intent:
                if self.memory:
                    self.memory.clear_pending_confirmation(user_id)
                return {
                    "success": False,
                    "message": f"❌ {intent_error}",
                    "chart_path": None
                }

            # Validate date range if date filter exists
            if date_filter:
                # Get date column using DateFilterBuilder
                date_column = confirmation_data.get("date_column")
                
                if not date_column and self.date_filter_builder:
                    # Discover from schema
                    schema = self.semantic_loader.schema_discovery.discover_full_schema() if self.semantic_loader else None
                    discovered_date_col = self.date_filter_builder.discover_date_column(table, schema)
                    date_column = discovered_date_col
                
                if not date_column and self.semantic_loader:
                    # Fallback: try to get from schema mapping
                    mapping = self.semantic_loader.map_metric_to_schema(metric_name)
                    if mapping:
                        date_column = mapping.get("date_column")
                
                # Last resort fallback
                if not date_column:
                    date_column = "created_at"
                    logger.warning("router_using_fallback_date_column", table=table)
                    
                is_valid_date, date_error = self._validate_date_range(table, date_filter, date_column)

                if not is_valid_date:
                    if self.memory:
                        self.memory.clear_pending_confirmation(user_id)
                    return {
                        "success": False,
                        "message": f"❌ {date_error}",
                        "chart_path": None
                    }
            
            # Validate and find correct column from actual database schema
            actual_column = column
            if self.semantic_loader and self.semantic_loader.schema_discovery:
                try:
                    # Get actual columns from the table
                    schema = self.semantic_loader.schema_discovery.discover_full_schema()
                    
                    if table in schema:
                        actual_columns = [col["name"] for col in schema[table]]
                        
                        # Check if suggested column exists
                        if column not in actual_columns:
                            # Try to find a matching column (e.g., "amount" → "price", "value", etc.)
                            column_lower = column.lower()
                            matching_columns = [
                                col for col in actual_columns 
                                if column_lower in col.lower() or col.lower() in column_lower
                            ]
                            
                            # Also try common revenue-related column names
                            if not matching_columns and ("revenue" in metric_name.lower() or "amount" in column_lower or "price" in column_lower):
                                revenue_keywords = ["amount", "price", "value", "total", "revenue", "cost", "fee"]
                                for keyword in revenue_keywords:
                                    matches = [col for col in actual_columns if keyword in col.lower()]
                                    if matches:
                                        matching_columns = matches
                                        break
                            
                            if matching_columns:
                                actual_column = matching_columns[0]
                                logger.info(
                                    "router_column_corrected",
                                    original=column,
                                    corrected=actual_column,
                                    table=table
                                )
                            else:
                                # No matching column found - return error with available columns
                                available_columns = ", ".join(actual_columns[:10])  # Show first 10
                                if self.memory:
                                    self.memory.clear_pending_confirmation(user_id)
                                return {
                                    "success": False,
                                    "message": (
                                        f"❌ Column `{column}` not found in `{table}`.\n\n"
                                        f"Available columns: {available_columns}\n"
                                        f"Please specify the correct column name."
                                    ),
                                    "chart_path": None
                                }
                        else:
                            # Column exists, use it
                            actual_column = column
                    else:
                        # Table not found - this shouldn't happen but handle it
                        logger.warning("router_table_not_found", table=table)
                        if self.memory:
                            self.memory.clear_pending_confirmation(user_id)
                        return {
                            "success": False,
                            "message": f"❌ Table `{table}` not found in database.",
                            "chart_path": None
                        }
                except Exception as e:
                    logger.warning("router_schema_validation_failed", error=str(e), table=table, column=column)
                    # Continue with original column if schema validation fails
                    actual_column = column
            
            # Validate column for aggregation before building SQL
            if self.semantic_loader:
                is_valid, error_msg = self.semantic_loader.validate_column_for_aggregation(
                    column_name=actual_column,
                    aggregation=aggregation,
                    table=table
                )
                if not is_valid:
                    if self.memory:
                        self.memory.clear_pending_confirmation(user_id)
                    return {
                        "success": False,
                        "message": f"❌ {error_msg}\n\nCannot calculate {concept}: invalid column type for {aggregation}.",
                        "chart_path": None
                    }

            # Only add date filter if user explicitly requested it
            # Check if user query explicitly mentions a date/time period
            if not date_filter and self.date_filter_builder and query_text:
                # Extract date requirements from user query (only if explicitly mentioned)
                date_requirements = self.date_filter_builder.extract_date_requirements(query_text)
                
                # ONLY add date filter if user explicitly requested it
                if date_requirements.has_date_filter:
                    schema = self.semantic_loader.schema_discovery.discover_full_schema() if self.semantic_loader else None
                    date_filter_result = self.date_filter_builder.build_date_filter(
                        table, date_requirements, schema
                    )
                    date_filter = date_filter_result.date_filter
                    date_column = date_filter_result.date_column
                    logger.info("router_execute_using_explicit_date_filter", metric=metric_name, date_type=date_requirements.date_type)
                # DO NOT add default date filter - respect user intent
            
            # Build WHERE clause (only if date filter exists)
            where_clause = ""
            if date_filter:
                where_clause = f"WHERE {date_filter}"
            # DO NOT add default WHERE clause - only add if user explicitly requested date filter
            
            # Build HAVING clause if needed (for conditional aggregations like "more than X")
            having_clause = ""
            # Check query text for conditional aggregations
            query_text = confirmation_data.get("text", "")
            if query_text:
                query_lower = query_text.lower()
                # Patterns like "more than 100", "greater than X", "at least X"
                if any(phrase in query_lower for phrase in ["more than", "greater than", "at least", "over", "above"]):
                    # Extract number from query (simplified - could be enhanced)
                    import re
                    number_match = re.search(r'(?:more than|greater than|at least|over|above)\s+(\d+)', query_lower)
                    if number_match:
                        threshold = number_match.group(1)
                        # Add HAVING clause for aggregation threshold
                        having_clause = f"HAVING {aggregation}({actual_column}) > {threshold}"
            
            # Build SQL query with validated/corrected column
            if aggregation == "COUNT DISTINCT":
                sql = f"SELECT COUNT(DISTINCT {actual_column}) as {metric_name} FROM {table} {where_clause} {having_clause}".strip()
            else:
                sql = f"SELECT {aggregation}({actual_column}) as {metric_name} FROM {table} {where_clause} {having_clause}".strip()
            
            # Execute SQL
            result = self.sql_tool.run_safe_query(sql)
            
            # Clear pending confirmation
            if self.memory:
                self.memory.clear_pending_confirmation(user_id)
            
            # Format response
            rows = result.get("rows", [])
            row_count = result.get("row_count", 0)
            execution_time = result.get("execution_time_ms", 0)
            
            # Extract metric value from result with descriptive error messages
            metric_value = "N/A"
            if rows and len(rows) > 0:
                value = None
                
                # Handle different result formats
                row_obj = rows[0]
                
                # Check if it's a tuple
                if isinstance(row_obj, tuple):
                    if len(row_obj) > 0:
                        value = row_obj[0]
                    else:
                        metric_value = "Cannot compute value: result tuple is empty"
                # Check if it's SQLAlchemy Row (has tuple-like access)
                elif hasattr(row_obj, '__getitem__') and hasattr(row_obj, '__len__'):
                    try:
                        if len(row_obj) > 0:
                            value = row_obj[0]
                        else:
                            metric_value = "Cannot compute value: result row is empty"
                    except (IndexError, TypeError):
                        metric_value = "Cannot compute value: unable to access row value"
                # Handle dict format
                elif isinstance(row_obj, dict):
                    if row_obj:
                        value = list(row_obj.values())[0]
                    else:
                        metric_value = "Cannot compute value: result dictionary is empty"
                
                # Process the extracted value
                if value is not None:
                    # Handle empty string
                    if value == "":
                        metric_value = "Cannot compute value: field is empty"
                    # Handle boolean True
                    elif value is True:
                        metric_value = "Cannot compute value: field contains boolean True (expected numeric value)"
                    # Handle boolean False
                    elif value is False:
                        metric_value = "Cannot compute value: field contains boolean False (expected numeric value)"
                    # Handle empty collections
                    elif isinstance(value, (list, tuple, dict, set)) and len(value) == 0:
                        metric_value = f"Cannot compute value: field is empty {type(value).__name__}"
                    # Handle valid numeric values (including 0 and Decimal)
                    else:
                        # Try to convert any value to displayable format
                        try:
                            # Import Decimal check
                            from decimal import Decimal
                            
                            # Check if it's Decimal type
                            if isinstance(value, Decimal):
                                metric_value = f"{float(value):.2f}"
                            # Check if it's directly numeric
                            elif isinstance(value, (int, float)):
                                metric_value = f"{value:.2f}"
                            # Check if it can be converted to float
                            elif hasattr(value, '__float__'):
                                metric_value = f"{float(value):.2f}"
                            # For everything else, convert to string
                            else:
                                metric_value = str(value)
                        except (ValueError, OverflowError, TypeError) as e:
                            # If conversion fails, use string representation
                            metric_value = str(value)
                # If value extraction failed
                else:
                    if metric_value == "N/A":
                        metric_value = "Cannot compute value: unable to extract value from result"
            else:
                # No rows returned
                metric_value = "Cannot compute value: no rows returned from query"
            
            # Analyze empty result if no rows returned
            if row_count == 0 or (metric_value and "no rows returned" in metric_value.lower()):
                analysis = self._analyze_empty_result(sql, table, date_filter)
                if analysis:
                    if self.memory:
                        self.memory.clear_pending_confirmation(user_id)
                    return {
                        "success": False,
                        "message": f"❌ {analysis}",
                        "chart_path": None
                    }
            
            # Show corrected column in message if it was changed
            message = (
                f"✅ {metric_name.replace('_', ' ').title()} calculated.\n"
                f"• Value: {metric_value}\n"
                f"• Rows: {row_count}\n"
                f"• Time: {execution_time:.1f}ms"
            )
            if actual_column != column:
                message += f"\n• Used column: `{actual_column}` (instead of `{column}`)"
            
            return {
                "success": True,
                "message": message,
                "chart_path": None,
                "raw": result,
            }
            
        except Exception as e:
            logger.exception("router_confirmed_query_error", error=str(e))
            if self.memory:
                self.memory.clear_pending_confirmation(user_id)
            return {
                "success": False,
                "message": f"⚠️ Error executing query: {str(e)}",
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

            # Check if it's already valid SQL (explicit SQL - bypass validation)
            is_explicit_sql = query.strip().lower().startswith(("select", "with"))
            
            # Only validate intent for natural language queries, not explicit SQL
            if not is_explicit_sql and self.semantic_loader:
                is_valid_intent, intent_error = self._validate_query_intent(query, None)
                if not is_valid_intent:
                    return {
                        "success": False,
                        "message": f"❌ {intent_error}",
                        "chart_path": None,
                    }

            # Convert natural language to SQL if needed
            if not is_explicit_sql:
                converted_query = self._convert_natural_language_to_sql(query)
                if converted_query:
                    query = converted_query
                    logger.info("router_nl_to_sql_converted", original=text[:50], sql=query)
                else:
                    return {
                        "success": False,
                        "message": "Could not convert query to SQL. Please use SQL syntax or 'sql: SELECT ...'",
                        "chart_path": None,
                    }

            # Execute SQL (with error handling)
            try:
                result = self.sql_tool.run_safe_query(query)
            except RuntimeError as e:
                # Handle SQL execution errors gracefully
                error_msg = str(e)
                if "Database error" in error_msg:
                    # Extract the actual database error
                    db_error = error_msg.replace("Database error: ", "")
                    return {
                        "success": False,
                        "message": f"❌ SQL Error: {db_error}\n\n💡 Tip: Check your SQL syntax and ensure all table/column names are correct.",
                        "chart_path": None,
                    }
                elif "timed out" in error_msg.lower():
                    return {
                        "success": False,
                        "message": f"❌ {error_msg}\n\n💡 Tip: Your query took too long. Try adding filters or limits.",
                        "chart_path": None,
                    }
                else:
                    return {
                        "success": False,
                        "message": f"❌ {error_msg}",
                        "chart_path": None,
                    }

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

    def _convert_natural_language_to_sql(self, text: str) -> Optional[str]:
        """
        Convert simple natural language queries to SQL.
        
        Args:
            text: Natural language query
            
        Returns:
            SQL query string or None if conversion fails
        """
        text_lower = text.lower().strip()
        
        # Pattern: "show me total rows in <table>" or "count rows in <table>"
        patterns = [
            (r"(?:show me|get|count|total)\s+(?:total\s+)?(?:rows?|records?)\s+in\s+(\w+)", 
             lambda m: f"SELECT COUNT(*) as total FROM {m.group(1)}"),
            (r"how many\s+(?:rows?|records?)\s+in\s+(\w+)",
             lambda m: f"SELECT COUNT(*) as total FROM {m.group(1)}"),
            (r"count\s+(\w+)",
             lambda m: f"SELECT COUNT(*) as total FROM {m.group(1)}"),
            (r"show\s+me\s+all\s+(?:rows?|records?|data)\s+from\s+(\w+)",
             lambda m: f"SELECT * FROM {m.group(1)} LIMIT 100"),
            (r"list\s+(?:all\s+)?(?:rows?|records?)\s+from\s+(\w+)",
             lambda m: f"SELECT * FROM {m.group(1)} LIMIT 100"),
            (r"select\s+(?:all\s+)?(?:from\s+)?(\w+)",
             lambda m: f"SELECT * FROM {m.group(1)} LIMIT 100"),
        ]
        
        for pattern, converter in patterns:
            match = re.search(pattern, text_lower)
            if match:
                try:
                    sql = converter(match)
                    logger.info("router_nl_to_sql_pattern_matched", pattern=pattern, sql=sql)
                    return sql
                except Exception as e:
                    logger.warning("router_nl_to_sql_pattern_failed", pattern=pattern, error=str(e))
                    continue
        
        # If no pattern matches, try using LLM reasoner to generate SQL
        if self.reasoner and self.reasoner.enabled:
            try:
                # Use LLM to convert NL to SQL
                sql = self._llm_convert_to_sql(text)
                if sql:
                    return sql
            except Exception as e:
                logger.warning("router_llm_sql_conversion_failed", error=str(e))
        
        return None

    def _llm_convert_to_sql(self, text: str) -> Optional[str]:
        """Use LLM to convert natural language to SQL."""
        if not self.reasoner or not self.reasoner.enabled:
            return None
        
        try:
            import json
            
            # Get actual schema dynamically (database-agnostic)
            schema_context = ""
            if self.semantic_loader:
                schema = self.semantic_loader.schema_discovery.discover_full_schema()
                # Build schema context from actual database
                tables_info = []
                for table_name, columns in list(schema.items())[:5]:  # Limit to 5 tables for prompt size
                    column_names = [col["name"] for col in columns[:10]]  # Limit to 10 columns per table
                    tables_info.append(f"- {table_name}: {', '.join(column_names)}")
                
                if tables_info:
                    schema_context = f"\n\nDatabase Schema (actual tables and columns):\n" + "\n".join(tables_info)
            
            # Use PromptManager if available
            if self.prompt_manager:
                prompt = self.prompt_manager.build_nl_to_sql_prompt(
                    query=text,
                    schema_context=schema_context
                )
            else:
                # Fallback to old hardcoded prompt (for backward compatibility)
                prompt = f"""Convert this natural language query to a safe PostgreSQL SELECT query.
Only return the SQL query, nothing else.

Query: {text}{schema_context}

Rules:
- Only use SELECT statements
- Use COUNT(*) for counting rows
- Use COUNT(DISTINCT column_name) for counting unique values (NOT COUNT DISTINCT(column_name))
- Use AVG(column_name) for averages
- Use SUM(column_name) for totals
- Use WHERE column_name = 'value' for filtering
- Use actual table and column names from the schema above
- Return only the SQL query
- Do not include markdown code blocks
- Do not include explanations

SQL:"""
            
            response = self.reasoner._call_llm(prompt)

            # Extract SQL from response (might have markdown code blocks)
            sql = response.strip()
            if "```sql" in sql:
                sql = sql.split("```sql")[1].split("```")[0].strip()
            elif "```" in sql:
                sql = sql.split("```")[1].split("```")[0].strip()

            # Fix common LLM errors: restore missing asterisks
            import re
            # Fix COUNT() -> COUNT(*) (when COUNT is followed by empty parentheses)
            sql = re.sub(r'\bCOUNT\s*\(\s*\)', 'COUNT(*)', sql, flags=re.IGNORECASE)
            # Fix SELECT FROM -> SELECT * FROM (when SELECT is directly followed by FROM)
            sql = re.sub(r'\bSELECT\s+FROM\b', 'SELECT * FROM', sql, flags=re.IGNORECASE)
            
            # Validate it's a SELECT query
            if sql and sql.strip().upper().startswith("SELECT"):
                return sql.strip()

        except Exception as e:
            logger.warning("router_llm_sql_conversion_failed", error=str(e))
        
        return None

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
