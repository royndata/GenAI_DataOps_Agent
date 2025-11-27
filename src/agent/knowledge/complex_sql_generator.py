# src/agent/knowledge/complex_sql_generator.py

"""
Complex SQL Generator Module

Generates SQL for complex queries including JOINs, window functions, subqueries, HAVING clauses.
Fixes issues:
- Missing JOIN support
- Missing window functions
- Missing subqueries
- Missing HAVING clause
- Missing GROUP BY for daily aggregations
- Multi-metric queries
"""

import re
from typing import Dict, List, Optional, Any, Tuple
from pydantic import BaseModel, Field

from agent.knowledge.query_analyzer import QueryComplexity, QueryAnalyzer
from agent.knowledge.date_filter_builder import DateFilterBuilder, DateFilterResult
from agent.knowledge.prompt_manager import PromptManager
from agent.knowledge.schema_discovery import SchemaDiscovery
from agent.cognition.llm_reasoner import LLMReasoner
from agent.logging_config import logger


class SQLGenerationParams(BaseModel):
    """Pydantic model for SQL generation parameters."""
    
    query: str = Field(description="Original user query")
    mapping: Dict[str, Any] = Field(description="Metric mapping from semantic layer")
    complexity: QueryComplexity = Field(description="Query complexity analysis")
    date_filter_result: Optional[DateFilterResult] = Field(default=None, description="Date filter result")
    schema: Dict[str, Any] = Field(description="Database schema")


class ComplexSQLGenerator:
    """
    Generates SQL for complex queries.
    
    Handles JOINs, window functions, subqueries, HAVING clauses, and GROUP BY.
    Routes to appropriate generation method based on query complexity.
    """
    
    def __init__(
        self,
        schema_discovery: SchemaDiscovery,
        query_analyzer: QueryAnalyzer,
        date_filter_builder: DateFilterBuilder,
        reasoner: Optional[LLMReasoner] = None
    ):
        """
        Initialize complex SQL generator.
        
        Args:
            schema_discovery: SchemaDiscovery instance
            query_analyzer: QueryAnalyzer instance
            date_filter_builder: DateFilterBuilder instance
            reasoner: Optional LLMReasoner for LLM-based generation
        """
        self.schema_discovery = schema_discovery
        self.query_analyzer = query_analyzer
        self.date_filter_builder = date_filter_builder
        self.reasoner = reasoner

    def generate_sql(
        self,
        mapping: Dict[str, Any],
        query_text: str,
        date_filter: Optional[str] = None,
        complexity: Optional[QueryComplexity] = None
    ) -> Optional[str]:
        """
        Generate SQL for complex queries (wrapper method called by semantic_loader).
        
        Args:
            mapping: Metric mapping from semantic layer
            query_text: Original user query text
            date_filter: Optional date filter clause
            complexity: Query complexity analysis (optional, will be analyzed if not provided)
            
        Returns:
            Generated SQL query or None if fails
        """
        # Analyze complexity if not provided
        if not complexity and self.query_analyzer:
            try:
                complexity = self.query_analyzer.analyze_query_complexity(
                    query_text,
                    metric_table=mapping.get("table") if mapping else None
                )
            except Exception as e:
                logger.warning("complex_sql_complexity_analysis_failed", error=str(e))
                # Fall through to use generate_complex_sql without complexity
        
        # Use generate_complex_sql with the correct signature
        # Pass complexity to avoid re-analysis
        return self.generate_complex_sql(
            query=query_text,
            mapping=mapping,
            date_filter=date_filter,
            complexity=complexity
        )

    def generate_complex_sql(
        self,
        query: str,
        mapping: Dict[str, Any],
        date_filter: Optional[str] = None,
        complexity: Optional[QueryComplexity] = None  
    ) -> Optional[str]:
        """
        Generate SQL for complex queries.
        
        Routes to appropriate generator based on query complexity.
        
        Args:
            query: User query text
            mapping: Metric mapping from semantic layer
            date_filter: Optional date filter clause
            complexity: Optional pre-analyzed complexity (avoids re-analysis)
            
        Returns:
            Generated SQL query or None if fails
        """
        # Analyze query complexity if not provided
        if not complexity:
            if not self.query_analyzer:
                logger.error("complex_sql_no_analyzer", message="query_analyzer not initialized")
                return None
            try:
                complexity = self.query_analyzer.analyze_query_complexity(
                    query,
                    metric_table=mapping.get("table") if mapping else None
                )
            except Exception as e:
                logger.exception("complex_sql_complexity_analysis_failed", error=str(e))
                return None
        
        # Get date filter result
        date_filter_result = None
        if date_filter or complexity.requires_grouping:
            table = mapping.get("table")
            if table:
                requirements = self.date_filter_builder.extract_date_requirements(query)
                date_filter_result = self.date_filter_builder.build_date_filter(
                    table, requirements
                )
        
        # Get schema
        schema = self.schema_discovery.discover_full_schema()
        
        # Route to appropriate generator
        # Handle combined requirements (e.g., joins + grouping)
        if complexity.requires_joins:
            # If also requires grouping, pass that info to join generator
            if complexity.requires_grouping:
                # Use LLM to generate SQL with both JOINs and GROUP BY
                return self.generate_join_with_grouping_sql(
                    query, mapping, complexity, date_filter_result, schema
                )
            else:
                return self.generate_join_sql(query, mapping, complexity, date_filter_result, schema)
        
        elif complexity.requires_window_functions:
            return self.generate_window_function_sql(query, mapping, complexity, date_filter_result, schema)
        
        elif complexity.requires_subqueries:
            return self.generate_subquery_sql(query, mapping, complexity, date_filter_result, schema)
        
        elif complexity.requires_grouping:
            return self.generate_grouped_sql(query, mapping, complexity, date_filter_result, schema)
        
        elif complexity.requires_case_statements:
            return self.generate_conditional_sql(query, mapping, complexity, date_filter_result, schema)
        
        else:
            # Fallback to simple SQL generation
            return self.generate_simple_sql(mapping, date_filter_result)
    
    def generate_join_sql(
        self,
        query: str,
        mapping: Dict[str, Any],
        complexity: QueryComplexity,
        date_filter_result: Optional[DateFilterResult],
        schema: Dict[str, Any]
    ) -> Optional[str]:
        """
        Generate SQL with JOINs.
        
        Args:
            query: User query
            mapping: Metric mapping
            complexity: Query complexity
            date_filter_result: Date filter result
            schema: Database schema
            
        Returns:
            SQL query with JOINs
        """
        if not self.reasoner or not self.reasoner.enabled:
            logger.warning("complex_sql_llm_not_available", query_type="join")
            return None
        
        # Use LLM to generate JOIN SQL
        requirements = {
            "requires_joins": True,
            "tables": complexity.tables_mentioned,
            "metrics": complexity.metrics_mentioned,
            "requires_grouping": complexity.requires_grouping, 
            "group_by_columns": complexity.group_by_columns if complexity.requires_grouping else []  
        }
        
        prompt = PromptManager.build_complex_sql_prompt(schema, query, requirements)
        
        try:
            response = self.reasoner._call_llm(prompt)
            
            # Extract SQL from response
            sql = self._extract_sql_from_response(response)
            
            if sql and sql.lower().startswith("select"):
                logger.info("complex_sql_join_generated", sql_preview=sql[:100])
                return sql
            
        except Exception as e:
            logger.exception("complex_sql_join_generation_failed", error=str(e))
        
        return None
    
    def generate_join_with_grouping_sql(
        self,
        query: str,
        mapping: Dict[str, Any],
        complexity: QueryComplexity,
        date_filter_result: Optional[DateFilterResult],
        schema: Dict[str, Any]
    ) -> Optional[str]:
        """
        Generate SQL with JOINs AND GROUP BY (for multi-table grouped queries).
        
        Args:
            query: User query
            mapping: Metric mapping
            complexity: Query complexity
            date_filter_result: Date filter result
            schema: Database schema
            
        Returns:
            SQL query with JOINs and GROUP BY
        """
        if not self.reasoner:
            logger.error("complex_sql_reasoner_not_initialized", query_type="join_with_grouping")
            return None
        
        if not self.reasoner.enabled:
            logger.warning("complex_sql_reasoner_not_enabled", query_type="join_with_grouping")
            return None
        
        # Use LLM to generate SQL with both JOINs and GROUP BY
        # Include metric mapping so LLM knows which table/column to use for the metric
        requirements = {
            "requires_joins": True,
            "requires_grouping": True,
            "requires_having": complexity.requires_having,
            "tables": complexity.tables_mentioned,
            "metrics": complexity.metrics_mentioned,
            "group_by_columns": complexity.group_by_columns,
            "metric_mapping": {
                "table": mapping.get("table"),
                "column": mapping.get("column"),
                "aggregation": mapping.get("aggregation", "COUNT"),
                "concept": mapping.get("concept")
            }
        }
        
        prompt = PromptManager.build_complex_sql_prompt(schema, query, requirements)
        
        try:
            response = self.reasoner._call_llm(prompt)
            
            # Log raw response for debugging
            logger.info(
                "complex_sql_llm_response_received",
                response_preview=response[:200] if response else "None",
                response_length=len(response) if response else 0
            )
            
            # Extract SQL from response
            sql = self._extract_sql_from_response(response)
            
            # Log extracted SQL for debugging
            logger.info(
                "complex_sql_extracted",
                sql_preview=sql[:200] if sql else "None",
                sql_length=len(sql) if sql else 0,
                starts_with_select=sql.lower().startswith("select") if sql else False
            )
            
            if sql and sql.lower().startswith("select"):
                logger.info("complex_sql_join_with_grouping_generated", sql_preview=sql[:100])
                return sql
            else:
                # Log why SQL was rejected
                logger.warning(
                    "complex_sql_rejected",
                    reason="SQL does not start with SELECT" if sql else "SQL extraction returned None",
                    extracted_sql_preview=sql[:200] if sql else None
                )
            
        except Exception as e:
            logger.exception("complex_sql_join_with_grouping_generation_failed", error=str(e))
        
        return None

    def generate_window_function_sql(
        self,
        query: str,
        mapping: Dict[str, Any],
        complexity: QueryComplexity,
        date_filter_result: Optional[DateFilterResult],
        schema: Dict[str, Any]
    ) -> Optional[str]:
        """
        Generate SQL with window functions (for rankings, running totals).
        
        Args:
            query: User query
            mapping: Metric mapping
            complexity: Query complexity
            date_filter_result: Date filter result
            schema: Database schema
            
        Returns:
            SQL query with window functions
        """
        if not self.reasoner or not self.reasoner.enabled:
            logger.warning("complex_sql_llm_not_available", query_type="window_function")
            return None
        
        requirements = {
            "requires_window_functions": True,
            "query_type": "ranking" if "top" in query.lower() or "rank" in query.lower() else "running_total"
        }
        
        prompt = PromptManager.build_complex_sql_prompt(schema, query, requirements)
        
        try:
            response = self.reasoner._call_llm(prompt)
            sql = self._extract_sql_from_response(response)
            
            if sql and sql.lower().startswith("select"):
                logger.info("complex_sql_window_generated", sql_preview=sql[:100])
                return sql
            
        except Exception as e:
            logger.exception("complex_sql_window_generation_failed", error=str(e))
        
        return None
    
    def generate_subquery_sql(
        self,
        query: str,
        mapping: Dict[str, Any],
        complexity: QueryComplexity,
        date_filter_result: Optional[DateFilterResult],
        schema: Dict[str, Any]
    ) -> Optional[str]:
        """
        Generate SQL with subqueries.
        
        Args:
            query: User query
            mapping: Metric mapping
            complexity: Query complexity
            date_filter_result: Date filter result
            schema: Database schema
            
        Returns:
            SQL query with subqueries
        """
        if not self.reasoner or not self.reasoner.enabled:
            logger.warning("complex_sql_llm_not_available", query_type="subquery")
            return None
        
        requirements = {
            "requires_subqueries": True,
            "tables": complexity.tables_mentioned
        }
        
        prompt = PromptManager.build_complex_sql_prompt(schema, query, requirements)
        
        try:
            response = self.reasoner._call_llm(prompt)
            sql = self._extract_sql_from_response(response)
            
            if sql and sql.lower().startswith("select"):
                logger.info("complex_sql_subquery_generated", sql_preview=sql[:100])
                return sql
            
        except Exception as e:
            logger.exception("complex_sql_subquery_generation_failed", error=str(e))
        
        return None
    
    def generate_grouped_sql(
        self,
        query: str,
        mapping: Dict[str, Any],
        complexity: QueryComplexity,
        date_filter_result: Optional[DateFilterResult],
        schema: Dict[str, Any]
    ) -> Optional[str]:
        """
        Generate SQL with GROUP BY (for daily/weekly/monthly aggregations).
        
        Args:
            query: User query
            mapping: Metric mapping
            complexity: Query complexity
            date_filter_result: Date filter result
            schema: Database schema
            
        Returns:
            SQL query with GROUP BY
        """
        table = mapping.get("table")
        column = mapping.get("column")
        aggregation = mapping.get("aggregation", "SUM")

        # Discover date column from schema if not available
        if date_filter_result:
            date_column = date_filter_result.date_column
        elif mapping.get("date_column"):
            date_column = mapping["date_column"]
        else:
            # Discover from schema using date_filter_builder
            
            if table:
                # Use the schema parameter that was passed in
                discovered_date_col = self.date_filter_builder.discover_date_column(table, schema)
                date_column = discovered_date_col or "created_at"  # Last resort fallback
                if not discovered_date_col:
                    logger.warning("complex_sql_using_fallback_date_column", table=table)
            else:
                date_column = "created_at"  # Last resort

        # Check if any grouping column is from a different table (requires JOIN)
        needs_join_for_grouping = False
        if complexity.group_by_columns:
            for col in complexity.group_by_columns:
                if col not in ["date", "day", "daily", "week", "weekly", "month", "monthly", "biweekly", "bimonthly", "year", "yearly", "quarter", "quarterly"]:
                    # Normalize: try both the exact name and singular/plural variants
                    col_normalized = col.lower()
                    # Better plural/singular conversion
                    if col_normalized.endswith('ies') and len(col_normalized) > 4:
                        col_singular = col_normalized[:-3] + 'y'  # "countries" -> "country"
                    elif col_normalized.endswith('es') and len(col_normalized) > 3:
                        # Check if removing 'es' makes sense (e.g., "countries" -> "countr" is wrong)
                        # Only remove 'es' if the word without it is meaningful
                        potential_singular = col_normalized[:-2]
                        if len(potential_singular) > 2:
                            col_singular = potential_singular
                        else:
                            col_singular = col_normalized
                    elif col_normalized.endswith('s') and len(col_normalized) > 3:
                        col_singular = col_normalized[:-1]  # Remove single 's': "users" -> "user"
                    else:
                        col_singular = col_normalized
                    
                    # Generate plural form
                    if col_singular.endswith('y'):
                        col_plural = col_singular[:-1] + 'ies'  # "country" -> "countries"
                    elif not col_singular.endswith('s'):
                        col_plural = col_singular + 's'  # "user" -> "users"
                    else:
                        col_plural = col_singular
                    
                    for table_name, columns in schema.items():
                        for col_info in columns:
                            col_name_lower = col_info.get("name", "").lower()
                            # Try exact match
                            if col_name_lower == col_normalized:
                                if table_name != table:
                                    needs_join_for_grouping = True
                                    break
                            # Try singular/plural matching
                            elif col_name_lower == col_singular or col_name_lower == col_plural:
                                if table_name != table:
                                    needs_join_for_grouping = True
                                    break

                            # Try word-level matching with word boundaries
                            # Only match if one is a substring of the other AND they're similar length
                            # This prevents "user" matching "user_id" or "id" matching "user_id"
                            elif (col_normalized in col_name_lower or col_name_lower in col_normalized):
                                # Additional check: lengths should be similar (within 2 chars)
                                # This ensures "country" matches "countries" but not "user" matches "user_id"
                                length_diff = abs(len(col_normalized) - len(col_name_lower))
                                if length_diff <= 2:
                                    if table_name != table:
                                        needs_join_for_grouping = True
                                        break
                        if needs_join_for_grouping:
                            break
                    if needs_join_for_grouping:
                        break
        
        # If grouping requires JOINs but requires_joins wasn't detected, use LLM generator
        if needs_join_for_grouping and not complexity.requires_joins:
            logger.warning(
                "complex_sql_grouping_requires_join",
                message=f"Grouping column from different table detected, routing to JOIN generator"
            )
            # Force requires_joins to trigger JOIN generator
            complexity.requires_joins = True
            return self.generate_join_with_grouping_sql(
                query, mapping, complexity, date_filter_result, schema
            )

        # Build GROUP BY clause based on grouping requirements
        group_by_clause = ""
        if complexity.group_by_columns:
            # Use actual column names from group_by_columns (e.g., "country")
            # Map column names to actual table columns if needed
            group_by_cols = []
            for col in complexity.group_by_columns:
                # Check if it's a date grouping keyword
                if col in ["date", "day", "daily"]:
                    group_by_cols.append(f"DATE({date_column})")
                elif col in ["week", "weekly"]:
                    group_by_cols.append(f"DATE_TRUNC('week', {date_column})")
                elif col in ["month", "monthly"]:
                    group_by_cols.append(f"DATE_TRUNC('month', {date_column})")
                elif col in ["year", "yearly"]:
                    group_by_cols.append(f"DATE_TRUNC('year', {date_column})")
                elif col in ["quarter", "quarterly"]:
                    group_by_cols.append(f"DATE_TRUNC('quarter', {date_column})")
                elif col == "biweekly":
                    # Biweekly: every 2 weeks
                    group_by_cols.append(f"DATE_TRUNC('week', {date_column})")
                elif col == "bimonthly":
                    # Bimonthly: every 2 months
                    group_by_cols.append(f"DATE_TRUNC('month', {date_column})")
                else:
                    # It's an actual column name (e.g., "country" or "countries")
                    # Try to find it in schema with plural/singular matching
                    col_found = False
                    # Normalize: try both the exact name and singular/plural variants
                    col_normalized = col.lower()
                    # Better plural/singular conversion
                    if col_normalized.endswith('ies') and len(col_normalized) > 4:
                        col_singular = col_normalized[:-3] + 'y'  # "countries" -> "country"
                    elif col_normalized.endswith('es') and len(col_normalized) > 3:
                        # Check if removing 'es' makes sense (e.g., "countries" -> "countr" is wrong)
                        # Only remove 'es' if the word without it is meaningful
                        potential_singular = col_normalized[:-2]
                        if len(potential_singular) > 2:
                            col_singular = potential_singular
                        else:
                            col_singular = col_normalized
                    elif col_normalized.endswith('s') and len(col_normalized) > 3:
                        col_singular = col_normalized[:-1]  # Remove single 's': "users" -> "user"
                    else:
                        col_singular = col_normalized
                    
                    # Generate plural form
                    if col_singular.endswith('y'):
                        col_plural = col_singular[:-1] + 'ies'  # "country" -> "countries"
                    elif not col_singular.endswith('s'):
                        col_plural = col_singular + 's'  # "user" -> "users"
                    else:
                        col_plural = col_singular
                    
                    for table_name, columns in schema.items():
                        for col_info in columns:
                            col_name_lower = col_info.get("name", "").lower()
                            # Try exact match first
                            if col_name_lower == col_normalized:
                                col_found = True
                            # Try singular/plural matching
                            elif col_name_lower == col_singular or col_name_lower == col_plural:
                                col_found = True

                            # Try word-level matching with length check
                            # Only match if one is a substring of the other AND they're similar length
                            # This prevents "user" matching "user_id" or "id" matching "user_id"
                            elif (col_normalized in col_name_lower or col_name_lower in col_normalized):
                                # Additional check: lengths should be similar (within 2 chars)
                                # This ensures "country" matches "countries" but not "user" matches "user_id"
                                length_diff = abs(len(col_normalized) - len(col_name_lower))
                                if length_diff <= 2:
                                    col_found = True
                            
                            if col_found:
                                # Use table.column format if different table
                                if table_name != table:
                                    group_by_cols.append(f"{table_name}.{col_info['name']}")
                                else:
                                    group_by_cols.append(col_info['name'])
                                break
                        if col_found:
                            break
                    # If not found, use as-is (might need JOIN)
                    if not col_found:
                        group_by_cols.append(col)
            
            # don't default to date if no grouping columns
            if group_by_cols:
                group_by_clause = f"GROUP BY {', '.join(group_by_cols)}"
            else:
                # Don't default to date grouping - this is likely an error
                # If no grouping columns detected, log warning and return None or use LLM
                logger.warning(
                    "complex_sql_no_grouping_columns",
                    message="No grouping columns detected but grouping is required"
                )
                # Try to use LLM to generate SQL if reasoner is available
                if self.reasoner and self.reasoner.enabled:
                    return self.generate_join_with_grouping_sql(
                        query, mapping, complexity, date_filter_result, schema
                    )
                # Last resort: return None to trigger fallback
                return None
        else:
            # If no grouping required, don't add GROUP BY
            group_by_clause = ""
        
        # Build HAVING clause if needed
        having_clause = ""
        if complexity.requires_having:

            # Extract threshold from query            
            query_lower = query.lower()
            # Patterns like "more than 100", "greater than X", etc.
            threshold_match = re.search(r'(?:more than|greater than|at least|over|above)\s+(\d+)', query_lower)
            if threshold_match:
                threshold = threshold_match.group(1)
                aggregation = mapping.get("aggregation", "COUNT")
                if aggregation == "COUNT DISTINCT":
                    having_clause = f"HAVING COUNT(DISTINCT {column}) > {threshold}"
                else:
                    having_clause = f"HAVING {aggregation}({column}) > {threshold}"

        # Build WHERE clause if date filter is provided
        where_clause = ""
        if date_filter_result and date_filter_result.date_filter:
            where_clause = f"WHERE {date_filter_result.date_filter}"

        # Build SELECT clause
        select_parts = []
        # Add grouping columns to SELECT (use same logic as GROUP BY)
        if complexity.group_by_columns:
            for col in complexity.group_by_columns:
                # Check if it's a date grouping keyword
                if col in ["date", "day", "daily"]:
                    select_parts.append(f"DATE({date_column}) as date")
                elif col in ["week", "weekly"]:
                    select_parts.append(f"DATE_TRUNC('week', {date_column}) as week")
                elif col in ["month", "monthly"]:
                    select_parts.append(f"DATE_TRUNC('month', {date_column}) as month")
                elif col in ["year", "yearly"]:
                    select_parts.append(f"DATE_TRUNC('year', {date_column}) as year")
                elif col in ["quarter", "quarterly"]:
                    select_parts.append(f"DATE_TRUNC('quarter', {date_column}) as quarter")
                elif col == "biweekly":
                    select_parts.append(f"DATE_TRUNC('week', {date_column}) as week")
                elif col == "bimonthly":
                    select_parts.append(f"DATE_TRUNC('month', {date_column}) as month")

                else:
                    # It's an actual column name (e.g., "country" or "countries")
                    # Find it in schema with plural/singular matching
                    col_found = False
                    # Normalize: try both the exact name and singular/plural variants
                    col_normalized = col.lower()
                    # Better plural/singular conversion
                    if col_normalized.endswith('ies') and len(col_normalized) > 4:
                        col_singular = col_normalized[:-3] + 'y'  # "countries" -> "country"
                    elif col_normalized.endswith('es') and len(col_normalized) > 3:
                        # Check if removing 'es' makes sense (e.g., "countries" -> "countr" is wrong)
                        # Only remove 'es' if the word without it is meaningful
                        potential_singular = col_normalized[:-2]
                        if len(potential_singular) > 2:
                            col_singular = potential_singular
                        else:
                            col_singular = col_normalized
                    elif col_normalized.endswith('s') and len(col_normalized) > 3:
                        col_singular = col_normalized[:-1]  # Remove single 's': "users" -> "user"
                    else:
                        col_singular = col_normalized
                    
                    # Generate plural form
                    if col_singular.endswith('y'):
                        col_plural = col_singular[:-1] + 'ies'  # "country" -> "countries"
                    elif not col_singular.endswith('s'):
                        col_plural = col_singular + 's'  # "user" -> "users"
                    else:
                        col_plural = col_singular

                    for table_name, columns in schema.items():
                        for col_info in columns:
                            col_name_lower = col_info.get("name", "").lower()
                            # Try exact match first
                            if col_name_lower == col_normalized:
                                col_found = True
                            # Try singular/plural matching
                            elif col_name_lower == col_singular or col_name_lower == col_plural:
                                col_found = True

                            # Try word-level matching with length check
                            # Only match if one is a substring of the other AND they're similar length
                            # This prevents "user" matching "user_id" or "id" matching "user_id"
                            elif (col_normalized in col_name_lower or col_name_lower in col_normalized):
                                # Additional check: lengths should be similar (within 2 chars)
                                # This ensures "country" matches "countries" but not "user" matches "user_id"
                                length_diff = abs(len(col_normalized) - len(col_name_lower))
                                if length_diff <= 2:
                                    col_found = True
                            
                            if col_found:
                                # Use table.column format if different table
                                if table_name != table:
                                    select_parts.append(f"{table_name}.{col_info['name']} as {col_info['name']}")
                                else:
                                    select_parts.append(f"{col_info['name']}")
                                break
                        if col_found:
                            break
                    # If not found, use as-is (might need JOIN)
                    if not col_found:
                        select_parts.append(col)
        
        # Only add date if ALL grouping columns are date keywords (not actual columns)
        # Don't add date if we're grouping by actual columns like "country"
        has_actual_columns = False
        if complexity.group_by_columns:
            has_actual_columns = any(c not in ["date", "day", "daily", "week", "weekly", "month", "monthly", "biweekly", "bimonthly", "year", "yearly", "quarter", "quarterly"] for c in complexity.group_by_columns)
        
        # Only add date if no actual columns are being grouped by
        if not has_actual_columns:
            # Only add if not already added above
            if not any("DATE(" in part or "DATE_TRUNC" in part for part in select_parts):
                select_parts.append(f"DATE({date_column}) as date")
        
        # Add aggregation
        if aggregation == "COUNT DISTINCT":
            select_parts.append(f"COUNT(DISTINCT {column}) as {mapping.get('concept', 'metric')}")
        else:
            select_parts.append(f"{aggregation}({column}) as {mapping.get('concept', 'metric')}")
        
        select_clause = f"SELECT {', '.join(select_parts)}"
        
        # Determine ORDER BY column (use first grouping column or first SELECT part)
        order_by_col = None
        if complexity.group_by_columns and select_parts:
            # Use first grouping column
            first_group_col = complexity.group_by_columns[0]
            if first_group_col in ["date", "day", "daily"]:
                order_by_col = "date"
            elif first_group_col in ["week", "weekly"]:
                order_by_col = "week"
            elif first_group_col in ["month", "monthly"]:
                order_by_col = "month"
            elif first_group_col in ["year", "yearly"]:
                order_by_col = "year"
            elif first_group_col in ["quarter", "quarterly"]:
                order_by_col = "quarter"
            elif first_group_col == "biweekly":
                order_by_col = "week"
            elif first_group_col == "bimonthly":
                order_by_col = "month"
            else:
                # Find the corresponding SELECT part for this grouping column
                for part in select_parts:
                    # Check if this part corresponds to the grouping column
                    if first_group_col.lower() in part.lower() or (first_group_col in part and " as " in part):
                        # Extract alias if present, otherwise use column name
                        if " as " in part:
                            order_by_col = part.split(" as ")[1].strip()
                        elif "." in part and " as " in part:
                            # Handle table.column as alias
                            order_by_col = part.split(" as ")[1].strip()
                        elif "." in part:
                            # Extract column name from table.column
                            order_by_col = part.split(".")[1].strip()
                        else:
                            order_by_col = part.strip()
                        break
        
        # Fallback: use first SELECT part if no grouping column found
        if not order_by_col and select_parts:
            first_part = select_parts[0]
            # Extract alias or column name
            if " as " in first_part:
                order_by_col = first_part.split(" as ")[1].strip()
            elif "." in first_part:
                # Extract column name from table.column
                order_by_col = first_part.split(".")[-1].strip()
            else:
                order_by_col = first_part.strip()
        
        # Last resort fallback
        if not order_by_col:
            order_by_col = "1"  # Order by constant if nothing else works
        
        # Build full SQL
        sql = f"{select_clause} FROM {table} {where_clause} {group_by_clause} {having_clause} ORDER BY {order_by_col} DESC"
                
        logger.info("complex_sql_grouped_generated", sql_preview=sql[:100])
        return sql.strip()
    
    def generate_conditional_sql(
        self,
        query: str,
        mapping: Dict[str, Any],
        complexity: QueryComplexity,
        date_filter_result: Optional[DateFilterResult],
        schema: Dict[str, Any]
    ) -> Optional[str]:
        """
        Generate SQL with CASE statements.
        
        Args:
            query: User query
            mapping: Metric mapping
            complexity: Query complexity
            date_filter_result: Date filter result
            schema: Database schema
            
        Returns:
            SQL query with CASE statements
        """
        if not self.reasoner or not self.reasoner.enabled:
            logger.warning("complex_sql_llm_not_available", query_type="conditional")
            return None
        
        requirements = {
            "requires_case_statements": True
        }
        
        prompt = PromptManager.build_complex_sql_prompt(schema, query, requirements)
        
        try:
            response = self.reasoner._call_llm(prompt)
            sql = self._extract_sql_from_response(response)
            
            if sql and sql.lower().startswith("select"):
                logger.info("complex_sql_conditional_generated", sql_preview=sql[:100])
                return sql
            
        except Exception as e:
            logger.exception("complex_sql_conditional_generation_failed", error=str(e))
        
        return None
    
    def generate_simple_sql(
        self,
        mapping: Dict[str, Any],
        date_filter_result: Optional[DateFilterResult]
    ) -> Optional[str]:
        """
        Generate simple SQL (fallback for non-complex queries).
        
        Args:
            mapping: Metric mapping
            date_filter_result: Date filter result
            
        Returns:
            Simple SQL query
        """
        table = mapping.get("table")
        column = mapping.get("column")
        aggregation = mapping.get("aggregation", "SUM")
        metric_name = mapping.get("concept", "metric")
        
        if aggregation == "COUNT DISTINCT":
            select_clause = f"SELECT COUNT(DISTINCT {column}) as {metric_name}"
        else:
            select_clause = f"SELECT {aggregation}({column}) as {metric_name}"
        
        where_clause = ""
        if date_filter_result and date_filter_result.date_filter:
            where_clause = f"WHERE {date_filter_result.date_filter}"
        
        sql = f"{select_clause} FROM {table} {where_clause}".strip()
        
        logger.info("complex_sql_simple_generated", sql_preview=sql[:100])
        return sql
    
    def _extract_sql_from_response(self, response: str) -> str:
        """
        Extract SQL from LLM response (handles markdown code blocks and JSON).
        
        Args:
            response: LLM response text
            
        Returns:
            Extracted SQL query
        """
        sql = response.strip()
        
        # First, try to extract from JSON format (e.g., {"query": "SELECT ..."})
        try:
            import json
            # Check if response is JSON
            if sql.startswith("{") or sql.startswith("["):
                parsed = json.loads(sql)
                # Try common JSON keys
                if isinstance(parsed, dict):
                    # Check for "query" key
                    if "query" in parsed:
                        sql = parsed["query"]
                    # Check for "sql" key
                    elif "sql" in parsed:
                        sql = parsed["sql"]
                    # Check for "SELECT" key (some LLMs use this)
                    elif "SELECT" in parsed:
                        sql = parsed["SELECT"]
        except (json.JSONDecodeError, KeyError, TypeError):
            # Not JSON or doesn't have expected keys, continue with other extraction methods
            pass
        
        # Extract from markdown code blocks
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
        
        return sql

