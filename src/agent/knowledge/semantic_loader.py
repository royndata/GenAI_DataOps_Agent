# src/agent/knowledge/semantic_loader.py

"""
Semantic layer loader and mapper.

Bridges between semantic_layer.yaml patterns and actual database schema.
Maps metric concepts to real tables/columns using schema discovery.
Now supports LLM-based dynamic mapping for any database schema.
"""

import yaml
import json
from pathlib import Path
from typing import Dict, List, Optional, Any, Tuple
from sqlalchemy import text

from agent.knowledge.database import Database
from agent.knowledge.schema_discovery import SchemaDiscovery
from agent.logging_config import logger

try:
    from agent.knowledge.sql_validator import SQLValidator
    SQL_VALIDATOR_AVAILABLE = True
except ImportError:
    SQL_VALIDATOR_AVAILABLE = False
    SQLValidator = None

# Optional new module imports (graceful degradation)
try:
    from agent.knowledge.date_filter_builder import DateFilterBuilder
    from agent.knowledge.query_analyzer import QueryAnalyzer
    from agent.knowledge.complex_sql_generator import ComplexSQLGenerator
    from agent.knowledge.prompt_manager import PromptManager
    NEW_MODULES_AVAILABLE = True
except ImportError:
    NEW_MODULES_AVAILABLE = False
    DateFilterBuilder = None
    QueryAnalyzer = None
    ComplexSQLGenerator = None
    PromptManager = None

# Optional LLM reasoner import (graceful degradation)
try:
    from agent.cognition.llm_reasoner import LLMReasoner
    LLM_REASONER_AVAILABLE = True
except ImportError:
    LLM_REASONER_AVAILABLE = False
    LLMReasoner = None

class SemanticLoader:
    """
    Loads semantic layer definitions and maps them to actual database schema.
    Works dynamically with any database by using pattern matching.
    Falls back to LLM-based mapping when patterns don't match.
    """

    def __init__(
        self, 
        database: Database, 
        semantic_layer_path: Optional[Path] = None,
        reasoner: Optional[LLMReasoner] = None
    ):
        """
        Initialize semantic loader.
        
        Args:
            database: Database instance for schema discovery
            semantic_layer_path: Optional path to semantic_layer.yaml (default: knowledge/semantic_layer.yaml)
            reasoner: Optional LLMReasoner instance for dynamic mapping
        """
        self.database = database
        self.schema_discovery = SchemaDiscovery(database)
        self.reasoner = reasoner
        
        if semantic_layer_path:
            self.semantic_layer_path = Path(semantic_layer_path)
        else:
            # Default location
            self.semantic_layer_path = Path(__file__).parent / "semantic_layer.yaml"
        
        self._semantic_data: Optional[Dict[str, Any]] = None
        self._schema_map: Optional[Dict[str, Dict[str, str]]] = None
        # Initialize new modules (optional)
        self.date_filter_builder = None
        self.query_analyzer = None
        self.complex_sql_generator = None
        self.prompt_manager = None
        
        if NEW_MODULES_AVAILABLE:
            try:
                self.date_filter_builder = DateFilterBuilder(self.schema_discovery)
                self.query_analyzer = QueryAnalyzer(self.schema_discovery)
                self.complex_sql_generator = ComplexSQLGenerator(
                    schema_discovery=self.schema_discovery,
                    query_analyzer=self.query_analyzer,  
                    date_filter_builder=self.date_filter_builder,
                    reasoner=self.reasoner
                )
                self.prompt_manager = PromptManager()
                logger.info("semantic_loader_new_modules_initialized")
            except Exception as e:
                logger.warning("semantic_loader_new_modules_init_failed", error=str(e))

    def load_semantic_layer(self) -> Dict[str, Any]:
        """
        Load semantic layer YAML file.
        
        Returns:
            Parsed YAML content as dict
        """
        if self._semantic_data:
            return self._semantic_data

        try:
            if not self.semantic_layer_path.exists():
                logger.warning("semantic_layer_not_found", path=str(self.semantic_layer_path))
                return {}

            with open(self.semantic_layer_path, 'r') as f:
                data = yaml.safe_load(f) or {}
            
            self._semantic_data = data
            logger.info("semantic_layer_loaded", path=str(self.semantic_layer_path))
            return data
            
        except Exception as e:
            logger.exception("semantic_layer_load_failed", error=str(e))
            return {}

    def map_metric_to_schema(self, metric_name: str) -> Optional[Dict[str, str]]:
        """
        Map a metric pattern to actual database schema.
        Falls back to LLM-based mapping if pattern matching fails.
        
        Args:
            metric_name: Name of metric from semantic layer or concept name
            
        Returns:
            Dict with keys: table, column, date_column (if found), or None if not found
        """
        semantic_data = self.load_semantic_layer()
        metric_patterns = semantic_data.get("metric_patterns", {})
        
        # Try pattern-based mapping first if metric exists in YAML
        if metric_name in metric_patterns:
            pattern = metric_patterns[metric_name]
            # Schema not needed here - table selection uses find_tables_by_pattern()
            
            # Find matching table with semantic scoring
            # For "active" concepts, prefer activity-based tables over static tables
            table = None
            all_table_matches = []
            
            # Collect all matching tables from all patterns
            for table_pattern in pattern.get("table_patterns", []):
                matches = self.schema_discovery.find_tables_by_pattern(table_pattern)
                if matches:
                    all_table_matches.extend(matches)
            
            if all_table_matches:
                # Score tables based on semantic relevance
                metric_lower = metric_name.lower()
                scored_tables = []
                
                for match_table in all_table_matches:
                    score = 0
                    table_lower = match_table.lower()
                    
                    # For "active" concepts, prefer activity-based tables
                    if "active" in metric_lower:
                        if any(activity_word in table_lower for activity_word in ["session", "activity", "event", "log"]):
                            score += 10  # High priority for activity tables
                        elif "user" in table_lower and "session" not in table_lower and "activity" not in table_lower:
                            score -= 5  # Lower priority for static user tables
                    
                    # Prefer tables that match the metric domain
                    metric_words = metric_lower.replace("_", " ").split()
                    for word in metric_words:
                        if word in table_lower:
                            score += 3
                    
                    # Pattern order consideration (tiebreaker)
                    # Tables matching earlier patterns get small bonus
                    pattern_index = next(
                        (i for i, p in enumerate(pattern.get("table_patterns", [])) 
                         if self.schema_discovery.find_tables_by_pattern(p) and 
                         match_table in self.schema_discovery.find_tables_by_pattern(p)),
                        len(pattern.get("table_patterns", []))
                    )
                    # Earlier patterns (lower index) get higher bonus
                    score += (len(pattern.get("table_patterns", [])) - pattern_index) * 0.5
                    
                    scored_tables.append((match_table, score))
                
                # Sort by score (descending) and select highest
                scored_tables.sort(key=lambda x: x[1], reverse=True)
                table = scored_tables[0][0] if scored_tables else None
                
                logger.info(
                    "semantic_table_selected_with_scoring",
                    metric=metric_name,
                    selected_table=table,
                    all_matches=[t[0] for t in scored_tables[:5]],  # Log top 5
                    scores=[t[1] for t in scored_tables[:5]]
                )
            
            if not table:
                logger.warning("semantic_table_not_found", metric=metric_name, patterns=pattern.get("table_patterns"))
                # Fall through to LLM-based mapping
            else:
                # Find matching column with semantic scoring
                column = None
                all_column_matches = []
                
                # Collect all matching columns from all patterns
                for col_pattern in pattern.get("column_patterns", []):
                    if col_pattern == "*":
                        # For "*", get all columns from table
                        schema = self.schema_discovery.discover_full_schema()
                        if table in schema:
                            all_column_matches.extend([col["name"] for col in schema[table]])
                        column = "*"  # Special case - use immediately
                        break
                    matches = self.schema_discovery.find_columns_by_pattern(table, col_pattern)
                    if matches:
                        all_column_matches.extend(matches)
                
                if all_column_matches and column != "*":
                    # Score columns based on semantic relevance
                    scored_columns = []
                    metric_lower = metric_name.lower()
                    
                    for match_col in all_column_matches:
                        score = 0
                        col_lower = match_col.lower()
                        
                        # Prefer columns that match metric keywords
                        metric_words = metric_lower.replace("_", " ").split()
                        for word in metric_words:
                            if word in col_lower:
                                score += 5  # Exact match with metric name
                        
                        # Prefer ID columns for counting metrics
                        if "count" in pattern.get("aggregation", "").lower() or "distinct" in pattern.get("aggregation", "").lower():
                            if any(id_word in col_lower for id_word in ["_id", "id", "user_id", "order_id", "customer_id"]):
                                score += 3
                        
                        # Prefer amount/price columns for revenue metrics
                        if "revenue" in metric_lower or "sales" in metric_lower or "income" in metric_lower:
                            if any(amount_word in col_lower for amount_word in ["amount", "price", "revenue", "total", "value"]):
                                score += 5
                        
                        # Pattern order consideration (tiebreaker)
                        pattern_index = next(
                            (i for i, p in enumerate(pattern.get("column_patterns", [])) 
                             if p != "*" and self.schema_discovery.find_columns_by_pattern(table, p) and 
                             match_col in self.schema_discovery.find_columns_by_pattern(table, p)),
                            len(pattern.get("column_patterns", []))
                        )
                        # Earlier patterns get higher bonus
                        score += (len(pattern.get("column_patterns", [])) - pattern_index) * 0.5
                        
                        scored_columns.append((match_col, score))
                    
                    # Sort by score and select highest
                    scored_columns.sort(key=lambda x: x[1], reverse=True)
                    column = scored_columns[0][0] if scored_columns else None
                    
                    logger.info(
                        "semantic_column_selected_with_scoring",
                        metric=metric_name,
                        table=table,
                        selected_column=column,
                        top_matches=[c[0] for c in scored_columns[:3]]  # Log top 3
                    )
                elif not all_column_matches and column != "*":
                    column = None

                if not column:
                    logger.warning("semantic_column_not_found", metric=metric_name, table=table)
                    # Fall through to LLM-based mapping
                else:
                    # Find date column if needed (use DateFilterBuilder if available)
                    date_column = None
                    if pattern.get("requires_date_filter"):
                        # Use DateFilterBuilder if available (better date column discovery)
                        if self.date_filter_builder:
                            date_column = self.date_filter_builder.discover_date_column(table)
                            if date_column:
                                logger.info(
                                    "semantic_date_column_discovered",
                                    table=table,
                                    date_column=date_column,
                                    method="DateFilterBuilder"
                                )
                        
                        # Fallback to pattern matching if DateFilterBuilder didn't find one
                        if not date_column:
                            all_date_matches = []
                            date_patterns = pattern.get("date_column_patterns", ["created_at", "date", "timestamp"])
                            
                            for date_pattern in date_patterns:
                                matches = self.schema_discovery.find_columns_by_pattern(table, date_pattern)
                                if matches:
                                    all_date_matches.extend(matches)
                            
                            if all_date_matches:
                                # Score date columns with semantic relevance
                                scored_dates = []
                                metric_lower = metric_name.lower()
                                
                                for match_date in all_date_matches:
                                    score = 0
                                    date_lower = match_date.lower()
                                    
                                    # Prefer activity-related date columns for active metrics
                                    if "active" in metric_lower:
                                        if any(activity_word in date_lower for activity_word in ["session", "activity", "event"]):
                                            score += 5
                                    
                                    # Prefer common date column names
                                    if any(common_date in date_lower for common_date in ["date", "created_at", "timestamp"]):
                                        score += 3
                                    
                                    # Pattern order consideration (tiebreaker)
                                    pattern_index = next(
                                        (i for i, p in enumerate(date_patterns) 
                                         if self.schema_discovery.find_columns_by_pattern(table, p) and 
                                         match_date in self.schema_discovery.find_columns_by_pattern(table, p)),
                                        len(date_patterns)
                                    )
                                    score += (len(date_patterns) - pattern_index) * 0.5
                                    
                                    scored_dates.append((match_date, score))
                                
                                scored_dates.sort(key=lambda x: x[1], reverse=True)
                                date_column = scored_dates[0][0] if scored_dates else None
                                
                                logger.info(
                                    "semantic_date_column_selected_with_scoring",
                                    table=table,
                                    selected_date_column=date_column,
                                    method="pattern_matching"
                                )
                            
                            # Final fallback
                            if not date_column:
                                date_column = "created_at"
                                logger.warning(
                                    "semantic_date_column_fallback",
                                    table=table,
                                    fallback=date_column
                                )

                    result = {
                        "table": table,
                        "column": column,
                        "date_column": date_column or "created_at",  # Default fallback
                        "aggregation": pattern.get("aggregation", "SUM"),
                        "sql_template": pattern.get("sql_template", "")
                    }

                    # Check if concept name matches column name (needs confirmation if not)
                    # Similar to LLM mapping logic
                    concept_lower = metric_name.lower()
                    column_lower = column.lower()
                    needs_confirmation = concept_lower not in column_lower and column_lower not in concept_lower
                    
                    if needs_confirmation:
                        result["needs_confirmation"] = True
                        result["concept"] = metric_name
                    else:
                        result["needs_confirmation"] = False
                        
                    logger.info("semantic_metric_mapped", metric=metric_name, table=table, column=column)
                    return result

        # Fallback to LLM-based dynamic mapping
        # Try LLM mapping if reasoner is available (even if pattern matching failed)
        if self.reasoner and self.reasoner.enabled:
            logger.info("semantic_falling_back_to_llm", metric=metric_name)
            llm_result = self._llm_map_concept_to_schema(metric_name)
            if llm_result:
                return llm_result
        
        # If we get here, both pattern matching and LLM mapping failed
        logger.warning(
            "semantic_mapping_failed", 
            metric=metric_name,
            has_reasoner=self.reasoner is not None,
            reasoner_enabled=self.reasoner.enabled if self.reasoner else False
        )
        return None

    def _llm_map_concept_to_schema(self, concept: str) -> Optional[Dict[str, str]]:
        """
        Use LLM to dynamically map a concept to database schema.
        
        Args:
            concept: Concept name (e.g., "revenue", "active users", "total orders")
            
        Returns:
            Dict with keys: table, column, date_column, aggregation, or None if fails
        """
        if not self.reasoner or not self.reasoner.enabled:
            return None

        try:
            # Get all tables and their columns
            schema = self.schema_discovery.discover_full_schema()
            tables_info = []
            
            for table_name, columns in schema.items():
                column_names = [col["name"] for col in columns]
                tables_info.append({
                    "table": table_name,
                    "columns": column_names
                })

            # Build prompt for LLM
            prompt = f"""Given this database schema and a concept, select the best table and column to represent it.

Database Schema:
{json.dumps(tables_info, indent=2)}

Concept: {concept}

Rules:
- For "revenue", "sales", "income" → find table with amount/price/revenue columns
- For "users", "active users" → find table with user_id/user columns
- For "orders", "count" → find table with order_id/id columns
- For date filtering → find created_at/date/timestamp columns

Respond in JSON format only:
{{
    "table": "table_name",
    "column": "column_name",
    "date_column": "date_column_name",
    "aggregation": "SUM|COUNT|COUNT DISTINCT|AVG"
}}

If no suitable match, return null."""

            response = self.reasoner._call_llm(prompt)
            logger.info(
                "semantic_llm_mapping_response",
                concept=concept,
                response_preview=response[:200] if response else "None",
                response_length=len(response) if response else 0
            )

            # Parse JSON from response
            # First, try to extract from code blocks if present
            if "```json" in response:
                json_start = response.find("```json") + 7
                json_end = response.find("```", json_start)
                if json_end != -1:
                    response = response[json_start:json_end].strip()
            elif "```" in response:
                json_start = response.find("```") + 3
                json_end = response.find("```", json_start)
                if json_end != -1:
                    response = response[json_start:json_end].strip()

            # If no code blocks were found, extract JSON object directly
            # Always extract to handle cases where response has extra text or formatting
            json_start = response.find("{")
            if json_start != -1:
                # Find the matching closing brace
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
            
            # Try to parse JSON
            try:
                result = json.loads(response)
            except json.JSONDecodeError as e:
                logger.warning(
                    "semantic_llm_mapping_json_parse_error",
                    concept=concept,
                    error=str(e),
                    response_preview=response[:200]
                )
                return None
            
            # Check if JSON parsing succeeded and has required fields
            if not result:
                logger.warning("semantic_llm_mapping_no_result", concept=concept, response_preview=response[:200])
                return None
            
            if not result.get("table") or not result.get("column"):
                logger.warning(
                    "semantic_llm_mapping_missing_fields",
                    concept=concept,
                    has_table="table" in result,
                    has_column="column" in result,
                    result_keys=list(result.keys()) if result else []
                )
                return None
            
            if result and result.get("table") and result.get("column"):
                # Validate table and column exist
                # If schema discovery failed, still allow LLM mapping (with confirmation)
                if result["table"] in schema:
                    columns = [col["name"] for col in schema[result["table"]]]
                    if result["column"] in columns or result["column"] == "*":
                        logger.info("semantic_llm_mapping_success", concept=concept, table=result["table"], column=result["column"])

                        # Check if concept name matches column name (needs confirmation if not)
                        concept_lower = concept.lower()
                        column_lower = result["column"].lower()
                        needs_confirmation = concept_lower not in column_lower and column_lower not in concept_lower
                        
                        return {
                            "table": result["table"],
                            "column": result["column"],
                            "date_column": result.get("date_column", "created_at"),
                            "aggregation": result.get("aggregation", "SUM"),
                            "sql_template": "",
                            "needs_confirmation": needs_confirmation,
                            "concept": concept
                        }
                    else:
                        # Column not found in schema, but LLM suggested it - still allow with confirmation
                        logger.warning(
                            "semantic_llm_mapping_column_not_in_schema",
                            concept=concept,
                            table=result["table"],
                            column=result["column"],
                            available_columns=columns[:10]  # Log first 10 columns
                        )
                        # Still return mapping but require confirmation
                        return {
                            "table": result["table"],
                            "column": result["column"],
                            "date_column": result.get("date_column", "created_at"),
                            "aggregation": result.get("aggregation", "SUM"),
                            "sql_template": "",
                            "needs_confirmation": True,  # Always confirm if column not in schema
                            "concept": concept
                        }
                else:
                    # Table not found in schema, but LLM suggested it - log warning but still allow
                    logger.warning(
                        "semantic_llm_mapping_table_not_in_schema",
                        concept=concept,
                        table=result["table"],
                        available_tables=list(schema.keys())[:10]  # Log first 10 tables
                    )
                    # Still return mapping but require confirmation
                    return {
                        "table": result["table"],
                        "column": result["column"],
                        "date_column": result.get("date_column", "created_at"),
                        "aggregation": result.get("aggregation", "SUM"),
                        "sql_template": "",
                        "needs_confirmation": True,  # Always confirm if table not in schema
                        "concept": concept
                    }                            
            
            logger.warning("semantic_llm_mapping_invalid", concept=concept, response=response[:100])
            return None

        except Exception as e:
            logger.exception("semantic_llm_mapping_failed", concept=concept, error=str(e))
            return None

    def generate_sql(self, metric_name: str, date_filter: Optional[str] = None, query_text: Optional[str] = None) -> Optional[str]:
        """
        Generate SQL query for a metric using actual schema mapping.
        Now supports complex queries via ComplexSQLGenerator.
        
        Args:
            metric_name: Name of metric or concept
            date_filter: Optional date filter clause
            query_text: Optional original query text for complexity detection
            
        Returns:
            Generated SQL query string, or None if mapping fails
        """
        # Try pattern-based mapping first
        mapping = self.map_metric_to_schema(metric_name)
        if not mapping:
            # Try LLM-based SQL generation directly
            if self.reasoner and self.reasoner.enabled:
                return self._llm_generate_sql_from_concept(metric_name, date_filter)
            return None

        # Detect query complexity if query_text provided
        complexity = None
        if query_text and self.query_analyzer:
            try:
                complexity = self.query_analyzer.analyze_query_complexity(
                    query_text, 
                    metric_table=mapping.get("table") if mapping else None
                )
                # Use complex SQL generator for complex queries
                if complexity.complexity in ["medium", "high"]:
                    if self.complex_sql_generator:
                        return self.complex_sql_generator.generate_sql(
                            mapping=mapping,
                            query_text=query_text,
                            date_filter=date_filter,
                            complexity=complexity
                        )
            except Exception as e:
                logger.warning("semantic_complexity_detection_failed", error=str(e))
                # Fall through to simple generation

        # Simple query generation (existing logic)
        semantic_data = self.load_semantic_layer()
        pattern = semantic_data.get("metric_patterns", {}).get(metric_name, {})

        # Only add date filter if user explicitly requested it
        # Check if user query explicitly mentions a date/time period
        if not date_filter and query_text and self.date_filter_builder:
            # Extract date requirements from user query (only if explicitly mentioned)
            date_requirements = self.date_filter_builder.extract_date_requirements(query_text)
            
            # ONLY add date filter if user explicitly requested it
            if date_requirements.has_date_filter:
                date_filter_result = self.date_filter_builder.build_date_filter(
                    mapping["table"], date_requirements
                )
                date_filter = date_filter_result.date_filter
                logger.info("semantic_using_explicit_date_filter", metric=metric_name, date_type=date_requirements.date_type)
            # DO NOT add default date filter - respect user intent
        
        # Use template if available
        template = pattern.get("sql_template", "")
        if template:
            # For templates, only include date_filter if it exists (user requested it)
            # Replace {date_filter} with empty string if not provided
            date_filter_str = date_filter if date_filter else ""
            sql = template.format(
                table=mapping["table"],
                column=mapping["column"],
                date_column=mapping.get("date_column", "created_at"),
                date_filter=date_filter_str
            )
            # Remove empty WHERE clause if date_filter was empty
            if "WHERE " in sql and not date_filter:
                sql = sql.replace("WHERE ", "").strip()
        else:
            # Fallback: simple query generation
            aggregation = mapping.get("aggregation", "SUM")
            metric_name_alias = mapping.get("concept", metric_name)
            
            # Build WHERE clause (only if date filter exists)
            where_clause = ""
            if date_filter:
                where_clause = f"WHERE {date_filter}"
            # DO NOT add default WHERE clause - only add if user explicitly requested date filter
            
            # Build HAVING clause if needed (for conditional aggregations)
            having_clause = ""
            # TODO: Add HAVING clause detection from query text if needed
            
            if aggregation == "COUNT DISTINCT":
                sql = f"SELECT COUNT(DISTINCT {mapping['column']}) as {metric_name_alias} FROM {mapping['table']} {where_clause} {having_clause}".strip()
            else:
                aggregation_sql = aggregation
                sql = f"SELECT {aggregation_sql}({mapping['column']}) as {metric_name_alias} FROM {mapping['table']} {where_clause} {having_clause}".strip()
            
        logger.info("semantic_sql_generated", metric=metric_name, sql_preview=sql[:100], has_where=bool(where_clause))
        return sql

    def _llm_generate_sql_from_concept(self, concept: str, date_filter: Optional[str] = None) -> Optional[str]:
        """
        Use LLM to generate SQL directly from concept without pattern matching.
        Now uses PromptManager for database-agnostic prompts.
        
        Args:
            concept: Concept name (e.g., "revenue", "active users")
            date_filter: Optional date filter clause
            
        Returns:
            SQL query string or None if fails
        """
        if not self.reasoner or not self.reasoner.enabled:
            return None

        try:
            # Get schema info
            schema = self.schema_discovery.discover_full_schema()
            
            # Use PromptManager if available
            if self.prompt_manager:
                prompt = self.prompt_manager.build_sql_generation_prompt(
                    schema=schema,
                    concept=concept,
                    date_filter=date_filter or ""
                )
            else:
                # Fallback to old prompt (for backward compatibility)
                tables_info = []
                for table_name, columns in schema.items():
                    column_names = [col["name"] for col in columns]
                    tables_info.append({
                        "table": table_name,
                        "columns": column_names
                    })
                date_clause = date_filter or "created_at >= NOW() - INTERVAL '30 days'"
                prompt = f"""Generate a safe PostgreSQL SELECT query for this concept.

Database Schema:
{json.dumps(tables_info, indent=2)}

Concept: {concept}
Date Filter: {date_clause}

Rules:
- Only use SELECT statements
- Use appropriate aggregation (SUM for revenue, COUNT for users/orders)
- Include date filter if provided
- Return only the SQL query, no explanations
- Use actual table and column names from the schema

SQL:"""

            response = self.reasoner._call_llm(prompt)
            
            # Extract SQL from response
            sql = response.strip()
            if "```sql" in sql:
                sql = sql.split("```sql")[1].split("```")[0].strip()
            elif "```" in sql:
                sql = sql.split("```")[1].split("```")[0].strip()
            
            # Validate it's a SELECT query
            if sql.lower().startswith("select"):
                logger.info("semantic_llm_sql_generated", concept=concept, sql_preview=sql[:100])
                return sql
            else:
                logger.warning("semantic_llm_sql_invalid", concept=concept, sql_preview=sql[:100])
                return None

        except Exception as e:
            logger.exception("semantic_llm_sql_generation_failed", concept=concept, error=str(e))
            return None

    def get_routing_hints(self) -> Dict[str, List[str]]:
        """
        Get routing hints from semantic layer.
        
        Returns:
            Dict with keys: sql_keywords, pandasai_keywords, dataset_info_keywords
        """
        semantic_data = self.load_semantic_layer()
        return semantic_data.get("routing_hints", {})

    def get_metric_keywords(self) -> Dict[str, List[str]]:
        """
        Get all metric keywords for intent detection.
        
        Returns:
            Dict mapping metric_name -> keywords
        """
        semantic_data = self.load_semantic_layer()
        patterns = semantic_data.get("metric_patterns", {})
        
        return {
            name: pattern.get("keywords", [])
            for name, pattern in patterns.items()
        }

    def is_metric_available(self, metric_name: str) -> bool:
        """
        Check if a metric can be mapped to current database schema.
        Now supports LLM-based mapping as fallback.
        
        Args:
            metric_name: Name of metric to check
            
        Returns:
            True if metric can be mapped, False otherwise
        """
        mapping = self.map_metric_to_schema(metric_name)
        return mapping is not None

    def get_available_metrics(self) -> List[str]:
        """
        Get list of all available metrics from semantic layer.
        
        Returns:
            List of metric names
        """
        semantic_data = self.load_semantic_layer()
        patterns = semantic_data.get("metric_patterns", {})
        return list(patterns.keys())
    
    def get_database_context(self) -> Dict[str, Any]:
        """
        Get database context for query validation.
        
        Returns:
            Dict with keys: available_tables, available_metrics, database_name
        """
        # Get available tables from schema
        schema = self.schema_discovery.discover_full_schema()
        available_tables = list(schema.keys())
        
        # Get available metrics from semantic layer
        available_metrics = self.get_available_metrics()
        
        # Get database name if available
        database_name = None
        if hasattr(self.schema_discovery, 'get_database_name'):
            database_name = self.schema_discovery.get_database_name()
        
        return {
            "available_tables": available_tables,
            "available_metrics": available_metrics,
            "database_name": database_name
        }

    def validate_column_for_aggregation(
        self,
        column_name: str,
        aggregation: str,
        table: str
    ) -> Tuple[bool, Optional[str]]:
        """
        Validate that column can be used with aggregation.
        
        Args:
            column_name: Column name
            aggregation: Aggregation function (SUM, AVG, COUNT, etc.)
            table: Table name
            
        Returns:
            (is_valid, error_message)
        """
        if not SQL_VALIDATOR_AVAILABLE:
            # Fallback: basic existence check
            schema = self.schema_discovery.discover_full_schema()
            if table not in schema:
                return False, f"Table '{table}' not found"
            columns = [col["name"] for col in schema[table]]
            if column_name not in columns:
                return False, f"Column '{column_name}' not found in table '{table}'"
            return True, None
        
        schema = self.schema_discovery.discover_full_schema()
        column_info = next(
            (col for col in schema[table] if col["name"] == column_name),
            None
        )
        
        if not column_info:
            return False, f"Column '{column_name}' not found in table '{table}'"
        
        return SQLValidator.validate_column_for_operation(
            column_name=column_name,
            column_type=column_info.get("type", ""),
            operation="aggregation",
            table=table,
            schema=schema,
            aggregation=aggregation
        )
    
    def get_available_columns(self, table: str) -> List[str]:
        """
        Get list of available columns for a table.
        
        Args:
            table: Table name
            
        Returns:
            List of column names
        """
        schema = self.schema_discovery.discover_full_schema()
        if table not in schema:
            return []
        return [col["name"] for col in schema[table]]
    
    def suggest_expression_with_llm(
        self,
        concept: str,
        table: str,
        available_columns: List[str]
    ) -> Optional[Dict[str, Any]]:
        """
        Use LLM to suggest SQL expression based on available columns.
        Now uses PromptManager for database-agnostic prompts.
        
        Args:
            concept: Concept name (e.g., "revenue")
            table: Table name
            available_columns: List of available column names
            
        Returns:
            Dict with keys: expression, column, aggregation, or None if fails
        """
        if not self.reasoner or not self.reasoner.enabled:
            return None
        
        try:
            # Get column types if available
            column_types = {}
            schema = self.schema_discovery.discover_full_schema()
            if table in schema:
                column_types = {
                    col["name"]: col.get("type", "")
                    for col in schema[table]
                    if col["name"] in available_columns
                }
            
            # Use PromptManager if available
            if self.prompt_manager:
                prompt = self.prompt_manager.build_expression_suggestion_prompt(
                    table=table,
                    columns=available_columns,
                    column_types=column_types,
                    concept=concept
                )
            else:
                # Fallback to old prompt (for backward compatibility)
                prompt = f"""Given this table and available columns, suggest a SQL expression to calculate the concept.

Table: {table}
Available columns: {', '.join(available_columns)}
Concept: {concept}

Rules:
- For "revenue", "sales", "income" → use SUM() on amount/price/value columns
- For "users", "active users" → use COUNT DISTINCT on user_id/user columns
- For "orders", "count" → use COUNT on order_id/id columns
- Use appropriate aggregation function
- Return only the SQL SELECT expression (e.g., "SUM(amount_usd) as revenue")

Respond in JSON format:
{{
    "expression": "SUM(amount_usd) as revenue",
    "column": "amount_usd",
    "aggregation": "SUM",
    "reasoning": "brief explanation"
}}

If no suitable column exists, return null."""
            
            response = self.reasoner._call_llm(prompt)
            
            # Parse JSON from response
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
            
            # Try to parse JSON
            try:
                result = json.loads(response)
            except json.JSONDecodeError as e:
                logger.warning(
                    "semantic_llm_expression_json_parse_error",
                    concept=concept,
                    error=str(e),
                    response_preview=response[:200]
                )
                return None
            
            if result and result.get("expression") and result.get("column"):
                # Validate suggested column exists
                if result["column"] in available_columns:
                    return result
                else:
                    logger.warning(
                        "semantic_llm_suggested_invalid_column",
                        concept=concept,
                        suggested_column=result["column"],
                        available_columns=available_columns[:10]
                    )
            
            return None
            
        except Exception as e:
            logger.exception("semantic_llm_expression_suggestion_failed", concept=concept, error=str(e))
            return None    