# src/agent/knowledge/query_analyzer.py

"""
Query Analyzer Module

Analyzes query complexity and extracts requirements for SQL generation.
Fixes issues:
- Wrong table routing (sessions vs users)
- Wrong metric detection
- Missing complexity detection
- Missing JOIN/window function/subquery detection
"""

from typing import Dict, List, Optional, Set, Any, Tuple
from pydantic import BaseModel, Field
import re

from agent.knowledge.schema_discovery import SchemaDiscovery
from agent.logging_config import logger


class QueryComplexity(BaseModel):
    """Pydantic model for query complexity analysis."""
    
    query_type: str = Field(description="Type: simple, multi_table, time_series, ranking, conditional, subquery")
    complexity: str = Field(description="Complexity level: low, medium, high")
    requires_joins: bool = Field(default=False, description="Whether query needs JOINs")
    requires_window_functions: bool = Field(default=False, description="Whether query needs window functions")
    requires_subqueries: bool = Field(default=False, description="Whether query needs subqueries")
    requires_case_statements: bool = Field(default=False, description="Whether query needs CASE statements")
    requires_grouping: bool = Field(default=False, description="Whether query needs GROUP BY")
    requires_having: bool = Field(default=False, description="Whether query needs HAVING clause")
    group_by_columns: List[str] = Field(default_factory=list, description="Columns to group by")
    tables_mentioned: List[str] = Field(default_factory=list, description="Tables mentioned in query")
    metrics_mentioned: List[str] = Field(default_factory=list, description="Metrics mentioned in query")


class QueryAnalyzer:
    """
    Analyzes query complexity and extracts requirements.
    
    Detects query type, complexity level, and required SQL features.
    Helps route queries to appropriate SQL generators.
    """
    
    # Keywords for different query types
    JOIN_KEYWORDS = ["join", "with", "across", "between", "compare", "per", "by"]
    TIME_SERIES_KEYWORDS = ["daily", "weekly", "monthly", "by day", "by week", "by month", "over time", "trend", "over the past"]
    RANKING_KEYWORDS = ["top", "bottom", "rank", "highest", "lowest", "best", "worst", "first", "last"]
    CONDITIONAL_KEYWORDS = ["if", "when", "case", "conditional", "depending", "based on"]
    SUBQUERY_KEYWORDS = ["with more than", "that have", "which have", "where exists"]
    GROUPING_KEYWORDS = ["by", "grouped by", "per", "each"]
    
    # Common metric keywords
    METRIC_KEYWORDS = {
        "revenue": ["revenue", "sales", "income", "money", "earnings"],
        "active_users": ["active users", "dau", "daily active users", "users", "unique users"],
        "total_orders": ["orders", "order count", "total orders"],
        "average_order_value": ["aov", "average order", "order value", "avg order"]
    }
    
    def __init__(self, schema_discovery: SchemaDiscovery):
        """
        Initialize query analyzer.
        
        Args:
            schema_discovery: SchemaDiscovery instance for table/column info
        """
        self.schema_discovery = schema_discovery
    
    def analyze_query_complexity(
        self, 
        query: str, 
        schema: Optional[Dict[str, Any]] = None,
        metric_table: Optional[str] = None
    ) -> QueryComplexity:
        """
        Analyze query complexity and extract requirements.
        
        Args:
            query: User query text
            schema: Optional pre-discovered schema
            
        Returns:
            QueryComplexity model with analysis results
        """
        query_lower = query.lower()
        
        # Discover schema if not provided
        if schema is None:
            schema = self.schema_discovery.discover_full_schema()
        
        # Initialize result
        complexity = QueryComplexity(
            query_type="simple",
            complexity="low"
        )
        
        # Detect tables mentioned in query
        complexity.tables_mentioned = self._detect_tables(query_lower, schema)
        
        # Detect metrics mentioned
        complexity.metrics_mentioned = self._detect_metrics(query_lower)

        # Use metric_table parameter if provided (from router)
        # Only try to infer if not provided and metric is detected
        if not metric_table and complexity.metrics_mentioned:
            # Fallback: Try to infer from schema (e.g., "active_users" → "sessions")
            # This is a fallback - router should pass metric_table from mapping
            # For now, keep as None and let semantic inference work without it
            pass
        
        # SEMANTIC INFERENCE: Detect grouping requirements from schema
        # Instead of hardcoding "countries", semantically match query to actual columns
        semantic_grouping, semantic_group_by_cols = self._infer_grouping_from_schema(
            query, schema, metric_table
        )
        if semantic_grouping:
            complexity.requires_grouping = True
            complexity.group_by_columns.extend(semantic_group_by_cols)
            if complexity.query_type == "simple":
                complexity.query_type = "multi_table"  # Grouping often implies multi-table
        
        # SEMANTIC INFERENCE: Detect JOIN requirements from schema
        # Check if query mentions columns from different tables
        semantic_joins = self._infer_join_requirements_from_schema(
            query, schema, metric_table, complexity.tables_mentioned
        )
        if semantic_joins:
            complexity.requires_joins = True
            if complexity.query_type == "simple":
                complexity.query_type = "multi_table"
            complexity.complexity = "medium"
        
        # Check for time series (don't overwrite if already set)
        if self._is_time_series(query_lower):
            complexity.requires_grouping = True
            if complexity.query_type == "simple":
                complexity.query_type = "time_series"
            complexity.complexity = "medium"
            time_grouping = self._extract_grouping_requirements(query_lower)
            for col in time_grouping:
                if col not in complexity.group_by_columns:
                    complexity.group_by_columns.append(col)
        
        # Check for "by X" patterns (keep as additional detection)
        if "by" in query_lower:
            complexity.requires_grouping = True
            by_matches = re.findall(r'by\s+(\w+)', query_lower)
            for match in by_matches:
                if match not in complexity.group_by_columns:
                    complexity.group_by_columns.append(match)
        
        # Check for "Which X" patterns (e.g., "Which countries", "Which users")
        which_matches = re.findall(r'which\s+(\w+)', query_lower)
        for match in which_matches:
            # Skip common words that aren't grouping columns
            if match not in ["one", "way", "table", "column", "query"]:
                if match not in complexity.group_by_columns:
                    complexity.group_by_columns.append(match)
                    complexity.requires_grouping = True
        
        # Check for ranking (don't overwrite time_series)
        if self._is_ranking(query_lower):
            complexity.requires_window_functions = True
            if complexity.query_type == "simple":
                complexity.query_type = "ranking"
            complexity.complexity = "high"
        
        # Check for conditional logic
        if self._is_conditional(query_lower):
            complexity.requires_case_statements = True
            if complexity.query_type == "simple":
                complexity.query_type = "conditional"
            complexity.complexity = "medium"
        
        # Check for subqueries
        if self._requires_subqueries(query_lower):
            complexity.requires_subqueries = True
            if complexity.query_type == "simple":
                complexity.query_type = "subquery"
            complexity.complexity = "high"

        # Check for HAVING clause requirements (conditional aggregations)
        if self._requires_having(query_lower):
            complexity.requires_having = True
            complexity.requires_grouping = True  # HAVING requires GROUP BY
            if complexity.query_type == "simple":
                complexity.query_type = "conditional"
            complexity.complexity = "medium" if complexity.complexity == "low" else complexity.complexity

        # Update complexity level based on multiple requirements (INCLUDE grouping)
        requirement_count = sum([
            complexity.requires_joins,
            complexity.requires_grouping,  # ADD THIS
            complexity.requires_window_functions,
            complexity.requires_subqueries,
            complexity.requires_having,
            complexity.requires_case_statements            
        ])
        
        if requirement_count >= 2:
            complexity.complexity = "high"
        elif requirement_count == 1:
            complexity.complexity = "medium"
        
        logger.info(
            "query_complexity_analyzed",
            query_type=complexity.query_type,
            complexity=complexity.complexity,
            requires_joins=complexity.requires_joins,
            requires_grouping=complexity.requires_grouping,
            requires_having=complexity.requires_having,
            tables=complexity.tables_mentioned,
            group_by_columns=complexity.group_by_columns
        )
        
        return complexity
    
    def _detect_tables(self, query_lower: str, schema: Dict[str, Any]) -> List[str]:
        """
        Detect tables mentioned in query.
        
        Args:
            query_lower: Lowercase query text
            schema: Database schema
            
        Returns:
            List of table names mentioned
        """
        tables = list(schema.keys())
        mentioned = []
        
        for table in tables:
            # Check if table name or related keywords appear in query
            if table.lower() in query_lower:
                mentioned.append(table)
            else:
                # Check for related keywords (e.g., "sessions" → "session", "session data")
                table_singular = table.rstrip('s')  # Remove plural 's'
                if table_singular in query_lower:
                    mentioned.append(table)
        
        return mentioned
    
    def _detect_metrics(self, query_lower: str) -> List[str]:
        """
        Detect metrics mentioned in query.
        
        Args:
            query_lower: Lowercase query text
            
        Returns:
            List of metric names detected
        """
        metrics = []
        
        for metric_name, keywords in self.METRIC_KEYWORDS.items():
            if any(keyword in query_lower for keyword in keywords):
                metrics.append(metric_name)
        
        return metrics
    
    def _requires_joins(self, query_lower: str, tables_mentioned: List[str]) -> bool:
        """Check if query requires JOINs."""
        # Multiple tables mentioned
        if len(tables_mentioned) > 1:
            return True
        
        # JOIN keywords present
        if any(keyword in query_lower for keyword in self.JOIN_KEYWORDS):
            # Check if it's actually about multiple tables
            if "per" in query_lower or "by" in query_lower:
                return True
        
        return False
    
    def _is_time_series(self, query_lower: str) -> bool:
        """Check if query is time series."""
        return any(keyword in query_lower for keyword in self.TIME_SERIES_KEYWORDS)
    
    def _is_ranking(self, query_lower: str) -> bool:
        """Check if query requires ranking."""
        return any(keyword in query_lower for keyword in self.RANKING_KEYWORDS)
    
    def _is_conditional(self, query_lower: str) -> bool:
        """Check if query requires conditional logic."""
        return any(keyword in query_lower for keyword in self.CONDITIONAL_KEYWORDS)
    
    def _requires_subqueries(self, query_lower: str) -> bool:
        """Check if query requires subqueries."""
        # Patterns like "with more than", "that have", "which have"
        if any(keyword in query_lower for keyword in self.SUBQUERY_KEYWORDS):
            return True
        
        # Pattern: "only for X that/which Y"
        if re.search(r'only\s+for\s+\w+\s+(?:that|which)', query_lower):
            return True
        
        return False
    
    def _requires_having(self, query_lower: str) -> bool:
        """
        Check if query requires HAVING clause for conditional aggregations.
        
        Args:
            query_lower: Lowercase query text
            
        Returns:
            True if HAVING clause is needed
        """
        # Patterns like "more than X", "greater than X", "at least X", "over X", "above X"
        having_patterns = [
            r"more than\s+\d+",
            r"greater than\s+\d+",
            r"at least\s+\d+",
            r"over\s+\d+",
            r"above\s+\d+",
            r"less than\s+\d+",
            r"fewer than\s+\d+",
            r"exactly\s+\d+",
            r"equal to\s+\d+"
        ]
        return any(re.search(pattern, query_lower) for pattern in having_patterns)

    def _extract_grouping_requirements(self, query_lower: str) -> List[str]:
        """
        Extract grouping requirements from query.
        
        Args:
            query_lower: Lowercase query text
            
        Returns:
            List of columns to group by
        """
        grouping = []
        
        # Extract "by X" patterns
        by_matches = re.findall(r'by\s+(\w+)', query_lower)
        grouping.extend(by_matches)
        
        # Extract "per X" patterns
        per_matches = re.findall(r'per\s+(\w+)', query_lower)
        grouping.extend(per_matches)
        
        # Check for date grouping
        if any(word in query_lower for word in ["daily", "by day", "day"]):
            grouping.append("date")
        if any(word in query_lower for word in ["weekly", "by week", "week"]):
            grouping.append("week")
        if any(word in query_lower for word in ["monthly", "by month", "month"]):
            grouping.append("month")
        
        return list(set(grouping))  # Remove duplicates

    def _semantically_match_query_to_columns(
        self, 
        query: str, 
        schema: Dict[str, Any],
        metric_table: Optional[str] = None
    ) -> Dict[str, Any]:
        """
        Semantically match query terms to actual database columns.
        Returns which columns match and which tables they're in.
        
        Args:
            query: User query text
            schema: Full database schema
            metric_table: Table where the metric is calculated (if known)
            
        Returns:
            Dict with:
            - matched_columns: List of (table, column, score) tuples
            - grouping_columns: Columns that likely need grouping
            - requires_joins: Whether JOINs are needed
        """
        query_lower = query.lower()
        # Filter out stop words and extract meaningful words
        stop_words = {"which", "what", "where", "when", "who", "how", "the", "a", "an", "is", "are", "have", "has", "had", "do", "does", "did", "than", "that", "this", "these", "those"}
        query_words = set(word for word in query_lower.split() if word not in stop_words and len(word) > 2)
        
        matched_columns = []
        grouping_columns = []
        
        # Scan all tables and columns in schema
        for table_name, columns in schema.items():
            for col_info in columns:
                col_name = col_info.get("name", "")
                col_lower = col_name.lower()
                
                # Semantic matching: check if query words match column name
                score = 0
                
                # Exact match
                if col_lower in query_lower:
                    score += 10
                
                # Word-level matching (e.g., "countries" matches "country")
                col_words = set(col_lower.split("_"))
                for q_word in query_words:
                    # Exact word match
                    if q_word in col_words:
                        score += 8
                    # Partial match (e.g., "countries" contains "country")
                    elif q_word in col_lower or col_lower in q_word:
                        score += 5

                    # Better plural/singular matching
                    # "countries" → "country", "users" → "user"
                    # Only strip 's' if word is longer than 3 chars (avoid "is" → "i")
                    q_stem = q_word.rstrip('s') if q_word.endswith('s') and len(q_word) > 3 else q_word
                    col_stem = col_lower.rstrip('s') if col_lower.endswith('s') and len(col_lower) > 3 else col_lower
                    # Check stem match (both directions)
                    if q_stem == col_stem or q_stem in col_stem or col_stem in q_stem:
                        score += 6  # Higher score for stem matches
                
                # If score is high enough, consider it a match
                if score >= 4:
                    matched_columns.append((table_name, col_name, score))
                    
                    # If this column is in a different table than metric, likely needs grouping
                    if metric_table and table_name != metric_table:
                        if col_name not in grouping_columns:
                            grouping_columns.append(col_name)
        
        # Sort by score
        matched_columns.sort(key=lambda x: x[2], reverse=True)
        
        # Infer JOIN requirements: if grouping columns are in different table than metric
        requires_joins = False
        if metric_table and grouping_columns:
            grouping_tables = set(
                table for table, col, _ in matched_columns 
                if col in grouping_columns and table != metric_table
            )
            if grouping_tables:
                requires_joins = True
        
        return {
            "matched_columns": matched_columns,
            "grouping_columns": grouping_columns,
            "requires_joins": requires_joins
        }
    
    def _infer_grouping_from_schema(
        self,
        query: str,
        schema: Dict[str, Any],
        metric_table: Optional[str] = None
    ) -> Tuple[bool, List[str]]:
        """
        Infer grouping requirements by semantically matching query to schema columns.
        
        Args:
            query: User query text
            schema: Full database schema
            metric_table: Table where metric is calculated
            
        Returns:
            (requires_grouping, group_by_columns)
        """
        # Use semantic matching to find columns mentioned in query
        matches = self._semantically_match_query_to_columns(query, schema, metric_table)
        
        grouping_columns = matches["grouping_columns"]
        requires_grouping = len(grouping_columns) > 0
        
        return requires_grouping, grouping_columns
    
    def _infer_join_requirements_from_schema(
        self,
        query: str,
        schema: Dict[str, Any],
        metric_table: Optional[str] = None,
        tables_mentioned: List[str] = None
    ) -> bool:
        """
        Infer JOIN requirements by checking if query mentions columns from multiple tables.
        
        Args:
            query: User query text
            schema: Full database schema
            metric_table: Table where metric is calculated
            tables_mentioned: Tables already detected in query
            
        Returns:
            True if JOINs are needed
        """
        # Multiple tables explicitly mentioned
        if tables_mentioned and len(tables_mentioned) > 1:
            return True
        
        # Use semantic matching to find columns
        matches = self._semantically_match_query_to_columns(query, schema, metric_table)
        
        # If semantic matching found columns in different tables, JOIN needed
        if matches["requires_joins"]:
            return True
        
        # Check if matched columns span multiple tables
        if metric_table:
            matched_tables = set(table for table, _, _ in matches["matched_columns"])
            if len(matched_tables) > 1 or (len(matched_tables) == 1 and metric_table not in matched_tables):
                return True
        
        return False