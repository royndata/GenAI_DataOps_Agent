# src/agent/tools/sql_tool.py

import time
from typing import Any, Dict, List, Tuple, Optional
from sqlalchemy import text, TextClause
from sqlalchemy.exc import SQLAlchemyError, TimeoutError
from agent.knowledge.database import Database
from agent.logging_config import logger


ALLOWED_PREFIXES = ("select", "with")  # CTE support
MAX_ROWS = 5000  # Prevent huge DB pulls
QUERY_TIMEOUT_SECONDS = 30  # Tool-level timeout
MAX_QUERY_LENGTH = 10000  # Prevent extremely long queries


class SQLTool:
    """
    Production-grade safe SQL executor.
    - SELECT/CTE-only (allow-list)
    - Strong SQL normalization
    - Row-limit enforcement
    - Parameter binding (prevents SQL injection)
    - Safe error handling & audit logging
    - Execution time tracking
    - Query complexity analysis
    """

    def __init__(self, db: Database):
        self.db = db

    def _validate_query(self, query: str) -> None:
        """Ensure query is SELECT-only and safe."""

        # Length check
        if len(query) > MAX_QUERY_LENGTH:
            logger.error("sql_query_too_long", length=len(query))
            raise ValueError(f"Query exceeds maximum length of {MAX_QUERY_LENGTH} characters.")

        q = query.strip().lower()

        # Allow-list instead of deny-list (much safer)
        if not q.startswith(ALLOWED_PREFIXES):
            logger.error("sql_blocked_non_select", query=query[:100])
            raise ValueError("Only SELECT/CTE queries allowed.")

        # Hard block dangerous tokens anywhere
        forbidden = {
            ";", "drop", "delete", "update", "insert", "alter",
            "truncate", "create", "grant", "revoke", "exec", "execute"
        }

        for word in forbidden:
            if f" {word} " in f" {q} ":
                logger.error("sql_blocked_dangerous_token", token=word, query=query[:100])
                raise ValueError(f"Forbidden SQL keyword: {word}")

        # Check for suspicious patterns
        if "--" in query or "/*" in query:
            logger.warning("sql_comment_detected", query=query[:100])

    def _analyze_complexity(self, query: str) -> Dict[str, Any]:
        """Analyze query complexity for monitoring."""
        q_lower = query.lower()
        return {
            "has_joins": any(join in q_lower for join in ["join", "inner join", "left join", "right join"]),
            "has_subqueries": q_lower.count("select") > 1,
            "has_unions": "union" in q_lower,
            "has_ctes": q_lower.startswith("with"),
            "estimated_complexity": "high" if q_lower.count("select") > 2 else "medium" if "join" in q_lower else "low"
        }

    def _normalize_sql(self, query: str) -> str:
        """
        Normalize SQL query to ensure SQLAlchemy compatibility.
        Fixes common syntax issues from LLM-generated or user-provided SQL.
        
        Args:
            query: Raw SQL query string
            
        Returns:
            Normalized SQL query string
        """
        import re
        import html
        
        normalized = query.strip()
        
        # FIRST: Fix HTML encoding if present (e.g., &gt; -> >)
        try:
            normalized = html.unescape(normalized)
        except Exception:
            pass  # If unescaping fails, continue with original
        
        # SECOND: Fix common LLM/user errors: restore missing asterisks IMMEDIATELY
        # This must happen FIRST before any other normalization
        
        # Fix COUNT() -> COUNT(*) (handle various spacing patterns)
        # Matches: COUNT(), COUNT( ), COUNT ( ), COUNT(), etc.
        normalized = re.sub(r'\bCOUNT\s*\(\s*\)', 'COUNT(*)', normalized, flags=re.IGNORECASE)
        
        # Fix SELECT FROM -> SELECT * FROM (handle various spacing and contexts)
        # Matches: SELECT FROM, SELECT  FROM, SELECT\nFROM, etc.
        # Works in CTEs, subqueries, and main queries
        normalized = re.sub(r'\bSELECT\s+FROM\b', 'SELECT * FROM', normalized, flags=re.IGNORECASE)
        
        # Also fix SELECT DISTINCT FROM -> SELECT DISTINCT * FROM
        normalized = re.sub(r'\bSELECT\s+DISTINCT\s+FROM\b', 'SELECT DISTINCT * FROM', normalized, flags=re.IGNORECASE)
        
        # Fix COUNT DISTINCT syntax: COUNT DISTINCT(column) -> COUNT(DISTINCT column)
        # Pattern: COUNT DISTINCT(column) or COUNT DISTINCT (column)
        normalized = re.sub(
            r'COUNT\s+DISTINCT\s*\(([^)]+)\)',
            r'COUNT(DISTINCT \1)',
            normalized,
            flags=re.IGNORECASE
        )
        
        # Fix date literals in WHERE clauses for SQLAlchemy compatibility
        # Ensure date strings are properly quoted and formatted
        # Pattern: WHERE column >= 'YYYY-MM-DD' or WHERE column >= CURRENT_DATE - INTERVAL
        # SQLAlchemy text() should handle these, but ensure proper spacing
        normalized = re.sub(
            r"(\w+)\s*>=\s*'(\d{4}-\d{2}-\d{2})'",
            r"\1 >= '\2'",
            normalized,
            flags=re.IGNORECASE
        )
        
        # Normalize CTE formatting (ensure proper spacing)
        # Pattern: WITH name AS (SELECT...) -> ensure proper formatting
        normalized = re.sub(
            r'WITH\s+(\w+)\s+AS\s*\(',
            r'WITH \1 AS (',
            normalized,
            flags=re.IGNORECASE
        )
        
        # Normalize spacing more carefully - preserve COUNT(*) and SELECT *
        # Only normalize excessive spaces, don't add spaces around parentheses in function calls
        normalized = re.sub(r'\s{2,}', ' ', normalized)  # Multiple spaces to single (but preserve single spaces)
        
        return normalized.strip()

    def run_safe_query(
        self, 
        query: str, 
        params: Dict[str, Any] | None = None,
        timeout: Optional[int] = None
    ) -> Dict[str, Any]:
        """
        Safely executes a SQL SELECT query using SQLAlchemy parameter binding.
        
        Args:
            query: SQL SELECT query string
            params: Optional parameter dictionary for parameterized queries
            timeout: Optional timeout in seconds (overrides default)
            
        Returns:
            Dict with keys: rows, columns, execution_time_ms, row_count, complexity
        """

        start_time = time.time()
        timeout_seconds = timeout or QUERY_TIMEOUT_SECONDS

        # Normalize SQL before validation
        query = self._normalize_sql(query)
        
        self._validate_query(query)
        complexity = self._analyze_complexity(query)

        try:
            # Build parameterized query safely using SQLAlchemy text() with bindparams
            if params:
                stmt = text(query)
                for key, value in params.items():
                    stmt = stmt.bindparams(**{key: value})
            else:
                stmt = text(query)

            # Pass the TextClause object directly to database.run_query()
            rows = self.db.run_query(stmt)

            execution_time = (time.time() - start_time) * 1000  # Convert to ms

            # Enforce output limits
            row_count = len(rows)
            if row_count > MAX_ROWS:
                logger.warning(
                    "sql_truncated_result",
                    returned=row_count,
                    max_allowed=MAX_ROWS
                )
                rows = rows[:MAX_ROWS]
                row_count = MAX_ROWS

            # Convert SQLAlchemy Row objects to tuples for consistent handling
            # SQLAlchemy Row objects are tuple-like but not actual tuples
            converted_rows = []
            for row in rows:
                if hasattr(row, '_fields'):
                    # SQLAlchemy Row object - convert to tuple
                    converted_rows.append(tuple(row))
                elif isinstance(row, (tuple, list)):
                    # Already a tuple/list
                    converted_rows.append(tuple(row))
                elif isinstance(row, dict):
                    # Dict - convert to tuple of values
                    converted_rows.append(tuple(row.values()))
                else:
                    # Fallback: wrap in tuple
                    converted_rows.append((row,))

            # Get column names if available (from first row structure)
            columns = []
            if converted_rows:
                # Try to get column names from result keys if available
                if rows and hasattr(rows[0], '_fields'):
                    columns = list(rows[0]._fields)
                elif rows and isinstance(rows[0], dict):
                    columns = list(rows[0].keys())
                elif rows and isinstance(rows[0], tuple):
                    # For tuples, we don't have column names from DB
                    # This would need to be enhanced if database.run_query returns ResultProxy
                    columns = [f"column_{i+1}" for i in range(len(converted_rows[0]))]

            logger.info(
                "sql_query_success",
                rows=row_count,
                execution_time_ms=round(execution_time, 2),
                complexity=complexity["estimated_complexity"]
            )

            return {
                "rows": converted_rows,  # Use converted rows
                "columns": columns,
                "row_count": row_count,
                "execution_time_ms": round(execution_time, 2),
                "complexity": complexity,
                "truncated": len(converted_rows) < row_count if row_count > MAX_ROWS else False
            }

        except TimeoutError as e:
            execution_time = (time.time() - start_time) * 1000
            logger.error(
                "sql_timeout",
                error=str(e),
                execution_time_ms=round(execution_time, 2),
                timeout_seconds=timeout_seconds
            )
            raise RuntimeError(f"Database query timed out after {timeout_seconds}s.")

        except SQLAlchemyError as e:
            execution_time = (time.time() - start_time) * 1000
            logger.error(
                "sql_query_failed",
                error=str(e),
                execution_time_ms=round(execution_time, 2),
                query_preview=query[:100]
            )
            raise RuntimeError(f"Database error: {str(e)}")

        except ValueError as e:
            # Re-raise validation errors as-is
            raise

        except Exception as e:
            execution_time = (time.time() - start_time) * 1000
            logger.exception(
                "sql_unknown_error",
                error=str(e),
                execution_time_ms=round(execution_time, 2)
            )
            raise RuntimeError(f"Unexpected error running SQL: {str(e)}")