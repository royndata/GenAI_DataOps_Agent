# src/agent/knowledge/sql_validator.py

"""
SQL validation module for comprehensive column and operation validation.
Validates columns for aggregation, date filtering, grouping, joins, etc.
"""

from typing import Dict, List, Optional, Tuple, Any
from agent.logging_config import logger


class SQLValidator:
    """
    Validates SQL operations and column types for safe query generation.
    """
    
    NUMERIC_TYPES = ["integer", "bigint", "numeric", "decimal", "float", "double", "real", "money"]
    DATE_TYPES = ["date", "timestamp", "timestamptz", "datetime"]
    CATEGORICAL_TYPES = ["varchar", "text", "char", "enum"]
    
    @staticmethod
    def validate_column_for_operation(
        column_name: str,
        column_type: str,
        operation: str,
        table: str,
        schema: Dict[str, List[Dict[str, str]]],
        aggregation: Optional[str] = None,
        comparison_value: Optional[Any] = None
    ) -> Tuple[bool, Optional[str]]:
        """
        Validate column for any SQL operation.
        
        Args:
            column_name: Column name to validate
            column_type: Column data type
            operation: Operation type ("aggregation", "date_filter", "group_by", "join", "order_by", "where")
            table: Table name
            schema: Full schema dict
            aggregation: Aggregation function (if operation is "aggregation")
            comparison_value: Value being compared (if operation is "where")
            
        Returns:
            (is_valid, error_message)
        """
        # Get column metadata
        if table not in schema:
            return False, f"Table '{table}' not found in schema"
        
        column_info = next(
            (col for col in schema[table] if col["name"] == column_name),
            None
        )
        
        if not column_info:
            return False, f"Column '{column_name}' not found in table '{table}'"
        
        data_type = column_info.get("type", column_type).lower()
        
        # Operation-specific validation
        if operation == "aggregation":
            if aggregation in ["SUM", "AVG"]:
                if not any(numeric in data_type for numeric in SQLValidator.NUMERIC_TYPES):
                    return False, f"Cannot use {aggregation} on non-numeric column '{column_name}' (type: {data_type})"
            # COUNT and COUNT DISTINCT work on any type
            return True, None
        
        elif operation == "date_filter" or operation == "group_by_date":
            if not any(date in data_type for date in SQLValidator.DATE_TYPES):
                return False, f"Cannot use '{column_name}' for date operations (type: {data_type})"
            return True, None
        
        elif operation == "group_by":
            if not (any(cat in data_type for cat in SQLValidator.CATEGORICAL_TYPES) or 
                    any(date in data_type for date in SQLValidator.DATE_TYPES)):
                return False, f"Cannot GROUP BY numeric column '{column_name}' (type: {data_type})"
            return True, None
        
        elif operation == "where":
            if comparison_value:
                if isinstance(comparison_value, (int, float)) and "varchar" in data_type:
                    return False, f"Cannot compare string column '{column_name}' to numeric value"
                if isinstance(comparison_value, str) and "integer" in data_type:
                    return False, f"Cannot compare numeric column '{column_name}' to string value"
            return True, None
        
        elif operation == "join":
            # JOIN columns should exist in both tables (basic check)
            return True, None
        
        return True, None
    
    @staticmethod
    def validate_sql_expression(
        expression: str,
        table: str,
        schema: Dict[str, List[Dict[str, str]]],
        concept: str
    ) -> Tuple[bool, Optional[str], Optional[Dict[str, Any]]]:
        """
        Validate a complete SQL expression and extract column/aggregation info.
        
        Args:
            expression: SQL expression string
            table: Table name
            schema: Full schema dict
            concept: Concept being calculated
            
        Returns:
            (is_valid, error_message, parsed_info)
        """
        # Basic validation - check if expression contains valid table/column references
        if table not in schema:
            return False, f"Table '{table}' not found in schema", None
        
        available_columns = [col["name"] for col in schema[table]]
        
        # Extract column names from expression (simple pattern matching)
        # This is a basic check - full SQL parsing would be more robust
        import re
        column_pattern = r'\b([a-z_][a-z0-9_]*)\b'
        found_columns = re.findall(column_pattern, expression.lower())
        
        # Check if any found "columns" are actually SQL keywords
        sql_keywords = {"select", "from", "where", "group", "by", "order", "as", "sum", "count", "avg", "max", "min", "distinct", "case", "when", "then", "else", "end"}
        actual_columns = [col for col in found_columns if col not in sql_keywords]
        
        # Validate columns exist
        for col in actual_columns:
            if col not in available_columns:
                return False, f"Column '{col}' not found in table '{table}'", None
        
        parsed_info = {
            "table": table,
            "columns": actual_columns,
            "expression": expression
        }
        
        return True, None, parsed_info