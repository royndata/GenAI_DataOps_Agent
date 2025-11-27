# src/agent/knowledge/date_filter_builder.py

"""
Date Filter Builder Module

Handles date filter generation, date column discovery, and date range validation.
Fixes issues:
- Date column mismatch (created_at vs actual date columns)
- Missing date filters in SQL
- Year filter extraction and application
- Date range validation before query execution
"""

from typing import Dict, List, Optional, Tuple, Any
from pydantic import BaseModel, Field
from datetime import datetime, date
import re

from agent.knowledge.schema_discovery import SchemaDiscovery
from agent.logging_config import logger


class DateRequirements(BaseModel):
    """Pydantic model for date requirements extracted from query."""
    
    has_date_filter: bool = Field(default=False, description="Whether query has date requirements")
    date_type: Optional[str] = Field(default=None, description="Type: 'last_n_days', 'year', 'month', 'range'")
    value: Optional[int] = Field(default=None, description="Numeric value (e.g., 30 for 'last 30 days')")
    year: Optional[int] = Field(default=None, description="Year value if specified")
    month: Optional[int] = Field(default=None, description="Month value if specified")
    start_date: Optional[str] = Field(default=None, description="Start date string")
    end_date: Optional[str] = Field(default=None, description="End date string")


class DateFilterResult(BaseModel):
    """Pydantic model for date filter result."""
    
    date_filter: str = Field(description="SQL date filter clause")
    date_column: str = Field(description="Actual date column name from schema")
    is_valid: bool = Field(default=True, description="Whether date range is valid")
    error_message: Optional[str] = Field(default=None, description="Error message if invalid")


class DateFilterBuilder:
    """
    Builds correct date filters using actual schema date columns.
    
    Discovers date columns from schema instead of hardcoding 'created_at'.
    Handles various date filter patterns: last N days, year, month, date ranges.
    """
    
    # Common date column patterns to search for
    DATE_COLUMN_PATTERNS = [
        "date", "created_at", "updated_at", "timestamp", 
        "payment_date", "order_date", "signup_date", "session_date"
    ]
    
    def __init__(self, schema_discovery: SchemaDiscovery):
        """
        Initialize date filter builder.
        
        Args:
            schema_discovery: SchemaDiscovery instance for discovering date columns
        """
        self.schema_discovery = schema_discovery
    
    def extract_date_requirements(self, query: str) -> DateRequirements:
        """
        Extract date requirements from natural language query.
        
        Args:
            query: User query text
            
        Returns:
            DateRequirements model with extracted date information
        """
        query_lower = query.lower()
        requirements = DateRequirements()
        
        # Pattern: "last N days" or "past N days"
        last_days_match = re.search(r'last\s+(\d+)\s+days?', query_lower)
        if last_days_match:
            requirements.has_date_filter = True
            requirements.date_type = "last_n_days"
            requirements.value = int(last_days_match.group(1))
            return requirements
        
        # Pattern: "this month" or "current month"
        if "this month" in query_lower or "current month" in query_lower:
            requirements.has_date_filter = True
            requirements.date_type = "month"
            return requirements
        
        # Pattern: "last 7 days" or "past week"
        if "last 7 days" in query_lower or "past week" in query_lower:
            requirements.has_date_filter = True
            requirements.date_type = "last_n_days"
            requirements.value = 7
            return requirements
        
        # Pattern: "year YYYY" or "for year YYYY"
        year_match = re.search(r'(?:for\s+)?year\s+(\d{4})', query_lower)
        if year_match:
            requirements.has_date_filter = True
            requirements.date_type = "year"
            requirements.year = int(year_match.group(1))
            return requirements
        
        # Pattern: "month YYYY-MM" or "in month"
        month_match = re.search(r'month\s+(\d{4})-(\d{1,2})', query_lower)
        if month_match:
            requirements.has_date_filter = True
            requirements.date_type = "month"
            requirements.year = int(month_match.group(1))
            requirements.month = int(month_match.group(2))
            return requirements
        
        # Pattern: "daily", "weekly", "monthly" (implies date grouping)
        if any(word in query_lower for word in ["daily", "weekly", "monthly", "by day", "by week"]):
            requirements.has_date_filter = True
            requirements.date_type = "grouping"  # Special type for GROUP BY date
            return requirements
        
        return requirements
    
    def discover_date_column(self, table: str, schema: Optional[Dict[str, Any]] = None) -> Optional[str]:
        """
        Discover actual date column from table schema.
        
        Searches for common date column patterns instead of hardcoding 'created_at'.
        
        Args:
            table: Table name
            schema: Optional pre-discovered schema (if None, will discover)
            
        Returns:
            Date column name or None if not found
        """
        if schema is None:
            schema = self.schema_discovery.discover_full_schema()
        
        if table not in schema:
            logger.warning("date_filter_table_not_found", table=table)
            return None
        
        columns = schema[table]
        
        # Search for date columns using patterns
        for pattern in self.DATE_COLUMN_PATTERNS:
            for col in columns:
                col_name_lower = col["name"].lower()
                col_type = col.get("type", "").lower()
                
                # Check if column name matches pattern
                if pattern in col_name_lower:
                    # Also verify it's a date type
                    if any(date_type in col_type for date_type in ["date", "timestamp", "time"]):
                        logger.info("date_column_discovered", table=table, column=col["name"])
                        return col["name"]
        
        # Fallback: check by type only
        for col in columns:
            col_type = col.get("type", "").lower()
            if any(date_type in col_type for date_type in ["date", "timestamp", "timestamptz"]):
                logger.info("date_column_discovered_by_type", table=table, column=col["name"])
                return col["name"]
        
        logger.warning("date_column_not_found", table=table)
        return None
    
    def build_date_filter(
        self, 
        table: str, 
        requirements: DateRequirements, 
        schema: Optional[Dict[str, Any]] = None
    ) -> DateFilterResult:
        """
        Build SQL date filter clause based on requirements.
        
        Args:
            table: Table name
            requirements: DateRequirements model
            schema: Optional pre-discovered schema
            
        Returns:
            DateFilterResult with SQL filter and date column
        """
        # Discover actual date column from schema
        date_column = self.discover_date_column(table, schema)
        
        if not date_column:
            # Fallback to common name (but log warning)
            date_column = "created_at"
            logger.warning("date_filter_using_fallback", table=table, fallback=date_column)
        
        # Build filter based on date type
        if requirements.date_type == "last_n_days":
            date_filter = f"{date_column} >= NOW() - INTERVAL '{requirements.value} days'"
        
        elif requirements.date_type == "month":
            if requirements.year and requirements.month:
                # Specific month: YYYY-MM
                date_filter = f"{date_column} >= '{requirements.year}-{requirements.month:02d}-01' AND {date_column} < '{requirements.year}-{requirements.month+1:02d}-01'"
            else:
                # Current month
                date_filter = f"{date_column} >= DATE_TRUNC('month', NOW())"
        
        elif requirements.date_type == "year":
            if requirements.year:
                # Specific year: YYYY
                date_filter = f"{date_column} >= '{requirements.year}-01-01' AND {date_column} < '{requirements.year + 1}-01-01'"
            else:
                # Current year
                date_filter = f"{date_column} >= DATE_TRUNC('year', NOW())"
        
        elif requirements.date_type == "grouping":
            # For daily/weekly/monthly queries, return filter for reasonable default range
            date_filter = f"{date_column} >= NOW() - INTERVAL '90 days'"
        
        else:
            # No specific date filter
            date_filter = ""
        
        return DateFilterResult(
            date_filter=date_filter,
            date_column=date_column or "created_at",
            is_valid=True
        )
    
    def validate_date_range(
        self, 
        table: str, 
        date_filter: str, 
        schema: Optional[Dict[str, Any]] = None
    ) -> Tuple[bool, Optional[str]]:
        """
        Validate if date filter is within available data range.
        
        Args:
            table: Table name
            date_filter: SQL date filter clause
            schema: Optional pre-discovered schema
            
        Returns:
            (is_valid, error_message)
        """
        if not date_filter:
            return True, None
        
        try:
            # Extract year from date filter if present
            year_match = re.search(r'(\d{4})', date_filter)
            if not year_match:
                return True, None  # Can't validate, allow query
            
            filter_year = int(year_match.group(1))
            current_year = datetime.now().year
            
            # Basic validation: reject years too far in future or past
            if filter_year > current_year + 10:
                return False, f"Date {filter_year} is too far in the future. Available data up to {current_year}"
            
            if filter_year < current_year - 50:
                return False, f"Date {filter_year} is too far in the past. Available data from {current_year - 10}"
            
            # TODO: Could query actual min/max dates from table for precise validation
            # For now, basic validation is sufficient
            
            return True, None
            
        except Exception as e:
            logger.warning("date_range_validation_failed", error=str(e))
            return True, None  # On error, allow query