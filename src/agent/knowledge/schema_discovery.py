# src/agent/knowledge/schema_discovery.py

"""
Database schema discovery module.

Automatically discovers table and column metadata from PostgreSQL database.
Used by semantic_loader to map semantic layer concepts to actual database schema.
"""

from typing import Dict, List, Optional, Any
from sqlalchemy import text, inspect
from sqlalchemy.engine import Engine

from agent.knowledge.database import Database
from agent.logging_config import logger


class SchemaDiscovery:
    """
    Discovers database schema structure dynamically.
    Works with any PostgreSQL database by querying information_schema.
    """

    def __init__(self, database: Database):
        """
        Initialize schema discovery with database connection.
        
        Args:
            database: Database instance with active connection
        """
        self.database = database
        self._schema_cache: Optional[Dict[str, Any]] = None

    def discover_tables(self) -> List[str]:
        """
        Discover all tables in the database.
        
        Returns:
            List of table names
        """
        try:
            query = text("""
                SELECT table_name 
                FROM information_schema.tables 
                WHERE table_schema = 'public' 
                AND table_type = 'BASE TABLE'
                ORDER BY table_name
            """)
            
            results = self.database.run_query(query)
            tables = [row[0] for row in results]
            
            logger.info("schema_discovery_tables_found", table_count=len(tables))
            return tables
            
        except Exception as e:
            logger.exception("schema_discovery_tables_failed", error=str(e))
            return []

    def discover_columns(self, table_name: str) -> List[Dict[str, str]]:
        """
        Discover columns for a specific table.
        
        Args:
            table_name: Name of the table
            
        Returns:
            List of column metadata dicts with keys: name, type, nullable
        """
        try:
            query = text("""
                SELECT 
                    column_name,
                    data_type,
                    is_nullable
                FROM information_schema.columns
                WHERE table_schema = 'public' 
                AND table_name = :table_name
                ORDER BY ordinal_position
            """)
            
            results = self.database.run_query(query, params={"table_name": table_name})
            
            columns = [
                {
                    "name": row[0],
                    "type": row[1],
                    "nullable": row[2] == "YES"
                }
                for row in results
            ]
            
            logger.info("schema_discovery_columns_found", table=table_name, column_count=len(columns))
            return columns
            
        except Exception as e:
            logger.exception("schema_discovery_columns_failed", table=table_name, error=str(e))
            return []

    def discover_full_schema(self) -> Dict[str, Any]:
        """
        Discover complete database schema (tables + columns).
        Results are cached for performance.
        
        Returns:
            Dict with structure: {table_name: [column_metadata]}
        """
        if self._schema_cache:
            logger.info("schema_discovery_cache_hit")
            return self._schema_cache

        try:
            tables = self.discover_tables()
            schema = {}
            
            for table in tables:
                columns = self.discover_columns(table)
                schema[table] = columns
            
            self._schema_cache = schema
            logger.info("schema_discovery_complete", table_count=len(schema))
            return schema
            
        except Exception as e:
            logger.exception("schema_discovery_failed", error=str(e))
            return {}

    def find_tables_by_pattern(self, pattern: str) -> List[str]:
        """
        Find tables matching a pattern (case-insensitive).
        
        Args:
            pattern: Pattern to match (e.g., "sales", "user*")
            
        Returns:
            List of matching table names
        """
        tables = self.discover_tables()
        pattern_lower = pattern.lower()
        
        # Simple pattern matching
        if "*" in pattern_lower:
            # Wildcard pattern
            prefix = pattern_lower.replace("*", "")
            matches = [t for t in tables if t.lower().startswith(prefix)]
        else:
            # Exact or substring match
            matches = [t for t in tables if pattern_lower in t.lower()]
        
        logger.info("schema_discovery_pattern_match", pattern=pattern, matches=len(matches))
        return matches

    def find_columns_by_pattern(self, table_name: str, pattern: str) -> List[str]:
        """
        Find columns in a table matching a pattern.
        
        Args:
            table_name: Table to search
            pattern: Column name pattern (e.g., "amount", "price", "revenue")
            
        Returns:
            List of matching column names
        """
        columns = self.discover_columns(table_name)
        pattern_lower = pattern.lower()
        
        matches = [
            col["name"] 
            for col in columns 
            if pattern_lower in col["name"].lower()
        ]
        
        return matches

    def clear_cache(self) -> None:
        """Clear schema cache to force re-discovery."""
        self._schema_cache = None
        logger.info("schema_discovery_cache_cleared")