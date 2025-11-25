# src/agent/knowledge/semantic_loader.py

"""
Semantic layer loader and mapper.

Bridges between semantic_layer.yaml patterns and actual database schema.
Maps metric concepts to real tables/columns using schema discovery.
"""

import yaml
from pathlib import Path
from typing import Dict, List, Optional, Any
from sqlalchemy import text

from agent.knowledge.database import Database
from agent.knowledge.schema_discovery import SchemaDiscovery
from agent.logging_config import logger


class SemanticLoader:
    """
    Loads semantic layer definitions and maps them to actual database schema.
    Works dynamically with any database by using pattern matching.
    """

    def __init__(self, database: Database, semantic_layer_path: Optional[Path] = None):
        """
        Initialize semantic loader.
        
        Args:
            database: Database instance for schema discovery
            semantic_layer_path: Optional path to semantic_layer.yaml (default: knowledge/semantic_layer.yaml)
        """
        self.database = database
        self.schema_discovery = SchemaDiscovery(database)
        
        if semantic_layer_path:
            self.semantic_layer_path = Path(semantic_layer_path)
        else:
            # Default location
            self.semantic_layer_path = Path(__file__).parent / "semantic_layer.yaml"
        
        self._semantic_data: Optional[Dict[str, Any]] = None
        self._schema_map: Optional[Dict[str, Dict[str, str]]] = None

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
        
        Args:
            metric_name: Name of metric from semantic layer
            
        Returns:
            Dict with keys: table, column, date_column (if found), or None if not found
        """
        semantic_data = self.load_semantic_layer()
        metric_patterns = semantic_data.get("metric_patterns", {})
        
        if metric_name not in metric_patterns:
            logger.warning("semantic_metric_not_found", metric=metric_name)
            return None

        pattern = metric_patterns[metric_name]
        schema = self.schema_discovery.discover_full_schema()
        
        # Find matching table
        table = None
        for table_pattern in pattern.get("table_patterns", []):
            matches = self.schema_discovery.find_tables_by_pattern(table_pattern)
            if matches:
                table = matches[0]  # Use first match
                break
        
        if not table:
            logger.warning("semantic_table_not_found", metric=metric_name, patterns=pattern.get("table_patterns"))
            return None

        # Find matching column
        column = None
        for col_pattern in pattern.get("column_patterns", []):
            if col_pattern == "*":
                column = "*"
                break
            matches = self.schema_discovery.find_columns_by_pattern(table, col_pattern)
            if matches:
                column = matches[0]
                break

        if not column:
            logger.warning("semantic_column_not_found", metric=metric_name, table=table)
            return None

        # Find date column if needed
        date_column = None
        if pattern.get("requires_date_filter"):
            date_patterns = pattern.get("date_column_patterns", ["created_at", "date", "timestamp"])
            for date_pattern in date_patterns:
                matches = self.schema_discovery.find_columns_by_pattern(table, date_pattern)
                if matches:
                    date_column = matches[0]
                    break

        result = {
            "table": table,
            "column": column,
            "date_column": date_column or "created_at",  # Default fallback
            "aggregation": pattern.get("aggregation", "SUM"),
            "sql_template": pattern.get("sql_template", "")
        }

        logger.info("semantic_metric_mapped", metric=metric_name, table=table, column=column)
        return result

    def generate_sql(self, metric_name: str, date_filter: Optional[str] = None) -> Optional[str]:
        """
        Generate SQL query for a metric using actual schema mapping.
        
        Args:
            metric_name: Name of metric
            date_filter: Optional date filter clause (e.g., "created_at >= NOW() - INTERVAL '30 days'")
            
        Returns:
            Generated SQL query string, or None if mapping fails
        """
        mapping = self.map_metric_to_schema(metric_name)
        if not mapping:
            return None

        semantic_data = self.load_semantic_layer()
        pattern = semantic_data.get("metric_patterns", {}).get(metric_name, {})
        
        # Use template if available
        template = pattern.get("sql_template", "")
        if template:
            sql = template.format(
                table=mapping["table"],
                column=mapping["column"],
                date_column=mapping["date_column"],
                date_filter=date_filter or f"{mapping['date_column']} >= NOW() - INTERVAL '30 days'"
            )
        else:
            # Fallback: simple query generation
            if date_filter:
                sql = f"SELECT {mapping['aggregation']}({mapping['column']}) as {metric_name} FROM {mapping['table']} WHERE {date_filter}"
            else:
                sql = f"SELECT {mapping['aggregation']}({mapping['column']}) as {metric_name} FROM {mapping['table']}"

        logger.info("semantic_sql_generated", metric=metric_name, sql_preview=sql[:100])
        return sql

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
        
        Args:
            metric_name: Name of metric to check
            
        Returns:
            True if metric can be mapped, False otherwise
        """
        mapping = self.map_metric_to_schema(metric_name)
        return mapping is not None