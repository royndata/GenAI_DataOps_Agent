# src/agent/output/output_formatter.py

"""
Output formatter for the GenAI DataOps Agent.

Formats tool results into Slack-friendly messages with:
- Table formatting for SQL results
- Text summarization for PandasAI results
- Chart path handling
- Truncation for Slack's 4000 character limit
- Error message formatting
- Metric result formatting

Follows AGENTS.md rules: Output module never touches DB.
"""

from typing import Any, Dict, List, Optional, Tuple
from agent.logging_config import logger


# Slack formatting constants
SLACK_MAX_MESSAGE_LENGTH = 4000
SLACK_TRUNCATE_AT = 3900
MAX_TABLE_ROWS_DISPLAY = 20
MAX_TABLE_COLUMNS_DISPLAY = 10


class OutputFormatter:
    """
    Production-grade output formatter for Slack messages.
    Converts structured tool results into human-readable Slack messages.
    """

    def __init__(self, max_message_length: int = SLACK_MAX_MESSAGE_LENGTH):
        """
        Initialize output formatter.
        
        Args:
            max_message_length: Maximum message length (default: Slack's 4000)
        """
        self.max_message_length = max_message_length
        self.truncate_at = max_message_length - 100  # Leave room for truncation notice
        logger.info("output_formatter_initialized", max_length=max_message_length)

    def format_router_response(self, response: Dict[str, Any]) -> str:
        """
        Format router response into Slack message.
        
        Args:
            response: Router response dict with keys: success, message, chart_path, raw
            
        Returns:
            Formatted Slack message string
        """
        if not response.get("success"):
            return self._format_error(response.get("message", "Unknown error"))

        # Router already provides formatted message
        message = response.get("message", "✅ Completed")
        
        # Add raw data formatting if available and message is short
        raw = response.get("raw")
        if raw and len(message) < 2000:
            formatted_raw = self._format_raw_data(raw)
            if formatted_raw:
                message += f"\n\n{formatted_raw}"

        return self._truncate_message(message)

    def format_sql_result(self, result: Dict[str, Any], include_raw: bool = False) -> str:
        """
        Format SQL query result into Slack message.
        
        Args:
            result: SQLTool result dict with keys: rows, columns, row_count, execution_time_ms
            include_raw: Whether to include raw data table
            
        Returns:
            Formatted Slack message
        """
        row_count = result.get("row_count", 0)
        execution_time = result.get("execution_time_ms", 0)
        rows = result.get("rows", [])
        columns = result.get("columns", [])
        truncated = result.get("truncated", False)

        # Header
        message = f"✅ *SQL Query Executed*\n"
        message += f"• Rows: {row_count}"
        if truncated:
            message += f" (truncated to {len(rows)})"
        message += f"\n• Time: {execution_time:.1f}ms"

        # Add complexity info if available
        complexity = result.get("complexity", {})
        if complexity.get("estimated_complexity"):
            message += f"\n• Complexity: {complexity['estimated_complexity']}"

        # Format table if requested and rows available
        if include_raw and rows:
            table = self._format_table(rows, columns)
            if table:
                message += f"\n\n*Results:*\n```{table}```"

        return self._truncate_message(message)

    def format_pandasai_result(self, result: Dict[str, Any], include_raw: bool = False) -> str:
        """
        Format PandasAI analysis result into Slack message.
        
        Args:
            result: PandasAITool result dict
            include_raw: Whether to include raw result data
            
        Returns:
            Formatted Slack message
        """
        if not result.get("success"):
            return self._format_error(result.get("error", "Analysis failed"))

        execution_time = result.get("execution_time_ms", 0)
        dataset_rows = result.get("dataset_rows", 0)
        chart_path = result.get("chart_path")
        parsed_result = result.get("result", {})

        # Header
        message = f"✅ *Analysis Completed*\n"
        message += f"• Dataset rows: {dataset_rows:,}\n"
        message += f"• Time: {execution_time:.1f}ms"

        if chart_path:
            message += f"\n• Chart: ✅ Generated"

        # Add parsed result if available
        if include_raw and parsed_result:
            result_text = self._format_parsed_result(parsed_result)
            if result_text:
                message += f"\n\n*Analysis Result:*\n{result_text}"

        return self._truncate_message(message)

    def format_metric_result(
        self,
        metric_name: str,
        value: Any,
        row_count: int = 0,
        execution_time: float = 0.0
    ) -> str:
        """
        Format metric calculation result.
        
        Args:
            metric_name: Name of the metric
            value: Calculated metric value
            row_count: Number of rows processed
            execution_time: Execution time in milliseconds
            
        Returns:
            Formatted Slack message
        """
        # Format metric name (replace underscores, title case)
        display_name = metric_name.replace("_", " ").title()

        message = f"✅ *{display_name}*\n"
        message += f"• Value: {self._format_value(value)}\n"
        message += f"• Rows: {row_count:,}\n"
        message += f"• Time: {execution_time:.1f}ms"

        return self._truncate_message(message)

    def format_dataset_info(self, info: Dict[str, Any]) -> str:
        """
        Format dataset metadata into Slack message.
        
        Args:
            info: Dataset info dict with keys: filename, rows, columns, etc.
            
        Returns:
            Formatted Slack message
        """
        filename = info.get("filename", "unknown")
        rows = info.get("rows", 0)
        columns = info.get("columns", [])

        message = f"📊 *Dataset: `{filename}`*\n"
        message += f"• Rows: {rows:,}\n"
        message += f"• Columns ({len(columns)}): {', '.join(columns[:10])}"
        
        if len(columns) > 10:
            message += f" ... (+{len(columns) - 10} more)"

        # Add memory usage if available
        memory_mb = info.get("memory_usage_mb")
        if memory_mb:
            message += f"\n• Memory: {memory_mb:.2f} MB"

        return self._truncate_message(message)

    # ------------------------------------------------------
    # Internal Formatting Helpers
    # ------------------------------------------------------

    def _format_table(self, rows: List[Any], columns: Optional[List[str]] = None) -> str:
        """
        Format rows into a text table.
        
        Args:
            rows: List of row data (tuples or dicts)
            columns: Optional column names
            
        Returns:
            Formatted table string, or empty if too large
        """
        if not rows:
            return ""

        # Limit rows and columns for display
        display_rows = rows[:MAX_TABLE_ROWS_DISPLAY]
        is_truncated = len(rows) > MAX_TABLE_ROWS_DISPLAY

        # Convert rows to list of lists
        table_data = []
        for row in display_rows:
            if isinstance(row, dict):
                values = list(row.values())
            elif isinstance(row, tuple):
                values = list(row)
            else:
                values = [str(row)]

            # Limit columns
            if columns and len(values) > MAX_TABLE_COLUMNS_DISPLAY:
                values = values[:MAX_TABLE_COLUMNS_DISPLAY]
            table_data.append(values)

        if not table_data:
            return ""

        # Get column names
        if not columns:
            columns = [f"col_{i+1}" for i in range(len(table_data[0]))]
        else:
            columns = columns[:MAX_TABLE_COLUMNS_DISPLAY]

        # Calculate column widths
        col_widths = [len(str(col)) for col in columns]
        for row in table_data:
            for i, val in enumerate(row):
                if i < len(col_widths):
                    col_widths[i] = max(col_widths[i], len(str(val)))

        # Build table
        lines = []
        
        # Header
        header = " | ".join(str(col).ljust(col_widths[i]) for i, col in enumerate(columns))
        lines.append(header)
        lines.append("-" * len(header))

        # Rows
        for row in table_data:
            row_str = " | ".join(
                str(val).ljust(col_widths[i]) if i < len(col_widths) else str(val)
                for i, val in enumerate(row)
            )
            lines.append(row_str)

        # Truncation notice
        if is_truncated:
            lines.append(f"\n... ({len(rows) - MAX_TABLE_ROWS_DISPLAY} more rows)")

        table = "\n".join(lines)
        
        # Check if table is too large for Slack
        if len(table) > 2000:
            return f"Table too large to display ({len(rows)} rows, {len(columns)} columns). Use SQL tool directly for full results."

        return table

    def _format_parsed_result(self, parsed: Dict[str, Any]) -> str:
        """
        Format PandasAI parsed result into text.
        
        Args:
            parsed: Parsed result dict with type and data
            
        Returns:
            Formatted text string
        """
        result_type = parsed.get("type", "text")
        data = parsed.get("data") or parsed.get("rows") or parsed.get("result")

        if result_type == "table" and isinstance(data, list):
            # Format as simple list
            if len(data) <= 5:
                return "\n".join(str(item) for item in data[:5])
            else:
                preview = "\n".join(str(item) for item in data[:3])
                return f"{preview}\n... ({len(data) - 3} more items)"

        elif result_type == "text":
            return str(data) if data else "No result"

        elif result_type == "dict":
            # Format dict as key-value pairs
            if isinstance(data, dict):
                lines = [f"• {k}: {v}" for k, v in list(data.items())[:10]]
                if len(data) > 10:
                    lines.append(f"... ({len(data) - 10} more keys)")
                return "\n".join(lines)

        return str(data) if data else "No result available"

    def _format_raw_data(self, raw: Dict[str, Any]) -> str:
        """
        Format raw tool result data.
        
        Args:
            raw: Raw result dict from tool
            
        Returns:
            Formatted text string, or empty if not applicable
        """
        if not raw:
            return ""

        # Check if it's a SQL result
        if "rows" in raw and "row_count" in raw:
            return self._format_table(raw.get("rows", []), raw.get("columns"))

        # Check if it's a PandasAI result
        if "result" in raw:
            return self._format_parsed_result(raw.get("result", {}))

        return ""

    def _format_value(self, value: Any) -> str:
        """
        Format a metric value for display.
        
        Args:
            value: Value to format
            
        Returns:
            Formatted string
        """
        if value is None:
            return "N/A"

        if isinstance(value, (int, float)):
            # Format numbers with commas
            if isinstance(value, float):
                return f"{value:,.2f}"
            return f"{value:,}"

        return str(value)

    def _format_error(self, error_message: str) -> str:
        """
        Format error message for Slack.
        
        Args:
            error_message: Error message string
            
        Returns:
            Formatted error message
        """
        if not error_message.startswith("⚠️"):
            return f"⚠️ {error_message}"
        return error_message

    def _truncate_message(self, message: str) -> str:
        """
        Truncate message to fit Slack's character limit.
        
        Args:
            message: Message to truncate
            
        Returns:
            Truncated message with notice if needed
        """
        if len(message) <= self.max_message_length:
            return message

        truncated = message[:self.truncate_at]
        # Try to truncate at a newline
        last_newline = truncated.rfind("\n")
        if last_newline > self.truncate_at - 200:  # If newline is reasonably close
            truncated = truncated[:last_newline]

        truncated += f"\n\n... (message truncated, {len(message)} chars total)"
        
        logger.warning(
            "output_formatter_message_truncated",
            original_length=len(message),
            truncated_length=len(truncated)
        )

        return truncated


# Convenience function for simple formatting
def format_response(response: Dict[str, Any]) -> str:
    """
    Convenience function to format router response.
    
    Args:
        response: Router response dict
        
    Returns:
        Formatted Slack message
    """
    formatter = OutputFormatter()
    return formatter.format_router_response(response)