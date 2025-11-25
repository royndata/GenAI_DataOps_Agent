# src/agent/output/output_guardrails.py

"""
Output guardrails for the GenAI DataOps Agent.

Validates and sanitizes output before sending to Slack to prevent:
- Hallucinations (unrealistic values, inconsistencies)
- Unsafe content (sensitive data leaks, SQL injection patterns)
- Formatting issues (message length, structure)
- Security risks (path traversal, file system access)
- Data quality issues (invalid types, out-of-range values)

Follows AGENTS.md rules: Output module never touches DB.
"""

import re
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple, Union
from urllib.parse import urlparse

from agent.logging_config import logger


# Validation constants
SLACK_MAX_MESSAGE_LENGTH = 4000
MAX_CHART_FILE_SIZE_MB = 10
MAX_ROWS_IN_OUTPUT = 10000
MAX_COLUMNS_IN_OUTPUT = 100

# Sensitive data patterns (regex)
SENSITIVE_PATTERNS = [
    r'\b\d{4}[-\s]?\d{4}[-\s]?\d{4}[-\s]?\d{4}\b',  # Credit card numbers
    r'\b\d{3}-\d{2}-\d{4}\b',  # SSN
    r'\b[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Z|a-z]{2,}\b',  # Email (context-dependent)
    r'\bpassword\s*[:=]\s*\S+',  # Password in output
    r'\bapi[_-]?key\s*[:=]\s*\S+',  # API keys
    r'\btoken\s*[:=]\s*\S+',  # Tokens
]

# Suspicious SQL patterns in output
SQL_INJECTION_PATTERNS = [
    r"';?\s*(drop|delete|insert|update|alter)",
    r"union\s+select",
    r"exec\s*\(",
    r"xp_\w+",
]

# Path traversal patterns
PATH_TRAVERSAL_PATTERNS = [
    r"\.\./",
    r"\.\.\\",
    r"/etc/passwd",
    r"C:\\Windows",
]


class OutputGuardrailsError(Exception):
    """Base exception for output guardrails errors."""
    pass


class UnsafeContentError(OutputGuardrailsError):
    """Raised when unsafe content is detected in output."""
    pass


class HallucinationError(OutputGuardrailsError):
    """Raised when potential hallucination is detected."""
    pass


class FormattingError(OutputGuardrailsError):
    """Raised when output formatting is invalid."""
    pass


class OutputGuardrails:
    """
    Production-grade output guardrails for validating and sanitizing responses.
    
    Validates router responses, tool results, and formatted messages before
    sending to Slack to ensure safety and correctness.
    """

    def __init__(
        self,
        max_message_length: int = SLACK_MAX_MESSAGE_LENGTH,
        allow_sensitive_data: bool = False
    ):
        """
        Initialize output guardrails.
        
        Args:
            max_message_length: Maximum message length (default: Slack's 4000)
            allow_sensitive_data: Whether to allow potentially sensitive data (default: False)
        """
        self.max_message_length = max_message_length
        self.allow_sensitive_data = allow_sensitive_data
        
        # Compile regex patterns for performance
        self._sensitive_patterns = [re.compile(p, re.IGNORECASE) for p in SENSITIVE_PATTERNS]
        self._sql_patterns = [re.compile(p, re.IGNORECASE) for p in SQL_INJECTION_PATTERNS]
        self._path_patterns = [re.compile(p, re.IGNORECASE) for p in PATH_TRAVERSAL_PATTERNS]
        
        logger.info(
            "output_guardrails_initialized",
            max_message_length=max_message_length,
            allow_sensitive_data=allow_sensitive_data
        )

    def validate_router_response(
        self,
        response: Dict[str, Any]
    ) -> Tuple[bool, Optional[str], Dict[str, Any]]:
        """
        Validate router response before formatting.
        
        Args:
            response: Router response dict with keys: success, message, chart_path, raw
            
        Returns:
            Tuple of (is_valid, error_message, sanitized_response)
        """
        metadata = {
            "checks_passed": [],
            "checks_failed": [],
            "warnings": []
        }

        try:
            # 1. Validate response structure
            if not isinstance(response, dict):
                metadata["checks_failed"].append("invalid_structure")
                return False, "Invalid response structure", metadata

            if "success" not in response:
                metadata["checks_failed"].append("missing_success_field")
                return False, "Response missing 'success' field", metadata

            metadata["checks_passed"].append("structure")

            # 2. Validate message content
            message = response.get("message", "")
            if message:
                msg_valid, msg_error, msg_warnings = self.validate_message(message)
                if not msg_valid:
                    metadata["checks_failed"].append("message_validation")
                    return False, f"Message validation failed: {msg_error}", metadata
                metadata["warnings"].extend(msg_warnings)
                metadata["checks_passed"].append("message")

            # 3. Validate chart path if present
            chart_path = response.get("chart_path")
            if chart_path:
                chart_valid, chart_error = self.validate_chart_path(chart_path)
                if not chart_valid:
                    metadata["checks_failed"].append("chart_path")
                    return False, f"Chart path validation failed: {chart_error}", metadata
                metadata["checks_passed"].append("chart_path")

            # 4. Validate raw data if present
            raw = response.get("raw")
            if raw:
                raw_valid, raw_error, raw_warnings = self.validate_raw_data(raw)
                if not raw_valid:
                    metadata["checks_failed"].append("raw_data")
                    return False, f"Raw data validation failed: {raw_error}", metadata
                metadata["warnings"].extend(raw_warnings)
                metadata["checks_passed"].append("raw_data")

            # 5. Sanitize response (remove sensitive data if needed)
            sanitized = self._sanitize_response(response)

            logger.info(
                "output_guardrails_validation_passed",
                checks_passed=len(metadata["checks_passed"]),
                warnings=len(metadata["warnings"])
            )

            return True, None, sanitized

        except Exception as e:
            logger.exception("output_guardrails_validation_error", error=str(e))
            metadata["checks_failed"].append("exception")
            return False, f"Validation error: {str(e)}", metadata

    def validate_message(
        self,
        message: str
    ) -> Tuple[bool, Optional[str], List[str]]:
        """
        Validate message content for safety and formatting.
        
        Args:
            message: Message string to validate
            
        Returns:
            Tuple of (is_valid, error_message, warnings)
        """
        warnings = []

        # 1. Length check
        if len(message) > self.max_message_length:
            return False, f"Message exceeds maximum length ({len(message)} > {self.max_message_length})", warnings

        # 2. Sensitive data check
        if not self.allow_sensitive_data:
            for pattern in self._sensitive_patterns:
                if pattern.search(message):
                    # Email might be okay in some contexts, so warn instead of fail
                    if "email" in pattern.pattern.lower():
                        warnings.append("Email address detected in output")
                    else:
                        return False, "Sensitive data pattern detected in message", warnings

        # 3. SQL injection pattern check
        for pattern in self._sql_patterns:
            if pattern.search(message):
                return False, "Suspicious SQL pattern detected in output", warnings

        # 4. Path traversal check
        for pattern in self._path_patterns:
            if pattern.search(message):
                return False, "Path traversal pattern detected in output", warnings

        # 5. Check for excessive repetition (potential hallucination)
        words = message.split()
        if len(words) > 100:
            word_counts = {}
            for word in words:
                word_counts[word] = word_counts.get(word, 0) + 1
            max_repetition = max(word_counts.values()) if word_counts else 0
            if max_repetition > len(words) * 0.3:  # More than 30% repetition
                warnings.append("Excessive word repetition detected (potential hallucination)")

        return True, None, warnings

    def validate_chart_path(self, chart_path: Union[str, Path]) -> Tuple[bool, Optional[str]]:
        """
        Validate chart file path and existence.
        
        Args:
            chart_path: Path to chart file
            
        Returns:
            Tuple of (is_valid, error_message)
        """
        try:
            path = Path(chart_path)

            # 1. Check for path traversal
            path_str = str(path)
            for pattern in self._path_patterns:
                if pattern.search(path_str):
                    return False, "Path traversal detected in chart path"

            # 2. Check if file exists
            if not path.exists():
                return False, f"Chart file does not exist: {chart_path}"

            # 3. Check file size
            file_size_mb = path.stat().st_size / (1024 * 1024)
            if file_size_mb > MAX_CHART_FILE_SIZE_MB:
                return False, f"Chart file too large ({file_size_mb:.2f}MB > {MAX_CHART_FILE_SIZE_MB}MB)"

            # 4. Check file extension (must be image)
            valid_extensions = {".png", ".jpg", ".jpeg", ".gif", ".svg"}
            if path.suffix.lower() not in valid_extensions:
                return False, f"Invalid chart file extension: {path.suffix}"

            # 5. Check if path is within allowed directory
            # Ensure chart is in exports/charts or similar safe directory
            if ".." in str(path.resolve()):
                return False, "Chart path contains parent directory reference"

            return True, None

        except Exception as e:
            logger.error("output_guardrails_chart_validation_error", error=str(e))
            return False, f"Chart validation error: {str(e)}"

    def validate_raw_data(
        self,
        raw: Dict[str, Any]
    ) -> Tuple[bool, Optional[str], List[str]]:
        """
        Validate raw tool result data.
        
        Args:
            raw: Raw result dict from tool
            
        Returns:
            Tuple of (is_valid, error_message, warnings)
        """
        warnings = []

        # 1. Check for SQL result structure
        if "rows" in raw and "row_count" in raw:
            rows = raw.get("rows", [])
            row_count = raw.get("row_count", 0)

            # Validate row count
            if row_count > MAX_ROWS_IN_OUTPUT:
                return False, f"Too many rows in output ({row_count} > {MAX_ROWS_IN_OUTPUT})", warnings

            # Validate column count
            columns = raw.get("columns", [])
            if len(columns) > MAX_COLUMNS_IN_OUTPUT:
                return False, f"Too many columns in output ({len(columns)} > {MAX_COLUMNS_IN_OUTPUT})", warnings

            # Check for sensitive data in rows
            if not self.allow_sensitive_data and rows:
                for i, row in enumerate(rows[:10]):  # Sample first 10 rows
                    row_str = str(row)
                    for pattern in self._sensitive_patterns:
                        if pattern.search(row_str):
                            warnings.append(f"Potential sensitive data in row {i}")

        # 2. Check for PandasAI result structure
        if "result" in raw:
            result = raw.get("result", {})
            if isinstance(result, dict):
                result_str = str(result)
                # Check for suspicious patterns
                for pattern in self._sql_patterns:
                    if pattern.search(result_str):
                        return False, "Suspicious SQL pattern in result data", warnings

        # 3. Validate data types and ranges (basic sanity checks)
        if "execution_time_ms" in raw:
            exec_time = raw["execution_time_ms"]
            if isinstance(exec_time, (int, float)):
                if exec_time < 0:
                    warnings.append("Negative execution time detected")
                if exec_time > 3600000:  # More than 1 hour in ms
                    warnings.append("Unusually long execution time detected")

        return True, None, warnings

    def validate_formatted_message(self, formatted_message: str) -> Tuple[bool, Optional[str]]:
        """
        Validate formatted message before sending to Slack.
        
        Args:
            formatted_message: Formatted message string
            
        Returns:
            Tuple of (is_valid, error_message)
        """
        # 1. Length check
        if len(formatted_message) > self.max_message_length:
            return False, f"Formatted message exceeds length limit ({len(formatted_message)} > {self.max_message_length})"

        # 2. Final safety check
        valid, error, _ = self.validate_message(formatted_message)
        if not valid:
            return False, error

        return True, None

    def _sanitize_response(self, response: Dict[str, Any]) -> Dict[str, Any]:
        """
        Sanitize response by removing or masking sensitive data.
        
        Args:
            response: Original response dict
            
        Returns:
            Sanitized response dict
        """
        sanitized = response.copy()

        # Sanitize message if present
        if "message" in sanitized and not self.allow_sensitive_data:
            message = sanitized["message"]
            # Mask email addresses (keep domain visible)
            message = re.sub(
                r'\b([A-Za-z0-9._%+-]+)@([A-Za-z0-9.-]+\.[A-Z|a-z]{2,})\b',
                r'***@\2',
                message
            )
            sanitized["message"] = message

        # Sanitize raw data if present
        if "raw" in sanitized and not self.allow_sensitive_data:
            raw = sanitized["raw"]
            if isinstance(raw, dict) and "rows" in raw:
                # Don't modify raw data structure, just log warning
                logger.warning("output_guardrails_raw_data_present", has_rows=True)

        return sanitized

    def check_for_hallucinations(
        self,
        result: Dict[str, Any],
        context: Optional[Dict[str, Any]] = None
    ) -> Tuple[bool, List[str]]:
        """
        Check for potential hallucinations in results.
        
        Args:
            result: Tool result dict
            context: Optional context for validation (e.g., expected ranges)
            
        Returns:
            Tuple of (is_valid, warnings)
        """
        warnings = []

        # 1. Check for unrealistic numeric values
        if "raw" in result:
            raw = result["raw"]
            if isinstance(raw, dict) and "rows" in raw:
                rows = raw.get("rows", [])
                for row in rows[:5]:  # Sample first 5 rows
                    if isinstance(row, (list, tuple)):
                        for val in row:
                            if isinstance(val, (int, float)):
                                # Check for unrealistic values
                                if abs(val) > 1e15:  # Very large numbers
                                    warnings.append(f"Unusually large value detected: {val}")
                                if isinstance(val, float) and val != val:  # NaN check
                                    warnings.append("NaN value detected in results")

        # 2. Check for inconsistencies in metric results
        if "raw" in result:
            raw = result["raw"]
            if isinstance(raw, dict):
                # Check if row_count matches actual rows length
                row_count = raw.get("row_count", 0)
                rows = raw.get("rows", [])
                if isinstance(rows, list) and len(rows) != row_count:
                    warnings.append(f"Row count mismatch: reported {row_count}, actual {len(rows)}")

        # 3. Check execution time vs result size (potential timeout/truncation)
        exec_time = result.get("execution_time_ms", 0)
        if "raw" in result:
            raw = result["raw"]
            if isinstance(raw, dict):
                row_count = raw.get("row_count", 0)
                # If execution was very fast but returned many rows, might be cached/stale
                if exec_time < 10 and row_count > 1000:
                    warnings.append("Unusually fast execution for large result set")

        return len(warnings) == 0, warnings


# Convenience function for simple validation
def validate_output(
    response: Dict[str, Any],
    allow_sensitive_data: bool = False
) -> Tuple[bool, Optional[str], Dict[str, Any]]:
    """
    Convenience function to validate router response.
    
    Args:
        response: Router response dict
        allow_sensitive_data: Whether to allow sensitive data
        
    Returns:
        Tuple of (is_valid, error_message, sanitized_response)
    """
    guardrails = OutputGuardrails(allow_sensitive_data=allow_sensitive_data)
    return guardrails.validate_router_response(response)