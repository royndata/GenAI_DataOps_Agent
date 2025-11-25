# src/agent/ingestion/input_guardrails.py

"""
Input guardrails for the GenAI DataOps Agent.

Validates user queries before routing to prevent:
- Unbounded time ranges
- Unsafe SQL patterns
- Invalid metric requests
- System-breaking queries
- Long-running operations

Returns structured validation results with clear error messages.
"""

import re
from typing import Dict, Any, Optional, Tuple
from datetime import datetime, timedelta

from agent.logging_config import logger

# Optional semantic layer import (graceful degradation)
try:
    from agent.knowledge.semantic_loader import SemanticLoader
    SEMANTIC_LAYER_AVAILABLE = True
except ImportError:
    SEMANTIC_LAYER_AVAILABLE = False
    SemanticLoader = None


# Validation constants
MAX_INPUT_LENGTH = 3000
MIN_INPUT_LENGTH = 1
MAX_DATE_RANGE_DAYS = 365
MIN_DATE_RANGE_DAYS = 1
FORBIDDEN_KEYWORDS = {
    "drop", "delete", "truncate", "alter", "create",
    "grant", "revoke", "exec", "execute", "shutdown"
}
UNBOUNDED_TIME_PATTERNS = [
    r"all\s+(logs|data|records|history)",
    r"(entire|full|complete)\s+(history|dataset|database)",
    r"(\d+)\s+years?\s+(of|ago)",
    r"since\s+(the\s+)?beginning",
    r"everything",
    r"all\s+time"
]


class InputGuardrails:
    """
    Production-grade input validation and guardrails.
    Validates queries before routing to prevent dangerous operations.
    """

    def __init__(self, semantic_loader: Optional[SemanticLoader] = None):
        """
        Initialize input guardrails.
        
        Args:
            semantic_loader: Optional semantic loader for metric validation
        """
        self.semantic_loader = semantic_loader
        logger.info("input_guardrails_initialized", semantic_layer_enabled=semantic_loader is not None)

    def validate(self, text: str) -> Tuple[bool, Optional[str], Dict[str, Any]]:
        """
        Comprehensive input validation.
        
        Args:
            text: User input text to validate
            
        Returns:
            Tuple of (is_valid, error_message, validation_metadata)
        """
        metadata = {
            "length": len(text),
            "checks_passed": [],
            "checks_failed": []
        }

        # 1. Basic length validation
        if not text or not text.strip():
            metadata["checks_failed"].append("empty")
            return False, "Please provide a query or question.", metadata

        if len(text) < MIN_INPUT_LENGTH:
            metadata["checks_failed"].append("too_short")
            return False, f"Query too short (minimum {MIN_INPUT_LENGTH} character).", metadata

        if len(text) > MAX_INPUT_LENGTH:
            metadata["checks_failed"].append("too_long")
            return False, f"Query too long ({len(text)} chars). Maximum {MAX_INPUT_LENGTH} characters allowed.", metadata

        metadata["checks_passed"].append("length")

        # 2. Unbounded time range detection
        text_lower = text.lower()
        for pattern in UNBOUNDED_TIME_PATTERNS:
            if re.search(pattern, text_lower):
                metadata["checks_failed"].append("unbounded_time_range")
                return False, (
                    f"Unbounded time range detected. Please specify a date range "
                    f"(e.g., 'last 30 days', 'this month'). Maximum range: {MAX_DATE_RANGE_DAYS} days."
                ), metadata

        metadata["checks_passed"].append("time_range")

        # 3. Forbidden keyword detection
        for keyword in FORBIDDEN_KEYWORDS:
            if re.search(rf"\b{keyword}\b", text_lower):
                metadata["checks_failed"].append("forbidden_keyword")
                return False, f"Forbidden keyword detected: '{keyword}'. This operation is not allowed.", metadata

        metadata["checks_passed"].append("forbidden_keywords")

        # 4. Date range validation (if date range is specified)
        date_range_result = self._validate_date_range(text)
        if not date_range_result[0]:
            metadata["checks_failed"].append("date_range")
            return False, date_range_result[1], metadata

        metadata["checks_passed"].append("date_range")

        # 5. SQL injection pattern detection
        sql_injection_patterns = [
            r"';?\s*(drop|delete|insert|update)",
            r"union\s+select",
            r"exec\s*\(",
            r"xp_\w+",
        ]
        for pattern in sql_injection_patterns:
            if re.search(pattern, text_lower):
                metadata["checks_failed"].append("sql_injection_pattern")
                return False, "Potentially unsafe SQL pattern detected.", metadata

        metadata["checks_passed"].append("sql_safety")

        # 6. Semantic layer validation (if available)
        if self.semantic_loader:
            semantic_result = self._validate_against_semantic_layer(text)
            if not semantic_result[0]:
                # Warning, not error - allow query to proceed
                logger.warning("input_guardrails_semantic_warning", warning=semantic_result[1])
                metadata["warnings"] = metadata.get("warnings", []) + [semantic_result[1]]

        metadata["checks_passed"].append("semantic_validation")

        logger.info("input_guardrails_validation_passed", checks_passed=len(metadata["checks_passed"]))
        return True, None, metadata

    def _validate_date_range(self, text: str) -> Tuple[bool, Optional[str]]:
        """
        Validate date range if specified in query.
        
        Args:
            text: Query text
            
        Returns:
            Tuple of (is_valid, error_message)
        """
        text_lower = text.lower()

        # Extract numeric date ranges
        patterns = [
            (r"last\s+(\d+)\s+days?", 1),
            (r"(\d+)\s+days?\s+ago", 1),
            (r"last\s+(\d+)\s+months?", 30),
            (r"(\d+)\s+months?\s+ago", 30),
            (r"last\s+(\d+)\s+years?", 365),
            (r"(\d+)\s+years?\s+ago", 365),
        ]

        for pattern, multiplier in patterns:
            match = re.search(pattern, text_lower)
            if match:
                number = int(match.group(1))
                days = number * multiplier

                if days < MIN_DATE_RANGE_DAYS:
                    return False, f"Date range too short (minimum {MIN_DATE_RANGE_DAYS} day)."

                if days > MAX_DATE_RANGE_DAYS:
                    return False, (
                        f"Date range too large ({days} days). "
                        f"Maximum allowed: {MAX_DATE_RANGE_DAYS} days."
                    )

        return True, None

    def _validate_against_semantic_layer(self, text: str) -> Tuple[bool, Optional[str]]:
        """
        Validate query against semantic layer (if available).
        Checks if requested metrics are available.
        
        Args:
            text: Query text
            
        Returns:
            Tuple of (is_valid, warning_message)
        """
        if not self.semantic_loader:
            return True, None

        try:
            # Check if query mentions metrics that don't exist
            metric_keywords = self.semantic_loader.get_metric_keywords()
            text_lower = text.lower()

            mentioned_metrics = []
            for metric_name, keywords in metric_keywords.items():
                if any(keyword in text_lower for keyword in keywords):
                    mentioned_metrics.append(metric_name)

            # Check availability of mentioned metrics
            unavailable = []
            for metric in mentioned_metrics:
                if not self.semantic_loader.is_metric_available(metric):
                    unavailable.append(metric)

            if unavailable:
                return False, f"Metrics not available in current database: {', '.join(unavailable)}"

            return True, None

        except Exception as e:
            logger.warning("input_guardrails_semantic_check_failed", error=str(e))
            return True, None  # Don't block on semantic layer errors

    def normalize(self, text: str) -> str:
        """
        Normalize input text before routing.
        Removes extra whitespace, normalizes case for keywords, etc.
        
        Args:
            text: Raw input text
            
        Returns:
            Normalized text
        """
        # Remove extra whitespace
        text = " ".join(text.split())

        # Normalize common date expressions
        text = re.sub(r"\blast\s+(\d+)\s+days?\b", r"last \1 days", text, flags=re.IGNORECASE)
        text = re.sub(r"\bthis\s+month\b", "this month", text, flags=re.IGNORECASE)
        text = re.sub(r"\bthis\s+year\b", "this year", text, flags=re.IGNORECASE)

        return text.strip()


def validate_input(text: str, semantic_loader: Optional[SemanticLoader] = None) -> Tuple[bool, Optional[str]]:
    """
    Convenience function for simple validation.
    
    Args:
        text: Input text to validate
        semantic_loader: Optional semantic loader
        
    Returns:
        Tuple of (is_valid, error_message)
    """
    guardrails = InputGuardrails(semantic_loader)
    is_valid, error_msg, _ = guardrails.validate(text)
    return is_valid, error_msg