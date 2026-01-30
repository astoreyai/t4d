"""
API error handling utilities.

Provides safe error message sanitization to prevent information leakage.
"""

import logging
import re

logger = logging.getLogger(__name__)

# Patterns to redact from error messages
SENSITIVE_PATTERNS = [
    # Connection strings with credentials
    r"(bolt|neo4j)://[^@]+:[^@]+@",
    # API keys and tokens
    r"(api[_-]?key|token|secret|password|auth)\s*[:=]\s*['\"]?[a-zA-Z0-9_-]+",
    # File paths with usernames
    r"/home/[^/\s]+",
    r"/Users/[^/\s]+",
    # IP addresses with ports
    r"\d{1,3}\.\d{1,3}\.\d{1,3}\.\d{1,3}:\d+",
    # Stack trace internal paths
    r'File "[^"]+site-packages[^"]*"',
]

# Generic error messages by exception type
GENERIC_MESSAGES = {
    "ConnectionError": "Service connection failed",
    "TimeoutError": "Request timed out",
    "ValueError": "Invalid input data",
    "TypeError": "Data type mismatch",
    "KeyError": "Required field missing",
    "PermissionError": "Access denied",
    "RuntimeError": "Internal processing error",
}


def sanitize_error(
    error: Exception,
    context: str = "operation",
    include_type: bool = True,
) -> str:
    """
    Sanitize an error message for safe API response.

    Args:
        error: The exception to sanitize
        context: Short description of what operation failed
        include_type: Whether to include error type in message

    Returns:
        Safe error message without sensitive information
    """
    error_type = type(error).__name__
    error_msg = str(error)

    # Log full error for debugging
    logger.error(f"Error in {context}: {error_type}: {error_msg}")

    # Check for known exception types with generic messages
    if error_type in GENERIC_MESSAGES:
        if include_type:
            return f"{context}: {GENERIC_MESSAGES[error_type]}"
        return GENERIC_MESSAGES[error_type]

    # Redact sensitive patterns
    sanitized = error_msg
    for pattern in SENSITIVE_PATTERNS:
        sanitized = re.sub(pattern, "[REDACTED]", sanitized, flags=re.IGNORECASE)

    # Truncate long messages
    if len(sanitized) > 200:
        sanitized = sanitized[:200] + "..."

    # Remove any remaining potentially sensitive info
    # Keep only alphanumeric, spaces, and basic punctuation
    sanitized = re.sub(r"[^\w\s.,!?()-]", "", sanitized)

    if include_type:
        return f"Failed to {context}: {sanitized}"
    return sanitized


def create_error_response(
    status_code: int,
    error: Exception,
    context: str = "process request",
) -> dict:
    """
    Create a standardized error response dict.

    Args:
        status_code: HTTP status code
        error: The exception
        context: Description of failed operation

    Returns:
        Error response dict
    """
    return {
        "status": "error",
        "code": status_code,
        "detail": sanitize_error(error, context),
    }
