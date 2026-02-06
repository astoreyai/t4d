"""
Input validation utilities for T4DM.

Provides consistent validation and error handling across all modules.
"""

import logging
import re
from datetime import datetime, timezone
from enum import Enum
from typing import Any, TypeVar
from uuid import UUID

import numpy as np

logger = logging.getLogger(__name__)

# =============================================================================
# Constants
# =============================================================================

# Maximum content length for memory content validation
MAX_CONTENT_LENGTH = 100000

# Maximum embedding dimension for vector validation
MAX_EMBEDDING_DIM = 4096

T = TypeVar("T", bound=Enum)


class ValidationError(ValueError):
    """Raised when input validation fails."""

    def __init__(self, field: str, message: str, value: Any = None):
        self.field = field
        self.message = message
        self.value = value
        super().__init__(f"{field}: {message}")

    def to_dict(self) -> dict:
        """Convert to error response format."""
        return {
            "error": "validation_error",
            "field": self.field,
            "message": self.message,
        }


# =============================================================================
# Session ID Validation
# =============================================================================


# Session ID validation pattern (alphanumeric, underscore, hyphen only)
SESSION_ID_PATTERN = re.compile(r"^[a-zA-Z0-9_\-]{1,128}$")

# Reserved session IDs that cannot be used
RESERVED_SESSION_IDS = frozenset([
    "admin", "system", "root", "default", "test",
    "null", "none", "undefined", "anonymous",
])


class SessionValidationError(ValidationError):
    """Session ID validation error."""


def validate_session_id(
    session_id: str | None,
    allow_none: bool = True,
    allow_reserved: bool = False,
) -> str | None:
    """
    Validate and sanitize session ID with comprehensive security checks.

    Prevents injection attacks by validating:
    - Type and length constraints
    - Character whitelist (alphanumeric, underscore, hyphen only)
    - Path traversal attempts
    - Null bytes and control characters
    - Reserved session IDs

    Args:
        session_id: Session ID to validate
        allow_none: Whether None/empty is allowed (default: True)
        allow_reserved: Whether reserved IDs are allowed (default: False)

    Returns:
        Validated session ID or None

    Raises:
        SessionValidationError: If session ID is invalid
    """
    if session_id is None or session_id == "":
        if allow_none:
            return None
        raise SessionValidationError(
            field="session_id",
            message="Session ID is required",
        )

    # Type check
    if not isinstance(session_id, str):
        raise SessionValidationError(
            field="session_id",
            message=f"Session ID must be string, got {type(session_id).__name__}",
        )

    # Strip whitespace
    session_id = session_id.strip()

    # Length check (max 128 chars)
    if len(session_id) > 128:
        raise SessionValidationError(
            field="session_id",
            message=f"Session ID too long ({len(session_id)} > 128 chars)",
        )

    if len(session_id) < 1:
        if allow_none:
            return None
        raise SessionValidationError(
            field="session_id",
            message="Session ID cannot be empty",
        )

    # Pattern check (alphanumeric, underscore, hyphen only)
    if not SESSION_ID_PATTERN.match(session_id):
        raise SessionValidationError(
            field="session_id",
            message=(
                "Session ID contains invalid characters. "
                "Only alphanumeric, underscore, and hyphen allowed."
            ),
        )

    # Reserved check (case-insensitive)
    if not allow_reserved and session_id.lower() in RESERVED_SESSION_IDS:
        raise SessionValidationError(
            field="session_id",
            message=f"Session ID '{session_id}' is reserved",
        )

    # Path traversal check (redundant with pattern check, but explicit)
    if ".." in session_id or "/" in session_id or "\\" in session_id:
        raise SessionValidationError(
            field="session_id",
            message="Session ID contains path traversal characters",
        )

    # Null byte check (security critical)
    if "\x00" in session_id:
        raise SessionValidationError(
            field="session_id",
            message="Session ID contains null bytes",
        )

    return session_id


def sanitize_session_id(session_id: str | None) -> str | None:
    """
    Sanitize session ID by removing dangerous characters.

    Use validate_session_id() for strict validation.
    This function is for cases where you want to clean
    rather than reject invalid input.

    Args:
        session_id: Session ID to sanitize

    Returns:
        Sanitized session ID or None
    """
    if session_id is None:
        return None

    if not isinstance(session_id, str):
        return None

    # Remove dangerous characters
    sanitized = session_id.strip()
    sanitized = re.sub(r"[^a-zA-Z0-9_\-]", "", sanitized)

    # Truncate to max length
    sanitized = sanitized[:128]

    return sanitized if sanitized else None


# =============================================================================
# UUID Validation
# =============================================================================


def validate_uuid(value: str, field: str = "id") -> UUID:
    """
    Validate and convert string to UUID.

    Args:
        value: String to validate as UUID
        field: Field name for error messages

    Returns:
        Parsed UUID object

    Raises:
        ValidationError: If value is not a valid UUID
    """
    if value is None:
        raise ValidationError(field, "UUID cannot be None")

    if not isinstance(value, str):
        raise ValidationError(field, f"Expected string, got {type(value).__name__}")

    try:
        return UUID(value)
    except (ValueError, AttributeError, TypeError) as e:
        raise ValidationError(field, f"Invalid UUID format: {value}") from e


def validate_uuid_list(values: list[str], field: str = "ids") -> list[UUID]:
    """
    Validate a list of UUID strings.

    Args:
        values: List of strings to validate as UUIDs
        field: Field name for error messages

    Returns:
        List of parsed UUID objects

    Raises:
        ValidationError: If any value is not a valid UUID
    """
    if not isinstance(values, list):
        raise ValidationError(field, f"Expected list, got {type(values).__name__}")

    result = []
    for i, v in enumerate(values):
        try:
            result.append(validate_uuid(v, f"{field}[{i}]"))
        except ValidationError:
            raise ValidationError(field, f"Invalid UUID at index {i}: {v}")

    return result


# =============================================================================
# Numeric Validation
# =============================================================================


def validate_range(
    value: float,
    min_val: float,
    max_val: float,
    field: str,
    allow_none: bool = False,
) -> float | None:
    """
    Validate float is within range.

    Args:
        value: Value to validate
        min_val: Minimum allowed value (inclusive)
        max_val: Maximum allowed value (inclusive)
        field: Field name for error messages
        allow_none: Whether None is acceptable

    Returns:
        Validated value

    Raises:
        ValidationError: If value is out of range
    """
    if value is None:
        if allow_none:
            return None
        raise ValidationError(field, "Value cannot be None")

    if not isinstance(value, (int, float)):
        raise ValidationError(field, f"Expected number, got {type(value).__name__}")

    if not (min_val <= value <= max_val):
        raise ValidationError(
            field,
            f"Must be between {min_val} and {max_val}, got {value}"
        )

    return float(value)


def validate_positive_int(
    value: int,
    field: str,
    max_val: int | None = None,
) -> int:
    """
    Validate positive integer (>= 1).

    Args:
        value: Value to validate
        field: Field name for error messages
        max_val: Optional maximum value

    Returns:
        Validated value

    Raises:
        ValidationError: If value is not a positive integer
    """
    if not isinstance(value, int) or isinstance(value, bool):
        raise ValidationError(field, f"Expected integer, got {type(value).__name__}")

    if value < 1:
        raise ValidationError(field, f"Must be positive, got {value}")

    if max_val is not None and value > max_val:
        raise ValidationError(field, f"Must be <= {max_val}, got {value}")

    return value


def validate_non_negative_int(
    value: int,
    field: str,
    max_val: int | None = None,
) -> int:
    """
    Validate non-negative integer (>= 0).

    Args:
        value: Value to validate
        field: Field name for error messages
        max_val: Optional maximum value

    Returns:
        Validated value

    Raises:
        ValidationError: If value is not a non-negative integer
    """
    if not isinstance(value, int) or isinstance(value, bool):
        raise ValidationError(field, f"Expected integer, got {type(value).__name__}")

    if value < 0:
        raise ValidationError(field, f"Must be non-negative, got {value}")

    if max_val is not None and value > max_val:
        raise ValidationError(field, f"Must be <= {max_val}, got {value}")

    return value


def validate_limit(value: Any, max_limit: int = 100, field: str = "limit") -> int:
    """
    Validate and constrain limit parameter.

    Silently caps at max_limit if exceeded.

    Args:
        value: Limit value to validate
        max_limit: Maximum allowed limit
        field: Field name for error messages

    Returns:
        Validated and capped limit

    Raises:
        ValidationError: If validation fails
    """
    try:
        limit = int(value)
    except (TypeError, ValueError):
        raise ValidationError(field, f"Must be an integer, got {type(value).__name__}")

    if limit < 1:
        raise ValidationError(field, "Must be at least 1")

    if limit > max_limit:
        logger.info(f"Limit {limit} exceeds max {max_limit}, capping at max")
        return max_limit

    return limit


def validate_float_range(
    value: Any,
    min_val: float = 0.0,
    max_val: float = 1.0,
    field: str = "value"
) -> float:
    """
    Validate float is within range.

    Args:
        value: Value to validate
        min_val: Minimum allowed value
        max_val: Maximum allowed value
        field: Field name for error messages

    Returns:
        Validated float

    Raises:
        ValidationError: If validation fails
    """
    try:
        f = float(value)
    except (TypeError, ValueError):
        raise ValidationError(field, f"Must be a number, got {type(value).__name__}")

    if f < min_val or f > max_val:
        raise ValidationError(field, f"Must be between {min_val} and {max_val}")

    return f


def validate_non_negative_float(
    value: Any,
    field: str = "value"
) -> float:
    """
    Validate value is a non-negative float.

    Args:
        value: Value to validate
        field: Field name for error messages

    Returns:
        Validated float

    Raises:
        ValidationError: If validation fails
    """
    try:
        f = float(value)
    except (TypeError, ValueError):
        raise ValidationError(field, f"Must be a number, got {type(value).__name__}")

    if f < 0:
        raise ValidationError(field, f"Must be non-negative, got {f}")

    return f


# =============================================================================
# String Validation
# =============================================================================


# XSS patterns to strip (common attack vectors)
_XSS_PATTERNS = [
    (re.compile(r"<script[^>]*>.*?</script>", re.IGNORECASE | re.DOTALL), ""),
    (re.compile(r"<script[^>]*>", re.IGNORECASE), ""),
    (re.compile(r"</script>", re.IGNORECASE), ""),
    (re.compile(r"javascript:", re.IGNORECASE), ""),
    (re.compile(r"on\w+\s*=", re.IGNORECASE), ""),  # onerror=, onclick=, etc.
    (re.compile(r"<iframe[^>]*>.*?</iframe>", re.IGNORECASE | re.DOTALL), ""),
    (re.compile(r"<iframe[^>]*/?>", re.IGNORECASE), ""),
    (re.compile(r"<embed[^>]*/?>", re.IGNORECASE), ""),
    (re.compile(r"<object[^>]*>.*?</object>", re.IGNORECASE | re.DOTALL), ""),
    (re.compile(r"<object[^>]*/?>", re.IGNORECASE), ""),
    (re.compile(r"<img[^>]*onerror[^>]*>", re.IGNORECASE), ""),  # img with onerror
    (re.compile(r"vbscript:", re.IGNORECASE), ""),
    (re.compile(r"data:text/html", re.IGNORECASE), ""),
]


def _sanitize_xss(value: str) -> str:
    """
    Remove common XSS attack patterns from string.

    This is a defense-in-depth measure. Content should also be
    escaped when rendered in HTML contexts.

    Args:
        value: String to sanitize

    Returns:
        Sanitized string with XSS patterns removed
    """
    result = value
    for pattern, replacement in _XSS_PATTERNS:
        result = pattern.sub(replacement, result)
    return result


def validate_non_empty_string(
    value: str,
    field: str,
    max_length: int | None = None,
    sanitize: bool = True
) -> str:
    """
    Validate and sanitize non-empty string.

    Args:
        value: String to validate
        field: Field name for error messages
        max_length: Optional maximum length
        sanitize: Whether to sanitize XSS/null bytes (default: True)

    Returns:
        Validated and sanitized string

    Raises:
        ValidationError: If value is empty, too long, or contains null bytes
    """
    if value is None:
        raise ValidationError(field, "Value cannot be None")

    if not isinstance(value, str):
        raise ValidationError(field, f"Expected string, got {type(value).__name__}")

    # Check for null bytes (security critical)
    if "\x00" in value:
        raise ValidationError(field, "Cannot contain null bytes")

    if not value.strip():
        raise ValidationError(field, "Cannot be empty")

    if max_length is not None and len(value) > max_length:
        raise ValidationError(field, f"Exceeds maximum length of {max_length}")

    if sanitize:
        value = _sanitize_xss(value)

    return value


def sanitize_string(value: str, max_length: int = 10000, field: str = "content") -> str:
    """
    Sanitize user input string.

    Removes dangerous control characters and XSS vectors while preserving readability.

    Args:
        value: String to sanitize
        max_length: Maximum allowed length
        field: Field name for error messages

    Returns:
        Sanitized string

    Raises:
        ValidationError: If validation fails
    """
    if not isinstance(value, str):
        raise ValidationError(field, f"Expected string, got {type(value).__name__}")

    if len(value) > max_length:
        raise ValidationError(field, f"Exceeds max length of {max_length} characters")

    # Remove null bytes (can cause issues in databases)
    value = value.replace("\x00", "")

    # Remove other potentially dangerous control characters (keep newlines, tabs, carriage returns)
    value = re.sub(r"[\x01-\x08\x0b\x0c\x0e-\x1f]", "", value)

    # Apply XSS sanitization
    value = _sanitize_xss(value)

    return value


def sanitize_identifier(value: str, field: str = "identifier") -> str:
    """
    Sanitize identifiers (names, labels, IDs).

    Only allows alphanumeric characters, underscores, and hyphens.

    Args:
        value: String to sanitize as identifier
        field: Field name for error messages

    Returns:
        Validated identifier

    Raises:
        ValidationError: If validation fails
    """
    if not isinstance(value, str):
        raise ValidationError(field, f"Expected string, got {type(value).__name__}")

    if not value:
        raise ValidationError(field, "Cannot be empty")

    if len(value) > 100:
        raise ValidationError(field, "Exceeds max length of 100 characters")

    if not re.match(r"^[a-zA-Z0-9_-]+$", value):
        raise ValidationError(field, "Only alphanumeric, underscore, and hyphen allowed")

    return value


# =============================================================================
# Memory Content Validation
# =============================================================================


def validate_memory_content(
    content: str,
    max_length: int = MAX_CONTENT_LENGTH,
    min_length: int = 1,
    field: str = "content",
) -> str:
    """
    Sanitize and validate memory content text input.

    Performs comprehensive validation:
    - Type check (must be string)
    - Length constraints (min/max)
    - Null byte removal
    - Control character removal (preserves newlines, tabs)
    - XSS pattern sanitization
    - Whitespace normalization

    Args:
        content: Text content to validate
        max_length: Maximum allowed length (default: MAX_CONTENT_LENGTH)
        min_length: Minimum required length (default: 1)
        field: Field name for error messages

    Returns:
        Sanitized and validated content string

    Raises:
        ValidationError: If validation fails
    """
    if content is None:
        raise ValidationError(field, "Content cannot be None")

    if not isinstance(content, str):
        raise ValidationError(
            field, f"Expected string, got {type(content).__name__}"
        )

    # Check for null bytes (security critical - reject early)
    if "\x00" in content:
        raise ValidationError(field, "Content contains null bytes")

    # Remove dangerous control characters (keep \n, \r, \t)
    sanitized = re.sub(r"[\x01-\x08\x0b\x0c\x0e-\x1f]", "", content)

    # Apply XSS sanitization
    sanitized = _sanitize_xss(sanitized)

    # Check length after sanitization
    if len(sanitized) > max_length:
        raise ValidationError(
            field,
            f"Content exceeds maximum length of {max_length} characters "
            f"(got {len(sanitized)})"
        )

    # Strip leading/trailing whitespace for length check
    stripped = sanitized.strip()
    if len(stripped) < min_length:
        if min_length == 1:
            raise ValidationError(field, "Content cannot be empty")
        raise ValidationError(
            field,
            f"Content must be at least {min_length} characters (got {len(stripped)})"
        )

    return sanitized


# =============================================================================
# Kappa Validation
# =============================================================================


def validate_kappa(kappa: float, field: str = "kappa") -> float:
    """
    Validate kappa consolidation level is in valid range [0, 1].

    Kappa represents memory consolidation progress:
    - 0.0: Raw episodic (just encoded)
    - ~0.15: Replayed (NREM strengthened)
    - ~0.4: Transitional (being abstracted)
    - ~0.85: Semantic concept (REM prototype)
    - 1.0: Stable knowledge (fully consolidated)

    Args:
        kappa: Consolidation level to validate
        field: Field name for error messages

    Returns:
        Validated kappa value as float

    Raises:
        ValidationError: If kappa is not in [0, 1] or not a number
    """
    if kappa is None:
        raise ValidationError(field, "Kappa cannot be None")

    if not isinstance(kappa, (int, float)):
        raise ValidationError(
            field, f"Expected number, got {type(kappa).__name__}"
        )

    # Check for NaN/Inf
    if not np.isfinite(kappa):
        raise ValidationError(field, "Kappa must be finite (not NaN or Inf)")

    # Convert to float
    kappa_float = float(kappa)

    # Check bounds
    if kappa_float < 0.0:
        raise ValidationError(
            field, f"Kappa must be >= 0, got {kappa_float}"
        )
    if kappa_float > 1.0:
        raise ValidationError(
            field, f"Kappa must be <= 1, got {kappa_float}"
        )

    return kappa_float


# =============================================================================
# Enum Validation
# =============================================================================


def validate_enum(value: str, enum_class: type[T], field: str) -> T:
    """
    Validate string is valid enum value.

    Args:
        value: String to validate
        enum_class: Enum class to validate against
        field: Field name for error messages

    Returns:
        Enum member

    Raises:
        ValidationError: If value is not a valid enum value
    """
    if value is None:
        raise ValidationError(field, "Value cannot be None")

    if not isinstance(value, str):
        raise ValidationError(field, f"Expected string, got {type(value).__name__}")

    try:
        return enum_class(value)
    except ValueError:
        valid_values = [e.value for e in enum_class]
        raise ValidationError(
            field,
            f"Invalid value '{value}'. Must be one of: {valid_values}"
        )


# =============================================================================
# Collection Validation
# =============================================================================


def validate_dict(value: Any, field: str, required_keys: list[str] | None = None) -> dict:
    """
    Validate dictionary with optional required keys.

    Args:
        value: Value to validate as dict
        field: Field name for error messages
        required_keys: Optional list of required keys

    Returns:
        Validated dict

    Raises:
        ValidationError: If value is not a dict or missing required keys
    """
    if value is None:
        return {}

    if not isinstance(value, dict):
        raise ValidationError(field, f"Expected dict, got {type(value).__name__}")

    if required_keys:
        missing = [k for k in required_keys if k not in value]
        if missing:
            raise ValidationError(field, f"Missing required keys: {missing}")

    return value


def validate_list(
    value: Any,
    field: str,
    min_length: int = 0,
    max_length: int | None = None,
) -> list:
    """
    Validate list with length constraints.

    Args:
        value: Value to validate as list
        field: Field name for error messages
        min_length: Minimum required length
        max_length: Optional maximum length

    Returns:
        Validated list

    Raises:
        ValidationError: If value is not a list or wrong length
    """
    if value is None:
        if min_length > 0:
            raise ValidationError(field, "Cannot be None")
        return []

    if not isinstance(value, list):
        raise ValidationError(field, f"Expected list, got {type(value).__name__}")

    if len(value) < min_length:
        raise ValidationError(field, f"Must have at least {min_length} items")

    if max_length is not None and len(value) > max_length:
        raise ValidationError(field, f"Cannot exceed {max_length} items")

    return value


def validate_metadata(metadata: Any, field: str = "metadata", max_depth: int = 5, _depth: int = 0) -> dict:
    """
    Validate metadata dictionary with recursive sanitization.

    Args:
        metadata: Metadata to validate
        field: Field name for error messages
        max_depth: Maximum nesting depth
        _depth: Internal depth tracker

    Returns:
        Validated and sanitized metadata

    Raises:
        ValidationError: If validation fails
    """
    if metadata is None:
        return {}

    if not isinstance(metadata, dict):
        raise ValidationError(field, f"Must be a dict, got {type(metadata).__name__}")

    if _depth >= max_depth:
        raise ValidationError(field, f"Exceeds maximum nesting depth of {max_depth}")

    # Recursively sanitize string values
    result = {}
    for k, v in metadata.items():
        # Sanitize key
        try:
            key = sanitize_identifier(str(k), field=f"{field}.key")
        except ValidationError:
            # Allow more permissive keys in metadata (but still sanitize)
            key = sanitize_string(str(k), max_length=100, field=f"{field}.key")

        # Sanitize value based on type
        if isinstance(v, str):
            result[key] = sanitize_string(v, max_length=1000, field=f"{field}.{key}")
        elif isinstance(v, (int, float, bool)):
            result[key] = v
        elif isinstance(v, dict):
            result[key] = validate_metadata(v, field=f"{field}.{key}", max_depth=max_depth, _depth=_depth + 1)
        elif isinstance(v, list):
            # Sanitize list elements
            result[key] = [
                sanitize_string(item, max_length=1000, field=f"{field}.{key}[{i}]")
                if isinstance(item, str) else item
                for i, item in enumerate(v[:100])  # Cap list length
            ]
        elif v is None:
            result[key] = None
        else:
            raise ValidationError(f"{field}.{key}", f"Invalid type: {type(v).__name__}")

    return result


# =============================================================================
# Convenience Validators
# =============================================================================


def validate_valence(value: float, field: str = "valence", allow_none: bool = False) -> float | None:
    """Validate valence is in [0, 1] range."""
    return validate_range(value, 0.0, 1.0, field, allow_none)


def validate_weight(value: float, field: str = "weight", allow_none: bool = False) -> float | None:
    """Validate weight is in [0, 1] range."""
    return validate_range(value, 0.0, 1.0, field, allow_none)


def validate_score(value: float, field: str = "score", allow_none: bool = False) -> float | None:
    """Validate score is in [0, 1] range."""
    return validate_range(value, 0.0, 1.0, field, allow_none)


# =============================================================================
# Array and Embedding Validation (NCA layer boundaries)
# =============================================================================

def validate_array(arr, *, expected_dim: int, name: str, allow_zero: bool = False, dtype=np.float32):
    """Validate numpy array: shape, dtype, NaN/Inf, zero-vector, norm bounds.

    Args:
        arr: Array to validate
        expected_dim: Expected dimensionality
        name: Name for error messages
        allow_zero: Whether zero vectors are allowed
        dtype: Expected dtype (default float32, also accepts float64)

    Raises:
        ValidationError: If validation fails
    """
    if arr is None:
        raise ValidationError(name, "Array cannot be None")
    if not isinstance(arr, np.ndarray):
        raise ValidationError(name, f"Expected numpy array, got {type(arr).__name__}")
    if arr.ndim != 1 or arr.shape[0] != expected_dim:
        raise ValidationError(name, f"Expected shape ({expected_dim},), got {arr.shape}")
    if not np.issubdtype(arr.dtype, np.floating):
        raise ValidationError(name, f"Expected floating dtype, got {arr.dtype}")
    if not np.all(np.isfinite(arr)):
        raise ValidationError(name, "Array contains NaN or Inf values")
    if not allow_zero and np.linalg.norm(arr) < 1e-10:
        raise ValidationError(name, "Zero-vector not allowed")


def validate_embedding(vec, *, dim: int, name: str):
    """Validate embedding vector: dimension, finite, non-zero.

    Args:
        vec: Embedding vector to validate
        dim: Expected dimension
        name: Name for error messages

    Raises:
        ValidationError: If validation fails
    """
    validate_array(vec, expected_dim=dim, name=name, allow_zero=False)


def validate_embedding_vector(
    embedding: np.ndarray,
    expected_dim: int,
    field: str = "embedding",
) -> np.ndarray:
    """
    Validate embedding vector with comprehensive checks.

    Validates:
    - Type (must be numpy array)
    - Shape (must be 1D with expected dimension)
    - No NaN or Inf values
    - Dimension within MAX_EMBEDDING_DIM limit
    - Floating point dtype

    Args:
        embedding: Embedding vector to validate
        expected_dim: Expected dimensionality of the embedding
        field: Field name for error messages

    Returns:
        Validated embedding array (may be converted to float32)

    Raises:
        ValidationError: If validation fails
    """
    if embedding is None:
        raise ValidationError(field, "Embedding cannot be None")

    if not isinstance(embedding, np.ndarray):
        raise ValidationError(
            field, f"Expected numpy array, got {type(embedding).__name__}"
        )

    # Check dimension limit
    if expected_dim > MAX_EMBEDDING_DIM:
        raise ValidationError(
            field,
            f"Expected dimension {expected_dim} exceeds maximum of {MAX_EMBEDDING_DIM}"
        )

    # Check shape
    if embedding.ndim != 1:
        raise ValidationError(
            field,
            f"Expected 1D array, got {embedding.ndim}D array with shape {embedding.shape}"
        )

    if embedding.shape[0] != expected_dim:
        raise ValidationError(
            field,
            f"Expected dimension {expected_dim}, got {embedding.shape[0]}"
        )

    # Check for empty embedding
    if embedding.size == 0:
        raise ValidationError(field, "Embedding cannot be empty")

    # Check for NaN/Inf values
    if not np.all(np.isfinite(embedding)):
        nan_count = np.sum(np.isnan(embedding))
        inf_count = np.sum(np.isinf(embedding))
        if nan_count > 0 and inf_count > 0:
            raise ValidationError(
                field, f"Embedding contains {nan_count} NaN and {inf_count} Inf values"
            )
        elif nan_count > 0:
            raise ValidationError(field, f"Embedding contains {nan_count} NaN values")
        else:
            raise ValidationError(field, f"Embedding contains {inf_count} Inf values")

    # Convert to float32 if needed for consistency
    if not np.issubdtype(embedding.dtype, np.floating):
        try:
            embedding = embedding.astype(np.float32)
        except (ValueError, TypeError) as e:
            raise ValidationError(
                field, f"Cannot convert embedding to float: {e}"
            ) from e
    elif embedding.dtype != np.float32:
        # Convert float64 to float32 for memory efficiency
        embedding = embedding.astype(np.float32)

    return embedding


def validate_timestamp(ts, *, max_future_s: float = 5.0, max_past_s: float = 86400 * 365):
    """Reject future timestamps, implausibly old timestamps.

    Args:
        ts: Timestamp as float (unix epoch) or datetime
        max_future_s: Max seconds into the future allowed (default 5s)
        max_past_s: Max seconds into the past allowed

    Raises:
        ValidationError: If timestamp is invalid
    """
    if ts is None:
        raise ValidationError("timestamp", "Timestamp cannot be None")

    if isinstance(ts, datetime):
        ts_epoch = ts.timestamp()
    elif isinstance(ts, (int, float)):
        ts_epoch = float(ts)
    else:
        raise ValidationError("timestamp", f"Expected float or datetime, got {type(ts).__name__}")

    if not np.isfinite(ts_epoch):
        raise ValidationError("timestamp", "Timestamp is not finite")

    now = datetime.now(timezone.utc).timestamp()
    if ts_epoch > now + max_future_s:
        raise ValidationError("timestamp", f"Timestamp is {ts_epoch - now:.1f}s in the future")
    if ts_epoch < now - max_past_s:
        raise ValidationError("timestamp", f"Timestamp is more than {max_past_s}s in the past")


def validate_nt_level(level: float, *, name: str, floor: float = 0.05, ceiling: float = 0.95) -> float:
    """Clamp and warn on out-of-range neuromodulator levels.

    Args:
        level: Neuromodulator level
        name: Name (e.g., "da", "ach")
        floor: Minimum allowed level
        ceiling: Maximum allowed level

    Returns:
        Clamped level
    """
    if not isinstance(level, (int, float)):
        raise ValidationError(name, f"Expected number, got {type(level).__name__}")
    if not np.isfinite(level):
        raise ValidationError(name, "Level is not finite")
    clamped = float(np.clip(level, floor, ceiling))
    if clamped != level:
        logger.warning(f"NT level {name} clamped from {level} to {clamped}")
    return clamped


def validate_memory_id(mid) -> str:
    """Validate UUID format for memory IDs.

    Args:
        mid: Memory ID (string or UUID)

    Returns:
        String UUID

    Raises:
        ValidationError: If not valid UUID
    """
    if mid is None:
        raise ValidationError("memory_id", "Memory ID cannot be None")
    mid_str = str(mid)
    # Use existing validate_uuid
    validate_uuid(mid_str, field="memory_id")
    return mid_str


def validate_spike_interval(prev_ts: float, curr_ts: float, *, min_isi_ms: float = 1.0):
    """Reject physiologically implausible inter-spike intervals.

    Args:
        prev_ts: Previous spike timestamp (seconds)
        curr_ts: Current spike timestamp (seconds)
        min_isi_ms: Minimum inter-spike interval in milliseconds (default 1.0ms)

    Raises:
        ValidationError: If interval is too short
    """
    dt_ms = (curr_ts - prev_ts) * 1000.0
    if dt_ms < min_isi_ms - 1e-9:  # Use small epsilon for floating point comparison
        raise ValidationError(
            "spike_interval",
            f"Inter-spike interval {dt_ms:.2f}ms < minimum {min_isi_ms}ms"
        )
