# Security Fixes - Implementation Guide

Concrete code changes to address findings from security audit.

## Priority 1: Critical Path Fixes (Deploy Before Production)

### Fix M-1: Cypher Injection Prevention

**File**: `src/t4dm/storage/neo4j_store.py`

Add label/type validation at the beginning of the file:

```python
# Add after imports, before Neo4jStore class

# Whitelist of allowed node labels
ALLOWED_LABELS = {"Episode", "Entity", "Procedure"}

# Whitelist of allowed relationship types (import from types.py)
from ww.core.types import RelationType
ALLOWED_REL_TYPES = {rt.value for rt in RelationType}

def validate_label(label: str) -> str:
    """Validate node label against whitelist."""
    if label not in ALLOWED_LABELS:
        raise ValueError(
            f"Invalid node label: {label}. Must be one of {ALLOWED_LABELS}"
        )
    return label

def validate_rel_type(rel_type: str) -> str:
    """Validate relationship type against whitelist."""
    if rel_type not in ALLOWED_REL_TYPES:
        raise ValueError(
            f"Invalid relationship type: {rel_type}. Must be one of {ALLOWED_REL_TYPES}"
        )
    return rel_type
```

Update `create_node()` method (line 137):

```python
async def create_node(
    self,
    label: str,
    properties: dict[str, Any],
) -> str:
    # SECURITY: Validate label before use in query
    label = validate_label(label)

    async def _create():
        driver = await self._get_driver()
        props = self._serialize_props(properties)

        async with driver.session(database=self.database) as session:
            result = await session.run(
                f"""
                CREATE (n:{label} $props)
                RETURN n.id as id
                """,
                props=props,
            )
            record = await result.single()
            return record["id"]

    return await self._with_timeout(_create(), f"create_node({label})")
```

Similarly update `create_relationship()` method (line 266):

```python
async def create_relationship(
    self,
    source_id: str,
    target_id: str,
    rel_type: str,
    properties: dict[str, Any],
) -> None:
    # SECURITY: Validate relationship type
    rel_type = validate_rel_type(rel_type)

    async def _create_rel():
        # ... rest of implementation
```

Update all other methods that use `label` or `rel_type` parameters:
- `get_node()` (line 169)
- `update_node()` (line 204)
- `delete_node()` (line 236)
- `get_relationships()` (line 302)
- `update_relationship()` (line 356)
- `strengthen_relationship()` (line 392)
- `batch_create_nodes()` (line 542)
- `batch_create_with_relationships()` (line 566)

---

### Fix M-2: Session Authentication

**File**: `src/t4dm/mcp/session_manager.py` (new file)

Create session management system:

```python
"""
Session authentication and authorization for World Weaver.
"""

import secrets
from typing import Optional
from datetime import datetime, timedelta


class SessionManager:
    """Manages session authentication and access control."""

    def __init__(self):
        self._sessions: dict[str, dict] = {}
        self._session_timeout = timedelta(hours=24)

    def create_session(self, user_id: str) -> str:
        """
        Create new authenticated session.

        Args:
            user_id: Authenticated user identifier

        Returns:
            Cryptographically secure session token
        """
        session_token = secrets.token_urlsafe(32)  # 256-bit security

        self._sessions[session_token] = {
            "user_id": user_id,
            "created_at": datetime.now(),
            "last_accessed": datetime.now(),
        }

        return session_token

    def validate_session(self, session_token: str) -> bool:
        """
        Validate session token and check expiration.

        Args:
            session_token: Token to validate

        Returns:
            True if valid and not expired
        """
        session = self._sessions.get(session_token)
        if not session:
            return False

        # Check expiration
        if datetime.now() - session["created_at"] > self._session_timeout:
            del self._sessions[session_token]
            return False

        # Update last access
        session["last_accessed"] = datetime.now()
        return True

    def get_user_id(self, session_token: str) -> Optional[str]:
        """Get user ID for session token."""
        session = self._sessions.get(session_token)
        return session["user_id"] if session else None

    def verify_access(self, session_token: str, resource_session_id: str) -> bool:
        """
        Verify session has access to resource.

        Args:
            session_token: Authentication token
            resource_session_id: Session ID of resource being accessed

        Returns:
            True if access allowed
        """
        if not self.validate_session(session_token):
            return False

        user_id = self.get_user_id(session_token)
        # User can only access their own sessions
        return resource_session_id == f"session_{user_id}"

    def revoke_session(self, session_token: str) -> None:
        """Revoke session token."""
        self._sessions.pop(session_token, None)

    def cleanup_expired(self) -> int:
        """Remove expired sessions. Returns count removed."""
        now = datetime.now()
        expired = [
            token for token, session in self._sessions.items()
            if now - session["created_at"] > self._session_timeout
        ]

        for token in expired:
            del self._sessions[token]

        return len(expired)


# Global session manager instance
_session_manager: Optional[SessionManager] = None


def get_session_manager() -> SessionManager:
    """Get or create singleton session manager."""
    global _session_manager
    if _session_manager is None:
        _session_manager = SessionManager()
    return _session_manager
```

**File**: `src/t4dm/mcp/memory_gateway.py`

Add authentication to MCP tools:

```python
# Add import
from ww.mcp.session_manager import get_session_manager

# Add authentication decorator
def require_auth(func):
    """Decorator to require session authentication."""
    async def wrapper(*args, session_token: Optional[str] = None, **kwargs):
        session_manager = get_session_manager()

        # For backward compatibility, allow unauthenticated in dev mode
        if session_token is None:
            settings = get_settings()
            if settings.require_auth:  # New config option
                return _make_error_response(
                    "authentication_required",
                    "Session token required"
                )
            # Dev mode - use default session
            return await func(*args, **kwargs)

        # Validate token
        if not session_manager.validate_session(session_token):
            return _make_error_response(
                "invalid_session",
                "Session token invalid or expired"
            )

        # Execute with authenticated context
        return await func(*args, **kwargs)

    return wrapper

# Apply to tools
@mcp_app.tool()
@require_auth
async def create_episode(...):
    # Implementation unchanged
```

**File**: `src/t4dm/core/config.py`

Add authentication configuration:

```python
# Add to Settings class
require_auth: bool = Field(
    default=False,
    description="Require session authentication (set True for production)",
)
```

---

### Fix M-3: Remove Default Credentials

**File**: `src/t4dm/core/config.py`

Replace hardcoded defaults with required fields:

```python
# BEFORE:
neo4j_password: str = Field(
    default="password",
    description="Neo4j password",
)

# AFTER:
neo4j_password: str = Field(
    ...,  # Required, no default
    description="Neo4j password (set via WW_NEO4J_PASSWORD env var)",
)

# OR with runtime validation:
neo4j_password: str = Field(
    default="",
    description="Neo4j password",
)

def model_post_init(self, __context):
    """Validate configuration after initialization."""
    # Check for insecure defaults
    if self.neo4j_password in ("", "password", "neo4j"):
        raise ValueError(
            "Insecure Neo4j password detected. "
            "Set WW_NEO4J_PASSWORD environment variable to a strong password."
        )

    # Check for production deployment with defaults
    if not self.neo4j_uri.startswith("bolt://localhost"):
        if self.neo4j_password == "password":
            raise ValueError(
                "Cannot use default password with remote Neo4j instance"
            )
```

---

### Fix M-4: Rate Limiting

**File**: `src/t4dm/mcp/rate_limiter.py` (new file)

```python
"""
Rate limiting for World Weaver MCP tools.
"""

from collections import defaultdict
from datetime import datetime, timedelta
from typing import Optional


class RateLimiter:
    """Token bucket rate limiter."""

    def __init__(
        self,
        requests_per_minute: int = 60,
        burst_size: int = 10,
    ):
        """
        Initialize rate limiter.

        Args:
            requests_per_minute: Sustained rate limit
            burst_size: Additional requests allowed in burst
        """
        self.rate = requests_per_minute
        self.burst = burst_size
        self._buckets: dict[str, dict] = defaultdict(
            lambda: {
                "tokens": burst_size,
                "last_update": datetime.now()
            }
        )

    def check_limit(self, key: str) -> tuple[bool, Optional[int]]:
        """
        Check if request is allowed under rate limit.

        Args:
            key: Rate limit key (typically session_id)

        Returns:
            Tuple of (allowed: bool, retry_after_seconds: Optional[int])
        """
        now = datetime.now()
        bucket = self._buckets[key]

        # Refill tokens based on time elapsed
        elapsed = (now - bucket["last_update"]).total_seconds()
        tokens_to_add = elapsed * (self.rate / 60.0)
        bucket["tokens"] = min(self.burst, bucket["tokens"] + tokens_to_add)
        bucket["last_update"] = now

        # Check if request allowed
        if bucket["tokens"] >= 1.0:
            bucket["tokens"] -= 1.0
            return True, None
        else:
            # Calculate retry after
            tokens_needed = 1.0 - bucket["tokens"]
            retry_seconds = int((tokens_needed / (self.rate / 60.0)) + 1)
            return False, retry_seconds

    def reset(self, key: str) -> None:
        """Reset rate limit for key."""
        if key in self._buckets:
            del self._buckets[key]


# Global rate limiter instances
_rate_limiters: dict[str, RateLimiter] = {
    "create": RateLimiter(requests_per_minute=60, burst_size=10),
    "query": RateLimiter(requests_per_minute=120, burst_size=20),
    "expensive": RateLimiter(requests_per_minute=10, burst_size=2),
}


def get_rate_limiter(operation: str = "default") -> RateLimiter:
    """Get rate limiter for operation type."""
    return _rate_limiters.get(operation, _rate_limiters["create"])
```

**File**: `src/t4dm/mcp/memory_gateway.py`

Add rate limiting to tools:

```python
from ww.mcp.rate_limiter import get_rate_limiter

# Add decorator
def rate_limit(operation: str = "create"):
    """Decorator to enforce rate limiting."""
    def decorator(func):
        async def wrapper(*args, **kwargs):
            settings = get_settings()
            limiter = get_rate_limiter(operation)

            allowed, retry_after = limiter.check_limit(settings.session_id)

            if not allowed:
                return _make_error_response(
                    "rate_limit_exceeded",
                    f"Too many requests. Retry after {retry_after} seconds.",
                    field="rate_limit"
                )

            return await func(*args, **kwargs)
        return wrapper
    return decorator

# Apply to tools
@mcp_app.tool()
@rate_limit("create")
async def create_episode(...):
    # Implementation unchanged

@mcp_app.tool()
@rate_limit("query")
async def recall_episodes(...):
    # Implementation unchanged

@mcp_app.tool()
@rate_limit("expensive")
async def spread_activation(...):
    # Implementation unchanged
```

---

## Priority 2: Hardening Improvements

### Fix L-1: Error Message Sanitization

**File**: `src/t4dm/mcp/memory_gateway.py`

Update error handling:

```python
from uuid import uuid4

def _make_error_response(
    error_type: str,
    message: str,
    field: Optional[str] = None,
    safe_message: bool = False
) -> dict:
    """
    Create standardized error response.

    Args:
        error_type: Error category
        message: Error message (logged internally)
        field: Field that caused error
        safe_message: If False, replace with generic message
    """
    # Generate unique error ID for correlation
    error_id = str(uuid4())

    # Log full details internally
    logger.error(f"Error {error_id}: {message}")

    # Return safe message to client
    if not safe_message:
        message = f"An error occurred. Reference: {error_id}"

    response = {
        "error": error_type,
        "message": message,
        "error_id": error_id,
    }
    if field:
        response["field"] = field

    return response

# Update exception handlers
except ValidationError as e:
    # Validation errors are safe to expose
    return e.to_dict()
except DatabaseTimeoutError as e:
    # Generic timeout message
    return _make_error_response(
        "timeout",
        f"Database operation timed out: {e}",  # Logged
        safe_message=False  # Client gets generic message
    )
except Exception as e:
    # All other errors - sanitize
    return _make_error_response(
        "internal_error",
        str(e),  # Logged
        safe_message=False  # Client gets generic message
    )
```

---

### Fix L-2: Content Sanitization

**File**: `src/t4dm/mcp/validation.py`

Add sanitization function:

```python
import html
import unicodedata
import re

def sanitize_text(value: str, allow_newlines: bool = True) -> str:
    """
    Sanitize user input for safe storage and display.

    Args:
        value: Text to sanitize
        allow_newlines: Whether to preserve newline characters

    Returns:
        Sanitized text
    """
    # Normalize unicode to prevent homograph attacks
    value = unicodedata.normalize('NFKC', value)

    # Remove null bytes
    value = value.replace('\x00', '')

    # Escape HTML entities
    value = html.escape(value)

    # Remove or escape control characters (except allowed)
    if allow_newlines:
        allowed_chars = '\n\t'
    else:
        allowed_chars = '\t'

    value = ''.join(
        c for c in value
        if c >= ' ' or c in allowed_chars
    )

    # Remove CRLF injection attempts
    value = value.replace('\r\n', '\n')
    value = re.sub(r'\n{3,}', '\n\n', value)  # Limit consecutive newlines

    return value

# Update validate_non_empty_string
def validate_non_empty_string(
    value: str,
    field: str,
    max_length: Optional[int] = None,
    sanitize: bool = True
) -> str:
    """Validate and optionally sanitize string."""
    if value is None:
        raise ValidationError(field, "Value cannot be None")

    if not isinstance(value, str):
        raise ValidationError(field, f"Expected string, got {type(value).__name__}")

    if not value.strip():
        raise ValidationError(field, "Cannot be empty")

    # Sanitize if enabled
    if sanitize:
        value = sanitize_text(value)

    if max_length is not None and len(value) > max_length:
        raise ValidationError(field, f"Exceeds maximum length of {max_length}")

    return value
```

---

### Fix L-3: Secure Cache Directory

**File**: `src/t4dm/core/config.py`

```python
from pathlib import Path
import os

class Settings(BaseSettings):
    # ... other fields ...

    embedding_cache_dir: str = Field(
        default_factory=lambda: str(
            Path.home() / ".cache" / "world_weaver" / "models"
        ),
        description="Directory for model caching",
    )

    def model_post_init(self, __context):
        """Validate and secure configuration."""
        # Ensure cache directory is absolute path
        cache_path = Path(self.embedding_cache_dir)

        if not cache_path.is_absolute():
            raise ValueError(
                "embedding_cache_dir must be an absolute path. "
                f"Got: {self.embedding_cache_dir}"
            )

        # Create directory with restricted permissions
        try:
            cache_path.mkdir(parents=True, exist_ok=True, mode=0o700)
        except Exception as e:
            logger.warning(f"Could not create cache directory: {e}")

        # Verify permissions (Unix only)
        if os.name != 'nt':  # Not Windows
            stat_info = cache_path.stat()
            if stat_info.st_mode & 0o077:  # Others have permissions
                logger.warning(
                    f"Cache directory {cache_path} has insecure permissions. "
                    "Should be 0700 (owner-only access)."
                )
```

---

## Testing Security Fixes

After implementing fixes, run security test suite:

```bash
# Run security tests
pytest tests/security/test_injection.py -v

# Run full test suite
pytest tests/ -v --cov=src/ww

# Security scan
bandit -r src/t4dm/ -ll

# Dependency check
safety check
```

## Deployment Checklist

Before deploying with security fixes:

- [ ] All Priority 1 fixes implemented
- [ ] Security tests passing
- [ ] Configuration validated (no default passwords)
- [ ] Rate limiting tested with load testing
- [ ] Error messages sanitized and tested
- [ ] Session authentication enabled in production
- [ ] Audit logging configured
- [ ] Documentation updated

---

**Implementation Priority**: Address Priority 1 fixes before production deployment.
**Estimated Effort**: 2-3 days for experienced developer
**Testing Required**: Full regression + security test suite
