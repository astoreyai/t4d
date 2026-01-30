"""
Structured Logging for World Weaver.

Provides JSON-structured logging with contextual information
for debugging and monitoring memory operations.
"""

import json
import logging
import re
import sys
import time
from collections.abc import Callable
from contextvars import ContextVar
from dataclasses import asdict, dataclass, field
from datetime import datetime
from functools import wraps

# Context variables for request tracking
_session_id: ContextVar[str] = ContextVar("session_id", default="unknown")
_operation_id: ContextVar[str] = ContextVar("operation_id", default="unknown")
_request_start: ContextVar[float] = ContextVar("request_start", default=0.0)

# LOG-001 FIX: Control character pattern for log sanitization
# Matches CR, LF, null bytes, and other control chars except tab
_CONTROL_CHAR_PATTERN = re.compile(r"[\x00-\x08\x0b\x0c\x0e-\x1f\x7f]")


def _sanitize_log_message(message: str) -> str:
    """
    LOG-001 FIX: Sanitize log message to prevent log injection attacks.

    Escapes newlines, carriage returns, and other control characters that
    could be used to forge log entries or corrupt log parsers.

    Args:
        message: Raw log message

    Returns:
        Sanitized message safe for logging
    """
    if not isinstance(message, str):
        message = str(message)

    # Replace newlines and carriage returns with visible escape sequences
    message = message.replace("\r\n", "\\r\\n")
    message = message.replace("\n", "\\n")
    message = message.replace("\r", "\\r")

    # Remove null bytes and other dangerous control characters
    message = _CONTROL_CHAR_PATTERN.sub("", message)

    return message


@dataclass
class LogContext:
    """Structured log context."""
    timestamp: str = field(default_factory=lambda: datetime.utcnow().isoformat())
    level: str = "INFO"
    logger: str = "ww"
    message: str = ""
    session_id: str = ""
    operation_id: str = ""
    duration_ms: float | None = None
    extra: dict = field(default_factory=dict)

    def to_json(self) -> str:
        """Convert to JSON string."""
        data = asdict(self)
        # Remove None values
        data = {k: v for k, v in data.items() if v is not None}
        # Flatten extra into main dict
        extra = data.pop("extra", {})
        data.update(extra)
        return json.dumps(data)


class StructuredFormatter(logging.Formatter):
    """JSON formatter for structured logging."""

    def format(self, record: logging.LogRecord) -> str:
        """Format log record as JSON."""
        # LOG-001 FIX: Sanitize message to prevent log injection
        sanitized_message = _sanitize_log_message(record.getMessage())

        ctx = LogContext(
            timestamp=datetime.utcnow().isoformat(),
            level=record.levelname,
            logger=record.name,
            message=sanitized_message,
            session_id=_sanitize_log_message(_session_id.get()),
            operation_id=_sanitize_log_message(_operation_id.get()),
        )

        # Add duration if available
        start = _request_start.get()
        if start > 0:
            ctx.duration_ms = (time.time() - start) * 1000

        # Add extra fields from record
        if hasattr(record, "extra_fields"):
            ctx.extra = record.extra_fields

        # Add exception info if present
        if record.exc_info:
            ctx.extra["exception"] = self.formatException(record.exc_info)

        return ctx.to_json()


class ContextAdapter(logging.LoggerAdapter):
    """Logger adapter that adds context to log messages."""

    def process(self, msg: str, kwargs: dict) -> tuple[str, dict]:
        """Add context to log record."""
        extra = kwargs.get("extra", {})
        extra["session_id"] = _session_id.get()
        extra["operation_id"] = _operation_id.get()
        kwargs["extra"] = extra
        return msg, kwargs


def configure_logging(
    level: str = "INFO",
    json_output: bool = True,
    log_file: str | None = None,
) -> None:
    """
    Configure structured logging for World Weaver.

    Args:
        level: Log level (DEBUG, INFO, WARNING, ERROR)
        json_output: Use JSON format (default: True)
        log_file: Optional file path for logs
    """
    root_logger = logging.getLogger()
    root_logger.setLevel(getattr(logging, level.upper()))

    # Remove existing handlers
    for handler in root_logger.handlers[:]:
        root_logger.removeHandler(handler)

    # Create console handler
    console_handler = logging.StreamHandler(sys.stderr)
    console_handler.setLevel(getattr(logging, level.upper()))

    if json_output:
        console_handler.setFormatter(StructuredFormatter())
    else:
        console_handler.setFormatter(
            logging.Formatter(
                "%(asctime)s [%(levelname)s] %(name)s: %(message)s"
            )
        )

    root_logger.addHandler(console_handler)

    # Add file handler if specified
    if log_file:
        file_handler = logging.FileHandler(log_file)
        file_handler.setLevel(getattr(logging, level.upper()))
        file_handler.setFormatter(StructuredFormatter())
        root_logger.addHandler(file_handler)

    # Configure ww loggers
    for name in ["ww", "ww.memory", "ww.storage", "ww.mcp"]:
        logger = logging.getLogger(name)
        logger.setLevel(getattr(logging, level.upper()))


def get_logger(name: str) -> logging.Logger:
    """
    Get a logger with structured output.

    Args:
        name: Logger name (usually __name__)

    Returns:
        Configured logger
    """
    return logging.getLogger(name)


def set_context(session_id: str, operation_id: str) -> None:
    """
    Set logging context for current async task.

    Args:
        session_id: Current session ID
        operation_id: Current operation ID
    """
    _session_id.set(session_id)
    _operation_id.set(operation_id)
    _request_start.set(time.time())


def clear_context() -> None:
    """Clear logging context."""
    _session_id.set("unknown")
    _operation_id.set("unknown")
    _request_start.set(0.0)


class OperationLogger:
    """
    Context manager for logging operations with timing.

    Example:
        async with OperationLogger("create_episode", session_id="abc") as log:
            episode = await memory.create(content)
            log.set_result(episode_id=str(episode.id))
    """

    def __init__(
        self,
        operation: str,
        session_id: str | None = None,
        **extra,
    ):
        """
        Initialize operation logger.

        Args:
            operation: Operation name
            session_id: Session ID (uses context if not provided)
            **extra: Additional fields to log
        """
        self.operation = operation
        self.session_id = session_id or _session_id.get()
        self.extra = extra
        self.result = {}
        self.start_time = 0.0
        self.logger = get_logger("ww.operations")

    async def __aenter__(self) -> "OperationLogger":
        """Start operation logging."""
        self.start_time = time.time()
        set_context(self.session_id, self.operation)

        self.logger.info(
            f"Starting {self.operation}",
            extra={"extra_fields": {"operation": self.operation, **self.extra}},
        )
        return self

    async def __aexit__(self, exc_type, exc_val, exc_tb) -> None:
        """End operation logging."""
        duration_ms = (time.time() - self.start_time) * 1000

        fields = {
            "operation": self.operation,
            "duration_ms": round(duration_ms, 2),
            **self.extra,
            **self.result,
        }

        if exc_type:
            fields["error"] = str(exc_val)
            fields["error_type"] = exc_type.__name__
            self.logger.error(
                f"Failed {self.operation}: {exc_val}",
                extra={"extra_fields": fields},
            )
        else:
            self.logger.info(
                f"Completed {self.operation} in {duration_ms:.2f}ms",
                extra={"extra_fields": fields},
            )

        clear_context()

    def set_result(self, **kwargs) -> None:
        """Set result fields to include in log."""
        self.result.update(kwargs)


def log_operation(
    operation: str,
    session_id: str | None = None,
    **extra,
) -> Callable:
    """
    Decorator for logging async operations with timing.

    Args:
        operation: Operation name
        session_id: Session ID (uses context if not provided)
        **extra: Additional fields to log

    Example:
        @log_operation("recall_episodes")
        async def recall(self, query: str) -> list:
            ...
    """
    def decorator(func: Callable) -> Callable:
        @wraps(func)
        async def wrapper(*args, **kwargs):
            async with OperationLogger(operation, session_id, **extra):
                return await func(*args, **kwargs)
        return wrapper
    return decorator


# =============================================================================
# P3-SEC-L5: Audit Logger for Security-Sensitive Operations
# =============================================================================


class AuditEventType:
    """Audit event types for security monitoring."""
    SESSION_CREATED = "session.created"
    SESSION_DELETED = "session.deleted"
    BULK_DELETE = "bulk.delete"
    AUTH_FAILURE = "auth.failure"
    AUTH_SUCCESS = "auth.success"
    RATE_LIMIT_EXCEEDED = "rate_limit.exceeded"
    PERMISSION_DENIED = "permission.denied"
    CONFIG_CHANGE = "config.change"
    ADMIN_ACTION = "admin.action"


@dataclass
class AuditEvent:
    """
    P3-SEC-L5: Structured audit event for security forensics.

    All audit events include:
    - timestamp (ISO 8601)
    - event_type (from AuditEventType)
    - session_id (if available)
    - ip_address (if available)
    - user_agent (if available)
    - details (event-specific data)
    """
    event_type: str
    timestamp: str = field(default_factory=lambda: datetime.utcnow().isoformat())
    session_id: str | None = None
    ip_address: str | None = None
    user_agent: str | None = None
    details: dict = field(default_factory=dict)
    severity: str = "INFO"  # INFO, WARNING, ERROR, CRITICAL

    def to_dict(self) -> dict:
        """Convert to dictionary for logging."""
        return {
            "audit": True,  # Mark as audit log for filtering
            "event_type": self.event_type,
            "timestamp": self.timestamp,
            "session_id": self.session_id,
            "ip_address": self.ip_address,
            "user_agent": self.user_agent,
            "severity": self.severity,
            **self.details,
        }


class AuditLogger:
    """
    P3-SEC-L5: Security audit logger for sensitive operations.

    Provides structured audit logs for:
    - Session lifecycle (creation, deletion)
    - Bulk operations (deletions)
    - Authentication events (failures, rate limits)
    - Administrative actions

    Usage:
        audit = AuditLogger()
        audit.log_session_created("session-123", ip="192.168.1.1")
        audit.log_auth_failure(ip="192.168.1.1", reason="invalid_api_key")
    """

    def __init__(self, logger_name: str = "ww.audit"):
        self._logger = logging.getLogger(logger_name)

    def _log(self, event: AuditEvent) -> None:
        """Log an audit event."""
        level = getattr(logging, event.severity.upper(), logging.INFO)
        self._logger.log(
            level,
            f"AUDIT: {event.event_type}",
            extra={"extra_fields": event.to_dict()},
        )

    def log_session_created(
        self,
        session_id: str,
        ip: str | None = None,
        user_agent: str | None = None,
        **details,
    ) -> None:
        """Log session creation."""
        self._log(AuditEvent(
            event_type=AuditEventType.SESSION_CREATED,
            session_id=session_id,
            ip_address=ip,
            user_agent=user_agent,
            details=details,
        ))

    def log_session_deleted(
        self,
        session_id: str,
        ip: str | None = None,
        reason: str | None = None,
        **details,
    ) -> None:
        """Log session deletion."""
        self._log(AuditEvent(
            event_type=AuditEventType.SESSION_DELETED,
            session_id=session_id,
            ip_address=ip,
            details={"reason": reason, **details},
        ))

    def log_bulk_delete(
        self,
        session_id: str | None,
        collection: str,
        count: int,
        ip: str | None = None,
        **details,
    ) -> None:
        """Log bulk deletion operation."""
        self._log(AuditEvent(
            event_type=AuditEventType.BULK_DELETE,
            session_id=session_id,
            ip_address=ip,
            severity="WARNING",
            details={"collection": collection, "count": count, **details},
        ))

    def log_auth_failure(
        self,
        ip: str | None = None,
        reason: str = "unknown",
        session_id: str | None = None,
        user_agent: str | None = None,
        **details,
    ) -> None:
        """Log authentication failure."""
        self._log(AuditEvent(
            event_type=AuditEventType.AUTH_FAILURE,
            session_id=session_id,
            ip_address=ip,
            user_agent=user_agent,
            severity="WARNING",
            details={"reason": reason, **details},
        ))

    def log_auth_success(
        self,
        session_id: str,
        ip: str | None = None,
        method: str = "api_key",
        **details,
    ) -> None:
        """Log successful authentication."""
        self._log(AuditEvent(
            event_type=AuditEventType.AUTH_SUCCESS,
            session_id=session_id,
            ip_address=ip,
            details={"method": method, **details},
        ))

    def log_rate_limit_exceeded(
        self,
        ip: str | None = None,
        session_id: str | None = None,
        limit: int = 0,
        window_seconds: int = 0,
        **details,
    ) -> None:
        """Log rate limit violation."""
        self._log(AuditEvent(
            event_type=AuditEventType.RATE_LIMIT_EXCEEDED,
            session_id=session_id,
            ip_address=ip,
            severity="WARNING",
            details={"limit": limit, "window_seconds": window_seconds, **details},
        ))

    def log_permission_denied(
        self,
        session_id: str | None,
        resource: str,
        action: str,
        ip: str | None = None,
        **details,
    ) -> None:
        """Log permission denied event."""
        self._log(AuditEvent(
            event_type=AuditEventType.PERMISSION_DENIED,
            session_id=session_id,
            ip_address=ip,
            severity="WARNING",
            details={"resource": resource, "action": action, **details},
        ))

    def log_admin_action(
        self,
        action: str,
        session_id: str | None = None,
        ip: str | None = None,
        **details,
    ) -> None:
        """Log administrative action."""
        self._log(AuditEvent(
            event_type=AuditEventType.ADMIN_ACTION,
            session_id=session_id,
            ip_address=ip,
            severity="WARNING",
            details={"action": action, **details},
        ))


# Global audit logger instance
_audit_logger: AuditLogger | None = None


def get_audit_logger() -> AuditLogger:
    """Get the global audit logger instance."""
    global _audit_logger
    if _audit_logger is None:
        _audit_logger = AuditLogger()
    return _audit_logger
