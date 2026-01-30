"""Tests for structured logging module."""

import json
import logging
import pytest
from io import StringIO
from unittest.mock import patch

from ww.observability.logging import (
    StructuredFormatter,
    LogContext,
    configure_logging,
    get_logger,
    set_context,
    clear_context,
    OperationLogger,
    _sanitize_log_message,
)


class TestLogInjection:
    """LOG-001: Tests for log injection prevention."""

    def test_sanitize_log_message_newlines(self):
        """Test that newlines are escaped in log messages."""
        # Newlines could be used to forge log entries
        malicious = "Normal message\nFake log entry: [ERROR] Something bad"
        sanitized = _sanitize_log_message(malicious)

        assert "\n" not in sanitized
        assert "\\n" in sanitized or "[LF]" in sanitized

    def test_sanitize_log_message_carriage_return(self):
        """Test that carriage returns are escaped."""
        malicious = "Normal message\r[CRITICAL] Forged entry"
        sanitized = _sanitize_log_message(malicious)

        assert "\r" not in sanitized

    def test_sanitize_log_message_tabs(self):
        """Test that tabs are preserved (they're generally safe)."""
        msg = "Message with\ttab"
        sanitized = _sanitize_log_message(msg)
        # Tabs can be kept or escaped - both are acceptable
        assert "tab" in sanitized

    def test_sanitize_log_message_null_bytes(self):
        """Test that null bytes are removed."""
        malicious = "Message with \x00 null byte"
        sanitized = _sanitize_log_message(malicious)

        assert "\x00" not in sanitized

    def test_sanitize_log_message_control_chars(self):
        """Test that control characters are escaped."""
        # Bell, backspace, form feed, etc.
        malicious = "Message with \x07 bell and \x08 backspace"
        sanitized = _sanitize_log_message(malicious)

        assert "\x07" not in sanitized
        assert "\x08" not in sanitized

    def test_sanitize_log_message_normal_text(self):
        """Test that normal text is unchanged."""
        normal = "This is a normal log message with special chars: @#$%^&*()[]{}|;':\",./<>?"
        sanitized = _sanitize_log_message(normal)

        assert sanitized == normal

    def test_structured_formatter_escapes_newlines(self):
        """Test that StructuredFormatter prevents log injection."""
        formatter = StructuredFormatter()

        # Create a log record with injection attempt
        record = logging.LogRecord(
            name="test",
            level=logging.INFO,
            pathname="",
            lineno=0,
            msg="User input: %s",
            args=("malicious\n[ERROR] Forged entry",),
            exc_info=None,
        )

        output = formatter.format(record)
        parsed = json.loads(output)

        # The message should not contain raw newlines
        assert "\n" not in parsed["message"]

    def test_log_context_json_escapes_injection(self):
        """Test that LogContext JSON output is safe."""
        ctx = LogContext(
            message="User said: hello\nFake: [ERROR] hacked",
            session_id="test\n[ADMIN] escalation",
        )

        json_output = ctx.to_json()
        parsed = json.loads(json_output)

        # JSON should be parseable (no injection broke the format)
        assert "message" in parsed
        # Newlines should be escaped by json.dumps
        assert "\n" not in json_output or "\\n" in json_output


class TestStructuredLogging:
    """Tests for structured logging functionality."""

    def test_log_context_to_json(self):
        """Test LogContext serialization."""
        ctx = LogContext(
            message="Test message",
            session_id="sess-123",
            operation_id="op-456",
        )

        json_str = ctx.to_json()
        parsed = json.loads(json_str)

        assert parsed["message"] == "Test message"
        assert parsed["session_id"] == "sess-123"
        assert parsed["operation_id"] == "op-456"

    def test_log_context_removes_none(self):
        """Test that None values are removed from JSON."""
        ctx = LogContext(
            message="Test",
            duration_ms=None,
        )

        json_str = ctx.to_json()
        parsed = json.loads(json_str)

        assert "duration_ms" not in parsed

    def test_log_context_flattens_extra(self):
        """Test that extra dict is flattened."""
        ctx = LogContext(
            message="Test",
            extra={"custom_field": "value"},
        )

        json_str = ctx.to_json()
        parsed = json.loads(json_str)

        assert parsed["custom_field"] == "value"
        assert "extra" not in parsed

    def test_set_and_clear_context(self):
        """Test context variable management."""
        set_context("session-abc", "operation-xyz")

        # Context should be set
        from ww.observability.logging import _session_id, _operation_id
        assert _session_id.get() == "session-abc"
        assert _operation_id.get() == "operation-xyz"

        clear_context()

        assert _session_id.get() == "unknown"
        assert _operation_id.get() == "unknown"


class TestOperationLogger:
    """Tests for OperationLogger context manager."""

    @pytest.mark.asyncio
    async def test_operation_logger_timing(self):
        """Test that operation logger tracks timing."""
        import time

        async with OperationLogger("test_op", session_id="test") as log:
            time.sleep(0.01)  # 10ms

        # Should complete without error

    @pytest.mark.asyncio
    async def test_operation_logger_set_result(self):
        """Test setting result fields."""
        async with OperationLogger("test_op", session_id="test") as log:
            log.set_result(items_processed=42)

        assert log.result["items_processed"] == 42

    @pytest.mark.asyncio
    async def test_operation_logger_handles_exception(self):
        """Test exception handling in operation logger."""
        with pytest.raises(ValueError):
            async with OperationLogger("test_op", session_id="test"):
                raise ValueError("Test error")
