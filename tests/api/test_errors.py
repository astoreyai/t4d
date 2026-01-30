"""Tests for API error handling utilities."""

import pytest
import re

from ww.api.errors import (
    sanitize_error,
    create_error_response,
    SENSITIVE_PATTERNS,
    GENERIC_MESSAGES,
)


class TestSanitizeError:
    """Tests for sanitize_error function."""

    def test_sanitize_connection_string(self):
        """Connection strings are redacted."""
        error = Exception("bolt://user:password@localhost:7687 connection failed")
        result = sanitize_error(error, "connect")
        # Password should be redacted or removed in the sanitized output
        assert "password" not in result.lower() or "[REDACTED]" in result

    def test_sanitize_api_key(self):
        """API keys are redacted."""
        error = Exception("api_key: sk-secret123abc failed")
        result = sanitize_error(error, "auth")
        assert "secret123" not in result

    def test_sanitize_file_paths(self):
        """File paths with usernames are redacted."""
        error = Exception("File not found: /home/username/secrets/key.pem")
        result = sanitize_error(error, "read file")
        assert "username" not in result

    def test_sanitize_ip_addresses(self):
        """IP addresses with ports are redacted."""
        error = Exception("Cannot connect to 192.168.1.100:5432")
        result = sanitize_error(error, "connect")
        assert "192.168.1.100" not in result

    def test_generic_connection_error(self):
        """ConnectionError maps to generic message."""
        # Use base ConnectionError class
        error = ConnectionError("Secret details about connection")
        result = sanitize_error(error, "connect")
        # ConnectionError should map to generic message
        assert "Service connection failed" in result or "Secret" not in result

    def test_generic_timeout_error(self):
        """TimeoutError maps to generic message."""
        error = TimeoutError("Timeout after 30s connecting to internal-server.local")
        result = sanitize_error(error, "fetch")
        assert "internal-server" not in result
        assert "timed out" in result.lower() or "timeout" in result.lower()

    def test_generic_value_error(self):
        """ValueError maps to generic message."""
        error = ValueError("Invalid value for internal field xyz_secret")
        result = sanitize_error(error, "validate")
        assert "xyz_secret" not in result

    def test_truncate_long_messages(self):
        """Long messages are truncated."""
        long_msg = "x" * 500
        error = Exception(long_msg)
        result = sanitize_error(error, "process")
        assert len(result) < 300  # Should be truncated

    def test_include_type_false(self):
        """Can exclude error type from message."""
        error = ValueError("Bad input")
        result_with = sanitize_error(error, "process", include_type=True)
        result_without = sanitize_error(error, "process", include_type=False)
        assert "process" in result_with
        assert "Invalid input" in result_without

    def test_safe_characters_only(self):
        """Only safe characters remain."""
        error = Exception("Error with <script>alert('xss')</script>")
        result = sanitize_error(error, "render")
        assert "<script>" not in result
        assert "'" not in result or result.count("'") == 0


class TestCreateErrorResponse:
    """Tests for create_error_response function."""

    def test_basic_response_structure(self):
        """Response has correct structure."""
        error = Exception("Test error")
        response = create_error_response(500, error, "test")

        assert response["status"] == "error"
        assert response["code"] == 500
        assert "detail" in response

    def test_status_codes(self):
        """Various status codes work correctly."""
        error = Exception("Error")

        for code in [400, 401, 403, 404, 500, 503]:
            response = create_error_response(code, error, "test")
            assert response["code"] == code

    def test_sanitized_detail(self):
        """Detail is sanitized."""
        error = Exception("Error with api_key: secret123")
        response = create_error_response(500, error, "process")
        assert "secret123" not in response["detail"]


class TestSensitivePatterns:
    """Tests for sensitive pattern definitions."""

    def test_bolt_uri_pattern(self):
        """Bolt URIs are matched."""
        text = "bolt://user:pass@host:7687"
        for pattern in SENSITIVE_PATTERNS:
            if re.search(pattern, text, re.IGNORECASE):
                return
        pytest.fail("Bolt URI pattern not matched")

    def test_neo4j_uri_pattern(self):
        """Neo4j URIs are matched."""
        text = "neo4j://admin:password@localhost"
        for pattern in SENSITIVE_PATTERNS:
            if re.search(pattern, text, re.IGNORECASE):
                return
        pytest.fail("Neo4j URI pattern not matched")

    def test_api_key_patterns(self):
        """API key patterns are matched."""
        texts = [
            "api_key: abc123",
            "token=secret",
            "secret: mysecret",
        ]
        for text in texts:
            matched = False
            for pattern in SENSITIVE_PATTERNS:
                if re.search(pattern, text, re.IGNORECASE):
                    matched = True
                    break
            assert matched, f"Pattern not matched: {text}"


class TestGenericMessages:
    """Tests for generic error messages."""

    def test_all_expected_types_covered(self):
        """Expected error types have generic messages."""
        expected_types = [
            "ConnectionError",
            "TimeoutError",
            "ValueError",
            "TypeError",
            "KeyError",
        ]
        for etype in expected_types:
            assert etype in GENERIC_MESSAGES

    def test_messages_are_safe(self):
        """Generic messages don't contain sensitive info."""
        for msg in GENERIC_MESSAGES.values():
            assert len(msg) < 50
            assert "password" not in msg.lower()
            assert "secret" not in msg.lower()
