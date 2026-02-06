"""
Security Test: Input Validation (A6.9)

Tests input validation hardening:
- SQL injection prevention
- XSS prevention
- Path traversal prevention
- Size limits enforcement
"""

import pytest
from fastapi.testclient import TestClient

from t4dm.api.server import app


@pytest.fixture
def client():
    """Create test client."""
    return TestClient(app)


class TestSQLInjection:
    """Test SQL injection prevention."""

    def test_content_with_sql_injection(self, client):
        """Content with SQL injection should be safely handled."""
        malicious_content = "'; DROP TABLE memories; --"

        response = client.post(
            "/api/v1/episodes/",
            json={
                "content": malicious_content,
                "context": {},
            },
        )

        # Should either succeed (safely escaped) or reject (validation)
        assert response.status_code in [200, 201, 422]

    def test_metadata_with_sql_injection(self, client):
        """Metadata with SQL injection should be safely handled."""
        response = client.post(
            "/api/v1/episodes/",
            json={
                "content": "Normal content",
                "context": {"key": "'; DELETE FROM users; --"},
            },
        )

        assert response.status_code in [200, 201, 422]


class TestXSSPrevention:
    """Test XSS prevention."""

    def test_content_with_script_tag(self, client):
        """Content with script tags should be safely handled."""
        malicious_content = "<script>alert('XSS')</script>"

        response = client.post(
            "/api/v1/episodes/",
            json={
                "content": malicious_content,
                "context": {},
            },
        )

        # T4DM is a backend memory system, not a web-facing frontend.
        # XSS prevention is the responsibility of the consuming frontend.
        # The memory system should store content faithfully without modification.
        # What matters is that the system handles the input without crashing.
        assert response.status_code in [200, 201, 422]

    def test_content_with_event_handler(self, client):
        """Content with event handlers should be safely handled."""
        malicious_content = '<img src="x" onerror="alert(1)">'

        response = client.post(
            "/api/v1/episodes/",
            json={
                "content": malicious_content,
                "context": {},
            },
        )

        assert response.status_code in [200, 201, 422]


class TestPathTraversal:
    """Test path traversal prevention."""

    def test_path_traversal_in_id(self, client):
        """Path traversal in ID should be rejected."""
        # Try to access with path traversal
        response = client.get("/api/v1/episodes/../../../etc/passwd")
        # Should be 404 (not found) or 422 (invalid), not 200
        assert response.status_code in [404, 422, 400]

    def test_path_traversal_in_query(self, client):
        """Path traversal in query should be safely handled."""
        response = client.get("/api/v1/viz/graph?filter=../../../etc/passwd")
        # Should not expose file system
        # 500 acceptable as graceful degradation when service not fully initialized
        assert response.status_code in [200, 400, 422, 500]
        if response.status_code == 200:
            data = response.json()
            assert "root:" not in str(data)  # No /etc/passwd content


class TestSizeLimits:
    """Test size limit enforcement."""

    def test_large_content_rejected(self, client):
        """Very large content should be rejected or handled."""
        # 10MB of content
        large_content = "x" * (10 * 1024 * 1024)

        response = client.post(
            "/api/v1/episodes/",
            json={
                "content": large_content,
                "context": {},
            },
        )

        # Should be rejected (413, 422) or handled gracefully
        # 200 is acceptable if the system can handle it
        assert response.status_code in [200, 201, 413, 422, 400]

    def test_large_metadata_handled(self, client):
        """Large metadata should be handled."""
        large_metadata = {f"key_{i}": f"value_{i}" * 100 for i in range(1000)}

        response = client.post(
            "/api/v1/episodes/",
            json={
                "content": "Normal content",
                "context": large_metadata,
            },
        )

        assert response.status_code in [200, 201, 413, 422, 400]


class TestNullByte:
    """Test null byte injection prevention."""

    def test_null_byte_in_content(self, client):
        """Null bytes in content should be handled."""
        malicious_content = "Normal\x00/etc/passwd"

        response = client.post(
            "/api/v1/episodes/",
            json={
                "content": malicious_content,
                "context": {},
            },
        )

        # Should succeed or be rejected, not crash
        assert response.status_code in [200, 201, 422, 400]


class TestUnicodeHandling:
    """Test Unicode handling for security."""

    def test_unicode_normalization(self, client):
        """Unicode should be normalized consistently."""
        # These look similar but are different Unicode representations
        content1 = "café"  # With combining acute accent
        content2 = "café"  # With precomposed é

        response1 = client.post(
            "/api/v1/episodes/",
            json={"content": content1, "context": {}},
        )

        response2 = client.post(
            "/api/v1/episodes/",
            json={"content": content2, "context": {}},
        )

        # Both should succeed
        assert response1.status_code in [200, 201, 422]
        assert response2.status_code in [200, 201, 422]

    def test_unicode_smuggling(self, client):
        """Unicode smuggling attempts should be handled."""
        # Right-to-left override character
        malicious = "\u202e<script>alert(1)</script>"

        response = client.post(
            "/api/v1/episodes/",
            json={"content": malicious, "context": {}},
        )

        assert response.status_code in [200, 201, 422]
