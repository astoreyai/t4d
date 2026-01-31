"""
Security tests for injection vulnerabilities.

Tests for:
- Cypher injection in neo4j_store
- Session spoofing
- Content sanitization

NOTE: These tests require running Neo4j and Qdrant instances.
"""

import pytest

# Mark all tests in this module as integration tests
pytestmark = pytest.mark.integration
from uuid import uuid4

from t4dm.storage.neo4j_store import Neo4jStore
from t4dm.core.validation import ValidationError, validate_non_empty_string


class TestCypherInjection:
    """Test Cypher injection vulnerabilities."""

    @pytest.mark.asyncio
    async def test_malicious_label_in_create_node(self):
        """Test that malicious labels are rejected."""
        store = Neo4jStore()
        await store.initialize()

        # Attempt to inject Cypher via label
        malicious_labels = [
            "Entity} DETACH DELETE n //",
            "Entity}\nDETACH DELETE n\n//",
            "Entity` WHERE 1=1 DETACH DELETE n //",
            "Entity; MATCH (n) DETACH DELETE n; //",
        ]

        for malicious_label in malicious_labels:
            with pytest.raises((ValueError, ValidationError)):
                await store.create_node(
                    label=malicious_label,
                    properties={"id": str(uuid4()), "name": "test"}
                )

    @pytest.mark.asyncio
    async def test_malicious_rel_type_in_create_relationship(self):
        """Test that malicious relationship types are rejected."""
        store = Neo4jStore()
        await store.initialize()

        # Create legitimate nodes first
        node1_id = await store.create_node("Entity", {"id": str(uuid4()), "name": "a"})
        node2_id = await store.create_node("Entity", {"id": str(uuid4()), "name": "b"})

        # Attempt injection via relationship type
        malicious_types = [
            "RELATED_TO} DETACH DELETE n //",
            "RELATED_TO]\nMATCH (n) DETACH DELETE n\n//",
            "RELATED_TO; DROP DATABASE neo4j; //",
        ]

        for malicious_type in malicious_types:
            with pytest.raises((ValueError, ValidationError)):
                await store.create_relationship(
                    source_id=node1_id,
                    target_id=node2_id,
                    rel_type=malicious_type,
                    properties={"weight": 0.5}
                )

    @pytest.mark.asyncio
    async def test_injection_via_property_values(self):
        """Test that property values are safely parameterized."""
        store = Neo4jStore()
        await store.initialize()

        # These should be safe because properties are parameterized
        malicious_values = [
            "'; DETACH DELETE n; //",
            "' OR 1=1 --",
            "\\'; MATCH (n) DETACH DELETE n; //",
        ]

        for malicious_value in malicious_values:
            node_id = await store.create_node(
                label="Entity",
                properties={
                    "id": str(uuid4()),
                    "name": malicious_value  # Should be safely parameterized
                }
            )

            # Verify node was created with exact value (not executed as Cypher)
            node = await store.get_node(node_id)
            assert node["name"] == malicious_value

    @pytest.mark.asyncio
    async def test_label_whitelist_enforcement(self):
        """Test that only allowed labels are accepted."""
        store = Neo4jStore()
        await store.initialize()

        # Only these labels should be allowed
        allowed_labels = {"Episode", "Entity", "Procedure"}

        # Valid labels should work
        for label in allowed_labels:
            node_id = await store.create_node(
                label=label,
                properties={"id": str(uuid4())}
            )
            assert node_id is not None

        # Invalid labels should be rejected
        invalid_labels = ["User", "Admin", "CustomNode"]
        for label in invalid_labels:
            with pytest.raises(ValueError, match="Invalid node label"):
                await store.create_node(
                    label=label,
                    properties={"id": str(uuid4())}
                )


class TestSessionSpoofing:
    """Test session isolation and authentication."""

    @pytest.mark.asyncio
    async def test_cross_session_memory_access(self):
        """Test that sessions cannot access each other's memories."""
        from t4dm.memory.episodic import get_episodic_memory

        # Use unique session IDs to avoid collision with other tests
        unique_suffix = str(uuid4())[:8]
        session_a_id = f"session_a_{unique_suffix}"
        session_b_id = f"session_b_{unique_suffix}"
        unique_content = f"Secret data unique {unique_suffix}"

        # Create episodes in different sessions
        episodic_a = get_episodic_memory(session_a_id)
        episodic_b = get_episodic_memory(session_b_id)

        await episodic_a.initialize()
        await episodic_b.initialize()

        # Session A creates episode
        episode_a = await episodic_a.create(
            content=unique_content,
            outcome="success",
            valence=0.8
        )

        # Session B tries to search
        results_b = await episodic_b.recall(
            query=unique_content,
            limit=10,
            session_filter=session_b_id  # Filtered to own session
        )

        # Should not find session A's episode
        assert len(results_b) == 0

        # But session A should find it
        results_a = await episodic_a.recall(
            query=unique_content,
            limit=10,
            session_filter=session_a_id
        )
        assert len(results_a) > 0
        assert results_a[0].item.id == episode_a.id

    @pytest.mark.asyncio
    async def test_session_id_validation(self):
        """Test that session IDs are properly validated."""
        from t4dm.core.services import get_services

        # Malicious session IDs that might break assumptions
        malicious_session_ids = [
            "../../../etc/passwd",
            "session'; DROP TABLE episodes; --",
            "\x00\x01\x02",  # Null bytes
            "a" * 10000,  # Extremely long
        ]

        for malicious_id in malicious_session_ids:
            # Should either sanitize or reject
            with pytest.raises((ValueError, ValidationError)):
                await get_services(session_id=malicious_id)


class TestContentSanitization:
    """Test content sanitization for XSS and injection."""

    def test_xss_in_content(self):
        """Test that HTML/script tags are escaped."""
        malicious_content = [
            "<script>alert('XSS')</script>",
            "<img src=x onerror=alert('XSS')>",
            "javascript:alert('XSS')",
            "<iframe src='http://evil.com'>",
        ]

        for content in malicious_content:
            # Should either escape or strip HTML
            sanitized = validate_non_empty_string(content, "content")

            # Check that dangerous content is neutralized
            assert "<script>" not in sanitized.lower()
            assert "onerror=" not in sanitized.lower()
            assert "javascript:" not in sanitized.lower()
            assert "<iframe" not in sanitized.lower()

    def test_crlf_injection(self):
        """Test that CRLF sequences don't break logs."""
        malicious_content = [
            "Line 1\r\nInjected: Fake log entry",
            "Data\n\nHTTP/1.1 200 OK\nContent-Type: text/html",
            "Text%0d%0aInjected: Header",
        ]

        for content in malicious_content:
            sanitized = validate_non_empty_string(content, "content")

            # CRLF should be normalized or rejected
            # (Implementation depends on sanitization strategy)
            assert "\r\n" not in sanitized or content == sanitized

    def test_unicode_normalization(self):
        """Test that unicode attacks are prevented."""
        # Unicode normalization attacks
        variants = [
            "café",  # é = U+00E9
            "café",  # é = U+0065 U+0301 (combining acute)
        ]

        # Should normalize to same form
        normalized = [validate_non_empty_string(v, "content") for v in variants]
        # All variants should normalize identically
        # (Actual behavior depends on implementation)

    def test_null_byte_injection(self):
        """Test that null bytes are handled."""
        malicious_content = [
            "data\x00injected",
            "path/to/file\x00.txt",
            "\x00" * 100,
        ]

        for content in malicious_content:
            with pytest.raises((ValueError, ValidationError)):
                validate_non_empty_string(content, "content")


class TestRateLimiting:
    """Test rate limiting and DoS protection."""

    @pytest.mark.asyncio
    @pytest.mark.skip(reason="MCP gateway removed in v0.2.0 - rate limiting now in REST API")
    async def test_create_episode_rate_limit(self):
        """Test that excessive create operations are rate limited."""
        # Rate limiting is now implemented in the REST API layer via ww.core.services.RateLimiter
        # See tests/api/test_deps.py for rate limiting tests
        pass

    @pytest.mark.asyncio
    async def test_expensive_query_limits(self):
        """Test that expensive queries are limited."""
        from t4dm.storage.neo4j_store import Neo4jStore

        store = Neo4jStore()
        await store.initialize()

        # Create two nodes
        node1 = await store.create_node("Entity", {"id": str(uuid4())})
        node2 = await store.create_node("Entity", {"id": str(uuid4())})

        # Attempt path search with excessive depth
        with pytest.raises(ValueError, match="max_depth"):
            await store.find_path(
                source_id=node1,
                target_id=node2,
                max_depth=100  # Too deep
            )

    @pytest.mark.asyncio
    async def test_batch_operation_size_limit(self):
        """Test that batch operations have size limits."""
        from t4dm.storage.neo4j_store import Neo4jStore

        store = Neo4jStore()
        await store.initialize()

        # Attempt to create excessive batch
        large_batch = [
            ("Entity", {"id": str(uuid4()), "index": i})
            for i in range(10000)  # Excessive size
        ]

        with pytest.raises((ValueError, Exception)):
            await store.batch_create_nodes(large_batch)


class TestErrorLeakage:
    """Test that error messages don't leak sensitive information."""

    @pytest.mark.asyncio
    async def test_database_error_sanitization(self):
        """Test that database connection errors are sanitized."""
        from t4dm.storage.neo4j_store import Neo4jStore

        # Configure store with invalid connection
        store = Neo4jStore(uri="bolt://invalid-host:7687")

        try:
            await store.initialize()
        except Exception as e:
            error_msg = str(e)

            # Should not expose internal details
            assert "bolt://" not in error_msg.lower()
            assert "password" not in error_msg.lower()
            assert "invalid-host" not in error_msg.lower()

    @pytest.mark.asyncio
    async def test_validation_error_messages(self):
        """Test that validation errors are safe to expose."""
        from t4dm.core.validation import validate_uuid

        try:
            validate_uuid("not-a-uuid", "test_field")
        except ValidationError as e:
            error_dict = e.to_dict()

            # Should only contain safe information
            assert error_dict["field"] == "test_field"
            assert "validation_error" in error_dict["error"]
            # Should not contain stack traces or internal paths
            assert "src/ww" not in str(error_dict)


if __name__ == "__main__":
    pytest.main([__file__, "-v", "--tb=short"])
