"""Tests for API dependencies."""

import pytest
from unittest.mock import AsyncMock, MagicMock, patch

from fastapi import HTTPException


class TestGetSessionId:
    """Tests for get_session_id dependency."""

    @pytest.mark.asyncio
    async def test_returns_header_session(self):
        """Returns session ID from header when valid."""
        from ww.api.deps import get_session_id

        with patch("ww.api.deps.validate_session_id", return_value="valid-session"):
            result = await get_session_id("my-session")
            assert result == "valid-session"

    @pytest.mark.asyncio
    async def test_returns_default_when_none(self):
        """Returns default session when header is None."""
        from ww.api.deps import get_session_id

        with patch("ww.api.deps.validate_session_id", return_value=None):
            with patch("ww.api.deps.get_settings") as mock_settings:
                mock_settings.return_value.session_id = "default-session"
                result = await get_session_id(None)
                assert result == "default-session"

    @pytest.mark.asyncio
    async def test_raises_on_invalid_session(self):
        """Raises HTTPException for invalid session."""
        from ww.api.deps import get_session_id
        from ww.core.validation import SessionValidationError

        with patch(
            "ww.api.deps.validate_session_id",
            side_effect=SessionValidationError("session_id", "Invalid session"),
        ):
            with pytest.raises(HTTPException) as exc:
                await get_session_id("bad\nsession")
            assert exc.value.status_code == 400

    @pytest.mark.asyncio
    async def test_rejects_reserved_session_ids(self):
        """P2-SEC-M2: Reserved session IDs are rejected for security."""
        from ww.api.deps import get_session_id

        # Reserved IDs like "admin", "system", "root" should be rejected
        reserved_ids = ["admin", "system", "root", "default", "test"]

        for reserved_id in reserved_ids:
            with pytest.raises(HTTPException) as exc:
                await get_session_id(reserved_id)
            assert exc.value.status_code == 400, f"Expected {reserved_id} to be rejected"


class TestRateLimiter:
    """Tests for rate limiting."""

    @pytest.mark.asyncio
    async def test_allows_under_limit(self):
        """Allows requests under rate limit."""
        from ww.api.deps import check_rate_limit, _rate_limiter

        # Clear any existing state
        _rate_limiter.requests.clear()

        with patch("ww.api.deps.get_session_id", return_value="test-session"):
            result = await check_rate_limit("test-session")
            assert result == "test-session"

    @pytest.mark.asyncio
    async def test_blocks_over_limit(self):
        """Blocks requests over rate limit."""
        from ww.api.deps import check_rate_limit, _rate_limiter

        # Simulate hitting rate limit
        session = "rate-limited-session"
        _rate_limiter.requests[session] = []

        # Fill up requests
        import time

        current = time.time()
        for _ in range(101):  # Over limit
            _rate_limiter.requests[session].append(current)

        with pytest.raises(HTTPException) as exc:
            await check_rate_limit(session)
        assert exc.value.status_code == 429
        assert "Retry-After" in exc.value.headers


class TestUUIDValidation:
    """Tests for UUID validation helpers."""

    def test_parse_uuid_valid(self):
        """Valid UUID string is parsed correctly."""
        from uuid import UUID
        from ww.api.deps import parse_uuid

        valid_uuid = "12345678-1234-5678-1234-567812345678"
        result = parse_uuid(valid_uuid, "test_id")

        assert isinstance(result, UUID)
        assert str(result) == valid_uuid

    def test_parse_uuid_invalid_format(self):
        """Invalid UUID format raises 400 HTTPException."""
        from ww.api.deps import parse_uuid

        with pytest.raises(HTTPException) as exc:
            parse_uuid("not-a-valid-uuid", "test_id")

        assert exc.value.status_code == 400
        assert "Invalid UUID format" in exc.value.detail
        assert "test_id" in exc.value.detail

    def test_parse_uuid_empty_string(self):
        """Empty string raises 400 HTTPException."""
        from ww.api.deps import parse_uuid

        with pytest.raises(HTTPException) as exc:
            parse_uuid("", "test_id")

        assert exc.value.status_code == 400
        assert "Missing required" in exc.value.detail

    def test_parse_uuid_malformed(self):
        """Malformed UUID raises 400 HTTPException."""
        from ww.api.deps import parse_uuid

        # Too short
        with pytest.raises(HTTPException) as exc:
            parse_uuid("12345", "memory_id")
        assert exc.value.status_code == 400

        # Invalid characters
        with pytest.raises(HTTPException) as exc:
            parse_uuid("gggggggg-gggg-gggg-gggg-gggggggggggg", "memory_id")
        assert exc.value.status_code == 400

    def test_validate_uuid_path(self):
        """validate_uuid_path works as dependency."""
        from uuid import UUID
        from ww.api.deps import validate_uuid_path

        valid_uuid = "12345678-1234-5678-1234-567812345678"
        result = validate_uuid_path(valid_uuid)

        assert isinstance(result, UUID)

    def test_validate_uuid_path_invalid(self):
        """validate_uuid_path raises 400 for invalid input."""
        from ww.api.deps import validate_uuid_path

        with pytest.raises(HTTPException) as exc:
            validate_uuid_path("bad-uuid")

        assert exc.value.status_code == 400


class TestMemoryServices:
    """Tests for memory services dependency."""

    @pytest.mark.asyncio
    async def test_returns_services(self):
        """Returns initialized services."""
        from ww.api.deps import get_memory_services

        mock_episodic = MagicMock()
        mock_semantic = MagicMock()
        mock_procedural = MagicMock()

        with patch(
            "ww.api.deps.get_services",
            new=AsyncMock(
                return_value=(mock_episodic, mock_semantic, mock_procedural)
            ),
        ):
            result = await get_memory_services("test-session")

            assert result["session_id"] == "test-session"
            assert result["episodic"] is mock_episodic
            assert result["semantic"] is mock_semantic
            assert result["procedural"] is mock_procedural

    @pytest.mark.asyncio
    async def test_raises_on_service_error(self):
        """Raises 503 when services unavailable."""
        from ww.api.deps import get_memory_services

        with patch(
            "ww.api.deps.get_services",
            new=AsyncMock(side_effect=Exception("Connection failed")),
        ):
            with pytest.raises(HTTPException) as exc:
                await get_memory_services("test-session")
            assert exc.value.status_code == 503
            assert "unavailable" in exc.value.detail.lower()
