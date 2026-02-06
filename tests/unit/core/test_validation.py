"""Tests for input validation layer.

Tests validate_memory_content, validate_embedding_vector, validate_kappa,
and validate_uuid with comprehensive edge cases.
"""

import numpy as np
import pytest
from uuid import UUID

from t4dm.core.validation import (
    MAX_CONTENT_LENGTH,
    MAX_EMBEDDING_DIM,
    ValidationError,
    validate_embedding_vector,
    validate_kappa,
    validate_memory_content,
    validate_uuid,
)


class TestValidateMemoryContent:
    """Tests for validate_memory_content function."""

    def test_valid_content(self):
        """Valid content passes through."""
        result = validate_memory_content("Hello, world!")
        assert result == "Hello, world!"

    def test_content_with_newlines_preserved(self):
        """Newlines and tabs are preserved."""
        content = "Line 1\nLine 2\tTabbed"
        result = validate_memory_content(content)
        assert result == content

    def test_none_raises_error(self):
        """None content raises ValidationError."""
        with pytest.raises(ValidationError) as exc:
            validate_memory_content(None)
        assert exc.value.field == "content"
        assert "None" in exc.value.message

    def test_non_string_raises_error(self):
        """Non-string types raise ValidationError."""
        with pytest.raises(ValidationError) as exc:
            validate_memory_content(123)
        assert "string" in exc.value.message
        assert "int" in exc.value.message

    def test_empty_string_raises_error(self):
        """Empty string raises ValidationError."""
        with pytest.raises(ValidationError) as exc:
            validate_memory_content("")
        assert "empty" in exc.value.message.lower()

    def test_whitespace_only_raises_error(self):
        """Whitespace-only content raises ValidationError."""
        with pytest.raises(ValidationError) as exc:
            validate_memory_content("   \t\n  ")
        assert "empty" in exc.value.message.lower()

    def test_null_bytes_raise_error(self):
        """Null bytes raise ValidationError."""
        with pytest.raises(ValidationError) as exc:
            validate_memory_content("hello\x00world")
        assert "null" in exc.value.message.lower()

    def test_control_characters_removed(self):
        """Control characters (except newlines/tabs) are removed."""
        content = "hello\x01\x02\x03world"
        result = validate_memory_content(content)
        assert result == "helloworld"

    def test_xss_patterns_sanitized(self):
        """XSS attack patterns are sanitized."""
        xss_content = "<script>alert('xss')</script>hello"
        result = validate_memory_content(xss_content)
        assert "<script>" not in result.lower()
        assert "hello" in result

    def test_javascript_protocol_sanitized(self):
        """javascript: protocol is sanitized."""
        content = "click javascript:alert(1)"
        result = validate_memory_content(content)
        assert "javascript:" not in result.lower()

    def test_max_length_exceeded_raises_error(self):
        """Content exceeding max length raises ValidationError."""
        content = "x" * (MAX_CONTENT_LENGTH + 1)
        with pytest.raises(ValidationError) as exc:
            validate_memory_content(content)
        assert "maximum length" in exc.value.message.lower()

    def test_max_length_exact_passes(self):
        """Content at exactly max length passes."""
        content = "x" * MAX_CONTENT_LENGTH
        result = validate_memory_content(content)
        assert len(result) == MAX_CONTENT_LENGTH

    def test_custom_max_length(self):
        """Custom max_length is respected."""
        content = "x" * 100
        with pytest.raises(ValidationError):
            validate_memory_content(content, max_length=50)

    def test_custom_min_length(self):
        """Custom min_length is respected."""
        with pytest.raises(ValidationError) as exc:
            validate_memory_content("hi", min_length=5)
        assert "at least 5" in exc.value.message

    def test_custom_field_name(self):
        """Custom field name appears in error."""
        with pytest.raises(ValidationError) as exc:
            validate_memory_content(None, field="message")
        assert exc.value.field == "message"

    def test_unicode_content_preserved(self):
        """Unicode content is preserved."""
        content = "Hello, 世界! 🌍 Привет!"
        result = validate_memory_content(content)
        assert result == content

    def test_iframe_sanitized(self):
        """iframe tags are sanitized."""
        content = '<iframe src="evil.com"></iframe>text'
        result = validate_memory_content(content)
        assert "<iframe" not in result.lower()
        assert "text" in result

    def test_onerror_handler_sanitized(self):
        """Event handlers like onerror are sanitized."""
        content = '<img src="x" onerror="alert(1)">'
        result = validate_memory_content(content)
        assert "onerror" not in result.lower()

    def test_leading_trailing_whitespace_kept(self):
        """Leading/trailing whitespace is kept but checked after strip."""
        content = "  hello world  "
        result = validate_memory_content(content)
        # Content is preserved, but stripped version is checked for min length
        assert result == "  hello world  "


class TestValidateEmbeddingVector:
    """Tests for validate_embedding_vector function."""

    def test_valid_embedding(self):
        """Valid embedding passes through."""
        embedding = np.array([0.1, 0.2, 0.3], dtype=np.float32)
        result = validate_embedding_vector(embedding, expected_dim=3)
        assert np.array_equal(result, embedding)
        assert result.dtype == np.float32

    def test_none_raises_error(self):
        """None embedding raises ValidationError."""
        with pytest.raises(ValidationError) as exc:
            validate_embedding_vector(None, expected_dim=3)
        assert exc.value.field == "embedding"
        assert "None" in exc.value.message

    def test_non_array_raises_error(self):
        """Non-numpy types raise ValidationError."""
        with pytest.raises(ValidationError) as exc:
            validate_embedding_vector([0.1, 0.2, 0.3], expected_dim=3)
        assert "numpy array" in exc.value.message

    def test_wrong_dimension_raises_error(self):
        """Wrong dimension raises ValidationError."""
        embedding = np.array([0.1, 0.2, 0.3], dtype=np.float32)
        with pytest.raises(ValidationError) as exc:
            validate_embedding_vector(embedding, expected_dim=5)
        assert "dimension" in exc.value.message.lower()
        assert "5" in exc.value.message
        assert "3" in exc.value.message

    def test_2d_array_raises_error(self):
        """2D array raises ValidationError."""
        embedding = np.array([[0.1, 0.2], [0.3, 0.4]], dtype=np.float32)
        with pytest.raises(ValidationError) as exc:
            validate_embedding_vector(embedding, expected_dim=4)
        assert "1D" in exc.value.message

    def test_nan_values_raise_error(self):
        """NaN values raise ValidationError."""
        embedding = np.array([0.1, np.nan, 0.3], dtype=np.float32)
        with pytest.raises(ValidationError) as exc:
            validate_embedding_vector(embedding, expected_dim=3)
        assert "NaN" in exc.value.message

    def test_inf_values_raise_error(self):
        """Inf values raise ValidationError."""
        embedding = np.array([0.1, np.inf, 0.3], dtype=np.float32)
        with pytest.raises(ValidationError) as exc:
            validate_embedding_vector(embedding, expected_dim=3)
        assert "Inf" in exc.value.message

    def test_negative_inf_raises_error(self):
        """Negative infinity raises ValidationError."""
        embedding = np.array([0.1, -np.inf, 0.3], dtype=np.float32)
        with pytest.raises(ValidationError) as exc:
            validate_embedding_vector(embedding, expected_dim=3)
        assert "Inf" in exc.value.message

    def test_nan_and_inf_combined_message(self):
        """Both NaN and Inf in same array produces combined message."""
        embedding = np.array([np.nan, np.inf, 0.3], dtype=np.float32)
        with pytest.raises(ValidationError) as exc:
            validate_embedding_vector(embedding, expected_dim=3)
        assert "NaN" in exc.value.message
        assert "Inf" in exc.value.message

    def test_float64_converted_to_float32(self):
        """float64 arrays are converted to float32."""
        embedding = np.array([0.1, 0.2, 0.3], dtype=np.float64)
        result = validate_embedding_vector(embedding, expected_dim=3)
        assert result.dtype == np.float32

    def test_int_array_converted(self):
        """Integer arrays are converted to float32."""
        embedding = np.array([1, 2, 3], dtype=np.int32)
        result = validate_embedding_vector(embedding, expected_dim=3)
        assert result.dtype == np.float32
        assert np.allclose(result, [1.0, 2.0, 3.0])

    def test_empty_embedding_raises_error(self):
        """Empty embedding raises ValidationError."""
        embedding = np.array([], dtype=np.float32)
        with pytest.raises(ValidationError) as exc:
            validate_embedding_vector(embedding, expected_dim=0)
        assert "empty" in exc.value.message.lower()

    def test_expected_dim_exceeds_max_raises_error(self):
        """Expected dim exceeding MAX_EMBEDDING_DIM raises ValidationError."""
        embedding = np.zeros(MAX_EMBEDDING_DIM + 1, dtype=np.float32)
        with pytest.raises(ValidationError) as exc:
            validate_embedding_vector(embedding, expected_dim=MAX_EMBEDDING_DIM + 1)
        assert "maximum" in exc.value.message.lower()
        assert str(MAX_EMBEDDING_DIM) in exc.value.message

    def test_max_embedding_dim_passes(self):
        """Embedding at exactly MAX_EMBEDDING_DIM passes."""
        embedding = np.random.randn(MAX_EMBEDDING_DIM).astype(np.float32)
        result = validate_embedding_vector(embedding, expected_dim=MAX_EMBEDDING_DIM)
        assert result.shape[0] == MAX_EMBEDDING_DIM

    def test_custom_field_name(self):
        """Custom field name appears in error."""
        with pytest.raises(ValidationError) as exc:
            validate_embedding_vector(None, expected_dim=3, field="vector")
        assert exc.value.field == "vector"

    def test_zero_vector_passes(self):
        """Zero vector passes validation (no norm check in this function)."""
        embedding = np.zeros(3, dtype=np.float32)
        result = validate_embedding_vector(embedding, expected_dim=3)
        assert np.array_equal(result, embedding)

    def test_large_values_pass(self):
        """Large but finite values pass."""
        embedding = np.array([1e30, -1e30, 1e30], dtype=np.float32)
        result = validate_embedding_vector(embedding, expected_dim=3)
        assert np.all(np.isfinite(result))


class TestValidateKappa:
    """Tests for validate_kappa function."""

    def test_valid_kappa_zero(self):
        """Kappa 0.0 is valid."""
        result = validate_kappa(0.0)
        assert result == 0.0
        assert isinstance(result, float)

    def test_valid_kappa_one(self):
        """Kappa 1.0 is valid."""
        result = validate_kappa(1.0)
        assert result == 1.0

    def test_valid_kappa_middle(self):
        """Kappa in the middle of range is valid."""
        result = validate_kappa(0.5)
        assert result == 0.5

    def test_valid_kappa_typical_values(self):
        """Typical kappa values are valid."""
        # Raw episodic
        assert validate_kappa(0.0) == 0.0
        # Replayed
        assert validate_kappa(0.15) == 0.15
        # Transitional
        assert validate_kappa(0.4) == 0.4
        # Semantic
        assert validate_kappa(0.85) == 0.85
        # Stable
        assert validate_kappa(1.0) == 1.0

    def test_integer_converted_to_float(self):
        """Integer inputs are converted to float."""
        result = validate_kappa(1)
        assert result == 1.0
        assert isinstance(result, float)

    def test_none_raises_error(self):
        """None kappa raises ValidationError."""
        with pytest.raises(ValidationError) as exc:
            validate_kappa(None)
        assert exc.value.field == "kappa"
        assert "None" in exc.value.message

    def test_non_numeric_raises_error(self):
        """Non-numeric types raise ValidationError."""
        with pytest.raises(ValidationError) as exc:
            validate_kappa("0.5")
        assert "number" in exc.value.message.lower()

    def test_negative_raises_error(self):
        """Negative kappa raises ValidationError."""
        with pytest.raises(ValidationError) as exc:
            validate_kappa(-0.1)
        assert ">= 0" in exc.value.message

    def test_greater_than_one_raises_error(self):
        """Kappa > 1 raises ValidationError."""
        with pytest.raises(ValidationError) as exc:
            validate_kappa(1.1)
        assert "<= 1" in exc.value.message

    def test_nan_raises_error(self):
        """NaN kappa raises ValidationError."""
        with pytest.raises(ValidationError) as exc:
            validate_kappa(float("nan"))
        assert "finite" in exc.value.message.lower() or "NaN" in exc.value.message

    def test_inf_raises_error(self):
        """Inf kappa raises ValidationError."""
        with pytest.raises(ValidationError) as exc:
            validate_kappa(float("inf"))
        assert "finite" in exc.value.message.lower() or "Inf" in exc.value.message

    def test_negative_inf_raises_error(self):
        """Negative infinity raises ValidationError."""
        with pytest.raises(ValidationError) as exc:
            validate_kappa(float("-inf"))
        assert "finite" in exc.value.message.lower() or "Inf" in exc.value.message

    def test_custom_field_name(self):
        """Custom field name appears in error."""
        with pytest.raises(ValidationError) as exc:
            validate_kappa(None, field="consolidation_level")
        assert exc.value.field == "consolidation_level"

    def test_boundary_values(self):
        """Boundary values are handled correctly."""
        # Just inside bounds
        assert validate_kappa(0.0001) == 0.0001
        assert validate_kappa(0.9999) == 0.9999

        # Just outside bounds
        with pytest.raises(ValidationError):
            validate_kappa(-0.0001)
        with pytest.raises(ValidationError):
            validate_kappa(1.0001)


class TestValidateUUID:
    """Tests for validate_uuid function."""

    def test_valid_uuid_string(self):
        """Valid UUID string is parsed."""
        uuid_str = "550e8400-e29b-41d4-a716-446655440000"
        result = validate_uuid(uuid_str)
        assert isinstance(result, UUID)
        assert str(result) == uuid_str

    def test_valid_uuid_uppercase(self):
        """Uppercase UUID string is parsed."""
        uuid_str = "550E8400-E29B-41D4-A716-446655440000"
        result = validate_uuid(uuid_str)
        assert isinstance(result, UUID)

    def test_valid_uuid_no_hyphens(self):
        """UUID string without hyphens is parsed."""
        uuid_str = "550e8400e29b41d4a716446655440000"
        result = validate_uuid(uuid_str)
        assert isinstance(result, UUID)

    def test_none_raises_error(self):
        """None raises ValidationError."""
        with pytest.raises(ValidationError) as exc:
            validate_uuid(None)
        assert "None" in exc.value.message

    def test_non_string_raises_error(self):
        """Non-string types raise ValidationError."""
        with pytest.raises(ValidationError) as exc:
            validate_uuid(123)
        assert "string" in exc.value.message

    def test_invalid_format_raises_error(self):
        """Invalid UUID format raises ValidationError."""
        with pytest.raises(ValidationError) as exc:
            validate_uuid("not-a-uuid")
        assert "Invalid UUID" in exc.value.message

    def test_empty_string_raises_error(self):
        """Empty string raises ValidationError."""
        with pytest.raises(ValidationError) as exc:
            validate_uuid("")
        assert "Invalid UUID" in exc.value.message

    def test_custom_field_name(self):
        """Custom field name appears in error."""
        with pytest.raises(ValidationError) as exc:
            validate_uuid("invalid", field="memory_id")
        assert exc.value.field == "memory_id"

    def test_uuid_with_braces(self):
        """UUID with braces is parsed."""
        uuid_str = "{550e8400-e29b-41d4-a716-446655440000}"
        result = validate_uuid(uuid_str)
        assert isinstance(result, UUID)


class TestValidationErrorClass:
    """Tests for ValidationError exception class."""

    def test_basic_properties(self):
        """ValidationError has expected properties."""
        error = ValidationError("field_name", "error message", value="bad_value")
        assert error.field == "field_name"
        assert error.message == "error message"
        assert error.value == "bad_value"

    def test_string_representation(self):
        """ValidationError str includes field and message."""
        error = ValidationError("test_field", "test message")
        assert "test_field" in str(error)
        assert "test message" in str(error)

    def test_to_dict(self):
        """to_dict returns expected format."""
        error = ValidationError("my_field", "my message")
        result = error.to_dict()
        assert result["error"] == "validation_error"
        assert result["field"] == "my_field"
        assert result["message"] == "my message"

    def test_inherits_from_valueerror(self):
        """ValidationError inherits from ValueError."""
        error = ValidationError("field", "message")
        assert isinstance(error, ValueError)


class TestConstants:
    """Tests for validation constants."""

    def test_max_content_length_value(self):
        """MAX_CONTENT_LENGTH has expected value."""
        assert MAX_CONTENT_LENGTH == 100000

    def test_max_embedding_dim_value(self):
        """MAX_EMBEDDING_DIM has expected value."""
        assert MAX_EMBEDDING_DIM == 4096
