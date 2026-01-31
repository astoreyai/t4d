"""
Unit tests for the validation framework.

Tests all validators and error handling in the validation module:
- ValidationError class and its methods
- UUID validation
- Valence/weight/score validation (0-1 range)
- Enum validation
- Range validation
- Positive integer validation
- Non-empty string validation
- List validation
- Dictionary validation
"""

import pytest
from uuid import UUID, uuid4
from enum import Enum

from datetime import datetime, timezone
import numpy as np

from t4dm.core.validation import (
    ValidationError,
    validate_uuid,
    validate_uuid_list,
    validate_range,
    validate_positive_int,
    validate_enum,
    validate_non_empty_string,
    validate_dict,
    validate_list,
    validate_valence,
    validate_weight,
    validate_score,
    sanitize_string,
    sanitize_identifier,
    sanitize_session_id,
    validate_limit,
    validate_float_range,
    validate_metadata,
    validate_array,
    validate_embedding,
    validate_timestamp,
    validate_nt_level,
    validate_memory_id,
    validate_spike_interval,
)


# Test fixtures and helper classes

class SampleEnum(Enum):
    """Enum for testing enum validation."""
    OPTION_A = "option_a"
    OPTION_B = "option_b"
    OPTION_C = "option_c"


@pytest.fixture
def valid_uuid_string():
    """Fixture providing a valid UUID string."""
    return str(uuid4())


@pytest.fixture
def valid_uuid_object():
    """Fixture providing a valid UUID object."""
    return uuid4()


# ValidationError Tests

class TestValidationError:
    """Tests for ValidationError class."""

    def test_validation_error_creation(self):
        """Test creating ValidationError with all attributes."""
        error = ValidationError("username", "Username is required", "")
        assert error.field == "username"
        assert error.message == "Username is required"
        assert error.value == ""

    def test_validation_error_str_representation(self):
        """Test string representation of ValidationError."""
        error = ValidationError("email", "Invalid email format", "notanemail")
        assert str(error) == "email: Invalid email format"

    def test_validation_error_to_dict(self):
        """Test to_dict() method returns proper MCP error format."""
        error = ValidationError("field_name", "Field is invalid", "bad_value")
        error_dict = error.to_dict()

        assert error_dict["error"] == "validation_error"
        assert error_dict["field"] == "field_name"
        assert error_dict["message"] == "Field is invalid"
        assert "value" not in error_dict  # value is not included in dict output

    def test_validation_error_to_dict_with_none_value(self):
        """Test to_dict() when value is None."""
        error = ValidationError("field", "Cannot be None")
        error_dict = error.to_dict()

        assert error_dict["error"] == "validation_error"
        assert error_dict["field"] == "field"

    def test_validation_error_inheritance(self):
        """Test that ValidationError inherits from ValueError."""
        error = ValidationError("field", "Invalid")
        assert isinstance(error, ValueError)

    def test_validation_error_with_special_characters(self):
        """Test ValidationError with special characters in message."""
        error = ValidationError("field", "Invalid: value must be > 0 & < 100")
        error_dict = error.to_dict()
        assert error_dict["message"] == "Invalid: value must be > 0 & < 100"


# UUID Validation Tests

class TestValidateUUID:
    """Tests for validate_uuid function."""

    def test_validate_uuid_valid_string(self, valid_uuid_string):
        """Test validating a valid UUID string."""
        result = validate_uuid(valid_uuid_string)
        assert isinstance(result, UUID)
        assert str(result) == valid_uuid_string

    def test_validate_uuid_valid_string_different_field_name(self, valid_uuid_string):
        """Test validating UUID with custom field name."""
        result = validate_uuid(valid_uuid_string, field="entity_id")
        assert isinstance(result, UUID)

    def test_validate_uuid_none_value(self):
        """Test that None value raises ValidationError."""
        with pytest.raises(ValidationError) as exc_info:
            validate_uuid(None)
        assert exc_info.value.field == "id"
        assert "cannot be none" in exc_info.value.message.lower()

    def test_validate_uuid_non_string_type(self):
        """Test that non-string type raises ValidationError."""
        with pytest.raises(ValidationError) as exc_info:
            validate_uuid(12345)
        assert exc_info.value.field == "id"
        assert "expected string" in exc_info.value.message.lower()

    def test_validate_uuid_invalid_format(self):
        """Test that invalid UUID format raises ValidationError."""
        with pytest.raises(ValidationError) as exc_info:
            validate_uuid("not-a-uuid")
        assert exc_info.value.field == "id"
        assert "invalid uuid" in exc_info.value.message.lower()

    def test_validate_uuid_empty_string(self):
        """Test that empty string raises ValidationError."""
        with pytest.raises(ValidationError) as exc_info:
            validate_uuid("")
        assert exc_info.value.field == "id"

    def test_validate_uuid_custom_field_name(self, valid_uuid_string):
        """Test validate_uuid with custom field name in error."""
        with pytest.raises(ValidationError) as exc_info:
            validate_uuid("invalid-uuid", field="resource_id")
        assert exc_info.value.field == "resource_id"

    def test_validate_uuid_with_uuid_object(self, valid_uuid_object):
        """Test that UUID object (not string) raises ValidationError."""
        with pytest.raises(ValidationError) as exc_info:
            validate_uuid(valid_uuid_object)
        assert "expected string" in exc_info.value.message.lower()

    def test_validate_uuid_list_valid(self, valid_uuid_string):
        """Test validating a list of valid UUID strings."""
        uuid2 = str(uuid4())
        uuids = [valid_uuid_string, uuid2]
        result = validate_uuid_list(uuids)

        assert len(result) == 2
        assert all(isinstance(u, UUID) for u in result)
        assert str(result[0]) == valid_uuid_string
        assert str(result[1]) == uuid2

    def test_validate_uuid_list_empty(self):
        """Test validating empty UUID list."""
        result = validate_uuid_list([])
        assert result == []

    def test_validate_uuid_list_invalid_type(self):
        """Test that non-list type raises ValidationError."""
        with pytest.raises(ValidationError) as exc_info:
            validate_uuid_list("not-a-list")
        assert exc_info.value.field == "ids"
        assert "expected list" in exc_info.value.message.lower()

    def test_validate_uuid_list_with_invalid_uuid(self):
        """Test that invalid UUID in list raises ValidationError."""
        with pytest.raises(ValidationError) as exc_info:
            validate_uuid_list([str(uuid4()), "invalid-uuid"])
        assert exc_info.value.field == "ids"
        assert "invalid uuid" in exc_info.value.message.lower()

    def test_validate_uuid_list_with_none_in_list(self):
        """Test that None in list raises ValidationError."""
        with pytest.raises(ValidationError) as exc_info:
            validate_uuid_list([str(uuid4()), None])
        assert exc_info.value.field == "ids"


# Range Validation Tests

class TestValidateRange:
    """Tests for validate_range function."""

    def test_validate_range_valid_float(self):
        """Test validating float within range."""
        result = validate_range(0.5, 0.0, 1.0, "score")
        assert result == 0.5

    def test_validate_range_valid_int(self):
        """Test validating int within range."""
        result = validate_range(5, 0, 10, "count")
        assert result == 5.0
        assert isinstance(result, float)

    def test_validate_range_min_boundary(self):
        """Test validating at minimum boundary (inclusive)."""
        result = validate_range(0.0, 0.0, 1.0, "field")
        assert result == 0.0

    def test_validate_range_max_boundary(self):
        """Test validating at maximum boundary (inclusive)."""
        result = validate_range(1.0, 0.0, 1.0, "field")
        assert result == 1.0

    def test_validate_range_below_min(self):
        """Test value below minimum raises ValidationError."""
        with pytest.raises(ValidationError) as exc_info:
            validate_range(-0.5, 0.0, 1.0, "score")
        assert exc_info.value.field == "score"
        assert "between" in exc_info.value.message.lower()

    def test_validate_range_above_max(self):
        """Test value above maximum raises ValidationError."""
        with pytest.raises(ValidationError) as exc_info:
            validate_range(1.5, 0.0, 1.0, "score")
        assert exc_info.value.field == "score"
        assert "between" in exc_info.value.message.lower()

    def test_validate_range_none_not_allowed(self):
        """Test that None raises ValidationError when not allowed."""
        with pytest.raises(ValidationError) as exc_info:
            validate_range(None, 0.0, 1.0, "field", allow_none=False)
        assert exc_info.value.field == "field"
        assert "cannot be none" in exc_info.value.message.lower()

    def test_validate_range_none_allowed(self):
        """Test that None returns None when allowed."""
        result = validate_range(None, 0.0, 1.0, "field", allow_none=True)
        assert result is None

    def test_validate_range_invalid_type(self):
        """Test that non-numeric type raises ValidationError."""
        with pytest.raises(ValidationError) as exc_info:
            validate_range("not-a-number", 0.0, 1.0, "field")
        assert exc_info.value.field == "field"
        assert "expected number" in exc_info.value.message.lower()

    def test_validate_range_negative_range(self):
        """Test range with negative values."""
        result = validate_range(-5.5, -10.0, 0.0, "temperature")
        assert result == -5.5

    def test_validate_range_negative_out_of_bounds(self):
        """Test negative value outside negative range."""
        with pytest.raises(ValidationError):
            validate_range(-15.0, -10.0, 0.0, "temperature")


# Positive Integer Validation Tests

class TestValidatePositiveInt:
    """Tests for validate_positive_int function."""

    def test_validate_positive_int_valid(self):
        """Test validating positive integer."""
        result = validate_positive_int(5, "count")
        assert result == 5

    def test_validate_positive_int_one(self):
        """Test validating positive integer equals 1."""
        result = validate_positive_int(1, "count")
        assert result == 1

    def test_validate_positive_int_large(self):
        """Test validating large positive integer."""
        result = validate_positive_int(999999, "count")
        assert result == 999999

    def test_validate_positive_int_zero(self):
        """Test that zero raises ValidationError."""
        with pytest.raises(ValidationError) as exc_info:
            validate_positive_int(0, "count")
        assert exc_info.value.field == "count"
        assert "positive" in exc_info.value.message.lower()

    def test_validate_positive_int_negative(self):
        """Test that negative integer raises ValidationError."""
        with pytest.raises(ValidationError) as exc_info:
            validate_positive_int(-5, "count")
        assert exc_info.value.field == "count"
        assert "positive" in exc_info.value.message.lower()

    def test_validate_positive_int_float_rejects(self):
        """Test that float raises ValidationError."""
        with pytest.raises(ValidationError) as exc_info:
            validate_positive_int(5.5, "count")
        assert exc_info.value.field == "count"
        assert "expected integer" in exc_info.value.message.lower()

    def test_validate_positive_int_bool_rejects(self):
        """Test that bool raises ValidationError."""
        with pytest.raises(ValidationError) as exc_info:
            validate_positive_int(True, "count")
        assert exc_info.value.field == "count"
        assert "expected integer" in exc_info.value.message.lower()

    def test_validate_positive_int_string_rejects(self):
        """Test that string raises ValidationError."""
        with pytest.raises(ValidationError) as exc_info:
            validate_positive_int("5", "count")
        assert exc_info.value.field == "count"

    def test_validate_positive_int_with_max_value(self):
        """Test validating positive int with max constraint."""
        result = validate_positive_int(5, "count", max_val=10)
        assert result == 5

    def test_validate_positive_int_exceeds_max(self):
        """Test that exceeding max_val raises ValidationError."""
        with pytest.raises(ValidationError) as exc_info:
            validate_positive_int(15, "count", max_val=10)
        assert exc_info.value.field == "count"
        assert "must be <=" in exc_info.value.message.lower()

    def test_validate_positive_int_at_max_boundary(self):
        """Test value at max boundary."""
        result = validate_positive_int(10, "count", max_val=10)
        assert result == 10


# Enum Validation Tests

class TestValidateEnum:
    """Tests for validate_enum function."""

    def test_validate_enum_valid_value(self):
        """Test validating valid enum value."""
        result = validate_enum("option_a", SampleEnum, "choice")
        assert result == SampleEnum.OPTION_A
        assert result.value == "option_a"

    def test_validate_enum_valid_different_value(self):
        """Test validating different valid enum value."""
        result = validate_enum("option_c", SampleEnum, "choice")
        assert result == SampleEnum.OPTION_C

    def test_validate_enum_none_value(self):
        """Test that None raises ValidationError."""
        with pytest.raises(ValidationError) as exc_info:
            validate_enum(None, SampleEnum, "choice")
        assert exc_info.value.field == "choice"
        assert "cannot be none" in exc_info.value.message.lower()

    def test_validate_enum_invalid_value(self):
        """Test that invalid enum value raises ValidationError."""
        with pytest.raises(ValidationError) as exc_info:
            validate_enum("invalid_option", SampleEnum, "choice")
        assert exc_info.value.field == "choice"
        assert "invalid value" in exc_info.value.message.lower()

    def test_validate_enum_non_string_type(self):
        """Test that non-string type raises ValidationError."""
        with pytest.raises(ValidationError) as exc_info:
            validate_enum(123, SampleEnum, "choice")
        assert exc_info.value.field == "choice"
        assert "expected string" in exc_info.value.message.lower()

    def test_validate_enum_error_message_contains_valid_values(self):
        """Test that error message lists valid enum values."""
        with pytest.raises(ValidationError) as exc_info:
            validate_enum("wrong", SampleEnum, "choice")
        message = exc_info.value.message
        assert "option_a" in message
        assert "option_b" in message
        assert "option_c" in message

    def test_validate_enum_custom_field_name(self):
        """Test enum validation with custom field name."""
        with pytest.raises(ValidationError) as exc_info:
            validate_enum("bad", SampleEnum, "status_type")
        assert exc_info.value.field == "status_type"

    def test_validate_enum_empty_string(self):
        """Test that empty string raises ValidationError."""
        with pytest.raises(ValidationError):
            validate_enum("", SampleEnum, "choice")


# Non-Empty String Validation Tests

class TestValidateNonEmptyString:
    """Tests for validate_non_empty_string function."""

    def test_validate_non_empty_string_valid(self):
        """Test validating non-empty string."""
        result = validate_non_empty_string("hello", "name")
        assert result == "hello"

    def test_validate_non_empty_string_with_spaces(self):
        """Test validating string with internal spaces."""
        result = validate_non_empty_string("hello world", "description")
        assert result == "hello world"

    def test_validate_non_empty_string_special_chars(self):
        """Test validating string with special characters."""
        result = validate_non_empty_string("hello-world_123!", "value")
        assert result == "hello-world_123!"

    def test_validate_non_empty_string_unicode(self):
        """Test validating unicode string."""
        result = validate_non_empty_string("こんにちは", "greeting")
        assert result == "こんにちは"

    def test_validate_non_empty_string_none(self):
        """Test that None raises ValidationError."""
        with pytest.raises(ValidationError) as exc_info:
            validate_non_empty_string(None, "name")
        assert exc_info.value.field == "name"
        assert "cannot be none" in exc_info.value.message.lower()

    def test_validate_non_empty_string_empty(self):
        """Test that empty string raises ValidationError."""
        with pytest.raises(ValidationError) as exc_info:
            validate_non_empty_string("", "name")
        assert exc_info.value.field == "name"
        assert "cannot be empty" in exc_info.value.message.lower()

    def test_validate_non_empty_string_whitespace_only(self):
        """Test that whitespace-only string raises ValidationError."""
        with pytest.raises(ValidationError) as exc_info:
            validate_non_empty_string("   ", "name")
        assert exc_info.value.field == "name"
        assert "cannot be empty" in exc_info.value.message.lower()

    def test_validate_non_empty_string_non_string_type(self):
        """Test that non-string type raises ValidationError."""
        with pytest.raises(ValidationError) as exc_info:
            validate_non_empty_string(123, "name")
        assert exc_info.value.field == "name"
        assert "expected string" in exc_info.value.message.lower()

    def test_validate_non_empty_string_with_max_length(self):
        """Test string validation with max_length constraint."""
        result = validate_non_empty_string("hello", "text", max_length=10)
        assert result == "hello"

    def test_validate_non_empty_string_exceeds_max_length(self):
        """Test that string exceeding max_length raises ValidationError."""
        with pytest.raises(ValidationError) as exc_info:
            validate_non_empty_string("hello world", "text", max_length=5)
        assert exc_info.value.field == "text"
        assert "exceeds maximum length" in exc_info.value.message.lower()

    def test_validate_non_empty_string_at_max_length_boundary(self):
        """Test string exactly at max_length boundary."""
        result = validate_non_empty_string("hello", "text", max_length=5)
        assert result == "hello"

    def test_validate_non_empty_string_max_length_with_long_string(self):
        """Test max_length constraint with longer string."""
        with pytest.raises(ValidationError) as exc_info:
            validate_non_empty_string("a" * 101, "text", max_length=100)
        assert "exceeds maximum length" in exc_info.value.message.lower()


# Dictionary Validation Tests

class TestValidateDict:
    """Tests for validate_dict function."""

    def test_validate_dict_valid_empty(self):
        """Test validating empty dict."""
        result = validate_dict({}, "config")
        assert result == {}

    def test_validate_dict_valid_with_data(self):
        """Test validating dict with data."""
        test_dict = {"key": "value", "number": 42}
        result = validate_dict(test_dict, "config")
        assert result == test_dict

    def test_validate_dict_none_returns_empty(self):
        """Test that None returns empty dict."""
        result = validate_dict(None, "config")
        assert result == {}

    def test_validate_dict_invalid_type(self):
        """Test that non-dict type raises ValidationError."""
        with pytest.raises(ValidationError) as exc_info:
            validate_dict("not-a-dict", "config")
        assert exc_info.value.field == "config"
        assert "expected dict" in exc_info.value.message.lower()

    def test_validate_dict_list_raises_error(self):
        """Test that list raises ValidationError."""
        with pytest.raises(ValidationError):
            validate_dict([], "config")

    def test_validate_dict_with_required_keys_present(self):
        """Test dict with all required keys."""
        test_dict = {"name": "John", "age": 30}
        result = validate_dict(test_dict, "user", required_keys=["name", "age"])
        assert result == test_dict

    def test_validate_dict_with_required_keys_missing_one(self):
        """Test dict missing one required key."""
        test_dict = {"name": "John"}
        with pytest.raises(ValidationError) as exc_info:
            validate_dict(test_dict, "user", required_keys=["name", "age"])
        assert exc_info.value.field == "user"
        assert "missing required keys" in exc_info.value.message.lower()
        assert "age" in exc_info.value.message

    def test_validate_dict_with_required_keys_missing_multiple(self):
        """Test dict missing multiple required keys."""
        test_dict = {"name": "John"}
        with pytest.raises(ValidationError) as exc_info:
            validate_dict(test_dict, "user", required_keys=["name", "age", "email"])
        message = exc_info.value.message.lower()
        assert "missing required keys" in message

    def test_validate_dict_extra_keys_allowed(self):
        """Test that extra keys are allowed."""
        test_dict = {"name": "John", "age": 30, "email": "john@example.com"}
        result = validate_dict(test_dict, "user", required_keys=["name", "age"])
        assert result == test_dict

    def test_validate_dict_nested_dict(self):
        """Test validating dict with nested dicts."""
        test_dict = {"user": {"name": "John"}, "settings": {"theme": "dark"}}
        result = validate_dict(test_dict, "config")
        assert result == test_dict


# List Validation Tests

class TestValidateList:
    """Tests for validate_list function."""

    def test_validate_list_empty(self):
        """Test validating empty list."""
        result = validate_list([], "items")
        assert result == []

    def test_validate_list_with_items(self):
        """Test validating list with items."""
        test_list = [1, 2, 3]
        result = validate_list(test_list, "items")
        assert result == test_list

    def test_validate_list_with_strings(self):
        """Test validating list of strings."""
        test_list = ["a", "b", "c"]
        result = validate_list(test_list, "tags")
        assert result == test_list

    def test_validate_list_none_min_length_zero(self):
        """Test that None returns empty list when min_length=0."""
        result = validate_list(None, "items", min_length=0)
        assert result == []

    def test_validate_list_none_min_length_gt_zero(self):
        """Test that None raises error when min_length > 0."""
        with pytest.raises(ValidationError) as exc_info:
            validate_list(None, "items", min_length=1)
        assert exc_info.value.field == "items"
        assert "cannot be none" in exc_info.value.message.lower()

    def test_validate_list_invalid_type(self):
        """Test that non-list type raises ValidationError."""
        with pytest.raises(ValidationError) as exc_info:
            validate_list("not-a-list", "items")
        assert exc_info.value.field == "items"
        assert "expected list" in exc_info.value.message.lower()

    def test_validate_list_min_length_satisfied(self):
        """Test list with length >= min_length."""
        result = validate_list([1, 2, 3], "items", min_length=2)
        assert result == [1, 2, 3]

    def test_validate_list_min_length_not_satisfied(self):
        """Test list with length < min_length."""
        with pytest.raises(ValidationError) as exc_info:
            validate_list([1], "items", min_length=2)
        assert exc_info.value.field == "items"
        assert "at least" in exc_info.value.message.lower()

    def test_validate_list_max_length_satisfied(self):
        """Test list with length <= max_length."""
        result = validate_list([1, 2], "items", max_length=3)
        assert result == [1, 2]

    def test_validate_list_max_length_exceeded(self):
        """Test list with length > max_length."""
        with pytest.raises(ValidationError) as exc_info:
            validate_list([1, 2, 3, 4, 5], "items", max_length=3)
        assert exc_info.value.field == "items"
        assert "cannot exceed" in exc_info.value.message.lower()

    def test_validate_list_both_constraints(self):
        """Test list with both min and max length constraints."""
        result = validate_list([1, 2, 3], "items", min_length=2, max_length=5)
        assert result == [1, 2, 3]

    def test_validate_list_at_min_boundary(self):
        """Test list exactly at min_length boundary."""
        result = validate_list([1, 2], "items", min_length=2)
        assert result == [1, 2]

    def test_validate_list_at_max_boundary(self):
        """Test list exactly at max_length boundary."""
        result = validate_list([1, 2, 3], "items", max_length=3)
        assert result == [1, 2, 3]

    def test_validate_list_mixed_types(self):
        """Test list with mixed types."""
        test_list = [1, "string", 3.14, True]
        result = validate_list(test_list, "mixed")
        assert result == test_list

    def test_validate_list_dict_raises_error(self):
        """Test that dict raises ValidationError."""
        with pytest.raises(ValidationError):
            validate_list({"key": "value"}, "items")


# Convenience Validator Tests

class TestValenceValidator:
    """Tests for convenience validate_valence function."""

    def test_validate_valence_valid(self):
        """Test validating valid valence value."""
        result = validate_valence(0.5)
        assert result == 0.5

    def test_validate_valence_zero(self):
        """Test valence at minimum boundary."""
        result = validate_valence(0.0)
        assert result == 0.0

    def test_validate_valence_one(self):
        """Test valence at maximum boundary."""
        result = validate_valence(1.0)
        assert result == 1.0

    def test_validate_valence_out_of_range(self):
        """Test that valence outside [0,1] raises error."""
        with pytest.raises(ValidationError) as exc_info:
            validate_valence(1.5)
        assert exc_info.value.field == "valence"

    def test_validate_valence_negative(self):
        """Test that negative valence raises error."""
        with pytest.raises(ValidationError):
            validate_valence(-0.1)

    def test_validate_valence_none_not_allowed(self):
        """Test that None raises error by default."""
        with pytest.raises(ValidationError) as exc_info:
            validate_valence(None, allow_none=False)
        assert "cannot be none" in exc_info.value.message.lower()

    def test_validate_valence_none_allowed(self):
        """Test that None returns None when allowed."""
        result = validate_valence(None, allow_none=True)
        assert result is None

    def test_validate_valence_custom_field_name(self):
        """Test valence with custom field name."""
        with pytest.raises(ValidationError) as exc_info:
            validate_valence(1.5, field="emotion_valence")
        assert exc_info.value.field == "emotion_valence"


class TestWeightValidator:
    """Tests for convenience validate_weight function."""

    def test_validate_weight_valid(self):
        """Test validating valid weight value."""
        result = validate_weight(0.75)
        assert result == 0.75

    def test_validate_weight_zero(self):
        """Test weight at minimum boundary."""
        result = validate_weight(0.0)
        assert result == 0.0

    def test_validate_weight_one(self):
        """Test weight at maximum boundary."""
        result = validate_weight(1.0)
        assert result == 1.0

    def test_validate_weight_out_of_range(self):
        """Test that weight outside [0,1] raises error."""
        with pytest.raises(ValidationError) as exc_info:
            validate_weight(2.0)
        assert exc_info.value.field == "weight"

    def test_validate_weight_negative(self):
        """Test that negative weight raises error."""
        with pytest.raises(ValidationError):
            validate_weight(-0.05)

    def test_validate_weight_none_allowed(self):
        """Test that None returns None when allowed."""
        result = validate_weight(None, allow_none=True)
        assert result is None


class TestScoreValidator:
    """Tests for convenience validate_score function."""

    def test_validate_score_valid(self):
        """Test validating valid score value."""
        result = validate_score(0.9)
        assert result == 0.9

    def test_validate_score_zero(self):
        """Test score at minimum boundary."""
        result = validate_score(0.0)
        assert result == 0.0

    def test_validate_score_one(self):
        """Test score at maximum boundary."""
        result = validate_score(1.0)
        assert result == 1.0

    def test_validate_score_out_of_range(self):
        """Test that score outside [0,1] raises error."""
        with pytest.raises(ValidationError) as exc_info:
            validate_score(1.1)
        assert exc_info.value.field == "score"

    def test_validate_score_negative(self):
        """Test that negative score raises error."""
        with pytest.raises(ValidationError):
            validate_score(-0.01)

    def test_validate_score_none_allowed(self):
        """Test that None returns None when allowed."""
        result = validate_score(None, allow_none=True)
        assert result is None


# Integration Tests - Testing combinations of validators

class TestValidationIntegration:
    """Integration tests using multiple validators together."""

    def test_validate_user_data(self, valid_uuid_string):
        """Test validating a complete user data object."""
        user_data = {
            "id": valid_uuid_string,
            "name": "John Doe",
            "status": "option_a",
            "score": 0.85,
        }

        # Validate each field
        user_id = validate_uuid(user_data["id"])
        name = validate_non_empty_string(user_data["name"], "name", max_length=100)
        status = validate_enum(user_data["status"], SampleEnum, "status")
        score = validate_score(user_data["score"])

        assert user_id is not None
        assert name == "John Doe"
        assert status == SampleEnum.OPTION_A
        assert score == 0.85

    def test_validate_config_dict_with_items(self):
        """Test validating complex config structure."""
        config = {
            "items": [1, 2, 3, 4],
            "metadata": {"version": 1, "author": "test"},
        }

        validated_config = validate_dict(
            config,
            "config",
            required_keys=["items", "metadata"]
        )
        validated_items = validate_list(
            validated_config["items"],
            "items",
            min_length=1,
            max_length=10
        )

        assert validated_config["metadata"]["version"] == 1
        assert len(validated_items) == 4

    def test_validation_error_chain(self):
        """Test catching multiple validation errors in sequence."""
        errors = []

        test_cases = [
            (lambda: validate_uuid("bad-uuid"), "UUID validation"),
            (lambda: validate_positive_int(-5, "count"), "Positive int validation"),
            (lambda: validate_valence(1.5), "Valence validation"),
        ]

        for test_func, label in test_cases:
            with pytest.raises(ValidationError):
                test_func()


# Edge case and stress tests

class TestValidationEdgeCases:
    """Tests for edge cases and boundary conditions."""

    def test_very_long_string_validation(self):
        """Test validation of very long string."""
        long_string = "a" * 10000
        result = validate_non_empty_string(long_string, "text")
        assert len(result) == 10000

    def test_very_long_string_exceeds_max(self):
        """Test that very long string exceeding max raises error."""
        long_string = "a" * 1001
        with pytest.raises(ValidationError):
            validate_non_empty_string(long_string, "text", max_length=1000)

    def test_large_list_validation(self):
        """Test validation of large list."""
        large_list = list(range(1000))
        result = validate_list(large_list, "items", max_length=1000)
        assert len(result) == 1000

    def test_large_list_exceeds_max(self):
        """Test that large list exceeding max raises error."""
        large_list = list(range(1001))
        with pytest.raises(ValidationError):
            validate_list(large_list, "items", max_length=1000)

    def test_deeply_nested_dict(self):
        """Test validation of deeply nested dict."""
        nested = {"level1": {"level2": {"level3": {"value": "deep"}}}}
        result = validate_dict(nested, "config")
        assert result["level1"]["level2"]["level3"]["value"] == "deep"

    def test_float_precision_edge_case(self):
        """Test float validation with precision edge cases."""
        # Test very small positive number
        result = validate_range(0.0000001, 0.0, 1.0, "field")
        assert result > 0

        # Test number very close to boundary
        result = validate_range(0.9999999, 0.0, 1.0, "field")
        assert result < 1.0

    def test_int_boundary_validation(self):
        """Test integer validation at system boundaries."""
        large_int = 2**31 - 1  # Max 32-bit int
        result = validate_positive_int(large_int, "count")
        assert result == large_int


# =============================================================================
# Phase 3 Security: Sanitization Function Tests
# =============================================================================


class TestSanitizeString:
    """Tests for sanitize_string function."""

    def test_sanitize_string_normal_text(self):
        """Test sanitizing normal text."""
        result = sanitize_string("Hello, world!")
        assert result == "Hello, world!"

    def test_sanitize_string_with_newlines(self):
        """Test that newlines are preserved."""
        text = "Line 1\nLine 2\nLine 3"
        result = sanitize_string(text)
        assert result == text
        assert "\n" in result

    def test_sanitize_string_with_tabs(self):
        """Test that tabs are preserved."""
        text = "Col1\tCol2\tCol3"
        result = sanitize_string(text)
        assert result == text
        assert "\t" in result

    def test_sanitize_string_removes_null_bytes(self):
        """Test that null bytes are removed."""
        text = "Hello\x00World"
        result = sanitize_string(text)
        assert result == "HelloWorld"
        assert "\x00" not in result

    def test_sanitize_string_removes_control_chars(self):
        """Test that dangerous control characters are removed."""
        text = "Hello\x01\x02\x03World"
        result = sanitize_string(text)
        assert result == "HelloWorld"

    def test_sanitize_string_preserves_carriage_return(self):
        """Test that carriage return is preserved."""
        text = "Line 1\r\nLine 2"
        result = sanitize_string(text)
        assert "\r" in result

    def test_sanitize_string_exceeds_max_length(self):
        """Test that exceeding max length raises error."""
        long_text = "a" * 10001
        with pytest.raises(ValidationError) as exc_info:
            sanitize_string(long_text, max_length=10000)
        assert exc_info.value.field == "content"
        assert "exceeds max length" in exc_info.value.message.lower()

    def test_sanitize_string_custom_field_name(self):
        """Test sanitize with custom field name."""
        with pytest.raises(ValidationError) as exc_info:
            sanitize_string(123, field="description")
        assert exc_info.value.field == "description"

    def test_sanitize_string_non_string_type(self):
        """Test that non-string raises error."""
        with pytest.raises(ValidationError) as exc_info:
            sanitize_string(123)
        assert "expected string" in exc_info.value.message.lower()

    def test_sanitize_string_empty_allowed(self):
        """Test that empty string is allowed."""
        result = sanitize_string("")
        assert result == ""

    def test_sanitize_string_unicode(self):
        """Test sanitizing unicode text."""
        text = "Hello 世界 🌍"
        result = sanitize_string(text)
        assert result == text


class TestSanitizeIdentifier:
    """Tests for sanitize_identifier function."""

    def test_sanitize_identifier_valid_alphanumeric(self):
        """Test valid alphanumeric identifier."""
        result = sanitize_identifier("user123")
        assert result == "user123"

    def test_sanitize_identifier_with_underscore(self):
        """Test identifier with underscores."""
        result = sanitize_identifier("my_variable_name")
        assert result == "my_variable_name"

    def test_sanitize_identifier_with_hyphen(self):
        """Test identifier with hyphens."""
        result = sanitize_identifier("my-session-id")
        assert result == "my-session-id"

    def test_sanitize_identifier_mixed_valid_chars(self):
        """Test identifier with mixed valid characters."""
        result = sanitize_identifier("user-123_ABC")
        assert result == "user-123_ABC"

    def test_sanitize_identifier_empty_string(self):
        """Test that empty string raises error."""
        with pytest.raises(ValidationError) as exc_info:
            sanitize_identifier("")
        assert "cannot be empty" in exc_info.value.message.lower()

    def test_sanitize_identifier_too_long(self):
        """Test that string over 100 chars raises error."""
        long_id = "a" * 101
        with pytest.raises(ValidationError) as exc_info:
            sanitize_identifier(long_id)
        assert "exceeds max length of 100" in exc_info.value.message.lower()

    def test_sanitize_identifier_with_spaces(self):
        """Test that spaces are rejected."""
        with pytest.raises(ValidationError) as exc_info:
            sanitize_identifier("my identifier")
        assert "only alphanumeric" in exc_info.value.message.lower()

    def test_sanitize_identifier_with_special_chars(self):
        """Test that special characters are rejected."""
        with pytest.raises(ValidationError) as exc_info:
            sanitize_identifier("user@123")
        assert "only alphanumeric" in exc_info.value.message.lower()

    def test_sanitize_identifier_with_dots(self):
        """Test that dots are rejected."""
        with pytest.raises(ValidationError):
            sanitize_identifier("user.name")

    def test_sanitize_identifier_non_string(self):
        """Test that non-string raises error."""
        with pytest.raises(ValidationError) as exc_info:
            sanitize_identifier(123)
        assert "expected string" in exc_info.value.message.lower()


class TestSanitizeSessionId:
    """Tests for sanitize_session_id function."""

    def test_sanitize_session_id_valid(self):
        """Test valid session ID."""
        result = sanitize_session_id("session-123")
        assert result == "session-123"

    def test_sanitize_session_id_invalid(self):
        """Test sanitizing session ID with invalid characters."""
        # sanitize_session_id cleans rather than raises - it removes invalid chars
        result = sanitize_session_id("session@123")
        assert result == "session123"  # @ is removed


class TestValidateLimit:
    """Tests for validate_limit function."""

    def test_validate_limit_valid_value(self):
        """Test valid limit value."""
        result = validate_limit(50)
        assert result == 50

    def test_validate_limit_caps_at_max(self):
        """Test that limit is capped at max_limit."""
        result = validate_limit(200, max_limit=100)
        assert result == 100

    def test_validate_limit_at_max_boundary(self):
        """Test limit at max boundary."""
        result = validate_limit(100, max_limit=100)
        assert result == 100

    def test_validate_limit_below_one(self):
        """Test that limit < 1 raises error."""
        with pytest.raises(ValidationError) as exc_info:
            validate_limit(0)
        assert "must be at least 1" in exc_info.value.message.lower()

    def test_validate_limit_negative(self):
        """Test that negative limit raises error."""
        with pytest.raises(ValidationError) as exc_info:
            validate_limit(-5)
        assert "must be at least 1" in exc_info.value.message.lower()

    def test_validate_limit_string_number(self):
        """Test that string number is converted."""
        result = validate_limit("42")
        assert result == 42
        assert isinstance(result, int)

    def test_validate_limit_non_numeric_string(self):
        """Test that non-numeric string raises error."""
        with pytest.raises(ValidationError) as exc_info:
            validate_limit("abc")
        assert "must be an integer" in exc_info.value.message.lower()

    def test_validate_limit_float(self):
        """Test that float is converted to int."""
        result = validate_limit(50.7)
        assert result == 50
        assert isinstance(result, int)

    def test_validate_limit_custom_field_name(self):
        """Test with custom field name."""
        with pytest.raises(ValidationError) as exc_info:
            validate_limit(0, field="page_size")
        assert exc_info.value.field == "page_size"


class TestValidateFloatRange:
    """Tests for validate_float_range function."""

    def test_validate_float_range_valid(self):
        """Test valid float in range."""
        result = validate_float_range(0.5)
        assert result == 0.5

    def test_validate_float_range_min_boundary(self):
        """Test float at min boundary."""
        result = validate_float_range(0.0, min_val=0.0, max_val=1.0)
        assert result == 0.0

    def test_validate_float_range_max_boundary(self):
        """Test float at max boundary."""
        result = validate_float_range(1.0, min_val=0.0, max_val=1.0)
        assert result == 1.0

    def test_validate_float_range_below_min(self):
        """Test that value below min raises error."""
        with pytest.raises(ValidationError) as exc_info:
            validate_float_range(-0.1, min_val=0.0, max_val=1.0)
        assert "must be between" in exc_info.value.message.lower()

    def test_validate_float_range_above_max(self):
        """Test that value above max raises error."""
        with pytest.raises(ValidationError) as exc_info:
            validate_float_range(1.5, min_val=0.0, max_val=1.0)
        assert "must be between" in exc_info.value.message.lower()

    def test_validate_float_range_custom_range(self):
        """Test with custom min/max values."""
        result = validate_float_range(50.0, min_val=0.0, max_val=100.0)
        assert result == 50.0

    def test_validate_float_range_int_value(self):
        """Test that int is converted to float."""
        result = validate_float_range(1, min_val=0.0, max_val=10.0)
        assert result == 1.0
        assert isinstance(result, float)

    def test_validate_float_range_string_number(self):
        """Test that string number is converted."""
        result = validate_float_range("0.75")
        assert result == 0.75

    def test_validate_float_range_non_numeric(self):
        """Test that non-numeric raises error."""
        with pytest.raises(ValidationError) as exc_info:
            validate_float_range("abc")
        assert "must be a number" in exc_info.value.message.lower()


class TestValidateMetadata:
    """Tests for validate_metadata function."""

    def test_validate_metadata_none(self):
        """Test that None returns empty dict."""
        result = validate_metadata(None)
        assert result == {}

    def test_validate_metadata_empty_dict(self):
        """Test empty metadata dict."""
        result = validate_metadata({})
        assert result == {}

    def test_validate_metadata_simple_strings(self):
        """Test metadata with simple string values."""
        metadata = {"key1": "value1", "key2": "value2"}
        result = validate_metadata(metadata)
        assert result == metadata

    def test_validate_metadata_mixed_types(self):
        """Test metadata with mixed types."""
        metadata = {
            "string": "value",
            "int": 42,
            "float": 3.14,
            "bool": True,
        }
        result = validate_metadata(metadata)
        assert result == metadata

    def test_validate_metadata_nested_dict(self):
        """Test nested metadata dicts."""
        metadata = {
            "level1": {
                "level2": {
                    "value": "deep"
                }
            }
        }
        result = validate_metadata(metadata)
        assert result == metadata

    def test_validate_metadata_with_list(self):
        """Test metadata with list values."""
        metadata = {"tags": ["tag1", "tag2", "tag3"]}
        result = validate_metadata(metadata)
        assert result == metadata

    def test_validate_metadata_list_capped_at_100(self):
        """Test that lists are capped at 100 items."""
        metadata = {"items": list(range(150))}
        result = validate_metadata(metadata)
        assert len(result["items"]) == 100

    def test_validate_metadata_sanitizes_string_values(self):
        """Test that string values are sanitized."""
        metadata = {"text": "Hello\x00World"}
        result = validate_metadata(metadata)
        assert result["text"] == "HelloWorld"

    def test_validate_metadata_max_depth_exceeded(self):
        """Test that exceeding max depth raises error."""
        # Create deeply nested dict
        metadata = {"l1": {"l2": {"l3": {"l4": {"l5": {"l6": "too deep"}}}}}}
        with pytest.raises(ValidationError) as exc_info:
            validate_metadata(metadata, max_depth=5)
        assert "exceeds maximum nesting depth" in exc_info.value.message.lower()

    def test_validate_metadata_invalid_type(self):
        """Test that non-dict raises error."""
        with pytest.raises(ValidationError) as exc_info:
            validate_metadata("not a dict")
        assert "must be a dict" in exc_info.value.message.lower()

    def test_validate_metadata_invalid_value_type(self):
        """Test that invalid value type raises error."""
        metadata = {"key": object()}
        with pytest.raises(ValidationError) as exc_info:
            validate_metadata(metadata)
        assert "invalid type" in exc_info.value.message.lower()

    def test_validate_metadata_null_value(self):
        """Test that None values are preserved."""
        metadata = {"key": None}
        result = validate_metadata(metadata)
        assert result["key"] is None

    def test_validate_metadata_long_string_truncated(self):
        """Test that long strings over 1000 chars raise error."""
        metadata = {"text": "a" * 1001}
        with pytest.raises(ValidationError) as exc_info:
            validate_metadata(metadata)
        assert "exceeds max length" in exc_info.value.message.lower()

    def test_validate_metadata_sanitizes_keys(self):
        """Test that keys are sanitized."""
        metadata = {"valid-key_123": "value"}
        result = validate_metadata(metadata)
        assert "valid-key_123" in result


# =============================================================================
# Array and Embedding Validation Tests (NCA layer boundaries)
# =============================================================================


class TestValidateArray:
    """Tests for validate_array function."""

    def test_validate_array_valid_float32(self):
        """Test validating valid float32 array."""
        arr = np.ones(128, dtype=np.float32)
        validate_array(arr, expected_dim=128, name="embedding")  # Should not raise

    def test_validate_array_valid_float64(self):
        """Test validating valid float64 array."""
        arr = np.ones(128, dtype=np.float64)
        validate_array(arr, expected_dim=128, name="embedding")  # Should not raise

    def test_validate_array_none(self):
        """Test that None raises ValidationError."""
        with pytest.raises(ValidationError) as exc_info:
            validate_array(None, expected_dim=128, name="embedding")
        assert "cannot be none" in exc_info.value.message.lower()

    def test_validate_array_not_numpy(self):
        """Test that non-numpy array raises ValidationError."""
        with pytest.raises(ValidationError) as exc_info:
            validate_array([1, 2, 3], expected_dim=3, name="embedding")
        assert "expected numpy array" in exc_info.value.message.lower()

    def test_validate_array_wrong_shape(self):
        """Test that wrong shape raises ValidationError."""
        arr = np.ones((128, 2), dtype=np.float32)  # 2D instead of 1D
        with pytest.raises(ValidationError) as exc_info:
            validate_array(arr, expected_dim=128, name="embedding")
        assert "expected shape" in exc_info.value.message.lower()

    def test_validate_array_wrong_dim(self):
        """Test that wrong dimension raises ValidationError."""
        arr = np.ones(64, dtype=np.float32)
        with pytest.raises(ValidationError) as exc_info:
            validate_array(arr, expected_dim=128, name="embedding")
        assert "expected shape (128,)" in exc_info.value.message.lower()

    def test_validate_array_wrong_dtype(self):
        """Test that non-floating dtype raises ValidationError."""
        arr = np.ones(128, dtype=np.int32)
        with pytest.raises(ValidationError) as exc_info:
            validate_array(arr, expected_dim=128, name="embedding")
        assert "expected floating dtype" in exc_info.value.message.lower()

    def test_validate_array_contains_nan(self):
        """Test that NaN values raise ValidationError."""
        arr = np.ones(128, dtype=np.float32)
        arr[0] = np.nan
        with pytest.raises(ValidationError) as exc_info:
            validate_array(arr, expected_dim=128, name="embedding")
        assert "nan or inf" in exc_info.value.message.lower()

    def test_validate_array_contains_inf(self):
        """Test that Inf values raise ValidationError."""
        arr = np.ones(128, dtype=np.float32)
        arr[0] = np.inf
        with pytest.raises(ValidationError) as exc_info:
            validate_array(arr, expected_dim=128, name="embedding")
        assert "nan or inf" in exc_info.value.message.lower()

    def test_validate_array_zero_vector_not_allowed(self):
        """Test that zero vector raises error when not allowed."""
        arr = np.zeros(128, dtype=np.float32)
        with pytest.raises(ValidationError) as exc_info:
            validate_array(arr, expected_dim=128, name="embedding", allow_zero=False)
        assert "zero-vector not allowed" in exc_info.value.message.lower()

    def test_validate_array_zero_vector_allowed(self):
        """Test that zero vector passes when allowed."""
        arr = np.zeros(128, dtype=np.float32)
        validate_array(arr, expected_dim=128, name="embedding", allow_zero=True)  # Should not raise


class TestValidateEmbedding:
    """Tests for validate_embedding function."""

    def test_validate_embedding_valid(self):
        """Test validating valid embedding."""
        emb = np.random.default_rng(42).random(128).astype(np.float32)
        validate_embedding(emb, dim=128, name="embedding")  # Should not raise

    def test_validate_embedding_zero_not_allowed(self):
        """Test that zero embedding raises error."""
        emb = np.zeros(128, dtype=np.float32)
        with pytest.raises(ValidationError) as exc_info:
            validate_embedding(emb, dim=128, name="embedding")
        assert "zero-vector not allowed" in exc_info.value.message.lower()


class TestValidateTimestamp:
    """Tests for validate_timestamp function."""

    def test_validate_timestamp_valid_float(self):
        """Test validating valid float timestamp."""
        now = datetime.now(timezone.utc).timestamp()
        validate_timestamp(now)  # Should not raise

    def test_validate_timestamp_valid_datetime(self):
        """Test validating valid datetime."""
        now = datetime.now(timezone.utc)
        validate_timestamp(now)  # Should not raise

    def test_validate_timestamp_none(self):
        """Test that None raises ValidationError."""
        with pytest.raises(ValidationError) as exc_info:
            validate_timestamp(None)
        assert "cannot be none" in exc_info.value.message.lower()

    def test_validate_timestamp_future(self):
        """Test that future timestamp raises error."""
        future = datetime.now(timezone.utc).timestamp() + 100  # 100s in future
        with pytest.raises(ValidationError) as exc_info:
            validate_timestamp(future)
        assert "future" in exc_info.value.message.lower()

    def test_validate_timestamp_far_past(self):
        """Test that very old timestamp raises error."""
        past = datetime.now(timezone.utc).timestamp() - (86400 * 400)  # 400 days ago
        with pytest.raises(ValidationError) as exc_info:
            validate_timestamp(past)
        assert "past" in exc_info.value.message.lower()

    def test_validate_timestamp_not_finite(self):
        """Test that non-finite timestamp raises error."""
        with pytest.raises(ValidationError) as exc_info:
            validate_timestamp(np.inf)
        assert "not finite" in exc_info.value.message.lower()

    def test_validate_timestamp_invalid_type(self):
        """Test that invalid type raises error."""
        with pytest.raises(ValidationError) as exc_info:
            validate_timestamp("not-a-timestamp")
        assert "expected float or datetime" in exc_info.value.message.lower()

    def test_validate_timestamp_within_tolerance(self):
        """Test timestamp within 5s future tolerance."""
        future = datetime.now(timezone.utc).timestamp() + 2  # 2s in future
        validate_timestamp(future)  # Should not raise


class TestValidateNTLevel:
    """Tests for validate_nt_level function."""

    def test_validate_nt_level_valid(self):
        """Test validating valid NT level."""
        result = validate_nt_level(0.5, name="da")
        assert result == 0.5

    def test_validate_nt_level_clamps_high(self):
        """Test that high level is clamped."""
        result = validate_nt_level(1.5, name="da")
        assert result == 0.95

    def test_validate_nt_level_clamps_low(self):
        """Test that low level is clamped."""
        result = validate_nt_level(-0.5, name="da")
        assert result == 0.05

    def test_validate_nt_level_at_floor(self):
        """Test level at floor boundary."""
        result = validate_nt_level(0.05, name="da")
        assert result == 0.05

    def test_validate_nt_level_at_ceiling(self):
        """Test level at ceiling boundary."""
        result = validate_nt_level(0.95, name="da")
        assert result == 0.95

    def test_validate_nt_level_not_finite(self):
        """Test that non-finite level raises error."""
        with pytest.raises(ValidationError) as exc_info:
            validate_nt_level(np.nan, name="da")
        assert "not finite" in exc_info.value.message.lower()

    def test_validate_nt_level_invalid_type(self):
        """Test that invalid type raises error."""
        with pytest.raises(ValidationError) as exc_info:
            validate_nt_level("not-a-number", name="da")
        assert "expected number" in exc_info.value.message.lower()


class TestValidateMemoryId:
    """Tests for validate_memory_id function."""

    def test_validate_memory_id_valid_string(self):
        """Test validating valid UUID string."""
        mid = str(uuid4())
        result = validate_memory_id(mid)
        assert result == mid

    def test_validate_memory_id_valid_uuid(self):
        """Test validating UUID object."""
        mid = uuid4()
        result = validate_memory_id(mid)
        assert result == str(mid)

    def test_validate_memory_id_none(self):
        """Test that None raises ValidationError."""
        with pytest.raises(ValidationError) as exc_info:
            validate_memory_id(None)
        assert "cannot be none" in exc_info.value.message.lower()

    def test_validate_memory_id_invalid(self):
        """Test that invalid UUID raises error."""
        with pytest.raises(ValidationError) as exc_info:
            validate_memory_id("not-a-uuid")
        assert "invalid uuid" in exc_info.value.message.lower()


class TestValidateSpikeInterval:
    """Tests for validate_spike_interval function."""

    def test_validate_spike_interval_valid(self):
        """Test validating valid spike interval."""
        prev_ts = 1.0
        curr_ts = 1.01  # 10ms later
        validate_spike_interval(prev_ts, curr_ts)  # Should not raise

    def test_validate_spike_interval_too_short(self):
        """Test that too short interval raises error."""
        prev_ts = 1.0
        curr_ts = 1.0005  # 0.5ms later
        with pytest.raises(ValidationError) as exc_info:
            validate_spike_interval(prev_ts, curr_ts, min_isi_ms=1.0)
        assert "inter-spike interval" in exc_info.value.message.lower()

    def test_validate_spike_interval_at_minimum(self):
        """Test interval exactly at minimum."""
        prev_ts = 1.0
        curr_ts = 1.001  # Exactly 1ms later
        validate_spike_interval(prev_ts, curr_ts, min_isi_ms=1.0)  # Should not raise

    def test_validate_spike_interval_custom_minimum(self):
        """Test with custom minimum ISI."""
        prev_ts = 1.0
        curr_ts = 1.002  # 2ms later
        with pytest.raises(ValidationError):
            validate_spike_interval(prev_ts, curr_ts, min_isi_ms=5.0)


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
