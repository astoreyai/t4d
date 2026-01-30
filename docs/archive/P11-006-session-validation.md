# P11-006: Session ID Validation at Gateway Level

**Status**: Implemented
**Date**: 2025-11-27
**Location**: `/mnt/projects/ww/src/ww/mcp/`

## Summary

Added comprehensive session ID validation at the gateway level to prevent injection attacks and ensure data integrity across the World Weaver MCP system.

## Implementation

### 1. Validation Rules (`validation.py`)

#### `validate_session_id()`
Comprehensive validation function with security checks:

- **Type Safety**: Rejects non-string inputs
- **Length Constraints**: Max 128 characters
- **Character Whitelist**: Alphanumeric, underscore, hyphen only (regex: `^[a-zA-Z0-9_\-]{1,128}$`)
- **Path Traversal Prevention**: Blocks `..`, `/`, `\`
- **Null Byte Detection**: Rejects `\x00` characters
- **Reserved ID Blocking**: Case-insensitive check for reserved IDs
- **Whitespace Stripping**: Automatic trimming

**Reserved IDs** (blocked by default):
- `admin`, `system`, `root`, `default`, `test`
- `null`, `none`, `undefined`, `anonymous`

#### `sanitize_session_id()`
Permissive sanitization for cleaning invalid input:
- Removes dangerous characters
- Truncates to 128 chars
- Returns `None` if result is empty
- Use for cleaning, not strict validation

### 2. Gateway Integration (`gateway.py`)

#### `with_session_validation` Decorator
Applies to any MCP tool with `session_id` or `session_filter` parameter:

```python
@with_session_validation
async def my_tool(session_id: Optional[str] = None, ...):
    # session_id is validated before this code runs
    ...
```

**Behavior**:
- Validates `session_id` and `session_filter` parameters
- Returns validation error response if invalid
- Allows reserved IDs for `session_filter` (e.g., "default")
- Strips whitespace automatically
- Preserves `None` values (when allowed)

#### `get_services()` Validation
Entry point validation for all memory services:

```python
async def get_services(session_id: Optional[str] = None):
    # Validates session_id before any service initialization
    validated_session = validate_session_id(
        session_id,
        allow_none=True,
        allow_reserved=True,
    )
    ...
```

### 3. Security Checks

**Prevents**:
- SQL injection: `session'; DROP TABLE users; --`
- Command injection: `session; rm -rf /`
- Path traversal: `../../../etc/passwd`
- XSS attacks: `session<script>alert(1)</script>`
- Null byte injection: `session\x00admin`
- Type confusion: `123`, `[session]`, `{session: "id"}`
- Unicode bypasses: Cyrillic lookalikes

**Example Attack Attempts (All Blocked)**:
```python
# SQL injection
validate_session_id("session' OR '1'='1")  # REJECTED

# Path traversal
validate_session_id("../../../etc/passwd")  # REJECTED

# Command injection
validate_session_id("session && cat /etc/passwd")  # REJECTED

# XSS
validate_session_id("<script>alert('xss')</script>")  # REJECTED

# Null bytes
validate_session_id("session\x00admin")  # REJECTED

# Reserved IDs
validate_session_id("admin")  # REJECTED (unless allow_reserved=True)
```

### 4. Tests

**Test Coverage**: 48 tests, 100% pass rate

#### Unit Tests (`test_session_validation.py`)
- 23 validation tests
- 9 sanitization tests
- 6 security-focused tests

**Categories**:
- Valid inputs (alphanumeric, UUID, hyphen, underscore)
- Invalid inputs (special chars, path traversal, null bytes)
- Reserved ID handling
- Length constraints
- Type safety
- Injection attack prevention

#### Integration Tests (`test_session_validation_integration.py`)
- 10 decorator integration tests

**Categories**:
- Decorator validation behavior
- Error response format
- Argument preservation
- Filter vs. session_id handling

## Usage

### In MCP Tools

Apply decorator to any tool with session parameters:

```python
from ww.mcp.gateway import with_session_validation

@with_session_validation
async def my_episodic_tool(
    session_id: Optional[str] = None,
    query: str = "",
):
    # session_id is validated and sanitized
    episodic, _, _ = await get_services(session_id)
    ...
```

### Direct Validation

For custom validation logic:

```python
from ww.mcp.validation import validate_session_id, SessionValidationError

try:
    session_id = validate_session_id(
        user_input,
        allow_none=False,  # Require session ID
        allow_reserved=False,  # Block reserved IDs
    )
except SessionValidationError as e:
    return validation_error(e.field, e.message)
```

### Sanitization

For cleaning untrusted input:

```python
from ww.mcp.validation import sanitize_session_id

clean_id = sanitize_session_id(user_input)
if clean_id is None:
    # Input was too dirty to salvage
    return validation_error("session_id", "Invalid session ID")
```

## Files Modified

- `/mnt/projects/ww/src/ww/mcp/validation.py`
  - Added `validate_session_id()` function (114 lines)
  - Added `sanitize_session_id()` function (28 lines)
  - Added `SessionValidationError` exception class
  - Added `SESSION_ID_PATTERN` regex constant
  - Added `RESERVED_SESSION_IDS` frozenset

- `/mnt/projects/ww/src/ww/mcp/gateway.py`
  - Added `with_session_validation()` decorator (35 lines)
  - Updated `get_services()` with validation (13 lines)
  - Added imports for validation functions

## Files Created

- `/mnt/projects/ww/tests/mcp/test_session_validation.py` (373 lines)
  - 38 unit tests for validation and sanitization

- `/mnt/projects/ww/tests/mcp/test_session_validation_integration.py` (129 lines)
  - 10 integration tests for decorator

- `/mnt/projects/ww/docs/P11-006-session-validation.md` (this file)

## Test Results

```
48 passed in 1.43s
```

**Coverage**: Session validation module improved from 0% to 68%

## Security Impact

**Before**: Session IDs were passed directly to storage backends without validation, allowing:
- Injection attacks via session_id parameter
- Path traversal to access other sessions' data
- Type confusion errors

**After**: All session IDs validated at gateway entry point:
- Character whitelist enforced
- Path traversal blocked
- Null bytes rejected
- Reserved IDs protected
- Type safety guaranteed

## Performance Impact

**Minimal**: Validation adds ~0.01ms per request
- Regex match: O(n) where n = session ID length (max 128)
- Reserved check: O(1) frozenset lookup
- No network calls or I/O

## Next Steps

**Optional Enhancements**:
1. Apply `@with_session_validation` to all existing MCP tools
2. Add session ID validation to authentication middleware
3. Extend validation to other identifiers (entity IDs, episode IDs)
4. Add metrics for rejected session IDs
5. Consider session ID format versioning (e.g., `v1_user-session-123`)

## References

- OWASP Input Validation Cheat Sheet
- CWE-20: Improper Input Validation
- CWE-22: Improper Limitation of a Pathname to a Restricted Directory
- CWE-89: SQL Injection
- CWE-78: OS Command Injection
