# Phase 3 Security - Quick Reference

## Input Sanitization Functions

### sanitize_string(value, max_length=10000, field="content")
```python
from t4dm.mcp.validation import sanitize_string

# Clean user input
content = sanitize_string(user_input, max_length=100000)

# What it does:
# ✓ Removes null bytes (\x00)
# ✓ Removes dangerous control chars (\x01-\x08, \x0b-\x0c, \x0e-\x1f)
# ✓ Preserves newlines, tabs, carriage returns
# ✓ Enforces max length
```

### sanitize_identifier(value, field="identifier")
```python
from t4dm.mcp.validation import sanitize_identifier

# Validate session IDs, names, labels
session_id = sanitize_identifier(raw_session_id)

# Allowed: a-z, A-Z, 0-9, _, -
# Max length: 100 chars
```

### validate_limit(value, max_limit=100, field="limit")
```python
from t4dm.mcp.validation import validate_limit

# Cap pagination/query limits
limit = validate_limit(user_limit, max_limit=100)

# Silently caps at max_limit
# Prevents resource exhaustion
```

### validate_metadata(metadata, field="metadata", max_depth=5)
```python
from t4dm.mcp.validation import validate_metadata

# Sanitize nested dicts
clean_meta = validate_metadata(user_metadata)

# Features:
# ✓ Sanitizes all string values
# ✓ Max nesting depth: 5
# ✓ Caps lists at 100 items
# ✓ Supports: str, int, float, bool, dict, list, None
```

## Authentication Context

### Setting Authentication
```python
from t4dm.mcp.memory_gateway import set_auth_context

# At request start (e.g., in middleware)
set_auth_context(
    session_id="session-123",
    user_id="user-456",
    roles=["user", "developer"]
)
```

### Protecting Tools
```python
from t4dm.mcp.memory_gateway import require_auth, require_role

@mcp_app.tool()
@require_auth
async def my_tool(data: str):
    """Only authenticated users can call this."""
    # Tool implementation
    pass

@mcp_app.tool()
@require_role("admin")
async def admin_tool(config: dict):
    """Only users with 'admin' role can call this."""
    # Admin operations
    pass
```

### Checking Context
```python
from t4dm.mcp.memory_gateway import get_auth_context

ctx = get_auth_context()

if ctx.get("authenticated"):
    user_id = ctx["user_id"]
    roles = ctx["roles"]
    
    if "developer" in roles:
        # Developer-specific logic
        pass
```

## Error Handling

All validation functions raise `ValidationError`:
```python
from t4dm.mcp.validation import ValidationError, sanitize_identifier

try:
    clean_id = sanitize_identifier(user_input)
except ValidationError as e:
    return {
        "error": "validation_error",
        "field": e.field,
        "message": e.message
    }
```

## Common Patterns

### MCP Tool with Full Validation
```python
from t4dm.mcp.validation import (
    sanitize_string,
    validate_limit,
    validate_metadata,
    ValidationError,
)
from t4dm.mcp.memory_gateway import require_auth

@mcp_app.tool()
@require_auth
async def create_item(
    title: str,
    description: str,
    metadata: dict,
    limit: int = 10
) -> dict:
    """Create item with full input validation."""
    try:
        # Sanitize all inputs
        title = sanitize_string(title, max_length=200, field="title")
        description = sanitize_string(description, max_length=10000, field="description")
        metadata = validate_metadata(metadata, field="metadata")
        limit = validate_limit(limit, max_limit=100, field="limit")
        
        # Process item...
        item = await create_item_in_db(title, description, metadata)
        
        return {
            "id": str(item.id),
            "title": item.title,
            "created": True
        }
    except ValidationError as e:
        return e.to_dict()
    except Exception as e:
        return {
            "error": "internal_error",
            "message": str(e)
        }
```

### Role-Based Access with Context
```python
from t4dm.mcp.memory_gateway import get_auth_context, require_auth

@mcp_app.tool()
@require_auth
async def flexible_tool(action: str) -> dict:
    """Tool with different behavior based on roles."""
    ctx = get_auth_context()
    
    # All authenticated users can read
    if action == "read":
        return await read_data()
    
    # Only developers can write
    if action == "write":
        if "developer" not in ctx.get("roles", []):
            return {
                "error": "forbidden",
                "message": "Developer role required for write"
            }
        return await write_data()
    
    # Only admins can delete
    if action == "delete":
        if "admin" not in ctx.get("roles", []):
            return {
                "error": "forbidden",
                "message": "Admin role required for delete"
            }
        return await delete_data()
```

## Testing

All functions are fully tested:
```bash
# Run validation tests
pytest tests/unit/test_validation.py -v

# Run specific test class
pytest tests/unit/test_validation.py::TestSanitizeString -v

# Test with coverage
pytest tests/unit/test_validation.py --cov=t4dm.mcp.validation
```
