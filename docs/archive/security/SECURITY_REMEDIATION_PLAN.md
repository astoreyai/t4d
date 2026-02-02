# World Weaver Security Remediation Plan

**Generated**: 2025-12-06
**Target Completion**: Before production deployment

---

## Priority 1: Critical for Production (Complete Before Launch)

### 1. Implement API Authentication (H-2, M-4)
**Effort**: 4 hours | **Risk**: HIGH

```python
# Add to src/t4dm/core/config.py
api_key: Optional[str] = Field(default=None, description="API key for auth")

@field_validator("api_key")
@classmethod
def validate_api_key_production(cls, v: Optional[str]) -> Optional[str]:
    env = os.getenv("WW_ENVIRONMENT", "development")
    if env == "production" and not v:
        raise ValueError("API key required in production (set WW_API_KEY)")
    if v and len(v) < 32:
        raise ValueError("API key must be at least 32 characters")
    return v
```

```python
# Add to src/t4dm/api/deps.py
async def verify_api_key(x_api_key: Annotated[Optional[str], Header()] = None):
    settings = get_settings()
    if settings.api_key:
        if not x_api_key or not secrets.compare_digest(x_api_key, settings.api_key):
            raise HTTPException(401, detail="Invalid or missing API key")
```

**Files to modify**:
- `src/t4dm/core/config.py` (add field + validator)
- `src/t4dm/api/deps.py` (add verify_api_key dependency)
- `src/t4dm/api/server.py` (add to router dependencies)
- `.env.example` (add WW_API_KEY with generation instructions)

**Testing**:
```bash
# Generate API key
openssl rand -hex 32

# Test authentication
curl -H "X-API-Key: invalid" http://localhost:8765/api/v1/health  # Should 401
curl -H "X-API-Key: <valid>" http://localhost:8765/api/v1/health  # Should 200
```

---

### 2. Sanitize API Error Messages (H-2)
**Effort**: 3 hours | **Risk**: HIGH

**Create error utilities** (`src/t4dm/api/errors.py`):
```python
from ww.mcp.gateway import get_request_id
import logging

logger = logging.getLogger(__name__)

def handle_api_error(e: Exception, operation: str) -> HTTPException:
    """Sanitize and log errors, return safe HTTP exception."""
    error_id = get_request_id()
    logger.error(f"[{error_id}] {operation} failed: {e}", exc_info=True)

    return HTTPException(
        status_code=500,
        detail=f"Operation failed (ref: {error_id}). Contact support if issue persists."
    )
```

**Apply to all routes**:
```python
# Before (INSECURE):
except Exception as e:
    logger.error(f"Failed to create episode: {e}")
    raise HTTPException(500, detail=f"Failed to create episode: {str(e)}")

# After (SECURE):
except HTTPException:
    raise  # Re-raise validation errors
except Exception as e:
    raise handle_api_error(e, "create episode")
```

**Files to modify**:
- `src/t4dm/api/errors.py` (new file)
- `src/t4dm/api/routes/episodes.py` (6 locations)
- `src/t4dm/api/routes/entities.py` (7 locations)
- `src/t4dm/api/routes/skills.py` (5 locations)
- `src/t4dm/api/routes/visualization.py` (3 locations)
- `src/t4dm/api/routes/system.py` (2 locations)

---

### 3. Enforce Production TLS (H-3)
**Effort**: 2 hours | **Risk**: HIGH

**Add production validator** (`src/t4dm/core/config.py`):
```python
@model_validator(mode="after")
def validate_production_tls(self) -> "Settings":
    """Enforce TLS in production."""
    env = os.getenv("WW_ENVIRONMENT", "development")

    if env == "production":
        issues = []

        # Check Neo4j uses TLS
        if self.neo4j_uri.startswith("bolt://"):
            issues.append(
                "Neo4j must use bolt+s:// or neo4j+s:// in production"
            )

        # Check OTLP uses TLS
        if self.otel_enabled and self.otel_insecure:
            issues.append("OTLP must use TLS in production")

        if issues:
            raise ValueError(
                f"Production TLS validation failed:\n  - " + "\n  - ".join(issues)
            )

    return self
```

**Create production docker-compose** (`docker-compose.production.yml`):
```yaml
services:
  neo4j:
    environment:
      - NEO4J_server_bolt_tls__level=REQUIRED
      - NEO4J_server_https_enabled=true
      - NEO4J_dbms_ssl_policy_bolt_enabled=true
    volumes:
      - ./certs/neo4j:/certificates:ro
```

**Files to modify**:
- `src/t4dm/core/config.py` (add validator)
- `docker-compose.production.yml` (new file)
- `docs/DEPLOYMENT.md` (add TLS setup instructions)

---

### 4. Fix CORS Configuration (H-1)
**Effort**: 1 hour | **Risk**: HIGH

**Add CORS validator** (`src/t4dm/core/config.py`):
```python
@field_validator("cors_allowed_origins")
@classmethod
def validate_cors_origins(cls, v: list[str]) -> list[str]:
    env = os.getenv("WW_ENVIRONMENT", "development")

    if env == "production":
        if "*" in v:
            raise ValueError(
                "CORS wildcard (*) not allowed in production. "
                "Specify exact origins (e.g., https://app.example.com)"
            )

        # Warn about non-HTTPS
        for origin in v:
            if not origin.startswith("https://") and not origin.startswith("http://localhost"):
                logger.warning(f"Non-HTTPS origin in production: {origin}")

    return v
```

**Update .env.example**:
```bash
# CORS Configuration (API Server)
# Development: localhost origins
WW_API_CORS_ORIGINS=http://localhost:3000,http://localhost:8080

# Production: ONLY HTTPS origins, comma-separated
# WW_API_CORS_ORIGINS=https://app.example.com,https://dashboard.example.com
```

**Files to modify**:
- `src/t4dm/core/config.py` (add validator)
- `.env.example` (update with clear guidance)

---

## Priority 2: High Impact Security Improvements

### 5. Strengthen Password Validation (M-1)
**Effort**: 2 hours | **Risk**: MEDIUM

```python
def validate_password_strength(password: str, field_name: str = "password") -> str:
    """Enhanced password validation with pattern detection."""
    if len(password) < 12:  # Increased from 8
        raise ValueError(f"{field_name} must be at least 12 characters")

    # Check common weak patterns
    weak_patterns = [
        (r'^[A-Z][a-z]+\d+$', "Capital + word + digits"),
        (r'^\d+[A-Z][a-z]+$', "Digits + capital + word"),
        (r'^[A-Za-z]+\d{1,4}$', "Letters + 1-4 digits"),
    ]

    for pattern, description in weak_patterns:
        if re.match(pattern, password):
            raise ValueError(
                f"{field_name} follows weak pattern: {description}. "
                f"Use a passphrase or password manager."
            )

    # Require 3 of 4 character classes
    complexity = sum([
        bool(re.search(r'[a-z]', password)),
        bool(re.search(r'[A-Z]', password)),
        bool(re.search(r'[0-9]', password)),
        bool(re.search(r'[^a-zA-Z0-9]', password)),
    ])

    if complexity < 3:
        raise ValueError(
            f"{field_name} needs more complexity. Include at least 3 of: "
            f"uppercase, lowercase, digits, special characters"
        )

    return password
```

**Files to modify**:
- `src/t4dm/core/config.py:32-80` (replace function)

---

### 6. Block Reserved Session IDs (M-2)
**Effort**: 30 minutes | **Risk**: MEDIUM

```python
# In src/t4dm/api/deps.py
validated = validate_session_id(
    x_session_id,
    allow_none=True,
    allow_reserved=False,  # Changed from True
)
```

**Files to modify**:
- `src/t4dm/api/deps.py:42` (change allow_reserved)

---

### 7. Add Security Headers Middleware (M-6)
**Effort**: 1 hour | **Risk**: MEDIUM

```python
# src/t4dm/api/middleware.py (new file)
from starlette.middleware.base import BaseHTTPMiddleware

class SecurityHeadersMiddleware(BaseHTTPMiddleware):
    async def dispatch(self, request, call_next):
        response = await call_next(request)
        response.headers["X-Content-Type-Options"] = "nosniff"
        response.headers["X-Frame-Options"] = "DENY"
        response.headers["X-XSS-Protection"] = "1; mode=block"
        response.headers["Strict-Transport-Security"] = "max-age=31536000"
        response.headers["Content-Security-Policy"] = (
            "default-src 'self'; "
            "script-src 'self' 'unsafe-inline' https://cdn.jsdelivr.net; "
            "style-src 'self' 'unsafe-inline' https://cdn.jsdelivr.net; "
            "img-src 'self' data: https:;"
        )
        return response
```

```python
# In src/t4dm/api/server.py
from ww.api.middleware import SecurityHeadersMiddleware
app.add_middleware(SecurityHeadersMiddleware)
```

**Files to modify**:
- `src/t4dm/api/middleware.py` (new file)
- `src/t4dm/api/server.py` (add middleware)

---

### 8. Document Rate Limiting with Multiple Workers (M-3)
**Effort**: 30 minutes | **Risk**: MEDIUM (document now, fix later with Redis)

**Update .env.example**:
```bash
# API Server Configuration
# WARNING: Rate limiting (100 req/min) is per-worker
# With 4 workers, effective limit is 400 req/min
# For production with multiple workers, use external rate limiter (nginx, Redis)
WW_API_WORKERS=1
```

**Files to modify**:
- `.env.example` (add warning)
- `README.md` (document limitation)

**Future enhancement** (Priority 3):
- Implement Redis-based distributed rate limiter

---

## Priority 3: Defense in Depth Improvements

### 9. Run Docker Containers as Non-Root (L-6)
**Effort**: 1 hour | **Risk**: LOW

```dockerfile
FROM python:3.11-slim

# Create non-root user
RUN groupadd -r ww && useradd -r -g ww -m ww

# Install system dependencies
RUN apt-get update && apt-get install -y --no-install-recommends \
    gcc \
    && rm -rf /var/lib/apt/lists/*

# Install Python dependencies as root
COPY requirements.txt /tmp/
RUN pip install --no-cache-dir -r /tmp/requirements.txt

# Switch to non-root user
USER ww
WORKDIR /home/t4dm/app

# Copy application
COPY --chown=ww:ww . .

CMD ["python", "-m", "ww.api.server"]
```

**Files to modify**:
- `Dockerfile` (add USER directive)

---

### 10. Add Request Size Limits (L-7)
**Effort**: 1 hour | **Risk**: LOW

```python
# src/t4dm/api/middleware.py
class RequestSizeLimitMiddleware(BaseHTTPMiddleware):
    def __init__(self, app, max_body_size: int = 5 * 1024 * 1024):
        super().__init__(app)
        self.max_body_size = max_body_size

    async def dispatch(self, request, call_next):
        if "content-length" in request.headers:
            if int(request.headers["content-length"]) > self.max_body_size:
                return JSONResponse(
                    status_code=413,
                    content={"detail": "Request too large"},
                )
        return await call_next(request)
```

**Files to modify**:
- `src/t4dm/api/middleware.py` (add class)
- `src/t4dm/api/server.py` (add middleware)

---

### 11. Implement Audit Logging (L-5)
**Effort**: 2 hours | **Risk**: LOW

```python
# src/t4dm/observability/audit.py (new file)
import logging
from fastapi import Request

audit_logger = logging.getLogger("ww.audit")

def log_api_event(event: str, request: Request, **extra):
    """Log audit event with context."""
    audit_logger.info(
        event,
        extra={
            "ip": request.client.host,
            "user_agent": request.headers.get("user-agent"),
            "path": request.url.path,
            "session_id": request.headers.get("x-session-id"),
            **extra,
        }
    )
```

**Apply to sensitive operations**:
```python
# In create_episode
from ww.observability.audit import log_api_event
log_api_event("episode_created", request, episode_id=str(episode.id))

# In delete operations
log_api_event("episode_deleted", request, episode_id=str(episode_id))
```

**Files to modify**:
- `src/t4dm/observability/audit.py` (new file)
- All route files (add audit logging)

---

### 12. Enforce .env Permissions (L-1)
**Effort**: 30 minutes | **Risk**: LOW

```python
# In src/t4dm/core/config.py
def check_file_permissions(path: Path) -> None:
    """Check and enforce file permissions."""
    if not path.exists():
        return

    mode = path.stat().st_mode
    if mode & 0o077:
        env = os.getenv("WW_ENVIRONMENT", "development")
        if env == "production":
            raise ValueError(
                f"Config file '{path}' has insecure permissions: {oct(mode)}. "
                f"Run: chmod 600 {path}"
            )
        else:
            logger.warning(f"Config file '{path}' has permissive permissions. Run: chmod 600 {path}")
```

**Files to modify**:
- `src/t4dm/core/config.py:99-117` (update function)

---

## Testing Plan

### Security Test Suite

**Create** `tests/security/test_authentication.py`:
```python
import pytest
from fastapi.testclient import TestClient

def test_api_key_required(client: TestClient):
    """Test API key authentication enforcement."""
    # Without API key
    response = client.get("/api/v1/health")
    assert response.status_code == 401

    # With invalid API key
    response = client.get("/api/v1/health", headers={"X-API-Key": "invalid"})
    assert response.status_code == 401

    # With valid API key
    response = client.get("/api/v1/health", headers={"X-API-Key": settings.api_key})
    assert response.status_code == 200

def test_session_id_injection(client: TestClient):
    """Test session ID injection prevention."""
    malicious_ids = [
        "../../../etc/passwd",
        "'; DROP TABLE episodes; --",
        "<script>alert('xss')</script>",
        "admin",
        "system",
    ]

    for sid in malicious_ids:
        response = client.get("/api/v1/episodes", headers={"X-Session-ID": sid})
        assert response.status_code in [400, 401]

def test_xss_sanitization(client: TestClient, auth_headers: dict):
    """Test XSS pattern removal."""
    payloads = [
        "<script>alert('xss')</script>",
        "<img src=x onerror=alert(1)>",
        "javascript:alert(1)",
    ]

    for payload in payloads:
        response = client.post(
            "/api/v1/episodes",
            headers=auth_headers,
            json={"content": payload}
        )

        if response.status_code == 201:
            assert "<script>" not in response.json()["content"]
            assert "javascript:" not in response.json()["content"]

def test_rate_limiting(client: TestClient, auth_headers: dict):
    """Test rate limit enforcement."""
    for i in range(101):
        response = client.get("/api/v1/health", headers=auth_headers)
        if i < 100:
            assert response.status_code == 200
        else:
            assert response.status_code == 429
            assert "Retry-After" in response.headers
```

**Create** `tests/security/test_error_sanitization.py`:
```python
def test_error_messages_sanitized(client: TestClient, auth_headers: dict):
    """Test database errors don't leak sensitive info."""
    # Trigger database error (e.g., invalid UUID)
    response = client.get("/api/v1/episodes/invalid-uuid", headers=auth_headers)

    # Should get generic error, not database details
    detail = response.json().get("detail", "")
    assert "bolt://" not in detail
    assert "neo4j" not in detail.lower()
    assert "Connection" not in detail
    assert "ref:" in detail  # Should have error correlation ID
```

---

## Rollout Plan

### Phase 1: Critical Fixes (Week 1)
- [ ] Implement API authentication
- [ ] Sanitize error messages
- [ ] Enforce production TLS
- [ ] Fix CORS configuration
- [ ] Test all critical fixes

### Phase 2: Security Hardening (Week 2)
- [ ] Strengthen password validation
- [ ] Block reserved session IDs
- [ ] Add security headers
- [ ] Document rate limiting
- [ ] Run security test suite

### Phase 3: Defense in Depth (Week 3)
- [ ] Non-root containers
- [ ] Request size limits
- [ ] Audit logging
- [ ] Enforce .env permissions
- [ ] Update deployment docs

### Phase 4: Production Readiness (Week 4)
- [ ] Full security regression testing
- [ ] Penetration testing (optional)
- [ ] Security documentation review
- [ ] Incident response plan
- [ ] Production deployment checklist

---

## Success Metrics

### Before Remediation
- **CORS**: Wildcard allowed
- **Error Messages**: Full stack traces exposed
- **TLS**: Optional
- **Authentication**: None
- **Security Rating**: B+

### After Remediation
- **CORS**: Explicit origins only, HTTPS in prod
- **Error Messages**: Sanitized with correlation IDs
- **TLS**: Enforced in production
- **Authentication**: API key required
- **Security Rating**: A

---

## Resources Required

### Development Time
- **Phase 1**: 10 hours (1 developer, 2 days)
- **Phase 2**: 6 hours (1 developer, 1 day)
- **Phase 3**: 5 hours (1 developer, 1 day)
- **Phase 4**: 8 hours (testing + docs)

**Total**: 29 hours (~4 developer days)

### Infrastructure
- **Redis** (for distributed rate limiting - Phase 3 future enhancement)
- **TLS Certificates** (for Neo4j, OTLP - production only)

### Documentation Updates
- `README.md`
- `docs/DEPLOYMENT.md`
- `docs/SECURITY.md` (new)
- `.env.example`

---

**Next Steps**:
1. Review and approve this remediation plan
2. Create GitHub issues for each priority 1 item
3. Schedule Phase 1 implementation sprint
4. Set up security testing environment

**Contact**: Security team for questions or assistance
