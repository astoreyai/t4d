# World Weaver Security Checklist

Quick reference for security verification before deployment.

## Pre-Deployment Checklist

### Configuration Security

- [ ] **Database Credentials**
  - [ ] `WW_NEO4J_PASSWORD` set to strong password (not "password")
  - [ ] `WW_QDRANT_API_KEY` set for production Qdrant instances
  - [ ] No credentials in version control (.env in .gitignore)
  - [ ] Credentials rotated regularly (every 90 days)

- [ ] **Network Configuration**
  - [ ] Neo4j not exposed to public internet
  - [ ] Qdrant not exposed to public internet
  - [ ] TLS/SSL enabled for all database connections
  - [ ] Firewall rules restrict access to trusted IPs only

- [ ] **Session Management**
  - [ ] Session IDs are cryptographically random (not "default")
  - [ ] Session authentication implemented for multi-user deployments
  - [ ] Session timeout configured appropriately

### Code Security

- [ ] **Input Validation**
  - [ ] All MCP tool inputs validated via `validation.py`
  - [ ] Label/type whitelisting added to `neo4j_store.py`
  - [ ] Content length limits enforced (max 100KB per episode)
  - [ ] Enum validation for all type fields

- [ ] **Query Safety**
  - [ ] All Cypher queries use parameterization
  - [ ] No user input in f-string query construction
  - [ ] Query complexity limits enforced (max depth 10)
  - [ ] Query timeout configured (30s default)

- [ ] **Error Handling**
  - [ ] Error messages sanitized (no internal details exposed)
  - [ ] Stack traces logged but not returned to client
  - [ ] Generic error messages for authentication failures

### Operational Security

- [ ] **Monitoring**
  - [ ] Audit logging enabled for sensitive operations
  - [ ] Failed authentication attempts logged
  - [ ] Unusual query patterns detected
  - [ ] Resource usage monitored

- [ ] **Rate Limiting**
  - [ ] Per-session rate limits configured
  - [ ] Global rate limits for expensive operations
  - [ ] Backoff strategy for repeated failures

- [ ] **Data Protection**
  - [ ] Database backups encrypted
  - [ ] Backups stored securely (not in application directory)
  - [ ] Backup restoration tested
  - [ ] GDPR data export/deletion implemented if required

### Dependency Security

- [ ] **Package Management**
  - [ ] All dependencies pinned to exact versions
  - [ ] No packages with known critical CVEs
  - [ ] Vulnerability scanning in CI/CD pipeline
  - [ ] Dependencies reviewed quarterly

- [ ] **Model Security**
  - [ ] Embedding model downloaded from trusted source
  - [ ] Model cache directory permissions restricted (0700)
  - [ ] Model checksums verified

## Production Hardening

### Minimal Configuration

```bash
# .env.production
WW_SESSION_ID=$(openssl rand -hex 16)
WW_NEO4J_URI=bolt://neo4j.internal:7687
WW_NEO4J_USER=ww_app
WW_NEO4J_PASSWORD=$(openssl rand -base64 32)
WW_NEO4J_DATABASE=worldweaver_prod

WW_QDRANT_URL=https://qdrant.internal:6333
WW_QDRANT_API_KEY=$(openssl rand -base64 32)

WW_EMBEDDING_CACHE_DIR=/var/lib/world_weaver/models
```

### Docker Security

```dockerfile
# Run as non-root user
USER 1000:1000

# Read-only root filesystem
RUN chmod -R 755 /app && \
    chown -R 1000:1000 /app

# Drop capabilities
SECURITY_OPT="no-new-privileges"
CAP_DROP="ALL"
```

### Nginx Reverse Proxy

```nginx
# Rate limiting
limit_req_zone $binary_remote_addr zone=mcp:10m rate=10r/s;

server {
    listen 443 ssl http2;
    server_name ww.example.com;

    # TLS configuration
    ssl_certificate /etc/ssl/certs/ww.crt;
    ssl_certificate_key /etc/ssl/private/ww.key;
    ssl_protocols TLSv1.3;
    ssl_ciphers HIGH:!aNULL:!MD5;

    # Security headers
    add_header Strict-Transport-Security "max-age=31536000" always;
    add_header X-Content-Type-Options "nosniff" always;
    add_header X-Frame-Options "DENY" always;

    # Rate limiting
    limit_req zone=mcp burst=20 nodelay;

    location / {
        proxy_pass http://127.0.0.1:8080;
        proxy_set_header X-Real-IP $remote_addr;
        proxy_set_header X-Forwarded-For $proxy_add_x_forwarded_for;
    }
}
```

## Security Testing

### Manual Tests

```bash
# Test 1: Default password detection
grep -r "password" .env && echo "FAIL: Default password in use"

# Test 2: Secret exposure in logs
grep -r "neo4j_password\|api_key" /var/log/world_weaver/ && echo "FAIL: Secrets in logs"

# Test 3: File permissions
find /var/lib/world_weaver -type f -perm /022 && echo "FAIL: Insecure permissions"

# Test 4: Open ports
nmap localhost -p 7687,6333 && echo "WARNING: Databases exposed"
```

### Automated Security Scan

```bash
# Install security tools
pip install bandit safety

# Run static analysis
bandit -r src/t4dm/ -f json -o security_report.json

# Check dependencies
safety check --json

# Check for secrets
pip install detect-secrets
detect-secrets scan > .secrets.baseline
```

## Incident Response Plan

### If Breach Suspected

1. **Isolate**
   - Disconnect affected systems from network
   - Revoke all API keys and passwords
   - Enable enhanced logging

2. **Investigate**
   - Check audit logs for unauthorized access
   - Review database query logs
   - Identify compromised session IDs

3. **Remediate**
   - Rotate all credentials
   - Patch vulnerable components
   - Restore from clean backup if needed

4. **Notify**
   - Inform users of affected sessions
   - Report to security team
   - Document incident for review

### Emergency Contacts

```
Security Team: security@example.com
Database Admin: dba@example.com
On-Call Engineer: +1-555-ONCALL
```

## Compliance Requirements

### GDPR (if applicable)

- [ ] Privacy policy documented
- [ ] Data retention policy defined (default: 365 days)
- [ ] User consent recorded per session
- [ ] Data export API implemented
- [ ] Data deletion API implemented
- [ ] Data processing agreement with hosting provider

### SOC 2 (if applicable)

- [ ] Access logs retained for 1 year
- [ ] Change management process documented
- [ ] Security training completed
- [ ] Penetration test conducted annually
- [ ] Vulnerability disclosure policy published

## Regular Maintenance

### Weekly

- Review access logs for anomalies
- Check error rates and performance metrics
- Verify backup completion

### Monthly

- Rotate database passwords
- Update dependencies (after testing)
- Review and prune old sessions

### Quarterly

- Security audit by external team
- Dependency vulnerability scan
- Update threat model
- Review and update this checklist

---

**Last Updated**: 2025-11-27
**Next Review**: 2025-12-27
