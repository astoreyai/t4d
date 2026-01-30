# Security Policy

## Supported Versions

| Version | Supported          |
| ------- | ------------------ |
| 0.2.x   | :white_check_mark: |
| 0.1.x   | :x:                |

## Reporting a Vulnerability

We take security seriously. If you discover a security vulnerability in World Weaver, please report it responsibly.

### How to Report

1. **DO NOT** open a public GitHub issue for security vulnerabilities
2. Email security concerns to: security@astoreyai.com
3. Include:
   - Description of the vulnerability
   - Steps to reproduce
   - Potential impact
   - Any suggested fixes (optional)

### Response Timeline

- **Initial Response**: Within 48 hours
- **Triage & Assessment**: Within 7 days
- **Fix Development**: Depends on severity
  - Critical: Within 7 days
  - High: Within 14 days
  - Medium: Within 30 days
  - Low: Within 90 days

### What to Expect

1. Acknowledgment of your report
2. Regular updates on fix progress
3. Credit in the security advisory (if desired)
4. Notification when the fix is released

## Security Best Practices

### Deployment

1. **Environment Variables**: Never commit secrets to version control
   ```bash
   # Use .env files (excluded from git)
   cp .env.example .env
   ./scripts/setup-env.sh
   ```

2. **Network Security**: Bind services to localhost only in development
   ```yaml
   # docker-compose.yml binds to 127.0.0.1 by default
   ports:
     - "127.0.0.1:7687:7687"  # Neo4j
     - "127.0.0.1:6333:6333"  # Qdrant
   ```

3. **Authentication**: Enable Neo4j authentication in production
   ```bash
   NEO4J_AUTH=neo4j/your-strong-password
   ```

4. **API Security**: Configure CORS appropriately
   ```bash
   WW_API_CORS_ORIGINS=https://your-domain.com
   ```

### Session Isolation

World Weaver implements strict session isolation:
- Each session has separate memory namespaces
- Session IDs are validated and sanitized
- Cross-session data access is prevented by design

### Input Validation

- All user inputs are validated using Pydantic models
- SQL/Cypher injection prevention through parameterized queries
- Content length limits prevent DoS attacks

### Rate Limiting

The API includes built-in rate limiting:
- Default: 100 requests per minute per session
- Configurable via environment variables

### Secure Defaults

- TLS disabled by default for local development
- Enable TLS in production with proper certificates
- API keys optional for development, recommended for production

## Known Security Considerations

### Embedding Model Security

- BGE-M3 models are downloaded from HuggingFace
- Verify model checksums when possible
- Use `WW_EMBEDDING_CACHE_DIR` to control model storage location

### Storage Security

- Neo4j: Enable SSL/TLS in production
- Qdrant: Enable API key authentication in production
- Both: Use encrypted volumes for sensitive data

### Memory Content

- World Weaver stores user-provided content
- Implement appropriate content filtering for your use case
- Consider PII detection and masking for sensitive applications

## Security Updates

Security updates are released as patch versions (e.g., 0.2.1, 0.2.2).

Subscribe to security advisories:
- Watch the GitHub repository
- Check the [CHANGELOG](CHANGELOG.md) for security-related updates

## Acknowledgments

We thank the following individuals for responsible disclosure:
- (None yet - be the first!)
