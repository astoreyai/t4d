# Contributing to World Weaver

## Development Setup

### Prerequisites
- Python 3.11+
- Docker (for local Qdrant/Neo4j)
- Git

### Setup
```bash
# Clone repo
git clone https://github.com/user/world-weaver.git
cd world-weaver

# Create virtual environment
python -m venv venv
source venv/bin/activate

# Install with dev dependencies
pip install -e ".[dev]"

# Start services
docker-compose up -d

# Run tests
pytest tests/ -v
```

## Code Style

- **Formatter**: Black (line length 100)
- **Linter**: Ruff
- **Type Hints**: Required for all public functions
- **Docstrings**: Google style

```bash
# Format code
black src/ tests/

# Lint
ruff check src/ tests/

# Type check
mypy src/
```

## Testing Requirements

- Minimum 70% coverage (enforced by CI)
- All new features require tests
- Use pytest-asyncio for async tests
- Use hypothesis for algorithm properties

```bash
# Run all tests
pytest tests/ -v --cov=src/ww

# Run specific test file
pytest tests/unit/test_episodic.py -v

# Skip slow tests
pytest tests/ -m "not slow"
```

## Pull Request Process

1. Fork the repository
2. Create feature branch (`git checkout -b feature/amazing-feature`)
3. Make changes with tests
4. Ensure CI passes (`pytest`, `black`, `ruff`)
5. Update CHANGELOG.md
6. Submit PR with description

### PR Checklist
- [ ] Tests added/updated
- [ ] Documentation updated
- [ ] CHANGELOG.md updated
- [ ] No new linter warnings
- [ ] Coverage maintained ≥70%

## Commit Messages

Use conventional commits:
- `feat:` New feature
- `fix:` Bug fix
- `docs:` Documentation
- `test:` Tests
- `refactor:` Code refactoring
- `perf:` Performance improvement

Example: `feat: add pagination to recall_episodes`

## Architecture Guidelines

- Memory layer should not contain Cypher queries
- All MCP tools use TypedDict responses
- Rate limiting applied at gateway level
- Session isolation via payload filtering
