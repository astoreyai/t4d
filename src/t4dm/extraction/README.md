# Extraction Module

**2 files | ~500 lines | Centrality: 1**

The extraction module provides named entity recognition (NER) services for automatic entity extraction from episodic content, supporting regex patterns, LLM-based semantic extraction, and hybrid approaches.

## Architecture Overview

```
┌─────────────────────────────────────────────────────────────────────────┐
│                       ENTITY EXTRACTION                                  │
├─────────────────────────────────────────────────────────────────────────┤
│  ┌─────────────────────────────────────────────────────────────────────┐│
│  │                    REGEX EXTRACTOR (Fast)                           ││
│  │  9 patterns: EMAIL, URL, PHONE, DATE, MONEY, FILE_PATH, GIT_HASH,   ││
│  │              PYTHON_PACKAGE, NPM_PACKAGE                            ││
│  │  100% confidence | Deterministic | High precision                   ││
│  └───────────────────────────────┬─────────────────────────────────────┘│
│                                  │                                      │
│  ┌───────────────────────────────┼─────────────────────────────────────┐│
│  │                    LLM EXTRACTOR (Semantic)                         ││
│  │  Entity types: PERSON, ORG, LOCATION, CONCEPT, TECHNOLOGY,          ││
│  │               EVENT, PROJECT, TOOL                                  ││
│  │  OpenAI API | Variable confidence | Semantic understanding          ││
│  └───────────────────────────────┬─────────────────────────────────────┘│
│                                  │                                      │
│  ┌───────────────────────────────▼─────────────────────────────────────┐│
│  │                    COMPOSITE EXTRACTOR                              ││
│  │  Parallel execution | Deduplication | Highest confidence wins       ││
│  └─────────────────────────────────────────────────────────────────────┘│
└─────────────────────────────────────────────────────────────────────────┘
```

## File Organization

| File | Lines | Purpose |
|------|-------|---------|
| `entity_extractor.py` | ~480 | Core extraction implementations |
| `__init__.py` | ~20 | Public API exports |

## Entity Data Structure

```python
from ww.extraction import ExtractedEntity

entity = ExtractedEntity(
    name="John Smith",
    entity_type="PERSON",
    confidence=0.95,
    span=(10, 20),           # Character positions
    context="...meeting with John Smith about..."
)

# Deduplication support
hash(entity)  # Case-insensitive hashing
entity1 == entity2  # Normalized name + type comparison
```

## Regex Extractor

Fast, deterministic extraction for structured patterns:

```python
from ww.extraction import RegexEntityExtractor

extractor = RegexEntityExtractor(context_window=50)

entities = await extractor.extract(
    "Contact john@example.com or visit https://example.com"
)

# [
#     ExtractedEntity(name="john@example.com", entity_type="CONTACT", confidence=1.0),
#     ExtractedEntity(name="https://example.com", entity_type="RESOURCE", confidence=1.0)
# ]
```

### Built-in Patterns (9 types)

| Pattern | Type | Example |
|---------|------|---------|
| EMAIL | CONTACT | john@example.com |
| URL | RESOURCE | https://example.com |
| PHONE | CONTACT | 555-123-4567 |
| DATE | TEMPORAL | 2025-11-27 |
| MONEY | FINANCIAL | $1,234.56 |
| FILE_PATH | RESOURCE | /home/user/file.py |
| GIT_HASH | RESOURCE | abc123def |
| PYTHON_PACKAGE | TECHNOLOGY | import pandas |
| NPM_PACKAGE | TECHNOLOGY | npm install react |

**Features**:
- 100% confidence (deterministic)
- Configurable context window (default 50 chars)
- Case-insensitive matching
- Group extraction support

## LLM Extractor

Semantic extraction using OpenAI API:

```python
from ww.extraction import LLMEntityExtractor

extractor = LLMEntityExtractor(
    api_key="sk-...",           # Or from OPENAI_API_KEY env
    model="gpt-4o-mini",        # Cost-efficient default
    max_text_length=4000,
    timeout=30.0
)

entities = await extractor.extract(
    "Dr. Jane Smith from MIT presented on quantum computing at AAAI 2025"
)

# [
#     ExtractedEntity(name="Dr. Jane Smith", entity_type="PERSON", confidence=0.9),
#     ExtractedEntity(name="MIT", entity_type="ORGANIZATION", confidence=0.95),
#     ExtractedEntity(name="quantum computing", entity_type="CONCEPT", confidence=0.85),
#     ExtractedEntity(name="AAAI 2025", entity_type="EVENT", confidence=0.9)
# ]
```

### Semantic Entity Types

| Type | Description |
|------|-------------|
| PERSON | Named individuals |
| ORGANIZATION | Companies, institutions |
| LOCATION | Places, addresses |
| CONCEPT | Abstract ideas, topics |
| TECHNOLOGY | Software, frameworks, languages |
| EVENT | Conferences, meetings, dates |
| PROJECT | Named projects |
| TOOL | Specific tools, utilities |

**Features**:
- JSON response parsing with recovery
- Multiple format support (arrays, wrapped objects)
- Confidence clamping to [0.0, 1.0]
- Case-insensitive span finding
- Graceful API error handling

## Composite Extractor

Hybrid approach combining multiple extractors:

```python
from ww.extraction import CompositeEntityExtractor, RegexEntityExtractor, LLMEntityExtractor

composite = CompositeEntityExtractor([
    RegexEntityExtractor(),
    LLMEntityExtractor(api_key="sk-...")
])

entities = await composite.extract(text)
# Runs extractors in parallel
# Deduplicates by (normalized_name, entity_type)
# Keeps highest confidence for duplicates
```

## Factory Function

```python
from ww.extraction import create_default_extractor

# Regex only (default)
extractor = create_default_extractor()

# With LLM
extractor = create_default_extractor(use_llm=True)
# Uses OPENAI_API_KEY from environment
# Falls back to regex if no API key
```

## Integration with Consolidation

The extraction module integrates with memory consolidation:

```python
# In consolidation/service.py
async def extract_entities_from_recent_episodes():
    """Background job during consolidation."""

    # Get recent episodes
    episodes = await episodic.get_recent(limit=batch_size)

    for episode in episodes:
        # Extract entities
        entities = await extractor.extract(episode.content)

        # Filter by confidence
        entities = [e for e in entities if e.confidence >= threshold]

        for entity in entities:
            # Create in semantic memory if new
            if not await semantic.exists(entity.name, entity.entity_type):
                await semantic.store_entity(
                    name=entity.name,
                    entity_type=entity.entity_type,
                    source=episode.id
                )

            # Link entity to episode
            await semantic.create_relation(
                source_id=entity_id,
                target_id=episode.id,
                relation_type="EXTRACTED_FROM"
            )
```

**Configuration** (from Settings):
- `extraction_use_llm`: Enable LLM extraction
- `extraction_llm_model`: Model name (default: gpt-4o-mini)
- `extraction_batch_size`: Episodes per batch
- `extraction_confidence_threshold`: Minimum confidence

## Error Handling

```python
# LLM API failures
try:
    entities = await llm_extractor.extract(text)
except Exception as e:
    logger.warning(f"LLM extraction failed: {e}")
    entities = []  # Graceful fallback

# Malformed JSON recovery
response = '{"entities": [{"name": "John"...'  # Truncated
# Attempts regex-based recovery
entities = _recover_from_malformed(response)

# Missing fields use defaults
entity = {"name": "Smith"}  # No type, no confidence
# → type="CONCEPT", confidence=0.7
```

## Testing

```bash
# Run extraction tests
pytest tests/extraction/ -v

# With coverage
pytest tests/extraction/ --cov=ww.extraction
```

**Test Coverage**: 51 test methods (693 lines)

### Test Categories

| Category | Tests | Coverage |
|----------|-------|----------|
| ExtractedEntity | 3 | Creation, equality, hashing |
| RegexEntityExtractor | 11 | All 9 patterns, context, multiples |
| LLMEntityExtractor | 26 | Parsing, recovery, defaults, errors |
| CompositeEntityExtractor | 5 | Parallel, dedup, failures |
| DefaultFactory | 3 | Regex/LLM/fallback |
| Integration | 3 | Code review, research, trading |

## Installation

```bash
# Core extraction (regex)
pip install -e "."

# With LLM support
pip install -e ".[llm]"
# Requires: openai
```

## Public API

```python
# Protocol
EntityExtractor (Protocol)

# Implementations
RegexEntityExtractor
LLMEntityExtractor
CompositeEntityExtractor

# Data class
ExtractedEntity

# Factory
create_default_extractor
```

## Performance

| Extractor | Speed | Cost |
|-----------|-------|------|
| Regex | ~1ms | Free |
| LLM (gpt-4o-mini) | ~500ms | ~$0.0001/call |
| Composite | ~500ms | Parallel overhead |

## Usage Examples

### Code Review Extraction

```python
text = """
Fixed bug in authentication.py using FastAPI's OAuth2PasswordBearer.
Reviewed by john.doe@company.com. See PR #1234.
"""

entities = await extractor.extract(text)
# RESOURCE: authentication.py
# TECHNOLOGY: FastAPI, OAuth2PasswordBearer
# CONTACT: john.doe@company.com
# RESOURCE: PR #1234
```

### Research Paper Extraction

```python
text = """
Dr. Smith from Stanford presents BERT-based NER at ACL 2025.
The model achieves 95% F1 on CoNLL-2003.
"""

entities = await extractor.extract(text)
# PERSON: Dr. Smith
# ORGANIZATION: Stanford
# TECHNOLOGY: BERT
# EVENT: ACL 2025
# CONCEPT: NER
```

## Design Patterns

| Pattern | Usage |
|---------|-------|
| Protocol | EntityExtractor interface |
| Strategy | Pluggable extractors |
| Composite | Multiple extractors combined |
| Factory | create_default_extractor |
| Graceful Degradation | Fallback on failures |
