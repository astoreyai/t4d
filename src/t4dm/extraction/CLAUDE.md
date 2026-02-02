# Extraction
**Path**: `/mnt/projects/t4d/t4dm/src/t4dm/extraction/`

## What
Named entity extraction from episodic memory content. Identifies structured entities (people, organizations, technologies, URLs, dates, etc.) from free-text memory content using regex patterns and optional LLM-based semantic extraction.

## How
- **RegexEntityExtractor**: Pattern-matching for deterministic entities (emails, URLs, dates, money, file paths, git hashes, package names). Maps patterns to entity types (CONTACT, RESOURCE, TEMPORAL, FINANCIAL, TECHNOLOGY).
- **LLMEntityExtractor**: Uses OpenAI API (gpt-4o-mini default) to extract semantic entities (PERSON, ORGANIZATION, CONCEPT, PROJECT, TOOL, etc.) with confidence scores.
- **CompositeEntityExtractor**: Runs multiple extractors, deduplicates by normalized name+type, keeps highest confidence.
- **ExtractedEntity**: Dataclass with name, type, confidence, character span, and surrounding context.

## Why
Enables automatic knowledge graph construction from raw episodic content. Extracted entities feed into semantic memory and Neo4j graph storage, supporting structured queries and relationship discovery across memories.

## Key Files
| File | Purpose |
|------|---------|
| `entity_extractor.py` | All extractor implementations, `ExtractedEntity` dataclass, `create_default_extractor()` factory |
| `__init__.py` | Public API exports |

## Data Flow
```
Episodic content (text)
    -> RegexEntityExtractor (deterministic patterns)
    -> LLMEntityExtractor (semantic, optional)
    -> CompositeEntityExtractor (dedup, rank by confidence)
    -> list[ExtractedEntity]
    -> Consolidation hooks (EntityExtractedHook)
    -> Semantic memory / Neo4j graph
```

## Integration Points
- **consolidation/**: `EntityExtractedHook` fires when entities are found during consolidation
- **memory/semantic.py**: Extracted entities become semantic memory nodes
- **bridges/neo4j**: Entities stored as graph nodes with relationships
