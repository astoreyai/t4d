"""
Entity extraction service for T4DM.

Automatically extracts named entities from episode content using:
- Regex patterns (emails, URLs, dates, etc.)
- LLM-based extraction (people, concepts, technologies)
- Composite extraction (combines multiple methods)
"""

import json
import logging
import os
import re
from dataclasses import dataclass
from typing import Any, Protocol

logger = logging.getLogger(__name__)


@dataclass
class ExtractedEntity:
    """Entity extracted from text content."""

    name: str
    entity_type: str  # PERSON, ORG, LOCATION, CONCEPT, etc.
    confidence: float
    span: tuple[int, int]  # Character positions in source text
    context: str  # Surrounding text for disambiguation

    def __hash__(self):
        """Make hashable for deduplication."""
        return hash((self.name.lower(), self.entity_type))

    def __eq__(self, other):
        """Equality based on normalized name and type."""
        if not isinstance(other, ExtractedEntity):
            return False
        return (self.name.lower() == other.name.lower() and
                self.entity_type == other.entity_type)


class EntityExtractor(Protocol):
    """Protocol for entity extraction implementations."""

    async def extract(self, text: str) -> list[ExtractedEntity]:
        """Extract entities from text."""
        ...


class RegexEntityExtractor:
    """
    Simple regex-based entity extraction for common patterns.

    Extracts:
    - Email addresses
    - URLs
    - Phone numbers
    - Dates (ISO format)
    - Money amounts
    - File paths
    - Git hashes
    - Package names (Python, npm)
    """

    PATTERNS = {
        "EMAIL": r"\b[\w.+-]+@[\w.-]+\.[a-zA-Z]{2,}\b",
        "URL": r'https?://[^\s<>"{}|\\^`\[\]]+',
        "PHONE": r"\b(?:\+\d{1,3}[-.]?)?\d{3}[-.]?\d{3}[-.]?\d{4}\b",
        "DATE": r"\b\d{4}-\d{2}-\d{2}\b",
        "MONEY": r"\$\d+(?:,\d{3})*(?:\.\d{2})?",
        "FILE_PATH": r"(?:/[\w.-]+)+\.[\w]+",
        "GIT_HASH": r"\b[0-9a-f]{7,40}\b",
        "PYTHON_PACKAGE": r"\bimport\s+([\w.]+)",
        "NPM_PACKAGE": r"(?:npm|yarn)\s+(?:install|add)\s+([@\w/-]+)",
    }

    # Map pattern types to entity types
    TYPE_MAPPING = {
        "EMAIL": "CONTACT",
        "URL": "RESOURCE",
        "PHONE": "CONTACT",
        "DATE": "TEMPORAL",
        "MONEY": "FINANCIAL",
        "FILE_PATH": "RESOURCE",
        "GIT_HASH": "RESOURCE",
        "PYTHON_PACKAGE": "TECHNOLOGY",
        "NPM_PACKAGE": "TECHNOLOGY",
    }

    def __init__(self, context_window: int = 50):
        """
        Initialize regex extractor.

        Args:
            context_window: Number of characters to include before/after match
        """
        self.context_window = context_window

    async def extract(self, text: str) -> list[ExtractedEntity]:
        """
        Extract entities using regex patterns.

        Args:
            text: Input text to extract from

        Returns:
            List of extracted entities
        """
        entities = []

        for pattern_name, pattern in self.PATTERNS.items():
            for match in re.finditer(pattern, text, re.IGNORECASE):
                # Extract matched text (use group 1 if available, else group 0)
                entity_name = match.group(1) if match.lastindex else match.group()

                # Get context window
                start_idx = max(0, match.start() - self.context_window)
                end_idx = min(len(text), match.end() + self.context_window)
                context = text[start_idx:end_idx]

                # Map to entity type
                entity_type = self.TYPE_MAPPING.get(pattern_name, "RESOURCE")

                entities.append(ExtractedEntity(
                    name=entity_name.strip(),
                    entity_type=entity_type,
                    confidence=1.0,  # Regex patterns are deterministic
                    span=(match.start(), match.end()),
                    context=context,
                ))

        logger.debug(f"Regex extraction found {len(entities)} entities")
        return entities


class LLMEntityExtractor:
    """
    LLM-based entity extraction for semantic entities.

    Extracts:
    - People (PERSON)
    - Organizations (ORGANIZATION)
    - Concepts (CONCEPT)
    - Technologies (TECHNOLOGY)
    - Events (EVENT)
    - Locations (LOCATION)
    - Projects (PROJECT)
    - Tools (TOOL)
    """

    EXTRACTION_PROMPT = """Extract named entities from the following text.
Return a JSON array of objects with these fields:
- "name": The entity name as it appears in text
- "type": One of: PERSON, ORGANIZATION, LOCATION, CONCEPT, TECHNOLOGY, EVENT, PROJECT, TOOL
- "confidence": Float 0.0-1.0 based on certainty

Only include clearly identifiable entities. Be conservative with confidence scores.

Text: {text}

JSON array (no markdown, just valid JSON):"""

    def __init__(
        self,
        api_key: str | None = None,
        model: str = "gpt-4o-mini",
        max_text_length: int = 4000,
        timeout: float = 30.0,
    ):
        """
        Initialize LLM entity extractor.

        Args:
            api_key: OpenAI API key (defaults to OPENAI_API_KEY env var)
            model: Model to use (default gpt-4o-mini for cost efficiency)
            max_text_length: Max characters to send to LLM
            timeout: Request timeout in seconds
        """
        self.model = model
        self.max_text_length = max_text_length
        self.timeout = timeout
        self._client: Any | None = None

        # Initialize client
        self._api_key = api_key or os.getenv("OPENAI_API_KEY")
        if self._api_key:
            try:
                from openai import AsyncOpenAI
                self._client = AsyncOpenAI(
                    api_key=self._api_key,
                    timeout=timeout,
                )
                logger.info(f"LLM entity extractor initialized with model {model}")
            except ImportError:
                logger.warning(
                    "openai package not installed. "
                    "Install with: pip install openai"
                )
        else:
            logger.debug("No OpenAI API key provided, LLM extraction disabled")

    @property
    def is_available(self) -> bool:
        """Check if LLM extraction is available."""
        return self._client is not None

    async def extract(self, text: str) -> list[ExtractedEntity]:
        """
        Extract entities from text using LLM.

        Args:
            text: Text to extract entities from

        Returns:
            List of extracted entities
        """
        if not self.is_available:
            logger.debug("LLM extraction not available, returning empty")
            return []

        if not text or not text.strip():
            return []

        # Truncate if needed
        if len(text) > self.max_text_length:
            text = text[:self.max_text_length] + "..."
            logger.debug(f"Truncated text to {self.max_text_length} chars")

        try:
            response_text = await self._call_llm(text)
            entities = self._parse_response(response_text, text)
            logger.debug(f"LLM extracted {len(entities)} entities")
            return entities
        except Exception as e:
            logger.error(f"LLM extraction failed: {e}")
            return []

    async def _call_llm(self, text: str) -> str:
        """
        Call LLM API for entity extraction.

        Args:
            text: Text to analyze

        Returns:
            JSON string response from LLM
        """
        prompt = self.EXTRACTION_PROMPT.format(text=text)

        try:
            response = await self._client.chat.completions.create(
                model=self.model,
                messages=[
                    {
                        "role": "system",
                        "content": "You are an entity extraction assistant. Return only valid JSON arrays."
                    },
                    {
                        "role": "user",
                        "content": prompt
                    }
                ],
                temperature=0.1,  # Low temperature for consistency
                max_tokens=1000,
                response_format={"type": "json_object"} if "gpt-4" in self.model else None,
            )

            content = response.choices[0].message.content
            return content.strip()

        except Exception as e:
            logger.error(f"OpenAI API call failed: {e}")
            raise

    def _parse_response(
        self,
        response: str,
        original_text: str,
    ) -> list[ExtractedEntity]:
        """
        Parse LLM JSON response into ExtractedEntity objects.

        Args:
            response: JSON string from LLM
            original_text: Original text for context extraction

        Returns:
            List of ExtractedEntity objects
        """
        entities = []

        try:
            # Handle potential wrapper object
            data = json.loads(response)
            if isinstance(data, dict):
                # Look for array in common keys
                for key in ["entities", "results", "items", "data"]:
                    if key in data and isinstance(data[key], list):
                        data = data[key]
                        break
                else:
                    # Single entity in object form
                    if "name" in data:
                        data = [data]
                    else:
                        logger.warning(f"Unexpected response format: {response[:100]}")
                        return []

            if not isinstance(data, list):
                logger.warning(f"Expected array, got {type(data)}")
                return []

            for item in data:
                if not isinstance(item, dict):
                    continue

                name = item.get("name", "").strip()
                entity_type = item.get("type", "CONCEPT").upper()
                confidence = float(item.get("confidence", 0.7))

                if not name:
                    continue

                # Find span in original text
                span = (0, 0)
                idx = original_text.lower().find(name.lower())
                if idx >= 0:
                    span = (idx, idx + len(name))

                # Extract context
                context_start = max(0, span[0] - 50)
                context_end = min(len(original_text), span[1] + 50)
                context = original_text[context_start:context_end]

                entities.append(ExtractedEntity(
                    name=name,
                    entity_type=entity_type,
                    confidence=min(1.0, max(0.0, confidence)),
                    span=span,
                    context=context,
                ))

        except json.JSONDecodeError as e:
            logger.warning(f"Failed to parse LLM response as JSON: {e}")
            # Try to extract from malformed response
            entities = self._extract_from_malformed(response, original_text)

        return entities

    def _extract_from_malformed(
        self,
        response: str,
        original_text: str,
    ) -> list[ExtractedEntity]:
        """Attempt to extract entities from malformed JSON response."""
        entities = []

        # Try to find JSON array in response
        array_match = re.search(r"\[[\s\S]*?\]", response)
        if array_match:
            try:
                data = json.loads(array_match.group())
                return self._parse_response(json.dumps(data), original_text)
            except json.JSONDecodeError:
                pass

        return entities


class CompositeEntityExtractor:
    """
    Combines multiple extraction methods.

    Runs all extractors in parallel and deduplicates results,
    keeping highest confidence for each unique entity.
    """

    def __init__(self, extractors: list[EntityExtractor]):
        """
        Initialize composite extractor.

        Args:
            extractors: List of extractors to use
        """
        self.extractors = extractors

    async def extract(self, text: str) -> list[ExtractedEntity]:
        """
        Run all extractors and deduplicate results.

        Args:
            text: Input text to extract from

        Returns:
            Deduplicated list of extracted entities
        """
        all_entities = []

        # Run all extractors
        for extractor in self.extractors:
            try:
                entities = await extractor.extract(text)
                all_entities.extend(entities)
            except Exception as e:
                logger.error(f"Extractor {type(extractor).__name__} failed: {e}")

        # Deduplicate
        deduplicated = self._deduplicate(all_entities)

        logger.info(
            f"Composite extraction: {len(all_entities)} total → "
            f"{len(deduplicated)} after deduplication"
        )

        return deduplicated

    def _deduplicate(self, entities: list[ExtractedEntity]) -> list[ExtractedEntity]:
        """
        Remove duplicate entities, keeping highest confidence.

        Entities are considered duplicates if they have the same
        normalized name and entity type.

        Args:
            entities: List of entities to deduplicate

        Returns:
            Deduplicated list
        """
        seen: dict[tuple[str, str], ExtractedEntity] = {}

        for entity in entities:
            key = (entity.name.lower(), entity.entity_type)

            if key not in seen or entity.confidence > seen[key].confidence:
                seen[key] = entity

        # Sort by confidence (descending)
        result = sorted(seen.values(), key=lambda e: e.confidence, reverse=True)

        logger.debug(
            f"Deduplication: {len(entities)} entities → {len(result)} unique"
        )

        return result


# Factory function for creating default extractor
def create_default_extractor(
    use_llm: bool = False,
    llm_client: Any | None = None,
    llm_model: str = "gpt-4o-mini",
    api_key: str | None = None,
) -> EntityExtractor:
    """
    Create default entity extractor.

    Args:
        use_llm: Whether to include LLM-based extraction
        llm_client: DEPRECATED: Use api_key instead
        llm_model: Model name for LLM
        api_key: OpenAI API key (or set OPENAI_API_KEY env var)

    Returns:
        Configured entity extractor
    """
    extractors = [RegexEntityExtractor()]

    if use_llm:
        # Support new API key parameter
        llm_extractor = LLMEntityExtractor(api_key=api_key, model=llm_model)
        if llm_extractor.is_available:
            extractors.append(llm_extractor)

    if len(extractors) > 1:
        return CompositeEntityExtractor(extractors)
    return extractors[0]
