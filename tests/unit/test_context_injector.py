"""
Tests for ContextInjector ToonJSON integration.

E1: Validates token-optimized context injection.
"""

import pytest
from unittest.mock import AsyncMock, MagicMock

from ww.integrations.kymera.context_injector import ContextInjector, InjectionConfig
from ww.integrations.kymera.bridge import MemoryContext
from ww.learning.events import ToonJSON


class TestToonJSONIntegration:
    """Test ToonJSON context formatting."""

    def test_toon_json_key_mappings(self):
        """Verify ToonJSON has context injection keys."""
        toon = ToonJSON()

        # E1: Context injection keys added
        assert toon.KEY_MAP["episodes"] == "eps"
        assert toon.KEY_MAP["entities"] == "ents"
        assert toon.KEY_MAP["skills"] == "sk"
        assert toon.KEY_MAP["personal_context"] == "pc"
        assert toon.KEY_MAP["name"] == "n"
        assert toon.KEY_MAP["summary"] == "sum"
        assert toon.KEY_MAP["description"] == "d"

    def test_toon_json_encoding(self):
        """Test ToonJSON encodes context correctly."""
        toon = ToonJSON()

        context_dict = {
            "episodes": [
                {"content": "Test conversation about Python", "timestamp": "2025-12-05T10:30:00"},
            ],
            "entities": [
                {"name": "Python", "summary": "Programming language"},
            ],
            "skills": [
                {"name": "coding", "description": "Write Python code"},
            ],
            "personal_context": "Working on project",
        }

        encoded = toon.encode(context_dict)

        # Should use short keys
        assert '"eps"' in encoded
        assert '"ents"' in encoded
        assert '"sk"' in encoded
        assert '"pc"' in encoded
        assert '"c"' in encoded  # content
        assert '"n"' in encoded  # name
        assert '"sum"' in encoded  # summary
        assert '"d"' in encoded  # description

        # Should NOT have long keys
        assert '"episodes"' not in encoded
        assert '"entities"' not in encoded
        assert '"skills"' not in encoded
        assert '"personal_context"' not in encoded
        assert '"content"' not in encoded
        assert '"summary"' not in encoded
        assert '"description"' not in encoded

    def test_token_reduction(self):
        """Verify ToonJSON achieves significant key compression."""
        toon = ToonJSON()

        # Larger context to amortize encoding overhead
        context_dict = {
            "episodes": [
                {"content": f"Conversation {i} about machine learning and neural networks", "timestamp": f"2025-12-0{i%9+1}T10:30:00"}
                for i in range(5)
            ],
            "entities": [
                {"name": f"Entity{i}", "summary": f"Description of entity {i} with more detail"}
                for i in range(5)
            ],
            "skills": [
                {"name": "model_training", "description": "Train models with custom datasets"},
                {"name": "data_preprocessing", "description": "Clean and prepare data"},
            ],
            "personal_context": "Working on a complex deep learning project with many components",
        }

        verbose = str(context_dict)
        compact = toon.encode(context_dict)

        reduction = 1 - len(compact) / len(verbose)
        # ToonJSON reduces key lengths - minimum 15% for structured data
        assert reduction > 0.15, f"Expected >15% reduction, got {reduction:.1%}"

    def test_toon_json_roundtrip(self):
        """Verify ToonJSON decode reconstructs structure."""
        toon = ToonJSON()

        context_dict = {
            "episodes": [{"content": "Test", "timestamp": "2025-12-05T10:30:00"}],
            "entities": [{"name": "Test", "summary": "Test summary"}],
        }

        encoded = toon.encode(context_dict)
        decoded = toon.decode(encoded)

        # Keys should be expanded back
        assert "episodes" in decoded
        assert "entities" in decoded
        assert decoded["episodes"][0]["content"] == "Test"
        assert decoded["entities"][0]["name"] == "Test"


class TestInjectionConfig:
    """Test InjectionConfig defaults."""

    def test_toon_json_enabled_by_default(self):
        """E1: ToonJSON should be enabled by default."""
        config = InjectionConfig()
        assert config.use_toon_json is True

    def test_can_disable_toon_json(self):
        """Can disable ToonJSON for verbose format."""
        config = InjectionConfig(use_toon_json=False)
        assert config.use_toon_json is False


class TestContextInjectorFormatting:
    """Test ContextInjector _format_context methods."""

    @pytest.fixture
    def mock_bridge(self):
        """Create mock VoiceMemoryBridge."""
        bridge = MagicMock()
        bridge.get_relevant_context = AsyncMock()
        return bridge

    @pytest.fixture
    def sample_memory_ctx(self):
        """Create sample MemoryContext."""
        return MemoryContext(
            episodes=[
                {"content": "Discussed Python optimization", "timestamp": "2025-12-05T10:30:00"},
                {"content": "Reviewed architecture patterns", "timestamp": "2025-12-04T14:00:00"},
            ],
            entities=[
                {"name": "Python", "summary": "Programming language for scripting"},
                {"name": "Docker", "summary": "Container platform"},
            ],
            skills=[
                {"name": "debugging", "description": "Debug complex issues"},
            ],
            personal_context="Active development session",
        )

    def test_format_context_toon(self, mock_bridge, sample_memory_ctx):
        """Test ToonJSON context formatting."""
        config = InjectionConfig(use_toon_json=True)
        injector = ContextInjector(mock_bridge, config)

        result = injector._format_context(sample_memory_ctx)

        # Should include legend
        assert "[Legend:" in result
        assert "eps=history" in result
        assert "ents=known" in result

        # Should have compact JSON
        assert '"eps"' in result
        assert '"ents"' in result
        assert '"sk"' in result
        assert '"pc"' in result

    def test_format_context_verbose(self, mock_bridge, sample_memory_ctx):
        """Test verbose markdown formatting."""
        config = InjectionConfig(use_toon_json=False)
        injector = ContextInjector(mock_bridge, config)

        result = injector._format_context(sample_memory_ctx)

        # Should use markdown headers
        assert "## Relevant History" in result
        assert "## Known Information" in result
        assert "## You Know How To" in result
        assert "## Current Status" in result

        # Should have bullet points
        assert "- " in result
        assert "**Python**" in result

    def test_format_context_empty(self, mock_bridge):
        """Empty context returns empty string."""
        config = InjectionConfig(use_toon_json=True)
        injector = ContextInjector(mock_bridge, config)

        empty_ctx = MemoryContext(
            episodes=[],
            entities=[],
            skills=[],
            personal_context=None,
        )

        result = injector._format_context(empty_ctx)
        assert result == ""

    def test_toon_vs_verbose_format_difference(self, mock_bridge, sample_memory_ctx):
        """Verify ToonJSON uses different format than verbose."""
        toon_config = InjectionConfig(use_toon_json=True)
        verbose_config = InjectionConfig(use_toon_json=False)

        toon_injector = ContextInjector(mock_bridge, toon_config)
        verbose_injector = ContextInjector(mock_bridge, verbose_config)

        toon_result = toon_injector._format_context(sample_memory_ctx)
        verbose_result = verbose_injector._format_context(sample_memory_ctx)

        # ToonJSON uses JSON structure with short keys
        assert '"eps"' in toon_result
        assert '"ents"' in toon_result
        assert "[Legend:" in toon_result

        # Verbose uses markdown
        assert "## Relevant History" in verbose_result
        assert "**Python**" in verbose_result

    def test_toon_json_provides_structure(self, mock_bridge):
        """ToonJSON provides structured JSON for LLM parsing."""
        import json

        large_ctx = MemoryContext(
            episodes=[
                {"content": f"Episode {i} content", "timestamp": f"2025-12-0{i%9+1}T10:00:00"}
                for i in range(5)
            ],
            entities=[
                {"name": f"Entity{i}", "summary": f"Summary {i}"}
                for i in range(10)
            ],
            skills=[
                {"name": f"skill_{i}", "description": f"Desc {i}"}
                for i in range(3)
            ],
            personal_context="Working on project",
        )

        toon_config = InjectionConfig(use_toon_json=True)
        injector = ContextInjector(mock_bridge, toon_config)

        result = injector._format_context(large_ctx)

        # ToonJSON should produce valid, parseable JSON after legend
        lines = result.split("\n", 1)
        assert len(lines) == 2
        assert "[Legend:" in lines[0]

        # JSON should be parseable
        data = json.loads(lines[1])

        # Structure should be intact
        assert "eps" in data
        assert "ents" in data
        assert "sk" in data
        assert "pc" in data
        assert len(data["eps"]) == 5
        assert len(data["ents"]) == 10
        assert len(data["sk"]) == 3

        # Values should use short keys
        assert "c" in data["eps"][0]  # content
        assert "n" in data["ents"][0]  # name
        assert "d" in data["sk"][0]  # description

    def test_respects_max_episodes(self, mock_bridge):
        """Respects max_episodes config."""
        config = InjectionConfig(use_toon_json=True, max_episodes=2)
        injector = ContextInjector(mock_bridge, config)

        many_episodes = MemoryContext(
            episodes=[
                {"content": f"Episode {i}", "timestamp": f"2025-12-0{i}T10:00:00"}
                for i in range(5)
            ],
            entities=[],
            skills=[],
            personal_context=None,
        )

        result = injector._format_context(many_episodes)

        # Should only have 2 episodes
        assert result.count('"c"') == 2

    def test_respects_max_entities(self, mock_bridge):
        """Respects max_entities config."""
        config = InjectionConfig(use_toon_json=True, max_entities=2)
        injector = ContextInjector(mock_bridge, config)

        many_entities = MemoryContext(
            episodes=[],
            entities=[
                {"name": f"Entity{i}", "summary": f"Summary {i}"}
                for i in range(5)
            ],
            skills=[],
            personal_context=None,
        )

        result = injector._format_context(many_entities)

        # Count entity entries (each has name and summary)
        import json
        # Remove legend line
        json_part = result.split("\n", 1)[1]
        decoded = json.loads(json_part)
        assert len(decoded.get("ents", [])) == 2


class TestContextInjectorLegend:
    """Test ToonJSON legend in context."""

    def test_legend_format(self):
        """Legend should be human-readable."""
        legend = ContextInjector.TOON_LEGEND

        # Should explain key abbreviations
        assert "eps=history" in legend
        assert "ents=known" in legend
        assert "sk=skills" in legend
        assert "pc=status" in legend
        assert "c=content" in legend
        assert "n=name" in legend
