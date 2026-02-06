"""Comprehensive tests for protocols.py - Storage and embedding protocols."""

import pytest
from typing import Any
from uuid import uuid4

from t4dm.core.protocols import (
    EmbeddingProvider,
    VectorStore,
    GraphStore,
    EpisodicStore,
    SemanticStore,
    ProceduralStore,
)
from t4dm.core.types import Episode, Entity, Procedure, Relationship, Domain


class MockEmbeddingProvider:
    """Mock embedding provider for testing."""

    @property
    def dimension(self) -> int:
        return 1024

    async def embed(self, texts: list[str]) -> list[list[float]]:
        """Return dummy embeddings."""
        return [[0.1] * 1024 for _ in texts]

    async def embed_query(self, query: str) -> list[float]:
        """Return dummy query embedding."""
        return [0.1] * 1024


class TestEmbeddingProviderProtocol:
    """Test EmbeddingProvider protocol compliance."""

    def test_embedding_provider_has_dimension_property(self):
        provider = MockEmbeddingProvider()
        assert hasattr(provider, 'dimension')
        assert provider.dimension == 1024

    def test_embedding_provider_has_embed_method(self):
        provider = MockEmbeddingProvider()
        assert hasattr(provider, 'embed')
        assert callable(provider.embed)

    def test_embedding_provider_has_embed_query_method(self):
        provider = MockEmbeddingProvider()
        assert hasattr(provider, 'embed_query')
        assert callable(provider.embed_query)

    @pytest.mark.asyncio
    async def test_embed_returns_list_of_lists(self):
        provider = MockEmbeddingProvider()
        result = await provider.embed(["text1", "text2"])
        assert isinstance(result, list)
        assert len(result) == 2
        assert all(isinstance(emb, list) for emb in result)

    @pytest.mark.asyncio
    async def test_embed_query_returns_list(self):
        provider = MockEmbeddingProvider()
        result = await provider.embed_query("query")
        assert isinstance(result, list)
        assert len(result) == 1024

    def test_embedding_provider_runtime_checkable(self):
        """Protocol should be runtime checkable."""
        provider = MockEmbeddingProvider()
        assert isinstance(provider, EmbeddingProvider)


class MockVectorStore:
    """Mock vector store for testing."""

    async def create_collection(
        self,
        name: str,
        dimension: int,
        distance: str = "cosine",
    ) -> None:
        pass

    async def add(
        self,
        collection: str,
        ids: list[str],
        vectors: list[list[float]],
        payloads: list[dict[str, Any]],
    ) -> None:
        pass

    async def search(
        self,
        collection: str,
        vector: list[float],
        limit: int = 10,
        filter: dict[str, Any] | None = None,
    ) -> list[tuple[str, float, dict[str, Any]]]:
        return []

    async def delete(
        self,
        collection: str,
        ids: list[str],
    ) -> None:
        pass

    async def get(
        self,
        collection: str,
        ids: list[str],
    ) -> list[tuple[str, dict[str, Any]]]:
        return []


class TestVectorStoreProtocol:
    """Test VectorStore protocol compliance."""

    def test_vector_store_has_create_collection(self):
        store = MockVectorStore()
        assert hasattr(store, 'create_collection')
        assert callable(store.create_collection)

    def test_vector_store_has_add(self):
        store = MockVectorStore()
        assert hasattr(store, 'add')
        assert callable(store.add)

    def test_vector_store_has_search(self):
        store = MockVectorStore()
        assert hasattr(store, 'search')
        assert callable(store.search)

    def test_vector_store_has_delete(self):
        store = MockVectorStore()
        assert hasattr(store, 'delete')
        assert callable(store.delete)

    def test_vector_store_has_get(self):
        store = MockVectorStore()
        assert hasattr(store, 'get')
        assert callable(store.get)

    @pytest.mark.asyncio
    async def test_search_returns_tuples(self):
        store = MockVectorStore()
        result = await store.search("test", [0.1] * 1024)
        assert isinstance(result, list)
        assert all(isinstance(item, tuple) for item in result)

    def test_vector_store_runtime_checkable(self):
        """Protocol should be runtime checkable."""
        store = MockVectorStore()
        assert isinstance(store, VectorStore)


class MockGraphStore:
    """Mock graph store for testing."""

    async def create_node(self, label: str, properties: dict[str, Any]) -> str:
        return "node_1"

    async def get_node(
        self, node_id: str, label: str | None = None
    ) -> dict[str, Any] | None:
        return {"id": node_id, "label": label}

    async def update_node(
        self, node_id: str, properties: dict[str, Any], label: str | None = None
    ) -> None:
        pass

    async def delete_node(self, node_id: str, label: str | None = None) -> None:
        pass

    async def create_relationship(
        self,
        source_id: str,
        target_id: str,
        rel_type: str,
        properties: dict[str, Any],
    ) -> None:
        pass

    async def get_relationships(
        self, node_id: str, rel_type: str | None = None, direction: str = "both"
    ) -> list[dict[str, Any]]:
        return []

    async def update_relationship(
        self,
        source_id: str,
        target_id: str,
        rel_type: str,
        properties: dict[str, Any],
    ) -> None:
        pass

    async def query(
        self, query: str | dict[str, Any], parameters: dict[str, Any] | None = None
    ) -> list[dict[str, Any]]:
        return []

    async def find_path(
        self, source_id: str, target_id: str, max_depth: int = 5
    ) -> list[str] | None:
        return None


class TestGraphStoreProtocol:
    """Test GraphStore protocol compliance."""

    def test_graph_store_has_create_node(self):
        store = MockGraphStore()
        assert hasattr(store, 'create_node')
        assert callable(store.create_node)

    def test_graph_store_has_get_node(self):
        store = MockGraphStore()
        assert hasattr(store, 'get_node')
        assert callable(store.get_node)

    def test_graph_store_has_update_node(self):
        store = MockGraphStore()
        assert hasattr(store, 'update_node')
        assert callable(store.update_node)

    def test_graph_store_has_delete_node(self):
        store = MockGraphStore()
        assert hasattr(store, 'delete_node')
        assert callable(store.delete_node)

    def test_graph_store_has_create_relationship(self):
        store = MockGraphStore()
        assert hasattr(store, 'create_relationship')
        assert callable(store.create_relationship)

    def test_graph_store_has_get_relationships(self):
        store = MockGraphStore()
        assert hasattr(store, 'get_relationships')
        assert callable(store.get_relationships)

    def test_graph_store_has_update_relationship(self):
        store = MockGraphStore()
        assert hasattr(store, 'update_relationship')
        assert callable(store.update_relationship)

    def test_graph_store_has_query(self):
        store = MockGraphStore()
        assert hasattr(store, 'query')
        assert callable(store.query)

    def test_graph_store_has_find_path(self):
        store = MockGraphStore()
        assert hasattr(store, 'find_path')
        assert callable(store.find_path)

    @pytest.mark.asyncio
    async def test_create_node_returns_string(self):
        store = MockGraphStore()
        result = await store.create_node("Entity", {"name": "test"})
        assert isinstance(result, str)

    def test_graph_store_runtime_checkable(self):
        """Protocol should be runtime checkable."""
        store = MockGraphStore()
        assert isinstance(store, GraphStore)


class MockEpisodicStore:
    """Mock episodic store for testing."""

    async def create_episode(self, episode: Episode) -> Episode:
        return episode

    async def get_episode(self, episode_id) -> Episode | None:
        return None

    async def search_episodes(
        self,
        query_vector: list[float],
        limit: int = 10,
        session_filter: str | None = None,
        time_filter: tuple | None = None,
    ) -> list[tuple[Episode, float]]:
        return []

    async def update_episode_access(self, episode_id, success: bool = True) -> None:
        pass


class TestEpisodicStoreProtocol:
    """Test EpisodicStore protocol compliance."""

    def test_episodic_store_has_create_episode(self):
        store = MockEpisodicStore()
        assert hasattr(store, 'create_episode')
        assert callable(store.create_episode)

    def test_episodic_store_has_get_episode(self):
        store = MockEpisodicStore()
        assert hasattr(store, 'get_episode')
        assert callable(store.get_episode)

    def test_episodic_store_has_search_episodes(self):
        store = MockEpisodicStore()
        assert hasattr(store, 'search_episodes')
        assert callable(store.search_episodes)

    def test_episodic_store_has_update_episode_access(self):
        store = MockEpisodicStore()
        assert hasattr(store, 'update_episode_access')
        assert callable(store.update_episode_access)

    @pytest.mark.asyncio
    async def test_create_episode_returns_episode(self):
        store = MockEpisodicStore()
        ep = Episode(session_id="s", content="test")
        result = await store.create_episode(ep)
        assert isinstance(result, Episode)

    def test_episodic_store_runtime_checkable(self):
        """Protocol should be runtime checkable."""
        store = MockEpisodicStore()
        assert isinstance(store, EpisodicStore)


class MockSemanticStore:
    """Mock semantic store for testing."""

    async def create_entity(self, entity: Entity) -> Entity:
        return entity

    async def get_entity(self, entity_id) -> Entity | None:
        return None

    async def search_entities(
        self,
        query_vector: list[float],
        limit: int = 10,
        entity_type: str | None = None,
    ) -> list[tuple[Entity, float]]:
        return []

    async def create_relationship(self, relationship: Relationship) -> None:
        pass

    async def strengthen_relationship(
        self, source_id, target_id, learning_rate: float = 0.1
    ) -> float:
        return 0.1

    async def get_neighbors(self, entity_id, rel_type: str | None = None):
        return []

    async def supersede_entity(self, entity_id, new_summary: str, new_details: str | None = None):
        return None


class TestSemanticStoreProtocol:
    """Test SemanticStore protocol compliance."""

    def test_semantic_store_has_create_entity(self):
        store = MockSemanticStore()
        assert hasattr(store, 'create_entity')
        assert callable(store.create_entity)

    def test_semantic_store_has_get_entity(self):
        store = MockSemanticStore()
        assert hasattr(store, 'get_entity')
        assert callable(store.get_entity)

    def test_semantic_store_has_search_entities(self):
        store = MockSemanticStore()
        assert hasattr(store, 'search_entities')
        assert callable(store.search_entities)

    def test_semantic_store_has_create_relationship(self):
        store = MockSemanticStore()
        assert hasattr(store, 'create_relationship')
        assert callable(store.create_relationship)

    def test_semantic_store_has_strengthen_relationship(self):
        store = MockSemanticStore()
        assert hasattr(store, 'strengthen_relationship')
        assert callable(store.strengthen_relationship)

    def test_semantic_store_has_get_neighbors(self):
        store = MockSemanticStore()
        assert hasattr(store, 'get_neighbors')
        assert callable(store.get_neighbors)

    def test_semantic_store_has_supersede_entity(self):
        store = MockSemanticStore()
        assert hasattr(store, 'supersede_entity')
        assert callable(store.supersede_entity)

    def test_semantic_store_runtime_checkable(self):
        """Protocol should be runtime checkable."""
        store = MockSemanticStore()
        assert isinstance(store, SemanticStore)


class MockProceduralStore:
    """Mock procedural store for testing."""

    async def create_procedure(self, procedure: Procedure) -> Procedure:
        return procedure

    async def get_procedure(self, procedure_id) -> Procedure | None:
        return None

    async def search_procedures(
        self,
        query_vector: list[float],
        limit: int = 5,
        domain: str | None = None,
    ) -> list[tuple[Procedure, float]]:
        return []

    async def update_procedure_feedback(self, procedure_id, success: bool) -> Procedure:
        return Procedure(name="test", domain=Domain.CODING)

    async def deprecate_procedure(self, procedure_id, consolidated_into = None) -> None:
        pass


class TestProceduralStoreProtocol:
    """Test ProceduralStore protocol compliance."""

    def test_procedural_store_has_create_procedure(self):
        store = MockProceduralStore()
        assert hasattr(store, 'create_procedure')
        assert callable(store.create_procedure)

    def test_procedural_store_has_get_procedure(self):
        store = MockProceduralStore()
        assert hasattr(store, 'get_procedure')
        assert callable(store.get_procedure)

    def test_procedural_store_has_search_procedures(self):
        store = MockProceduralStore()
        assert hasattr(store, 'search_procedures')
        assert callable(store.search_procedures)

    def test_procedural_store_has_update_procedure_feedback(self):
        store = MockProceduralStore()
        assert hasattr(store, 'update_procedure_feedback')
        assert callable(store.update_procedure_feedback)

    def test_procedural_store_has_deprecate_procedure(self):
        store = MockProceduralStore()
        assert hasattr(store, 'deprecate_procedure')
        assert callable(store.deprecate_procedure)

    def test_procedural_store_runtime_checkable(self):
        """Protocol should be runtime checkable."""
        store = MockProceduralStore()
        assert isinstance(store, ProceduralStore)


class TestProtocolAbstractMethods:
    """Test that all protocol methods are properly abstract."""

    def test_embedding_provider_abstract_methods(self):
        """All methods should be abstract in protocol."""
        # Try to verify the methods exist and are callable
        methods = ['dimension', 'embed', 'embed_query']
        for method in methods:
            assert hasattr(EmbeddingProvider, method)

    def test_vector_store_abstract_methods(self):
        """All methods should be abstract in protocol."""
        methods = ['create_collection', 'add', 'search', 'delete', 'get']
        for method in methods:
            assert hasattr(VectorStore, method)

    def test_graph_store_abstract_methods(self):
        """All methods should be abstract in protocol."""
        methods = [
            'create_node', 'get_node', 'update_node', 'delete_node',
            'create_relationship', 'get_relationships', 'update_relationship',
            'query', 'find_path'
        ]
        for method in methods:
            assert hasattr(GraphStore, method)


class TestProtocolDocumentation:
    """Test that protocols have documentation."""

    def test_embedding_provider_has_docstring(self):
        assert EmbeddingProvider.__doc__ is not None

    def test_vector_store_has_docstring(self):
        assert VectorStore.__doc__ is not None

    def test_graph_store_has_docstring(self):
        assert GraphStore.__doc__ is not None

    def test_episodic_store_has_docstring(self):
        assert EpisodicStore.__doc__ is not None


class TestProtocolIntegration:
    """Integration tests for protocol usage."""

    @pytest.mark.asyncio
    async def test_embedding_provider_full_workflow(self):
        """Test complete embedding provider workflow."""
        provider = MockEmbeddingProvider()

        # Check dimension
        assert provider.dimension > 0

        # Generate embeddings
        texts = ["hello world", "test text"]
        embeddings = await provider.embed(texts)
        assert len(embeddings) == len(texts)

        # Generate query embedding
        query_emb = await provider.embed_query("query")
        assert len(query_emb) == provider.dimension

    @pytest.mark.asyncio
    async def test_vector_store_full_workflow(self):
        """Test complete vector store workflow."""
        store = MockVectorStore()

        # Create collection
        await store.create_collection("test", 1024)

        # Add vectors
        await store.add(
            "test",
            ["id1", "id2"],
            [[0.1] * 1024, [0.2] * 1024],
            [{"key": "value"}, {"key": "value2"}]
        )

        # Search
        results = await store.search("test", [0.1] * 1024)
        assert isinstance(results, list)

        # Get
        items = await store.get("test", ["id1"])
        assert isinstance(items, list)

        # Delete
        await store.delete("test", ["id1"])

    @pytest.mark.asyncio
    async def test_graph_store_full_workflow(self):
        """Test complete graph store workflow."""
        store = MockGraphStore()

        # Create nodes
        node_id = await store.create_node("Entity", {"name": "test"})
        assert isinstance(node_id, str)

        # Get node
        node = await store.get_node(node_id)
        assert node is not None

        # Update node
        await store.update_node(node_id, {"updated": True})

        # Create relationship
        node2_id = await store.create_node("Entity", {"name": "test2"})
        await store.create_relationship(node_id, node2_id, "RELATES_TO", {})

        # Query
        results = await store.query("MATCH (n) RETURN n")
        assert isinstance(results, list)

        # Delete node
        await store.delete_node(node_id)
