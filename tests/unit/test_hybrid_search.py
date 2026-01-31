"""
Unit tests for hybrid search functionality.

Tests the BGE-M3 sparse vector generation and Qdrant hybrid search integration.
"""

import pytest
from unittest.mock import AsyncMock, MagicMock, patch
import numpy as np


class TestBGEM3HybridEmbedding:
    """Tests for BGEM3Embedding hybrid methods."""

    @pytest.fixture
    def mock_flag_model(self):
        """Mock FlagEmbedding model."""
        model = MagicMock()
        model.encode.return_value = {
            "dense_vecs": np.array([[0.1] * 1024, [0.2] * 1024]),
            "lexical_weights": [
                {100: 0.5, 200: 0.3, 300: 0.2},
                {100: 0.4, 400: 0.6},
            ],
        }
        return model

    @pytest.fixture
    def embedding_provider(self, mock_flag_model):
        """Create embedding provider with mocked model."""
        with patch("t4dm.embedding.bge_m3.get_settings") as mock_settings:
            mock_settings.return_value = MagicMock(
                embedding_model="BAAI/bge-m3",
                embedding_device="cpu",
                embedding_use_fp16=False,
                embedding_batch_size=32,
                embedding_max_length=512,
                embedding_cache_dir="./models",
                embedding_dimension=1024,
                embedding_cache_size=100,
                embedding_cache_ttl=3600,
            )
            from t4dm.embedding.bge_m3 import BGEM3Embedding
            provider = BGEM3Embedding()
            provider._model = mock_flag_model
            provider._initialized = True
            return provider

    @pytest.mark.asyncio
    async def test_embed_hybrid_returns_dense_and_sparse(self, embedding_provider):
        """Test embed_hybrid returns both vector types."""
        texts = ["test query one", "test query two"]
        dense_vecs, sparse_vecs = await embedding_provider.embed_hybrid(texts)

        # Check dense vectors
        assert len(dense_vecs) == 2
        assert len(dense_vecs[0]) == 1024
        assert isinstance(dense_vecs[0][0], float)

        # Check sparse vectors
        assert len(sparse_vecs) == 2
        assert isinstance(sparse_vecs[0], dict)
        assert all(isinstance(k, int) for k in sparse_vecs[0].keys())
        assert all(isinstance(v, float) for v in sparse_vecs[0].values())

    @pytest.mark.asyncio
    async def test_embed_hybrid_empty_input(self, embedding_provider):
        """Test embed_hybrid handles empty input."""
        dense_vecs, sparse_vecs = await embedding_provider.embed_hybrid([])

        assert dense_vecs == []
        assert sparse_vecs == []

    def test_convert_sparse_format(self, embedding_provider):
        """Test _convert_sparse converts to Qdrant format."""
        lexical_weights = [
            {"100": 0.5, "200": 0.3},  # String keys (from JSON)
            {"300": 0.7},
            None,  # Handle None gracefully
        ]

        sparse_vecs = embedding_provider._convert_sparse(lexical_weights)

        assert len(sparse_vecs) == 3
        # First vector
        assert 100 in sparse_vecs[0]
        assert sparse_vecs[0][100] == 0.5
        # Second vector
        assert 300 in sparse_vecs[1]
        # None becomes empty dict
        assert sparse_vecs[2] == {}

    @pytest.mark.asyncio
    async def test_embed_query_hybrid_caches_result(self, embedding_provider):
        """Test embed_query_hybrid uses cache."""
        query = "test query"

        # First call - should hit model
        dense1, sparse1 = await embedding_provider.embed_query_hybrid(query)
        call_count_1 = embedding_provider._model.encode.call_count

        # Second call - should hit cache
        dense2, sparse2 = await embedding_provider.embed_query_hybrid(query)
        call_count_2 = embedding_provider._model.encode.call_count

        # Model should only be called once
        assert call_count_2 == call_count_1
        assert dense1 == dense2
        assert sparse1 == sparse2

    @pytest.mark.asyncio
    async def test_embed_query_hybrid_adds_instruction_prefix(self, embedding_provider):
        """Test embed_query_hybrid adds BGE instruction prefix."""
        query = "find similar documents"
        await embedding_provider.embed_query_hybrid(query)

        # Check the model was called with prefixed query
        call_args = embedding_provider._model.encode.call_args
        texts = call_args[0][0]
        assert texts[0].startswith("Represent this sentence")
        assert query in texts[0]


class TestQdrantStoreHybrid:
    """Tests for QdrantStore hybrid search methods."""

    @pytest.fixture
    def mock_qdrant_client(self):
        """Mock async Qdrant client."""
        from qdrant_client.http.exceptions import UnexpectedResponse
        from unittest.mock import MagicMock
        from httpx import Headers

        client = AsyncMock()
        # UnexpectedResponse triggers collection creation
        client.get_collection.side_effect = UnexpectedResponse(
            status_code=404,
            reason_phrase="Not Found",
            content=b"Collection not found",
            headers=Headers({}),
        )
        client.create_collection = AsyncMock()
        client.upsert = AsyncMock()
        client.query_points = AsyncMock(return_value=MagicMock(points=[]))
        return client

    @pytest.fixture
    def qdrant_store(self, mock_qdrant_client):
        """Create QdrantStore with mocked client."""
        with patch("t4dm.storage.qdrant_store.get_settings") as mock_settings:
            mock_settings.return_value = MagicMock(
                qdrant_url="http://localhost:6333",
                qdrant_api_key=None,
                embedding_dimension=1024,
                qdrant_collection_episodes="ww_episodes",
                qdrant_collection_entities="ww_entities",
                qdrant_collection_procedures="ww_procedures",
            )
            from t4dm.storage.qdrant_store import QdrantStore
            store = QdrantStore()
            store._client = mock_qdrant_client
            return store

    @pytest.mark.asyncio
    async def test_ensure_collection_hybrid(self, qdrant_store, mock_qdrant_client):
        """Test _ensure_collection creates hybrid collection."""
        from qdrant_client.http import models

        await qdrant_store._ensure_collection(mock_qdrant_client, "test_hybrid", hybrid=True)

        mock_qdrant_client.create_collection.assert_called_once()
        call_kwargs = mock_qdrant_client.create_collection.call_args[1]

        # Check vectors_config has named "dense" vector
        assert "dense" in call_kwargs["vectors_config"]
        # Check sparse_vectors_config has "sparse"
        assert "sparse" in call_kwargs["sparse_vectors_config"]

    @pytest.mark.asyncio
    async def test_search_hybrid_builds_prefetch(self, qdrant_store, mock_qdrant_client):
        """Test search_hybrid builds correct prefetch queries."""
        from qdrant_client.http import models

        dense_vec = [0.1] * 1024
        sparse_vec = {100: 0.5, 200: 0.3}

        await qdrant_store.search_hybrid(
            collection="test",
            dense_vector=dense_vec,
            sparse_vector=sparse_vec,
            limit=10,
        )

        mock_qdrant_client.query_points.assert_called_once()
        call_kwargs = mock_qdrant_client.query_points.call_args[1]

        # Check prefetch contains both dense and sparse
        prefetch = call_kwargs["prefetch"]
        assert len(prefetch) == 2

        # Check fusion query uses RRF
        assert isinstance(call_kwargs["query"], models.FusionQuery)

    @pytest.mark.asyncio
    async def test_search_hybrid_handles_empty_sparse(self, qdrant_store, mock_qdrant_client):
        """Test search_hybrid handles empty sparse vector."""
        dense_vec = [0.1] * 1024
        sparse_vec = {}  # Empty sparse

        await qdrant_store.search_hybrid(
            collection="test",
            dense_vector=dense_vec,
            sparse_vector=sparse_vec,
            limit=10,
        )

        call_kwargs = mock_qdrant_client.query_points.call_args[1]
        # Should only have dense prefetch when sparse is empty
        prefetch = call_kwargs["prefetch"]
        assert len(prefetch) == 1

    @pytest.mark.asyncio
    async def test_add_hybrid_creates_points(self, qdrant_store, mock_qdrant_client):
        """Test add_hybrid creates points with named vectors."""
        from qdrant_client.http import models

        ids = ["id1", "id2"]
        dense_vecs = [[0.1] * 1024, [0.2] * 1024]
        sparse_vecs = [{100: 0.5}, {200: 0.3}]
        payloads = [{"content": "a"}, {"content": "b"}]

        await qdrant_store.add_hybrid(
            collection="test",
            ids=ids,
            dense_vectors=dense_vecs,
            sparse_vectors=sparse_vecs,
            payloads=payloads,
        )

        mock_qdrant_client.upsert.assert_called()
        call_kwargs = mock_qdrant_client.upsert.call_args[1]

        points = call_kwargs["points"]
        assert len(points) == 2
        # Check point has named vectors
        assert "dense" in points[0].vector
        assert "sparse" in points[0].vector


class TestEpisodicMemoryHybrid:
    """Tests for EpisodicMemory hybrid integration."""

    @pytest.fixture
    def mock_embedding(self):
        """Mock embedding provider."""
        embedding = AsyncMock()
        embedding.embed_query.return_value = [0.1] * 1024
        embedding.embed_query_hybrid.return_value = ([0.1] * 1024, {100: 0.5, 200: 0.3})
        embedding.embed_hybrid.return_value = ([[0.1] * 1024], [{100: 0.5}])
        return embedding

    @pytest.fixture
    def mock_vector_store(self):
        """Mock vector store."""
        store = AsyncMock()
        store.episodes_collection = "ww_episodes"
        store.add = AsyncMock()
        store.add_hybrid = AsyncMock()
        store.search = AsyncMock(return_value=[])
        store.search_hybrid = AsyncMock(return_value=[])
        store.ensure_hybrid_collection = AsyncMock()
        return store

    @pytest.fixture
    def mock_graph_store(self):
        """Mock graph store."""
        store = AsyncMock()
        store.create_node = AsyncMock()
        return store

    @pytest.fixture
    def episodic_memory(self, mock_embedding, mock_vector_store, mock_graph_store):
        """Create EpisodicMemory with mocked dependencies."""
        with patch("t4dm.memory.episodic.get_settings") as mock_settings, \
             patch("t4dm.memory.episodic.get_embedding_provider") as mock_emb, \
             patch("t4dm.memory.episodic.get_qdrant_store") as mock_qdrant, \
             patch("t4dm.memory.episodic.get_neo4j_store") as mock_neo4j:

            mock_settings.return_value = MagicMock(
                session_id="test",
                embedding_hybrid_enabled=True,
                episodic_weight_semantic=0.4,
                episodic_weight_recency=0.25,
                episodic_weight_outcome=0.2,
                episodic_weight_importance=0.15,
                fsrs_default_stability=1.0,
                fsrs_decay_factor=0.9,
                fsrs_recency_decay=0.1,
                ff_encoder_enabled=False,  # Phase 5: Disable for unit tests
                capsule_layer_enabled=False,  # Phase 6: Disable for unit tests
                capsule_retrieval_enabled=False,  # Phase 6: Disable for unit tests
                embedding_dimension=1024,
            )
            mock_emb.return_value = mock_embedding
            mock_qdrant.return_value = mock_vector_store
            mock_neo4j.return_value = mock_graph_store

            with patch("t4dm.memory.episodic.get_ff_encoder", return_value=None):
                from t4dm.memory.episodic import EpisodicMemory
                memory = EpisodicMemory(session_id="test")
                memory._hybrid_initialized = True
                return memory

    @pytest.mark.asyncio
    async def test_recall_hybrid_uses_hybrid_search(self, episodic_memory):
        """Test recall_hybrid uses hybrid search method."""
        await episodic_memory.recall_hybrid("test query", limit=5)

        # Should use embed_query_hybrid
        episodic_memory.embedding.embed_query_hybrid.assert_called_once_with("test query")
        # Should use search_hybrid
        episodic_memory.vector_store.search_hybrid.assert_called_once()

    @pytest.mark.asyncio
    async def test_recall_hybrid_falls_back_when_not_initialized(self, episodic_memory):
        """Test recall_hybrid falls back to dense when hybrid not initialized."""
        episodic_memory._hybrid_initialized = False

        with patch.object(episodic_memory, "recall", new_callable=AsyncMock) as mock_recall:
            mock_recall.return_value = []
            await episodic_memory.recall_hybrid("test query")
            mock_recall.assert_called_once()

    @pytest.mark.asyncio
    async def test_initialize_creates_hybrid_collection(self, episodic_memory):
        """Test initialize creates hybrid collection when enabled."""
        episodic_memory._hybrid_initialized = False
        await episodic_memory.initialize()

        episodic_memory.vector_store.ensure_hybrid_collection.assert_called_once()


class TestHybridSearchIntegration:
    """Integration-style tests for hybrid search workflow."""

    @pytest.mark.asyncio
    async def test_full_hybrid_workflow(self):
        """Test complete hybrid search workflow with mocks."""
        # This test verifies the full flow from embedding to search

        with patch("t4dm.embedding.bge_m3.get_settings") as mock_emb_settings, \
             patch("t4dm.storage.qdrant_store.get_settings") as mock_store_settings:

            mock_emb_settings.return_value = MagicMock(
                embedding_model="BAAI/bge-m3",
                embedding_device="cpu",
                embedding_use_fp16=False,
                embedding_batch_size=32,
                embedding_max_length=512,
                embedding_cache_dir="./models",
                embedding_dimension=1024,
                embedding_cache_size=100,
                embedding_cache_ttl=3600,
            )

            from t4dm.embedding.bge_m3 import BGEM3Embedding

            # Create embedding provider with mock model
            provider = BGEM3Embedding()
            provider._model = MagicMock()
            provider._model.encode.return_value = {
                "dense_vecs": np.array([[0.1] * 1024]),
                "lexical_weights": [{100: 0.5, 200: 0.3}],
            }
            provider._initialized = True

            # Test hybrid embedding
            dense, sparse = await provider.embed_hybrid(["test content"])

            assert len(dense) == 1
            assert len(sparse) == 1
            assert 100 in sparse[0]

    def test_sparse_vector_format_compatibility(self):
        """Test sparse vector format is compatible with Qdrant."""
        from qdrant_client.http import models

        # Create sparse vector in expected format
        sparse_dict = {100: 0.5, 200: 0.3, 300: 0.2}

        # Should be convertible to Qdrant SparseVector
        qdrant_sparse = models.SparseVector(
            indices=list(sparse_dict.keys()),
            values=list(sparse_dict.values()),
        )

        assert len(qdrant_sparse.indices) == 3
        assert len(qdrant_sparse.values) == 3
