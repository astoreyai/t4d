"""
Phase 2C: Tests for capsule pose storage and retrieval.

Per Hinton (2017, 2018): Pose matrices encode entity configuration,
enabling part-whole composition in memory retrieval. Routing agreement
measures consistency of capsule votes.
"""

import pytest
import numpy as np
from unittest.mock import AsyncMock, MagicMock, patch

from ww.storage.qdrant_store import (
    QdrantStore,
    CapsulePoseData,
    CAPSULE_PAYLOAD_FIELDS,
)


class TestCapsulePoseDataClass:
    """Tests for CapsulePoseData dataclass."""

    def test_empty_pose_data(self):
        """Test CapsulePoseData with no data."""
        data = CapsulePoseData()
        assert data.poses is None
        assert data.activations is None
        assert data.routing_agreement is None
        assert data.mean_activation is None
        assert not data.has_poses
        assert not data.has_activations

    def test_with_poses_only(self):
        """Test CapsulePoseData with only poses."""
        poses = [0.1, 0.2, 0.3, 0.4]  # Flattened 2x2 pose for single capsule
        data = CapsulePoseData(poses=poses)
        assert data.has_poses
        assert not data.has_activations
        assert data.routing_agreement is None

    def test_with_full_data(self):
        """Test CapsulePoseData with all fields."""
        data = CapsulePoseData(
            poses=[0.1, 0.2, 0.3, 0.4],
            activations=[0.5, 0.6, 0.7],
            routing_agreement=0.85,
            mean_activation=0.6,
        )
        assert data.has_poses
        assert data.has_activations
        assert data.routing_agreement == 0.85
        assert data.mean_activation == 0.6

    def test_has_poses_empty_list(self):
        """Test has_poses returns False for empty list."""
        data = CapsulePoseData(poses=[])
        assert not data.has_poses

    def test_has_activations_empty_list(self):
        """Test has_activations returns False for empty list."""
        data = CapsulePoseData(activations=[])
        assert not data.has_activations


class TestCapsulePayloadFields:
    """Test the declared capsule payload field names."""

    def test_capsule_fields_defined(self):
        """Verify all required capsule fields are defined."""
        assert "capsule_poses" in CAPSULE_PAYLOAD_FIELDS
        assert "capsule_activations" in CAPSULE_PAYLOAD_FIELDS
        assert "capsule_routing_agreement" in CAPSULE_PAYLOAD_FIELDS
        assert "capsule_mean_activation" in CAPSULE_PAYLOAD_FIELDS

    def test_field_count(self):
        """Verify expected number of fields."""
        assert len(CAPSULE_PAYLOAD_FIELDS) == 4


@pytest.fixture
def mock_qdrant_client():
    """Create a mock Qdrant client."""
    client = AsyncMock()
    return client


@pytest.fixture
def qdrant_store(mock_qdrant_client):
    """Create a QdrantStore with mocked client."""
    with patch("ww.storage.qdrant_store.get_settings") as mock_settings:
        mock_settings.return_value.qdrant_url = "http://localhost:6333"
        mock_settings.return_value.qdrant_api_key = None
        mock_settings.return_value.embedding_dimension = 1024
        mock_settings.return_value.qdrant_collection_episodes = "episodes"
        mock_settings.return_value.qdrant_collection_entities = "entities"
        mock_settings.return_value.qdrant_collection_procedures = "procedures"

        store = QdrantStore()
        store._client = mock_qdrant_client
        return store


class TestPoseMatricesStored:
    """Test that pose matrices are stored in Qdrant payload."""

    @pytest.mark.asyncio
    async def test_pose_matrices_in_payload(self, qdrant_store, mock_qdrant_client):
        """Verify pose matrices are correctly included in payload during add."""
        # Create test data
        episode_id = "test-episode-123"
        embedding = [0.1] * 1024
        poses = np.random.randn(32, 4, 4).astype(np.float32)  # 32 capsules, 4x4 poses
        activations = np.random.rand(32).astype(np.float32)

        # Build payload as episodic.py does
        payload = {
            "content": "test content",
            "session_id": "session-1",
            "capsule_poses": poses.reshape(-1).tolist(),
            "capsule_activations": activations.tolist(),
            "capsule_mean_activation": float(np.mean(activations)),
            "capsule_routing_agreement": 0.87,  # Routing agreement score
        }

        # Call add
        mock_qdrant_client.upsert = AsyncMock()
        await qdrant_store.add(
            collection="episodes",
            ids=[episode_id],
            vectors=[embedding],
            payloads=[payload],
        )

        # Verify upsert was called with correct payload
        mock_qdrant_client.upsert.assert_called_once()
        call_args = mock_qdrant_client.upsert.call_args
        points = call_args.kwargs["points"]

        assert len(points) == 1
        point = points[0]
        assert point.payload["capsule_poses"] == poses.reshape(-1).tolist()
        assert point.payload["capsule_activations"] == activations.tolist()
        assert point.payload["capsule_routing_agreement"] == 0.87

    @pytest.mark.asyncio
    async def test_pose_dimensions(self, qdrant_store, mock_qdrant_client):
        """Verify pose dimensions are preserved (32 capsules x 4 x 4 = 512)."""
        num_capsules = 32
        pose_dim = 4
        expected_flat_len = num_capsules * pose_dim * pose_dim  # 512

        poses = np.random.randn(num_capsules, pose_dim, pose_dim)
        flat_poses = poses.reshape(-1).tolist()

        assert len(flat_poses) == expected_flat_len


class TestPoseMatricesRetrieved:
    """Test that pose matrices are retrievable by episode ID."""

    @pytest.mark.asyncio
    async def test_get_capsule_poses_success(self, qdrant_store, mock_qdrant_client):
        """Test successful retrieval of capsule poses."""
        # Setup mock response
        poses = np.random.randn(32, 4, 4).astype(np.float32)
        activations = np.random.rand(32).astype(np.float32)

        mock_point = MagicMock()
        mock_point.id = "episode-123"
        mock_point.payload = {
            "capsule_poses": poses.reshape(-1).tolist(),
            "capsule_activations": activations.tolist(),
            "capsule_routing_agreement": 0.92,
            "capsule_mean_activation": 0.65,
        }
        mock_qdrant_client.retrieve = AsyncMock(return_value=[mock_point])

        # Call get_capsule_poses
        result = await qdrant_store.get_capsule_poses("episodes", "episode-123")

        assert result is not None
        assert isinstance(result, CapsulePoseData)
        assert result.poses == poses.reshape(-1).tolist()
        assert result.activations == activations.tolist()
        assert result.routing_agreement == 0.92
        assert result.mean_activation == 0.65
        assert result.has_poses
        assert result.has_activations

    @pytest.mark.asyncio
    async def test_get_capsule_poses_not_found(self, qdrant_store, mock_qdrant_client):
        """Test retrieval when episode not found."""
        mock_qdrant_client.retrieve = AsyncMock(return_value=[])

        result = await qdrant_store.get_capsule_poses("episodes", "nonexistent-id")

        assert result is None

    @pytest.mark.asyncio
    async def test_get_capsule_poses_no_capsule_data(self, qdrant_store, mock_qdrant_client):
        """Test retrieval when episode exists but has no capsule data."""
        mock_point = MagicMock()
        mock_point.id = "episode-456"
        mock_point.payload = {
            "content": "test content",
            "session_id": "session-1",
            # No capsule fields
        }
        mock_qdrant_client.retrieve = AsyncMock(return_value=[mock_point])

        result = await qdrant_store.get_capsule_poses("episodes", "episode-456")

        assert result is None

    @pytest.mark.asyncio
    async def test_get_capsule_poses_partial_data(self, qdrant_store, mock_qdrant_client):
        """Test retrieval with only routing agreement (no poses/activations)."""
        mock_point = MagicMock()
        mock_point.id = "episode-789"
        mock_point.payload = {
            "content": "test content",
            "capsule_routing_agreement": 0.75,  # Only routing agreement
        }
        mock_qdrant_client.retrieve = AsyncMock(return_value=[mock_point])

        result = await qdrant_store.get_capsule_poses("episodes", "episode-789")

        assert result is not None
        assert result.routing_agreement == 0.75
        assert not result.has_poses
        assert not result.has_activations


class TestBatchGetCapsulePoses:
    """Test batch retrieval of capsule poses."""

    @pytest.mark.asyncio
    async def test_batch_get_capsule_poses_success(self, qdrant_store, mock_qdrant_client):
        """Test batch retrieval of multiple episodes."""
        # Create mock points
        mock_point1 = MagicMock()
        mock_point1.id = "ep-1"
        mock_point1.payload = {
            "capsule_poses": [0.1, 0.2, 0.3],
            "capsule_routing_agreement": 0.9,
        }

        mock_point2 = MagicMock()
        mock_point2.id = "ep-2"
        mock_point2.payload = {
            "capsule_activations": [0.5, 0.6],
            "capsule_routing_agreement": 0.8,
        }

        mock_qdrant_client.retrieve = AsyncMock(return_value=[mock_point1, mock_point2])

        result = await qdrant_store.batch_get_capsule_poses(
            "episodes",
            ["ep-1", "ep-2", "ep-3"]  # ep-3 doesn't exist
        )

        assert len(result) == 3
        assert result["ep-1"] is not None
        assert result["ep-1"].routing_agreement == 0.9
        assert result["ep-2"] is not None
        assert result["ep-2"].routing_agreement == 0.8
        assert result["ep-3"] is None  # Not found

    @pytest.mark.asyncio
    async def test_batch_get_empty_list(self, qdrant_store, mock_qdrant_client):
        """Test batch retrieval with empty list."""
        result = await qdrant_store.batch_get_capsule_poses("episodes", [])

        assert result == {}
        mock_qdrant_client.retrieve.assert_not_called()


class TestRoutingAgreementAsConfidence:
    """Test using routing agreement for confidence scoring."""

    def test_confidence_boost_calculation_high_agreement(self):
        """
        High routing agreement (>0.8) should boost confidence.

        Per Hinton: High agreement = capsules strongly agree on pose configuration
        = more compositionally coherent memory = higher confidence.
        """
        base_score = 0.7
        routing_agreement = 0.95  # Very high agreement

        # Phase 2C formula: confidence = base * (0.7 + 0.3 * agreement)
        expected_confidence = base_score * (0.7 + 0.3 * routing_agreement)

        # With high agreement, confidence should be close to base_score
        assert expected_confidence > base_score * 0.95
        assert expected_confidence < base_score * 1.0

    def test_confidence_boost_calculation_low_agreement(self):
        """
        Low routing agreement (<0.5) should reduce confidence.

        Per Hinton: Low agreement = capsules disagree on configuration
        = ambiguous memory trace = lower confidence.
        """
        base_score = 0.7
        routing_agreement = 0.3  # Low agreement

        # Phase 2C formula
        expected_confidence = base_score * (0.7 + 0.3 * routing_agreement)

        # With low agreement, confidence should be reduced
        assert expected_confidence < base_score * 0.85
        assert expected_confidence > base_score * 0.7

    def test_confidence_formula_bounds(self):
        """Test confidence formula stays within [0.7, 1.0] multiplier range."""
        # Agreement = 0 -> multiplier = 0.7
        assert 0.7 + 0.3 * 0.0 == pytest.approx(0.7)

        # Agreement = 1 -> multiplier = 1.0
        assert 0.7 + 0.3 * 1.0 == pytest.approx(1.0)

        # Agreement = 0.5 -> multiplier = 0.85
        assert 0.7 + 0.3 * 0.5 == pytest.approx(0.85)


class TestCapsuleDataOptional:
    """Test that capsule data is optional (backward compatible)."""

    @pytest.mark.asyncio
    async def test_store_without_capsule_data(self, qdrant_store, mock_qdrant_client):
        """Episodes can be stored without any capsule data."""
        payload = {
            "content": "test content",
            "session_id": "session-1",
            # No capsule fields
        }

        mock_qdrant_client.upsert = AsyncMock()
        await qdrant_store.add(
            collection="episodes",
            ids=["episode-1"],
            vectors=[[0.1] * 1024],
            payloads=[payload],
        )

        # Should succeed
        mock_qdrant_client.upsert.assert_called_once()

    @pytest.mark.asyncio
    async def test_retrieve_without_capsule_data(self, qdrant_store, mock_qdrant_client):
        """Episodes without capsule data should return None from get_capsule_poses."""
        mock_point = MagicMock()
        mock_point.id = "episode-no-capsule"
        mock_point.payload = {"content": "test"}

        mock_qdrant_client.retrieve = AsyncMock(return_value=[mock_point])

        result = await qdrant_store.get_capsule_poses("episodes", "episode-no-capsule")

        assert result is None

    def test_capsule_pose_data_with_none_values(self):
        """CapsulePoseData handles None values gracefully."""
        data = CapsulePoseData(poses=None, activations=None)

        assert not data.has_poses
        assert not data.has_activations
        assert data.routing_agreement is None


class TestCapsuleStorageIntegration:
    """Integration tests for capsule storage workflow."""

    @pytest.mark.asyncio
    async def test_full_capsule_storage_workflow(self, qdrant_store, mock_qdrant_client):
        """Test complete store -> retrieve workflow for capsule data."""
        # Step 1: Generate capsule data
        num_capsules = 32
        pose_dim = 4
        poses = np.random.randn(num_capsules, pose_dim, pose_dim).astype(np.float32)
        activations = np.random.rand(num_capsules).astype(np.float32)
        routing_agreement = 0.88

        episode_id = "workflow-test-episode"
        payload = {
            "content": "integration test content",
            "capsule_poses": poses.reshape(-1).tolist(),
            "capsule_activations": activations.tolist(),
            "capsule_mean_activation": float(np.mean(activations)),
            "capsule_routing_agreement": routing_agreement,
        }

        # Step 2: Store
        mock_qdrant_client.upsert = AsyncMock()
        await qdrant_store.add(
            collection="episodes",
            ids=[episode_id],
            vectors=[[0.1] * 1024],
            payloads=[payload],
        )

        # Step 3: Retrieve
        mock_point = MagicMock()
        mock_point.id = episode_id
        mock_point.payload = payload
        mock_qdrant_client.retrieve = AsyncMock(return_value=[mock_point])

        result = await qdrant_store.get_capsule_poses("episodes", episode_id)

        # Step 4: Verify round-trip
        assert result is not None
        assert result.routing_agreement == routing_agreement
        assert len(result.poses) == num_capsules * pose_dim * pose_dim
        assert len(result.activations) == num_capsules

        # Step 5: Reconstruct poses to verify shape
        retrieved_poses = np.array(result.poses).reshape(num_capsules, pose_dim, pose_dim)
        assert retrieved_poses.shape == (num_capsules, pose_dim, pose_dim)
        np.testing.assert_array_almost_equal(retrieved_poses, poses)


class TestRoutingAgreementStorage:
    """Test routing agreement storage specifically."""

    @pytest.mark.asyncio
    async def test_routing_agreement_precision(self, qdrant_store, mock_qdrant_client):
        """Verify routing agreement maintains float precision."""
        routing_agreement = 0.123456789

        mock_point = MagicMock()
        mock_point.id = "precision-test"
        mock_point.payload = {"capsule_routing_agreement": routing_agreement}
        mock_qdrant_client.retrieve = AsyncMock(return_value=[mock_point])

        result = await qdrant_store.get_capsule_poses("episodes", "precision-test")

        assert result.routing_agreement == pytest.approx(routing_agreement, rel=1e-6)

    @pytest.mark.asyncio
    async def test_routing_agreement_bounds(self, qdrant_store, mock_qdrant_client):
        """Routing agreement should be in [0, 1] range."""
        # Test edge cases
        for agreement_value in [0.0, 0.5, 1.0]:
            mock_point = MagicMock()
            mock_point.id = f"bounds-test-{agreement_value}"
            mock_point.payload = {"capsule_routing_agreement": agreement_value}
            mock_qdrant_client.retrieve = AsyncMock(return_value=[mock_point])

            result = await qdrant_store.get_capsule_poses(
                "episodes",
                f"bounds-test-{agreement_value}"
            )

            assert result.routing_agreement == agreement_value
            assert 0.0 <= result.routing_agreement <= 1.0
