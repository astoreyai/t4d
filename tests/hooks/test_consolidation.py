"""Tests for consolidation hooks module."""

import pytest
from datetime import datetime

from t4dm.hooks.consolidation import (
    ConsolidationHook,
    PreConsolidateHook,
    PostConsolidateHook,
    DuplicateFoundHook,
    ClusterFormHook,
    EntityExtractedHook,
    ConsolidationMetricsHook,
    DuplicateMergeHook,
    ClusterAnalysisHook,
    EntityValidationHook,
    ConsolidationProgressHook,
)
from t4dm.hooks.base import HookContext, HookPhase, HookPriority


class TestConsolidationHook:
    """Tests for ConsolidationHook base class."""

    def test_initialization(self):
        """Test hook initialization."""
        hook = PreConsolidateHook(
            name="test",
            consolidation_type="deep",
        )
        assert hook.name == "test"
        assert hook.consolidation_type == "deep"

    def test_should_execute_with_matching_type(self):
        """Test should_execute with matching consolidation type."""
        hook = PreConsolidateHook(name="test", consolidation_type="deep")
        ctx = HookContext(
            phase=HookPhase.PRE,
            operation="consolidate",
            input_data={"consolidation_type": "deep"},
        )
        assert hook.should_execute(ctx) is True

    def test_should_execute_with_non_matching_type(self):
        """Test should_execute with non-matching type."""
        hook = PreConsolidateHook(name="test", consolidation_type="deep")
        ctx = HookContext(
            phase=HookPhase.PRE,
            operation="consolidate",
            input_data={"consolidation_type": "light"},
        )
        assert hook.should_execute(ctx) is False


class TestPreConsolidateHook:
    """Tests for PreConsolidateHook class."""

    @pytest.mark.asyncio
    async def test_execute(self):
        """Test pre-consolidation execution."""
        hook = PreConsolidateHook(name="pre_test")
        ctx = HookContext(
            phase=HookPhase.PRE,
            operation="consolidate",
            input_data={
                "consolidation_type": "deep",
                "session_filter": "session-123",
                "dry_run": False,
            },
        )
        result = await hook.execute(ctx)
        assert result is ctx

    @pytest.mark.asyncio
    async def test_execute_dry_run(self):
        """Test dry run execution."""
        hook = PreConsolidateHook(name="pre_test")
        ctx = HookContext(
            phase=HookPhase.PRE,
            operation="consolidate",
            input_data={"consolidation_type": "light", "dry_run": True},
        )
        result = await hook.execute(ctx)
        assert result is ctx


class TestPostConsolidateHook:
    """Tests for PostConsolidateHook class."""

    @pytest.mark.asyncio
    async def test_execute(self):
        """Test post-consolidation execution."""
        hook = PostConsolidateHook(name="post_test")
        ctx = HookContext(
            phase=HookPhase.POST,
            operation="consolidate",
            output_data={
                "episodes_processed": 100,
                "duplicates_removed": 10,
                "entities_extracted": 25,
                "clusters_formed": 5,
                "duration_ms": 1500,
            },
        )
        result = await hook.execute(ctx)
        assert result is ctx

    @pytest.mark.asyncio
    async def test_execute_no_output(self):
        """Test execution without output data."""
        hook = PostConsolidateHook(name="post_test")
        ctx = HookContext(
            phase=HookPhase.POST,
            operation="consolidate",
        )
        result = await hook.execute(ctx)
        assert result is ctx


class TestDuplicateFoundHook:
    """Tests for DuplicateFoundHook class."""

    @pytest.mark.asyncio
    async def test_execute(self):
        """Test duplicate detection execution."""
        hook = DuplicateFoundHook(name="dup_test")
        ctx = HookContext(
            phase=HookPhase.ON,
            operation="duplicate_found",
            input_data={
                "memory_id_1": "uuid-1",
                "memory_id_2": "uuid-2",
                "similarity": 0.95,
                "memory_type": "episodic",
                "merge_strategy": "keep_first",
            },
        )
        result = await hook.execute(ctx)
        assert result is ctx


class TestClusterFormHook:
    """Tests for ClusterFormHook class."""

    @pytest.mark.asyncio
    async def test_execute(self):
        """Test cluster formation execution."""
        hook = ClusterFormHook(name="cluster_test")
        ctx = HookContext(
            phase=HookPhase.ON,
            operation="cluster_form",
            input_data={
                "cluster_id": "cluster-123",
                "memory_ids": ["m1", "m2", "m3"],
                "cluster_size": 3,
                "coherence": 0.85,
            },
        )
        result = await hook.execute(ctx)
        assert result is ctx


class TestEntityExtractedHook:
    """Tests for EntityExtractedHook class."""

    @pytest.mark.asyncio
    async def test_execute(self):
        """Test entity extraction execution."""
        hook = EntityExtractedHook(name="entity_test")
        ctx = HookContext(
            phase=HookPhase.ON,
            operation="entity_extracted",
            input_data={
                "entity_id": "entity-123",
                "entity_type": "person",
                "entity_name": "John Doe",
                "confidence": 0.92,
                "source_episode_id": "episode-456",
                "mentions": 3,
            },
        )
        result = await hook.execute(ctx)
        assert result is ctx


class TestConsolidationMetricsHook:
    """Tests for ConsolidationMetricsHook example implementation."""

    def test_initialization(self):
        """Test hook initialization."""
        hook = ConsolidationMetricsHook()
        assert hook.name == "consolidation_metrics"
        assert hook.consolidation_history == []

    @pytest.mark.asyncio
    async def test_metrics_collected(self):
        """Test metrics are collected."""
        hook = ConsolidationMetricsHook()
        ctx = HookContext(
            phase=HookPhase.POST,
            operation="consolidate",
            input_data={"consolidation_type": "deep"},
            output_data={
                "episodes_processed": 50,
                "duplicates_removed": 5,
                "entities_extracted": 10,
                "clusters_formed": 2,
            },
        )
        await hook.execute(ctx)

        assert len(hook.consolidation_history) == 1
        assert hook.consolidation_history[0]["episodes_processed"] == 50

    def test_get_stats(self):
        """Test getting consolidation statistics."""
        hook = ConsolidationMetricsHook()
        hook.consolidation_history = [
            {"success": True, "episodes_processed": 100, "duplicates_removed": 10,
             "entities_extracted": 20, "clusters_formed": 5},
            {"success": True, "episodes_processed": 50, "duplicates_removed": 5,
             "entities_extracted": 10, "clusters_formed": 2},
        ]

        stats = hook.get_stats()
        assert "consolidation" in stats
        assert stats["consolidation"]["total_runs"] == 2
        assert stats["consolidation"]["success_rate"] == 1.0
        assert stats["consolidation"]["total_episodes"] == 150


class TestDuplicateMergeHook:
    """Tests for DuplicateMergeHook example implementation."""

    def test_initialization(self):
        """Test hook initialization."""
        hook = DuplicateMergeHook(
            similarity_threshold=0.9,
            prefer_newer=True,
        )
        assert hook.name == "duplicate_merge"
        assert hook.similarity_threshold == 0.9
        assert hook.prefer_newer is True

    @pytest.mark.asyncio
    async def test_high_similarity_merge(self):
        """Test high similarity triggers merge strategy."""
        hook = DuplicateMergeHook(similarity_threshold=0.9)
        ctx = HookContext(
            phase=HookPhase.ON,
            operation="duplicate_found",
            input_data={"similarity": 0.95},
        )
        await hook.execute(ctx)
        assert ctx.input_data["merge_strategy"] == "merge"

    @pytest.mark.asyncio
    async def test_low_similarity_prefer_newer(self):
        """Test low similarity uses prefer_newer strategy."""
        hook = DuplicateMergeHook(similarity_threshold=0.9, prefer_newer=True)
        ctx = HookContext(
            phase=HookPhase.ON,
            operation="duplicate_found",
            input_data={"similarity": 0.8},
        )
        await hook.execute(ctx)
        assert ctx.input_data["merge_strategy"] == "keep_second"

    @pytest.mark.asyncio
    async def test_low_similarity_prefer_older(self):
        """Test low similarity uses keep_first strategy."""
        hook = DuplicateMergeHook(similarity_threshold=0.9, prefer_newer=False)
        ctx = HookContext(
            phase=HookPhase.ON,
            operation="duplicate_found",
            input_data={"similarity": 0.8},
        )
        await hook.execute(ctx)
        assert ctx.input_data["merge_strategy"] == "keep_first"


class TestClusterAnalysisHook:
    """Tests for ClusterAnalysisHook example implementation."""

    def test_initialization(self):
        """Test hook initialization."""
        hook = ClusterAnalysisHook(min_coherence=0.8)
        assert hook.name == "cluster_analysis"
        assert hook.min_coherence == 0.8
        assert hook.cluster_stats == {}

    @pytest.mark.asyncio
    async def test_high_coherence_cluster(self):
        """Test high coherence cluster is marked high quality."""
        hook = ClusterAnalysisHook(min_coherence=0.7)
        ctx = HookContext(
            phase=HookPhase.ON,
            operation="cluster_form",
            input_data={
                "cluster_id": "cluster-1",
                "cluster_size": 5,
                "coherence": 0.85,
            },
        )
        await hook.execute(ctx)

        assert ctx.metadata["cluster_quality"] == "high"
        assert hook.cluster_stats["cluster-1"]["quality"] == "high"

    @pytest.mark.asyncio
    async def test_low_coherence_cluster(self):
        """Test low coherence cluster is marked low quality."""
        hook = ClusterAnalysisHook(min_coherence=0.7)
        ctx = HookContext(
            phase=HookPhase.ON,
            operation="cluster_form",
            input_data={
                "cluster_id": "cluster-2",
                "cluster_size": 3,
                "coherence": 0.5,
            },
        )
        await hook.execute(ctx)

        assert ctx.metadata["cluster_quality"] == "low"


class TestEntityValidationHook:
    """Tests for EntityValidationHook example implementation."""

    def test_initialization(self):
        """Test hook initialization."""
        hook = EntityValidationHook(min_confidence=0.8)
        assert hook.name == "entity_validation"
        assert hook.priority == HookPriority.CRITICAL
        assert hook.min_confidence == 0.8
        assert hook.validated_entities == set()

    @pytest.mark.asyncio
    async def test_valid_entity(self):
        """Test validation passes for valid entity."""
        hook = EntityValidationHook(min_confidence=0.7)
        ctx = HookContext(
            phase=HookPhase.ON,
            operation="entity_extracted",
            input_data={
                "entity_name": "John Doe",
                "entity_type": "person",
                "confidence": 0.9,
            },
        )
        await hook.execute(ctx)

        assert ctx.metadata["validation_passed"] is True
        assert "John Doe" in hook.validated_entities

    @pytest.mark.asyncio
    async def test_low_confidence_fails(self):
        """Test validation fails for low confidence."""
        hook = EntityValidationHook(min_confidence=0.8)
        ctx = HookContext(
            phase=HookPhase.ON,
            operation="entity_extracted",
            input_data={
                "entity_name": "Unknown",
                "entity_type": "concept",
                "confidence": 0.5,
            },
        )
        await hook.execute(ctx)

        assert ctx.metadata["validation_passed"] is False

    @pytest.mark.asyncio
    async def test_invalid_name_fails(self):
        """Test validation fails for invalid entity name."""
        hook = EntityValidationHook()
        ctx = HookContext(
            phase=HookPhase.ON,
            operation="entity_extracted",
            input_data={
                "entity_name": "x",  # Too short
                "confidence": 0.9,
            },
        )
        await hook.execute(ctx)

        assert ctx.metadata["validation_passed"] is False


class TestConsolidationProgressHook:
    """Tests for ConsolidationProgressHook example implementation."""

    def test_initialization(self):
        """Test hook initialization."""
        hook = ConsolidationProgressHook(report_interval=50)
        assert hook.name == "consolidation_progress"
        assert hook.priority == HookPriority.LOW
        assert hook.report_interval == 50
        assert hook.processed_count == 0

    @pytest.mark.asyncio
    async def test_progress_tracking(self):
        """Test progress is tracked."""
        hook = ConsolidationProgressHook(report_interval=5)

        for i in range(10):
            ctx = HookContext(
                phase=HookPhase.PRE,
                operation="consolidate",
                input_data={"consolidation_type": "light"},
            )
            await hook.execute(ctx)

        assert hook.processed_count == 10
