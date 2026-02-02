"""
Consolidation module lifecycle hooks for T4DM.

Provides hooks for:
- Consolidation lifecycle (pre/post)
- Duplicate detection
- Cluster formation
- Entity extraction
"""

import logging

from t4dm.hooks.base import Hook, HookContext, HookPriority

logger = logging.getLogger(__name__)


class ConsolidationHook(Hook):
    """Base class for consolidation-related hooks."""

    def __init__(
        self,
        name: str,
        priority: HookPriority = HookPriority.NORMAL,
        enabled: bool = True,
        consolidation_type: str | None = None,
    ):
        """
        Initialize consolidation hook.

        Args:
            name: Hook identifier
            priority: Execution priority
            enabled: Whether hook is active
            consolidation_type: Filter by type (light, deep, skill, all)
        """
        super().__init__(name, priority, enabled)
        self.consolidation_type = consolidation_type

    def should_execute(self, context: HookContext) -> bool:
        """Check if hook should execute based on consolidation type."""
        if not super().should_execute(context):
            return False

        if self.consolidation_type:
            cons_type = context.input_data.get("consolidation_type")
            return cons_type == self.consolidation_type

        return True


class PreConsolidateHook(ConsolidationHook):
    """
    Hook executed before consolidation starts.

    Use for:
    - Pre-consolidation validation
    - Resource preparation
    - Metrics initialization
    """

    async def execute(self, context: HookContext) -> HookContext:
        """
        Execute pre-consolidation hook.

        Context data:
        - input_data["consolidation_type"]: light/deep/skill/all
        - input_data["session_filter"]: Session filter (optional)
        - input_data["dry_run"]: Whether this is a dry run

        Returns:
            Modified context
        """
        cons_type = context.input_data.get("consolidation_type", "unknown")
        dry_run = context.input_data.get("dry_run", False)

        logger.info(
            f"[{self.name}] Pre-consolidation: {cons_type} "
            f"{'(dry run)' if dry_run else ''}"
        )

        return context


class PostConsolidateHook(ConsolidationHook):
    """
    Hook executed after consolidation completes.

    Use for:
    - Result validation
    - Metrics reporting
    - Cleanup
    """

    async def execute(self, context: HookContext) -> HookContext:
        """
        Execute post-consolidation hook.

        Context data (output):
        - output_data["episodes_processed"]: Number of episodes processed
        - output_data["duplicates_removed"]: Number of duplicates removed
        - output_data["entities_extracted"]: Number of entities extracted
        - output_data["clusters_formed"]: Number of clusters formed
        - output_data["duration_ms"]: Consolidation duration

        Returns:
            Modified context
        """
        if not context.output_data:
            return context

        episodes = context.output_data.get("episodes_processed", 0)
        duplicates = context.output_data.get("duplicates_removed", 0)
        entities = context.output_data.get("entities_extracted", 0)
        clusters = context.output_data.get("clusters_formed", 0)

        logger.info(
            f"[{self.name}] Post-consolidation results: "
            f"{episodes} episodes, {duplicates} duplicates removed, "
            f"{entities} entities, {clusters} clusters"
        )

        return context


class DuplicateFoundHook(ConsolidationHook):
    """
    Hook executed when duplicate memory is detected.

    Use for:
    - Duplicate logging
    - Custom merge logic
    - Deduplication metrics
    """

    async def execute(self, context: HookContext) -> HookContext:
        """
        Execute duplicate detection hook.

        Context data:
        - input_data["memory_id_1"]: First memory UUID
        - input_data["memory_id_2"]: Second memory UUID
        - input_data["similarity"]: Similarity score [0-1]
        - input_data["memory_type"]: episodic/semantic/procedural
        - input_data["merge_strategy"]: "keep_first", "keep_second", "merge"

        Returns:
            Modified context
        """
        mem_id_1 = context.input_data.get("memory_id_1")
        mem_id_2 = context.input_data.get("memory_id_2")
        similarity = context.input_data.get("similarity", 0.0)
        mem_type = context.input_data.get("memory_type", "unknown")

        logger.info(
            f"[{self.name}] Duplicate detected: {mem_type} "
            f"{mem_id_1} ≈ {mem_id_2} (similarity: {similarity:.3f})"
        )

        return context


class ClusterFormHook(ConsolidationHook):
    """
    Hook executed when memory cluster is formed.

    Use for:
    - Cluster validation
    - Pattern extraction
    - Knowledge graph updates
    """

    async def execute(self, context: HookContext) -> HookContext:
        """
        Execute cluster formation hook.

        Context data:
        - input_data["cluster_id"]: Cluster identifier
        - input_data["memory_ids"]: List of memory UUIDs in cluster
        - input_data["cluster_size"]: Number of memories
        - input_data["centroid"]: Cluster centroid embedding
        - input_data["coherence"]: Cluster coherence score

        Returns:
            Modified context
        """
        cluster_id = context.input_data.get("cluster_id")
        cluster_size = context.input_data.get("cluster_size", 0)
        coherence = context.input_data.get("coherence", 0.0)

        logger.info(
            f"[{self.name}] Cluster formed: {cluster_id} "
            f"({cluster_size} memories, coherence: {coherence:.3f})"
        )

        return context


class EntityExtractedHook(ConsolidationHook):
    """
    Hook executed when entity is extracted from episode.

    Use for:
    - Entity validation
    - Knowledge graph integration
    - Entity linking
    """

    async def execute(self, context: HookContext) -> HookContext:
        """
        Execute entity extraction hook.

        Context data:
        - input_data["entity_id"]: Entity UUID
        - input_data["entity_type"]: person/organization/concept/etc.
        - input_data["entity_name"]: Entity name
        - input_data["confidence"]: Extraction confidence [0-1]
        - input_data["source_episode_id"]: Episode UUID
        - input_data["mentions"]: Number of mentions

        Returns:
            Modified context
        """
        entity_name = context.input_data.get("entity_name", "unknown")
        entity_type = context.input_data.get("entity_type", "unknown")
        confidence = context.input_data.get("confidence", 0.0)

        logger.info(
            f"[{self.name}] Entity extracted: {entity_name} "
            f"(type: {entity_type}, confidence: {confidence:.3f})"
        )

        return context


# Example implementations

class ConsolidationMetricsHook(PostConsolidateHook):
    """Example: Collect comprehensive consolidation metrics."""

    def __init__(self):
        super().__init__(
            name="consolidation_metrics",
            priority=HookPriority.NORMAL,
        )
        self.consolidation_history: list[dict] = []

    async def execute(self, context: HookContext) -> HookContext:
        if not context.output_data:
            return context

        # Collect metrics
        metrics = {
            "timestamp": context.start_time.isoformat(),
            "type": context.input_data.get("consolidation_type"),
            "duration_ms": context.elapsed_ms(),
            "episodes_processed": context.output_data.get("episodes_processed", 0),
            "duplicates_removed": context.output_data.get("duplicates_removed", 0),
            "entities_extracted": context.output_data.get("entities_extracted", 0),
            "clusters_formed": context.output_data.get("clusters_formed", 0),
            "success": context.error is None,
        }

        self.consolidation_history.append(metrics)
        logger.info(f"Consolidation metrics: {metrics}")

        return context

    def get_stats(self) -> dict:
        """Get consolidation statistics."""
        stats = super().get_stats()

        if self.consolidation_history:
            total = len(self.consolidation_history)
            successful = sum(1 for m in self.consolidation_history if m["success"])

            stats["consolidation"] = {
                "total_runs": total,
                "successful_runs": successful,
                "success_rate": successful / total,
                "total_episodes": sum(m["episodes_processed"] for m in self.consolidation_history),
                "total_duplicates": sum(m["duplicates_removed"] for m in self.consolidation_history),
                "total_entities": sum(m["entities_extracted"] for m in self.consolidation_history),
                "total_clusters": sum(m["clusters_formed"] for m in self.consolidation_history),
            }

        return stats


class DuplicateMergeHook(DuplicateFoundHook):
    """Example: Implement custom duplicate merge logic."""

    def __init__(
        self,
        similarity_threshold: float = 0.95,
        prefer_newer: bool = True,
    ):
        super().__init__(
            name="duplicate_merge",
            priority=HookPriority.NORMAL,
        )
        self.similarity_threshold = similarity_threshold
        self.prefer_newer = prefer_newer

    async def execute(self, context: HookContext) -> HookContext:
        similarity = context.input_data.get("similarity", 0.0)

        if similarity >= self.similarity_threshold:
            # High similarity - merge
            strategy = "merge"
            logger.info(
                f"Merging duplicates (similarity: {similarity:.3f} >= "
                f"{self.similarity_threshold})"
            )
        elif self.prefer_newer:
            strategy = "keep_second"
        else:
            strategy = "keep_first"

        # Update merge strategy
        if context.input_data:
            context.input_data["merge_strategy"] = strategy

        return context


class ClusterAnalysisHook(ClusterFormHook):
    """Example: Analyze cluster properties and extract insights."""

    def __init__(self, min_coherence: float = 0.7):
        super().__init__(
            name="cluster_analysis",
            priority=HookPriority.NORMAL,
        )
        self.min_coherence = min_coherence
        self.cluster_stats: dict[str, dict] = {}

    async def execute(self, context: HookContext) -> HookContext:
        cluster_id = context.input_data.get("cluster_id")
        cluster_size = context.input_data.get("cluster_size", 0)
        coherence = context.input_data.get("coherence", 0.0)

        # Analyze cluster quality
        quality = "high" if coherence >= self.min_coherence else "low"

        # Store cluster stats
        self.cluster_stats[cluster_id] = {
            "size": cluster_size,
            "coherence": coherence,
            "quality": quality,
        }

        if quality == "low":
            logger.warning(
                f"Low coherence cluster: {cluster_id} "
                f"(coherence: {coherence:.3f} < {self.min_coherence})"
            )

        context.metadata["cluster_quality"] = quality
        return context


class EntityValidationHook(EntityExtractedHook):
    """Example: Validate extracted entities against knowledge base."""

    def __init__(self, min_confidence: float = 0.7):
        super().__init__(
            name="entity_validation",
            priority=HookPriority.CRITICAL,
        )
        self.min_confidence = min_confidence
        self.validated_entities: set[str] = set()

    async def execute(self, context: HookContext) -> HookContext:
        entity_name = context.input_data.get("entity_name", "")
        entity_type = context.input_data.get("entity_type", "")
        confidence = context.input_data.get("confidence", 0.0)

        # Validate confidence threshold
        if confidence < self.min_confidence:
            logger.warning(
                f"Low confidence entity: {entity_name} "
                f"(confidence: {confidence:.3f} < {self.min_confidence})"
            )
            context.metadata["validation_passed"] = False
            return context

        # Validate entity name
        if not entity_name or len(entity_name) < 2:
            logger.warning(f"Invalid entity name: '{entity_name}'")
            context.metadata["validation_passed"] = False
            return context

        # Track validated entities
        self.validated_entities.add(entity_name)
        context.metadata["validation_passed"] = True

        logger.debug(f"Entity validated: {entity_name} ({entity_type})")
        return context


class ConsolidationProgressHook(PreConsolidateHook):
    """Example: Track and report consolidation progress."""

    def __init__(self, report_interval: int = 100):
        super().__init__(
            name="consolidation_progress",
            priority=HookPriority.LOW,
        )
        self.report_interval = report_interval
        self.processed_count = 0

    async def execute(self, context: HookContext) -> HookContext:

        self.processed_count += 1

        if self.processed_count % self.report_interval == 0:
            elapsed = context.elapsed_ms()
            rate = self.processed_count / (elapsed / 1000) if elapsed > 0 else 0

            logger.info(
                f"Consolidation progress: {self.processed_count} processed "
                f"({rate:.1f} items/sec)"
            )

        return context
