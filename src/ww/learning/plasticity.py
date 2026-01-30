"""
Biological Plasticity Mechanisms for World Weaver.

Implements biologically-inspired synaptic plasticity rules that complement
the Hebbian learning already present in the system.

Mechanisms:
1. LTD (Long-Term Depression) - Competitive weakening of non-co-activated connections
2. Homeostatic Synaptic Scaling - Global normalization to prevent runaway excitation
3. Metaplasticity - Learning rate adaptation based on activity history
4. Synaptic Tagging - Mark synapses for consolidation
5. EWC (Elastic Weight Consolidation) - Protect important weights from catastrophic forgetting

Biological Basis:
- LTD: BCM theory - synapses that fire below threshold weaken
- Homeostasis: TNFα-mediated scaling maintains network stability
- Metaplasticity: Bienenstock-Cooper-Munro sliding threshold
- Tagging: Synaptic tag-and-capture model (Frey & Morris, 1997)
- EWC: Analogous to synaptic consolidation in neocortex (slow weight changes)

References:
- Turrigiano (2008) "The self-tuning neuron"
- Abraham & Bear (1996) "Metaplasticity"
- Redondo & Morris (2011) "Making memories last"
- Kirkpatrick et al. (2017) "Overcoming catastrophic forgetting in neural networks"
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum
from typing import TYPE_CHECKING, Protocol

logger = logging.getLogger(__name__)

# Optional torch import for EWC
try:
    import torch
    import torch.nn as nn
    TORCH_AVAILABLE = True
except ImportError:
    TORCH_AVAILABLE = False
    logger.debug("PyTorch not available, EWC disabled")

if TYPE_CHECKING:
    from torch.utils.data import DataLoader


class PlasticityType(str, Enum):
    """Types of synaptic plasticity."""
    LTP = "ltp"  # Long-Term Potentiation
    LTD = "ltd"  # Long-Term Depression
    HOMEOSTATIC = "homeostatic"  # Synaptic scaling
    METAPLASTIC = "metaplastic"  # Threshold adjustment


@dataclass
class PlasticityEvent:
    """Record of a plasticity event."""
    event_type: PlasticityType
    source_id: str
    target_id: str
    old_weight: float
    new_weight: float
    timestamp: datetime = field(default_factory=datetime.now)

    @property
    def delta(self) -> float:
        """Change in weight."""
        return self.new_weight - self.old_weight


@dataclass
class SynapseState:
    """State of a synapse for plasticity computations."""
    source_id: str
    target_id: str
    weight: float
    last_activated: datetime | None = None
    activation_count: int = 0
    tagged_for_consolidation: bool = False


class RelationshipStore(Protocol):
    """Protocol for relationship storage."""

    async def get_relationships(
        self,
        node_id: str,
        direction: str = "out"
    ) -> list[dict]:
        """Get relationships for a node."""
        ...

    async def update_relationship_weight(
        self,
        source_id: str,
        target_id: str,
        new_weight: float
    ) -> None:
        """Update weight of a relationship."""
        ...


class LTDEngine:
    """
    Long-Term Depression engine.

    Implements competitive weakening: when a set of entities is activated,
    connections from those entities to their NON-activated neighbors are weakened.

    This creates winner-take-all dynamics that sharpen memory representations.

    BCM Theory Implementation:
    - θ_m (modification threshold) adapts based on activity
    - Activity below θ_m leads to depression
    - Activity above θ_m leads to potentiation (handled by LTP)
    """

    def __init__(
        self,
        ltd_rate: float = 0.05,
        min_weight: float = 0.01,
        competitive_radius: int = 1
    ):
        """
        Initialize LTD engine.

        Args:
            ltd_rate: Depression rate (fraction of weight to remove)
            min_weight: Minimum weight before connection is considered dead
            competitive_radius: How many hops for competition (1 = immediate neighbors)
        """
        self.ltd_rate = ltd_rate
        self.min_weight = min_weight
        self.competitive_radius = competitive_radius
        # MEM-002 FIX: Bounded event history
        self._events: list[PlasticityEvent] = []
        self._max_events = 10000

    async def apply_ltd(
        self,
        activated_ids: set[str],
        store: RelationshipStore,
        session_id: str | None = None
    ) -> list[PlasticityEvent]:
        """
        Apply LTD to non-co-activated neighbors.

        For each activated entity, weaken connections to neighbors
        that were NOT in the activated set.

        Args:
            activated_ids: Set of entity IDs that were activated
            store: Relationship store for queries and updates
            session_id: Optional session for isolation

        Returns:
            List of PlasticityEvent records
        """
        events = []

        for entity_id in activated_ids:
            try:
                # Get outgoing relationships
                relationships = await store.get_relationships(
                    node_id=entity_id,
                    direction="out"
                )

                for rel in relationships:
                    other_id = rel.get("other_id", "")
                    current_weight = rel.get("properties", {}).get("weight", 1.0)

                    # Only weaken if neighbor was NOT activated
                    if other_id not in activated_ids:
                        # LTD: competitive weakening
                        new_weight = max(
                            self.min_weight,
                            current_weight * (1 - self.ltd_rate)
                        )

                        if new_weight != current_weight:
                            await store.update_relationship_weight(
                                source_id=entity_id,
                                target_id=other_id,
                                new_weight=new_weight
                            )

                            event = PlasticityEvent(
                                event_type=PlasticityType.LTD,
                                source_id=entity_id,
                                target_id=other_id,
                                old_weight=current_weight,
                                new_weight=new_weight
                            )
                            events.append(event)
                            self._events.append(event)

            except Exception as e:
                logger.warning(f"LTD failed for {entity_id}: {e}")

        # MEM-002 FIX: Trim history if over limit
        if len(self._events) > self._max_events:
            self._events = self._events[-self._max_events:]

        logger.debug(f"Applied LTD to {len(events)} connections")
        return events

    def get_history(self, limit: int = 100) -> list[PlasticityEvent]:
        """Get recent LTD events."""
        return self._events[-limit:]


class HomeostaticScaler:
    """
    Homeostatic Synaptic Scaling.

    Maintains network stability by globally scaling weights when
    total activity exceeds or falls below target levels.

    Biological Basis:
    - TNFα-mediated multiplicative scaling
    - Maintains mean firing rate homeostasis
    - Acts over hours to days (we simulate per consolidation cycle)

    Implementation:
    - Calculate total outgoing weight per node
    - If above target, scale down all outgoing weights
    - If below target, scale up (with ceiling)
    """

    def __init__(
        self,
        target_total: float = 10.0,
        tolerance: float = 0.2,
        max_weight: float = 1.0,
        min_weight: float = 0.01
    ):
        """
        Initialize homeostatic scaler.

        Args:
            target_total: Target sum of outgoing weights per node
            tolerance: Fraction tolerance before scaling (0.2 = ±20%)
            max_weight: Maximum individual weight
            min_weight: Minimum individual weight
        """
        self.target_total = target_total
        self.tolerance = tolerance
        self.max_weight = max_weight
        self.min_weight = min_weight
        # MEM-002 FIX: Bounded event history
        self._events: list[PlasticityEvent] = []
        self._max_events = 10000

    async def scale_node(
        self,
        entity_id: str,
        store: RelationshipStore
    ) -> list[PlasticityEvent]:
        """
        Apply homeostatic scaling to a single node.

        Args:
            entity_id: Node to scale
            store: Relationship store

        Returns:
            List of scaling events
        """
        events = []

        try:
            relationships = await store.get_relationships(
                node_id=entity_id,
                direction="out"
            )

            if not relationships:
                return events

            # Calculate total outgoing weight
            total_weight = sum(
                rel.get("properties", {}).get("weight", 0.0)
                for rel in relationships
            )

            # Check if scaling needed
            lower = self.target_total * (1 - self.tolerance)
            upper = self.target_total * (1 + self.tolerance)

            if lower <= total_weight <= upper:
                return events  # Within tolerance

            # Calculate scale factor
            if total_weight > 0:
                scale_factor = self.target_total / total_weight
            else:
                return events

            # Apply scaling to each relationship
            for rel in relationships:
                other_id = rel.get("other_id", "")
                old_weight = rel.get("properties", {}).get("weight", 0.0)

                # Scale and bound
                new_weight = old_weight * scale_factor
                new_weight = max(self.min_weight, min(self.max_weight, new_weight))

                if abs(new_weight - old_weight) > 1e-6:
                    await store.update_relationship_weight(
                        source_id=entity_id,
                        target_id=other_id,
                        new_weight=new_weight
                    )

                    event = PlasticityEvent(
                        event_type=PlasticityType.HOMEOSTATIC,
                        source_id=entity_id,
                        target_id=other_id,
                        old_weight=old_weight,
                        new_weight=new_weight
                    )
                    events.append(event)
                    self._events.append(event)

            # MEM-002 FIX: Trim history if over limit
            if len(self._events) > self._max_events:
                self._events = self._events[-self._max_events:]

        except Exception as e:
            logger.warning(f"Homeostatic scaling failed for {entity_id}: {e}")

        return events

    async def scale_batch(
        self,
        entity_ids: list[str],
        store: RelationshipStore
    ) -> list[PlasticityEvent]:
        """
        Apply homeostatic scaling to multiple nodes.

        Args:
            entity_ids: Nodes to scale
            store: Relationship store

        Returns:
            List of all scaling events
        """
        all_events = []

        for entity_id in entity_ids:
            events = await self.scale_node(entity_id, store)
            all_events.extend(events)

        logger.debug(f"Homeostatic scaling: {len(all_events)} weight adjustments")
        return all_events

    def get_history(self, limit: int = 100) -> list[PlasticityEvent]:
        """Get recent scaling events."""
        return self._events[-limit:]


class MetaplasticityController:
    """
    Metaplasticity Controller.

    Implements the sliding modification threshold (θ_m) from BCM theory.
    High recent activity raises the threshold (harder to strengthen).
    Low recent activity lowers the threshold (easier to strengthen).

    This prevents runaway potentiation and stabilizes learning.
    """

    def __init__(
        self,
        base_threshold: float = 0.5,
        adaptation_rate: float = 0.1,
        min_threshold: float = 0.1,
        max_threshold: float = 0.9
    ):
        """
        Initialize metaplasticity controller.

        Args:
            base_threshold: Starting modification threshold
            adaptation_rate: How fast threshold adapts to activity
            min_threshold: Minimum threshold (easiest to potentiate)
            max_threshold: Maximum threshold (hardest to potentiate)
        """
        self.base_threshold = base_threshold
        self.adaptation_rate = adaptation_rate
        self.min_threshold = min_threshold
        self.max_threshold = max_threshold

        # Per-entity thresholds - MEM-002 FIX: Add size limit
        self._thresholds: dict[str, float] = {}
        # Activity history (exponential moving average)
        self._activity_ema: dict[str, float] = {}
        self._max_tracked_entities = 100000

    def get_threshold(self, entity_id: str) -> float:
        """
        Get current modification threshold for an entity.

        Args:
            entity_id: Entity to query

        Returns:
            Current threshold (higher = harder to potentiate)
        """
        return self._thresholds.get(entity_id, self.base_threshold)

    def update_activity(self, entity_id: str, activity_level: float) -> float:
        """
        Update activity history and adjust threshold.

        Args:
            entity_id: Entity that was active
            activity_level: Level of activation (0-1)

        Returns:
            New threshold for this entity
        """
        # Update EMA of activity
        current_ema = self._activity_ema.get(entity_id, 0.5)
        new_ema = (1 - self.adaptation_rate) * current_ema + self.adaptation_rate * activity_level
        self._activity_ema[entity_id] = new_ema

        # BCM rule: threshold proportional to squared activity
        # High activity -> high threshold -> harder to potentiate
        new_threshold = self.base_threshold * (1 + new_ema ** 2)
        new_threshold = max(self.min_threshold, min(self.max_threshold, new_threshold))

        self._thresholds[entity_id] = new_threshold

        # MEM-002 FIX: Cleanup old entries if over limit
        if len(self._thresholds) > self._max_tracked_entities:
            self._cleanup_old_entities()

        return new_threshold

    def _cleanup_old_entities(self) -> None:
        """Remove least active entities to enforce size limit."""
        if len(self._thresholds) <= self._max_tracked_entities:
            return
        # Keep entities with highest activity
        sorted_entities = sorted(
            self._activity_ema.items(),
            key=lambda x: x[1],
            reverse=True
        )
        keep_ids = set(e[0] for e in sorted_entities[:self._max_tracked_entities])
        self._thresholds = {k: v for k, v in self._thresholds.items() if k in keep_ids}
        self._activity_ema = {k: v for k, v in self._activity_ema.items() if k in keep_ids}

    def should_potentiate(self, entity_id: str, signal_strength: float) -> bool:
        """
        Determine if signal is strong enough for potentiation.

        Args:
            entity_id: Entity receiving signal
            signal_strength: Strength of incoming signal (0-1)

        Returns:
            True if signal exceeds threshold (potentiation)
        """
        threshold = self.get_threshold(entity_id)
        return signal_strength > threshold

    def should_depress(self, entity_id: str, signal_strength: float) -> bool:
        """
        Determine if signal is weak enough for depression.

        Args:
            entity_id: Entity receiving signal
            signal_strength: Strength of incoming signal (0-1)

        Returns:
            True if signal below threshold (depression)
        """
        threshold = self.get_threshold(entity_id)
        return signal_strength < threshold * 0.5  # Depression below half threshold

    def reset(self, entity_id: str | None = None) -> None:
        """
        Reset thresholds to baseline.

        Args:
            entity_id: Specific entity to reset, or None for all
        """
        if entity_id is not None:
            self._thresholds.pop(entity_id, None)
            self._activity_ema.pop(entity_id, None)
        else:
            self._thresholds.clear()
            self._activity_ema.clear()


@dataclass
class SynapticTag:
    """Synaptic tag for consolidation."""
    source_id: str
    target_id: str
    tag_type: str  # "early" or "late"
    created_at: datetime
    strength: float  # Tag strength (decays over time)
    captured: bool = False  # Whether tag has been captured by proteins


class SynapticTagger:
    """
    Synaptic Tagging and Capture.

    Implements the "synaptic tag-and-capture" model:
    1. Strong activation creates "late LTP" tags
    2. Weak activation creates "early LTP" tags
    3. Tags decay if not captured by plasticity proteins
    4. Protein synthesis (consolidation) captures tags

    This enables heterosynaptic metaplasticity and memory allocation.
    """

    def __init__(
        self,
        early_threshold: float = 0.3,
        late_threshold: float = 0.7,
        tag_lifetime_hours: float = 2.0
    ):
        """
        Initialize synaptic tagger.

        Args:
            early_threshold: Signal strength for early LTP tag
            late_threshold: Signal strength for late LTP tag
            tag_lifetime_hours: How long uncaptured tags persist
        """
        self.early_threshold = early_threshold
        self.late_threshold = late_threshold
        self.tag_lifetime_hours = tag_lifetime_hours

        # MEM-002 FIX: Add size limit
        self._tags: dict[tuple[str, str], SynapticTag] = {}
        self._max_tags = 100000

    def tag_synapse(
        self,
        source_id: str,
        target_id: str,
        signal_strength: float
    ) -> SynapticTag | None:
        """
        Tag a synapse based on signal strength.

        Args:
            source_id: Presynaptic entity
            target_id: Postsynaptic entity
            signal_strength: Strength of activation (0-1)

        Returns:
            Created tag, or None if below threshold
        """
        key = (source_id, target_id)

        if signal_strength >= self.late_threshold:
            tag_type = "late"
            strength = signal_strength
        elif signal_strength >= self.early_threshold:
            tag_type = "early"
            strength = signal_strength
        else:
            return None

        tag = SynapticTag(
            source_id=source_id,
            target_id=target_id,
            tag_type=tag_type,
            created_at=datetime.now(),
            strength=strength
        )

        # Overwrite any existing tag with stronger one
        existing = self._tags.get(key)
        if existing is None or tag.strength > existing.strength:
            # MEM-002 FIX: Check capacity before adding
            if key not in self._tags and len(self._tags) >= self._max_tags:
                self._prune_expired()  # Try to make room
                if len(self._tags) >= self._max_tags:
                    self._evict_weakest_tag()
            self._tags[key] = tag

        return tag

    def _evict_weakest_tag(self) -> None:
        """Evict weakest tag to make room."""
        if not self._tags:
            return
        weakest_key = min(self._tags.keys(), key=lambda k: self._tags[k].strength)
        del self._tags[weakest_key]

    def get_tagged_synapses(
        self,
        tag_type: str | None = None,
        captured_only: bool = False
    ) -> list[SynapticTag]:
        """
        Get synapses with active tags.

        Args:
            tag_type: Filter by "early" or "late"
            captured_only: Only return captured tags

        Returns:
            List of matching tags
        """
        self._prune_expired()

        tags = list(self._tags.values())

        if tag_type is not None:
            tags = [t for t in tags if t.tag_type == tag_type]

        if captured_only:
            tags = [t for t in tags if t.captured]

        return tags

    def capture_tags(self) -> list[SynapticTag]:
        """
        Capture all eligible tags (simulate protein synthesis).

        This should be called during consolidation.

        Returns:
            List of captured tags
        """
        self._prune_expired()

        captured = []
        for tag in self._tags.values():
            if not tag.captured:
                tag.captured = True
                captured.append(tag)

        logger.debug(f"Captured {len(captured)} synaptic tags")
        return captured

    def _prune_expired(self) -> int:
        """Remove expired (uncaptured) tags."""
        now = datetime.now()
        expired = []

        for key, tag in self._tags.items():
            if not tag.captured:
                age_hours = (now - tag.created_at).total_seconds() / 3600
                if age_hours > self.tag_lifetime_hours:
                    expired.append(key)

        for key in expired:
            del self._tags[key]

        return len(expired)

    def clear(self) -> None:
        """Clear all tags."""
        self._tags.clear()


class PlasticityManager:
    """
    Unified manager for all plasticity mechanisms.

    Coordinates:
    - LTD for competitive weakening
    - Homeostatic scaling for stability
    - Metaplasticity for adaptive thresholds
    - Synaptic tagging for consolidation
    """

    def __init__(
        self,
        ltd_engine: LTDEngine | None = None,
        homeostatic_scaler: HomeostaticScaler | None = None,
        metaplasticity: MetaplasticityController | None = None,
        synaptic_tagger: SynapticTagger | None = None
    ):
        """
        Initialize plasticity manager.

        Args:
            ltd_engine: LTD engine (default created if None)
            homeostatic_scaler: Scaler (default created if None)
            metaplasticity: Controller (default created if None)
            synaptic_tagger: Tagger (default created if None)
        """
        self.ltd = ltd_engine or LTDEngine()
        self.homeostatic = homeostatic_scaler or HomeostaticScaler()
        self.metaplasticity = metaplasticity or MetaplasticityController()
        self.tagger = synaptic_tagger or SynapticTagger()

    async def on_retrieval(
        self,
        activated_ids: set[str],
        store: RelationshipStore
    ) -> dict:
        """
        Process plasticity after retrieval event.

        Args:
            activated_ids: Entities that were activated
            store: Relationship store

        Returns:
            Summary of plasticity events
        """
        # Apply LTD to non-co-activated neighbors
        ltd_events = await self.ltd.apply_ltd(activated_ids, store)

        # Update metaplasticity thresholds
        for entity_id in activated_ids:
            self.metaplasticity.update_activity(entity_id, 0.8)

        # Tag synapses (between activated entities)
        tags_created = 0
        for src in activated_ids:
            for tgt in activated_ids:
                if src != tgt:
                    tag = self.tagger.tag_synapse(src, tgt, 0.7)
                    if tag:
                        tags_created += 1

        return {
            "ltd_events": len(ltd_events),
            "tags_created": tags_created
        }

    async def on_consolidation(
        self,
        all_entity_ids: list[str],
        store: RelationshipStore
    ) -> dict:
        """
        Process plasticity during consolidation.

        Args:
            all_entity_ids: All entities to process
            store: Relationship store

        Returns:
            Summary of consolidation plasticity
        """
        # Homeostatic scaling
        scaling_events = await self.homeostatic.scale_batch(
            all_entity_ids, store
        )

        # Capture synaptic tags
        captured_tags = self.tagger.capture_tags()

        return {
            "scaling_events": len(scaling_events),
            "tags_captured": len(captured_tags)
        }

    def get_stats(self) -> dict:
        """Get plasticity statistics."""
        return {
            "ltd_history_size": len(self.ltd.get_history()),
            "homeostatic_history_size": len(self.homeostatic.get_history()),
            "active_tags": len(self.tagger.get_tagged_synapses()),
            "captured_tags": len(self.tagger.get_tagged_synapses(captured_only=True))
        }


# =============================================================================
# P3.5: Elastic Weight Consolidation (EWC)
# =============================================================================


@dataclass
class EWCStats:
    """Statistics for EWC regularization."""
    n_consolidations: int = 0
    total_fisher_entries: int = 0
    avg_fisher_magnitude: float = 0.0
    last_consolidation_loss: float = 0.0


if TORCH_AVAILABLE:
    class EWCRegularizer:
        """
        Elastic Weight Consolidation for continual learning.

        Prevents catastrophic forgetting by penalizing changes to weights
        that were important for previous tasks. Importance is measured by
        the diagonal of the Fisher Information Matrix.

        Biological Analogy:
        - Fisher = synaptic importance (how much each synapse matters)
        - Optimal weights = consolidated memory traces in neocortex
        - λ_ewc = consolidation strength (like sleep-dependent consolidation)

        Algorithm:
        1. After training on task T, compute Fisher information F_T
        2. Store optimal weights θ*_T
        3. When training on task T+1, add penalty:
           L_EWC = (λ/2) * Σ_i F_i * (θ_i - θ*_i)²

        This allows learning new tasks while preserving old knowledge,
        following the Complementary Learning Systems theory.

        References:
        - Kirkpatrick et al. (2017) "Overcoming catastrophic forgetting"
        - Zenke et al. (2017) "Continual Learning Through Synaptic Intelligence"
        """

        def __init__(
            self,
            lambda_ewc: float = 1000.0,
            fisher_n_samples: int = 200,
            online: bool = True,
            gamma: float = 0.95,
        ):
            """
            Initialize EWC regularizer.

            Args:
                lambda_ewc: Regularization strength (higher = more protection)
                fisher_n_samples: Number of samples for Fisher estimation
                online: If True, use online EWC (exponential moving average)
                gamma: Decay factor for online EWC (0 < gamma <= 1)
            """
            self.lambda_ewc = lambda_ewc
            self.fisher_n_samples = fisher_n_samples
            self.online = online
            self.gamma = gamma

            # Stored Fisher information matrices (per-task or accumulated)
            self.fisher_diag: dict[str, torch.Tensor] = {}

            # Optimal weights after each task
            self.optimal_weights: dict[str, torch.Tensor] = {}

            # Statistics
            self.stats = EWCStats()

            logger.info(
                f"EWC initialized: λ={lambda_ewc}, online={online}, "
                f"gamma={gamma}, n_samples={fisher_n_samples}"
            )

        def compute_fisher(
            self,
            model: nn.Module,
            dataloader: DataLoader,
            device: torch.device | str = "cpu",
        ) -> dict[str, torch.Tensor]:
            """
            Compute diagonal of Fisher Information Matrix.

            Fisher diagonal approximates the importance of each parameter
            for the current task. Computed as:
                F_i = E[(∂L/∂θ_i)²]

            This is equivalent to the curvature of the loss landscape.

            Args:
                model: PyTorch model with trainable parameters
                dataloader: DataLoader with representative task samples
                device: Device for computation

            Returns:
                Dictionary mapping parameter names to Fisher diagonal tensors
            """
            model.eval()
            device = torch.device(device) if isinstance(device, str) else device

            # Initialize Fisher accumulator
            fisher = {
                name: torch.zeros_like(param, device=device)
                for name, param in model.named_parameters()
                if param.requires_grad
            }

            n_samples = 0
            for batch_idx, batch in enumerate(dataloader):
                if n_samples >= self.fisher_n_samples:
                    break

                # Handle different batch formats
                if isinstance(batch, (tuple, list)):
                    inputs = batch[0]
                    if isinstance(inputs, torch.Tensor):
                        inputs = inputs.to(device)
                else:
                    inputs = batch
                    if isinstance(inputs, torch.Tensor):
                        inputs = inputs.to(device)

                # Forward pass
                model.zero_grad()
                try:
                    outputs = model(inputs)

                    # For probabilistic outputs, use log-likelihood
                    if outputs.dim() > 1 and outputs.shape[-1] > 1:
                        # Classification: sample from output distribution
                        log_probs = torch.log_softmax(outputs, dim=-1)
                        # Use empirical Fisher (sample from model's own predictions)
                        with torch.no_grad():
                            samples = torch.multinomial(
                                torch.softmax(outputs, dim=-1), 1
                            ).squeeze(-1)
                        loss = -log_probs.gather(1, samples.unsqueeze(1)).mean()
                    else:
                        # Scalar output: use squared output as proxy
                        loss = outputs.pow(2).mean()

                    loss.backward()

                    # Accumulate squared gradients
                    for name, param in model.named_parameters():
                        if param.grad is not None:
                            fisher[name] += param.grad.pow(2)

                    n_samples += inputs.shape[0] if isinstance(inputs, torch.Tensor) else 1

                except Exception as e:
                    logger.warning(f"Fisher computation failed for batch {batch_idx}: {e}")
                    continue

            # Average
            if n_samples > 0:
                for name in fisher:
                    fisher[name] /= n_samples

            logger.debug(f"Computed Fisher for {len(fisher)} parameters over {n_samples} samples")
            return fisher

        def consolidate(
            self,
            model: nn.Module,
            dataloader: DataLoader,
            task_id: str | None = None,
            device: torch.device | str = "cpu",
        ) -> float:
            """
            Consolidate current task knowledge.

            Computes Fisher information and stores optimal weights.
            For online EWC, accumulates with exponential decay.

            Args:
                model: Model to consolidate
                dataloader: Representative data for current task
                task_id: Optional task identifier (for multi-task tracking)
                device: Device for computation

            Returns:
                Average Fisher magnitude (for monitoring)
            """
            device = torch.device(device) if isinstance(device, str) else device

            # Compute Fisher for current task
            fisher_new = self.compute_fisher(model, dataloader, device)

            if self.online:
                # Online EWC: exponential moving average
                for name, fisher_val in fisher_new.items():
                    if name in self.fisher_diag:
                        # Decay old Fisher and add new
                        self.fisher_diag[name] = (
                            self.gamma * self.fisher_diag[name] +
                            (1 - self.gamma) * fisher_val
                        )
                    else:
                        self.fisher_diag[name] = fisher_val.clone()

                    # Store current optimal weights
                    for n, p in model.named_parameters():
                        if n == name:
                            self.optimal_weights[name] = p.detach().clone()
                            break
            else:
                # Standard EWC: store per-task
                task_key = task_id or f"task_{self.stats.n_consolidations}"
                self.fisher_diag[task_key] = fisher_new
                self.optimal_weights[task_key] = {
                    name: param.detach().clone()
                    for name, param in model.named_parameters()
                    if param.requires_grad
                }

            # Update statistics
            self.stats.n_consolidations += 1
            total_entries = sum(f.numel() for f in fisher_new.values())
            avg_magnitude = sum(f.abs().mean().item() for f in fisher_new.values()) / len(fisher_new)
            self.stats.total_fisher_entries = total_entries
            self.stats.avg_fisher_magnitude = avg_magnitude

            logger.info(
                f"EWC consolidation #{self.stats.n_consolidations}: "
                f"{total_entries} params, avg_fisher={avg_magnitude:.4e}"
            )

            return avg_magnitude

        def penalty(self, model: nn.Module) -> torch.Tensor:
            """
            Compute EWC penalty for current parameters.

            Penalty = (λ/2) * Σ_i F_i * (θ_i - θ*_i)²

            Args:
                model: Model with current parameters

            Returns:
                Scalar penalty tensor (add to loss before backward)
            """
            if not self.fisher_diag:
                # No consolidation yet, no penalty
                return torch.tensor(0.0, requires_grad=True)

            total_penalty = torch.tensor(0.0, device=next(model.parameters()).device)

            if self.online:
                # Online EWC: single accumulated Fisher
                for name, param in model.named_parameters():
                    if name in self.fisher_diag and name in self.optimal_weights:
                        fisher = self.fisher_diag[name]
                        optimal = self.optimal_weights[name]

                        # Move to same device if needed
                        if fisher.device != param.device:
                            fisher = fisher.to(param.device)
                        if optimal.device != param.device:
                            optimal = optimal.to(param.device)

                        # EWC penalty: F * (θ - θ*)²
                        total_penalty = total_penalty + (
                            fisher * (param - optimal).pow(2)
                        ).sum()
            else:
                # Standard EWC: sum over all tasks
                for task_key, task_fisher in self.fisher_diag.items():
                    if isinstance(task_fisher, dict):
                        task_optimal = self.optimal_weights[task_key]
                        for name, param in model.named_parameters():
                            if name in task_fisher and name in task_optimal:
                                fisher = task_fisher[name]
                                optimal = task_optimal[name]

                                if fisher.device != param.device:
                                    fisher = fisher.to(param.device)
                                if optimal.device != param.device:
                                    optimal = optimal.to(param.device)

                                total_penalty = total_penalty + (
                                    fisher * (param - optimal).pow(2)
                                ).sum()

            return (self.lambda_ewc / 2) * total_penalty

        def ewc_loss(
            self,
            model: nn.Module,
            task_loss: torch.Tensor,
        ) -> torch.Tensor:
            """
            Compute total loss with EWC penalty.

            Args:
                model: Current model
                task_loss: Loss for current task

            Returns:
                Combined loss: task_loss + EWC_penalty
            """
            ewc_penalty = self.penalty(model)
            total_loss = task_loss + ewc_penalty
            self.stats.last_consolidation_loss = ewc_penalty.item()
            return total_loss

        def get_importance_scores(self, model: nn.Module) -> dict[str, float]:
            """
            Get normalized importance scores for each parameter.

            Useful for visualization and debugging.

            Args:
                model: Model to analyze

            Returns:
                Dictionary of parameter name -> importance score
            """
            if not self.fisher_diag:
                return {}

            scores = {}
            total = 0.0

            if self.online:
                for name in self.fisher_diag:
                    score = self.fisher_diag[name].abs().mean().item()
                    scores[name] = score
                    total += score
            else:
                # Average across tasks
                for task_fisher in self.fisher_diag.values():
                    if isinstance(task_fisher, dict):
                        for name, fisher in task_fisher.items():
                            if name not in scores:
                                scores[name] = 0.0
                            scores[name] += fisher.abs().mean().item()
                            total += fisher.abs().mean().item()

            # Normalize
            if total > 0:
                scores = {k: v / total for k, v in scores.items()}

            return scores

        def get_stats(self) -> dict:
            """Get EWC statistics."""
            return {
                "n_consolidations": self.stats.n_consolidations,
                "total_fisher_entries": self.stats.total_fisher_entries,
                "avg_fisher_magnitude": self.stats.avg_fisher_magnitude,
                "last_penalty": self.stats.last_consolidation_loss,
                "lambda_ewc": self.lambda_ewc,
                "online": self.online,
                "gamma": self.gamma,
            }

        def reset(self) -> None:
            """Reset EWC state (clear all consolidated knowledge)."""
            self.fisher_diag.clear()
            self.optimal_weights.clear()
            self.stats = EWCStats()
            logger.info("EWC state reset")

        def save(self, path: str) -> None:
            """
            Save EWC state to file.

            Args:
                path: Path to save file
            """
            state = {
                "fisher_diag": self.fisher_diag,
                "optimal_weights": self.optimal_weights,
                "stats": {
                    "n_consolidations": self.stats.n_consolidations,
                    "total_fisher_entries": self.stats.total_fisher_entries,
                    "avg_fisher_magnitude": self.stats.avg_fisher_magnitude,
                },
                "config": {
                    "lambda_ewc": self.lambda_ewc,
                    "online": self.online,
                    "gamma": self.gamma,
                }
            }
            torch.save(state, path)
            logger.info(f"EWC state saved to {path}")

        def load(self, path: str) -> None:
            """
            Load EWC state from file.

            Args:
                path: Path to load from
            """
            state = torch.load(path, weights_only=False)
            self.fisher_diag = state["fisher_diag"]
            self.optimal_weights = state["optimal_weights"]
            self.stats = EWCStats(**state["stats"])
            self.lambda_ewc = state["config"]["lambda_ewc"]
            self.online = state["config"]["online"]
            self.gamma = state["config"]["gamma"]
            logger.info(f"EWC state loaded from {path}")


    def create_ewc_regularizer(
        lambda_ewc: float = 1000.0,
        online: bool = True,
        gamma: float = 0.95,
    ) -> EWCRegularizer:
        """
        Factory function to create EWC regularizer.

        Args:
            lambda_ewc: Regularization strength
            online: Use online EWC (recommended)
            gamma: Decay factor for online mode

        Returns:
            Configured EWCRegularizer
        """
        return EWCRegularizer(lambda_ewc=lambda_ewc, online=online, gamma=gamma)

else:
    # Stub class when PyTorch not available
    class EWCRegularizer:  # type: ignore[no-redef]
        """Stub EWC regularizer when PyTorch is not available."""

        def __init__(self, **kwargs):
            logger.warning("EWC not available (PyTorch required)")

        def compute_fisher(self, *args, **kwargs):
            return {}

        def consolidate(self, *args, **kwargs):
            return 0.0

        def penalty(self, *args, **kwargs):
            return 0.0

        def ewc_loss(self, *args, **kwargs):
            return kwargs.get("task_loss", 0.0)

        def get_stats(self):
            return {"available": False}

    def create_ewc_regularizer(**kwargs):
        """Stub factory when PyTorch not available."""
        return EWCRegularizer(**kwargs)
