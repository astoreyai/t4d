"""
P7.3: STDP Integration with Sleep Consolidation.

Integrates Spike-Timing-Dependent Plasticity with the sleep consolidation
system to enable biologically-plausible weight updates during memory replay.

The integration follows biological principles:
1. During NREM replay, STDP is applied to strengthen temporal sequences
2. During consolidation, STDP weights are synced with synaptic tags
3. Weight decay is applied homeostatic regulation
4. Uses multiplicative STDP for stable weight dynamics (van Rossum et al. 2000)

References:
- Wilson & McNaughton (1994): Reactivation of hippocampal ensemble memories during sleep
- Ji & Wilson (2007): Hippocampal replay during sleep
- Clopath et al. (2010): STDP and memory consolidation
- Foster & Wilson (2006): Reverse replay of behavioural sequences
- van Rossum et al. (2000): Stable Hebbian learning from spike timing-dependent plasticity
"""

import logging
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from typing import Any

from t4dm.learning.plasticity import SynapticTagger
from t4dm.learning.stdp import STDPConfig, STDPLearner, STDPUpdate

logger = logging.getLogger(__name__)


@dataclass
class ConsolidationSTDPConfig:
    """Configuration for STDP during consolidation."""

    # STDP timing
    replay_interval_ms: float = 50.0  # Time between replayed memories
    stdp_config: STDPConfig | None = None

    # Weight synchronization
    sync_with_tags: bool = True
    tag_weight_influence: float = 0.3  # How much tags influence STDP weights

    # Homeostatic regulation
    apply_decay: bool = True
    decay_baseline: float = 0.5

    # Replay integration
    min_sequence_length: int = 3
    max_sequence_length: int = 10

    # Weight bounds
    consolidation_min_weight: float = 0.05
    consolidation_max_weight: float = 0.95


@dataclass
class STDPReplayResult:
    """Result of STDP applied during replay."""

    episode_ids: list[str]
    updates_applied: int
    ltp_count: int
    ltd_count: int
    mean_weight_change: float
    sequence_length: int
    replay_time: datetime = field(default_factory=datetime.now)


class ConsolidationSTDP:
    """
    P7.3: STDP integration with sleep consolidation.

    Applies STDP weight updates during memory replay to strengthen
    temporal sequences and maintain biologically-plausible learning.

    Uses multiplicative STDP for stable weight dynamics.

    Usage:
        # During consolidation
        consolidation_stdp = ConsolidationSTDP()

        # When replaying a sequence of episodes
        result = consolidation_stdp.apply_stdp_to_sequence(episode_ids)

        # After consolidation cycle
        consolidation_stdp.consolidate_weights()
    """

    def __init__(self, config: ConsolidationSTDPConfig | None = None):
        """
        Initialize consolidation STDP.

        Args:
            config: Configuration (uses defaults if None)
        """
        self.config = config or ConsolidationSTDPConfig()

        # Initialize STDP learner with multiplicative dynamics
        stdp_config = self.config.stdp_config or STDPConfig(
            # Wider window for replay timing
            spike_window_ms=200.0,
            # Moderate learning rates for consolidation
            a_plus=0.008,
            a_minus=0.0084,  # Slight LTD bias for homeostasis
            weight_decay=0.01,
            min_weight=self.config.consolidation_min_weight,
            max_weight=self.config.consolidation_max_weight,
            # Enable multiplicative STDP (van Rossum et al. 2000)
            multiplicative=True,
            mu=0.5,  # Weight dependence exponent
        )
        self.stdp = STDPLearner(stdp_config)

        # Optional synaptic tagger integration
        self.tagger: SynapticTagger | None = None
        if self.config.sync_with_tags:
            self.tagger = SynapticTagger()

        # Statistics
        self._total_sequences = 0
        self._total_updates = 0
        self._consolidation_cycles = 0

        logger.info(
            f"ConsolidationSTDP initialized: "
            f"replay_interval={self.config.replay_interval_ms}ms, "
            f"sync_tags={self.config.sync_with_tags}, "
            f"multiplicative={stdp_config.multiplicative}, mu={stdp_config.mu}"
        )

    def apply_stdp_to_sequence(
        self,
        episode_ids: list[str],
        replay_time: datetime | None = None,
        interval_ms: float | None = None,
    ) -> STDPReplayResult:
        """
        Apply STDP to a sequence of replayed episodes.

        During sleep replay, episodes are replayed in compressed time.
        This method records the spike times and computes STDP updates
        for adjacent pairs in the sequence.

        Args:
            episode_ids: List of episode IDs in replay order
            replay_time: Base time for replay (default: now)
            interval_ms: Time between replayed episodes in ms

        Returns:
            STDPReplayResult with update statistics
        """
        if len(episode_ids) < 2:
            return STDPReplayResult(
                episode_ids=episode_ids,
                updates_applied=0,
                ltp_count=0,
                ltd_count=0,
                mean_weight_change=0.0,
                sequence_length=len(episode_ids)
            )

        replay_time = replay_time or datetime.now()
        interval_ms = interval_ms or self.config.replay_interval_ms

        # Record spike times for each episode in sequence
        for i, ep_id in enumerate(episode_ids):
            spike_time = replay_time + timedelta(milliseconds=i * interval_ms)
            self.stdp.record_spike(ep_id, timestamp=spike_time)

        # Compute STDP updates for adjacent pairs
        updates: list[STDPUpdate] = []
        for i in range(len(episode_ids) - 1):
            pre_id = episode_ids[i]
            post_id = episode_ids[i + 1]

            current_weight = self.stdp.get_weight(pre_id, post_id)
            update = self.stdp.compute_update(pre_id, post_id, current_weight)

            if update:
                # Apply the update
                self.stdp.set_weight(pre_id, post_id, update.new_weight)
                updates.append(update)

                # Optionally create synaptic tag
                if self.tagger:
                    # Tag strength based on update magnitude
                    tag_strength = min(0.9, 0.5 + abs(update.delta_weight) * 5)
                    self.tagger.tag_synapse(pre_id, post_id, tag_strength)

        # Compute statistics
        ltp_count = sum(1 for u in updates if u.update_type == "ltp")
        ltd_count = sum(1 for u in updates if u.update_type == "ltd")
        mean_change = (
            sum(abs(u.delta_weight) for u in updates) / len(updates)
            if updates else 0.0
        )

        self._total_sequences += 1
        self._total_updates += len(updates)

        return STDPReplayResult(
            episode_ids=episode_ids,
            updates_applied=len(updates),
            ltp_count=ltp_count,
            ltd_count=ltd_count,
            mean_weight_change=mean_change,
            sequence_length=len(episode_ids),
            replay_time=replay_time
        )

    def apply_stdp_to_co_retrieval(
        self,
        query_id: str,
        retrieved_ids: list[str],
        similarities: list[float] | None = None,
    ) -> list[STDPUpdate]:
        """
        Apply STDP when memories are co-retrieved.

        When a query retrieves multiple memories, those memories should
        have their connections strengthened based on co-activation.

        Args:
            query_id: ID of the query
            retrieved_ids: IDs of retrieved memories (in order)
            similarities: Optional similarity scores (used for tag strength)

        Returns:
            List of weight updates applied
        """
        if len(retrieved_ids) < 2:
            return []

        now = datetime.now()
        updates = []

        # Record query spike
        self.stdp.record_spike(query_id, timestamp=now)

        # Record retrieved memory spikes in order
        for i, mem_id in enumerate(retrieved_ids):
            spike_time = now + timedelta(milliseconds=(i + 1) * 10)
            self.stdp.record_spike(mem_id, timestamp=spike_time)

        # Apply STDP between adjacent retrievals
        for i in range(len(retrieved_ids) - 1):
            pre_id = retrieved_ids[i]
            post_id = retrieved_ids[i + 1]

            current_weight = self.stdp.get_weight(pre_id, post_id)
            update = self.stdp.compute_update(pre_id, post_id, current_weight)

            if update:
                self.stdp.set_weight(pre_id, post_id, update.new_weight)
                updates.append(update)

                # Create tag with similarity-based strength
                if self.tagger and similarities and i < len(similarities):
                    self.tagger.tag_synapse(pre_id, post_id, similarities[i])

        return updates

    def consolidate_weights(self) -> dict[str, Any]:
        """
        Perform weight consolidation after replay phase.

        This applies:
        1. Weight decay toward baseline
        2. Tag capture (if enabled)
        3. Statistics collection

        Returns:
            Consolidation statistics
        """
        # Apply weight decay
        if self.config.apply_decay:
            self.stdp.apply_weight_decay(self.config.decay_baseline)

        # Capture tags
        captured_tags = []
        if self.tagger:
            captured_tags = self.tagger.capture_tags()

        self._consolidation_cycles += 1

        stats = self.stdp.get_stats()
        stats["consolidation_cycles"] = self._consolidation_cycles
        stats["total_sequences_replayed"] = self._total_sequences
        stats["captured_tags"] = len(captured_tags)

        logger.info(
            f"Consolidation cycle {self._consolidation_cycles}: "
            f"{stats['total_updates']} STDP updates, {len(captured_tags)} tags captured"
        )

        return stats

    def sync_weights_with_tags(self) -> int:
        """
        Synchronize STDP weights with synaptic tags.

        Tagged synapses that haven't been updated recently get their
        weights adjusted based on tag type (early vs late LTP).

        Returns:
            Number of weights synchronized
        """
        if not self.tagger:
            return 0

        synced = 0
        tags = self.tagger.get_tagged_synapses()

        for tag in tags:
            # Get current STDP weight
            current = self.stdp.get_weight(tag.source_id, tag.target_id)

            # Adjust based on tag type and influence
            if tag.tag_type == "late":
                # Late LTP tags strengthen significantly
                adjustment = self.config.tag_weight_influence * 0.2
            else:
                # Early LTP tags strengthen moderately
                adjustment = self.config.tag_weight_influence * 0.1

            new_weight = min(
                self.config.consolidation_max_weight,
                current + adjustment
            )
            self.stdp.set_weight(tag.source_id, tag.target_id, new_weight)
            synced += 1

        return synced

    def get_synapse_strength(self, pre_id: str, post_id: str) -> float:
        """
        Get the current strength of a synapse.

        Args:
            pre_id: Presynaptic entity ID
            post_id: Postsynaptic entity ID

        Returns:
            Current synaptic weight
        """
        return self.stdp.get_weight(pre_id, post_id)

    def get_strong_connections(
        self,
        threshold: float = 0.7
    ) -> list[tuple[str, str, float]]:
        """
        Get all synapses above a strength threshold.

        Args:
            threshold: Minimum weight threshold

        Returns:
            List of (pre_id, post_id, weight) tuples
        """
        strong = []
        for (pre_id, post_id), weight in self.stdp._weights.items():
            if weight >= threshold:
                strong.append((pre_id, post_id, weight))
        return sorted(strong, key=lambda x: x[2], reverse=True)

    def get_weak_connections(
        self,
        threshold: float = 0.3
    ) -> list[tuple[str, str, float]]:
        """
        Get all synapses below a strength threshold.

        Args:
            threshold: Maximum weight threshold

        Returns:
            List of (pre_id, post_id, weight) tuples
        """
        weak = []
        for (pre_id, post_id), weight in self.stdp._weights.items():
            if weight <= threshold:
                weak.append((pre_id, post_id, weight))
        return sorted(weak, key=lambda x: x[2])

    def prune_weak_synapses(
        self,
        threshold: float | None = None
    ) -> int:
        """
        Remove synapses below the pruning threshold.

        Args:
            threshold: Weight threshold (default: consolidation_min_weight)

        Returns:
            Number of synapses pruned
        """
        threshold = threshold or self.config.consolidation_min_weight

        to_prune = [
            key for key, weight in self.stdp._weights.items()
            if weight < threshold
        ]

        for key in to_prune:
            del self.stdp._weights[key]

        if to_prune:
            logger.info(f"Pruned {len(to_prune)} weak synapses")

        return len(to_prune)

    def get_stats(self) -> dict[str, Any]:
        """Get consolidation STDP statistics."""
        stats = self.stdp.get_stats()
        stats.update({
            "total_sequences_replayed": self._total_sequences,
            "consolidation_cycles": self._consolidation_cycles,
            "sync_with_tags": self.config.sync_with_tags,
        })
        return stats

    def save_state(self) -> dict[str, Any]:
        """Save consolidation STDP state."""
        return {
            "stdp_state": self.stdp.save_state(),
            "total_sequences": self._total_sequences,
            "total_updates": self._total_updates,
            "consolidation_cycles": self._consolidation_cycles,
        }

    def load_state(self, state: dict[str, Any]) -> None:
        """Load consolidation STDP state."""
        if "stdp_state" in state:
            self.stdp.load_state(state["stdp_state"])
        self._total_sequences = state.get("total_sequences", 0)
        self._total_updates = state.get("total_updates", 0)
        self._consolidation_cycles = state.get("consolidation_cycles", 0)

    def reset(self) -> None:
        """Reset all state."""
        self.stdp.clear_spikes()
        self.stdp._weights.clear()
        if self.tagger:
            self.tagger.capture_tags()  # Clear tags
        self._total_sequences = 0
        self._total_updates = 0
        self._consolidation_cycles = 0


# Singleton access for integration
_consolidation_stdp: ConsolidationSTDP | None = None


def get_consolidation_stdp(
    config: ConsolidationSTDPConfig | None = None
) -> ConsolidationSTDP:
    """Get or create the consolidation STDP instance."""
    global _consolidation_stdp
    if _consolidation_stdp is None:
        _consolidation_stdp = ConsolidationSTDP(config)
    return _consolidation_stdp


def reset_consolidation_stdp() -> None:
    """Reset the consolidation STDP singleton."""
    global _consolidation_stdp
    if _consolidation_stdp:
        _consolidation_stdp.reset()
    _consolidation_stdp = None
