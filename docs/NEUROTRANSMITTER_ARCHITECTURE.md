# Neurotransmitter-Inspired Systems for T4DM

**Author**: Geoffrey Hinton AI Architect Agent
**Date**: 2025-12-06 (Updated: 2026-01-03)
**Status**: Architectural Design + Phase 1 Implementation

---

## Executive Summary

T4DM already implements a dopamine-like reward prediction error (RPE) system that modulates learning based on surprise. This document extends the neuromodulatory framework with four additional systems: **Norepinephrine**, **Acetylcholine**, **Serotonin**, and **GABA/Glutamate balance**. Each provides computationally distinct functions that the brain uses to orchestrate memory encoding, retrieval, and consolidation.

The key insight from neuroscience is that these systems don't operate in isolation - they form an *interacting neuromodulatory orchestra* where each transmitter influences the others. We must design with this integration in mind.

---

## Current State Assessment

### What Exists

1. **DopamineSystem** (`/mnt/projects/t4d/t4dm/src/t4dm/learning/dopamine.py`)
   - Computes reward prediction error: delta = actual - expected
   - Modulates learning rate by surprise magnitude
   - Maintains per-memory value estimates via EMA
   - Well-integrated with EpisodicMemory's reconsolidation

2. **ReconsolidationEngine** (`/mnt/projects/t4d/t4dm/src/t4dm/learning/reconsolidation.py`)
   - Updates embeddings based on retrieval outcomes
   - Importance-weighted learning rate adjustment
   - Dopamine-modulated via `lr_modulation` parameter

3. **PatternSeparation/DentateGyrus** (`/mnt/projects/t4d/t4dm/src/t4dm/memory/pattern_separation.py`)
   - Orthogonalizes similar inputs to reduce interference
   - Adaptive separation strength based on similarity
   - Optional sparse coding

4. **NeuroSymbolicReasoner** (`/mnt/projects/t4d/t4dm/src/t4dm/learning/neuro_symbolic.py`)
   - Learned fusion weights for multi-source scoring
   - Symbolic graph traversal for credit attribution
   - ListMLE training for ranking optimization

### What's Missing

The current system lacks mechanisms for:
- **Attention/arousal modulation** (norepinephrine)
- **Encoding vs retrieval mode switching** (acetylcholine)
- **Long-term temporal credit assignment** (serotonin)
- **Competitive dynamics in retrieval** (GABA/glutamate)

---

## Architectural Critique (Hinton Perspective)

The existing dopamine implementation is sound - it captures the essential insight that learning should be driven by prediction error, not raw outcomes. However, several limitations emerge when viewed through a neural learning lens:

1. **Static arousal**: The system has no mechanism to modulate global gain based on context novelty or urgency. A surprising query should heighten attention across all systems.

2. **Fixed encoding/retrieval balance**: Biological systems shift between "learning mode" (high ACh, hippocampal encoding dominates) and "retrieval mode" (low ACh, cortical pattern completion dominates). WW currently has no such switching.

3. **Myopic credit assignment**: Dopamine handles immediate RPE, but longer temporal horizons require different mechanisms. Serotonin in the brain appears to encode patience and delayed reward expectations.

4. **Winner-take-all missing**: Retrieval returns ranked results, but there's no lateral inhibition to sharpen the representation. GABA-like inhibition would implement soft winner-take-all dynamics.

---

## Neurotransmitter System Designs

---

## 1. Norepinephrine System (Locus Coeruleus)

### Biological Basis

The locus coeruleus (LC) is a small brainstem nucleus that projects widely throughout the cortex and hippocampus. It modulates:

- **Arousal/alertness**: Tonic NE levels set overall gain
- **Novelty detection**: Phasic NE bursts signal unexpected stimuli
- **Network reset**: High NE promotes exploration over exploitation
- **Attention**: Enhances signal-to-noise ratio in neural processing

Key papers:
- Aston-Jones & Cohen (2005): Adaptive Gain Theory
- Dayan & Yu (2006): NE signals unexpected uncertainty

### Computational Analog

Norepinephrine provides a **global gain modulator** that scales the system's responsiveness based on:
1. **Query novelty**: How different is this query from recent history?
2. **Urgency signals**: External cues indicating importance
3. **Uncertainty estimation**: Entropy of retrieval results

High NE should:
- Increase learning rates across all systems
- Broaden retrieval (more exploration)
- Heighten pattern separation (reduce interference)
- Promote reconsolidation (memories become more labile)

### Implementation Design

```python
# /mnt/projects/t4d/t4dm/src/t4dm/learning/norepinephrine.py

"""
Norepinephrine-like Arousal/Attention System for T4DM.

Biological Basis:
- Locus coeruleus modulates cortical and hippocampal gain
- Tonic NE sets baseline arousal; phasic NE signals novelty
- High NE promotes exploration; low NE promotes exploitation
- NE enhances pattern separation in dentate gyrus

Implementation:
- Tracks query novelty via embedding distance from recent history
- Computes uncertainty from retrieval result entropy
- Modulates global gain affecting all downstream systems
- Influences exploration-exploitation balance in retrieval
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from datetime import datetime
from typing import Optional, Protocol
from collections import deque
from uuid import UUID

import numpy as np

logger = logging.getLogger(__name__)


@dataclass
class ArousalState:
    """Current state of the norepinephrine system."""

    tonic_level: float  # Baseline arousal [0, 1]
    phasic_burst: float  # Transient novelty signal [0, 1]
    combined_gain: float  # Effective gain multiplier
    novelty_score: float  # How novel was the current query
    uncertainty_score: float  # Entropy of retrieval results
    timestamp: datetime = field(default_factory=datetime.now)

    @property
    def exploration_bias(self) -> float:
        """Higher values favor exploration over exploitation."""
        return min(1.0, self.combined_gain * 0.5)


class NorepinephrineSystem:
    """
    Global arousal and attention modulator inspired by locus coeruleus.

    The LC-NE system implements adaptive gain control:
    - Novelty detection via query embedding distance
    - Uncertainty estimation from retrieval entropy
    - Global gain modulation affecting all learning systems
    - Exploration-exploitation balance

    Key insight: NE doesn't encode specific information, but rather
    modulates HOW information is processed across the entire system.
    """

    def __init__(
        self,
        baseline_arousal: float = 0.5,
        novelty_decay: float = 0.95,
        history_size: int = 50,
        phasic_decay: float = 0.7,
        min_gain: float = 0.5,
        max_gain: float = 2.0,
        uncertainty_weight: float = 0.3,
        novelty_weight: float = 0.7
    ):
        """
        Initialize norepinephrine system.

        Args:
            baseline_arousal: Tonic NE level when no novelty
            novelty_decay: How fast novelty habituates
            history_size: Number of recent queries to track
            phasic_decay: Decay rate for phasic bursts
            min_gain: Minimum gain multiplier
            max_gain: Maximum gain multiplier
            uncertainty_weight: Weight for uncertainty in gain
            novelty_weight: Weight for novelty in gain
        """
        self.baseline_arousal = baseline_arousal
        self.novelty_decay = novelty_decay
        self.history_size = history_size
        self.phasic_decay = phasic_decay
        self.min_gain = min_gain
        self.max_gain = max_gain
        self.uncertainty_weight = uncertainty_weight
        self.novelty_weight = novelty_weight

        # Recent query history for novelty detection
        self._query_history: deque[np.ndarray] = deque(maxlen=history_size)

        # Current state
        self._tonic_level = baseline_arousal
        self._phasic_level = 0.0
        self._current_state: Optional[ArousalState] = None

        # Arousal history for analysis
        self._arousal_history: list[ArousalState] = []

    def compute_novelty(self, query_embedding: np.ndarray) -> float:
        """
        Compute novelty of query relative to recent history.

        Uses average distance from recent queries. High distance = high novelty.

        Args:
            query_embedding: Current query vector

        Returns:
            Novelty score [0, 1]
        """
        if not self._query_history:
            return 1.0  # First query is maximally novel

        query = np.asarray(query_embedding, dtype=np.float32)
        query_norm = np.linalg.norm(query)
        if query_norm > 0:
            query = query / query_norm

        # Compute distance from each historical query
        distances = []
        for hist_query in self._query_history:
            # Cosine distance: 1 - cosine_similarity
            similarity = np.dot(query, hist_query)
            distance = 1.0 - similarity
            distances.append(distance)

        # Recency-weighted average (recent queries matter more)
        weights = np.array([
            self.novelty_decay ** i
            for i in range(len(distances) - 1, -1, -1)
        ])
        weights = weights / weights.sum()

        avg_distance = np.average(distances, weights=weights)

        # Normalize to [0, 1] (distance ranges from 0 to 2 for unit vectors)
        novelty = min(1.0, avg_distance / 1.5)

        return float(novelty)

    def compute_uncertainty(
        self,
        retrieval_scores: list[float]
    ) -> float:
        """
        Compute uncertainty from retrieval result distribution.

        Uses entropy of score distribution. Uniform = high uncertainty,
        peaked = low uncertainty.

        Args:
            retrieval_scores: Scores of retrieved items

        Returns:
            Uncertainty score [0, 1]
        """
        if not retrieval_scores or len(retrieval_scores) < 2:
            return 0.5  # Default uncertainty

        scores = np.array(retrieval_scores, dtype=np.float32)

        # Normalize to probability distribution
        scores = scores - scores.min() + 1e-8
        probs = scores / scores.sum()

        # Compute entropy
        entropy = -np.sum(probs * np.log(probs + 1e-10))

        # Normalize by maximum entropy (uniform distribution)
        max_entropy = np.log(len(scores))
        normalized_entropy = entropy / max_entropy if max_entropy > 0 else 0.5

        return float(normalized_entropy)

    def update(
        self,
        query_embedding: np.ndarray,
        retrieval_scores: Optional[list[float]] = None,
        external_urgency: float = 0.0
    ) -> ArousalState:
        """
        Update arousal state based on current query.

        Args:
            query_embedding: Current query vector
            retrieval_scores: Optional scores from retrieval
            external_urgency: Optional external urgency signal [0, 1]

        Returns:
            Current arousal state
        """
        # Normalize query
        query = np.asarray(query_embedding, dtype=np.float32)
        query_norm = np.linalg.norm(query)
        if query_norm > 0:
            query = query / query_norm

        # Compute novelty
        novelty = self.compute_novelty(query)

        # Compute uncertainty if scores provided
        uncertainty = 0.5
        if retrieval_scores:
            uncertainty = self.compute_uncertainty(retrieval_scores)

        # Update phasic level (novelty burst)
        # Phasic = max of current novelty and decayed previous phasic
        self._phasic_level = max(
            novelty * 0.8,  # New novelty burst
            self._phasic_level * self.phasic_decay  # Decay previous
        )

        # Update tonic level (slow adaptation)
        # High sustained novelty raises tonic; low novelty returns to baseline
        target_tonic = self.baseline_arousal + 0.3 * novelty
        self._tonic_level += 0.1 * (target_tonic - self._tonic_level)

        # Combine into gain
        arousal = (
            self._tonic_level +
            self._phasic_level +
            0.2 * external_urgency
        )

        # Weight by uncertainty and novelty
        weighted_arousal = (
            self.novelty_weight * novelty +
            self.uncertainty_weight * uncertainty
        ) * arousal

        # Compute final gain
        combined_gain = self.min_gain + (self.max_gain - self.min_gain) * weighted_arousal
        combined_gain = np.clip(combined_gain, self.min_gain, self.max_gain)

        # Store query in history
        self._query_history.append(query.copy())

        # Create state
        self._current_state = ArousalState(
            tonic_level=float(self._tonic_level),
            phasic_burst=float(self._phasic_level),
            combined_gain=float(combined_gain),
            novelty_score=novelty,
            uncertainty_score=uncertainty
        )

        self._arousal_history.append(self._current_state)

        logger.debug(
            f"NE update: novelty={novelty:.3f}, uncertainty={uncertainty:.3f}, "
            f"gain={combined_gain:.3f}"
        )

        return self._current_state

    def get_current_gain(self) -> float:
        """Get current gain multiplier for downstream systems."""
        if self._current_state is None:
            return 1.0
        return self._current_state.combined_gain

    def modulate_learning_rate(self, base_lr: float) -> float:
        """
        Modulate learning rate by current arousal.

        Higher arousal = higher learning rate (more plastic).

        Args:
            base_lr: Base learning rate

        Returns:
            Modulated learning rate
        """
        return base_lr * self.get_current_gain()

    def modulate_retrieval_threshold(self, base_threshold: float) -> float:
        """
        Modulate retrieval threshold by arousal.

        Higher arousal = lower threshold (broader search, more exploration).

        Args:
            base_threshold: Base similarity threshold

        Returns:
            Modulated threshold
        """
        gain = self.get_current_gain()
        # Inverse relationship: high gain = low threshold
        return base_threshold / gain

    def modulate_separation_strength(self, base_separation: float) -> float:
        """
        Modulate pattern separation strength by arousal.

        Higher arousal = stronger separation (reduce interference).

        Args:
            base_separation: Base separation magnitude

        Returns:
            Modulated separation
        """
        return base_separation * self.get_current_gain()

    def get_stats(self) -> dict:
        """Get norepinephrine system statistics."""
        if not self._arousal_history:
            return {
                "total_updates": 0,
                "avg_novelty": 0.0,
                "avg_uncertainty": 0.0,
                "avg_gain": 1.0,
                "current_tonic": self._tonic_level,
                "current_phasic": self._phasic_level
            }

        return {
            "total_updates": len(self._arousal_history),
            "avg_novelty": float(np.mean([s.novelty_score for s in self._arousal_history])),
            "avg_uncertainty": float(np.mean([s.uncertainty_score for s in self._arousal_history])),
            "avg_gain": float(np.mean([s.combined_gain for s in self._arousal_history])),
            "current_tonic": self._tonic_level,
            "current_phasic": self._phasic_level
        }

    def reset_history(self) -> None:
        """Clear query and arousal history."""
        self._query_history.clear()
        self._arousal_history.clear()
        self._tonic_level = self.baseline_arousal
        self._phasic_level = 0.0


__all__ = [
    "ArousalState",
    "NorepinephrineSystem",
]
```

### Integration Points

1. **EpisodicMemory.recall()**: Modulate retrieval threshold based on arousal
2. **DentateGyrus.encode()**: Increase separation strength under high arousal
3. **ReconsolidationEngine.reconsolidate()**: Scale learning rate by gain
4. **NeuroSymbolicReasoner**: Bias fusion weights toward exploration

```python
# Integration in EpisodicMemory.__init__():
from t4dm.learning.norepinephrine import NorepinephrineSystem
self.norepinephrine = NorepinephrineSystem()

# Integration in EpisodicMemory.recall():
async def recall(self, query: str, ...):
    query_emb = await self.embedding.embed_query(query)

    # Update arousal based on query novelty
    arousal_state = self.norepinephrine.update(query_emb)

    # Modulate retrieval threshold (lower = broader search)
    effective_threshold = self.norepinephrine.modulate_retrieval_threshold(0.5)

    results = await self.vector_store.search(
        ...,
        score_threshold=effective_threshold,
    )

    # Update arousal with retrieval uncertainty
    if results:
        scores = [r.score for r in results]
        self.norepinephrine.update(query_emb, retrieval_scores=scores)
```

---

## 2. Acetylcholine System (Basal Forebrain)

### Biological Basis

Acetylcholine from the basal forebrain modulates the hippocampus and cortex:

- **Encoding mode**: High ACh favors new learning over retrieval
- **Retrieval mode**: Low ACh favors pattern completion from cortex
- **Attention gating**: ACh enhances attended stimuli, suppresses unattended
- **Learning rate modulation**: ACh increases synaptic plasticity

Key papers:
- Hasselmo (2006): The role of ACh in memory
- Hasselmo & McGaughy (2004): High ACh supports encoding

### Computational Analog

Acetylcholine implements an **encoding/retrieval mode switch**:

- **High ACh (encoding mode)**:
  - Prioritize new information over stored patterns
  - Strengthen hippocampal-like fast learning
  - Reduce cortical pattern completion

- **Low ACh (retrieval mode)**:
  - Prioritize stored patterns over new encoding
  - Enhance pattern completion from partial cues
  - Reduce interference from new information

The system should detect when to encode (novel, important) vs retrieve (familiar, answerable from memory).

### Implementation Design

```python
# /mnt/projects/t4d/t4dm/src/t4dm/learning/acetylcholine.py

"""
Acetylcholine-like Encoding/Retrieval Mode Switch for T4DM.

Biological Basis:
- High ACh in hippocampus promotes encoding, suppresses retrieval
- Low ACh promotes retrieval/pattern completion from cortex
- ACh levels modulated by novelty, attention, and uncertainty
- Sleep-wake cycle shifts ACh levels (low during SWS consolidation)

Implementation:
- Tracks encoding vs retrieval demands from query characteristics
- Modulates balance between new learning and pattern completion
- Influences reconsolidation (high ACh = more labile memories)
- Gates attention between memory systems
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum
from typing import Optional
from uuid import UUID

import numpy as np

logger = logging.getLogger(__name__)


class CognitiveMode(str, Enum):
    """Current cognitive mode based on ACh level."""
    ENCODING = "encoding"      # High ACh: prioritize learning
    BALANCED = "balanced"      # Moderate ACh: normal operation
    RETRIEVAL = "retrieval"    # Low ACh: prioritize recall


@dataclass
class AcetylcholineState:
    """Current state of the acetylcholine system."""

    ach_level: float  # ACh concentration [0, 1]
    mode: CognitiveMode  # Current cognitive mode
    encoding_weight: float  # Weight for new encoding [0, 1]
    retrieval_weight: float  # Weight for pattern completion [0, 1]
    attention_gate: float  # Attention gating strength [0, 1]
    timestamp: datetime = field(default_factory=datetime.now)

    @property
    def learning_rate_modifier(self) -> float:
        """Higher ACh = higher learning rate."""
        return 0.5 + self.ach_level

    @property
    def pattern_completion_strength(self) -> float:
        """Lower ACh = stronger pattern completion."""
        return 1.0 - self.ach_level * 0.6


class AcetylcholineSystem:
    """
    Encoding/retrieval mode switch inspired by basal forebrain ACh.

    The cholinergic system dynamically balances:
    1. Encoding new information (high ACh)
    2. Retrieving stored patterns (low ACh)
    3. Attention allocation (moderate-high ACh)

    Mode detection is based on:
    - Query type (question vs statement)
    - Novelty (novel content -> encoding)
    - Memory match quality (good matches -> retrieval)
    - Explicit signals (user marking something important)
    """

    def __init__(
        self,
        baseline_ach: float = 0.5,
        encoding_threshold: float = 0.7,
        retrieval_threshold: float = 0.3,
        adaptation_rate: float = 0.2,
        min_ach: float = 0.1,
        max_ach: float = 0.9
    ):
        """
        Initialize acetylcholine system.

        Args:
            baseline_ach: Default ACh level
            encoding_threshold: ACh level above which encoding dominates
            retrieval_threshold: ACh level below which retrieval dominates
            adaptation_rate: How fast ACh adapts to demands
            min_ach: Minimum ACh level
            max_ach: Maximum ACh level
        """
        self.baseline_ach = baseline_ach
        self.encoding_threshold = encoding_threshold
        self.retrieval_threshold = retrieval_threshold
        self.adaptation_rate = adaptation_rate
        self.min_ach = min_ach
        self.max_ach = max_ach

        # Current ACh level
        self._ach_level = baseline_ach
        self._current_state: Optional[AcetylcholineState] = None

        # History for analysis
        self._state_history: list[AcetylcholineState] = []

    def compute_encoding_demand(
        self,
        query_novelty: float,
        is_statement: bool = False,
        explicit_importance: Optional[float] = None
    ) -> float:
        """
        Compute demand for encoding mode.

        Args:
            query_novelty: Novelty score from NE system [0, 1]
            is_statement: True if query is a statement (not question)
            explicit_importance: Optional user-indicated importance

        Returns:
            Encoding demand [0, 1]
        """
        demand = 0.0

        # Novel content demands encoding
        demand += 0.4 * query_novelty

        # Statements (vs questions) lean toward encoding
        if is_statement:
            demand += 0.2

        # Explicit importance strongly favors encoding
        if explicit_importance is not None:
            demand += 0.4 * explicit_importance

        return min(1.0, demand)

    def compute_retrieval_demand(
        self,
        is_question: bool = False,
        memory_match_quality: Optional[float] = None,
        query_specificity: float = 0.5
    ) -> float:
        """
        Compute demand for retrieval mode.

        Args:
            is_question: True if query is a question
            memory_match_quality: Quality of best memory match [0, 1]
            query_specificity: How specific/answerable the query is [0, 1]

        Returns:
            Retrieval demand [0, 1]
        """
        demand = 0.0

        # Questions demand retrieval
        if is_question:
            demand += 0.3

        # Good memory matches suggest retrieval
        if memory_match_quality is not None:
            demand += 0.4 * memory_match_quality

        # Specific queries lean toward retrieval
        demand += 0.2 * query_specificity

        return min(1.0, demand)

    def update(
        self,
        encoding_demand: float,
        retrieval_demand: float,
        arousal_gain: float = 1.0
    ) -> AcetylcholineState:
        """
        Update ACh level based on current demands.

        Args:
            encoding_demand: Demand for encoding mode [0, 1]
            retrieval_demand: Demand for retrieval mode [0, 1]
            arousal_gain: Gain from NE system (high arousal boosts ACh)

        Returns:
            Current ACh state
        """
        # Compute target ACh level
        # High encoding demand -> high ACh
        # High retrieval demand -> low ACh
        demand_diff = encoding_demand - retrieval_demand
        target_ach = self.baseline_ach + 0.4 * demand_diff

        # Arousal modulates ACh (novel/urgent -> higher ACh)
        target_ach *= arousal_gain

        # Clamp to bounds
        target_ach = np.clip(target_ach, self.min_ach, self.max_ach)

        # Adapt toward target
        self._ach_level += self.adaptation_rate * (target_ach - self._ach_level)

        # Determine mode
        if self._ach_level >= self.encoding_threshold:
            mode = CognitiveMode.ENCODING
        elif self._ach_level <= self.retrieval_threshold:
            mode = CognitiveMode.RETRIEVAL
        else:
            mode = CognitiveMode.BALANCED

        # Compute weights
        # Encoding weight increases with ACh
        encoding_weight = self._ach_level
        retrieval_weight = 1.0 - self._ach_level * 0.6

        # Attention gate (ACh enhances attention)
        attention_gate = 0.5 + 0.5 * self._ach_level

        # Create state
        self._current_state = AcetylcholineState(
            ach_level=float(self._ach_level),
            mode=mode,
            encoding_weight=float(encoding_weight),
            retrieval_weight=float(retrieval_weight),
            attention_gate=float(attention_gate)
        )

        self._state_history.append(self._current_state)

        logger.debug(
            f"ACh update: level={self._ach_level:.3f}, mode={mode.value}, "
            f"enc={encoding_weight:.2f}, ret={retrieval_weight:.2f}"
        )

        return self._current_state

    def get_current_mode(self) -> CognitiveMode:
        """Get current cognitive mode."""
        if self._current_state is None:
            return CognitiveMode.BALANCED
        return self._current_state.mode

    def should_prioritize_encoding(self) -> bool:
        """Check if system should prioritize encoding over retrieval."""
        return self.get_current_mode() == CognitiveMode.ENCODING

    def should_prioritize_retrieval(self) -> bool:
        """Check if system should prioritize retrieval over encoding."""
        return self.get_current_mode() == CognitiveMode.RETRIEVAL

    def modulate_learning_rate(self, base_lr: float) -> float:
        """
        Modulate learning rate by ACh level.

        Higher ACh = higher learning rate (encoding mode).

        Args:
            base_lr: Base learning rate

        Returns:
            Modulated learning rate
        """
        if self._current_state is None:
            return base_lr
        return base_lr * self._current_state.learning_rate_modifier

    def modulate_pattern_completion(self, base_strength: float) -> float:
        """
        Modulate pattern completion strength by ACh level.

        Lower ACh = stronger pattern completion (retrieval mode).

        Args:
            base_strength: Base pattern completion strength

        Returns:
            Modulated strength
        """
        if self._current_state is None:
            return base_strength
        return base_strength * self._current_state.pattern_completion_strength

    def get_attention_weights(
        self,
        memory_sources: list[str]
    ) -> dict[str, float]:
        """
        Compute attention weights for different memory sources.

        In encoding mode: weight hippocampal (episodic) sources higher
        In retrieval mode: weight cortical (semantic) sources higher

        Args:
            memory_sources: List of memory source names

        Returns:
            Source name -> attention weight
        """
        if self._current_state is None:
            return {src: 1.0 for src in memory_sources}

        weights = {}
        mode = self._current_state.mode

        for src in memory_sources:
            if mode == CognitiveMode.ENCODING:
                # Encoding: boost episodic, reduce semantic
                if "episodic" in src.lower():
                    weights[src] = 1.2
                elif "semantic" in src.lower():
                    weights[src] = 0.8
                else:
                    weights[src] = 1.0
            elif mode == CognitiveMode.RETRIEVAL:
                # Retrieval: boost semantic, reduce episodic
                if "semantic" in src.lower():
                    weights[src] = 1.2
                elif "episodic" in src.lower():
                    weights[src] = 0.8
                else:
                    weights[src] = 1.0
            else:
                weights[src] = 1.0

        return weights

    def get_stats(self) -> dict:
        """Get acetylcholine system statistics."""
        if not self._state_history:
            return {
                "total_updates": 0,
                "avg_ach": self.baseline_ach,
                "mode_counts": {},
                "current_mode": "balanced"
            }

        mode_counts = {}
        for state in self._state_history:
            mode_counts[state.mode.value] = mode_counts.get(state.mode.value, 0) + 1

        return {
            "total_updates": len(self._state_history),
            "avg_ach": float(np.mean([s.ach_level for s in self._state_history])),
            "mode_counts": mode_counts,
            "current_mode": self._current_state.mode.value if self._current_state else "balanced"
        }

    def reset(self) -> None:
        """Reset to baseline state."""
        self._ach_level = self.baseline_ach
        self._current_state = None
        self._state_history.clear()


__all__ = [
    "CognitiveMode",
    "AcetylcholineState",
    "AcetylcholineSystem",
]
```

### Integration Points

1. **EpisodicMemory**: Weight encoding vs pattern completion based on mode
2. **PatternCompletion (CA3)**: Modulate completion strength by ACh
3. **ReconsolidationEngine**: Only reconsolidate in encoding mode
4. **UnifiedMemory**: Attention-weight different memory sources

---

## 3. Serotonin System

### Biological Basis

Serotonin from the raphe nuclei modulates:

- **Temporal discounting**: 5-HT promotes patience for delayed rewards
- **Long-term credit assignment**: Connects actions to distant outcomes
- **Mood/baseline state**: Sets overall emotional valence
- **Impulse control**: Inhibits immediate reward-seeking

Key papers:
- Daw et al. (2002): Serotonin and temporal discounting
- Cools et al. (2008): 5-HT and behavioral inhibition

### Computational Analog

Serotonin provides **long-horizon temporal credit assignment**:

- **Patience parameter**: How far into the future to look for outcomes
- **Decay rate for eligibility**: How fast past actions become ineligible
- **Baseline mood**: Affects value estimates across the board
- **Temporal abstraction**: Group actions into macro-actions for distant credit

Dopamine handles immediate RPE; serotonin handles "was this useful for the long-term goal?"

### Implementation Design

```python
# /mnt/projects/t4d/t4dm/src/t4dm/learning/serotonin.py

"""
Serotonin-like Long-Term Credit Assignment for T4DM.

Biological Basis:
- 5-HT modulates patience and temporal discounting
- Low 5-HT leads to impulsive, short-term choices
- High 5-HT promotes waiting for larger delayed rewards
- 5-HT interacts with dopamine for temporal credit assignment

Implementation:
- Maintains eligibility traces that decay slowly
- Computes long-horizon value estimates
- Modulates temporal discounting in value learning
- Tracks session-level and multi-session outcomes
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from typing import Optional, Dict, List
from uuid import UUID
from collections import defaultdict

import numpy as np

logger = logging.getLogger(__name__)


@dataclass
class EligibilityTrace:
    """
    Trace connecting a memory to future outcomes.

    Eligibility decays exponentially with time, allowing
    credit to flow back to earlier memories that contributed
    to later outcomes.
    """

    memory_id: UUID
    initial_strength: float
    created_at: datetime
    decay_rate: float = 0.95  # Per-hour decay

    def get_current_strength(self) -> float:
        """Get current eligibility accounting for decay."""
        elapsed_hours = (datetime.now() - self.created_at).total_seconds() / 3600
        return self.initial_strength * (self.decay_rate ** elapsed_hours)

    @property
    def is_expired(self) -> bool:
        """Check if trace has decayed below threshold."""
        return self.get_current_strength() < 0.01


@dataclass
class TemporalContext:
    """Context for long-term outcome tracking."""

    session_id: str
    goal_description: Optional[str] = None
    start_time: datetime = field(default_factory=datetime.now)
    outcome_received: bool = False
    final_outcome: Optional[float] = None


class SerotoninSystem:
    """
    Long-term credit assignment inspired by serotonergic modulation.

    While dopamine signals immediate prediction errors, serotonin
    supports patience and long-term value estimation:

    1. Eligibility traces connect past memories to future outcomes
    2. Temporal discounting controls how far to look ahead
    3. Session-level outcomes update memories used throughout
    4. Mood/baseline affects all value estimates

    Key insight: Some memories are valuable not for immediate use,
    but because they set up future successes. Serotonin captures this.
    """

    def __init__(
        self,
        base_discount_rate: float = 0.99,  # Per-step discount
        eligibility_decay: float = 0.95,   # Per-hour decay
        trace_lifetime_hours: float = 24.0,
        baseline_mood: float = 0.5,
        mood_adaptation_rate: float = 0.1,
        max_traces_per_memory: int = 10
    ):
        """
        Initialize serotonin system.

        Args:
            base_discount_rate: Gamma for temporal discounting
            eligibility_decay: How fast traces decay per hour
            trace_lifetime_hours: Maximum trace lifetime
            baseline_mood: Default mood level [0, 1]
            mood_adaptation_rate: How fast mood adapts
            max_traces_per_memory: Max traces to maintain per memory
        """
        self.base_discount_rate = base_discount_rate
        self.eligibility_decay = eligibility_decay
        self.trace_lifetime_hours = trace_lifetime_hours
        self.baseline_mood = baseline_mood
        self.mood_adaptation_rate = mood_adaptation_rate
        self.max_traces_per_memory = max_traces_per_memory

        # Current mood level
        self._mood = baseline_mood

        # Eligibility traces per memory
        self._traces: Dict[str, List[EligibilityTrace]] = defaultdict(list)

        # Active temporal contexts (ongoing sessions/goals)
        self._active_contexts: Dict[str, TemporalContext] = {}

        # Long-term value estimates (learned over sessions)
        self._long_term_values: Dict[str, float] = {}

        # Statistics
        self._total_outcomes = 0
        self._positive_outcomes = 0

    def start_context(
        self,
        session_id: str,
        goal_description: Optional[str] = None
    ) -> None:
        """
        Start a new temporal context for tracking.

        Args:
            session_id: Unique session identifier
            goal_description: Optional description of session goal
        """
        self._active_contexts[session_id] = TemporalContext(
            session_id=session_id,
            goal_description=goal_description
        )
        logger.debug(f"Started temporal context: {session_id}")

    def add_eligibility(
        self,
        memory_id: UUID,
        strength: float = 1.0,
        context_id: Optional[str] = None
    ) -> None:
        """
        Add eligibility trace for a memory.

        Called when a memory is retrieved - makes it eligible
        for credit when outcomes arrive later.

        Args:
            memory_id: Memory that was used
            strength: Initial trace strength
            context_id: Optional context to associate with
        """
        mem_id_str = str(memory_id)

        trace = EligibilityTrace(
            memory_id=memory_id,
            initial_strength=strength,
            created_at=datetime.now(),
            decay_rate=self.eligibility_decay
        )

        self._traces[mem_id_str].append(trace)

        # Prune old traces
        self._traces[mem_id_str] = [
            t for t in self._traces[mem_id_str]
            if not t.is_expired
        ][-self.max_traces_per_memory:]

    def get_eligibility(self, memory_id: UUID) -> float:
        """
        Get current total eligibility for a memory.

        Args:
            memory_id: Memory to check

        Returns:
            Total eligibility (sum of active traces)
        """
        mem_id_str = str(memory_id)

        if mem_id_str not in self._traces:
            return 0.0

        total = sum(
            t.get_current_strength()
            for t in self._traces[mem_id_str]
            if not t.is_expired
        )

        return min(1.0, total)

    def receive_outcome(
        self,
        outcome_score: float,
        context_id: Optional[str] = None,
        decay_with_time: bool = True
    ) -> Dict[str, float]:
        """
        Receive an outcome and distribute credit via eligibility.

        This is the serotonin equivalent of dopamine's RPE, but
        it assigns credit across time, not just to immediate actions.

        Args:
            outcome_score: Outcome value [0, 1]
            context_id: Optional context this outcome belongs to
            decay_with_time: Whether to apply temporal decay

        Returns:
            Memory ID -> credit assigned
        """
        credits = {}

        # Update mood based on outcome
        self._mood += self.mood_adaptation_rate * (outcome_score - self._mood)
        self._mood = np.clip(self._mood, 0.0, 1.0)

        # Track statistics
        self._total_outcomes += 1
        if outcome_score > 0.5:
            self._positive_outcomes += 1

        # Distribute credit to eligible memories
        for mem_id_str, traces in self._traces.items():
            total_eligibility = sum(
                t.get_current_strength()
                for t in traces
                if not t.is_expired
            )

            if total_eligibility > 0:
                # Credit = eligibility * (outcome - baseline_mood)
                # This centers on mood, similar to advantage in RL
                advantage = outcome_score - self._mood
                credit = total_eligibility * advantage

                credits[mem_id_str] = credit

                # Update long-term value estimate
                current_value = self._long_term_values.get(mem_id_str, 0.5)
                learning_rate = 0.1 * total_eligibility
                new_value = current_value + learning_rate * advantage
                self._long_term_values[mem_id_str] = np.clip(new_value, 0.0, 1.0)

        # Mark context as complete if provided
        if context_id and context_id in self._active_contexts:
            self._active_contexts[context_id].outcome_received = True
            self._active_contexts[context_id].final_outcome = outcome_score

        logger.debug(
            f"Distributed credit to {len(credits)} memories, "
            f"outcome={outcome_score:.3f}, mood={self._mood:.3f}"
        )

        return credits

    def get_long_term_value(self, memory_id: UUID) -> float:
        """
        Get long-term value estimate for a memory.

        This represents how often this memory has led to
        positive long-term outcomes.

        Args:
            memory_id: Memory to check

        Returns:
            Long-term value estimate [0, 1]
        """
        return self._long_term_values.get(str(memory_id), 0.5)

    def compute_patience_factor(
        self,
        steps_to_outcome: int
    ) -> float:
        """
        Compute discount factor for delayed outcomes.

        Higher mood (5-HT) = more patience = less discounting.

        Args:
            steps_to_outcome: Expected steps until outcome

        Returns:
            Patience factor [0, 1]
        """
        # Base discount
        base_patience = self.base_discount_rate ** steps_to_outcome

        # Mood modulation: high mood reduces temporal discounting
        mood_bonus = 0.2 * self._mood
        effective_patience = base_patience + mood_bonus * (1 - base_patience)

        return float(np.clip(effective_patience, 0.0, 1.0))

    def modulate_value_by_mood(self, raw_value: float) -> float:
        """
        Modulate value estimate by current mood.

        Low mood pessimistically reduces values;
        high mood optimistically increases them.

        Args:
            raw_value: Unmodulated value

        Returns:
            Mood-modulated value
        """
        # Mood deviation from neutral
        mood_offset = self._mood - 0.5

        # Apply offset (bounded)
        modulated = raw_value + 0.2 * mood_offset
        return float(np.clip(modulated, 0.0, 1.0))

    def get_current_mood(self) -> float:
        """Get current mood level."""
        return self._mood

    def cleanup_expired_traces(self) -> int:
        """
        Remove expired eligibility traces.

        Returns:
            Number of traces removed
        """
        removed = 0
        for mem_id in list(self._traces.keys()):
            original_count = len(self._traces[mem_id])
            self._traces[mem_id] = [
                t for t in self._traces[mem_id]
                if not t.is_expired
            ]
            removed += original_count - len(self._traces[mem_id])

            if not self._traces[mem_id]:
                del self._traces[mem_id]

        return removed

    def get_stats(self) -> dict:
        """Get serotonin system statistics."""
        active_traces = sum(len(traces) for traces in self._traces.values())

        return {
            "current_mood": self._mood,
            "total_outcomes": self._total_outcomes,
            "positive_outcome_rate": (
                self._positive_outcomes / self._total_outcomes
                if self._total_outcomes > 0 else 0.5
            ),
            "memories_with_traces": len(self._traces),
            "active_traces": active_traces,
            "memories_with_long_term_values": len(self._long_term_values),
            "active_contexts": len(self._active_contexts)
        }

    def reset(self) -> None:
        """Reset to baseline state."""
        self._mood = self.baseline_mood
        self._traces.clear()
        self._active_contexts.clear()
        self._long_term_values.clear()
        self._total_outcomes = 0
        self._positive_outcomes = 0


__all__ = [
    "EligibilityTrace",
    "TemporalContext",
    "SerotoninSystem",
]
```

### Integration Points

1. **EpisodicMemory.recall()**: Add eligibility when memories are retrieved
2. **Session end hooks**: Trigger outcome distribution
3. **Reconsolidation**: Use long-term value for importance weighting
4. **NeuroSymbolicReasoner**: Include long-term value in fusion scoring

---

## 4. GABA/Glutamate Balance System

### Biological Basis

GABA (inhibitory) and glutamate (excitatory) create local circuit dynamics:

- **Lateral inhibition**: Activated neurons suppress neighbors
- **Winner-take-all**: Competition sharpens representations
- **Sparse coding**: Only a few neurons active at once
- **Oscillations**: Balance creates rhythmic activity (gamma, theta)

Key papers:
- Douglas & Martin (2004): Recurrent excitation in neocortex
- Rolls & Treves (1998): Sparse coding in hippocampus

### Computational Analog

GABA/glutamate implements **competitive dynamics in retrieval**:

- **Soft winner-take-all**: Top results inhibit lower-ranked ones
- **Lateral inhibition**: Similar memories compete
- **Sparse activation**: Only a few memories "win"
- **Oscillatory dynamics**: Iterative refinement of retrieval

This sharpens the retrieval output and reduces interference between similar memories.

### Implementation Design

```python
# /mnt/projects/t4d/t4dm/src/t4dm/learning/inhibition.py

"""
GABA/Glutamate-like Inhibitory Dynamics for T4DM.

Biological Basis:
- GABA neurons provide lateral inhibition in cortex/hippocampus
- Winner-take-all dynamics sharpen neural representations
- Balance of E/I determines network stability and sparsity
- Oscillations (gamma/theta) emerge from E/I balance

Implementation:
- Soft winner-take-all sharpens retrieval rankings
- Lateral inhibition between similar memories
- Sparse activation via competitive dynamics
- Configurable inhibition strength and sparsity
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from datetime import datetime
from typing import Optional, List, Tuple
from uuid import UUID

import numpy as np

logger = logging.getLogger(__name__)


@dataclass
class InhibitionResult:
    """Result of applying inhibitory dynamics."""

    original_scores: dict[str, float]
    inhibited_scores: dict[str, float]
    winners: list[str]  # IDs that "won" the competition
    sparsity: float  # Fraction of items surviving
    iterations: int  # Convergence iterations
    timestamp: datetime = field(default_factory=datetime.now)


class InhibitoryNetwork:
    """
    Competitive inhibition network inspired by GABA/glutamate balance.

    Implements soft winner-take-all dynamics that:
    1. Sharpen retrieval rankings
    2. Suppress weakly activated memories
    3. Reduce interference between similar items
    4. Create sparse output representations

    The dynamics are:
    - Excitation: Each item's score contributes to activation
    - Inhibition: Active items suppress similar/competing items
    - Convergence: Iterate until stable or max iterations
    """

    def __init__(
        self,
        inhibition_strength: float = 0.5,
        sparsity_target: float = 0.2,
        similarity_inhibition: bool = True,
        max_iterations: int = 5,
        convergence_threshold: float = 0.01,
        temperature: float = 1.0
    ):
        """
        Initialize inhibitory network.

        Args:
            inhibition_strength: How strongly winners suppress losers
            sparsity_target: Target fraction of surviving items
            similarity_inhibition: Whether similar items inhibit each other
            max_iterations: Maximum competition iterations
            convergence_threshold: Threshold for early stopping
            temperature: Softmax temperature for competition
        """
        self.inhibition_strength = inhibition_strength
        self.sparsity_target = sparsity_target
        self.similarity_inhibition = similarity_inhibition
        self.max_iterations = max_iterations
        self.convergence_threshold = convergence_threshold
        self.temperature = temperature

        # History for analysis
        self._history: list[InhibitionResult] = []

    def apply_inhibition(
        self,
        scores: dict[str, float],
        embeddings: Optional[dict[str, np.ndarray]] = None
    ) -> InhibitionResult:
        """
        Apply competitive inhibition to retrieval scores.

        Args:
            scores: Memory ID -> initial retrieval score
            embeddings: Optional Memory ID -> embedding for similarity

        Returns:
            InhibitionResult with sharpened scores
        """
        if not scores:
            return InhibitionResult(
                original_scores={},
                inhibited_scores={},
                winners=[],
                sparsity=0.0,
                iterations=0
            )

        ids = list(scores.keys())
        n = len(ids)

        # Convert to array
        activations = np.array([scores[id_] for id_ in ids], dtype=np.float32)
        original_activations = activations.copy()

        # Compute similarity matrix if embeddings provided
        similarity_matrix = None
        if self.similarity_inhibition and embeddings:
            similarity_matrix = self._compute_similarity_matrix(ids, embeddings)

        # Iterative competition
        for iteration in range(self.max_iterations):
            prev_activations = activations.copy()

            # Softmax for competition weights
            exp_act = np.exp(activations / self.temperature)
            competition_weights = exp_act / exp_act.sum()

            # Compute inhibition
            inhibition = np.zeros(n, dtype=np.float32)

            for i in range(n):
                for j in range(n):
                    if i == j:
                        continue

                    # Base inhibition from competing item j
                    base_inhibit = competition_weights[j] * self.inhibition_strength

                    # Scale by similarity if available
                    if similarity_matrix is not None:
                        base_inhibit *= similarity_matrix[i, j]

                    inhibition[i] += base_inhibit

            # Apply inhibition (bounded)
            activations = activations - inhibition
            activations = np.maximum(activations, 0.0)

            # Normalize to preserve total activation
            if activations.sum() > 0:
                activations = activations * (original_activations.sum() / activations.sum())

            # Check convergence
            delta = np.linalg.norm(activations - prev_activations)
            if delta < self.convergence_threshold:
                break

        # Determine winners (above dynamic threshold)
        if len(activations) > 0:
            threshold = np.percentile(activations, (1 - self.sparsity_target) * 100)
            winners = [ids[i] for i in range(n) if activations[i] >= threshold]
        else:
            winners = []

        # Convert back to dict
        inhibited_scores = {ids[i]: float(activations[i]) for i in range(n)}

        # Compute sparsity
        active_count = sum(1 for a in activations if a > 0.01)
        sparsity = active_count / n if n > 0 else 0.0

        result = InhibitionResult(
            original_scores=scores.copy(),
            inhibited_scores=inhibited_scores,
            winners=winners,
            sparsity=sparsity,
            iterations=iteration + 1
        )

        self._history.append(result)

        logger.debug(
            f"Inhibition: {n} items -> {len(winners)} winners, "
            f"sparsity={sparsity:.2f}, iterations={iteration + 1}"
        )

        return result

    def _compute_similarity_matrix(
        self,
        ids: list[str],
        embeddings: dict[str, np.ndarray]
    ) -> np.ndarray:
        """
        Compute pairwise similarity matrix.

        Args:
            ids: List of memory IDs
            embeddings: Memory ID -> embedding

        Returns:
            Similarity matrix [n, n]
        """
        n = len(ids)
        matrix = np.zeros((n, n), dtype=np.float32)

        # Get normalized embeddings
        normalized = {}
        for id_ in ids:
            if id_ in embeddings:
                emb = np.asarray(embeddings[id_], dtype=np.float32)
                norm = np.linalg.norm(emb)
                if norm > 0:
                    normalized[id_] = emb / norm

        # Compute similarities
        for i, id_i in enumerate(ids):
            for j, id_j in enumerate(ids):
                if i == j:
                    matrix[i, j] = 1.0
                elif id_i in normalized and id_j in normalized:
                    matrix[i, j] = float(np.dot(normalized[id_i], normalized[id_j]))

        return matrix

    def apply_lateral_inhibition(
        self,
        target_id: str,
        target_score: float,
        competitors: list[Tuple[str, float, float]]  # (id, score, similarity)
    ) -> float:
        """
        Apply lateral inhibition from competitors to target.

        Args:
            target_id: ID of target memory
            target_score: Target's initial score
            competitors: List of (id, score, similarity) tuples

        Returns:
            Inhibited score for target
        """
        inhibition = 0.0

        for comp_id, comp_score, similarity in competitors:
            if comp_id == target_id:
                continue

            # Inhibition proportional to competitor's score and similarity
            inhibition += comp_score * similarity * self.inhibition_strength

        # Apply inhibition (bounded)
        inhibited_score = max(0.0, target_score - inhibition)

        return inhibited_score

    def sharpen_ranking(
        self,
        ranked_results: list[Tuple[str, float]]
    ) -> list[Tuple[str, float]]:
        """
        Sharpen a ranked list by increasing separation.

        Higher-ranked items maintain scores; lower-ranked are suppressed.

        Args:
            ranked_results: List of (id, score) in descending order

        Returns:
            Re-ranked list with sharpened scores
        """
        if not ranked_results:
            return []

        # Convert to arrays
        ids = [r[0] for r in ranked_results]
        scores = np.array([r[1] for r in ranked_results], dtype=np.float32)

        # Compute rank-based suppression
        ranks = np.arange(len(scores))
        suppression = self.inhibition_strength * (ranks / len(scores))

        # Apply suppression
        sharpened = scores * (1 - suppression)
        sharpened = np.maximum(sharpened, 0.0)

        return [(ids[i], float(sharpened[i])) for i in range(len(ids))]

    def get_stats(self) -> dict:
        """Get inhibitory network statistics."""
        if not self._history:
            return {
                "total_applications": 0,
                "avg_sparsity": 0.0,
                "avg_iterations": 0.0,
                "avg_winners": 0.0
            }

        return {
            "total_applications": len(self._history),
            "avg_sparsity": float(np.mean([r.sparsity for r in self._history])),
            "avg_iterations": float(np.mean([r.iterations for r in self._history])),
            "avg_winners": float(np.mean([len(r.winners) for r in self._history]))
        }

    def reset_history(self) -> None:
        """Clear history."""
        self._history.clear()


class SparseRetrieval:
    """
    Sparse retrieval layer that applies GABA-like dynamics.

    Wraps a retrieval function and applies competitive inhibition
    to sharpen the output.
    """

    def __init__(
        self,
        inhibitory_network: InhibitoryNetwork,
        min_score_threshold: float = 0.1,
        max_results: int = 10
    ):
        """
        Initialize sparse retrieval layer.

        Args:
            inhibitory_network: Network for applying inhibition
            min_score_threshold: Minimum score to return
            max_results: Maximum results after sparsification
        """
        self.inhibitory = inhibitory_network
        self.min_score_threshold = min_score_threshold
        self.max_results = max_results

    def sparsify_results(
        self,
        results: list[Tuple[str, float]],
        embeddings: Optional[dict[str, np.ndarray]] = None
    ) -> list[Tuple[str, float]]:
        """
        Apply sparse coding to retrieval results.

        Args:
            results: List of (id, score) tuples
            embeddings: Optional embeddings for similarity-based inhibition

        Returns:
            Sparsified results (fewer items, sharper distribution)
        """
        if not results:
            return []

        # Convert to dict
        scores = {r[0]: r[1] for r in results}

        # Apply inhibition
        result = self.inhibitory.apply_inhibition(scores, embeddings)

        # Filter and sort
        filtered = [
            (id_, score)
            for id_, score in result.inhibited_scores.items()
            if score >= self.min_score_threshold
        ]

        # Sort by score descending
        filtered.sort(key=lambda x: x[1], reverse=True)

        # Limit results
        return filtered[:self.max_results]


__all__ = [
    "InhibitionResult",
    "InhibitoryNetwork",
    "SparseRetrieval",
]
```

### Integration Points

1. **EpisodicMemory.recall()**: Apply inhibition to sharpen rankings
2. **PatternSeparation**: Reinforce sparse activation
3. **NeuroSymbolicReasoner.fuse_scores()**: Apply after fusion
4. **UnifiedMemory**: Competitive dynamics across memory systems

---

## Integration Architecture

### Neuromodulator Orchestra

The four systems should interact in a coordinated manner:

```python
# /mnt/projects/t4d/t4dm/src/t4dm/learning/neuromodulators.py

"""
Integrated Neuromodulator System for T4DM.

Coordinates the four neuromodulatory systems:
1. Dopamine - Reward prediction error, surprise-driven learning
2. Norepinephrine - Arousal, attention, novelty detection
3. Acetylcholine - Encoding/retrieval mode switching
4. Serotonin - Long-term credit assignment, patience
5. GABA/Glutamate - Competitive inhibition, sparse representations

These systems interact:
- NE novelty -> ACh encoding mode
- Dopamine surprise -> NE arousal boost
- Serotonin mood -> Dopamine baseline adjustment
- GABA dynamics apply to all retrieval outputs
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from datetime import datetime
from typing import Optional, Dict
from uuid import UUID

import numpy as np

from t4dm.learning.dopamine import DopamineSystem
from t4dm.learning.norepinephrine import NorepinephrineSystem
from t4dm.learning.acetylcholine import AcetylcholineSystem
from t4dm.learning.serotonin import SerotoninSystem
from t4dm.learning.inhibition import InhibitoryNetwork

logger = logging.getLogger(__name__)


@dataclass
class NeuromodulatorState:
    """Combined state of all neuromodulatory systems."""

    dopamine_rpe: float  # Recent reward prediction error
    norepinephrine_gain: float  # Current arousal gain
    acetylcholine_mode: str  # encoding/balanced/retrieval
    serotonin_mood: float  # Current mood
    inhibition_sparsity: float  # Recent sparsity

    timestamp: datetime = field(default_factory=datetime.now)

    @property
    def effective_learning_rate(self) -> float:
        """Compute combined learning rate modifier."""
        # Base from NE arousal
        lr = self.norepinephrine_gain

        # Boost in encoding mode
        if self.acetylcholine_mode == "encoding":
            lr *= 1.3
        elif self.acetylcholine_mode == "retrieval":
            lr *= 0.7

        # Mood modulation
        lr *= (0.8 + 0.4 * self.serotonin_mood)

        return lr

    @property
    def exploration_exploitation_balance(self) -> float:
        """
        Compute balance between exploration and exploitation.

        Returns value in [-1, 1] where:
        - Negative = exploitation (use known patterns)
        - Positive = exploration (try new things)
        """
        balance = 0.0

        # High arousal -> exploration
        balance += (self.norepinephrine_gain - 1.0) * 0.5

        # Encoding mode -> exploration
        if self.acetylcholine_mode == "encoding":
            balance += 0.3
        elif self.acetylcholine_mode == "retrieval":
            balance -= 0.3

        # High mood -> slight exploitation (trust current approach)
        balance -= (self.serotonin_mood - 0.5) * 0.2

        return np.clip(balance, -1.0, 1.0)


class NeuromodulatorOrchestra:
    """
    Coordinates all neuromodulatory systems for unified brain-like dynamics.

    The orchestra ensures systems interact appropriately:
    - High novelty (NE) triggers encoding mode (ACh)
    - Surprise (DA) boosts arousal (NE)
    - Outcomes update both DA expectations and 5-HT traces
    - All retrieval passes through inhibitory sharpening (GABA)
    """

    def __init__(
        self,
        dopamine: Optional[DopamineSystem] = None,
        norepinephrine: Optional[NorepinephrineSystem] = None,
        acetylcholine: Optional[AcetylcholineSystem] = None,
        serotonin: Optional[SerotoninSystem] = None,
        inhibitory: Optional[InhibitoryNetwork] = None
    ):
        """
        Initialize neuromodulator orchestra.

        Args:
            dopamine: Dopamine system (created if None)
            norepinephrine: NE system (created if None)
            acetylcholine: ACh system (created if None)
            serotonin: 5-HT system (created if None)
            inhibitory: GABA network (created if None)
        """
        self.dopamine = dopamine or DopamineSystem()
        self.norepinephrine = norepinephrine or NorepinephrineSystem()
        self.acetylcholine = acetylcholine or AcetylcholineSystem()
        self.serotonin = serotonin or SerotoninSystem()
        self.inhibitory = inhibitory or InhibitoryNetwork()

        self._current_state: Optional[NeuromodulatorState] = None
        self._state_history: list[NeuromodulatorState] = []

    def process_query(
        self,
        query_embedding: np.ndarray,
        is_question: bool = False,
        explicit_importance: Optional[float] = None
    ) -> NeuromodulatorState:
        """
        Process a query through all neuromodulatory systems.

        Args:
            query_embedding: Query vector
            is_question: Whether query is a question
            explicit_importance: User-indicated importance

        Returns:
            Combined neuromodulator state
        """
        # 1. Update norepinephrine (novelty/arousal)
        ne_state = self.norepinephrine.update(query_embedding)

        # 2. Update acetylcholine (encoding/retrieval mode)
        encoding_demand = self.acetylcholine.compute_encoding_demand(
            query_novelty=ne_state.novelty_score,
            is_statement=not is_question,
            explicit_importance=explicit_importance
        )
        retrieval_demand = self.acetylcholine.compute_retrieval_demand(
            is_question=is_question
        )
        ach_state = self.acetylcholine.update(
            encoding_demand=encoding_demand,
            retrieval_demand=retrieval_demand,
            arousal_gain=ne_state.combined_gain
        )

        # 3. Create combined state (DA and 5-HT update on outcomes)
        self._current_state = NeuromodulatorState(
            dopamine_rpe=0.0,  # Updated when outcome received
            norepinephrine_gain=ne_state.combined_gain,
            acetylcholine_mode=ach_state.mode.value,
            serotonin_mood=self.serotonin.get_current_mood(),
            inhibition_sparsity=0.0  # Updated after retrieval
        )

        self._state_history.append(self._current_state)

        return self._current_state

    def process_retrieval(
        self,
        retrieved_ids: list[UUID],
        scores: dict[str, float],
        embeddings: Optional[dict[str, np.ndarray]] = None
    ) -> dict[str, float]:
        """
        Process retrieval results through inhibitory dynamics.

        Also adds eligibility traces for serotonin credit assignment.

        Args:
            retrieved_ids: IDs of retrieved memories
            scores: Memory ID -> score
            embeddings: Optional embeddings for similarity-based inhibition

        Returns:
            Inhibited scores
        """
        # Add eligibility traces for 5-HT
        for mem_id in retrieved_ids:
            self.serotonin.add_eligibility(mem_id, strength=scores.get(str(mem_id), 0.5))

        # Apply inhibitory dynamics
        result = self.inhibitory.apply_inhibition(scores, embeddings)

        # Update state
        if self._current_state:
            self._current_state.inhibition_sparsity = result.sparsity

        return result.inhibited_scores

    def process_outcome(
        self,
        memory_outcomes: Dict[str, float],
        session_outcome: Optional[float] = None
    ) -> Dict[str, float]:
        """
        Process outcomes through DA (immediate) and 5-HT (long-term).

        Args:
            memory_outcomes: Memory ID -> immediate outcome
            session_outcome: Optional overall session outcome

        Returns:
            Memory ID -> combined learning signal
        """
        learning_signals = {}

        # Process through dopamine (immediate RPE)
        rpes = self.dopamine.batch_compute_rpe(memory_outcomes)
        for mem_id, rpe in rpes.items():
            learning_signals[mem_id] = rpe.rpe

            # Surprise boosts arousal
            if rpe.surprise_magnitude > 0.3:
                # Could trigger NE boost here
                pass

        # Update dopamine expectations
        self.dopamine.batch_update_expectations(memory_outcomes)

        # Process through serotonin (long-term credit)
        if session_outcome is not None:
            credits = self.serotonin.receive_outcome(session_outcome)

            # Combine DA and 5-HT signals
            for mem_id, credit in credits.items():
                if mem_id in learning_signals:
                    # Weighted combination
                    learning_signals[mem_id] = (
                        0.7 * learning_signals[mem_id] +
                        0.3 * credit
                    )
                else:
                    learning_signals[mem_id] = credit

        # Update state
        if self._current_state and rpes:
            avg_rpe = np.mean([r.rpe for r in rpes.values()])
            self._current_state.dopamine_rpe = avg_rpe

        return learning_signals

    def get_learning_rate(self, base_lr: float) -> float:
        """
        Get combined learning rate from all systems.

        Args:
            base_lr: Base learning rate

        Returns:
            Modulated learning rate
        """
        if self._current_state is None:
            return base_lr

        return base_lr * self._current_state.effective_learning_rate

    def get_retrieval_threshold(self, base_threshold: float) -> float:
        """
        Get modulated retrieval threshold.

        Args:
            base_threshold: Base similarity threshold

        Returns:
            Modulated threshold
        """
        return self.norepinephrine.modulate_retrieval_threshold(base_threshold)

    def should_encode(self) -> bool:
        """Check if system should prioritize encoding."""
        return self.acetylcholine.should_prioritize_encoding()

    def should_retrieve(self) -> bool:
        """Check if system should prioritize retrieval."""
        return self.acetylcholine.should_prioritize_retrieval()

    def get_current_state(self) -> Optional[NeuromodulatorState]:
        """Get current neuromodulator state."""
        return self._current_state

    def get_stats(self) -> dict:
        """Get combined statistics from all systems."""
        return {
            "dopamine": self.dopamine.get_stats(),
            "norepinephrine": self.norepinephrine.get_stats(),
            "acetylcholine": self.acetylcholine.get_stats(),
            "serotonin": self.serotonin.get_stats(),
            "inhibitory": self.inhibitory.get_stats(),
            "total_states": len(self._state_history)
        }

    def reset(self) -> None:
        """Reset all systems to baseline."""
        self.dopamine.reset_history()
        self.norepinephrine.reset_history()
        self.acetylcholine.reset()
        self.serotonin.reset()
        self.inhibitory.reset_history()
        self._current_state = None
        self._state_history.clear()


__all__ = [
    "NeuromodulatorState",
    "NeuromodulatorOrchestra",
]
```

---

## Recommended Improvements

### Priority 1: Core Infrastructure

1. **Create norepinephrine.py** with NorepinephrineSystem
2. **Create acetylcholine.py** with AcetylcholineSystem
3. **Create serotonin.py** with SerotoninSystem
4. **Create inhibition.py** with InhibitoryNetwork
5. **Create neuromodulators.py** with NeuromodulatorOrchestra

### Priority 2: Integration

1. **Modify EpisodicMemory** to use NeuromodulatorOrchestra
2. **Update reconsolidation** to respect ACh mode
3. **Add pattern separation** NE modulation
4. **Integrate GABA** in recall methods

### Priority 3: Testing

1. **Unit tests** for each neuromodulator system
2. **Integration tests** for orchestra coordination
3. **Behavioral tests** for mode switching
4. **Performance tests** for inhibition scaling

---

## Research Directions

### Implemented (Phase 1 - v0.5.0)

1. ✅ **Sleep-like consolidation**: Now implemented via `t4dm.nca.oscillators.DeltaOscillator` and `t4dm.nca.sleep_spindles`
   - Delta oscillations (0.5-4 Hz) with up/down state dynamics
   - Sleep spindles (11-16 Hz) coupled to delta up-states
   - Adenosine-sensitive sleep pressure integration
   - See: `DeltaOscillator`, `SleepSpindleGenerator`, `SpindleDeltaCoupler`

2. ✅ **Oscillatory dynamics**: Theta/gamma/delta/alpha all implemented
   - Theta-gamma PAC for encoding/retrieval (`t4dm.nca.theta_gamma_integration`)
   - Delta for slow-wave sleep consolidation (`t4dm.nca.oscillators.DeltaOscillator`)
   - Alpha for inhibition (`t4dm.nca.oscillators.AlphaOscillator`)
   - See: `FrequencyBandGenerator`, `ThetaGammaIntegration`

### Remaining Ideas

3. **Predictive coding integration**: The neuromodulators could modulate prediction error magnitude at different levels of a hierarchical predictive model

4. **Neuromodulator learning**: The neuromodulator parameters themselves could be learned from experience (meta-learning)

5. **Stress response**: Add cortisol-like system that affects all neuromodulators during high-stakes situations

---

## Implementation Notes

### File Locations

- `/mnt/projects/t4d/t4dm/src/t4dm/learning/dopamine.py` (exists)
- `/mnt/projects/t4d/t4dm/src/t4dm/learning/norepinephrine.py` (to create)
- `/mnt/projects/t4d/t4dm/src/t4dm/learning/acetylcholine.py` (to create)
- `/mnt/projects/t4d/t4dm/src/t4dm/learning/serotonin.py` (to create)
- `/mnt/projects/t4d/t4dm/src/t4dm/learning/inhibition.py` (to create)
- `/mnt/projects/t4d/t4dm/src/t4dm/learning/neuromodulators.py` (to create)

### Dependencies

No new external dependencies required. All implementations use numpy, which is already a dependency.

### Configuration

Add to settings:

```python
# Neuromodulator settings
ne_baseline_arousal: float = 0.5
ne_novelty_decay: float = 0.95
ach_encoding_threshold: float = 0.7
ach_retrieval_threshold: float = 0.3
serotonin_discount_rate: float = 0.99
gaba_inhibition_strength: float = 0.5
gaba_sparsity_target: float = 0.2
```

---

## Conclusion

This design extends WW's dopamine-based learning with a complete neuromodulatory orchestra. The key architectural insight is that these systems must interact - they are not independent modules but a coordinated ensemble that shapes how the entire memory system operates.

The implementation prioritizes:
1. **Biological plausibility** where it aids function
2. **Computational efficiency** for real-time operation
3. **Modularity** for testing and extension
4. **Integration** with existing WW components

The brain doesn't have separate "attention," "learning," and "retrieval" modules - it has neuromodulatory systems that orchestrate these functions dynamically. This design brings that principle to artificial memory systems.
