"""
P7.1: Bridge Container for World Weaver.

Centralized factory for all bridge instances. Provides dependency injection
for NCA subsystems into memory operations.

Integration Pattern:
    container = get_bridge_container(session_id)

    # Use in episodic memory
    guidance = container.ff_bridge.process(embedding)

    # Use in semantic memory
    boosts = container.capsule_bridge.compute_boosts(query, candidates)

    # Use in consolidation
    rpe = container.dopamine_bridge.compute_pe_signal()

References:
- P6.2: CapsuleRetrievalBridge for semantic retrieval boosting
- P6.3: FFEncodingBridge for novelty detection in encoding
- P5.2: PredictiveCodingDopamineBridge for consolidation RPE
- Bridge module: MemoryNCABridge for state-dependent memory ops
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from datetime import datetime
from typing import TYPE_CHECKING, Any

from ww.bridges import (
    CapsuleRetrievalBridge,
    FFEncodingBridge,
    FFRetrievalScorer,
    MemoryNCABridge,  # Backwards compat alias
    # P10.1: NCA bridge now imported from ww.bridges
    NCABridgeConfig,
    PredictiveCodingDopamineBridge,
    create_capsule_bridge,
    create_ff_encoding_bridge,
)

if TYPE_CHECKING:
    from ww.learning.dopamine import DopamineSystem
    from ww.nca.attractors import StateTransitionManager
    from ww.nca.capsules import CapsuleLayer
    from ww.nca.coupling import LearnableCoupling
    from ww.nca.energy import EnergyLandscape
    from ww.nca.forward_forward import ForwardForwardLayer
    from ww.nca.neural_field import NeuralFieldSolver
    from ww.prediction.predictive_coding import PredictiveCodingHierarchy

logger = logging.getLogger(__name__)


@dataclass
class BridgeContainerConfig:
    """Configuration for bridge container.

    Attributes:
        ff_enabled: Enable Forward-Forward encoding bridge
        capsule_enabled: Enable capsule retrieval bridge
        dopamine_enabled: Enable dopamine consolidation bridge
        nca_enabled: Enable NCA memory bridge
        lazy_init: Defer bridge initialization until first use
    """
    ff_enabled: bool = True
    capsule_enabled: bool = True
    dopamine_enabled: bool = True
    nca_enabled: bool = True
    lazy_init: bool = True


@dataclass
class BridgeContainerState:
    """Runtime state of bridge container.

    Attributes:
        initialized: Whether bridges have been initialized
        ff_calls: Number of FF encoding bridge calls
        ff_retrieval_calls: Number of FF retrieval scorer calls
        capsule_calls: Number of capsule bridge calls
        dopamine_calls: Number of dopamine bridge calls
        nca_calls: Number of NCA bridge calls
        last_access: Last access timestamp
    """
    initialized: bool = False
    ff_calls: int = 0
    ff_retrieval_calls: int = 0
    capsule_calls: int = 0
    dopamine_calls: int = 0
    nca_calls: int = 0
    last_access: datetime = field(default_factory=datetime.now)


class BridgeContainer:
    """
    Centralized container for all WW bridges.

    Implements P7.1: Bridge wiring for production code paths.

    Each bridge connects NCA subsystems to memory operations:
    - FFEncodingBridge: Novelty detection for episodic encoding
    - CapsuleRetrievalBridge: Part-whole scoring for semantic retrieval
    - PredictiveCodingDopamineBridge: RPE for consolidation priority
    - MemoryNCABridge: State-dependent memory operations

    Usage:
        ```python
        container = BridgeContainer()

        # Lazy initialization with NCA components
        container.set_ff_layer(ff_layer)
        container.set_capsule_layer(capsule_layer)

        # Use bridges
        guidance = container.get_ff_bridge().process(embedding)
        boosts = container.get_capsule_bridge().compute_boosts(query, candidates)
        ```
    """

    def __init__(
        self,
        config: BridgeContainerConfig | None = None,
        session_id: str = "default",
    ):
        """
        Initialize bridge container.

        Args:
            config: Container configuration
            session_id: Session ID for logging/tracking
        """
        self.config = config or BridgeContainerConfig()
        self.session_id = session_id
        self.state = BridgeContainerState()

        # Bridge instances (created on demand if lazy_init)
        self._ff_bridge: FFEncodingBridge | None = None
        self._ff_retrieval_scorer: FFRetrievalScorer | None = None
        self._capsule_bridge: CapsuleRetrievalBridge | None = None
        self._dopamine_bridge: PredictiveCodingDopamineBridge | None = None
        self._nca_bridge: MemoryNCABridge | None = None

        # NCA component references (set externally)
        self._ff_layer: ForwardForwardLayer | None = None
        self._capsule_layer: CapsuleLayer | None = None
        self._hierarchy: PredictiveCodingHierarchy | None = None
        self._dopamine_system: DopamineSystem | None = None
        self._neural_field: NeuralFieldSolver | None = None
        self._coupling: LearnableCoupling | None = None
        self._state_manager: StateTransitionManager | None = None
        self._energy_landscape: EnergyLandscape | None = None

        if not self.config.lazy_init:
            self._initialize_bridges()

        logger.info(
            f"P7.1: BridgeContainer created (session={session_id}, "
            f"lazy_init={self.config.lazy_init})"
        )

    def _initialize_bridges(self) -> None:
        """Initialize all enabled bridges."""
        if self.state.initialized:
            return

        if self.config.ff_enabled:
            self._ff_bridge = create_ff_encoding_bridge(
                ff_layer=self._ff_layer,
                goodness_threshold=2.0,
                novelty_boost=0.5,
                online_learning=True,
            )

        if self.config.capsule_enabled:
            self._capsule_bridge = create_capsule_bridge(
                capsule_layer=self._capsule_layer,
                activation_weight=0.4,
                pose_weight=0.6,
                max_boost=0.3,
            )

        if self.config.nca_enabled:
            self._nca_bridge = MemoryNCABridge(
                config=NCABridgeConfig(),
                neural_field=self._neural_field,
                coupling=self._coupling,
                state_manager=self._state_manager,
                energy_landscape=self._energy_landscape,
                dopamine=self._dopamine_system,
            )

        # Dopamine bridge requires hierarchy - only create if available
        if self.config.dopamine_enabled and self._hierarchy is not None:
            from ww.bridges import create_pc_dopamine_bridge
            self._dopamine_bridge = create_pc_dopamine_bridge(
                hierarchy=self._hierarchy,
                dopamine=self._dopamine_system,
                blend_ratio=0.5,
            )

        self.state.initialized = True
        logger.info("P7.1: All bridges initialized")

    # -------------------------------------------------------------------------
    # NCA Component Setters
    # -------------------------------------------------------------------------

    def set_ff_layer(self, ff_layer: ForwardForwardLayer) -> None:
        """Set Forward-Forward layer for encoding bridge."""
        self._ff_layer = ff_layer
        if self._ff_bridge is not None:
            self._ff_bridge.set_ff_layer(ff_layer)
        logger.debug("P7.1: FF layer set")

    def set_capsule_layer(self, capsule_layer: CapsuleLayer) -> None:
        """Set capsule layer for retrieval bridge."""
        self._capsule_layer = capsule_layer
        if self._capsule_bridge is not None:
            self._capsule_bridge.set_capsule_layer(capsule_layer)
        logger.debug("P7.1: Capsule layer set")

    def set_hierarchy(self, hierarchy: PredictiveCodingHierarchy) -> None:
        """Set predictive coding hierarchy for dopamine bridge."""
        self._hierarchy = hierarchy
        # May need to recreate dopamine bridge if hierarchy wasn't available at init
        if self._dopamine_bridge is None and self.config.dopamine_enabled:
            from ww.bridges import create_pc_dopamine_bridge
            self._dopamine_bridge = create_pc_dopamine_bridge(
                hierarchy=hierarchy,
                dopamine=self._dopamine_system,
                blend_ratio=0.5,
            )
        logger.debug("P7.1: Predictive hierarchy set")

    def set_dopamine_system(self, dopamine: DopamineSystem) -> None:
        """Set dopamine system for bridges."""
        self._dopamine_system = dopamine
        if self._nca_bridge is not None:
            self._nca_bridge.dopamine = dopamine
        logger.debug("P7.1: Dopamine system set")

    def set_neural_field(self, neural_field: NeuralFieldSolver) -> None:
        """Set neural field for NCA bridge."""
        self._neural_field = neural_field
        if self._nca_bridge is not None:
            self._nca_bridge.neural_field = neural_field
        logger.debug("P7.1: Neural field set")

    def set_coupling(self, coupling: LearnableCoupling) -> None:
        """Set coupling for NCA bridge."""
        self._coupling = coupling
        if self._nca_bridge is not None:
            self._nca_bridge.coupling = coupling
        logger.debug("P7.1: Coupling set")

    def set_state_manager(self, state_manager: StateTransitionManager) -> None:
        """Set state manager for NCA bridge."""
        self._state_manager = state_manager
        if self._nca_bridge is not None:
            self._nca_bridge.state_manager = state_manager
        logger.debug("P7.1: State manager set")

    def set_energy_landscape(self, energy_landscape: EnergyLandscape) -> None:
        """Set energy landscape for NCA bridge."""
        self._energy_landscape = energy_landscape
        if self._nca_bridge is not None:
            self._nca_bridge.energy_landscape = energy_landscape
        logger.debug("P7.1: Energy landscape set")

    # -------------------------------------------------------------------------
    # Bridge Accessors
    # -------------------------------------------------------------------------

    def get_ff_bridge(self) -> FFEncodingBridge | None:
        """Get FF encoding bridge, initializing if needed."""
        if not self.config.ff_enabled:
            return None

        if self._ff_bridge is None and self.config.lazy_init:
            self._ff_bridge = create_ff_encoding_bridge(
                ff_layer=self._ff_layer,
                goodness_threshold=2.0,
                novelty_boost=0.5,
                online_learning=True,
            )

        self.state.ff_calls += 1
        self.state.last_access = datetime.now()
        return self._ff_bridge

    def get_ff_retrieval_scorer(self) -> FFRetrievalScorer | None:
        """Get FF retrieval scorer, initializing if needed.

        P6.4: Uses FF goodness to score retrieval candidates.
        High goodness = confident pattern match.
        """
        if not self.config.ff_enabled:
            return None

        if self._ff_retrieval_scorer is None and self.config.lazy_init:
            from ww.bridges import FFRetrievalConfig
            self._ff_retrieval_scorer = FFRetrievalScorer(
                ff_layer=self._ff_layer,
                config=FFRetrievalConfig(
                    max_boost=0.3,
                    learn_from_outcomes=True,
                ),
            )

        self.state.ff_retrieval_calls += 1
        self.state.last_access = datetime.now()
        return self._ff_retrieval_scorer

    def get_capsule_bridge(self) -> CapsuleRetrievalBridge | None:
        """Get capsule retrieval bridge, initializing if needed."""
        if not self.config.capsule_enabled:
            return None

        if self._capsule_bridge is None and self.config.lazy_init:
            self._capsule_bridge = create_capsule_bridge(
                capsule_layer=self._capsule_layer,
                activation_weight=0.4,
                pose_weight=0.6,
                max_boost=0.3,
            )

        self.state.capsule_calls += 1
        self.state.last_access = datetime.now()
        return self._capsule_bridge

    def get_dopamine_bridge(self) -> PredictiveCodingDopamineBridge | None:
        """Get dopamine consolidation bridge, initializing if needed."""
        if not self.config.dopamine_enabled:
            return None

        if self._dopamine_bridge is None and self._hierarchy is not None:
            from ww.bridges import create_pc_dopamine_bridge
            self._dopamine_bridge = create_pc_dopamine_bridge(
                hierarchy=self._hierarchy,
                dopamine=self._dopamine_system,
                blend_ratio=0.5,
            )

        if self._dopamine_bridge is not None:
            self.state.dopamine_calls += 1
            self.state.last_access = datetime.now()

        return self._dopamine_bridge

    def get_nca_bridge(self) -> MemoryNCABridge | None:
        """Get NCA memory bridge, initializing if needed."""
        if not self.config.nca_enabled:
            return None

        if self._nca_bridge is None and self.config.lazy_init:
            self._nca_bridge = MemoryNCABridge(
                config=NCABridgeConfig(),
                neural_field=self._neural_field,
                coupling=self._coupling,
                state_manager=self._state_manager,
                energy_landscape=self._energy_landscape,
                dopamine=self._dopamine_system,
            )

        self.state.nca_calls += 1
        self.state.last_access = datetime.now()
        return self._nca_bridge

    # -------------------------------------------------------------------------
    # Convenience Properties
    # -------------------------------------------------------------------------

    @property
    def ff_bridge(self) -> FFEncodingBridge | None:
        """Direct access to FF bridge."""
        return self.get_ff_bridge()

    @property
    def ff_retrieval_scorer(self) -> FFRetrievalScorer | None:
        """Direct access to FF retrieval scorer (P6.4)."""
        return self.get_ff_retrieval_scorer()

    @property
    def capsule_bridge(self) -> CapsuleRetrievalBridge | None:
        """Direct access to capsule bridge."""
        return self.get_capsule_bridge()

    @property
    def dopamine_bridge(self) -> PredictiveCodingDopamineBridge | None:
        """Direct access to dopamine bridge."""
        return self.get_dopamine_bridge()

    @property
    def nca_bridge(self) -> MemoryNCABridge | None:
        """Direct access to NCA bridge."""
        return self.get_nca_bridge()

    # -------------------------------------------------------------------------
    # Statistics
    # -------------------------------------------------------------------------

    def get_statistics(self) -> dict[str, Any]:
        """Get container statistics."""
        stats = {
            "session_id": self.session_id,
            "initialized": self.state.initialized,
            "config": {
                "ff_enabled": self.config.ff_enabled,
                "capsule_enabled": self.config.capsule_enabled,
                "dopamine_enabled": self.config.dopamine_enabled,
                "nca_enabled": self.config.nca_enabled,
            },
            "calls": {
                "ff": self.state.ff_calls,
                "ff_retrieval": self.state.ff_retrieval_calls,
                "capsule": self.state.capsule_calls,
                "dopamine": self.state.dopamine_calls,
                "nca": self.state.nca_calls,
            },
            "last_access": self.state.last_access.isoformat(),
        }

        # Add per-bridge stats if available
        if self._ff_bridge is not None:
            stats["ff_stats"] = self._ff_bridge.get_statistics()
        if self._ff_retrieval_scorer is not None:
            stats["ff_retrieval_stats"] = self._ff_retrieval_scorer.get_stats()
        if self._capsule_bridge is not None:
            stats["capsule_stats"] = self._capsule_bridge.get_statistics()
        if self._dopamine_bridge is not None:
            stats["dopamine_stats"] = self._dopamine_bridge.get_statistics()
        if self._nca_bridge is not None:
            stats["nca_stats"] = self._nca_bridge.get_stats()

        return stats


# -----------------------------------------------------------------------------
# Singleton Pattern for Session-Scoped Containers
# -----------------------------------------------------------------------------

_containers: dict[str, BridgeContainer] = {}


def get_bridge_container(
    session_id: str = "default",
    config: BridgeContainerConfig | None = None,
) -> BridgeContainer:
    """
    Get or create bridge container for session.

    Uses singleton pattern per session_id to ensure consistent
    bridge state across memory operations.

    Args:
        session_id: Session identifier
        config: Optional configuration (only used on first creation)

    Returns:
        BridgeContainer for the session
    """
    if session_id not in _containers:
        _containers[session_id] = BridgeContainer(
            config=config,
            session_id=session_id,
        )
    return _containers[session_id]


def clear_bridge_containers() -> None:
    """Clear all bridge containers (for testing)."""
    _containers.clear()
    logger.info("P7.1: All bridge containers cleared")


__all__ = [
    "BridgeContainer",
    "BridgeContainerConfig",
    "BridgeContainerState",
    "get_bridge_container",
    "clear_bridge_containers",
]
