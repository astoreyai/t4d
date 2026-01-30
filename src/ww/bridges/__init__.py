"""
World Weaver Bridges Module.

P5.x: Integration bridges connecting neural subsystems.
P6.x: Enhanced capability bridges.
P10.x: Consolidated bridge module.

Bridges:
- nca_bridge: Memory-NCA integration bridge (P10.1 - migrated from bridge/)
- dopamine_bridge: PredictiveCoding -> DopamineSystem RPE (P5.2)
- capsule_bridge: CapsuleLayer -> Memory retrieval scoring (P6.2)
- ff_encoding_bridge: ForwardForward -> Encoding pipeline novelty (P6.3)
- ff_retrieval_scorer: ForwardForward -> Retrieval confidence scoring (P6.4)
- ff_capsule_bridge: FF goodness + Capsule routing agreement (Phase 6A)
"""

from ww.bridges.capsule_bridge import (
    CapsuleBridgeConfig,
    CapsuleBridgeState,
    CapsuleRepresentation,
    CapsuleRetrievalBridge,
    create_capsule_bridge,
)
from ww.bridges.dopamine_bridge import (
    PredictiveCodingDopamineBridge,
    create_pc_dopamine_bridge,
)
from ww.bridges.ff_capsule_bridge import (
    CapsuleState,
    FFCapsuleBridge,
    FFCapsuleBridgeConfig,
    FFCapsuleBridgeState,
    create_ff_capsule_bridge,
)
from ww.bridges.ff_encoding_bridge import (
    EncodingGuidance,
    FFEncodingBridge,
    FFEncodingConfig,
    FFEncodingState,
    create_ff_encoding_bridge,
)
from ww.bridges.ff_retrieval_scorer import (
    FFRetrievalConfig,
    FFRetrievalScorer,
    FFRetrievalState,
    RetrievalScore,
)
from ww.bridges.nca_bridge import (
    BridgeConfig,
    EncodingContext,
    # Backwards compatibility
    MemoryNCABridge,
    NCABridge,
    NCABridgeConfig,
    RetrievalContext,
)

__all__ = [
    # P10.1: NCA Bridge (migrated from bridge/)
    "NCABridge",
    "NCABridgeConfig",
    "EncodingContext",
    "RetrievalContext",
    # Backwards compatibility aliases
    "MemoryNCABridge",
    "BridgeConfig",
    # P5.2: Dopamine Bridge
    "PredictiveCodingDopamineBridge",
    "create_pc_dopamine_bridge",
    # P6.2: Capsule Bridge
    "CapsuleBridgeConfig",
    "CapsuleBridgeState",
    "CapsuleRepresentation",
    "CapsuleRetrievalBridge",
    "create_capsule_bridge",
    # P6.3: FF-Encoding Bridge
    "EncodingGuidance",
    "FFEncodingBridge",
    "FFEncodingConfig",
    "FFEncodingState",
    "create_ff_encoding_bridge",
    # P6.4: FF-Retrieval Scorer
    "FFRetrievalConfig",
    "FFRetrievalScorer",
    "FFRetrievalState",
    "RetrievalScore",
    # Phase 6A: FF-Capsule Bridge
    "FFCapsuleBridge",
    "FFCapsuleBridgeConfig",
    "FFCapsuleBridgeState",
    "CapsuleState",
    "create_ff_capsule_bridge",
]
