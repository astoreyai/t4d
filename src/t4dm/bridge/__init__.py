"""
DEPRECATED: This module has been migrated to ww.bridges.nca_bridge.

This file re-exports from t4dm.bridges for backwards compatibility.
New code should import from t4dm.bridges directly.

Migration (Phase 10.1):
    Old: from t4dm.bridge import MemoryNCABridge
    New: from t4dm.bridges import NCABridge  # or MemoryNCABridge for compat
"""

import warnings

warnings.warn(
    "t4dm.bridge is deprecated, use t4dm.bridges instead",
    DeprecationWarning,
    stacklevel=2
)

# Re-export from new location for backwards compatibility
from t4dm.bridges.nca_bridge import (
    BridgeConfig,  # Alias
    EncodingContext,
    MemoryNCABridge,  # Alias
    NCABridge,
    NCABridgeConfig,
    RetrievalContext,
)

__all__ = [
    "NCABridge",
    "NCABridgeConfig",
    "MemoryNCABridge",
    "BridgeConfig",
    "EncodingContext",
    "RetrievalContext",
]
