"""
DEPRECATED: This module has been migrated to ww.bridges.nca_bridge.

This file re-exports from ww.bridges for backwards compatibility.
New code should import from ww.bridges directly.

Migration (Phase 10.1):
    Old: from ww.bridge import MemoryNCABridge
    New: from ww.bridges import NCABridge  # or MemoryNCABridge for compat
"""

import warnings

warnings.warn(
    "ww.bridge is deprecated, use ww.bridges instead",
    DeprecationWarning,
    stacklevel=2
)

# Re-export from new location for backwards compatibility
from ww.bridges.nca_bridge import (
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
