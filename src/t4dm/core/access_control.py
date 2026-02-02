"""
Internal caller authentication for T4DM.

Distinguishes trusted internal callers from untrusted external ones
using a simple capability-based token system. Not full RBAC -- just
enough to prevent cross-module privilege escalation.

Usage:
    from t4dm.core.access_control import require_capability, HIPPOCAMPUS_TOKEN, API_TOKEN

    def receive_ach(self, level: float, token: CallerToken):
        require_capability(token, "set_neuromod")
        ...
"""

from __future__ import annotations

import logging
from dataclasses import dataclass

logger = logging.getLogger(__name__)


class AccessDenied(PermissionError):
    """Raised when a caller lacks the required capability."""

    def __init__(self, module: str, capability: str, trust_level: str):
        self.module = module
        self.capability = capability
        self.trust_level = trust_level
        super().__init__(
            f"Access denied: module='{module}' (trust={trust_level}) "
            f"lacks capability '{capability}'"
        )


@dataclass(frozen=True)
class CallerToken:
    """Immutable token identifying a caller and its capabilities."""

    module: str              # e.g., "hippocampus", "consolidation_service"
    trust_level: str         # "internal" | "external"
    capabilities: frozenset  # e.g., frozenset({"write_ca3", "set_neuromod"})

    def has_capability(self, capability: str) -> bool:
        """Check if this token has a specific capability."""
        return capability in self.capabilities


def require_capability(token: CallerToken | None, capability: str) -> None:
    """Require a caller token to have a specific capability.

    Args:
        token: Caller token (None = anonymous/external)
        capability: Required capability string

    Raises:
        AccessDenied: If token is None or lacks the capability
    """
    if token is None:
        raise AccessDenied("anonymous", capability, "none")

    if not isinstance(token, CallerToken):
        raise AccessDenied("unknown", capability, "invalid_token")

    if not token.has_capability(capability):
        raise AccessDenied(token.module, capability, token.trust_level)


# =============================================================================
# Pre-built tokens for internal modules
# =============================================================================

HIPPOCAMPUS_TOKEN = CallerToken(
    module="hippocampus",
    trust_level="internal",
    capabilities=frozenset({
        "write_ca3", "set_neuromod", "set_oscillator", "record_spike",
        "submit_salience", "read", "store_episodic",
    }),
)

CONSOLIDATION_TOKEN = CallerToken(
    module="consolidation",
    trust_level="internal",
    capabilities=frozenset({
        "trigger_swr", "trigger_replay", "write_ca3", "set_neuromod",
        "trigger_consolidation", "set_sleep_state", "set_adenosine",
        "read", "store_episodic",
    }),
)

VTA_TOKEN = CallerToken(
    module="vta",
    trust_level="internal",
    capabilities=frozenset({
        "submit_reward", "set_neuromod", "record_spike", "read",
    }),
)

LEARNING_TOKEN = CallerToken(
    module="learning",
    trust_level="internal",
    capabilities=frozenset({
        "record_spike", "submit_reward", "set_neuromod", "read",
        "write_ca3",
    }),
)

ADENOSINE_TOKEN = CallerToken(
    module="adenosine",
    trust_level="internal",
    capabilities=frozenset({
        "set_sleep_state", "set_adenosine", "read",
    }),
)

SLEEP_TOKEN = CallerToken(
    module="sleep",
    trust_level="internal",
    capabilities=frozenset({
        "trigger_swr", "trigger_replay", "write_ca3", "set_neuromod",
        "set_sleep_state", "set_adenosine", "read", "trigger_consolidation",
    }),
)

API_TOKEN = CallerToken(
    module="api",
    trust_level="external",
    capabilities=frozenset({
        "read", "store_episodic", "submit_feedback",
    }),
)

DEBUG_TOKEN = CallerToken(
    module="debug",
    trust_level="internal",
    capabilities=frozenset({
        "write_ca3", "set_neuromod", "set_oscillator", "record_spike",
        "submit_salience", "submit_reward", "trigger_swr", "trigger_replay",
        "trigger_consolidation", "set_sleep_state", "set_adenosine",
        "read", "store_episodic", "submit_feedback", "debug",
    }),
)
