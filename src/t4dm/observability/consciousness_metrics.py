"""
DEPRECATED: Use integration_metrics.py instead.

This module is maintained for backward compatibility only.
All functionality has been moved to integration_metrics.py.

The rename reflects that these are *integration metrics* measuring
computational coupling between subsystems, not consciousness claims.
"""

from t4dm.observability.integration_metrics import (
    ConsciousnessMetrics,
    IITMetricsComputer,
    IntegrationMetrics,
)

__all__ = [
    "ConsciousnessMetrics",  # Backward compat alias for IntegrationMetrics
    "IITMetricsComputer",
    "IntegrationMetrics",
]
