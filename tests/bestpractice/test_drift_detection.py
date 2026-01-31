"""Tests for learning drift detection."""
import numpy as np


class TestDriftDetection:
    def test_drift_detection_exists(self):
        """P2-16: Coupling tracks weight history."""
        from t4dm.nca.coupling import LearnableCoupling
        coupling = LearnableCoupling()
        assert hasattr(coupling, '_weight_history')
