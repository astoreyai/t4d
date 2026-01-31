"""
Comprehensive tests for World Weaver Bio-inspired API endpoints.

Tests the neural architecture correctness, biological plausibility, and
API contract compliance for:
- Neuromodulator live tuning (PUT /bio/neuromodulators)
- Neuromodulator reset (POST /bio/neuromodulators/reset)
- Homeostatic system (GET/PUT /bio/homeostatic, POST /bio/homeostatic/force-scaling)
- ACh mode switching (POST /bio/acetylcholine/switch-mode)

From a neural architecture perspective, these tests verify:
1. Parameter ranges respect biological plausibility
2. Changes propagate correctly to learning systems
3. Edge cases (min/max values, invalid inputs) are handled
4. The learning dynamics are properly modulated
"""

import pytest
from datetime import datetime
from unittest.mock import MagicMock, AsyncMock, patch
from fastapi import FastAPI
from fastapi.testclient import TestClient

from t4dm.api.routes.visualization import (
    router,
    NeuromodulatorTuning,  # Alias for NeuromodulatorTuningRequest
    NeuromodulatorTuningRequest,
    NeuromodulatorTuningResponse,
    HomeostaticConfigUpdate,
    HomeostaticStateResponse,
    AchModeSwitchRequest,
    AchModeSwitchResponse,
)


class TestNeuromodulatorTuningModel:
    """Tests for NeuromodulatorTuning Pydantic model.

    Validates that the parameter ranges are biologically plausible.
    """

    def test_valid_norepinephrine_params(self):
        """NE parameters respect LC-NE firing rate dynamics."""
        tuning = NeuromodulatorTuning(
            ne_baseline_arousal=0.5,  # Mid-range tonic activity
            ne_min_gain=0.5,  # Allow exploration
            ne_max_gain=2.5,  # Upper bound on alertness
            ne_novelty_decay=0.95,  # Slow decay for sustained attention
            ne_phasic_decay=0.8,  # Fast decay for burst responses
        )
        assert tuning.ne_baseline_arousal == 0.5
        assert tuning.ne_novelty_decay > tuning.ne_phasic_decay  # Biologically expected

    def test_valid_acetylcholine_params(self):
        """ACh parameters respect encoding/retrieval tradeoff."""
        tuning = NeuromodulatorTuning(
            ach_baseline=0.5,
            ach_adaptation_rate=0.1,
            ach_encoding_threshold=0.7,  # High threshold for encoding
            ach_retrieval_threshold=0.3,  # Low threshold for retrieval
        )
        # Biologically, encoding threshold > retrieval threshold
        assert tuning.ach_encoding_threshold > tuning.ach_retrieval_threshold

    def test_valid_dopamine_params(self):
        """DA parameters respect reward prediction error dynamics."""
        tuning = NeuromodulatorTuning(
            da_value_learning_rate=0.1,  # Moderate TD learning
            da_default_expected=0.5,  # Neutral expectation
            da_surprise_threshold=0.05,  # Sensitive to surprises
        )
        assert tuning.da_value_learning_rate == 0.1

    def test_valid_serotonin_params(self):
        """5-HT parameters respect temporal discounting dynamics."""
        tuning = NeuromodulatorTuning(
            serotonin_baseline_mood=0.5,
            serotonin_mood_adaptation_rate=0.1,
            serotonin_discount_rate=0.95,  # Patient, long-horizon
            serotonin_eligibility_decay=0.9,  # Sustained eligibility traces
        )
        assert tuning.serotonin_discount_rate == 0.95

    def test_valid_inhibition_params(self):
        """GABA parameters respect sparse coding principles."""
        tuning = NeuromodulatorTuning(
            inhibition_strength=0.5,
            sparsity_target=0.1,  # DG-like sparsity
            inhibition_temperature=1.0,  # Standard softmax
        )
        assert tuning.sparsity_target == 0.1  # ~10% activation

    def test_boundary_ne_arousal_min(self):
        """NE baseline can be 0 (low arousal state)."""
        tuning = NeuromodulatorTuning(ne_baseline_arousal=0.0)
        assert tuning.ne_baseline_arousal == 0.0

    def test_boundary_ne_arousal_max(self):
        """NE baseline can be 1 (high arousal state)."""
        tuning = NeuromodulatorTuning(ne_baseline_arousal=1.0)
        assert tuning.ne_baseline_arousal == 1.0

    def test_boundary_ne_gain_range(self):
        """NE gain follows Yerkes-Dodson range."""
        tuning = NeuromodulatorTuning(
            ne_min_gain=0.1,  # Minimum for drowsy state
            ne_max_gain=5.0,  # Maximum for emergency response
        )
        assert tuning.ne_min_gain == 0.1
        assert tuning.ne_max_gain == 5.0

    def test_boundary_ach_baseline(self):
        """ACh baseline respects cholinergic bounds."""
        # Min
        tuning_min = NeuromodulatorTuning(ach_baseline=0.1)
        assert tuning_min.ach_baseline == 0.1
        # Max
        tuning_max = NeuromodulatorTuning(ach_baseline=0.9)
        assert tuning_max.ach_baseline == 0.9

    def test_boundary_serotonin_discount(self):
        """5-HT discount respects temporal patience bounds."""
        # Near-sighted (low 5-HT)
        tuning_low = NeuromodulatorTuning(serotonin_discount_rate=0.9)
        # Far-sighted (high 5-HT)
        tuning_high = NeuromodulatorTuning(serotonin_discount_rate=1.0)
        assert tuning_low.serotonin_discount_rate < tuning_high.serotonin_discount_rate

    def test_boundary_sparsity_target(self):
        """GABA sparsity matches biological DG range (1-10%)."""
        tuning = NeuromodulatorTuning(sparsity_target=0.05)  # 5% like DG
        assert tuning.sparsity_target == 0.05

    def test_invalid_ne_arousal_below_min(self):
        """Reject NE arousal below 0."""
        with pytest.raises(ValueError):
            NeuromodulatorTuning(ne_baseline_arousal=-0.1)

    def test_invalid_ne_arousal_above_max(self):
        """Reject NE arousal above 1."""
        with pytest.raises(ValueError):
            NeuromodulatorTuning(ne_baseline_arousal=1.5)

    def test_invalid_ne_min_gain_too_low(self):
        """Reject NE min gain below biological minimum."""
        with pytest.raises(ValueError):
            NeuromodulatorTuning(ne_min_gain=0.05)  # Below 0.1

    def test_invalid_ne_max_gain_too_high(self):
        """Reject NE max gain above biological maximum."""
        with pytest.raises(ValueError):
            NeuromodulatorTuning(ne_max_gain=6.0)  # Above 5.0

    def test_invalid_ach_baseline_below_min(self):
        """Reject ACh baseline below minimum (complete cholinergic shutdown)."""
        with pytest.raises(ValueError):
            NeuromodulatorTuning(ach_baseline=0.05)  # Below 0.1

    def test_invalid_ach_baseline_above_max(self):
        """Reject ACh baseline above maximum (cholinergic overload)."""
        with pytest.raises(ValueError):
            NeuromodulatorTuning(ach_baseline=0.95)  # Above 0.9

    def test_invalid_serotonin_mood_below_min(self):
        """Reject 5-HT mood below 0 (undefined state)."""
        with pytest.raises(ValueError):
            NeuromodulatorTuning(serotonin_baseline_mood=-0.1)

    def test_invalid_serotonin_discount_below_min(self):
        """Reject discount rate below minimum (pathological impatience)."""
        with pytest.raises(ValueError):
            NeuromodulatorTuning(serotonin_discount_rate=0.85)  # Below 0.9

    def test_invalid_sparsity_too_low(self):
        """Reject sparsity below biological minimum."""
        with pytest.raises(ValueError):
            NeuromodulatorTuning(sparsity_target=0.01)  # Below 0.05

    def test_invalid_sparsity_too_high(self):
        """Reject sparsity above biological maximum (defeats sparsity purpose)."""
        with pytest.raises(ValueError):
            NeuromodulatorTuning(sparsity_target=0.6)  # Above 0.5

    def test_empty_update_is_valid(self):
        """Empty update (all None) is valid for partial updates."""
        tuning = NeuromodulatorTuning()
        assert tuning.ne_baseline_arousal is None
        assert tuning.ach_baseline is None
        assert tuning.serotonin_baseline_mood is None


class TestHomeostaticConfigUpdateModel:
    """Tests for HomeostaticConfigUpdate Pydantic model.

    Validates BCM/homeostatic plasticity parameter ranges.
    """

    def test_valid_target_norm(self):
        """Target norm within embedding space bounds."""
        update = HomeostaticConfigUpdate(target_norm=1.0)
        assert update.target_norm == 1.0

    def test_valid_norm_tolerance(self):
        """Norm tolerance for triggering scaling."""
        update = HomeostaticConfigUpdate(norm_tolerance=0.2)
        assert update.norm_tolerance == 0.2

    def test_valid_ema_alpha(self):
        """EMA rate for statistics tracking."""
        update = HomeostaticConfigUpdate(ema_alpha=0.01)
        assert update.ema_alpha == 0.01

    def test_valid_decorrelation_strength(self):
        """Decorrelation for reducing embedding interference."""
        update = HomeostaticConfigUpdate(decorrelation_strength=0.01)
        assert update.decorrelation_strength == 0.01

    def test_valid_sliding_threshold_rate(self):
        """BCM sliding threshold adaptation rate."""
        update = HomeostaticConfigUpdate(sliding_threshold_rate=0.001)
        assert update.sliding_threshold_rate == 0.001

    def test_boundary_target_norm_min(self):
        """Minimum target norm (compressed representations)."""
        update = HomeostaticConfigUpdate(target_norm=0.5)
        assert update.target_norm == 0.5

    def test_boundary_target_norm_max(self):
        """Maximum target norm (expanded representations)."""
        update = HomeostaticConfigUpdate(target_norm=2.0)
        assert update.target_norm == 2.0

    def test_boundary_ema_alpha_min(self):
        """Slow EMA (long memory of statistics)."""
        update = HomeostaticConfigUpdate(ema_alpha=0.001)
        assert update.ema_alpha == 0.001

    def test_boundary_ema_alpha_max(self):
        """Fast EMA (rapid adaptation)."""
        update = HomeostaticConfigUpdate(ema_alpha=0.1)
        assert update.ema_alpha == 0.1

    def test_invalid_target_norm_below_min(self):
        """Reject target norm below minimum."""
        with pytest.raises(ValueError):
            HomeostaticConfigUpdate(target_norm=0.3)  # Below 0.5

    def test_invalid_target_norm_above_max(self):
        """Reject target norm above maximum."""
        with pytest.raises(ValueError):
            HomeostaticConfigUpdate(target_norm=2.5)  # Above 2.0

    def test_invalid_norm_tolerance_below_min(self):
        """Reject norm tolerance too tight."""
        with pytest.raises(ValueError):
            HomeostaticConfigUpdate(norm_tolerance=0.01)  # Below 0.05

    def test_invalid_ema_alpha_below_min(self):
        """Reject EMA rate too slow (numerical instability)."""
        with pytest.raises(ValueError):
            HomeostaticConfigUpdate(ema_alpha=0.0001)  # Below 0.001

    def test_invalid_ema_alpha_above_max(self):
        """Reject EMA rate too fast (no smoothing)."""
        with pytest.raises(ValueError):
            HomeostaticConfigUpdate(ema_alpha=0.5)  # Above 0.1

    def test_invalid_decorrelation_above_max(self):
        """Reject decorrelation too strong (destroys information)."""
        with pytest.raises(ValueError):
            HomeostaticConfigUpdate(decorrelation_strength=0.2)  # Above 0.1

    def test_empty_update_is_valid(self):
        """Empty update is valid."""
        update = HomeostaticConfigUpdate()
        assert update.target_norm is None


class TestAchModeSwitchRequest:
    """Tests for ACh mode switching request model."""

    def test_valid_encoding_mode(self):
        """Encoding mode is valid."""
        request = AchModeSwitchRequest(mode="encoding")
        assert request.mode == "encoding"

    def test_valid_balanced_mode(self):
        """Balanced mode is valid."""
        request = AchModeSwitchRequest(mode="balanced")
        assert request.mode == "balanced"

    def test_valid_retrieval_mode(self):
        """Retrieval mode is valid."""
        request = AchModeSwitchRequest(mode="retrieval")
        assert request.mode == "retrieval"


class TestBioEndpointsIntegration:
    """Integration tests for bio-inspired API endpoints.

    These tests verify the API contracts and propagation of changes
    to the underlying learning systems.
    """

    ADMIN_KEY = "test-admin-key-12345"

    @pytest.fixture
    def mock_episodic(self):
        """Create mock episodic memory with full orchestra."""
        episodic = MagicMock()

        # Mock neuromodulator orchestra
        orchestra = MagicMock()

        # Norepinephrine mock
        ne = MagicMock()
        ne.min_gain = 0.5
        ne.max_gain = 2.0
        orchestra.norepinephrine = ne

        # Acetylcholine mock
        ach = MagicMock()
        ach.encoding_threshold = 0.7
        ach.retrieval_threshold = 0.3
        ach.get_current_mode.return_value = MagicMock(value="balanced")
        ach.force_mode.return_value = MagicMock(
            mode=MagicMock(value="encoding"),
            ach_level=0.8,
            encoding_weight=0.9,
            retrieval_weight=0.1,
        )
        orchestra.acetylcholine = ach

        # Dopamine mock
        da = MagicMock()
        orchestra.dopamine = da

        # Serotonin mock
        serotonin = MagicMock()
        orchestra.serotonin = serotonin

        # Inhibitory mock
        inhib = MagicMock()
        orchestra.inhibitory = inhib

        episodic.orchestra = orchestra

        # Mock neuromodulator state
        episodic.get_current_neuromodulator_state.return_value = {
            "dopamine_rpe": 0.1,
            "norepinephrine_gain": 1.2,
            "acetylcholine_mode": "balanced",
            "serotonin_mood": 0.6,
            "inhibition_sparsity": 0.8,
        }

        # Mock homeostatic plasticity
        homeostatic = MagicMock()
        homeostatic.get_state.return_value = MagicMock(
            mean_norm=1.0,
            std_norm=0.1,
            mean_activation=0.5,
            sliding_threshold=0.5,
            last_update=datetime.now(),
        )
        homeostatic.get_stats.return_value = {
            "scaling_count": 5,
            "decorrelation_count": 3,
            "config": {
                "target_norm": 1.0,
                "norm_tolerance": 0.2,
                "ema_alpha": 0.01,
            },
        }
        homeostatic.needs_scaling.return_value = False
        homeostatic.compute_scaling_factor.return_value = 1.0
        homeostatic.force_scaling.return_value = 0.98
        episodic.homeostatic = homeostatic

        return episodic

    @pytest.fixture
    def mock_services(self, mock_episodic):
        """Create mock services dict."""
        return {
            "session_id": "test-session",
            "episodic": mock_episodic,
            "semantic": MagicMock(),
            "procedural": MagicMock(),
        }

    @pytest.fixture
    def client(self, mock_services):
        """Create test client with mocked dependencies."""
        app = FastAPI()
        app.include_router(router)

        # Override dependency
        from t4dm.api.deps import get_memory_services
        app.dependency_overrides[get_memory_services] = lambda: mock_services

        return TestClient(app)

    @pytest.fixture(autouse=True)
    def setup_admin_key(self):
        """Set admin key for auth tests."""
        from t4dm.core.config import get_settings
        settings = get_settings()
        original_key = getattr(settings, 'admin_api_key', None)
        settings.admin_api_key = self.ADMIN_KEY
        yield
        settings.admin_api_key = original_key

    @property
    def admin_headers(self):
        """Headers with admin authentication."""
        return {"X-Admin-Key": self.ADMIN_KEY}

    # =========================================================================
    # Homeostatic Endpoints Tests
    # =========================================================================

    def test_get_homeostatic_state(self, client):
        """GET /bio/homeostatic returns current state."""
        response = client.get("/bio/homeostatic")
        assert response.status_code == 200

        data = response.json()
        assert "mean_norm" in data
        assert "std_norm" in data
        assert "sliding_threshold" in data
        assert "needs_scaling" in data
        assert "current_scaling_factor" in data
        assert "config" in data

    def test_get_homeostatic_includes_bcm_threshold(self, client):
        """Homeostatic state includes BCM sliding threshold."""
        response = client.get("/bio/homeostatic")
        data = response.json()

        # BCM threshold should be present
        assert "sliding_threshold" in data
        assert isinstance(data["sliding_threshold"], float)

    def test_put_homeostatic_config(self, client, mock_services):
        """PUT /bio/homeostatic updates configuration."""
        update = {
            "target_norm": 1.2,
            "norm_tolerance": 0.15,
            "ema_alpha": 0.02,
        }

        response = client.put(
            "/bio/homeostatic",
            json=update,
            headers=self.admin_headers,
        )
        assert response.status_code == 200

        # Verify setters were called
        homeostatic = mock_services["episodic"].homeostatic
        homeostatic.set_target_norm.assert_called_once_with(1.2)
        homeostatic.set_norm_tolerance.assert_called_once_with(0.15)
        homeostatic.set_ema_alpha.assert_called_once_with(0.02)

    def test_put_homeostatic_partial_update(self, client, mock_services):
        """Partial updates only change specified fields."""
        update = {"target_norm": 1.5}  # Only update norm

        response = client.put(
            "/bio/homeostatic",
            json=update,
            headers=self.admin_headers,
        )
        assert response.status_code == 200

        homeostatic = mock_services["episodic"].homeostatic
        homeostatic.set_target_norm.assert_called_once_with(1.5)
        homeostatic.set_norm_tolerance.assert_not_called()
        homeostatic.set_ema_alpha.assert_not_called()

    def test_put_homeostatic_requires_auth(self, client):
        """PUT /bio/homeostatic requires admin auth."""
        response = client.put("/bio/homeostatic", json={"target_norm": 1.0})
        assert response.status_code == 401

    def test_put_homeostatic_rejects_invalid_key(self, client):
        """PUT /bio/homeostatic rejects invalid admin key."""
        response = client.put(
            "/bio/homeostatic",
            json={"target_norm": 1.0},
            headers={"X-Admin-Key": "wrong-key"},
        )
        assert response.status_code == 403

    def test_put_homeostatic_validates_target_norm(self, client):
        """Reject invalid target_norm values."""
        response = client.put(
            "/bio/homeostatic",
            json={"target_norm": 3.0},  # Above max 2.0
            headers=self.admin_headers,
        )
        assert response.status_code == 422  # Validation error

    def test_force_homeostatic_scaling(self, client, mock_services):
        """POST /bio/homeostatic/force-scaling triggers immediate scaling."""
        response = client.post(
            "/bio/homeostatic/force-scaling",
            headers=self.admin_headers,
        )
        assert response.status_code == 200

        data = response.json()
        assert data["success"] is True
        assert "scaling_factor" in data
        assert data["scaling_factor"] == 0.98  # From mock

        # Verify force_scaling was called
        mock_services["episodic"].homeostatic.force_scaling.assert_called_once()

    def test_force_scaling_requires_auth(self, client):
        """POST /bio/homeostatic/force-scaling requires admin auth."""
        response = client.post("/bio/homeostatic/force-scaling")
        assert response.status_code == 401

    # =========================================================================
    # Neuromodulator Tuning Endpoints Tests
    # =========================================================================

    def test_put_neuromodulators_ne(self, client, mock_services):
        """PUT /bio/neuromodulators updates norepinephrine."""
        tuning = {
            "ne_baseline_arousal": 0.6,
            "ne_min_gain": 0.3,
            "ne_max_gain": 3.0,
        }

        response = client.put(
            "/bio/neuromodulators",
            json=tuning,
            headers=self.admin_headers,
        )
        assert response.status_code == 200

        data = response.json()
        assert data["success"] is True
        assert "norepinephrine" in data["updated_systems"]

        # Verify NE setters were called
        ne = mock_services["episodic"].orchestra.norepinephrine
        ne.set_baseline_arousal.assert_called_once_with(0.6)
        ne.set_arousal_bounds.assert_called_once_with(0.3, 3.0)

    def test_put_neuromodulators_ach(self, client, mock_services):
        """PUT /bio/neuromodulators updates acetylcholine."""
        tuning = {
            "ach_baseline": 0.6,
            "ach_adaptation_rate": 0.2,
        }

        response = client.put(
            "/bio/neuromodulators",
            json=tuning,
            headers=self.admin_headers,
        )
        assert response.status_code == 200

        data = response.json()
        assert "acetylcholine" in data["updated_systems"]

        ach = mock_services["episodic"].orchestra.acetylcholine
        ach.set_baseline_ach.assert_called_once_with(0.6)
        ach.set_adaptation_rate.assert_called_once_with(0.2)

    def test_put_neuromodulators_dopamine(self, client, mock_services):
        """PUT /bio/neuromodulators updates dopamine."""
        tuning = {
            "da_value_learning_rate": 0.15,
            "da_default_expected": 0.4,
            "da_surprise_threshold": 0.08,
        }

        response = client.put(
            "/bio/neuromodulators",
            json=tuning,
            headers=self.admin_headers,
        )
        assert response.status_code == 200

        data = response.json()
        assert "dopamine" in data["updated_systems"]

        da = mock_services["episodic"].orchestra.dopamine
        da.set_value_learning_rate.assert_called_once_with(0.15)
        da.set_default_expected.assert_called_once_with(0.4)
        da.set_surprise_threshold.assert_called_once_with(0.08)

    def test_put_neuromodulators_serotonin(self, client, mock_services):
        """PUT /bio/neuromodulators updates serotonin."""
        tuning = {
            "serotonin_baseline_mood": 0.7,
            "serotonin_discount_rate": 0.98,
            "serotonin_eligibility_decay": 0.92,
        }

        response = client.put(
            "/bio/neuromodulators",
            json=tuning,
            headers=self.admin_headers,
        )
        assert response.status_code == 200

        data = response.json()
        assert "serotonin" in data["updated_systems"]

        serotonin = mock_services["episodic"].orchestra.serotonin
        serotonin.set_baseline_mood.assert_called_once_with(0.7)
        serotonin.set_discount_rate.assert_called_once_with(0.98)
        serotonin.set_eligibility_decay.assert_called_once_with(0.92)

    def test_put_neuromodulators_inhibition(self, client, mock_services):
        """PUT /bio/neuromodulators updates GABA inhibition."""
        tuning = {
            "inhibition_strength": 0.6,
            "sparsity_target": 0.08,
            "inhibition_temperature": 2.0,
        }

        response = client.put(
            "/bio/neuromodulators",
            json=tuning,
            headers=self.admin_headers,
        )
        assert response.status_code == 200

        data = response.json()
        assert "inhibition" in data["updated_systems"]

        inhib = mock_services["episodic"].orchestra.inhibitory
        inhib.set_inhibition_strength.assert_called_once_with(0.6)
        inhib.set_sparsity_target.assert_called_once_with(0.08)
        inhib.set_temperature.assert_called_once_with(2.0)

    def test_put_neuromodulators_multiple_systems(self, client, mock_services):
        """Update multiple neuromodulator systems at once."""
        tuning = {
            "ne_baseline_arousal": 0.7,
            "ach_baseline": 0.6,
            "serotonin_baseline_mood": 0.5,
        }

        response = client.put(
            "/bio/neuromodulators",
            json=tuning,
            headers=self.admin_headers,
        )
        assert response.status_code == 200

        data = response.json()
        # Should have updated all three systems
        assert "norepinephrine" in data["updated_systems"]
        assert "acetylcholine" in data["updated_systems"]
        assert "serotonin" in data["updated_systems"]

    def test_put_neuromodulators_returns_current_state(self, client):
        """Response includes current neuromodulator state."""
        tuning = {"ne_baseline_arousal": 0.5}

        response = client.put(
            "/bio/neuromodulators",
            json=tuning,
            headers=self.admin_headers,
        )
        assert response.status_code == 200

        data = response.json()
        assert "current_state" in data
        # State comes from mock
        assert "dopamine_rpe" in data["current_state"]

    def test_put_neuromodulators_empty_update(self, client):
        """Empty update is valid but updates nothing."""
        response = client.put(
            "/bio/neuromodulators",
            json={},
            headers=self.admin_headers,
        )
        assert response.status_code == 200

        data = response.json()
        assert data["updated_systems"] == []

    def test_put_neuromodulators_requires_auth(self, client):
        """PUT /bio/neuromodulators requires admin auth."""
        response = client.put("/bio/neuromodulators", json={"ne_baseline_arousal": 0.5})
        assert response.status_code == 401

    def test_put_neuromodulators_validates_params(self, client):
        """Reject invalid neuromodulator parameters."""
        # Invalid NE arousal
        response = client.put(
            "/bio/neuromodulators",
            json={"ne_baseline_arousal": 1.5},  # Above max
            headers=self.admin_headers,
        )
        assert response.status_code == 422

    def test_reset_neuromodulators(self, client, mock_services):
        """POST /bio/neuromodulators/reset resets all systems."""
        response = client.post(
            "/bio/neuromodulators/reset",
            headers=self.admin_headers,
        )
        assert response.status_code == 200

        data = response.json()
        assert data["success"] is True

        # Verify all reset methods were called
        orchestra = mock_services["episodic"].orchestra
        orchestra.norepinephrine.reset_history.assert_called_once()
        orchestra.acetylcholine.reset.assert_called_once()
        orchestra.dopamine.reset_history.assert_called_once()
        orchestra.serotonin.reset.assert_called_once()
        orchestra.inhibitory.reset_history.assert_called_once()

    def test_reset_neuromodulators_requires_auth(self, client):
        """POST /bio/neuromodulators/reset requires admin auth."""
        response = client.post("/bio/neuromodulators/reset")
        assert response.status_code == 401

    # =========================================================================
    # ACh Mode Switching Tests
    # =========================================================================

    def test_switch_ach_mode_to_encoding(self, client, mock_services):
        """Switch to encoding mode (high ACh, new information priority)."""
        # Update mock for encoding mode
        ach = mock_services["episodic"].orchestra.acetylcholine
        ach.force_mode.return_value = MagicMock(
            mode=MagicMock(value="encoding"),
            ach_level=0.8,
            encoding_weight=0.9,
            retrieval_weight=0.1,
        )

        response = client.post(
            "/bio/acetylcholine/switch-mode",
            json={"mode": "encoding"},
            headers=self.admin_headers,
        )
        assert response.status_code == 200

        data = response.json()
        assert data["success"] is True
        assert data["new_mode"] == "encoding"
        assert data["encoding_weight"] > data["retrieval_weight"]

        ach.force_mode.assert_called_once_with("encoding")

    def test_switch_ach_mode_to_retrieval(self, client, mock_services):
        """Switch to retrieval mode (low ACh, pattern completion priority)."""
        ach = mock_services["episodic"].orchestra.acetylcholine
        ach.force_mode.return_value = MagicMock(
            mode=MagicMock(value="retrieval"),
            ach_level=0.2,
            encoding_weight=0.1,
            retrieval_weight=0.9,
        )

        response = client.post(
            "/bio/acetylcholine/switch-mode",
            json={"mode": "retrieval"},
            headers=self.admin_headers,
        )
        assert response.status_code == 200

        data = response.json()
        assert data["new_mode"] == "retrieval"
        assert data["retrieval_weight"] > data["encoding_weight"]

    def test_switch_ach_mode_to_balanced(self, client, mock_services):
        """Switch to balanced mode (moderate ACh)."""
        ach = mock_services["episodic"].orchestra.acetylcholine
        ach.force_mode.return_value = MagicMock(
            mode=MagicMock(value="balanced"),
            ach_level=0.5,
            encoding_weight=0.5,
            retrieval_weight=0.5,
        )

        response = client.post(
            "/bio/acetylcholine/switch-mode",
            json={"mode": "balanced"},
            headers=self.admin_headers,
        )
        assert response.status_code == 200

        data = response.json()
        assert data["new_mode"] == "balanced"

    def test_switch_ach_mode_returns_previous_mode(self, client, mock_services):
        """Response includes previous mode."""
        response = client.post(
            "/bio/acetylcholine/switch-mode",
            json={"mode": "encoding"},
            headers=self.admin_headers,
        )
        assert response.status_code == 200

        data = response.json()
        assert "previous_mode" in data
        assert data["previous_mode"] == "balanced"  # From mock

    def test_switch_ach_mode_returns_ach_level(self, client):
        """Response includes current ACh level."""
        response = client.post(
            "/bio/acetylcholine/switch-mode",
            json={"mode": "encoding"},
            headers=self.admin_headers,
        )

        data = response.json()
        assert "ach_level" in data
        assert 0.0 <= data["ach_level"] <= 1.0

    def test_switch_ach_invalid_mode(self, client):
        """Reject invalid ACh modes."""
        response = client.post(
            "/bio/acetylcholine/switch-mode",
            json={"mode": "invalid_mode"},
            headers=self.admin_headers,
        )
        assert response.status_code == 400
        assert "Invalid mode" in response.json()["detail"]

    def test_switch_ach_case_insensitive(self, client, mock_services):
        """Mode switching is case-insensitive."""
        response = client.post(
            "/bio/acetylcholine/switch-mode",
            json={"mode": "ENCODING"},
            headers=self.admin_headers,
        )
        assert response.status_code == 200

        # Should have called force_mode with lowercase
        ach = mock_services["episodic"].orchestra.acetylcholine
        ach.force_mode.assert_called_with("encoding")

    def test_switch_ach_requires_auth(self, client):
        """POST /bio/acetylcholine/switch-mode requires admin auth."""
        response = client.post(
            "/bio/acetylcholine/switch-mode",
            json={"mode": "encoding"},
        )
        assert response.status_code == 401


class TestBiologicalPlausibilityConstraints:
    """Tests verifying biological plausibility of parameter constraints.

    These tests validate that the API enforces constraints that align
    with known neuroscience findings.
    """

    def test_ne_yerkes_dodson_range(self):
        """NE gain follows Yerkes-Dodson inverted U curve range."""
        # Too low (drowsy) - should be allowed but bounded
        tuning_low = NeuromodulatorTuning(ne_min_gain=0.1)
        assert tuning_low.ne_min_gain == 0.1

        # Optimal range
        tuning_mid = NeuromodulatorTuning(ne_min_gain=0.5, ne_max_gain=2.0)
        assert tuning_mid.ne_min_gain == 0.5

        # Too high (stress) - bounded
        with pytest.raises(ValueError):
            NeuromodulatorTuning(ne_max_gain=10.0)  # Above 5.0

    def test_ach_encoding_retrieval_tradeoff(self):
        """ACh thresholds respect encoding/retrieval balance."""
        # This test documents expected relationship
        tuning = NeuromodulatorTuning(
            ach_encoding_threshold=0.7,
            ach_retrieval_threshold=0.3,
        )
        # Encoding threshold should be higher - this is biologically expected
        # High ACh -> encoding mode, Low ACh -> retrieval mode
        assert tuning.ach_encoding_threshold > tuning.ach_retrieval_threshold

    def test_serotonin_patience_bounds(self):
        """5-HT discount rate bounded for temporal patience."""
        # Near gamma=1 is patient (high 5-HT)
        tuning_patient = NeuromodulatorTuning(serotonin_discount_rate=0.99)
        assert tuning_patient.serotonin_discount_rate == 0.99

        # gamma < 0.9 is pathologically impatient
        with pytest.raises(ValueError):
            NeuromodulatorTuning(serotonin_discount_rate=0.8)

    def test_gaba_sparsity_matches_dg(self):
        """GABA sparsity matches dentate gyrus firing rates."""
        # DG has ~2-5% active neurons
        tuning_dg = NeuromodulatorTuning(sparsity_target=0.05)
        assert tuning_dg.sparsity_target == 0.05

        # CA3 is denser (~10-20%)
        tuning_ca3 = NeuromodulatorTuning(sparsity_target=0.15)
        assert tuning_ca3.sparsity_target == 0.15

        # Above 50% defeats sparse coding
        with pytest.raises(ValueError):
            NeuromodulatorTuning(sparsity_target=0.6)

    def test_homeostatic_bcm_threshold_rate(self):
        """BCM sliding threshold rate bounded for stability."""
        # Slow rate for stable learning
        update_slow = HomeostaticConfigUpdate(sliding_threshold_rate=0.0001)
        assert update_slow.sliding_threshold_rate == 0.0001

        # Fast rate for rapid adaptation (less stable)
        update_fast = HomeostaticConfigUpdate(sliding_threshold_rate=0.01)
        assert update_fast.sliding_threshold_rate == 0.01

        # Rate too fast causes instability
        with pytest.raises(ValueError):
            HomeostaticConfigUpdate(sliding_threshold_rate=0.1)


class TestLearningSystemPropagation:
    """Tests verifying changes propagate to underlying learning systems.

    These tests check that API changes actually affect the learning dynamics.
    """

    @pytest.fixture
    def mock_episodic(self):
        """Create mock with inspectable learning systems."""
        episodic = MagicMock()

        # Track calls to learning rate modulation
        learning_rates = {"ne": 1.0, "ach": 1.0, "da": 1.0}

        orchestra = MagicMock()

        # NE affects attention/exploration
        ne = MagicMock()
        ne.current_gain = 1.0
        def set_arousal(val):
            ne.current_gain = 1.0 + val  # Higher arousal = higher gain
            learning_rates["ne"] = ne.current_gain
        ne.set_baseline_arousal = MagicMock(side_effect=set_arousal)
        orchestra.norepinephrine = ne

        # ACh affects encoding strength
        ach = MagicMock()
        ach.encoding_weight = 0.5
        def set_baseline(val):
            ach.encoding_weight = val
            learning_rates["ach"] = val
        ach.set_baseline_ach = MagicMock(side_effect=set_baseline)
        ach.get_current_mode.return_value = MagicMock(value="balanced")
        ach.force_mode.return_value = MagicMock(
            mode=MagicMock(value="encoding"),
            ach_level=0.8,
            encoding_weight=0.9,
            retrieval_weight=0.1,
        )
        orchestra.acetylcholine = ach

        # DA affects reward prediction
        da = MagicMock()
        orchestra.dopamine = da

        # 5-HT affects temporal credit
        serotonin = MagicMock()
        orchestra.serotonin = serotonin

        # GABA affects sparsity
        inhib = MagicMock()
        orchestra.inhibitory = inhib

        episodic.orchestra = orchestra
        episodic.get_current_neuromodulator_state.return_value = {}

        # Homeostatic mock
        homeostatic = MagicMock()
        homeostatic.get_state.return_value = MagicMock(
            mean_norm=1.0,
            std_norm=0.1,
            mean_activation=0.5,
            sliding_threshold=0.5,
            last_update=datetime.now(),
        )
        homeostatic.get_stats.return_value = {"config": {}}
        homeostatic.needs_scaling.return_value = False
        homeostatic.compute_scaling_factor.return_value = 1.0
        episodic.homeostatic = homeostatic

        episodic._learning_rates = learning_rates
        return episodic

    @pytest.fixture
    def client_with_learning(self, mock_episodic):
        """Create test client with learning system mocks."""
        from t4dm.api.deps import get_memory_services
        from t4dm.core.config import get_settings

        services = {
            "session_id": "test",
            "episodic": mock_episodic,
            "semantic": MagicMock(),
            "procedural": MagicMock(),
        }

        app = FastAPI()
        app.include_router(router)
        app.dependency_overrides[get_memory_services] = lambda: services

        settings = get_settings()
        settings.admin_api_key = "test-key"

        return TestClient(app), mock_episodic

    def test_ne_change_affects_attention(self, client_with_learning):
        """Changing NE arousal affects attention/exploration gain."""
        client, episodic = client_with_learning

        initial_gain = episodic.orchestra.norepinephrine.current_gain

        response = client.put(
            "/bio/neuromodulators",
            json={"ne_baseline_arousal": 0.8},
            headers={"X-Admin-Key": "test-key"},
        )
        assert response.status_code == 200

        # Verify gain increased
        assert episodic._learning_rates["ne"] > initial_gain

    def test_ach_mode_affects_encoding_strength(self, client_with_learning):
        """ACh encoding mode increases encoding weight."""
        client, episodic = client_with_learning

        response = client.post(
            "/bio/acetylcholine/switch-mode",
            json={"mode": "encoding"},
            headers={"X-Admin-Key": "test-key"},
        )
        assert response.status_code == 200

        data = response.json()
        assert data["encoding_weight"] > data["retrieval_weight"]
