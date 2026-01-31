"""Tests for configuration API routes."""

import pytest
from fastapi.testclient import TestClient
from unittest.mock import patch, MagicMock

from t4dm.api.routes.config import (
    router,
    FSRSConfig,
    ACTRConfig,
    HebbianConfig,
    NeuromodConfig,
    PatternSepConfig,
    MemoryGateConfig,
    ConsolidationConfig,
    EpisodicWeightsConfig,
    SemanticWeightsConfig,
    ProceduralWeightsConfig,
    ThreeFactorConfig,
    LearnedGateConfig,
    BioinspiredConfig,
    SystemConfigResponse,
    SystemConfigUpdate,
    PRESETS,
    _runtime_config,
)


class TestFSRSConfig:
    """Tests for FSRSConfig model."""

    def test_valid_config(self):
        """Create valid FSRS config."""
        config = FSRSConfig(
            defaultStability=1.0,
            retentionTarget=0.9,
            decayFactor=0.5,
            recencyDecay=0.1,
        )
        assert config.defaultStability == 1.0
        assert config.retentionTarget == 0.9

    def test_boundary_values(self):
        """Test boundary values."""
        config = FSRSConfig(
            defaultStability=0.1,  # min
            retentionTarget=1.0,  # max
            decayFactor=0.1,
            recencyDecay=1.0,
        )
        assert config.defaultStability == 0.1

    def test_invalid_stability_low(self):
        """Reject stability below minimum."""
        with pytest.raises(ValueError):
            FSRSConfig(
                defaultStability=0.05,  # Below 0.1 minimum
                retentionTarget=0.9,
                decayFactor=0.5,
                recencyDecay=0.1,
            )

    def test_invalid_retention_high(self):
        """Reject retention above maximum."""
        with pytest.raises(ValueError):
            FSRSConfig(
                defaultStability=1.0,
                retentionTarget=1.5,  # Above 1.0 maximum
                decayFactor=0.5,
                recencyDecay=0.1,
            )


class TestACTRConfig:
    """Tests for ACTRConfig model."""

    def test_valid_config(self):
        """Create valid ACT-R config."""
        config = ACTRConfig(
            decay=0.5,
            noise=0.2,
            threshold=0.0,
            spreadingWeight=1.5,
        )
        assert config.decay == 0.5
        assert config.noise == 0.2

    def test_negative_threshold(self):
        """Threshold can be negative."""
        config = ACTRConfig(
            decay=0.5,
            noise=0.2,
            threshold=-3.0,
            spreadingWeight=1.5,
        )
        assert config.threshold == -3.0

    def test_invalid_decay(self):
        """Reject invalid decay value."""
        with pytest.raises(ValueError):
            ACTRConfig(
                decay=0.05,  # Below 0.1
                noise=0.2,
                threshold=0.0,
                spreadingWeight=1.5,
            )


class TestHebbianConfig:
    """Tests for HebbianConfig model."""

    def test_valid_config(self):
        """Create valid Hebbian config."""
        config = HebbianConfig(
            learningRate=0.1,
            initialWeight=0.5,
            minWeight=0.01,
            decayRate=0.01,
            staleDays=30,
        )
        assert config.learningRate == 0.1
        assert config.staleDays == 30

    def test_invalid_learning_rate(self):
        """Reject invalid learning rate."""
        with pytest.raises(ValueError):
            HebbianConfig(
                learningRate=0.6,  # Above 0.5
                initialWeight=0.5,
                minWeight=0.01,
                decayRate=0.01,
                staleDays=30,
            )


class TestNeuromodConfig:
    """Tests for NeuromodConfig model."""

    def test_valid_config(self):
        """Create valid neuromodulation config."""
        config = NeuromodConfig(
            dopamineBaseline=0.5,
            norepinephrineGain=1.0,
            serotoninDiscount=0.5,
            acetylcholineThreshold=0.3,
            gabaInhibition=0.2,
        )
        assert config.dopamineBaseline == 0.5

    def test_zero_baseline(self):
        """Zero baseline is valid."""
        config = NeuromodConfig(
            dopamineBaseline=0.0,
            norepinephrineGain=0.1,
            serotoninDiscount=0.0,
            acetylcholineThreshold=0.0,
            gabaInhibition=0.0,
        )
        assert config.dopamineBaseline == 0.0


class TestPatternSepConfig:
    """Tests for PatternSepConfig model."""

    def test_valid_config(self):
        """Create valid pattern separation config."""
        config = PatternSepConfig(
            targetSparsity=0.1,
            maxNeighbors=50,
            maxNodes=1000,
        )
        assert config.targetSparsity == 0.1
        assert config.maxNeighbors == 50

    def test_boundary_values(self):
        """Test boundary values."""
        config = PatternSepConfig(
            targetSparsity=0.01,  # min
            maxNeighbors=200,  # max
            maxNodes=10000,  # max
        )
        assert config.maxNeighbors == 200


class TestMemoryGateConfig:
    """Tests for MemoryGateConfig model."""

    def test_valid_config(self):
        """Create valid memory gate config."""
        config = MemoryGateConfig(
            baseThreshold=0.5,
            noveltyWeight=0.3,
            importanceWeight=0.3,
            contextWeight=0.4,
        )
        assert config.baseThreshold == 0.5


class TestConsolidationConfig:
    """Tests for ConsolidationConfig model."""

    def test_valid_config(self):
        """Create valid consolidation config."""
        config = ConsolidationConfig(
            minSimilarity=0.8,
            minOccurrences=3,
            skillSimilarity=0.9,
            clusterSize=10,
        )
        assert config.minSimilarity == 0.8
        assert config.minOccurrences == 3


class TestEpisodicWeightsConfig:
    """Tests for EpisodicWeightsConfig model."""

    def test_valid_config(self):
        """Create valid episodic weights config."""
        config = EpisodicWeightsConfig(
            semanticWeight=0.4,
            recencyWeight=0.3,
            outcomeWeight=0.2,
            importanceWeight=0.1,
        )
        assert config.semanticWeight == 0.4

    def test_weights_can_be_zero(self):
        """Weights can be zero."""
        config = EpisodicWeightsConfig(
            semanticWeight=1.0,
            recencyWeight=0.0,
            outcomeWeight=0.0,
            importanceWeight=0.0,
        )
        assert config.recencyWeight == 0.0


class TestSemanticWeightsConfig:
    """Tests for SemanticWeightsConfig model."""

    def test_valid_config(self):
        """Create valid semantic weights config."""
        config = SemanticWeightsConfig(
            similarityWeight=0.5,
            activationWeight=0.3,
            retrievabilityWeight=0.2,
        )
        assert config.similarityWeight == 0.5


class TestProceduralWeightsConfig:
    """Tests for ProceduralWeightsConfig model."""

    def test_valid_config(self):
        """Create valid procedural weights config."""
        config = ProceduralWeightsConfig(
            similarityWeight=0.4,
            successWeight=0.4,
            experienceWeight=0.2,
        )
        assert config.successWeight == 0.4


class TestThreeFactorConfig:
    """Tests for ThreeFactorConfig model (three-factor learning rule)."""

    def test_valid_config(self):
        """Create valid three-factor learning config."""
        config = ThreeFactorConfig(
            achWeight=0.4,
            neWeight=0.35,
            serotoninWeight=0.25,
            minEffectiveLr=0.1,
            maxEffectiveLr=3.0,
            bootstrapRate=0.01,
        )
        assert config.achWeight == 0.4
        assert config.neWeight == 0.35
        assert config.serotoninWeight == 0.25

    def test_defaults(self):
        """Default values match expected neuroscience-inspired values."""
        config = ThreeFactorConfig()
        assert config.achWeight == 0.4
        assert config.neWeight == 0.35
        assert config.serotoninWeight == 0.25
        assert config.minEffectiveLr == 0.1
        assert config.maxEffectiveLr == 3.0
        assert config.bootstrapRate == 0.01

    def test_boundary_values(self):
        """Test boundary values (weights must sum to 1.0)."""
        config = ThreeFactorConfig(
            achWeight=0.0,  # min
            neWeight=0.6,
            serotoninWeight=0.4,  # sum = 1.0
            minEffectiveLr=0.01,  # min
            maxEffectiveLr=10.0,  # max
            bootstrapRate=0.1,  # max
        )
        assert config.achWeight == 0.0
        assert config.neWeight == 0.6

    def test_weight_sum_validation(self):
        """Weights must sum to 1.0."""
        with pytest.raises(ValueError, match="must sum to 1.0"):
            ThreeFactorConfig(
                achWeight=0.4,
                neWeight=0.4,
                serotoninWeight=0.4,  # sum = 1.2
            )

    def test_invalid_ach_weight_high(self):
        """Reject ACh weight above maximum."""
        with pytest.raises(ValueError):
            ThreeFactorConfig(achWeight=1.5)

    def test_invalid_min_lr_too_low(self):
        """Reject min effective LR below threshold."""
        with pytest.raises(ValueError):
            ThreeFactorConfig(minEffectiveLr=0.001)

    def test_invalid_max_lr_too_high(self):
        """Reject max effective LR above ceiling."""
        with pytest.raises(ValueError):
            ThreeFactorConfig(maxEffectiveLr=15.0)


class TestLearnedGateConfig:
    """Tests for LearnedGateConfig model (Thompson sampling memory gate)."""

    def test_valid_config(self):
        """Create valid learned gate config."""
        config = LearnedGateConfig(
            storeThreshold=0.6,
            bufferThreshold=0.3,
            learningRateMean=0.1,
            learningRateVar=0.05,
            coldStartThreshold=100,
            thompsonTemperature=1.0,
        )
        assert config.storeThreshold == 0.6
        assert config.coldStartThreshold == 100

    def test_defaults(self):
        """Default values match expected values."""
        config = LearnedGateConfig()
        assert config.storeThreshold == 0.6
        assert config.bufferThreshold == 0.3
        assert config.learningRateMean == 0.1
        assert config.learningRateVar == 0.05
        assert config.coldStartThreshold == 100
        assert config.thompsonTemperature == 1.0

    def test_boundary_values(self):
        """Test boundary values."""
        config = LearnedGateConfig(
            storeThreshold=0.3,  # min
            bufferThreshold=0.5,  # max
            learningRateMean=0.5,  # max
            learningRateVar=0.2,  # max
            coldStartThreshold=1000,  # max
            thompsonTemperature=5.0,  # max
        )
        assert config.storeThreshold == 0.3
        assert config.coldStartThreshold == 1000

    def test_invalid_store_threshold_low(self):
        """Reject store threshold below minimum."""
        with pytest.raises(ValueError):
            LearnedGateConfig(storeThreshold=0.1)

    def test_invalid_store_threshold_high(self):
        """Reject store threshold above maximum."""
        with pytest.raises(ValueError):
            LearnedGateConfig(storeThreshold=0.95)

    def test_invalid_cold_start(self):
        """Reject cold start threshold below minimum."""
        with pytest.raises(ValueError):
            LearnedGateConfig(coldStartThreshold=5)


class TestBioinspiredConfig:
    """Tests for BioinspiredConfig including three-factor and learned gate."""

    def test_includes_three_factor(self):
        """BioinspiredConfig includes threeFactor field."""
        config = BioinspiredConfig()
        assert hasattr(config, 'threeFactor')
        assert isinstance(config.threeFactor, ThreeFactorConfig)

    def test_includes_learned_gate(self):
        """BioinspiredConfig includes learnedGate field."""
        config = BioinspiredConfig()
        assert hasattr(config, 'learnedGate')
        assert isinstance(config.learnedGate, LearnedGateConfig)

    def test_custom_three_factor(self):
        """Set custom three-factor values."""
        config = BioinspiredConfig(
            threeFactor=ThreeFactorConfig(achWeight=0.5, neWeight=0.3, serotoninWeight=0.2)
        )
        assert config.threeFactor.achWeight == 0.5

    def test_custom_learned_gate(self):
        """Set custom learned gate values."""
        config = BioinspiredConfig(
            learnedGate=LearnedGateConfig(storeThreshold=0.7, coldStartThreshold=200)
        )
        assert config.learnedGate.storeThreshold == 0.7
        assert config.learnedGate.coldStartThreshold == 200


class TestSystemConfigResponse:
    """Tests for SystemConfigResponse model."""

    def test_full_config(self):
        """Create full system config response."""
        config = SystemConfigResponse(
            fsrs=FSRSConfig(
                defaultStability=1.0,
                retentionTarget=0.9,
                decayFactor=0.5,
                recencyDecay=0.1,
            ),
            actr=ACTRConfig(
                decay=0.5,
                noise=0.2,
                threshold=0.0,
                spreadingWeight=1.5,
            ),
            hebbian=HebbianConfig(
                learningRate=0.1,
                initialWeight=0.5,
                minWeight=0.01,
                decayRate=0.01,
                staleDays=30,
            ),
            neuromod=NeuromodConfig(
                dopamineBaseline=0.5,
                norepinephrineGain=1.0,
                serotoninDiscount=0.5,
                acetylcholineThreshold=0.3,
                gabaInhibition=0.2,
            ),
            patternSep=PatternSepConfig(
                targetSparsity=0.1,
                maxNeighbors=50,
                maxNodes=1000,
            ),
            memoryGate=MemoryGateConfig(
                baseThreshold=0.5,
                noveltyWeight=0.3,
                importanceWeight=0.3,
                contextWeight=0.4,
            ),
            consolidation=ConsolidationConfig(
                minSimilarity=0.8,
                minOccurrences=3,
                skillSimilarity=0.9,
                clusterSize=10,
            ),
            episodicWeights=EpisodicWeightsConfig(
                semanticWeight=0.4,
                recencyWeight=0.3,
                outcomeWeight=0.2,
                importanceWeight=0.1,
            ),
            semanticWeights=SemanticWeightsConfig(
                similarityWeight=0.5,
                activationWeight=0.3,
                retrievabilityWeight=0.2,
            ),
            proceduralWeights=ProceduralWeightsConfig(
                similarityWeight=0.4,
                successWeight=0.4,
                experienceWeight=0.2,
            ),
        )
        assert config.fsrs.defaultStability == 1.0
        assert config.actr.decay == 0.5


class TestSystemConfigUpdate:
    """Tests for SystemConfigUpdate model."""

    def test_empty_update(self):
        """Create empty update (all optional)."""
        update = SystemConfigUpdate()
        assert update.fsrs is None
        assert update.actr is None

    def test_partial_update(self):
        """Create partial update."""
        update = SystemConfigUpdate(
            fsrs=FSRSConfig(
                defaultStability=2.0,
                retentionTarget=0.85,
                decayFactor=0.6,
                recencyDecay=0.2,
            ),
        )
        assert update.fsrs is not None
        assert update.fsrs.defaultStability == 2.0
        assert update.actr is None

    def test_full_update(self):
        """Create full update."""
        update = SystemConfigUpdate(
            fsrs=FSRSConfig(
                defaultStability=1.0,
                retentionTarget=0.9,
                decayFactor=0.5,
                recencyDecay=0.1,
            ),
            actr=ACTRConfig(
                decay=0.5,
                noise=0.2,
                threshold=0.0,
                spreadingWeight=1.5,
            ),
        )
        assert update.fsrs is not None
        assert update.actr is not None


class TestConfigRouteHelpers:
    """Tests for config route helper functions."""

    def test_runtime_config_is_dict(self):
        """Runtime config is a dictionary."""
        from t4dm.api.routes.config import _runtime_config
        assert isinstance(_runtime_config, dict)


class TestConfigRouteEndpoints:
    """Integration tests for config route endpoints."""

    ADMIN_KEY = "test-admin-key-12345"  # Test admin key

    @pytest.fixture
    def client(self):
        """Create test client with admin key configured."""
        from fastapi import FastAPI
        app = FastAPI()
        app.include_router(router, prefix="/config")
        return TestClient(app)

    @pytest.fixture(autouse=True)
    def reset_runtime_config(self):
        """Reset runtime config and set admin key before each test."""
        from t4dm.api.routes import config as config_module
        from t4dm.core.config import get_settings
        config_module._runtime_config = {}
        # Set admin key for tests
        settings = get_settings()
        original_key = getattr(settings, 'admin_api_key', None)
        settings.admin_api_key = self.ADMIN_KEY
        yield
        settings.admin_api_key = original_key
        config_module._runtime_config = {}

    @property
    def admin_headers(self):
        """Headers with admin authentication."""
        return {"X-Admin-Key": self.ADMIN_KEY}

    def test_get_config(self, client):
        """Get current configuration."""
        response = client.get("/config")
        assert response.status_code == 200

        data = response.json()
        assert "fsrs" in data
        assert "actr" in data
        assert "hebbian" in data
        assert "neuromod" in data
        assert "patternSep" in data
        assert "memoryGate" in data
        assert "consolidation" in data

    def test_update_fsrs_config(self, client):
        """Update FSRS configuration."""
        update = {
            "fsrs": {
                "defaultStability": 2.0,
                "retentionTarget": 0.85,
                "decayFactor": 0.6,
                "recencyDecay": 0.15,
            }
        }
        response = client.put("/config", json=update, headers=self.admin_headers)
        assert response.status_code == 200

        data = response.json()
        assert data["fsrs"]["defaultStability"] == 2.0

    def test_update_actr_config(self, client):
        """Update ACT-R configuration."""
        update = {
            "actr": {
                "decay": 0.6,
                "noise": 0.3,
                "threshold": -1.0,
                "spreadingWeight": 2.0,
            }
        }
        response = client.put("/config", json=update, headers=self.admin_headers)
        assert response.status_code == 200

        data = response.json()
        assert data["actr"]["decay"] == 0.6

    def test_update_hebbian_config(self, client):
        """Update Hebbian configuration."""
        update = {
            "hebbian": {
                "learningRate": 0.15,
                "initialWeight": 0.6,
                "minWeight": 0.02,
                "decayRate": 0.02,
                "staleDays": 60,
            }
        }
        response = client.put("/config", json=update, headers=self.admin_headers)
        assert response.status_code == 200

    def test_update_neuromod_config(self, client):
        """Update neuromodulation configuration."""
        update = {
            "neuromod": {
                "dopamineBaseline": 0.6,
                "norepinephrineGain": 1.5,
                "serotoninDiscount": 0.7,
                "acetylcholineThreshold": 0.4,
                "gabaInhibition": 0.3,
            }
        }
        response = client.put("/config", json=update, headers=self.admin_headers)
        assert response.status_code == 200

    def test_update_pattern_sep_config(self, client):
        """Update pattern separation configuration."""
        update = {
            "patternSep": {
                "targetSparsity": 0.15,
                "maxNeighbors": 75,
                "maxNodes": 2000,
            }
        }
        response = client.put("/config", json=update, headers=self.admin_headers)
        assert response.status_code == 200

    def test_update_memory_gate_config(self, client):
        """Update memory gate configuration."""
        update = {
            "memoryGate": {
                "baseThreshold": 0.6,
                "noveltyWeight": 0.4,
                "importanceWeight": 0.4,
                "contextWeight": 0.2,
            }
        }
        response = client.put("/config", json=update, headers=self.admin_headers)
        assert response.status_code == 200

    def test_update_consolidation_config(self, client):
        """Update consolidation configuration."""
        update = {
            "consolidation": {
                "minSimilarity": 0.85,
                "minOccurrences": 4,
                "skillSimilarity": 0.92,
                "clusterSize": 15,
            }
        }
        response = client.put("/config", json=update, headers=self.admin_headers)
        assert response.status_code == 200

    def test_update_episodic_weights_valid(self, client):
        """Update episodic weights that sum to 1.0."""
        update = {
            "episodicWeights": {
                "semanticWeight": 0.3,
                "recencyWeight": 0.3,
                "outcomeWeight": 0.2,
                "importanceWeight": 0.2,
            }
        }
        response = client.put("/config", json=update, headers=self.admin_headers)
        assert response.status_code == 200

    def test_update_episodic_weights_invalid_sum(self, client):
        """Reject episodic weights not summing to 1.0."""
        update = {
            "episodicWeights": {
                "semanticWeight": 0.5,
                "recencyWeight": 0.5,
                "outcomeWeight": 0.5,
                "importanceWeight": 0.5,
            }
        }
        response = client.put("/config", json=update, headers=self.admin_headers)
        assert response.status_code == 422
        assert "must sum to 1.0" in response.json()["detail"]

    def test_update_semantic_weights_valid(self, client):
        """Update semantic weights that sum to 1.0."""
        update = {
            "semanticWeights": {
                "similarityWeight": 0.4,
                "activationWeight": 0.4,
                "retrievabilityWeight": 0.2,
            }
        }
        response = client.put("/config", json=update, headers=self.admin_headers)
        assert response.status_code == 200

    def test_update_semantic_weights_invalid_sum(self, client):
        """Reject semantic weights not summing to 1.0."""
        update = {
            "semanticWeights": {
                "similarityWeight": 0.5,
                "activationWeight": 0.5,
                "retrievabilityWeight": 0.5,
            }
        }
        response = client.put("/config", json=update, headers=self.admin_headers)
        assert response.status_code == 422

    def test_update_procedural_weights_valid(self, client):
        """Update procedural weights that sum to 1.0."""
        update = {
            "proceduralWeights": {
                "similarityWeight": 0.33,
                "successWeight": 0.34,
                "experienceWeight": 0.33,
            }
        }
        response = client.put("/config", json=update, headers=self.admin_headers)
        assert response.status_code == 200

    def test_update_procedural_weights_invalid_sum(self, client):
        """Reject procedural weights not summing to 1.0."""
        update = {
            "proceduralWeights": {
                "similarityWeight": 0.2,
                "successWeight": 0.2,
                "experienceWeight": 0.2,
            }
        }
        response = client.put("/config", json=update, headers=self.admin_headers)
        assert response.status_code == 422

    def test_reset_config(self, client):
        """Reset configuration to defaults."""
        # First update something
        update = {"fsrs": {"defaultStability": 5.0, "retentionTarget": 0.8, "decayFactor": 0.4, "recencyDecay": 0.05}}
        client.put("/config", json=update, headers=self.admin_headers)

        # Then reset
        response = client.post("/config/reset", headers=self.admin_headers)
        assert response.status_code == 200
        assert response.json()["status"] == "reset"

    def test_update_multiple_configs(self, client):
        """Update multiple config sections at once."""
        update = {
            "fsrs": {
                "defaultStability": 2.0,
                "retentionTarget": 0.85,
                "decayFactor": 0.6,
                "recencyDecay": 0.15,
            },
            "actr": {
                "decay": 0.6,
                "noise": 0.3,
                "threshold": -0.5,
                "spreadingWeight": 2.0,
            },
        }
        response = client.put("/config", json=update, headers=self.admin_headers)
        assert response.status_code == 200

        data = response.json()
        assert data["fsrs"]["defaultStability"] == 2.0
        assert data["actr"]["decay"] == 0.6

    def test_update_config_requires_auth(self, client):
        """Update config requires admin authentication."""
        update = {"fsrs": {"defaultStability": 2.0, "retentionTarget": 0.85, "decayFactor": 0.6, "recencyDecay": 0.15}}
        response = client.put("/config", json=update)  # No headers
        assert response.status_code == 401

    def test_update_config_rejects_invalid_key(self, client):
        """Update config rejects invalid admin key."""
        update = {"fsrs": {"defaultStability": 2.0, "retentionTarget": 0.85, "decayFactor": 0.6, "recencyDecay": 0.15}}
        response = client.put("/config", json=update, headers={"X-Admin-Key": "wrong-key"})
        assert response.status_code == 403

    def test_reset_config_requires_auth(self, client):
        """Reset config requires admin authentication."""
        response = client.post("/config/reset")  # No headers
        assert response.status_code == 401

    def test_reset_config_rejects_invalid_key(self, client):
        """Reset config rejects invalid admin key."""
        response = client.post("/config/reset", headers={"X-Admin-Key": "wrong-key"})
        assert response.status_code == 403

    def test_get_config_includes_bioinspired(self, client):
        """Get config includes bioinspired section with three-factor and learned gate."""
        response = client.get("/config")
        assert response.status_code == 200

        data = response.json()
        assert "bioinspired" in data
        assert "threeFactor" in data["bioinspired"]
        assert "learnedGate" in data["bioinspired"]

        # Verify three-factor defaults
        tf = data["bioinspired"]["threeFactor"]
        assert tf["achWeight"] == 0.4
        assert tf["neWeight"] == 0.35
        assert tf["serotoninWeight"] == 0.25
        assert tf["minEffectiveLr"] == 0.1
        assert tf["maxEffectiveLr"] == 3.0
        assert tf["bootstrapRate"] == 0.01

        # Verify learned gate defaults
        lg = data["bioinspired"]["learnedGate"]
        assert lg["storeThreshold"] == 0.6
        assert lg["bufferThreshold"] == 0.3
        assert lg["coldStartThreshold"] == 100

    def test_update_three_factor_config(self, client):
        """Update three-factor learning configuration."""
        update = {
            "bioinspired": {
                "enabled": True,
                "dendritic": {"hiddenDim": 512, "contextDim": 512, "couplingStrength": 0.5, "tauDendrite": 10.0, "tauSoma": 15.0},
                "sparseEncoder": {"hiddenDim": 8192, "sparsity": 0.02, "useKwta": True, "lateralInhibition": 0.2},
                "attractor": {"settlingSteps": 10, "noiseStd": 0.01, "adaptationTau": 5.0, "stepSize": 0.1},
                "fastEpisodic": {"capacity": 10000, "learningRate": 0.1, "consolidationThreshold": 0.7},
                "neuromodGains": {"rhoDa": 1.0, "rhoNe": 1.0, "rhoAchFast": 1.0, "rhoAchSlow": 0.5, "alphaNe": 0.1},
                "eligibility": {"decay": 0.95, "tauTrace": 20.0},
                "threeFactor": {
                    "achWeight": 0.5,
                    "neWeight": 0.3,
                    "serotoninWeight": 0.2,
                    "minEffectiveLr": 0.15,
                    "maxEffectiveLr": 4.0,
                    "bootstrapRate": 0.02,
                },
                "learnedGate": {"storeThreshold": 0.6, "bufferThreshold": 0.3, "learningRateMean": 0.1, "learningRateVar": 0.05, "coldStartThreshold": 100, "thompsonTemperature": 1.0},
            }
        }
        response = client.put("/config", json=update, headers=self.admin_headers)
        assert response.status_code == 200

        data = response.json()
        tf = data["bioinspired"]["threeFactor"]
        assert tf["achWeight"] == 0.5
        assert tf["neWeight"] == 0.3
        assert tf["serotoninWeight"] == 0.2
        assert tf["minEffectiveLr"] == 0.15
        assert tf["maxEffectiveLr"] == 4.0
        assert tf["bootstrapRate"] == 0.02

    def test_update_learned_gate_config(self, client):
        """Update learned gate (Thompson sampling) configuration."""
        update = {
            "bioinspired": {
                "enabled": True,
                "dendritic": {"hiddenDim": 512, "contextDim": 512, "couplingStrength": 0.5, "tauDendrite": 10.0, "tauSoma": 15.0},
                "sparseEncoder": {"hiddenDim": 8192, "sparsity": 0.02, "useKwta": True, "lateralInhibition": 0.2},
                "attractor": {"settlingSteps": 10, "noiseStd": 0.01, "adaptationTau": 5.0, "stepSize": 0.1},
                "fastEpisodic": {"capacity": 10000, "learningRate": 0.1, "consolidationThreshold": 0.7},
                "neuromodGains": {"rhoDa": 1.0, "rhoNe": 1.0, "rhoAchFast": 1.0, "rhoAchSlow": 0.5, "alphaNe": 0.1},
                "eligibility": {"decay": 0.95, "tauTrace": 20.0},
                "threeFactor": {"achWeight": 0.4, "neWeight": 0.35, "serotoninWeight": 0.25, "minEffectiveLr": 0.1, "maxEffectiveLr": 3.0, "bootstrapRate": 0.01},
                "learnedGate": {
                    "storeThreshold": 0.7,
                    "bufferThreshold": 0.4,
                    "learningRateMean": 0.15,
                    "learningRateVar": 0.08,
                    "coldStartThreshold": 200,
                    "thompsonTemperature": 1.5,
                },
            }
        }
        response = client.put("/config", json=update, headers=self.admin_headers)
        assert response.status_code == 200

        data = response.json()
        lg = data["bioinspired"]["learnedGate"]
        assert lg["storeThreshold"] == 0.7
        assert lg["bufferThreshold"] == 0.4
        assert lg["learningRateMean"] == 0.15
        assert lg["learningRateVar"] == 0.08
        assert lg["coldStartThreshold"] == 200
        assert lg["thompsonTemperature"] == 1.5

    def test_update_three_factor_persists_after_reset(self, client):
        """Config resets clear three-factor customizations."""
        # First update three-factor
        update = {
            "bioinspired": {
                "enabled": True,
                "dendritic": {"hiddenDim": 512, "contextDim": 512, "couplingStrength": 0.5, "tauDendrite": 10.0, "tauSoma": 15.0},
                "sparseEncoder": {"hiddenDim": 8192, "sparsity": 0.02, "useKwta": True, "lateralInhibition": 0.2},
                "attractor": {"settlingSteps": 10, "noiseStd": 0.01, "adaptationTau": 5.0, "stepSize": 0.1},
                "fastEpisodic": {"capacity": 10000, "learningRate": 0.1, "consolidationThreshold": 0.7},
                "neuromodGains": {"rhoDa": 1.0, "rhoNe": 1.0, "rhoAchFast": 1.0, "rhoAchSlow": 0.5, "alphaNe": 0.1},
                "eligibility": {"decay": 0.95, "tauTrace": 20.0},
                "threeFactor": {"achWeight": 0.6, "neWeight": 0.25, "serotoninWeight": 0.15, "minEffectiveLr": 0.2, "maxEffectiveLr": 5.0, "bootstrapRate": 0.03},
                "learnedGate": {"storeThreshold": 0.6, "bufferThreshold": 0.3, "learningRateMean": 0.1, "learningRateVar": 0.05, "coldStartThreshold": 100, "thompsonTemperature": 1.0},
            }
        }
        client.put("/config", json=update, headers=self.admin_headers)

        # Verify update
        response = client.get("/config")
        assert response.json()["bioinspired"]["threeFactor"]["achWeight"] == 0.6

        # Reset
        client.post("/config/reset", headers=self.admin_headers)

        # Verify reset to defaults
        response = client.get("/config")
        assert response.json()["bioinspired"]["threeFactor"]["achWeight"] == 0.4

    def test_list_presets(self, client):
        """List available configuration presets."""
        response = client.get("/config/presets")
        assert response.status_code == 200

        data = response.json()
        assert "presets" in data
        preset_names = [p["name"] for p in data["presets"]]
        assert "bio-plausible" in preset_names
        assert "performance" in preset_names
        assert "conservative" in preset_names
        assert "exploration" in preset_names

    def test_list_presets_includes_descriptions(self, client):
        """Preset list includes descriptions and changes."""
        response = client.get("/config/presets")
        data = response.json()

        bio_preset = next(p for p in data["presets"] if p["name"] == "bio-plausible")
        assert "description" in bio_preset
        assert "CompBio" in bio_preset["description"]
        assert "changes" in bio_preset
        assert len(bio_preset["changes"]) > 0

    def test_apply_bio_plausible_preset(self, client):
        """Apply bio-plausible preset."""
        response = client.post("/config/presets/bio-plausible", headers=self.admin_headers)
        assert response.status_code == 200

        data = response.json()
        assert data["preset"] == "bio-plausible"
        assert "applied_changes" in data
        assert "config" in data

        # Verify specific bio-plausible changes were applied
        assert "gaba_inhibition" in data["applied_changes"]
        assert data["applied_changes"]["gaba_inhibition"] == 0.75

    def test_apply_performance_preset(self, client):
        """Apply performance preset."""
        response = client.post("/config/presets/performance", headers=self.admin_headers)
        assert response.status_code == 200

        data = response.json()
        assert data["preset"] == "performance"
        assert "sparse_sparsity" in data["applied_changes"]
        assert data["applied_changes"]["sparse_sparsity"] == 0.01

    def test_apply_conservative_preset(self, client):
        """Apply conservative preset."""
        response = client.post("/config/presets/conservative", headers=self.admin_headers)
        assert response.status_code == 200

        data = response.json()
        assert data["preset"] == "conservative"
        assert "fsrs_retention_target" in data["applied_changes"]
        assert data["applied_changes"]["fsrs_retention_target"] == 0.95

    def test_apply_exploration_preset(self, client):
        """Apply exploration preset."""
        response = client.post("/config/presets/exploration", headers=self.admin_headers)
        assert response.status_code == 200

        data = response.json()
        assert data["preset"] == "exploration"
        assert "norepinephrine_gain" in data["applied_changes"]
        assert data["applied_changes"]["norepinephrine_gain"] == 1.5

    def test_apply_preset_requires_auth(self, client):
        """Apply preset requires admin authentication."""
        response = client.post("/config/presets/bio-plausible")  # No headers
        assert response.status_code == 401

    def test_apply_preset_rejects_invalid_key(self, client):
        """Apply preset rejects invalid admin key."""
        response = client.post("/config/presets/bio-plausible", headers={"X-Admin-Key": "wrong-key"})
        assert response.status_code == 403

    def test_apply_unknown_preset(self, client):
        """Reject unknown preset name."""
        response = client.post("/config/presets/unknown-preset", headers=self.admin_headers)
        assert response.status_code == 404
        assert "not found" in response.json()["detail"]

    def test_preset_then_reset(self, client):
        """Applying preset then reset clears changes."""
        # Apply bio-plausible preset
        client.post("/config/presets/bio-plausible", headers=self.admin_headers)

        # Verify preset was applied (gaba_inhibition = 0.75)
        response = client.get("/config")
        assert response.json()["neuromod"]["gabaInhibition"] == 0.75

        # Reset
        client.post("/config/reset", headers=self.admin_headers)

        # Verify reset to default (0.3)
        response = client.get("/config")
        assert response.json()["neuromod"]["gabaInhibition"] == 0.3


class TestPresetsModule:
    """Tests for PRESETS data structure."""

    def test_presets_dict_exists(self):
        """PRESETS dict is defined."""
        assert isinstance(PRESETS, dict)
        assert len(PRESETS) >= 4

    def test_bio_plausible_preset_structure(self):
        """Bio-plausible preset has required structure."""
        assert "bio-plausible" in PRESETS
        preset = PRESETS["bio-plausible"]
        assert "description" in preset
        assert "changes" in preset
        assert isinstance(preset["changes"], dict)

    def test_bio_plausible_includes_compbio_recommendations(self):
        """Bio-plausible preset includes CompBio agent recommendations."""
        changes = PRESETS["bio-plausible"]["changes"]
        # Key CompBio recommendations
        assert changes.get("gaba_inhibition") == 0.75  # E/I ratio
        assert changes.get("sparse_sparsity") == 0.05  # DG sparsity
        assert changes.get("neuromod_alpha_ne") == 0.3  # LC-NE decay

    def test_all_presets_have_required_fields(self):
        """All presets have description and changes."""
        for name, preset in PRESETS.items():
            assert "description" in preset, f"Preset {name} missing description"
            assert "changes" in preset, f"Preset {name} missing changes"
            assert isinstance(preset["changes"], dict), f"Preset {name} changes not dict"
