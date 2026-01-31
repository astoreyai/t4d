"""
Unit tests for unified learning signals.

Tests the Phase 6B integration of all learning signals:
1. Three-factor computation
2. FF goodness modulation
3. Capsule agreement modulation
4. Neurogenesis structural updates
5. Combined update computation
"""

import pytest
import numpy as np
from uuid import uuid4
from datetime import datetime
from dataclasses import dataclass

from t4dm.learning.unified_signals import (
    UnifiedLearningSignal,
    UnifiedSignalConfig,
    LearningContext,
    LearningUpdate,
    StructuralUpdate,
    UpdateType,
    SignalSource,
    create_unified_signal,
    create_fully_integrated_signal,
)
from t4dm.learning.three_factor import ThreeFactorLearningRule, ThreeFactorSignal


class TestLearningContext:
    """Tests for LearningContext dataclass."""

    def test_context_creation_minimal(self):
        """Context can be created with minimal inputs."""
        memory_id = uuid4()
        context = LearningContext(memory_id=memory_id)

        assert context.memory_id == memory_id
        assert context.eligibility == 0.0
        assert context.neuromod_gate == 1.0
        assert context.dopamine_surprise == 0.0

    def test_context_creation_full(self):
        """Context can be created with all inputs."""
        memory_id = uuid4()
        embedding = np.random.randn(128)

        context = LearningContext(
            memory_id=memory_id,
            eligibility=0.8,
            neuromod_gate=1.5,
            dopamine_surprise=0.4,
            rpe_signed=0.3,
            ff_goodness=3.5,
            ff_threshold=2.0,
            ff_is_positive=True,
            ff_probability=0.75,
            capsule_agreement=0.6,
            novelty_score=0.8,
            activity_level=0.7,
            importance=2.0,
            embedding=embedding,
            base_lr=0.01,
        )

        assert context.eligibility == 0.8
        assert context.neuromod_gate == 1.5
        assert context.dopamine_surprise == 0.4
        assert context.ff_goodness == 3.5
        assert context.capsule_agreement == 0.6
        assert context.novelty_score == 0.8
        assert context.importance == 2.0

    def test_context_to_dict(self):
        """Context serializes to dictionary."""
        memory_id = uuid4()
        context = LearningContext(
            memory_id=memory_id,
            eligibility=0.5,
            dopamine_surprise=0.3,
        )

        result = context.to_dict()

        assert isinstance(result, dict)
        assert result["memory_id"] == str(memory_id)
        assert result["eligibility"] == 0.5
        assert result["dopamine_surprise"] == 0.3
        assert "timestamp" in result


class TestStructuralUpdate:
    """Tests for StructuralUpdate dataclass."""

    def test_structural_update_defaults(self):
        """StructuralUpdate has sensible defaults."""
        update = StructuralUpdate()

        assert update.should_add_neuron is False
        assert update.should_prune is False
        assert update.target_layer_idx == 0
        assert update.birth_probability == 0.0

    def test_structural_update_with_values(self):
        """StructuralUpdate accepts values."""
        update = StructuralUpdate(
            should_add_neuron=True,
            novelty_score=1.5,
            birth_probability=0.8,
        )

        assert update.should_add_neuron is True
        assert update.novelty_score == 1.5
        assert update.birth_probability == 0.8

    def test_structural_update_to_dict(self):
        """StructuralUpdate serializes to dictionary."""
        update = StructuralUpdate(should_add_neuron=True)

        result = update.to_dict()

        assert isinstance(result, dict)
        assert result["should_add_neuron"] is True


class TestLearningUpdate:
    """Tests for LearningUpdate dataclass."""

    def test_learning_update_defaults(self):
        """LearningUpdate has sensible defaults."""
        update = LearningUpdate()

        assert update.weight_delta == 0.0
        assert update.effective_lr == 0.0
        assert update.should_update is False
        assert update.confidence == 0.0
        assert isinstance(update.structure_delta, StructuralUpdate)

    def test_learning_update_with_values(self):
        """LearningUpdate accepts values."""
        update = LearningUpdate(
            weight_delta=0.5,
            effective_lr=0.005,
            three_factor_contrib=0.3,
            ff_contrib=0.1,
            capsule_contrib=0.1,
            update_type=UpdateType.WEIGHT,
            signal_sources=[SignalSource.THREE_FACTOR, SignalSource.FF_GOODNESS],
            should_update=True,
            confidence=0.8,
        )

        assert update.weight_delta == 0.5
        assert update.effective_lr == 0.005
        assert update.three_factor_contrib == 0.3
        assert update.update_type == UpdateType.WEIGHT
        assert len(update.signal_sources) == 2

    def test_learning_update_to_dict(self):
        """LearningUpdate serializes to dictionary."""
        update = LearningUpdate(
            weight_delta=0.5,
            update_type=UpdateType.BOTH,
            signal_sources=[SignalSource.THREE_FACTOR],
        )

        result = update.to_dict()

        assert isinstance(result, dict)
        assert result["weight_delta"] == 0.5
        assert result["update_type"] == "BOTH"
        assert "structure_delta" in result


class TestUnifiedSignalConfig:
    """Tests for UnifiedSignalConfig."""

    def test_config_defaults(self):
        """Config has sensible defaults."""
        config = UnifiedSignalConfig()

        assert config.ff_weight == 0.3
        assert config.capsule_weight == 0.3
        assert config.neurogenesis_threshold == 0.5
        assert config.enable_ff_modulation is True

    def test_config_validation_ff_weight(self):
        """Config validates ff_weight bounds."""
        with pytest.raises(AssertionError):
            UnifiedSignalConfig(ff_weight=1.5)

        with pytest.raises(AssertionError):
            UnifiedSignalConfig(ff_weight=-0.1)

    def test_config_validation_capsule_weight(self):
        """Config validates capsule_weight bounds."""
        with pytest.raises(AssertionError):
            UnifiedSignalConfig(capsule_weight=2.0)

    def test_config_validation_min_eligibility(self):
        """Config validates min_eligibility."""
        with pytest.raises(AssertionError):
            UnifiedSignalConfig(min_eligibility=-0.1)

    def test_config_validation_max_effective_lr(self):
        """Config validates max_effective_lr."""
        with pytest.raises(AssertionError):
            UnifiedSignalConfig(max_effective_lr=0)


class TestUnifiedLearningSignalBasic:
    """Basic tests for UnifiedLearningSignal."""

    @pytest.fixture
    def unified(self):
        """Create unified learning signal with defaults."""
        return UnifiedLearningSignal()

    def test_initialization_default(self, unified):
        """Signal initializes with default config."""
        assert unified.config is not None
        assert unified._three_factor is not None
        assert unified._ff_encoder is None
        assert unified._capsule_layer is None

    def test_initialization_with_config(self):
        """Signal accepts custom config."""
        config = UnifiedSignalConfig(ff_weight=0.5, capsule_weight=0.2)
        unified = UnifiedLearningSignal(config=config)

        assert unified.config.ff_weight == 0.5
        assert unified.config.capsule_weight == 0.2

    def test_initialization_with_three_factor(self):
        """Signal accepts custom three-factor rule."""
        three_factor = ThreeFactorLearningRule(
            min_eligibility_threshold=0.001,
        )
        unified = UnifiedLearningSignal(three_factor=three_factor)

        assert unified._three_factor is three_factor


class TestThreeFactorComputation:
    """Tests for three-factor computation in unified signal."""

    @pytest.fixture
    def unified(self):
        """Create unified learning signal."""
        return UnifiedLearningSignal(
            config=UnifiedSignalConfig(
                enable_ff_modulation=False,
                enable_capsule_modulation=False,
            )
        )

    def test_three_factor_basic(self, unified):
        """Three-factor computation works correctly."""
        context = LearningContext(
            memory_id=uuid4(),
            eligibility=0.8,
            neuromod_gate=1.0,
            dopamine_surprise=0.5,
            base_lr=0.01,
        )

        update = unified.compute_update(context)

        # weight_delta = eligibility * neuromod * DA * (1 + 0 + 0)
        expected = 0.8 * 1.0 * 0.5
        assert abs(update.three_factor_contrib - expected) < 1e-6
        assert abs(update.weight_delta - expected) < 1e-6

    def test_three_factor_multiplicative(self, unified):
        """Three factors multiply correctly."""
        context = LearningContext(
            memory_id=uuid4(),
            eligibility=0.5,
            neuromod_gate=2.0,
            dopamine_surprise=0.4,
            base_lr=0.01,
        )

        update = unified.compute_update(context)

        expected = 0.5 * 2.0 * 0.4
        assert abs(update.three_factor_contrib - expected) < 1e-6

    def test_three_factor_zero_eligibility(self, unified):
        """Zero eligibility produces minimal update."""
        context = LearningContext(
            memory_id=uuid4(),
            eligibility=0.0,
            neuromod_gate=2.0,
            dopamine_surprise=0.8,
            base_lr=0.01,
        )

        update = unified.compute_update(context)

        assert update.weight_delta == 0.0
        assert update.should_update is False

    def test_three_factor_zero_dopamine(self, unified):
        """Zero dopamine surprise produces minimal update."""
        context = LearningContext(
            memory_id=uuid4(),
            eligibility=0.8,
            neuromod_gate=1.5,
            dopamine_surprise=0.0,
            base_lr=0.01,
        )

        update = unified.compute_update(context)

        assert update.weight_delta == 0.0

    def test_three_factor_high_neuromod(self, unified):
        """High neuromod gate boosts update."""
        context_low = LearningContext(
            memory_id=uuid4(),
            eligibility=0.5,
            neuromod_gate=1.0,
            dopamine_surprise=0.3,
        )
        context_high = LearningContext(
            memory_id=uuid4(),
            eligibility=0.5,
            neuromod_gate=2.0,
            dopamine_surprise=0.3,
        )

        update_low = unified.compute_update(context_low)
        update_high = unified.compute_update(context_high)

        assert update_high.weight_delta > update_low.weight_delta
        assert abs(update_high.weight_delta / update_low.weight_delta - 2.0) < 1e-6


class TestCombinedModulation:
    """Tests for combined FF and capsule modulation."""

    @pytest.fixture
    def unified(self):
        """Create unified learning signal with modulation enabled."""
        return UnifiedLearningSignal(
            config=UnifiedSignalConfig(
                ff_weight=0.3,
                capsule_weight=0.3,
                enable_ff_modulation=True,
                enable_capsule_modulation=True,
                enable_neurogenesis=False,
            )
        )

    def test_ff_goodness_positive_modulation(self, unified):
        """High FF goodness increases weight delta."""
        context_low = LearningContext(
            memory_id=uuid4(),
            eligibility=0.5,
            neuromod_gate=1.0,
            dopamine_surprise=0.4,
            ff_goodness=0.0,  # Below threshold
            ff_threshold=2.0,
        )
        context_high = LearningContext(
            memory_id=uuid4(),
            eligibility=0.5,
            neuromod_gate=1.0,
            dopamine_surprise=0.4,
            ff_goodness=4.0,  # Above threshold
            ff_threshold=2.0,
        )

        update_low = unified.compute_update(context_low)
        update_high = unified.compute_update(context_high)

        # Higher goodness should give higher update (positive modulation)
        assert update_high.weight_delta > update_low.weight_delta

    def test_capsule_agreement_modulation(self, unified):
        """Low capsule agreement increases weight delta."""
        # Low agreement = need more learning
        context_low_agree = LearningContext(
            memory_id=uuid4(),
            eligibility=0.5,
            neuromod_gate=1.0,
            dopamine_surprise=0.4,
            capsule_agreement=0.2,  # Low agreement
        )
        # High agreement = stable, less learning needed
        context_high_agree = LearningContext(
            memory_id=uuid4(),
            eligibility=0.5,
            neuromod_gate=1.0,
            dopamine_surprise=0.4,
            capsule_agreement=0.9,  # High agreement
        )

        update_low = unified.compute_update(context_low_agree)
        update_high = unified.compute_update(context_high_agree)

        # Low agreement should produce larger update
        assert update_low.weight_delta > update_high.weight_delta

    def test_combined_ff_and_capsule(self, unified):
        """FF and capsule modulation combine."""
        context = LearningContext(
            memory_id=uuid4(),
            eligibility=0.5,
            neuromod_gate=1.0,
            dopamine_surprise=0.4,
            ff_goodness=3.0,
            ff_threshold=2.0,
            capsule_agreement=0.5,
        )

        update = unified.compute_update(context)

        assert update.ff_contrib != 0.0
        assert update.capsule_contrib != 0.0
        assert SignalSource.FF_GOODNESS in update.signal_sources
        assert SignalSource.CAPSULE_AGREEMENT in update.signal_sources

    def test_modulation_disabled(self):
        """Modulation can be disabled via config."""
        unified = UnifiedLearningSignal(
            config=UnifiedSignalConfig(
                enable_ff_modulation=False,
                enable_capsule_modulation=False,
            )
        )

        context = LearningContext(
            memory_id=uuid4(),
            eligibility=0.5,
            neuromod_gate=1.0,
            dopamine_surprise=0.4,
            ff_goodness=5.0,
            capsule_agreement=0.1,
        )

        update = unified.compute_update(context)

        # Should not have FF or capsule contributions
        assert update.ff_contrib == 0.0
        assert update.capsule_contrib == 0.0
        assert SignalSource.FF_GOODNESS not in update.signal_sources


# =============================================================================
# Mock Neurogenesis Manager for Testing
# =============================================================================

@dataclass
class MockNeurogenesisConfig:
    """Mock neurogenesis configuration."""
    birth_rate: float = 0.1
    survival_threshold: float = 0.1
    integration_rate: float = 0.1


class MockNeurogenesisManager:
    """Mock neurogenesis manager for testing."""

    def __init__(self, birth_rate: float = 0.1):
        self.config = MockNeurogenesisConfig(birth_rate=birth_rate)


class TestNeurogenesisIntegration:
    """Tests for neurogenesis structural updates."""

    @pytest.fixture
    def unified_with_neurogenesis(self):
        """Create unified learning signal with mock neurogenesis manager."""
        mock_neuro = MockNeurogenesisManager(birth_rate=0.5)
        unified = UnifiedLearningSignal(
            config=UnifiedSignalConfig(
                neurogenesis_threshold=0.5,
                enable_neurogenesis=True,
            ),
            neurogenesis=mock_neuro,
        )
        return unified

    def test_neurogenesis_high_novelty(self, unified_with_neurogenesis):
        """High novelty triggers neuron birth signal when manager present."""
        context = LearningContext(
            memory_id=uuid4(),
            eligibility=0.5,
            neuromod_gate=1.0,
            dopamine_surprise=0.3,
            novelty_score=1.0,  # Above threshold
            activity_level=0.5,
        )

        update = unified_with_neurogenesis.compute_update(context)

        assert update.structure_delta.should_add_neuron is True
        assert update.structure_delta.birth_probability > 0

    def test_neurogenesis_low_novelty(self, unified_with_neurogenesis):
        """Low novelty does not trigger neuron birth."""
        context = LearningContext(
            memory_id=uuid4(),
            eligibility=0.5,
            neuromod_gate=1.0,
            dopamine_surprise=0.3,
            novelty_score=0.1,  # Below threshold
            activity_level=0.5,
        )

        update = unified_with_neurogenesis.compute_update(context)

        assert update.structure_delta.should_add_neuron is False

    def test_neurogenesis_disabled(self):
        """Neurogenesis can be disabled via config."""
        mock_neuro = MockNeurogenesisManager()
        unified = UnifiedLearningSignal(
            config=UnifiedSignalConfig(enable_neurogenesis=False),
            neurogenesis=mock_neuro,
        )

        context = LearningContext(
            memory_id=uuid4(),
            novelty_score=2.0,  # Very high
        )

        update = unified.compute_update(context)

        assert update.structure_delta.should_add_neuron is False

    def test_neurogenesis_without_manager(self):
        """Without neurogenesis manager, no structural signals are produced."""
        unified = UnifiedLearningSignal(
            config=UnifiedSignalConfig(
                neurogenesis_threshold=0.5,
                enable_neurogenesis=True,
            )
            # No neurogenesis manager provided
        )

        context = LearningContext(
            memory_id=uuid4(),
            novelty_score=2.0,  # Very high
        )

        update = unified.compute_update(context)

        # Without manager, no structural signals
        assert update.structure_delta.should_add_neuron is False
        assert update.structure_delta.birth_probability == 0.0

    def test_update_type_both(self, unified_with_neurogenesis):
        """Update type is BOTH when weight and structure updates."""
        context = LearningContext(
            memory_id=uuid4(),
            eligibility=0.5,
            neuromod_gate=1.0,
            dopamine_surprise=0.5,
            novelty_score=1.0,  # Triggers neurogenesis
        )

        update = unified_with_neurogenesis.compute_update(context)

        # Has both weight update and structure update
        assert update.weight_delta > 0
        assert update.structure_delta.should_add_neuron is True
        assert update.update_type == UpdateType.BOTH

    def test_low_activity_triggers_pruning(self, unified_with_neurogenesis):
        """Low activity level triggers pruning signal."""
        context = LearningContext(
            memory_id=uuid4(),
            eligibility=0.5,
            neuromod_gate=1.0,
            dopamine_surprise=0.3,
            novelty_score=0.0,
            activity_level=0.01,  # Below survival threshold
        )

        update = unified_with_neurogenesis.compute_update(context)

        assert update.structure_delta.should_prune is True


class TestWeightUpdateBounds:
    """Tests for weight update bounds and clamping."""

    def test_max_effective_lr_enforced(self):
        """Weight delta is clamped to max_effective_lr."""
        unified = UnifiedLearningSignal(
            config=UnifiedSignalConfig(
                max_effective_lr=1.0,
                enable_ff_modulation=True,
            )
        )

        # Context designed to produce very large update
        context = LearningContext(
            memory_id=uuid4(),
            eligibility=1.0,
            neuromod_gate=10.0,
            dopamine_surprise=1.0,
            ff_goodness=10.0,
            ff_threshold=1.0,
        )

        update = unified.compute_update(context)

        assert update.weight_delta <= 1.0

    def test_min_update_threshold(self):
        """Updates below threshold are skipped."""
        unified = UnifiedLearningSignal(
            config=UnifiedSignalConfig(
                min_update_threshold=0.01,
                enable_ff_modulation=False,
                enable_capsule_modulation=False,
            )
        )

        # Tiny update
        context = LearningContext(
            memory_id=uuid4(),
            eligibility=0.001,
            neuromod_gate=1.0,
            dopamine_surprise=0.001,
        )

        update = unified.compute_update(context)

        assert update.should_update is False

    def test_importance_protection(self):
        """High importance reduces update magnitude."""
        unified = UnifiedLearningSignal(
            config=UnifiedSignalConfig(
                importance_protection=1.0,
                enable_ff_modulation=False,
                enable_capsule_modulation=False,
            )
        )

        context_low = LearningContext(
            memory_id=uuid4(),
            eligibility=0.5,
            neuromod_gate=1.0,
            dopamine_surprise=0.5,
            importance=0.0,
        )
        context_high = LearningContext(
            memory_id=uuid4(),
            eligibility=0.5,
            neuromod_gate=1.0,
            dopamine_surprise=0.5,
            importance=5.0,
        )

        update_low = unified.compute_update(context_low)
        update_high = unified.compute_update(context_high)

        # High importance should reduce update
        assert update_high.weight_delta < update_low.weight_delta

    def test_nan_handling(self):
        """NaN values are handled gracefully."""
        unified = UnifiedLearningSignal(
            config=UnifiedSignalConfig(
                enable_ff_modulation=False,
                enable_capsule_modulation=False,
            )
        )

        context = LearningContext(
            memory_id=uuid4(),
            eligibility=float("nan"),
            neuromod_gate=1.0,
            dopamine_surprise=0.5,
        )

        update = unified.compute_update(context)

        # Should not produce NaN
        assert np.isfinite(update.weight_delta)

    def test_effective_lr_computation(self):
        """Effective LR is base_lr * weight_delta."""
        unified = UnifiedLearningSignal(
            config=UnifiedSignalConfig(
                enable_ff_modulation=False,
                enable_capsule_modulation=False,
            )
        )

        context = LearningContext(
            memory_id=uuid4(),
            eligibility=0.5,
            neuromod_gate=1.0,
            dopamine_surprise=0.4,
            base_lr=0.01,
        )

        update = unified.compute_update(context)

        expected_lr = 0.01 * update.weight_delta
        assert abs(update.effective_lr - expected_lr) < 1e-9


class TestSignalTracking:
    """Tests for signal source tracking and confidence."""

    @pytest.fixture
    def unified(self):
        """Create unified learning signal."""
        return UnifiedLearningSignal()

    def test_signal_sources_tracked(self, unified):
        """Signal sources are tracked."""
        context = LearningContext(
            memory_id=uuid4(),
            eligibility=0.5,
            neuromod_gate=1.0,
            dopamine_surprise=0.4,
            ff_goodness=3.0,
            ff_threshold=2.0,
            capsule_agreement=0.5,
        )

        update = unified.compute_update(context)

        assert SignalSource.THREE_FACTOR in update.signal_sources

    def test_confidence_increases_with_signals(self, unified):
        """Confidence increases with more aligned signals."""
        context_minimal = LearningContext(
            memory_id=uuid4(),
            eligibility=0.5,
            neuromod_gate=1.0,
            dopamine_surprise=0.4,
        )
        context_full = LearningContext(
            memory_id=uuid4(),
            eligibility=0.5,
            neuromod_gate=1.0,
            dopamine_surprise=0.4,
            ff_goodness=3.0,
            ff_threshold=2.0,
            capsule_agreement=0.3,  # Low agreement = high contribution
        )

        update_minimal = unified.compute_update(context_minimal)
        update_full = unified.compute_update(context_full)

        assert update_full.confidence >= update_minimal.confidence


class TestHistoryAndStatistics:
    """Tests for history tracking and statistics."""

    @pytest.fixture
    def unified(self):
        """Create unified learning signal."""
        return UnifiedLearningSignal(
            config=UnifiedSignalConfig(history_size=100)
        )

    def test_history_tracking(self, unified):
        """Updates are tracked in history."""
        for _ in range(10):
            context = LearningContext(
                memory_id=uuid4(),
                eligibility=0.5,
                neuromod_gate=1.0,
                dopamine_surprise=0.3,
            )
            unified.compute_update(context)

        assert len(unified._update_history) == 10

    def test_history_bounded(self, unified):
        """History is bounded to configured size."""
        for _ in range(150):
            context = LearningContext(
                memory_id=uuid4(),
                eligibility=0.5,
                neuromod_gate=1.0,
                dopamine_surprise=0.3,
            )
            unified.compute_update(context)

        assert len(unified._update_history) <= 100

    def test_get_recent_updates(self, unified):
        """Can retrieve recent updates."""
        for _ in range(10):
            context = LearningContext(
                memory_id=uuid4(),
                eligibility=0.5,
                neuromod_gate=1.0,
                dopamine_surprise=0.3,
            )
            unified.compute_update(context)

        recent = unified.get_recent_updates(n=5)

        assert len(recent) == 5
        assert all(isinstance(u, LearningUpdate) for u in recent)

    def test_get_statistics(self, unified):
        """Statistics are computed correctly."""
        for i in range(10):
            context = LearningContext(
                memory_id=uuid4(),
                eligibility=0.5 + i * 0.05,
                neuromod_gate=1.0,
                dopamine_surprise=0.3,
            )
            unified.compute_update(context)

        stats = unified.get_statistics()

        assert stats["total_updates"] == 10
        assert "components" in stats
        assert "recent" in stats
        assert stats["recent"]["mean_weight_delta"] > 0

    def test_clear_history(self, unified):
        """History can be cleared."""
        for _ in range(10):
            context = LearningContext(
                memory_id=uuid4(),
                eligibility=0.5,
                neuromod_gate=1.0,
                dopamine_surprise=0.3,
            )
            unified.compute_update(context)

        assert len(unified._update_history) == 10

        unified.clear_history()

        assert len(unified._update_history) == 0

    def test_reset_statistics(self, unified):
        """Statistics can be reset."""
        for _ in range(10):
            context = LearningContext(
                memory_id=uuid4(),
                eligibility=0.5,
                neuromod_gate=1.0,
                dopamine_surprise=0.3,
            )
            unified.compute_update(context)

        assert unified._total_updates == 10

        unified.reset_statistics()

        assert unified._total_updates == 0


class TestConvenienceMethods:
    """Tests for convenience methods."""

    @pytest.fixture
    def unified(self):
        """Create unified learning signal."""
        return UnifiedLearningSignal()

    def test_compute_from_three_factor_signal(self, unified):
        """Can compute from pre-existing ThreeFactorSignal."""
        memory_id = uuid4()
        signal = ThreeFactorSignal(
            memory_id=memory_id,
            eligibility=0.8,
            neuromod_gate=1.2,
            ach_mode_factor=1.5,
            ne_arousal_factor=1.0,
            serotonin_mood_factor=0.9,
            dopamine_surprise=0.5,
            rpe_raw=0.3,
            effective_lr_multiplier=0.48,
        )

        update = unified.compute_from_three_factor_signal(
            signal=signal,
            ff_goodness=2.5,
            base_lr=0.01,
        )

        assert isinstance(update, LearningUpdate)
        assert update.three_factor_contrib > 0

    def test_batch_compute(self, unified):
        """Can batch compute multiple contexts."""
        contexts = [
            LearningContext(
                memory_id=uuid4(),
                eligibility=0.5 + i * 0.1,
                neuromod_gate=1.0,
                dopamine_surprise=0.3,
            )
            for i in range(5)
        ]

        updates = unified.batch_compute(contexts)

        assert len(updates) == 5
        assert all(isinstance(u, LearningUpdate) for u in updates)


class TestFactoryFunctions:
    """Tests for factory functions."""

    def test_create_unified_signal(self):
        """create_unified_signal creates configured signal."""
        unified = create_unified_signal(
            ff_weight=0.4,
            capsule_weight=0.2,
            max_effective_lr=2.0,
        )

        assert unified.config.ff_weight == 0.4
        assert unified.config.capsule_weight == 0.2
        assert unified.config.max_effective_lr == 2.0

    def test_create_fully_integrated_signal_minimal(self):
        """create_fully_integrated_signal works with no components."""
        unified = create_fully_integrated_signal()

        assert unified._three_factor is not None
        assert unified._ff_encoder is None

    def test_create_fully_integrated_signal_with_config(self):
        """create_fully_integrated_signal accepts config."""
        config = UnifiedSignalConfig(ff_weight=0.5)
        unified = create_fully_integrated_signal(config=config)

        assert unified.config.ff_weight == 0.5


class TestComponentSetters:
    """Tests for component setter methods."""

    @pytest.fixture
    def unified(self):
        """Create unified learning signal."""
        return UnifiedLearningSignal()

    def test_set_ff_encoder(self, unified):
        """FF encoder can be set after initialization."""
        assert unified._ff_encoder is None

        # Mock FF encoder (we just need an object)
        class MockFF:
            pass

        unified.set_ff_encoder(MockFF())

        assert unified._ff_encoder is not None

    def test_set_capsule_layer(self, unified):
        """Capsule layer can be set after initialization."""
        assert unified._capsule_layer is None

        class MockCapsule:
            pass

        unified.set_capsule_layer(MockCapsule())

        assert unified._capsule_layer is not None

    def test_set_neurogenesis(self, unified):
        """Neurogenesis manager can be set after initialization."""
        assert unified._neurogenesis is None

        class MockNeurogenesis:
            pass

        unified.set_neurogenesis(MockNeurogenesis())

        assert unified._neurogenesis is not None

    def test_set_pose_learner(self, unified):
        """Pose learner can be set after initialization."""
        assert unified._pose_learner is None

        class MockPose:
            pass

        unified.set_pose_learner(MockPose())

        assert unified._pose_learner is not None


class TestEdgeCases:
    """Edge case and boundary tests."""

    def test_all_zeros(self):
        """Handles all-zero context gracefully."""
        unified = UnifiedLearningSignal()
        context = LearningContext(
            memory_id=uuid4(),
            eligibility=0.0,
            neuromod_gate=0.0,
            dopamine_surprise=0.0,
        )

        update = unified.compute_update(context)

        assert update.weight_delta == 0.0
        assert update.should_update is False

    def test_very_large_values(self):
        """Handles very large values without overflow."""
        unified = UnifiedLearningSignal(
            config=UnifiedSignalConfig(max_effective_lr=1.0)
        )
        context = LearningContext(
            memory_id=uuid4(),
            eligibility=1000.0,
            neuromod_gate=1000.0,
            dopamine_surprise=1000.0,
        )

        update = unified.compute_update(context)

        assert np.isfinite(update.weight_delta)
        assert update.weight_delta <= 1.0

    def test_negative_ff_goodness(self):
        """Handles negative FF goodness."""
        unified = UnifiedLearningSignal()
        context = LearningContext(
            memory_id=uuid4(),
            eligibility=0.5,
            neuromod_gate=1.0,
            dopamine_surprise=0.3,
            ff_goodness=-1.0,  # Negative
            ff_threshold=2.0,
        )

        update = unified.compute_update(context)

        assert np.isfinite(update.weight_delta)

    def test_update_type_structure_only(self):
        """UpdateType is STRUCTURE when only neurogenesis triggers."""
        mock_neuro = MockNeurogenesisManager(birth_rate=1.0)
        unified = UnifiedLearningSignal(
            config=UnifiedSignalConfig(
                neurogenesis_threshold=0.1,
                min_update_threshold=1.0,  # Very high to skip weight update
                enable_neurogenesis=True,
            ),
            neurogenesis=mock_neuro,
        )
        context = LearningContext(
            memory_id=uuid4(),
            eligibility=0.0,  # No weight update
            neuromod_gate=1.0,
            dopamine_surprise=0.0,
            novelty_score=0.5,  # Triggers neurogenesis
        )

        update = unified.compute_update(context)

        assert update.weight_delta < 1.0
        assert update.structure_delta.should_add_neuron is True
        assert update.update_type == UpdateType.STRUCTURE

    def test_update_type_weight_only(self):
        """UpdateType is WEIGHT when only weight update occurs."""
        unified = UnifiedLearningSignal(
            config=UnifiedSignalConfig(
                enable_neurogenesis=False,
            )
        )
        context = LearningContext(
            memory_id=uuid4(),
            eligibility=0.5,
            neuromod_gate=1.0,
            dopamine_surprise=0.4,
        )

        update = unified.compute_update(context)

        assert update.weight_delta > 0
        assert update.structure_delta.should_add_neuron is False
        assert update.update_type == UpdateType.WEIGHT
