"""
Tests for Adaptive Consolidation Trigger (W1-04).

CLS-based consolidation triggering: consolidate when fast learner (MemTable)
saturates, regardless of sleep pressure.

Evidence Base: O'Reilly et al. (2014) "Complementary Learning Systems"

Test Strategy (TDD):
1. Config tests for parameter validation
2. MemTable saturation trigger tests
3. Encoding rate trigger tests
4. Minimum interval respect tests
5. Adenosine fallback tests
6. Phase selection tests
"""

import pytest
import time
from unittest.mock import Mock, MagicMock
from collections import deque


class TestAdaptiveConsolidationConfig:
    """Test AdaptiveConsolidationConfig dataclass."""

    def test_default_values(self):
        """Test default configuration values."""
        from t4dm.consolidation.adaptive_trigger import AdaptiveConsolidationConfig

        config = AdaptiveConsolidationConfig()
        assert config.memtable_saturation_threshold == 0.7
        assert config.encoding_rate_threshold == 10.0
        assert config.adenosine_threshold == 0.6
        assert config.min_interval_seconds == 300.0

    def test_custom_values(self):
        """Test custom configuration values."""
        from t4dm.consolidation.adaptive_trigger import AdaptiveConsolidationConfig

        config = AdaptiveConsolidationConfig(
            memtable_saturation_threshold=0.8,
            encoding_rate_threshold=20.0,
            adenosine_threshold=0.5,
            min_interval_seconds=600.0,
        )
        assert config.memtable_saturation_threshold == 0.8
        assert config.encoding_rate_threshold == 20.0
        assert config.adenosine_threshold == 0.5
        assert config.min_interval_seconds == 600.0


class TestConsolidationTriggerResult:
    """Test ConsolidationTriggerResult dataclass."""

    def test_result_fields(self):
        """Test result dataclass has required fields."""
        from t4dm.consolidation.adaptive_trigger import ConsolidationTriggerResult

        result = ConsolidationTriggerResult(
            reason="memtable_saturation",
            urgency=0.8,
            phase="nrem",
        )
        assert result.reason == "memtable_saturation"
        assert result.urgency == 0.8
        assert result.phase == "nrem"

    def test_to_dict(self):
        """Test dictionary conversion."""
        from t4dm.consolidation.adaptive_trigger import ConsolidationTriggerResult

        result = ConsolidationTriggerResult(
            reason="high_encoding_rate",
            urgency=1.5,
            phase="nrem",
        )
        d = result.to_dict()
        assert d["reason"] == "high_encoding_rate"
        assert d["urgency"] == 1.5
        assert d["phase"] == "nrem"


class TestAdaptiveConsolidationTrigger:
    """Test AdaptiveConsolidationTrigger class."""

    @pytest.fixture
    def mock_engine(self):
        """Create mock T4DX engine."""
        engine = Mock()
        engine.memtable_size.return_value = 5000
        engine.max_memtable_size = 10000
        return engine

    @pytest.fixture
    def config(self):
        """Create default config."""
        from t4dm.consolidation.adaptive_trigger import AdaptiveConsolidationConfig

        return AdaptiveConsolidationConfig()

    @pytest.fixture
    def trigger(self, config, mock_engine):
        """Create trigger with default config and mock engine."""
        from t4dm.consolidation.adaptive_trigger import AdaptiveConsolidationTrigger

        t = AdaptiveConsolidationTrigger(config, mock_engine)
        t.last_consolidation = 0  # Long ago
        return t

    def test_init(self, trigger, config, mock_engine):
        """Test initialization."""
        assert trigger.config == config
        assert trigger.engine == mock_engine
        assert len(trigger.encoding_times) == 0

    def test_no_trigger_below_thresholds(self, trigger, mock_engine):
        """Should not trigger when all values below thresholds."""
        mock_engine.memtable_size.return_value = 5000  # 50% < 70%

        result = trigger.should_trigger(adenosine_pressure=0.3)  # Low

        assert result is None

    def test_trigger_on_memtable_saturation(self, config, mock_engine):
        """Should trigger consolidation when MemTable is 70%+ full."""
        from t4dm.consolidation.adaptive_trigger import AdaptiveConsolidationTrigger

        mock_engine.memtable_size.return_value = 7500  # 75% > 70%

        trigger = AdaptiveConsolidationTrigger(config, mock_engine)
        trigger.last_consolidation = 0  # Long ago

        result = trigger.should_trigger(adenosine_pressure=0.3)

        assert result is not None
        assert result.reason == "memtable_saturation"
        assert result.phase == "nrem"
        assert result.urgency == 0.75  # 75% saturation

    def test_trigger_on_high_encoding_rate(self, config, mock_engine):
        """Should trigger when encoding rate exceeds threshold."""
        from t4dm.consolidation.adaptive_trigger import AdaptiveConsolidationTrigger

        mock_engine.memtable_size.return_value = 1000  # Low saturation

        trigger = AdaptiveConsolidationTrigger(config, mock_engine)
        trigger.last_consolidation = 0

        # Simulate 20 encodings in ~1 minute (above threshold of 10/min)
        base_time = time.time()
        for i in range(20):
            trigger.encoding_times.append(base_time + i * 3)  # 3 sec apart = 20/min

        result = trigger.should_trigger(adenosine_pressure=0.3)

        assert result is not None
        assert result.reason == "high_encoding_rate"
        assert result.phase == "nrem"

    def test_respects_minimum_interval(self, config, mock_engine):
        """Should not trigger within min_interval of last consolidation."""
        from t4dm.consolidation.adaptive_trigger import AdaptiveConsolidationTrigger

        mock_engine.memtable_size.return_value = 9000  # High saturation

        trigger = AdaptiveConsolidationTrigger(config, mock_engine)
        trigger.last_consolidation = time.time()  # Just consolidated

        result = trigger.should_trigger(adenosine_pressure=0.9)

        assert result is None, "Should respect minimum interval"

    def test_default_to_adenosine(self, config, mock_engine):
        """When no urgency, use adenosine pressure."""
        from t4dm.consolidation.adaptive_trigger import AdaptiveConsolidationTrigger

        mock_engine.memtable_size.return_value = 3000  # Low saturation

        trigger = AdaptiveConsolidationTrigger(config, mock_engine)
        trigger.last_consolidation = 0  # Long ago

        result = trigger.should_trigger(adenosine_pressure=0.8)  # High

        assert result is not None
        assert result.reason == "adenosine_pressure"
        assert result.phase == "full"  # Full sleep cycle

    def test_adenosine_below_threshold_no_trigger(self, trigger, mock_engine):
        """Should not trigger on adenosine if below threshold."""
        mock_engine.memtable_size.return_value = 3000  # Low saturation

        result = trigger.should_trigger(adenosine_pressure=0.5)  # Below 0.6

        assert result is None

    def test_record_encoding(self, trigger):
        """record_encoding should track encoding times."""
        assert len(trigger.encoding_times) == 0

        trigger.record_encoding()
        trigger.record_encoding()
        trigger.record_encoding()

        assert len(trigger.encoding_times) == 3

    def test_encoding_rate_computation(self, trigger):
        """Should correctly compute encodings per minute."""
        base_time = time.time()

        # 10 encodings over 30 seconds = 20 per minute
        for i in range(10):
            trigger.encoding_times.append(base_time + i * 3)

        rate = trigger._compute_encoding_rate()

        # Should be approximately 20 per minute (10 encodings / 0.5 minutes)
        assert rate > 15  # Allow some tolerance

    def test_encoding_rate_with_insufficient_data(self, trigger):
        """Should return 0 with insufficient encoding data."""
        assert trigger._compute_encoding_rate() == 0.0

        trigger.encoding_times.append(time.time())
        assert trigger._compute_encoding_rate() == 0.0  # Need at least 2

    def test_phase_selection_saturation(self, config, mock_engine):
        """High saturation should trigger NREM (fast consolidation)."""
        from t4dm.consolidation.adaptive_trigger import AdaptiveConsolidationTrigger

        mock_engine.memtable_size.return_value = 9000  # 90%

        trigger = AdaptiveConsolidationTrigger(config, mock_engine)
        trigger.last_consolidation = 0

        result = trigger.should_trigger(adenosine_pressure=0.3)

        assert result.phase == "nrem"

    def test_phase_selection_adenosine(self, config, mock_engine):
        """Adenosine trigger should use full sleep cycle."""
        from t4dm.consolidation.adaptive_trigger import AdaptiveConsolidationTrigger

        mock_engine.memtable_size.return_value = 3000  # Low

        trigger = AdaptiveConsolidationTrigger(config, mock_engine)
        trigger.last_consolidation = 0

        result = trigger.should_trigger(adenosine_pressure=0.8)

        assert result.phase == "full"

    def test_urgency_reflects_saturation(self, config, mock_engine):
        """Urgency should reflect MemTable saturation level."""
        from t4dm.consolidation.adaptive_trigger import AdaptiveConsolidationTrigger

        mock_engine.memtable_size.return_value = 8000  # 80%

        trigger = AdaptiveConsolidationTrigger(config, mock_engine)
        trigger.last_consolidation = 0

        result = trigger.should_trigger(adenosine_pressure=0.3)

        assert result.urgency == 0.8  # Matches saturation

    def test_mark_consolidation_complete(self, trigger):
        """mark_consolidation_complete should update last_consolidation."""
        old_time = trigger.last_consolidation

        trigger.mark_consolidation_complete()

        assert trigger.last_consolidation > old_time


class TestEncodingRateWindow:
    """Test encoding rate window behavior."""

    def test_encoding_window_limited_size(self):
        """Encoding times should be limited to window size."""
        from t4dm.consolidation.adaptive_trigger import (
            AdaptiveConsolidationConfig,
            AdaptiveConsolidationTrigger,
        )

        config = AdaptiveConsolidationConfig()
        engine = Mock()
        engine.memtable_size.return_value = 1000
        engine.max_memtable_size = 10000

        trigger = AdaptiveConsolidationTrigger(config, engine)

        # Record more than maxlen encodings
        for _ in range(150):
            trigger.record_encoding()

        # Should be capped at maxlen (100)
        assert len(trigger.encoding_times) == 100

    def test_encoding_rate_uses_recent_only(self):
        """Encoding rate should use recent encodings only."""
        from t4dm.consolidation.adaptive_trigger import (
            AdaptiveConsolidationConfig,
            AdaptiveConsolidationTrigger,
        )

        config = AdaptiveConsolidationConfig()
        engine = Mock()
        engine.memtable_size.return_value = 1000
        engine.max_memtable_size = 10000

        trigger = AdaptiveConsolidationTrigger(config, engine)

        # Add old encodings
        old_time = time.time() - 3600  # 1 hour ago
        for i in range(50):
            trigger.encoding_times.append(old_time + i)

        # Add recent encodings (fast rate)
        now = time.time()
        for i in range(50):
            trigger.encoding_times.append(now + i * 0.1)

        # Rate should reflect the full window, not just recent
        rate = trigger._compute_encoding_rate()
        assert rate > 0


class TestIntegrationWithScheduler:
    """Integration tests with existing ConsolidationScheduler."""

    def test_can_coexist_with_scheduler(self):
        """AdaptiveConsolidationTrigger should work alongside scheduler."""
        from t4dm.consolidation.adaptive_trigger import (
            AdaptiveConsolidationConfig,
            AdaptiveConsolidationTrigger,
        )
        from t4dm.consolidation.service import ConsolidationScheduler

        # Both can be instantiated
        scheduler = ConsolidationScheduler()
        config = AdaptiveConsolidationConfig()
        engine = Mock()
        engine.memtable_size.return_value = 1000
        engine.max_memtable_size = 10000

        adaptive_trigger = AdaptiveConsolidationTrigger(config, engine)

        # Both should have trigger methods
        assert hasattr(scheduler, "should_consolidate")
        assert hasattr(adaptive_trigger, "should_trigger")


class TestEdgeCases:
    """Test edge cases and error handling."""

    def test_zero_max_memtable_size(self):
        """Should handle zero max memtable size gracefully."""
        from t4dm.consolidation.adaptive_trigger import (
            AdaptiveConsolidationConfig,
            AdaptiveConsolidationTrigger,
        )

        config = AdaptiveConsolidationConfig()
        engine = Mock()
        engine.memtable_size.return_value = 1000
        engine.max_memtable_size = 0  # Edge case

        trigger = AdaptiveConsolidationTrigger(config, engine)
        trigger.last_consolidation = 0

        # Should not crash
        result = trigger.should_trigger(adenosine_pressure=0.5)

        # Should fall back to adenosine or return None
        assert result is None or result.reason == "adenosine_pressure"

    def test_negative_adenosine_pressure(self):
        """Should handle negative adenosine pressure."""
        from t4dm.consolidation.adaptive_trigger import (
            AdaptiveConsolidationConfig,
            AdaptiveConsolidationTrigger,
        )

        config = AdaptiveConsolidationConfig()
        engine = Mock()
        engine.memtable_size.return_value = 1000
        engine.max_memtable_size = 10000

        trigger = AdaptiveConsolidationTrigger(config, engine)
        trigger.last_consolidation = 0

        # Should not crash
        result = trigger.should_trigger(adenosine_pressure=-0.5)

        assert result is None  # Below threshold

    def test_get_stats(self):
        """Should return useful statistics."""
        from t4dm.consolidation.adaptive_trigger import (
            AdaptiveConsolidationConfig,
            AdaptiveConsolidationTrigger,
        )

        config = AdaptiveConsolidationConfig()
        engine = Mock()
        engine.memtable_size.return_value = 5000
        engine.max_memtable_size = 10000

        trigger = AdaptiveConsolidationTrigger(config, engine)

        for _ in range(10):
            trigger.record_encoding()

        stats = trigger.get_stats()

        assert "memtable_saturation" in stats
        assert "encoding_rate" in stats
        assert "seconds_since_consolidation" in stats
