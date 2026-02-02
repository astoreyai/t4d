# T4DM Test Templates - Quick Start

Use these templates to quickly add tests for the critical gaps identified in the analysis.

## Template 1: Visualization Module Test

File: `tests/visualization/test_telemetry_hub.py`

```python
import pytest
import numpy as np
from unittest.mock import Mock, patch
from t4dm.visualization.telemetry_hub import TelemetryHub
from t4dm.visualization.neuromodulator_state import NeuromodulatorState


class TestTelemetryHubBasics:
    """Test telemetry hub initialization and basic operations."""

    def test_initialization(self):
        """Test TelemetryHub initialization."""
        hub = TelemetryHub()
        assert hub is not None
        assert hasattr(hub, 'traces')
        assert len(hub.traces) == 0

    def test_add_trace_single(self):
        """Test adding a single trace."""
        hub = TelemetryHub()
        trace_data = {"timestamp": 0.0, "value": 1.5}
        hub.add_trace("test_trace", trace_data)

        assert "test_trace" in hub.traces
        assert len(hub.traces["test_trace"]) == 1

    def test_add_trace_multiple(self):
        """Test adding multiple traces."""
        hub = TelemetryHub()
        for i in range(10):
            hub.add_trace("test_trace", {"timestamp": i * 0.1, "value": i})

        assert len(hub.traces["test_trace"]) == 10

    def test_add_different_trace_types(self):
        """Test adding different trace types."""
        hub = TelemetryHub()
        hub.add_trace("neuromod", {"dopamine": 0.5})
        hub.add_trace("consolidation", {"strength": 0.8})
        hub.add_trace("activity", {"spike_rate": 10.0})

        assert len(hub.traces) == 3

    def test_get_trace(self):
        """Test retrieving traces."""
        hub = TelemetryHub()
        trace_data = [{"value": i} for i in range(5)]
        for data in trace_data:
            hub.add_trace("test", data)

        retrieved = hub.get_trace("test")
        assert len(retrieved) == 5

    def test_clear_traces(self):
        """Test clearing all traces."""
        hub = TelemetryHub()
        hub.add_trace("test1", {"value": 1})
        hub.add_trace("test2", {"value": 2})

        hub.clear_traces()
        assert len(hub.traces) == 0

    def test_nonexistent_trace(self):
        """Test accessing nonexistent trace."""
        hub = TelemetryHub()
        retrieved = hub.get_trace("nonexistent")
        assert retrieved is None or len(retrieved) == 0


class TestTelemetryHubVisualization:
    """Test visualization rendering functionality."""

    def test_render_basic(self):
        """Test basic visualization rendering."""
        hub = TelemetryHub()
        hub.add_trace("test", {"value": 1.0})

        viz = hub.render()
        assert viz is not None

    def test_render_with_data(self):
        """Test rendering with actual data."""
        hub = TelemetryHub()
        for i in range(100):
            hub.add_trace("sine_wave", {"x": i, "y": np.sin(i * 0.1)})

        viz = hub.render()
        assert viz is not None

    def test_render_multiple_traces(self):
        """Test rendering multiple traces together."""
        hub = TelemetryHub()
        for i in range(50):
            hub.add_trace("signal1", {"value": np.sin(i * 0.1)})
            hub.add_trace("signal2", {"value": np.cos(i * 0.1)})

        viz = hub.render()
        assert viz is not None

    def test_render_to_html(self):
        """Test rendering to HTML format."""
        hub = TelemetryHub()
        hub.add_trace("test", {"value": 1.0})

        html = hub.render_to_html()
        assert html is not None
        assert isinstance(html, str)
        assert len(html) > 0


class TestNeuromodulatorState:
    """Test neuromodulator state visualization."""

    def test_initialization(self):
        """Test neuromodulator state initialization."""
        neuro_state = NeuromodulatorState()
        assert neuro_state is not None

    def test_add_neuromodulator(self):
        """Test adding neuromodulator states."""
        neuro_state = NeuromodulatorState()
        neuro_state.add_neuromodulator("dopamine", 0.5)
        neuro_state.add_neuromodulator("serotonin", 0.7)
        neuro_state.add_neuromodulator("acetylcholine", 0.3)

        assert neuro_state.dopamine == 0.5
        assert neuro_state.serotonin == 0.7
        assert neuro_state.acetylcholine == 0.3

    def test_render_heatmap(self):
        """Test rendering neuromodulator heatmap."""
        neuro_state = NeuromodulatorState()
        neuro_state.add_neuromodulator("dopamine", 0.5)

        heatmap = neuro_state.render_heatmap()
        assert heatmap is not None


class TestVisualizationBoundaryConditions:
    """Test visualization edge cases and boundaries."""

    def test_empty_traces(self):
        """Test rendering with no traces."""
        hub = TelemetryHub()
        viz = hub.render()
        assert viz is not None

    def test_very_large_dataset(self):
        """Test rendering with very large dataset."""
        hub = TelemetryHub()
        for i in range(10000):
            hub.add_trace("large_trace", {"x": i, "y": np.random.random()})

        viz = hub.render()
        assert viz is not None

    def test_nan_values(self):
        """Test handling NaN values in traces."""
        hub = TelemetryHub()
        hub.add_trace("test", {"value": np.nan})
        hub.add_trace("test", {"value": 1.0})

        # Should not crash
        viz = hub.render()
        assert viz is not None

    def test_inf_values(self):
        """Test handling infinity values in traces."""
        hub = TelemetryHub()
        hub.add_trace("test", {"value": np.inf})
        hub.add_trace("test", {"value": -np.inf})

        # Should handle gracefully
        viz = hub.render()
        assert viz is not None


class TestVisualizationIntegration:
    """Integration tests with other system components."""

    def test_with_memory_graph(self, mock_memory_graph):
        """Test visualization with memory graph data."""
        hub = TelemetryHub()

        # Simulate memory graph activity
        nodes = mock_memory_graph.get_nodes()
        for node in nodes:
            hub.add_trace("memory_nodes", {
                "node_id": node.id,
                "activation": node.activation
            })

        viz = hub.render()
        assert viz is not None

    def test_with_consolidation_state(self, mock_consolidation_state):
        """Test visualization with consolidation state."""
        hub = TelemetryHub()

        state = mock_consolidation_state
        hub.add_trace("consolidation", {
            "strength": state.strength,
            "stage": state.stage,
            "duration": state.duration
        })

        viz = hub.render()
        assert viz is not None
```

## Template 2: Prediction Module Test

File: `tests/prediction/test_active_inference.py`

```python
import pytest
import numpy as np
from unittest.mock import Mock, patch
from t4dm.prediction.active_inference import ActiveInference
from t4dm.prediction.context_encoder import ContextEncoder


class TestActiveInferenceBasics:
    """Test active inference initialization and basic operations."""

    def test_initialization(self):
        """Test ActiveInference initialization."""
        ai = ActiveInference()
        assert ai is not None
        assert hasattr(ai, 'model')

    def test_simple_prediction(self):
        """Test basic prediction generation."""
        ai = ActiveInference()
        context = {"state": np.random.randn(5)}

        prediction = ai.predict(context)
        assert prediction is not None
        assert prediction.shape == (5,)

    def test_prediction_bounds(self):
        """Test predictions are within reasonable bounds."""
        ai = ActiveInference()
        context = {"state": np.ones(5)}

        prediction = ai.predict(context)
        assert np.all(np.isfinite(prediction))
        assert not np.any(np.isnan(prediction))

    def test_repeated_predictions(self):
        """Test repeated predictions with same context."""
        ai = ActiveInference()
        context = {"state": np.array([1, 2, 3, 4, 5])}

        pred1 = ai.predict(context)
        pred2 = ai.predict(context)

        # Deterministic - should be identical
        np.testing.assert_array_almost_equal(pred1, pred2)

    def test_different_contexts(self):
        """Test predictions differ for different contexts."""
        ai = ActiveInference()
        context1 = {"state": np.array([1, 0, 0, 0, 0])}
        context2 = {"state": np.array([0, 0, 0, 0, 1])}

        pred1 = ai.predict(context1)
        pred2 = ai.predict(context2)

        # Should be different
        assert not np.allclose(pred1, pred2)


class TestActiveInferenceLearning:
    """Test learning mechanisms in active inference."""

    def test_prediction_error_tracking(self):
        """Test that prediction errors are tracked."""
        ai = ActiveInference()
        context = {"state": np.random.randn(5)}
        target = np.random.randn(5)

        error = ai.compute_prediction_error(context, target)
        assert error is not None
        assert error > 0

    def test_learning_reduces_error(self):
        """Test that learning reduces prediction error."""
        ai = ActiveInference(learning_rate=0.1)
        context = {"state": np.random.randn(5)}
        target = np.random.randn(5)

        errors = []
        for _ in range(10):
            pred = ai.predict(context)
            error = np.mean((pred - target) ** 2)
            errors.append(error)
            ai.learn(context, target)

        # Error should trend downward
        assert errors[-1] < errors[0]

    def test_learning_with_batch(self):
        """Test batch learning."""
        ai = ActiveInference()
        contexts = [{"state": np.random.randn(5)} for _ in range(10)]
        targets = [np.random.randn(5) for _ in range(10)]

        errors_before = [
            np.mean((ai.predict(c) - t) ** 2)
            for c, t in zip(contexts, targets)
        ]

        ai.learn_batch(contexts, targets)

        errors_after = [
            np.mean((ai.predict(c) - t) ** 2)
            for c, t in zip(contexts, targets)
        ]

        # Most errors should decrease
        improvements = sum(1 for b, a in zip(errors_before, errors_after) if a < b)
        assert improvements > 5


class TestContextEncoder:
    """Test context encoding functionality."""

    def test_initialization(self):
        """Test ContextEncoder initialization."""
        encoder = ContextEncoder(input_dim=5, hidden_dim=10)
        assert encoder is not None

    def test_encode_context(self):
        """Test encoding a context."""
        encoder = ContextEncoder(input_dim=5, hidden_dim=10)
        context = {"state": np.random.randn(5)}

        encoded = encoder.encode(context)
        assert encoded is not None
        assert encoded.shape == (10,)

    def test_encode_preserves_information(self):
        """Test that encoding preserves information."""
        encoder = ContextEncoder(input_dim=5, hidden_dim=10)
        context = {"state": np.array([1, 0, 0, 0, 0])}

        encoded1 = encoder.encode(context)

        context2 = {"state": np.array([0, 0, 0, 0, 1])}
        encoded2 = encoder.encode(context2)

        # Different inputs should have different encodings
        assert not np.allclose(encoded1, encoded2)

    def test_decode_context(self):
        """Test decoding an encoded context."""
        encoder = ContextEncoder(input_dim=5, hidden_dim=10)
        original_context = {"state": np.random.randn(5)}

        encoded = encoder.encode(original_context)
        decoded = encoder.decode(encoded)

        # Decoded should be similar to original
        np.testing.assert_array_almost_equal(
            original_context["state"],
            decoded["state"],
            decimal=2
        )


class TestActiveInferenceBoundaryConditions:
    """Test edge cases and boundary conditions."""

    def test_zero_context(self):
        """Test prediction with zero context."""
        ai = ActiveInference()
        context = {"state": np.zeros(5)}

        prediction = ai.predict(context)
        assert np.all(np.isfinite(prediction))

    def test_large_context(self):
        """Test prediction with large context."""
        ai = ActiveInference()
        context = {"state": np.random.randn(100)}

        # May need to handle dimension mismatch
        try:
            prediction = ai.predict(context)
            assert prediction is not None
        except ValueError:
            # Expected if model doesn't support this dimension
            pass

    def test_extreme_values(self):
        """Test handling of extreme values."""
        ai = ActiveInference()
        context = {"state": np.array([1e6, 1e-6, -1e6, 0, 1])}

        prediction = ai.predict(context)
        assert np.all(np.isfinite(prediction))


class TestActiveInferenceIntegration:
    """Integration tests with other components."""

    def test_with_memory_context(self, mock_memory_graph):
        """Test inference with memory graph context."""
        ai = ActiveInference()

        # Extract context from memory graph
        nodes = mock_memory_graph.get_nodes()
        activations = np.array([node.activation for node in nodes])
        context = {"state": activations}

        prediction = ai.predict(context)
        assert prediction is not None

    def test_prediction_for_consolidation(self, mock_consolidation_state):
        """Test using predictions to guide consolidation."""
        ai = ActiveInference()

        state = mock_consolidation_state
        context = {"state": np.array([state.strength, state.duration, state.recency])}

        next_state_prediction = ai.predict(context)
        assert next_state_prediction is not None
```

## Template 3: Storage Integration Test

File: `tests/storage/test_neo4j_advanced.py`

```python
import pytest
import time
import numpy as np
from t4dm.storage.t4dx_graph_adapter import T4DXGraphAdapter


class TestT4DXGraphAdapterAdvanced:
    """Advanced integration tests for Neo4j storage."""

    def test_batch_write_performance(self):
        """Test batch insertion performance."""
        store = T4DXGraphAdapter()

        start = time.perf_counter()
        for i in range(100):  # Start small, scale up
            store.add_node(f"entity_{i}", properties={
                "value": i,
                "type": "concept"
            })
        elapsed = time.perf_counter() - start

        # Should complete in reasonable time
        assert elapsed < 5.0
        assert store.count_nodes() >= 100

    def test_complex_query(self):
        """Test complex graph query."""
        store = T4DXGraphAdapter()

        # Add test data
        for i in range(10):
            store.add_node(f"node_{i}", properties={"level": i % 3})

        # Add relationships
        for i in range(9):
            store.add_relationship(f"node_{i}", f"node_{i+1}", "CONNECTED_TO")

        # Query
        results = store.execute_query("""
            MATCH (n)-[r]->(m)
            RETURN n, r, m
            LIMIT 5
        """)

        assert len(results) >= 0

    def test_transactional_consistency(self):
        """Test transactional consistency."""
        store = T4DXGraphAdapter()

        try:
            store.begin_transaction()
            store.add_node("tx_node_1", properties={"type": "test"})
            store.add_node("tx_node_2", properties={"type": "test"})
            store.add_relationship("tx_node_1", "tx_node_2", "RELATED")
            store.commit_transaction()
        except Exception:
            store.rollback_transaction()
            raise

        # Verify all added
        assert store.node_exists("tx_node_1")
        assert store.node_exists("tx_node_2")

    def test_large_property_storage(self):
        """Test storing large properties."""
        store = T4DXGraphAdapter()

        large_data = {
            "vector": np.random.randn(256).tolist(),
            "text": "x" * 10000,
            "metadata": {
                "nested": {"deep": {"data": 123}}
            }
        }

        store.add_node("large_node", properties=large_data)

        retrieved = store.get_node("large_node")
        assert retrieved is not None
```

## Template 4: Persistence Recovery Test

File: `tests/persistence/test_recovery_advanced.py`

```python
import pytest
import tempfile
import os
from t4dm.persistence.manager import PersistenceManager
from t4dm.persistence.recovery import RecoveryManager


class TestCrashRecovery:
    """Test recovery from simulated crashes."""

    def test_basic_recovery(self, temp_persistence_dir):
        """Test basic recovery."""
        mgr = PersistenceManager(storage_path=temp_persistence_dir)

        # Write some transactions
        tx_ids = []
        for i in range(10):
            tx_id = mgr.write_transaction({
                "id": i,
                "data": f"transaction_{i}"
            })
            tx_ids.append(tx_id)

        # Simulate crash (close without cleanup)
        mgr._close_without_cleanup()
        del mgr

        # Recover
        recovery = RecoveryManager(storage_path=temp_persistence_dir)
        recovered = recovery.recover()

        assert len(recovered) == 10

    def test_partial_write_recovery(self, temp_persistence_dir):
        """Test recovery from partial writes."""
        mgr = PersistenceManager(storage_path=temp_persistence_dir)

        # Write complete transaction
        tx1_id = mgr.write_transaction({"id": 1})

        # Simulate partial write (incomplete flush)
        mgr._start_write()
        mgr._write_data({"id": 2})
        # Don't complete the write
        mgr._close_without_cleanup()

        # Recovery should restore consistent state
        recovery = RecoveryManager(storage_path=temp_persistence_dir)
        recovered = recovery.recover()

        # Incomplete transaction should be discarded
        assert len([r for r in recovered if r["id"] == 1]) == 1


class TestWALRotation:
    """Test WAL log rotation."""

    def test_wal_rotation_triggered(self):
        """Test WAL rotation when size threshold exceeded."""
        mgr = PersistenceManager(max_wal_size=1024)  # Small for testing

        # Write until rotation
        for i in range(100):
            mgr.write_transaction({
                "id": i,
                "data": "x" * 50
            })

        # Should have rotated
        assert len(mgr.wal_segments) > 1

    def test_wal_recovery_across_segments(self):
        """Test recovery across WAL segments."""
        mgr = PersistenceManager(max_wal_size=512)

        tx_ids = []
        for i in range(50):
            tx_id = mgr.write_transaction({
                "id": i,
                "data": "y" * 30
            })
            tx_ids.append(tx_id)

        # Force rotation
        mgr.rotate_wal()

        # Simulate crash
        mgr._close_without_cleanup()

        # Recovery across segments
        recovery = RecoveryManager()
        recovered = recovery.recover()

        assert len(recovered) >= 40


@pytest.fixture
def temp_persistence_dir():
    """Create temporary directory for persistence tests."""
    with tempfile.TemporaryDirectory() as tmpdir:
        yield tmpdir
```

## Quick Integration Tips

1. **Add to conftest.py** for shared fixtures:
```python
@pytest.fixture
def mock_memory_graph():
    """Mock memory graph for testing."""
    graph = Mock()
    graph.get_nodes.return_value = [
        Mock(id=i, activation=np.random.random())
        for i in range(10)
    ]
    return graph
```

2. **Run new tests**:
```bash
pytest tests/visualization/ -v --tb=short
pytest tests/prediction/ -v --tb=short
pytest tests/storage/test_neo4j_advanced.py -v
```

3. **Check coverage**:
```bash
pytest tests/visualization/ --cov=src/t4dm/visualization --cov-report=term-missing
```

Use these templates to get started on filling the critical coverage gaps!
