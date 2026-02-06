"""
Unit Tests for Decision Tracing Infrastructure.

Verifies the decision tracing decorator, tracer buffer management,
and JSON serialization for bio-inspired component debugging.
"""

import json
import time
import threading
import pytest
import numpy as np
import torch

from t4dm.observability.decision_trace import (
    DecisionTrace,
    DecisionTracer,
    disable_decision_tracing,
    enable_decision_tracing,
    get_decision_tracer,
    is_decision_tracing_enabled,
    reset_decision_tracer,
    traced_decision,
    _serialize_value,
)


@pytest.fixture(autouse=True)
def reset_tracing():
    """Reset tracing state before and after each test."""
    disable_decision_tracing()
    reset_decision_tracer()
    yield
    disable_decision_tracing()
    reset_decision_tracer()


class TestDecisionTrace:
    """Test DecisionTrace dataclass."""

    def test_trace_creation(self):
        """Should create a trace with all required fields."""
        trace = DecisionTrace(
            component="dopamine",
            decision_type="compute_rpe",
            inputs={"actual": 0.8, "expected": 0.5},
            output=0.3,
        )

        assert trace.component == "dopamine"
        assert trace.decision_type == "compute_rpe"
        assert trace.inputs == {"actual": 0.8, "expected": 0.5}
        assert trace.output == 0.3
        assert trace.timestamp is not None

    def test_trace_to_dict(self):
        """Should convert trace to dictionary."""
        trace = DecisionTrace(
            component="dopamine",
            decision_type="compute_rpe",
            inputs={"actual": 0.8, "expected": 0.5},
            output=0.3,
        )

        d = trace.to_dict()

        assert d["component"] == "dopamine"
        assert d["decision"] == "compute_rpe"
        assert d["inputs"] == {"actual": 0.8, "expected": 0.5}
        assert d["output"] == 0.3
        assert "ts" in d

    def test_trace_to_json(self):
        """Should convert trace to valid JSON string."""
        trace = DecisionTrace(
            component="dopamine",
            decision_type="compute_rpe",
            inputs={"actual": 0.8, "expected": 0.5},
            output=0.3,
        )

        json_str = trace.to_json()
        parsed = json.loads(json_str)

        assert parsed["component"] == "dopamine"
        assert parsed["decision"] == "compute_rpe"
        assert parsed["inputs"]["actual"] == 0.8
        assert parsed["output"] == 0.3

    def test_trace_with_duration(self):
        """Should include duration_ms when provided."""
        trace = DecisionTrace(
            component="lif_neuron",
            decision_type="spike",
            inputs={"membrane_potential": 1.5},
            output=True,
            duration_ms=0.5,
        )

        d = trace.to_dict()
        assert d["duration_ms"] == 0.5

    def test_trace_with_metadata(self):
        """Should include metadata when provided."""
        trace = DecisionTrace(
            component="stdp",
            decision_type="update_weight",
            inputs={"pre_spike": True, "post_spike": True},
            output=0.01,
            metadata={"layer": 3, "synapse_id": "syn_123"},
        )

        d = trace.to_dict()
        assert d["metadata"]["layer"] == 3
        assert d["metadata"]["synapse_id"] == "syn_123"


class TestDecisionTracer:
    """Test DecisionTracer class."""

    def test_tracer_creation(self):
        """Should create tracer with default buffer size."""
        tracer = DecisionTracer()
        assert tracer.max_buffer_size == 10000
        assert len(tracer) == 0

    def test_tracer_custom_buffer_size(self):
        """Should respect custom buffer size."""
        tracer = DecisionTracer(max_buffer_size=100)
        assert tracer.max_buffer_size == 100

    def test_record_trace(self):
        """Should record traces."""
        tracer = DecisionTracer()

        trace = DecisionTrace(
            component="test",
            decision_type="test_decision",
            inputs={"x": 1},
            output=2,
        )
        tracer.record(trace)

        assert len(tracer) == 1

    def test_get_traces(self):
        """Should retrieve all traces."""
        tracer = DecisionTracer()

        for i in range(5):
            tracer.record(
                DecisionTrace(
                    component=f"comp_{i}",
                    decision_type="test",
                    inputs={"i": i},
                    output=i * 2,
                )
            )

        traces = tracer.get_traces()
        assert len(traces) == 5

    def test_get_traces_filter_by_component(self):
        """Should filter traces by component."""
        tracer = DecisionTracer()

        tracer.record(
            DecisionTrace(component="dopamine", decision_type="rpe", inputs={}, output=0.1)
        )
        tracer.record(
            DecisionTrace(component="serotonin", decision_type="mood", inputs={}, output=0.5)
        )
        tracer.record(
            DecisionTrace(component="dopamine", decision_type="rpe", inputs={}, output=0.2)
        )

        traces = tracer.get_traces(component="dopamine")
        assert len(traces) == 2
        assert all(t.component == "dopamine" for t in traces)

    def test_get_traces_filter_by_decision_type(self):
        """Should filter traces by decision type."""
        tracer = DecisionTracer()

        tracer.record(
            DecisionTrace(component="neuron", decision_type="spike", inputs={}, output=True)
        )
        tracer.record(
            DecisionTrace(component="neuron", decision_type="reset", inputs={}, output=0.0)
        )
        tracer.record(
            DecisionTrace(component="neuron", decision_type="spike", inputs={}, output=False)
        )

        traces = tracer.get_traces(decision_type="spike")
        assert len(traces) == 2
        assert all(t.decision_type == "spike" for t in traces)

    def test_get_traces_with_limit(self):
        """Should limit number of returned traces."""
        tracer = DecisionTracer()

        for i in range(10):
            tracer.record(
                DecisionTrace(component="test", decision_type="op", inputs={}, output=i)
            )

        traces = tracer.get_traces(limit=3)
        assert len(traces) == 3
        # Should return last 3
        assert traces[-1].output == 9

    def test_buffer_overflow(self):
        """Should drop oldest traces when buffer overflows."""
        tracer = DecisionTracer(max_buffer_size=5)

        for i in range(10):
            tracer.record(
                DecisionTrace(component="test", decision_type="op", inputs={"i": i}, output=i)
            )

        assert len(tracer) == 5
        traces = tracer.get_traces()
        # Should have traces 5-9 (dropped 0-4)
        outputs = [t.output for t in traces]
        assert outputs == [5, 6, 7, 8, 9]

    def test_clear(self):
        """Should clear all traces."""
        tracer = DecisionTracer()

        for i in range(5):
            tracer.record(
                DecisionTrace(component="test", decision_type="op", inputs={}, output=i)
            )

        assert len(tracer) == 5
        tracer.clear()
        assert len(tracer) == 0

    def test_get_traces_json(self):
        """Should return traces as JSON strings."""
        tracer = DecisionTracer()

        tracer.record(
            DecisionTrace(component="test", decision_type="op", inputs={"x": 1}, output=2)
        )

        json_traces = tracer.get_traces_json()
        assert len(json_traces) == 1
        parsed = json.loads(json_traces[0])
        assert parsed["component"] == "test"

    def test_thread_safety(self):
        """Should handle concurrent access safely."""
        tracer = DecisionTracer(max_buffer_size=1000)
        errors = []

        def record_traces(thread_id):
            try:
                for i in range(100):
                    tracer.record(
                        DecisionTrace(
                            component=f"thread_{thread_id}",
                            decision_type="op",
                            inputs={"i": i},
                            output=i,
                        )
                    )
            except Exception as e:
                errors.append(e)

        threads = [threading.Thread(target=record_traces, args=(i,)) for i in range(10)]

        for t in threads:
            t.start()
        for t in threads:
            t.join()

        assert len(errors) == 0
        assert len(tracer) == 1000  # 10 threads * 100 traces


class TestTracingControl:
    """Test enable/disable tracing functions."""

    def test_tracing_disabled_by_default(self):
        """Tracing should be disabled by default."""
        assert not is_decision_tracing_enabled()

    def test_enable_tracing(self):
        """Should enable tracing."""
        enable_decision_tracing()
        assert is_decision_tracing_enabled()

    def test_disable_tracing(self):
        """Should disable tracing."""
        enable_decision_tracing()
        disable_decision_tracing()
        assert not is_decision_tracing_enabled()


class TestSingletonTracer:
    """Test get_decision_tracer singleton."""

    def test_singleton(self):
        """Should return same tracer instance."""
        tracer1 = get_decision_tracer()
        tracer2 = get_decision_tracer()
        assert tracer1 is tracer2

    def test_reset_tracer(self):
        """Should reset singleton tracer."""
        tracer1 = get_decision_tracer()
        tracer1.record(
            DecisionTrace(component="test", decision_type="op", inputs={}, output=1)
        )

        reset_decision_tracer()

        tracer2 = get_decision_tracer()
        assert tracer1 is not tracer2
        assert len(tracer2) == 0


class TestTracedDecisionDecorator:
    """Test @traced_decision decorator."""

    def test_decorator_no_trace_when_disabled(self):
        """Should not trace when tracing is disabled."""

        @traced_decision("test", "add")
        def add(a: int, b: int) -> int:
            return a + b

        result = add(1, 2)

        assert result == 3
        tracer = get_decision_tracer()
        assert len(tracer) == 0

    def test_decorator_traces_when_enabled(self):
        """Should trace when tracing is enabled."""
        enable_decision_tracing()

        @traced_decision("test", "add")
        def add(a: int, b: int) -> int:
            return a + b

        result = add(1, 2)

        assert result == 3
        tracer = get_decision_tracer()
        assert len(tracer) == 1

        trace = tracer.get_traces()[0]
        assert trace.component == "test"
        assert trace.decision_type == "add"
        assert trace.inputs == {"a": 1, "b": 2}
        assert trace.output == 3

    def test_decorator_captures_kwargs(self):
        """Should capture keyword arguments."""
        enable_decision_tracing()

        @traced_decision("dopamine", "compute_rpe")
        def compute_rpe(actual: float, expected: float) -> float:
            return actual - expected

        result = compute_rpe(actual=0.8, expected=0.5)

        assert result == pytest.approx(0.3)
        tracer = get_decision_tracer()
        trace = tracer.get_traces()[0]
        assert trace.inputs == {"actual": 0.8, "expected": 0.5}

    def test_decorator_records_duration(self):
        """Should record execution duration."""
        enable_decision_tracing()

        @traced_decision("test", "slow_op")
        def slow_op() -> int:
            time.sleep(0.01)
            return 42

        slow_op()

        tracer = get_decision_tracer()
        trace = tracer.get_traces()[0]
        assert trace.duration_ms is not None
        assert trace.duration_ms >= 10  # At least 10ms

    def test_decorator_captures_selected_args(self):
        """Should only capture specified arguments."""
        enable_decision_tracing()

        @traced_decision("test", "op", capture_args=["a"])
        def op(a: int, b: int, c: int) -> int:
            return a + b + c

        op(1, 2, 3)

        tracer = get_decision_tracer()
        trace = tracer.get_traces()[0]
        assert trace.inputs == {"a": 1}

    def test_decorator_includes_metadata(self):
        """Should include static metadata."""
        enable_decision_tracing()

        @traced_decision("test", "op", metadata={"layer": 5})
        def op(x: int) -> int:
            return x * 2

        op(10)

        tracer = get_decision_tracer()
        trace = tracer.get_traces()[0]
        assert trace.metadata == {"layer": 5}

    def test_decorator_handles_exception(self):
        """Should trace exceptions."""
        enable_decision_tracing()

        @traced_decision("test", "failing_op")
        def failing_op(x: int) -> int:
            raise ValueError("Test error")

        with pytest.raises(ValueError):
            failing_op(5)

        tracer = get_decision_tracer()
        assert len(tracer) == 1

        trace = tracer.get_traces()[0]
        assert trace.output["error"] == "Test error"
        assert trace.output["type"] == "ValueError"
        assert trace.metadata.get("error") is True

    def test_decorator_preserves_function_metadata(self):
        """Should preserve decorated function's metadata."""

        @traced_decision("test", "documented_fn")
        def documented_fn(x: int) -> int:
            """This is a documented function."""
            return x

        assert documented_fn.__name__ == "documented_fn"
        assert documented_fn.__doc__ == "This is a documented function."


class TestValueSerialization:
    """Test _serialize_value function."""

    def test_serialize_primitives(self):
        """Should serialize primitives directly."""
        assert _serialize_value(None) is None
        assert _serialize_value(42) == 42
        assert _serialize_value(3.14) == 3.14
        assert _serialize_value("hello") == "hello"
        assert _serialize_value(True) is True

    def test_serialize_list(self):
        """Should serialize lists recursively."""
        assert _serialize_value([1, 2, 3]) == [1, 2, 3]
        assert _serialize_value([1, [2, 3]]) == [1, [2, 3]]

    def test_serialize_dict(self):
        """Should serialize dicts recursively."""
        assert _serialize_value({"a": 1, "b": 2}) == {"a": 1, "b": 2}

    def test_serialize_numpy_small_array(self):
        """Should serialize small numpy arrays as lists."""
        arr = np.array([1, 2, 3])
        result = _serialize_value(arr)
        assert result == [1, 2, 3]

    def test_serialize_numpy_large_array(self):
        """Should summarize large numpy arrays."""
        arr = np.arange(100)
        result = _serialize_value(arr)

        assert result["type"] == "ndarray"
        assert result["shape"] == [100]
        assert "summary" in result
        assert result["summary"]["min"] == 0
        assert result["summary"]["max"] == 99

    def test_serialize_numpy_scalar(self):
        """Should serialize numpy scalars as floats."""
        assert _serialize_value(np.float64(3.14)) == 3.14
        assert _serialize_value(np.int32(42)) == 42

    def test_serialize_torch_small_tensor(self):
        """Should serialize small tensors as lists."""
        t = torch.tensor([1, 2, 3])
        result = _serialize_value(t)
        assert result == [1, 2, 3]

    def test_serialize_torch_large_tensor(self):
        """Should summarize large tensors."""
        t = torch.arange(100, dtype=torch.float32)
        result = _serialize_value(t)

        assert result["type"] == "tensor"
        assert result["shape"] == [100]
        assert "summary" in result
        assert result["summary"]["min"] == 0
        assert result["summary"]["max"] == 99

    def test_serialize_unknown_object(self):
        """Should stringify unknown objects."""

        class CustomObj:
            def __str__(self):
                return "CustomObj()"

        result = _serialize_value(CustomObj())
        assert result == "CustomObj()"


class TestIntegration:
    """Integration tests for decision tracing."""

    def test_dopamine_rpe_trace_format(self):
        """Should produce expected JSON format for dopamine RPE."""
        enable_decision_tracing()

        @traced_decision("dopamine", "compute_rpe")
        def compute_rpe(actual: float, expected: float) -> float:
            return actual - expected

        compute_rpe(0.8, 0.5)

        tracer = get_decision_tracer()
        json_str = tracer.get_traces_json()[0]
        parsed = json.loads(json_str)

        # Verify format matches specification
        assert "component" in parsed
        assert "decision" in parsed
        assert "inputs" in parsed
        assert "output" in parsed
        assert "ts" in parsed

        assert parsed["component"] == "dopamine"
        assert parsed["decision"] == "compute_rpe"
        assert parsed["inputs"]["actual"] == 0.8
        assert parsed["inputs"]["expected"] == 0.5
        assert parsed["output"] == pytest.approx(0.3)

    def test_lif_neuron_spike_trace(self):
        """Should trace LIF neuron spike decisions."""
        enable_decision_tracing()

        @traced_decision("lif_neuron", "check_spike")
        def check_spike(
            membrane_potential: torch.Tensor, threshold: float
        ) -> torch.Tensor:
            return membrane_potential > threshold

        v = torch.tensor([0.5, 1.2, 0.8, 1.5])
        spikes = check_spike(v, 1.0)

        tracer = get_decision_tracer()
        trace = tracer.get_traces()[0]

        assert trace.component == "lif_neuron"
        assert trace.decision_type == "check_spike"
        assert trace.inputs["threshold"] == 1.0
        # Small tensor serialized as list (with float32 precision)
        assert trace.inputs["membrane_potential"] == pytest.approx([0.5, 1.2, 0.8, 1.5], rel=1e-5)
        assert trace.output == [False, True, False, True]

    def test_multiple_component_traces(self):
        """Should trace multiple components independently."""
        enable_decision_tracing()

        @traced_decision("dopamine", "compute_rpe")
        def compute_rpe(actual: float, expected: float) -> float:
            return actual - expected

        @traced_decision("serotonin", "compute_mood")
        def compute_mood(baseline: float, stimulus: float) -> float:
            return baseline + stimulus * 0.5

        compute_rpe(0.8, 0.5)
        compute_mood(0.5, 0.3)

        tracer = get_decision_tracer()

        dopamine_traces = tracer.get_traces(component="dopamine")
        serotonin_traces = tracer.get_traces(component="serotonin")

        assert len(dopamine_traces) == 1
        assert len(serotonin_traces) == 1
        assert dopamine_traces[0].decision_type == "compute_rpe"
        assert serotonin_traces[0].decision_type == "compute_mood"
