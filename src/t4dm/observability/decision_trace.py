"""
Decision Tracing Infrastructure for Bio-Inspired Components.

Provides structured tracing of decisions made by bio-inspired components
(neuromodulators, spiking blocks, consolidation phases, etc.) to enable
debugging and analysis of decision paths.

Example usage:
    from t4dm.observability.decision_trace import traced_decision, get_decision_tracer

    @traced_decision("dopamine", "compute_rpe")
    def compute_rpe(actual: float, expected: float) -> float:
        return actual - expected

    # Later, retrieve traces
    tracer = get_decision_tracer()
    for trace in tracer.get_traces():
        print(trace.to_json())
"""

import json
import threading
import time
from collections import deque
from dataclasses import dataclass, field
from datetime import datetime, timezone
from functools import wraps
from typing import Any, Callable, TypeVar

F = TypeVar("F", bound=Callable[..., Any])

# Global tracing state
_tracing_enabled = False
_tracing_lock = threading.Lock()


@dataclass
class DecisionTrace:
    """
    Represents a single traced decision from a bio-inspired component.

    Attributes:
        component: Name of the component making the decision (e.g., "dopamine", "lif_neuron")
        decision_type: Type of decision being made (e.g., "compute_rpe", "spike_threshold")
        inputs: Dictionary of input parameters to the decision
        output: The output/result of the decision
        timestamp: ISO 8601 timestamp of when the decision was made
        duration_ms: Optional duration of the decision in milliseconds
        metadata: Optional additional metadata about the decision
    """

    component: str
    decision_type: str
    inputs: dict[str, Any]
    output: Any
    timestamp: str = field(default_factory=lambda: datetime.now(timezone.utc).isoformat())
    duration_ms: float | None = None
    metadata: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        """Convert to dictionary representation."""
        result = {
            "component": self.component,
            "decision": self.decision_type,
            "inputs": self.inputs,
            "output": self.output,
            "ts": self.timestamp,
        }
        if self.duration_ms is not None:
            result["duration_ms"] = self.duration_ms
        if self.metadata:
            result["metadata"] = self.metadata
        return result

    def to_json(self) -> str:
        """Convert to JSON string representation."""
        return json.dumps(self.to_dict(), default=str)


class DecisionTracer:
    """
    Collects and manages decision traces with a configurable buffer.

    The tracer maintains a ring buffer of traces to prevent unbounded memory growth.
    Traces are collected in a thread-safe manner.

    Attributes:
        max_buffer_size: Maximum number of traces to retain in the buffer
    """

    def __init__(self, max_buffer_size: int = 10000):
        """
        Initialize the decision tracer.

        Args:
            max_buffer_size: Maximum number of traces to retain (default: 10000)
        """
        self._buffer: deque[DecisionTrace] = deque(maxlen=max_buffer_size)
        self._lock = threading.Lock()
        self._max_buffer_size = max_buffer_size

    @property
    def max_buffer_size(self) -> int:
        """Get the maximum buffer size."""
        return self._max_buffer_size

    def record(self, trace: DecisionTrace) -> None:
        """
        Record a decision trace.

        Args:
            trace: The DecisionTrace to record
        """
        with self._lock:
            self._buffer.append(trace)

    def get_traces(
        self,
        component: str | None = None,
        decision_type: str | None = None,
        limit: int | None = None,
    ) -> list[DecisionTrace]:
        """
        Retrieve traces, optionally filtered by component or decision type.

        Args:
            component: Filter by component name (optional)
            decision_type: Filter by decision type (optional)
            limit: Maximum number of traces to return (optional)

        Returns:
            List of matching DecisionTrace objects
        """
        with self._lock:
            traces = list(self._buffer)

        # Apply filters
        if component is not None:
            traces = [t for t in traces if t.component == component]
        if decision_type is not None:
            traces = [t for t in traces if t.decision_type == decision_type]

        # Apply limit
        if limit is not None:
            traces = traces[-limit:]

        return traces

    def get_traces_json(
        self,
        component: str | None = None,
        decision_type: str | None = None,
        limit: int | None = None,
    ) -> list[str]:
        """
        Retrieve traces as JSON strings.

        Args:
            component: Filter by component name (optional)
            decision_type: Filter by decision type (optional)
            limit: Maximum number of traces to return (optional)

        Returns:
            List of JSON string representations of traces
        """
        return [t.to_json() for t in self.get_traces(component, decision_type, limit)]

    def clear(self) -> None:
        """Clear all traces from the buffer."""
        with self._lock:
            self._buffer.clear()

    def __len__(self) -> int:
        """Return the current number of traces in the buffer."""
        with self._lock:
            return len(self._buffer)


# Global singleton tracer
_tracer: DecisionTracer | None = None
_tracer_init_lock = threading.Lock()


def get_decision_tracer(max_buffer_size: int = 10000) -> DecisionTracer:
    """
    Get the singleton DecisionTracer instance.

    Args:
        max_buffer_size: Maximum buffer size (only used on first call)

    Returns:
        The global DecisionTracer instance
    """
    global _tracer
    if _tracer is None:
        with _tracer_init_lock:
            if _tracer is None:
                _tracer = DecisionTracer(max_buffer_size=max_buffer_size)
    return _tracer


def enable_decision_tracing() -> None:
    """Enable decision tracing globally."""
    global _tracing_enabled
    with _tracing_lock:
        _tracing_enabled = True


def disable_decision_tracing() -> None:
    """Disable decision tracing globally."""
    global _tracing_enabled
    with _tracing_lock:
        _tracing_enabled = False


def is_decision_tracing_enabled() -> bool:
    """Check if decision tracing is currently enabled."""
    with _tracing_lock:
        return _tracing_enabled


def traced_decision(
    component: str,
    decision_type: str,
    capture_args: list[str] | None = None,
    metadata: dict[str, Any] | None = None,
) -> Callable[[F], F]:
    """
    Decorator that traces function calls as decisions.

    When tracing is enabled, this decorator captures the inputs and outputs
    of the decorated function and records them as a DecisionTrace.

    Args:
        component: Name of the component (e.g., "dopamine", "lif_neuron")
        decision_type: Type of decision (e.g., "compute_rpe", "spike")
        capture_args: List of argument names to capture (None = all)
        metadata: Additional static metadata to include in traces

    Returns:
        Decorated function

    Example:
        @traced_decision("dopamine", "compute_rpe")
        def compute_rpe(actual: float, expected: float) -> float:
            return actual - expected
    """

    def decorator(func: F) -> F:
        @wraps(func)
        def wrapper(*args: Any, **kwargs: Any) -> Any:
            # Check if tracing is enabled
            if not is_decision_tracing_enabled():
                return func(*args, **kwargs)

            # Capture inputs
            inputs: dict[str, Any] = {}

            # Get function signature for mapping positional args
            import inspect

            sig = inspect.signature(func)
            params = list(sig.parameters.keys())

            # Map positional args to param names
            for i, arg in enumerate(args):
                if i < len(params):
                    param_name = params[i]
                    if capture_args is None or param_name in capture_args:
                        inputs[param_name] = _serialize_value(arg)

            # Add kwargs
            for key, value in kwargs.items():
                if capture_args is None or key in capture_args:
                    inputs[key] = _serialize_value(value)

            # Execute and time the function
            start_time = time.perf_counter()
            try:
                result = func(*args, **kwargs)
                duration_ms = (time.perf_counter() - start_time) * 1000

                # Create and record trace
                trace = DecisionTrace(
                    component=component,
                    decision_type=decision_type,
                    inputs=inputs,
                    output=_serialize_value(result),
                    duration_ms=duration_ms,
                    metadata=metadata or {},
                )
                get_decision_tracer().record(trace)

                return result
            except Exception as e:
                duration_ms = (time.perf_counter() - start_time) * 1000

                # Record error trace
                trace = DecisionTrace(
                    component=component,
                    decision_type=decision_type,
                    inputs=inputs,
                    output={"error": str(e), "type": type(e).__name__},
                    duration_ms=duration_ms,
                    metadata={**(metadata or {}), "error": True},
                )
                get_decision_tracer().record(trace)
                raise

        return wrapper  # type: ignore[return-value]

    return decorator


def _serialize_value(value: Any) -> Any:
    """
    Serialize a value for JSON-safe storage in traces.

    Handles common types like tensors, numpy arrays, and complex objects.
    """
    # Handle None
    if value is None:
        return None

    # Handle primitives
    if isinstance(value, (bool, int, float, str)):
        return value

    # Handle lists and tuples
    if isinstance(value, (list, tuple)):
        return [_serialize_value(v) for v in value]

    # Handle dicts
    if isinstance(value, dict):
        return {str(k): _serialize_value(v) for k, v in value.items()}

    # Handle numpy arrays
    try:
        import numpy as np

        if isinstance(value, np.ndarray):
            if value.size <= 10:
                return value.tolist()
            return {
                "type": "ndarray",
                "shape": list(value.shape),
                "dtype": str(value.dtype),
                "summary": {
                    "min": float(np.min(value)),
                    "max": float(np.max(value)),
                    "mean": float(np.mean(value)),
                },
            }
        if isinstance(value, (np.integer, np.floating)):
            return float(value)
    except ImportError:
        pass

    # Handle torch tensors
    try:
        import torch

        if isinstance(value, torch.Tensor):
            if value.numel() <= 10:
                return value.detach().cpu().tolist()
            return {
                "type": "tensor",
                "shape": list(value.shape),
                "dtype": str(value.dtype),
                "device": str(value.device),
                "summary": {
                    "min": float(value.min().item()),
                    "max": float(value.max().item()),
                    "mean": float(value.float().mean().item()),
                },
            }
    except ImportError:
        pass

    # Fallback: convert to string
    return str(value)


def reset_decision_tracer() -> None:
    """
    Reset the global decision tracer.

    This clears all traces and resets the singleton. Useful for testing.
    """
    global _tracer
    with _tracer_init_lock:
        if _tracer is not None:
            _tracer.clear()
        _tracer = None
