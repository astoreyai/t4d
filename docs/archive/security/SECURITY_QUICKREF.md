# Security Quick Reference: Eligibility Trace System

**For Developers:** Quick reference card for secure usage of EligibilityTrace

---

## Safe Usage Patterns

### ✓ SAFE: Basic Usage
```python
from t4dm.learning.eligibility import EligibilityTrace

# Create with defaults (safe)
trace = EligibilityTrace()

# Update with validated inputs
if len(memory_id) < 256 and 0 <= activity <= 10:
    trace.update(memory_id, activity=activity)

# Assign credit with validated reward
if abs(reward) < 1000 and np.isfinite(reward):
    credits = trace.assign_credit(reward)

# Step with reasonable time delta
if 0 <= dt <= 86400:  # Max 24 hours
    trace.step(dt=dt)
```

### ✗ UNSAFE: Common Mistakes
```python
# ❌ VULNERABLE: No validation
user_id = request.get("memory_id")  # Could be huge!
trace.update(user_id)  # Memory exhaustion

# ❌ VULNERABLE: NaN injection
activity = compute_activity()  # Could be NaN
trace.update("mem1", activity=activity)  # Corrupts system

# ❌ VULNERABLE: Extreme reward
reward = external_api.get_reward()  # Untrusted source
trace.assign_credit(reward)  # Could overflow

# ❌ VULNERABLE: No dt validation
dt = time.time() - last_time  # Could be huge if clock skew
trace.step(dt=dt)  # Instant decay
```

---

## Input Validation Checklist

Before calling EligibilityTrace methods, validate:

### update(memory_id, activity)
- [ ] `isinstance(memory_id, str)`
- [ ] `len(memory_id) <= 256`
- [ ] `memory_id.isprintable()`
- [ ] `0 <= activity < float('inf')`
- [ ] `not np.isnan(activity)`

### assign_credit(reward)
- [ ] `isinstance(reward, (int, float))`
- [ ] `np.isfinite(reward)`
- [ ] `abs(reward) <= 1000` (or clip)

### step(dt)
- [ ] `dt >= 0`
- [ ] `dt <= 86400` (24 hours)
- [ ] `np.isfinite(dt)`

---

## Defensive Coding Template

```python
import numpy as np
from t4dm.learning.eligibility import EligibilityTrace

class SafeEligibilityWrapper:
    """Wrapper that adds input validation."""

    MAX_MEMORY_ID_LENGTH = 256
    MAX_REWARD = 1000.0
    MAX_DT = 86400.0
    MAX_ACTIVITY = 10.0

    def __init__(self, **kwargs):
        self._trace = EligibilityTrace(**kwargs)

    def update(self, memory_id: str, activity: float = 1.0):
        """Validated update."""
        # Validate memory_id
        if not isinstance(memory_id, str):
            raise TypeError(f"memory_id must be str, got {type(memory_id)}")
        if len(memory_id) > self.MAX_MEMORY_ID_LENGTH:
            raise ValueError(f"memory_id too long: {len(memory_id)}")
        if not memory_id.isprintable():
            raise ValueError("memory_id must be printable")

        # Validate activity
        if not isinstance(activity, (int, float)):
            raise TypeError(f"activity must be numeric")
        if not np.isfinite(activity):
            raise ValueError(f"activity must be finite, got {activity}")
        if activity < 0:
            raise ValueError(f"activity must be non-negative")

        # Clip to safe range
        activity = min(activity, self.MAX_ACTIVITY)

        return self._trace.update(memory_id, activity)

    def assign_credit(self, reward: float):
        """Validated credit assignment."""
        if not isinstance(reward, (int, float)):
            raise TypeError(f"reward must be numeric")
        if not np.isfinite(reward):
            raise ValueError(f"reward must be finite, got {reward}")

        # Clip to safe range
        reward = float(np.clip(reward, -self.MAX_REWARD, self.MAX_REWARD))

        return self._trace.assign_credit(reward)

    def step(self, dt: float = None):
        """Validated step."""
        if dt is not None:
            if not isinstance(dt, (int, float)):
                raise TypeError(f"dt must be numeric")
            if not np.isfinite(dt):
                raise ValueError(f"dt must be finite")
            if dt < 0:
                raise ValueError(f"dt must be non-negative")

            # Clip to safe range
            dt = min(dt, self.MAX_DT)

        return self._trace.step(dt)

    def __getattr__(self, name):
        """Forward other methods to underlying trace."""
        return getattr(self._trace, name)
```

---

## Thread Safety

### ⚠️ WARNING: Not Thread-Safe

EligibilityTrace is **NOT thread-safe**. Concurrent access can cause:
- Data corruption
- Lost updates
- Capacity limit bypass

### Solution 1: External Locking
```python
import threading

trace = EligibilityTrace()
lock = threading.RLock()

def safe_update(memory_id, activity):
    with lock:
        trace.update(memory_id, activity)

def safe_step(dt):
    with lock:
        trace.step(dt)
```

### Solution 2: Thread-Local Storage
```python
import threading

trace_storage = threading.local()

def get_trace():
    """Get thread-local trace."""
    if not hasattr(trace_storage, 'trace'):
        trace_storage.trace = EligibilityTrace()
    return trace_storage.trace

# Each thread has isolated trace
trace = get_trace()
trace.update("mem1")
```

---

## Known Vulnerabilities (Until Patched)

| Issue | Severity | Workaround |
|-------|----------|-----------|
| Huge memory IDs | HIGH | Validate `len(memory_id) <= 256` |
| NaN activity | MEDIUM | Check `np.isfinite(activity)` |
| Extreme reward | HIGH | Clip to `[-1000, 1000]` |
| Huge dt | MEDIUM | Clip to `86400` (24h) |
| Layered capacity bypass | MEDIUM | Use base EligibilityTrace instead |
| No thread safety | CRITICAL | Add external locking |

---

## Security Testing

### Before Deployment
```bash
# Run security proof-of-concept
python security_poc_eligibility.py

# Expected: 0/6 exploits successful after fixes
```

### Unit Test Template
```python
import pytest
import numpy as np
from t4dm.learning.eligibility import EligibilityTrace

def test_reject_nan_activity():
    """System must reject NaN activity."""
    trace = EligibilityTrace()
    with pytest.raises(ValueError):
        trace.update("mem1", activity=float('nan'))

def test_reject_huge_memory_id():
    """System must reject huge memory IDs."""
    trace = EligibilityTrace()
    huge_id = "A" * 10_000_000
    with pytest.raises(ValueError):
        trace.update(huge_id)

def test_clip_extreme_reward():
    """System must clip extreme rewards."""
    trace = EligibilityTrace()
    trace.update("mem1")
    credits = trace.assign_credit(reward=1e100)
    # Should be clipped, not overflow
    assert np.isfinite(credits["mem1"])
    assert abs(credits["mem1"]) <= 100_000  # MAX_REWARD * MAX_TRACE
```

---

## Monitoring & Logging

### Security Events to Log
```python
import logging
logger = logging.getLogger(__name__)

# Log capacity exceeded
if trace.count >= trace.max_traces * 0.9:
    logger.warning(f"Trace capacity at 90%: {trace.count}/{trace.max_traces}")

# Log extreme values (potential attack)
if abs(reward) > 100:
    logger.info(f"Extreme reward detected: {reward}")

# Log validation failures
try:
    trace.update(memory_id, activity)
except ValueError as e:
    logger.error(f"Validation failed: {e}", extra={"memory_id": memory_id})
```

### Metrics to Track
- `trace.count` (capacity utilization)
- `trace._total_updates` (update rate)
- `trace._total_credits_assigned` (credit flow)
- Validation failures per second (attack detection)

---

## Quick Fixes (Copy-Paste)

### Add to update() Method
```python
def update(self, memory_id: str, activity: float = 1.0):
    # BEGIN SECURITY PATCH
    if not isinstance(memory_id, str):
        raise TypeError(f"memory_id must be str")
    if len(memory_id) > 256:
        raise ValueError(f"memory_id too long: {len(memory_id)}")
    if not isinstance(activity, (int, float)):
        raise TypeError(f"activity must be numeric")
    if not np.isfinite(activity):
        raise ValueError(f"activity must be finite")
    if activity < 0:
        raise ValueError(f"activity must be non-negative")
    activity = min(activity, 10.0)  # Clip
    # END SECURITY PATCH

    # ... existing logic ...
```

### Add to assign_credit() Method
```python
def assign_credit(self, reward: float) -> Dict[str, float]:
    # BEGIN SECURITY PATCH
    if not isinstance(reward, (int, float)):
        raise TypeError(f"reward must be numeric")
    if not np.isfinite(reward):
        raise ValueError(f"reward must be finite")
    reward = float(np.clip(reward, -1000.0, 1000.0))  # Clip
    # END SECURITY PATCH

    # ... existing logic ...
```

### Add to step() Method
```python
def step(self, dt: Optional[float] = None):
    current_time = time.time()
    if dt is None:
        dt = current_time - self._last_step_time

    # BEGIN SECURITY PATCH
    if not isinstance(dt, (int, float)):
        raise TypeError(f"dt must be numeric")
    if not np.isfinite(dt):
        raise ValueError(f"dt must be finite")
    if dt < 0:
        raise ValueError(f"dt must be non-negative")
    dt = min(dt, 86400.0)  # Clip to 24 hours
    # END SECURITY PATCH

    # ... existing logic ...
```

---

## Resources

- **Detailed Analysis:** `security_analysis_eligibility_trace.md`
- **Executive Summary:** `SECURITY_ASSESSMENT_SUMMARY.md`
- **Proof-of-Concept:** `security_poc_eligibility.py`
- **This Quick Ref:** `SECURITY_QUICKREF.md`

---

## Contact

**Questions?** See full security assessment in `security_analysis_eligibility_trace.md`

**Found a vulnerability?** Report via project issue tracker (for research use only)

---

**Last Updated:** 2025-12-06
**Status:** System requires patching before production use
