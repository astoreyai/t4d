# Phase 1D: Quick Reference Commands

## Run Tests

```bash
# Run all Phase 1D tests
cd /mnt/projects/ww
python -m pytest tests/consolidation/test_sleep_rpe.py -v

# Run specific test
python -m pytest tests/consolidation/test_sleep_rpe.py::test_replay_generates_rpe -v

# Run with coverage
python -m pytest tests/consolidation/test_sleep_rpe.py --cov=src/ww/consolidation/sleep --cov-report=term-missing
```

## Test Categories

```bash
# RPE generation tests
python -m pytest tests/consolidation/test_sleep_rpe.py -v -k "replay_generates or rpe_reflects"

# Prioritization tests
python -m pytest tests/consolidation/test_sleep_rpe.py -v -k "priority or probability"

# Integration tests
python -m pytest tests/consolidation/test_sleep_rpe.py -v -k "vta_active or integration"

# Robustness tests
python -m pytest tests/consolidation/test_sleep_rpe.py -v -k "without_vta or empty"
```

## Usage Example

```python
# Quick test of RPE generation
from ww.consolidation.sleep import SleepConsolidation
from ww.nca.vta import VTACircuit

# Create mocks
from tests.consolidation.test_sleep_rpe import MockMemory, MockSemanticMemory, MockGraphStore

episodic = MockMemory()
semantic = MockSemanticMemory()
graph = MockGraphStore()

# Create consolidation + VTA
consolidation = SleepConsolidation(episodic, semantic, graph, vae_enabled=False)
vta = VTACircuit()
consolidation.set_vta_circuit(vta)

# Run NREM (will generate RPE)
import asyncio
events = asyncio.run(consolidation.nrem_phase(session_id="test"))

# Check results
print(f"Replayed {len(events)} episodes")
for e in events:
    if e.rpe != 0.0:
        print(f"  Episode {e.episode_id}: RPE={e.rpe:.3f}, pos={e.sequence_position}")
```

## Check Statistics

```python
# Get consolidation statistics
stats = consolidation.get_stats()

# VTA statistics
if 'vta_circuit' in stats:
    vta_stats = stats['vta_circuit']
    print(f"Current DA: {vta_stats['current_da']:.3f}")
    print(f"Last RPE: {vta_stats['last_rpe']:.3f}")
    print(f"Firing mode: {vta_stats['firing_mode']}")

# Replay history
history = consolidation.get_replay_history(limit=10)
print(f"Recent replays: {len(history)}")
for event in history:
    print(f"  {event.episode_id}: RPE={event.rpe:.3f}")
```

## Files Modified

- `/mnt/projects/ww/src/ww/consolidation/sleep.py` - Main implementation
- `/mnt/projects/ww/tests/consolidation/test_sleep_rpe.py` - Test suite

## Success Criteria Checklist

- [x] Replay sequences generate RPE via VTA
- [x] RPE fed back to credit assignment
- [x] High-RPE sequences prioritized for replay
- [x] Tests pass (16/16)
- [x] Biological accuracy (90% reverse replay)
- [x] Integration with NREM consolidation
- [x] Graceful degradation without VTA

## Quick Verification

```bash
# Verify all tests pass
pytest tests/consolidation/test_sleep_rpe.py -v --tb=short

# Check coverage
pytest tests/consolidation/test_sleep_rpe.py --cov=src/ww/consolidation/sleep --cov-report=term-missing | grep "sleep.py"

# Run integration test
pytest tests/consolidation/test_sleep_rpe.py::test_integration_full_sleep_cycle_with_vta -v
```
