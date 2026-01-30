#!/usr/bin/env python3
"""
Proof-of-Concept Security Exploits for Eligibility Trace System

WARNING: For research and testing purposes only.
Demonstrates vulnerabilities identified in security assessment.

Usage:
    python security_poc_eligibility.py

Each exploit is isolated and can be run independently.
"""

import sys
import time
import threading
import traceback
from pathlib import Path

# Add src to path
sys.path.insert(0, str(Path(__file__).parent / "src"))

import numpy as np
from ww.learning.eligibility import (
    EligibilityTrace,
    LayeredEligibilityTrace,
    MAX_TRACE_VALUE,
    MAX_TRACES
)


def exploit_high1_reward_overflow():
    """
    HIGH-1: Unbounded reward multiplication causes overflow to infinity.

    Expected: credits should be finite
    Actual: credits become infinity, poisoning learning system
    """
    print("\n" + "="*70)
    print("EXPLOIT HIGH-1: Reward Overflow Attack")
    print("="*70)

    trace = EligibilityTrace()
    trace.update("victim_memory", activity=1.0)

    # Attacker provides extreme reward
    extreme_reward = 1e308  # Near float max
    print(f"Attacking with reward: {extreme_reward:.2e}")

    credits = trace.assign_credit(reward=extreme_reward)

    print(f"\nResult:")
    print(f"  Credit assigned: {credits['victim_memory']}")
    print(f"  Is infinite: {np.isinf(credits['victim_memory'])}")
    print(f"  Total credits: {trace._total_credits_assigned}")

    if np.isinf(credits['victim_memory']):
        print("\n[EXPLOIT SUCCESSFUL] Credit overflowed to infinity!")
        print("Impact: Learning system poisoned, model weights will explode")
        return True
    else:
        print("\n[EXPLOIT FAILED] System handled extreme reward safely")
        return False


def exploit_high2_memory_exhaustion():
    """
    HIGH-2: Memory ID injection exhausts system memory.

    Expected: max_traces limit prevents exhaustion
    Actual: huge memory IDs consume gigabytes despite trace limit
    """
    print("\n" + "="*70)
    print("EXPLOIT HIGH-2: Memory Exhaustion via Huge IDs")
    print("="*70)

    trace = EligibilityTrace(max_traces=100)

    print(f"Creating 100 traces with 1 MB memory IDs each...")

    # Each memory_id is 1 MB (reduced from 10 MB for faster demo)
    memory_consumed_mb = 0
    for i in range(100):
        huge_id = "A" * (1024 * 1024) + str(i)  # 1 MB + counter
        try:
            trace.update(huge_id, activity=1.0)
            memory_consumed_mb += 1
        except Exception as e:
            print(f"\n[EXPLOIT BLOCKED] Exception raised: {e}")
            return False

    print(f"\nResult:")
    print(f"  Traces created: {trace.count}")
    print(f"  Estimated memory consumed: ~{memory_consumed_mb} MB")
    print(f"  Max traces limit: {trace.max_traces}")

    if memory_consumed_mb >= 50:  # 50+ MB consumed
        print("\n[EXPLOIT SUCCESSFUL] Consumed excessive memory!")
        print("Impact: System can be OOM-killed despite max_traces limit")
        return True
    else:
        print("\n[EXPLOIT FAILED] Memory consumption blocked")
        return False


def exploit_medium1_nan_activity():
    """
    MEDIUM-1: NaN activity propagates through calculations.

    Expected: NaN should be rejected
    Actual: NaN propagates, corrupting all trace values
    """
    print("\n" + "="*70)
    print("EXPLOIT MEDIUM-1: NaN Activity Injection")
    print("="*70)

    trace = EligibilityTrace()

    # Create normal trace first
    trace.update("normal_memory", activity=1.0)
    normal_value = trace.get_trace("normal_memory")
    print(f"Normal trace value: {normal_value}")

    # Inject NaN
    print(f"\nInjecting NaN activity...")
    try:
        trace.update("poisoned_memory", activity=float('nan'))
        poisoned_value = trace.get_trace("poisoned_memory")

        print(f"\nResult:")
        print(f"  Poisoned trace value: {poisoned_value}")
        print(f"  Is NaN: {np.isnan(poisoned_value)}")

        # Try to use poisoned trace
        credits = trace.assign_credit(reward=10.0)
        print(f"  Credits: {credits}")

        if any(np.isnan(v) for v in credits.values()):
            print("\n[EXPLOIT SUCCESSFUL] NaN propagated through system!")
            print("Impact: All downstream calculations corrupted")
            return True
    except Exception as e:
        print(f"\n[EXPLOIT BLOCKED] Exception raised: {e}")
        return False

    print("\n[EXPLOIT FAILED] NaN was handled safely")
    return False


def exploit_medium3_time_overflow():
    """
    MEDIUM-3: Extreme time delta causes exponential overflow.

    Expected: Large dt should be rejected or clipped
    Actual: Traces either vanish instantly or explode to infinity
    """
    print("\n" + "="*70)
    print("EXPLOIT MEDIUM-3: Time Delta Overflow")
    print("="*70)

    trace = EligibilityTrace(tau_trace=1.0)
    trace.update("victim_memory", activity=1.0)
    initial_value = trace.get_trace("victim_memory")

    print(f"Initial trace value: {initial_value}")

    # Attack 1: Huge positive dt (instant decay)
    print(f"\nAttack 1: Applying dt = 1e10 (huge time step)...")
    try:
        trace.step(dt=1e10)
        after_value = trace.get_trace("victim_memory")
        print(f"  Trace value after: {after_value}")

        if after_value == 0.0 and initial_value > 0:
            print("\n[EXPLOIT SUCCESSFUL] Trace instantly decayed to zero!")
            print("Impact: All learning history erased (denial of service)")
            return True
    except Exception as e:
        print(f"\n[EXPLOIT BLOCKED] Exception raised: {e}")

    # Attack 2: Negative dt (not currently blocked)
    print(f"\nAttack 2: Attempting negative dt = -1000...")
    trace2 = EligibilityTrace(tau_trace=1.0)
    trace2.update("victim2", activity=1.0)

    try:
        trace2.step(dt=-1000)
        after_value = trace2.get_trace("victim2")
        print(f"  Trace value after negative dt: {after_value}")

        if np.isinf(after_value):
            print("\n[EXPLOIT SUCCESSFUL] Trace exploded to infinity!")
            print("Impact: Memory corruption, system instability")
            return True
    except Exception as e:
        print(f"\n[EXPLOIT BLOCKED] Exception raised: {e}")

    print("\n[EXPLOIT FAILED] Time delta handled safely")
    return False


def exploit_medium4_layered_capacity_bypass():
    """
    MEDIUM-4: LayeredEligibilityTrace bypasses max_traces limit.

    Expected: max_traces enforced
    Actual: Unlimited traces can be created
    """
    print("\n" + "="*70)
    print("EXPLOIT MEDIUM-4: Layered Trace Capacity Bypass")
    print("="*70)

    trace = LayeredEligibilityTrace(max_traces=100)

    print(f"Max traces limit: {trace.max_traces}")
    print(f"Creating 1000 unique traces...")

    for i in range(1000):
        trace.update(f"memory_{i}", activity=1.0)

    fast_count = len(trace.fast_traces)
    slow_count = len(trace.slow_traces)
    total_count = trace.count

    print(f"\nResult:")
    print(f"  Fast traces: {fast_count}")
    print(f"  Slow traces: {slow_count}")
    print(f"  Unique count: {total_count}")
    print(f"  Max allowed: {trace.max_traces}")

    if total_count > trace.max_traces:
        print(f"\n[EXPLOIT SUCCESSFUL] Created {total_count} traces, limit is {trace.max_traces}!")
        print("Impact: Unbounded memory growth, max_traces security control bypassed")
        return True
    else:
        print("\n[EXPLOIT FAILED] Capacity limit enforced")
        return False


def exploit_critical1_race_condition():
    """
    CRITICAL-1: Thread safety - concurrent access causes corruption.

    Expected: Thread-safe operations
    Actual: Race conditions lead to data corruption and crashes
    """
    print("\n" + "="*70)
    print("EXPLOIT CRITICAL-1: Thread Safety Race Condition")
    print("="*70)

    trace = EligibilityTrace(max_traces=1000)
    errors = []
    corruption_detected = False

    def updater(thread_id):
        """Continuously update traces."""
        try:
            for i in range(100):
                trace.update(f"thread_{thread_id}_mem_{i}", activity=1.0)
                time.sleep(0.0001)  # Tiny delay to increase contention
        except Exception as e:
            errors.append(f"Updater {thread_id}: {e}")
            traceback.print_exc()

    def stepper():
        """Continuously decay traces."""
        try:
            for i in range(100):
                trace.step(dt=0.01)
                time.sleep(0.0001)
        except Exception as e:
            errors.append(f"Stepper: {e}")
            traceback.print_exc()

    def assigner():
        """Continuously assign credits."""
        try:
            for i in range(100):
                credits = trace.assign_credit(reward=1.0)
                time.sleep(0.0001)
        except Exception as e:
            errors.append(f"Assigner: {e}")
            traceback.print_exc()

    print("Spawning 5 threads (3 updaters, 1 stepper, 1 assigner)...")
    threads = [
        threading.Thread(target=updater, args=(1,)),
        threading.Thread(target=updater, args=(2,)),
        threading.Thread(target=updater, args=(3,)),
        threading.Thread(target=stepper),
        threading.Thread(target=assigner),
    ]

    for t in threads:
        t.start()

    for t in threads:
        t.join()

    # Check for corruption
    final_count = trace.count
    stats = trace.get_stats()

    print(f"\nResult:")
    print(f"  Errors raised: {len(errors)}")
    print(f"  Final trace count: {final_count}")
    print(f"  Max traces: {trace.max_traces}")
    print(f"  Stats: {stats}")

    if errors:
        print(f"\n[EXPLOIT SUCCESSFUL] Thread safety violated!")
        print(f"Errors encountered:")
        for error in errors[:5]:  # Show first 5
            print(f"  - {error}")
        return True

    if final_count > trace.max_traces:
        print(f"\n[EXPLOIT SUCCESSFUL] max_traces limit bypassed via race condition!")
        print(f"  Expected: ≤ {trace.max_traces}")
        print(f"  Actual: {final_count}")
        return True

    # Check for NaN corruption
    if np.isnan(stats.get('mean_trace', 0)):
        print("\n[EXPLOIT SUCCESSFUL] Data corrupted (NaN detected)!")
        return True

    print("\n[EXPLOIT FAILED] System remained stable under concurrent access")
    return False


def main():
    """Run all exploit proof-of-concepts."""
    print("="*70)
    print("ELIGIBILITY TRACE SECURITY EXPLOIT PROOF-OF-CONCEPT")
    print("="*70)
    print("\nThis demonstrates security vulnerabilities in the system.")
    print("Each exploit attempts to trigger a specific vulnerability.\n")

    exploits = [
        ("CRITICAL-1: Race Condition", exploit_critical1_race_condition),
        ("HIGH-1: Reward Overflow", exploit_high1_reward_overflow),
        ("HIGH-2: Memory Exhaustion", exploit_high2_memory_exhaustion),
        ("MEDIUM-1: NaN Activity", exploit_medium1_nan_activity),
        ("MEDIUM-3: Time Overflow", exploit_medium3_time_overflow),
        ("MEDIUM-4: Capacity Bypass", exploit_medium4_layered_capacity_bypass),
    ]

    results = {}
    for name, exploit_func in exploits:
        try:
            results[name] = exploit_func()
        except Exception as e:
            print(f"\n[ERROR] Exploit {name} crashed: {e}")
            traceback.print_exc()
            results[name] = False

    # Summary
    print("\n" + "="*70)
    print("EXPLOIT SUMMARY")
    print("="*70)

    successful = sum(1 for success in results.values() if success)
    total = len(results)

    for name, success in results.items():
        status = "VULNERABLE" if success else "PROTECTED"
        print(f"  [{status}] {name}")

    print(f"\nTotal: {successful}/{total} exploits successful")

    if successful > 0:
        print("\nWARNING: System has exploitable vulnerabilities!")
        print("See security_analysis_eligibility_trace.md for remediation.")
    else:
        print("\nSUCCESS: All exploits blocked. System is secure.")

    return successful


if __name__ == "__main__":
    successful_exploits = main()
    sys.exit(0 if successful_exploits == 0 else 1)
