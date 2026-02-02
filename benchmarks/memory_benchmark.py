"""T4DM Memory System Benchmark.

Tests storage performance at various scales.
Usage: python benchmarks/memory_benchmark.py [--items 1000] [--dims 1024]
"""

from __future__ import annotations

import argparse
import sys
import tempfile
import time
import uuid
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from t4dm.storage.t4dx.engine import T4DXEngine
from t4dm.storage.t4dx.types import EdgeRecord, ItemRecord


def _make_item(dims: int) -> ItemRecord:
    now = time.time()
    return ItemRecord(
        id=uuid.uuid4().bytes,
        vector=np.random.randn(dims).astype(np.float32).tolist(),
        kappa=np.random.uniform(0.0, 1.0),
        importance=np.random.uniform(0.0, 1.0),
        event_time=now,
        record_time=now,
        valid_from=now,
        valid_until=None,
        item_type=np.random.choice(["episodic", "semantic", "procedural"]),
        content=f"benchmark item {uuid.uuid4().hex[:8]}",
        access_count=0,
        session_id="bench",
    )


def bench_insert(engine: T4DXEngine, n: int, dims: int) -> float:
    """Returns items/sec."""
    items = [_make_item(dims) for _ in range(n)]
    start = time.perf_counter()
    for item in items:
        engine.insert(item)
    elapsed = time.perf_counter() - start
    return n / elapsed if elapsed > 0 else float("inf")


def bench_search(engine: T4DXEngine, dims: int, k: int, n_queries: int = 50) -> float:
    """Returns average latency in ms."""
    queries = [np.random.randn(dims).astype(np.float32).tolist() for _ in range(n_queries)]
    start = time.perf_counter()
    for q in queries:
        engine.search(q, k=k)
    elapsed = time.perf_counter() - start
    return (elapsed / n_queries) * 1000


def bench_traverse(engine: T4DXEngine, item_ids: list[bytes], n_queries: int = 50) -> float:
    """Returns average latency in ms."""
    if not item_ids:
        return 0.0
    indices = np.random.randint(0, len(item_ids), size=n_queries)
    start = time.perf_counter()
    for idx in indices:
        engine.traverse(item_ids[idx])
    elapsed = time.perf_counter() - start
    return (elapsed / n_queries) * 1000


def bench_compaction(engine: T4DXEngine) -> dict[str, float]:
    """Returns compaction times in ms."""
    results = {}

    start = time.perf_counter()
    engine.flush()
    results["flush"] = (time.perf_counter() - start) * 1000

    start = time.perf_counter()
    engine.nrem_compact()
    results["nrem"] = (time.perf_counter() - start) * 1000

    start = time.perf_counter()
    engine.rem_compact()
    results["rem"] = (time.perf_counter() - start) * 1000

    return results


def _print_table(rows: list[list[str]], headers: list[str]) -> None:
    widths = [max(len(h), *(len(r[i]) for r in rows)) for i, h in enumerate(headers)]
    fmt = " | ".join(f"{{:<{w}}}" for w in widths)
    sep = "-+-".join("-" * w for w in widths)
    print(fmt.format(*headers))
    print(sep)
    for row in rows:
        print(fmt.format(*row))


def main() -> None:
    parser = argparse.ArgumentParser(description="T4DM Memory Benchmark")
    parser.add_argument("--items", type=int, default=1000, help="Max items to insert (scales: items/10, items, items*10 if <=10K)")
    parser.add_argument("--dims", type=int, default=1024, help="Vector dimensions")
    args = parser.parse_args()

    dims = args.dims
    base = args.items
    scales = [max(1, base // 10), base]
    if base <= 10000:
        scales.append(base * 10)

    print(f"T4DM Memory Benchmark (dims={dims})")
    print("=" * 60)

    # --- INSERT throughput ---
    print("\n## INSERT Throughput")
    rows = []
    for n in scales:
        with tempfile.TemporaryDirectory() as td:
            eng = T4DXEngine(td, flush_threshold=max(n + 1, 100000))
            eng.startup()
            rate = bench_insert(eng, n, dims)
            rows.append([str(n), f"{rate:,.0f}"])
            eng.shutdown()
    _print_table(rows, ["Items", "Items/sec"])

    # --- SEARCH latency ---
    print("\n## SEARCH Latency (ms)")
    rows = []
    for n in scales:
        with tempfile.TemporaryDirectory() as td:
            eng = T4DXEngine(td, flush_threshold=max(n + 1, 100000))
            eng.startup()
            items = [_make_item(dims) for _ in range(n)]
            for item in items:
                eng.insert(item)
            for k in [5, 10, 20]:
                lat = bench_search(eng, dims, k)
                rows.append([str(n), str(k), f"{lat:.2f}"])
            eng.shutdown()
    _print_table(rows, ["Items", "top-k", "Latency(ms)"])

    # --- TRAVERSE latency ---
    print("\n## TRAVERSE Latency (ms)")
    rows = []
    for n in scales:
        with tempfile.TemporaryDirectory() as td:
            eng = T4DXEngine(td, flush_threshold=max(n + 1, 100000))
            eng.startup()
            items = [_make_item(dims) for _ in range(n)]
            for item in items:
                eng.insert(item)
            # Add some edges
            for i in range(min(n - 1, 500)):
                edge = EdgeRecord(
                    source_id=items[i].id,
                    target_id=items[i + 1].id,
                    edge_type="SEQUENCE",
                    weight=0.5,
                )
                eng.insert_edge(edge)
            lat = bench_traverse(eng, [it.id for it in items])
            rows.append([str(n), f"{lat:.2f}"])
            eng.shutdown()
    _print_table(rows, ["Items", "Latency(ms)"])

    # --- Compaction ---
    print("\n## Compaction Time (ms)")
    with tempfile.TemporaryDirectory() as td:
        eng = T4DXEngine(td, flush_threshold=100000)
        eng.startup()
        items = [_make_item(dims) for _ in range(base)]
        for item in items:
            eng.insert(item)
        times = bench_compaction(eng)
        rows = [[k, f"{v:.2f}"] for k, v in times.items()]
        _print_table(rows, ["Phase", "Time(ms)"])
        eng.shutdown()

    print("\nDone.")


if __name__ == "__main__":
    main()
