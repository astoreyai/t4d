"""Shared fixtures for consolidation tests."""

from __future__ import annotations

import sys
import time
import uuid
from pathlib import Path

import numpy as np
import pytest
import torch.nn as nn

sys.path.insert(0, str(Path(__file__).resolve().parents[3] / "src"))

from t4dm.storage.t4dx.engine import T4DXEngine
from t4dm.storage.t4dx.types import EdgeRecord, ItemRecord


def make_item(
    dim: int = 32,
    kappa: float = 0.0,
    importance: float = 0.5,
    item_type: str = "episodic",
    content: str = "test",
) -> ItemRecord:
    return ItemRecord(
        id=uuid.uuid4().bytes,
        vector=np.random.randn(dim).tolist(),
        kappa=kappa,
        importance=importance,
        event_time=time.time(),
        record_time=time.time(),
        valid_from=time.time(),
        valid_until=None,
        item_type=item_type,
        content=content,
        access_count=1,
        session_id="test",
    )


@pytest.fixture
def engine(tmp_path):
    e = T4DXEngine(tmp_path / "data", flush_threshold=1000)
    e.startup()
    yield e
    e.shutdown()


@pytest.fixture
def populated_engine(engine):
    """Engine with mixed-kappa items and edges."""
    items = []
    for i in range(20):
        kappa = (i % 5) * 0.2  # 0.0, 0.2, 0.4, 0.6, 0.8
        item = make_item(kappa=kappa, importance=0.3 + i * 0.03, content=f"item-{i}")
        engine.insert(item)
        items.append(item)

    # Add some edges
    for i in range(len(items) - 1):
        edge = EdgeRecord(
            source_id=items[i].id,
            target_id=items[i + 1].id,
            edge_type="SEQUENCE",
            weight=0.5,
        )
        engine.insert_edge(edge)

    return engine, items
