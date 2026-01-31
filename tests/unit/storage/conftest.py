"""Shared fixtures for T4DX storage tests."""

from __future__ import annotations

import sys
import time
import uuid
from pathlib import Path

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[3] / "src"))

from t4dm.storage.t4dx.types import EdgeRecord, ItemRecord


def make_item(
    dim: int = 8,
    kappa: float = 0.0,
    item_type: str = "episodic",
    event_time: float | None = None,
    content: str = "test content",
    importance: float = 0.5,
) -> ItemRecord:
    vec = np.random.randn(dim).tolist()
    now = event_time or time.time()
    return ItemRecord(
        id=uuid.uuid4().bytes,
        vector=vec,
        kappa=kappa,
        importance=importance,
        event_time=now,
        record_time=now,
        valid_from=now,
        valid_until=None,
        item_type=item_type,
        content=content,
        access_count=0,
        session_id="test-session",
    )


def make_edge(
    source: ItemRecord,
    target: ItemRecord,
    edge_type: str = "USES",
    weight: float = 0.5,
) -> EdgeRecord:
    return EdgeRecord(
        source_id=source.id,
        target_id=target.id,
        edge_type=edge_type,
        weight=weight,
    )


@pytest.fixture
def tmp_data_dir(tmp_path: Path) -> Path:
    d = tmp_path / "t4dx_data"
    d.mkdir()
    return d


@pytest.fixture
def sample_items() -> list[ItemRecord]:
    np.random.seed(42)
    return [make_item(content=f"item {i}", kappa=i * 0.2) for i in range(5)]


@pytest.fixture
def sample_edges(sample_items: list[ItemRecord]) -> list[EdgeRecord]:
    return [
        make_edge(sample_items[0], sample_items[1]),
        make_edge(sample_items[1], sample_items[2], edge_type="CAUSES"),
    ]
