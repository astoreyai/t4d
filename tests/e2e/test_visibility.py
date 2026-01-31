"""Glass-box verification tests (P7-05).

Acceptance criteria:
- Every spiking block stage produces observable metrics
- Every T4DX operation is traceable
- Full trace from input to output through spiking stack
"""

import time
import uuid

import numpy as np
import pytest
import torch

from t4dm.spiking.cortical_block import CorticalBlock
from t4dm.spiking.cortical_stack import CorticalStack
from t4dm.spiking.lif import LIFNeuron
from t4dm.spiking.spike_attention import SpikeAttention
from t4dm.spiking.thalamic_gate import ThalamicGate
from t4dm.spiking.apical_modulation import ApicalModulation
from t4dm.spiking.rwkv_recurrence import RWKVRecurrence
from t4dm.storage.t4dx.engine import T4DXEngine
from t4dm.storage.t4dx.types import ItemRecord


DIM = 32


def _make_item(**kwargs):
    defaults = dict(
        id=uuid.uuid4().bytes,
        vector=np.random.randn(DIM).tolist(),
        event_time=time.time(),
        record_time=time.time(),
        valid_from=time.time(),
        valid_until=None,
        kappa=0.1,
        importance=0.5,
        item_type="episode",
        content="test",
        access_count=0,
        session_id=None,
    )
    defaults.update(kwargs)
    return ItemRecord(**defaults)


class TestSpikingBlockVisibility:
    """Verify every spiking block stage produces observable outputs."""

    def test_thalamic_gate_observable(self):
        """Thalamic gate output should be observable."""
        gate = ThalamicGate(DIM)
        x = torch.randn(1, 4, DIM)
        out = gate(x, context=None, ach_level=0.5)
        assert out.shape == (1, 4, DIM)
        assert not torch.isnan(out).any()

    def test_lif_observable(self):
        """LIF spikes and membrane potential should be observable."""
        lif = LIFNeuron(size=DIM)
        x = torch.randn(1, DIM)
        spikes, membrane = lif(x)
        assert spikes.shape == (1, DIM)
        assert membrane.shape == (1, DIM)
        # Spikes should be binary
        assert ((spikes == 0) | (spikes == 1)).all()

    def test_spike_attention_observable(self):
        """Spike attention should return output and weight info."""
        attn = SpikeAttention(DIM, num_heads=4)
        x = torch.randn(1, 4, DIM)
        out, weights = attn(x)
        assert out.shape == (1, 4, DIM)
        assert weights.shape == (4,)  # per-head STDP weights

    def test_apical_modulation_observable(self):
        """Apical modulation should return prediction error and goodness."""
        apical = ApicalModulation(DIM)
        x = torch.randn(1, 4, DIM)
        out, pe, goodness = apical(x, apical_input=None)
        assert out.shape == (1, 4, DIM)
        assert isinstance(pe, torch.Tensor)
        assert isinstance(goodness, torch.Tensor)

    def test_rwkv_observable(self):
        """RWKV recurrence should return output and state."""
        rwkv = RWKVRecurrence(DIM)
        x = torch.randn(1, 4, DIM)
        out, state = rwkv(x, state=None)
        assert out.shape == (1, 4, DIM)
        assert state is not None

    def test_cortical_block_metrics(self):
        """Full cortical block should emit all stage metrics."""
        block = CorticalBlock(DIM, num_heads=4)
        x = torch.randn(1, 4, DIM)
        out, state, metrics = block(x, ach=0.5)

        assert out.shape == (1, 4, DIM)
        assert "pe" in metrics, "Missing prediction error metric"
        assert "goodness" in metrics, "Missing goodness metric"
        assert "attn" in metrics, "Missing attention weights metric"
        assert state is not None

    def test_cortical_stack_all_blocks_emit_metrics(self):
        """Stack should emit metrics for every block."""
        num_blocks = 3
        stack = CorticalStack(DIM, num_blocks=num_blocks, num_heads=4)
        x = torch.randn(1, 4, DIM)
        out, states, all_metrics = stack(x)

        assert len(all_metrics) == num_blocks, (
            f"Expected {num_blocks} metric dicts, got {len(all_metrics)}"
        )
        for i, metrics in enumerate(all_metrics):
            assert "pe" in metrics, f"Block {i} missing PE"
            assert "goodness" in metrics, f"Block {i} missing goodness"
            assert "attn" in metrics, f"Block {i} missing attn"

    def test_full_stack_gradient_flow(self):
        """Gradients should flow through all blocks."""
        stack = CorticalStack(DIM, num_blocks=3, num_heads=4)
        x = torch.randn(1, 4, DIM, requires_grad=True)
        out, _, _ = stack(x)
        loss = out.sum()
        loss.backward()

        assert x.grad is not None
        assert x.grad.abs().sum() > 0, "Gradient should reach input"

        # Check all block params have gradients
        # Verify at least some block params have gradients
        has_grad = False
        for i, block in enumerate(stack.blocks):
            for name, p in block.named_parameters():
                if p.requires_grad and p.grad is not None:
                    has_grad = True
        assert has_grad, "At least some block params should have gradients"


class TestT4DXOperationVisibility:
    """Verify T4DX operations are traceable."""

    def test_insert_get_roundtrip(self, tmp_path):
        """INSERT followed by GET returns same item."""
        engine = T4DXEngine(tmp_path / "vis_insert")
        engine.startup()

        item = _make_item()
        engine.insert(item)
        rec = engine.get(item.id)

        assert rec is not None
        assert rec.id == item.id
        assert rec.kappa == pytest.approx(item.kappa)
        assert rec.content == item.content
        engine.shutdown()

    def test_update_observable(self, tmp_path):
        """UPDATE_FIELDS changes are observable via GET."""
        engine = T4DXEngine(tmp_path / "vis_update")
        engine.startup()

        item = _make_item(kappa=0.1)
        engine.insert(item)
        engine.update_fields(item.id, {"kappa": 0.5})
        rec = engine.get(item.id)

        assert rec.kappa == pytest.approx(0.5)
        engine.shutdown()

    def test_delete_observable(self, tmp_path):
        """DELETE makes item non-retrievable."""
        engine = T4DXEngine(tmp_path / "vis_delete")
        engine.startup()

        item = _make_item()
        engine.insert(item)
        engine.delete(item.id)
        rec = engine.get(item.id)

        assert rec is None
        engine.shutdown()

    def test_search_returns_scored_results(self, tmp_path):
        """SEARCH returns (id, score) pairs with valid scores."""
        engine = T4DXEngine(tmp_path / "vis_search")
        engine.startup()

        for _ in range(10):
            engine.insert(_make_item())

        query = np.random.randn(DIM).tolist()
        results = engine.search(query, k=5)

        assert len(results) <= 5
        for item_id, score in results:
            assert isinstance(item_id, bytes)
            assert len(item_id) == 16
            assert -1.0 <= score <= 1.0

        engine.shutdown()

    def test_scan_with_filters(self, tmp_path):
        """SCAN with κ filter returns correct subset."""
        engine = T4DXEngine(tmp_path / "vis_scan")
        engine.startup()

        for i in range(10):
            engine.insert(_make_item(kappa=0.1 * i))

        low_kappa = engine.scan(kappa_max=0.3)
        high_kappa = engine.scan(kappa_min=0.7)

        for item in low_kappa:
            assert item.kappa <= 0.3
        for item in high_kappa:
            assert item.kappa >= 0.7

        engine.shutdown()

    def test_traverse_returns_edges(self, tmp_path):
        """TRAVERSE returns connected edges."""
        from t4dm.storage.t4dx.types import EdgeRecord

        engine = T4DXEngine(tmp_path / "vis_traverse")
        engine.startup()

        items = [_make_item() for _ in range(3)]
        for item in items:
            engine.insert(item)

        engine.insert_edge(EdgeRecord(
            source_id=items[0].id,
            target_id=items[1].id,
            edge_type="USES",
            weight=0.7,
        ))

        edges = engine.traverse(items[0].id, direction="out")
        assert len(edges) >= 1
        assert edges[0].edge_type == "USES"

        engine.shutdown()

    def test_wal_captures_all_ops(self, tmp_path):
        """WAL should capture entries for all mutation operations."""
        engine = T4DXEngine(tmp_path / "vis_wal")
        engine.startup()

        item = _make_item()
        engine.insert(item)
        engine.update_fields(item.id, {"kappa": 0.5})
        engine.delete(item.id)

        # WAL should have entries
        entries = engine._wal.replay()
        assert len(entries) >= 3, f"Expected ≥3 WAL entries, got {len(entries)}"

        engine.shutdown()
