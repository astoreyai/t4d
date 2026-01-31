"""Tests for QLoRA rank search (P3-11)."""

import pytest

from ww.qwen.rank_search import (
    RankSearchConfig,
    estimate_lora_params,
    estimate_vram_mb,
    run_rank_search,
)


class TestRankSearch:
    def test_estimate_lora_params(self):
        # 36 layers, 2 targets, r=16, dim=2048
        params = estimate_lora_params(2048, 36, 16, 2)
        assert params == 36 * 2 * 2 * 2048 * 16
        assert params > 0

    def test_higher_rank_more_params(self):
        p8 = estimate_lora_params(2048, 36, 8)
        p16 = estimate_lora_params(2048, 36, 16)
        p32 = estimate_lora_params(2048, 36, 32)
        assert p8 < p16 < p32

    def test_estimate_vram(self):
        params = estimate_lora_params(2048, 36, 16)
        vram = estimate_vram_mb(2000.0, params)
        assert vram > 2000.0

    def test_run_rank_search(self):
        cfg = RankSearchConfig(ranks=[4, 8, 16])
        results = run_rank_search(None, cfg, hidden_dim=64, num_layers=4)
        assert len(results) == 3
        assert results[0].rank == 4
        assert results[1].rank == 8
        assert results[0].trainable_params < results[1].trainable_params
