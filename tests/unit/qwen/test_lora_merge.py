"""Tests for LoRA merge/export (P3-08)."""

import pytest

from ww.qwen.lora_merge import merge_lora, save_lora_only


class TestLoraMerge:
    def test_merge_non_peft_passthrough(self, tiny_model):
        """Non-PEFT model should pass through unchanged."""
        result = merge_lora(tiny_model)
        assert result is tiny_model

    def test_save_lora_only_non_peft(self, tiny_model, tmp_path):
        """Should warn but not crash on non-PEFT model."""
        save_lora_only(tiny_model, tmp_path / "lora")
