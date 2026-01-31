"""Shared fixtures for Qwen tests — uses tiny mock models to avoid downloads."""

from __future__ import annotations

import sys
from pathlib import Path
from unittest.mock import MagicMock

import torch
import torch.nn as nn
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[3] / "src"))


class TinyTransformerLayer(nn.Module):
    """Minimal transformer layer for testing."""

    def __init__(self, dim: int):
        super().__init__()
        self.norm = nn.LayerNorm(dim)
        self.linear = nn.Linear(dim, dim)

    def forward(self, x, attention_mask=None, **kwargs):
        return (self.linear(self.norm(x)),)


class TinyQwenModel(nn.Module):
    """Minimal model mimicking Qwen2 structure for testing."""

    def __init__(self, dim: int = 64, num_layers: int = 4, vocab_size: int = 100):
        super().__init__()
        self.config = MagicMock()
        self.config.hidden_size = dim
        self.config.num_hidden_layers = num_layers
        self.config.vocab_size = vocab_size

        self.model = nn.Module()
        self.model.embed_tokens = nn.Embedding(vocab_size, dim)
        self.model.layers = nn.ModuleList(
            [TinyTransformerLayer(dim) for _ in range(num_layers)]
        )
        self.model.norm = nn.LayerNorm(dim)
        self.lm_head = nn.Linear(dim, vocab_size, bias=False)

    def forward(self, input_ids, attention_mask=None, **kwargs):
        x = self.model.embed_tokens(input_ids)
        for layer in self.model.layers:
            x = layer(x, attention_mask=attention_mask)[0]
        x = self.model.norm(x)
        logits = self.lm_head(x)
        return MagicMock(logits=logits, hidden_states=None)

    def parameters(self, recurse=True):
        return super().parameters(recurse)

    def named_parameters(self, prefix='', recurse=True):
        return super().named_parameters(prefix, recurse)


class TinyTokenizer:
    """Minimal tokenizer mock."""

    def __init__(self, vocab_size: int = 100):
        self.vocab_size = vocab_size
        self.pad_token = "<pad>"
        self.eos_token = "<eos>"
        self.pad_token_id = 0
        self.eos_token_id = 1

    def __call__(self, text, return_tensors="pt", **kwargs):
        # Return object with .to() method mimicking BatchEncoding
        ids = torch.randint(2, self.vocab_size, (1, 10))
        mask = torch.ones_like(ids)

        class TokenizerOutput(dict):
            def to(self, device):
                return self
        return TokenizerOutput(input_ids=ids, attention_mask=mask)

    def decode(self, ids, skip_special_tokens=True):
        return "generated text"


@pytest.fixture
def tiny_model():
    return TinyQwenModel(dim=64, num_layers=4, vocab_size=100)


@pytest.fixture
def tiny_tokenizer():
    return TinyTokenizer()
