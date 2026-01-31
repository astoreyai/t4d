"""Tests for neuromodulator bus."""

import pytest
from datetime import datetime

from ww.learning.neuromodulators import NeuromodulatorState
from ww.spiking.neuromod_bus import NeuromodBus


class TestNeuromodBus:
    @pytest.fixture
    def bus(self):
        return NeuromodBus()

    @pytest.fixture
    def state(self):
        return NeuromodulatorState(
            dopamine_rpe=0.7,
            norepinephrine_gain=1.3,
            acetylcholine_mode="encoding",
            serotonin_mood=0.6,
            inhibition_sparsity=0.4,
        )

    def test_correct_routing(self, bus, state):
        mods = bus.get_layer_modulation(state)
        assert mods["thalamic_ach"] == "encoding"
        assert mods["lif_ne_gain"] == 1.3
        assert mods["attention_da_lr"] == 0.7
        assert mods["apical_da"] == 0.7
        assert mods["rwkv_5ht_patience"] == 0.6
        assert mods["output_ne_gain"] == 1.3

    def test_ach_level_encoding(self, bus, state):
        assert bus.ach_level(state) == 0.8

    def test_ach_level_retrieval(self, bus):
        state = NeuromodulatorState(
            dopamine_rpe=0.0,
            norepinephrine_gain=1.0,
            acetylcholine_mode="retrieval",
            serotonin_mood=0.5,
            inhibition_sparsity=0.5,
        )
        assert bus.ach_level(state) == 0.3

    def test_ach_level_balanced(self, bus):
        state = NeuromodulatorState(
            dopamine_rpe=0.0,
            norepinephrine_gain=1.0,
            acetylcholine_mode="balanced",
            serotonin_mood=0.5,
            inhibition_sparsity=0.5,
        )
        assert bus.ach_level(state) == 0.5
