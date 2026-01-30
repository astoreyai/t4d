"""
Theta-Gamma Coupling Integration.

P4-4: Connect oscillator system to learning and memory.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from typing import Any, Protocol
from uuid import UUID

from ww.nca.oscillators import (
    CognitivePhase,
    FrequencyBandGenerator,
    OscillatorConfig,
)

logger = logging.getLogger(__name__)


class PlasticityModuleProtocol(Protocol):
    def get_learning_rate(self) -> float: ...
    def set_learning_rate(self, lr: float) -> None: ...


@dataclass
class ThetaGammaConfig:
    plasticity_gating: bool = True
    encoding_boost: float = 2.0
    retrieval_suppression: float = 0.3
    pac_reward_learning: bool = True
    pac_reward_scale: float = 0.1
    enable_wm_slots: bool = True
    max_wm_items: int = 7
    alpha_inhibition_weight: float = 0.5
    # ATOM-P4-8: Configurable WM activation decay factor
    wm_decay: float = 0.9  # Default 0.9 decay per theta cycle
    oscillator_config: OscillatorConfig = field(default_factory=OscillatorConfig)


@dataclass
class WMSlot:
    slot_index: int
    theta_cycle: int
    gamma_phase: float
    content_id: UUID | None = None
    activation: float = 0.0


@dataclass
class ThetaGammaState:
    theta_cycle_count: int = 0
    current_cognitive_phase: CognitivePhase = CognitivePhase.ENCODING
    encoding_signal: float = 1.0
    retrieval_signal: float = 0.0
    pac_modulation_index: float = 0.0
    alpha_inhibition: float = 0.0
    wm_capacity: int = 7
    wm_slots: list[WMSlot] = field(default_factory=list)

    def to_dict(self) -> dict[str, Any]:
        return {
            "theta_cycle_count": self.theta_cycle_count,
            "cognitive_phase": self.current_cognitive_phase.value,
            "encoding_signal": self.encoding_signal,
            "retrieval_signal": self.retrieval_signal,
            "pac_mi": self.pac_modulation_index,
            "alpha_inhibition": self.alpha_inhibition,
            "wm_capacity": self.wm_capacity,
            "wm_slots_used": len([s for s in self.wm_slots if s.content_id]),
        }


class ThetaGammaIntegration:
    """Integrate theta-gamma oscillations with learning and memory."""

    def __init__(self, config: ThetaGammaConfig | None = None):
        self.config = config or ThetaGammaConfig()
        self.oscillator = FrequencyBandGenerator(self.config.oscillator_config)
        self.state = ThetaGammaState()

        self._last_theta_phase = 0.0
        self._theta_cycle_count = 0
        self._wm_slots: list[WMSlot] = []
        self._init_wm_slots()

        self._plasticity_modules: list[PlasticityModuleProtocol] = []
        self._base_learning_rates: dict[int, float] = {}

        self._total_steps = 0
        self._reward_events = 0
        self._wm_stores = 0
        self._wm_retrievals = 0

        # ATOM-P2-26: Encoding streak protection
        self._encoding_streak = 0
        self._max_encoding_streak = 10

        logger.info("ThetaGammaIntegration initialized")

    def _init_wm_slots(self) -> None:
        n_slots = self.oscillator.get_working_memory_capacity()
        self._wm_slots = [
            WMSlot(slot_index=i, theta_cycle=0, gamma_phase=0.0)
            for i in range(n_slots)
        ]
        self.state.wm_capacity = n_slots
        self.state.wm_slots = self._wm_slots

    def step(
        self,
        ach_level: float = 0.5,
        da_level: float = 0.5,
        ne_level: float = 0.3,
        glu_level: float = 0.5,
        gaba_level: float = 0.5,
        attention_level: float = 0.5,
        dt_ms: float = 1.0,
    ) -> dict[str, float]:
        outputs = self.oscillator.step(
            ach_level=ach_level,
            da_level=da_level,
            ne_level=ne_level,
            glu_level=glu_level,
            gaba_level=gaba_level,
            attention_level=attention_level,
            dt_ms=dt_ms,
        )

        theta_phase = outputs["theta_phase"]
        if theta_phase < self._last_theta_phase:
            self._theta_cycle_count += 1
            self._on_theta_cycle()
        self._last_theta_phase = theta_phase

        self.state.theta_cycle_count = self._theta_cycle_count
        self.state.current_cognitive_phase = CognitivePhase(outputs["cognitive_phase"])
        self.state.encoding_signal = self.oscillator.get_encoding_signal()
        self.state.retrieval_signal = self.oscillator.get_retrieval_signal()
        self.state.pac_modulation_index = self.oscillator.get_modulation_index()
        self.state.alpha_inhibition = outputs["alpha_inhibition"]

        gamma_phase = outputs["gamma_phase"]
        for slot in self._wm_slots:
            slot.gamma_phase = gamma_phase
            slot.theta_cycle = self._theta_cycle_count

        if self.config.plasticity_gating:
            self._gate_plasticity()

        self._total_steps += 1

        outputs["encoding_signal"] = self.state.encoding_signal
        outputs["retrieval_signal"] = self.state.retrieval_signal
        outputs["theta_cycle"] = self._theta_cycle_count

        return outputs

    def _on_theta_cycle(self) -> None:
        # ATOM-P4-8: Use configurable decay factor
        for slot in self._wm_slots:
            slot.activation *= self.config.wm_decay

    def _gate_plasticity(self) -> None:
        encoding = self.state.encoding_signal

        # ATOM-P2-26: Track encoding streak and reset if exceeded
        if self.state.current_cognitive_phase == CognitivePhase.ENCODING:
            self._encoding_streak += 1
            if self._encoding_streak > self._max_encoding_streak:
                logger.warning(
                    f"Encoding streak exceeded ({self._encoding_streak} > {self._max_encoding_streak}), "
                    f"resetting to balanced mode"
                )
                self._encoding_streak = 0
                # Force balanced mode by reducing encoding signal
                encoding = 0.5
        else:
            # Reset streak when not in encoding phase
            self._encoding_streak = 0

        for i, module in enumerate(self._plasticity_modules):
            if i in self._base_learning_rates:
                base_lr = self._base_learning_rates[i]
                if self.state.current_cognitive_phase == CognitivePhase.ENCODING:
                    gated_lr = base_lr * (1.0 + (self.config.encoding_boost - 1.0) * encoding)
                else:
                    gated_lr = base_lr * self.config.retrieval_suppression
                module.set_learning_rate(gated_lr)

    def register_plasticity_module(self, module: PlasticityModuleProtocol) -> int:
        module_id = len(self._plasticity_modules)
        self._plasticity_modules.append(module)
        self._base_learning_rates[module_id] = module.get_learning_rate()
        return module_id

    def get_gated_learning_rate(self, base_lr: float) -> float:
        if not self.config.plasticity_gating:
            return base_lr
        encoding = self.state.encoding_signal
        if self.state.current_cognitive_phase == CognitivePhase.ENCODING:
            return base_lr * (1.0 + (self.config.encoding_boost - 1.0) * encoding)
        else:
            return base_lr * self.config.retrieval_suppression

    def on_reward(self, rpe: float) -> None:
        if self.config.pac_reward_learning:
            scaled_reward = rpe * self.config.pac_reward_scale
            self.oscillator.update_pac_from_reward(scaled_reward)
            self._reward_events += 1

    def store_in_wm(self, content_id: UUID) -> bool:
        if not self.config.enable_wm_slots:
            return False

        min_activation = float("inf")
        best_slot = None

        for slot in self._wm_slots:
            if slot.content_id is None:
                best_slot = slot
                break
            elif slot.activation < min_activation:
                min_activation = slot.activation
                best_slot = slot

        if best_slot:
            best_slot.content_id = content_id
            best_slot.activation = 1.0
            best_slot.theta_cycle = self._theta_cycle_count
            best_slot.gamma_phase = self.oscillator.state.gamma_phase
            self._wm_stores += 1
            return True

        return False

    def retrieve_from_wm(self, content_id: UUID) -> bool:
        for slot in self._wm_slots:
            if slot.content_id == content_id:
                slot.activation = 1.0
                self._wm_retrievals += 1
                return True
        return False

    def get_wm_contents(self) -> list[UUID]:
        items = [
            (slot.content_id, slot.activation)
            for slot in self._wm_slots
            if slot.content_id is not None
        ]
        items.sort(key=lambda x: x[1], reverse=True)
        return [item[0] for item in items]

    def clear_wm(self) -> None:
        for slot in self._wm_slots:
            slot.content_id = None
            slot.activation = 0.0

    def get_inhibition_signal(self) -> float:
        return self.state.alpha_inhibition * self.config.alpha_inhibition_weight

    def get_statistics(self) -> dict[str, Any]:
        osc_stats = self.oscillator.get_stats()
        return {
            "total_steps": self._total_steps,
            "theta_cycles": self._theta_cycle_count,
            "reward_events": self._reward_events,
            "wm_stores": self._wm_stores,
            "wm_retrievals": self._wm_retrievals,
            "wm_items": len([s for s in self._wm_slots if s.content_id]),
            "wm_capacity": self.state.wm_capacity,
            "current_phase": self.state.current_cognitive_phase.value,
            "pac_strength": osc_stats["pac_strength"],
            "pac_mi": self.state.pac_modulation_index,
            "alpha_inhibition": self.state.alpha_inhibition,
        }

    def validate(self) -> dict[str, Any]:
        osc_validation = self.oscillator.validate_oscillations()
        results = {
            "oscillators_valid": osc_validation["all_pass"],
            "wm_capacity_valid": 4 <= self.state.wm_capacity <= 10,
            "encoding_signal_valid": 0.0 <= self.state.encoding_signal <= 1.0,
            "retrieval_signal_valid": 0.0 <= self.state.retrieval_signal <= 1.0,
            "signals_complementary": abs(
                self.state.encoding_signal + self.state.retrieval_signal - 1.0
            ) < 0.1,
        }
        results["all_pass"] = all(results.values())
        return results

    def save_state(self) -> dict[str, Any]:
        return {
            "theta_cycle_count": self._theta_cycle_count,
            "last_theta_phase": self._last_theta_phase,
            "wm_slots": [
                {
                    "slot_index": s.slot_index,
                    "content_id": str(s.content_id) if s.content_id else None,
                    "activation": s.activation,
                }
                for s in self._wm_slots
            ],
            "statistics": self.get_statistics(),
        }

    def load_state(self, state: dict[str, Any]) -> None:
        self._theta_cycle_count = state.get("theta_cycle_count", 0)
        self._last_theta_phase = state.get("last_theta_phase", 0.0)

        for i, slot_data in enumerate(state.get("wm_slots", [])):
            if i < len(self._wm_slots):
                self._wm_slots[i].content_id = (
                    UUID(slot_data["content_id"])
                    if slot_data.get("content_id")
                    else None
                )
                self._wm_slots[i].activation = slot_data.get("activation", 0.0)
