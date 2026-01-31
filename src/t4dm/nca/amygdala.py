"""
Amygdala Emotional Valence and Salience Circuit for World Weaver.

Biological Basis:
- Amygdala processes emotional significance and salience
- Central to emotional memory encoding (McGaugh 2004)
- Basolateral amygdala (BLA) modulates hippocampal memory consolidation
- Central amygdala (CeA) drives autonomic and behavioral responses
- Emotional arousal enhances memory encoding strength

Key Features:
- Valence detection (positive/negative emotional tone)
- Salience computation (how important/attention-worthy)
- Emotional tagging of memories for prioritized consolidation
- Fear learning and extinction
- Integration with stress and arousal systems

Architecture:
- Receives sensory input for rapid threat detection
- Computes valence and arousal from input features
- Tags memories with emotional weight
- Sends salience signals to NBM (acetylcholine), LC (norepinephrine)
- Modulates hippocampal encoding via emotional arousal

Integration:
- Memory encoding: emotional salience boosts encoding strength
- NBM: drives cholinergic attention to salient events
- LC: triggers noradrenergic arousal response
- Hippocampus: BLA→HPC pathway enhances consolidation

References:
- McGaugh (2004): The amygdala modulates the consolidation of memories
- LeDoux (2000): Emotion circuits in the brain
- Phelps (2004): Human emotion and memory
- Roozendaal et al. (2009): Stress and memory consolidation
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from enum import Enum

import numpy as np

logger = logging.getLogger(__name__)


class EmotionalState(Enum):
    """Emotional state classifications."""
    NEUTRAL = "neutral"
    POSITIVE = "positive"      # Positive valence
    NEGATIVE = "negative"      # Negative valence (threat/fear)
    AMBIVALENT = "ambivalent"  # Mixed valence


@dataclass
class AmygdalaConfig:
    """Configuration for amygdala dynamics."""

    # Valence sensitivity
    valence_sensitivity: float = 0.8    # How strongly valence is detected
    valence_threshold: float = 0.2      # Minimum for emotional classification

    # Salience computation
    salience_threshold: float = 0.3     # Minimum salience for tagging
    salience_decay: float = 0.1         # Salience decay rate

    # Fear learning
    fear_learning_rate: float = 0.3     # Rate of fear association learning
    fear_extinction_rate: float = 0.05  # Rate of fear extinction
    fear_threshold: float = 0.5         # Threshold for fear response

    # Emotional memory tagging
    encoding_boost_max: float = 2.0     # Max encoding strength multiplier
    arousal_to_encoding: float = 0.8    # How much arousal boosts encoding

    # Temporal dynamics
    tau_valence: float = 0.3            # Valence integration time constant
    tau_arousal: float = 0.5            # Arousal decay time constant

    # Biological constraints
    max_arousal: float = 1.0
    min_arousal: float = 0.0


@dataclass
class AmygdalaState:
    """Current state of amygdala circuit."""

    emotional_state: EmotionalState = EmotionalState.NEUTRAL
    valence: float = 0.0               # Current valence [-1=negative, +1=positive]
    arousal: float = 0.0               # Current arousal/activation [0, 1]
    salience_signal: float = 0.0       # Salience output [0, 1]

    # Fear state
    fear_level: float = 0.0            # Current fear activation [0, 1]
    fear_associations: dict[str, float] = None  # Stimulus -> fear mapping

    # Memory tagging
    encoding_modulation: float = 1.0   # Memory encoding strength multiplier

    def __post_init__(self):
        if self.fear_associations is None:
            self.fear_associations = {}

    def to_dict(self) -> dict:
        """Convert to dictionary for serialization."""
        return {
            "emotional_state": self.emotional_state.value,
            "valence": self.valence,
            "arousal": self.arousal,
            "salience_signal": self.salience_signal,
            "fear_level": self.fear_level,
            "encoding_modulation": self.encoding_modulation,
        }


class AmygdalaCircuit:
    """
    Amygdala emotional processing circuit.

    Models the dynamics of amygdala nuclei that:
    1. Detect emotional valence (positive/negative)
    2. Compute salience for attention allocation
    3. Tag memories with emotional weight for prioritization
    4. Learn and extinguish fear associations
    5. Modulate memory encoding strength via arousal

    Key innovation: Emotional salience as memory prioritization signal,
    implementing McGaugh's (2004) model of arousal-enhanced consolidation.
    """

    def __init__(self, config: AmygdalaConfig | None = None):
        """
        Initialize amygdala circuit.

        Args:
            config: Amygdala configuration parameters
        """
        self.config = config or AmygdalaConfig()
        self.state = AmygdalaState()

        # Track history
        self._valence_history: list[float] = []
        self._arousal_history: list[float] = []
        self._max_history = 1000

        logger.info(
            f"AmygdalaCircuit initialized: valence_sensitivity={self.config.valence_sensitivity}, "
            f"fear_learning_rate={self.config.fear_learning_rate}"
        )

    # =========================================================================
    # Core Emotional Processing
    # =========================================================================

    def process_sensory_input(
        self,
        input_features: np.ndarray | None = None,
        valence: float | None = None,
        arousal: float | None = None,
        dt: float = 0.1,
    ) -> tuple[float, float, float]:
        """
        Process sensory input to extract emotional content.

        Args:
            input_features: Raw sensory features (for internal valence detection)
            valence: Explicit valence score [-1, 1] (if pre-computed)
            arousal: Explicit arousal score [0, 1] (if pre-computed)
            dt: Timestep (seconds)

        Returns:
            (valence, arousal, salience) tuple
        """
        # Compute or use provided valence
        if valence is not None:
            detected_valence = float(np.clip(valence, -1, 1))
        elif input_features is not None:
            detected_valence = self._detect_valence(input_features)
        else:
            detected_valence = 0.0

        # Compute or use provided arousal
        if arousal is not None:
            detected_arousal = float(np.clip(arousal, 0, 1))
        elif input_features is not None:
            detected_arousal = self._detect_arousal(input_features)
        else:
            detected_arousal = 0.0

        # Update internal state
        self._update_valence(detected_valence, dt)
        self._update_arousal(detected_arousal, dt)

        # Compute salience
        salience = self._compute_salience()
        self.state.salience_signal = salience

        # Update emotional state classification
        self._classify_emotional_state()

        # Update encoding modulation
        self._update_encoding_modulation()

        # Track history
        self._valence_history.append(self.state.valence)
        self._arousal_history.append(self.state.arousal)
        if len(self._valence_history) > self._max_history:
            self._valence_history = self._valence_history[-self._max_history:]
            self._arousal_history = self._arousal_history[-self._max_history:]

        return self.state.valence, self.state.arousal, salience

    def _detect_valence(self, features: np.ndarray) -> float:
        """
        Detect emotional valence from sensory features.

        Simplified model: projects features to valence dimension.

        Args:
            features: Input feature vector

        Returns:
            Valence score [-1, 1]
        """
        # Simple projection (in real system, would use learned weights)
        # Positive if mean > 0.5, negative otherwise
        mean_activation = float(np.mean(features))
        valence = (mean_activation - 0.5) * 2 * self.config.valence_sensitivity
        return float(np.clip(valence, -1, 1))

    def _detect_arousal(self, features: np.ndarray) -> float:
        """
        Detect arousal from sensory features.

        Args:
            features: Input feature vector

        Returns:
            Arousal score [0, 1]
        """
        # Arousal related to variance/intensity
        intensity = float(np.std(features))
        arousal = intensity * self.config.valence_sensitivity
        return float(np.clip(arousal, 0, 1))

    def _update_valence(self, detected_valence: float, dt: float) -> None:
        """Update internal valence state with temporal smoothing."""
        alpha = dt / self.config.tau_valence
        self.state.valence += alpha * (detected_valence - self.state.valence)
        self.state.valence = float(np.clip(self.state.valence, -1, 1))

    def _update_arousal(self, detected_arousal: float, dt: float) -> None:
        """Update internal arousal state with decay."""
        alpha = dt / self.config.tau_arousal
        self.state.arousal += alpha * (detected_arousal - self.state.arousal)
        self.state.arousal = float(np.clip(
            self.state.arousal,
            self.config.min_arousal,
            self.config.max_arousal
        ))

    def _compute_salience(self) -> float:
        """
        Compute salience signal from valence and arousal.

        Salience = |valence| * arousal (extremity × intensity)

        Returns:
            Salience [0, 1]
        """
        # Salience is magnitude of valence weighted by arousal
        valence_magnitude = abs(self.state.valence)
        salience = valence_magnitude * self.state.arousal

        # Apply threshold
        if salience < self.config.salience_threshold:
            salience = 0.0

        return float(np.clip(salience, 0, 1))

    def _classify_emotional_state(self) -> None:
        """Classify current emotional state."""
        v = self.state.valence
        threshold = self.config.valence_threshold

        if abs(v) < threshold:
            self.state.emotional_state = EmotionalState.NEUTRAL
        elif v > threshold:
            self.state.emotional_state = EmotionalState.POSITIVE
        elif v < -threshold:
            self.state.emotional_state = EmotionalState.NEGATIVE
        else:
            self.state.emotional_state = EmotionalState.AMBIVALENT

    def _update_encoding_modulation(self) -> None:
        """
        Update memory encoding modulation based on emotional arousal.

        McGaugh (2004): Emotional arousal enhances memory consolidation.
        """
        # Encoding boost proportional to arousal
        boost = 1.0 + (self.config.arousal_to_encoding * self.state.arousal)
        self.state.encoding_modulation = float(np.clip(
            boost,
            1.0,
            self.config.encoding_boost_max
        ))

    # =========================================================================
    # Fear Learning
    # =========================================================================

    def learn_fear_association(
        self,
        stimulus_id: str,
        threat_level: float,
    ) -> float:
        """
        Learn fear association to stimulus.

        Args:
            stimulus_id: Identifier for stimulus
            threat_level: Perceived threat [0, 1]

        Returns:
            Updated fear level for this stimulus
        """
        # Get current fear level for this stimulus
        current_fear = self.state.fear_associations.get(stimulus_id, 0.0)

        # Hebbian-like learning: strengthen association
        delta_fear = self.config.fear_learning_rate * (threat_level - current_fear)

        # Update fear association
        new_fear = float(np.clip(current_fear + delta_fear, 0, 1))
        self.state.fear_associations[stimulus_id] = new_fear

        # Update global fear level
        self.state.fear_level = max(self.state.fear_level, new_fear)

        return new_fear

    def extinguish_fear(
        self,
        stimulus_id: str,
        safety_signal: float = 1.0,
    ) -> float:
        """
        Extinguish fear association through safety learning.

        Args:
            stimulus_id: Identifier for stimulus
            safety_signal: Safety evidence [0, 1]

        Returns:
            Updated fear level
        """
        if stimulus_id not in self.state.fear_associations:
            return 0.0

        current_fear = self.state.fear_associations[stimulus_id]

        # Extinction: gradual decrease with safety evidence
        delta_fear = -self.config.fear_extinction_rate * safety_signal * current_fear

        new_fear = float(np.clip(current_fear + delta_fear, 0, 1))
        self.state.fear_associations[stimulus_id] = new_fear

        # Update global fear level
        self.state.fear_level = max(self.state.fear_associations.values(), default=0.0)

        return new_fear

    # =========================================================================
    # Memory Tagging
    # =========================================================================

    def tag_memory(self, memory_id: str, valence: float | None = None) -> dict:
        """
        Tag memory with emotional weight for prioritized consolidation.

        Args:
            memory_id: Memory identifier
            valence: Optional explicit valence (uses current if None)

        Returns:
            Emotional tag metadata
        """
        use_valence = valence if valence is not None else self.state.valence

        tag = {
            "memory_id": memory_id,
            "valence": use_valence,
            "arousal": self.state.arousal,
            "salience": self.state.salience_signal,
            "encoding_boost": self.state.encoding_modulation,
            "emotional_state": self.state.emotional_state.value,
            "timestamp": 0.0,  # Would be set by caller
        }

        return tag

    def get_consolidation_priority(self, memory_id: str) -> float:
        """
        Get consolidation priority for emotionally-tagged memory.

        High arousal + high salience = high priority.

        Args:
            memory_id: Memory identifier

        Returns:
            Priority score [0, 1]
        """
        # Priority is combination of salience and arousal
        priority = 0.5 * self.state.salience_signal + 0.5 * self.state.arousal
        return float(np.clip(priority, 0, 1))

    # =========================================================================
    # Integration Methods
    # =========================================================================

    def receive_sensory(
        self,
        sensory_input: np.ndarray | None = None,
        valence: float | None = None,
        arousal: float | None = None,
    ) -> tuple[float, float]:
        """
        Receive and process sensory input.

        Args:
            sensory_input: Raw sensory features
            valence: Explicit valence (optional)
            arousal: Explicit arousal (optional)

        Returns:
            (valence, arousal) tuple
        """
        v, a, _ = self.process_sensory_input(
            input_features=sensory_input,
            valence=valence,
            arousal=arousal,
        )
        return v, a

    def emit_salience(self) -> float:
        """
        Emit salience signal for downstream targets (NBM, LC).

        Returns:
            Salience [0, 1]
        """
        return self.state.salience_signal

    def get_encoding_boost(self) -> float:
        """
        Get memory encoding strength boost.

        Returns:
            Encoding multiplier [1.0, encoding_boost_max]
        """
        return self.state.encoding_modulation

    def step(self, dt: float = 0.1) -> None:
        """
        Advance amygdala dynamics by one timestep.

        Handles decay of arousal and salience.

        Args:
            dt: Timestep in seconds
        """
        # Decay arousal
        decay_rate = dt / self.config.tau_arousal
        self.state.arousal *= (1 - decay_rate)

        # Decay salience
        self.state.salience_signal *= (1 - self.config.salience_decay * dt)

        # Decay fear gradually
        for stim_id in list(self.state.fear_associations.keys()):
            self.extinguish_fear(stim_id, safety_signal=0.1)

        # Update encoding modulation
        self._update_encoding_modulation()

        # Update state classification
        self._classify_emotional_state()

    # =========================================================================
    # Statistics and State
    # =========================================================================

    def get_stats(self) -> dict:
        """Get amygdala circuit statistics."""
        stats = {
            "emotional_state": self.state.emotional_state.value,
            "valence": self.state.valence,
            "arousal": self.state.arousal,
            "salience_signal": self.state.salience_signal,
            "fear_level": self.state.fear_level,
            "n_fear_associations": len(self.state.fear_associations),
            "encoding_modulation": self.state.encoding_modulation,
        }

        if self._valence_history:
            stats["avg_valence"] = float(np.mean(self._valence_history))
        if self._arousal_history:
            stats["avg_arousal"] = float(np.mean(self._arousal_history))

        return stats

    def reset(self) -> None:
        """Reset to initial state."""
        self.state = AmygdalaState()
        self._valence_history.clear()
        self._arousal_history.clear()
        logger.info("AmygdalaCircuit reset")

    def save_state(self) -> dict:
        """Save state for persistence."""
        return {
            "state": self.state.to_dict(),
            "fear_associations": self.state.fear_associations.copy(),
            "config": {
                "valence_sensitivity": self.config.valence_sensitivity,
                "fear_learning_rate": self.config.fear_learning_rate,
            }
        }

    def load_state(self, saved: dict) -> None:
        """Load state from persistence."""
        if "state" in saved:
            s = saved["state"]
            self.state.valence = s.get("valence", 0.0)
            self.state.arousal = s.get("arousal", 0.0)
            self.state.fear_level = s.get("fear_level", 0.0)

        if "fear_associations" in saved:
            self.state.fear_associations = saved["fear_associations"]


def create_amygdala_circuit(
    valence_sensitivity: float = 0.8,
    fear_learning_rate: float = 0.3,
) -> AmygdalaCircuit:
    """
    Factory function to create amygdala circuit with common configurations.

    Args:
        valence_sensitivity: Sensitivity to emotional valence
        fear_learning_rate: Rate of fear association learning

    Returns:
        Configured AmygdalaCircuit
    """
    config = AmygdalaConfig(
        valence_sensitivity=valence_sensitivity,
        fear_learning_rate=fear_learning_rate,
    )
    return AmygdalaCircuit(config)


__all__ = [
    "AmygdalaCircuit",
    "AmygdalaConfig",
    "AmygdalaState",
    "EmotionalState",
    "create_amygdala_circuit",
]
