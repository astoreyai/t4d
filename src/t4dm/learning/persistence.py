"""
Persistence layer for learned component state.

This module handles saving/loading learned parameters for:
- LearnedMemoryGate: Bayesian logistic regression weights
- LearnedRetrievalScorer: MLP fusion weights
- NeuromodulatorOrchestra: Dopamine expectations, serotonin traces

Key insight: Persistence enables resuming learning across sessions,
avoiding cold start from scratch every time.

Usage:
    persister = StatePersister(storage_path="~/.ww/learned_state")

    # Save state
    persister.save_gate_state(learned_gate)

    # Load state (returns None if not found)
    state = persister.load_gate_state()
    if state:
        learned_gate.restore_from_state(state)
"""

from __future__ import annotations

import gzip
import json
import logging
import pickle
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from typing import TYPE_CHECKING, Any

import numpy as np

if TYPE_CHECKING:
    from t4dm.core.learned_gate import LearnedMemoryGate
    from t4dm.learning.neuromodulators import NeuromodulatorOrchestra
    from t4dm.learning.scorer import LearnedRetrievalScorer

logger = logging.getLogger(__name__)


@dataclass
class LearnedGateState:
    """Serializable state for LearnedMemoryGate."""

    # Model parameters
    weight_mean: np.ndarray  # μ - weight means
    weight_covariance: np.ndarray  # Σ - variances (diagonal) or full covariance
    bias: float  # b - bias term
    use_diagonal: bool  # Whether covariance is diagonal

    # Training state
    n_observations: int

    # Statistics
    decisions: dict[str, int]  # store/buffer/skip counts

    # Metadata
    saved_at: datetime = field(default_factory=datetime.now)
    version: str = "1.0.0"

    def to_dict(self) -> dict[str, Any]:
        """Convert to JSON-serializable dict (numpy arrays -> lists)."""
        return {
            "weight_mean": self.weight_mean.tolist(),
            "weight_covariance": self.weight_covariance.tolist(),
            "bias": self.bias,
            "use_diagonal": self.use_diagonal,
            "n_observations": self.n_observations,
            "decisions": self.decisions,
            "saved_at": self.saved_at.isoformat(),
            "version": self.version,
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> LearnedGateState:
        """Reconstruct from dict."""
        return cls(
            weight_mean=np.array(data["weight_mean"], dtype=np.float32),
            weight_covariance=np.array(data["weight_covariance"], dtype=np.float32),
            bias=data["bias"],
            use_diagonal=data["use_diagonal"],
            n_observations=data["n_observations"],
            decisions=data["decisions"],
            saved_at=datetime.fromisoformat(data["saved_at"]),
            version=data.get("version", "1.0.0"),
        )


@dataclass
class ScorerState:
    """Serializable state for LearnedRetrievalScorer."""

    # MLP weights (layer name -> weight matrix)
    layer_weights: dict[str, np.ndarray]
    layer_biases: dict[str, np.ndarray]

    # Training statistics
    n_training_steps: int
    recent_losses: list[float]

    # Metadata
    saved_at: datetime = field(default_factory=datetime.now)
    version: str = "1.0.0"


@dataclass
class NeuromodulatorState:
    """Serializable state for neuromodulator systems."""

    # Dopamine: memory_id -> expected_value
    dopamine_expectations: dict[str, float]

    # Serotonin: memory_id -> long_term_value
    serotonin_values: dict[str, float]
    serotonin_mood: float

    # NE: reference distribution for novelty
    ne_reference_mean: np.ndarray | None
    ne_reference_std: np.ndarray | None

    # ACh: baseline mode
    ach_baseline_mode: str  # "encoding" / "balanced" / "retrieval"

    # Metadata
    saved_at: datetime = field(default_factory=datetime.now)
    version: str = "1.0.0"


class StatePersister:
    """
    Persists learned state to disk for cross-session continuity.

    Uses JSON for human-readable state and gzip for compression.
    Falls back gracefully if state doesn't exist or is corrupted.
    """

    def __init__(
        self,
        storage_path: str | None = None,
        compress: bool = True
    ):
        """
        Initialize state persister.

        Args:
            storage_path: Directory for state files (default: ~/.ww/learned_state)
            compress: Whether to gzip compress state files
        """
        if storage_path:
            self.storage_path = Path(storage_path).expanduser()
        else:
            self.storage_path = Path.home() / ".ww" / "learned_state"

        self.compress = compress
        self.storage_path.mkdir(parents=True, exist_ok=True)

        logger.info(f"StatePersister initialized at {self.storage_path}")

    def _get_path(self, name: str) -> Path:
        """Get path for a state file."""
        ext = ".json.gz" if self.compress else ".json"
        return self.storage_path / f"{name}{ext}"

    def _save_json(self, path: Path, data: dict[str, Any]) -> None:
        """Save data as JSON (optionally compressed)."""
        json_str = json.dumps(data, indent=2, default=str)

        if self.compress:
            with gzip.open(path, "wt", encoding="utf-8") as f:
                f.write(json_str)
        else:
            with open(path, "w", encoding="utf-8") as f:
                f.write(json_str)

    def _load_json(self, path: Path) -> dict[str, Any] | None:
        """Load JSON data (handles compression)."""
        if not path.exists():
            return None

        try:
            if self.compress or path.suffix == ".gz":
                with gzip.open(path, "rt", encoding="utf-8") as f:
                    return json.load(f)
            else:
                with open(path, encoding="utf-8") as f:
                    return json.load(f)
        except (json.JSONDecodeError, gzip.BadGzipFile, OSError) as e:
            logger.warning(f"Failed to load state from {path}: {e}")
            return None

    # --- LearnedMemoryGate ---

    def save_gate_state(self, gate: LearnedMemoryGate) -> Path:
        """
        Save LearnedMemoryGate state.

        Args:
            gate: Gate to save

        Returns:
            Path where state was saved
        """
        state = LearnedGateState(
            weight_mean=gate.μ,
            weight_covariance=gate.Σ,
            bias=gate.b,
            use_diagonal=gate.use_diagonal,
            n_observations=gate.n_observations,
            decisions=dict(gate.decisions),
        )

        path = self._get_path("learned_gate")
        self._save_json(path, state.to_dict())

        logger.info(
            f"Saved gate state: n_obs={state.n_observations}, "
            f"decisions={state.decisions}"
        )
        return path

    def load_gate_state(self) -> LearnedGateState | None:
        """
        Load LearnedMemoryGate state.

        Returns:
            State if found, None otherwise
        """
        path = self._get_path("learned_gate")
        data = self._load_json(path)

        if data is None:
            logger.debug("No saved gate state found")
            return None

        try:
            state = LearnedGateState.from_dict(data)
            logger.info(
                f"Loaded gate state: n_obs={state.n_observations}, "
                f"saved_at={state.saved_at}"
            )
            return state
        except (KeyError, ValueError) as e:
            logger.warning(f"Failed to parse gate state: {e}")
            return None

    def restore_gate(self, gate: LearnedMemoryGate) -> bool:
        """
        Restore gate from saved state.

        Args:
            gate: Gate to restore into

        Returns:
            True if restored, False if no state found
        """
        state = self.load_gate_state()
        if state is None:
            return False

        # Validate dimensions match
        if len(state.weight_mean) != gate.feature_dim:
            logger.warning(
                f"Dimension mismatch: saved={len(state.weight_mean)}, "
                f"expected={gate.feature_dim}. State not restored."
            )
            return False

        # Restore parameters
        gate.μ = state.weight_mean
        gate.Σ = state.weight_covariance
        gate.b = state.bias
        gate.n_observations = state.n_observations
        gate.decisions = state.decisions

        logger.info(f"Restored gate from state (n_obs={state.n_observations})")
        return True

    # --- LearnedRetrievalScorer ---

    def save_scorer_state(
        self,
        scorer: LearnedRetrievalScorer,
        n_steps: int = 0,
        losses: list[float] | None = None
    ) -> Path:
        """
        Save LearnedRetrievalScorer state.

        Args:
            scorer: Scorer to save
            n_steps: Number of training steps completed
            losses: Recent loss values

        Returns:
            Path where state was saved
        """
        # Extract MLP weights using state_dict pattern
        layer_weights = {}
        layer_biases = {}

        for i, layer in enumerate(scorer.fusion_net):
            if hasattr(layer, "weight"):
                layer_weights[f"layer_{i}"] = layer.weight.detach().cpu().numpy()
            if hasattr(layer, "bias") and layer.bias is not None:
                layer_biases[f"layer_{i}"] = layer.bias.detach().cpu().numpy()

        state = ScorerState(
            layer_weights=layer_weights,
            layer_biases=layer_biases,
            n_training_steps=n_steps,
            recent_losses=losses or [],
        )

        # Use pickle for numpy arrays in complex structures
        path = self.storage_path / "learned_scorer.pkl.gz"
        with gzip.open(path, "wb") as f:
            pickle.dump(state, f)

        logger.info(f"Saved scorer state: n_steps={n_steps}")
        return path

    def load_scorer_state(self) -> ScorerState | None:
        """Load LearnedRetrievalScorer state."""
        path = self.storage_path / "learned_scorer.pkl.gz"
        if not path.exists():
            return None

        try:
            with gzip.open(path, "rb") as f:
                return pickle.load(f)
        except Exception as e:
            logger.warning(f"Failed to load scorer state: {e}")
            return None

    # --- NeuromodulatorOrchestra ---

    def save_neuromodulator_state(
        self,
        orchestra: NeuromodulatorOrchestra
    ) -> Path:
        """
        Save neuromodulator orchestra state.

        Args:
            orchestra: Orchestra to save

        Returns:
            Path where state was saved
        """
        # Extract dopamine expectations
        da_expectations = {}
        if hasattr(orchestra.dopamine, "_expectations"):
            da_expectations = {
                str(k): v for k, v in orchestra.dopamine._expectations.items()
            }

        # Extract serotonin values
        serotonin_values = {}
        if hasattr(orchestra.serotonin, "_long_term_values"):
            serotonin_values = {
                str(k): v for k, v in orchestra.serotonin._long_term_values.items()
            }

        # Extract NE reference distribution
        ne_mean = None
        ne_std = None
        if hasattr(orchestra.norepinephrine, "_reference_mean"):
            ne_mean = orchestra.norepinephrine._reference_mean
        if hasattr(orchestra.norepinephrine, "_reference_std"):
            ne_std = orchestra.norepinephrine._reference_std

        # ACh baseline
        ach_mode = "balanced"
        if hasattr(orchestra.acetylcholine, "_baseline_mode"):
            ach_mode = orchestra.acetylcholine._baseline_mode.value

        state = NeuromodulatorState(
            dopamine_expectations=da_expectations,
            serotonin_values=serotonin_values,
            serotonin_mood=orchestra.serotonin.get_current_mood(),
            ne_reference_mean=ne_mean,
            ne_reference_std=ne_std,
            ach_baseline_mode=ach_mode,
        )

        # Use pickle for numpy arrays
        path = self.storage_path / "neuromodulators.pkl.gz"
        with gzip.open(path, "wb") as f:
            pickle.dump(state, f)

        logger.info(
            f"Saved neuromodulator state: "
            f"da_expectations={len(da_expectations)}, "
            f"5ht_values={len(serotonin_values)}"
        )
        return path

    def load_neuromodulator_state(self) -> NeuromodulatorState | None:
        """Load neuromodulator orchestra state."""
        path = self.storage_path / "neuromodulators.pkl.gz"
        if not path.exists():
            return None

        try:
            with gzip.open(path, "rb") as f:
                return pickle.load(f)
        except Exception as e:
            logger.warning(f"Failed to load neuromodulator state: {e}")
            return None

    # --- Bulk Operations ---

    def save_all(
        self,
        gate: LearnedMemoryGate | None = None,
        scorer: LearnedRetrievalScorer | None = None,
        orchestra: NeuromodulatorOrchestra | None = None,
    ) -> dict[str, Path]:
        """
        Save all learned state.

        Returns:
            Dict of component name -> saved path
        """
        saved = {}

        if gate is not None:
            saved["gate"] = self.save_gate_state(gate)

        if scorer is not None:
            saved["scorer"] = self.save_scorer_state(scorer)

        if orchestra is not None:
            saved["neuromodulators"] = self.save_neuromodulator_state(orchestra)

        return saved

    def get_storage_info(self) -> dict[str, Any]:
        """Get information about stored state."""
        info = {
            "storage_path": str(self.storage_path),
            "files": [],
        }

        for path in self.storage_path.iterdir():
            if path.is_file():
                info["files"].append({
                    "name": path.name,
                    "size_bytes": path.stat().st_size,
                    "modified": datetime.fromtimestamp(
                        path.stat().st_mtime
                    ).isoformat(),
                })

        return info


__all__ = [
    "LearnedGateState",
    "NeuromodulatorState",
    "ScorerState",
    "StatePersister",
]
