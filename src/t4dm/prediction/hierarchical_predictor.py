"""
Hierarchical Multi-Timescale Prediction.

P4-1: Predict at multiple temporal horizons.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from datetime import datetime
from typing import Any
from uuid import UUID

import numpy as np

from t4dm.prediction.context_encoder import ContextEncoder
from t4dm.prediction.latent_predictor import (
    LatentPredictor,
    LatentPredictorConfig,
    Prediction,
    PredictionError,
)

logger = logging.getLogger(__name__)


@dataclass
class HierarchicalConfig:
    fast_horizon: int = 1
    medium_horizon: int = 5
    slow_horizon: int = 15
    fast_lr: float = 0.01
    medium_lr: float = 0.001
    slow_lr: float = 0.0001
    fast_weight: float = 0.5
    medium_weight: float = 0.3
    slow_weight: float = 0.2
    embedding_dim: int = 1024
    hidden_dim: int = 512
    fast_context_size: int = 3
    medium_context_size: int = 10
    slow_context_size: int = 30


@dataclass
class HierarchicalPrediction:
    fast_prediction: Prediction
    medium_prediction: Prediction
    slow_prediction: Prediction
    combined_confidence: float = 0.0
    timestamp: datetime = field(default_factory=datetime.now)

    def to_dict(self) -> dict[str, Any]:
        return {
            "fast": self.fast_prediction.to_dict(),
            "medium": self.medium_prediction.to_dict(),
            "slow": self.slow_prediction.to_dict(),
            "combined_confidence": self.combined_confidence,
            "timestamp": self.timestamp.isoformat(),
        }


@dataclass
class HierarchicalError:
    episode_id: UUID
    fast_error: PredictionError | None = None
    medium_error: PredictionError | None = None
    slow_error: PredictionError | None = None

    @property
    def combined_error(self) -> float:
        errors = []
        weights = [0.5, 0.3, 0.2]
        for error, weight in zip([self.fast_error, self.medium_error, self.slow_error], weights):
            if error:
                errors.append(error.combined_error * weight)
        return sum(errors) if errors else 0.0

    def to_dict(self) -> dict[str, Any]:
        return {
            "episode_id": str(self.episode_id),
            "fast_error": self.fast_error.to_dict() if self.fast_error else None,
            "medium_error": self.medium_error.to_dict() if self.medium_error else None,
            "slow_error": self.slow_error.to_dict() if self.slow_error else None,
            "combined_error": self.combined_error,
        }


class HierarchicalPredictor:
    """Multi-timescale hierarchical prediction."""

    def __init__(self, context_encoder: ContextEncoder | None = None, config: HierarchicalConfig | None = None):
        self.config = config or HierarchicalConfig()
        self.encoder = context_encoder or ContextEncoder()

        self.fast_predictor = LatentPredictor(LatentPredictorConfig(
            context_dim=self.config.embedding_dim,
            hidden_dim=self.config.hidden_dim,
            output_dim=self.config.embedding_dim,
            learning_rate=self.config.fast_lr,
            prediction_horizon=self.config.fast_horizon,
        ))

        self.medium_predictor = LatentPredictor(LatentPredictorConfig(
            context_dim=self.config.embedding_dim,
            hidden_dim=self.config.hidden_dim,
            output_dim=self.config.embedding_dim,
            learning_rate=self.config.medium_lr,
            prediction_horizon=self.config.medium_horizon,
        ))

        self.slow_predictor = LatentPredictor(LatentPredictorConfig(
            context_dim=self.config.embedding_dim,
            hidden_dim=self.config.hidden_dim,
            output_dim=self.config.embedding_dim,
            learning_rate=self.config.slow_lr,
            prediction_horizon=self.config.slow_horizon,
        ))

        self._episode_buffer: list[tuple[UUID, np.ndarray]] = []
        self._max_buffer_size = max(self.config.slow_context_size + self.config.slow_horizon, 100)

        self._pending_predictions: dict[str, list[tuple[int, int, Prediction]]] = {
            "fast": [], "medium": [], "slow": [],
        }

        self._error_history: list[HierarchicalError] = []
        self._max_error_history = 500
        self._step_count = 0
        self._total_predictions = 0
        self._resolved_predictions = 0

        logger.info(f"HierarchicalPredictor initialized: horizons=({self.config.fast_horizon}, {self.config.medium_horizon}, {self.config.slow_horizon})")

    def add_episode(self, episode_id: UUID, embedding: np.ndarray) -> HierarchicalError | None:
        norm = np.linalg.norm(embedding)
        if norm > 0:
            embedding = embedding / norm

        self._episode_buffer.append((episode_id, embedding))
        if len(self._episode_buffer) > self._max_buffer_size:
            self._episode_buffer = self._episode_buffer[-self._max_buffer_size:]

        self._step_count += 1
        error = self._resolve_predictions(episode_id, embedding)
        return error

    def predict(self) -> HierarchicalPrediction | None:
        if len(self._episode_buffer) < self.config.fast_context_size:
            return None

        fast_context = self._get_context(self.config.fast_context_size)
        medium_context = self._get_context(self.config.medium_context_size)
        slow_context = self._get_context(self.config.slow_context_size)

        fast_pred = self.fast_predictor.predict(fast_context)
        medium_pred = self.medium_predictor.predict(medium_context)
        slow_pred = self.slow_predictor.predict(slow_context)

        current_step = self._step_count
        self._pending_predictions["fast"].append((current_step, current_step + self.config.fast_horizon, fast_pred))
        self._pending_predictions["medium"].append((current_step, current_step + self.config.medium_horizon, medium_pred))
        self._pending_predictions["slow"].append((current_step, current_step + self.config.slow_horizon, slow_pred))

        for key in self._pending_predictions:
            if len(self._pending_predictions[key]) > 100:
                self._pending_predictions[key] = self._pending_predictions[key][-100:]

        self._total_predictions += 1

        combined = (
            fast_pred.confidence * self.config.fast_weight
            + medium_pred.confidence * self.config.medium_weight
            + slow_pred.confidence * self.config.slow_weight
        )

        return HierarchicalPrediction(
            fast_prediction=fast_pred,
            medium_prediction=medium_pred,
            slow_prediction=slow_pred,
            combined_confidence=combined,
        )

    def _get_context(self, context_size: int) -> np.ndarray:
        n = min(context_size, len(self._episode_buffer))
        if n == 0:
            return np.zeros(self.config.embedding_dim, dtype=np.float32)
        embeddings = [emb for _, emb in self._episode_buffer[-n:]]
        encoded = self.encoder.encode(embeddings)
        return encoded.context_vector

    def _resolve_predictions(self, actual_id: UUID, actual_embedding: np.ndarray) -> HierarchicalError | None:
        current_step = self._step_count
        resolved_any = False
        error = HierarchicalError(episode_id=actual_id)

        for horizon_type, predictor, lr_attr in [
            ("fast", self.fast_predictor, self.config.fast_lr),
            ("medium", self.medium_predictor, self.config.medium_lr),
            ("slow", self.slow_predictor, self.config.slow_lr),
        ]:
            pending = self._pending_predictions[horizon_type]
            new_pending = []

            for step_made, target_step, prediction in pending:
                if target_step == current_step:
                    pred_error = predictor.compute_error(prediction, actual_embedding, actual_id)

                    context_offset = current_step - step_made
                    if context_offset <= len(self._episode_buffer):
                        context = self._get_historical_context(context_offset, self._get_context_size(horizon_type))
                        if context is not None:
                            predictor.train_step(context, actual_embedding)

                    if horizon_type == "fast":
                        error.fast_error = pred_error
                    elif horizon_type == "medium":
                        error.medium_error = pred_error
                    else:
                        error.slow_error = pred_error

                    resolved_any = True
                    self._resolved_predictions += 1

                elif target_step > current_step:
                    new_pending.append((step_made, target_step, prediction))

            self._pending_predictions[horizon_type] = new_pending

        if resolved_any:
            self._error_history.append(error)
            if len(self._error_history) > self._max_error_history:
                self._error_history = self._error_history[-self._max_error_history:]
            return error

        return None

    def _get_context_size(self, horizon_type: str) -> int:
        if horizon_type == "fast":
            return self.config.fast_context_size
        elif horizon_type == "medium":
            return self.config.medium_context_size
        else:
            return self.config.slow_context_size

    def _get_historical_context(self, steps_ago: int, context_size: int) -> np.ndarray | None:
        if steps_ago > len(self._episode_buffer):
            return None
        end_idx = len(self._episode_buffer) - steps_ago
        start_idx = max(0, end_idx - context_size)
        if start_idx >= end_idx:
            return None
        embeddings = [emb for _, emb in self._episode_buffer[start_idx:end_idx]]
        if not embeddings:
            return None
        encoded = self.encoder.encode(embeddings)
        return encoded.context_vector

    def get_recent_errors(self, n: int = 10) -> list[HierarchicalError]:
        return self._error_history[-n:]

    def get_high_error_episodes(self, k: int = 10, horizon: str = "combined") -> list[tuple[UUID, float]]:
        episode_errors: dict[UUID, float] = {}
        for error in self._error_history:
            if horizon == "combined":
                err_val = error.combined_error
            elif horizon == "fast" and error.fast_error:
                err_val = error.fast_error.combined_error
            elif horizon == "medium" and error.medium_error:
                err_val = error.medium_error.combined_error
            elif horizon == "slow" and error.slow_error:
                err_val = error.slow_error.combined_error
            else:
                continue
            episode_errors[error.episode_id] = max(episode_errors.get(error.episode_id, 0), err_val)
        sorted_episodes = sorted(episode_errors.items(), key=lambda x: x[1], reverse=True)
        return sorted_episodes[:k]

    def get_statistics(self) -> dict[str, Any]:
        return {
            "total_predictions": self._total_predictions,
            "resolved_predictions": self._resolved_predictions,
            "pending_fast": len(self._pending_predictions["fast"]),
            "pending_medium": len(self._pending_predictions["medium"]),
            "pending_slow": len(self._pending_predictions["slow"]),
            "buffer_size": len(self._episode_buffer),
            "error_history_size": len(self._error_history),
            "fast_stats": self.fast_predictor.get_statistics(),
            "medium_stats": self.medium_predictor.get_statistics(),
            "slow_stats": self.slow_predictor.get_statistics(),
        }

    def save_state(self) -> dict[str, Any]:
        return {
            "config": {
                "fast_horizon": self.config.fast_horizon,
                "medium_horizon": self.config.medium_horizon,
                "slow_horizon": self.config.slow_horizon,
            },
            "fast_state": self.fast_predictor.save_state(),
            "medium_state": self.medium_predictor.save_state(),
            "slow_state": self.slow_predictor.save_state(),
            "step_count": self._step_count,
            "statistics": self.get_statistics(),
        }

    def load_state(self, state: dict[str, Any]) -> None:
        if "fast_state" in state:
            self.fast_predictor.load_state(state["fast_state"])
        if "medium_state" in state:
            self.medium_predictor.load_state(state["medium_state"])
        if "slow_state" in state:
            self.slow_predictor.load_state(state["slow_state"])
        self._step_count = state.get("step_count", 0)
