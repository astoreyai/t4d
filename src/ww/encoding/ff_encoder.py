"""
Learnable Forward-Forward Encoder for World Weaver.

Phase 5: THE LEARNING GAP FIX

Problem (identified by Hinton agent):
    Current flow (NO LEARNING):
        Input -> Frozen Embedder -> Store in Vector DB -> Retrieve

    This means the system stores representations but doesn't LEARN representations.

Solution:
    Target flow (LEARNS):
        Input -> Frozen Embedder -> FF Encoder (learnable) -> Store -> Retrieve
                                          ^
                              Three-Factor Learning Signal

The FFEncoder sits between the frozen embedding model (BGE-M3) and storage.
It learns to refine representations based on retrieval outcomes using the
Forward-Forward algorithm with three-factor learning rate modulation.

Key Features:
1. Trainable FF layers that update via positive/negative phases
2. Three-factor LR: eligibility x neuromod x dopamine -> weight updates
3. Generative replay during consolidation to prevent catastrophic forgetting
4. Goodness-based confidence scoring for retrieval ranking

Biological Mapping:
- Frozen embedder = sensory cortex (feature extraction)
- FF encoder = hippocampal pattern completion + separation
- Three-factor = neuromodulated Hebbian plasticity
- Sleep replay = systems consolidation

References:
- Hinton (2022): The Forward-Forward Algorithm
- McClelland et al. (1995): Complementary Learning Systems
- Frémaux & Gerstner (2016): Neuromodulated STDP

Integration Points:
- EpisodicMemory.store(): Encode before storage
- EpisodicMemory.recall(): Encode query, use goodness for ranking
- ThreeFactorLearningRule: Apply to FF layer weights
- SleepConsolidation: Generative replay through encoder
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from datetime import datetime
from typing import TYPE_CHECKING

import numpy as np

from ww.nca.forward_forward import ForwardForwardConfig, ForwardForwardLayer

if TYPE_CHECKING:
    from ww.learning.three_factor import ThreeFactorLearningRule, ThreeFactorSignal

logger = logging.getLogger(__name__)


@dataclass
class FFEncoderConfig:
    """
    Configuration for the learnable FF Encoder.

    Attributes:
        input_dim: Input embedding dimension (from frozen embedder)
        hidden_dims: Hidden layer dimensions for FF stack
        output_dim: Output dimension (can match input for residual)
        learning_rate: Base learning rate for FF layers
        threshold_theta: Goodness threshold for classification
        use_residual: Add skip connection from input to output
        normalize_output: L2 normalize output embeddings
        use_neuromod_gating: Enable neuromodulator integration
        dropout_rate: Dropout during training (not used in inference)
        max_history: Maximum samples to keep for replay
    """
    input_dim: int = 1024
    hidden_dims: tuple[int, ...] = (512, 256)
    output_dim: int = 1024
    learning_rate: float = 0.03
    threshold_theta: float = 2.0
    use_residual: bool = True
    normalize_output: bool = True
    use_neuromod_gating: bool = True
    dropout_rate: float = 0.1
    max_history: int = 5000

    def __post_init__(self) -> None:
        """Validate configuration."""
        assert self.input_dim > 0, "input_dim must be positive"
        assert len(self.hidden_dims) > 0, "Need at least one hidden layer"
        assert self.output_dim > 0, "output_dim must be positive"
        assert 0 < self.learning_rate < 1, "learning_rate must be in (0, 1)"


@dataclass
class FFEncoderState:
    """State tracking for FF Encoder."""
    total_encodes: int = 0
    total_positive_updates: int = 0
    total_negative_updates: int = 0
    mean_goodness: float = 0.0
    mean_effective_lr: float = 0.0
    last_encode_time: datetime = field(default_factory=datetime.now)

    def to_dict(self) -> dict:
        """Serialize for logging."""
        return {
            "total_encodes": self.total_encodes,
            "total_positive_updates": self.total_positive_updates,
            "total_negative_updates": self.total_negative_updates,
            "mean_goodness": self.mean_goodness,
            "mean_effective_lr": self.mean_effective_lr,
        }


class FFEncoder:
    """
    Learnable Forward-Forward encoder for representation refinement.

    This is THE key component that makes World Weaver actually LEARN
    representations through use, rather than just storing frozen embeddings.

    Architecture:
        Input (1024) -> FF Layer 1 (512) -> FF Layer 2 (256) -> Project (1024)
              |                                                      |
              +---------------------> Residual -----------------------+

    Learning Flow:
        1. Encode input through FF layers
        2. Store encoded representation
        3. On retrieval outcome:
           - Positive outcome (helpful): train_positive() on encoder
           - Negative outcome (unhelpful): train_negative() on encoder
        4. Three-factor rule modulates learning rate

    Example:
        ```python
        from ww.encoding.ff_encoder import FFEncoder, FFEncoderConfig

        # Create encoder
        config = FFEncoderConfig(hidden_dims=(512, 256))
        encoder = FFEncoder(config)

        # Encode embedding (during store)
        raw_embedding = frozen_embedder.embed(text)
        encoded = encoder.encode(raw_embedding)

        # After positive retrieval outcome
        encoder.learn_from_outcome(
            embedding=encoded,
            outcome_score=0.9,  # Helpful retrieval
            three_factor_signal=signal
        )
        ```
    """

    def __init__(
        self,
        config: FFEncoderConfig | None = None,
        random_seed: int | None = None,
    ):
        """
        Initialize FF Encoder.

        Args:
            config: Encoder configuration
            random_seed: Random seed for reproducibility
        """
        self.config = config or FFEncoderConfig()
        self._rng = np.random.default_rng(random_seed)
        self.state = FFEncoderState()

        # Build FF layer stack
        self._layers: list[ForwardForwardLayer] = []
        in_dim = self.config.input_dim

        for i, hidden_dim in enumerate(self.config.hidden_dims):
            layer_config = ForwardForwardConfig(
                input_dim=in_dim,
                hidden_dim=hidden_dim,
                learning_rate=self.config.learning_rate,
                threshold_theta=self.config.threshold_theta,
                use_neuromod_gating=self.config.use_neuromod_gating,
            )
            layer = ForwardForwardLayer(
                layer_config,
                layer_idx=i,
                random_seed=int(self._rng.integers(0, 2**31)) if random_seed else None,
            )
            self._layers.append(layer)
            in_dim = hidden_dim

        # Output projection (last hidden -> output_dim)
        self._output_weights = self._rng.normal(
            0, np.sqrt(2.0 / (in_dim + self.config.output_dim)),
            size=(in_dim, self.config.output_dim)
        ).astype(np.float32)
        self._output_bias = np.zeros(self.config.output_dim, dtype=np.float32)

        # Residual projection if dimensions don't match
        if self.config.use_residual and self.config.input_dim != self.config.output_dim:
            self._residual_proj = self._rng.normal(
                0, np.sqrt(2.0 / (self.config.input_dim + self.config.output_dim)),
                size=(self.config.input_dim, self.config.output_dim)
            ).astype(np.float32)
        else:
            self._residual_proj = None

        # History for replay (bounded - MEM-007 pattern)
        self._encode_history: list[np.ndarray] = []
        self._goodness_history: list[float] = []

        logger.info(
            f"Phase 5: FFEncoder initialized "
            f"(input={self.config.input_dim}, "
            f"hidden={self.config.hidden_dims}, "
            f"output={self.config.output_dim}, "
            f"residual={self.config.use_residual})"
        )

    def encode(self, embedding: np.ndarray, training: bool = False) -> np.ndarray:
        """
        Encode embedding through learnable FF layers.

        This is the main encoding function. It transforms the frozen
        embedding from the base model into a learned representation.

        Args:
            embedding: Input embedding from frozen embedder [dim] or [batch, dim]
            training: Enable training mode (affects dropout)

        Returns:
            Encoded embedding with same shape as input
        """
        embedding = np.atleast_2d(embedding).astype(np.float32)
        batch_size = embedding.shape[0]

        # Store input for residual
        input_embedding = embedding.copy()

        # Forward through FF layers
        h = embedding
        layer_outputs = []

        for layer in self._layers:
            h = layer.forward(h, training=training)
            layer_outputs.append(h.copy())

        # Output projection
        output = h @ self._output_weights + self._output_bias

        # Apply residual connection
        if self.config.use_residual:
            if self._residual_proj is not None:
                residual = input_embedding @ self._residual_proj
            else:
                residual = input_embedding
            output = output + residual

        # Normalize output
        if self.config.normalize_output:
            norm = np.linalg.norm(output, axis=-1, keepdims=True)
            output = output / (norm + 1e-8)

        # Update state
        self.state.total_encodes += batch_size
        self.state.last_encode_time = datetime.now()

        # Track goodness from last layer
        if self._layers:
            goodness = self._layers[-1].state.goodness
            self._goodness_history.append(goodness)
            if len(self._goodness_history) > self.config.max_history:
                self._goodness_history = self._goodness_history[-self.config.max_history:]

            # Update mean goodness with EMA
            alpha = 0.01
            self.state.mean_goodness = alpha * goodness + (1 - alpha) * self.state.mean_goodness

        # Store for replay (flatten if batch)
        for i in range(batch_size):
            self._encode_history.append(input_embedding[i].copy())
        if len(self._encode_history) > self.config.max_history:
            self._encode_history = self._encode_history[-self.config.max_history:]

        return output.squeeze() if batch_size == 1 else output

    def get_goodness(self, embedding: np.ndarray) -> float:
        """
        Get goodness score for an embedding without learning.

        Useful for retrieval ranking - higher goodness = more confident match.

        Args:
            embedding: Embedding to score

        Returns:
            Goodness value (sum of squared activations)
        """
        embedding = np.atleast_2d(embedding).astype(np.float32)

        # Forward through FF layers
        h = embedding
        for layer in self._layers:
            h = layer.forward(h, training=False)

        # Return last layer's goodness
        return self._layers[-1].state.goodness if self._layers else 0.0

    def learn_from_outcome(
        self,
        embedding: np.ndarray,
        outcome_score: float,
        three_factor_signal: ThreeFactorSignal | None = None,
        effective_lr: float | None = None,
    ) -> dict:
        """
        Update FF encoder weights based on retrieval outcome.

        THIS IS WHERE THE LEARNING HAPPENS.

        Positive outcomes (outcome_score > 0.5):
            - Train FF layers to INCREASE goodness for this pattern
            - Makes similar patterns easier to retrieve/recognize

        Negative outcomes (outcome_score < 0.5):
            - Train FF layers to DECREASE goodness for this pattern
            - Reduces confidence for unhelpful patterns

        Three-factor modulation:
            - effective_lr = base_lr * eligibility * neuromod * dopamine
            - High eligibility: recently active memories learn more
            - High neuromod: encoding state enables learning
            - High dopamine: surprising outcomes learn more

        Args:
            embedding: The embedding that was retrieved/used
            outcome_score: How helpful was this retrieval [0, 1]
            three_factor_signal: Full three-factor signal (optional)
            effective_lr: Override effective learning rate

        Returns:
            Learning statistics
        """
        embedding = np.atleast_2d(embedding).astype(np.float32)

        # Determine effective learning rate
        if effective_lr is not None:
            lr = effective_lr
        elif three_factor_signal is not None:
            lr = self.config.learning_rate * three_factor_signal.effective_lr_multiplier
        else:
            lr = self.config.learning_rate

        # Update state
        alpha = 0.01
        self.state.mean_effective_lr = alpha * lr + (1 - alpha) * self.state.mean_effective_lr

        stats = {
            "outcome_score": outcome_score,
            "effective_lr": lr,
            "layer_stats": [],
        }

        # Determine phase based on outcome
        is_positive = outcome_score > 0.5

        # Forward through layers, collecting inputs
        h = embedding
        layer_inputs = [embedding.copy()]

        for layer in self._layers:
            h = layer.forward(h, training=True)
            layer_inputs.append(h.copy())

        # Update each layer based on outcome
        for i, layer in enumerate(self._layers):
            x_in = layer_inputs[i]
            h_out = layer_inputs[i + 1]

            # Temporarily set layer learning rate
            original_lr = layer.config.learning_rate
            layer.config.learning_rate = lr

            if is_positive:
                layer_stats = layer.learn_positive(x_in, h_out)
                self.state.total_positive_updates += 1
            else:
                layer_stats = layer.learn_negative(x_in, h_out)
                self.state.total_negative_updates += 1

            # Restore original learning rate
            layer.config.learning_rate = original_lr

            layer_stats["layer_idx"] = i
            stats["layer_stats"].append(layer_stats)

        # Update output projection weights (simple gradient step)
        if is_positive:
            # Positive: increase projection magnitude slightly
            gradient = np.outer(layer_inputs[-1].flatten(), np.ones(self.config.output_dim))
            self._output_weights += lr * 0.01 * gradient[:self._output_weights.shape[0], :]
        else:
            # Negative: decrease projection magnitude slightly
            gradient = np.outer(layer_inputs[-1].flatten(), np.ones(self.config.output_dim))
            self._output_weights -= lr * 0.01 * gradient[:self._output_weights.shape[0], :]

        # Clip output weights
        self._output_weights = np.clip(self._output_weights, -2.0, 2.0)

        stats["phase"] = "positive" if is_positive else "negative"
        stats["final_goodness"] = self._layers[-1].state.goodness if self._layers else 0.0

        logger.debug(
            f"Phase 5: FFEncoder learned from {stats['phase']} outcome "
            f"(score={outcome_score:.2f}, lr={lr:.4f}, "
            f"goodness={stats['final_goodness']:.2f})"
        )

        return stats

    def replay_consolidation(
        self,
        n_samples: int = 100,
        positive_ratio: float = 0.7,
        noise_scale: float = 0.1,
    ) -> dict:
        """
        Replay stored patterns for consolidation (sleep phase).

        During sleep, we replay stored embeddings to maintain encoder
        stability and prevent catastrophic forgetting.

        Interleaved learning:
        - Real patterns (from history) as positive
        - Noisy/shuffled patterns as negative

        Args:
            n_samples: Number of replay samples
            positive_ratio: Ratio of positive (real) samples
            noise_scale: Noise magnitude for negative samples

        Returns:
            Replay statistics
        """
        if len(self._encode_history) < 10:
            return {"status": "insufficient_history", "n_available": len(self._encode_history)}

        n_positive = int(n_samples * positive_ratio)
        n_negative = n_samples - n_positive

        stats = {
            "n_positive": n_positive,
            "n_negative": n_negative,
            "positive_stats": [],
            "negative_stats": [],
        }

        # Sample and replay positive patterns
        if n_positive > 0:
            indices = self._rng.choice(
                len(self._encode_history),
                size=min(n_positive, len(self._encode_history)),
                replace=False
            )
            for idx in indices:
                pattern = self._encode_history[idx]
                # Forward pass, collecting layer inputs
                h = np.atleast_2d(pattern).astype(np.float32)
                layer_inputs = [h.copy()]
                for layer in self._layers:
                    h = layer.forward(h, training=True)
                    layer_inputs.append(h.copy())
                # Train each layer as positive (real pattern)
                for i, layer in enumerate(self._layers):
                    layer.learn_positive(layer_inputs[i], layer_inputs[i + 1])
                stats["positive_stats"].append(self._layers[-1].state.goodness if self._layers else 0.0)

        # Generate and replay negative patterns
        if n_negative > 0:
            for _ in range(n_negative):
                # Pick random pattern and corrupt it
                idx = self._rng.choice(len(self._encode_history))
                pattern = self._encode_history[idx].copy()

                # Corruption: add noise or shuffle
                if self._rng.random() < 0.5:
                    # Add noise
                    noise = self._rng.normal(0, noise_scale, pattern.shape)
                    pattern = pattern + noise
                    pattern = pattern / (np.linalg.norm(pattern) + 1e-8)
                else:
                    # Shuffle features
                    self._rng.shuffle(pattern)

                # Forward pass, collecting layer inputs
                h = np.atleast_2d(pattern).astype(np.float32)
                layer_inputs = [h.copy()]
                for layer in self._layers:
                    h = layer.forward(h, training=True)
                    layer_inputs.append(h.copy())
                # Train each layer as negative (corrupted pattern)
                for i, layer in enumerate(self._layers):
                    layer.learn_negative(layer_inputs[i], layer_inputs[i + 1])
                stats["negative_stats"].append(self._layers[-1].state.goodness if self._layers else 0.0)

        stats["mean_positive_goodness"] = float(np.mean(stats["positive_stats"])) if stats["positive_stats"] else 0.0
        stats["mean_negative_goodness"] = float(np.mean(stats["negative_stats"])) if stats["negative_stats"] else 0.0

        logger.debug(
            f"Phase 5: FFEncoder consolidation replay "
            f"(pos={n_positive}, neg={n_negative}, "
            f"mean_pos_g={stats['mean_positive_goodness']:.2f}, "
            f"mean_neg_g={stats['mean_negative_goodness']:.2f})"
        )

        return stats

    def get_layer_stats(self) -> list[dict]:
        """Get statistics from each FF layer."""
        return [layer.get_stats() for layer in self._layers]

    def get_stats(self) -> dict:
        """Get encoder statistics."""
        return {
            "state": self.state.to_dict(),
            "config": {
                "input_dim": self.config.input_dim,
                "hidden_dims": self.config.hidden_dims,
                "output_dim": self.config.output_dim,
                "use_residual": self.config.use_residual,
            },
            "n_layers": len(self._layers),
            "history_size": len(self._encode_history),
            "layer_stats": self.get_layer_stats(),
        }

    def save_weights(self) -> dict:
        """Save all weights for persistence."""
        return {
            "layers": [
                {
                    "W": layer.W.tolist(),
                    "b": layer.b.tolist(),
                }
                for layer in self._layers
            ],
            "output_weights": self._output_weights.tolist(),
            "output_bias": self._output_bias.tolist(),
            "residual_proj": self._residual_proj.tolist() if self._residual_proj is not None else None,
            "state": self.state.to_dict(),
        }

    def load_weights(self, weights: dict) -> None:
        """Load weights from persistence."""
        for i, layer_weights in enumerate(weights["layers"]):
            if i < len(self._layers):
                self._layers[i].W = np.array(layer_weights["W"], dtype=np.float32)
                self._layers[i].b = np.array(layer_weights["b"], dtype=np.float32)

        self._output_weights = np.array(weights["output_weights"], dtype=np.float32)
        self._output_bias = np.array(weights["output_bias"], dtype=np.float32)

        if weights.get("residual_proj") is not None:
            self._residual_proj = np.array(weights["residual_proj"], dtype=np.float32)

        if "state" in weights:
            self.state.total_encodes = weights["state"].get("total_encodes", 0)
            self.state.total_positive_updates = weights["state"].get("total_positive_updates", 0)
            self.state.total_negative_updates = weights["state"].get("total_negative_updates", 0)

        logger.info(f"Phase 5: FFEncoder weights loaded")

    def reset(self) -> None:
        """Reset encoder to initial state."""
        for layer in self._layers:
            layer.reset()

        # Reinitialize output projection
        in_dim = self.config.hidden_dims[-1] if self.config.hidden_dims else self.config.input_dim
        self._output_weights = self._rng.normal(
            0, np.sqrt(2.0 / (in_dim + self.config.output_dim)),
            size=(in_dim, self.config.output_dim)
        ).astype(np.float32)
        self._output_bias = np.zeros(self.config.output_dim, dtype=np.float32)

        self._encode_history.clear()
        self._goodness_history.clear()
        self.state = FFEncoderState()

        logger.info("Phase 5: FFEncoder reset")


# Singleton instance for global access
_ff_encoder_instance: FFEncoder | None = None


def get_ff_encoder(config: FFEncoderConfig | None = None) -> FFEncoder:
    """
    Get the global FF encoder instance.

    Args:
        config: Configuration (only used on first call)

    Returns:
        Global FFEncoder instance
    """
    global _ff_encoder_instance

    if _ff_encoder_instance is None:
        _ff_encoder_instance = FFEncoder(config)

    return _ff_encoder_instance


def reset_ff_encoder() -> None:
    """Reset the global FF encoder instance."""
    global _ff_encoder_instance
    _ff_encoder_instance = None


def create_ff_encoder(
    input_dim: int = 1024,
    hidden_dims: tuple[int, ...] = (512, 256),
    output_dim: int = 1024,
    learning_rate: float = 0.03,
    use_residual: bool = True,
) -> FFEncoder:
    """
    Factory function for FF Encoder.

    Args:
        input_dim: Input embedding dimension
        hidden_dims: Hidden layer dimensions
        output_dim: Output dimension
        learning_rate: Base learning rate
        use_residual: Enable residual connection

    Returns:
        Configured FFEncoder
    """
    config = FFEncoderConfig(
        input_dim=input_dim,
        hidden_dims=hidden_dims,
        output_dim=output_dim,
        learning_rate=learning_rate,
        use_residual=use_residual,
    )
    return FFEncoder(config)


__all__ = [
    "FFEncoder",
    "FFEncoderConfig",
    "FFEncoderState",
    "get_ff_encoder",
    "reset_ff_encoder",
    "create_ff_encoder",
]
