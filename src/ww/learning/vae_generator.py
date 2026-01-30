"""
Variational Autoencoder Generator for Wake-Sleep Algorithm.

P6.1: Real VAE implementation for GenerativeReplaySystem.

Implements the Generator protocol with proper encode/decode functionality
using a variational autoencoder architecture. This enables true generative
replay during sleep consolidation phases.

Architecture:
    Input (embedding_dim=1024)
        |
    Encoder MLP
        |
    [mu, log_var] → Reparameterization → z
        |
    Decoder MLP
        |
    Output (embedding_dim=1024)

Training Objective (ELBO):
    L = E[log p(x|z)] - KL(q(z|x) || p(z))

    Where:
    - Reconstruction: MSE between input and output
    - KL Divergence: Regularizes latent space to N(0,1)

References:
- Kingma & Welling (2014): Auto-Encoding Variational Bayes
- Hinton et al. (1995): The wake-sleep algorithm
- Shin et al. (2017): Continual Learning with Deep Generative Replay
"""

from __future__ import annotations

import logging
from dataclasses import dataclass

import numpy as np

logger = logging.getLogger(__name__)


@dataclass
class VAEConfig:
    """Configuration for VAE Generator.

    Attributes:
        embedding_dim: Input/output embedding dimension
        latent_dim: Latent space dimension (compressed)
        hidden_dims: Hidden layer sizes for encoder/decoder
        learning_rate: Training learning rate
        kl_weight: Weight for KL divergence term (beta-VAE)
        dropout_rate: Dropout probability during training
        weight_decay: L2 regularization strength
    """
    embedding_dim: int = 1024
    latent_dim: int = 128
    hidden_dims: tuple[int, ...] = (512, 256)
    learning_rate: float = 0.001
    kl_weight: float = 0.1  # Beta for beta-VAE
    dropout_rate: float = 0.1
    weight_decay: float = 1e-5


@dataclass
class VAEState:
    """Current state of the VAE.

    Attributes:
        n_training_steps: Total training iterations
        total_loss: Cumulative total loss
        reconstruction_loss: Cumulative reconstruction loss
        kl_loss: Cumulative KL divergence loss
        mean_latent_norm: Running average latent vector norm
    """
    n_training_steps: int = 0
    total_loss: float = 0.0
    reconstruction_loss: float = 0.0
    kl_loss: float = 0.0
    mean_latent_norm: float = 0.0


@dataclass
class MLPLayer:
    """Simple MLP layer with weights and biases.

    Attributes:
        weights: Weight matrix (in_dim, out_dim)
        biases: Bias vector (out_dim,)
    """
    weights: np.ndarray
    biases: np.ndarray

    @classmethod
    def create(
        cls,
        in_dim: int,
        out_dim: int,
        init_scale: float = 0.01
    ) -> MLPLayer:
        """Initialize layer with Xavier/He initialization."""
        # He initialization for ReLU activations
        scale = np.sqrt(2.0 / in_dim) * init_scale
        weights = np.random.randn(in_dim, out_dim) * scale
        biases = np.zeros(out_dim)
        return cls(weights=weights, biases=biases)

    def forward(self, x: np.ndarray) -> np.ndarray:
        """Forward pass: x @ W + b."""
        return x @ self.weights + self.biases

    def backward(
        self,
        x: np.ndarray,
        grad_output: np.ndarray,
        learning_rate: float,
        weight_decay: float = 0.0
    ) -> np.ndarray:
        """Backward pass with gradient update.

        Args:
            x: Input to this layer
            grad_output: Gradient from next layer
            learning_rate: Update step size
            weight_decay: L2 regularization

        Returns:
            Gradient for previous layer
        """
        # Compute gradients
        grad_weights = x.T @ grad_output / len(x)
        grad_biases = np.mean(grad_output, axis=0)

        # Gradient for previous layer
        grad_input = grad_output @ self.weights.T

        # Apply updates with weight decay
        self.weights -= learning_rate * (grad_weights + weight_decay * self.weights)
        self.biases -= learning_rate * grad_biases

        return grad_input


def relu(x: np.ndarray) -> np.ndarray:
    """ReLU activation function."""
    return np.maximum(0, x)


def relu_backward(x: np.ndarray, grad: np.ndarray) -> np.ndarray:
    """ReLU gradient."""
    return grad * (x > 0).astype(float)


class VAEGenerator:
    """
    Variational Autoencoder implementing Generator protocol.

    Provides true generative capability for wake-sleep algorithm:
    - Encode: Map embeddings to latent space
    - Decode: Reconstruct from latent space
    - Generate: Sample new embeddings from learned distribution

    Example:
        ```python
        from ww.learning.vae_generator import VAEGenerator, VAEConfig
        from ww.learning.generative_replay import GenerativeReplaySystem

        # Create VAE generator
        config = VAEConfig(embedding_dim=1024, latent_dim=128)
        vae = VAEGenerator(config)

        # Use with generative replay
        replay = GenerativeReplaySystem(generator=vae)

        # Training during wake phase
        for batch in training_data:
            vae.train_step(batch)

        # Generation during sleep phase
        samples = vae.generate(n_samples=100, temperature=1.0)
        ```
    """

    def __init__(self, config: VAEConfig | None = None):
        """
        Initialize VAE Generator.

        Args:
            config: VAE configuration
        """
        self.config = config or VAEConfig()
        self.state = VAEState()

        # Build encoder layers
        self._encoder_layers: list[MLPLayer] = []
        in_dim = self.config.embedding_dim
        for hidden_dim in self.config.hidden_dims:
            layer = MLPLayer.create(in_dim, hidden_dim)
            self._encoder_layers.append(layer)
            in_dim = hidden_dim

        # Latent projection (mu and log_var)
        self._mu_layer = MLPLayer.create(in_dim, self.config.latent_dim)
        self._logvar_layer = MLPLayer.create(in_dim, self.config.latent_dim)

        # Build decoder layers (reverse order)
        self._decoder_layers: list[MLPLayer] = []
        in_dim = self.config.latent_dim
        for hidden_dim in reversed(self.config.hidden_dims):
            layer = MLPLayer.create(in_dim, hidden_dim)
            self._decoder_layers.append(layer)
            in_dim = hidden_dim

        # Output projection
        self._output_layer = MLPLayer.create(in_dim, self.config.embedding_dim)

        # Cache for backward pass
        self._cache: dict = {}

        logger.info(
            f"P6.1: VAEGenerator initialized "
            f"(embedding={self.config.embedding_dim}, "
            f"latent={self.config.latent_dim}, "
            f"hidden={self.config.hidden_dims})"
        )

    def _encode_forward(self, x: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        """
        Encode input to latent distribution parameters.

        Args:
            x: Input embeddings (batch, embedding_dim)

        Returns:
            Tuple of (mu, log_var) for latent distribution
        """
        h = x
        activations = [x]

        for layer in self._encoder_layers:
            h = layer.forward(h)
            pre_activation = h.copy()
            h = relu(h)
            activations.append((pre_activation, h))

        mu = self._mu_layer.forward(h)
        log_var = self._logvar_layer.forward(h)

        # Cache for backward pass
        self._cache['encoder_activations'] = activations
        self._cache['final_hidden'] = h

        return mu, log_var

    def _reparameterize(
        self,
        mu: np.ndarray,
        log_var: np.ndarray,
        temperature: float = 1.0
    ) -> np.ndarray:
        """
        Reparameterization trick: z = mu + std * epsilon.

        This allows gradients to flow through the sampling operation.

        Args:
            mu: Mean of latent distribution
            log_var: Log variance of latent distribution
            temperature: Sampling temperature (higher = more diverse)

        Returns:
            Sampled latent vector z
        """
        std = np.exp(0.5 * log_var) * temperature
        epsilon = np.random.randn(*mu.shape)
        z = mu + std * epsilon

        # Cache for backward pass
        self._cache['mu'] = mu
        self._cache['log_var'] = log_var
        self._cache['epsilon'] = epsilon
        self._cache['std'] = std

        return z

    def _decode_forward(self, z: np.ndarray) -> np.ndarray:
        """
        Decode latent vector to embedding space.

        Args:
            z: Latent vector (batch, latent_dim)

        Returns:
            Reconstructed embedding (batch, embedding_dim)
        """
        h = z
        activations = [z]

        for layer in self._decoder_layers:
            h = layer.forward(h)
            pre_activation = h.copy()
            h = relu(h)
            activations.append((pre_activation, h))

        output = self._output_layer.forward(h)

        # Normalize output to unit sphere (embeddings are typically normalized)
        output = output / (np.linalg.norm(output, axis=-1, keepdims=True) + 1e-8)

        # Cache for backward pass
        self._cache['decoder_activations'] = activations
        self._cache['output_pre_norm'] = output

        return output

    def encode(self, embeddings: list[np.ndarray]) -> list[np.ndarray]:
        """
        Encode embeddings to latent space (Generator protocol).

        Args:
            embeddings: List of embedding vectors

        Returns:
            List of latent vectors (mean of distribution)
        """
        if not embeddings:
            return []

        x = np.stack(embeddings, axis=0)
        mu, _ = self._encode_forward(x)

        return [mu[i] for i in range(len(mu))]

    def decode(self, latents: list[np.ndarray]) -> list[np.ndarray]:
        """
        Decode latent vectors to embedding space (Generator protocol).

        Args:
            latents: List of latent vectors

        Returns:
            List of reconstructed embeddings
        """
        if not latents:
            return []

        z = np.stack(latents, axis=0)
        output = self._decode_forward(z)

        return [output[i] for i in range(len(output))]

    def generate(
        self,
        n_samples: int,
        temperature: float = 1.0
    ) -> list[np.ndarray]:
        """
        Generate synthetic embeddings (Generator protocol).

        Samples from the prior N(0,1) and decodes to embedding space.

        Args:
            n_samples: Number of samples to generate
            temperature: Sampling temperature (higher = more diverse)

        Returns:
            List of generated embedding vectors
        """
        # Sample from prior N(0, 1)
        z = np.random.randn(n_samples, self.config.latent_dim) * temperature

        # Decode to embedding space
        output = self._decode_forward(z)

        return [output[i] for i in range(n_samples)]

    def forward(self, x: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        """
        Full forward pass through VAE.

        Args:
            x: Input embeddings (batch, embedding_dim)

        Returns:
            Tuple of (reconstruction, mu, log_var)
        """
        mu, log_var = self._encode_forward(x)
        z = self._reparameterize(mu, log_var)
        reconstruction = self._decode_forward(z)

        return reconstruction, mu, log_var

    def compute_loss(
        self,
        x: np.ndarray,
        reconstruction: np.ndarray,
        mu: np.ndarray,
        log_var: np.ndarray
    ) -> tuple[float, float, float]:
        """
        Compute ELBO loss.

        Loss = Reconstruction + beta * KL

        Args:
            x: Input embeddings
            reconstruction: Reconstructed embeddings
            mu: Latent mean
            log_var: Latent log variance

        Returns:
            Tuple of (total_loss, reconstruction_loss, kl_loss)
        """
        # Reconstruction loss (MSE)
        recon_loss = np.mean(np.sum((x - reconstruction) ** 2, axis=-1))

        # KL divergence: KL(N(mu, sigma) || N(0, 1))
        # = -0.5 * sum(1 + log_var - mu^2 - exp(log_var))
        kl_loss = -0.5 * np.mean(np.sum(
            1 + log_var - mu ** 2 - np.exp(log_var),
            axis=-1
        ))

        # Total loss with beta weighting
        total_loss = recon_loss + self.config.kl_weight * kl_loss

        return float(total_loss), float(recon_loss), float(kl_loss)

    def train_step(self, embeddings) -> dict:
        """
        Perform one training step.

        Args:
            embeddings: Batch of training embeddings (either list[np.ndarray] or np.ndarray)

        Returns:
            Dict with loss values
        """
        # Handle both list of arrays and pre-stacked arrays
        if isinstance(embeddings, np.ndarray):
            x = embeddings
        else:
            # Assume it's a list or iterable
            try:
                if len(embeddings) == 0:
                    return {"total_loss": 0.0, "recon_loss": 0.0, "kl_loss": 0.0}
            except (TypeError, ValueError):
                return {"total_loss": 0.0, "recon_loss": 0.0, "kl_loss": 0.0}
            x = np.stack(embeddings, axis=0)
        
        batch_size = len(x)

        # Forward pass
        reconstruction, mu, log_var = self.forward(x)

        # Compute loss
        total_loss, recon_loss, kl_loss = self.compute_loss(
            x, reconstruction, mu, log_var
        )

        # Backward pass (gradient computation)
        self._backward(x, reconstruction, mu, log_var)

        # Update state
        self.state.n_training_steps += 1
        alpha = 0.1  # EMA smoothing
        self.state.total_loss = alpha * total_loss + (1 - alpha) * self.state.total_loss
        self.state.reconstruction_loss = alpha * recon_loss + (1 - alpha) * self.state.reconstruction_loss
        self.state.kl_loss = alpha * kl_loss + (1 - alpha) * self.state.kl_loss
        self.state.mean_latent_norm = alpha * np.mean(np.linalg.norm(mu, axis=-1)) + \
                                       (1 - alpha) * self.state.mean_latent_norm

        return {
            "total_loss": total_loss,
            "recon_loss": recon_loss,
            "kl_loss": kl_loss,
            "latent_norm": float(np.mean(np.linalg.norm(mu, axis=-1)))
        }

    def _backward(
        self,
        x: np.ndarray,
        reconstruction: np.ndarray,
        mu: np.ndarray,
        log_var: np.ndarray
    ) -> None:
        """
        Backward pass with gradient updates.

        Args:
            x: Original input
            reconstruction: VAE output
            mu: Latent mean
            log_var: Latent log variance
        """
        lr = self.config.learning_rate
        wd = self.config.weight_decay
        batch_size = len(x)

        # Reconstruction gradient
        grad_output = 2 * (reconstruction - x) / batch_size

        # Gradient for output normalization (approximate)
        # Since we normalized, the gradient is modified
        grad_decoder = grad_output

        # Decoder backward
        # decoder_activations[0] = z (latent), decoder_activations[1+] = (pre_act, post_act) tuples
        decoder_acts = self._cache['decoder_activations']
        if len(decoder_acts) > 1 and isinstance(decoder_acts[-1], tuple):
            h = decoder_acts[-1][1]  # Last post-activation
        else:
            h = decoder_acts[0]  # Just z
        grad_decoder = self._output_layer.backward(h, grad_decoder, lr, wd)

        # Through decoder hidden layers (in reverse order)
        for i, layer in enumerate(reversed(self._decoder_layers)):
            # Index into activations: layer i (reversed) corresponds to decoder_activations[len-1-i]
            act_idx = len(self._decoder_layers) - i  # 1-indexed into activations

            # Apply ReLU backward if we have tuple activation at this index
            if act_idx < len(decoder_acts) and isinstance(decoder_acts[act_idx], tuple):
                pre_act, _ = decoder_acts[act_idx]
                grad_decoder = relu_backward(pre_act, grad_decoder)

            # Get input for this layer
            if act_idx > 1 and isinstance(decoder_acts[act_idx - 1], tuple):
                layer_input = decoder_acts[act_idx - 1][1]  # post_act from previous
            else:
                layer_input = decoder_acts[0]  # z

            grad_decoder = layer.backward(layer_input, grad_decoder, lr, wd)

        # Gradient to latent z
        grad_z = grad_decoder

        # Through reparameterization
        # z = mu + std * epsilon
        # grad_mu = grad_z
        # grad_std = grad_z * epsilon
        # grad_log_var = grad_std * 0.5 * std
        epsilon = self._cache['epsilon']
        std = self._cache['std']

        grad_mu_from_z = grad_z
        grad_std = grad_z * epsilon
        grad_logvar_from_z = grad_std * 0.5 * std

        # KL gradients
        # d/d_mu KL = mu
        # d/d_logvar KL = 0.5 * (exp(logvar) - 1)
        grad_mu_kl = self.config.kl_weight * mu / batch_size
        grad_logvar_kl = self.config.kl_weight * 0.5 * (np.exp(log_var) - 1) / batch_size

        # Combine gradients
        grad_mu = grad_mu_from_z + grad_mu_kl
        grad_logvar = grad_logvar_from_z + grad_logvar_kl

        # Update mu and logvar layers
        final_hidden = self._cache['final_hidden']
        self._mu_layer.backward(final_hidden, grad_mu, lr, wd)
        self._logvar_layer.backward(final_hidden, grad_logvar, lr, wd)

        # Combine gradients for encoder
        grad_encoder = self._mu_layer.weights @ grad_mu.T + self._logvar_layer.weights @ grad_logvar.T
        grad_encoder = grad_encoder.T

        # Through encoder hidden layers
        # encoder_activations[0] = x (input), encoder_activations[1+] = (pre_act, post_act) tuples
        encoder_acts = self._cache['encoder_activations']

        for i, layer in enumerate(reversed(self._encoder_layers)):
            # Index into activations: layer i (reversed) corresponds to encoder_activations[len-i]
            act_idx = len(self._encoder_layers) - i  # 1-indexed into activations

            # Apply ReLU backward if we have tuple activation at this index
            if act_idx < len(encoder_acts) and isinstance(encoder_acts[act_idx], tuple):
                pre_act, _ = encoder_acts[act_idx]
                grad_encoder = relu_backward(pre_act, grad_encoder)

            # Get input for this layer
            if act_idx > 1 and isinstance(encoder_acts[act_idx - 1], tuple):
                layer_input = encoder_acts[act_idx - 1][1]  # post_act from previous
            else:
                layer_input = encoder_acts[0]  # original input x

            grad_encoder = layer.backward(layer_input, grad_encoder, lr, wd)

    def get_statistics(self) -> dict:
        """Get VAE statistics."""
        return {
            "n_training_steps": self.state.n_training_steps,
            "total_loss": self.state.total_loss,
            "reconstruction_loss": self.state.reconstruction_loss,
            "kl_loss": self.state.kl_loss,
            "mean_latent_norm": self.state.mean_latent_norm,
            "config": {
                "embedding_dim": self.config.embedding_dim,
                "latent_dim": self.config.latent_dim,
                "hidden_dims": self.config.hidden_dims,
                "kl_weight": self.config.kl_weight,
            }
        }

    def save_weights(self) -> dict:
        """Save all weights to dictionary."""
        weights = {
            "encoder_layers": [
                {"weights": layer.weights.tolist(), "biases": layer.biases.tolist()}
                for layer in self._encoder_layers
            ],
            "decoder_layers": [
                {"weights": layer.weights.tolist(), "biases": layer.biases.tolist()}
                for layer in self._decoder_layers
            ],
            "mu_layer": {
                "weights": self._mu_layer.weights.tolist(),
                "biases": self._mu_layer.biases.tolist()
            },
            "logvar_layer": {
                "weights": self._logvar_layer.weights.tolist(),
                "biases": self._logvar_layer.biases.tolist()
            },
            "output_layer": {
                "weights": self._output_layer.weights.tolist(),
                "biases": self._output_layer.biases.tolist()
            },
            "state": {
                "n_training_steps": self.state.n_training_steps,
                "total_loss": self.state.total_loss,
                "reconstruction_loss": self.state.reconstruction_loss,
                "kl_loss": self.state.kl_loss,
            }
        }
        return weights

    def load_weights(self, weights: dict) -> None:
        """Load weights from dictionary."""
        for i, layer_data in enumerate(weights["encoder_layers"]):
            self._encoder_layers[i].weights = np.array(layer_data["weights"])
            self._encoder_layers[i].biases = np.array(layer_data["biases"])

        for i, layer_data in enumerate(weights["decoder_layers"]):
            self._decoder_layers[i].weights = np.array(layer_data["weights"])
            self._decoder_layers[i].biases = np.array(layer_data["biases"])

        self._mu_layer.weights = np.array(weights["mu_layer"]["weights"])
        self._mu_layer.biases = np.array(weights["mu_layer"]["biases"])

        self._logvar_layer.weights = np.array(weights["logvar_layer"]["weights"])
        self._logvar_layer.biases = np.array(weights["logvar_layer"]["biases"])

        self._output_layer.weights = np.array(weights["output_layer"]["weights"])
        self._output_layer.biases = np.array(weights["output_layer"]["biases"])

        if "state" in weights:
            self.state.n_training_steps = weights["state"]["n_training_steps"]
            self.state.total_loss = weights["state"]["total_loss"]
            self.state.reconstruction_loss = weights["state"]["reconstruction_loss"]
            self.state.kl_loss = weights["state"]["kl_loss"]

        logger.info(f"Loaded VAE weights (steps={self.state.n_training_steps})")


def create_vae_generator(
    embedding_dim: int = 1024,
    latent_dim: int = 128,
    hidden_dims: tuple[int, ...] = (512, 256),
    kl_weight: float = 0.1
) -> VAEGenerator:
    """
    Factory function for VAE Generator.

    Args:
        embedding_dim: Input/output embedding dimension
        latent_dim: Latent space dimension
        hidden_dims: Hidden layer sizes
        kl_weight: Beta for beta-VAE

    Returns:
        Configured VAEGenerator
    """
    config = VAEConfig(
        embedding_dim=embedding_dim,
        latent_dim=latent_dim,
        hidden_dims=hidden_dims,
        kl_weight=kl_weight
    )
    return VAEGenerator(config)
