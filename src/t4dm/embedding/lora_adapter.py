"""
LoRA Embedding Adapter for T4DM.

Implements Low-Rank Adaptation (LoRA) of frozen embeddings, enabling
task-specific representation learning without full model fine-tuning.

Biological Basis:
- Prefrontal cortex modulates sensory representations for task relevance
- Attention mechanisms gate information flow based on context
- LoRA mirrors synaptic plasticity in projection neurons

Architecture:
- Frozen backbone embeddings (BGE-M3)
- Low-rank adapters (A, B matrices) for residual adaptation
- Optional query/memory asymmetric heads
- Training via contrastive loss on retrieval outcomes

References:
- Hu et al. (2021) "LoRA: Low-Rank Adaptation of Large Language Models"
- Hinton Analysis: Frozen embeddings limit task adaptation
"""

from __future__ import annotations

import json
import logging
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from typing import TYPE_CHECKING

import numpy as np

if TYPE_CHECKING:
    from t4dm.embedding.bge_m3 import BGEM3Embedding

logger = logging.getLogger(__name__)

# Optional torch import for gradient-based training
try:
    import torch
    import torch.nn as nn
    import torch.nn.functional as F
    TORCH_AVAILABLE = True
except ImportError:
    TORCH_AVAILABLE = False
    logger.warning("PyTorch not available, LoRA adapter training disabled")


@dataclass
class LoRAConfig:
    """Configuration for LoRA embedding adapter."""
    # Architecture
    embedding_dim: int = 1024          # BGE-M3 dimension
    rank: int = 16                     # LoRA rank (16-64 typical)
    alpha: float = 16.0                # LoRA scaling factor
    dropout: float = 0.1               # Dropout for regularization

    # Training
    learning_rate: float = 1e-4        # Adam LR
    weight_decay: float = 0.01         # L2 regularization
    warmup_steps: int = 100            # LR warmup
    temperature: float = 0.07          # Contrastive loss temperature
    margin: float = 0.5                # Triplet loss margin
    batch_size: int = 32               # Training mini-batch size

    # Asymmetric encoding
    use_asymmetric: bool = False       # Separate query/memory heads

    # EWC (P3.5 - Elastic Weight Consolidation)
    ewc_enabled: bool = False          # Enable EWC regularization
    ewc_lambda: float = 1000.0         # EWC regularization strength
    ewc_online: bool = True            # Use online EWC (recommended)
    ewc_gamma: float = 0.95            # Decay factor for online EWC
    ewc_consolidation_interval: int = 50  # Steps between EWC consolidations

    # Persistence
    save_dir: str = ".ww_lora"         # Save directory
    checkpoint_interval: int = 100     # Steps between checkpoints


@dataclass
class RetrievalOutcome:
    """Outcome of a retrieval operation for training."""
    query_embedding: np.ndarray        # Query that was issued
    positive_embeddings: list[np.ndarray]  # Relevant retrieved items
    negative_embeddings: list[np.ndarray]  # Non-relevant items
    timestamp: datetime = field(default_factory=datetime.now)
    session_id: str | None = None


@dataclass
class LoRAState:
    """Serializable adapter state for persistence."""
    config: LoRAConfig
    step_count: int = 0
    training_losses: list[float] = field(default_factory=list)
    created_at: datetime = field(default_factory=datetime.now)
    updated_at: datetime = field(default_factory=datetime.now)

    def to_dict(self) -> dict:
        """Convert to JSON-serializable dict."""
        return {
            "config": {
                "embedding_dim": self.config.embedding_dim,
                "rank": self.config.rank,
                "alpha": self.config.alpha,
                "dropout": self.config.dropout,
                "learning_rate": self.config.learning_rate,
                "weight_decay": self.config.weight_decay,
                "warmup_steps": self.config.warmup_steps,
                "temperature": self.config.temperature,
                "margin": self.config.margin,
                "batch_size": self.config.batch_size,
                "use_asymmetric": self.config.use_asymmetric,
                "save_dir": self.config.save_dir,
                "checkpoint_interval": self.config.checkpoint_interval,
            },
            "step_count": self.step_count,
            "training_losses": self.training_losses[-100:],  # Keep last 100
            "created_at": self.created_at.isoformat(),
            "updated_at": self.updated_at.isoformat(),
        }

    @classmethod
    def from_dict(cls, data: dict) -> LoRAState:
        """Create from dict."""
        config = LoRAConfig(**data["config"])
        return cls(
            config=config,
            step_count=data["step_count"],
            training_losses=data["training_losses"],
            created_at=datetime.fromisoformat(data["created_at"]),
            updated_at=datetime.fromisoformat(data["updated_at"]),
        )


if TORCH_AVAILABLE:
    class LoRAModule(nn.Module):
        """
        LoRA adapter module for embedding transformation.

        Implements low-rank residual adaptation:
            adapted = x + scale * B(dropout(A(x)))

        where scale = alpha / rank for stability.

        This allows task-specific tuning with far fewer parameters
        than full fine-tuning (rank * dim * 2 vs dim * dim).
        """

        def __init__(self, config: LoRAConfig):
            """
            Initialize LoRA module.

            Args:
                config: Adapter configuration
            """
            super().__init__()
            self.config = config
            dim = config.embedding_dim
            rank = config.rank

            # Low-rank decomposition: W_adapted = W + BA
            # A: dim -> rank (down-projection)
            # B: rank -> dim (up-projection)
            self.A = nn.Linear(dim, rank, bias=False)
            self.B = nn.Linear(rank, dim, bias=False)

            # Dropout for regularization
            self.dropout = nn.Dropout(config.dropout)

            # Scaling factor (following LoRA paper)
            self.scale = config.alpha / config.rank

            # Initialize: A with small random, B with zeros
            # This ensures adapter starts as identity
            nn.init.kaiming_uniform_(self.A.weight, a=np.sqrt(5))
            nn.init.zeros_(self.B.weight)

            logger.info(
                f"LoRA module initialized: dim={dim}, rank={rank}, "
                f"params={self.num_parameters()}"
            )

        def forward(self, x: torch.Tensor) -> torch.Tensor:
            """
            Apply LoRA adaptation.

            Args:
                x: Input embeddings [batch, dim] or [dim]

            Returns:
                Adapted embeddings with same shape
            """
            # Low-rank transformation with residual connection
            delta = self.B(self.dropout(self.A(x)))
            return x + self.scale * delta

        def num_parameters(self) -> int:
            """Return number of trainable parameters."""
            return sum(p.numel() for p in self.parameters() if p.requires_grad)


    class AsymmetricLoRA(nn.Module):
        """
        Asymmetric query/memory LoRA adapter.

        Uses separate LoRA adapters for queries vs memories,
        following Dense Passage Retrieval (DPR) architecture insight
        that queries and documents benefit from different representations.
        """

        def __init__(self, config: LoRAConfig):
            """
            Initialize asymmetric adapter.

            Args:
                config: Adapter configuration
            """
            super().__init__()
            self.config = config

            # Separate adapters for query and memory
            self.query_adapter = LoRAModule(config)
            self.memory_adapter = LoRAModule(config)

            logger.info(
                f"Asymmetric LoRA initialized: "
                f"params={self.num_parameters()}"
            )

        def encode_query(self, x: torch.Tensor) -> torch.Tensor:
            """Adapt query embedding."""
            return self.query_adapter(x)

        def encode_memory(self, x: torch.Tensor) -> torch.Tensor:
            """Adapt memory embedding."""
            return self.memory_adapter(x)

        def forward(self, x: torch.Tensor, is_query: bool = True) -> torch.Tensor:
            """Apply appropriate adapter based on type."""
            if is_query:
                return self.encode_query(x)
            return self.encode_memory(x)

        def num_parameters(self) -> int:
            """Return number of trainable parameters."""
            return sum(p.numel() for p in self.parameters() if p.requires_grad)


class LoRAEmbeddingAdapter:
    """
    High-level LoRA adapter interface for T4DM embeddings.

    Wraps LoRA modules with:
    - Training loop using retrieval outcomes
    - State persistence
    - Integration with BGEM3Embedding
    - Numpy interface for compatibility
    - EWC regularization for continual learning (P3.5)
    """

    def __init__(
        self,
        config: LoRAConfig | None = None,
        device: str = "cuda:0"
    ):
        """
        Initialize LoRA embedding adapter.

        Args:
            config: Adapter configuration
            device: Torch device for computation
        """
        self.config = config or LoRAConfig()
        self.device = device if TORCH_AVAILABLE and torch.cuda.is_available() else "cpu"
        self.state = LoRAState(config=self.config)

        # Initialize adapter model
        self._adapter: nn.Module | None = None
        self._optimizer: torch.optim.Optimizer | None = None
        self._scheduler: torch.optim.lr_scheduler.LambdaLR | None = None

        # EWC regularizer (P3.5)
        self._ewc = None

        # Training buffer for outcomes
        self._training_buffer: list[RetrievalOutcome] = []

        if TORCH_AVAILABLE:
            self._init_model()
            self._init_ewc()

    def _init_model(self) -> None:
        """Initialize PyTorch model and optimizer."""
        if self.config.use_asymmetric:
            self._adapter = AsymmetricLoRA(self.config).to(self.device)
        else:
            self._adapter = LoRAModule(self.config).to(self.device)

        self._optimizer = torch.optim.AdamW(
            self._adapter.parameters(),
            lr=self.config.learning_rate,
            weight_decay=self.config.weight_decay
        )

        # Warmup scheduler
        def lr_lambda(step):
            if step < self.config.warmup_steps:
                return step / max(1, self.config.warmup_steps)
            return 1.0

        self._scheduler = torch.optim.lr_scheduler.LambdaLR(
            self._optimizer, lr_lambda
        )

    def _init_ewc(self) -> None:
        """Initialize EWC regularizer if enabled (P3.5)."""
        if not self.config.ewc_enabled:
            return

        try:
            from t4dm.learning.plasticity import EWCRegularizer
            self._ewc = EWCRegularizer(
                lambda_ewc=self.config.ewc_lambda,
                online=self.config.ewc_online,
                gamma=self.config.ewc_gamma,
            )
            logger.info(
                f"EWC initialized: λ={self.config.ewc_lambda}, "
                f"online={self.config.ewc_online}"
            )
        except ImportError:
            logger.warning("EWC not available (plasticity module not found)")
            self._ewc = None

    def adapt(self, embedding: np.ndarray, is_query: bool = True) -> np.ndarray:
        """
        Apply adapter to embedding.

        Args:
            embedding: Input embedding [dim] or [batch, dim]
            is_query: Whether this is a query (vs memory) embedding

        Returns:
            Adapted embedding with same shape
        """
        if not TORCH_AVAILABLE or self._adapter is None:
            return embedding

        # Convert to tensor
        was_1d = embedding.ndim == 1
        if was_1d:
            embedding = embedding[np.newaxis, :]

        x = torch.from_numpy(embedding).float().to(self.device)

        # Apply adapter
        self._adapter.eval()
        with torch.no_grad():
            if self.config.use_asymmetric and isinstance(self._adapter, AsymmetricLoRA):
                y = self._adapter(x, is_query=is_query)
            else:
                y = self._adapter(x)

        # Convert back to numpy
        result = y.cpu().numpy()
        if was_1d:
            result = result[0]

        return result

    def adapt_query(self, embedding: np.ndarray) -> np.ndarray:
        """Adapt query embedding."""
        return self.adapt(embedding, is_query=True)

    def adapt_memory(self, embedding: np.ndarray) -> np.ndarray:
        """Adapt memory embedding."""
        return self.adapt(embedding, is_query=False)

    def record_outcome(self, outcome: RetrievalOutcome) -> None:
        """
        Record retrieval outcome for training.

        Args:
            outcome: Retrieval outcome with query and positive/negative examples
        """
        self._training_buffer.append(outcome)

        # Train when buffer is full
        if len(self._training_buffer) >= self.config.batch_size:
            self._train_step()

    def _train_step(self) -> float | None:
        """
        Execute training step using buffered outcomes.

        Includes EWC penalty if enabled (P3.5) to prevent catastrophic forgetting.

        Returns:
            Training loss or None if training not possible
        """
        if not TORCH_AVAILABLE or self._adapter is None:
            self._training_buffer.clear()
            return None

        if not self._training_buffer:
            return None

        self._adapter.train()

        # Prepare batch
        losses = []
        for outcome in self._training_buffer:
            loss = self._compute_contrastive_loss(outcome)
            if loss is not None:
                losses.append(loss)

        if not losses:
            self._training_buffer.clear()
            return None

        # Compute task loss
        task_loss = torch.stack(losses).mean()

        # Add EWC penalty if enabled (P3.5)
        if self._ewc is not None:
            total_loss = self._ewc.ewc_loss(self._adapter, task_loss)
        else:
            total_loss = task_loss

        # Backward pass
        self._optimizer.zero_grad()
        total_loss.backward()

        # Gradient clipping for stability
        torch.nn.utils.clip_grad_norm_(self._adapter.parameters(), max_norm=1.0)

        self._optimizer.step()
        self._scheduler.step()

        # Update state
        loss_value = total_loss.item()
        self.state.step_count += 1
        self.state.training_losses.append(loss_value)
        self.state.updated_at = datetime.now()

        # Clear buffer
        self._training_buffer.clear()

        # EWC consolidation at intervals (P3.5)
        if (self._ewc is not None and
            self.state.step_count % self.config.ewc_consolidation_interval == 0):
            self._ewc_consolidate()

        # Checkpoint if needed
        if self.state.step_count % self.config.checkpoint_interval == 0:
            self.save()

        logger.debug(
            f"LoRA training step {self.state.step_count}: loss={loss_value:.4f}"
            f"{' (with EWC)' if self._ewc else ''}"
        )

        return loss_value

    def _ewc_consolidate(self) -> None:
        """
        Consolidate current knowledge with EWC (P3.5).

        Creates a simple dataloader from recent training buffer for
        Fisher information estimation.
        """
        if self._ewc is None or self._adapter is None:
            return

        # Create simple dataloader from recent embeddings
        # We use random embeddings as a proxy since we don't store training data
        try:
            # Generate synthetic data for Fisher estimation
            n_samples = min(50, self._ewc.fisher_n_samples)
            synthetic_data = [
                torch.randn(1, self.config.embedding_dim, device=self.device)
                for _ in range(n_samples)
            ]

            # Create simple iterable
            class SimpleLoader:
                def __init__(self, data):
                    self.data = data
                def __iter__(self):
                    return iter(self.data)

            dataloader = SimpleLoader(synthetic_data)
            self._ewc.consolidate(self._adapter, dataloader, device=self.device)
            logger.debug(f"EWC consolidation at step {self.state.step_count}")
        except Exception as e:
            logger.warning(f"EWC consolidation failed: {e}")

    def _compute_contrastive_loss(
        self,
        outcome: RetrievalOutcome
    ) -> torch.Tensor | None:
        """
        Compute InfoNCE contrastive loss for retrieval outcome.

        Loss encourages:
        - High similarity between query and positives
        - Low similarity between query and negatives

        Args:
            outcome: Retrieval outcome

        Returns:
            Loss tensor or None if insufficient examples
        """
        if not outcome.positive_embeddings:
            return None

        # Convert to tensors
        query = torch.from_numpy(outcome.query_embedding).float().to(self.device)

        positives = torch.stack([
            torch.from_numpy(e).float() for e in outcome.positive_embeddings
        ]).to(self.device)

        # Apply adapter
        if self.config.use_asymmetric and isinstance(self._adapter, AsymmetricLoRA):
            query_adapted = self._adapter(query.unsqueeze(0), is_query=True).squeeze(0)
            positives_adapted = self._adapter(positives, is_query=False)
        else:
            query_adapted = self._adapter(query.unsqueeze(0)).squeeze(0)
            positives_adapted = self._adapter(positives)

        # Normalize for cosine similarity
        query_adapted = F.normalize(query_adapted, dim=-1)
        positives_adapted = F.normalize(positives_adapted, dim=-1)

        # Positive similarities
        pos_sim = torch.matmul(positives_adapted, query_adapted)  # [num_pos]

        # If we have negatives, use full InfoNCE
        if outcome.negative_embeddings:
            negatives = torch.stack([
                torch.from_numpy(e).float() for e in outcome.negative_embeddings
            ]).to(self.device)

            if self.config.use_asymmetric and isinstance(self._adapter, AsymmetricLoRA):
                negatives_adapted = self._adapter(negatives, is_query=False)
            else:
                negatives_adapted = self._adapter(negatives)

            negatives_adapted = F.normalize(negatives_adapted, dim=-1)
            neg_sim = torch.matmul(negatives_adapted, query_adapted)  # [num_neg]

            # InfoNCE: -log(exp(pos) / (exp(pos) + sum(exp(neg))))
            all_sim = torch.cat([pos_sim, neg_sim]) / self.config.temperature
            labels = torch.zeros(len(pos_sim), dtype=torch.long, device=self.device)
            loss = F.cross_entropy(
                all_sim.unsqueeze(0).expand(len(pos_sim), -1),
                labels
            )
        else:
            # Simple margin loss without negatives
            # Encourage high similarity to positives
            loss = (1.0 - pos_sim).mean()

        return loss

    def train_on_outcomes(
        self,
        outcomes: list[RetrievalOutcome],
        epochs: int = 1
    ) -> list[float]:
        """
        Train adapter on batch of outcomes.

        Args:
            outcomes: List of retrieval outcomes
            epochs: Number of passes over data

        Returns:
            List of training losses
        """
        if not TORCH_AVAILABLE or self._adapter is None:
            logger.warning("Training not available (PyTorch missing)")
            return []

        losses = []
        for epoch in range(epochs):
            for outcome in outcomes:
                self._training_buffer.append(outcome)
                if len(self._training_buffer) >= self.config.batch_size:
                    loss = self._train_step()
                    if loss is not None:
                        losses.append(loss)

            # Flush remaining
            if self._training_buffer:
                loss = self._train_step()
                if loss is not None:
                    losses.append(loss)

        return losses

    def save(self, path: str | None = None) -> str:
        """
        Save adapter state and weights.

        Args:
            path: Save directory (default: config.save_dir)

        Returns:
            Path where saved
        """
        save_dir = Path(path or self.config.save_dir)
        save_dir.mkdir(parents=True, exist_ok=True)

        # Save state
        state_path = save_dir / "lora_state.json"
        with open(state_path, "w") as f:
            json.dump(self.state.to_dict(), f, indent=2)

        # Save weights
        if TORCH_AVAILABLE and self._adapter is not None:
            weights_path = save_dir / "lora_weights.pt"
            torch.save(self._adapter.state_dict(), weights_path)
            logger.info(f"LoRA adapter saved to {save_dir}")
        else:
            logger.info(f"LoRA adapter state saved to {save_dir} (no weights)")

        return str(save_dir)

    def load(self, path: str | None = None) -> bool:
        """
        Load adapter state and weights.

        Args:
            path: Load directory (default: config.save_dir)

        Returns:
            True if loaded successfully
        """
        load_dir = Path(path or self.config.save_dir)

        # Load state
        state_path = load_dir / "lora_state.json"
        if state_path.exists():
            with open(state_path) as f:
                self.state = LoRAState.from_dict(json.load(f))
            self.config = self.state.config
            logger.info(f"Loaded LoRA adapter state: step={self.state.step_count}")
        else:
            logger.warning(f"No state found at {state_path}")
            return False

        # Reinitialize model with loaded config
        if TORCH_AVAILABLE:
            self._init_model()

            # Load weights
            weights_path = load_dir / "lora_weights.pt"
            if weights_path.exists():
                self._adapter.load_state_dict(
                    torch.load(weights_path, map_location=self.device, weights_only=True)
                )
                logger.info(f"Loaded LoRA weights from {weights_path}")
            else:
                logger.warning(f"No weights found at {weights_path}")

        return True

    def get_stats(self) -> dict:
        """Get adapter statistics."""
        stats = {
            "step_count": self.state.step_count,
            "torch_available": TORCH_AVAILABLE,
            "device": self.device,
            "config": {
                "rank": self.config.rank,
                "alpha": self.config.alpha,
                "embedding_dim": self.config.embedding_dim,
                "use_asymmetric": self.config.use_asymmetric,
                "ewc_enabled": self.config.ewc_enabled,
            },
        }

        if TORCH_AVAILABLE and self._adapter is not None:
            stats["num_parameters"] = self._adapter.num_parameters()
            stats["parameter_efficiency"] = (
                self._adapter.num_parameters() /
                (self.config.embedding_dim ** 2) * 100
            )  # % of full matrix

        if self.state.training_losses:
            recent_losses = self.state.training_losses[-10:]
            stats["recent_avg_loss"] = sum(recent_losses) / len(recent_losses)

        # EWC stats (P3.5)
        if self._ewc is not None:
            stats["ewc"] = self._ewc.get_stats()

        return stats

    def reset(self) -> None:
        """Reset adapter to initial state."""
        self.state = LoRAState(config=self.config)
        self._training_buffer.clear()
        if TORCH_AVAILABLE:
            self._init_model()
        logger.info("LoRA adapter reset to initial state")


class AdaptedBGEM3Provider:
    """
    BGE-M3 embedding provider with LoRA adaptation.

    Wraps BGEM3Embedding to provide adapted embeddings transparently.
    """

    def __init__(
        self,
        base_provider: BGEM3Embedding,
        adapter: LoRAEmbeddingAdapter | None = None,
        adapter_config: LoRAConfig | None = None,
    ):
        """
        Initialize adapted provider.

        Args:
            base_provider: Base embedding provider (BGEM3Embedding)
            adapter: Pre-configured adapter or None to create
            adapter_config: Config for new adapter if not provided
        """
        self.base = base_provider
        self.adapter = adapter or LoRAEmbeddingAdapter(
            config=adapter_config or LoRAConfig()
        )

    @property
    def dimension(self) -> int:
        """Return embedding dimension (same as base)."""
        return self.base.dimension

    async def embed_query(self, query: str) -> list[float]:
        """
        Generate adapted query embedding.

        Args:
            query: Query text

        Returns:
            Adapted embedding vector
        """
        # Get base embedding
        base_emb = await self.base.embed_query(query)

        # Apply adapter
        adapted = self.adapter.adapt_query(np.array(base_emb))

        return adapted.tolist()

    async def embed(self, texts: list[str]) -> list[list[float]]:
        """
        Generate adapted embeddings for texts.

        Args:
            texts: List of texts

        Returns:
            List of adapted embeddings
        """
        if not texts:
            return []

        # Get base embeddings
        base_embs = await self.base.embed(texts)

        # Apply adapter (as memory embeddings)
        adapted = [
            self.adapter.adapt_memory(np.array(e)).tolist()
            for e in base_embs
        ]

        return adapted

    async def embed_batch(
        self,
        texts: list[str],
        show_progress: bool = False
    ) -> list[list[float]]:
        """
        Generate adapted embeddings for large batch.

        Args:
            texts: Large list of texts
            show_progress: Show progress

        Returns:
            List of adapted embeddings
        """
        if not texts:
            return []

        base_embs = await self.base.embed_batch(texts, show_progress)

        # Batch adapt
        base_array = np.array(base_embs)
        adapted_array = self.adapter.adapt_memory(base_array)

        return adapted_array.tolist()

    def record_retrieval_outcome(
        self,
        query_embedding: np.ndarray,
        retrieved_embeddings: list[np.ndarray],
        relevance_scores: list[float],
        relevance_threshold: float = 0.5
    ) -> None:
        """
        Record retrieval outcome for adapter training.

        Args:
            query_embedding: Query embedding used
            retrieved_embeddings: Retrieved item embeddings
            relevance_scores: Relevance scores (0-1)
            relevance_threshold: Threshold for positive classification
        """
        # Split into positive/negative based on relevance
        positives = []
        negatives = []

        for emb, score in zip(retrieved_embeddings, relevance_scores):
            if score >= relevance_threshold:
                positives.append(emb)
            else:
                negatives.append(emb)

        if positives:  # Only record if we have positives
            outcome = RetrievalOutcome(
                query_embedding=query_embedding,
                positive_embeddings=positives,
                negative_embeddings=negatives,
            )
            self.adapter.record_outcome(outcome)

    def get_adapter_stats(self) -> dict:
        """Get adapter statistics."""
        return self.adapter.get_stats()

    def save_adapter(self, path: str | None = None) -> str:
        """Save adapter state."""
        return self.adapter.save(path)

    def load_adapter(self, path: str | None = None) -> bool:
        """Load adapter state."""
        return self.adapter.load(path)


# Convenience factory functions

def create_lora_adapter(
    rank: int = 16,
    use_asymmetric: bool = False,
    device: str = "cuda:0",
    **kwargs
) -> LoRAEmbeddingAdapter:
    """
    Create LoRA embedding adapter.

    Args:
        rank: LoRA rank (lower = fewer params, higher = more capacity)
        use_asymmetric: Use separate query/memory adapters
        device: Torch device
        **kwargs: Additional config options

    Returns:
        Configured LoRA adapter
    """
    config = LoRAConfig(
        rank=rank,
        use_asymmetric=use_asymmetric,
        **kwargs
    )
    return LoRAEmbeddingAdapter(config=config, device=device)


def create_adapted_provider(
    base_provider: BGEM3Embedding,
    rank: int = 16,
    use_asymmetric: bool = False,
    **kwargs
) -> AdaptedBGEM3Provider:
    """
    Create adapted BGE-M3 provider with LoRA.

    Args:
        base_provider: Base embedding provider
        rank: LoRA rank
        use_asymmetric: Use separate query/memory adapters
        **kwargs: Additional config options

    Returns:
        Adapted provider
    """
    adapter = create_lora_adapter(rank=rank, use_asymmetric=use_asymmetric, **kwargs)
    return AdaptedBGEM3Provider(base_provider=base_provider, adapter=adapter)
