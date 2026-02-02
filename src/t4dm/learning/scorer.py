"""
Neural Retrieval Scorer for T4DM.

Implements learnable retrieval scoring using PyTorch:
1. LearnedRetrievalScorer: Neural network for scoring memories
2. PrioritizedReplayBuffer: Experience replay with TD-error priority
3. ListMLELoss: ListMLE ranking loss for learning-to-rank
4. ScorerTrainer: Training loop with online learning support
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from pathlib import Path

import torch
import torch.nn as nn
import torch.nn.functional as F

from t4dm.learning.events import Experience, MemoryType

logger = logging.getLogger(__name__)


# =============================================================================
# Neural Scorer
# =============================================================================

class LearnedRetrievalScorer(nn.Module):
    """
    Neural network for scoring retrieved memories.

    Takes component scores (similarity, recency, importance, etc.) and
    learns optimal weights for combining them based on outcome feedback.

    Architecture:
    - Input: Component score vector per memory
    - Hidden: 2-layer MLP with residual connection
    - Output: Scalar relevance score

    The model learns to predict which memories are most useful for
    achieving positive outcomes.
    """

    def __init__(
        self,
        input_dim: int = 4,  # similarity, recency, importance, outcome_history
        hidden_dim: int = 32,
        dropout: float = 0.1,
    ):
        super().__init__()
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim

        # Feature transformation
        self.fc1 = nn.Linear(input_dim, hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, hidden_dim)
        self.fc_out = nn.Linear(hidden_dim, 1)

        # Normalization and regularization
        self.norm1 = nn.LayerNorm(hidden_dim)
        self.norm2 = nn.LayerNorm(hidden_dim)
        self.dropout = nn.Dropout(dropout)

        # Residual projection if dimensions differ
        self.residual_proj = nn.Linear(input_dim, hidden_dim) if input_dim != hidden_dim else None

        # Initialize with small weights for stable learning
        self._init_weights()

    def _init_weights(self):
        """Initialize weights for stable training."""
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight, gain=0.1)
                if m.bias is not None:
                    nn.init.zeros_(m.bias)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass.

        Args:
            x: Component scores [batch, n_memories, input_dim]

        Returns:
            Relevance scores [batch, n_memories]
        """
        # First layer
        h = self.fc1(x)
        h = self.norm1(h)
        h = F.gelu(h)
        h = self.dropout(h)

        # Second layer with residual
        h2 = self.fc2(h)
        h2 = self.norm2(h2)
        h2 = F.gelu(h2)

        # Residual connection
        if self.residual_proj is not None:
            h2 = h2 + self.residual_proj(x)
        else:
            h2 = h2 + h

        # Output score
        score = self.fc_out(h2).squeeze(-1)
        return score

    def score_memories(
        self,
        component_vectors: list[list[float]],
        device: torch.device | None = None,
    ) -> list[float]:
        """
        Score a list of memories.

        Args:
            component_vectors: List of component score vectors per memory
            device: Device to run on

        Returns:
            List of relevance scores
        """
        if not component_vectors:
            return []

        device = device or next(self.parameters()).device
        x = torch.tensor(component_vectors, dtype=torch.float32, device=device)

        # Add batch dimension if needed
        if x.dim() == 1:
            x = x.unsqueeze(0)
        if x.dim() == 2:
            x = x.unsqueeze(0)

        with torch.no_grad():
            scores = self(x)

        return scores.squeeze(0).tolist()


# =============================================================================
# Prioritized Replay Buffer
# =============================================================================

@dataclass
class ReplayItem:
    """Single replay buffer item."""
    experience_id: str
    query: str
    memory_type: MemoryType
    component_vectors: list[list[float]]
    rewards: list[float]
    priority: float = 1.0


class PrioritizedReplayBuffer:
    """
    Prioritized experience replay buffer.

    Stores experiences and samples them proportional to their
    TD-error priority. Higher-error experiences are sampled more
    frequently for faster learning.

    Uses sum-tree for O(log n) sampling.
    """

    def __init__(
        self,
        capacity: int = 10000,
        alpha: float = 0.6,  # Priority exponent
        beta: float = 0.4,  # Importance sampling exponent
        beta_increment: float = 0.001,
    ):
        self.capacity = capacity
        self.alpha = alpha
        self.beta = beta
        self.beta_increment = beta_increment

        self.buffer: list[ReplayItem] = []
        self.priorities: list[float] = []
        self.position = 0
        self.max_priority = 1.0

    def __len__(self) -> int:
        return len(self.buffer)

    def add(self, item: ReplayItem) -> None:
        """Add experience to buffer."""
        if len(self.buffer) < self.capacity:
            self.buffer.append(item)
            self.priorities.append(self.max_priority)
        else:
            self.buffer[self.position] = item
            self.priorities[self.position] = self.max_priority

        self.position = (self.position + 1) % self.capacity

    def add_from_experience(self, exp: Experience) -> None:
        """Add from Experience dataclass."""
        # Extract rewards as list matching component_vectors order
        rewards = list(exp.per_memory_rewards.values())

        item = ReplayItem(
            experience_id=str(exp.experience_id),
            query=exp.query,
            memory_type=exp.memory_type,
            component_vectors=exp.component_vectors,
            rewards=rewards,
            priority=exp.priority,
        )
        self.add(item)

    def sample(self, batch_size: int) -> tuple[list[ReplayItem], list[int], list[float]]:
        """
        Sample batch with prioritized sampling.

        Returns:
            (items, indices, importance_weights)
        """
        if len(self.buffer) == 0:
            return [], [], []

        n = len(self.buffer)
        batch_size = min(batch_size, n)

        # Compute sampling probabilities
        priorities = torch.tensor(self.priorities[:n], dtype=torch.float32)
        probs = (priorities ** self.alpha)
        probs = probs / probs.sum()

        # Sample indices
        indices = torch.multinomial(probs, batch_size, replacement=False).tolist()

        # Compute importance sampling weights
        min_prob = probs.min()
        max_weight = (n * min_prob) ** (-self.beta)
        weights = []
        for idx in indices:
            weight = (n * probs[idx]) ** (-self.beta) / max_weight
            weights.append(weight.item())

        # Anneal beta
        self.beta = min(1.0, self.beta + self.beta_increment)

        items = [self.buffer[i] for i in indices]
        return items, indices, weights

    def update_priorities(self, indices: list[int], td_errors: list[float]) -> None:
        """Update priorities based on TD errors."""
        for idx, td_error in zip(indices, td_errors):
            priority = abs(td_error) + 1e-6
            self.priorities[idx] = priority
            self.max_priority = max(self.max_priority, priority)


# =============================================================================
# ListMLE Loss
# =============================================================================

class ListMLELoss(nn.Module):
    """
    ListMLE loss for learning-to-rank.

    Given predicted scores and ground truth relevance (rewards),
    computes the negative log-likelihood of the correct ranking.

    Reference: Xia et al., "Listwise Approach to Learning to Rank"
    """

    def __init__(self, temperature: float = 1.0, eps: float = 1e-10):
        super().__init__()
        self.temperature = temperature
        self.eps = eps

    def forward(
        self,
        scores: torch.Tensor,
        relevance: torch.Tensor,
        mask: torch.Tensor | None = None,
    ) -> torch.Tensor:
        """
        Compute ListMLE loss.

        Args:
            scores: Predicted scores [batch, n_items]
            relevance: Ground truth relevance [batch, n_items]
            mask: Optional mask for valid items [batch, n_items]

        Returns:
            Scalar loss
        """
        batch_size, n_items = scores.shape

        if mask is None:
            mask = torch.ones_like(scores)

        # Sort by relevance (descending)
        sorted_relevance, sorted_idx = relevance.sort(descending=True, dim=1)
        sorted_scores = scores.gather(1, sorted_idx)
        sorted_mask = mask.gather(1, sorted_idx)

        # Apply mask
        sorted_scores = sorted_scores.masked_fill(sorted_mask == 0, float("-inf"))

        # Compute log-likelihood
        # For each position, compute log(softmax) over remaining items
        loss = torch.zeros(batch_size, device=scores.device)

        for i in range(n_items - 1):
            # Scores from position i onwards
            remaining_scores = sorted_scores[:, i:] / self.temperature

            # Log softmax
            log_probs = F.log_softmax(remaining_scores, dim=1)

            # We want the first item to be selected
            loss = loss - log_probs[:, 0] * sorted_mask[:, i]

        return loss.mean()


# =============================================================================
# Trainer
# =============================================================================

@dataclass
class TrainerConfig:
    """Configuration for scorer training."""
    learning_rate: float = 1e-4
    weight_decay: float = 0.01
    batch_size: int = 32
    epochs_per_update: int = 1
    warmup_steps: int = 100
    grad_clip: float = 1.0
    checkpoint_dir: Path | None = None
    device: str = "auto"

    # EWC (P3.5 - Elastic Weight Consolidation)
    ewc_enabled: bool = False          # Enable EWC regularization
    ewc_lambda: float = 1000.0         # EWC regularization strength
    ewc_online: bool = True            # Use online EWC (recommended)
    ewc_gamma: float = 0.95            # Decay factor for online EWC
    ewc_consolidation_interval: int = 100  # Steps between EWC consolidations


class ScorerTrainer:
    """
    Trainer for LearnedRetrievalScorer.

    Supports both batch training from replay buffer and
    online learning from individual experiences.
    Includes EWC regularization for continual learning (P3.5).
    """

    def __init__(
        self,
        scorer: LearnedRetrievalScorer,
        config: TrainerConfig,
        replay_buffer: PrioritizedReplayBuffer | None = None,
    ):
        self.scorer = scorer
        self.config = config
        self.replay_buffer = replay_buffer or PrioritizedReplayBuffer()

        # Setup device
        if config.device == "auto":
            self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        else:
            self.device = torch.device(config.device)

        self.scorer.to(self.device)

        # Optimizer
        self.optimizer = torch.optim.AdamW(
            scorer.parameters(),
            lr=config.learning_rate,
            weight_decay=config.weight_decay,
        )

        # Loss
        self.loss_fn = ListMLELoss()

        # EWC regularizer (P3.5)
        self._ewc = None
        if config.ewc_enabled:
            self._init_ewc()

        # Training state
        self.step = 0
        self.total_loss = 0.0

    def _init_ewc(self) -> None:
        """Initialize EWC regularizer if enabled (P3.5)."""
        try:
            from t4dm.learning.plasticity import EWCRegularizer
            self._ewc = EWCRegularizer(
                lambda_ewc=self.config.ewc_lambda,
                online=self.config.ewc_online,
                gamma=self.config.ewc_gamma,
            )
            logger.info(
                f"Scorer EWC initialized: λ={self.config.ewc_lambda}, "
                f"online={self.config.ewc_online}"
            )
        except ImportError:
            logger.warning("EWC not available (plasticity module not found)")
            self._ewc = None

    def train_step(self) -> float | None:
        """
        Single training step from replay buffer.

        Includes EWC penalty if enabled (P3.5) to prevent catastrophic forgetting.

        Returns:
            Loss value or None if buffer empty
        """
        if len(self.replay_buffer) < self.config.batch_size:
            return None

        self.scorer.train()

        # Sample batch
        items, indices, weights = self.replay_buffer.sample(self.config.batch_size)

        # Prepare tensors
        max_len = max(len(item.component_vectors) for item in items)
        input_dim = self.scorer.input_dim

        # Pad sequences
        batch_components = torch.zeros(len(items), max_len, input_dim, device=self.device)
        batch_rewards = torch.zeros(len(items), max_len, device=self.device)
        batch_mask = torch.zeros(len(items), max_len, device=self.device)

        for i, item in enumerate(items):
            n = len(item.component_vectors)
            if n > 0:
                batch_components[i, :n] = torch.tensor(item.component_vectors, device=self.device)
                batch_rewards[i, :n] = torch.tensor(item.rewards, device=self.device)
                batch_mask[i, :n] = 1.0

        # Forward pass
        scores = self.scorer(batch_components)

        # Compute task loss with importance sampling weights
        task_loss = self.loss_fn(scores, batch_rewards, batch_mask)
        weights_tensor = torch.tensor(weights, device=self.device)
        weighted_loss = (task_loss * weights_tensor).mean()

        # Add EWC penalty if enabled (P3.5)
        if self._ewc is not None:
            total_loss = self._ewc.ewc_loss(self.scorer, weighted_loss)
        else:
            total_loss = weighted_loss

        # Backward pass
        self.optimizer.zero_grad()
        total_loss.backward()

        # Gradient clipping
        torch.nn.utils.clip_grad_norm_(
            self.scorer.parameters(),
            self.config.grad_clip,
        )

        self.optimizer.step()

        # Update priorities based on TD error
        with torch.no_grad():
            td_errors = (scores - batch_rewards).abs().sum(dim=1).tolist()
        self.replay_buffer.update_priorities(indices, td_errors)

        self.step += 1
        self.total_loss += total_loss.item()

        # EWC consolidation at intervals (P3.5)
        if (self._ewc is not None and
            self.step % self.config.ewc_consolidation_interval == 0):
            self._ewc_consolidate()

        return total_loss.item()

    def _ewc_consolidate(self) -> None:
        """
        Consolidate current knowledge with EWC (P3.5).

        Creates a dataloader from replay buffer for Fisher estimation.
        """
        if self._ewc is None:
            return

        try:
            # Use replay buffer data for Fisher estimation
            items = list(self.replay_buffer.buffer)[:self._ewc.fisher_n_samples]
            if not items:
                return

            # Create dataloader from buffer items
            input_dim = self.scorer.input_dim

            class ReplayLoader:
                def __init__(self, replay_items, input_dim, device):
                    self.items = replay_items
                    self.input_dim = input_dim
                    self.device = device

                def __iter__(self):
                    for item in self.items:
                        if item.component_vectors:
                            x = torch.tensor(
                                item.component_vectors,
                                dtype=torch.float32,
                                device=self.device
                            ).unsqueeze(0)  # [1, n_memories, input_dim]
                            yield x

            dataloader = ReplayLoader(items, input_dim, self.device)
            self._ewc.consolidate(self.scorer, dataloader, device=self.device)
            logger.debug(f"Scorer EWC consolidation at step {self.step}")
        except Exception as e:
            logger.warning(f"Scorer EWC consolidation failed: {e}")

    def train_epoch(self) -> float:
        """
        Train for one epoch.

        Returns:
            Average loss for epoch
        """
        steps = len(self.replay_buffer) // self.config.batch_size
        epoch_loss = 0.0

        for _ in range(steps):
            loss = self.train_step()
            if loss is not None:
                epoch_loss += loss

        return epoch_loss / max(steps, 1)

    def add_experience(self, exp: Experience) -> None:
        """Add experience to replay buffer."""
        self.replay_buffer.add_from_experience(exp)

    def save_checkpoint(self, path: Path) -> None:
        """Save model checkpoint."""
        torch.save({
            "model_state_dict": self.scorer.state_dict(),
            "optimizer_state_dict": self.optimizer.state_dict(),
            "step": self.step,
            "total_loss": self.total_loss,
        }, path)
        logger.info(f"Saved checkpoint to {path}")

    def load_checkpoint(self, path: Path) -> None:
        """Load model checkpoint."""
        checkpoint = torch.load(path, map_location=self.device)
        self.scorer.load_state_dict(checkpoint["model_state_dict"])
        self.optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
        self.step = checkpoint["step"]
        self.total_loss = checkpoint["total_loss"]
        logger.info(f"Loaded checkpoint from {path}")

    @property
    def avg_loss(self) -> float:
        """Average loss over all steps."""
        return self.total_loss / max(self.step, 1)


# =============================================================================
# Factory Functions
# =============================================================================

def create_scorer(
    input_dim: int = 4,
    hidden_dim: int = 32,
    device: str = "auto",
) -> LearnedRetrievalScorer:
    """Create a new scorer."""
    scorer = LearnedRetrievalScorer(input_dim=input_dim, hidden_dim=hidden_dim)

    if device == "auto":
        device = "cuda" if torch.cuda.is_available() else "cpu"

    return scorer.to(device)


def create_trainer(
    scorer: LearnedRetrievalScorer | None = None,
    config: TrainerConfig | None = None,
) -> ScorerTrainer:
    """Create a trainer with default configuration."""
    scorer = scorer or create_scorer()
    config = config or TrainerConfig()
    return ScorerTrainer(scorer, config)


# =============================================================================
# Module Exports
# =============================================================================

__all__ = [
    "LearnedRetrievalScorer",
    "ListMLELoss",
    "PrioritizedReplayBuffer",
    "ReplayItem",
    "ScorerTrainer",
    "TrainerConfig",
    "create_scorer",
    "create_trainer",
]
