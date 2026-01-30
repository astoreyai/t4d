"""
Learned fusion weights for episodic memory retrieval scoring.

Implements query-dependent weight learning for semantic/recency/outcome/importance components.
"""

import numpy as np


class LearnedFusionWeights:
    """
    R1: Query-dependent fusion weights for retrieval scoring.

    Replaces fixed weights (0.4/0.3/0.2/0.1) with learned, query-adaptive weights.
    Uses a simple 2-layer MLP in numpy for speed (no torch in hot path).

    Components:
    - semantic: Vector similarity weight
    - recency: Temporal relevance weight
    - outcome: Success history weight
    - importance: Emotional valence weight

    Training:
    - Online gradient descent from retrieval outcomes
    - Query embedding → hidden → softmax weights
    """

    def __init__(
        self,
        embed_dim: int = 1024,
        hidden_dim: int = 32,
        learning_rate: float = 0.01,
        n_components: int = 4,
    ):
        self.embed_dim = embed_dim
        self.hidden_dim = hidden_dim
        self.lr = learning_rate
        self.n_components = n_components

        # Xavier initialization
        self.W1 = np.random.randn(hidden_dim, embed_dim).astype(np.float32) * np.sqrt(2.0 / (embed_dim + hidden_dim))
        self.b1 = np.zeros(hidden_dim, dtype=np.float32)
        self.W2 = np.random.randn(n_components, hidden_dim).astype(np.float32) * np.sqrt(2.0 / (hidden_dim + n_components))
        self.b2 = np.zeros(n_components, dtype=np.float32)

        # Default weights (used during cold start)
        self.default_weights = np.array([0.4, 0.3, 0.2, 0.1], dtype=np.float32)

        # Training stats
        self.n_updates = 0
        self.cold_start_threshold = 50

        # Component names
        self.component_names = ["semantic", "recency", "outcome", "importance"]

    def _softmax(self, x: np.ndarray) -> np.ndarray:
        """Numerically stable softmax."""
        exp_x = np.exp(x - np.max(x))
        return exp_x / exp_x.sum()

    def compute_weights(self, query_embedding: np.ndarray) -> np.ndarray:
        """
        Compute query-dependent fusion weights.

        Args:
            query_embedding: Query vector (1024-dim)

        Returns:
            4-component weights [semantic, recency, outcome, importance]
        """
        # Ensure correct dimension
        if len(query_embedding) > self.embed_dim:
            query_embedding = query_embedding[:self.embed_dim]
        elif len(query_embedding) < self.embed_dim:
            query_embedding = np.pad(query_embedding, (0, self.embed_dim - len(query_embedding)))

        # Forward pass
        hidden = np.tanh(self.W1 @ query_embedding + self.b1)
        logits = self.W2 @ hidden + self.b2
        weights = self._softmax(logits)

        # Blend with default during cold start
        if self.n_updates < self.cold_start_threshold:
            blend = self.n_updates / self.cold_start_threshold
            weights = blend * weights + (1 - blend) * self.default_weights

        return weights

    def get_weights_dict(self, query_embedding: np.ndarray) -> dict[str, float]:
        """Get interpretable weight dictionary."""
        weights = self.compute_weights(query_embedding)
        return {name: float(w) for name, w in zip(self.component_names, weights)}

    def update(
        self,
        query_embedding: np.ndarray,
        component_scores: dict[str, float],
        outcome_utility: float,
    ) -> None:
        """
        Update weights based on retrieval outcome.

        Uses gradient descent to increase weights of components that
        correlated positively with outcome.

        Args:
            query_embedding: Query used for retrieval
            component_scores: {component: score} from retrieval
            outcome_utility: How useful was the retrieval (0-1)
        """
        if len(query_embedding) > self.embed_dim:
            query_embedding = query_embedding[:self.embed_dim]
        elif len(query_embedding) < self.embed_dim:
            query_embedding = np.pad(query_embedding, (0, self.embed_dim - len(query_embedding)))

        # Forward pass (cache for backprop)
        hidden = np.tanh(self.W1 @ query_embedding + self.b1)
        logits = self.W2 @ hidden + self.b2
        weights = self._softmax(logits)

        # Compute gradient signal: increase weights for components
        # that were high when outcome was good
        scores = np.array([
            component_scores.get(name, 0.0)
            for name in self.component_names
        ], dtype=np.float32)

        # Target: shift weights toward components that correlated with utility
        # If utility > 0.5, emphasize high-scoring components
        # If utility < 0.5, de-emphasize them
        utility_centered = outcome_utility - 0.5
        target_shift = utility_centered * (scores - scores.mean())
        target_shift = target_shift / (np.abs(target_shift).max() + 1e-8) * 0.1

        # Gradient of cross-entropy-like loss
        error = weights - (weights + target_shift)
        error = np.clip(error, -0.1, 0.1)

        # Backprop through softmax → W2
        grad_logits = error * weights * (1 - weights)  # Simplified Jacobian
        grad_W2 = np.outer(grad_logits, hidden)
        grad_b2 = grad_logits

        # Backprop through tanh → W1
        tanh_grad = 1 - hidden ** 2
        grad_hidden = (self.W2.T @ grad_logits) * tanh_grad
        grad_W1 = np.outer(grad_hidden, query_embedding)
        grad_b1 = grad_hidden

        # Update
        self.W2 -= self.lr * grad_W2
        self.b2 -= self.lr * grad_b2
        self.W1 -= self.lr * grad_W1
        self.b1 -= self.lr * grad_b1

        self.n_updates += 1

    # MEMORY-HIGH-004 FIX: Add save/load methods for weight persistence
    def save_state(self) -> dict:
        """
        Save model weights and state to dictionary.

        Returns:
            Dictionary with all model parameters
        """
        return {
            "W1": self.W1.tolist(),
            "b1": self.b1.tolist(),
            "W2": self.W2.tolist(),
            "b2": self.b2.tolist(),
            "n_updates": self.n_updates,
            "embed_dim": self.embed_dim,
            "hidden_dim": self.hidden_dim,
            "lr": self.lr,
            "n_components": self.n_components,
        }

    def load_state(self, state: dict) -> None:
        """
        Load model weights and state from dictionary.

        Args:
            state: Dictionary from save_state()
        """
        self.W1 = np.array(state["W1"], dtype=np.float32)
        self.b1 = np.array(state["b1"], dtype=np.float32)
        self.W2 = np.array(state["W2"], dtype=np.float32)
        self.b2 = np.array(state["b2"], dtype=np.float32)
        self.n_updates = state.get("n_updates", 0)
        # Optionally update config if dimensions match
        if state.get("embed_dim") == self.embed_dim and state.get("hidden_dim") == self.hidden_dim:
            self.lr = state.get("lr", self.lr)


class LearnedReranker:
    """
    P0c: Learned re-ranking for retrieval results.

    Post-retrieval re-ranking using a 2-layer MLP that considers:
    - Component scores (semantic, recency, outcome, importance)
    - Query context (via query embedding projection)

    This provides a second-pass scoring that can capture
    cross-component interactions and query-specific patterns.

    Architecture:
    - Input: [component_scores (4), query_context (16)] = 20-dim
    - Hidden: 32 units with tanh
    - Output: scalar score with residual from initial score

    The residual connection allows the model to learn small adjustments
    while preserving the base scoring behavior.
    """

    COMPONENT_DIM = 4  # semantic, recency, outcome, importance
    QUERY_CONTEXT_DIM = 16  # Compressed query representation
    INPUT_DIM = COMPONENT_DIM + QUERY_CONTEXT_DIM  # 20
    HIDDEN_DIM = 32

    def __init__(
        self,
        embed_dim: int = 1024,
        learning_rate: float = 0.005,
    ):
        self.embed_dim = embed_dim
        self.lr = learning_rate

        # Query compression: 1024 → 16
        self.W_query = np.random.randn(
            self.QUERY_CONTEXT_DIM, embed_dim
        ).astype(np.float32) * np.sqrt(2.0 / (embed_dim + self.QUERY_CONTEXT_DIM))

        # Reranking MLP: 20 → 32 → 1
        self.W1 = np.random.randn(
            self.HIDDEN_DIM, self.INPUT_DIM
        ).astype(np.float32) * np.sqrt(2.0 / (self.INPUT_DIM + self.HIDDEN_DIM))
        self.b1 = np.zeros(self.HIDDEN_DIM, dtype=np.float32)

        self.W2 = np.random.randn(
            1, self.HIDDEN_DIM
        ).astype(np.float32) * np.sqrt(2.0 / (self.HIDDEN_DIM + 1))
        self.b2 = np.zeros(1, dtype=np.float32)

        # Residual weight (how much to trust learned adjustment vs initial)
        self.residual_weight = 0.3  # Start with 30% learned, 70% initial

        # Training stats
        self.n_updates = 0
        self.cold_start_threshold = 100  # Need more data for reliable reranking

    def _compress_query(self, query_embedding: np.ndarray) -> np.ndarray:
        """Compress query embedding to context vector."""
        if len(query_embedding) > self.embed_dim:
            query_embedding = query_embedding[:self.embed_dim]
        elif len(query_embedding) < self.embed_dim:
            query_embedding = np.pad(query_embedding, (0, self.embed_dim - len(query_embedding)))

        context = np.tanh(self.W_query @ query_embedding)
        return context

    def rerank(
        self,
        scored_results: list,
        query_embedding: np.ndarray,
        component_names: tuple = ("semantic", "recency", "outcome", "importance"),
    ) -> list:
        """
        Re-rank scored results using learned model.

        Args:
            scored_results: List of ScoredResult with .score and .components
            query_embedding: Query vector (1024-dim)
            component_names: Order of component scores

        Returns:
            Re-ordered list with updated scores
        """
        if not scored_results:
            return scored_results

        # Cold start: don't modify results until we have enough training data
        if self.n_updates < self.cold_start_threshold:
            return scored_results

        # Compress query
        query_context = self._compress_query(query_embedding)

        # Score each result
        for result in scored_results:
            # Extract component scores in order
            components = np.array([
                result.components.get(name, 0.0)
                for name in component_names
            ], dtype=np.float32)

            # Concatenate [components, query_context]
            features = np.concatenate([components, query_context])

            # Forward pass
            hidden = np.tanh(self.W1 @ features + self.b1)
            adjustment = (self.W2 @ hidden + self.b2)[0]
            adjustment = np.tanh(adjustment) * 0.2  # Limit to [-0.2, 0.2]

            # Residual: blend learned adjustment with initial score
            initial_score = result.score
            result.score = (1 - self.residual_weight) * initial_score + self.residual_weight * (initial_score + adjustment)

            # Track for debugging
            result.components["rerank_adjustment"] = float(adjustment)

        # Re-sort by new scores
        scored_results.sort(key=lambda x: x.score, reverse=True)

        return scored_results

    def update(
        self,
        query_embedding: np.ndarray,
        component_scores_list: list[dict[str, float]],
        outcome_utilities: list[float],
        component_names: tuple = ("semantic", "recency", "outcome", "importance"),
    ) -> None:
        """
        Update reranker from retrieval outcomes.

        Args:
            query_embedding: Query used for retrieval
            component_scores_list: List of component dicts per retrieved item
            outcome_utilities: How useful each item was (0-1)
            component_names: Order of component scores
        """
        if not component_scores_list or not outcome_utilities:
            return

        query_context = self._compress_query(query_embedding)

        # Process each item
        for components_dict, utility in zip(component_scores_list, outcome_utilities):
            components = np.array([
                components_dict.get(name, 0.0)
                for name in component_names
            ], dtype=np.float32)

            features = np.concatenate([components, query_context])

            # Forward pass
            hidden = np.tanh(self.W1 @ features + self.b1)
            pred_adjustment = (self.W2 @ hidden + self.b2)[0]

            # Target: adjust score toward utility
            # If utility is high, we want positive adjustment
            # If utility is low, we want negative adjustment
            target_adjustment = (utility - 0.5) * 0.4  # Scale to [-0.2, 0.2]

            # Loss gradient (MSE)
            error = pred_adjustment - target_adjustment
            error = np.clip(error, -0.1, 0.1)  # Clip for stability

            # Backprop
            grad_W2 = error * hidden.reshape(1, -1)
            grad_b2 = np.array([error])

            tanh_grad = 1 - hidden ** 2
            grad_hidden = (self.W2.T @ np.array([error])).flatten() * tanh_grad
            grad_W1 = np.outer(grad_hidden, features)
            grad_b1 = grad_hidden

            # Also update query compression (small LR)
            grad_hidden[:self.QUERY_CONTEXT_DIM] if len(grad_hidden) >= self.QUERY_CONTEXT_DIM else grad_hidden
            # Skip query compression update for now (too slow)

            # Apply gradients
            self.W2 -= self.lr * grad_W2
            self.b2 -= self.lr * grad_b2
            self.W1 -= self.lr * grad_W1
            self.b1 -= self.lr * grad_b1

        self.n_updates += 1

        # Gradually increase residual weight as we learn
        if self.n_updates >= self.cold_start_threshold:
            # Slowly ramp up to 0.5 max
            self.residual_weight = min(0.5, 0.3 + 0.002 * (self.n_updates - self.cold_start_threshold))

    # MEMORY-HIGH-004 FIX: Add save/load methods for weight persistence
    def save_state(self) -> dict:
        """
        Save model weights and state to dictionary.

        Returns:
            Dictionary with all model parameters
        """
        return {
            "W_query": self.W_query.tolist(),
            "W1": self.W1.tolist(),
            "b1": self.b1.tolist(),
            "W2": self.W2.tolist(),
            "b2": self.b2.tolist(),
            "residual_weight": self.residual_weight,
            "n_updates": self.n_updates,
            "embed_dim": self.embed_dim,
            "lr": self.lr,
        }

    def load_state(self, state: dict) -> None:
        """
        Load model weights and state from dictionary.

        Args:
            state: Dictionary from save_state()
        """
        self.W_query = np.array(state["W_query"], dtype=np.float32)
        self.W1 = np.array(state["W1"], dtype=np.float32)
        self.b1 = np.array(state["b1"], dtype=np.float32)
        self.W2 = np.array(state["W2"], dtype=np.float32)
        self.b2 = np.array(state["b2"], dtype=np.float32)
        self.residual_weight = state.get("residual_weight", 0.3)
        self.n_updates = state.get("n_updates", 0)
