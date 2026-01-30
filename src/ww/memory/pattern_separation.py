"""
Pattern Separation for World Weaver.

Addresses Hinton critique: Similar inputs produce similar embeddings, causing
interference during retrieval. The dentate gyrus in the hippocampus performs
pattern separation - orthogonalizing similar patterns to make them more
distinct and reduce interference.

Biological Basis:
- DG receives input from entorhinal cortex with ~1M neurons
- DG has ~10M granule cells with very sparse activation (~0.5%)
- This expansion and sparsification creates orthogonal representations
- Similar inputs are mapped to dissimilar sparse codes

Implementation:
1. For new content, generate base embedding
2. Search for similar recent episodes (high similarity = interference risk)
3. If similar items found, apply orthogonalization:
   - Compute average direction of similar items
   - Project new embedding away from this direction
   - Normalize to maintain unit sphere constraint
4. The separation strength scales with similarity (more similar = more separation)

Modern Hopfield Networks (P3.2):
- Ramsauer et al. (2020) "Hopfield Networks is All You Need"
- Uses softmax with inverse temperature beta for exponential storage capacity
- Storage capacity: O(d^(n-1)) vs classical O(d) for n-body interactions
- Higher beta → sharper attention → better pattern retrieval
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum
from typing import Protocol

import numpy as np

logger = logging.getLogger(__name__)


# =============================================================================
# Modern Hopfield Network Functions (P3.2)
# =============================================================================

class HopfieldMode(Enum):
    """Hopfield network update mode."""
    CLASSICAL = "classical"      # Linear capacity, binary patterns
    MODERN = "modern"            # Exponential capacity, continuous patterns
    SPARSE = "sparse"            # Sparse attention variant


@dataclass
class HopfieldConfig:
    """Configuration for Modern Hopfield network."""
    beta: float = 1.0                 # Inverse temperature (higher = sharper)
    max_iterations: int = 10          # Maximum update iterations
    convergence_threshold: float = 0.001  # Convergence epsilon
    normalize_patterns: bool = True   # L2 normalize patterns
    mode: HopfieldMode = HopfieldMode.MODERN

    # Sparse mode parameters
    top_k: int | None = None          # If set, use top-k sparse attention


@dataclass
class HopfieldResult:
    """Result of Hopfield pattern completion."""
    completed_pattern: np.ndarray
    iterations: int
    converged: bool
    final_energy: float
    attention_entropy: float          # Lower = more confident retrieval


def modern_hopfield_update(
    query: np.ndarray,
    memories: np.ndarray,
    beta: float = 1.0,
    normalize: bool = True
) -> np.ndarray:
    """
    Modern Hopfield network update with exponential storage capacity.

    Implements the continuous Hopfield update from Ramsauer et al. (2020):
        new_query = softmax(beta * query @ memories.T) @ memories

    This achieves exponential storage capacity O(d^(n-1)) compared to
    classical Hopfield's O(d), where d is dimension and n is pattern count.

    Biological basis: Models CA3 recurrent dynamics with graded synaptic
    weights and soft winner-take-all competition.

    Args:
        query: Query pattern [dim] or [batch, dim]
        memories: Stored patterns [num_patterns, dim]
        beta: Inverse temperature (higher = sharper attention)
              - beta ≈ 1: Soft attention, pattern mixing
              - beta ≈ 10: Sharp attention, near-exact retrieval
              - beta → ∞: Hard attention, exact nearest neighbor
        normalize: Whether to L2 normalize output

    Returns:
        Updated pattern after one Hopfield step [dim] or [batch, dim]

    References:
        Ramsauer et al. (2020) "Hopfield Networks is All You Need"
        arXiv:2008.02217
    """
    query = np.asarray(query, dtype=np.float32)
    memories = np.asarray(memories, dtype=np.float32)

    was_1d = query.ndim == 1
    if was_1d:
        query = query[np.newaxis, :]

    # Compute similarities: [batch, num_patterns]
    similarities = query @ memories.T

    # Softmax with temperature: attention weights
    # Subtract max for numerical stability
    scaled_sim = beta * similarities
    scaled_sim = scaled_sim - scaled_sim.max(axis=1, keepdims=True)
    attention = np.exp(scaled_sim)
    attention = attention / attention.sum(axis=1, keepdims=True)

    # Weighted combination of memories
    result = attention @ memories  # [batch, dim]

    # Normalize if requested
    if normalize:
        norms = np.linalg.norm(result, axis=1, keepdims=True)
        norms = np.maximum(norms, 1e-8)  # Avoid division by zero
        result = result / norms

    if was_1d:
        result = result[0]

    return result


def sparse_hopfield_update(
    query: np.ndarray,
    memories: np.ndarray,
    beta: float = 1.0,
    top_k: int = 5,
    normalize: bool = True
) -> np.ndarray:
    """
    Sparse variant of Modern Hopfield using top-k attention.

    Only attends to top-k most similar patterns, providing:
    - Faster computation for large memory banks
    - Sparser, more interpretable attention
    - Better separation between patterns

    Args:
        query: Query pattern [dim] or [batch, dim]
        memories: Stored patterns [num_patterns, dim]
        beta: Inverse temperature
        top_k: Number of top patterns to attend to
        normalize: Whether to L2 normalize output

    Returns:
        Updated pattern after sparse Hopfield step
    """
    query = np.asarray(query, dtype=np.float32)
    memories = np.asarray(memories, dtype=np.float32)

    was_1d = query.ndim == 1
    if was_1d:
        query = query[np.newaxis, :]

    batch_size = query.shape[0]
    num_patterns = memories.shape[0]
    k = min(top_k, num_patterns)

    # Compute similarities
    similarities = query @ memories.T  # [batch, num_patterns]

    # Find top-k indices for each query
    top_indices = np.argpartition(similarities, -k, axis=1)[:, -k:]

    # Gather top-k similarities and apply softmax
    results = []
    for i in range(batch_size):
        idx = top_indices[i]
        top_sim = similarities[i, idx]
        top_mem = memories[idx]

        # Softmax over top-k
        scaled = beta * top_sim
        scaled = scaled - scaled.max()
        attn = np.exp(scaled)
        attn = attn / attn.sum()

        # Weighted combination
        result = attn @ top_mem
        results.append(result)

    result = np.array(results, dtype=np.float32)

    if normalize:
        norms = np.linalg.norm(result, axis=1, keepdims=True)
        norms = np.maximum(norms, 1e-8)
        result = result / norms

    if was_1d:
        result = result[0]

    return result


def hopfield_energy(
    pattern: np.ndarray,
    memories: np.ndarray,
    beta: float = 1.0
) -> float:
    """
    Compute Hopfield energy for a pattern.

    Lower energy indicates the pattern is closer to a stored attractor.
    The energy landscape has minima at stored patterns.

    E(x) = -log(sum_i exp(beta * x @ m_i)) / beta

    Args:
        pattern: Current pattern [dim]
        memories: Stored patterns [num_patterns, dim]
        beta: Inverse temperature

    Returns:
        Energy value (lower = more stable)
    """
    pattern = np.asarray(pattern, dtype=np.float32)
    memories = np.asarray(memories, dtype=np.float32)

    if pattern.ndim > 1:
        pattern = pattern.flatten()

    # Normalize pattern
    norm = np.linalg.norm(pattern)
    if norm > 0:
        pattern = pattern / norm

    similarities = pattern @ memories.T

    # LogSumExp for numerical stability
    max_sim = similarities.max()
    energy = -(np.log(np.sum(np.exp(beta * (similarities - max_sim)))) + beta * max_sim) / beta

    return float(energy)


def attention_entropy(
    query: np.ndarray,
    memories: np.ndarray,
    beta: float = 1.0
) -> float:
    """
    Compute entropy of attention distribution.

    Lower entropy indicates more confident/focused retrieval.
    High entropy means the query is ambiguous between patterns.

    Args:
        query: Query pattern [dim]
        memories: Stored patterns [num_patterns, dim]
        beta: Inverse temperature

    Returns:
        Entropy in nats (0 = perfectly focused, log(n) = uniform)
    """
    query = np.asarray(query, dtype=np.float32)
    memories = np.asarray(memories, dtype=np.float32)

    if query.ndim > 1:
        query = query.flatten()

    similarities = query @ memories.T

    # Softmax
    scaled = beta * similarities
    scaled = scaled - scaled.max()
    attention = np.exp(scaled)
    attention = attention / attention.sum()

    # Entropy: -sum(p * log(p))
    # Add epsilon to avoid log(0)
    entropy = -np.sum(attention * np.log(attention + 1e-10))

    return float(entropy)


@dataclass
class SeparationResult:
    """Result of pattern separation encoding."""

    original_embedding: np.ndarray
    separated_embedding: np.ndarray
    similar_count: int
    max_similarity: float
    separation_magnitude: float
    timestamp: datetime = field(default_factory=datetime.now)

    @property
    def was_separated(self) -> bool:
        """Whether separation was actually applied."""
        return self.separation_magnitude > 0


class EmbeddingProvider(Protocol):
    """Protocol for embedding providers."""

    async def embed_query(self, text: str) -> np.ndarray:
        """Generate embedding for text."""
        ...


class VectorStore(Protocol):
    """Protocol for vector stores."""

    async def search(
        self,
        collection: str,
        query_vector: np.ndarray,
        limit: int = 10,
        score_threshold: float | None = None
    ) -> list[dict]:
        """Search for similar vectors."""
        ...


class DentateGyrus:
    """
    Pattern separator inspired by hippocampal dentate gyrus.

    Orthogonalizes similar inputs to reduce interference during memory
    encoding. Uses adaptive separation strength based on the degree of
    similarity with existing memories.

    The separation works by:
    1. Finding similar recent memories
    2. Computing the "interference direction" (centroid of similar items)
    3. Projecting the new embedding orthogonal to this direction
    4. Mixing original and orthogonalized embeddings based on similarity

    Safeguards:
    - Maximum separation bounded to prevent over-distortion
    - Minimum similarity threshold prevents spurious separation
    - Normalized output maintains unit sphere constraint
    """

    def __init__(
        self,
        embedding_provider: EmbeddingProvider,
        vector_store: VectorStore,
        collection_name: str = "episodes",
        # MEMORY-HIGH-005 FIX: Lowered from 0.75 to 0.55 to reduce interference
        # 0.75 allowed too many similar patterns to coexist without separation
        # 0.55 triggers separation earlier while not over-separating distinct items
        similarity_threshold: float = 0.55,
        search_limit: int = 10,
        max_separation: float = 0.3,
        min_separation: float = 0.05,
        use_sparse_coding: bool = True,
        sparsity_ratio: float = 0.01
    ):
        """
        Initialize pattern separator.

        Args:
            embedding_provider: Provider for generating embeddings
            vector_store: Store for searching similar vectors
            collection_name: Collection to search for similar items
            similarity_threshold: Minimum similarity to trigger separation
                (MEMORY-HIGH-005: Lowered from 0.75 to 0.55 for better separation)
            search_limit: Max similar items to consider
            max_separation: Maximum separation magnitude
            min_separation: Minimum separation to apply
            use_sparse_coding: Apply sparse coding after separation
            sparsity_ratio: Target sparsity (fraction of non-zero).
                Biological DG has ~0.5% activation; 4% is a practical compromise
                that maintains pattern separation while preserving information.
        """
        self.embedding = embedding_provider
        self.vector_store = vector_store
        self.collection_name = collection_name
        self.similarity_threshold = similarity_threshold
        self.search_limit = search_limit
        self.max_separation = max_separation
        self.min_separation = min_separation
        self.use_sparse_coding = use_sparse_coding
        self.sparsity_ratio = sparsity_ratio

        # History for analysis
        self._separation_history: list[SeparationResult] = []

    async def encode(
        self,
        content: str,
        apply_separation: bool = True,
        ne_gain: float = 1.0,
    ) -> np.ndarray:
        """
        Generate pattern-separated embedding for content.

        P2.3 NE-Modulated Pattern Separation:
        Higher NE arousal (ne_gain > 1.0) increases separation strength,
        enhancing orthogonalization to reduce interference. This is
        biologically accurate: NE enhances DG granule cell activation.

        Reference: Aston-Jones & Cohen (2005) - Adaptive Gain Theory

        Args:
            content: Text content to encode
            apply_separation: Whether to apply pattern separation
            ne_gain: Norepinephrine arousal gain (1.0 = baseline).
                     Higher values increase separation strength.

        Returns:
            Pattern-separated embedding (normalized)
        """
        # Generate base embedding
        base_emb = await self.embedding.embed_query(content)
        base_emb = np.asarray(base_emb, dtype=np.float32)

        if not apply_separation:
            return base_emb

        # Search for similar items
        try:
            similar = await self.vector_store.search(
                collection=self.collection_name,
                vector=base_emb.tolist(),
                limit=self.search_limit,
                score_threshold=self.similarity_threshold
            )
        except Exception as e:
            logger.warning(f"Pattern separation search failed: {e}")
            return base_emb

        if not similar:
            # No similar items - no separation needed
            result = SeparationResult(
                original_embedding=base_emb.copy(),
                separated_embedding=base_emb,
                similar_count=0,
                max_similarity=0.0,
                separation_magnitude=0.0
            )
            self._separation_history.append(result)
            return base_emb

        # Apply separation with NE modulation (P2.3)
        separated_emb = self._orthogonalize(base_emb, similar, ne_gain=ne_gain)

        # Optional: Apply sparse coding with NE modulation (P2.3)
        # Higher NE → more extreme sparsity (sharper activation)
        if self.use_sparse_coding:
            separated_emb = self._sparsify(separated_emb, ne_gain=ne_gain)

        # Normalize
        norm = np.linalg.norm(separated_emb)
        if norm > 0:
            separated_emb = separated_emb / norm

        # Compute metrics
        max_sim = max(s.get("score", 0) for s in similar)
        sep_magnitude = float(np.linalg.norm(separated_emb - base_emb))

        result = SeparationResult(
            original_embedding=base_emb.copy(),
            separated_embedding=separated_emb,
            similar_count=len(similar),
            max_similarity=max_sim,
            separation_magnitude=sep_magnitude
        )
        self._separation_history.append(result)

        logger.debug(
            f"Pattern separation: {len(similar)} similar items, "
            f"max_sim={max_sim:.3f}, sep_mag={sep_magnitude:.4f}"
        )

        return separated_emb

    def _orthogonalize(
        self,
        target: np.ndarray,
        similar_items: list[dict],
        ne_gain: float = 1.0,
    ) -> np.ndarray:
        """
        Orthogonalize target embedding away from similar items.

        Uses adaptive Gram-Schmidt: removes component in direction of
        similar items' centroid, weighted by similarity.

        P2.3 NE Modulation (Aston-Jones & Cohen 2005):
        Higher NE arousal increases separation strength, reducing
        interference during high-novelty encoding.

        Args:
            target: Embedding to orthogonalize
            similar_items: Similar items with scores and vectors
            ne_gain: Norepinephrine gain (1.0 = baseline)

        Returns:
            Orthogonalized embedding
        """
        if not similar_items:
            return target

        # Extract vectors and similarities
        vectors = []
        similarities = []

        for item in similar_items:
            vec = item.get("vector")
            if vec is None:
                vec = item.get("embedding")
            if vec is not None:
                vectors.append(np.asarray(vec, dtype=np.float32))
                similarities.append(item.get("score", 0.8))

        if not vectors:
            return target

        vectors = np.array(vectors)
        similarities = np.array(similarities)

        # Compute similarity-weighted centroid (interference direction)
        weights = similarities / similarities.sum()
        centroid = np.average(vectors, axis=0, weights=weights)
        centroid_norm = np.linalg.norm(centroid)

        if centroid_norm < 1e-8:
            return target

        centroid = centroid / centroid_norm

        # Compute separation strength based on max similarity
        max_sim = similarities.max()

        # Separation increases with similarity (more similar = more separation)
        # Formula: sep_strength = min(max_sep, base + scale * (sim - threshold))
        base_sep = self.min_separation
        scale = (self.max_separation - self.min_separation) / (1.0 - self.similarity_threshold)
        sep_strength = base_sep + scale * (max_sim - self.similarity_threshold)
        sep_strength = np.clip(sep_strength, self.min_separation, self.max_separation)

        # P2.3: NE modulation of separation strength
        # Higher NE gain → stronger separation (reduces interference)
        # Biological basis: NE enhances DG granule cell activation
        ne_modulated_sep = sep_strength * ne_gain
        ne_modulated_sep = np.clip(ne_modulated_sep, self.min_separation, self.max_separation * 1.5)

        # Project target onto centroid
        projection = np.dot(target, centroid) * centroid

        # Remove scaled projection (partial Gram-Schmidt)
        # Uses NE-modulated separation strength (P2.3)
        orthogonalized = target - ne_modulated_sep * projection

        # Add random perturbation for additional separation
        # (inspired by DG's expansion recoding)
        # Also modulated by NE for consistent effect
        noise = np.random.randn(len(target)).astype(np.float32)
        noise = noise / np.linalg.norm(noise)
        orthogonalized = orthogonalized + 0.01 * ne_modulated_sep * noise

        return orthogonalized

    def _sparsify(
        self,
        embedding: np.ndarray,
        use_soft_threshold: bool = True,
        ne_gain: float = 1.0,
    ) -> np.ndarray:
        """
        Apply sparse coding to embedding.

        Uses soft shrinkage thresholding for biologically plausible sparsification.
        Soft thresholding preserves gradient information for near-threshold
        activations, unlike hard top-k which creates discontinuities.

        Formula: sparse = sign(x) * max(0, |x| - threshold)

        This is more neurally plausible as biological neurons have graded
        responses near threshold rather than binary on/off.

        P2.3 NE Modulation:
        Higher NE gain → more extreme sparsity (sharper activation pattern).
        This is biologically accurate: NE enhances signal-to-noise in DG.

        Args:
            embedding: Dense embedding
            use_soft_threshold: If True, use soft shrinkage. If False, use hard top-k.
            ne_gain: Norepinephrine gain (1.0 = baseline)

        Returns:
            Sparse embedding
        """
        # P2.3: NE modulates sparsity - higher NE → fewer but stronger activations
        # Inverse relationship: high NE reduces the number of active units
        ne_modulated_sparsity = self.sparsity_ratio / max(0.5, ne_gain)
        ne_modulated_sparsity = np.clip(ne_modulated_sparsity, 0.01, 0.2)  # Bounds
        k = max(1, int(len(embedding) * ne_modulated_sparsity))
        abs_vals = np.abs(embedding)

        # Find threshold at k-th largest absolute value
        threshold = np.partition(abs_vals, -k)[-k]

        if use_soft_threshold:
            # Soft shrinkage: sign(x) * max(0, |x| - threshold)
            # This preserves gradient flow for near-threshold values
            shrunk = abs_vals - threshold
            sparse = np.sign(embedding) * np.maximum(0, shrunk)
        else:
            # Hard thresholding: zero out below threshold
            sparse = np.where(abs_vals >= threshold, embedding, 0)

        return sparse

    def compute_separation(
        self,
        embedding_a: np.ndarray,
        embedding_b: np.ndarray
    ) -> float:
        """
        Compute separation between two embeddings.

        Returns cosine distance (1 - cosine_similarity).
        Higher values indicate more separation.

        Args:
            embedding_a: First embedding
            embedding_b: Second embedding

        Returns:
            Separation score [0, 2]
        """
        a = np.asarray(embedding_a)
        b = np.asarray(embedding_b)

        norm_a = np.linalg.norm(a)
        norm_b = np.linalg.norm(b)

        if norm_a < 1e-8 or norm_b < 1e-8:
            return 1.0

        cosine_sim = np.dot(a, b) / (norm_a * norm_b)
        return 1.0 - cosine_sim

    def get_separation_history(
        self,
        limit: int = 100,
        only_separated: bool = False
    ) -> list[SeparationResult]:
        """
        Get recent separation history.

        Args:
            limit: Maximum results to return
            only_separated: Only include results where separation was applied

        Returns:
            List of SeparationResult objects
        """
        history = self._separation_history

        if only_separated:
            history = [r for r in history if r.was_separated]

        return history[-limit:]

    def get_stats(self) -> dict:
        """
        Get pattern separation statistics.

        Returns:
            Dict with separation counts and averages
        """
        if not self._separation_history:
            return {
                "total_encodings": 0,
                "separations_applied": 0,
                "separation_rate": 0.0,
                "avg_separation_magnitude": 0.0,
                "avg_similar_count": 0.0,
                "avg_max_similarity": 0.0
            }

        separated = [r for r in self._separation_history if r.was_separated]

        return {
            "total_encodings": len(self._separation_history),
            "separations_applied": len(separated),
            "separation_rate": len(separated) / len(self._separation_history),
            "avg_separation_magnitude": float(np.mean([
                r.separation_magnitude for r in separated
            ])) if separated else 0.0,
            "avg_similar_count": float(np.mean([
                r.similar_count for r in self._separation_history
            ])),
            "avg_max_similarity": float(np.mean([
                r.max_similarity for r in separated
            ])) if separated else 0.0
        }

    def clear_history(self) -> None:
        """Clear separation history."""
        self._separation_history.clear()

    # ==================== P2.3: NE-Modulated Pattern Separation ====================

    @staticmethod
    def get_ne_modulated_separation(
        base_separation: float,
        ne_gain: float,
        min_sep: float = 0.05,
        max_sep: float = 0.45,
    ) -> float:
        """
        Compute NE-modulated separation strength.

        P2.3: Higher NE arousal increases separation strength,
        enhancing orthogonalization during high-novelty encoding.

        Reference: Aston-Jones & Cohen (2005) - Adaptive Gain Theory

        Args:
            base_separation: Base separation magnitude
            ne_gain: Norepinephrine gain (1.0 = baseline)
            min_sep: Minimum allowed separation
            max_sep: Maximum allowed separation

        Returns:
            NE-modulated separation strength
        """
        modulated = base_separation * ne_gain
        return float(np.clip(modulated, min_sep, max_sep))

    @staticmethod
    def get_ne_modulated_sparsity(
        base_sparsity: float,
        ne_gain: float,
        min_sparsity: float = 0.01,
        max_sparsity: float = 0.2,
    ) -> float:
        """
        Compute NE-modulated sparsity ratio.

        P2.3: Higher NE → sparser representations (fewer, stronger activations).
        Biological basis: NE enhances signal-to-noise in DG granule cells.

        Args:
            base_sparsity: Base sparsity ratio
            ne_gain: Norepinephrine gain (1.0 = baseline)
            min_sparsity: Minimum allowed sparsity
            max_sparsity: Maximum allowed sparsity

        Returns:
            NE-modulated sparsity ratio
        """
        # Inverse relationship: high NE reduces active units
        modulated = base_sparsity / max(0.5, ne_gain)
        return float(np.clip(modulated, min_sparsity, max_sparsity))


class PatternCompletion:
    """
    Pattern completion using Modern Hopfield dynamics (P3.2).

    Complementary to pattern separation: given a partial or noisy cue,
    reconstruct the full pattern through associative recall.

    This implements Modern Hopfield networks (Ramsauer et al. 2020) which
    provide exponential storage capacity compared to classical Hopfield.

    Key improvements over classical Hopfield:
    - Exponential capacity: O(d^(n-1)) vs O(d) patterns
    - Continuous patterns (not just binary)
    - Configurable sharpness via beta (inverse temperature)
    - Energy-based convergence guarantees

    Biological basis: Models CA3 recurrent dynamics with:
    - Graded synaptic weights (continuous valued)
    - Soft winner-take-all competition (softmax attention)
    - Iterative settling to attractor states
    """

    def __init__(
        self,
        embedding_dim: int = 1024,
        num_attractors: int = 100,
        convergence_threshold: float = 0.001,
        max_iterations: int = 10,
        beta: float = 8.0,
        mode: HopfieldMode = HopfieldMode.MODERN,
        sparse_top_k: int | None = None
    ):
        """
        Initialize Modern Hopfield pattern completion network.

        Args:
            embedding_dim: Dimension of embeddings
            num_attractors: Maximum attractors to maintain
            convergence_threshold: Threshold for stopping iteration
            max_iterations: Maximum completion iterations
            beta: Inverse temperature for attention sharpness
                  - beta ≈ 1: Soft attention, pattern mixing
                  - beta ≈ 8: Default, balanced retrieval
                  - beta ≈ 20+: Very sharp, near-exact retrieval
            mode: Hopfield network mode (CLASSICAL, MODERN, SPARSE)
            sparse_top_k: If mode=SPARSE, number of patterns to attend to
        """
        self.embedding_dim = embedding_dim
        self.num_attractors = num_attractors
        self.convergence_threshold = convergence_threshold
        self.max_iterations = max_iterations
        self.beta = beta
        self.mode = mode
        self.sparse_top_k = sparse_top_k or 5

        # Stored attractors (full patterns)
        self._attractors: list[np.ndarray] = []

        # Completion history for analysis
        self._completion_history: list[HopfieldResult] = []

    def add_attractor(self, pattern: np.ndarray) -> None:
        """
        Add a pattern as an attractor.

        Args:
            pattern: Full pattern to store
        """
        pattern = np.asarray(pattern, dtype=np.float32)
        norm = np.linalg.norm(pattern)
        if norm > 0:
            pattern = pattern / norm

        self._attractors.append(pattern)

        # Keep only most recent attractors
        if len(self._attractors) > self.num_attractors:
            self._attractors.pop(0)

    def complete(
        self,
        partial_pattern: np.ndarray,
        mask: np.ndarray | None = None,
        beta: float | None = None
    ) -> tuple[np.ndarray, int]:
        """
        Complete a partial pattern using Modern Hopfield dynamics.

        Args:
            partial_pattern: Noisy or partial input
            mask: Optional mask indicating known dimensions
            beta: Override default beta for this completion

        Returns:
            Tuple of (completed pattern, iterations used)
        """
        if not self._attractors:
            return partial_pattern.copy(), 0

        current = np.asarray(partial_pattern, dtype=np.float32)

        # Normalize input
        norm = np.linalg.norm(current)
        if norm > 0:
            current = current / norm

        attractors = np.array(self._attractors)
        effective_beta = beta if beta is not None else self.beta

        converged = False
        for iteration in range(self.max_iterations):
            # Modern Hopfield update
            if self.mode == HopfieldMode.SPARSE:
                next_pattern = sparse_hopfield_update(
                    current, attractors,
                    beta=effective_beta,
                    top_k=self.sparse_top_k,
                    normalize=True
                )
            else:
                next_pattern = modern_hopfield_update(
                    current, attractors,
                    beta=effective_beta,
                    normalize=True
                )

            # Apply mask if provided (keep known dimensions)
            if mask is not None:
                # Renormalize partial_pattern
                partial_norm = np.linalg.norm(partial_pattern)
                if partial_norm > 0:
                    partial_normed = partial_pattern / partial_norm
                else:
                    partial_normed = partial_pattern
                next_pattern = np.where(mask, partial_normed, next_pattern)
                # Renormalize after mask application
                next_norm = np.linalg.norm(next_pattern)
                if next_norm > 0:
                    next_pattern = next_pattern / next_norm

            # Check convergence
            delta = np.linalg.norm(next_pattern - current)
            current = next_pattern

            if delta < self.convergence_threshold:
                converged = True
                break

        # Compute metrics
        energy = hopfield_energy(current, attractors, effective_beta)
        entropy = attention_entropy(current, attractors, effective_beta)

        result = HopfieldResult(
            completed_pattern=current,
            iterations=iteration + 1,
            converged=converged,
            final_energy=energy,
            attention_entropy=entropy
        )
        self._completion_history.append(result)

        logger.debug(
            f"Hopfield completion: {iteration + 1} iters, "
            f"converged={converged}, energy={energy:.4f}, entropy={entropy:.4f}"
        )

        return current, iteration + 1

    def complete_with_details(
        self,
        partial_pattern: np.ndarray,
        mask: np.ndarray | None = None,
        beta: float | None = None
    ) -> HopfieldResult:
        """
        Complete pattern and return detailed result.

        Args:
            partial_pattern: Noisy or partial input
            mask: Optional mask indicating known dimensions
            beta: Override default beta

        Returns:
            HopfieldResult with completed pattern and metrics
        """
        self.complete(partial_pattern, mask, beta)
        return self._completion_history[-1]

    def find_nearest_attractor(
        self,
        pattern: np.ndarray
    ) -> tuple[np.ndarray | None, float]:
        """
        Find nearest stored attractor to pattern.

        Args:
            pattern: Input pattern

        Returns:
            Tuple of (nearest attractor, similarity)
        """
        if not self._attractors:
            return None, 0.0

        pattern = np.asarray(pattern, dtype=np.float32)
        norm = np.linalg.norm(pattern)
        if norm > 0:
            pattern = pattern / norm

        attractors = np.array(self._attractors)
        similarities = attractors @ pattern

        best_idx = np.argmax(similarities)
        return self._attractors[best_idx], float(similarities[best_idx])

    def get_attention_weights(
        self,
        query: np.ndarray,
        beta: float | None = None
    ) -> np.ndarray:
        """
        Get attention weights for a query over stored attractors.

        Useful for interpretability and debugging.

        Args:
            query: Query pattern
            beta: Override default beta

        Returns:
            Attention weights [num_attractors]
        """
        if not self._attractors:
            return np.array([])

        query = np.asarray(query, dtype=np.float32)
        norm = np.linalg.norm(query)
        if norm > 0:
            query = query / norm

        attractors = np.array(self._attractors)
        effective_beta = beta if beta is not None else self.beta

        similarities = query @ attractors.T
        scaled = effective_beta * similarities
        scaled = scaled - scaled.max()
        weights = np.exp(scaled)
        weights = weights / weights.sum()

        return weights

    def get_attractor_count(self) -> int:
        """Get number of stored attractors."""
        return len(self._attractors)

    def get_attractors(self) -> np.ndarray:
        """Get all stored attractors as array."""
        if not self._attractors:
            return np.array([])
        return np.array(self._attractors)

    def get_completion_history(self, limit: int = 100) -> list[HopfieldResult]:
        """Get recent completion history."""
        return self._completion_history[-limit:]

    def get_stats(self) -> dict:
        """Get completion statistics."""
        if not self._completion_history:
            return {
                "total_completions": 0,
                "avg_iterations": 0.0,
                "convergence_rate": 0.0,
                "avg_energy": 0.0,
                "avg_entropy": 0.0,
                "beta": self.beta,
                "mode": self.mode.value
            }

        history = self._completion_history
        converged = [r for r in history if r.converged]

        return {
            "total_completions": len(history),
            "avg_iterations": float(np.mean([r.iterations for r in history])),
            "convergence_rate": len(converged) / len(history),
            "avg_energy": float(np.mean([r.final_energy for r in history])),
            "avg_entropy": float(np.mean([r.attention_entropy for r in history])),
            "beta": self.beta,
            "mode": self.mode.value
        }

    def clear(self) -> None:
        """Clear all attractors and history."""
        self._attractors.clear()
        self._completion_history.clear()


# Factory functions

def create_dentate_gyrus(
    embedding_provider: EmbeddingProvider,
    vector_store: VectorStore,
    collection_name: str = "episodes",
    **kwargs
) -> DentateGyrus:
    """
    Create a DentateGyrus pattern separator.

    Args:
        embedding_provider: Provider for generating embeddings
        vector_store: Store for searching similar vectors
        collection_name: Collection to search
        **kwargs: Additional arguments passed to DentateGyrus

    Returns:
        Configured DentateGyrus instance
    """
    return DentateGyrus(
        embedding_provider=embedding_provider,
        vector_store=vector_store,
        collection_name=collection_name,
        **kwargs
    )


def create_pattern_completion(
    embedding_dim: int = 1024,
    num_attractors: int = 100,
    beta: float = 8.0,
    mode: str | HopfieldMode = "modern",
    **kwargs
) -> PatternCompletion:
    """
    Create a Modern Hopfield pattern completion network.

    Args:
        embedding_dim: Dimension of embeddings
        num_attractors: Maximum attractors to maintain
        beta: Inverse temperature (higher = sharper retrieval)
              Recommended values:
              - 1.0: Soft mixing of patterns
              - 8.0: Default, balanced retrieval
              - 20.0: Sharp, near-exact retrieval
        mode: Hopfield mode ("classical", "modern", "sparse")
        **kwargs: Additional arguments passed to PatternCompletion

    Returns:
        Configured PatternCompletion instance
    """
    if isinstance(mode, str):
        mode = HopfieldMode(mode)

    return PatternCompletion(
        embedding_dim=embedding_dim,
        num_attractors=num_attractors,
        beta=beta,
        mode=mode,
        **kwargs
    )


def benchmark_hopfield_capacity(
    dim: int = 128,
    num_patterns: int = 50,
    noise_level: float = 0.3,
    beta_values: list[float] | None = None,
    num_trials: int = 10
) -> dict:
    """
    Benchmark Hopfield network pattern completion capacity.

    Tests how well Modern Hopfield recovers stored patterns from
    noisy queries at different beta values.

    Args:
        dim: Embedding dimension
        num_patterns: Number of patterns to store
        noise_level: Noise magnitude (fraction of pattern norm)
        beta_values: Beta values to test (default: [1, 2, 4, 8, 16])
        num_trials: Trials per beta value

    Returns:
        Dict with benchmark results per beta value
    """
    if beta_values is None:
        beta_values = [1.0, 2.0, 4.0, 8.0, 16.0]

    # Generate random orthogonal-ish patterns
    patterns = np.random.randn(num_patterns, dim).astype(np.float32)
    patterns = patterns / np.linalg.norm(patterns, axis=1, keepdims=True)

    results = {}

    for beta in beta_values:
        network = PatternCompletion(
            embedding_dim=dim,
            num_attractors=num_patterns + 10,
            beta=beta
        )

        # Add patterns
        for p in patterns:
            network.add_attractor(p)

        correct = 0
        total_similarity = 0.0
        total_iterations = 0

        for _ in range(num_trials):
            # Pick random pattern
            idx = np.random.randint(num_patterns)
            target = patterns[idx]

            # Add noise
            noise = np.random.randn(dim).astype(np.float32)
            noise = noise / np.linalg.norm(noise)
            noisy = target + noise_level * noise
            noisy = noisy / np.linalg.norm(noisy)

            # Complete
            completed, iters = network.complete(noisy)
            total_iterations += iters

            # Check if closest to correct pattern
            similarities = patterns @ completed
            recovered_idx = np.argmax(similarities)
            if recovered_idx == idx:
                correct += 1
            total_similarity += float(similarities[idx])

        results[beta] = {
            "accuracy": correct / num_trials,
            "avg_similarity": total_similarity / num_trials,
            "avg_iterations": total_iterations / num_trials
        }

        network.clear()

    return results
