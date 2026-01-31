"""
Cross-Modal Binding for Episodic, Semantic, and Procedural Memories.

Biological Basis:
- Gamma synchrony binds distributed representations across brain regions
- Hippocampus coordinates retrieval across memory systems
- Phase-locking enables coherent multi-modal recall

Mathematical Formulation:

Cross-Modal Attention:
    Q = f_q(episodic)
    A_semantic = softmax(Q @ K_semantic.T / sqrt(d))
    A_procedural = softmax(Q @ K_procedural.T / sqrt(d))

Binding via Synchrony:
    Bound = gamma_sync * (A_semantic @ V_semantic + A_procedural @ V_procedural)
    gamma_sync = PLV(gamma_phase_episodic, gamma_phase_semantic)

PLV (Phase-Locking Value):
    PLV = |mean(exp(i * (phase1 - phase2)))|

References:
- Fell & Axmacher (2011): Role of phase synchronization in memory
- Fries (2005): Communication through coherence
- Varela et al. (2001): Brainweb: Phase synchronization and large-scale integration
"""

from __future__ import annotations

import logging
from dataclasses import dataclass

import numpy as np

logger = logging.getLogger(__name__)


@dataclass
class CrossModalBindingConfig:
    """Configuration for cross-modal binding.

    Attributes:
        embed_dim: Embedding dimension
        binding_dim: Binding space dimension
        binding_temperature: Softmax temperature for attention
        synchrony_threshold: Threshold for binding activation
        contrastive_temperature: Temperature for contrastive learning
        orthogonality_weight: Weight for orthogonality loss
        num_gamma_bins: Number of gamma phase bins
        plv_window_size: Window size for PLV computation
    """
    embed_dim: int = 1024
    binding_dim: int = 256
    binding_temperature: float = 0.5
    synchrony_threshold: float = 0.3
    contrastive_temperature: float = 0.07
    orthogonality_weight: float = 0.1
    num_gamma_bins: int = 8
    plv_window_size: int = 10


class ModalityProjector:
    """
    Projects embeddings from modality-specific to shared binding space.

    Each memory modality (episodic, semantic, procedural) has its own
    projector to a common binding space where cross-modal attention operates.
    """

    def __init__(
        self,
        input_dim: int,
        output_dim: int,
        name: str = "projector",
    ):
        """
        Initialize modality projector.

        Args:
            input_dim: Input embedding dimension
            output_dim: Output binding dimension
            name: Projector name for logging
        """
        self.input_dim = input_dim
        self.output_dim = output_dim
        self.name = name

        # Projection matrix
        self.W = np.random.randn(output_dim, input_dim).astype(np.float32)
        # Xavier initialization
        self.W *= np.sqrt(2.0 / (input_dim + output_dim))

        # Bias
        self.b = np.zeros(output_dim, dtype=np.float32)

        logger.debug(f"ModalityProjector '{name}': {input_dim} -> {output_dim}")

    def project(self, embedding: np.ndarray) -> np.ndarray:
        """
        Project embedding to binding space.

        Args:
            embedding: Input embedding [embed_dim] or [batch, embed_dim]

        Returns:
            Projected embedding [binding_dim] or [batch, binding_dim]
        """
        embedding = np.atleast_2d(embedding)
        projected = embedding @ self.W.T + self.b

        # L2 normalize
        norms = np.linalg.norm(projected, axis=-1, keepdims=True)
        projected = projected / (norms + 1e-8)

        if projected.shape[0] == 1:
            projected = projected.squeeze(0)

        return projected

    def update(
        self,
        embedding: np.ndarray,
        target: np.ndarray,
        lr: float = 0.01,
    ) -> float:
        """
        Update projection to better align embedding with target.

        Args:
            embedding: Input embedding
            target: Target in binding space
            lr: Learning rate

        Returns:
            Update magnitude
        """
        embedding = np.atleast_2d(embedding)
        target = np.atleast_2d(target)

        # Current projection
        projected = self.project(embedding)
        projected = np.atleast_2d(projected)

        # Error
        error = target - projected

        # Gradient update
        dW = (error.T @ embedding) / embedding.shape[0]
        db = error.mean(axis=0)

        self.W += lr * dW
        self.b += lr * db

        return float(np.linalg.norm(dW))


class GammaSynchronyDetector:
    """
    Detects gamma synchrony between brain regions.

    Uses Phase-Locking Value (PLV) to measure synchronization
    between gamma oscillations in different memory systems.

    High PLV indicates coherent binding.
    """

    def __init__(
        self,
        n_bins: int = 8,
        window_size: int = 10,
    ):
        """
        Initialize synchrony detector.

        Args:
            n_bins: Number of phase bins
            window_size: Window for PLV computation
        """
        self.n_bins = n_bins
        self.window_size = window_size

        # Phase history buffers
        self._phase_history_1: list = []
        self._phase_history_2: list = []

        logger.debug(f"GammaSynchronyDetector: bins={n_bins}, window={window_size}")

    def compute_synchrony(
        self,
        phases_1: np.ndarray,
        phases_2: np.ndarray,
    ) -> float:
        """
        Compute Phase-Locking Value between two phase sequences.

        PLV = |mean(exp(i * (phase1 - phase2)))|

        Args:
            phases_1: Phase sequence from region 1 [n_samples]
            phases_2: Phase sequence from region 2 [n_samples]

        Returns:
            PLV value in [0, 1]
        """
        if len(phases_1) != len(phases_2):
            min_len = min(len(phases_1), len(phases_2))
            phases_1 = phases_1[:min_len]
            phases_2 = phases_2[:min_len]

        # Phase difference
        phase_diff = phases_1 - phases_2

        # Complex exponential
        complex_phases = np.exp(1j * phase_diff)

        # PLV = |mean|
        plv = np.abs(np.mean(complex_phases))

        return float(plv)

    def update_phases(
        self,
        phase_1: float,
        phase_2: float,
    ) -> float:
        """
        Update phase history and compute running PLV.

        Args:
            phase_1: Current phase from region 1
            phase_2: Current phase from region 2

        Returns:
            Current PLV
        """
        self._phase_history_1.append(phase_1)
        self._phase_history_2.append(phase_2)

        # Maintain window size
        if len(self._phase_history_1) > self.window_size:
            self._phase_history_1.pop(0)
            self._phase_history_2.pop(0)

        # Compute PLV
        return self.compute_synchrony(
            np.array(self._phase_history_1),
            np.array(self._phase_history_2),
        )

    def is_synchronized(self, synchrony: float, threshold: float = 0.3) -> bool:
        """Check if synchrony exceeds threshold."""
        return synchrony > threshold

    def reset(self) -> None:
        """Reset phase history."""
        self._phase_history_1.clear()
        self._phase_history_2.clear()


class CrossModalBinding:
    """
    Cross-modal binding system for multi-memory integration.

    Binds representations from episodic, semantic, and procedural
    memory systems using gamma synchrony and attention.

    Key mechanisms:
    1. Project each modality to common binding space
    2. Compute cross-modal attention
    3. Gate by gamma synchrony
    4. Learn via contrastive objectives
    """

    def __init__(self, config: CrossModalBindingConfig | None = None):
        """
        Initialize cross-modal binding.

        Args:
            config: Binding configuration
        """
        self.config = config or CrossModalBindingConfig()

        # Projectors for each modality
        self.projectors = {
            "episodic": ModalityProjector(
                self.config.embed_dim,
                self.config.binding_dim,
                "episodic",
            ),
            "semantic": ModalityProjector(
                self.config.embed_dim,
                self.config.binding_dim,
                "semantic",
            ),
            "procedural": ModalityProjector(
                self.config.embed_dim,
                self.config.binding_dim,
                "procedural",
            ),
        }

        # Synchrony detectors
        self.synchrony_detectors = {
            ("episodic", "semantic"): GammaSynchronyDetector(
                self.config.num_gamma_bins,
                self.config.plv_window_size,
            ),
            ("episodic", "procedural"): GammaSynchronyDetector(
                self.config.num_gamma_bins,
                self.config.plv_window_size,
            ),
            ("semantic", "procedural"): GammaSynchronyDetector(
                self.config.num_gamma_bins,
                self.config.plv_window_size,
            ),
        }

        logger.info(
            f"CrossModalBinding initialized: embed_dim={self.config.embed_dim}, "
            f"binding_dim={self.config.binding_dim}"
        )

    def bind(
        self,
        episodic: np.ndarray | None = None,
        semantic: np.ndarray | None = None,
        procedural: np.ndarray | None = None,
        gamma_phases: dict[str, float] | None = None,
    ) -> dict:
        """
        Bind representations across modalities.

        Args:
            episodic: Episodic memory embeddings [n, embed_dim] or [embed_dim]
            semantic: Semantic memory embeddings [m, embed_dim] or [embed_dim]
            procedural: Procedural memory embeddings [p, embed_dim] or [embed_dim]
            gamma_phases: Gamma phases for each modality

        Returns:
            Dict with bound representations and synchrony values
        """
        result = {
            "bound_representations": {},
            "synchrony": {},
            "attention_weights": {},
        }

        # Project each modality to binding space
        projections = {}
        if episodic is not None:
            projections["episodic"] = self.projectors["episodic"].project(episodic)
        if semantic is not None:
            projections["semantic"] = self.projectors["semantic"].project(semantic)
        if procedural is not None:
            projections["procedural"] = self.projectors["procedural"].project(procedural)

        # Store projections
        result["projections"] = projections

        # Compute synchrony if phases provided
        if gamma_phases is not None:
            for (mod1, mod2), detector in self.synchrony_detectors.items():
                if mod1 in gamma_phases and mod2 in gamma_phases:
                    sync = detector.update_phases(
                        gamma_phases[mod1],
                        gamma_phases[mod2],
                    )
                    result["synchrony"][(mod1, mod2)] = sync

        # Cross-modal attention
        if len(projections) >= 2:
            modalities = list(projections.keys())

            for query_mod in modalities:
                query = np.atleast_2d(projections[query_mod])
                bound = np.zeros_like(query)

                for key_mod in modalities:
                    if key_mod != query_mod:
                        key = np.atleast_2d(projections[key_mod])

                        # Attention
                        attention = self._cross_attention(query, key)
                        result["attention_weights"][(query_mod, key_mod)] = attention

                        # Weighted sum
                        attended = attention @ key

                        # Gate by synchrony
                        sync_key = tuple(sorted([query_mod, key_mod]))
                        if sync_key in result["synchrony"]:
                            sync = result["synchrony"][sync_key]
                            if sync > self.config.synchrony_threshold:
                                bound = bound + sync * attended

                result["bound_representations"][query_mod] = bound.squeeze()

        return result

    def query_across_modalities(
        self,
        query: np.ndarray,
        query_mod: str,
        target_mod: str,
        candidates: np.ndarray,
    ) -> list[tuple[int, float]]:
        """
        Query one modality using representation from another.

        Args:
            query: Query embedding [embed_dim]
            query_mod: Query modality name
            target_mod: Target modality name
            candidates: Candidate embeddings [n, embed_dim]

        Returns:
            List of (index, score) tuples sorted by score
        """
        # Project query
        query_proj = self.projectors[query_mod].project(query)
        query_proj = np.atleast_2d(query_proj)

        # Project candidates
        cand_proj = self.projectors[target_mod].project(candidates)
        cand_proj = np.atleast_2d(cand_proj)

        # Compute attention scores
        scores = (query_proj @ cand_proj.T).squeeze()

        # Sort by score
        indices = np.argsort(scores)[::-1]
        results = [(int(idx), float(scores[idx])) for idx in indices]

        return results

    def learn_binding(
        self,
        anchor: np.ndarray,
        anchor_mod: str,
        positive: np.ndarray,
        pos_mod: str,
        negatives: list[np.ndarray],
        neg_mods: list[str] | None = None,
    ) -> float:
        """
        Learn binding via contrastive loss.

        Brings anchor and positive closer, pushes negatives away.

        Args:
            anchor: Anchor embedding
            anchor_mod: Anchor modality
            positive: Positive example (should bind with anchor)
            pos_mod: Positive modality
            negatives: Negative examples (should not bind)
            neg_mods: Negative modalities

        Returns:
            Contrastive loss
        """
        if neg_mods is None:
            neg_mods = [pos_mod] * len(negatives)

        # Project anchor
        anchor_proj = self.projectors[anchor_mod].project(anchor)

        # Project positive
        pos_proj = self.projectors[pos_mod].project(positive)

        # Project negatives
        neg_projs = [
            self.projectors[neg_mods[i]].project(neg)
            for i, neg in enumerate(negatives)
        ]

        # Compute similarities
        pos_sim = np.dot(anchor_proj, pos_proj)
        neg_sims = [np.dot(anchor_proj, neg_proj) for neg_proj in neg_projs]

        # InfoNCE loss
        temperature = self.config.contrastive_temperature
        numerator = np.exp(pos_sim / temperature)
        denominator = numerator + sum(np.exp(s / temperature) for s in neg_sims)
        loss = -np.log(numerator / (denominator + 1e-8))

        # Update projectors
        lr = 0.01

        # Gradient approximation: move anchor toward positive
        target = pos_proj + 0.1 * (pos_proj - anchor_proj)
        self.projectors[anchor_mod].update(anchor, target, lr)

        return float(loss)

    def _cross_attention(
        self,
        query: np.ndarray,
        key: np.ndarray,
    ) -> np.ndarray:
        """Compute cross-attention scores."""
        # Scaled dot product
        scale = np.sqrt(query.shape[-1])
        scores = (query @ key.T) / scale

        # Softmax
        scores = scores / self.config.binding_temperature
        exp_scores = np.exp(scores - np.max(scores, axis=-1, keepdims=True))
        attention = exp_scores / (np.sum(exp_scores, axis=-1, keepdims=True) + 1e-8)

        return attention

    def get_stats(self) -> dict:
        """Get binding statistics."""
        sync_values = {}
        for (mod1, mod2), detector in self.synchrony_detectors.items():
            if detector._phase_history_1:
                sync_values[(mod1, mod2)] = detector.compute_synchrony(
                    np.array(detector._phase_history_1),
                    np.array(detector._phase_history_2),
                )

        return {
            "binding_dim": self.config.binding_dim,
            "synchrony_threshold": self.config.synchrony_threshold,
            "current_synchrony": sync_values,
            "projector_norms": {
                name: float(np.linalg.norm(proj.W))
                for name, proj in self.projectors.items()
            },
        }


class TripartiteMemoryAttention:
    """
    Holistic attention across all three memory systems.

    Coordinates simultaneous retrieval from episodic, semantic,
    and procedural memories with cross-modal coherence checking.
    """

    def __init__(
        self,
        config: CrossModalBindingConfig | None = None,
    ):
        """
        Initialize tripartite attention.

        Args:
            config: Binding configuration
        """
        self.config = config or CrossModalBindingConfig()
        self.binding = CrossModalBinding(config)

        # Memory buffers (for demonstration)
        self._episodic_buffer: list[np.ndarray] = []
        self._semantic_buffer: list[np.ndarray] = []
        self._procedural_buffer: list[np.ndarray] = []

        logger.info("TripartiteMemoryAttention initialized")

    def add_memory(
        self,
        embedding: np.ndarray,
        modality: str,
    ) -> None:
        """Add memory to buffer."""
        if modality == "episodic":
            self._episodic_buffer.append(embedding)
        elif modality == "semantic":
            self._semantic_buffer.append(embedding)
        elif modality == "procedural":
            self._procedural_buffer.append(embedding)

    def holistic_recall(
        self,
        query: np.ndarray,
        query_modality: str = "episodic",
        top_k: int = 5,
    ) -> dict:
        """
        Perform holistic recall across all memory systems.

        Args:
            query: Query embedding
            query_modality: Which modality the query comes from
            top_k: Number of results per modality

        Returns:
            Dict with retrieval results and coherence
        """
        results = {
            "episodic": [],
            "semantic": [],
            "procedural": [],
        }

        # Query each modality
        for target_mod, buffer in [
            ("episodic", self._episodic_buffer),
            ("semantic", self._semantic_buffer),
            ("procedural", self._procedural_buffer),
        ]:
            if buffer and target_mod != query_modality:
                candidates = np.stack(buffer)
                retrieved = self.binding.query_across_modalities(
                    query, query_modality, target_mod, candidates
                )
                results[target_mod] = retrieved[:top_k]

        # Compute cross-modal coherence
        coherence = self._compute_retrieval_coherence(results, query_modality)
        results["coherence"] = coherence

        return results

    def compute_cross_modal_coherence(
        self,
        episodic: np.ndarray,
        semantic: np.ndarray,
        procedural: np.ndarray,
    ) -> float:
        """
        Compute coherence across three representations.

        High coherence indicates successful binding.

        Args:
            episodic: Episodic embedding
            semantic: Semantic embedding
            procedural: Procedural embedding

        Returns:
            Coherence score [0, 1]
        """
        # Project to binding space
        e_proj = self.binding.projectors["episodic"].project(episodic)
        s_proj = self.binding.projectors["semantic"].project(semantic)
        p_proj = self.binding.projectors["procedural"].project(procedural)

        # Pairwise similarities
        es_sim = np.dot(e_proj, s_proj)
        ep_sim = np.dot(e_proj, p_proj)
        sp_sim = np.dot(s_proj, p_proj)

        # Average similarity
        coherence = (es_sim + ep_sim + sp_sim) / 3.0

        # Normalize to [0, 1]
        coherence = (coherence + 1) / 2

        return float(coherence)

    def _compute_retrieval_coherence(
        self,
        results: dict,
        query_modality: str,
    ) -> float:
        """Compute coherence of retrieval results."""
        # Simplified coherence based on score distributions
        all_scores = []
        for mod, retrieved in results.items():
            if mod != "coherence" and retrieved:
                scores = [score for _, score in retrieved]
                all_scores.extend(scores)

        if not all_scores:
            return 0.0

        # High coherence = similar top scores across modalities
        return float(np.std(all_scores) + 0.1)


__all__ = [
    "CrossModalBindingConfig",
    "ModalityProjector",
    "GammaSynchronyDetector",
    "CrossModalBinding",
    "TripartiteMemoryAttention",
]
