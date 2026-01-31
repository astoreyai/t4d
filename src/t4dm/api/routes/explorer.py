"""
Phase 11: Memory Explorer Demo API.

Provides interactive visualization of hippocampal memory processes:
- Pattern separation (Dentate Gyrus)
- Pattern completion (CA3)
- Sequence binding (CA1)
- Memory consolidation (HPC → Cortex)

Biological basis:
- Jung & McNaughton (1993): DG sparsity 2-5%
- Rolls (2013): CA3 completion >90% from 30% cue
- Leutgeb (2007): Pattern separation gain >3x
"""

from __future__ import annotations

import logging
from datetime import datetime

import numpy as np
from fastapi import APIRouter
from pydantic import BaseModel, Field

logger = logging.getLogger(__name__)

explorer_router = APIRouter()


# -----------------------------------------------------------------------------
# Models
# -----------------------------------------------------------------------------

class MemoryExplorerState(BaseModel):
    """Current state of the memory explorer."""

    # Hippocampal stages
    dg_sparsity: float = Field(0.03, description="DG sparsity (2-5% active neurons)")
    ca3_attractor_state: str = Field("idle", description="CA3 state: encoding/retrieval/settling")
    ca1_binding_strength: float = Field(0.0, description="CA1 temporal binding strength")

    # Pattern separation metrics
    input_similarity: float = Field(0.0, description="Input pattern similarity")
    output_similarity: float = Field(0.0, description="Output pattern similarity")
    separation_gain: float = Field(1.0, description="Separation gain (target >3x)")

    # Memory flow
    working_memory_items: int = Field(0, description="Items in working memory")
    episodic_recent: int = Field(0, description="Recent episodic memories")
    semantic_links: int = Field(0, description="Active semantic links")

    # Embeddings for visualization
    embedding_2d: list[list[float]] = Field(default_factory=list, description="t-SNE/UMAP projection")


class EncodeDemoRequest(BaseModel):
    """Request for encoding demonstration."""
    content: str = Field(..., description="Content to encode")
    cognitive_state: str = Field("focus", description="Cognitive state: focus/explore/rest")


class EncodeDemoResponse(BaseModel):
    """Response from encoding demonstration."""
    success: bool
    memory_id: str
    dg_sparsity: float
    separation_gain: float
    ca3_activity: list[float]
    embedding_2d: list[float]


class RetrieveDemoRequest(BaseModel):
    """Request for retrieval demonstration."""
    cue: str = Field(..., description="Retrieval cue (partial pattern)")
    cue_completeness: float = Field(0.3, description="Fraction of cue present (0.1-1.0)")


class RetrieveDemoResponse(BaseModel):
    """Response from retrieval demonstration."""
    success: bool
    completion_rate: float
    retrieved_content: str | None
    ca3_settling_steps: int
    attractor_basin: str
    embedding_2d: list[float]


# -----------------------------------------------------------------------------
# State Management
# -----------------------------------------------------------------------------

class ExplorerStateManager:
    """Manages demo state for memory explorer."""

    def __init__(self):
        self._state = MemoryExplorerState()
        self._pattern_history: list[np.ndarray] = []
        self._memories: dict = {}

    def get_state(self) -> MemoryExplorerState:
        return self._state

    def simulate_encoding(
        self,
        content: str,
        cognitive_state: str = "focus"
    ) -> EncodeDemoResponse:
        """Simulate memory encoding with hippocampal dynamics."""
        # Generate random embedding for demo
        embedding = np.random.randn(128).astype(np.float32)

        # Simulate DG pattern separation
        if self._pattern_history:
            # Compute input similarity to last pattern
            last = self._pattern_history[-1]
            input_sim = float(np.dot(embedding, last) / (
                np.linalg.norm(embedding) * np.linalg.norm(last) + 1e-8
            ))
            input_sim = (input_sim + 1) / 2  # Normalize to [0, 1]
        else:
            input_sim = 0.5

        # DG creates sparse, orthogonalized output
        dg_sparsity = np.random.uniform(0.02, 0.05)
        sparse_output = embedding.copy()
        mask = np.random.rand(128) > (1 - dg_sparsity)
        sparse_output *= mask

        # Compute separation gain
        if self._pattern_history:
            output_sim = float(np.dot(sparse_output, self._pattern_history[-1]) / (
                np.linalg.norm(sparse_output) * np.linalg.norm(self._pattern_history[-1]) + 1e-8
            ))
            output_sim = (output_sim + 1) / 2
            separation_gain = (1 - output_sim) / max(1 - input_sim, 0.01)
        else:
            output_sim = 0.5
            separation_gain = 3.0

        # Store pattern
        self._pattern_history.append(sparse_output)
        if len(self._pattern_history) > 100:
            self._pattern_history = self._pattern_history[-100:]

        # Generate memory ID
        import uuid
        memory_id = str(uuid.uuid4())
        self._memories[memory_id] = {
            "content": content,
            "embedding": sparse_output,
            "timestamp": datetime.now(),
        }

        # Update state
        self._state.dg_sparsity = dg_sparsity
        self._state.input_similarity = input_sim
        self._state.output_similarity = output_sim
        self._state.separation_gain = separation_gain
        self._state.ca3_attractor_state = "encoding"
        self._state.episodic_recent = len(self._memories)

        # 2D projection for visualization
        embedding_2d = [float(sparse_output[0]), float(sparse_output[1])]

        return EncodeDemoResponse(
            success=True,
            memory_id=memory_id,
            dg_sparsity=dg_sparsity,
            separation_gain=separation_gain,
            ca3_activity=sparse_output[:10].tolist(),
            embedding_2d=embedding_2d,
        )

    def simulate_retrieval(
        self,
        cue: str,
        cue_completeness: float = 0.3
    ) -> RetrieveDemoResponse:
        """Simulate pattern completion retrieval."""
        if not self._memories:
            return RetrieveDemoResponse(
                success=False,
                completion_rate=0.0,
                retrieved_content=None,
                ca3_settling_steps=0,
                attractor_basin="empty",
                embedding_2d=[0.0, 0.0],
            )

        # Find best matching memory (simplified)
        best_id = None
        best_score = -1.0

        for mem_id, mem_data in self._memories.items():
            # Simple text matching for demo
            content = mem_data["content"].lower()
            cue_lower = cue.lower()
            score = sum(1 for c in cue_lower if c in content) / max(len(cue_lower), 1)

            if score > best_score:
                best_score = score
                best_id = mem_id

        if best_id is None or best_score < 0.1:
            return RetrieveDemoResponse(
                success=False,
                completion_rate=0.0,
                retrieved_content=None,
                ca3_settling_steps=0,
                attractor_basin="no_match",
                embedding_2d=[0.0, 0.0],
            )

        # Simulate CA3 settling
        settling_steps = int(5 + 10 * (1 - cue_completeness))

        # Completion rate (biological: >90% from 30% cue)
        completion_rate = 0.7 + 0.25 * cue_completeness + np.random.uniform(-0.05, 0.05)
        completion_rate = min(1.0, max(0.0, completion_rate))

        # Update state
        self._state.ca3_attractor_state = "retrieval"
        self._state.ca1_binding_strength = completion_rate

        mem_data = self._memories[best_id]
        embedding_2d = [float(mem_data["embedding"][0]), float(mem_data["embedding"][1])]

        return RetrieveDemoResponse(
            success=True,
            completion_rate=completion_rate,
            retrieved_content=mem_data["content"],
            ca3_settling_steps=settling_steps,
            attractor_basin=f"attractor_{best_id[:8]}",
            embedding_2d=embedding_2d,
        )


# Global state manager
_explorer_state = ExplorerStateManager()


# -----------------------------------------------------------------------------
# Endpoints
# -----------------------------------------------------------------------------

@explorer_router.get(
    "/state",
    response_model=MemoryExplorerState,
    summary="Get explorer state",
    description="Get current state of the memory explorer visualization",
)
async def get_explorer_state():
    """Get current memory explorer state."""
    return _explorer_state.get_state()


@explorer_router.post(
    "/demo/encode",
    response_model=EncodeDemoResponse,
    summary="Demo encoding",
    description="Demonstrate memory encoding with pattern separation visualization",
)
async def demo_encode(request: EncodeDemoRequest):
    """
    Demonstrate memory encoding.

    Visualizes:
    - Dentate Gyrus pattern separation (2-5% sparsity)
    - CA3 encoding state
    - Separation gain (target >3x orthogonalization)
    """
    return _explorer_state.simulate_encoding(
        content=request.content,
        cognitive_state=request.cognitive_state,
    )


@explorer_router.post(
    "/demo/retrieve",
    response_model=RetrieveDemoResponse,
    summary="Demo retrieval",
    description="Demonstrate pattern completion retrieval from partial cue",
)
async def demo_retrieve(request: RetrieveDemoRequest):
    """
    Demonstrate pattern completion retrieval.

    Visualizes:
    - CA3 attractor settling
    - Completion rate (>90% from 30% cue)
    - Attractor basin dynamics
    """
    return _explorer_state.simulate_retrieval(
        cue=request.cue,
        cue_completeness=request.cue_completeness,
    )


@explorer_router.post(
    "/reset",
    summary="Reset explorer",
    description="Reset explorer state for fresh demonstration",
)
async def reset_explorer():
    """Reset explorer to initial state."""
    global _explorer_state
    _explorer_state = ExplorerStateManager()
    return {"status": "reset", "message": "Explorer state reset"}


__all__ = ["explorer_router"]
