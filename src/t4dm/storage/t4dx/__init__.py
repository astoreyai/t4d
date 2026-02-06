"""T4DX embedded storage engine for T4DM."""

from t4dm.storage.t4dx.engine import T4DXEngine
from t4dm.storage.t4dx.graph_adapter import T4DXGraphStore
from t4dm.storage.t4dx.vector_adapter import T4DXVectorStore
from t4dm.storage.t4dx.uncertainty import (
    UncertaintyAwareItem,
    UncertaintyAwareSearch,
    UncertaintyConfig,
    UncertaintyEstimator,
)
from t4dm.storage.t4dx.markov_retrieval import (
    MarkovBlanketConfig,
    MarkovBlanketRetriever,
)
from t4dm.storage.t4dx.learned_edges import (
    EdgeImportanceTrainer,
    LearnedEdgeImportance,
    TraversalWithLearnedEdges,
)

__all__ = [
    "T4DXEngine",
    "T4DXGraphStore",
    "T4DXVectorStore",
    # W2-01: Uncertainty-aware storage (Friston)
    "UncertaintyAwareItem",
    "UncertaintyAwareSearch",
    "UncertaintyConfig",
    "UncertaintyEstimator",
    # W3-01: Markov Blanket Retrieval (Pearl/Friston)
    "MarkovBlanketConfig",
    "MarkovBlanketRetriever",
    # W3-02: Learned Edge Importance (Graves)
    "EdgeImportanceTrainer",
    "LearnedEdgeImportance",
    "TraversalWithLearnedEdges",
]
