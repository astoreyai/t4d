"""
World Weaver REST API Visualization Routes.

Endpoints for 3D memory graph visualization.
"""

import logging
from datetime import datetime
from enum import Enum
from uuid import UUID

import numpy as np
from fastapi import APIRouter, HTTPException, Query, status
from pydantic import BaseModel, Field

from t4dm.api.deps import AdminAuth, MemoryServices, parse_uuid

logger = logging.getLogger(__name__)

router = APIRouter()


# ============================================================================
# Response Models
# ============================================================================

class MemoryType(str, Enum):
    """Memory subsystem types."""
    EPISODIC = "episodic"
    SEMANTIC = "semantic"
    PROCEDURAL = "procedural"


class EdgeType(str, Enum):
    """Relationship types for edges."""
    CAUSED = "CAUSED"
    SIMILAR_TO = "SIMILAR_TO"
    PREREQUISITE = "PREREQUISITE"
    CONTRADICTS = "CONTRADICTS"
    REFERENCES = "REFERENCES"
    DERIVED_FROM = "DERIVED_FROM"


# ============================================================================
# Biological Mechanism Visualization Types
# ============================================================================

class FSRSState(BaseModel):
    """FSRS spaced repetition state for a memory."""
    memory_id: str
    stability: float  # Days until 90% retention
    difficulty: float  # 0-1 difficulty rating
    retrievability: float  # Current probability of recall
    last_review: float  # Unix timestamp
    next_review: float  # Predicted optimal review time
    review_count: int
    decay_curve: list[tuple[float, float]]  # [(days, retrievability), ...]


class HebbianWeight(BaseModel):
    """Hebbian synaptic weight between memories."""
    source_id: str
    target_id: str
    weight: float  # Current weight [0, 1]
    weight_history: list[tuple[float, float]]  # [(timestamp, weight), ...]
    co_activation_count: int
    last_potentiation: float | None = None  # LTP timestamp
    last_depression: float | None = None  # LTD timestamp
    eligibility_trace: float = 0.0  # Current trace value


class ActivationSpread(BaseModel):
    """ACT-R style activation spreading state."""
    source_id: str
    target_id: str
    base_level: float  # Base-level activation
    spreading_activation: float  # From connected nodes
    total_activation: float  # Combined
    decay_rate: float  # Per-second decay
    time_since_activation: float  # Seconds


class SleepPhaseViz(str, Enum):
    """Sleep consolidation phases."""
    NREM = "nrem"
    REM = "rem"
    PRUNE = "prune"
    COMPLETE = "complete"


class SleepConsolidationState(BaseModel):
    """Current sleep consolidation status."""
    is_active: bool
    current_phase: SleepPhaseViz | None = None
    phase_progress: float = 0.0  # 0-1 progress through phase
    replays_completed: int = 0
    abstractions_created: int = 0
    connections_pruned: int = 0
    replay_events: list[dict] = []  # Recent replay events
    last_cycle: float | None = None  # Timestamp


class WorkingMemoryState(BaseModel):
    """Working memory buffer visualization."""
    capacity: int
    current_size: int
    items: list[dict]  # Active WM items with priority
    attention_weights: list[float]
    decay_rate: float
    eviction_history: list[dict]  # Recent evictions
    is_full: bool
    attentional_blink_active: bool = False


class PatternSeparationMetrics(BaseModel):
    """Pattern separation (Dentate Gyrus) metrics."""
    input_similarity: float  # Similarity of input patterns
    output_similarity: float  # Similarity after separation
    separation_ratio: float  # input_sim / output_sim
    sparsity: float  # Proportion of active units
    orthogonalization_strength: float


class PatternCompletionMetrics(BaseModel):
    """Pattern completion (CA3) metrics."""
    input_completeness: float  # Fraction of pattern provided
    output_confidence: float  # Attractor network confidence
    convergence_iterations: int  # Steps to stable state
    best_match_id: str | None = None
    similarity_to_match: float = 0.0


class DopamineRPEMetrics(BaseModel):
    """Dopamine reward prediction error metrics."""
    total_signals: int  # Total RPE computations
    positive_surprises: int  # Better than expected (δ > 0)
    negative_surprises: int  # Worse than expected (δ < 0)
    avg_rpe: float  # Average prediction error
    avg_surprise: float  # Average |δ|
    memories_tracked: int  # Memories with value estimates


class ReconsolidationMetrics(BaseModel):
    """Memory reconsolidation metrics."""
    total_updates: int
    positive_updates: int  # Pulled toward query
    negative_updates: int  # Pushed away from query
    avg_update_magnitude: float
    memories_in_cooldown: int
    avg_learning_rate: float


class LearnedFusionMetrics(BaseModel):
    """Learned fusion training metrics."""
    enabled: bool
    train_steps: int
    avg_loss: float
    current_weights: dict | None = None  # Query-dependent weights


# ============================================================================
# Neuromodulator Orchestra Visualization Types
# ============================================================================

class NorepinephrineState(BaseModel):
    """Norepinephrine (locus coeruleus) arousal state."""
    current_gain: float = Field(ge=0.5, le=3.0, description="Arousal multiplier")
    novelty_score: float = Field(ge=0.0, le=1.0, description="Recent novelty detection")
    tonic_level: float = Field(ge=0.0, le=1.0, description="Baseline arousal")
    phasic_response: float = Field(description="Recent phasic burst")
    exploration_bonus: float = Field(description="Threshold reduction for search")
    history_length: int = Field(description="Tracked embeddings count")


class AcetylcholineState(BaseModel):
    """Acetylcholine encoding/retrieval mode state."""
    mode: str = Field(description="Current mode: encoding/balanced/retrieval")
    encoding_level: float = Field(ge=0.0, le=1.0)
    retrieval_level: float = Field(ge=0.0, le=1.0)
    attention_weights: dict[str, float] | None = None
    mode_switches: int = Field(description="Total mode transitions")
    time_in_current_mode: float = Field(description="Seconds in current mode")


class SerotoninState(BaseModel):
    """Serotonin long-term credit assignment state."""
    current_mood: float = Field(ge=0.0, le=1.0, description="Mood level")
    total_outcomes: int = Field(description="Total outcomes processed")
    positive_rate: float = Field(ge=0.0, le=1.0, description="Positive outcome rate")
    memories_with_traces: int = Field(description="Memories with eligibility traces")
    active_traces: int = Field(description="Total active traces")
    active_contexts: int = Field(description="Ongoing sessions/goals")


class InhibitionState(BaseModel):
    """GABA/Glutamate inhibitory network state."""
    recent_sparsity: float = Field(ge=0.0, le=1.0, description="Recent output sparsity")
    avg_sparsity: float = Field(ge=0.0, le=1.0, description="Average sparsity")
    inhibition_events: int = Field(description="Total inhibition applications")
    k_winners: int = Field(description="Current k-winners-take-all k")
    lateral_inhibition_strength: float = Field(ge=0.0, le=1.0)


class LearnedGateMetrics(BaseModel):
    """Learned memory gate (Bayesian logistic) metrics."""
    enabled: bool
    n_observations: int = Field(description="Total training samples")
    cold_start_progress: float = Field(ge=0.0, le=1.0, description="Progress through cold start")
    store_rate: float = Field(ge=0.0, le=1.0, description="Fraction stored")
    buffer_rate: float = Field(ge=0.0, le=1.0, description="Fraction buffered")
    skip_rate: float = Field(ge=0.0, le=1.0, description="Fraction skipped")
    avg_accuracy: float = Field(ge=0.0, le=1.0, description="Prediction accuracy")
    calibration_ece: float = Field(ge=0.0, description="Expected calibration error")


class NeuromodulatorOrchestraState(BaseModel):
    """Full neuromodulator orchestra state for visualization."""
    # Current combined state
    dopamine_rpe: float = Field(description="Recent reward prediction error")
    norepinephrine_gain: float = Field(description="Current arousal gain")
    acetylcholine_mode: str = Field(description="encoding/balanced/retrieval")
    serotonin_mood: float = Field(description="Current mood level")
    inhibition_sparsity: float = Field(description="Recent output sparsity")

    # Derived metrics
    effective_learning_rate: float = Field(description="Combined LR modifier")
    exploration_exploitation: float = Field(ge=-1.0, le=1.0, description="Balance (-1=exploit, +1=explore)")

    # Individual system states
    norepinephrine: NorepinephrineState | None = None
    acetylcholine: AcetylcholineState | None = None
    serotonin: SerotoninState | None = None
    inhibition: InhibitionState | None = None

    # Gate metrics
    learned_gate: LearnedGateMetrics | None = None

    # History
    state_count: int = Field(description="Total states recorded")
    timestamp: str | None = None


class BiologicalMechanismsResponse(BaseModel):
    """Combined biological mechanism states."""
    fsrs_states: list[FSRSState] = []
    hebbian_weights: list[HebbianWeight] = []
    activation_spreading: list[ActivationSpread] = []
    sleep_consolidation: SleepConsolidationState | None = None
    working_memory: WorkingMemoryState | None = None
    pattern_separation: PatternSeparationMetrics | None = None
    pattern_completion: PatternCompletionMetrics | None = None
    dopamine_rpe: DopamineRPEMetrics | None = None
    reconsolidation: ReconsolidationMetrics | None = None
    learned_fusion: LearnedFusionMetrics | None = None
    # New neuromodulator orchestra state
    neuromodulator_orchestra: NeuromodulatorOrchestraState | None = None


class Position3D(BaseModel):
    """3D position for layout."""
    x: float
    y: float
    z: float


class NodeMetadata(BaseModel):
    """Metadata for visualization node."""
    created_at: float  # Unix timestamp
    last_accessed: float
    access_count: int
    importance: float = Field(ge=0.0, le=1.0)
    tags: list[str] | None = None
    source: str | None = None


class MemoryNodeResponse(BaseModel):
    """Memory node for 3D visualization."""
    id: str
    type: MemoryType
    content: str
    metadata: NodeMetadata
    position: Position3D | None = None


class EdgeMetadata(BaseModel):
    """Metadata for edge."""
    created_at: float  # Unix timestamp
    last_activated: float | None = None


class MemoryEdgeResponse(BaseModel):
    """Memory edge for 3D visualization."""
    id: str
    source: str
    target: str
    type: EdgeType
    weight: float = Field(ge=0.0, le=1.0)
    metadata: EdgeMetadata | None = None


class GraphResponse(BaseModel):
    """Full graph response for visualization."""
    nodes: list[MemoryNodeResponse]
    edges: list[MemoryEdgeResponse]
    metrics: dict


class EmbeddingPoint(BaseModel):
    """Embedding with projected coordinates."""
    id: str
    type: MemoryType
    content: str
    embedding: list[float] | None = None  # Raw embedding
    position_2d: tuple[float, float] | None = None  # UMAP/t-SNE
    position_3d: tuple[float, float, float] | None = None


class EmbeddingsResponse(BaseModel):
    """Response with embeddings for projection."""
    points: list[EmbeddingPoint]
    embedding_dim: int
    projection_method: str | None = None


class TimelineEvent(BaseModel):
    """Event on the timeline."""
    id: str
    type: MemoryType
    timestamp: float
    content: str
    importance: float


class TimelineResponse(BaseModel):
    """Temporal data for timeline animation."""
    events: list[TimelineEvent]
    start_time: float
    end_time: float
    total_events: int


class ActivityMetrics(BaseModel):
    """Activity metrics for a memory."""
    id: str
    type: MemoryType
    activation: float  # Current activation level
    recency: float  # Time-based decay
    frequency: float  # Access frequency
    last_accessed: float


class ActivityResponse(BaseModel):
    """Recent activity metrics."""
    memories: list[ActivityMetrics]
    most_active: list[str]  # Top N active memory IDs
    recently_created: list[str]
    recently_accessed: list[str]


# ============================================================================
# Helper Functions
# ============================================================================

def _compute_activation(
    access_count: int,
    last_accessed: datetime,
    created_at: datetime,
    importance: float = 0.5,
    decay_rate: float = 0.1
) -> tuple[float, float, float]:
    """
    Compute activation metrics.

    Returns:
        (activation, recency, frequency)
    """
    now = datetime.now()

    # Recency: exponential decay from last access
    time_since_access = (now - last_accessed).total_seconds()
    recency = np.exp(-decay_rate * time_since_access / 86400)  # Decay per day

    # Frequency: normalized access count
    frequency = min(access_count / 100.0, 1.0)

    # Activation: weighted combination
    activation = 0.3 * recency + 0.3 * frequency + 0.4 * importance

    return float(activation), float(recency), float(frequency)


def _assign_3d_positions(
    nodes: list[dict],
    edges: list[dict],
    algorithm: str = "force-directed"
) -> dict[str, Position3D]:
    """
    Assign 3D positions to nodes.

    Simple layout algorithms - for production use d3-force-3d or similar.
    """
    positions = {}
    n = len(nodes)

    if n == 0:
        return positions

    if algorithm == "force-directed":
        # Simple force-directed starting positions
        for i, node in enumerate(nodes):
            angle = (i / n) * 2 * np.pi
            radius = 20 + np.random.uniform(-2, 2)
            z = np.random.uniform(-5, 5)
            positions[node["id"]] = Position3D(
                x=float(np.cos(angle) * radius),
                y=float(np.sin(angle) * radius),
                z=float(z)
            )

    elif algorithm == "hierarchical":
        # Layer by type
        type_layers = {"episodic": 0, "semantic": 1, "procedural": 2}
        type_counts = {"episodic": 0, "semantic": 0, "procedural": 0}

        for node in nodes:
            node_type = node["type"]
            layer = type_layers.get(node_type, 0)
            count = type_counts.get(node_type, 0)

            angle = (count * 0.5) % (2 * np.pi)
            radius = 10 + (count // 12) * 5

            positions[node["id"]] = Position3D(
                x=float(np.cos(angle) * radius),
                y=float(layer * 15 - 15),
                z=float(np.sin(angle) * radius)
            )

            type_counts[node_type] = count + 1

    elif algorithm == "circular":
        radius = max(10, n * 1.5)
        for i, node in enumerate(nodes):
            angle = (i / n) * 2 * np.pi
            positions[node["id"]] = Position3D(
                x=float(np.cos(angle) * radius),
                y=float(np.sin(angle) * radius),
                z=0.0
            )

    return positions


# ============================================================================
# Endpoints
# ============================================================================

@router.get("/graph", response_model=GraphResponse)
async def get_memory_graph(
    services: MemoryServices,
    layout: str = Query(default="force-directed", description="Layout algorithm"),
    limit: int = Query(default=500, ge=1, le=5000, description="Max nodes"),
    include_edges: bool = Query(default=True, description="Include edges"),
):
    """
    Get full memory graph with 3D positions.

    Returns all memory nodes (episodic, semantic, procedural) and their
    relationships for 3D visualization.
    """
    episodic = services["episodic"]
    semantic = services["semantic"]
    procedural = services["procedural"]

    try:
        nodes = []
        edges = []

        # Collect episodic memories
        episodes = await episodic.recent(limit=limit)
        for ep in episodes:
            # Extract tags from context safely
            tags = []
            if ep.context:
                ctx_dict = getattr(ep.context, "__dict__", {})
                for v in ctx_dict.values():
                    if isinstance(v, str) and v:
                        tags.append(v)

            nodes.append({
                "id": str(ep.id),
                "type": "episodic",
                "content": ep.content[:200],  # Truncate for viz
                "metadata": {
                    "created_at": ep.timestamp.timestamp(),
                    "last_accessed": ep.last_accessed.timestamp() if hasattr(ep, "last_accessed") else ep.timestamp.timestamp(),
                    "access_count": ep.access_count,
                    "importance": ep.emotional_valence,
                    "tags": tags,
                    "source": getattr(ep.context, "project", None) if ep.context else None,
                }
            })

        # Collect semantic entities
        try:
            entities = await semantic.list_entities(limit=limit)
            for entity in entities:
                entity_id = str(entity.get("id", entity.get("name", "")))
                nodes.append({
                    "id": entity_id,
                    "type": "semantic",
                    "content": entity.get("name", entity.get("content", ""))[:200],
                    "metadata": {
                        "created_at": entity.get("created_at", datetime.now()).timestamp() if isinstance(entity.get("created_at"), datetime) else entity.get("created_at", datetime.now().timestamp()),
                        "last_accessed": entity.get("last_accessed", datetime.now()).timestamp() if isinstance(entity.get("last_accessed"), datetime) else entity.get("last_accessed", datetime.now().timestamp()),
                        "access_count": entity.get("access_count", 0),
                        "importance": entity.get("importance", 0.5),
                        "tags": entity.get("tags", []),
                        "source": entity.get("source"),
                    }
                })
        except Exception as e:
            logger.warning(f"Could not fetch entities: {e}")

        # Collect procedural skills
        try:
            skills = await procedural.list_skills(limit=limit)
            for skill in skills:
                skill_id = str(skill.get("id", skill.get("name", "")))
                nodes.append({
                    "id": skill_id,
                    "type": "procedural",
                    "content": skill.get("name", skill.get("content", ""))[:200],
                    "metadata": {
                        "created_at": skill.get("created_at", datetime.now()).timestamp() if isinstance(skill.get("created_at"), datetime) else skill.get("created_at", datetime.now().timestamp()),
                        "last_accessed": skill.get("last_executed", datetime.now()).timestamp() if isinstance(skill.get("last_executed"), datetime) else skill.get("last_executed", datetime.now().timestamp()),
                        "access_count": skill.get("execution_count", 0),
                        "importance": skill.get("success_rate", 0.5),
                        "tags": skill.get("tags", []),
                        "source": skill.get("source"),
                    }
                })
        except Exception as e:
            logger.warning(f"Could not fetch skills: {e}")

        # Get edges from semantic graph if available
        if include_edges:
            try:
                relationships = await semantic.get_relationships(limit=limit * 2)
                for rel in relationships:
                    edge_type = rel.get("type", "REFERENCES").upper()
                    if edge_type not in [e.value for e in EdgeType]:
                        edge_type = "REFERENCES"

                    edges.append({
                        "id": f"{rel['source']}_{rel['target']}",
                        "source": str(rel["source"]),
                        "target": str(rel["target"]),
                        "type": edge_type,
                        "weight": rel.get("weight", 0.5),
                        "metadata": {
                            "created_at": rel.get("created_at", datetime.now()).timestamp() if isinstance(rel.get("created_at"), datetime) else rel.get("created_at", datetime.now().timestamp()),
                            "last_activated": rel.get("last_activated"),
                        }
                    })
            except Exception as e:
                logger.warning(f"Could not fetch relationships: {e}")

        # Compute 3D positions
        positions = _assign_3d_positions(nodes, edges, layout)

        # Build response
        node_responses = []
        for node in nodes:
            node_responses.append(MemoryNodeResponse(
                id=node["id"],
                type=MemoryType(node["type"]),
                content=node["content"],
                metadata=NodeMetadata(**node["metadata"]),
                position=positions.get(node["id"]),
            ))

        edge_responses = []
        for edge in edges:
            edge_responses.append(MemoryEdgeResponse(
                id=edge["id"],
                source=edge["source"],
                target=edge["target"],
                type=EdgeType(edge["type"]),
                weight=edge["weight"],
                metadata=EdgeMetadata(**edge["metadata"]) if edge.get("metadata") else None,
            ))

        # Compute metrics
        metrics = {
            "total_nodes": len(node_responses),
            "total_edges": len(edge_responses),
            "nodes_by_type": {
                "episodic": sum(1 for n in node_responses if n.type == MemoryType.EPISODIC),
                "semantic": sum(1 for n in node_responses if n.type == MemoryType.SEMANTIC),
                "procedural": sum(1 for n in node_responses if n.type == MemoryType.PROCEDURAL),
            },
            "average_importance": float(np.mean([n.metadata.importance for n in node_responses])) if node_responses else 0.0,
            "density": len(edge_responses) / (len(node_responses) * (len(node_responses) - 1)) if len(node_responses) > 1 else 0.0,
        }

        return GraphResponse(
            nodes=node_responses,
            edges=edge_responses,
            metrics=metrics,
        )

    except Exception as e:
        logger.error(f"Failed to get memory graph: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Failed to get memory graph: {e!s}",
        )


@router.get("/embeddings", response_model=EmbeddingsResponse)
async def get_embeddings(
    services: MemoryServices,
    limit: int = Query(default=200, ge=1, le=1000, description="Max embeddings"),
    include_raw: bool = Query(default=False, description="Include raw embeddings"),
    project_to: str = Query(default="3d", description="Projection: 2d, 3d, or none"),
):
    """
    Get memory embeddings for dimensionality reduction.

    Returns raw embeddings and optionally UMAP/t-SNE projections.
    """
    episodic = services["episodic"]

    try:
        points = []
        embeddings = []

        # Get episodes with embeddings
        episodes = await episodic.recent(limit=limit)

        for ep in episodes:
            emb = getattr(ep, "embedding", None)
            if emb is not None:
                embeddings.append(emb)
                points.append({
                    "id": str(ep.id),
                    "type": "episodic",
                    "content": ep.content[:100],
                    "embedding": emb.tolist() if include_raw and hasattr(emb, "tolist") else None,
                })

        # Project embeddings if requested
        if embeddings and project_to != "none":
            try:
                from sklearn.manifold import TSNE

                emb_array = np.array(embeddings)

                if project_to == "2d":
                    perplexity = min(30, len(embeddings) - 1)
                    if perplexity > 0:
                        tsne = TSNE(n_components=2, perplexity=perplexity, random_state=42)
                        projected = tsne.fit_transform(emb_array)

                        for i, point in enumerate(points):
                            point["position_2d"] = (float(projected[i, 0]), float(projected[i, 1]))

                elif project_to == "3d":
                    perplexity = min(30, len(embeddings) - 1)
                    if perplexity > 0:
                        tsne = TSNE(n_components=3, perplexity=perplexity, random_state=42)
                        projected = tsne.fit_transform(emb_array)

                        for i, point in enumerate(points):
                            point["position_3d"] = (
                                float(projected[i, 0]),
                                float(projected[i, 1]),
                                float(projected[i, 2]),
                            )

            except ImportError:
                logger.warning("sklearn not available for projection")
            except Exception as e:
                logger.warning(f"Projection failed: {e}")

        return EmbeddingsResponse(
            points=[EmbeddingPoint(**p) for p in points],
            embedding_dim=len(embeddings[0]) if embeddings else 0,
            projection_method="t-SNE" if project_to != "none" else None,
        )

    except Exception as e:
        logger.error(f"Failed to get embeddings: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Failed to get embeddings: {e!s}",
        )


@router.get("/timeline", response_model=TimelineResponse)
async def get_timeline(
    services: MemoryServices,
    days: int = Query(default=30, ge=1, le=365, description="Days of history"),
    limit: int = Query(default=500, ge=1, le=5000, description="Max events"),
):
    """
    Get temporal data for timeline animation.

    Returns events ordered by timestamp for timeline playback.
    """
    episodic = services["episodic"]

    try:
        now = datetime.now()
        start_time = (now - timedelta(days=days)).timestamp()
        end_time = now.timestamp()

        episodes = await episodic.recent(limit=limit)

        events = []
        for ep in episodes:
            timestamp = ep.timestamp.timestamp()
            if timestamp >= start_time:
                events.append(TimelineEvent(
                    id=str(ep.id),
                    type=MemoryType.EPISODIC,
                    timestamp=timestamp,
                    content=ep.content[:100],
                    importance=ep.emotional_valence,
                ))

        # Sort by timestamp
        events.sort(key=lambda e: e.timestamp)

        # Update start/end based on actual data
        if events:
            start_time = min(e.timestamp for e in events)
            end_time = max(e.timestamp for e in events)

        return TimelineResponse(
            events=events,
            start_time=start_time,
            end_time=end_time,
            total_events=len(events),
        )

    except Exception as e:
        logger.error(f"Failed to get timeline: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Failed to get timeline: {e!s}",
        )


@router.get("/activity", response_model=ActivityResponse)
async def get_activity(
    services: MemoryServices,
    limit: int = Query(default=100, ge=1, le=500, description="Max results"),
    top_n: int = Query(default=10, ge=1, le=50, description="Top N for each category"),
):
    """
    Get recent activity metrics.

    Returns activation levels, recency, and frequency for memories.
    """
    episodic = services["episodic"]

    try:
        episodes = await episodic.recent(limit=limit)

        memories = []
        for ep in episodes:
            activation, recency, frequency = _compute_activation(
                access_count=ep.access_count,
                last_accessed=getattr(ep, "last_accessed", ep.timestamp),
                created_at=ep.timestamp,
                importance=ep.emotional_valence,
            )

            memories.append(ActivityMetrics(
                id=str(ep.id),
                type=MemoryType.EPISODIC,
                activation=activation,
                recency=recency,
                frequency=frequency,
                last_accessed=getattr(ep, "last_accessed", ep.timestamp).timestamp() if isinstance(getattr(ep, "last_accessed", ep.timestamp), datetime) else ep.timestamp.timestamp(),
            ))

        # Sort for top N lists
        by_activation = sorted(memories, key=lambda m: m.activation, reverse=True)
        by_recency = sorted(memories, key=lambda m: m.last_accessed, reverse=True)
        by_created = sorted(memories, key=lambda m: m.last_accessed, reverse=True)  # Approximate

        return ActivityResponse(
            memories=memories,
            most_active=[m.id for m in by_activation[:top_n]],
            recently_created=[m.id for m in by_created[:top_n]],
            recently_accessed=[m.id for m in by_recency[:top_n]],
        )

    except Exception as e:
        logger.error(f"Failed to get activity: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Failed to get activity: {e!s}",
        )


class ExportRequest(BaseModel):
    """Request for graph export."""
    format: str = Field(default="json", description="Export format: json, gexf, graphml")
    include_embeddings: bool = Field(default=False)
    include_positions: bool = Field(default=True)


class ExportResponse(BaseModel):
    """Export response."""
    format: str
    content: str  # Base64 encoded for binary formats
    filename: str


@router.post("/export", response_model=ExportResponse)
async def export_graph(
    request: ExportRequest,
    services: MemoryServices,
):
    """
    Export graph to various formats.

    Supports JSON, GEXF (for Gephi), and GraphML.
    """
    try:
        # Get graph data
        graph_response = await get_memory_graph(
            services=services,
            layout="force-directed",
            limit=1000,
            include_edges=True,
        )

        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")

        if request.format == "json":
            import json
            content = json.dumps({
                "nodes": [n.model_dump() for n in graph_response.nodes],
                "edges": [e.model_dump() for e in graph_response.edges],
                "metrics": graph_response.metrics,
            }, indent=2, default=str)
            filename = f"memory_graph_{timestamp}.json"

        elif request.format == "gexf":
            # GEXF format for Gephi
            content = _generate_gexf(graph_response)
            filename = f"memory_graph_{timestamp}.gexf"

        elif request.format == "graphml":
            # GraphML format
            content = _generate_graphml(graph_response)
            filename = f"memory_graph_{timestamp}.graphml"

        else:
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail=f"Unsupported format: {request.format}",
            )

        return ExportResponse(
            format=request.format,
            content=content,
            filename=filename,
        )

    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Failed to export graph: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Failed to export graph: {e!s}",
        )


def _generate_gexf(graph: GraphResponse) -> str:
    """Generate GEXF format for Gephi."""
    lines = [
        '<?xml version="1.0" encoding="UTF-8"?>',
        '<gexf xmlns="http://www.gexf.net/1.2draft" version="1.2">',
        '  <graph mode="static" defaultedgetype="directed">',
        "    <nodes>",
    ]

    for node in graph.nodes:
        label = node.content[:50].replace('"', "'").replace("<", "&lt;").replace(">", "&gt;")
        lines.append(f'      <node id="{node.id}" label="{label}">')
        lines.append("        <attvalues>")
        lines.append(f'          <attvalue for="type" value="{node.type.value}"/>')
        lines.append(f'          <attvalue for="importance" value="{node.metadata.importance}"/>')
        lines.append("        </attvalues>")
        if node.position:
            lines.append(f'        <viz:position x="{node.position.x}" y="{node.position.y}" z="{node.position.z}"/>')
        lines.append("      </node>")

    lines.append("    </nodes>")
    lines.append("    <edges>")

    for i, edge in enumerate(graph.edges):
        lines.append(f'      <edge id="{i}" source="{edge.source}" target="{edge.target}" weight="{edge.weight}"/>')

    lines.append("    </edges>")
    lines.append("  </graph>")
    lines.append("</gexf>")

    return "\n".join(lines)


def _generate_graphml(graph: GraphResponse) -> str:
    """Generate GraphML format."""
    lines = [
        '<?xml version="1.0" encoding="UTF-8"?>',
        '<graphml xmlns="http://graphml.graphdrawing.org/xmlns">',
        '  <key id="type" for="node" attr.name="type" attr.type="string"/>',
        '  <key id="importance" for="node" attr.name="importance" attr.type="double"/>',
        '  <key id="weight" for="edge" attr.name="weight" attr.type="double"/>',
        '  <graph id="memory_graph" edgedefault="directed">',
    ]

    for node in graph.nodes:
        lines.append(f'    <node id="{node.id}">')
        lines.append(f'      <data key="type">{node.type.value}</data>')
        lines.append(f'      <data key="importance">{node.metadata.importance}</data>')
        lines.append("    </node>")

    for edge in graph.edges:
        lines.append(f'    <edge source="{edge.source}" target="{edge.target}">')
        lines.append(f'      <data key="weight">{edge.weight}</data>')
        lines.append("    </edge>")

    lines.append("  </graph>")
    lines.append("</graphml>")

    return "\n".join(lines)


# ============================================================================
# Biological Mechanism Visualization Endpoints
# ============================================================================

@router.get("/bio/fsrs", response_model=list[FSRSState])
async def get_fsrs_states(
    services: MemoryServices,
    limit: int = Query(default=50, ge=1, le=200, description="Max memories"),
    include_decay_curve: bool = Query(default=True, description="Include decay curves"),
    forecast_days: int = Query(default=30, ge=1, le=365, description="Days to forecast"),
):
    """
    Get FSRS spaced repetition states for memories.

    Returns stability, difficulty, retrievability and decay curves.
    """
    episodic = services["episodic"]

    try:
        episodes = await episodic.recent(limit=limit)
        states = []

        for ep in episodes:
            now = datetime.now()
            stability = getattr(ep, "stability", 1.0)
            difficulty = getattr(ep, "difficulty", 0.3)

            # Calculate retrievability using FSRS formula
            days_since = (now - ep.timestamp).days
            retrievability = np.exp(-days_since / max(stability, 0.1))

            # Generate decay curve
            decay_curve = []
            if include_decay_curve:
                for d in range(0, forecast_days + 1, max(1, forecast_days // 20)):
                    r = np.exp(-d / max(stability, 0.1))
                    decay_curve.append((float(d), float(r)))

            # Predict next optimal review (when R drops to 0.9)
            next_review_days = stability * np.log(1 / 0.9) if stability > 0 else 1
            next_review = (now + timedelta(days=next_review_days)).timestamp()

            states.append(FSRSState(
                memory_id=str(ep.id),
                stability=float(stability),
                difficulty=float(difficulty),
                retrievability=float(retrievability),
                last_review=ep.timestamp.timestamp(),
                next_review=next_review,
                review_count=ep.access_count,
                decay_curve=decay_curve,
            ))

        return states

    except Exception as e:
        logger.error(f"Failed to get FSRS states: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Failed to get FSRS states: {e!s}",
        )


@router.get("/bio/hebbian", response_model=list[HebbianWeight])
async def get_hebbian_weights(
    services: MemoryServices,
    limit: int = Query(default=100, ge=1, le=500, description="Max connections"),
    min_weight: float = Query(default=0.1, ge=0.0, le=1.0, description="Min weight threshold"),
):
    """
    Get Hebbian synaptic weights between memories.

    Returns current weights, history, and plasticity states.
    """
    semantic = services["semantic"]

    try:
        weights = []

        # Get relationships from semantic memory
        try:
            relationships = await semantic.get_relationships(limit=limit)

            for rel in relationships:
                weight = rel.get("weight", 0.5)
                if weight < min_weight:
                    continue

                # Build weight history if available
                history = rel.get("weight_history", [])
                if not history and rel.get("created_at"):
                    created = rel["created_at"]
                    if isinstance(created, datetime):
                        created = created.timestamp()
                    history = [(created, weight)]

                weights.append(HebbianWeight(
                    source_id=str(rel["source"]),
                    target_id=str(rel["target"]),
                    weight=float(weight),
                    weight_history=history,
                    co_activation_count=rel.get("co_activation_count", 0),
                    last_potentiation=rel.get("last_potentiation"),
                    last_depression=rel.get("last_depression"),
                    eligibility_trace=rel.get("eligibility_trace", 0.0),
                ))

        except Exception as e:
            logger.warning(f"Could not fetch relationships: {e}")

        return weights

    except Exception as e:
        logger.error(f"Failed to get Hebbian weights: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Failed to get Hebbian weights: {e!s}",
        )


@router.get("/bio/activation", response_model=list[ActivationSpread])
async def get_activation_spreading(
    services: MemoryServices,
    source_id: str | None = Query(default=None, description="Source node for spreading"),
    limit: int = Query(default=50, ge=1, le=200, description="Max activations"),
):
    """
    Get ACT-R style activation spreading from a source memory.

    Returns base-level and spreading activation for connected nodes.
    """
    semantic = services["semantic"]

    try:
        activations = []

        try:
            relationships = await semantic.get_relationships(limit=limit)

            for rel in relationships:
                if source_id and str(rel["source"]) != source_id:
                    continue

                # ACT-R base-level activation
                access_count = rel.get("access_count", 1)
                base_level = np.log(access_count + 1)

                # Spreading activation from weight
                weight = rel.get("weight", 0.5)
                spreading = weight * 0.5  # Simplified spreading

                # Time decay
                last_activated = rel.get("last_activated")
                if last_activated:
                    if isinstance(last_activated, datetime):
                        time_since = (datetime.now() - last_activated).total_seconds()
                    else:
                        time_since = datetime.now().timestamp() - last_activated
                else:
                    time_since = 0.0

                decay_rate = 0.5  # Per-second decay rate

                activations.append(ActivationSpread(
                    source_id=str(rel["source"]),
                    target_id=str(rel["target"]),
                    base_level=float(base_level),
                    spreading_activation=float(spreading),
                    total_activation=float(base_level + spreading),
                    decay_rate=decay_rate,
                    time_since_activation=float(time_since),
                ))

        except Exception as e:
            logger.warning(f"Could not compute activation: {e}")

        return activations

    except Exception as e:
        logger.error(f"Failed to get activation spreading: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Failed to get activation spreading: {e!s}",
        )


@router.get("/bio/sleep", response_model=SleepConsolidationState)
async def get_sleep_consolidation_state(
    services: MemoryServices,
):
    """
    Get current sleep consolidation status.

    Returns phase, progress, and recent consolidation events.
    """
    try:
        # Try to get sleep consolidation service from context
        # For now, return default state
        return SleepConsolidationState(
            is_active=False,
            current_phase=None,
            phase_progress=0.0,
            replays_completed=0,
            abstractions_created=0,
            connections_pruned=0,
            replay_events=[],
            last_cycle=None,
        )

    except Exception as e:
        logger.error(f"Failed to get sleep state: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Failed to get sleep state: {e!s}",
        )


@router.get("/bio/working-memory", response_model=WorkingMemoryState)
async def get_working_memory_state(
    services: MemoryServices,
):
    """
    Get current working memory buffer state.

    Returns capacity, items, attention weights, and eviction history.
    """
    try:
        # Try to get working memory from services if available
        # For now, return default state
        return WorkingMemoryState(
            capacity=4,
            current_size=0,
            items=[],
            attention_weights=[],
            decay_rate=0.1,
            eviction_history=[],
            is_full=False,
            attentional_blink_active=False,
        )

    except Exception as e:
        logger.error(f"Failed to get working memory state: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Failed to get working memory state: {e!s}",
        )


@router.get("/bio/pattern-separation", response_model=PatternSeparationMetrics)
async def get_pattern_separation_metrics(
    services: MemoryServices,
    input_a: str | None = Query(default=None, description="First pattern content"),
    input_b: str | None = Query(default=None, description="Second pattern content"),
):
    """
    Get pattern separation (Dentate Gyrus) metrics.

    Compares input similarity vs output similarity after orthogonalization.
    """
    try:
        # Default metrics when no inputs provided
        if not input_a or not input_b:
            return PatternSeparationMetrics(
                input_similarity=0.0,
                output_similarity=0.0,
                separation_ratio=1.0,
                sparsity=0.1,
                orthogonalization_strength=0.0,
            )

        # For actual computation, would need embedding provider
        # Returning mock metrics for now
        return PatternSeparationMetrics(
            input_similarity=0.85,
            output_similarity=0.35,
            separation_ratio=0.85 / 0.35,
            sparsity=0.1,
            orthogonalization_strength=0.5,
        )

    except Exception as e:
        logger.error(f"Failed to get pattern separation metrics: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Failed to get pattern separation metrics: {e!s}",
        )


@router.get("/bio/pattern-completion", response_model=PatternCompletionMetrics)
async def get_pattern_completion_metrics(
    services: MemoryServices,
    partial_input: str | None = Query(default=None, description="Partial pattern"),
):
    """
    Get pattern completion (CA3 attractor) metrics.

    Shows completion confidence and convergence for partial patterns.
    """
    try:
        if not partial_input:
            return PatternCompletionMetrics(
                input_completeness=0.0,
                output_confidence=0.0,
                convergence_iterations=0,
                best_match_id=None,
                similarity_to_match=0.0,
            )

        # Mock completion metrics
        return PatternCompletionMetrics(
            input_completeness=0.6,
            output_confidence=0.85,
            convergence_iterations=5,
            best_match_id=None,
            similarity_to_match=0.92,
        )

    except Exception as e:
        logger.error(f"Failed to get pattern completion metrics: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Failed to get pattern completion metrics: {e!s}",
        )


@router.get("/bio/dopamine", response_model=DopamineRPEMetrics)
async def get_dopamine_metrics(services: MemoryServices):
    """
    Get dopamine reward prediction error metrics.

    Shows learning statistics: surprise signals, prediction errors,
    and per-memory value estimates.
    """
    try:
        stats = services["episodic"].get_dopamine_stats()
        return DopamineRPEMetrics(
            total_signals=stats.get("total_signals", 0),
            positive_surprises=stats.get("positive_surprises", 0),
            negative_surprises=stats.get("negative_surprises", 0),
            avg_rpe=stats.get("avg_rpe", 0.0),
            avg_surprise=stats.get("avg_surprise", 0.0),
            memories_tracked=stats.get("memories_tracked", 0),
        )
    except Exception as e:
        logger.error(f"Failed to get dopamine metrics: {e}")
        return DopamineRPEMetrics(
            total_signals=0,
            positive_surprises=0,
            negative_surprises=0,
            avg_rpe=0.0,
            avg_surprise=0.0,
            memories_tracked=0,
        )


@router.get("/bio/reconsolidation", response_model=ReconsolidationMetrics)
async def get_reconsolidation_metrics(services: MemoryServices):
    """
    Get memory reconsolidation metrics.

    Shows embedding update statistics from retrieval-based learning.
    """
    try:
        stats = services["episodic"].get_reconsolidation_stats()
        return ReconsolidationMetrics(
            total_updates=stats.get("total_updates", 0),
            positive_updates=stats.get("positive_updates", 0),
            negative_updates=stats.get("negative_updates", 0),
            avg_update_magnitude=stats.get("avg_update_magnitude", 0.0),
            memories_in_cooldown=stats.get("memories_in_cooldown", 0),
            avg_learning_rate=stats.get("avg_learning_rate", 0.0),
        )
    except Exception as e:
        logger.error(f"Failed to get reconsolidation metrics: {e}")
        return ReconsolidationMetrics(
            total_updates=0,
            positive_updates=0,
            negative_updates=0,
            avg_update_magnitude=0.0,
            memories_in_cooldown=0,
            avg_learning_rate=0.0,
        )


@router.get("/bio/learned-fusion", response_model=LearnedFusionMetrics)
async def get_learned_fusion_metrics(services: MemoryServices):
    """
    Get learned fusion training metrics.

    Shows neural network training statistics for query-dependent
    fusion weight learning.
    """
    try:
        stats = services["episodic"].get_fusion_training_stats()
        return LearnedFusionMetrics(
            enabled=stats.get("enabled", False),
            train_steps=stats.get("train_steps", 0),
            avg_loss=stats.get("avg_loss", 0.0),
            current_weights=stats.get("current_weights", None),
        )
    except Exception as e:
        logger.error(f"Failed to get learned fusion metrics: {e}")
        return LearnedFusionMetrics(
            enabled=False,
            train_steps=0,
            avg_loss=0.0,
            current_weights=None,
        )


# ============================================================================
# Neuromodulator Orchestra Endpoints
# ============================================================================

@router.get("/bio/neuromodulators", response_model=NeuromodulatorOrchestraState)
async def get_neuromodulator_state(services: MemoryServices):
    """
    Get full neuromodulator orchestra state.

    Returns the coordinated state of all five neuromodulatory systems:
    - Dopamine (DA): Reward prediction error, surprise-driven learning
    - Norepinephrine (NE): Arousal, attention, novelty detection
    - Acetylcholine (ACh): Encoding/retrieval mode switching
    - Serotonin (5-HT): Long-term credit assignment, patience
    - GABA/Glutamate: Competitive inhibition, sparse representations

    Plus the learned memory gate metrics.
    """
    try:
        # Get orchestra stats
        orchestra_stats = services["episodic"].get_orchestra_stats()
        current_state = services["episodic"].get_current_neuromodulator_state()
        gate_stats = services["episodic"].get_learned_gate_stats()

        # Build individual system states
        ne_stats = orchestra_stats.get("norepinephrine", {})
        norepinephrine = NorepinephrineState(
            current_gain=ne_stats.get("current_gain", 1.0),
            novelty_score=ne_stats.get("novelty_score", 0.0),
            tonic_level=ne_stats.get("tonic_level", 0.5),
            phasic_response=ne_stats.get("phasic_response", 0.0),
            exploration_bonus=ne_stats.get("exploration_bonus", 0.0),
            history_length=ne_stats.get("history_length", 0),
        )

        ach_stats = orchestra_stats.get("acetylcholine", {})
        acetylcholine = AcetylcholineState(
            mode=ach_stats.get("current_mode", "balanced"),
            encoding_level=ach_stats.get("encoding_level", 0.5),
            retrieval_level=ach_stats.get("retrieval_level", 0.5),
            attention_weights=ach_stats.get("attention_weights"),
            mode_switches=ach_stats.get("mode_switches", 0),
            time_in_current_mode=ach_stats.get("time_in_mode", 0.0),
        )

        serotonin_stats = orchestra_stats.get("serotonin", {})
        serotonin = SerotoninState(
            current_mood=serotonin_stats.get("current_mood", 0.5),
            total_outcomes=serotonin_stats.get("total_outcomes", 0),
            positive_rate=serotonin_stats.get("positive_outcome_rate", 0.5),
            memories_with_traces=serotonin_stats.get("memories_with_traces", 0),
            active_traces=serotonin_stats.get("active_traces", 0),
            active_contexts=serotonin_stats.get("active_contexts", 0),
        )

        inhib_stats = orchestra_stats.get("inhibitory", {})
        inhibition = InhibitionState(
            recent_sparsity=inhib_stats.get("recent_sparsity", 0.0),
            avg_sparsity=inhib_stats.get("avg_sparsity", 0.0),
            inhibition_events=inhib_stats.get("inhibition_events", 0),
            k_winners=inhib_stats.get("k", 10),
            lateral_inhibition_strength=inhib_stats.get("lateral_strength", 0.5),
        )

        # Build gate metrics
        learned_gate = None
        if gate_stats.get("enabled", False):
            learned_gate = LearnedGateMetrics(
                enabled=True,
                n_observations=gate_stats.get("n_observations", 0),
                cold_start_progress=gate_stats.get("cold_start_progress", 0.0),
                store_rate=gate_stats.get("store_rate", 0.0),
                buffer_rate=gate_stats.get("buffer_rate", 0.0),
                skip_rate=gate_stats.get("skip_rate", 0.0),
                avg_accuracy=gate_stats.get("avg_accuracy", 0.0),
                calibration_ece=gate_stats.get("calibration_ece", 0.0),
            )

        # Get current state values
        if current_state:
            return NeuromodulatorOrchestraState(
                dopamine_rpe=current_state.get("dopamine_rpe", 0.0),
                norepinephrine_gain=current_state.get("norepinephrine_gain", 1.0),
                acetylcholine_mode=current_state.get("acetylcholine_mode", "balanced"),
                serotonin_mood=current_state.get("serotonin_mood", 0.5),
                inhibition_sparsity=current_state.get("inhibition_sparsity", 0.0),
                effective_learning_rate=current_state.get("effective_learning_rate", 1.0),
                exploration_exploitation=current_state.get("exploration_balance", 0.0),
                norepinephrine=norepinephrine,
                acetylcholine=acetylcholine,
                serotonin=serotonin,
                inhibition=inhibition,
                learned_gate=learned_gate,
                state_count=orchestra_stats.get("total_states", 0),
                timestamp=current_state.get("timestamp"),
            )
        # No state yet - return defaults
        return NeuromodulatorOrchestraState(
            dopamine_rpe=0.0,
            norepinephrine_gain=1.0,
            acetylcholine_mode="balanced",
            serotonin_mood=0.5,
            inhibition_sparsity=0.0,
            effective_learning_rate=1.0,
            exploration_exploitation=0.0,
            norepinephrine=norepinephrine,
            acetylcholine=acetylcholine,
            serotonin=serotonin,
            inhibition=inhibition,
            learned_gate=learned_gate,
            state_count=0,
            timestamp=None,
        )
    except Exception as e:
        logger.error(f"Failed to get neuromodulator state: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Failed to get neuromodulator state: {e}"
        )


@router.get("/bio/learned-gate", response_model=LearnedGateMetrics)
async def get_learned_gate_metrics(services: MemoryServices):
    """
    Get learned memory gate metrics.

    Shows Bayesian logistic regression + Thompson sampling statistics
    for the "learning what to remember" system.
    """
    try:
        stats = services["episodic"].get_learned_gate_stats()
        return LearnedGateMetrics(
            enabled=stats.get("enabled", False),
            n_observations=stats.get("n_observations", 0),
            cold_start_progress=stats.get("cold_start_progress", 0.0),
            store_rate=stats.get("store_rate", 0.0),
            buffer_rate=stats.get("buffer_rate", 0.0),
            skip_rate=stats.get("skip_rate", 0.0),
            avg_accuracy=stats.get("avg_accuracy", 0.0),
            calibration_ece=stats.get("calibration_ece", 0.0),
        )
    except Exception as e:
        logger.error(f"Failed to get learned gate metrics: {e}")
        return LearnedGateMetrics(
            enabled=False,
            n_observations=0,
            cold_start_progress=0.0,
            store_rate=0.0,
            buffer_rate=0.0,
            skip_rate=0.0,
            avg_accuracy=0.0,
            calibration_ece=0.0,
        )


# =============================================================================
# Neuromodulator Tuning Endpoints (for live parameter adjustment)
# =============================================================================

class NeuromodulatorTuningRequest(BaseModel):
    """Request to tune neuromodulator parameters."""
    # Norepinephrine (LC-NE system)
    ne_baseline_arousal: float | None = Field(default=None, ge=0.0, le=1.0)
    ne_min_gain: float | None = Field(default=None, ge=0.1, le=1.0)
    ne_max_gain: float | None = Field(default=None, ge=1.0, le=5.0)
    ne_novelty_decay: float | None = Field(default=None, ge=0.8, le=0.99)
    ne_phasic_decay: float | None = Field(default=None, ge=0.5, le=0.95)

    # Acetylcholine (cholinergic system)
    ach_baseline: float | None = Field(default=None, ge=0.1, le=0.9)
    ach_adaptation_rate: float | None = Field(default=None, ge=0.01, le=1.0)
    ach_encoding_threshold: float | None = Field(default=None, ge=0.5, le=0.95)
    ach_retrieval_threshold: float | None = Field(default=None, ge=0.1, le=0.5)

    # Dopamine (reward prediction error)
    da_value_learning_rate: float | None = Field(default=None, ge=0.01, le=0.5)
    da_default_expected: float | None = Field(default=None, ge=0.0, le=1.0)
    da_surprise_threshold: float | None = Field(default=None, ge=0.01, le=0.2)

    # Serotonin (temporal discounting)
    serotonin_baseline_mood: float | None = Field(default=None, ge=0.0, le=1.0)
    serotonin_mood_adaptation_rate: float | None = Field(default=None, ge=0.01, le=0.5)
    serotonin_discount_rate: float | None = Field(default=None, ge=0.9, le=1.0)
    serotonin_eligibility_decay: float | None = Field(default=None, ge=0.8, le=0.99)

    # Inhibition (GABA-ergic sparse coding)
    inhibition_strength: float | None = Field(default=None, ge=0.0, le=1.0)
    sparsity_target: float | None = Field(default=None, ge=0.05, le=0.5)  # DG-like 5-50%
    inhibition_temperature: float | None = Field(default=None, ge=0.1, le=5.0)


# Backward compatibility alias
NeuromodulatorTuning = NeuromodulatorTuningRequest


class NeuromodulatorTuningResponse(BaseModel):
    """Response after tuning neuromodulators."""
    success: bool = True
    updated_systems: list[str] = []
    applied: dict[str, float] = {}
    current_state: NeuromodulatorOrchestraState | None = None


@router.put("/bio/neuromodulators", response_model=NeuromodulatorTuningResponse)
async def tune_neuromodulators(
    request: NeuromodulatorTuningRequest,
    services: MemoryServices,
    _: AdminAuth,
):
    """
    Tune neuromodulator parameters at runtime.

    Requires admin authentication. Changes take effect immediately
    but do not persist across restarts.
    """
    try:
        episodic = services.get("episodic")
        applied = {}
        updated_systems = set()

        if hasattr(episodic, "orchestra") and episodic.orchestra:
            orchestra = episodic.orchestra

            # Apply NE tuning (using setter methods if available, fallback to direct)
            if hasattr(orchestra, "norepinephrine"):
                ne = orchestra.norepinephrine
                ne_updated = False
                if request.ne_baseline_arousal is not None:
                    if hasattr(ne, "set_baseline_arousal"):
                        ne.set_baseline_arousal(request.ne_baseline_arousal)
                    elif hasattr(ne, "baseline_arousal"):
                        ne.baseline_arousal = request.ne_baseline_arousal
                    applied["ne_baseline_arousal"] = request.ne_baseline_arousal
                    ne_updated = True
                if request.ne_min_gain is not None or request.ne_max_gain is not None:
                    min_gain = request.ne_min_gain if request.ne_min_gain is not None else getattr(ne, "min_gain", 0.5)
                    max_gain = request.ne_max_gain if request.ne_max_gain is not None else getattr(ne, "max_gain", 2.0)
                    if hasattr(ne, "set_arousal_bounds"):
                        ne.set_arousal_bounds(min_gain, max_gain)
                    else:
                        if request.ne_min_gain is not None and hasattr(ne, "min_gain"):
                            ne.min_gain = request.ne_min_gain
                        if request.ne_max_gain is not None and hasattr(ne, "max_gain"):
                            ne.max_gain = request.ne_max_gain
                    if request.ne_min_gain is not None:
                        applied["ne_min_gain"] = request.ne_min_gain
                    if request.ne_max_gain is not None:
                        applied["ne_max_gain"] = request.ne_max_gain
                    ne_updated = True
                if ne_updated:
                    updated_systems.add("norepinephrine")

            # Apply ACh tuning
            if hasattr(orchestra, "acetylcholine"):
                ach = orchestra.acetylcholine
                ach_updated = False
                if request.ach_baseline is not None:
                    if hasattr(ach, "set_baseline_ach"):
                        ach.set_baseline_ach(request.ach_baseline)
                    elif hasattr(ach, "baseline"):
                        ach.baseline = request.ach_baseline
                    applied["ach_baseline"] = request.ach_baseline
                    ach_updated = True
                if request.ach_adaptation_rate is not None:
                    if hasattr(ach, "set_adaptation_rate"):
                        ach.set_adaptation_rate(request.ach_adaptation_rate)
                    elif hasattr(ach, "adaptation_rate"):
                        ach.adaptation_rate = request.ach_adaptation_rate
                    applied["ach_adaptation_rate"] = request.ach_adaptation_rate
                    ach_updated = True
                if ach_updated:
                    updated_systems.add("acetylcholine")

            # Apply DA tuning
            if hasattr(orchestra, "dopamine"):
                da = orchestra.dopamine
                da_updated = False
                if request.da_value_learning_rate is not None:
                    if hasattr(da, "set_value_learning_rate"):
                        da.set_value_learning_rate(request.da_value_learning_rate)
                    elif hasattr(da, "learning_rate"):
                        da.learning_rate = request.da_value_learning_rate
                    applied["da_value_learning_rate"] = request.da_value_learning_rate
                    da_updated = True
                if request.da_default_expected is not None:
                    if hasattr(da, "set_default_expected"):
                        da.set_default_expected(request.da_default_expected)
                    elif hasattr(da, "default_expected"):
                        da.default_expected = request.da_default_expected
                    applied["da_default_expected"] = request.da_default_expected
                    da_updated = True
                if request.da_surprise_threshold is not None:
                    if hasattr(da, "set_surprise_threshold"):
                        da.set_surprise_threshold(request.da_surprise_threshold)
                    elif hasattr(da, "surprise_threshold"):
                        da.surprise_threshold = request.da_surprise_threshold
                    applied["da_surprise_threshold"] = request.da_surprise_threshold
                    da_updated = True
                if da_updated:
                    updated_systems.add("dopamine")

            # Apply Serotonin tuning
            if hasattr(orchestra, "serotonin"):
                ser = orchestra.serotonin
                ser_updated = False
                if request.serotonin_baseline_mood is not None:
                    if hasattr(ser, "set_baseline_mood"):
                        ser.set_baseline_mood(request.serotonin_baseline_mood)
                    elif hasattr(ser, "baseline_mood"):
                        ser.baseline_mood = request.serotonin_baseline_mood
                    applied["serotonin_baseline_mood"] = request.serotonin_baseline_mood
                    ser_updated = True
                if request.serotonin_discount_rate is not None:
                    if hasattr(ser, "set_discount_rate"):
                        ser.set_discount_rate(request.serotonin_discount_rate)
                    elif hasattr(ser, "discount_rate"):
                        ser.discount_rate = request.serotonin_discount_rate
                    applied["serotonin_discount_rate"] = request.serotonin_discount_rate
                    ser_updated = True
                if request.serotonin_eligibility_decay is not None:
                    if hasattr(ser, "set_eligibility_decay"):
                        ser.set_eligibility_decay(request.serotonin_eligibility_decay)
                    elif hasattr(ser, "eligibility_decay"):
                        ser.eligibility_decay = request.serotonin_eligibility_decay
                    applied["serotonin_eligibility_decay"] = request.serotonin_eligibility_decay
                    ser_updated = True
                if ser_updated:
                    updated_systems.add("serotonin")

            # Apply Inhibition tuning (may be named 'inhibitory' or 'inhibition')
            # Try inhibitory first (common naming), then fallback to inhibition
            inh = None
            if hasattr(orchestra, "inhibitory"):
                inh = orchestra.inhibitory
            elif hasattr(orchestra, "inhibition"):
                inh = orchestra.inhibition
            if inh:
                inh_updated = False
                if request.inhibition_strength is not None:
                    if hasattr(inh, "set_inhibition_strength"):
                        inh.set_inhibition_strength(request.inhibition_strength)
                    elif hasattr(inh, "set_strength"):
                        inh.set_strength(request.inhibition_strength)
                    elif hasattr(inh, "strength"):
                        inh.strength = request.inhibition_strength
                    applied["inhibition_strength"] = request.inhibition_strength
                    inh_updated = True
                if request.sparsity_target is not None:
                    if hasattr(inh, "set_sparsity_target"):
                        inh.set_sparsity_target(request.sparsity_target)
                    elif hasattr(inh, "sparsity_target"):
                        inh.sparsity_target = request.sparsity_target
                    applied["sparsity_target"] = request.sparsity_target
                    inh_updated = True
                if request.inhibition_temperature is not None:
                    if hasattr(inh, "set_temperature"):
                        inh.set_temperature(request.inhibition_temperature)
                    elif hasattr(inh, "temperature"):
                        inh.temperature = request.inhibition_temperature
                    applied["inhibition_temperature"] = request.inhibition_temperature
                    inh_updated = True
                if inh_updated:
                    updated_systems.add("inhibition")

        # Get current state after tuning
        try:
            orchestra_stats = services["episodic"].get_orchestra_stats()
            current_state_data = services["episodic"].get_current_neuromodulator_state()
            current_state = NeuromodulatorOrchestraState(
                dopamine_rpe=current_state_data.get("dopamine_rpe", 0.0),
                norepinephrine_gain=current_state_data.get("norepinephrine_gain", 1.0),
                acetylcholine_mode=current_state_data.get("acetylcholine_mode", "balanced"),
                serotonin_mood=current_state_data.get("serotonin_mood", 0.5),
                inhibition_sparsity=current_state_data.get("inhibition_sparsity", 0.1),
                effective_learning_rate=current_state_data.get("effective_learning_rate", 1.0),
                exploration_exploitation=current_state_data.get("exploration_exploitation", 0.0),
                state_count=orchestra_stats.get("total_updates", 0),
            )
        except Exception:
            current_state = None

        logger.info(f"Neuromodulators tuned: {applied}")
        return NeuromodulatorTuningResponse(
            success=True,
            updated_systems=list(updated_systems),
            applied=applied,
            current_state=current_state,
        )

    except Exception as e:
        logger.error(f"Failed to tune neuromodulators: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@router.post("/bio/neuromodulators/reset")
async def reset_neuromodulators(
    services: MemoryServices,
    _: AdminAuth,
):
    """
    Reset all neuromodulator parameters to defaults.

    Requires admin authentication.
    """
    try:
        episodic = services.get("episodic")

        if hasattr(episodic, "orchestra") and episodic.orchestra:
            orchestra = episodic.orchestra

            # Reset each system - some use reset(), others reset_history()
            # NE: uses reset_history() for arousal/gain history
            ne = getattr(orchestra, "norepinephrine", None)
            if ne:
                if hasattr(ne, "reset_history"):
                    ne.reset_history()
                elif hasattr(ne, "reset"):
                    ne.reset()

            # ACh: uses reset() for mode state
            ach = getattr(orchestra, "acetylcholine", None)
            if ach and hasattr(ach, "reset"):
                ach.reset()

            # DA: uses reset_history() for prediction error history
            da = getattr(orchestra, "dopamine", None)
            if da:
                if hasattr(da, "reset_history"):
                    da.reset_history()
                elif hasattr(da, "reset"):
                    da.reset()

            # 5-HT: uses reset() for mood/eligibility
            ser = getattr(orchestra, "serotonin", None)
            if ser and hasattr(ser, "reset"):
                ser.reset()

            # GABA: uses reset_history() for inhibition history
            # Check inhibitory first (common naming), then inhibition
            inh = None
            if hasattr(orchestra, "inhibitory"):
                inh = orchestra.inhibitory
            elif hasattr(orchestra, "inhibition"):
                inh = orchestra.inhibition
            if inh:
                if hasattr(inh, "reset_history"):
                    inh.reset_history()
                elif hasattr(inh, "reset"):
                    inh.reset()

        return {"success": True, "status": "reset", "message": "Neuromodulator systems reset to defaults"}
    except Exception as e:
        logger.error(f"Failed to reset neuromodulators: {e}")
        raise HTTPException(status_code=500, detail=str(e))


class InteractionWeightsRequest(BaseModel):
    """Request to tune neuromodulator interaction weights."""
    # DA-NE interaction (arousal modulates reward learning)
    da_ne_weight: float | None = Field(default=None, ge=0.0, le=1.0)
    # ACh-5HT interaction (mode affects credit assignment)
    ach_serotonin_weight: float | None = Field(default=None, ge=0.0, le=1.0)
    # NE-GABA interaction (arousal affects inhibition)
    ne_inhibition_weight: float | None = Field(default=None, ge=0.0, le=1.0)
    # Three-factor learning weights
    three_factor_ach: float | None = Field(default=None, ge=0.0, le=1.0)
    three_factor_ne: float | None = Field(default=None, ge=0.0, le=1.0)
    three_factor_serotonin: float | None = Field(default=None, ge=0.0, le=1.0)


@router.put("/bio/neuromodulators/interactions")
async def tune_neuromodulator_interactions(
    request: InteractionWeightsRequest,
    services: MemoryServices,
    _: AdminAuth,
):
    """
    Tune neuromodulator interaction weights for three-factor learning.

    Controls how neuromodulators combine to modulate learning:
    - DA-NE: Arousal modulates surprise-driven learning
    - ACh-5HT: Encoding/retrieval mode affects credit assignment
    - NE-GABA: Arousal affects lateral inhibition strength
    - Three-factor weights: How DA, NE, ACh, 5-HT combine for learning rate

    Requires admin authentication.
    """
    try:
        episodic = services.get("episodic")
        applied = {}

        if hasattr(episodic, "orchestra") and episodic.orchestra:
            orchestra = episodic.orchestra

            # Apply three-factor weights if orchestra tracks them
            if hasattr(orchestra, "three_factor_weights"):
                if request.three_factor_ach is not None:
                    orchestra.three_factor_weights["ach"] = request.three_factor_ach
                    applied["three_factor_ach"] = request.three_factor_ach
                if request.three_factor_ne is not None:
                    orchestra.three_factor_weights["ne"] = request.three_factor_ne
                    applied["three_factor_ne"] = request.three_factor_ne
                if request.three_factor_serotonin is not None:
                    orchestra.three_factor_weights["serotonin"] = request.three_factor_serotonin
                    applied["three_factor_serotonin"] = request.three_factor_serotonin

            # Apply interaction weights if available
            if hasattr(orchestra, "interaction_weights"):
                if request.da_ne_weight is not None:
                    orchestra.interaction_weights["da_ne"] = request.da_ne_weight
                    applied["da_ne_weight"] = request.da_ne_weight
                if request.ach_serotonin_weight is not None:
                    orchestra.interaction_weights["ach_serotonin"] = request.ach_serotonin_weight
                    applied["ach_serotonin_weight"] = request.ach_serotonin_weight
                if request.ne_inhibition_weight is not None:
                    orchestra.interaction_weights["ne_inhibition"] = request.ne_inhibition_weight
                    applied["ne_inhibition_weight"] = request.ne_inhibition_weight

        logger.info(f"Neuromodulator interactions tuned: {applied}")
        return {"applied": applied, "status": "success"}

    except Exception as e:
        logger.error(f"Failed to tune interactions: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/bio/all", response_model=BiologicalMechanismsResponse)
async def get_all_biological_mechanisms(
    services: MemoryServices,
    fsrs_limit: int = Query(default=20, ge=1, le=100),
    hebbian_limit: int = Query(default=50, ge=1, le=200),
):
    """
    Get all biological mechanism states in one call.

    Combines FSRS, Hebbian, activation, sleep, WM, pattern, dopamine,
    reconsolidation, and learned fusion metrics.
    """
    try:
        # Gather all mechanism states
        fsrs_states = await get_fsrs_states(
            services=services,
            limit=fsrs_limit,
            include_decay_curve=False,
            forecast_days=30,
        )

        hebbian_weights = await get_hebbian_weights(
            services=services,
            limit=hebbian_limit,
            min_weight=0.1,
        )

        activation = await get_activation_spreading(
            services=services,
            source_id=None,
            limit=50,
        )

        sleep_state = await get_sleep_consolidation_state(services=services)
        wm_state = await get_working_memory_state(services=services)
        pattern_sep = await get_pattern_separation_metrics(
            services=services, input_a=None, input_b=None
        )
        pattern_comp = await get_pattern_completion_metrics(
            services=services, partial_input=None
        )

        # New biological mechanisms
        dopamine = await get_dopamine_metrics(services=services)
        reconsolidation = await get_reconsolidation_metrics(services=services)
        learned_fusion = await get_learned_fusion_metrics(services=services)

        # Neuromodulator orchestra state
        neuromodulator_orchestra = await get_neuromodulator_state(services=services)

        return BiologicalMechanismsResponse(
            fsrs_states=fsrs_states,
            hebbian_weights=hebbian_weights,
            activation_spreading=activation,
            sleep_consolidation=sleep_state,
            working_memory=wm_state,
            pattern_separation=pattern_sep,
            pattern_completion=pattern_comp,
            dopamine_rpe=dopamine,
            reconsolidation=reconsolidation,
            learned_fusion=learned_fusion,
            neuromodulator_orchestra=neuromodulator_orchestra,
        )

    except Exception as e:
        logger.error(f"Failed to get biological mechanisms: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Failed to get biological mechanisms: {e!s}",
        )


# Add missing import
from datetime import timedelta

# ============================================================================
# Neurocomputational Surgery Interface
# ============================================================================

class MemoryProvenance(BaseModel):
    """Full provenance and history of a memory."""
    memory_id: str
    memory_type: str
    content_preview: str
    created_at: float | None = None
    last_accessed: float | None = None
    access_count: int = 0
    stability: float = 1.0
    emotional_valence: float = 0.5
    reconsolidation_history: list[dict] = []  # Embedding updates
    dopamine_history: list[dict] = []  # RPE signals
    eligibility_trace: float = 0.0  # Current trace strength
    symbolic_connections: list[dict] = []  # Graph relationships
    co_retrieved_with: list[str] = []  # Memories often retrieved together


class CreditPath(BaseModel):
    """Credit assignment path from memory to outcome."""
    memory_id: str
    outcome_id: str
    path_weight: float  # Product of edge weights along path
    path_length: int
    hops: list[dict]  # Each hop: {from, to, relationship, weight}


class SurgeryResult(BaseModel):
    """Result of a surgery operation."""
    success: bool
    operation: str
    affected_memories: list[str]
    affected_edges: int = 0
    details: dict = {}


@router.get("/surgery/provenance/{memory_id}", response_model=MemoryProvenance)
async def get_memory_provenance(
    memory_id: str,
    services: MemoryServices,
    include_history: bool = Query(default=True, description="Include modification history"),
    max_history: int = Query(default=50, ge=1, le=500, description="Max history entries"),
):
    """
    Get complete provenance and history of a memory.

    Shows all modifications, credit assignments, and relationships
    for neurocomputational debugging.
    """
    # Validate UUID early - returns 400 for invalid format
    mem_uuid = parse_uuid(memory_id, "memory_id")

    try:

        # Get dopamine history
        dopamine_history = []
        if include_history:
            for rpe in services["episodic"].dopamine._rpe_history[-max_history:]:
                if str(rpe.memory_id) == memory_id:
                    dopamine_history.append({
                        "expected": rpe.expected,
                        "actual": rpe.actual,
                        "rpe": rpe.rpe,
                        "timestamp": rpe.timestamp.isoformat(),
                        "was_surprising": rpe.is_positive_surprise or rpe.is_negative_surprise
                    })

        # Get reconsolidation history
        recon_history = []
        if include_history:
            for update in services["episodic"].reconsolidation.get_update_history(
                memory_id=mem_uuid, limit=max_history
            ):
                recon_history.append({
                    "advantage": update.advantage,
                    "learning_rate": update.learning_rate,
                    "timestamp": update.timestamp.isoformat(),
                    "update_magnitude": float(np.linalg.norm(
                        update.updated_embedding - update.original_embedding
                    ))
                })

        # Get expected value from dopamine system
        expected_value = services["episodic"].dopamine.get_expected_value(mem_uuid)

        return MemoryProvenance(
            memory_id=memory_id,
            memory_type="episodic",
            content_preview="[Memory content would be loaded here]",
            access_count=0,
            stability=1.0,
            emotional_valence=expected_value,
            reconsolidation_history=recon_history,
            dopamine_history=dopamine_history,
            eligibility_trace=0.0,
            symbolic_connections=[],
            co_retrieved_with=[]
        )

    except Exception as e:
        logger.error(f"Failed to get memory provenance: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Failed to get provenance: {e!s}"
        )


@router.get("/surgery/trace-credit/{memory_id}", response_model=list[CreditPath])
async def trace_credit_paths(
    memory_id: str,
    services: MemoryServices,
    max_depth: int = Query(default=3, ge=1, le=10, description="Maximum path depth"),
    min_weight: float = Query(default=0.1, ge=0.0, le=1.0, description="Minimum path weight"),
):
    """
    Trace credit assignment paths from a memory to outcomes.

    Shows how this memory contributes to downstream decisions
    through symbolic graph traversal.
    """
    try:
        # Placeholder - would query Neo4j for paths to outcome nodes
        paths = []
        return paths

    except Exception as e:
        logger.error(f"Failed to trace credit paths: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Failed to trace credit: {e!s}"
        )


@router.post("/surgery/reset-dopamine/{memory_id}", response_model=SurgeryResult)
async def reset_dopamine_for_memory(
    memory_id: str,
    services: MemoryServices,
    new_expected: float = Query(default=0.5, ge=0.0, le=1.0, description="New expected value"),
):
    """
    Reset dopamine value estimate for a memory.

    Use when a memory's expected value has drifted incorrectly.
    """
    # Validate UUID early - returns 400 for invalid format
    mem_uuid = parse_uuid(memory_id, "memory_id")
    mem_str = str(mem_uuid)

    try:

        # Reset value estimate
        old_value = services["episodic"].dopamine._value_estimates.get(mem_str, 0.5)
        services["episodic"].dopamine._value_estimates[mem_str] = new_expected
        services["episodic"].dopamine._outcome_counts[mem_str] = 0  # Reset uncertainty

        return SurgeryResult(
            success=True,
            operation="reset_dopamine",
            affected_memories=[memory_id],
            details={
                "old_expected": old_value,
                "new_expected": new_expected,
                "uncertainty_reset": True
            }
        )

    except Exception as e:
        logger.error(f"Failed to reset dopamine: {e}")
        return SurgeryResult(
            success=False,
            operation="reset_dopamine",
            affected_memories=[],
            details={"error": str(e)}
        )


@router.post("/surgery/clear-reconsolidation-cooldown/{memory_id}", response_model=SurgeryResult)
async def clear_reconsolidation_cooldown(
    memory_id: str,
    services: MemoryServices,
):
    """
    Clear reconsolidation cooldown for a memory.

    Allows immediate re-reconsolidation (use carefully - may cause instability).
    """
    # Validate UUID early - returns 400 for invalid format
    mem_str = str(parse_uuid(memory_id, "memory_id"))

    try:

        # Remove from last_update tracking
        was_in_cooldown = mem_str in services["episodic"].reconsolidation._last_update
        if was_in_cooldown:
            del services["episodic"].reconsolidation._last_update[mem_str]

        return SurgeryResult(
            success=True,
            operation="clear_reconsolidation_cooldown",
            affected_memories=[memory_id],
            details={
                "was_in_cooldown": was_in_cooldown
            }
        )

    except Exception as e:
        logger.error(f"Failed to clear cooldown: {e}")
        return SurgeryResult(
            success=False,
            operation="clear_reconsolidation_cooldown",
            affected_memories=[],
            details={"error": str(e)}
        )


@router.get("/surgery/dopamine-value-map")
async def get_dopamine_value_map(
    services: MemoryServices,
    min_observations: int = Query(default=1, ge=0, description="Minimum observations"),
):
    """
    Get map of all memories' expected values.

    Useful for understanding which memories are expected to be
    useful (high value) vs not useful (low value).
    """
    try:
        value_map = []
        for mem_id, value in services["episodic"].dopamine._value_estimates.items():
            count = services["episodic"].dopamine._outcome_counts.get(mem_id, 0)
            if count >= min_observations:
                uncertainty = services["episodic"].dopamine.get_uncertainty(UUID(mem_id))
                value_map.append({
                    "memory_id": mem_id,
                    "expected_value": value,
                    "observation_count": count,
                    "uncertainty": uncertainty,
                    "confidence": 1.0 - uncertainty
                })

        # Sort by value descending
        value_map.sort(key=lambda x: x["expected_value"], reverse=True)
        return value_map

    except Exception as e:
        logger.error(f"Failed to get value map: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Failed to get value map: {e!s}"
        )


@router.get("/surgery/reconsolidation-history")
async def get_reconsolidation_history(
    services: MemoryServices,
    limit: int = Query(default=100, ge=1, le=1000),
    memory_id: str | None = Query(default=None, description="Filter to specific memory"),
):
    """
    Get complete reconsolidation history.

    Shows all embedding updates across all memories for debugging
    learning dynamics.
    """
    # Validate optional UUID - returns 400 for invalid format
    mem_uuid = parse_uuid(memory_id, "memory_id") if memory_id else None

    try:
        history = services["episodic"].reconsolidation.get_update_history(
            memory_id=mem_uuid,
            limit=limit
        )

        return [
            {
                "memory_id": str(u.memory_id),
                "advantage": u.advantage,
                "learning_rate": u.learning_rate,
                "outcome_score": u.outcome_score,
                "update_magnitude": float(np.linalg.norm(
                    u.updated_embedding - u.original_embedding
                )),
                "timestamp": u.timestamp.isoformat()
            }
            for u in history
        ]

    except Exception as e:
        logger.error(f"Failed to get reconsolidation history: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Failed to get history: {e!s}"
        )


@router.get("/surgery/learning-stats")
async def get_comprehensive_learning_stats(services: MemoryServices):
    """
    Get comprehensive learning system statistics.

    Combines dopamine, reconsolidation, and fusion training stats
    for system-wide learning health monitoring.
    """
    try:
        dopamine_stats = services["episodic"].get_dopamine_stats()
        recon_stats = services["episodic"].get_reconsolidation_stats()
        fusion_stats = services["episodic"].get_fusion_training_stats()

        return {
            "dopamine": dopamine_stats,
            "reconsolidation": recon_stats,
            "learned_fusion": fusion_stats,
            "learning_health": {
                "dopamine_active": dopamine_stats.get("total_signals", 0) > 0,
                "reconsolidation_active": recon_stats.get("total_updates", 0) > 0,
                "fusion_training_active": fusion_stats.get("train_steps", 0) > 0,
                "positive_surprise_ratio": (
                    dopamine_stats.get("positive_surprises", 0) /
                    max(dopamine_stats.get("total_signals", 1), 1)
                ),
                "avg_update_magnitude": recon_stats.get("avg_update_magnitude", 0.0),
            }
        }

    except Exception as e:
        logger.error(f"Failed to get learning stats: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Failed to get stats: {e!s}"
        )


# =============================================================================
# Eligibility Traces Endpoints (for EligibilityTracesPanel)
# =============================================================================

class EligibilityTraceEntry(BaseModel):
    """Single eligibility trace entry."""
    memoryId: str
    value: float
    activations: int
    lastUpdate: str


class EligibilityTracesResponse(BaseModel):
    """Response for eligibility traces list."""
    traces: list[EligibilityTraceEntry]
    traceType: str = "standard"


class EligibilityStatsResponse(BaseModel):
    """Eligibility trace statistics."""
    count: int
    meanTrace: float
    maxTrace: float
    totalUpdates: int
    totalCreditsAssigned: float
    traceType: str = "standard"


class CreditAssignmentResponse(BaseModel):
    """Response after assigning credit."""
    topCredits: dict[str, float]
    totalAssigned: float


@router.get("/bio/eligibility/traces", response_model=EligibilityTracesResponse)
async def get_eligibility_traces(
    services: MemoryServices,
    layered: bool = Query(default=False, description="Use layered traces (fast/slow)"),
):
    """
    Get current eligibility traces for all tracked memories.
    """
    try:
        # Try to get serotonin system for traces
        episodic = services.get("episodic")
        traces = []

        if hasattr(episodic, "orchestra") and episodic.orchestra:
            serotonin = episodic.orchestra.serotonin
            if hasattr(serotonin, "eligibility_traces"):
                for mem_id, trace in serotonin.eligibility_traces.items():
                    traces.append(EligibilityTraceEntry(
                        memoryId=str(mem_id),
                        value=trace.strength if hasattr(trace, "strength") else trace,
                        activations=trace.activations if hasattr(trace, "activations") else 1,
                        lastUpdate=trace.timestamp.isoformat() if hasattr(trace, "timestamp") else datetime.now().isoformat(),
                    ))

        return EligibilityTracesResponse(
            traces=traces,
            traceType="layered" if layered else "standard"
        )
    except Exception as e:
        logger.error(f"Failed to get eligibility traces: {e}")
        return EligibilityTracesResponse(traces=[], traceType="standard")


@router.get("/bio/eligibility/stats", response_model=EligibilityStatsResponse)
async def get_eligibility_stats(
    services: MemoryServices,
    layered: bool = Query(default=False, description="Use layered traces"),
):
    """
    Get eligibility trace statistics.
    """
    try:
        episodic = services.get("episodic")
        traces = []
        total_updates = 0
        total_credits = 0.0

        if hasattr(episodic, "orchestra") and episodic.orchestra:
            serotonin = episodic.orchestra.serotonin
            if hasattr(serotonin, "eligibility_traces"):
                for trace in serotonin.eligibility_traces.values():
                    val = trace.strength if hasattr(trace, "strength") else trace
                    traces.append(val)
            if hasattr(serotonin, "total_updates"):
                total_updates = serotonin.total_updates
            if hasattr(serotonin, "total_credits_assigned"):
                total_credits = serotonin.total_credits_assigned

        if traces:
            return EligibilityStatsResponse(
                count=len(traces),
                meanTrace=sum(traces) / len(traces),
                maxTrace=max(traces),
                totalUpdates=total_updates,
                totalCreditsAssigned=total_credits,
                traceType="layered" if layered else "standard"
            )
        return EligibilityStatsResponse(
            count=0,
            meanTrace=0.0,
            maxTrace=0.0,
            totalUpdates=0,
            totalCreditsAssigned=0.0,
            traceType="layered" if layered else "standard"
        )
    except Exception as e:
        logger.error(f"Failed to get eligibility stats: {e}")
        return EligibilityStatsResponse(
            count=0, meanTrace=0.0, maxTrace=0.0,
            totalUpdates=0, totalCreditsAssigned=0.0
        )


@router.post("/bio/eligibility/step")
async def step_eligibility_decay(
    services: MemoryServices,
    dt: float = 1.0,
    useLayered: bool = False,
):
    """
    Trigger a decay step on all eligibility traces.
    """
    try:
        episodic = services.get("episodic")
        decayed_count = 0

        if hasattr(episodic, "orchestra") and episodic.orchestra:
            serotonin = episodic.orchestra.serotonin
            if hasattr(serotonin, "decay_traces"):
                decayed_count = serotonin.decay_traces(dt=dt)
            elif hasattr(serotonin, "eligibility_traces"):
                # Manual decay
                import math
                tau = 5.0 if useLayered else 20.0
                decay_factor = math.exp(-dt / tau)
                for mem_id in list(serotonin.eligibility_traces.keys()):
                    trace = serotonin.eligibility_traces[mem_id]
                    if hasattr(trace, "strength"):
                        trace.strength *= decay_factor
                        if trace.strength < 0.001:
                            del serotonin.eligibility_traces[mem_id]
                    else:
                        serotonin.eligibility_traces[mem_id] *= decay_factor
                        if serotonin.eligibility_traces[mem_id] < 0.001:
                            del serotonin.eligibility_traces[mem_id]
                decayed_count = len(serotonin.eligibility_traces)

        return {"decayed": True, "remaining_traces": decayed_count}
    except Exception as e:
        logger.error(f"Failed to decay traces: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@router.post("/bio/eligibility/credit", response_model=CreditAssignmentResponse)
async def assign_eligibility_credit(
    services: MemoryServices,
    reward: float = 1.0,
    useLayered: bool = False,
):
    """
    Assign credit to memories based on their eligibility traces.
    """
    try:
        episodic = services.get("episodic")
        credits = {}
        total = 0.0

        if hasattr(episodic, "orchestra") and episodic.orchestra:
            serotonin = episodic.orchestra.serotonin
            if hasattr(serotonin, "receive_outcome"):
                credits = serotonin.receive_outcome(reward)
                total = sum(credits.values())
            elif hasattr(serotonin, "eligibility_traces"):
                # Manual credit assignment
                for mem_id, trace in serotonin.eligibility_traces.items():
                    val = trace.strength if hasattr(trace, "strength") else trace
                    credit = reward * val
                    credits[str(mem_id)] = credit
                    total += credit

        # Return top 10 by absolute credit
        sorted_credits = dict(
            sorted(credits.items(), key=lambda x: abs(x[1]), reverse=True)[:10]
        )
        return CreditAssignmentResponse(topCredits=sorted_credits, totalAssigned=total)
    except Exception as e:
        logger.error(f"Failed to assign credit: {e}")
        raise HTTPException(status_code=500, detail=str(e))


class TopEligibilityResponse(BaseModel):
    """Top-k eligibility traces by strength."""
    traces: list[EligibilityTraceEntry]
    total_tracked: int
    cutoff_value: float


@router.get("/bio/eligibility/top-k", response_model=TopEligibilityResponse)
async def get_top_eligibility_traces(
    services: MemoryServices,
    k: int = Query(default=10, ge=1, le=100, description="Number of top traces"),
):
    """
    Get the top-k eligibility traces by strength.

    Useful for monitoring which memories are most eligible for credit assignment.
    """
    try:
        episodic = services.get("episodic")
        all_traces = []

        if hasattr(episodic, "orchestra") and episodic.orchestra:
            serotonin = episodic.orchestra.serotonin
            if hasattr(serotonin, "eligibility_traces"):
                for mem_id, trace in serotonin.eligibility_traces.items():
                    val = trace.strength if hasattr(trace, "strength") else trace
                    all_traces.append((str(mem_id), val, trace))

        # Sort by value descending
        all_traces.sort(key=lambda x: x[1], reverse=True)
        top_k = all_traces[:k]

        entries = []
        cutoff = 0.0
        for mem_id, val, trace in top_k:
            entries.append(EligibilityTraceEntry(
                memoryId=mem_id,
                value=val,
                activations=trace.activations if hasattr(trace, "activations") else 1,
                lastUpdate=trace.timestamp.isoformat() if hasattr(trace, "timestamp") else datetime.now().isoformat(),
            ))
            cutoff = val

        return TopEligibilityResponse(
            traces=entries,
            total_tracked=len(all_traces),
            cutoff_value=cutoff,
        )
    except Exception as e:
        logger.error(f"Failed to get top-k traces: {e}")
        return TopEligibilityResponse(traces=[], total_tracked=0, cutoff_value=0.0)


# =============================================================================
# Learning Dynamics Endpoints
# =============================================================================

class EffectiveLRResponse(BaseModel):
    """Current effective learning rate computed from neuromodulators."""
    effective_lr: float
    base_lr: float
    components: dict[str, float]  # Individual neuromodulator contributions
    three_factor_signal: float
    bootstrap_contribution: float


class CreditFlowResponse(BaseModel):
    """Credit flow through the system."""
    total_credit_assigned: float
    credit_by_memory_type: dict[str, float]
    recent_outcomes: list[dict]
    eligibility_decay_rate: float
    active_traces_count: int


class LearningDynamicsResponse(BaseModel):
    """Combined learning dynamics state."""
    effective_lr: EffectiveLRResponse
    credit_flow: CreditFlowResponse
    timestamp: str


@router.get("/bio/learning/effective-lr", response_model=EffectiveLRResponse)
async def get_effective_learning_rate(services: MemoryServices):
    """
    Get the current effective learning rate.

    Shows how DA, NE, ACh, and 5-HT combine via three-factor learning
    to modulate the base learning rate.
    """
    try:
        episodic = services.get("episodic")
        components = {}
        three_factor = 1.0
        bootstrap = 0.01
        base_lr = 0.1

        if hasattr(episodic, "orchestra") and episodic.orchestra:
            orchestra = episodic.orchestra

            # Get individual contributions
            if hasattr(orchestra, "get_learning_signal"):
                signal = orchestra.get_learning_signal()
                if hasattr(signal, "effective_lr_multiplier"):
                    three_factor = signal.effective_lr_multiplier
                if hasattr(signal, "bootstrap"):
                    bootstrap = signal.bootstrap

            # Gather component contributions
            if hasattr(orchestra, "norepinephrine"):
                ne = orchestra.norepinephrine
                components["norepinephrine"] = getattr(ne, "current_gain", 1.0)

            if hasattr(orchestra, "acetylcholine"):
                ach = orchestra.acetylcholine
                components["acetylcholine"] = getattr(ach, "ach_level", 0.5)

            if hasattr(orchestra, "dopamine"):
                da = orchestra.dopamine
                components["dopamine"] = getattr(da, "last_rpe", 0.0)

            if hasattr(orchestra, "serotonin"):
                ser = orchestra.serotonin
                components["serotonin"] = getattr(ser, "current_mood", 0.5)

        effective = base_lr * three_factor + bootstrap

        return EffectiveLRResponse(
            effective_lr=effective,
            base_lr=base_lr,
            components=components,
            three_factor_signal=three_factor,
            bootstrap_contribution=bootstrap,
        )
    except Exception as e:
        logger.error(f"Failed to get effective LR: {e}")
        return EffectiveLRResponse(
            effective_lr=0.1,
            base_lr=0.1,
            components={},
            three_factor_signal=1.0,
            bootstrap_contribution=0.01,
        )


@router.get("/bio/learning/credit-flow", response_model=CreditFlowResponse)
async def get_credit_flow(services: MemoryServices):
    """
    Get credit flow through the learning system.

    Shows how temporal difference signals and eligibility traces
    combine to assign credit to memories.
    """
    try:
        episodic = services.get("episodic")
        total_credit = 0.0
        by_type = {"episodic": 0.0, "semantic": 0.0, "procedural": 0.0}
        recent_outcomes = []
        decay_rate = 0.95
        active_count = 0

        if hasattr(episodic, "orchestra") and episodic.orchestra:
            serotonin = episodic.orchestra.serotonin

            if hasattr(serotonin, "total_credits_assigned"):
                total_credit = serotonin.total_credits_assigned

            if hasattr(serotonin, "eligibility_traces"):
                active_count = len(serotonin.eligibility_traces)

            if hasattr(serotonin, "decay_rate"):
                decay_rate = serotonin.decay_rate

            if hasattr(serotonin, "recent_outcomes"):
                recent_outcomes = list(serotonin.recent_outcomes)[-10:]

        return CreditFlowResponse(
            total_credit_assigned=total_credit,
            credit_by_memory_type=by_type,
            recent_outcomes=recent_outcomes,
            eligibility_decay_rate=decay_rate,
            active_traces_count=active_count,
        )
    except Exception as e:
        logger.error(f"Failed to get credit flow: {e}")
        return CreditFlowResponse(
            total_credit_assigned=0.0,
            credit_by_memory_type={},
            recent_outcomes=[],
            eligibility_decay_rate=0.95,
            active_traces_count=0,
        )


@router.get("/bio/learning/dynamics", response_model=LearningDynamicsResponse)
async def get_learning_dynamics(services: MemoryServices):
    """
    Get combined learning dynamics state.

    Includes effective learning rate and credit flow in one call.
    """
    try:
        effective_lr = await get_effective_learning_rate(services)
        credit_flow = await get_credit_flow(services)

        return LearningDynamicsResponse(
            effective_lr=effective_lr,
            credit_flow=credit_flow,
            timestamp=datetime.now().isoformat(),
        )
    except Exception as e:
        logger.error(f"Failed to get learning dynamics: {e}")
        raise HTTPException(status_code=500, detail=str(e))


# =============================================================================
# Temporal Dynamics Endpoints
# =============================================================================

class NeuromodulatorTracePoint(BaseModel):
    """Single point in a neuromodulator time series."""
    timestamp: str
    da: float
    ne: float
    ach: float
    serotonin: float
    inhibition: float


class NeuromodulatorTracesResponse(BaseModel):
    """Historical traces of neuromodulator levels."""
    traces: list[NeuromodulatorTracePoint]
    window_seconds: float
    sample_count: int


class SimulationRequest(BaseModel):
    """Request to simulate neuromodulator dynamics."""
    steps: int = Field(default=100, ge=1, le=1000)
    dt: float = Field(default=0.1, ge=0.01, le=1.0)
    # Input signals
    novelty_input: float = Field(default=0.5, ge=0.0, le=1.0)
    reward_input: float = Field(default=0.0, ge=-1.0, le=1.0)
    attention_input: float = Field(default=0.5, ge=0.0, le=1.0)


class SimulationResponse(BaseModel):
    """Response from neuromodulator simulation."""
    traces: list[NeuromodulatorTracePoint]
    final_state: dict[str, float]
    mode_switches: int


@router.get("/bio/neuromodulators/traces", response_model=NeuromodulatorTracesResponse)
async def get_neuromodulator_traces(
    services: MemoryServices,
    window: float = Query(default=60.0, ge=1.0, le=3600.0, description="Seconds of history"),
    max_points: int = Query(default=100, ge=10, le=1000, description="Max data points"),
):
    """
    Get recent neuromodulator traces for visualization.

    Returns time series of DA, NE, ACh, 5-HT, and GABA levels
    over the specified time window.
    """
    try:
        episodic = services.get("episodic")
        traces = []

        if hasattr(episodic, "orchestra") and episodic.orchestra:
            orchestra = episodic.orchestra

            # Check if orchestra has history
            if hasattr(orchestra, "state_history"):
                history = list(orchestra.state_history)[-max_points:]
                for state in history:
                    traces.append(NeuromodulatorTracePoint(
                        timestamp=state.get("timestamp", datetime.now().isoformat()),
                        da=state.get("dopamine_rpe", 0.0),
                        ne=state.get("norepinephrine_gain", 1.0),
                        ach=state.get("ach_level", 0.5),
                        serotonin=state.get("serotonin_mood", 0.5),
                        inhibition=state.get("inhibition_sparsity", 0.1),
                    ))

        return NeuromodulatorTracesResponse(
            traces=traces,
            window_seconds=window,
            sample_count=len(traces),
        )
    except Exception as e:
        logger.error(f"Failed to get neuromodulator traces: {e}")
        return NeuromodulatorTracesResponse(traces=[], window_seconds=window, sample_count=0)


@router.post("/bio/neuromodulators/simulate", response_model=SimulationResponse)
async def simulate_neuromodulators(
    request: SimulationRequest,
    services: MemoryServices,
):
    """
    Simulate neuromodulator dynamics forward in time.

    Runs a forward simulation with specified inputs to predict
    system behavior. Does NOT modify actual system state.
    """
    try:

        traces = []
        mode_switches = 0

        # Initial state from current system
        da = 0.5
        ne = 1.0
        ach = 0.5
        serotonin = 0.5
        inhibition = 0.1
        current_mode = "balanced"

        # Simulation parameters (simplified dynamics)
        tau_da = 5.0
        tau_ne = 10.0
        tau_ach = 20.0
        tau_5ht = 50.0

        for step in range(request.steps):
            t = step * request.dt

            # Dopamine: responds to reward prediction error
            da_target = 0.5 + request.reward_input * 0.5
            da += (da_target - da) * request.dt / tau_da

            # Norepinephrine: responds to novelty/arousal
            ne_target = 1.0 + request.novelty_input * 1.0
            ne += (ne_target - ne) * request.dt / tau_ne

            # Acetylcholine: responds to attention, controls mode
            ach_target = request.attention_input
            ach += (ach_target - ach) * request.dt / tau_ach

            # Track mode switches
            new_mode = "balanced"
            if ach > 0.7:
                new_mode = "encoding"
            elif ach < 0.3:
                new_mode = "retrieval"
            if new_mode != current_mode:
                mode_switches += 1
                current_mode = new_mode

            # Serotonin: slow mood dynamics
            serotonin_target = 0.5 + da * 0.2
            serotonin += (serotonin_target - serotonin) * request.dt / tau_5ht

            # Inhibition: modulated by arousal
            inhibition = max(0.05, min(0.5, 0.1 * ne))

            traces.append(NeuromodulatorTracePoint(
                timestamp=f"t+{t:.2f}s",
                da=round(da, 4),
                ne=round(ne, 4),
                ach=round(ach, 4),
                serotonin=round(serotonin, 4),
                inhibition=round(inhibition, 4),
            ))

        return SimulationResponse(
            traces=traces,
            final_state={
                "dopamine": da,
                "norepinephrine": ne,
                "acetylcholine": ach,
                "serotonin": serotonin,
                "inhibition": inhibition,
                "mode": current_mode,
            },
            mode_switches=mode_switches,
        )
    except Exception as e:
        logger.error(f"Simulation failed: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@router.post("/bio/neuromodulators/step")
async def step_neuromodulators(
    services: MemoryServices,
    dt: float = Query(default=1.0, ge=0.01, le=10.0, description="Time step in seconds"),
    novelty: float = Query(default=0.0, ge=0.0, le=1.0, description="Novelty input"),
    reward: float = Query(default=0.0, ge=-1.0, le=1.0, description="Reward signal"),
):
    """
    Step the neuromodulator system forward in time.

    Updates actual system state with specified inputs.
    Use for testing/debugging dynamics.
    """
    try:
        episodic = services.get("episodic")
        updates = {}

        if hasattr(episodic, "orchestra") and episodic.orchestra:
            orchestra = episodic.orchestra

            # Update NE with novelty
            if hasattr(orchestra, "norepinephrine") and hasattr(orchestra.norepinephrine, "update"):
                ne_state = orchestra.norepinephrine.update(novelty_score=novelty)
                updates["norepinephrine_gain"] = getattr(ne_state, "gain", 1.0)

            # Update DA with reward
            if hasattr(orchestra, "dopamine") and hasattr(orchestra.dopamine, "compute_rpe"):
                rpe = orchestra.dopamine.compute_rpe(reward, 0.5)
                updates["dopamine_rpe"] = rpe

            # Update ACh
            if hasattr(orchestra, "acetylcholine") and hasattr(orchestra.acetylcholine, "update"):
                ach_state = orchestra.acetylcholine.update()
                updates["acetylcholine_mode"] = str(getattr(ach_state, "mode", "balanced"))

            # Update serotonin with reward signal
            if hasattr(orchestra, "serotonin") and hasattr(orchestra.serotonin, "receive_outcome"):
                orchestra.serotonin.receive_outcome(reward)
                updates["serotonin_mood"] = getattr(orchestra.serotonin, "current_mood", 0.5)

        return {"stepped": True, "dt": dt, "updates": updates}
    except Exception as e:
        logger.error(f"Failed to step neuromodulators: {e}")
        raise HTTPException(status_code=500, detail=str(e))


# =============================================================================
# Sparse Encoding Endpoints (for SparseEncodingPanel)
# =============================================================================

class SparseEncoderStatsResponse(BaseModel):
    """Sparse encoder configuration and stats."""
    inputDim: int = 1024
    hiddenDim: int = 8192
    sparsity: float = 0.02
    expansionRatio: float = 8.0


class SparseEncodingResponse(BaseModel):
    """Response from encoding content."""
    contentLength: int
    sparseDim: int
    activeCount: int
    sparsity: float
    targetSparsity: float
    activeIndices: list[int] | None = None


@router.get("/bio/sparse-encoder/stats", response_model=SparseEncoderStatsResponse)
async def get_sparse_encoder_stats(services: MemoryServices):
    """
    Get sparse encoder configuration.
    """
    try:
        episodic = services.get("episodic")

        # Try to get pattern separator config
        if hasattr(episodic, "pattern_separator") and episodic.pattern_separator:
            ps = episodic.pattern_separator
            input_dim = getattr(ps, "input_dim", 1024)
            hidden_dim = getattr(ps, "hidden_dim", 8192)
            sparsity = getattr(ps, "target_sparsity", 0.02)
            return SparseEncoderStatsResponse(
                inputDim=input_dim,
                hiddenDim=hidden_dim,
                sparsity=sparsity,
                expansionRatio=hidden_dim / input_dim
            )

        # Default values based on DG biology
        return SparseEncoderStatsResponse()
    except Exception as e:
        logger.error(f"Failed to get encoder stats: {e}")
        return SparseEncoderStatsResponse()


@router.post("/bio/encode", response_model=SparseEncodingResponse)
async def encode_content_sparse(
    services: MemoryServices,
    content: str = "",
    returnIndices: bool = False,
):
    """
    Encode content using sparse k-WTA encoding.
    """
    try:
        import numpy as np

        services.get("episodic")
        embedding_service = services.get("embedding")

        # Get embedding first
        embedding = await embedding_service.embed_query(content)
        embedding_np = np.array(embedding)

        hidden_dim = 8192
        target_sparsity = 0.02
        k = int(hidden_dim * target_sparsity)

        # Simulate sparse encoding (k-WTA)
        # Project to higher dimension
        np.random.seed(hash(content) % 2**31)
        projection = np.random.randn(len(embedding), hidden_dim) * 0.1
        hidden = embedding_np @ projection

        # k-WTA: keep only top k activations
        threshold = np.partition(hidden, -k)[-k]
        sparse = np.where(hidden >= threshold, hidden, 0)
        active_indices = np.where(sparse > 0)[0].tolist()

        actual_sparsity = len(active_indices) / hidden_dim

        return SparseEncodingResponse(
            contentLength=len(content),
            sparseDim=hidden_dim,
            activeCount=len(active_indices),
            sparsity=actual_sparsity,
            targetSparsity=target_sparsity,
            activeIndices=active_indices if returnIndices else None
        )
    except Exception as e:
        logger.error(f"Failed to encode: {e}")
        raise HTTPException(status_code=500, detail=str(e))


# =============================================================================
# Fast Episodic Store Endpoints (for FastEpisodicPanel)
# =============================================================================

class FESStatsResponse(BaseModel):
    """Fast Episodic Store statistics."""
    count: int
    capacity: int
    capacityUsage: float
    avgSalience: float
    consolidationCandidates: int


class EpisodePreview(BaseModel):
    """Episode preview for listing."""
    id: str
    contentPreview: str
    salience: float
    timestamp: str
    replayCount: int


class FESEpisodesResponse(BaseModel):
    """Response with episode previews."""
    episodes: list[EpisodePreview]


@router.get("/bio/fes/stats", response_model=FESStatsResponse)
async def get_fes_stats(services: MemoryServices):
    """
    Get Fast Episodic Store statistics.
    """
    try:
        episodic = services.get("episodic")

        # Get episode count
        count = 0
        capacity = 10000
        avg_salience = 0.5
        consolidation_candidates = 0

        if hasattr(episodic, "count"):
            count = episodic.count()
        elif hasattr(episodic, "_episodes"):
            count = len(episodic._episodes)

        if hasattr(episodic, "capacity"):
            capacity = episodic.capacity

        # Estimate consolidation candidates (episodes with high salience and age)
        consolidation_candidates = int(count * 0.1)  # Estimate 10%

        return FESStatsResponse(
            count=count,
            capacity=capacity,
            capacityUsage=count / capacity if capacity > 0 else 0,
            avgSalience=avg_salience,
            consolidationCandidates=consolidation_candidates
        )
    except Exception as e:
        logger.error(f"Failed to get FES stats: {e}")
        return FESStatsResponse(
            count=0, capacity=10000, capacityUsage=0,
            avgSalience=0.5, consolidationCandidates=0
        )


@router.get("/bio/fes/recent", response_model=FESEpisodesResponse)
async def get_recent_episodes(
    services: MemoryServices,
    limit: int = Query(default=20, ge=1, le=100),
):
    """
    Get recent episodes from the Fast Episodic Store.
    """
    try:
        episodic = services.get("episodic")
        episodes = []

        # Try to get recent episodes
        if hasattr(episodic, "get_recent"):
            recent = await episodic.get_recent(limit=limit)
            for ep in recent:
                episodes.append(EpisodePreview(
                    id=str(ep.id) if hasattr(ep, "id") else str(ep.get("id", "")),
                    contentPreview=(ep.content if hasattr(ep, "content") else ep.get("content", ""))[:100],
                    salience=ep.emotional_valence if hasattr(ep, "emotional_valence") else 0.5,
                    timestamp=ep.timestamp.isoformat() if hasattr(ep, "timestamp") else datetime.now().isoformat(),
                    replayCount=ep.access_count if hasattr(ep, "access_count") else 0,
                ))

        return FESEpisodesResponse(episodes=episodes)
    except Exception as e:
        logger.error(f"Failed to get recent episodes: {e}")
        return FESEpisodesResponse(episodes=[])


@router.post("/bio/fes/write")
async def write_episode_fes(
    services: MemoryServices,
    content: str = "",
    neuromodState: dict | None = None,
):
    """
    Write a new episode to the Fast Episodic Store.
    """
    try:
        episodic = services.get("episodic")

        # Compute salience from neuromod state
        salience = 0.5
        if neuromodState:
            da = neuromodState.get("dopamine", 0.5)
            ne = neuromodState.get("norepinephrine", 0.5)
            ach = neuromodState.get("acetylcholine", 0.5)
            salience = da * ne * ach

        # Create episode
        if hasattr(episodic, "create"):
            episode = await episodic.create(
                content=content,
                valence=salience,
                outcome="neutral"
            )
            return {
                "success": True,
                "episodeId": str(episode.id) if hasattr(episode, "id") else None,
                "salience": salience
            }

        return {"success": False, "error": "Episodic memory not available"}
    except Exception as e:
        logger.error(f"Failed to write episode: {e}")
        raise HTTPException(status_code=500, detail=str(e))


# ============================================================================
# Homeostatic Plasticity Endpoints
# ============================================================================

class HomeostaticStateResponse(BaseModel):
    """Homeostatic plasticity state."""
    mean_norm: float = Field(description="Current mean embedding norm")
    std_norm: float = Field(description="Std of embedding norms")
    mean_activation: float = Field(description="Mean recent activation")
    sliding_threshold: float = Field(description="BCM threshold for LTP/LTD")
    last_update: str | None = Field(description="Last update timestamp")
    needs_scaling: bool = Field(description="Whether scaling is needed")
    current_scaling_factor: float = Field(description="Factor if scaling applied")
    scaling_count: int = Field(description="Total scaling events")
    decorrelation_count: int = Field(description="Total decorrelation events")
    config: dict = Field(description="Current configuration")


class HomeostaticConfigUpdate(BaseModel):
    """Homeostatic configuration update."""
    target_norm: float | None = Field(default=None, ge=0.5, le=2.0, description="Target L2 norm")
    norm_tolerance: float | None = Field(default=None, ge=0.05, le=0.5, description="Tolerance before scaling")
    ema_alpha: float | None = Field(default=None, ge=0.001, le=0.1, description="EMA rate")
    decorrelation_strength: float | None = Field(default=None, ge=0.0, le=0.1, description="Decorrelation strength")
    sliding_threshold_rate: float | None = Field(default=None, ge=0.0001, le=0.01, description="BCM threshold rate")


@router.get("/bio/homeostatic", response_model=HomeostaticStateResponse)
async def get_homeostatic_state(services: MemoryServices):
    """
    Get homeostatic plasticity state.

    Homeostatic plasticity prevents runaway potentiation by:
    - Tracking running statistics of embedding norms
    - Applying global scaling when norms drift from target
    - Implementing BCM-like sliding threshold for updates
    """
    try:
        episodic = services["episodic"]
        homeostatic = episodic.homeostatic

        state = homeostatic.get_state()
        stats = homeostatic.get_stats()

        return HomeostaticStateResponse(
            mean_norm=state.mean_norm,
            std_norm=state.std_norm,
            mean_activation=state.mean_activation,
            sliding_threshold=state.sliding_threshold,
            last_update=state.last_update.isoformat() if state.last_update else None,
            needs_scaling=homeostatic.needs_scaling(),
            current_scaling_factor=homeostatic.compute_scaling_factor(),
            scaling_count=stats.get("scaling_count", 0),
            decorrelation_count=stats.get("decorrelation_count", 0),
            config=stats.get("config", {}),
        )
    except Exception as e:
        logger.error(f"Failed to get homeostatic state: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Failed to get homeostatic state: {e}"
        )


@router.put("/bio/homeostatic", response_model=HomeostaticStateResponse)
async def update_homeostatic_config(
    update: HomeostaticConfigUpdate,
    services: MemoryServices,
    _: AdminAuth,
):
    """
    Update homeostatic plasticity configuration.

    Requires admin authentication.
    """
    try:
        episodic = services["episodic"]
        homeostatic = episodic.homeostatic

        if update.target_norm is not None:
            homeostatic.set_target_norm(update.target_norm)
        if update.norm_tolerance is not None:
            homeostatic.set_norm_tolerance(update.norm_tolerance)
        if update.ema_alpha is not None:
            homeostatic.set_ema_alpha(update.ema_alpha)
        if update.decorrelation_strength is not None:
            homeostatic.set_decorrelation_strength(update.decorrelation_strength)
        if update.sliding_threshold_rate is not None:
            homeostatic.set_sliding_threshold_rate(update.sliding_threshold_rate)

        # Return updated state
        return await get_homeostatic_state(services)
    except Exception as e:
        logger.error(f"Failed to update homeostatic config: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Failed to update homeostatic config: {e}"
        )


@router.post("/bio/homeostatic/force-scaling")
async def force_homeostatic_scaling(
    services: MemoryServices,
    _: AdminAuth,
):
    """
    Force immediate homeostatic scaling.

    Requires admin authentication.
    """
    try:
        episodic = services["episodic"]
        homeostatic = episodic.homeostatic

        scaling_factor = homeostatic.force_scaling()

        return {
            "success": True,
            "scaling_factor": scaling_factor,
            "message": f"Applied homeostatic scaling with factor {scaling_factor:.4f}",
        }
    except Exception as e:
        logger.error(f"Failed to force homeostatic scaling: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Failed to force scaling: {e}"
        )


# ============================================================================
# ACh Mode Switching Endpoint
# ============================================================================

class AchModeSwitchRequest(BaseModel):
    """Request to force ACh mode."""
    mode: str = Field(description="Target mode: encoding, balanced, or retrieval")


class AchModeSwitchResponse(BaseModel):
    """Response after mode switch."""
    success: bool
    previous_mode: str
    new_mode: str
    ach_level: float
    encoding_weight: float
    retrieval_weight: float


@router.post("/bio/acetylcholine/switch-mode", response_model=AchModeSwitchResponse)
async def switch_ach_mode(
    request: AchModeSwitchRequest,
    services: MemoryServices,
    _: AdminAuth,
):
    """
    Force ACh cognitive mode switch.

    Valid modes:
    - encoding: High ACh, prioritize new information
    - balanced: Moderate ACh, normal operation
    - retrieval: Low ACh, prioritize pattern completion

    Requires admin authentication.
    """
    try:
        episodic = services["episodic"]
        ach = episodic.orchestra.acetylcholine

        # Get previous mode
        previous_mode = ach.get_current_mode().value

        # Force new mode
        if request.mode.lower() not in ["encoding", "balanced", "retrieval"]:
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail=f"Invalid mode: {request.mode}. Must be encoding, balanced, or retrieval"
            )

        new_state = ach.force_mode(request.mode.lower())

        return AchModeSwitchResponse(
            success=True,
            previous_mode=previous_mode,
            new_mode=new_state.mode.value,
            ach_level=new_state.ach_level,
            encoding_weight=new_state.encoding_weight,
            retrieval_weight=new_state.retrieval_weight,
        )
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Failed to switch ACh mode: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Failed to switch ACh mode: {e}"
        )
