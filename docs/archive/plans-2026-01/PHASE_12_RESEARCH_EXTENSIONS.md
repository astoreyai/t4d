# Phase 12: Research Extensions for World Weaver

**Version**: 0.4.0+ | **Status**: Planning Document
**Author**: Geoffrey Hinton AI Architect Agent
**Date**: 2026-01-04
**Priority**: P2 (Research) | **Timeline**: Q2-Q3 2026

---

## Executive Summary

This document outlines four cutting-edge research extensions for World Weaver that push beyond standard memory system capabilities into novel territory. Each extension is grounded in neural learning principles and designed to integrate with the existing biologically-inspired architecture.

**Key Insight**: The existing infrastructure already implements the *substrate* for these extensions. The dreaming system (P3), hierarchical prediction (P4), and neuromodulator dynamics (P5) provide the foundation. What's needed is the *learning objectives* and *integration patterns* to unlock emergent capabilities.

---

## Table of Contents

1. [Extension 1: Narrative Dreaming with LLM Integration](#extension-1-narrative-dreaming)
2. [Extension 2: Multi-Agent Memory](#extension-2-multi-agent-memory)
3. [Extension 3: Continual Learning Benchmarks](#extension-3-continual-learning-benchmarks)
4. [Extension 4: Embodied Grounding (Future)](#extension-4-embodied-grounding)
5. [Priority Matrix and Dependencies](#priority-matrix-and-dependencies)
6. [Implementation Timeline](#implementation-timeline)

---

## Extension 1: Narrative Dreaming

### Overview

Current dreaming operates in embedding space - generating latent trajectories for consolidation. **Narrative Dreaming** adds a language layer: LLM-generated narratives that explain, elaborate, and connect memory clusters. This is analogous to how human dreams construct narrative meaning from fragmented memory traces.

### Biological Basis

- **Hippocampal Replay + Prefrontal Integration**: During REM, hippocampal traces are replayed while prefrontal cortex constructs narrative coherence (Walker & Stickgold, 2006)
- **Dream Bizarreness as Feature Binding**: Dreams combine disparate memory elements, which may serve associative learning (Hobson, 2009)
- **Schema Consolidation**: Narratives help consolidate episodic memories into semantic schemas (Lewis & Durrant, 2011)

### Architecture

```
                    NARRATIVE DREAMING PIPELINE

    Memory Clusters                   LLM Layer                  Consolidation
    (from DreamingSystem)            (Narrative Gen)            (Schema Update)

    +-----------------+         +-------------------+         +----------------+
    | Cluster 1       |         |                   |         |                |
    | - Episode A     |-------->| Prompt Assembly   |-------->| Narrative      |
    | - Episode B     |         | (structure + ctx) |         | Embedding      |
    | - Episode C     |         |                   |         |                |
    +-----------------+         +--------+----------+         +-------+--------+
                                         |                            |
    +-----------------+                  v                            v
    | Cluster 2       |         +-------------------+         +----------------+
    | - Episode D     |-------->| LLM Generation    |-------->| Schema Node    |
    | - Episode E     |         | (Claude/GPT-4)    |         | Creation       |
    +-----------------+         +--------+----------+         +-------+--------+
                                         |                            |
                                         v                            v
                                +-------------------+         +----------------+
                                | Quality Scoring   |         | Cross-Cluster  |
                                | - Coherence       |         | Linking        |
                                | - Novelty         |         | (Hebbian)      |
                                | - Utility         |         |                |
                                +-------------------+         +----------------+
```

### Implementation Design

#### 1.1 Memory Cluster Extraction

```python
# File: src/t4dm/dreaming/narrative_extractor.py

@dataclass
class MemoryCluster:
    """A cluster of related memories for narrative synthesis."""
    cluster_id: UUID
    episode_ids: list[UUID]
    centroid_embedding: np.ndarray
    theme_keywords: list[str]  # Extracted via TF-IDF or LLM
    temporal_span: tuple[datetime, datetime]
    emotional_valence: float  # Mean valence
    coherence_score: float  # Intra-cluster similarity

@dataclass
class NarrativeContext:
    """Context for LLM narrative generation."""
    clusters: list[MemoryCluster]
    cross_cluster_links: list[tuple[UUID, UUID, float]]  # Links with weight
    temporal_ordering: list[UUID]  # Clusters in time order
    meta_themes: list[str]  # High-level themes across clusters


class ClusterNarrativeExtractor:
    """
    Extract memory clusters suitable for narrative synthesis.

    Uses DreamingSystem's high-quality dreams + HDBSCAN clustering
    to identify coherent memory groups.
    """

    def __init__(
        self,
        dreaming_system: DreamingSystem,
        consolidation_service: ConsolidationService,
        min_cluster_size: int = 3,
        max_clusters_per_cycle: int = 5,
    ):
        self.dreamer = dreaming_system
        self.consolidator = consolidation_service
        self.min_size = min_cluster_size
        self.max_clusters = max_clusters_per_cycle

    async def extract_clusters_for_narrative(
        self,
        episode_embeddings: list[tuple[UUID, np.ndarray]],
        time_window_hours: float = 24.0,
    ) -> list[MemoryCluster]:
        """
        Extract memory clusters suitable for narrative synthesis.

        Strategy:
        1. Filter to recent, high-salience episodes
        2. Cluster using HDBSCAN (from consolidation)
        3. Score clusters by coherence and narrative potential
        4. Return top clusters
        """
        # Step 1: Temporal filtering
        recent = self._filter_recent(episode_embeddings, time_window_hours)

        # Step 2: Cluster using existing infrastructure
        embeddings = [emb for _, emb in recent]
        ids = [id for id, _ in recent]

        # Leverage consolidation's HDBSCAN
        cluster_labels = self.consolidator.cluster_embeddings(embeddings)

        # Step 3: Build cluster objects
        clusters = self._build_clusters(ids, embeddings, cluster_labels)

        # Step 4: Score and filter
        scored = [(c, self._narrative_potential(c)) for c in clusters]
        scored.sort(key=lambda x: x[1], reverse=True)

        return [c for c, _ in scored[:self.max_clusters]]

    def _narrative_potential(self, cluster: MemoryCluster) -> float:
        """
        Score cluster for narrative synthesis potential.

        High potential = coherent + diverse + emotionally salient
        """
        coherence = cluster.coherence_score
        diversity = len(cluster.episode_ids) / 10  # More episodes = more material
        salience = abs(cluster.emotional_valence)  # High valence = important

        # Prefer moderate coherence (too high = redundant, too low = incoherent)
        coherence_factor = 1 - abs(coherence - 0.6)

        return 0.4 * coherence_factor + 0.3 * diversity + 0.3 * salience
```

#### 1.2 LLM Prompt Assembly

```python
# File: src/t4dm/dreaming/narrative_prompt.py

NARRATIVE_SYSTEM_PROMPT = """
You are a memory consolidation system that synthesizes coherent narratives from episodic memories.

Your role is to:
1. Identify the central theme connecting these memory fragments
2. Create a coherent narrative that captures the essential meaning
3. Highlight patterns, insights, or lessons that emerge
4. Note any surprising connections between seemingly unrelated memories

Output Format:
- THEME: One-sentence theme statement
- NARRATIVE: 2-3 paragraph synthesis (200-400 words)
- INSIGHTS: Bullet list of key takeaways
- CONNECTIONS: Relationships to other knowledge domains
- QUESTIONS: Open questions for future exploration
"""

@dataclass
class NarrativePrompt:
    """Assembled prompt for LLM narrative generation."""
    system_prompt: str
    cluster_summaries: list[str]
    temporal_context: str
    cross_references: list[str]
    constraints: dict[str, Any]


class PromptAssembler:
    """
    Assemble prompts for LLM narrative generation.

    Key insight: We're not asking the LLM to remember (it can't) -
    we're asking it to synthesize meaning from our memory traces.
    """

    def __init__(
        self,
        episodic_store: EpisodicMemory,
        max_tokens_per_cluster: int = 500,
    ):
        self.episodic = episodic_store
        self.max_tokens = max_tokens_per_cluster

    async def assemble_prompt(
        self,
        cluster: MemoryCluster,
        related_clusters: list[MemoryCluster] | None = None,
    ) -> NarrativePrompt:
        """
        Assemble a prompt for narrative generation.

        Uses ToonJSON for efficient token usage.
        """
        # Fetch episode content
        episodes = await self.episodic.get_batch(cluster.episode_ids)

        # Convert to ToonJSON for efficiency
        summaries = []
        for ep in episodes:
            toon = get_representation(ep, ToonJSON)
            summaries.append(toon)

        # Build temporal context
        temporal = self._build_temporal_context(cluster)

        # Add cross-references if available
        cross_refs = []
        if related_clusters:
            for rc in related_clusters:
                cross_refs.append(f"Related theme: {', '.join(rc.theme_keywords)}")

        return NarrativePrompt(
            system_prompt=NARRATIVE_SYSTEM_PROMPT,
            cluster_summaries=summaries,
            temporal_context=temporal,
            cross_references=cross_refs,
            constraints={
                "max_narrative_words": 400,
                "require_insights": True,
                "prefer_actionable": True,
            }
        )
```

#### 1.3 Narrative Generation and Consolidation

```python
# File: src/t4dm/dreaming/narrative_generator.py

@dataclass
class GeneratedNarrative:
    """A narrative generated from memory clusters."""
    narrative_id: UUID
    source_cluster_id: UUID
    theme: str
    narrative_text: str
    insights: list[str]
    connections: list[str]
    questions: list[str]
    embedding: np.ndarray  # Embedded narrative for retrieval
    generation_time: datetime
    quality_score: float


class NarrativeGenerator:
    """
    Generate narratives from memory clusters using LLM.

    Integration points:
    - Uses existing EmbeddingService for narrative embedding
    - Stores narratives as semantic entities (schema nodes)
    - Creates Hebbian links to source episodes
    - Triggers during REM phase of SleepConsolidation
    """

    def __init__(
        self,
        llm_provider: LLMProvider,  # Abstract interface to Claude/GPT-4/etc
        embedding_service: EmbeddingService,
        semantic_store: SemanticMemory,
        config: NarrativeConfig | None = None,
    ):
        self.llm = llm_provider
        self.embedder = embedding_service
        self.semantic = semantic_store
        self.config = config or NarrativeConfig()

    async def generate_narrative(
        self,
        cluster: MemoryCluster,
        prompt: NarrativePrompt,
    ) -> GeneratedNarrative:
        """
        Generate a narrative from a memory cluster.

        This is the core LLM integration point.
        """
        # Build the full prompt
        messages = self._build_messages(prompt)

        # Call LLM
        response = await self.llm.generate(
            messages=messages,
            max_tokens=self.config.max_generation_tokens,
            temperature=self.config.generation_temperature,
        )

        # Parse structured output
        parsed = self._parse_narrative_response(response)

        # Embed the narrative for retrieval
        narrative_embedding = await self.embedder.embed(parsed.narrative_text)

        # Score quality
        quality = await self._score_quality(parsed, cluster)

        return GeneratedNarrative(
            narrative_id=uuid4(),
            source_cluster_id=cluster.cluster_id,
            theme=parsed.theme,
            narrative_text=parsed.narrative_text,
            insights=parsed.insights,
            connections=parsed.connections,
            questions=parsed.questions,
            embedding=narrative_embedding,
            generation_time=datetime.now(),
            quality_score=quality,
        )

    async def consolidate_narrative(
        self,
        narrative: GeneratedNarrative,
        source_episodes: list[UUID],
    ) -> UUID:
        """
        Consolidate narrative into semantic memory.

        Creates a schema node and links to source episodes.
        """
        # Create semantic entity for the narrative
        entity = Entity(
            name=narrative.theme,
            entity_type=EntityType.SCHEMA,  # New type for narrative schemas
            summary=narrative.narrative_text[:200],
            details=narrative.narrative_text,
            embedding=narrative.embedding,
            metadata={
                "insights": narrative.insights,
                "connections": narrative.connections,
                "questions": narrative.questions,
                "source_episodes": [str(ep) for ep in source_episodes],
                "quality_score": narrative.quality_score,
            }
        )

        entity_id = await self.semantic.create_entity(entity)

        # Create Hebbian links to source episodes
        for ep_id in source_episodes:
            await self.semantic.create_relationship(
                source_id=str(entity_id),
                target_id=str(ep_id),
                relation_type=RelationType.DERIVED_FROM,
                weight=0.5,  # Initial weight
            )

        return entity_id

    async def _score_quality(
        self,
        parsed: ParsedNarrative,
        cluster: MemoryCluster,
    ) -> float:
        """
        Score narrative quality.

        Metrics:
        1. Coherence: Does narrative address cluster themes?
        2. Novelty: Does it add insight beyond raw episodes?
        3. Utility: Are insights actionable?
        """
        # Semantic similarity to cluster centroid
        narrative_emb = await self.embedder.embed(parsed.narrative_text)
        coherence = float(np.dot(narrative_emb, cluster.centroid_embedding))

        # Novelty: dissimilarity from source summaries
        # (Lower similarity to sources = more synthesis, not just summary)
        # ... implementation details ...

        # Utility: presence of actionable language
        action_words = {"implement", "try", "apply", "use", "consider", "avoid"}
        utility = sum(1 for word in parsed.insights if any(a in word.lower() for a in action_words))
        utility_score = min(utility / 3, 1.0)

        return 0.4 * coherence + 0.3 * novelty_score + 0.3 * utility_score
```

#### 1.4 Integration with Sleep Consolidation

```python
# File: src/t4dm/consolidation/narrative_integration.py

class NarrativeIntegration:
    """
    Integrate narrative dreaming with sleep consolidation.

    Modifies REM phase to include narrative generation.
    """

    def __init__(
        self,
        sleep_consolidation: SleepConsolidation,
        narrative_generator: NarrativeGenerator,
        cluster_extractor: ClusterNarrativeExtractor,
    ):
        self.sleep = sleep_consolidation
        self.narrator = narrative_generator
        self.extractor = cluster_extractor

    async def enhanced_rem_phase(
        self,
        episode_embeddings: list[tuple[UUID, np.ndarray]],
    ) -> NarrativePhaseResult:
        """
        Enhanced REM phase with narrative generation.

        Original REM: Generate dream trajectories, train predictor
        Enhanced REM: + Generate narratives, consolidate schemas
        """
        # Step 1: Run original REM
        original_result = await self.sleep.rem_phase(episode_embeddings)

        # Step 2: Extract clusters for narrative
        clusters = await self.extractor.extract_clusters_for_narrative(
            episode_embeddings
        )

        # Step 3: Generate narratives (if clusters found)
        narratives = []
        for cluster in clusters:
            prompt = await self.extractor.assemble_prompt(cluster)
            narrative = await self.narrator.generate_narrative(cluster, prompt)

            if narrative.quality_score >= self.config.min_quality:
                # Consolidate high-quality narratives
                schema_id = await self.narrator.consolidate_narrative(
                    narrative,
                    cluster.episode_ids,
                )
                narratives.append((narrative, schema_id))

        return NarrativePhaseResult(
            rem_result=original_result,
            clusters_processed=len(clusters),
            narratives_generated=len(narratives),
            schemas_created=[n[1] for n in narratives],
        )
```

### Feasibility Assessment

| Factor | Assessment | Notes |
|--------|------------|-------|
| **Technical Feasibility** | High | LLM integration is well-understood; infrastructure exists |
| **Required Infrastructure** | Medium | Need LLM provider abstraction, prompt engineering |
| **Integration Complexity** | Medium | Hooks into existing dreaming/consolidation |
| **Research Value** | High | Novel contribution: memory narrativization |
| **Implementation Cost** | 2-3 weeks | Core implementation + testing |

### Risks and Mitigations

1. **LLM Hallucination**: Narratives may invent details not in memories
   - Mitigation: Quality scoring includes source fidelity check
   - Mitigation: Narratives marked as "synthesized" not "recalled"

2. **Token Cost**: LLM calls are expensive
   - Mitigation: Use ToonJSON for efficient prompts
   - Mitigation: Rate limit narrative generation (1-2/hour)

3. **Latency**: LLM generation is slow
   - Mitigation: Run during sleep phase (offline)
   - Mitigation: Async batch processing

---

## Extension 2: Multi-Agent Memory

### Overview

Enable multiple agent instances to share, coordinate, and collaboratively consolidate memory while maintaining appropriate isolation and conflict resolution.

### Biological Basis

- **Social Memory**: Humans share memories through language; this extends to AI agents
- **Collective Intelligence**: Ant colonies, bee hives share information for collective benefit
- **Distributed Cognition**: Knowledge exists across individuals and artifacts

### Architecture

```
              MULTI-AGENT MEMORY ARCHITECTURE

    Agent A              Shared Memory Layer              Agent B
    (Session: A)         (Global Namespace)               (Session: B)

    +-------------+      +-------------------+            +-------------+
    | Private     |      |                   |            | Private     |
    | Episodic    |      |   Shared          |            | Episodic    |
    | (my actions)|      |   Semantic        |            | (my actions)|
    +------+------+      |   (facts, skills) |            +------+------+
           |             |                   |                   |
           v             +--------+----------+                   v
    +-------------+               |                       +-------------+
    | Private     |<--------------+-------------->        | Private     |
    | Working     |     Publish/Subscribe                 | Working     |
    | Memory      |     + Conflict Resolution             | Memory      |
    +------+------+               |                       +------+------+
           |             +--------v----------+                   |
           +------------>|   Consensus       |<------------------+
                         |   Layer           |
                         |   (CRDT + Vector  |
                         |    Clock)         |
                         +-------------------+
                                  |
                         +--------v----------+
                         |   Collaborative   |
                         |   Consolidation   |
                         |   (merged dreams) |
                         +-------------------+
```

### Implementation Design

#### 2.1 Agent Identity and Namespacing

```python
# File: src/t4dm/multiagent/identity.py

@dataclass
class AgentIdentity:
    """Unique identity for an agent instance."""
    agent_id: UUID
    agent_type: str  # "claude-code", "research-assistant", etc.
    trust_level: float  # 0.0 to 1.0 (for memory sharing)
    capabilities: set[str]  # What operations this agent can perform
    created_at: datetime
    last_active: datetime


class MemoryNamespace:
    """
    Namespace for memory isolation and sharing.

    Each agent has:
    - Private namespace: Only accessible to this agent
    - Shared namespace: Accessible to trusted agents
    - Global namespace: Accessible to all agents
    """

    PRIVATE = "private"
    SHARED = "shared"
    GLOBAL = "global"

    def __init__(self, agent_id: UUID):
        self.agent_id = agent_id
        self._namespace_prefix = {
            self.PRIVATE: f"agent:{agent_id}:private:",
            self.SHARED: f"agent:{agent_id}:shared:",
            self.GLOBAL: "global:",
        }

    def qualify_key(self, key: str, namespace: str = PRIVATE) -> str:
        """Add namespace prefix to memory key."""
        return self._namespace_prefix[namespace] + key

    def can_access(
        self,
        memory_namespace: str,
        requesting_agent: AgentIdentity,
    ) -> bool:
        """Check if requesting agent can access this namespace."""
        if memory_namespace == self.GLOBAL:
            return True
        if memory_namespace == self.SHARED:
            return requesting_agent.trust_level >= 0.5
        # Private namespace
        return requesting_agent.agent_id == self.agent_id


class AgentRegistry:
    """
    Registry of active agents and their capabilities.
    """

    def __init__(self, storage: KeyValueStore):
        self.storage = storage
        self._active_agents: dict[UUID, AgentIdentity] = {}
        self._capabilities_index: dict[str, set[UUID]] = {}

    async def register(self, agent: AgentIdentity) -> None:
        """Register a new agent."""
        self._active_agents[agent.agent_id] = agent
        for cap in agent.capabilities:
            self._capabilities_index.setdefault(cap, set()).add(agent.agent_id)
        await self._persist_agent(agent)

    async def find_agents_with_capability(
        self,
        capability: str,
    ) -> list[AgentIdentity]:
        """Find agents with a specific capability."""
        agent_ids = self._capabilities_index.get(capability, set())
        return [self._active_agents[aid] for aid in agent_ids if aid in self._active_agents]

    async def get_trusted_peers(
        self,
        agent_id: UUID,
        min_trust: float = 0.5,
    ) -> list[AgentIdentity]:
        """Get agents trusted by the specified agent."""
        return [
            a for a in self._active_agents.values()
            if a.agent_id != agent_id and a.trust_level >= min_trust
        ]
```

#### 2.2 Shared Memory Protocol

```python
# File: src/t4dm/multiagent/shared_memory.py

@dataclass
class MemoryPublication:
    """A memory item published to shared namespace."""
    memory_id: UUID
    memory_type: MemoryType  # episodic, semantic, procedural
    publisher_id: UUID
    namespace: str
    content_hash: str  # For deduplication
    embedding: np.ndarray
    metadata: dict[str, Any]
    timestamp: datetime
    version: int
    vector_clock: dict[str, int]  # For CRDT conflict resolution


class SharedMemoryBroker:
    """
    Broker for multi-agent memory sharing.

    Implements publish-subscribe pattern with CRDT conflict resolution.
    """

    def __init__(
        self,
        registry: AgentRegistry,
        semantic_store: SemanticMemory,
        vector_store: QdrantStore,
    ):
        self.registry = registry
        self.semantic = semantic_store
        self.vector = vector_store
        self._subscriptions: dict[UUID, set[str]] = {}  # agent_id -> topics
        self._publications: dict[str, list[MemoryPublication]] = {}

    async def publish(
        self,
        agent: AgentIdentity,
        memory: MemoryItem,
        namespace: str = MemoryNamespace.SHARED,
    ) -> MemoryPublication:
        """
        Publish a memory to shared namespace.

        Steps:
        1. Validate agent can publish to namespace
        2. Check for duplicates (content hash)
        3. Resolve conflicts if updating existing
        4. Notify subscribed agents
        """
        # Validate
        if namespace == MemoryNamespace.PRIVATE:
            raise ValueError("Cannot publish to private namespace")

        if namespace == MemoryNamespace.GLOBAL and agent.trust_level < 0.8:
            raise PermissionError("Insufficient trust for global publication")

        # Check for duplicate
        existing = await self._find_by_content_hash(memory.content_hash)

        if existing:
            # Merge using CRDT
            merged = await self._crdt_merge(existing, memory, agent)
            publication = await self._update_publication(merged)
        else:
            publication = await self._create_publication(memory, agent, namespace)

        # Store in shared layer
        await self._store_shared(publication)

        # Notify subscribers
        await self._notify_subscribers(publication, namespace)

        return publication

    async def subscribe(
        self,
        agent: AgentIdentity,
        topics: list[str],
    ) -> None:
        """Subscribe agent to memory topics."""
        self._subscriptions.setdefault(agent.agent_id, set()).update(topics)

    async def query_shared(
        self,
        agent: AgentIdentity,
        query_embedding: np.ndarray,
        namespace: str = MemoryNamespace.SHARED,
        limit: int = 10,
    ) -> list[MemoryPublication]:
        """
        Query shared memory.

        Respects namespace access rules.
        """
        # Vector search in shared namespace
        results = await self.vector.search(
            query_embedding,
            namespace=namespace,
            limit=limit,
        )

        # Filter by access permissions
        accessible = []
        for pub in results:
            ns = MemoryNamespace(pub.publisher_id)
            if ns.can_access(pub.namespace, agent):
                accessible.append(pub)

        return accessible


class CRDTResolver:
    """
    Conflict-free Replicated Data Type resolver.

    Uses OR-Set for conflict resolution with vector clocks for causality.
    """

    def merge(
        self,
        local: MemoryPublication,
        remote: MemoryPublication,
    ) -> MemoryPublication:
        """
        Merge conflicting publications.

        Strategy:
        1. Merge vector clocks
        2. For semantic content: keep higher version
        3. For embeddings: weighted average based on trust
        4. For metadata: union with preference for newer
        """
        # Merge vector clocks
        merged_clock = {}
        for agent_id in set(local.vector_clock.keys()) | set(remote.vector_clock.keys()):
            merged_clock[agent_id] = max(
                local.vector_clock.get(agent_id, 0),
                remote.vector_clock.get(agent_id, 0),
            )

        # Determine which version is "newer"
        if self._happens_before(local.vector_clock, remote.vector_clock):
            base = remote
        elif self._happens_before(remote.vector_clock, local.vector_clock):
            base = local
        else:
            # Concurrent updates - merge content
            base = self._merge_concurrent(local, remote)

        base.vector_clock = merged_clock
        base.version = max(local.version, remote.version) + 1

        return base
```

#### 2.3 Collaborative Consolidation

```python
# File: src/t4dm/multiagent/collaborative_consolidation.py

class CollaborativeConsolidation:
    """
    Coordinate memory consolidation across multiple agents.

    Key insight: Different agents see different parts of the world.
    Merging their memories creates more complete knowledge.
    """

    def __init__(
        self,
        broker: SharedMemoryBroker,
        consolidation_service: ConsolidationService,
        dreaming_system: DreamingSystem,
    ):
        self.broker = broker
        self.consolidator = consolidation_service
        self.dreamer = dreaming_system

    async def collaborative_sleep_cycle(
        self,
        participating_agents: list[AgentIdentity],
    ) -> CollaborativeConsolidationResult:
        """
        Run collaborative consolidation across agents.

        Steps:
        1. Collect recent memories from all agents
        2. Identify cross-agent patterns (shared themes)
        3. Generate merged dream trajectories
        4. Create shared semantic entities
        5. Distribute back to agent namespaces
        """
        # Step 1: Collect memories
        all_memories = []
        for agent in participating_agents:
            agent_memories = await self._collect_agent_memories(agent)
            all_memories.extend(agent_memories)

        # Step 2: Cross-agent clustering
        clusters = await self._cross_agent_clustering(all_memories)

        # Step 3: Merged dreaming
        # Use weighted embeddings based on agent trust
        merged_dreams = []
        for cluster in clusters:
            if self._is_cross_agent(cluster):
                dream = await self._merged_dream(cluster)
                merged_dreams.append(dream)

        # Step 4: Create shared entities
        shared_entities = []
        for dream in merged_dreams:
            if dream.quality_score >= 0.5:
                entity = await self._create_shared_entity(dream)
                shared_entities.append(entity)

        # Step 5: Distribute
        for entity in shared_entities:
            await self.broker.publish(
                agent=AgentIdentity.SYSTEM,  # System-level publication
                memory=entity,
                namespace=MemoryNamespace.GLOBAL,
            )

        return CollaborativeConsolidationResult(
            agents_participated=len(participating_agents),
            total_memories=len(all_memories),
            cross_agent_clusters=len([c for c in clusters if self._is_cross_agent(c)]),
            dreams_generated=len(merged_dreams),
            shared_entities_created=len(shared_entities),
        )

    async def _merged_dream(
        self,
        cluster: CrossAgentCluster,
    ) -> DreamTrajectory:
        """
        Generate dream from cross-agent cluster.

        Weighted average of agent perspectives, trust-adjusted.
        """
        # Compute weighted centroid
        weights = []
        embeddings = []
        for memory, agent in cluster.memories_with_agents:
            weights.append(agent.trust_level)
            embeddings.append(memory.embedding)

        weights = np.array(weights)
        weights /= weights.sum()

        weighted_centroid = np.sum(
            [w * e for w, e in zip(weights, embeddings)],
            axis=0,
        )

        # Dream from weighted centroid
        trajectory = self.dreamer.dream(
            seed_embedding=weighted_centroid,
            context_embeddings=embeddings,
        )

        return trajectory
```

### Feasibility Assessment

| Factor | Assessment | Notes |
|--------|------------|-------|
| **Technical Feasibility** | High | CRDT patterns well-established |
| **Required Infrastructure** | High | Significant new infrastructure needed |
| **Integration Complexity** | High | Changes to storage, retrieval, consolidation |
| **Research Value** | High | Novel: biologically-inspired multi-agent memory |
| **Implementation Cost** | 4-6 weeks | Core + integration + testing |

### Risks and Mitigations

1. **Consensus Overhead**: CRDT sync can be expensive
   - Mitigation: Lazy synchronization during idle periods
   - Mitigation: Hierarchical resolution (local first, then global)

2. **Trust Exploitation**: Malicious agent pollutes shared memory
   - Mitigation: Reputation system based on memory utility
   - Mitigation: Quarantine for new/untrusted agents

3. **Privacy Leakage**: Shared memories expose private information
   - Mitigation: Explicit opt-in for sharing
   - Mitigation: Differential privacy for aggregated knowledge

---

## Extension 3: Continual Learning Benchmarks

### Overview

Integrate standard continual learning benchmarks to validate catastrophic forgetting mitigation and measure learning-without-forgetting capabilities.

### Scientific Context

World Weaver implements multiple anti-forgetting mechanisms:
- **Complementary Learning Systems**: Fast hippocampal + slow neocortical
- **Elastic Weight Consolidation (EWC)**: Fisher information regularization
- **Generative Replay**: Wake-sleep style pseudo-rehearsal
- **Interleaved Replay**: Mixing old and new during consolidation

We need rigorous benchmarks to validate these mechanisms.

### Benchmark Suite

#### 3.1 CLEAR Benchmark Integration

```python
# File: src/t4dm/benchmarks/clear_benchmark.py

"""
CLEAR Benchmark (Continual LEArning on Real-World Imagery)

10 tasks from time-ordered ImageNet subsets (2004-2014)
Tests: forward transfer, backward transfer, forgetting

Integration Strategy:
- Use CLIP embeddings as memory content
- Measure retrieval accuracy across tasks
- Track semantic memory growth patterns
"""

@dataclass
class CLEARTask:
    """A single CLEAR benchmark task."""
    task_id: int
    year: int
    class_names: list[str]
    image_embeddings: np.ndarray  # CLIP embeddings
    labels: np.ndarray


class CLEARBenchmark:
    """
    CLEAR benchmark integration for World Weaver.

    Measures:
    1. Forward Transfer (FWT): Does learning task T help with T+1?
    2. Backward Transfer (BWT): Does learning T+1 hurt recall of T?
    3. Forgetting: Difference between peak and final accuracy
    4. Cumulative Accuracy: Average across all seen tasks
    """

    def __init__(
        self,
        memory_service: UnifiedMemoryService,
        embedding_service: EmbeddingService,
        consolidation_service: ConsolidationService,
    ):
        self.memory = memory_service
        self.embedder = embedding_service
        self.consolidator = consolidation_service

        self.task_accuracies: dict[int, list[float]] = {}  # task -> accuracy over time

    async def run_benchmark(
        self,
        tasks: list[CLEARTask],
        consolidation_after_each: bool = True,
    ) -> CLEARResults:
        """
        Run full CLEAR benchmark.

        For each task:
        1. Store task examples as episodes
        2. Run consolidation (optional)
        3. Test retrieval on all seen tasks
        4. Record accuracy matrix
        """
        results = []

        for task_idx, task in enumerate(tasks):
            # Learn current task
            await self._learn_task(task)

            # Consolidate
            if consolidation_after_each:
                await self.consolidator.run_sleep_cycle()

            # Test all tasks seen so far
            task_results = {}
            for prev_task in tasks[:task_idx + 1]:
                accuracy = await self._test_task(prev_task)
                task_results[prev_task.task_id] = accuracy

                # Track history
                self.task_accuracies.setdefault(prev_task.task_id, []).append(accuracy)

            results.append(task_results)

        return self._compute_metrics(results)

    async def _learn_task(self, task: CLEARTask) -> None:
        """Store task examples as episodic memories."""
        for i, (embedding, label) in enumerate(zip(task.image_embeddings, task.labels)):
            episode = Episode(
                content=f"Image {i} from task {task.task_id} (year {task.year})",
                embedding=embedding,
                context={"task_id": task.task_id, "label": int(label)},
                emotional_valence=0.5,  # Neutral importance
            )
            await self.memory.episodic.store(episode)

    async def _test_task(self, task: CLEARTask) -> float:
        """Test retrieval accuracy on a task."""
        correct = 0
        total = 0

        for embedding, label in zip(task.image_embeddings, task.labels):
            results = await self.memory.episodic.recall(
                query_embedding=embedding,
                limit=1,
            )

            if results and results[0].context.get("label") == label:
                correct += 1
            total += 1

        return correct / max(total, 1)

    def _compute_metrics(self, results: list[dict]) -> CLEARResults:
        """Compute standard continual learning metrics."""
        n_tasks = len(results)

        # Accuracy matrix: A[i,j] = accuracy on task j after learning task i
        A = np.zeros((n_tasks, n_tasks))
        for i, task_results in enumerate(results):
            for j, (task_id, acc) in enumerate(sorted(task_results.items())):
                A[i, j] = acc

        # Forward Transfer
        fwt = 0.0
        for j in range(1, n_tasks):
            # Random baseline
            random_acc = 1.0 / (j + 1)  # Assume uniform classes
            fwt += A[j-1, j] - random_acc
        fwt /= (n_tasks - 1)

        # Backward Transfer (negative = forgetting)
        bwt = 0.0
        for j in range(n_tasks - 1):
            bwt += A[n_tasks-1, j] - A[j, j]  # Final - peak
        bwt /= (n_tasks - 1)

        # Forgetting
        forgetting = {}
        for task_id, history in self.task_accuracies.items():
            if len(history) > 1:
                forgetting[task_id] = max(history) - history[-1]

        # Cumulative Accuracy (last row average)
        cumulative = np.mean(A[-1, :])

        return CLEARResults(
            accuracy_matrix=A,
            forward_transfer=fwt,
            backward_transfer=bwt,
            forgetting=forgetting,
            cumulative_accuracy=cumulative,
        )
```

#### 3.2 Split-MNIST/CIFAR Evaluation

```python
# File: src/t4dm/benchmarks/split_benchmarks.py

"""
Split-MNIST and Split-CIFAR benchmarks.

Standard continual learning benchmarks:
- Split-MNIST: 5 tasks of 2 digits each
- Split-CIFAR10: 5 tasks of 2 classes each
- Split-CIFAR100: 10 tasks of 10 classes each

These are simpler than CLEAR but more widely reported.
"""

@dataclass
class SplitBenchmarkConfig:
    """Configuration for split benchmarks."""
    dataset: str  # "mnist", "cifar10", "cifar100"
    n_tasks: int
    classes_per_task: int
    embedding_model: str  # "clip", "resnet", "vit"
    use_replay: bool = True
    use_ewc: bool = True
    replay_buffer_size: int = 1000


class SplitBenchmark:
    """
    Split benchmark implementation.

    Compares World Weaver's anti-forgetting mechanisms against:
    - Naive (no protection)
    - Replay only
    - EWC only
    - Full WW stack (replay + EWC + consolidation)
    """

    async def run_ablation_study(
        self,
        config: SplitBenchmarkConfig,
    ) -> AblationResults:
        """
        Run ablation study to isolate mechanism contributions.
        """
        conditions = [
            ("naive", {"use_replay": False, "use_ewc": False, "consolidate": False}),
            ("replay_only", {"use_replay": True, "use_ewc": False, "consolidate": False}),
            ("ewc_only", {"use_replay": False, "use_ewc": True, "consolidate": False}),
            ("consolidation_only", {"use_replay": False, "use_ewc": False, "consolidate": True}),
            ("full_stack", {"use_replay": True, "use_ewc": True, "consolidate": True}),
        ]

        results = {}
        for condition_name, settings in conditions:
            memory = self._create_memory_system(**settings)
            benchmark = CLEARBenchmark(memory, self.embedder, self.consolidator)

            tasks = self._load_split_tasks(config)
            condition_results = await benchmark.run_benchmark(tasks)
            results[condition_name] = condition_results

        return AblationResults(
            conditions=results,
            relative_improvements=self._compute_relative_improvements(results),
        )
```

#### 3.3 Catastrophic Forgetting Metrics

```python
# File: src/t4dm/benchmarks/forgetting_metrics.py

"""
Comprehensive forgetting metrics for memory system evaluation.

Beyond standard CL metrics, we measure:
1. Semantic Stability: Do entity meanings drift?
2. Graph Coherence: Do relationship patterns degrade?
3. Retrieval Drift: Does ranking stability decrease?
4. Consolidation Quality: Does abstraction improve over time?
"""

@dataclass
class ForgettingMetrics:
    """Comprehensive forgetting metrics."""

    # Standard CL metrics
    accuracy_retention: float  # Final / Peak accuracy
    backward_transfer: float  # Negative = forgetting
    forward_transfer: float  # Positive = transfer

    # World Weaver specific
    semantic_stability: float  # Entity embedding drift
    graph_coherence: float  # Relationship pattern stability
    retrieval_drift: float  # Ranking stability over time
    consolidation_quality: float  # Schema abstraction quality


class ForgettingAnalyzer:
    """
    Analyze forgetting patterns in World Weaver memory.
    """

    async def analyze_semantic_stability(
        self,
        entity_id: UUID,
        history_window: int = 100,
    ) -> float:
        """
        Measure semantic stability of an entity.

        Tracks embedding drift over consolidation cycles.
        """
        history = await self.semantic_store.get_embedding_history(
            entity_id,
            limit=history_window,
        )

        if len(history) < 2:
            return 1.0  # No drift if no history

        # Compute average pairwise similarity
        drifts = []
        for i in range(1, len(history)):
            sim = float(np.dot(history[i], history[i-1]))
            drifts.append(1 - sim)  # Drift = 1 - similarity

        # Stability = 1 - average drift
        return 1.0 - np.mean(drifts)

    async def analyze_graph_coherence(
        self,
        sample_size: int = 100,
    ) -> float:
        """
        Measure knowledge graph coherence.

        Checks if relationships still make semantic sense.
        """
        relationships = await self.semantic_store.sample_relationships(sample_size)

        coherent = 0
        for rel in relationships:
            source_emb = await self.semantic_store.get_embedding(rel.source_id)
            target_emb = await self.semantic_store.get_embedding(rel.target_id)

            # Relationship should imply some similarity
            similarity = float(np.dot(source_emb, target_emb))

            # Weighted by relationship strength
            if similarity > 0.3 * rel.weight:  # Threshold scaled by weight
                coherent += 1

        return coherent / max(sample_size, 1)

    async def analyze_retrieval_drift(
        self,
        test_queries: list[np.ndarray],
        check_interval_episodes: int = 1000,
    ) -> float:
        """
        Measure retrieval ranking stability.

        Periodic queries should return consistent rankings.
        """
        baseline_rankings = {}

        # Get baseline rankings
        for i, query in enumerate(test_queries):
            results = await self.memory.recall(query_embedding=query, limit=10)
            baseline_rankings[i] = [r.item.id for r in results]

        # Check rankings after interval
        current_rankings = {}
        for i, query in enumerate(test_queries):
            results = await self.memory.recall(query_embedding=query, limit=10)
            current_rankings[i] = [r.item.id for r in results]

        # Compute rank correlation (Kendall tau)
        correlations = []
        for i in range(len(test_queries)):
            tau = self._kendall_tau(baseline_rankings[i], current_rankings[i])
            correlations.append(tau)

        return np.mean(correlations)
```

### Feasibility Assessment

| Factor | Assessment | Notes |
|--------|------------|-------|
| **Technical Feasibility** | Very High | Standard benchmarks with established implementations |
| **Required Infrastructure** | Low | Just need benchmark data and metrics |
| **Integration Complexity** | Low | Minimal changes to core system |
| **Research Value** | Very High | Required for publication credibility |
| **Implementation Cost** | 1-2 weeks | Straightforward implementation |

---

## Extension 4: Embodied Grounding (Future)

### Overview

Connect World Weaver to sensorimotor simulation for embodied memory grounding. This is the most speculative extension but potentially the most impactful.

### Rationale

Current memories are purely symbolic/embedding-based. Embodied grounding would:
- Connect memories to simulated sensorimotor experience
- Enable procedural memories grounded in action
- Support simulation-based retrieval ("imagine doing X")

### Preliminary Architecture

```
                    EMBODIED GROUNDING ARCHITECTURE

    World Weaver Memory          Sensorimotor Layer           Simulation

    +-----------------+         +-------------------+         +----------+
    | Procedural      |<------->| Action Encoder    |<------->| MuJoCo   |
    | Memory          |         | (proprioception)  |         | IsaacGym |
    +-----------------+         +-------------------+         | PyBullet |
                                                              +----------+
    +-----------------+         +-------------------+
    | Episodic        |<------->| Scene Encoder     |<------->   VLM
    | Memory          |         | (visual grounding)|         (Vision)
    +-----------------+         +-------------------+

    +-----------------+         +-------------------+
    | Semantic        |<------->| Concept Grounding |<------->  Language
    | Memory          |         | (symbol→sensory)  |          Model
    +-----------------+         +-------------------+
```

### Interface Specification

```python
# File: src/t4dm/embodied/interfaces.py (STUB - Future Work)

class EmbodiedInterface(Protocol):
    """Protocol for embodied grounding interfaces."""

    async def ground_action(
        self,
        action_description: str,
        context: SimulationContext,
    ) -> ActionGrounding:
        """Ground action description in sensorimotor simulation."""
        ...

    async def simulate_episode(
        self,
        episode_id: UUID,
        duration_steps: int,
    ) -> SimulatedExperience:
        """Re-simulate an episodic memory in embodied context."""
        ...

    async def retrieve_by_sensation(
        self,
        sensory_query: SensoryPattern,
    ) -> list[ScoredResult]:
        """Retrieve memories matching sensory pattern."""
        ...


@dataclass
class SimulationContext:
    """Context for embodied simulation."""
    environment: str  # "kitchen", "office", "outdoor"
    agent_type: str  # "humanoid", "robot_arm", "vehicle"
    physics_params: dict[str, float]
    initial_state: np.ndarray


@dataclass
class ActionGrounding:
    """Grounded action representation."""
    action_id: UUID
    action_description: str
    motor_commands: np.ndarray  # Sequence of motor commands
    proprioceptive_feedback: np.ndarray  # Expected feedback
    visual_sequence: list[np.ndarray]  # Expected visual frames
    success_criteria: Callable[[SimState], bool]
```

### Feasibility Assessment

| Factor | Assessment | Notes |
|--------|------------|-------|
| **Technical Feasibility** | Medium | Requires simulation integration |
| **Required Infrastructure** | Very High | Physics engine, rendering, VLM |
| **Integration Complexity** | Very High | Major architectural changes |
| **Research Value** | Very High | Novel: embodied memory for AI agents |
| **Implementation Cost** | 3-6 months | Major research project |

### Recommendation

**Defer to Phase 15+**. The infrastructure requirements are substantial, and the other extensions provide more immediate value. However, maintain interface specifications for future compatibility.

---

## Priority Matrix and Dependencies

### Priority Ranking

| Extension | Priority | Effort | Impact | Dependencies |
|-----------|----------|--------|--------|--------------|
| **E3: Continual Learning Benchmarks** | P1 | Low | High | None |
| **E1: Narrative Dreaming** | P2 | Medium | High | Dreaming (P3) |
| **E2: Multi-Agent Memory** | P2 | High | Very High | Storage, Consolidation |
| **E4: Embodied Grounding** | P3 | Very High | Very High | E1, E2 |

### Dependency Graph

```
                         DEPENDENCY GRAPH

    [E3: Benchmarks] -----> No dependencies (START HERE)
           |
           v
    [E1: Narrative] -----> Dreaming System (P3) [COMPLETE]
           |               Consolidation Service [COMPLETE]
           |               LLM Provider [NEW]
           |
           v
    [E2: Multi-Agent] ---> Storage Layer [COMPLETE]
           |               Session Isolation [COMPLETE]
           |               CRDT Implementation [NEW]
           |
           v
    [E4: Embodied] ------> E1, E2 [PARTIAL]
                           Simulation Engine [NEW]
                           VLM Integration [NEW]
```

### Recommended Implementation Order

1. **E3: Continual Learning Benchmarks** (Week 1-2)
   - No dependencies
   - Provides metrics for validating other extensions
   - Required for publication credibility

2. **E1: Narrative Dreaming** (Week 3-5)
   - Builds on existing dreaming infrastructure
   - Moderate complexity
   - High research novelty

3. **E2: Multi-Agent Memory** (Week 6-11)
   - Complex but high impact
   - Enables collaborative learning scenarios
   - Foundation for production multi-agent systems

4. **E4: Embodied Grounding** (Q3+ 2026)
   - Defer until E1-E3 stable
   - Maintain interface compatibility

---

## Implementation Timeline

### Q2 2026 (April - June)

**Sprint 12: Continual Learning Benchmarks** (2 weeks)
- Week 1: CLEAR benchmark integration
- Week 2: Split-MNIST/CIFAR + forgetting metrics

**Sprint 13: Narrative Dreaming** (3 weeks)
- Week 1: Cluster extraction + prompt assembly
- Week 2: LLM integration + generation
- Week 3: Consolidation integration + testing

**Sprint 14: Multi-Agent Memory (Part 1)** (3 weeks)
- Week 1: Agent identity + namespacing
- Week 2: Shared memory broker + CRDT
- Week 3: Publishing + subscription

### Q3 2026 (July - September)

**Sprint 15: Multi-Agent Memory (Part 2)** (3 weeks)
- Week 1: Collaborative consolidation
- Week 2: Trust + reputation system
- Week 3: Integration testing + benchmarks

**Sprint 16: Embodied Grounding (Interface Design)** (1 week)
- Define interfaces only
- No implementation

**Sprint 17: Validation and Publication** (2 weeks)
- Run full benchmark suite
- Document findings
- Prepare technical report

---

## Files to Create

### Phase 12 Files

```
src/t4dm/
├── benchmarks/
│   ├── __init__.py
│   ├── clear_benchmark.py          # CLEAR integration
│   ├── split_benchmarks.py         # Split-MNIST/CIFAR
│   ├── forgetting_metrics.py       # Comprehensive metrics
│   └── benchmark_runner.py         # CLI for running benchmarks
├── dreaming/
│   ├── narrative_extractor.py      # Cluster extraction
│   ├── narrative_prompt.py         # Prompt assembly
│   ├── narrative_generator.py      # LLM generation
│   └── narrative_integration.py    # Sleep integration
├── multiagent/
│   ├── __init__.py
│   ├── identity.py                 # Agent identity
│   ├── namespace.py                # Memory namespacing
│   ├── shared_memory.py            # Shared broker + CRDT
│   ├── collaborative.py            # Collaborative consolidation
│   └── trust.py                    # Trust/reputation system
└── embodied/
    ├── __init__.py
    └── interfaces.py               # Stub interfaces only

tests/
├── benchmarks/
│   ├── test_clear.py
│   ├── test_split.py
│   └── test_forgetting_metrics.py
├── dreaming/
│   ├── test_narrative_extractor.py
│   ├── test_narrative_generator.py
│   └── test_narrative_integration.py
└── multiagent/
    ├── test_identity.py
    ├── test_shared_memory.py
    ├── test_crdt.py
    └── test_collaborative.py
```

---

## Research Contribution Summary

### Novel Contributions

1. **Narrative Dreaming**: First system to use LLMs for dream-like narrative synthesis in AI memory
   - Connects hippocampal replay with language-mediated consolidation
   - Publishable as: "Narrative-Guided Memory Consolidation in AI Systems"

2. **Multi-Agent Memory with CRDT**: First biologically-inspired distributed memory system
   - Combines CLS theory with distributed systems
   - Publishable as: "Collective Memory in Multi-Agent AI: A Complementary Learning Systems Approach"

3. **Comprehensive Forgetting Analysis**: Goes beyond standard CL metrics
   - Semantic stability, graph coherence, retrieval drift
   - Useful for the broader CL community

### Publication Venues

- **NeurIPS 2026**: E1 (Narrative Dreaming) - cognitive science track
- **ICML 2026**: E2 (Multi-Agent) - multi-agent learning track
- **ICLR 2027**: Full system paper with all extensions
- **Nature Machine Intelligence**: If results are strong

---

## Appendix: LLM Provider Interface

```python
# File: src/t4dm/llm/provider.py

class LLMProvider(Protocol):
    """Abstract interface for LLM providers."""

    async def generate(
        self,
        messages: list[dict[str, str]],
        max_tokens: int = 1000,
        temperature: float = 0.7,
        stop: list[str] | None = None,
    ) -> str:
        """Generate text completion."""
        ...

    async def embed(
        self,
        text: str,
    ) -> np.ndarray:
        """Get embedding for text."""
        ...


class ClaudeProvider(LLMProvider):
    """Claude API provider."""

    def __init__(self, api_key: str, model: str = "claude-sonnet-4-20250514"):
        self.client = anthropic.Anthropic(api_key=api_key)
        self.model = model

    async def generate(self, messages, max_tokens=1000, temperature=0.7, stop=None):
        response = await self.client.messages.create(
            model=self.model,
            max_tokens=max_tokens,
            temperature=temperature,
            messages=messages,
        )
        return response.content[0].text


class OpenAIProvider(LLMProvider):
    """OpenAI API provider."""

    def __init__(self, api_key: str, model: str = "gpt-4o"):
        self.client = openai.AsyncOpenAI(api_key=api_key)
        self.model = model

    async def generate(self, messages, max_tokens=1000, temperature=0.7, stop=None):
        response = await self.client.chat.completions.create(
            model=self.model,
            messages=messages,
            max_tokens=max_tokens,
            temperature=temperature,
            stop=stop,
        )
        return response.choices[0].message.content
```

---

**End of Phase 12 Research Extensions Plan**

*This document represents cutting-edge research directions. Implementation should proceed incrementally with validation at each stage.*

*Last Updated: 2026-01-04*
