# Class Diagrams

UML class diagrams for World Weaver subsystems.

## Memory Subsystem Classes

```mermaid
classDiagram
    class MemoryAPI {
        +session_id: str
        +config: WWConfig
        +store(content, importance, tags) Episode
        +recall(query, limit, memory_type) list~MemoryResult~
        +forget(episode_id) bool
        +consolidate() ConsolidationResult
        +stats() SystemStats
    }

    class Episode {
        +id: UUID
        +content: str
        +embedding: ndarray
        +importance: float
        +created_at: datetime
        +accessed_at: datetime
        +access_count: int
        +outcome: float
        +tags: list~str~
        +metadata: dict
    }

    class EpisodicMemory {
        -storage: StorageBackend
        -embedding_provider: EmbeddingProvider
        +store(episode) str
        +recall(query, limit) list~Episode~
        +update_access(episode_id)
        +get_recent(hours) list~Episode~
        +get_by_importance(threshold) list~Episode~
    }

    class SemanticMemory {
        -neo4j: Neo4jBackend
        -activation_threshold: float
        +store_entity(entity) str
        +recall_entities(query, limit) list~Entity~
        +spread_activation(source, depth) dict
        +get_relationships(entity_id) list~Relationship~
        +update_hebbian(entity_ids)
    }

    class ProceduralMemory {
        -storage: StorageBackend
        +store_procedure(procedure) str
        +recall_procedures(task, limit) list~Procedure~
        +update_success_rate(proc_id, success)
        +get_by_skill(skill_name) list~Procedure~
    }

    class WorkingMemory {
        -buffer: deque
        -capacity: int
        +add(item) bool
        +probe(query) list~BufferItem~
        +evict_oldest()
        +get_pending() list~BufferItem~
        +clear()
    }

    class MemoryResult {
        +episode: Episode
        +score: float
        +memory_type: MemoryType
        +retrieval_path: str
    }

    MemoryAPI --> EpisodicMemory
    MemoryAPI --> SemanticMemory
    MemoryAPI --> ProceduralMemory
    MemoryAPI --> WorkingMemory
    EpisodicMemory --> Episode
    MemoryAPI ..> MemoryResult
```

## Learning Subsystem Classes

```mermaid
classDiagram
    class LearnedMemoryGate {
        -feature_extractor: FeatureExtractor
        -bayesian_lr: BayesianLogisticRegression
        -thompson_sampler: ThompsonSampler
        -evidence_threshold: float
        +evaluate(embedding, content) GateDecision
        +update_on_outcome(episode_id, outcome)
        +get_pending_count() int
        +decay_pending()
    }

    class GateDecision {
        <<enumeration>>
        ACCEPT
        REJECT
        PENDING
    }

    class FeatureExtractor {
        +extract(embedding, content) ndarray
        -compute_novelty(embedding) float
        -compute_complexity(content) float
        -compute_importance(content) float
        -compute_coherence(embedding) float
    }

    class BayesianLogisticRegression {
        -weights_mean: ndarray
        -weights_cov: ndarray
        +predict_proba(features) float
        +update(features, outcome)
        +get_uncertainty() float
    }

    class ThompsonSampler {
        -alpha: ndarray
        -beta: ndarray
        +sample() float
        +update(reward)
        +get_exploitation_rate() float
    }

    class EligibilityTrace {
        -traces: dict~str, float~
        -decay_rate: float
        +tag(synapse_id, strength)
        +decay_all()
        +get_eligible() dict
        +clear()
    }

    class STDPUpdater {
        -tau_plus: float
        -tau_minus: float
        -a_plus: float
        -a_minus: float
        +compute_update(pre_time, post_time) float
        +apply_update(synapse_id, delta_w)
    }

    class HebbianLearning {
        -learning_rate: float
        -decay_rate: float
        +update_on_coactivation(ids: list)
        +get_connection_strength(id1, id2) float
        +prune_weak_connections(threshold)
    }

    class DopamineSystem {
        -baseline: float
        -current_level: float
        -decay_tau: float
        +compute_rpe(actual, expected) float
        +emit_signal(rpe)
        +get_level() float
        +decay()
    }

    LearnedMemoryGate --> FeatureExtractor
    LearnedMemoryGate --> BayesianLogisticRegression
    LearnedMemoryGate --> ThompsonSampler
    LearnedMemoryGate ..> GateDecision
    DopamineSystem --> EligibilityTrace
    DopamineSystem --> STDPUpdater
```

## Storage Subsystem Classes

```mermaid
classDiagram
    class StorageBackend {
        <<abstract>>
        +store(data) str
        +retrieve(id) dict
        +delete(id) bool
        +search(query, limit) list
        +health_check() bool
    }

    class Neo4jBackend {
        -driver: Driver
        -pool_size: int
        +create_node(labels, properties) str
        +create_relationship(from_id, to_id, type) str
        +query(cypher, params) list
        +get_neighbors(node_id, depth) list
        +update_property(node_id, key, value)
    }

    class QdrantBackend {
        -client: QdrantClient
        -collection_name: str
        +upsert(id, vector, payload) str
        +search(vector, limit, filter) list
        +delete(id) bool
        +scroll(filter, limit) list
        +create_collection(config)
    }

    class SagaCoordinator {
        -neo4j: Neo4jBackend
        -qdrant: QdrantBackend
        -state: SagaState
        +begin_transaction() str
        +execute(operations) SagaResult
        +commit(tx_id) bool
        +rollback(tx_id) bool
        +get_state(tx_id) SagaState
    }

    class SagaState {
        <<enumeration>>
        PENDING
        PREPARING
        PREPARED
        EXECUTING
        EXECUTED
        COMMITTING
        COMMITTED
        COMPENSATING
        COMPENSATED
        FAILED
    }

    class CircuitBreaker {
        -state: CircuitState
        -failure_count: int
        -success_count: int
        -failure_threshold: int
        -reset_timeout: float
        +call(operation) Result
        +is_open() bool
        +record_success()
        +record_failure()
        +reset()
    }

    class CircuitState {
        <<enumeration>>
        CLOSED
        OPEN
        HALF_OPEN
    }

    class FallbackCache {
        -lru_cache: LRUCache
        -pending_queue: Queue
        -ttl: float
        +get(key) Optional~Value~
        +put(key, value)
        +queue_pending(operation)
        +drain_pending() list
    }

    StorageBackend <|-- Neo4jBackend
    StorageBackend <|-- QdrantBackend
    SagaCoordinator --> Neo4jBackend
    SagaCoordinator --> QdrantBackend
    SagaCoordinator ..> SagaState
    CircuitBreaker ..> CircuitState
    SagaCoordinator --> CircuitBreaker
    SagaCoordinator --> FallbackCache
```

## Neuromodulation Subsystem Classes

```mermaid
classDiagram
    class NeuromodulatorOrchestra {
        -modulators: dict~str, Neuromodulator~
        -coupling_matrix: ndarray
        +step(dt: float)
        +get_state() NeuroState
        +process_event(event: MemoryEvent)
        +reset_to_baseline()
    }

    class Neuromodulator {
        <<abstract>>
        +name: str
        +baseline: float
        +current: float
        +tau: float
        +step(dt: float)
        +excite(amount: float)
        +inhibit(amount: float)
        +get_normalized() float
    }

    class Dopamine {
        +compute_rpe(actual, expected) float
        +burst()
        +dip()
    }

    class Norepinephrine {
        +arousal_level: ArousalLevel
        +respond_to_novelty(novelty: float)
        +respond_to_threat(threat: float)
    }

    class Acetylcholine {
        +mode: ACHMode
        +set_encoding_mode()
        +set_retrieval_mode()
        +get_plasticity_factor() float
    }

    class Serotonin {
        +eligibility_traces: EligibilityTrace
        +credit_assignment(reward, delay) float
        +update_traces()
    }

    class GABA {
        +inhibition_level: float
        +lateral_inhibition(activations) ndarray
        +winner_take_all(activations, k) ndarray
    }

    class Glutamate {
        +excitation_level: float
        +compute_excitation(input) float
    }

    class NeuroState {
        +dopamine: float
        +norepinephrine: float
        +acetylcholine: float
        +serotonin: float
        +gaba: float
        +glutamate: float
        +timestamp: datetime
    }

    class ArousalLevel {
        <<enumeration>>
        DROWSY
        ALERT
        VIGILANT
        HYPERAROUSED
    }

    class ACHMode {
        <<enumeration>>
        ENCODING
        RETRIEVAL
        BALANCED
    }

    NeuromodulatorOrchestra --> Neuromodulator
    Neuromodulator <|-- Dopamine
    Neuromodulator <|-- Norepinephrine
    Neuromodulator <|-- Acetylcholine
    Neuromodulator <|-- Serotonin
    Neuromodulator <|-- GABA
    Neuromodulator <|-- Glutamate
    NeuromodulatorOrchestra ..> NeuroState
    Norepinephrine ..> ArousalLevel
    Acetylcholine ..> ACHMode
    Serotonin --> EligibilityTrace
```

## NCA Dynamics Classes

```mermaid
classDiagram
    class NCAEngine {
        -field: NeuralField
        -theta_oscillator: ThetaOscillator
        -gamma_oscillator: GammaOscillator
        -spatial_system: SpatialSystem
        -bridge: MemoryBridge
        +step(dt: float)
        +process_memory_event(event)
        +get_state() NCAState
    }

    class NeuralField {
        -state: ndarray
        -coupling: ndarray
        -tau: float
        +integrate(dt: float)
        +set_input(input: ndarray)
        +get_activity() ndarray
        +get_pattern() ndarray
    }

    class ThetaOscillator {
        -frequency: float
        -phase: float
        -amplitude: float
        +step(dt: float)
        +get_phase() float
        +get_amplitude() float
        +reset_phase()
    }

    class GammaOscillator {
        -frequency: float
        -phase: float
        -theta_modulation: float
        +step(dt: float, theta_phase: float)
        +get_phase() float
        +get_coupling_strength() float
    }

    class SpatialSystem {
        -place_cells: PlaceCellLayer
        -grid_cells: GridCellLayer
        -position: ndarray
        +update_position(movement)
        +get_place_activation() ndarray
        +get_grid_activation() ndarray
        +path_integrate(velocity, dt)
    }

    class PlaceCellLayer {
        -centers: ndarray
        -widths: ndarray
        +compute_activation(position) ndarray
        +learn_place(position)
    }

    class GridCellLayer {
        -scales: list~float~
        -orientations: list~float~
        +compute_activation(position) ndarray
        +get_hexagonal_pattern(scale) ndarray
    }

    class MemoryBridge {
        -nca: NCAEngine
        -memory_api: MemoryAPI
        +on_store(episode)
        +on_recall(query, results)
        +modulate_retrieval(nca_state) float
        +get_context_signal() ndarray
    }

    NCAEngine --> NeuralField
    NCAEngine --> ThetaOscillator
    NCAEngine --> GammaOscillator
    NCAEngine --> SpatialSystem
    NCAEngine --> MemoryBridge
    SpatialSystem --> PlaceCellLayer
    SpatialSystem --> GridCellLayer
```

## Prediction Subsystem Classes

```mermaid
classDiagram
    class HierarchicalPredictor {
        -fast_head: PredictionHead
        -medium_head: PredictionHead
        -slow_head: PredictionHead
        -context_encoder: ContextEncoder
        +predict(context) PredictionResult
        +update_on_outcome(predicted, actual)
        +get_prediction_error() float
    }

    class PredictionHead {
        -horizon: int
        -hidden_dim: int
        -model: nn.Module
        +forward(context) ndarray
        +compute_loss(predicted, actual) float
        +update(loss)
    }

    class ContextEncoder {
        -embedding_dim: int
        -sequence_length: int
        +encode(episodes: list) ndarray
        +get_context_vector() ndarray
    }

    class CausalDiscovery {
        -graph: CausalGraph
        -intervention_log: list
        +discover_causes(target) list~Cause~
        +intervene(variable, value) Effect
        +estimate_effect(cause, effect) float
        +get_causal_graph() CausalGraph
    }

    class CausalGraph {
        -nodes: set
        -edges: dict
        +add_edge(cause, effect, strength)
        +get_parents(node) list
        +get_children(node) list
        +topological_sort() list
    }

    class DreamingSystem {
        -predictor: HierarchicalPredictor
        -causal: CausalDiscovery
        -trajectory_buffer: list
        +start_dream(seeds: list)
        +dream_step() DreamState
        +compile_insights() list~Insight~
        +get_trajectories() list~Trajectory~
    }

    class PredictionResult {
        +fast_prediction: ndarray
        +medium_prediction: ndarray
        +slow_prediction: ndarray
        +confidence: float
        +timestamp: datetime
    }

    HierarchicalPredictor --> PredictionHead
    HierarchicalPredictor --> ContextEncoder
    HierarchicalPredictor ..> PredictionResult
    CausalDiscovery --> CausalGraph
    DreamingSystem --> HierarchicalPredictor
    DreamingSystem --> CausalDiscovery
```

## Consolidation Classes

```mermaid
classDiagram
    class ConsolidationCoordinator {
        -nrem_phase: NREMPhase
        -rem_phase: REMPhase
        -pruning_phase: PruningPhase
        -state: ConsolidationState
        +start_consolidation()
        +get_state() ConsolidationState
        +get_stats() ConsolidationStats
    }

    class ConsolidationState {
        <<enumeration>>
        AWAKE
        NREM_EARLY
        NREM_DEEP
        REM
        PRUNING
    }

    class NREMPhase {
        -episodic: EpisodicMemory
        -semantic: SemanticMemory
        -swr_generator: SWRGenerator
        +select_episodes(criteria) list~Episode~
        +replay_sequence(episodes)
        +extract_entities(patterns) list~Entity~
        +transfer_to_semantic(entities)
    }

    class SWRGenerator {
        -compression_ratio: float
        -sequence_threshold: float
        +generate_burst(episodes) SWRBurst
        +temporal_compress(sequence) ndarray
    }

    class REMPhase {
        -clusterer: HDBSCAN
        -abstractor: LLMAbstractor
        +cluster_semantic() list~Cluster~
        +abstract_cluster(cluster) Concept
        +integrate_cross_cluster(clusters) list~Link~
    }

    class PruningPhase {
        -target_activity: float
        -weak_threshold: float
        +identify_weak_connections() list~Connection~
        +scale_weights(factor: float)
        +prune(connections: list)
        +compute_activity_level() float
    }

    class ConsolidationStats {
        +episodes_processed: int
        +entities_created: int
        +concepts_abstracted: int
        +connections_pruned: int
        +connections_strengthened: int
        +duration_seconds: float
    }

    ConsolidationCoordinator --> NREMPhase
    ConsolidationCoordinator --> REMPhase
    ConsolidationCoordinator --> PruningPhase
    ConsolidationCoordinator ..> ConsolidationState
    ConsolidationCoordinator ..> ConsolidationStats
    NREMPhase --> SWRGenerator
```

## Hook System Classes

```mermaid
classDiagram
    class HookRegistry {
        -hooks: dict~HookPoint, list~Hook~~
        +register(hook: Hook)
        +unregister(hook_id: str)
        +execute_pre(operation, context) HookResult
        +execute_on(operation, context) HookResult
        +execute_post(operation, context) HookResult
        +get_hooks(point: HookPoint) list~Hook~
    }

    class Hook {
        <<abstract>>
        +id: str
        +priority: int
        +enabled: bool
        +execute(context: HookContext) HookResult
    }

    class HookPoint {
        <<enumeration>>
        PRE_STORE
        ON_STORE
        POST_STORE
        PRE_RECALL
        ON_RECALL
        POST_RECALL
        PRE_CONSOLIDATE
        POST_CONSOLIDATE
    }

    class HookContext {
        +operation: str
        +session_id: str
        +data: dict
        +metadata: dict
        +timestamp: datetime
    }

    class HookResult {
        +success: bool
        +modified_data: Optional~dict~
        +abort: bool
        +message: str
    }

    class CachingHook {
        -cache: LRUCache
        +execute(context) HookResult
        -check_cache(key) Optional~Value~
        -update_cache(key, value)
    }

    class ValidationHook {
        -validators: list~Validator~
        +execute(context) HookResult
        -validate_content(content) bool
        -validate_importance(importance) bool
    }

    class AuditHook {
        -logger: Logger
        +execute(context) HookResult
        -log_operation(context)
    }

    class HebbianHook {
        -learning: HebbianLearning
        +execute(context) HookResult
        -update_coactivation(ids)
    }

    HookRegistry --> Hook
    Hook ..> HookContext
    Hook ..> HookResult
    Hook <|-- CachingHook
    Hook <|-- ValidationHook
    Hook <|-- AuditHook
    Hook <|-- HebbianHook
    HookRegistry ..> HookPoint
```
