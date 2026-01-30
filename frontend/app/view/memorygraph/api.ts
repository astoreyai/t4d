/**
 * API client for World Weaver Memory Graph visualization.
 *
 * Connects to the WW REST API for memory graph data.
 */

import type { MemoryNode, MemoryEdge, MemoryType, EdgeType } from "./memorygraph-types";
import { Vector3 } from "three";

// ============================================================================
// Configuration
// ============================================================================

const API_BASE_URL = process.env.NEXT_PUBLIC_WW_API_URL || "http://localhost:8765/api/v1";

// ============================================================================
// Response Types (matching backend)
// ============================================================================

interface Position3D {
    x: number;
    y: number;
    z: number;
}

interface NodeMetadataResponse {
    created_at: number;
    last_accessed: number;
    access_count: number;
    importance: number;
    tags?: string[];
    source?: string;
}

interface MemoryNodeResponse {
    id: string;
    type: MemoryType;
    content: string;
    metadata: NodeMetadataResponse;
    position?: Position3D;
}

interface EdgeMetadataResponse {
    created_at: number;
    last_activated?: number;
}

interface MemoryEdgeResponse {
    id: string;
    source: string;
    target: string;
    type: EdgeType;
    weight: number;
    metadata?: EdgeMetadataResponse;
}

interface GraphResponse {
    nodes: MemoryNodeResponse[];
    edges: MemoryEdgeResponse[];
    metrics: Record<string, unknown>;
}

interface TimelineEvent {
    id: string;
    type: MemoryType;
    timestamp: number;
    content: string;
    importance: number;
}

interface TimelineResponse {
    events: TimelineEvent[];
    start_time: number;
    end_time: number;
    total_events: number;
}

interface ActivityMetrics {
    id: string;
    type: MemoryType;
    activation: number;
    recency: number;
    frequency: number;
    last_accessed: number;
}

interface ActivityResponse {
    memories: ActivityMetrics[];
    most_active: string[];
    recently_created: string[];
    recently_accessed: string[];
}

// Biological mechanism types
interface FSRSState {
    memory_id: string;
    stability: number;
    difficulty: number;
    retrievability: number;
    last_review: number;
    next_review: number;
    review_count: number;
    decay_curve: [number, number][];
}

interface HebbianWeight {
    source_id: string;
    target_id: string;
    weight: number;
    weight_history: [number, number][];
    co_activation_count: number;
    last_potentiation?: number;
    last_depression?: number;
    eligibility_trace: number;
}

interface ActivationSpread {
    source_id: string;
    target_id: string;
    base_level: number;
    spreading_activation: number;
    total_activation: number;
    decay_rate: number;
    time_since_activation: number;
}

interface SleepConsolidationState {
    is_active: boolean;
    current_phase?: string;
    phase_progress: number;
    replays_completed: number;
    abstractions_created: number;
    connections_pruned: number;
    replay_events: Record<string, unknown>[];
    last_cycle?: number;
}

interface WorkingMemoryState {
    capacity: number;
    current_size: number;
    items: Record<string, unknown>[];
    attention_weights: number[];
    decay_rate: number;
    eviction_history: Record<string, unknown>[];
    is_full: boolean;
    attentional_blink_active: boolean;
}

interface PatternSeparationMetrics {
    input_similarity: number;
    output_similarity: number;
    separation_ratio: number;
    sparsity: number;
    orthogonalization_strength: number;
}

interface PatternCompletionMetrics {
    input_completeness: number;
    output_confidence: number;
    convergence_iterations: number;
    best_match_id?: string;
    similarity_to_match: number;
}

interface BiologicalMechanismsResponse {
    fsrs_states: FSRSState[];
    hebbian_weights: HebbianWeight[];
    activation_spreading: ActivationSpread[];
    sleep_consolidation?: SleepConsolidationState;
    working_memory?: WorkingMemoryState;
    pattern_separation?: PatternSeparationMetrics;
    pattern_completion?: PatternCompletionMetrics;
}

// ============================================================================
// API Client
// ============================================================================

class WWApiClient {
    private baseUrl: string;
    private sessionId?: string;

    constructor(baseUrl: string = API_BASE_URL) {
        this.baseUrl = baseUrl;
    }

    setSessionId(sessionId: string) {
        this.sessionId = sessionId;
    }

    private async fetch<T>(endpoint: string, options: RequestInit = {}): Promise<T> {
        const headers: Record<string, string> = {
            "Content-Type": "application/json",
            ...(options.headers as Record<string, string>),
        };

        if (this.sessionId) {
            headers["X-Session-ID"] = this.sessionId;
        }

        const response = await fetch(`${this.baseUrl}${endpoint}`, {
            ...options,
            headers,
        });

        if (!response.ok) {
            const error = await response.json().catch(() => ({}));
            throw new Error(error.detail || `API error: ${response.status}`);
        }

        return response.json();
    }

    // Graph endpoints
    async getGraph(
        layout: string = "force-directed",
        limit: number = 500,
        includeEdges: boolean = true
    ): Promise<{ nodes: MemoryNode[]; edges: MemoryEdge[] }> {
        const params = new URLSearchParams({
            layout,
            limit: limit.toString(),
            include_edges: includeEdges.toString(),
        });

        const response = await this.fetch<GraphResponse>(`/viz/graph?${params}`);

        // Transform response to frontend types
        const nodes: MemoryNode[] = response.nodes.map((node) => {
            // Validate position values to prevent THREE.js NaN errors
            let position: Vector3 | undefined;
            if (node.position &&
                isFinite(node.position.x) &&
                isFinite(node.position.y) &&
                isFinite(node.position.z)) {
                position = new Vector3(node.position.x, node.position.y, node.position.z);
            }

            return {
                id: node.id,
                type: node.type,
                content: node.content,
                metadata: {
                    created_at: node.metadata.created_at,
                    last_accessed: node.metadata.last_accessed,
                    access_count: node.metadata.access_count,
                    importance: node.metadata.importance,
                    tags: node.metadata.tags,
                    source: node.metadata.source,
                },
                position,
            };
        });

        const edges: MemoryEdge[] = response.edges.map((edge) => ({
            id: edge.id,
            source: edge.source,
            target: edge.target,
            type: edge.type,
            weight: edge.weight,
            metadata: edge.metadata
                ? {
                      created_at: edge.metadata.created_at,
                      last_activated: edge.metadata.last_activated,
                  }
                : undefined,
        }));

        return { nodes, edges };
    }

    async getTimeline(days: number = 30, limit: number = 500): Promise<TimelineResponse> {
        const params = new URLSearchParams({
            days: days.toString(),
            limit: limit.toString(),
        });

        return this.fetch<TimelineResponse>(`/viz/timeline?${params}`);
    }

    async getActivity(limit: number = 100, topN: number = 10): Promise<ActivityResponse> {
        const params = new URLSearchParams({
            limit: limit.toString(),
            top_n: topN.toString(),
        });

        return this.fetch<ActivityResponse>(`/viz/activity?${params}`);
    }

    // Biological mechanism endpoints
    async getFSRSStates(
        limit: number = 50,
        includeDecayCurve: boolean = true,
        forecastDays: number = 30
    ): Promise<FSRSState[]> {
        const params = new URLSearchParams({
            limit: limit.toString(),
            include_decay_curve: includeDecayCurve.toString(),
            forecast_days: forecastDays.toString(),
        });

        return this.fetch<FSRSState[]>(`/viz/bio/fsrs?${params}`);
    }

    async getHebbianWeights(limit: number = 100, minWeight: number = 0.1): Promise<HebbianWeight[]> {
        const params = new URLSearchParams({
            limit: limit.toString(),
            min_weight: minWeight.toString(),
        });

        return this.fetch<HebbianWeight[]>(`/viz/bio/hebbian?${params}`);
    }

    async getActivationSpreading(sourceId?: string, limit: number = 50): Promise<ActivationSpread[]> {
        const params = new URLSearchParams({
            limit: limit.toString(),
        });

        if (sourceId) {
            params.set("source_id", sourceId);
        }

        return this.fetch<ActivationSpread[]>(`/viz/bio/activation?${params}`);
    }

    async getSleepConsolidationState(): Promise<SleepConsolidationState> {
        return this.fetch<SleepConsolidationState>("/viz/bio/sleep");
    }

    async getWorkingMemoryState(): Promise<WorkingMemoryState> {
        return this.fetch<WorkingMemoryState>("/viz/bio/working-memory");
    }

    async getPatternSeparationMetrics(
        inputA?: string,
        inputB?: string
    ): Promise<PatternSeparationMetrics> {
        const params = new URLSearchParams();

        if (inputA) params.set("input_a", inputA);
        if (inputB) params.set("input_b", inputB);

        return this.fetch<PatternSeparationMetrics>(`/viz/bio/pattern-separation?${params}`);
    }

    async getPatternCompletionMetrics(partialInput?: string): Promise<PatternCompletionMetrics> {
        const params = new URLSearchParams();

        if (partialInput) {
            params.set("partial_input", partialInput);
        }

        return this.fetch<PatternCompletionMetrics>(`/viz/bio/pattern-completion?${params}`);
    }

    async getAllBiologicalMechanisms(
        fsrsLimit: number = 20,
        hebbianLimit: number = 50
    ): Promise<BiologicalMechanismsResponse> {
        const params = new URLSearchParams({
            fsrs_limit: fsrsLimit.toString(),
            hebbian_limit: hebbianLimit.toString(),
        });

        return this.fetch<BiologicalMechanismsResponse>(`/viz/bio/all?${params}`);
    }

    // Export
    async exportGraph(format: string = "json"): Promise<{ content: string; filename: string }> {
        const response = await this.fetch<{ format: string; content: string; filename: string }>(
            "/viz/export",
            {
                method: "POST",
                body: JSON.stringify({ format }),
            }
        );

        return { content: response.content, filename: response.filename };
    }
}

// ============================================================================
// Singleton instance
// ============================================================================

export const wwApi = new WWApiClient();

// Export types for use in components
export type {
    FSRSState,
    HebbianWeight,
    ActivationSpread,
    SleepConsolidationState,
    WorkingMemoryState,
    PatternSeparationMetrics,
    PatternCompletionMetrics,
    BiologicalMechanismsResponse,
    TimelineEvent,
    TimelineResponse,
    ActivityMetrics,
    ActivityResponse,
};
