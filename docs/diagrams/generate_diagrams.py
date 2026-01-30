#!/usr/bin/env python3
"""
World Weaver System Diagrams Generator
Creates PNG diagrams for system architecture, encoding, storage, retrieval, and learning.
"""

import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.patches import FancyBboxPatch, FancyArrowPatch, Circle, Rectangle, Polygon
from matplotlib.collections import PatchCollection
import numpy as np

# Color palette
COLORS = {
    'episodic': '#4A90D9',      # Blue
    'semantic': '#50B83C',       # Green
    'procedural': '#9C6ADE',     # Purple
    'storage': '#F5A623',        # Orange
    'embedding': '#E34F6F',      # Red
    'consolidation': '#47C1BF',  # Teal
    'learning': '#FF6B6B',       # Coral
    'mcp': '#2D3748',            # Dark gray
    'api': '#667EEA',            # Indigo
    'bg': '#FAFBFC',             # Light gray
    'text': '#1A202C',           # Dark text
    'arrow': '#718096',          # Gray arrows
}

def create_architecture_diagram():
    """Create main system architecture diagram."""
    fig, ax = plt.subplots(figsize=(16, 12))
    ax.set_xlim(0, 16)
    ax.set_ylim(0, 12)
    ax.set_aspect('equal')
    ax.axis('off')
    ax.set_facecolor(COLORS['bg'])
    fig.patch.set_facecolor(COLORS['bg'])

    # Title
    ax.text(8, 11.5, 'World Weaver Architecture', fontsize=24, fontweight='bold',
            ha='center', va='center', color=COLORS['text'])
    ax.text(8, 11, 'Neural Memory System for AI Agents', fontsize=12,
            ha='center', va='center', color=COLORS['arrow'])

    # Claude Code / MCP Layer (top)
    mcp_box = FancyBboxPatch((1, 9.5), 14, 1, boxstyle="round,pad=0.05",
                              facecolor=COLORS['mcp'], edgecolor='white', linewidth=2)
    ax.add_patch(mcp_box)
    ax.text(8, 10, 'Claude Code / MCP Interface', fontsize=14, fontweight='bold',
            ha='center', va='center', color='white')

    # API Gateway Layer
    api_box = FancyBboxPatch((1, 8), 14, 1.2, boxstyle="round,pad=0.05",
                              facecolor=COLORS['api'], edgecolor='white', linewidth=2)
    ax.add_patch(api_box)
    ax.text(8, 8.6, 'REST API Gateway (FastAPI)', fontsize=13, fontweight='bold',
            ha='center', va='center', color='white')
    ax.text(8, 8.2, 'Session Isolation | Rate Limiting | Saga Orchestration', fontsize=10,
            ha='center', va='center', color='white', alpha=0.9)

    # Memory Gateway
    gateway_box = FancyBboxPatch((5.5, 6.5), 5, 1.2, boxstyle="round,pad=0.05",
                                  facecolor='#2C5282', edgecolor='white', linewidth=2)
    ax.add_patch(gateway_box)
    ax.text(8, 7.1, 'Memory Gateway', fontsize=12, fontweight='bold',
            ha='center', va='center', color='white')
    ax.text(8, 6.7, 'Service Initialization | Context', fontsize=9,
            ha='center', va='center', color='white', alpha=0.9)

    # Tripartite Memory (3 boxes)
    # Episodic
    ep_box = FancyBboxPatch((1, 4), 4, 2, boxstyle="round,pad=0.05",
                             facecolor=COLORS['episodic'], edgecolor='white', linewidth=2)
    ax.add_patch(ep_box)
    ax.text(3, 5.5, 'Episodic Memory', fontsize=12, fontweight='bold',
            ha='center', va='center', color='white')
    ax.text(3, 5.1, 'Episodes | FSRS Decay', fontsize=9,
            ha='center', va='center', color='white', alpha=0.9)
    ax.text(3, 4.7, 'Pattern Separation', fontsize=9,
            ha='center', va='center', color='white', alpha=0.9)
    ax.text(3, 4.3, 'Working Memory Buffer', fontsize=9,
            ha='center', va='center', color='white', alpha=0.9)

    # Semantic
    sem_box = FancyBboxPatch((6, 4), 4, 2, boxstyle="round,pad=0.05",
                              facecolor=COLORS['semantic'], edgecolor='white', linewidth=2)
    ax.add_patch(sem_box)
    ax.text(8, 5.5, 'Semantic Memory', fontsize=12, fontweight='bold',
            ha='center', va='center', color='white')
    ax.text(8, 5.1, 'Entities | Relations', fontsize=9,
            ha='center', va='center', color='white', alpha=0.9)
    ax.text(8, 4.7, 'Spreading Activation', fontsize=9,
            ha='center', va='center', color='white', alpha=0.9)
    ax.text(8, 4.3, 'Bi-temporal Versioning', fontsize=9,
            ha='center', va='center', color='white', alpha=0.9)

    # Procedural
    proc_box = FancyBboxPatch((11, 4), 4, 2, boxstyle="round,pad=0.05",
                               facecolor=COLORS['procedural'], edgecolor='white', linewidth=2)
    ax.add_patch(proc_box)
    ax.text(13, 5.5, 'Procedural Memory', fontsize=12, fontweight='bold',
            ha='center', va='center', color='white')
    ax.text(13, 5.1, 'Skills | Execution Logs', fontsize=9,
            ha='center', va='center', color='white', alpha=0.9)
    ax.text(13, 4.7, 'Dependency Graphs', fontsize=9,
            ha='center', va='center', color='white', alpha=0.9)
    ax.text(13, 4.3, 'Success Rate Tracking', fontsize=9,
            ha='center', va='center', color='white', alpha=0.9)

    # Embedding Layer
    emb_box = FancyBboxPatch((1, 2.5), 6, 1, boxstyle="round,pad=0.05",
                              facecolor=COLORS['embedding'], edgecolor='white', linewidth=2)
    ax.add_patch(emb_box)
    ax.text(4, 3, 'BGE-M3 Embeddings (1024-dim)', fontsize=11, fontweight='bold',
            ha='center', va='center', color='white')
    ax.text(4, 2.7, 'Dense + Sparse + ColBERT | GPU Accelerated', fontsize=9,
            ha='center', va='center', color='white', alpha=0.9)

    # Learning Layer
    learn_box = FancyBboxPatch((9, 2.5), 6, 1, boxstyle="round,pad=0.05",
                                facecolor=COLORS['learning'], edgecolor='white', linewidth=2)
    ax.add_patch(learn_box)
    ax.text(12, 3, 'Learning Systems', fontsize=11, fontweight='bold',
            ha='center', va='center', color='white')
    ax.text(12, 2.7, 'Hebbian | Memory Gate | Neuromodulation', fontsize=9,
            ha='center', va='center', color='white', alpha=0.9)

    # Storage Layer
    qdrant_box = FancyBboxPatch((1, 0.5), 6, 1.5, boxstyle="round,pad=0.05",
                                 facecolor=COLORS['storage'], edgecolor='white', linewidth=2)
    ax.add_patch(qdrant_box)
    ax.text(4, 1.5, 'Qdrant Vector Store', fontsize=11, fontweight='bold',
            ha='center', va='center', color='white')
    ax.text(4, 1.1, 'Episodes | Entities | Skills', fontsize=9,
            ha='center', va='center', color='white', alpha=0.9)
    ax.text(4, 0.7, 'Hybrid Search | HNSW Index', fontsize=9,
            ha='center', va='center', color='white', alpha=0.9)

    neo4j_box = FancyBboxPatch((9, 0.5), 6, 1.5, boxstyle="round,pad=0.05",
                                facecolor='#68A063', edgecolor='white', linewidth=2)
    ax.add_patch(neo4j_box)
    ax.text(12, 1.5, 'Neo4j Graph Store', fontsize=11, fontweight='bold',
            ha='center', va='center', color='white')
    ax.text(12, 1.1, 'Entity Relationships', fontsize=9,
            ha='center', va='center', color='white', alpha=0.9)
    ax.text(12, 0.7, 'Cypher Queries | Knowledge Graph', fontsize=9,
            ha='center', va='center', color='white', alpha=0.9)

    # Arrows
    arrow_style = dict(arrowstyle='->', color=COLORS['arrow'], lw=2,
                       connectionstyle='arc3,rad=0')

    # MCP to API
    ax.annotate('', xy=(8, 9.2), xytext=(8, 9.5),
                arrowprops=arrow_style)

    # API to Gateway
    ax.annotate('', xy=(8, 7.7), xytext=(8, 8),
                arrowprops=arrow_style)

    # Gateway to Memories
    ax.annotate('', xy=(3, 6), xytext=(6.5, 6.5),
                arrowprops=arrow_style)
    ax.annotate('', xy=(8, 6), xytext=(8, 6.5),
                arrowprops=arrow_style)
    ax.annotate('', xy=(13, 6), xytext=(9.5, 6.5),
                arrowprops=arrow_style)

    # Memories to Embedding
    ax.annotate('', xy=(4, 3.5), xytext=(3, 4),
                arrowprops=arrow_style)
    ax.annotate('', xy=(4, 3.5), xytext=(8, 4),
                arrowprops=arrow_style)

    # Memories to Learning
    ax.annotate('', xy=(12, 3.5), xytext=(8, 4),
                arrowprops=arrow_style)
    ax.annotate('', xy=(12, 3.5), xytext=(13, 4),
                arrowprops=arrow_style)

    # Embedding to Storage
    ax.annotate('', xy=(4, 2), xytext=(4, 2.5),
                arrowprops=arrow_style)

    # Learning to Storage
    ax.annotate('', xy=(12, 2), xytext=(12, 2.5),
                arrowprops=arrow_style)

    plt.tight_layout()
    plt.savefig('/mnt/projects/ww/diagrams/01_system_architecture.png', dpi=150,
                bbox_inches='tight', facecolor=COLORS['bg'])
    plt.close()
    print("Created: 01_system_architecture.png")


def create_encoding_diagram():
    """Create memory encoding flow diagram."""
    fig, ax = plt.subplots(figsize=(14, 10))
    ax.set_xlim(0, 14)
    ax.set_ylim(0, 10)
    ax.set_aspect('equal')
    ax.axis('off')
    ax.set_facecolor(COLORS['bg'])
    fig.patch.set_facecolor(COLORS['bg'])

    # Title
    ax.text(7, 9.5, 'Memory Encoding Pipeline', fontsize=22, fontweight='bold',
            ha='center', va='center', color=COLORS['text'])

    # Input
    input_box = FancyBboxPatch((0.5, 7), 3, 1.5, boxstyle="round,pad=0.05",
                                facecolor='#2D3748', edgecolor='white', linewidth=2)
    ax.add_patch(input_box)
    ax.text(2, 8.1, 'Input Content', fontsize=11, fontweight='bold',
            ha='center', va='center', color='white')
    ax.text(2, 7.6, 'Text + Context + Outcome', fontsize=9,
            ha='center', va='center', color='white', alpha=0.9)
    ax.text(2, 7.2, 'Emotional Valence', fontsize=9,
            ha='center', va='center', color='white', alpha=0.9)

    # Memory Gate
    gate_box = FancyBboxPatch((5, 7), 4, 1.5, boxstyle="round,pad=0.05",
                               facecolor=COLORS['learning'], edgecolor='white', linewidth=2)
    ax.add_patch(gate_box)
    ax.text(7, 8.1, 'Memory Gate', fontsize=11, fontweight='bold',
            ha='center', va='center', color='white')
    ax.text(7, 7.6, 'Salience Scoring (0-1)', fontsize=9,
            ha='center', va='center', color='white', alpha=0.9)
    ax.text(7, 7.2, 'Threshold: 0.4', fontsize=9,
            ha='center', va='center', color='white', alpha=0.9)

    # Decision diamond
    diamond = plt.Polygon([(12, 7.75), (11, 7), (12, 6.25), (13, 7)],
                          facecolor='#9F7AEA', edgecolor='white', linewidth=2)
    ax.add_patch(diamond)
    ax.text(12, 7, 'Pass\nGate?', fontsize=9, fontweight='bold',
            ha='center', va='center', color='white')

    # BGE-M3 Embedding
    emb_box = FancyBboxPatch((5, 4.5), 4, 1.5, boxstyle="round,pad=0.05",
                              facecolor=COLORS['embedding'], edgecolor='white', linewidth=2)
    ax.add_patch(emb_box)
    ax.text(7, 5.6, 'BGE-M3 Embedding', fontsize=11, fontweight='bold',
            ha='center', va='center', color='white')
    ax.text(7, 5.2, 'Dense (1024-dim)', fontsize=9,
            ha='center', va='center', color='white', alpha=0.9)
    ax.text(7, 4.8, 'Sparse (BM25) + ColBERT', fontsize=9,
            ha='center', va='center', color='white', alpha=0.9)

    # Pattern Separation
    pattern_box = FancyBboxPatch((0.5, 4.5), 3.5, 1.5, boxstyle="round,pad=0.05",
                                  facecolor=COLORS['episodic'], edgecolor='white', linewidth=2)
    ax.add_patch(pattern_box)
    ax.text(2.25, 5.6, 'Pattern Separation', fontsize=10, fontweight='bold',
            ha='center', va='center', color='white')
    ax.text(2.25, 5.2, 'Orthogonalize similar', fontsize=9,
            ha='center', va='center', color='white', alpha=0.9)
    ax.text(2.25, 4.8, 'memories (DG model)', fontsize=9,
            ha='center', va='center', color='white', alpha=0.9)

    # FSRS Initialization
    fsrs_box = FancyBboxPatch((10, 4.5), 3.5, 1.5, boxstyle="round,pad=0.05",
                               facecolor=COLORS['consolidation'], edgecolor='white', linewidth=2)
    ax.add_patch(fsrs_box)
    ax.text(11.75, 5.6, 'FSRS Initialize', fontsize=10, fontweight='bold',
            ha='center', va='center', color='white')
    ax.text(11.75, 5.2, 'Stability: 1.0', fontsize=9,
            ha='center', va='center', color='white', alpha=0.9)
    ax.text(11.75, 4.8, 'Difficulty: 0.3', fontsize=9,
            ha='center', va='center', color='white', alpha=0.9)

    # Saga Transaction
    saga_box = FancyBboxPatch((5, 2), 4, 1.5, boxstyle="round,pad=0.05",
                               facecolor='#553C9A', edgecolor='white', linewidth=2)
    ax.add_patch(saga_box)
    ax.text(7, 3.1, 'Saga Transaction', fontsize=11, fontweight='bold',
            ha='center', va='center', color='white')
    ax.text(7, 2.7, 'Two-Phase Commit', fontsize=9,
            ha='center', va='center', color='white', alpha=0.9)
    ax.text(7, 2.3, 'Rollback on Failure', fontsize=9,
            ha='center', va='center', color='white', alpha=0.9)

    # Storage
    storage_box = FancyBboxPatch((5, 0.3), 4, 1.2, boxstyle="round,pad=0.05",
                                  facecolor=COLORS['storage'], edgecolor='white', linewidth=2)
    ax.add_patch(storage_box)
    ax.text(7, 1.15, 'Qdrant + Neo4j', fontsize=11, fontweight='bold',
            ha='center', va='center', color='white')
    ax.text(7, 0.7, 'Vector + Graph Storage', fontsize=9,
            ha='center', va='center', color='white', alpha=0.9)

    # Discard box
    discard_box = FancyBboxPatch((10.5, 7), 3, 1, boxstyle="round,pad=0.05",
                                  facecolor='#E53E3E', edgecolor='white', linewidth=2)
    ax.add_patch(discard_box)
    ax.text(12, 7.5, 'Discarded', fontsize=10, fontweight='bold',
            ha='center', va='center', color='white')

    # Arrows
    arrow_style = dict(arrowstyle='->', color=COLORS['arrow'], lw=2)

    ax.annotate('', xy=(5, 7.75), xytext=(3.5, 7.75), arrowprops=arrow_style)
    ax.annotate('', xy=(11, 7.5), xytext=(9, 7.75), arrowprops=arrow_style)
    ax.annotate('', xy=(7, 6), xytext=(11.5, 6.25), arrowprops=arrow_style)
    ax.annotate('', xy=(4, 5.25), xytext=(5, 5.25), arrowprops=arrow_style)
    ax.annotate('', xy=(10, 5.25), xytext=(9, 5.25), arrowprops=arrow_style)
    ax.annotate('', xy=(7, 3.5), xytext=(7, 4.5), arrowprops=arrow_style)
    ax.annotate('', xy=(7, 1.5), xytext=(7, 2), arrowprops=arrow_style)

    # Labels
    ax.text(4.3, 7.9, '1', fontsize=10, fontweight='bold', ha='center',
            bbox=dict(boxstyle='circle', facecolor='white', edgecolor=COLORS['arrow']))
    ax.text(10.2, 7.9, '2', fontsize=10, fontweight='bold', ha='center',
            bbox=dict(boxstyle='circle', facecolor='white', edgecolor=COLORS['arrow']))
    ax.text(9.5, 6.1, '3', fontsize=10, fontweight='bold', ha='center',
            bbox=dict(boxstyle='circle', facecolor='white', edgecolor=COLORS['arrow']))
    ax.text(7, 3.9, '4', fontsize=10, fontweight='bold', ha='center',
            bbox=dict(boxstyle='circle', facecolor='white', edgecolor=COLORS['arrow']))
    ax.text(7, 1.75, '5', fontsize=10, fontweight='bold', ha='center',
            bbox=dict(boxstyle='circle', facecolor='white', edgecolor=COLORS['arrow']))

    # Yes/No labels
    ax.text(9.7, 6.9, 'Yes', fontsize=9, color=COLORS['semantic'], fontweight='bold')
    ax.text(12.3, 8.2, 'No', fontsize=9, color='#E53E3E', fontweight='bold')
    ax.annotate('', xy=(12, 8), xytext=(12, 7.75), arrowprops=arrow_style)

    plt.tight_layout()
    plt.savefig('/mnt/projects/ww/diagrams/02_encoding_flow.png', dpi=150,
                bbox_inches='tight', facecolor=COLORS['bg'])
    plt.close()
    print("Created: 02_encoding_flow.png")


def create_storage_diagram():
    """Create storage architecture diagram."""
    fig, ax = plt.subplots(figsize=(16, 10))
    ax.set_xlim(0, 16)
    ax.set_ylim(0, 10)
    ax.set_aspect('equal')
    ax.axis('off')
    ax.set_facecolor(COLORS['bg'])
    fig.patch.set_facecolor(COLORS['bg'])

    # Title
    ax.text(8, 9.5, 'Storage Architecture', fontsize=22, fontweight='bold',
            ha='center', va='center', color=COLORS['text'])

    # Qdrant Section
    qdrant_outer = FancyBboxPatch((0.5, 1), 7, 7.5, boxstyle="round,pad=0.1",
                                   facecolor='#FFF5EB', edgecolor=COLORS['storage'], linewidth=3)
    ax.add_patch(qdrant_outer)
    ax.text(4, 8, 'Qdrant Vector Database', fontsize=14, fontweight='bold',
            ha='center', va='center', color=COLORS['storage'])

    # Qdrant Collections
    collections = [
        ('ww_episodes', COLORS['episodic'], 'Episodes\n1024-dim vectors\nHybrid index'),
        ('ww_entities', COLORS['semantic'], 'Entities\n1024-dim vectors\nSession filter'),
        ('ww_procedures', COLORS['procedural'], 'Skills\n1024-dim vectors\nDomain filter'),
        ('ww_episodes_hybrid', '#2B6CB0', 'Hybrid Index\nDense + Sparse\nColBERT tokens'),
    ]

    for i, (name, color, desc) in enumerate(collections):
        y = 6.5 - i * 1.4
        box = FancyBboxPatch((1, y), 6, 1.2, boxstyle="round,pad=0.03",
                              facecolor=color, edgecolor='white', linewidth=1.5)
        ax.add_patch(box)
        ax.text(4, y + 0.85, name, fontsize=10, fontweight='bold',
                ha='center', va='center', color='white')
        ax.text(4, y + 0.35, desc, fontsize=8,
                ha='center', va='center', color='white', alpha=0.9)

    # Neo4j Section
    neo4j_outer = FancyBboxPatch((8.5, 1), 7, 7.5, boxstyle="round,pad=0.1",
                                  facecolor='#F0FFF4', edgecolor='#68A063', linewidth=3)
    ax.add_patch(neo4j_outer)
    ax.text(12, 8, 'Neo4j Graph Database', fontsize=14, fontweight='bold',
            ha='center', va='center', color='#68A063')

    # Neo4j Node Types
    ax.text(10.5, 6.8, 'Node Types', fontsize=11, fontweight='bold',
            ha='center', va='center', color=COLORS['text'])

    node_types = ['Entity', 'Relationship', 'Session', 'Timestamp']
    for i, nt in enumerate(node_types):
        circle = Circle((9.5 + (i % 2) * 2, 5.8 - (i // 2) * 1.2), 0.4,
                        facecolor=COLORS['semantic'], edgecolor='white', linewidth=1.5)
        ax.add_patch(circle)
        ax.text(9.5 + (i % 2) * 2, 5.8 - (i // 2) * 1.2, nt[:3], fontsize=8,
                ha='center', va='center', color='white', fontweight='bold')

    ax.text(14, 6.8, 'Edge Types', fontsize=11, fontweight='bold',
            ha='center', va='center', color=COLORS['text'])

    edge_types = ['RELATES_TO', 'USES', 'SUPERSEDES', 'TRIGGERS']
    for i, et in enumerate(edge_types):
        ax.text(14, 5.8 - i * 0.5, et, fontsize=9,
                ha='center', va='center', color=COLORS['arrow'])

    # Knowledge Graph Visual
    ax.text(12, 3.2, 'Knowledge Graph', fontsize=11, fontweight='bold',
            ha='center', va='center', color=COLORS['text'])

    # Graph nodes
    nodes = [(10, 2.3), (12, 2.8), (14, 2.3), (11, 1.3), (13, 1.3)]
    for nx, ny in nodes:
        circle = Circle((nx, ny), 0.3, facecolor=COLORS['semantic'],
                        edgecolor='white', linewidth=1.5)
        ax.add_patch(circle)

    # Graph edges
    edges = [((10, 2.3), (12, 2.8)), ((12, 2.8), (14, 2.3)),
             ((10, 2.3), (11, 1.3)), ((14, 2.3), (13, 1.3)),
             ((11, 1.3), (13, 1.3)), ((12, 2.8), (11, 1.3))]
    for (x1, y1), (x2, y2) in edges:
        ax.plot([x1, x2], [y1, y2], color=COLORS['arrow'], lw=1.5, alpha=0.6)

    # Resilience Layer
    resilience_box = FancyBboxPatch((3, 0.2), 10, 0.6, boxstyle="round,pad=0.03",
                                     facecolor='#805AD5', edgecolor='white', linewidth=2)
    ax.add_patch(resilience_box)
    ax.text(8, 0.5, 'Resilience: Circuit Breaker | Retry with Backoff | Connection Pooling',
            fontsize=10, fontweight='bold', ha='center', va='center', color='white')

    plt.tight_layout()
    plt.savefig('/mnt/projects/ww/diagrams/03_storage_architecture.png', dpi=150,
                bbox_inches='tight', facecolor=COLORS['bg'])
    plt.close()
    print("Created: 03_storage_architecture.png")


def create_retrieval_diagram():
    """Create retrieval/decoding flow diagram."""
    fig, ax = plt.subplots(figsize=(14, 10))
    ax.set_xlim(0, 14)
    ax.set_ylim(0, 10)
    ax.set_aspect('equal')
    ax.axis('off')
    ax.set_facecolor(COLORS['bg'])
    fig.patch.set_facecolor(COLORS['bg'])

    # Title
    ax.text(7, 9.5, 'Memory Retrieval Pipeline', fontsize=22, fontweight='bold',
            ha='center', va='center', color=COLORS['text'])

    # Query Input
    query_box = FancyBboxPatch((0.5, 7.5), 3, 1.2, boxstyle="round,pad=0.05",
                                facecolor='#2D3748', edgecolor='white', linewidth=2)
    ax.add_patch(query_box)
    ax.text(2, 8.4, 'Query Input', fontsize=11, fontweight='bold',
            ha='center', va='center', color='white')
    ax.text(2, 7.9, 'Natural language', fontsize=9,
            ha='center', va='center', color='white', alpha=0.9)

    # Query Embedding
    emb_box = FancyBboxPatch((4.5, 7.5), 3.5, 1.2, boxstyle="round,pad=0.05",
                              facecolor=COLORS['embedding'], edgecolor='white', linewidth=2)
    ax.add_patch(emb_box)
    ax.text(6.25, 8.4, 'Query Embedding', fontsize=11, fontweight='bold',
            ha='center', va='center', color='white')
    ax.text(6.25, 7.9, 'BGE-M3 (cached)', fontsize=9,
            ha='center', va='center', color='white', alpha=0.9)

    # Hybrid Search
    hybrid_box = FancyBboxPatch((9, 7.5), 4.5, 1.2, boxstyle="round,pad=0.05",
                                 facecolor=COLORS['storage'], edgecolor='white', linewidth=2)
    ax.add_patch(hybrid_box)
    ax.text(11.25, 8.4, 'Hybrid Search', fontsize=11, fontweight='bold',
            ha='center', va='center', color='white')
    ax.text(11.25, 7.9, 'Dense + Sparse + ColBERT RRF', fontsize=9,
            ha='center', va='center', color='white', alpha=0.9)

    # FSRS Filter
    fsrs_box = FancyBboxPatch((0.5, 5.3), 4, 1.5, boxstyle="round,pad=0.05",
                               facecolor=COLORS['consolidation'], edgecolor='white', linewidth=2)
    ax.add_patch(fsrs_box)
    ax.text(2.5, 6.35, 'FSRS Retrievability', fontsize=11, fontweight='bold',
            ha='center', va='center', color='white')
    ax.text(2.5, 5.9, 'R(t) = e^(-t/S)', fontsize=10, family='monospace',
            ha='center', va='center', color='white')
    ax.text(2.5, 5.5, 'Filter decayed memories', fontsize=9,
            ha='center', va='center', color='white', alpha=0.9)

    # Pattern Completion
    pattern_box = FancyBboxPatch((5.5, 5.3), 3.5, 1.5, boxstyle="round,pad=0.05",
                                  facecolor=COLORS['episodic'], edgecolor='white', linewidth=2)
    ax.add_patch(pattern_box)
    ax.text(7.25, 6.35, 'Pattern Completion', fontsize=11, fontweight='bold',
            ha='center', va='center', color='white')
    ax.text(7.25, 5.9, 'CA3 Autoassociation', fontsize=9,
            ha='center', va='center', color='white', alpha=0.9)
    ax.text(7.25, 5.5, 'Expand partial cues', fontsize=9,
            ha='center', va='center', color='white', alpha=0.9)

    # Spreading Activation
    spread_box = FancyBboxPatch((10, 5.3), 3.5, 1.5, boxstyle="round,pad=0.05",
                                 facecolor=COLORS['semantic'], edgecolor='white', linewidth=2)
    ax.add_patch(spread_box)
    ax.text(11.75, 6.35, 'Spreading Activation', fontsize=11, fontweight='bold',
            ha='center', va='center', color='white')
    ax.text(11.75, 5.9, 'Traverse knowledge graph', fontsize=9,
            ha='center', va='center', color='white', alpha=0.9)
    ax.text(11.75, 5.5, 'Decay: 0.85^depth', fontsize=9,
            ha='center', va='center', color='white', alpha=0.9)

    # Hebbian Update
    hebb_box = FancyBboxPatch((3, 3), 4, 1.5, boxstyle="round,pad=0.05",
                               facecolor=COLORS['learning'], edgecolor='white', linewidth=2)
    ax.add_patch(hebb_box)
    ax.text(5, 4.05, 'Hebbian Strengthening', fontsize=11, fontweight='bold',
            ha='center', va='center', color='white')
    ax.text(5, 3.6, 'w += lr * pre * post', fontsize=10, family='monospace',
            ha='center', va='center', color='white')
    ax.text(5, 3.2, 'Co-access strengthens', fontsize=9,
            ha='center', va='center', color='white', alpha=0.9)

    # Ranking
    rank_box = FancyBboxPatch((8, 3), 4, 1.5, boxstyle="round,pad=0.05",
                               facecolor='#805AD5', edgecolor='white', linewidth=2)
    ax.add_patch(rank_box)
    ax.text(10, 4.05, 'Final Ranking', fontsize=11, fontweight='bold',
            ha='center', va='center', color='white')
    ax.text(10, 3.6, 'sim * retrievability * importance', fontsize=9,
            ha='center', va='center', color='white', alpha=0.9)
    ax.text(10, 3.2, 'Top-K results', fontsize=9,
            ha='center', va='center', color='white', alpha=0.9)

    # Output
    output_box = FancyBboxPatch((5.5, 0.8), 4, 1.5, boxstyle="round,pad=0.05",
                                 facecolor='#2D3748', edgecolor='white', linewidth=2)
    ax.add_patch(output_box)
    ax.text(7.5, 1.85, 'ScoredResult[]', fontsize=11, fontweight='bold',
            ha='center', va='center', color='white')
    ax.text(7.5, 1.4, 'item + score + components', fontsize=9,
            ha='center', va='center', color='white', alpha=0.9)
    ax.text(7.5, 1, 'similarity, recency, importance', fontsize=9,
            ha='center', va='center', color='white', alpha=0.9)

    # Arrows
    arrow_style = dict(arrowstyle='->', color=COLORS['arrow'], lw=2)

    ax.annotate('', xy=(4.5, 8.1), xytext=(3.5, 8.1), arrowprops=arrow_style)
    ax.annotate('', xy=(9, 8.1), xytext=(8, 8.1), arrowprops=arrow_style)
    ax.annotate('', xy=(2.5, 6.8), xytext=(11.25, 7.5), arrowprops=arrow_style)
    ax.annotate('', xy=(7.25, 6.8), xytext=(11.25, 7.5), arrowprops=arrow_style)
    ax.annotate('', xy=(11.75, 6.8), xytext=(11.25, 7.5), arrowprops=arrow_style)
    ax.annotate('', xy=(5, 4.5), xytext=(2.5, 5.3), arrowprops=arrow_style)
    ax.annotate('', xy=(5, 4.5), xytext=(7.25, 5.3), arrowprops=arrow_style)
    ax.annotate('', xy=(10, 4.5), xytext=(11.75, 5.3), arrowprops=arrow_style)
    ax.annotate('', xy=(8, 3.75), xytext=(7, 3.75), arrowprops=arrow_style)
    ax.annotate('', xy=(7.5, 2.3), xytext=(10, 3), arrowprops=arrow_style)

    plt.tight_layout()
    plt.savefig('/mnt/projects/ww/diagrams/04_retrieval_flow.png', dpi=150,
                bbox_inches='tight', facecolor=COLORS['bg'])
    plt.close()
    print("Created: 04_retrieval_flow.png")


def create_learning_diagram():
    """Create learning and consolidation diagram."""
    fig, ax = plt.subplots(figsize=(16, 12))
    ax.set_xlim(0, 16)
    ax.set_ylim(0, 12)
    ax.set_aspect('equal')
    ax.axis('off')
    ax.set_facecolor(COLORS['bg'])
    fig.patch.set_facecolor(COLORS['bg'])

    # Title
    ax.text(8, 11.5, 'Learning & Consolidation Systems', fontsize=22, fontweight='bold',
            ha='center', va='center', color=COLORS['text'])

    # === FSRS Memory Decay ===
    fsrs_outer = FancyBboxPatch((0.5, 7.5), 5, 3.5, boxstyle="round,pad=0.1",
                                 facecolor='#E6FFFA', edgecolor=COLORS['consolidation'], linewidth=2)
    ax.add_patch(fsrs_outer)
    ax.text(3, 10.6, 'FSRS Memory Decay', fontsize=13, fontweight='bold',
            ha='center', va='center', color=COLORS['consolidation'])

    ax.text(3, 10, 'Retrievability: R(t) = e^(-t/S)', fontsize=10, family='monospace',
            ha='center', va='center', color=COLORS['text'])
    ax.text(3, 9.5, 'Stability updates on access', fontsize=9,
            ha='center', va='center', color=COLORS['arrow'])

    # Decay curve
    t = np.linspace(0, 3, 100)
    r = np.exp(-t)
    ax.plot(t * 1.3 + 0.8, r * 1.5 + 7.7, color=COLORS['consolidation'], lw=2)
    ax.text(4.5, 8.5, 'time', fontsize=8, color=COLORS['arrow'])
    ax.text(0.7, 9.3, 'R', fontsize=8, color=COLORS['arrow'])

    # === Hebbian Learning ===
    hebb_outer = FancyBboxPatch((6, 7.5), 4.5, 3.5, boxstyle="round,pad=0.1",
                                 facecolor='#FFF5F5', edgecolor=COLORS['learning'], linewidth=2)
    ax.add_patch(hebb_outer)
    ax.text(8.25, 10.6, 'Hebbian Learning', fontsize=13, fontweight='bold',
            ha='center', va='center', color=COLORS['learning'])

    ax.text(8.25, 10, 'w_ij += lr * a_i * a_j', fontsize=10, family='monospace',
            ha='center', va='center', color=COLORS['text'])
    ax.text(8.25, 9.5, 'Co-activated = strengthened', fontsize=9,
            ha='center', va='center', color=COLORS['arrow'])

    # Synapse visual
    ax.plot([6.8, 7.5], [8.5, 8.5], color=COLORS['learning'], lw=3)
    ax.plot([7.5, 8.25], [8.5, 8.8], color=COLORS['learning'], lw=3)
    ax.plot([8.25, 9], [8.8, 8.5], color=COLORS['learning'], lw=3)
    ax.plot([9, 9.7], [8.5, 8.5], color=COLORS['learning'], lw=3)
    circle1 = Circle((6.8, 8.5), 0.2, facecolor=COLORS['episodic'], edgecolor='white')
    circle2 = Circle((9.7, 8.5), 0.2, facecolor=COLORS['episodic'], edgecolor='white')
    ax.add_patch(circle1)
    ax.add_patch(circle2)
    ax.text(8.25, 8, 'synapse', fontsize=8, ha='center', color=COLORS['arrow'])

    # === Memory Gate ===
    gate_outer = FancyBboxPatch((11, 7.5), 4.5, 3.5, boxstyle="round,pad=0.1",
                                 facecolor='#FAF5FF', edgecolor='#9F7AEA', linewidth=2)
    ax.add_patch(gate_outer)
    ax.text(13.25, 10.6, 'Learned Memory Gate', fontsize=13, fontweight='bold',
            ha='center', va='center', color='#9F7AEA')

    ax.text(13.25, 10, 'GMM Salience Model', fontsize=10,
            ha='center', va='center', color=COLORS['text'])
    ax.text(13.25, 9.5, 'Cold-start: 100 samples', fontsize=9,
            ha='center', va='center', color=COLORS['arrow'])

    # Gate visual
    gate_box = FancyBboxPatch((12.25, 8), 2, 1, boxstyle="round,pad=0.03",
                               facecolor='#9F7AEA', edgecolor='white', linewidth=1.5)
    ax.add_patch(gate_box)
    ax.text(13.25, 8.5, 'P(store|x) > 0.4', fontsize=9,
            ha='center', va='center', color='white', fontweight='bold')

    # === Neuromodulation ===
    neuro_outer = FancyBboxPatch((0.5, 3.5), 7.5, 3.5, boxstyle="round,pad=0.1",
                                  facecolor='#FFFFF0', edgecolor='#D69E2E', linewidth=2)
    ax.add_patch(neuro_outer)
    ax.text(4.25, 6.6, 'Neuromodulation System', fontsize=13, fontweight='bold',
            ha='center', va='center', color='#D69E2E')

    modulators = [
        ('NE', 'Norepinephrine', 'Arousal/Urgency', '#E53E3E'),
        ('DA', 'Dopamine', 'Reward/Learning', '#38A169'),
        ('ACh', 'Acetylcholine', 'Encode/Retrieve', '#3182CE'),
        ('GABA', 'GABA', 'Inhibition/Focus', '#805AD5'),
    ]

    for i, (abbr, name, role, color) in enumerate(modulators):
        x = 1.3 + (i % 2) * 3.5
        y = 5.5 - (i // 2) * 1.3
        circle = Circle((x, y), 0.35, facecolor=color, edgecolor='white', linewidth=1.5)
        ax.add_patch(circle)
        ax.text(x, y, abbr, fontsize=9, fontweight='bold', ha='center', va='center', color='white')
        ax.text(x + 1.2, y + 0.15, name, fontsize=9, ha='left', va='center', color=COLORS['text'])
        ax.text(x + 1.2, y - 0.2, role, fontsize=8, ha='left', va='center', color=COLORS['arrow'])

    # === Consolidation ===
    consol_outer = FancyBboxPatch((8.5, 3.5), 7, 3.5, boxstyle="round,pad=0.1",
                                   facecolor='#EBF8FF', edgecolor=COLORS['episodic'], linewidth=2)
    ax.add_patch(consol_outer)
    ax.text(12, 6.6, 'Memory Consolidation', fontsize=13, fontweight='bold',
            ha='center', va='center', color=COLORS['episodic'])

    stages = [
        ('Working Memory', 'Short-term buffer (50 items)'),
        ('Episode Clustering', 'HDBSCAN grouping'),
        ('Entity Extraction', 'Promote to semantic'),
        ('Skill Induction', 'Pattern to procedure'),
    ]

    for i, (stage, desc) in enumerate(stages):
        y = 5.8 - i * 0.7
        ax.text(9, y, f'{i+1}.', fontsize=10, fontweight='bold', color=COLORS['episodic'])
        ax.text(9.5, y, stage, fontsize=10, color=COLORS['text'])
        ax.text(12.5, y, desc, fontsize=9, color=COLORS['arrow'])

    # === Session State ===
    session_box = FancyBboxPatch((3, 0.5), 10, 2.5, boxstyle="round,pad=0.1",
                                  facecolor='#F7FAFC', edgecolor='#4A5568', linewidth=2)
    ax.add_patch(session_box)
    ax.text(8, 2.7, 'Session State Management', fontsize=13, fontweight='bold',
            ha='center', va='center', color='#4A5568')

    ax.text(5, 2, 'Session ID', fontsize=10, fontweight='bold', ha='center', color=COLORS['text'])
    ax.text(5, 1.5, 'Isolates memory per agent', fontsize=9, ha='center', color=COLORS['arrow'])
    ax.text(5, 1, 'Enables multi-tenancy', fontsize=9, ha='center', color=COLORS['arrow'])

    ax.text(11, 2, 'Temporal Context', fontsize=10, fontweight='bold', ha='center', color=COLORS['text'])
    ax.text(11, 1.5, 'Bi-temporal versioning', fontsize=9, ha='center', color=COLORS['arrow'])
    ax.text(11, 1, 'Valid-time + Transaction-time', fontsize=9, ha='center', color=COLORS['arrow'])

    plt.tight_layout()
    plt.savefig('/mnt/projects/ww/diagrams/05_learning_consolidation.png', dpi=150,
                bbox_inches='tight', facecolor=COLORS['bg'])
    plt.close()
    print("Created: 05_learning_consolidation.png")


def create_claude_integration_diagram():
    """Create Claude Code integration diagram."""
    fig, ax = plt.subplots(figsize=(14, 10))
    ax.set_xlim(0, 14)
    ax.set_ylim(0, 10)
    ax.set_aspect('equal')
    ax.axis('off')
    ax.set_facecolor(COLORS['bg'])
    fig.patch.set_facecolor(COLORS['bg'])

    # Title
    ax.text(7, 9.5, 'Claude Code Integration', fontsize=22, fontweight='bold',
            ha='center', va='center', color=COLORS['text'])

    # Claude Code CLI
    claude_box = FancyBboxPatch((0.5, 7), 4, 2, boxstyle="round,pad=0.05",
                                 facecolor='#1A1A2E', edgecolor='#E0E0E0', linewidth=2)
    ax.add_patch(claude_box)
    ax.text(2.5, 8.3, 'Claude Code CLI', fontsize=12, fontweight='bold',
            ha='center', va='center', color='white')
    ax.text(2.5, 7.8, 'SessionStart Hook', fontsize=10,
            ha='center', va='center', color='#E94560')
    ax.text(2.5, 7.3, 'SessionEnd Hook', fontsize=10,
            ha='center', va='center', color='#0F3460')

    # MCP Server
    mcp_box = FancyBboxPatch((5.5, 7), 3.5, 2, boxstyle="round,pad=0.05",
                              facecolor=COLORS['mcp'], edgecolor='white', linewidth=2)
    ax.add_patch(mcp_box)
    ax.text(7.25, 8.3, 'MCP Server', fontsize=12, fontweight='bold',
            ha='center', va='center', color='white')
    ax.text(7.25, 7.8, 'ww-memory', fontsize=10,
            ha='center', va='center', color='white', alpha=0.9)
    ax.text(7.25, 7.3, 'Tools + Resources', fontsize=10,
            ha='center', va='center', color='white', alpha=0.9)

    # REST API
    api_box = FancyBboxPatch((10, 7), 3.5, 2, boxstyle="round,pad=0.05",
                              facecolor=COLORS['api'], edgecolor='white', linewidth=2)
    ax.add_patch(api_box)
    ax.text(11.75, 8.3, 'REST API', fontsize=12, fontweight='bold',
            ha='center', va='center', color='white')
    ax.text(11.75, 7.8, 'ww-api :8765', fontsize=10,
            ha='center', va='center', color='white', alpha=0.9)
    ax.text(11.75, 7.3, 'OpenAPI + SDK', fontsize=10,
            ha='center', va='center', color='white', alpha=0.9)

    # MCP Tools
    tools_box = FancyBboxPatch((0.5, 4), 6, 2.5, boxstyle="round,pad=0.05",
                                facecolor='#2D3748', edgecolor='white', linewidth=2)
    ax.add_patch(tools_box)
    ax.text(3.5, 6.1, 'MCP Tools Available', fontsize=11, fontweight='bold',
            ha='center', va='center', color='white')

    tools = [
        'ww_remember - Store memory',
        'ww_recall - Retrieve similar',
        'ww_context - Load project context',
        'ww_entity - Knowledge graph',
        'ww_skill - Procedural memory',
    ]
    for i, tool in enumerate(tools):
        ax.text(1, 5.5 - i * 0.35, tool, fontsize=9,
                ha='left', va='center', color='white', family='monospace')

    # Hooks
    hooks_box = FancyBboxPatch((7.5, 4), 6, 2.5, boxstyle="round,pad=0.05",
                                facecolor='#553C9A', edgecolor='white', linewidth=2)
    ax.add_patch(hooks_box)
    ax.text(10.5, 6.1, 'Hook Integration Points', fontsize=11, fontweight='bold',
            ha='center', va='center', color='white')

    hooks = [
        'SessionStart: Prime context from WW',
        'SessionEnd: Synthesize & store session',
        'ToolCall: Log tool invocations',
        'Error: Record failures for learning',
        'Completion: Store successful outcomes',
    ]
    for i, hook in enumerate(hooks):
        ax.text(8, 5.5 - i * 0.35, hook, fontsize=9,
                ha='left', va='center', color='white')

    # Configuration
    config_box = FancyBboxPatch((2, 1), 10, 2.5, boxstyle="round,pad=0.05",
                                 facecolor='#F7FAFC', edgecolor='#4A5568', linewidth=2)
    ax.add_patch(config_box)
    ax.text(7, 3.1, 'Configuration: ~/.claude/settings.json', fontsize=11, fontweight='bold',
            ha='center', va='center', color=COLORS['text'])

    config_text = '''{
  "mcpServers": {
    "ww-memory": {
      "command": "python",
      "args": ["-m", "ww.mcp.server"]
    }
  }
}'''
    ax.text(7, 1.8, config_text, fontsize=8, family='monospace',
            ha='center', va='center', color=COLORS['text'])

    # Arrows
    arrow_style = dict(arrowstyle='<->', color=COLORS['arrow'], lw=2)
    ax.annotate('', xy=(5.5, 8), xytext=(4.5, 8), arrowprops=arrow_style)
    ax.annotate('', xy=(10, 8), xytext=(9, 8), arrowprops=arrow_style)

    arrow_style2 = dict(arrowstyle='->', color=COLORS['arrow'], lw=2)
    ax.annotate('', xy=(3.5, 6.5), xytext=(5.5, 7), arrowprops=arrow_style2)
    ax.annotate('', xy=(10.5, 6.5), xytext=(9.5, 7), arrowprops=arrow_style2)

    plt.tight_layout()
    plt.savefig('/mnt/projects/ww/diagrams/06_claude_integration.png', dpi=150,
                bbox_inches='tight', facecolor=COLORS['bg'])
    plt.close()
    print("Created: 06_claude_integration.png")


if __name__ == '__main__':
    print("Generating World Weaver diagrams...")
    create_architecture_diagram()
    create_encoding_diagram()
    create_storage_diagram()
    create_retrieval_diagram()
    create_learning_diagram()
    create_claude_integration_diagram()
    print("\nAll diagrams generated in /mnt/projects/ww/diagrams/")
