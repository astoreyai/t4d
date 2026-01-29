"""Pydantic models for the unified graph schema."""

from __future__ import annotations

from datetime import datetime
from enum import Enum
from typing import Any

from pydantic import BaseModel, Field


class NodeType(str, Enum):
    PROCESS = "process"
    DECISION = "decision"
    STORE = "store"
    STATE = "state"
    PARTICIPANT = "participant"
    CLASS = "class"
    START_END = "start_end"
    SUBROUTINE = "subroutine"
    CYLINDER = "cylinder"
    TRAPEZOID = "trapezoid"
    UNKNOWN = "unknown"


class EdgeType(str, Enum):
    SOLID = "solid"
    DOTTED = "dotted"
    THICK = "thick"


class DiagramType(str, Enum):
    FLOWCHART = "flowchart"
    STATE = "stateDiagram"
    SEQUENCE = "sequenceDiagram"
    CLASS = "classDiagram"
    UNKNOWN = "unknown"


class GraphNode(BaseModel):
    id: str
    label: str = ""
    node_type: NodeType = NodeType.PROCESS
    subgraph: str | None = None
    source_file: str = ""
    diagram_type: DiagramType = DiagramType.FLOWCHART
    style: dict[str, str] = Field(default_factory=dict)
    metadata: dict[str, Any] = Field(default_factory=dict)


class GraphEdge(BaseModel):
    source: str
    target: str
    label: str = ""
    edge_type: EdgeType = EdgeType.SOLID
    weight: float = 1.0
    source_file: str = ""


class Subgraph(BaseModel):
    id: str
    label: str = ""
    parent: str | None = None
    children: list[str] = Field(default_factory=list)


class GraphMetadata(BaseModel):
    node_count: int = 0
    edge_count: int = 0
    subgraph_count: int = 0
    diagram_count: int = 0
    parsed_at: str = Field(default_factory=lambda: datetime.now().isoformat())


class UnifiedGraph(BaseModel):
    nodes: list[GraphNode] = Field(default_factory=list)
    edges: list[GraphEdge] = Field(default_factory=list)
    subgraphs: list[Subgraph] = Field(default_factory=list)
    metadata: GraphMetadata = Field(default_factory=GraphMetadata)

    def node_by_id(self, node_id: str) -> GraphNode | None:
        for n in self.nodes:
            if n.id == node_id:
                return n
        return None

    def update_metadata(self) -> None:
        self.metadata.node_count = len(self.nodes)
        self.metadata.edge_count = len(self.edges)
        self.metadata.subgraph_count = len(self.subgraphs)
