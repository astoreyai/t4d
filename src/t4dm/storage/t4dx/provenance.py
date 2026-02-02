"""Forward/backward lineage tracing over T4DX provenance edges."""

from __future__ import annotations

from collections import deque
from dataclasses import dataclass
from typing import TYPE_CHECKING

from t4dm.storage.t4dx.types import EdgeType

if TYPE_CHECKING:
    from t4dm.storage.t4dx.engine import T4DXEngine

# Edge types used for provenance tracing
_PROVENANCE_TYPES = frozenset({
    EdgeType.DERIVED_FROM.value,
    EdgeType.SOURCE_OF.value,
    EdgeType.MERGED_FROM.value,
    EdgeType.SUPERSEDES.value,
})


@dataclass(slots=True)
class ProvenanceNode:
    """A node in a provenance trace."""

    item_id: bytes
    depth: int
    edge_type: str
    parent_id: bytes | None


class ProvenanceTracer:
    """Traces forward and backward lineage through provenance edges."""

    def __init__(self, engine: T4DXEngine) -> None:
        self._engine = engine

    def forward_trace(self, item_id: bytes, max_depth: int = 5) -> list[ProvenanceNode]:
        """Follow provenance edges forward (outgoing) from item_id via BFS."""
        return self._bfs(item_id, direction="out", max_depth=max_depth)

    def backward_trace(self, item_id: bytes, max_depth: int = 5) -> list[ProvenanceNode]:
        """Follow provenance edges backward (incoming) to item_id via BFS."""
        return self._bfs(item_id, direction="in", max_depth=max_depth)

    def lineage_graph(self, item_id: bytes) -> dict:
        """Return full lineage as an adjacency dict.

        Returns dict mapping hex(item_id) -> list of {target, edge_type, direction}.
        """
        forward = self.forward_trace(item_id)
        backward = self.backward_trace(item_id)

        adj: dict[str, list[dict]] = {}
        for node in forward:
            if node.parent_id is not None:
                src = node.parent_id.hex()
                adj.setdefault(src, []).append({
                    "target": node.item_id.hex(),
                    "edge_type": node.edge_type,
                    "direction": "forward",
                })
        for node in backward:
            if node.parent_id is not None:
                src = node.item_id.hex()
                adj.setdefault(src, []).append({
                    "target": node.parent_id.hex(),
                    "edge_type": node.edge_type,
                    "direction": "backward",
                })
        return adj

    def _bfs(
        self, start_id: bytes, direction: str, max_depth: int,
    ) -> list[ProvenanceNode]:
        visited: set[bytes] = {start_id}
        queue: deque[tuple[bytes, int, bytes | None]] = deque()
        queue.append((start_id, 0, None))
        results: list[ProvenanceNode] = []

        while queue:
            current_id, depth, parent_id = queue.popleft()
            if depth > 0:
                # We don't know the edge_type from the queue entry directly,
                # so we stored it when adding to the queue. Use a different approach.
                pass
            if depth >= max_depth:
                continue

            edges = self._engine.traverse(current_id, edge_type=None, direction=direction)
            for edge in edges:
                if edge.edge_type not in _PROVENANCE_TYPES:
                    continue
                neighbor = edge.target_id if direction == "out" else edge.source_id
                if neighbor in visited:
                    continue
                visited.add(neighbor)
                node = ProvenanceNode(
                    item_id=neighbor,
                    depth=depth + 1,
                    edge_type=edge.edge_type,
                    parent_id=current_id,
                )
                results.append(node)
                queue.append((neighbor, depth + 1, current_id))

        return results
