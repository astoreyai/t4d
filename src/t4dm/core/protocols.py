"""
Protocol definitions for World Weaver.

These define the interfaces that storage and embedding providers must implement,
enabling provider-agnostic memory operations.
"""

from abc import abstractmethod
from typing import Any, Protocol, runtime_checkable
from uuid import UUID

from t4dm.core.types import Entity, Episode, Procedure, Relationship


@runtime_checkable
class EmbeddingProvider(Protocol):
    """Protocol for embedding generation providers."""

    @property
    @abstractmethod
    def dimension(self) -> int:
        """Return the embedding dimension."""
        ...

    @abstractmethod
    async def embed(self, texts: list[str]) -> list[list[float]]:
        """Generate embeddings for a batch of texts."""
        ...

    @abstractmethod
    async def embed_query(self, query: str) -> list[float]:
        """Generate embedding optimized for query (may differ from document embedding)."""
        ...


@runtime_checkable
class VectorStore(Protocol):
    """Protocol for vector storage providers. Implemented by T4DX adapter."""

    @abstractmethod
    async def create_collection(
        self,
        name: str,
        dimension: int,
        distance: str = "cosine",
    ) -> None:
        """Create a new vector collection."""
        ...

    @abstractmethod
    async def add(
        self,
        collection: str,
        ids: list[str],
        vectors: list[list[float]],
        payloads: list[dict[str, Any]],
    ) -> None:
        """Add vectors with payloads to collection."""
        ...

    @abstractmethod
    async def search(
        self,
        collection: str,
        vector: list[float],
        limit: int = 10,
        filter: dict[str, Any] | None = None,
    ) -> list[tuple[str, float, dict[str, Any]]]:
        """Search for similar vectors. Returns (id, score, payload) tuples."""
        ...

    @abstractmethod
    async def delete(
        self,
        collection: str,
        ids: list[str],
    ) -> None:
        """Delete vectors by ID."""
        ...

    @abstractmethod
    async def get(
        self,
        collection: str,
        ids: list[str],
    ) -> list[tuple[str, dict[str, Any]]]:
        """Get vectors by ID. Returns (id, payload) tuples."""
        ...


@runtime_checkable
class GraphStore(Protocol):
    """Protocol for graph storage providers. Implemented by T4DX adapter."""

    # Node operations
    @abstractmethod
    async def create_node(
        self,
        label: str,
        properties: dict[str, Any],
    ) -> str:
        """Create a node and return its ID."""
        ...

    @abstractmethod
    async def get_node(
        self,
        node_id: str,
        label: str | None = None,
    ) -> dict[str, Any] | None:
        """Get node by ID."""
        ...

    @abstractmethod
    async def update_node(
        self,
        node_id: str,
        properties: dict[str, Any],
        label: str | None = None,
    ) -> None:
        """Update node properties."""
        ...

    @abstractmethod
    async def delete_node(
        self,
        node_id: str,
        label: str | None = None,
    ) -> None:
        """Delete a node."""
        ...

    # Relationship operations
    @abstractmethod
    async def create_relationship(
        self,
        source_id: str,
        target_id: str,
        rel_type: str,
        properties: dict[str, Any],
    ) -> None:
        """Create a relationship between nodes."""
        ...

    @abstractmethod
    async def get_relationships(
        self,
        node_id: str,
        rel_type: str | None = None,
        direction: str = "both",  # "in", "out", "both"
    ) -> list[dict[str, Any]]:
        """Get relationships for a node."""
        ...

    @abstractmethod
    async def update_relationship(
        self,
        source_id: str,
        target_id: str,
        rel_type: str,
        properties: dict[str, Any],
    ) -> None:
        """Update relationship properties."""
        ...

    # Query operations
    @abstractmethod
    async def query(
        self,
        cypher: str,
        parameters: dict[str, Any] | None = None,
    ) -> list[dict[str, Any]]:
        """Execute a Cypher query."""
        ...

    @abstractmethod
    async def find_path(
        self,
        source_id: str,
        target_id: str,
        max_depth: int = 5,
    ) -> list[str] | None:
        """Find shortest path between nodes."""
        ...


# High-level Memory Store Protocols

@runtime_checkable
class EpisodicStore(Protocol):
    """Protocol for episodic memory storage."""

    @abstractmethod
    async def create_episode(self, episode: Episode) -> Episode:
        """Store a new episode."""
        ...

    @abstractmethod
    async def get_episode(self, episode_id: UUID) -> Episode | None:
        """Retrieve episode by ID."""
        ...

    @abstractmethod
    async def search_episodes(
        self,
        query_vector: list[float],
        limit: int = 10,
        session_filter: str | None = None,
        time_filter: tuple | None = None,
    ) -> list[tuple[Episode, float]]:
        """Search episodes by vector similarity."""
        ...

    @abstractmethod
    async def update_episode_access(
        self,
        episode_id: UUID,
        success: bool = True,
    ) -> None:
        """Update episode access statistics."""
        ...


@runtime_checkable
class SemanticStore(Protocol):
    """Protocol for semantic memory storage."""

    @abstractmethod
    async def create_entity(self, entity: Entity) -> Entity:
        """Store a new entity."""
        ...

    @abstractmethod
    async def get_entity(self, entity_id: UUID) -> Entity | None:
        """Retrieve entity by ID."""
        ...

    @abstractmethod
    async def search_entities(
        self,
        query_vector: list[float],
        limit: int = 10,
        entity_type: str | None = None,
    ) -> list[tuple[Entity, float]]:
        """Search entities by vector similarity."""
        ...

    @abstractmethod
    async def create_relationship(self, relationship: Relationship) -> None:
        """Create relationship between entities."""
        ...

    @abstractmethod
    async def strengthen_relationship(
        self,
        source_id: UUID,
        target_id: UUID,
        learning_rate: float = 0.1,
    ) -> float:
        """Strengthen relationship via Hebbian update."""
        ...

    @abstractmethod
    async def get_neighbors(
        self,
        entity_id: UUID,
        rel_type: str | None = None,
    ) -> list[tuple[Entity, Relationship]]:
        """Get neighboring entities with relationship info."""
        ...

    @abstractmethod
    async def supersede_entity(
        self,
        entity_id: UUID,
        new_summary: str,
        new_details: str | None = None,
    ) -> Entity:
        """Update entity with bi-temporal versioning."""
        ...


@runtime_checkable
class ProceduralStore(Protocol):
    """Protocol for procedural memory storage."""

    @abstractmethod
    async def create_procedure(self, procedure: Procedure) -> Procedure:
        """Store a new procedure."""
        ...

    @abstractmethod
    async def get_procedure(self, procedure_id: UUID) -> Procedure | None:
        """Retrieve procedure by ID."""
        ...

    @abstractmethod
    async def search_procedures(
        self,
        query_vector: list[float],
        limit: int = 5,
        domain: str | None = None,
    ) -> list[tuple[Procedure, float]]:
        """Search procedures by vector similarity."""
        ...

    @abstractmethod
    async def update_procedure_feedback(
        self,
        procedure_id: UUID,
        success: bool,
    ) -> Procedure:
        """Update procedure after execution."""
        ...

    @abstractmethod
    async def deprecate_procedure(
        self,
        procedure_id: UUID,
        consolidated_into: UUID | None = None,
    ) -> None:
        """Mark procedure as deprecated."""
        ...
