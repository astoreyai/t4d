"""
Simplified Memory API for World Weaver.

Provides a clean, user-friendly interface to World Weaver memory:

    from t4dm import memory

    # Store content
    await memory.store("User discussed Python decorators")

    # Recall similar memories
    results = await memory.recall("decorators")

    # Use context manager for explicit session
    async with memory.session("my-project") as m:
        await m.store("Project-specific knowledge")
"""

import asyncio
from contextlib import asynccontextmanager
from datetime import datetime
from typing import Any

from t4dm.core.config import get_settings
from t4dm.core.services import get_services
from t4dm.core.types import Domain, Entity, EntityType, Episode, Procedure


class MemoryResult:
    """A single memory retrieval result."""

    def __init__(
        self,
        content: str,
        memory_type: str,
        score: float = 0.0,
        id: str | None = None,
        metadata: dict[str, Any] | None = None,
    ):
        self.content = content
        self.memory_type = memory_type
        self.score = score
        self.id = id
        self.metadata = metadata or {}

    def __repr__(self) -> str:
        return f"MemoryResult(type={self.memory_type}, score={self.score:.3f}, content={self.content[:50]}...)"


class Memory:
    """
    Simplified interface to World Weaver memory systems.

    Provides async methods for storing and retrieving memories
    across episodic, semantic, and procedural memory stores.
    """

    def __init__(self, session_id: str | None = None):
        """
        Initialize Memory interface.

        Args:
            session_id: Optional session identifier. If None, uses default from settings.
        """
        self._session_id = session_id
        self._services_cache = None

    @property
    def session_id(self) -> str:
        """Get the current session ID."""
        if self._session_id:
            return self._session_id
        return get_settings().session_id

    async def _get_services(self):
        """Get or initialize memory services."""
        if self._services_cache is None:
            self._services_cache = await get_services(self.session_id)
        return self._services_cache

    # =========================================================================
    # Store Methods
    # =========================================================================

    async def store(
        self,
        content: str,
        importance: float = 0.5,
        tags: list[str] | None = None,
        metadata: dict[str, Any] | None = None,
    ) -> str:
        """
        Store content as an episodic memory.

        This is the primary method for storing information. Content is
        automatically timestamped and made searchable via embeddings.

        Args:
            content: The text content to store
            importance: Importance score 0-1 (affects consolidation priority)
            tags: Optional tags for categorization
            metadata: Optional additional metadata

        Returns:
            The ID of the stored episode

        Example:
            >>> await memory.store("User prefers dark mode interfaces")
            'ep_abc123'
        """
        episodic, _, _ = await self._get_services()

        episode = Episode(
            session_id=self.session_id,
            content=content,
            timestamp=datetime.utcnow(),
            emotional_valence=importance,
        )

        result = await episodic.add_episode(episode)
        return str(result.id)

    async def store_episode(
        self,
        content: str,
        importance: float = 0.5,
        tags: list[str] | None = None,
    ) -> str:
        """
        Store an episodic memory (explicit alias for store).

        Args:
            content: Episode content
            importance: Importance score 0-1
            tags: Optional tags

        Returns:
            Episode ID
        """
        return await self.store(content, importance=importance, tags=tags)

    async def store_entity(
        self,
        name: str,
        description: str | None = None,
        entity_type: str = "concept",
        properties: dict[str, Any] | None = None,
    ) -> str:
        """
        Store a semantic entity (concept, person, place, etc.).

        Entities are structured knowledge that can be linked together
        in the semantic graph.

        Args:
            name: Entity name
            description: Entity description/summary
            entity_type: Type of entity (concept, person, project, tool, technique, fact)
            properties: Additional structured properties

        Returns:
            Entity ID

        Example:
            >>> await memory.store_entity(
            ...     "Python",
            ...     description="A high-level programming language",
            ...     entity_type="concept"
            ... )
            'ent_xyz789'
        """
        _, semantic, _ = await self._get_services()

        # Map string type to enum
        type_map = {
            "concept": EntityType.CONCEPT,
            "person": EntityType.PERSON,
            "project": EntityType.PROJECT,
            "tool": EntityType.TOOL,
            "technique": EntityType.TECHNIQUE,
            "fact": EntityType.FACT,
        }
        etype = type_map.get(entity_type.lower(), EntityType.CONCEPT)

        entity = Entity(
            name=name,
            entity_type=etype,
            summary=description or name,
            details=description if description and len(description) > 200 else None,
        )

        result = await semantic.add_entity(entity)
        return str(result.id)

    async def store_skill(
        self,
        name: str,
        script: str,
        description: str | None = None,
        domain: str = "general",
    ) -> str:
        """
        Store a procedural skill (how-to knowledge).

        Skills represent executable knowledge - patterns, scripts,
        or procedures that can be recalled and applied.

        Args:
            name: Skill name
            script: The script/procedure content
            description: Optional description
            domain: Domain (coding, research, trading, devops, writing)

        Returns:
            Skill ID

        Example:
            >>> await memory.store_skill(
            ...     "git_commit_pattern",
            ...     script="git add -A && git commit -m '...'",
            ...     domain="coding"
            ... )
            'sk_def456'
        """
        _, _, procedural = await self._get_services()

        # Map string domain to enum
        domain_map = {
            "coding": Domain.CODING,
            "research": Domain.RESEARCH,
            "trading": Domain.TRADING,
            "devops": Domain.DEVOPS,
            "writing": Domain.WRITING,
        }
        dom = domain_map.get(domain.lower(), Domain.CODING)

        procedure = Procedure(
            name=name,
            domain=dom,
            script=script,
        )

        result = await procedural.add_skill(procedure)
        return str(result.id)

    # =========================================================================
    # Recall Methods
    # =========================================================================

    async def recall(
        self,
        query: str,
        limit: int = 5,
        memory_types: list[str] | None = None,
    ) -> list[MemoryResult]:
        """
        Recall memories matching a query across all memory types.

        This is the primary retrieval method. It searches episodic,
        semantic, and procedural memories and returns unified results.

        Args:
            query: Search query (natural language)
            limit: Maximum results per memory type
            memory_types: Optional filter to specific types
                          ("episodic", "semantic", "procedural")

        Returns:
            List of MemoryResult objects sorted by relevance

        Example:
            >>> results = await memory.recall("Python decorators")
            >>> for r in results:
            ...     print(f"{r.memory_type}: {r.content[:50]}")
        """
        types = memory_types or ["episodic", "semantic", "procedural"]
        results = []

        episodic, semantic, procedural = await self._get_services()

        if "episodic" in types:
            episodes = await episodic.recall_similar(query, limit=limit)
            for ep in episodes:
                results.append(MemoryResult(
                    content=ep.item.content,
                    memory_type="episodic",
                    score=ep.score,
                    id=str(ep.item.id),
                    metadata={
                        "timestamp": str(ep.item.timestamp),
                        "valence": ep.item.emotional_valence,
                    },
                ))

        if "semantic" in types:
            entities = await semantic.search_similar(query, limit=limit)
            for ent in entities:
                results.append(MemoryResult(
                    content=f"{ent.name}: {ent.summary}" if hasattr(ent, 'summary') else str(ent),
                    memory_type="semantic",
                    score=0.0,  # Semantic search doesn't always return scores
                    id=str(ent.id) if hasattr(ent, 'id') else None,
                    metadata={
                        "entity_type": str(ent.entity_type) if hasattr(ent, 'entity_type') else None,
                    },
                ))

        if "procedural" in types:
            skills = await procedural.find_relevant_skills(query, limit=limit)
            for skill in skills:
                results.append(MemoryResult(
                    content=f"{skill.name}: {skill.script[:200] if skill.script else ''}",
                    memory_type="procedural",
                    score=0.0,
                    id=str(skill.id) if hasattr(skill, 'id') else None,
                    metadata={
                        "domain": str(skill.domain) if hasattr(skill, 'domain') else None,
                    },
                ))

        # Sort by score (descending)
        results.sort(key=lambda x: x.score, reverse=True)
        return results

    async def recall_episodes(
        self,
        query: str,
        limit: int = 5,
    ) -> list[MemoryResult]:
        """
        Recall episodic memories matching a query.

        Args:
            query: Search query
            limit: Maximum results

        Returns:
            List of episodic MemoryResult objects
        """
        return await self.recall(query, limit=limit, memory_types=["episodic"])

    async def recall_entities(
        self,
        query: str,
        limit: int = 5,
    ) -> list[MemoryResult]:
        """
        Recall semantic entities matching a query.

        Args:
            query: Search query
            limit: Maximum results

        Returns:
            List of semantic MemoryResult objects
        """
        return await self.recall(query, limit=limit, memory_types=["semantic"])

    async def recall_skills(
        self,
        query: str,
        limit: int = 5,
    ) -> list[MemoryResult]:
        """
        Recall procedural skills matching a query.

        Args:
            query: Search query
            limit: Maximum results

        Returns:
            List of procedural MemoryResult objects
        """
        return await self.recall(query, limit=limit, memory_types=["procedural"])

    async def get_recent(self, limit: int = 10) -> list[MemoryResult]:
        """
        Get recent episodic memories.

        Args:
            limit: Maximum results

        Returns:
            List of recent MemoryResult objects
        """
        episodic, _, _ = await self._get_services()
        episodes = await episodic.get_recent_episodes(limit=limit)

        return [
            MemoryResult(
                content=ep.content,
                memory_type="episodic",
                score=1.0,
                id=str(ep.id),
                metadata={
                    "timestamp": str(ep.timestamp),
                    "valence": ep.emotional_valence,
                },
            )
            for ep in episodes
        ]

    # =========================================================================
    # Session Management
    # =========================================================================

    @asynccontextmanager
    async def session(self, session_id: str):
        """
        Context manager for working with a specific session.

        Args:
            session_id: Session identifier

        Yields:
            Memory instance bound to the session

        Example:
            >>> async with memory.session("project-alpha") as m:
            ...     await m.store("Project-specific knowledge")
        """
        session_memory = Memory(session_id=session_id)
        try:
            yield session_memory
        finally:
            # Clear cached services on exit
            session_memory._services_cache = None


# Default memory instance (uses settings-based session)
memory = Memory()


# Convenience function for sync contexts
def run_sync(coro):
    """Run an async coroutine from sync context."""
    return asyncio.get_event_loop().run_until_complete(coro)


# =============================================================================
# Module-level convenience functions
# =============================================================================


async def store(
    content: str,
    importance: float = 0.5,
    tags: list[str] | None = None,
    metadata: dict[str, Any] | None = None,
) -> str:
    """Store content in memory. See Memory.store()."""
    return await memory.store(content, importance=importance, tags=tags, metadata=metadata)


async def recall(
    query: str,
    limit: int = 5,
    memory_types: list[str] | None = None,
) -> list[MemoryResult]:
    """Recall memories matching query. See Memory.recall()."""
    return await memory.recall(query, limit=limit, memory_types=memory_types)


async def store_episode(content: str, importance: float = 0.5, tags: list[str] | None = None) -> str:
    """Store an episodic memory. See Memory.store_episode()."""
    return await memory.store_episode(content, importance=importance, tags=tags)


async def store_entity(
    name: str,
    description: str | None = None,
    entity_type: str = "concept",
    properties: dict[str, Any] | None = None,
) -> str:
    """Store a semantic entity. See Memory.store_entity()."""
    return await memory.store_entity(name, description=description, entity_type=entity_type, properties=properties)


async def store_skill(
    name: str,
    script: str,
    description: str | None = None,
    domain: str = "general",
) -> str:
    """Store a procedural skill. See Memory.store_skill()."""
    return await memory.store_skill(name, script, description=description, domain=domain)


async def recall_episodes(query: str, limit: int = 5) -> list[MemoryResult]:
    """Recall episodic memories. See Memory.recall_episodes()."""
    return await memory.recall_episodes(query, limit=limit)


async def recall_entities(query: str, limit: int = 5) -> list[MemoryResult]:
    """Recall semantic entities. See Memory.recall_entities()."""
    return await memory.recall_entities(query, limit=limit)


async def recall_skills(query: str, limit: int = 5) -> list[MemoryResult]:
    """Recall procedural skills. See Memory.recall_skills()."""
    return await memory.recall_skills(query, limit=limit)


async def get_recent(limit: int = 10) -> list[MemoryResult]:
    """Get recent episodic memories. See Memory.get_recent()."""
    return await memory.get_recent(limit=limit)
