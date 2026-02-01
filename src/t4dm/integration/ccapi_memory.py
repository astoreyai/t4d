"""
World Weaver Memory Adapter for ccapi.

Implements the llm_agents Memory protocol using WW's tripartite memory system.
Enables ccapi agents to use World Weaver's episodic, semantic, and procedural
memory with neuro-symbolic learning.
"""

from __future__ import annotations

import asyncio
import json
import logging
from datetime import datetime
from pathlib import Path
from typing import Any

logger = logging.getLogger(__name__)


# Type stub for ccapi Message (avoid hard dependency)
class Message:
    """Stub for llm_agents.core.types.Message."""
    def __init__(
        self,
        role: str,
        content: str,
        name: str | None = None,
        tool_call_id: str | None = None,
        metadata: dict[str, Any] | None = None,
    ):
        self.role = role
        self.content = content
        self.name = name
        self.tool_call_id = tool_call_id
        self.metadata = metadata or {}
        self.timestamp = datetime.now()


class WWMemory:
    """
    World Weaver Memory adapter for ccapi.

    Provides the Memory protocol interface while using WW's:
    - Episodic memory for conversation history
    - Semantic memory for entity/concept storage
    - Procedural memory for learned skills
    - Learning hooks for adaptive retrieval

    Usage:
        from t4dm.integration.ccapi_memory import WWMemory

        memory = WWMemory(session_id="agent-session-001")
        memory.add(Message(role="user", content="How do I..."))
        results = memory.search("related query")
    """

    def __init__(
        self,
        session_id: str = "default",
        project: str = "",
        qdrant_url: str = "http://localhost:6333",
        neo4j_url: str | None = None,
        max_messages: int = 1000,
        enable_learning: bool = True,
    ):
        """
        Initialize WW Memory adapter.

        Args:
            session_id: Session identifier for isolation
            project: Project context
            qdrant_url: Qdrant vector store URL
            neo4j_url: Neo4j graph store URL (optional)
            max_messages: Maximum messages to keep
            enable_learning: Enable learning hooks
        """
        self.session_id = session_id
        self.project = project
        self.qdrant_url = qdrant_url
        self.neo4j_url = neo4j_url
        self.max_messages = max_messages
        self.enable_learning = enable_learning

        # Lazy-loaded WW components
        self._episodic = None
        self._semantic = None
        self._procedural = None
        self._embedder = None
        self._learning_collector = None

        # In-memory buffer for recent messages (before async commit)
        self._buffer: list[Message] = []
        self._initialized = False

    async def _ensure_initialized(self) -> None:
        """Lazy initialization of WW components."""
        if self._initialized:
            return

        try:
            from t4dm.embedding.bge_m3 import BGEM3Embedder
            from t4dm.memory.episodic import EpisodicMemory
            from t4dm.storage import T4DXVectorStore

            # Initialize embedder
            self._embedder = await BGEM3Embedder.create()

            # Initialize Qdrant store
            vector_store = await T4DXVectorStore.create(url=self.qdrant_url)

            # Initialize episodic memory
            self._episodic = EpisodicMemory(
                vector_store=vector_store,
                embedder=self._embedder,
            )

            # Initialize learning collector if enabled
            if self.enable_learning:
                try:
                    from t4dm.learning.collector import get_collector
                    self._learning_collector = get_collector()
                except ImportError:
                    logger.warning("Learning module not available")

            self._initialized = True
            logger.info(f"WWMemory initialized for session {self.session_id}")

        except Exception as e:
            logger.error(f"Failed to initialize WWMemory: {e}")
            raise

    def add(self, message: Message) -> None:
        """
        Add a message to memory.

        Stores in buffer immediately, commits to WW asynchronously.

        Args:
            message: Message to store
        """
        self._buffer.append(message)

        # Async commit to WW
        if self._initialized:
            asyncio.create_task(self._commit_message(message))

    def add_many(self, messages: list[Message]) -> None:
        """Add multiple messages."""
        for msg in messages:
            self.add(msg)

    async def _commit_message(self, message: Message) -> None:
        """Commit message to WW episodic memory."""
        try:
            from t4dm.core.types import Episode, Outcome

            episode = Episode(
                content=f"[{message.role}] {message.content}",
                session_id=self.session_id,
                outcome=Outcome.SUCCESS if message.role == "assistant" else Outcome.NEUTRAL,
                emotional_valence=0.0,
                metadata={
                    "role": message.role,
                    "name": message.name,
                    "tool_call_id": message.tool_call_id,
                    **message.metadata,
                },
            )

            await self._episodic.store(episode)

        except Exception as e:
            logger.warning(f"Failed to commit message to WW: {e}")

    def get_messages(self, limit: int | None = None) -> list[Message]:
        """
        Get recent messages from memory.

        Args:
            limit: Maximum messages to return

        Returns:
            List of messages, oldest to newest
        """
        messages = self._buffer.copy()
        if limit:
            messages = messages[-limit:]
        return messages

    def search(self, query: str, limit: int = 5) -> list[Message]:
        """
        Search memory for relevant messages (synchronous version).

        Uses WW's hybrid retrieval if initialized,
        falls back to simple text matching otherwise.

        ASYNC-001 FIX: For async contexts, use search_async() instead.
        This method is for backward compatibility with sync code.

        Args:
            query: Search query
            limit: Maximum results

        Returns:
            Relevant messages ordered by relevance
        """
        if not self._initialized:
            # Fallback: simple text search
            return self._simple_search(query, limit)

        # ASYNC-001 FIX: Check if we're in an async context
        try:
            loop = asyncio.get_running_loop()
            # We're in an async context - warn and return fallback
            logger.warning(
                "search() called from async context. Use search_async() instead. "
                "Falling back to simple search."
            )
            return self._simple_search(query, limit)
        except RuntimeError:
            # No running loop - safe to use run_until_complete
            pass

        # Create new event loop for sync context
        try:
            loop = asyncio.new_event_loop()
            asyncio.set_event_loop(loop)
            try:
                return loop.run_until_complete(self._async_search(query, limit))
            finally:
                loop.close()
        except Exception as e:
            logger.warning(f"Async search failed: {e}, using fallback")
            return self._simple_search(query, limit)

    async def search_async(self, query: str, limit: int = 5) -> list[Message]:
        """
        Search memory for relevant messages (async version).

        ASYNC-001 FIX: Proper async interface for async contexts.

        Args:
            query: Search query
            limit: Maximum results

        Returns:
            Relevant messages ordered by relevance
        """
        if not self._initialized:
            await self._ensure_initialized()

        return await self._async_search(query, limit)

    async def _async_search(self, query: str, limit: int) -> list[Message]:
        """Async search using WW retrieval."""
        await self._ensure_initialized()

        try:
            results = await self._episodic.recall(
                query=query,
                limit=limit,
                session_filter=self.session_id,
            )

            # Emit learning event if enabled
            if self._learning_collector:
                from t4dm.learning.events import MemoryType
                from t4dm.learning.hooks import emit_retrieval_event

                emit_retrieval_event(
                    query=query,
                    memory_type=MemoryType.EPISODIC,
                    results=[
                        {
                            "id": str(r.item.id),
                            "score": r.score,
                            "components": r.components,
                        }
                        for r in results
                    ],
                    session_id=self.session_id,
                    project=self.project,
                )

            # Convert to Message format
            messages = []
            for result in results:
                content = result.item.content
                # Parse role from content if stored as "[role] content"
                role = "assistant"
                if content.startswith("["):
                    bracket_end = content.find("]")
                    if bracket_end > 0:
                        role = content[1:bracket_end]
                        content = content[bracket_end + 2:]

                messages.append(Message(
                    role=role,
                    content=content,
                    metadata={"score": result.score, "id": str(result.item.id)},
                ))

            return messages

        except Exception as e:
            logger.error(f"WW search failed: {e}")
            return self._simple_search(query, limit)

    def _simple_search(self, query: str, limit: int) -> list[Message]:
        """Simple text-based search fallback."""
        query_lower = query.lower()
        scored = []

        for msg in self._buffer:
            if query_lower in msg.content.lower():
                score = msg.content.lower().count(query_lower)
                scored.append((score, msg))

        scored.sort(key=lambda x: x[0], reverse=True)
        return [msg for _, msg in scored[:limit]]

    def clear(self) -> None:
        """Clear all messages from memory."""
        self._buffer.clear()
        logger.info(f"WWMemory cleared for session {self.session_id}")

    def save(self, path: str) -> None:
        """
        Save memory to file.

        Args:
            path: Path to save to
        """
        data = {
            "session_id": self.session_id,
            "project": self.project,
            "messages": [
                {
                    "role": msg.role,
                    "content": msg.content,
                    "name": msg.name,
                    "tool_call_id": msg.tool_call_id,
                    "metadata": msg.metadata,
                    "timestamp": msg.timestamp.isoformat(),
                }
                for msg in self._buffer
            ],
        }

        Path(path).write_text(json.dumps(data, indent=2))
        logger.info(f"WWMemory saved to {path}")

    def load(self, path: str) -> None:
        """
        Load memory from file.

        Args:
            path: Path to load from
        """
        data = json.loads(Path(path).read_text())

        self.session_id = data.get("session_id", self.session_id)
        self.project = data.get("project", self.project)

        self._buffer.clear()
        for msg_data in data.get("messages", []):
            msg = Message(
                role=msg_data["role"],
                content=msg_data["content"],
                name=msg_data.get("name"),
                tool_call_id=msg_data.get("tool_call_id"),
                metadata=msg_data.get("metadata", {}),
            )
            msg.timestamp = datetime.fromisoformat(msg_data["timestamp"])
            self._buffer.append(msg)

        logger.info(f"WWMemory loaded from {path}: {len(self._buffer)} messages")

    def __len__(self) -> int:
        """Get number of messages."""
        return len(self._buffer)

    def get_context(
        self,
        max_messages: int | None = None,
        max_tokens: int | None = None,
        include_system: bool = True,
    ) -> list[Message]:
        """
        Get messages formatted for LLM context.

        Args:
            max_messages: Maximum messages
            max_tokens: Maximum tokens (estimated)
            include_system: Include system messages

        Returns:
            Messages for context
        """
        messages = self.get_messages(limit=max_messages)

        if not include_system:
            messages = [m for m in messages if m.role != "system"]

        if max_tokens:
            total = 0
            trimmed = []
            for msg in reversed(messages):
                msg_tokens = len(msg.content) // 4
                if total + msg_tokens > max_tokens:
                    break
                trimmed.insert(0, msg)
                total += msg_tokens
            return trimmed

        return messages

    def get_last_message(self) -> Message | None:
        """Get most recent message."""
        return self._buffer[-1] if self._buffer else None

    def get_messages_by_role(self, role: str) -> list[Message]:
        """Get messages with specific role."""
        return [m for m in self._buffer if m.role == role]


# Factory function for convenience
def create_ww_memory(
    session_id: str = "default",
    **kwargs,
) -> WWMemory:
    """
    Create a WWMemory instance.

    Args:
        session_id: Session identifier
        **kwargs: Additional configuration

    Returns:
        WWMemory instance
    """
    return WWMemory(session_id=session_id, **kwargs)


__all__ = ["Message", "WWMemory", "create_ww_memory"]
