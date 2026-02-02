"""
T4DM SDK Client.

Provides synchronous and asynchronous clients for the REST API.
"""

import logging
from datetime import datetime
from uuid import UUID

import httpx

from t4dm.sdk.models import (
    ActivationResult,
    Entity,
    Episode,
    EpisodeContext,
    HealthStatus,
    MemoryStats,
    RecallResult,
    Relationship,
    Skill,
    Step,
)

logger = logging.getLogger(__name__)


class WorldWeaverError(Exception):
    """Base exception for T4DM SDK errors."""

    def __init__(self, message: str, status_code: int = None, response: dict = None):
        super().__init__(message)
        self.status_code = status_code
        self.response = response


class ConnectionError(WorldWeaverError):
    """Connection to T4DM API failed."""



class NotFoundError(WorldWeaverError):
    """Requested resource not found."""



class RateLimitError(WorldWeaverError):
    """Rate limit exceeded."""

    def __init__(self, message: str, retry_after: int = None):
        super().__init__(message, status_code=429)
        self.retry_after = retry_after


class AsyncWorldWeaverClient:
    """
    Async client for T4DM REST API.

    Example:
        async with AsyncWorldWeaverClient() as ww:
            episode = await ww.create_episode("Learned about Python decorators")
            results = await ww.recall_episodes("decorators")
    """

    def __init__(
        self,
        base_url: str = "http://localhost:8765",
        session_id: str | None = None,
        timeout: float = 30.0,
    ):
        """
        Initialize the async client.

        Args:
            base_url: Base URL of the T4DM API
            session_id: Session ID for memory isolation
            timeout: Request timeout in seconds
        """
        self.base_url = base_url.rstrip("/")
        self.session_id = session_id
        self.timeout = timeout
        self._client: httpx.AsyncClient | None = None

    async def __aenter__(self) -> "AsyncWorldWeaverClient":
        """Enter async context manager."""
        await self.connect()
        return self

    async def __aexit__(self, exc_type, exc_val, exc_tb):
        """Exit async context manager."""
        await self.close()

    async def connect(self):
        """Initialize the HTTP client."""
        headers = {}
        if self.session_id:
            headers["X-Session-ID"] = self.session_id

        self._client = httpx.AsyncClient(
            base_url=self.base_url,
            headers=headers,
            timeout=self.timeout,
        )

    async def close(self):
        """Close the HTTP client."""
        if self._client:
            await self._client.aclose()
            self._client = None

    def _get_client(self) -> httpx.AsyncClient:
        """Get the HTTP client, raising if not connected."""
        if not self._client:
            raise WorldWeaverError("Client not connected. Use 'async with' or call connect()")
        return self._client

    async def _request(self, method: str, path: str, **kwargs) -> dict:
        """Make an HTTP request with error handling."""
        client = self._get_client()
        try:
            response = await client.request(method, f"/api/v1{path}", **kwargs)

            if response.status_code == 404:
                raise NotFoundError(
                    f"Resource not found: {path}",
                    status_code=404,
                    response=response.json() if response.content else None,
                )
            if response.status_code == 429:
                retry_after = response.headers.get("Retry-After")
                raise RateLimitError(
                    "Rate limit exceeded",
                    retry_after=int(retry_after) if retry_after else None,
                )
            if response.status_code >= 400:
                raise WorldWeaverError(
                    f"API error: {response.status_code}",
                    status_code=response.status_code,
                    response=response.json() if response.content else None,
                )

            return response.json() if response.content else {}
        except httpx.ConnectError as e:
            raise ConnectionError(f"Failed to connect to {self.base_url}: {e}")
        except httpx.TimeoutException as e:
            raise WorldWeaverError(f"Request timed out: {e}")

    # Health & System

    async def health(self) -> HealthStatus:
        """Check API health status."""
        data = await self._request("GET", "/health")
        return HealthStatus(**data)

    async def stats(self) -> MemoryStats:
        """Get memory statistics."""
        data = await self._request("GET", "/stats")
        return MemoryStats(**data)

    async def consolidate(self, deep: bool = False) -> dict:
        """Trigger memory consolidation."""
        data = await self._request("POST", "/consolidate", json={"deep": deep})
        return data

    # Episodic Memory

    async def create_episode(
        self,
        content: str,
        project: str | None = None,
        file: str | None = None,
        tool: str | None = None,
        outcome: str = "neutral",
        emotional_valence: float = 0.5,
        timestamp: datetime | None = None,
    ) -> Episode:
        """Create a new episode."""
        data = await self._request(
            "POST",
            "/episodes",
            json={
                "content": content,
                "project": project,
                "file": file,
                "tool": tool,
                "outcome": outcome,
                "emotional_valence": emotional_valence,
                "timestamp": timestamp.isoformat() if timestamp else None,
            },
        )
        data["context"] = EpisodeContext(**data.get("context", {}))
        return Episode(**data)

    async def get_episode(self, episode_id: UUID) -> Episode:
        """Get an episode by ID."""
        data = await self._request("GET", f"/episodes/{episode_id}")
        data["context"] = EpisodeContext(**data.get("context", {}))
        return Episode(**data)

    async def list_episodes(
        self,
        page: int = 1,
        page_size: int = 20,
        project: str | None = None,
        outcome: str | None = None,
    ) -> tuple[list[Episode], int]:
        """List episodes with pagination."""
        params = {"page": page, "page_size": page_size}
        if project:
            params["project"] = project
        if outcome:
            params["outcome"] = outcome

        data = await self._request("GET", "/episodes", params=params)
        episodes = [
            Episode(**{**e, "context": EpisodeContext(**e.get("context", {}))})
            for e in data["episodes"]
        ]
        return episodes, data["total"]

    async def recall_episodes(
        self,
        query: str,
        limit: int = 10,
        min_similarity: float = 0.5,
        project: str | None = None,
        outcome: str | None = None,
    ) -> RecallResult:
        """Semantic search for episodes."""
        data = await self._request(
            "POST",
            "/episodes/recall",
            json={
                "query": query,
                "limit": limit,
                "min_similarity": min_similarity,
                "project": project,
                "outcome": outcome,
            },
        )
        episodes = [
            Episode(**{**e, "context": EpisodeContext(**e.get("context", {}))})
            for e in data["episodes"]
        ]
        return RecallResult(query=data["query"], episodes=episodes, scores=data["scores"])

    async def delete_episode(self, episode_id: UUID) -> None:
        """Delete an episode."""
        await self._request("DELETE", f"/episodes/{episode_id}")

    async def mark_important(self, episode_id: UUID, importance: float = 1.0) -> Episode:
        """Mark an episode as important."""
        data = await self._request(
            "POST",
            f"/episodes/{episode_id}/mark-important",
            params={"importance": importance},
        )
        data["context"] = EpisodeContext(**data.get("context", {}))
        return Episode(**data)

    # Semantic Memory

    async def create_entity(
        self,
        name: str,
        entity_type: str,
        summary: str,
        details: str | None = None,
        source: str | None = None,
    ) -> Entity:
        """Create a new entity."""
        data = await self._request(
            "POST",
            "/entities",
            json={
                "name": name,
                "entity_type": entity_type,
                "summary": summary,
                "details": details,
                "source": source,
            },
        )
        return Entity(**data)

    async def get_entity(self, entity_id: UUID) -> Entity:
        """Get an entity by ID."""
        data = await self._request("GET", f"/entities/{entity_id}")
        return Entity(**data)

    async def list_entities(
        self,
        entity_type: str | None = None,
        limit: int = 50,
    ) -> list[Entity]:
        """List entities with optional type filtering."""
        params = {"limit": limit}
        if entity_type:
            params["entity_type"] = entity_type

        data = await self._request("GET", "/entities", params=params)
        return [Entity(**e) for e in data["entities"]]

    async def create_relation(
        self,
        source_id: UUID,
        target_id: UUID,
        relation_type: str,
        weight: float = 0.1,
    ) -> Relationship:
        """Create a relationship between entities."""
        data = await self._request(
            "POST",
            "/entities/relations",
            json={
                "source_id": str(source_id),
                "target_id": str(target_id),
                "relation_type": relation_type,
                "weight": weight,
            },
        )
        return Relationship(**data)

    async def recall_entities(
        self,
        query: str,
        limit: int = 10,
        entity_types: list[str] | None = None,
    ) -> list[Entity]:
        """Semantic search for entities."""
        data = await self._request(
            "POST",
            "/entities/recall",
            json={
                "query": query,
                "limit": limit,
                "entity_types": entity_types,
            },
        )
        return [Entity(**e) for e in data["entities"]]

    async def spread_activation(
        self,
        entity_id: UUID,
        depth: int = 2,
        threshold: float = 0.1,
    ) -> ActivationResult:
        """Perform spreading activation from an entity."""
        data = await self._request(
            "POST",
            "/entities/spread-activation",
            json={
                "entity_id": str(entity_id),
                "depth": depth,
                "threshold": threshold,
            },
        )
        return ActivationResult(
            entities=[Entity(**e) for e in data["entities"]],
            activations=data["activations"],
            paths=data["paths"],
        )

    async def supersede_entity(
        self,
        entity_id: UUID,
        name: str,
        entity_type: str,
        summary: str,
        details: str | None = None,
    ) -> Entity:
        """Supersede an entity with new information."""
        data = await self._request(
            "POST",
            f"/entities/{entity_id}/supersede",
            json={
                "name": name,
                "entity_type": entity_type,
                "summary": summary,
                "details": details,
            },
        )
        return Entity(**data)

    # Procedural Memory

    async def create_skill(
        self,
        name: str,
        domain: str,
        task: str,
        steps: list[dict] | None = None,
        script: str | None = None,
        trigger_pattern: str | None = None,
    ) -> Skill:
        """Create a new skill."""
        data = await self._request(
            "POST",
            "/skills",
            json={
                "name": name,
                "domain": domain,
                "task": task,
                "steps": steps or [],
                "script": script,
                "trigger_pattern": trigger_pattern,
            },
        )
        data["steps"] = [Step(**s) for s in data.get("steps", [])]
        return Skill(**data)

    async def get_skill(self, skill_id: UUID) -> Skill:
        """Get a skill by ID."""
        data = await self._request("GET", f"/skills/{skill_id}")
        data["steps"] = [Step(**s) for s in data.get("steps", [])]
        return Skill(**data)

    async def list_skills(
        self,
        domain: str | None = None,
        include_deprecated: bool = False,
        limit: int = 50,
    ) -> list[Skill]:
        """List skills with optional filtering."""
        params = {"limit": limit, "include_deprecated": include_deprecated}
        if domain:
            params["domain"] = domain

        data = await self._request("GET", "/skills", params=params)
        return [
            Skill(**{**s, "steps": [Step(**st) for st in s.get("steps", [])]})
            for s in data["skills"]
        ]

    async def recall_skills(
        self,
        query: str,
        domain: str | None = None,
        limit: int = 5,
    ) -> list[Skill]:
        """Semantic search for skills."""
        data = await self._request(
            "POST",
            "/skills/recall",
            json={
                "query": query,
                "domain": domain,
                "limit": limit,
            },
        )
        return [
            Skill(**{**s, "steps": [Step(**st) for st in s.get("steps", [])]})
            for s in data["skills"]
        ]

    async def record_execution(
        self,
        skill_id: UUID,
        success: bool,
        duration_ms: int | None = None,
        notes: str | None = None,
    ) -> Skill:
        """Record skill execution result."""
        data = await self._request(
            "POST",
            f"/skills/{skill_id}/execute",
            json={
                "success": success,
                "duration_ms": duration_ms,
                "notes": notes,
            },
        )
        data["steps"] = [Step(**s) for s in data.get("steps", [])]
        return Skill(**data)

    async def deprecate_skill(
        self,
        skill_id: UUID,
        replacement_id: UUID | None = None,
    ) -> Skill:
        """Deprecate a skill."""
        params = {}
        if replacement_id:
            params["replacement_id"] = str(replacement_id)

        data = await self._request(
            "POST",
            f"/skills/{skill_id}/deprecate",
            params=params,
        )
        data["steps"] = [Step(**s) for s in data.get("steps", [])]
        return Skill(**data)

    async def how_to(
        self,
        query: str,
        domain: str | None = None,
    ) -> tuple[Skill | None, list[str], float]:
        """Get step-by-step instructions for a task."""
        params = {}
        if domain:
            params["domain"] = domain

        data = await self._request("GET", f"/skills/how-to/{query}", params=params)

        skill = None
        if data.get("skill"):
            skill_data = data["skill"]
            skill_data["steps"] = [Step(**s) for s in skill_data.get("steps", [])]
            skill = Skill(**skill_data)

        return skill, data["steps"], data["confidence"]


class WorldWeaverClient:
    """
    Synchronous client for T4DM REST API.

    Example:
        with WorldWeaverClient() as ww:
            episode = ww.create_episode("Learned about Python decorators")
            results = ww.recall_episodes("decorators")

    Note: This wraps the async client using httpx's sync client.
    For better performance with many requests, use AsyncWorldWeaverClient.
    """

    def __init__(
        self,
        base_url: str = "http://localhost:8765",
        session_id: str | None = None,
        timeout: float = 30.0,
    ):
        """
        Initialize the sync client.

        Args:
            base_url: Base URL of the T4DM API
            session_id: Session ID for memory isolation
            timeout: Request timeout in seconds
        """
        self.base_url = base_url.rstrip("/")
        self.session_id = session_id
        self.timeout = timeout
        self._client: httpx.Client | None = None

    def __enter__(self) -> "WorldWeaverClient":
        """Enter context manager."""
        self.connect()
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        """Exit context manager."""
        self.close()

    def connect(self):
        """Initialize the HTTP client."""
        headers = {}
        if self.session_id:
            headers["X-Session-ID"] = self.session_id

        self._client = httpx.Client(
            base_url=self.base_url,
            headers=headers,
            timeout=self.timeout,
        )

    def close(self):
        """Close the HTTP client."""
        if self._client:
            self._client.close()
            self._client = None

    def _get_client(self) -> httpx.Client:
        """Get the HTTP client, raising if not connected."""
        if not self._client:
            raise WorldWeaverError("Client not connected. Use 'with' or call connect()")
        return self._client

    def _request(self, method: str, path: str, **kwargs) -> dict:
        """Make an HTTP request with error handling."""
        client = self._get_client()
        try:
            response = client.request(method, f"/api/v1{path}", **kwargs)

            if response.status_code == 404:
                raise NotFoundError(
                    f"Resource not found: {path}",
                    status_code=404,
                    response=response.json() if response.content else None,
                )
            if response.status_code == 429:
                retry_after = response.headers.get("Retry-After")
                raise RateLimitError(
                    "Rate limit exceeded",
                    retry_after=int(retry_after) if retry_after else None,
                )
            if response.status_code >= 400:
                raise WorldWeaverError(
                    f"API error: {response.status_code}",
                    status_code=response.status_code,
                    response=response.json() if response.content else None,
                )

            return response.json() if response.content else {}
        except httpx.ConnectError as e:
            raise ConnectionError(f"Failed to connect to {self.base_url}: {e}")
        except httpx.TimeoutException as e:
            raise WorldWeaverError(f"Request timed out: {e}")

    # Sync wrappers for all methods - simplified for common use cases

    def health(self) -> HealthStatus:
        """Check API health status."""
        data = self._request("GET", "/health")
        return HealthStatus(**data)

    def create_episode(
        self,
        content: str,
        project: str | None = None,
        outcome: str = "neutral",
        emotional_valence: float = 0.5,
    ) -> Episode:
        """Create a new episode."""
        data = self._request(
            "POST",
            "/episodes",
            json={
                "content": content,
                "project": project,
                "outcome": outcome,
                "emotional_valence": emotional_valence,
            },
        )
        data["context"] = EpisodeContext(**data.get("context", {}))
        return Episode(**data)

    def recall_episodes(self, query: str, limit: int = 10) -> RecallResult:
        """Semantic search for episodes."""
        data = self._request(
            "POST",
            "/episodes/recall",
            json={"query": query, "limit": limit, "min_similarity": 0.5},
        )
        episodes = [
            Episode(**{**e, "context": EpisodeContext(**e.get("context", {}))})
            for e in data["episodes"]
        ]
        return RecallResult(query=data["query"], episodes=episodes, scores=data["scores"])

    def create_entity(
        self,
        name: str,
        entity_type: str,
        summary: str,
    ) -> Entity:
        """Create a new entity."""
        data = self._request(
            "POST",
            "/entities",
            json={"name": name, "entity_type": entity_type, "summary": summary},
        )
        return Entity(**data)

    def recall_entities(self, query: str, limit: int = 10) -> list[Entity]:
        """Semantic search for entities."""
        data = self._request(
            "POST",
            "/entities/recall",
            json={"query": query, "limit": limit},
        )
        return [Entity(**e) for e in data["entities"]]

    def create_skill(
        self,
        name: str,
        domain: str,
        task: str,
        steps: list[dict] | None = None,
    ) -> Skill:
        """Create a new skill."""
        data = self._request(
            "POST",
            "/skills",
            json={"name": name, "domain": domain, "task": task, "steps": steps or []},
        )
        data["steps"] = [Step(**s) for s in data.get("steps", [])]
        return Skill(**data)

    def how_to(self, query: str) -> tuple[Skill | None, list[str], float]:
        """Get step-by-step instructions for a task."""
        data = self._request("GET", f"/skills/how-to/{query}")

        skill = None
        if data.get("skill"):
            skill_data = data["skill"]
            skill_data["steps"] = [Step(**s) for s in skill_data.get("steps", [])]
            skill = Skill(**skill_data)

        return skill, data["steps"], data["confidence"]
