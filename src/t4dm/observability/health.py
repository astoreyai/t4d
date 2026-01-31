"""
Health Check for World Weaver.

Provides health monitoring for all system components.
"""

import asyncio
import logging
from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum

from t4dm.observability.metrics import get_metrics

logger = logging.getLogger(__name__)


class HealthStatus(Enum):
    """Health check status."""
    HEALTHY = "healthy"
    DEGRADED = "degraded"
    UNHEALTHY = "unhealthy"
    UNKNOWN = "unknown"


@dataclass
class ComponentHealth:
    """Health status for a single component."""
    name: str
    status: HealthStatus
    message: str = ""
    latency_ms: float | None = None
    details: dict = field(default_factory=dict)
    checked_at: datetime = field(default_factory=datetime.utcnow)

    def to_dict(self) -> dict:
        """Convert to dictionary."""
        return {
            "name": self.name,
            "status": self.status.value,
            "message": self.message,
            "latency_ms": round(self.latency_ms, 2) if self.latency_ms else None,
            "details": self.details,
            "checked_at": self.checked_at.isoformat(),
        }


@dataclass
class SystemHealth:
    """Overall system health."""
    status: HealthStatus
    components: list[ComponentHealth]
    version: str = "1.0.0"
    uptime_seconds: float = 0.0

    def to_dict(self) -> dict:
        """Convert to dictionary."""
        return {
            "status": self.status.value,
            "version": self.version,
            "uptime_seconds": round(self.uptime_seconds, 2),
            "components": [c.to_dict() for c in self.components],
            "checked_at": datetime.utcnow().isoformat(),
        }


class HealthChecker:
    """
    Health checker for World Weaver components.

    Checks:
    - Qdrant vector store connectivity
    - Neo4j graph store connectivity
    - Embedding service availability
    - Memory service initialization
    """

    def __init__(self, timeout: float = 5.0):
        """
        Initialize health checker.

        Args:
            timeout: Maximum time for health checks in seconds
        """
        self.timeout = timeout
        self._start_time = datetime.utcnow()

    async def check_qdrant(self) -> ComponentHealth:
        """Check Qdrant vector store health."""
        import time
        start = time.time()

        try:
            from t4dm.storage.qdrant_store import get_qdrant_store

            store = get_qdrant_store()
            count = await asyncio.wait_for(
                store.count(store.episodes_collection),
                timeout=self.timeout,
            )

            latency = (time.time() - start) * 1000

            return ComponentHealth(
                name="qdrant",
                status=HealthStatus.HEALTHY,
                message=f"Connected, {count} episodes",
                latency_ms=latency,
                details={"episodes_count": count},
            )

        except TimeoutError:
            return ComponentHealth(
                name="qdrant",
                status=HealthStatus.UNHEALTHY,
                message=f"Connection timeout after {self.timeout}s",
                latency_ms=(time.time() - start) * 1000,
            )
        except Exception as e:
            return ComponentHealth(
                name="qdrant",
                status=HealthStatus.UNHEALTHY,
                message=f"Connection failed: {e}",
                latency_ms=(time.time() - start) * 1000,
            )

    async def check_neo4j(self) -> ComponentHealth:
        """Check Neo4j graph store health."""
        import time
        start = time.time()

        try:
            from t4dm.storage.neo4j_store import get_neo4j_store

            store = get_neo4j_store()
            result = await asyncio.wait_for(
                store.query("RETURN 1 as n"),
                timeout=self.timeout,
            )

            latency = (time.time() - start) * 1000

            return ComponentHealth(
                name="neo4j",
                status=HealthStatus.HEALTHY,
                message="Connected",
                latency_ms=latency,
                details={"query_result": result[0]["n"] if result else None},
            )

        except TimeoutError:
            return ComponentHealth(
                name="neo4j",
                status=HealthStatus.UNHEALTHY,
                message=f"Connection timeout after {self.timeout}s",
                latency_ms=(time.time() - start) * 1000,
            )
        except Exception as e:
            return ComponentHealth(
                name="neo4j",
                status=HealthStatus.UNHEALTHY,
                message=f"Connection failed: {e}",
                latency_ms=(time.time() - start) * 1000,
            )

    async def check_embedding(self) -> ComponentHealth:
        """Check embedding service health."""
        import time
        start = time.time()

        try:
            from t4dm.embedding.bge_m3 import get_embedding_provider

            provider = get_embedding_provider()
            embedding = await asyncio.wait_for(
                provider.embed_query("health check"),
                timeout=self.timeout,
            )

            latency = (time.time() - start) * 1000

            return ComponentHealth(
                name="embedding",
                status=HealthStatus.HEALTHY,
                message=f"Model loaded, dim={len(embedding)}",
                latency_ms=latency,
                details={"dimension": len(embedding)},
            )

        except TimeoutError:
            return ComponentHealth(
                name="embedding",
                status=HealthStatus.DEGRADED,
                message=f"Model loading timeout after {self.timeout}s",
                latency_ms=(time.time() - start) * 1000,
            )
        except Exception as e:
            return ComponentHealth(
                name="embedding",
                status=HealthStatus.DEGRADED,
                message=f"Model unavailable: {e}",
                latency_ms=(time.time() - start) * 1000,
            )

    async def check_metrics(self) -> ComponentHealth:
        """Check metrics system health."""
        try:
            metrics = get_metrics()
            summary = metrics.get_summary()

            return ComponentHealth(
                name="metrics",
                status=HealthStatus.HEALTHY,
                message=f"{summary['total_operations']} operations tracked",
                details=summary,
            )
        except Exception as e:
            return ComponentHealth(
                name="metrics",
                status=HealthStatus.DEGRADED,
                message=f"Metrics unavailable: {e}",
            )

    async def check_all(self) -> SystemHealth:
        """
        Run all health checks.

        Returns:
            SystemHealth with all component statuses
        """
        # Run checks in parallel
        results = await asyncio.gather(
            self.check_qdrant(),
            self.check_neo4j(),
            self.check_embedding(),
            self.check_metrics(),
            return_exceptions=True,
        )

        components = []
        for result in results:
            if isinstance(result, Exception):
                components.append(ComponentHealth(
                    name="unknown",
                    status=HealthStatus.UNKNOWN,
                    message=str(result),
                ))
            else:
                components.append(result)

        # Determine overall status
        statuses = [c.status for c in components]
        if all(s == HealthStatus.HEALTHY for s in statuses):
            overall = HealthStatus.HEALTHY
        elif any(s == HealthStatus.UNHEALTHY for s in statuses):
            overall = HealthStatus.UNHEALTHY
        elif any(s == HealthStatus.DEGRADED for s in statuses):
            overall = HealthStatus.DEGRADED
        else:
            overall = HealthStatus.UNKNOWN

        uptime = (datetime.utcnow() - self._start_time).total_seconds()

        return SystemHealth(
            status=overall,
            components=components,
            uptime_seconds=uptime,
        )

    async def check_liveness(self) -> bool:
        """
        Simple liveness check.

        Returns:
            True if service is alive
        """
        return True

    async def check_readiness(self) -> bool:
        """
        Readiness check for accepting traffic.

        Returns:
            True if ready to serve requests
        """
        health = await self.check_all()
        return health.status in (HealthStatus.HEALTHY, HealthStatus.DEGRADED)


# Singleton instance
_health_checker: HealthChecker | None = None


def get_health_checker() -> HealthChecker:
    """Get singleton health checker."""
    global _health_checker
    if _health_checker is None:
        _health_checker = HealthChecker()
    return _health_checker
