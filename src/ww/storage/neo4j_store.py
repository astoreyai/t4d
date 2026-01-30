"""
Neo4j Graph Store for World Weaver.

Provides graph storage for memory relationships and metadata.
"""

import asyncio
import logging
import re
import threading
import time
from datetime import datetime
from typing import Any

from neo4j import AsyncDriver, AsyncGraphDatabase, AsyncSession

from ww.core.config import get_settings
from ww.storage.resilience import (
    CircuitBreaker,
    CircuitBreakerConfig,
    get_circuit_breaker,
)

logger = logging.getLogger(__name__)

# Default timeout for database operations (seconds)
DEFAULT_DB_TIMEOUT = 30

# Allowed node labels to prevent Cypher injection
ALLOWED_NODE_LABELS = frozenset({"Episode", "Entity", "Procedure"})

# Allowed relationship types - synced with ww.core.types.RelationType + graph-specific types
ALLOWED_RELATIONSHIP_TYPES = frozenset({
    # From RelationType enum (core/types.py)
    "USES", "PRODUCES", "REQUIRES", "CAUSES", "PART_OF", "SIMILAR_TO",
    "IMPLEMENTS", "IMPROVES_ON", "CONSOLIDATED_INTO", "SOURCE_OF",
    # Graph-specific relationship types
    "RELATES_TO", "HAS_CONTEXT", "DERIVED_FROM", "SUPERSEDES",
    "DEPENDS_ON", "TEMPORAL_BEFORE", "TEMPORAL_AFTER"
})


# P3-SEC-L2: Cypher metacharacters to reject (defense-in-depth)
# These should never appear in labels/types even with whitelist validation
_CYPHER_METACHAR_PATTERN = re.compile(r'[`\'"\\\[\]{}()$;:\n\r]')


def _assert_no_cypher_injection(value: str, context: str) -> None:
    """
    P3-SEC-L2: Defense-in-depth check for Cypher metacharacters.

    This is a secondary check after whitelist validation. It should never
    trigger if the whitelist is working correctly, but provides defense
    against future whitelist changes or bugs.

    Args:
        value: String to check
        context: Description for error message

    Raises:
        AssertionError: If metacharacters detected (indicates bug in whitelist)
    """
    if _CYPHER_METACHAR_PATTERN.search(value):
        # This should never happen with whitelist validation
        logger.critical(
            f"SECURITY: Cypher metacharacters detected in {context}: '{value}'. "
            "This indicates a bug in whitelist validation."
        )
        raise AssertionError(
            f"Cypher metacharacters detected in {context}. "
            "This is a security violation - contact administrators."
        )


def validate_label(label: str) -> str:
    """
    Validate that a label is in the allowed set.

    Args:
        label: Node label to validate

    Returns:
        The validated label

    Raises:
        ValueError: If label is not allowed (prevents Cypher injection)
    """
    if label not in ALLOWED_NODE_LABELS:
        raise ValueError(
            f"Invalid node label: {label}. "
            f"Must be one of: {', '.join(sorted(ALLOWED_NODE_LABELS))}"
        )
    # P3-SEC-L2: Defense-in-depth assertion
    _assert_no_cypher_injection(label, "node label")
    return label


def validate_relationship_type(rel_type: str) -> str:
    """
    Validate that a relationship type is in the allowed set.

    Args:
        rel_type: Relationship type to validate

    Returns:
        The validated relationship type

    Raises:
        ValueError: If type is not allowed (prevents Cypher injection)
    """
    if rel_type not in ALLOWED_RELATIONSHIP_TYPES:
        raise ValueError(
            f"Invalid relationship type: {rel_type}. "
            f"Must be one of: {', '.join(sorted(ALLOWED_RELATIONSHIP_TYPES))}"
        )
    # P3-SEC-L2: Defense-in-depth assertion
    _assert_no_cypher_injection(rel_type, "relationship type")
    return rel_type


# Pattern for valid Neo4j property names: alphanumeric + underscore, starting with letter/underscore
_PROPERTY_NAME_PATTERN = re.compile(r"^[a-zA-Z_][a-zA-Z0-9_]*$")


def validate_property_name(prop_name: str) -> str:
    """
    Validate that a property name is safe for Cypher interpolation.

    Args:
        prop_name: Property name to validate

    Returns:
        The validated property name

    Raises:
        ValueError: If property name contains invalid characters (prevents Cypher injection)
    """
    if not isinstance(prop_name, str):
        raise ValueError(f"Property name must be a string, got {type(prop_name).__name__}")
    if len(prop_name) > 255:
        raise ValueError(f"Property name too long: {len(prop_name)} chars (max 255)")
    if not _PROPERTY_NAME_PATTERN.match(prop_name):
        raise ValueError(
            f"Invalid property name: '{prop_name}'. "
            f"Must start with letter/underscore and contain only alphanumeric/underscore characters."
        )
    return prop_name


def validate_property_names(properties: dict) -> None:
    """
    Validate all property names in a dictionary.

    Args:
        properties: Dictionary with property names as keys

    Raises:
        ValueError: If any property name is invalid
    """
    for prop_name in properties:
        validate_property_name(prop_name)


class DatabaseTimeoutError(Exception):
    """Raised when a database operation times out."""

    def __init__(self, operation: str, timeout: float):
        self.operation = operation
        self.timeout = timeout
        super().__init__(f"Database operation '{operation}' timed out after {timeout}s")


class DatabaseConnectionError(Exception):
    """Raised when a database connection fails (sanitized)."""

    def __init__(self, operation: str = "connection"):
        self.operation = operation
        super().__init__(f"Database {operation} failed. Check configuration and connectivity.")


def _sanitize_error_message(error: Exception) -> str:
    """
    Sanitize database error messages to prevent leaking sensitive info.

    Removes:
    - Hostnames and IP addresses
    - URIs (bolt://, neo4j://, etc.)
    - Passwords
    - Internal paths

    Args:
        error: The original exception

    Returns:
        Sanitized error message safe for user exposure
    """
    import re

    msg = str(error)

    # Remove URIs
    msg = re.sub(r"(bolt|neo4j|bolt\+s|neo4j\+s)://[^\s]+", "[DATABASE_URI]", msg)

    # Remove hostnames/IPs with ports
    msg = re.sub(r"[\w.-]+:\d{4,5}", "[HOST:PORT]", msg)

    # Remove paths that look like internal
    msg = re.sub(r"/home/[\w/.-]+", "[PATH]", msg)
    msg = re.sub(r"src/[\w/.-]+", "[PATH]", msg)

    # Remove password hints
    msg = re.sub(r"password[^\s]*", "[REDACTED]", msg, flags=re.IGNORECASE)

    return msg


class Neo4jStore:
    """
    Neo4j graph store implementation.

    Manages nodes and relationships for episodic, semantic, and procedural memory.
    """

    def __init__(
        self,
        uri: str | None = None,
        user: str | None = None,
        password: str | None = None,
        database: str | None = None,
        timeout: float | None = None,
        circuit_breaker_config: CircuitBreakerConfig | None = None,
    ):
        """
        Initialize Neo4j connection.

        Args:
            uri: Neo4j bolt URI
            user: Username
            password: Password
            database: Database name
            timeout: Operation timeout in seconds (default: 30)
            circuit_breaker_config: Optional circuit breaker configuration
        """
        settings = get_settings()

        self.uri = uri or settings.neo4j_uri
        self.user = user or settings.neo4j_user
        self.password = password or settings.neo4j_password
        self.database = database or settings.neo4j_database
        self.timeout = timeout or DEFAULT_DB_TIMEOUT

        self._driver: AsyncDriver | None = None

        # P2-OPT-B3.3: Connection pool metrics tracking
        self._pool_metrics = {
            "acquisitions": 0,
            "acquisition_failures": 0,
            "acquisition_total_ms": 0.0,
            "acquisition_max_ms": 0.0,
            "sessions_created": 0,
            "sessions_closed": 0,
        }
        self._metrics_lock = threading.Lock()

        # Circuit breaker for resilience
        cb_config = circuit_breaker_config or CircuitBreakerConfig(
            failure_threshold=5,
            success_threshold=2,
            reset_timeout=60.0,
        )
        self._circuit_breaker = get_circuit_breaker("neo4j", cb_config)

    async def _with_timeout(self, coro, operation: str):
        """Execute coroutine with timeout and circuit breaker protection."""
        async def _execute():
            try:
                async with asyncio.timeout(self.timeout):
                    return await coro
            except TimeoutError:
                logger.error(f"Timeout in Neo4j operation '{operation}' after {self.timeout}s")
                raise DatabaseTimeoutError(operation, self.timeout)

        return await self._circuit_breaker.execute(_execute)

    @property
    def circuit_breaker(self) -> CircuitBreaker:
        """Get the circuit breaker for this store."""
        return self._circuit_breaker

    async def _get_driver(self) -> AsyncDriver:
        """Get or create driver with connection pooling."""
        if self._driver is None:
            settings = get_settings()
            self._driver = AsyncGraphDatabase.driver(
                self.uri,
                auth=(self.user, self.password),
                max_connection_pool_size=settings.neo4j_pool_size,
                connection_acquisition_timeout=settings.neo4j_connection_timeout,
                max_connection_lifetime=settings.neo4j_connection_lifetime,
                connection_timeout=settings.neo4j_connection_timeout,
                keep_alive=True,
            )
            logger.info(
                f"Neo4j driver initialized with pool_size={settings.neo4j_pool_size}, "
                f"timeout={settings.neo4j_connection_timeout}s, "
                f"lifetime={settings.neo4j_connection_lifetime}s"
            )
        return self._driver

    async def _acquire_session(self) -> AsyncSession:
        """
        P2-OPT-B3.3: Acquire session with metrics tracking.

        Records acquisition time and success/failure counts for pool monitoring.
        """
        start = time.time()
        try:
            driver = await self._get_driver()
            session = driver.session(database=self.database)
            elapsed_ms = (time.time() - start) * 1000

            with self._metrics_lock:
                self._pool_metrics["acquisitions"] += 1
                self._pool_metrics["acquisition_total_ms"] += elapsed_ms
                self._pool_metrics["acquisition_max_ms"] = max(
                    self._pool_metrics["acquisition_max_ms"], elapsed_ms
                )
                self._pool_metrics["sessions_created"] += 1

            # Log slow acquisitions (>100ms indicates pool pressure)
            if elapsed_ms > 100:
                logger.warning(f"Slow session acquisition: {elapsed_ms:.1f}ms")

            return session

        except Exception as e:
            elapsed_ms = (time.time() - start) * 1000
            with self._metrics_lock:
                self._pool_metrics["acquisitions"] += 1
                self._pool_metrics["acquisition_failures"] += 1
            logger.error(f"Session acquisition failed after {elapsed_ms:.1f}ms: {e}")
            raise

    def get_pool_stats(self) -> dict:
        """
        P2-OPT-B3.3: Get connection pool statistics.

        Returns:
            Dict with pool metrics including:
            - acquisitions: Total session acquisition attempts
            - acquisition_failures: Failed acquisition attempts
            - avg_acquisition_ms: Average acquisition time
            - max_acquisition_ms: Maximum acquisition time
            - sessions_active: Currently active sessions (created - closed)
            - pool_size: Configured pool size
        """
        settings = get_settings()

        with self._metrics_lock:
            acquisitions = self._pool_metrics["acquisitions"]
            total_ms = self._pool_metrics["acquisition_total_ms"]
            avg_ms = total_ms / acquisitions if acquisitions > 0 else 0.0

            return {
                "acquisitions": acquisitions,
                "acquisition_failures": self._pool_metrics["acquisition_failures"],
                "avg_acquisition_ms": round(avg_ms, 2),
                "max_acquisition_ms": round(self._pool_metrics["acquisition_max_ms"], 2),
                "sessions_created": self._pool_metrics["sessions_created"],
                "sessions_closed": self._pool_metrics["sessions_closed"],
                "sessions_active": (
                    self._pool_metrics["sessions_created"] -
                    self._pool_metrics["sessions_closed"]
                ),
                "pool_size": settings.neo4j_pool_size,
                "connection_timeout_s": settings.neo4j_connection_timeout,
            }

    def _record_session_close(self) -> None:
        """Record that a session was closed."""
        with self._metrics_lock:
            self._pool_metrics["sessions_closed"] += 1

    async def initialize(self) -> None:
        """Initialize database schema and indexes."""
        try:
            driver = await self._get_driver()

            async with driver.session(database=self.database) as session:
                # Create constraints and indexes
                await self._create_schema(session)
        except DatabaseConnectionError:
            # Already sanitized, re-raise
            raise
        except Exception as e:
            # Log the full error for debugging, but sanitize for user
            logger.error(f"Neo4j initialization failed: {e}", exc_info=True)
            raise DatabaseConnectionError("initialization") from None

    async def _create_schema(self, session: AsyncSession) -> None:
        """Create database schema."""
        # Episode constraints
        await session.run("""
            CREATE CONSTRAINT episode_id IF NOT EXISTS
            FOR (e:Episode) REQUIRE e.id IS UNIQUE
        """)

        # Entity constraints
        await session.run("""
            CREATE CONSTRAINT entity_id IF NOT EXISTS
            FOR (e:Entity) REQUIRE e.id IS UNIQUE
        """)

        # Procedure constraints
        await session.run("""
            CREATE CONSTRAINT procedure_id IF NOT EXISTS
            FOR (p:Procedure) REQUIRE p.id IS UNIQUE
        """)

        # Indexes for common queries - session isolation
        await session.run("""
            CREATE INDEX episode_session IF NOT EXISTS
            FOR (e:Episode) ON (e.sessionId)
        """)

        await session.run("""
            CREATE INDEX entity_session IF NOT EXISTS
            FOR (e:Entity) ON (e.sessionId)
        """)

        await session.run("""
            CREATE INDEX procedure_session IF NOT EXISTS
            FOR (p:Procedure) ON (p.sessionId)
        """)

        # Indexes for common queries - other attributes
        await session.run("""
            CREATE INDEX episode_timestamp IF NOT EXISTS
            FOR (e:Episode) ON (e.timestamp)
        """)

        # Verify schema
        result = await session.run("SHOW CONSTRAINTS")
        constraints = await result.data()
        logger.debug(f"Created {len(constraints)} constraints/indexes")

    async def query(
        self,
        cypher: str,
        params: dict[str, Any] | None = None,
    ) -> list[dict[str, Any]]:
        """
        Execute a read query.

        Args:
            cypher: Cypher query string
            params: Query parameters

        Returns:
            List of result records
        """
        async def _query():
            driver = await self._get_driver()
            async with driver.session(database=self.database) as session:
                result = await session.run(cypher, params or {})
                return await result.data()

        return await self._with_timeout(_query(), "query")

    async def create_node(
        self,
        label: str,
        properties: dict[str, Any],
    ) -> str:
        """
        Create a node in the graph.

        Args:
            label: Node label (must be validated)
            properties: Node properties

        Returns:
            Node ID

        Raises:
            ValueError: If label or property names are invalid (Cypher injection prevention)
        """
        label = validate_label(label)
        validate_property_names(properties)  # SEC-001: Prevent Cypher injection
        properties = self._serialize_props(properties)

        async def _create():
            driver = await self._get_driver()
            async with driver.session(database=self.database) as session:
                # Build CREATE clause with property keys
                prop_keys = ", ".join(f"{k}: ${k}" for k in properties.keys())
                cypher = f"""
                    CREATE (n:{label} {{{prop_keys}}})
                    RETURN n.id as id
                """
                result = await session.run(cypher, properties)
                record = await result.single()
                return record["id"] if record else None

        return await self._with_timeout(_create(), "create_node")

    async def get_node(
        self,
        node_id: str,
        label: str | None = None,
    ) -> dict[str, Any] | None:
        """
        Get a node by ID.

        Args:
            node_id: Node ID
            label: Optional label filter

        Returns:
            Node data or None
        """
        if label:
            label = validate_label(label)

        async def _get():
            driver = await self._get_driver()
            async with driver.session(database=self.database) as session:
                cypher = f"""
                    MATCH (n{f':{label}' if label else ''} {{id: $id}})
                    RETURN properties(n) as data
                """
                result = await session.run(cypher, {"id": node_id})
                record = await result.single()
                if record:
                    return self._deserialize_props(record["data"])
                return None

        return await self._with_timeout(_get(), "get_node")

    async def update_node(
        self,
        node_id: str,
        properties: dict[str, Any],
        label: str | None = None,
    ) -> None:
        """
        Update node properties.

        Args:
            node_id: Node ID
            properties: Properties to update
            label: Optional label filter

        Raises:
            ValueError: If label or property names are invalid (Cypher injection prevention)
        """
        if label:
            label = validate_label(label)

        validate_property_names(properties)  # SEC-001: Prevent Cypher injection
        properties = self._serialize_props(properties)

        async def _update():
            driver = await self._get_driver()
            async with driver.session(database=self.database) as session:
                # Build SET clause
                set_clause = ", ".join(f"n.{k} = ${k}" for k in properties.keys())
                cypher = f"""
                    MATCH (n{f':{label}' if label else ''} {{id: $id}})
                    SET {set_clause}
                """
                await session.run(cypher, {"id": node_id, **properties})

        return await self._with_timeout(_update(), "update_node")

    async def delete_node(
        self,
        node_id: str,
        label: str | None = None,
    ) -> None:
        """
        Delete a node.

        Args:
            node_id: Node ID
            label: Optional label filter
        """
        if label:
            label = validate_label(label)

        async def _delete():
            driver = await self._get_driver()
            async with driver.session(database=self.database) as session:
                cypher = f"""
                    MATCH (n{f':{label}' if label else ''} {{id: $id}})
                    DETACH DELETE n
                """
                await session.run(cypher, {"id": node_id})

        return await self._with_timeout(_delete(), "delete_node")

    async def create_relationship(
        self,
        source_id: str,
        target_id: str,
        rel_type: str,
        properties: dict[str, Any] | None = None,
    ) -> None:
        """
        Create a relationship between nodes.

        Args:
            source_id: Source node ID
            target_id: Target node ID
            rel_type: Relationship type
            properties: Relationship properties
        """
        rel_type = validate_relationship_type(rel_type)
        properties = self._serialize_props(properties or {})

        async def _create():
            driver = await self._get_driver()
            async with driver.session(database=self.database) as session:
                # Build property clause
                prop_clause = ""
                if properties:
                    prop_keys = ", ".join(f"{k}: ${k}" for k in properties.keys())
                    prop_clause = f" {{{prop_keys}}}"

                cypher = f"""
                    MATCH (src {{id: $source_id}})
                    MATCH (tgt {{id: $target_id}})
                    CREATE (src)-[r:{rel_type}{prop_clause}]->(tgt)
                """
                await session.run(
                    cypher,
                    {"source_id": source_id, "target_id": target_id, **properties}
                )

        return await self._with_timeout(_create(), "create_relationship")

    async def batch_create_relationships(
        self,
        relationships: list[tuple[str, str, str, dict[str, Any] | None]],
    ) -> int:
        """
        Create multiple relationships in a single batch operation.

        P4.2: Eliminates N+1 query pattern by using UNWIND for batch creation.

        Args:
            relationships: List of (source_id, target_id, rel_type, properties) tuples

        Returns:
            Number of relationships created

        Raises:
            ValueError: If batch size exceeds 1000 or relationship types are invalid
        """
        MAX_BATCH_SIZE = 1000

        if not relationships:
            return 0

        if len(relationships) > MAX_BATCH_SIZE:
            raise ValueError(
                f"Batch size {len(relationships)} exceeds maximum allowed size of {MAX_BATCH_SIZE}"
            )

        # SEC-001: Validate all relationship types upfront
        for _, _, rel_type, _ in relationships:
            validate_relationship_type(rel_type)

        async def _batch_create():
            driver = await self._get_driver()
            async with driver.session(database=self.database) as session:
                # Group by relationship type for efficient UNWIND queries
                by_type: dict[str, list[dict]] = {}
                for source_id, target_id, rel_type, props in relationships:
                    if rel_type not in by_type:
                        by_type[rel_type] = []
                    by_type[rel_type].append({
                        "source_id": source_id,
                        "target_id": target_id,
                        "props": self._serialize_props(props or {}),
                    })

                total_created = 0

                async with await session.begin_transaction() as tx:
                    for rel_type, rels in by_type.items():
                        # Use UNWIND for batch relationship creation
                        cypher = f"""
                            UNWIND $rels AS rel
                            MATCH (src {{id: rel.source_id}})
                            MATCH (tgt {{id: rel.target_id}})
                            CREATE (src)-[r:{rel_type}]->(tgt)
                            SET r += rel.props
                            RETURN count(r) as created
                        """
                        result = await tx.run(cypher, {"rels": rels})
                        record = await result.single()
                        if record:
                            total_created += record["created"]

                    await tx.commit()
                    return total_created

        return await self._with_timeout(_batch_create(), "batch_create_relationships")

    async def get_relationships(
        self,
        node_id: str,
        rel_type: str | None = None,
        direction: str = "both",
    ) -> list[dict[str, Any]]:
        """
        Get relationships for a node.

        Args:
            node_id: Node ID
            rel_type: Optional relationship type filter
            direction: 'out', 'in', or 'both' (default: 'both')

        Returns:
            List of relationships
        """
        if rel_type:
            rel_type = validate_relationship_type(rel_type)

        async def _get():
            driver = await self._get_driver()
            async with driver.session(database=self.database) as session:
                # Build relationship pattern
                if direction == "out":
                    pattern = f"-[r{f':{rel_type}' if rel_type else ''}]->"
                elif direction == "in":
                    pattern = f"<-[r{f':{rel_type}' if rel_type else ''}]-"
                else:
                    pattern = f"-[r{f':{rel_type}' if rel_type else ''}]-"

                cypher = f"""
                    MATCH (n {{id: $id}}){pattern}(other)
                    RETURN {{
                        rel_type: type(r),
                        properties: properties(r),
                        source_id: n.id,
                        other_id: other.id
                    }} as rel
                """
                result = await session.run(cypher, {"id": node_id})
                records = await result.data()
                return [self._deserialize_props(r["rel"]) for r in records]

        return await self._with_timeout(_get(), "get_relationships")

    async def get_relationships_batch(
        self,
        node_ids: list[str],
        rel_type: str | None = None,
        direction: str = "both",
    ) -> dict[str, list[dict[str, Any]]]:
        """
        Get relationships for multiple nodes efficiently.

        Args:
            node_ids: List of node IDs
            rel_type: Optional relationship type filter
            direction: 'out', 'in', or 'both' (default: 'both')

        Returns:
            Dict mapping node_id to list of relationships
        """
        if rel_type:
            rel_type = validate_relationship_type(rel_type)
        if direction not in ("out", "in", "both"):
            raise ValueError(f"Invalid direction: {direction}. Must be 'out', 'in', or 'both'")

        rel_filter = f":{rel_type}" if rel_type else ""

        async def _get():
            driver = await self._get_driver()
            async with driver.session(database=self.database) as session:
                # Build direction-aware pattern
                # Note: 'id' is a variable from UNWIND, not a parameter ($id)
                if direction == "out":
                    pattern = f"(n {{id: id}})-[r{rel_filter}]->(other)"
                elif direction == "in":
                    pattern = f"(n {{id: id}})<-[r{rel_filter}]-(other)"
                else:  # both
                    pattern = f"(n {{id: id}})-[r{rel_filter}]-(other)"

                cypher = f"""
                    MATCH {pattern}
                    RETURN n.id as node_id, type(r) as rel_type,
                           properties(r) as rel_props, other.id as other_id
                """
                result = await session.run(
                    "UNWIND $ids as id " + cypher,
                    {"ids": node_ids}
                )
                records = await result.data()

                # Group by node_id
                relationships = {nid: [] for nid in node_ids}
                for record in records:
                    node_id = record["node_id"]
                    rel_data = {
                        "rel_type": record["rel_type"],
                        "other_id": record["other_id"],
                        "properties": self._deserialize_props(record["rel_props"]),
                    }
                    relationships[node_id].append(rel_data)

                return relationships

        return await self._with_timeout(_get(), "get_relationships_batch")

    async def update_property(
        self,
        node_id: str,
        property_name: str,
        property_value: Any,
    ) -> None:
        """
        Update a single property on a node.

        Args:
            node_id: Node ID
            property_name: Property name
            property_value: New value

        Raises:
            ValueError: If property_name contains invalid characters (Cypher injection prevention)
        """
        # SEC-001: Validate property name to prevent Cypher injection
        validate_property_name(property_name)

        async def _update():
            driver = await self._get_driver()
            async with driver.session(database=self.database) as session:
                value = self._serialize_props({property_name: property_value})[property_name]
                cypher = f"MATCH (n {{id: $id}}) SET n.{property_name} = $value"
                await session.run(cypher, {"id": node_id, "value": value})

        return await self._with_timeout(_update(), "update_property")

    def _serialize_props(self, props: dict[str, Any]) -> dict[str, Any]:
        """Serialize properties for Neo4j (JSON strings for complex types)."""
        import json

        result = {}
        for key, value in props.items():
            if isinstance(value, (dict, list)):
                result[key] = json.dumps(value)
            elif isinstance(value, datetime):
                result[key] = value.isoformat()
            else:
                result[key] = value
        return result

    def _deserialize_props(self, props: dict[str, Any]) -> dict[str, Any]:
        """Deserialize properties from Neo4j."""
        result = {}
        for key, value in props.items():
            if isinstance(value, str):
                # Try to parse as JSON
                if value.startswith(("[", "{")):
                    try:
                        import json
                        result[key] = json.loads(value)
                        continue
                    except json.JSONDecodeError:
                        pass
                # Try to parse as datetime
                if "T" in value and len(value) > 10:
                    try:
                        result[key] = datetime.fromisoformat(value)
                        continue
                    except ValueError:
                        pass
            result[key] = value
        return result

    async def health_check(self) -> dict[str, Any]:
        """
        Check database connectivity and connection pool health.

        Returns:
            Dict with health status and metrics
        """
        async def _check():
            driver = await self._get_driver()

            async with driver.session(database=self.database) as session:
                # Test connectivity
                result = await session.run("RETURN 1 as test")
                await result.single()

            # Get pool metrics if available
            pool_metrics = {}
            if hasattr(driver, "_pool"):
                pool = driver._pool
                if hasattr(pool, "in_use"):
                    pool_metrics["connections_in_use"] = pool.in_use
                if hasattr(pool, "size"):
                    pool_metrics["pool_size"] = pool.size
                if hasattr(pool, "max_size"):
                    pool_metrics["max_pool_size"] = pool.max_size

            return {
                "status": "healthy",
                "database": self.database,
                "uri": self.uri,
                "pool_metrics": pool_metrics,
            }

        try:
            return await self._with_timeout(_check(), "health_check")
        except Exception as e:
            logger.error(f"Health check failed: {e}")
            return {
                "status": "unhealthy",
                "error": str(e),
                "database": self.database,
                "uri": self.uri,
            }

    async def get_pool_metrics(self) -> dict[str, Any]:
        """
        Get connection pool metrics.

        Returns:
            Dict with pool statistics
        """
        if not self._driver:
            return {
                "driver_initialized": False,
            }

        metrics = {
            "driver_initialized": True,
            "uri": self.uri,
            "database": self.database,
        }

        # Attempt to get internal pool metrics
        if hasattr(self._driver, "_pool"):
            pool = self._driver._pool
            if hasattr(pool, "in_use"):
                metrics["connections_in_use"] = pool.in_use
            if hasattr(pool, "size"):
                metrics["current_pool_size"] = pool.size
            if hasattr(pool, "max_size"):
                metrics["max_pool_size"] = pool.max_size

        settings = get_settings()
        metrics["configured_max_pool_size"] = settings.neo4j_pool_size
        metrics["connection_timeout"] = settings.neo4j_connection_timeout
        metrics["connection_lifetime"] = settings.neo4j_connection_lifetime

        return metrics

    async def batch_decay_relationships(
        self,
        stale_days: int = 30,
        decay_rate: float = 0.01,
        min_weight: float = 0.01,
        session_id: str | None = None,
    ) -> dict[str, int]:
        """
        Apply Hebbian decay to stale relationships in batch.

        Two-query approach:
        1. Decay: Reduce weight of all relationships not accessed within stale_days
        2. Prune: Delete relationships with weight below min_weight

        Args:
            stale_days: Days since last access to consider relationship stale
            decay_rate: Decay factor to apply (new_weight = old_weight * (1 - decay_rate))
            min_weight: Minimum weight threshold; relationships below this are pruned
            session_id: Optional session filter

        Returns:
            Dict with 'decayed' and 'pruned' counts
        """
        from datetime import datetime, timedelta

        cutoff = datetime.now() - timedelta(days=stale_days)
        cutoff_str = cutoff.isoformat()

        # Query 1: Apply decay to stale relationships
        decay_cypher = """
            MATCH (a)-[r]->(b)
            WHERE r.lastAccessed < $cutoff
            AND ($session_id IS NULL OR a.sessionId = $session_id)
            SET r.weight = r.weight * (1 - $decay_rate)
            RETURN count(r) as decayed
        """

        decay_result = await self.query(
            decay_cypher,
            {"cutoff": cutoff_str, "decay_rate": decay_rate, "session_id": session_id}
        )
        decayed = decay_result[0]["decayed"] if decay_result else 0

        # Query 2: Prune relationships below threshold
        prune_cypher = """
            MATCH (a)-[r]->(b)
            WHERE r.weight < $min_weight
            AND ($session_id IS NULL OR a.sessionId = $session_id)
            DELETE r
            RETURN count(r) as pruned
        """

        prune_result = await self.query(
            prune_cypher,
            {"min_weight": min_weight, "session_id": session_id}
        )
        pruned = prune_result[0]["pruned"] if prune_result else 0

        return {"decayed": decayed, "pruned": pruned}

    async def count_stale_relationships(
        self,
        stale_days: int = 30,
        session_id: str | None = None,
    ) -> int:
        """
        Count relationships not accessed within stale_days.

        Args:
            stale_days: Days since last access to consider relationship stale
            session_id: Optional session filter

        Returns:
            Count of stale relationships
        """
        from datetime import datetime, timedelta

        cutoff = datetime.now() - timedelta(days=stale_days)
        cutoff_str = cutoff.isoformat()

        count_cypher = """
            MATCH (a)-[r]->(b)
            WHERE r.lastAccessed < $cutoff
            AND ($session_id IS NULL OR a.sessionId = $session_id)
            RETURN count(r) as count
        """

        result = await self.query(
            count_cypher,
            {"cutoff": cutoff_str, "session_id": session_id}
        )
        return result[0]["count"] if result else 0

    async def strengthen_relationship(
        self,
        source_id: str,
        target_id: str,
        learning_rate: float = 0.1,
    ) -> float:
        """
        Strengthen relationship via Hebbian update.

        Implements bounded Hebbian learning: w' = w + lr * (1 - w)
        This ensures weights stay in [0, 1] and approach 1.0 asymptotically.

        Also updates coAccessCount and lastCoAccess for tracking.

        Args:
            source_id: Source node ID
            target_id: Target node ID
            learning_rate: Learning rate for update (default 0.1)

        Returns:
            New weight value after strengthening
        """
        from datetime import datetime

        now = datetime.now().isoformat()

        # Hebbian update: w' = w + lr * (1 - w)
        # Also increment coAccessCount and update lastCoAccess
        cypher = """
            MATCH (a {id: $source_id})-[r]->(b {id: $target_id})
            SET r.weight = COALESCE(r.weight, 0.1) + $lr * (1 - COALESCE(r.weight, 0.1)),
                r.coAccessCount = COALESCE(r.coAccessCount, 0) + 1,
                r.lastCoAccess = $now
            RETURN r.weight as new_weight
        """

        # Also try reverse direction if forward doesn't match
        cypher_reverse = """
            MATCH (a {id: $target_id})-[r]->(b {id: $source_id})
            SET r.weight = COALESCE(r.weight, 0.1) + $lr * (1 - COALESCE(r.weight, 0.1)),
                r.coAccessCount = COALESCE(r.coAccessCount, 0) + 1,
                r.lastCoAccess = $now
            RETURN r.weight as new_weight
        """

        params = {
            "source_id": source_id,
            "target_id": target_id,
            "lr": learning_rate,
            "now": now,
        }

        # Try forward direction first
        result = await self.query(cypher, params)
        if result and result[0].get("new_weight") is not None:
            new_weight = result[0]["new_weight"]
            logger.debug(
                f"Strengthened relationship {source_id}->{target_id}: "
                f"new_weight={new_weight:.4f}"
            )
            return new_weight

        # Try reverse direction
        result = await self.query(cypher_reverse, params)
        if result and result[0].get("new_weight") is not None:
            new_weight = result[0]["new_weight"]
            logger.debug(
                f"Strengthened relationship {target_id}->{source_id}: "
                f"new_weight={new_weight:.4f}"
            )
            return new_weight

        # No relationship found
        logger.warning(
            f"No relationship found between {source_id} and {target_id} to strengthen"
        )
        return 0.0

    async def batch_create_nodes(
        self,
        nodes: list[tuple[str, dict[str, Any]]],
    ) -> list[str]:
        """
        Create multiple nodes in batch.

        Args:
            nodes: List of (label, properties) tuples

        Returns:
            List of created node IDs

        Raises:
            ValueError: If batch size exceeds limit (1000), or if labels/property names are invalid
        """
        MAX_BATCH_SIZE = 1000

        if len(nodes) > MAX_BATCH_SIZE:
            raise ValueError(
                f"Batch size {len(nodes)} exceeds maximum allowed size of {MAX_BATCH_SIZE}"
            )

        # SEC-001: Validate labels and property names to prevent Cypher injection
        for label, properties in nodes:
            validate_label(label)
            validate_property_names(properties)

        async def _batch_create():
            driver = await self._get_driver()
            async with driver.session(database=self.database) as session:
                # DATA-002: Use transaction for atomicity - if one fails, all rollback
                async with await session.begin_transaction() as tx:
                    node_ids = []

                    for label, properties in nodes:
                        properties = self._serialize_props(properties)
                        prop_keys = ", ".join(f"{k}: ${k}" for k in properties.keys())
                        cypher = f"""
                            CREATE (n:{label} {{{prop_keys}}})
                            RETURN n.id as id
                        """
                        result = await tx.run(cypher, properties)
                        record = await result.single()
                        if record:
                            node_ids.append(record["id"])

                    # Commit transaction on success
                    await tx.commit()
                    return node_ids

        return await self._with_timeout(_batch_create(), "batch_create_nodes")

    async def find_path(
        self,
        source_id: str,
        target_id: str,
        max_depth: int,
    ) -> list[dict[str, Any]] | None:
        """
        Find shortest path between two nodes.

        Args:
            source_id: Source node ID
            target_id: Target node ID
            max_depth: Maximum path depth to search

        Returns:
            List of path segments (nodes and relationships) or None if no path exists

        Raises:
            ValueError: If max_depth exceeds limit (10)
        """
        MAX_DEPTH_LIMIT = 10

        # SEC-001: Validate max_depth is a safe integer to prevent injection
        if not isinstance(max_depth, int) or max_depth < 1:
            raise ValueError(f"max_depth must be a positive integer, got {max_depth!r}")
        if max_depth > MAX_DEPTH_LIMIT:
            raise ValueError(
                f"max_depth {max_depth} exceeds maximum allowed depth of {MAX_DEPTH_LIMIT}"
            )

        async def _find_path():
            driver = await self._get_driver()
            async with driver.session(database=self.database) as session:
                cypher = f"""
                    MATCH path = shortestPath((source {{id: $source_id}})-[*..{max_depth}]-(target {{id: $target_id}}))
                    RETURN [node in nodes(path) | {{id: node.id, labels: labels(node)}}] as nodes,
                           [rel in relationships(path) | {{type: type(rel), properties: properties(rel)}}] as relationships
                """
                result = await session.run(
                    cypher,
                    {"source_id": source_id, "target_id": target_id}
                )
                record = await result.single()

                if not record:
                    return None

                # Combine nodes and relationships into path segments
                path_segments = []
                nodes = record["nodes"]
                relationships = record["relationships"]

                for i, node in enumerate(nodes):
                    path_segments.append({"type": "node", **node})
                    if i < len(relationships):
                        rel = relationships[i]
                        path_segments.append({
                            "type": "relationship",
                            "rel_type": rel["type"],
                            **self._deserialize_props(rel["properties"])
                        })

                return path_segments

        return await self._with_timeout(_find_path(), "find_path")

    async def close(self) -> None:
        """Close driver connection."""
        if self._driver:
            await self._driver.close()
            self._driver = None


# Thread-safe singleton management

_neo4j_instances: dict[str, Neo4jStore] = {}
_neo4j_lock = threading.Lock()
# BUG-003 FIX: Use loop-specific locks to handle event loop changes correctly
# STORAGE-HIGH-003 FIX: Store (loop, lock) tuples to enable cleanup of closed loops
_neo4j_async_locks: dict[int, tuple[asyncio.AbstractEventLoop, asyncio.Lock]] = {}
_neo4j_locks_lock = threading.Lock()  # Thread-safe access to locks dict


def _cleanup_neo4j_closed_loops() -> None:
    """
    STORAGE-HIGH-003 FIX: Remove locks for closed event loops.

    Must be called while holding _neo4j_locks_lock.
    """
    closed_ids = [
        loop_id for loop_id, (loop, _) in _neo4j_async_locks.items()
        if loop.is_closed()
    ]
    for loop_id in closed_ids:
        del _neo4j_async_locks[loop_id]
    if closed_ids:
        logger.debug(f"Cleaned up {len(closed_ids)} stale neo4j async locks")


def _get_neo4j_async_lock() -> asyncio.Lock:
    """
    Get or create async lock for the current event loop (must be called from async context).

    This lock is event-loop specific to avoid issues when locks are accessed
    from different event loops (e.g., in testing with pytest-asyncio).

    BUG-003 FIX: Use a dictionary keyed by event loop ID instead of checking
    private attributes.

    STORAGE-HIGH-003 FIX: Clean up locks for closed event loops to prevent memory leak.
    """
    try:
        current_loop = asyncio.get_running_loop()
        loop_id = id(current_loop)
    except RuntimeError:
        # No running event loop - shouldn't happen in async context
        raise RuntimeError("_get_neo4j_async_lock must be called from async context")

    # Thread-safe lock creation for the specific event loop
    with _neo4j_locks_lock:
        # STORAGE-HIGH-003 FIX: Clean up closed loops before adding new entry
        _cleanup_neo4j_closed_loops()

        if loop_id not in _neo4j_async_locks:
            _neo4j_async_locks[loop_id] = (current_loop, asyncio.Lock())
        return _neo4j_async_locks[loop_id][1]


def get_neo4j_store(session_id: str = "default") -> Neo4jStore:
    """
    Get or create singleton Neo4j store (thread-safe).

    Args:
        session_id: Session identifier for instance isolation

    Returns:
        Neo4jStore instance for the session
    """
    if session_id not in _neo4j_instances:
        with _neo4j_lock:
            # Double-check locking pattern
            if session_id not in _neo4j_instances:
                _neo4j_instances[session_id] = Neo4jStore()
                logger.debug(f"Created Neo4jStore for session: {session_id}")
    return _neo4j_instances[session_id]


async def close_neo4j_store(session_id: str | None = None) -> None:
    """
    Close Neo4j store connection(s).

    Args:
        session_id: Specific session to close, or None for all
    """
    try:
        async with _get_neo4j_async_lock():
            if session_id:
                if session_id in _neo4j_instances:
                    await _neo4j_instances[session_id].close()
                    del _neo4j_instances[session_id]
                    logger.info(f"Closed Neo4jStore for session: {session_id}")
            else:
                for sid, store in list(_neo4j_instances.items()):
                    await store.close()
                    logger.debug(f"Closed Neo4jStore for session: {sid}")
                _neo4j_instances.clear()
                _reset_neo4j_async_lock()
                logger.info("Closed all Neo4jStore instances")
    except Exception as e:
        # If we can't acquire the lock due to event loop issues, just close stores directly
        logger.warning(f"Error acquiring async lock in close_neo4j_store: {e}. Closing stores directly.")
        if session_id:
            if session_id in _neo4j_instances:
                try:
                    await _neo4j_instances[session_id].close()
                except Exception as close_err:
                    logger.error(f"Error closing store: {close_err}")
                del _neo4j_instances[session_id]
        else:
            for sid, store in list(_neo4j_instances.items()):
                try:
                    await store.close()
                except Exception as close_err:
                    logger.error(f"Error closing store {sid}: {close_err}")
            _neo4j_instances.clear()
            _reset_neo4j_async_lock()


def _reset_neo4j_async_lock() -> None:
    """Reset the async lock (for test isolation). Should only be called after clearing all instances."""
    global _neo4j_async_lock
    _neo4j_async_lock = None
    logger.debug("Reset Neo4j async lock")
