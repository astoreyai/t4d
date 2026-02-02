"""
MCP Server for T4DM Memory System.

Exposes T4DM as an MCP server for Claude Code/Desktop integration.
Provides tools for memory storage, retrieval, learning, and consolidation.

Usage:
    # As standalone server
    python -m t4dm.mcp.server

    # Or programmatically
    from t4dm.mcp import run_mcp_server
    asyncio.run(run_mcp_server())

Configuration via environment:
    T4DM_API_URL: T4DM API URL (default: http://localhost:8765)
    T4DM_SESSION_ID: Default session ID
    T4DM_API_KEY: Optional API key
"""

import asyncio
import json
import logging
import os
import sys
from datetime import datetime
from typing import Any
from uuid import uuid4

from t4dm.mcp.tools import MEMORY_PROMPTS, MEMORY_TOOLS, ToolResult
from t4dm.sdk.agent_client import AgentMemoryClient

logger = logging.getLogger(__name__)


class WorldWeaverMCPServer:
    """
    MCP Server implementation for T4DM.

    Handles tool execution and resource access for Claude Code/Desktop.
    """

    def __init__(
        self,
        api_url: str = "http://localhost:8765",
        session_id: str | None = None,
        api_key: str | None = None,
    ):
        """
        Initialize MCP server.

        Args:
            api_url: T4DM API URL
            session_id: Default session ID (generated if not provided)
            api_key: Optional API key for authentication
        """
        self.api_url = api_url
        self.session_id = session_id or f"mcp-{uuid4().hex[:8]}"
        self.api_key = api_key
        self._memory: AgentMemoryClient | None = None
        self._initialized = False

    async def initialize(self):
        """Initialize connection to T4DM."""
        if self._initialized:
            return

        self._memory = AgentMemoryClient(
            base_url=self.api_url,
            session_id=self.session_id,
            api_key=self.api_key,
        )
        await self._memory.connect()
        self._initialized = True
        logger.info(f"MCP server initialized: session={self.session_id}")

    async def shutdown(self):
        """Shutdown and cleanup."""
        if self._memory:
            await self._memory.close()
            self._memory = None
        self._initialized = False
        logger.info("MCP server shutdown complete")

    # =========================================================================
    # MCP Protocol Handlers
    # =========================================================================

    async def handle_initialize(self, params: dict) -> dict:
        """Handle MCP initialize request."""
        await self.initialize()

        return {
            "protocolVersion": "2024-11-05",
            "capabilities": {
                "tools": {},
                "resources": {},
                "prompts": {},
            },
            "serverInfo": {
                "name": "world-weaver",
                "version": "0.5.0",
            },
        }

    async def handle_list_tools(self) -> dict:
        """Handle tools/list request."""
        return {"tools": MEMORY_TOOLS}

    async def handle_call_tool(self, name: str, arguments: dict) -> dict:
        """
        Handle tools/call request.

        Routes to appropriate tool handler based on name.
        """
        if not self._initialized:
            await self.initialize()

        handlers = {
            "ww_store": self._handle_store,
            "ww_recall": self._handle_recall,
            "ww_learn_outcome": self._handle_learn_outcome,
            "ww_consolidate": self._handle_consolidate,
            "ww_get_context": self._handle_get_context,
            "ww_entity": self._handle_entity,
            "ww_skill": self._handle_skill,
        }

        handler = handlers.get(name)
        if not handler:
            return ToolResult(
                success=False,
                error=f"Unknown tool: {name}",
            ).to_mcp_response()

        try:
            result = await handler(arguments)
            return result.to_mcp_response()
        except Exception as e:
            logger.error(f"Tool {name} failed: {e}", exc_info=True)
            return ToolResult(
                success=False,
                error=str(e),
            ).to_mcp_response()

    async def handle_list_resources(self) -> dict:
        """Handle resources/list request."""
        # List available memory sessions as resources
        resources = [
            {
                "uri": f"t4dm://sessions/{self.session_id}",
                "name": f"Current Session ({self.session_id})",
                "mimeType": "application/json",
                "description": "Current memory session",
            }
        ]

        return {"resources": resources}

    async def handle_read_resource(self, uri: str) -> dict:
        """Handle resources/read request."""
        if uri.startswith("t4dm://sessions/"):
            session_id = uri.split("/")[-1]
            stats = self._memory.get_stats() if self._memory else {}
            return {
                "contents": [
                    {
                        "uri": uri,
                        "mimeType": "application/json",
                        "text": json.dumps(stats, indent=2, default=str),
                    }
                ]
            }

        return {"contents": []}

    async def handle_list_prompts(self) -> dict:
        """Handle prompts/list request."""
        return {
            "prompts": [
                {
                    "name": p["name"],
                    "description": p["description"],
                }
                for p in MEMORY_PROMPTS
            ]
        }

    async def handle_get_prompt(self, name: str, arguments: dict) -> dict:
        """Handle prompts/get request."""
        for prompt in MEMORY_PROMPTS:
            if prompt["name"] == name:
                # Simple template substitution
                text = prompt["template"]
                for key, value in arguments.items():
                    text = text.replace(f"{{{{{key}}}}}", str(value))

                return {
                    "messages": [
                        {
                            "role": "user",
                            "content": {"type": "text", "text": text},
                        }
                    ]
                }

        return {"messages": []}

    # =========================================================================
    # Tool Handlers
    # =========================================================================

    async def _handle_store(self, args: dict) -> ToolResult:
        """Handle ww_store tool."""
        content = args.get("content")
        if not content:
            return ToolResult(success=False, error="content is required")

        episode = await self._memory.store_experience(
            content=content,
            outcome=args.get("outcome", "neutral"),
            importance=args.get("importance", 0.5),
            project=args.get("project"),
        )

        return ToolResult(
            success=True,
            data={
                "id": str(episode.id),
                "content": content[:100] + "..." if len(content) > 100 else content,
                "outcome": args.get("outcome", "neutral"),
                "stored_at": datetime.now().isoformat(),
            },
        )

    async def _handle_recall(self, args: dict) -> ToolResult:
        """Handle ww_recall tool."""
        query = args.get("query")
        if not query:
            return ToolResult(success=False, error="query is required")

        # Generate task ID for potential outcome reporting
        task_id = f"recall-{uuid4().hex[:8]}"

        memories = await self._memory.retrieve_for_task(
            task_id=task_id,
            query=query,
            limit=args.get("limit", 5),
            min_similarity=args.get("min_similarity", 0.5),
            project=args.get("project"),
        )

        return ToolResult(
            success=True,
            data={
                "task_id": task_id,
                "query": query,
                "count": len(memories),
                "memories": [
                    {
                        "id": str(m.episode.id),
                        "content": m.episode.content,
                        "outcome": m.episode.outcome,
                        "similarity": m.similarity_score,
                        "timestamp": m.episode.timestamp.isoformat(),
                    }
                    for m in memories
                ],
                "hint": "Use ww_learn_outcome with this task_id after completing your task",
            },
        )

    async def _handle_learn_outcome(self, args: dict) -> ToolResult:
        """Handle ww_learn_outcome tool."""
        task_id = args.get("task_id")
        if not task_id:
            return ToolResult(success=False, error="task_id is required")

        success = args.get("success")
        partial_credit = args.get("partial_credit")

        result = await self._memory.report_task_outcome(
            task_id=task_id,
            success=success,
            partial_credit=partial_credit,
            feedback=args.get("feedback"),
        )

        return ToolResult(
            success=True,
            data={
                "task_id": task_id,
                "memories_credited": result.credited,
                "memories_updated": result.reconsolidated,
                "total_learning_rate": result.total_lr_applied,
                "message": (
                    f"Credit assigned to {result.credited} memories. "
                    "These memories will be strengthened for future retrieval."
                ),
            },
        )

    async def _handle_consolidate(self, args: dict) -> ToolResult:
        """Handle ww_consolidate tool."""
        mode = args.get("mode", "light")

        result = await self._memory.trigger_consolidation(mode=mode)

        return ToolResult(
            success=True,
            data={
                "mode": mode,
                "result": result,
                "message": f"Memory consolidation ({mode}) completed successfully.",
            },
        )

    async def _handle_get_context(self, args: dict) -> ToolResult:
        """Handle ww_get_context tool."""
        stats = self._memory.get_stats()

        context = {
            "session_id": self.session_id,
            "api_url": self.api_url,
        }

        if args.get("include_stats", True):
            context["stats"] = stats

        if args.get("include_recent", True):
            # Get recent memories for this session
            recent = await self._memory.retrieve_for_task(
                task_id=f"context-{uuid4().hex[:8]}",
                query="recent session activity",
                limit=5,
            )
            context["recent_memories"] = [
                {
                    "content": m.episode.content[:100],
                    "outcome": m.episode.outcome,
                    "timestamp": m.episode.timestamp.isoformat(),
                }
                for m in recent
            ]

        return ToolResult(success=True, data=context)

    async def _handle_entity(self, args: dict) -> ToolResult:
        """Handle ww_entity tool for semantic memory operations."""
        action = args.get("action")
        client = self._memory._get_client()

        try:
            if action == "create":
                name = args.get("name")
                entity_type = args.get("entity_type", "concept")
                summary = args.get("summary", "")
                if not name:
                    return ToolResult(success=False, error="Entity name is required")

                entity = await client.create_entity(
                    name=name,
                    entity_type=entity_type,
                    summary=summary,
                )
                return ToolResult(
                    success=True,
                    data={
                        "action": "create",
                        "entity_id": str(entity.id),
                        "name": entity.name,
                        "entity_type": entity.entity_type,
                        "message": f"Created entity '{name}'",
                    },
                )

            elif action == "get":
                name = args.get("name")
                entity_id = args.get("entity_id")
                if entity_id:
                    from uuid import UUID
                    entity = await client.get_entity(UUID(entity_id))
                    return ToolResult(
                        success=True,
                        data={
                            "action": "get",
                            "entity_id": str(entity.id),
                            "name": entity.name,
                            "entity_type": entity.entity_type,
                            "summary": entity.summary,
                            "details": entity.details,
                        },
                    )
                return ToolResult(success=False, error="entity_id required for get action")

            elif action == "search":
                query = args.get("query", args.get("name", ""))
                if not query:
                    return ToolResult(success=False, error="Query required for search")

                entities = await client.recall_entities(query, limit=10)
                return ToolResult(
                    success=True,
                    data={
                        "action": "search",
                        "query": query,
                        "results": [
                            {
                                "id": str(e.id),
                                "name": e.name,
                                "type": e.entity_type,
                                "summary": e.summary[:100] if e.summary else "",
                            }
                            for e in entities
                        ],
                        "count": len(entities),
                    },
                )

            elif action == "relate":
                source_id = args.get("source_id")
                target_id = args.get("target_id")
                relation_type = args.get("relation_type", "RELATED_TO")
                if not source_id or not target_id:
                    return ToolResult(success=False, error="source_id and target_id required")

                from uuid import UUID
                relation = await client.create_relation(
                    source_id=UUID(source_id),
                    target_id=UUID(target_id),
                    relation_type=relation_type,
                )
                return ToolResult(
                    success=True,
                    data={
                        "action": "relate",
                        "source_id": source_id,
                        "target_id": target_id,
                        "relation_type": relation_type,
                        "message": f"Created {relation_type} relation",
                    },
                )

            else:
                return ToolResult(
                    success=False,
                    error=f"Unknown entity action: {action}. Use create, get, search, or relate.",
                )

        except Exception as e:
            logger.error(f"Entity operation failed: {e}")
            return ToolResult(success=False, error=str(e))

    async def _handle_skill(self, args: dict) -> ToolResult:
        """Handle ww_skill tool for procedural memory operations."""
        action = args.get("action")
        client = self._memory._get_client()

        try:
            if action == "create":
                name = args.get("name")
                domain = args.get("domain", "general")
                task = args.get("task", "")
                steps = args.get("steps", [])
                if not name:
                    return ToolResult(success=False, error="Skill name is required")

                skill = await client.create_skill(
                    name=name,
                    domain=domain,
                    task=task or name,
                    steps=steps,
                )
                return ToolResult(
                    success=True,
                    data={
                        "action": "create",
                        "skill_id": str(skill.id),
                        "name": skill.name,
                        "domain": skill.domain,
                        "message": f"Created skill '{name}'",
                    },
                )

            elif action == "get":
                skill_id = args.get("skill_id")
                name = args.get("name")
                if skill_id:
                    from uuid import UUID
                    skill = await client.get_skill(UUID(skill_id))
                    return ToolResult(
                        success=True,
                        data={
                            "action": "get",
                            "skill_id": str(skill.id),
                            "name": skill.name,
                            "domain": skill.domain,
                            "steps": [
                                {"order": s.order, "action": s.action, "tool": s.tool}
                                for s in skill.steps
                            ],
                            "success_rate": skill.success_rate,
                            "execution_count": skill.execution_count,
                        },
                    )
                return ToolResult(success=False, error="skill_id required for get action")

            elif action == "search":
                query = args.get("query", args.get("name", ""))
                domain = args.get("domain")
                if not query:
                    return ToolResult(success=False, error="Query required for search")

                skills = await client.recall_skills(query, domain=domain, limit=10)
                return ToolResult(
                    success=True,
                    data={
                        "action": "search",
                        "query": query,
                        "results": [
                            {
                                "id": str(s.id),
                                "name": s.name,
                                "domain": s.domain,
                                "success_rate": s.success_rate,
                            }
                            for s in skills
                        ],
                        "count": len(skills),
                    },
                )

            elif action == "record_execution":
                skill_id = args.get("skill_id")
                success = args.get("success", True)
                if not skill_id:
                    return ToolResult(success=False, error="skill_id required")

                from uuid import UUID
                skill = await client.record_execution(
                    skill_id=UUID(skill_id),
                    success=success,
                )
                return ToolResult(
                    success=True,
                    data={
                        "action": "record_execution",
                        "skill_id": skill_id,
                        "success": success,
                        "new_success_rate": skill.success_rate,
                        "execution_count": skill.execution_count,
                    },
                )

            else:
                return ToolResult(
                    success=False,
                    error=f"Unknown skill action: {action}. Use create, get, search, or record_execution.",
                )

        except Exception as e:
            logger.error(f"Skill operation failed: {e}")
            return ToolResult(success=False, error=str(e))


async def run_stdio_server(server: WorldWeaverMCPServer):
    """
    Run MCP server over stdio.

    Reads JSON-RPC messages from stdin and writes responses to stdout.
    """
    reader = asyncio.StreamReader()
    protocol = asyncio.StreamReaderProtocol(reader)
    await asyncio.get_event_loop().connect_read_pipe(lambda: protocol, sys.stdin)

    writer_transport, writer_protocol = await asyncio.get_event_loop().connect_write_pipe(
        asyncio.streams.FlowControlMixin,
        sys.stdout,
    )
    writer = asyncio.StreamWriter(writer_transport, writer_protocol, None, asyncio.get_event_loop())

    try:
        while True:
            # Read content-length header
            header_line = await reader.readline()
            if not header_line:
                break

            header = header_line.decode().strip()
            if not header.startswith("Content-Length:"):
                continue

            content_length = int(header.split(":")[1].strip())

            # Read empty line
            await reader.readline()

            # Read JSON content
            content = await reader.read(content_length)
            request = json.loads(content.decode())

            # Handle request
            response = await handle_request(server, request)

            # Write response
            response_json = json.dumps(response)
            response_bytes = response_json.encode()

            header = f"Content-Length: {len(response_bytes)}\r\n\r\n"
            writer.write(header.encode() + response_bytes)
            await writer.drain()

    except asyncio.CancelledError:
        pass
    finally:
        await server.shutdown()


async def handle_request(server: WorldWeaverMCPServer, request: dict) -> dict:
    """Handle a single MCP JSON-RPC request."""
    method = request.get("method", "")
    params = request.get("params", {})
    request_id = request.get("id")

    result = None
    error = None

    try:
        if method == "initialize":
            result = await server.handle_initialize(params)
        elif method == "tools/list":
            result = await server.handle_list_tools()
        elif method == "tools/call":
            result = await server.handle_call_tool(
                params.get("name", ""),
                params.get("arguments", {}),
            )
        elif method == "resources/list":
            result = await server.handle_list_resources()
        elif method == "resources/read":
            result = await server.handle_read_resource(params.get("uri", ""))
        elif method == "prompts/list":
            result = await server.handle_list_prompts()
        elif method == "prompts/get":
            result = await server.handle_get_prompt(
                params.get("name", ""),
                params.get("arguments", {}),
            )
        elif method == "notifications/initialized":
            # Acknowledgment, no response needed
            return {}
        else:
            error = {"code": -32601, "message": f"Unknown method: {method}"}

    except Exception as e:
        logger.error(f"Request handling failed: {e}", exc_info=True)
        error = {"code": -32603, "message": str(e)}

    response = {"jsonrpc": "2.0", "id": request_id}
    if error:
        response["error"] = error
    else:
        response["result"] = result

    return response


def create_mcp_server(
    api_url: str | None = None,
    session_id: str | None = None,
    api_key: str | None = None,
) -> WorldWeaverMCPServer:
    """
    Create an MCP server instance.

    Args:
        api_url: T4DM API URL (default from T4DM_API_URL env)
        session_id: Session ID (default from T4DM_SESSION_ID env)
        api_key: API key (default from T4DM_API_KEY env)

    Returns:
        Configured MCP server
    """
    return WorldWeaverMCPServer(
        api_url=api_url or os.environ.get("T4DM_API_URL", "http://localhost:8765"),
        session_id=session_id or os.environ.get("T4DM_SESSION_ID"),
        api_key=api_key or os.environ.get("T4DM_API_KEY"),
    )


async def run_mcp_server():
    """Run MCP server with stdio transport."""
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
        stream=sys.stderr,  # Log to stderr, not stdout (used for MCP)
    )

    server = create_mcp_server()
    logger.info("Starting T4DM MCP server...")

    await run_stdio_server(server)


def main():
    """Entry point for MCP server."""
    asyncio.run(run_mcp_server())


if __name__ == "__main__":
    main()
