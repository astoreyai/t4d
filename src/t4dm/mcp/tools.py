"""
MCP Tool Definitions for T4DM.

Provides memory tools for Claude Code/Desktop integration:
- t4dm_store: Store experiences in episodic memory
- t4dm_search: Retrieve relevant memories
- t4dm_learn: Report task outcomes for learning
- t4dm_consolidate: Trigger memory consolidation
- t4dm_context: Get project/session context
- t4dm_entity: Create/retrieve semantic entities
"""

from dataclasses import dataclass
from typing import Any

# Tool definitions using MCP schema format
MEMORY_TOOLS = [
    {
        "name": "t4dm_store",
        "description": (
            "Store information in T4DM episodic memory. "
            "Use this to remember important learnings, patterns, solutions, "
            "or any information that might be useful in future sessions."
        ),
        "inputSchema": {
            "type": "object",
            "properties": {
                "content": {
                    "type": "string",
                    "description": "The content to store in memory",
                },
                "outcome": {
                    "type": "string",
                    "enum": ["success", "failure", "neutral"],
                    "description": "Outcome category for this experience",
                    "default": "neutral",
                },
                "importance": {
                    "type": "number",
                    "minimum": 0,
                    "maximum": 1,
                    "description": "Importance score (0-1) affecting memory strength",
                    "default": 0.5,
                },
                "project": {
                    "type": "string",
                    "description": "Optional project context for filtering",
                },
                "tags": {
                    "type": "array",
                    "items": {"type": "string"},
                    "description": "Optional tags for categorization",
                },
            },
            "required": ["content"],
        },
    },
    {
        "name": "t4dm_search",
        "description": (
            "Search episodic memory for relevant past experiences. "
            "Use this to retrieve context, patterns, or solutions from previous sessions. "
            "Returns semantically similar memories ranked by relevance."
        ),
        "inputSchema": {
            "type": "object",
            "properties": {
                "query": {
                    "type": "string",
                    "description": "Semantic search query",
                },
                "limit": {
                    "type": "integer",
                    "minimum": 1,
                    "maximum": 20,
                    "description": "Maximum memories to return",
                    "default": 5,
                },
                "project": {
                    "type": "string",
                    "description": "Filter by project context",
                },
                "outcome": {
                    "type": "string",
                    "enum": ["success", "failure", "neutral"],
                    "description": "Filter by outcome type",
                },
                "min_similarity": {
                    "type": "number",
                    "minimum": 0,
                    "maximum": 1,
                    "description": "Minimum similarity threshold",
                    "default": 0.5,
                },
            },
            "required": ["query"],
        },
    },
    {
        "name": "t4dm_learn",
        "description": (
            "Report task outcome to enable learning. "
            "Call this after completing a task where you used recalled memories. "
            "Positive outcomes strengthen the memories that helped; "
            "negative outcomes help avoid similar patterns in the future."
        ),
        "inputSchema": {
            "type": "object",
            "properties": {
                "task_id": {
                    "type": "string",
                    "description": "Task identifier from the recall operation",
                },
                "success": {
                    "type": "boolean",
                    "description": "Whether the task succeeded",
                },
                "partial_credit": {
                    "type": "number",
                    "minimum": 0,
                    "maximum": 1,
                    "description": "Partial success score (overrides success boolean)",
                },
                "feedback": {
                    "type": "string",
                    "description": "Optional feedback about what worked or didn't",
                },
            },
            "required": ["task_id", "success"],
        },
    },
    {
        "name": "t4dm_consolidate",
        "description": (
            "Trigger memory consolidation (mimics biological sleep). "
            "Use 'light' during work breaks, 'deep' at session end, "
            "'full' for overnight consolidation. "
            "Consolidation strengthens important memories and prunes weak ones."
        ),
        "inputSchema": {
            "type": "object",
            "properties": {
                "mode": {
                    "type": "string",
                    "enum": ["light", "deep", "full"],
                    "description": "Consolidation intensity",
                    "default": "light",
                },
                "session_id": {
                    "type": "string",
                    "description": "Session to consolidate (current if not specified)",
                },
            },
        },
    },
    {
        "name": "t4dm_context",
        "description": (
            "Get current session and project context. "
            "Returns recent memories, session statistics, and project-specific knowledge."
        ),
        "inputSchema": {
            "type": "object",
            "properties": {
                "project": {
                    "type": "string",
                    "description": "Project to get context for",
                },
                "include_recent": {
                    "type": "boolean",
                    "description": "Include recent session memories",
                    "default": True,
                },
                "include_stats": {
                    "type": "boolean",
                    "description": "Include memory statistics",
                    "default": True,
                },
            },
        },
    },
    {
        "name": "t4dm_entity",
        "description": (
            "Create or retrieve semantic entities (concepts, people, projects). "
            "Entities form the knowledge graph for structured memory."
        ),
        "inputSchema": {
            "type": "object",
            "properties": {
                "action": {
                    "type": "string",
                    "enum": ["create", "get", "search", "relate"],
                    "description": "Action to perform",
                },
                "name": {
                    "type": "string",
                    "description": "Entity name (for create/get)",
                },
                "entity_type": {
                    "type": "string",
                    "description": "Entity type (concept, person, project, etc.)",
                },
                "summary": {
                    "type": "string",
                    "description": "Entity summary (for create)",
                },
                "query": {
                    "type": "string",
                    "description": "Search query (for search action)",
                },
                "source_id": {
                    "type": "string",
                    "description": "Source entity ID (for relate action)",
                },
                "target_id": {
                    "type": "string",
                    "description": "Target entity ID (for relate action)",
                },
                "relation_type": {
                    "type": "string",
                    "description": "Relationship type (for relate action)",
                },
            },
            "required": ["action"],
        },
    },
    {
        "name": "t4dm_skill",
        "description": (
            "Store or retrieve procedural skills (how-to knowledge). "
            "Skills are step-by-step procedures that can be recalled and executed."
        ),
        "inputSchema": {
            "type": "object",
            "properties": {
                "action": {
                    "type": "string",
                    "enum": ["create", "get", "search", "record_execution"],
                    "description": "Action to perform",
                },
                "name": {
                    "type": "string",
                    "description": "Skill name (for create/get)",
                },
                "domain": {
                    "type": "string",
                    "description": "Skill domain (coding, debugging, etc.)",
                },
                "task": {
                    "type": "string",
                    "description": "Task description (for create)",
                },
                "steps": {
                    "type": "array",
                    "items": {
                        "type": "object",
                        "properties": {
                            "order": {"type": "integer"},
                            "action": {"type": "string"},
                            "tool": {"type": "string"},
                        },
                    },
                    "description": "Procedure steps (for create)",
                },
                "query": {
                    "type": "string",
                    "description": "Search query (for search action)",
                },
                "skill_id": {
                    "type": "string",
                    "description": "Skill ID (for record_execution)",
                },
                "success": {
                    "type": "boolean",
                    "description": "Execution success (for record_execution)",
                },
            },
            "required": ["action"],
        },
    },
]


# Prompt templates for common operations
MEMORY_PROMPTS = [
    {
        "name": "session_start",
        "description": "Initialize a new T4DM session with context loading",
        "template": """
You are starting a new session with T4DM memory enabled.

Session ID: {{session_id}}
Project: {{project}}

Available memory tools:
- t4dm_search: Search for relevant past experiences
- t4dm_store: Save important learnings
- t4dm_learn: Report task success/failure for learning

Before starting work, consider recalling relevant context:
1. What patterns or solutions have worked before?
2. What mistakes should be avoided?
3. What project-specific knowledge is available?
""",
    },
    {
        "name": "session_end",
        "description": "Summarize and consolidate session learnings",
        "template": """
Session is ending. Please:

1. Store any important learnings from this session
2. Report outcomes for tasks where memories were used
3. Trigger consolidation to strengthen useful memories

Use t4dm_store for each key insight, and t4dm_consolidate with mode="deep".
""",
    },
    {
        "name": "task_reflection",
        "description": "Reflect on task completion and store learnings",
        "template": """
Task completed: {{task_description}}
Outcome: {{outcome}}

Reflect on this task:
1. What approach worked well?
2. What could be improved?
3. What patterns should be remembered?

Store insights using t4dm_store and report outcome using t4dm_learn.
""",
    },
]


@dataclass
class ToolResult:
    """Result from tool execution."""

    success: bool
    data: dict[str, Any] | None = None
    error: str | None = None

    def to_mcp_response(self) -> dict[str, Any]:
        """Convert to MCP response format."""
        if self.success:
            return {
                "content": [
                    {
                        "type": "text",
                        "text": self._format_data(),
                    }
                ]
            }
        else:
            return {
                "content": [
                    {
                        "type": "text",
                        "text": f"Error: {self.error}",
                    }
                ],
                "isError": True,
            }

    def _format_data(self) -> str:
        """Format data for display."""
        if not self.data:
            return "Success"

        import json
        return json.dumps(self.data, indent=2, default=str)
