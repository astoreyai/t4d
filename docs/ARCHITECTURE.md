# World Weaver (WW) - Personal AI Assistant Framework

**Version**: 0.1.0-alpha
**Status**: Architecture Design Phase
**Codename**: Jarvis Core

---

## Vision

A modular, provider-agnostic AI assistant framework that orchestrates specialized agents for knowledge management, semantic processing, algorithm design, and domain-specific assistance. Built on principles from Anthropic's long-running agent research.

---

## Core Architecture Principles

### From Anthropic's Long-Running Agent Research

1. **Two-Agent Pattern**
   - **Initializer Agent**: Sets up environment, feature lists, progress tracking
   - **Task Agents**: Make incremental progress, leave clean states

2. **State Artifacts**
   - `ww-progress.json` - Structured progress log with context
   - `ww-features.json` - Feature requirements and completion status
   - Git commits with descriptive messages
   - `init.sh` - Environment bootstrapping

3. **Incremental Progress**
   - One task/feature at a time
   - Clean state between sessions
   - Self-verification before completion

---

## System Layers

```
┌─────────────────────────────────────────────────────────────────┐
│                      APPLICATION LAYER                          │
│  ┌─────────────┐ ┌─────────────┐ ┌─────────────┐ ┌───────────┐ │
│  │ Knowledge   │ │  Semantic   │ │   Graph     │ │  Domain   │ │
│  │ Base Agent  │ │   Agent     │ │   Agent     │ │  Agents   │ │
│  └─────────────┘ └─────────────┘ └─────────────┘ └───────────┘ │
├─────────────────────────────────────────────────────────────────┤
│                      ORCHESTRATION LAYER                        │
│  ┌─────────────────────────────────────────────────────────────┐│
│  │              Agent Coordinator & Router                     ││
│  │   • Session Management    • Context Bridging                ││
│  │   • Progress Tracking     • Feature Verification            ││
│  └─────────────────────────────────────────────────────────────┘│
├─────────────────────────────────────────────────────────────────┤
│                       MEMORY LAYER                              │
│  ┌───────────────────┐  ┌───────────────────────────────────┐  │
│  │   HOT MEMORY      │  │         WARM MEMORY               │  │
│  │  (In-Session)     │  │      (Cross-Session)              │  │
│  │  • Context Window │  │  • Vector Store (embeddings)      │  │
│  │  • Working Memory │  │  • Progress Files                 │  │
│  │  • Task State     │  │  • Feature Lists                  │  │
│  └───────────────────┘  └───────────────────────────────────┘  │
│  ┌─────────────────────────────────────────────────────────────┐│
│  │                    COLD MEMORY                              ││
│  │   • SQLite/PostgreSQL (structured data)                     ││
│  │   • Document Store (knowledge artifacts)                    ││
│  │   • Graph Database (relationships)                          ││
│  └─────────────────────────────────────────────────────────────┘│
├─────────────────────────────────────────────────────────────────┤
│                      EMBEDDING LAYER                            │
│  ┌─────────────────────────────────────────────────────────────┐│
│  │           Provider-Agnostic Embedding Interface             ││
│  │  ┌─────────┐ ┌─────────┐ ┌─────────┐ ┌─────────┐           ││
│  │  │ Voyage  │ │ OpenAI  │ │ Cohere  │ │ Local   │           ││
│  │  │  AI     │ │ Ada-003 │ │ Embed   │ │ (SBERT) │           ││
│  │  └─────────┘ └─────────┘ └─────────┘ └─────────┘           ││
│  └─────────────────────────────────────────────────────────────┘│
├─────────────────────────────────────────────────────────────────┤
│                      STORAGE LAYER                              │
│  ┌─────────────────────────────────────────────────────────────┐│
│  │           Provider-Agnostic Storage Interface               ││
│  │  ┌─────────┐ ┌─────────┐ ┌─────────┐ ┌─────────┐           ││
│  │  │ ChromaDB│ │ Pinecone│ │  FAISS  │ │ Qdrant  │           ││
│  │  └─────────┘ └─────────┘ └─────────┘ └─────────┘           ││
│  └─────────────────────────────────────────────────────────────┘│
└─────────────────────────────────────────────────────────────────┘
```

---

## Memory Layer Design

### 1. Hot Memory (In-Session)

**Purpose**: Fast access to current context and task state

```python
class HotMemory:
    """In-session working memory"""

    context_window: list[Message]      # Current conversation
    task_state: TaskState              # Active task tracking
    working_set: dict[str, Any]        # Scratch space
    attention_cache: dict[str, float]  # Relevance scores
```

### 2. Warm Memory (Cross-Session)

**Purpose**: Bridge context windows, track progress

**Components**:
- **Progress Log** (`ww-progress.json`): What agents have done
- **Feature List** (`ww-features.json`): Requirements tracking
- **Vector Index**: Fast semantic retrieval
- **Recent Context**: Last N session summaries

### 3. Cold Memory (Long-Term)

**Purpose**: Persistent knowledge, relationships, history

**Components**:
- **Document Store**: Full knowledge artifacts
- **Relational Store**: Structured metadata
- **Graph Store**: Entity relationships

---

## Provider-Agnostic Interfaces

### Embedding Provider Interface

```python
from abc import ABC, abstractmethod
from typing import Protocol

class EmbeddingProvider(Protocol):
    """Provider-agnostic embedding interface"""

    @abstractmethod
    async def embed(self, texts: list[str]) -> list[list[float]]:
        """Generate embeddings for text inputs"""
        ...

    @abstractmethod
    async def embed_query(self, query: str) -> list[float]:
        """Generate embedding optimized for query"""
        ...

    @property
    @abstractmethod
    def dimension(self) -> int:
        """Embedding dimension"""
        ...

    @property
    @abstractmethod
    def model_name(self) -> str:
        """Model identifier"""
        ...
```

### Vector Store Interface

```python
class VectorStore(Protocol):
    """Provider-agnostic vector store interface"""

    @abstractmethod
    async def add(
        self,
        ids: list[str],
        embeddings: list[list[float]],
        documents: list[str],
        metadata: list[dict] | None = None
    ) -> None:
        """Add documents with embeddings"""
        ...

    @abstractmethod
    async def query(
        self,
        embedding: list[float],
        top_k: int = 10,
        filter: dict | None = None
    ) -> list[SearchResult]:
        """Semantic search"""
        ...

    @abstractmethod
    async def delete(self, ids: list[str]) -> None:
        """Remove documents"""
        ...
```

---

## Agent Framework

### Base Agent Structure

```python
class BaseAgent(ABC):
    """Base class for all WW agents"""

    def __init__(
        self,
        memory: MemoryLayer,
        config: AgentConfig
    ):
        self.memory = memory
        self.config = config
        self.progress_file = "ww-progress.json"
        self.features_file = "ww-features.json"

    @abstractmethod
    async def initialize(self) -> None:
        """First-run initialization (Initializer pattern)"""
        ...

    @abstractmethod
    async def execute(self, task: Task) -> TaskResult:
        """Execute single task increment"""
        ...

    async def get_bearings(self) -> SessionContext:
        """Standard session start sequence"""
        # 1. Read progress log
        # 2. Read feature list
        # 3. Check git status
        # 4. Run verification tests
        return SessionContext(...)

    async def checkpoint(self, result: TaskResult) -> None:
        """Clean state checkpoint"""
        # 1. Update progress log
        # 2. Update feature status
        # 3. Git commit if appropriate
        ...
```

### Initializer Agent Pattern

```python
class InitializerAgent(BaseAgent):
    """Sets up environment for long-running work"""

    async def initialize(self) -> None:
        # 1. Create progress tracking file
        await self._create_progress_file()

        # 2. Generate comprehensive feature list from spec
        await self._generate_feature_list()

        # 3. Create init.sh for environment setup
        await self._create_init_script()

        # 4. Initial git commit
        await self._initial_commit()

        # 5. Create verification tests
        await self._create_verification_suite()
```

---

## Specialized Agents

### 1. Knowledge Base Agent

**Purpose**: Manage knowledge capture, retrieval, and organization

**Capabilities**:
- Extract knowledge from conversations
- Chunk and embed documents
- Semantic search and retrieval
- Knowledge graph construction
- Cross-reference linking

### 2. Semantic Agent

**Purpose**: Encoding/decoding semantic representations

**Capabilities**:
- Text-to-embedding conversion
- Semantic similarity computation
- Concept extraction
- Relationship detection
- Dimension reduction for visualization

### 3. Graph Agent

**Purpose**: Design and traverse knowledge graphs

**Capabilities**:
- Graph schema design
- Entity extraction and linking
- Path finding algorithms
- Community detection
- Graph neural network integration

### 4. Fine-Tuning Orchestrator

**Purpose**: Manage model fine-tuning workflows

**Capabilities**:
- Dataset preparation
- Training configuration
- Progress monitoring
- Model evaluation
- Version management

### 5. Domain Agents

**Neuroscience Agent**:
- Neural pathway modeling
- Brain region mapping
- Cognitive process simulation

**Computational Biology Agent**:
- Protein structure analysis
- Gene expression modeling
- Pathway analysis

---

## Progress Tracking Schema

### ww-progress.json

```json
{
  "project": "world-weaver",
  "version": "0.1.0",
  "last_updated": "2025-11-27T10:00:00Z",
  "sessions": [
    {
      "id": "session-001",
      "started": "2025-11-27T09:00:00Z",
      "ended": "2025-11-27T10:00:00Z",
      "agent": "initializer",
      "summary": "Set up project structure and memory layer",
      "tasks_completed": [
        "Created project directory structure",
        "Implemented embedding provider interface",
        "Set up SQLite for cold storage"
      ],
      "next_steps": [
        "Implement vector store abstraction",
        "Create knowledge base agent"
      ],
      "git_commits": ["abc123", "def456"]
    }
  ],
  "current_focus": "memory-layer-implementation",
  "blockers": []
}
```

### ww-features.json

```json
{
  "project": "world-weaver",
  "generated": "2025-11-27T09:00:00Z",
  "features": [
    {
      "id": "MEM-001",
      "category": "memory",
      "description": "Provider-agnostic embedding interface with support for Voyage, OpenAI, Cohere, and local models",
      "priority": 1,
      "passes": false,
      "verification_steps": [
        "Import EmbeddingProvider protocol",
        "Instantiate VoyageEmbedding provider",
        "Generate embedding for sample text",
        "Verify dimension matches expected"
      ]
    },
    {
      "id": "MEM-002",
      "category": "memory",
      "description": "Vector store abstraction supporting ChromaDB, FAISS, Pinecone, and Qdrant",
      "priority": 1,
      "passes": false,
      "verification_steps": [
        "Import VectorStore protocol",
        "Instantiate ChromaDB store",
        "Add sample documents",
        "Query and verify results"
      ]
    }
  ]
}
```

---

## Directory Structure

```
t4dm/
├── ARCHITECTURE.md           # This document
├── ww-progress.json          # Progress tracking
├── ww-features.json          # Feature requirements
├── init.sh                   # Environment setup
├── pyproject.toml            # Project configuration
├── src/
│   └── t4dm/
│       ├── __init__.py
│       ├── core/
│       │   ├── __init__.py
│       │   ├── config.py         # Configuration management
│       │   └── types.py          # Core type definitions
│       ├── memory/
│       │   ├── __init__.py
│       │   ├── hot.py            # In-session memory
│       │   ├── warm.py           # Cross-session memory
│       │   ├── cold.py           # Long-term storage
│       │   └── layer.py          # Unified memory layer
│       ├── embedding/
│       │   ├── __init__.py
│       │   ├── protocol.py       # Provider interface
│       │   ├── voyage.py         # Voyage AI provider
│       │   ├── openai.py         # OpenAI provider
│       │   ├── local.py          # Local/SBERT provider
│       │   └── factory.py        # Provider factory
│       ├── storage/
│       │   ├── __init__.py
│       │   ├── protocol.py       # Store interface
│       │   ├── chromadb.py       # ChromaDB implementation
│       │   ├── faiss_store.py    # FAISS implementation
│       │   └── factory.py        # Store factory
│       ├── agents/
│       │   ├── __init__.py
│       │   ├── base.py           # Base agent class
│       │   ├── initializer.py    # Initializer agent
│       │   ├── coordinator.py    # Agent orchestration
│       │   ├── knowledge/
│       │   │   ├── __init__.py
│       │   │   └── agent.py      # Knowledge base agent
│       │   ├── semantic/
│       │   │   ├── __init__.py
│       │   │   └── agent.py      # Semantic processing agent
│       │   ├── graph/
│       │   │   ├── __init__.py
│       │   │   └── agent.py      # Graph traversal agent
│       │   └── domain/
│       │       ├── __init__.py
│       │       ├── neuroscience.py
│       │       └── compbio.py
│       └── cli/
│           ├── __init__.py
│           └── main.py           # CLI interface
├── tests/
│   ├── __init__.py
│   ├── conftest.py
│   ├── test_memory/
│   ├── test_embedding/
│   ├── test_storage/
│   └── test_agents/
└── docs/
    ├── getting-started.md
    ├── memory-layer.md
    ├── agent-development.md
    └── provider-integration.md
```

---

## Implementation Phases

### Phase 0: Foundation (Current)
- [x] Architecture design
- [ ] Project structure
- [ ] Core type definitions
- [ ] Configuration management

### Phase 1: Memory Layer
- [ ] Embedding provider protocol
- [ ] Voyage AI implementation
- [ ] Local SBERT implementation
- [ ] Vector store protocol
- [ ] ChromaDB implementation
- [ ] FAISS implementation
- [ ] Memory layer integration

### Phase 2: Agent Framework
- [ ] Base agent class
- [ ] Initializer agent
- [ ] Progress tracking
- [ ] Feature verification
- [ ] Session management

### Phase 3: Specialized Agents
- [ ] Knowledge base agent
- [ ] Semantic agent
- [ ] Graph agent

### Phase 4: Domain Agents
- [ ] Fine-tuning orchestrator
- [ ] Neuroscience agent
- [ ] Computational biology agent

### Phase 5: Integration
- [ ] Claude Code CLI integration
- [ ] Skill packaging
- [ ] Documentation

---

## Claude Code Integration

This framework is designed to work directly with Claude Code CLI:

1. **Skills**: Each agent becomes a skill in `~/github/astoreyai/claude-skills/`
2. **Agents**: Specialized agents register as Claude Code subagents
3. **Commands**: Slash commands for common operations
4. **Hooks**: Session start/end for context bridging

---

## Next Steps

1. Create project structure and core files
2. Implement embedding provider protocol
3. Implement vector store protocol
4. Build memory layer integration
5. Create base agent class
6. Implement initializer agent

---

## References

- [Effective harnesses for long-running agents](https://www.anthropic.com/engineering/effective-agent-harnesses) - Anthropic Engineering
- [Claude Agent SDK](https://docs.anthropic.com/en/docs/claude-code)
- [ChromaDB](https://docs.trychroma.com/)
- [FAISS](https://github.com/facebookresearch/faiss)
- [Sentence Transformers](https://www.sbert.net/)
