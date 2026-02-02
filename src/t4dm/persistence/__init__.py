"""
T4DM Persistence Layer

Provides crash-safe durability for the memory system through:
- Write-Ahead Logging (WAL) for operation durability
- Periodic checkpointing for fast recovery
- Cold/warm start protocols
- Graceful shutdown handling

Architecture:
============

    ┌─────────────────────────────────────────────────────────┐
    │                    Application Layer                     │
    │         (BufferManager, LearnedGate, Scorer, etc.)      │
    └────────────────────────┬────────────────────────────────┘
                             │
                             ▼
    ┌─────────────────────────────────────────────────────────┐
    │                  Persistence Manager                     │
    │                                                         │
    │  ┌─────────────┐  ┌──────────────┐  ┌───────────────┐  │
    │  │    WAL      │  │  Checkpoint  │  │   Recovery    │  │
    │  │  (append)   │  │  (periodic)  │  │  (startup)    │  │
    │  └──────┬──────┘  └──────┬───────┘  └───────┬───────┘  │
    │         │                │                  │          │
    │         ▼                ▼                  ▼          │
    │  ┌─────────────────────────────────────────────────┐   │
    │  │              Durable Storage (disk)              │   │
    │  │   wal/        checkpoints/       recovery.log    │   │
    │  └─────────────────────────────────────────────────┘   │
    └─────────────────────────────────────────────────────────┘

Recovery Protocol:
=================

    STARTUP
       │
       ▼
    ┌──────────────────┐
    │ Checkpoint exists?│
    └────────┬─────────┘
             │
      ┌──────┴──────┐
      │             │
      ▼             ▼
   WARM START    COLD START
      │             │
      ▼             ▼
   Load checkpoint  Initialize defaults
      │             │
      ▼             │
   Replay WAL       │
      │             │
      └──────┬──────┘
             │
             ▼
    Verify consistency
             │
             ▼
    Start serving

Shutdown Protocol:
=================

    SIGTERM/SIGINT
           │
           ▼
    Stop accepting requests
           │
           ▼
    Drain in-flight ops (timeout: 30s)
           │
           ▼
    Force checkpoint
           │
           ▼
    Flush & close WAL
           │
           ▼
    Close storage connections
           │
           ▼
    Exit(0)
"""

from .checkpoint import (
    Checkpoint,
    CheckpointConfig,
    CheckpointManager,
)
from .manager import (
    PersistenceConfig,
    PersistenceManager,
    get_persistence,
    set_persistence,
)
from .recovery import (
    RecoveryManager,
    RecoveryMode,
    RecoveryResult,
)
from .shutdown import (
    ShutdownConfig,
    ShutdownManager,
    register_shutdown_handlers,
)
from .wal import (
    WALCorruptionError,
    WALEntry,
    WALOperation,
    WriteAheadLog,
)

__all__ = [
    # WAL
    "WriteAheadLog",
    "WALEntry",
    "WALOperation",
    "WALCorruptionError",
    # Checkpoint
    "CheckpointManager",
    "Checkpoint",
    "CheckpointConfig",
    # Recovery
    "RecoveryManager",
    "RecoveryMode",
    "RecoveryResult",
    # Shutdown
    "ShutdownManager",
    "ShutdownConfig",
    "register_shutdown_handlers",
    # Manager
    "PersistenceManager",
    "PersistenceConfig",
    "get_persistence",
    "set_persistence",
]
