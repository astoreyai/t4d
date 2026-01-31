"""
CRUD Manager - Create, Read, Update, Delete operations for World Weaver memories.

Provides batch operations, bulk imports, and safe deletion with confirmation.
"""

import asyncio
from datetime import datetime
from typing import Any, Optional
from uuid import UUID

try:
    from rich.console import Console
    from rich.panel import Panel
    from rich.progress import BarColumn, Progress, SpinnerColumn, TextColumn
    from rich.prompt import Confirm, Prompt
    from rich.table import Table
    RICH_AVAILABLE = True
except ImportError:
    RICH_AVAILABLE = False

import logging

from t4dm.core.types import (
    Domain,
    Entity,
    Episode,
    EpisodeContext,
    Outcome,
    Procedure,
    ProcedureStep,
)
from t4dm.memory.episodic import EpisodicMemory
from t4dm.memory.procedural import ProceduralMemory
from t4dm.memory.semantic import SemanticMemory

logger = logging.getLogger(__name__)


class CRUDManager:
    """
    CRUD operations interface for World Weaver memories.

    Features:
    - Create episodes, entities, skills with validation
    - Read with filtering and pagination
    - Update with confirmation
    - Delete with cascade options
    - Batch operations with progress tracking
    """

    def __init__(
        self,
        session_id: str | None = None,
        console: Optional["Console"] = None,
    ):
        """
        Initialize CRUD manager.

        Args:
            session_id: Session to manage
            console: Rich console instance
        """
        if not RICH_AVAILABLE:
            raise ImportError(
                "rich library required for CRUDManager. "
                "Install with: pip install rich"
            )

        self.session_id = session_id or "default"
        self.console = console or Console()

        self.episodic = EpisodicMemory(session_id=session_id)
        self.semantic = SemanticMemory(session_id=session_id)
        self.procedural = ProceduralMemory(session_id=session_id)

        self._initialized = False

    async def initialize(self) -> None:
        """Initialize storage backends."""
        if self._initialized:
            return

        with Progress(
            SpinnerColumn(),
            TextColumn("[progress.description]{task.description}"),
            console=self.console,
        ) as progress:
            task = progress.add_task("Initializing CRUD manager...", total=None)
            await self.episodic.initialize()
            await self.semantic.initialize()
            await self.procedural.initialize()
            progress.update(task, completed=True)

        self._initialized = True

    # ==================== CREATE ====================

    async def create_episode(
        self,
        content: str,
        outcome: str = "neutral",
        emotional_valence: float = 0.5,
        context: dict[str, Any] | None = None,
    ) -> Episode:
        """
        Create a new episode.

        Args:
            content: Episode content
            outcome: success, failure, partial, neutral
            emotional_valence: Importance [0, 1]
            context: Optional context dict

        Returns:
            Created episode
        """
        await self.initialize()

        self.console.print("\n[bold cyan]Creating Episode...[/bold cyan]")

        episode_context = EpisodeContext(**(context or {}))

        with self.console.status("[bold green]Storing episode..."):
            episode = await self.episodic.store(
                content=content,
                outcome=Outcome(outcome),
                emotional_valence=emotional_valence,
                context=episode_context,
            )

        self.console.print(f"[green]✓[/green] Episode created: {episode.id}")
        self.console.print(f"  Content: {content[:60]}...")
        self.console.print(f"  Outcome: {outcome}")
        self.console.print(f"  Valence: {emotional_valence}")

        return episode

    async def create_entity(
        self,
        name: str,
        entity_type: str,
        summary: str,
        details: str | None = None,
        source: str | None = None,
    ) -> Entity:
        """
        Create a new entity.

        Args:
            name: Entity name
            entity_type: CONCEPT, PERSON, PROJECT, TOOL, TECHNIQUE, FACT
            summary: Short description
            details: Expanded context
            source: Source episode ID or 'user_provided'

        Returns:
            Created entity
        """
        await self.initialize()

        self.console.print("\n[bold cyan]Creating Entity...[/bold cyan]")

        with self.console.status("[bold green]Storing entity..."):
            entity = await self.semantic.create_entity(
                name=name,
                entity_type=entity_type,
                summary=summary,
                details=details,
                source=source or "user_provided",
            )

        self.console.print(f"[green]✓[/green] Entity created: {entity.id}")
        self.console.print(f"  Name: {name}")
        self.console.print(f"  Type: {entity_type}")
        self.console.print(f"  Summary: {summary}")

        return entity

    async def create_skill(
        self,
        name: str,
        domain: str,
        steps: list[dict[str, Any]],
        trigger_pattern: str | None = None,
        script: str | None = None,
    ) -> Procedure:
        """
        Create a new procedural skill.

        Args:
            name: Skill name
            domain: coding, research, trading, devops, writing
            steps: List of step dicts with order, action, tool, parameters
            trigger_pattern: When to invoke
            script: High-level abstraction

        Returns:
            Created skill
        """
        await self.initialize()

        self.console.print("\n[bold cyan]Creating Skill...[/bold cyan]")

        # Convert steps
        procedure_steps = [
            ProcedureStep(
                order=s.get("order", i + 1),
                action=s["action"],
                tool=s.get("tool"),
                parameters=s.get("parameters", {}),
                expected_outcome=s.get("expected_outcome"),
            )
            for i, s in enumerate(steps)
        ]

        # Build procedure
        from t4dm.embedding.bge_m3 import get_embedding_provider
        embedding_provider = get_embedding_provider()

        # Generate embedding
        embed_text = script if script else f"{name}: {' | '.join(s.action for s in procedure_steps)}"
        embedding = await embedding_provider.embed_query(embed_text)

        procedure = Procedure(
            name=name,
            domain=Domain(domain),
            trigger_pattern=trigger_pattern,
            steps=procedure_steps,
            script=script,
            embedding=embedding,
            created_from="manual",
        )

        # Store via procedural memory (using internal saga logic)
        with self.console.status("[bold green]Storing skill..."):
            # We need to store manually since create_skill expects trajectory
            # Use the vector store directly
            await self.procedural.vector_store.add(
                collection=self.procedural.vector_store.procedures_collection,
                ids=[str(procedure.id)],
                vectors=[embedding],
                payloads=[self.procedural._to_payload(procedure)],
            )

            await self.procedural.graph_store.create_node(
                label="Procedure",
                properties=self.procedural._to_graph_props(procedure),
            )

        self.console.print(f"[green]✓[/green] Skill created: {procedure.id}")
        self.console.print(f"  Name: {name}")
        self.console.print(f"  Domain: {domain}")
        self.console.print(f"  Steps: {len(procedure_steps)}")

        return procedure

    # ==================== READ ====================

    async def get_episode(self, episode_id: str) -> Episode | None:
        """Get episode by ID."""
        await self.initialize()

        results = await self.episodic.vector_store.get(
            collection=self.episodic.vector_store.episodes_collection,
            ids=[episode_id],
        )

        if not results:
            return None

        id_str, payload = results[0]
        return Episode(
            id=UUID(id_str),
            session_id=payload["session_id"],
            content=payload["content"],
            timestamp=datetime.fromisoformat(payload["timestamp"]),
            outcome=Outcome(payload["outcome"]),
            emotional_valence=payload["emotional_valence"],
            context=payload.get("context", {}),
            access_count=payload["access_count"],
            last_accessed=datetime.fromisoformat(payload["last_accessed"]),
            stability=payload["stability"],
        )

    async def get_entity(self, entity_id: str) -> Entity | None:
        """Get entity by ID."""
        await self.initialize()
        return await self.semantic.get_entity(UUID(entity_id))

    async def get_skill(self, skill_id: str) -> Procedure | None:
        """Get skill by ID."""
        await self.initialize()
        return await self.procedural.get_procedure(UUID(skill_id))

    # ==================== UPDATE ====================

    async def update_entity(
        self,
        entity_id: str,
        new_summary: str,
        new_details: str | None = None,
    ) -> Entity:
        """
        Update entity (creates new version).

        Args:
            entity_id: Entity UUID
            new_summary: Updated summary
            new_details: Updated details

        Returns:
            New entity version
        """
        await self.initialize()

        self.console.print(f"\n[bold cyan]Updating Entity {entity_id}...[/bold cyan]")

        with self.console.status("[bold green]Creating new version..."):
            new_entity = await self.semantic.supersede(
                entity_id=UUID(entity_id),
                new_summary=new_summary,
                new_details=new_details,
            )

        self.console.print(f"[green]✓[/green] Entity updated: {new_entity.id}")
        self.console.print(f"  Old version invalidated: {entity_id}")
        self.console.print(f"  New version: {new_entity.id}")

        return new_entity

    async def update_skill_performance(
        self,
        skill_id: str,
        success: bool,
        error: str | None = None,
    ) -> Procedure:
        """
        Update skill execution stats.

        Args:
            skill_id: Skill UUID
            success: Whether execution succeeded
            error: Error message if failed

        Returns:
            Updated skill
        """
        await self.initialize()

        self.console.print("\n[bold cyan]Updating Skill Performance...[/bold cyan]")

        with self.console.status("[bold green]Updating stats..."):
            skill = await self.procedural.update(
                procedure_id=UUID(skill_id),
                success=success,
                error=error,
            )

        status = "[green]SUCCESS[/green]" if success else "[red]FAILURE[/red]"
        self.console.print(f"[green]✓[/green] Skill updated: {status}")
        self.console.print(f"  Success rate: {skill.success_rate:.1%}")
        self.console.print(f"  Executions: {skill.execution_count}")

        if skill.deprecated:
            self.console.print("  [yellow]⚠ Skill deprecated due to low success rate[/yellow]")

        return skill

    # ==================== DELETE ====================

    async def delete_episode(
        self,
        episode_id: str,
        confirm: bool = True,
    ) -> bool:
        """
        Delete episode.

        Args:
            episode_id: Episode UUID
            confirm: Require confirmation

        Returns:
            True if deleted
        """
        await self.initialize()

        if confirm:
            episode = await self.get_episode(episode_id)
            if not episode:
                self.console.print(f"[red]Episode {episode_id} not found[/red]")
                return False

            self.console.print("\n[bold yellow]Delete Episode?[/bold yellow]")
            self.console.print(f"  ID: {episode_id}")
            self.console.print(f"  Content: {episode.content[:60]}...")
            self.console.print(f"  Timestamp: {episode.timestamp}")

            if not Confirm.ask("Confirm deletion"):
                self.console.print("[yellow]Cancelled[/yellow]")
                return False

        with self.console.status("[bold red]Deleting episode..."):
            await self.episodic.vector_store.delete(
                collection=self.episodic.vector_store.episodes_collection,
                ids=[episode_id],
            )

        self.console.print(f"[green]✓[/green] Episode deleted: {episode_id}")
        return True

    async def delete_entity(
        self,
        entity_id: str,
        confirm: bool = True,
        cascade_relationships: bool = True,
    ) -> bool:
        """
        Delete entity.

        Args:
            entity_id: Entity UUID
            confirm: Require confirmation
            cascade_relationships: Also delete relationships

        Returns:
            True if deleted
        """
        await self.initialize()

        if confirm:
            entity = await self.get_entity(entity_id)
            if not entity:
                self.console.print(f"[red]Entity {entity_id} not found[/red]")
                return False

            # Check relationships
            relationships = await self.semantic.graph_store.get_relationships(
                node_id=entity_id,
                direction="both",
            )

            self.console.print("\n[bold yellow]Delete Entity?[/bold yellow]")
            self.console.print(f"  ID: {entity_id}")
            self.console.print(f"  Name: {entity.name}")
            self.console.print(f"  Type: {entity.entity_type.value}")
            self.console.print(f"  Relationships: {len(relationships)}")

            if not Confirm.ask("Confirm deletion"):
                self.console.print("[yellow]Cancelled[/yellow]")
                return False

        with self.console.status("[bold red]Deleting entity..."):
            # Delete from vector store
            await self.semantic.vector_store.delete(
                collection=self.semantic.vector_store.entities_collection,
                ids=[entity_id],
            )

            # Delete from graph
            await self.semantic.graph_store.delete_node(
                node_id=entity_id,
                label="Entity",
            )

        self.console.print(f"[green]✓[/green] Entity deleted: {entity_id}")
        return True

    async def delete_skill(
        self,
        skill_id: str,
        confirm: bool = True,
    ) -> bool:
        """
        Delete or deprecate skill.

        Args:
            skill_id: Skill UUID
            confirm: Require confirmation

        Returns:
            True if deleted/deprecated
        """
        await self.initialize()

        if confirm:
            skill = await self.get_skill(skill_id)
            if not skill:
                self.console.print(f"[red]Skill {skill_id} not found[/red]")
                return False

            self.console.print("\n[bold yellow]Deprecate Skill?[/bold yellow]")
            self.console.print(f"  ID: {skill_id}")
            self.console.print(f"  Name: {skill.name}")
            self.console.print(f"  Domain: {skill.domain.value}")
            self.console.print(f"  Success rate: {skill.success_rate:.1%}")

            if not Confirm.ask("Confirm deprecation"):
                self.console.print("[yellow]Cancelled[/yellow]")
                return False

        with self.console.status("[bold red]Deprecating skill..."):
            await self.procedural.deprecate(
                procedure_id=UUID(skill_id),
                reason="Manual deletion",
            )

        self.console.print(f"[green]✓[/green] Skill deprecated: {skill_id}")
        return True

    # ==================== BATCH OPERATIONS ====================

    async def batch_create_episodes(
        self,
        episodes: list[dict[str, Any]],
    ) -> list[Episode]:
        """
        Create multiple episodes.

        Args:
            episodes: List of episode dicts with content, outcome, etc.

        Returns:
            List of created episodes
        """
        await self.initialize()

        self.console.print(f"\n[bold cyan]Batch Creating {len(episodes)} Episodes...[/bold cyan]")

        created = []

        with Progress(
            SpinnerColumn(),
            TextColumn("[progress.description]{task.description}"),
            BarColumn(),
            TextColumn("[progress.percentage]{task.percentage:>3.0f}%"),
            console=self.console,
        ) as progress:
            task = progress.add_task("Creating episodes...", total=len(episodes))

            for ep_data in episodes:
                episode = await self.episodic.store(
                    content=ep_data["content"],
                    outcome=Outcome(ep_data.get("outcome", "neutral")),
                    emotional_valence=ep_data.get("emotional_valence", 0.5),
                    context=EpisodeContext(**(ep_data.get("context", {}))),
                )
                created.append(episode)
                progress.advance(task)

        self.console.print(f"[green]✓[/green] Created {len(created)} episodes")
        return created

    async def batch_create_entities(
        self,
        entities: list[dict[str, Any]],
    ) -> list[Entity]:
        """
        Create multiple entities.

        Args:
            entities: List of entity dicts with name, type, summary, etc.

        Returns:
            List of created entities
        """
        await self.initialize()

        self.console.print(f"\n[bold cyan]Batch Creating {len(entities)} Entities...[/bold cyan]")

        created = []

        with Progress(
            SpinnerColumn(),
            TextColumn("[progress.description]{task.description}"),
            BarColumn(),
            TextColumn("[progress.percentage]{task.percentage:>3.0f}%"),
            console=self.console,
        ) as progress:
            task = progress.add_task("Creating entities...", total=len(entities))

            for ent_data in entities:
                entity = await self.semantic.create_entity(
                    name=ent_data["name"],
                    entity_type=ent_data["entity_type"],
                    summary=ent_data["summary"],
                    details=ent_data.get("details"),
                    source=ent_data.get("source", "batch_import"),
                )
                created.append(entity)
                progress.advance(task)

        self.console.print(f"[green]✓[/green] Created {len(created)} entities")
        return created

    async def batch_delete_episodes(
        self,
        episode_ids: list[str],
        confirm: bool = True,
    ) -> int:
        """
        Delete multiple episodes.

        Args:
            episode_ids: List of episode UUIDs
            confirm: Require confirmation

        Returns:
            Number deleted
        """
        await self.initialize()

        if confirm:
            self.console.print(f"\n[bold yellow]Delete {len(episode_ids)} Episodes?[/bold yellow]")
            if not Confirm.ask("Confirm batch deletion"):
                self.console.print("[yellow]Cancelled[/yellow]")
                return 0

        with Progress(
            SpinnerColumn(),
            TextColumn("[progress.description]{task.description}"),
            console=self.console,
        ) as progress:
            task = progress.add_task("Deleting episodes...", total=None)

            await self.episodic.vector_store.delete(
                collection=self.episodic.vector_store.episodes_collection,
                ids=episode_ids,
            )

            progress.update(task, completed=True)

        self.console.print(f"[green]✓[/green] Deleted {len(episode_ids)} episodes")
        return len(episode_ids)


async def main():
    """CLI demo for CRUD manager."""
    manager = CRUDManager(session_id="demo")

    # Create demo episode
    episode = await manager.create_episode(
        content="Testing CRUD manager with sample episode",
        outcome="success",
        emotional_valence=0.8,
    )

    # Create demo entity
    entity = await manager.create_entity(
        name="CRUD Manager",
        entity_type="TOOL",
        summary="Interface for memory CRUD operations",
        details="Provides batch operations and safe deletion",
    )

    # Create demo skill
    skill = await manager.create_skill(
        name="Create Memory",
        domain="coding",
        steps=[
            {"order": 1, "action": "Initialize memory service", "tool": "t4dm"},
            {"order": 2, "action": "Store memory with embedding", "tool": "qdrant"},
        ],
        trigger_pattern="When user wants to create a memory",
    )

    logger.info(f"Created: Episode: {episode.id}, Entity: {entity.id}, Skill: {skill.id}")


if __name__ == "__main__":
    asyncio.run(main())
