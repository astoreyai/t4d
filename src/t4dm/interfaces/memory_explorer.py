"""
Memory Explorer - Rich terminal UI for browsing World Weaver memories.

Provides interactive browsing of episodes, entities, and skills with filtering,
detail views, and relationship exploration.
"""

import asyncio
from datetime import datetime
from typing import Optional
from uuid import UUID

try:
    from rich.console import Console
    from rich.layout import Layout
    from rich.panel import Panel
    from rich.progress import Progress, SpinnerColumn, TextColumn
    from rich.prompt import Confirm, Prompt
    from rich.table import Table
    from rich.text import Text
    from rich.tree import Tree
    RICH_AVAILABLE = True
except ImportError:
    RICH_AVAILABLE = False

from t4dm.core.types import Episode
from t4dm.memory.episodic import EpisodicMemory
from t4dm.memory.procedural import ProceduralMemory
from t4dm.memory.semantic import SemanticMemory


class MemoryExplorer:
    """
    Interactive memory browser with Rich terminal UI.

    Features:
    - List episodes with filtering by session, outcome, time range
    - View episode details with embeddings visualization
    - Show entity relationships in tree view
    - Display skill execution history
    - Search and filter across all memory types
    """

    def __init__(
        self,
        session_id: str | None = None,
        console: Optional["Console"] = None,
    ):
        """
        Initialize memory explorer.

        Args:
            session_id: Session to explore (None = all sessions)
            console: Rich console instance (creates new if None)
        """
        if not RICH_AVAILABLE:
            raise ImportError(
                "rich library required for MemoryExplorer. "
                "Install with: pip install rich"
            )

        self.session_id = session_id
        self.console = console or Console()

        # Initialize memory services
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
            task = progress.add_task("Initializing memory services...", total=None)
            await self.episodic.initialize()
            await self.semantic.initialize()
            await self.procedural.initialize()
            progress.update(task, completed=True)

        self._initialized = True

    async def list_episodes(
        self,
        limit: int = 20,
        session_filter: str | None = None,
        outcome_filter: str | None = None,
        start_time: datetime | None = None,
        end_time: datetime | None = None,
    ) -> None:
        """
        Display episodes in a rich table.

        Args:
            limit: Maximum episodes to show
            session_filter: Filter by session ID
            outcome_filter: Filter by outcome (success, failure, etc.)
            start_time: Show episodes after this time
            end_time: Show episodes before this time
        """
        await self.initialize()

        # Build filter
        filter_dict = {}
        if session_filter:
            filter_dict["session_id"] = session_filter
        if outcome_filter:
            filter_dict["outcome"] = outcome_filter

        # Scroll through episodes
        with self.console.status("[bold green]Loading episodes..."):
            episodes, _ = await self.episodic.vector_store.scroll(
                collection=self.episodic.vector_store.episodes_collection,
                scroll_filter=filter_dict if filter_dict else None,
                limit=limit,
                with_payload=True,
                with_vectors=False,
            )

        # Create table
        table = Table(title=f"Episodes (showing {len(episodes)} of {limit})")
        table.add_column("ID", style="cyan", no_wrap=True, width=8)
        table.add_column("Timestamp", style="magenta")
        table.add_column("Outcome", style="green")
        table.add_column("Valence", justify="right", style="yellow")
        table.add_column("Access", justify="right", style="blue")
        table.add_column("Content Preview", style="white")

        for id_str, payload, _ in episodes:
            # Filter by time if specified
            timestamp = datetime.fromisoformat(payload["timestamp"])
            if start_time and timestamp < start_time:
                continue
            if end_time and timestamp > end_time:
                continue

            # Format row
            short_id = id_str[:8]
            timestamp_str = timestamp.strftime("%Y-%m-%d %H:%M")
            outcome = payload.get("outcome", "neutral")
            valence = f"{payload.get('emotional_valence', 0.5):.2f}"
            access_count = str(payload.get("access_count", 1))

            content = payload.get("content", "")
            preview = content[:60] + "..." if len(content) > 60 else content

            # Color outcome
            outcome_style = {
                "success": "[green]success[/green]",
                "failure": "[red]failure[/red]",
                "partial": "[yellow]partial[/yellow]",
                "neutral": "[white]neutral[/white]",
            }.get(outcome, outcome)

            table.add_row(
                short_id,
                timestamp_str,
                outcome_style,
                valence,
                access_count,
                preview,
            )

        self.console.print(table)

    async def view_episode(self, episode_id: str) -> None:
        """
        Display detailed episode view.

        Args:
            episode_id: Episode UUID (full or prefix)
        """
        await self.initialize()

        # Load episode
        with self.console.status(f"[bold green]Loading episode {episode_id}..."):
            results = await self.episodic.vector_store.get(
                collection=self.episodic.vector_store.episodes_collection,
                ids=[episode_id],
            )

        if not results:
            self.console.print(f"[red]Episode {episode_id} not found[/red]")
            return

        id_str, payload = results[0]

        # Create episode object
        episode = Episode(
            id=UUID(id_str),
            session_id=payload["session_id"],
            content=payload["content"],
            timestamp=datetime.fromisoformat(payload["timestamp"]),
            outcome=payload.get("outcome", "neutral"),
            emotional_valence=payload.get("emotional_valence", 0.5),
            context=payload.get("context", {}),
            access_count=payload.get("access_count", 1),
            last_accessed=datetime.fromisoformat(payload["last_accessed"]),
            stability=payload.get("stability", 1.0),
        )

        # Build detail panel
        layout = Layout()
        layout.split_column(
            Layout(name="header", size=3),
            Layout(name="metadata", size=8),
            Layout(name="content"),
        )

        # Header
        header_text = Text(f"Episode: {str(episode.id)[:8]}...", style="bold cyan")
        layout["header"].update(Panel(header_text, border_style="cyan"))

        # Metadata table
        meta_table = Table(show_header=False, box=None)
        meta_table.add_column("Key", style="bold magenta")
        meta_table.add_column("Value", style="white")

        meta_table.add_row("Session ID", episode.session_id)
        meta_table.add_row("Timestamp", episode.timestamp.strftime("%Y-%m-%d %H:%M:%S"))
        meta_table.add_row("Outcome", episode.outcome.value)
        meta_table.add_row("Emotional Valence", f"{episode.emotional_valence:.2f}")
        meta_table.add_row("Access Count", str(episode.access_count))
        meta_table.add_row("Last Accessed", episode.last_accessed.strftime("%Y-%m-%d %H:%M:%S"))
        meta_table.add_row("Stability", f"{episode.stability:.2f} days")
        meta_table.add_row("Retrievability", f"{episode.retrievability():.2%}")

        if episode.context:
            if episode.context.project:
                meta_table.add_row("Project", episode.context.project)
            if episode.context.file:
                meta_table.add_row("File", episode.context.file)
            if episode.context.tool:
                meta_table.add_row("Tool", episode.context.tool)

        layout["metadata"].update(Panel(meta_table, title="Metadata", border_style="green"))

        # Content
        content_text = Text(episode.content, style="white")
        layout["content"].update(Panel(content_text, title="Content", border_style="blue"))

        self.console.print(layout)

    async def list_entities(
        self,
        limit: int = 20,
        entity_type: str | None = None,
        session_filter: str | None = None,
    ) -> None:
        """
        Display entities in a rich table.

        Args:
            limit: Maximum entities to show
            entity_type: Filter by entity type (CONCEPT, PERSON, etc.)
            session_filter: Filter by session ID
        """
        await self.initialize()

        # Build filter
        filter_dict = {}
        if session_filter:
            filter_dict["session_id"] = session_filter
        if entity_type:
            filter_dict["entity_type"] = entity_type

        # Scroll through entities
        with self.console.status("[bold green]Loading entities..."):
            entities, _ = await self.semantic.vector_store.scroll(
                collection=self.semantic.vector_store.entities_collection,
                scroll_filter=filter_dict if filter_dict else None,
                limit=limit,
                with_payload=True,
                with_vectors=False,
            )

        # Create table
        table = Table(title=f"Entities (showing {len(entities)})")
        table.add_column("ID", style="cyan", no_wrap=True, width=8)
        table.add_column("Name", style="bold magenta")
        table.add_column("Type", style="green")
        table.add_column("Access", justify="right", style="blue")
        table.add_column("Stability", justify="right", style="yellow")
        table.add_column("Summary", style="white")

        for id_str, payload, _ in entities:
            short_id = id_str[:8]
            name = payload.get("name", "")
            entity_type = payload.get("entity_type", "")
            access_count = str(payload.get("access_count", 1))
            stability = f"{payload.get('stability', 1.0):.1f}"
            summary = payload.get("summary", "")
            preview = summary[:50] + "..." if len(summary) > 50 else summary

            table.add_row(
                short_id,
                name,
                entity_type,
                access_count,
                stability,
                preview,
            )

        self.console.print(table)

    async def view_entity_graph(self, entity_id: str, depth: int = 2) -> None:
        """
        Display entity relationship graph as tree.

        Args:
            entity_id: Entity UUID (full or prefix)
            depth: Relationship traversal depth
        """
        await self.initialize()

        # Load entity
        with self.console.status(f"[bold green]Loading entity {entity_id}..."):
            entity = await self.semantic.get_entity(UUID(entity_id))

        if not entity:
            self.console.print(f"[red]Entity {entity_id} not found[/red]")
            return

        # Build tree
        tree = Tree(
            f"[bold cyan]{entity.name}[/bold cyan] ({entity.entity_type.value})",
            guide_style="bright_blue",
        )

        # Add entity details
        tree.add(f"[magenta]Summary:[/magenta] {entity.summary}")
        tree.add(f"[yellow]Access Count:[/yellow] {entity.access_count}")
        tree.add(f"[green]Stability:[/green] {entity.stability:.2f} days")

        # Get relationships
        with self.console.status("[bold green]Loading relationships..."):
            relationships = await self.semantic.graph_store.get_relationships(
                node_id=str(entity.id),
                direction="both",
            )

        if relationships:
            rel_branch = tree.add("[bold blue]Relationships[/bold blue]")
            for rel in relationships[:20]:  # Limit to 20
                other_id = rel["other_id"]
                rel_type = rel.get("type", "RELATED")
                weight = rel.get("properties", {}).get("weight", 0.0)

                # Load other entity name
                try:
                    other_entity = await self.semantic.get_entity(UUID(other_id))
                    other_name = other_entity.name if other_entity else other_id[:8]
                except Exception:
                    other_name = other_id[:8]

                rel_branch.add(
                    f"[white]{rel_type}[/white] → [cyan]{other_name}[/cyan] "
                    f"(weight: {weight:.2f})"
                )

        self.console.print(Panel(tree, title="Entity Graph", border_style="green"))

    async def list_skills(
        self,
        limit: int = 20,
        domain: str | None = None,
        session_filter: str | None = None,
    ) -> None:
        """
        Display procedural skills in a rich table.

        Args:
            limit: Maximum skills to show
            domain: Filter by domain (coding, research, etc.)
            session_filter: Filter by session ID
        """
        await self.initialize()

        # Build filter
        filter_dict = {"deprecated": False}
        if session_filter:
            filter_dict["session_id"] = session_filter
        if domain:
            filter_dict["domain"] = domain

        # Scroll through procedures
        with self.console.status("[bold green]Loading skills..."):
            skills, _ = await self.procedural.vector_store.scroll(
                collection=self.procedural.vector_store.procedures_collection,
                scroll_filter=filter_dict if filter_dict else None,
                limit=limit,
                with_payload=True,
                with_vectors=False,
            )

        # Create table
        table = Table(title=f"Procedural Skills (showing {len(skills)})")
        table.add_column("ID", style="cyan", no_wrap=True, width=8)
        table.add_column("Name", style="bold magenta")
        table.add_column("Domain", style="green")
        table.add_column("Success", justify="right", style="yellow")
        table.add_column("Executions", justify="right", style="blue")
        table.add_column("Steps", justify="right", style="white")

        for id_str, payload, _ in skills:
            short_id = id_str[:8]
            name = payload.get("name", "")
            domain = payload.get("domain", "")
            success_rate = f"{payload.get('success_rate', 1.0):.1%}"
            exec_count = str(payload.get("execution_count", 1))
            steps = str(len(payload.get("steps", [])))

            table.add_row(
                short_id,
                name,
                domain,
                success_rate,
                exec_count,
                steps,
            )

        self.console.print(table)

    async def search(self, query: str, limit: int = 10) -> None:
        """
        Search across all memory types.

        Args:
            query: Search query
            limit: Maximum results per memory type
        """
        await self.initialize()

        self.console.print(f"\n[bold cyan]Searching for:[/bold cyan] {query}\n")

        # Search episodes
        with self.console.status("[bold green]Searching episodes..."):
            episode_results = await self.episodic.recall(query=query, limit=limit)

        if episode_results:
            self.console.print("[bold magenta]Episodes:[/bold magenta]")
            ep_table = Table()
            ep_table.add_column("Score", justify="right", style="yellow")
            ep_table.add_column("Timestamp", style="cyan")
            ep_table.add_column("Content Preview", style="white")

            for result in episode_results:
                episode = result.item
                preview = episode.content[:60] + "..." if len(episode.content) > 60 else episode.content
                ep_table.add_row(
                    f"{result.score:.3f}",
                    episode.timestamp.strftime("%Y-%m-%d %H:%M"),
                    preview,
                )

            self.console.print(ep_table)
            self.console.print()

        # Search entities
        with self.console.status("[bold green]Searching entities..."):
            entity_results = await self.semantic.recall(query=query, limit=limit)

        if entity_results:
            self.console.print("[bold magenta]Entities:[/bold magenta]")
            ent_table = Table()
            ent_table.add_column("Score", justify="right", style="yellow")
            ent_table.add_column("Name", style="bold cyan")
            ent_table.add_column("Type", style="green")
            ent_table.add_column("Summary", style="white")

            for result in entity_results:
                entity = result.item
                summary = entity.summary[:50] + "..." if len(entity.summary) > 50 else entity.summary
                ent_table.add_row(
                    f"{result.score:.3f}",
                    entity.name,
                    entity.entity_type.value,
                    summary,
                )

            self.console.print(ent_table)
            self.console.print()

        # Search skills
        with self.console.status("[bold green]Searching skills..."):
            skill_results = await self.procedural.recall_skill(task=query, limit=limit)

        if skill_results:
            self.console.print("[bold magenta]Skills:[/bold magenta]")
            skill_table = Table()
            skill_table.add_column("Score", justify="right", style="yellow")
            skill_table.add_column("Name", style="bold cyan")
            skill_table.add_column("Domain", style="green")
            skill_table.add_column("Success Rate", justify="right", style="blue")

            for result in skill_results:
                skill = result.item
                skill_table.add_row(
                    f"{result.score:.3f}",
                    skill.name,
                    skill.domain.value,
                    f"{skill.success_rate:.1%}",
                )

            self.console.print(skill_table)

    async def interactive(self) -> None:
        """
        Run interactive memory explorer session.

        Provides menu-driven interface for browsing memories.
        """
        await self.initialize()

        self.console.print(Panel(
            "[bold cyan]World Weaver Memory Explorer[/bold cyan]\n"
            "Interactive memory browsing interface",
            border_style="cyan",
        ))

        while True:
            self.console.print("\n[bold magenta]Options:[/bold magenta]")
            self.console.print("1. List episodes")
            self.console.print("2. View episode details")
            self.console.print("3. List entities")
            self.console.print("4. View entity graph")
            self.console.print("5. List skills")
            self.console.print("6. Search all memories")
            self.console.print("7. Exit")

            choice = Prompt.ask(
                "\n[bold cyan]Select option[/bold cyan]",
                choices=["1", "2", "3", "4", "5", "6", "7"],
                default="7",
            )

            if choice == "1":
                limit = int(Prompt.ask("Limit", default="20"))
                await self.list_episodes(limit=limit)

            elif choice == "2":
                episode_id = Prompt.ask("Episode ID")
                await self.view_episode(episode_id)

            elif choice == "3":
                limit = int(Prompt.ask("Limit", default="20"))
                await self.list_entities(limit=limit)

            elif choice == "4":
                entity_id = Prompt.ask("Entity ID")
                await self.view_entity_graph(entity_id)

            elif choice == "5":
                limit = int(Prompt.ask("Limit", default="20"))
                await self.list_skills(limit=limit)

            elif choice == "6":
                query = Prompt.ask("Search query")
                limit = int(Prompt.ask("Results per type", default="10"))
                await self.search(query, limit=limit)

            elif choice == "7":
                self.console.print("\n[bold green]Goodbye![/bold green]")
                break


async def main():
    """CLI entry point for memory explorer."""
    import sys

    session_id = sys.argv[1] if len(sys.argv) > 1 else None

    explorer = MemoryExplorer(session_id=session_id)
    await explorer.interactive()


if __name__ == "__main__":
    asyncio.run(main())
