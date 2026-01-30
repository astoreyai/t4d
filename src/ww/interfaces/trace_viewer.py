"""
Memory Trace Viewer - Visualize memory access patterns and consolidation events.

Provides timeline visualization of memory accesses, activation decay curves,
and consolidation history.
"""

import asyncio
from collections import defaultdict
from datetime import datetime, timedelta
from typing import Optional

try:
    from rich import box
    from rich.console import Console
    from rich.layout import Layout
    from rich.panel import Panel
    from rich.progress import Progress, SpinnerColumn, TextColumn
    from rich.table import Table
    from rich.text import Text
    RICH_AVAILABLE = True
except ImportError:
    RICH_AVAILABLE = False

from ww.memory.episodic import EpisodicMemory
from ww.memory.semantic import SemanticMemory
from ww.storage.qdrant_store import get_qdrant_store


class TraceViewer:
    """
    Memory trace visualization tool.

    Features:
    - Timeline of memory accesses
    - Activation decay curves
    - Consolidation event history
    - Access pattern analysis
    - Retrieval heat maps
    """

    def __init__(
        self,
        session_id: str | None = None,
        console: Optional["Console"] = None,
    ):
        """
        Initialize trace viewer.

        Args:
            session_id: Session to analyze
            console: Rich console instance
        """
        if not RICH_AVAILABLE:
            raise ImportError(
                "rich library required for TraceViewer. "
                "Install with: pip install rich"
            )

        self.session_id = session_id or "default"
        self.console = console or Console()

        self.episodic = EpisodicMemory(session_id=session_id)
        self.semantic = SemanticMemory(session_id=session_id)
        self.vector_store = get_qdrant_store(session_id)

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
            task = progress.add_task("Initializing trace viewer...", total=None)
            await self.episodic.initialize()
            await self.semantic.initialize()
            progress.update(task, completed=True)

        self._initialized = True

    async def show_access_timeline(
        self,
        hours: int = 24,
        memory_type: str = "all",
    ) -> None:
        """
        Display timeline of memory accesses.

        Args:
            hours: Time window in hours
            memory_type: Filter by type (all, episode, entity, skill)
        """
        await self.initialize()

        start_time = datetime.now() - timedelta(hours=hours)
        self.console.print("\n[bold cyan]Memory Access Timeline[/bold cyan]")
        self.console.print(f"[white]Past {hours} hours from {start_time.strftime('%Y-%m-%d %H:%M')}[/white]\n")

        # Collect access events
        access_events = []

        if memory_type in ("all", "episode"):
            episodes, _ = await self.vector_store.scroll(
                collection=self.vector_store.episodes_collection,
                limit=1000,
                with_payload=True,
            )
            for id_str, payload, _ in episodes:
                last_accessed = datetime.fromisoformat(payload["last_accessed"])
                if last_accessed >= start_time:
                    access_events.append({
                        "time": last_accessed,
                        "type": "episode",
                        "id": id_str[:8],
                        "count": payload.get("access_count", 1),
                        "content": payload.get("content", "")[:40],
                    })

        if memory_type in ("all", "entity"):
            entities, _ = await self.vector_store.scroll(
                collection=self.vector_store.entities_collection,
                limit=1000,
                with_payload=True,
            )
            for id_str, payload, _ in entities:
                last_accessed = datetime.fromisoformat(payload["last_accessed"])
                if last_accessed >= start_time:
                    access_events.append({
                        "time": last_accessed,
                        "type": "entity",
                        "id": id_str[:8],
                        "count": payload.get("access_count", 1),
                        "content": payload.get("name", ""),
                    })

        # Sort by time
        access_events.sort(key=lambda x: x["time"])

        # Create timeline table
        table = Table(title=f"Access Timeline ({len(access_events)} accesses)", box=box.SIMPLE)
        table.add_column("Time", style="cyan")
        table.add_column("Type", style="magenta")
        table.add_column("ID", style="yellow")
        table.add_column("Total Accesses", justify="right", style="green")
        table.add_column("Content", style="white")

        for event in access_events[-50:]:  # Last 50 accesses
            time_str = event["time"].strftime("%H:%M:%S")
            type_colored = {
                "episode": "[blue]episode[/blue]",
                "entity": "[green]entity[/green]",
                "skill": "[yellow]skill[/yellow]",
            }.get(event["type"], event["type"])

            table.add_row(
                time_str,
                type_colored,
                event["id"],
                str(event["count"]),
                event["content"],
            )

        self.console.print(table)

        # Access frequency summary
        type_counts = defaultdict(int)
        for event in access_events:
            type_counts[event["type"]] += 1

        self.console.print("\n[bold magenta]Summary:[/bold magenta]")
        for mem_type, count in type_counts.items():
            self.console.print(f"  {mem_type}: {count} accesses")

    async def show_decay_curves(
        self,
        sample_size: int = 20,
    ) -> None:
        """
        Display FSRS decay curves for sample memories.

        Args:
            sample_size: Number of memories to sample
        """
        await self.initialize()

        self.console.print("\n[bold cyan]Memory Decay Curves (FSRS)[/bold cyan]\n")

        # Sample episodes
        episodes, _ = await self.vector_store.scroll(
            collection=self.vector_store.episodes_collection,
            limit=sample_size,
            with_payload=True,
        )

        current_time = datetime.now()

        table = Table(title="Episode Retrievability", box=box.ROUNDED)
        table.add_column("ID", style="cyan", width=8)
        table.add_column("Age (days)", justify="right", style="yellow")
        table.add_column("Stability", justify="right", style="blue")
        table.add_column("Access Count", justify="right", style="green")
        table.add_column("Retrievability", justify="right", style="magenta")
        table.add_column("Decay Curve", style="white")

        for id_str, payload, _ in episodes:
            timestamp = datetime.fromisoformat(payload["timestamp"])
            last_accessed = datetime.fromisoformat(payload["last_accessed"])
            stability = payload.get("stability", 1.0)
            access_count = payload.get("access_count", 1)

            # Calculate retrievability
            elapsed_days = (current_time - last_accessed).total_seconds() / 86400
            retrievability = (1 + 0.9 * elapsed_days / stability) ** (-0.5)

            # ASCII curve (simplified)
            curve = self._render_decay_curve(retrievability)

            age_days = (current_time - timestamp).total_seconds() / 86400

            table.add_row(
                id_str[:8],
                f"{age_days:.1f}",
                f"{stability:.1f}",
                str(access_count),
                f"{retrievability:.2%}",
                curve,
            )

        self.console.print(table)

        # Explanation
        self.console.print("\n[bold]FSRS Decay Formula:[/bold]")
        self.console.print("R(t, S) = (1 + 0.9 * t/S)^(-0.5)")
        self.console.print("  t = elapsed days since last access")
        self.console.print("  S = stability (learned from access pattern)")
        self.console.print("  Higher stability → slower decay")

    def _render_decay_curve(self, retrievability: float, width: int = 20) -> str:
        """Render ASCII bar for retrievability."""
        filled = int(retrievability * width)
        bar = "█" * filled + "░" * (width - filled)

        # Color based on retrievability
        if retrievability > 0.8:
            return f"[green]{bar}[/green]"
        if retrievability > 0.5:
            return f"[yellow]{bar}[/yellow]"
        return f"[red]{bar}[/red]"

    async def show_consolidation_events(
        self,
        limit: int = 20,
    ) -> None:
        """
        Display consolidation event history.

        Args:
            limit: Maximum events to show
        """
        await self.initialize()

        self.console.print("\n[bold cyan]Consolidation Events[/bold cyan]\n")

        # Query Neo4j for consolidation relationships
        # (In production, you'd track these in a dedicated collection)

        # For now, show entities created from episodes
        entities, _ = await self.vector_store.scroll(
            collection=self.vector_store.entities_collection,
            limit=limit,
            with_payload=True,
        )

        consolidated = []
        for id_str, payload, _ in entities:
            source = payload.get("source")
            if source and source != "user_provided":
                consolidated.append({
                    "entity_id": id_str,
                    "entity_name": payload.get("name", ""),
                    "source_episode": source,
                    "created_at": datetime.fromisoformat(payload["created_at"]),
                })

        if not consolidated:
            self.console.print("[yellow]No consolidation events found[/yellow]")
            return

        table = Table(title=f"Episodic → Semantic Consolidations ({len(consolidated)})")
        table.add_column("Timestamp", style="cyan")
        table.add_column("Entity", style="magenta")
        table.add_column("Entity ID", style="yellow", width=8)
        table.add_column("Source Episode", style="green", width=8)

        for event in sorted(consolidated, key=lambda x: x["created_at"], reverse=True):
            table.add_row(
                event["created_at"].strftime("%Y-%m-%d %H:%M"),
                event["entity_name"],
                event["entity_id"][:8],
                event["source_episode"][:8] if event["source_episode"] else "N/A",
            )

        self.console.print(table)

    async def show_activation_history(
        self,
        entity_id: str,
        limit: int = 50,
    ) -> None:
        """
        Show activation history for an entity.

        Args:
            entity_id: Entity UUID
            limit: Maximum history entries
        """
        await self.initialize()

        # Load entity
        entity = await self.semantic.get_entity(entity_id)
        if not entity:
            self.console.print(f"[red]Entity {entity_id} not found[/red]")
            return

        self.console.print(f"\n[bold cyan]Activation History: {entity.name}[/bold cyan]\n")

        # Get relationships to see co-activation patterns
        relationships = await self.semantic.graph_store.get_relationships(
            node_id=str(entity.id),
            direction="both",
        )

        if not relationships:
            self.console.print("[yellow]No relationship history found[/yellow]")
            return

        # Create activation table
        table = Table(title="Co-Activation Events", box=box.ROUNDED)
        table.add_column("Related Entity", style="cyan")
        table.add_column("Relation Type", style="magenta")
        table.add_column("Weight", justify="right", style="yellow")
        table.add_column("Co-Access Count", justify="right", style="green")
        table.add_column("Last Co-Access", style="blue")

        for rel in relationships[:limit]:
            other_id = rel["other_id"]
            rel_type = rel.get("type", "RELATED")
            props = rel.get("properties", {})
            weight = props.get("weight", 0.0)
            co_access = props.get("coAccessCount", 0)
            last_access = props.get("lastCoAccess", "")

            # Load other entity name
            try:
                other_entity = await self.semantic.get_entity(other_id)
                other_name = other_entity.name if other_entity else other_id[:8]
            except Exception:
                other_name = other_id[:8]

            # Parse timestamp
            last_str = ""
            if last_access:
                try:
                    last_dt = datetime.fromisoformat(last_access)
                    last_str = last_dt.strftime("%Y-%m-%d %H:%M")
                except Exception:
                    last_str = last_access

            table.add_row(
                other_name,
                rel_type,
                f"{weight:.3f}",
                str(co_access),
                last_str,
            )

        self.console.print(table)

        # Hebbian weight distribution
        weights = [rel.get("properties", {}).get("weight", 0.0) for rel in relationships]
        if weights:
            avg_weight = sum(weights) / len(weights)
            max_weight = max(weights)
            min_weight = min(weights)

            self.console.print("\n[bold magenta]Hebbian Weight Statistics:[/bold magenta]")
            self.console.print(f"  Average: {avg_weight:.3f}")
            self.console.print(f"  Max: {max_weight:.3f}")
            self.console.print(f"  Min: {min_weight:.3f}")

    async def show_access_heatmap(
        self,
        hours: int = 24,
        bucket_minutes: int = 60,
    ) -> None:
        """
        Display access frequency heatmap.

        Args:
            hours: Time window
            bucket_minutes: Time bucket size in minutes
        """
        await self.initialize()

        start_time = datetime.now() - timedelta(hours=hours)
        bucket_size = timedelta(minutes=bucket_minutes)
        num_buckets = int(hours * 60 / bucket_minutes)

        self.console.print("\n[bold cyan]Access Heatmap[/bold cyan]")
        self.console.print(f"[white]{hours} hours, {bucket_minutes}min buckets[/white]\n")

        # Initialize buckets
        buckets = [0] * num_buckets

        # Collect accesses
        episodes, _ = await self.vector_store.scroll(
            collection=self.vector_store.episodes_collection,
            limit=10000,
            with_payload=True,
        )

        for id_str, payload, _ in episodes:
            last_accessed = datetime.fromisoformat(payload["last_accessed"])
            if last_accessed >= start_time:
                elapsed = last_accessed - start_time
                bucket_idx = int(elapsed.total_seconds() / bucket_size.total_seconds())
                if 0 <= bucket_idx < num_buckets:
                    buckets[bucket_idx] += 1

        # Render heatmap
        max_count = max(buckets) if buckets else 1

        table = Table(box=None, show_header=False, padding=(0, 1))
        table.add_column("Time", style="cyan", width=12)
        table.add_column("Bar", width=60)
        table.add_column("Count", justify="right", style="yellow", width=6)

        for i, count in enumerate(buckets):
            time_label = start_time + i * bucket_size
            time_str = time_label.strftime("%H:%M")

            # Render bar
            bar_width = int((count / max_count) * 50) if max_count > 0 else 0
            bar = "█" * bar_width

            # Color by intensity
            if count == 0:
                bar_colored = f"[dim]{bar}[/dim]"
            elif count < max_count * 0.3:
                bar_colored = f"[blue]{bar}[/blue]"
            elif count < max_count * 0.7:
                bar_colored = f"[yellow]{bar}[/yellow]"
            else:
                bar_colored = f"[red]{bar}[/red]"

            table.add_row(time_str, bar_colored, str(count))

        self.console.print(table)


async def main():
    """CLI entry point for trace viewer."""
    import sys

    session_id = sys.argv[1] if len(sys.argv) > 1 else None
    viewer = TraceViewer(session_id=session_id)

    # Demo: show all views
    await viewer.show_access_timeline(hours=24)
    await viewer.show_decay_curves(sample_size=15)
    await viewer.show_consolidation_events(limit=20)
    await viewer.show_access_heatmap(hours=24, bucket_minutes=60)


if __name__ == "__main__":
    asyncio.run(main())
