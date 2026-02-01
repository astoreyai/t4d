"""
System Dashboard - Real-time monitoring of World Weaver health and metrics.

Displays memory counts, storage health, circuit breaker states, and cache statistics.
"""

import asyncio
from datetime import datetime, timedelta
from typing import Any, Optional

try:
    from rich import box
    from rich.console import Console
    from rich.layout import Layout
    from rich.live import Live
    from rich.panel import Panel
    from rich.table import Table
    from rich.text import Text
    RICH_AVAILABLE = True
except ImportError:
    RICH_AVAILABLE = False

from t4dm.memory.episodic import EpisodicMemory
from t4dm.memory.procedural import ProceduralMemory
from t4dm.memory.semantic import SemanticMemory
from t4dm.storage import get_graph_store
from t4dm.storage import get_vector_store


class SystemDashboard:
    """
    Real-time system health dashboard.

    Features:
    - Memory counts (episodes, entities, skills)
    - Storage health (Qdrant, Neo4j)
    - Circuit breaker states
    - Recent activity metrics
    - Performance statistics
    """

    def __init__(
        self,
        session_id: str | None = None,
        console: Optional["Console"] = None,
    ):
        """
        Initialize dashboard.

        Args:
            session_id: Session to monitor (None = all sessions)
            console: Rich console instance
        """
        if not RICH_AVAILABLE:
            raise ImportError(
                "rich library required for SystemDashboard. "
                "Install with: pip install rich"
            )

        self.session_id = session_id
        self.console = console or Console()

        self.episodic = EpisodicMemory(session_id=session_id)
        self.semantic = SemanticMemory(session_id=session_id)
        self.procedural = ProceduralMemory(session_id=session_id)

        self.vector_store = get_vector_store(session_id or "default")
        self.graph_store = get_graph_store(session_id or "default")

        self._initialized = False
        self._last_update = None

    async def initialize(self) -> None:
        """Initialize storage backends."""
        if self._initialized:
            return

        await self.episodic.initialize()
        await self.semantic.initialize()
        await self.procedural.initialize()

        self._initialized = True

    async def get_memory_counts(self) -> dict[str, int]:
        """Get counts for all memory types."""
        filter_dict = {}
        if self.session_id:
            filter_dict["session_id"] = self.session_id

        counts = {}

        # Episodes
        counts["episodes"] = await self.vector_store.count(
            collection=self.vector_store.episodes_collection,
            count_filter=filter_dict if filter_dict else None,
        )

        # Entities
        counts["entities"] = await self.vector_store.count(
            collection=self.vector_store.entities_collection,
            count_filter=filter_dict if filter_dict else None,
        )

        # Skills (non-deprecated)
        skill_filter = {**filter_dict, "deprecated": False} if filter_dict else {"deprecated": False}
        counts["skills"] = await self.vector_store.count(
            collection=self.vector_store.procedures_collection,
            count_filter=skill_filter,
        )

        return counts

    async def get_storage_health(self) -> dict[str, Any]:
        """Check storage backend health."""
        health = {}

        # Qdrant circuit breaker state
        cb = self.vector_store.circuit_breaker
        health["qdrant_circuit_breaker"] = {
            "state": cb.state.name,
            "failures": cb.failure_count,
            "successes": cb.success_count,
        }

        # Neo4j circuit breaker state
        neo_cb = self.graph_store.circuit_breaker
        health["neo4j_circuit_breaker"] = {
            "state": neo_cb.state.name,
            "failures": neo_cb.failure_count,
            "successes": neo_cb.success_count,
        }

        return health

    async def get_recent_activity(self) -> dict[str, Any]:
        """Get recent activity statistics."""
        activity = {}

        # Recent episodes (last hour)
        start_time = datetime.now() - timedelta(hours=1)

        recent_episodes, _ = await self.vector_store.scroll(
            collection=self.vector_store.episodes_collection,
            limit=1000,
            with_payload=True,
        )

        recent_count = 0
        for _, payload, _ in recent_episodes:
            timestamp = datetime.fromisoformat(payload["timestamp"])
            if timestamp >= start_time:
                recent_count += 1

        activity["episodes_last_hour"] = recent_count

        # Most accessed memories
        accessed_entities, _ = await self.vector_store.scroll(
            collection=self.vector_store.entities_collection,
            limit=100,
            with_payload=True,
        )

        # Sort by access count
        accessed_entities = sorted(
            accessed_entities,
            key=lambda x: x[1].get("access_count", 0),
            reverse=True,
        )[:5]

        activity["top_entities"] = [
            {
                "name": payload.get("name", ""),
                "access_count": payload.get("access_count", 0),
            }
            for _, payload, _ in accessed_entities
        ]

        return activity

    async def get_performance_stats(self) -> dict[str, Any]:
        """Get performance statistics."""
        stats = {}

        # Average retrievability (sample)
        episodes, _ = await self.vector_store.scroll(
            collection=self.vector_store.episodes_collection,
            limit=100,
            with_payload=True,
        )

        current_time = datetime.now()
        retrievabilities = []

        for _, payload, _ in episodes:
            last_accessed = datetime.fromisoformat(payload["last_accessed"])
            stability = payload.get("stability", 1.0)
            elapsed_days = (current_time - last_accessed).total_seconds() / 86400
            retrievability = (1 + 0.9 * elapsed_days / stability) ** (-0.5)
            retrievabilities.append(retrievability)

        if retrievabilities:
            stats["avg_retrievability"] = sum(retrievabilities) / len(retrievabilities)
            stats["min_retrievability"] = min(retrievabilities)
            stats["max_retrievability"] = max(retrievabilities)
        else:
            stats["avg_retrievability"] = 0.0
            stats["min_retrievability"] = 0.0
            stats["max_retrievability"] = 0.0

        # Skill success rates
        skills, _ = await self.vector_store.scroll(
            collection=self.vector_store.procedures_collection,
            scroll_filter={"deprecated": False},
            limit=100,
            with_payload=True,
        )

        success_rates = [
            payload.get("success_rate", 1.0)
            for _, payload, _ in skills
        ]

        if success_rates:
            stats["avg_skill_success"] = sum(success_rates) / len(success_rates)
        else:
            stats["avg_skill_success"] = 0.0

        return stats

    def _build_layout(self) -> Layout:
        """Build dashboard layout."""
        layout = Layout()

        layout.split_column(
            Layout(name="header", size=3),
            Layout(name="main", ratio=1),
            Layout(name="footer", size=3),
        )

        layout["main"].split_row(
            Layout(name="left"),
            Layout(name="right"),
        )

        return layout

    async def render_dashboard(self) -> Layout:
        """Render complete dashboard."""
        await self.initialize()

        layout = self._build_layout()

        # Header
        session_label = self.session_id or "ALL SESSIONS"
        header_text = Text(f"World Weaver System Dashboard - {session_label}", style="bold cyan")
        layout["header"].update(Panel(header_text, border_style="cyan"))

        # Get data
        counts = await self.get_memory_counts()
        health = await self.get_storage_health()
        activity = await self.get_recent_activity()
        stats = await self.get_performance_stats()

        # Left panel: Counts and health
        left_table = Table(title="System Status", box=box.ROUNDED, show_header=False)
        left_table.add_column("Metric", style="bold magenta")
        left_table.add_column("Value", style="white")

        left_table.add_section()
        left_table.add_row("Episodes", f"{counts['episodes']:,}")
        left_table.add_row("Entities", f"{counts['entities']:,}")
        left_table.add_row("Skills", f"{counts['skills']:,}")

        left_table.add_section()
        qdrant_state = health["qdrant_circuit_breaker"]["state"]
        qdrant_color = "green" if qdrant_state == "CLOSED" else "red"
        left_table.add_row("Qdrant Circuit", f"[{qdrant_color}]{qdrant_state}[/{qdrant_color}]")

        neo_state = health["neo4j_circuit_breaker"]["state"]
        neo_color = "green" if neo_state == "CLOSED" else "red"
        left_table.add_row("Neo4j Circuit", f"[{neo_color}]{neo_state}[/{neo_color}]")

        left_table.add_section()
        left_table.add_row("Avg Retrievability", f"{stats['avg_retrievability']:.2%}")
        left_table.add_row("Avg Skill Success", f"{stats['avg_skill_success']:.2%}")

        layout["left"].update(Panel(left_table, border_style="green"))

        # Right panel: Activity
        right_table = Table(title="Recent Activity", box=box.ROUNDED)
        right_table.add_column("Metric", style="bold yellow")
        right_table.add_column("Value", style="white")

        right_table.add_row("Episodes (1h)", str(activity["episodes_last_hour"]))

        right_table.add_section()
        if activity["top_entities"]:
            right_table.add_row("[bold]Top Entities:[/bold]", "")
            for ent in activity["top_entities"]:
                right_table.add_row(
                    f"  {ent['name'][:30]}",
                    f"{ent['access_count']} accesses"
                )

        layout["right"].update(Panel(right_table, border_style="blue"))

        # Footer
        now = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        footer_text = Text(f"Last updated: {now}", style="dim")
        layout["footer"].update(Panel(footer_text, border_style="dim"))

        self._last_update = datetime.now()

        return layout

    async def show(self, refresh_interval: float | None = None) -> None:
        """
        Display dashboard (static or live).

        Args:
            refresh_interval: Auto-refresh interval in seconds (None = static)
        """
        if refresh_interval:
            # Live updating dashboard
            with Live(await self.render_dashboard(), console=self.console, refresh_per_second=1) as live:
                try:
                    while True:
                        await asyncio.sleep(refresh_interval)
                        live.update(await self.render_dashboard())
                except KeyboardInterrupt:
                    pass
        else:
            # Static dashboard
            layout = await self.render_dashboard()
            self.console.print(layout)

    async def show_detailed_health(self) -> None:
        """Display detailed health report."""
        await self.initialize()

        self.console.print("\n[bold cyan]Detailed Health Report[/bold cyan]\n")

        # Memory counts
        counts = await self.get_memory_counts()
        count_table = Table(title="Memory Counts", box=box.ROUNDED)
        count_table.add_column("Type", style="bold magenta")
        count_table.add_column("Count", justify="right", style="green")

        count_table.add_row("Episodes", f"{counts['episodes']:,}")
        count_table.add_row("Entities", f"{counts['entities']:,}")
        count_table.add_row("Skills (Active)", f"{counts['skills']:,}")

        self.console.print(count_table)
        self.console.print()

        # Storage health
        health = await self.get_storage_health()
        health_table = Table(title="Storage Health", box=box.ROUNDED)
        health_table.add_column("Backend", style="bold yellow")
        health_table.add_column("Circuit State", style="cyan")
        health_table.add_column("Failures", justify="right", style="red")
        health_table.add_column("Successes", justify="right", style="green")

        qdrant = health["qdrant_circuit_breaker"]
        health_table.add_row(
            "Qdrant",
            qdrant["state"],
            str(qdrant["failures"]),
            str(qdrant["successes"]),
        )

        neo = health["neo4j_circuit_breaker"]
        health_table.add_row(
            "Neo4j",
            neo["state"],
            str(neo["failures"]),
            str(neo["successes"]),
        )

        self.console.print(health_table)
        self.console.print()

        # Performance stats
        stats = await self.get_performance_stats()
        perf_table = Table(title="Performance Statistics", box=box.ROUNDED)
        perf_table.add_column("Metric", style="bold blue")
        perf_table.add_column("Value", justify="right", style="white")

        perf_table.add_row("Avg Retrievability", f"{stats['avg_retrievability']:.2%}")
        perf_table.add_row("Min Retrievability", f"{stats['min_retrievability']:.2%}")
        perf_table.add_row("Max Retrievability", f"{stats['max_retrievability']:.2%}")
        perf_table.add_row("Avg Skill Success", f"{stats['avg_skill_success']:.2%}")

        self.console.print(perf_table)


async def main():
    """CLI entry point for dashboard."""
    import sys

    session_id = sys.argv[1] if len(sys.argv) > 1 else None
    mode = sys.argv[2] if len(sys.argv) > 2 else "static"

    dashboard = SystemDashboard(session_id=session_id)

    if mode == "live":
        # Live updating dashboard (refresh every 5 seconds)
        await dashboard.show(refresh_interval=5.0)
    elif mode == "detailed":
        # Detailed health report
        await dashboard.show_detailed_health()
    else:
        # Static dashboard
        await dashboard.show()


if __name__ == "__main__":
    asyncio.run(main())
