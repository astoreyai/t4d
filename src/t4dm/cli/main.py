"""
World Weaver CLI - Main Entry Point.

Provides the `ww` command for interacting with World Weaver memory.
"""

import asyncio
import json
import logging
from datetime import datetime
from pathlib import Path

import typer
from rich.console import Console
from rich.table import Table

# Initialize CLI app
app = typer.Typer(
    name="t4dm",
    help="World Weaver - Biologically-inspired memory for AI",
    no_args_is_help=True,
)

console = Console()

# Sub-apps for namespacing
episodic_app = typer.Typer(help="Episodic memory operations")
semantic_app = typer.Typer(help="Semantic memory operations")
procedural_app = typer.Typer(help="Procedural memory operations")

app.add_typer(episodic_app, name="episodic")
app.add_typer(semantic_app, name="semantic")
app.add_typer(procedural_app, name="procedural")


def get_session_id() -> str:
    """Get session ID from environment or generate one."""
    import os
    return os.environ.get("T4DM_SESSION_ID", "cli-session")


def run_async(coro):
    """Run async function from sync context."""
    return asyncio.get_event_loop().run_until_complete(coro)


# =============================================================================
# Core Commands
# =============================================================================


@app.command()
def store(
    content: str = typer.Argument(..., help="Content to store"),
    tags: str | None = typer.Option(None, "--tags", "-t", help="Comma-separated tags"),
    importance: float = typer.Option(0.5, "--importance", "-i", min=0.0, max=1.0, help="Importance (0-1)"),
    memory_type: str = typer.Option("episodic", "--type", "-T", help="Memory type: episodic, semantic, procedural"),
    metadata: str | None = typer.Option(None, "--metadata", "-m", help="JSON metadata"),
):
    """Store content in memory."""
    from t4dm.core.services import get_services

    tag_list = [t.strip() for t in tags.split(",")] if tags else []
    meta = json.loads(metadata) if metadata else {}

    session_id = get_session_id()

    async def _store():
        episodic, semantic, procedural = await get_services(session_id)

        if memory_type == "episodic":
            from t4dm.core.types import Episode
            episode = Episode(
                session_id=session_id,
                content=content,
                timestamp=datetime.utcnow(),
                emotional_valence=importance,
            )
            result = await episodic.add_episode(episode)
            console.print(f"[green]Stored episode:[/green] {result.id}")

        elif memory_type == "semantic":
            from t4dm.core.types import Entity, EntityType
            entity = Entity(
                name=content[:50],
                entity_type=EntityType.CONCEPT,
                summary=content[:200],
                details=content if len(content) > 200 else None,
            )
            result = await semantic.add_entity(entity)
            console.print(f"[green]Stored entity:[/green] {result.id}")

        elif memory_type == "procedural":
            from t4dm.core.types import Domain, Procedure
            procedure = Procedure(
                name=tag_list[0] if tag_list else "unnamed",
                domain=Domain.CODING,  # Default domain
                script=content,
            )
            result = await procedural.add_skill(procedure)
            console.print(f"[green]Stored skill:[/green] {result.id}")

        else:
            console.print(f"[red]Unknown memory type:[/red] {memory_type}")
            raise typer.Exit(1)

    try:
        run_async(_store())
    except Exception as e:
        console.print(f"[red]Error:[/red] {e}")
        raise typer.Exit(1)


@app.command()
def recall(
    query: str = typer.Argument(..., help="Query to search for"),
    k: int = typer.Option(5, "--k", "-k", help="Number of results"),
    memory_type: str = typer.Option("all", "--type", "-T", help="Memory type: episodic, semantic, procedural, all"),
    format_output: str = typer.Option("table", "--format", "-f", help="Output format: table, json"),
):
    """Recall memories matching a query."""
    from t4dm.core.services import get_services

    session_id = get_session_id()

    async def _recall():
        episodic, semantic, procedural = await get_services(session_id)
        results = []

        if memory_type in ("episodic", "all"):
            episodes = await episodic.recall_similar(query, limit=k)
            for ep in episodes:
                results.append({
                    "type": "episodic",
                    "id": str(ep.item.id),
                    "content": ep.item.content[:100],
                    "score": f"{ep.score:.3f}",
                    "timestamp": str(ep.item.timestamp),
                })

        if memory_type in ("semantic", "all"):
            entities = await semantic.search_similar(query, limit=k)
            for ent in entities:
                results.append({
                    "type": "semantic",
                    "id": str(ent.id) if hasattr(ent, 'id') else "N/A",
                    "content": ent.name if hasattr(ent, 'name') else str(ent)[:100],
                    "score": "N/A",
                    "timestamp": "N/A",
                })

        if memory_type in ("procedural", "all"):
            skills = await procedural.find_relevant_skills(query, limit=k)
            for skill in skills:
                results.append({
                    "type": "procedural",
                    "id": str(skill.id) if hasattr(skill, 'id') else "N/A",
                    "content": skill.name if hasattr(skill, 'name') else str(skill)[:100],
                    "score": "N/A",
                    "timestamp": "N/A",
                })

        return results

    try:
        results = run_async(_recall())

        if format_output == "json":
            console.print_json(data=results)
        else:
            table = Table(title=f"Recall Results for: {query}")
            table.add_column("Type", style="cyan")
            table.add_column("ID", style="dim")
            table.add_column("Content")
            table.add_column("Score", justify="right")

            for r in results:
                table.add_row(r["type"], r["id"][:8], r["content"], r["score"])

            console.print(table)
            console.print(f"\n[dim]Found {len(results)} results[/dim]")

    except Exception as e:
        console.print(f"[red]Error:[/red] {e}")
        raise typer.Exit(1)


@app.command()
def consolidate(
    full: bool = typer.Option(False, "--full", "-f", help="Run full consolidation (slow)"),
    dry_run: bool = typer.Option(False, "--dry-run", "-n", help="Show what would be consolidated"),
):
    """Run memory consolidation."""
    console.print("[yellow]Running consolidation...[/yellow]")

    async def _consolidate():
        from t4dm.consolidation.service import get_consolidation_service

        service = get_consolidation_service()

        if dry_run:
            # Just show stats
            stats = service.get_scheduler_stats()
            console.print(f"Pending consolidations: {stats.get('pending_count', 0)}")
            console.print(f"Last consolidation: {stats.get('last_consolidation', 'Never')}")
            return

        mode = "deep" if full else "light"
        result = await service.consolidate(mode=mode)

        console.print("[green]Consolidation complete:[/green]")
        console.print(f"  Episodes processed: {result.get('episodes_processed', 0)}")
        console.print(f"  Entities created: {result.get('entities_created', 0)}")
        console.print(f"  Skills consolidated: {result.get('procedures_consolidated', 0)}")

    try:
        run_async(_consolidate())
    except Exception as e:
        console.print(f"[red]Error:[/red] {e}")
        raise typer.Exit(1)


@app.command()
def status():
    """Show memory system status."""
    from t4dm.core.config import get_settings

    settings = get_settings()

    table = Table(title="World Weaver Status")
    table.add_column("Property", style="cyan")
    table.add_column("Value")

    table.add_row("Session ID", get_session_id())
    table.add_row("Environment", settings.environment)
    table.add_row("Qdrant Host", settings.qdrant_host)
    table.add_row("Neo4j URI", settings.neo4j_uri or "Not configured")
    table.add_row("Embedding Model", settings.embedding_model)

    console.print(table)

    # Check connectivity
    async def _check():
        from t4dm.core.services import get_services
        try:
            await get_services(get_session_id())
            console.print("[green]Memory services: Connected[/green]")
        except Exception as e:
            console.print(f"[red]Memory services: Error - {e}[/red]")

    run_async(_check())


@app.command()
def serve(
    host: str = typer.Option("0.0.0.0", "--host", "-h", help="Host to bind to"),
    port: int = typer.Option(8765, "--port", "-p", help="Port to bind to"),
    workers: int = typer.Option(1, "--workers", "-w", help="Number of workers"),
    reload: bool = typer.Option(False, "--reload", "-r", help="Enable auto-reload"),
):
    """Start the REST API server."""
    import uvicorn

    console.print(f"[green]Starting World Weaver API on {host}:{port}[/green]")

    uvicorn.run(
        "t4dm.api.server:app",
        host=host,
        port=port,
        workers=workers,
        reload=reload,
    )


@app.command()
def version():
    """Show version information."""
    try:
        from t4dm import __version__
    except ImportError:
        __version__ = "0.5.0-dev"

    console.print(f"World Weaver v{__version__}")


# =============================================================================
# Episodic Sub-commands
# =============================================================================


@episodic_app.command("add")
def episodic_add(
    content: str = typer.Argument(..., help="Episode content"),
    valence: float = typer.Option(0.5, "--valence", "-v", help="Emotional valence (0-1)"),
    tags: str | None = typer.Option(None, "--tags", "-t", help="Comma-separated tags"),
):
    """Add an episodic memory."""
    store(content, tags=tags, importance=valence, memory_type="episodic")


@episodic_app.command("search")
def episodic_search(
    query: str = typer.Argument(..., help="Search query"),
    k: int = typer.Option(5, "--k", "-k", help="Number of results"),
):
    """Search episodic memories."""
    recall(query, k=k, memory_type="episodic")


@episodic_app.command("recent")
def episodic_recent(
    limit: int = typer.Option(10, "--limit", "-l", help="Number of recent episodes"),
):
    """Show recent episodic memories."""
    from t4dm.core.services import get_services

    async def _recent():
        episodic, _, _ = await get_services(get_session_id())
        episodes = await episodic.get_recent_episodes(limit=limit)

        table = Table(title="Recent Episodes")
        table.add_column("ID", style="dim")
        table.add_column("Content")
        table.add_column("Timestamp")
        table.add_column("Valence", justify="right")

        for ep in episodes:
            table.add_row(
                str(ep.id)[:8],
                ep.content[:60] + "..." if len(ep.content) > 60 else ep.content,
                str(ep.timestamp)[:19],
                f"{ep.valence:.2f}",
            )

        console.print(table)

    try:
        run_async(_recent())
    except Exception as e:
        console.print(f"[red]Error:[/red] {e}")
        raise typer.Exit(1)


# =============================================================================
# Semantic Sub-commands
# =============================================================================


@semantic_app.command("add")
def semantic_add(
    name: str = typer.Argument(..., help="Entity name"),
    description: str | None = typer.Option(None, "--desc", "-d", help="Description"),
    entity_type: str = typer.Option("concept", "--type", "-t", help="Entity type"),
):
    """Add a semantic entity."""
    content = f"{name}: {description}" if description else name
    store(content, memory_type="semantic")


@semantic_app.command("search")
def semantic_search(
    query: str = typer.Argument(..., help="Search query"),
    k: int = typer.Option(5, "--k", "-k", help="Number of results"),
):
    """Search semantic memories."""
    recall(query, k=k, memory_type="semantic")


# =============================================================================
# Procedural Sub-commands
# =============================================================================


@procedural_app.command("add")
def procedural_add(
    name: str = typer.Argument(..., help="Skill name"),
    description: str | None = typer.Option(None, "--desc", "-d", help="Description"),
):
    """Add a procedural skill."""
    content = f"{name}: {description}" if description else name
    store(content, tags=name, memory_type="procedural")


@procedural_app.command("search")
def procedural_search(
    query: str = typer.Argument(..., help="Search query"),
    k: int = typer.Option(5, "--k", "-k", help="Number of results"),
):
    """Search procedural skills."""
    recall(query, k=k, memory_type="procedural")


# =============================================================================
# Configuration Command
# =============================================================================


@app.command()
def config(
    show: bool = typer.Option(False, "--show", "-s", help="Show current configuration"),
    init: bool = typer.Option(False, "--init", "-i", help="Initialize config file"),
    path: Path | None = typer.Option(None, "--path", "-p", help="Config file path"),
):
    """Manage configuration."""
    from t4dm.core.config import get_settings

    if init:
        config_path = path or Path.home() / ".ww" / "config.yaml"
        config_path.parent.mkdir(parents=True, exist_ok=True)

        default_config = """# World Weaver Configuration
session_id: default
environment: development

# Storage
qdrant_host: localhost
qdrant_port: 6333

# neo4j_uri: bolt://localhost:7687
# neo4j_user: neo4j
# neo4j_password: password

# Embedding
embedding_model: bge-m3
embedding_dim: 1024

# API
api_host: 0.0.0.0
api_port: 8765
"""
        config_path.write_text(default_config)
        console.print(f"[green]Created config file:[/green] {config_path}")
        return

    if show:
        settings = get_settings()
        table = Table(title="Current Configuration")
        table.add_column("Setting", style="cyan")
        table.add_column("Value")

        for field in settings.__fields__:
            value = getattr(settings, field)
            # Mask sensitive values
            if "password" in field.lower() or "key" in field.lower():
                value = "***" if value else "Not set"
            table.add_row(field, str(value))

        console.print(table)


# =============================================================================
# Entry Point
# =============================================================================


def main():
    """Main entry point for the CLI."""
    logging.basicConfig(
        level=logging.WARNING,
        format="%(levelname)s: %(message)s",
    )
    app()


if __name__ == "__main__":
    main()
