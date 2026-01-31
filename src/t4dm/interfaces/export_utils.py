"""
Export Utilities - Export World Weaver memories to various formats.

Supports JSON, CSV, and GraphML exports for analysis and backup.
"""

import asyncio
import csv
import json
from datetime import datetime
from pathlib import Path
from typing import Optional

try:
    from rich.console import Console
    from rich.progress import BarColumn, Progress, SpinnerColumn, TextColumn
    RICH_AVAILABLE = True
except ImportError:
    RICH_AVAILABLE = False

from t4dm.memory.episodic import EpisodicMemory
from t4dm.memory.procedural import ProceduralMemory
from t4dm.memory.semantic import SemanticMemory
from t4dm.storage.qdrant_store import get_qdrant_store

# SEC-003 FIX: Default allowed export directories
DEFAULT_ALLOWED_DIRS = [
    Path.home() / "ww_exports",
    Path.home() / "Documents",
    Path.home() / "Downloads",
    Path("/tmp"),
]


def _validate_export_path(
    output_path: str | Path,
    allowed_dirs: list[Path] | None = None
) -> Path:
    """
    SEC-003 FIX: Validate export path to prevent path traversal attacks.

    Args:
        output_path: User-provided output path (string or Path)
        allowed_dirs: List of allowed parent directories

    Returns:
        Validated Path object

    Raises:
        ValueError: If path is outside allowed directories or contains traversal
    """
    if allowed_dirs is None:
        allowed_dirs = DEFAULT_ALLOWED_DIRS

    # Convert to string for pattern checks
    path_str = str(output_path)

    # Resolve to absolute path (this collapses .. and symlinks)
    resolved = Path(output_path).resolve()

    # Check for path traversal patterns in original path
    if ".." in path_str or path_str.startswith("/etc") or path_str.startswith("/usr"):
        raise ValueError(
            f"Path traversal detected in '{output_path}'. "
            f"Export paths must be within allowed directories."
        )

    # Verify path is within allowed directories
    is_allowed = False
    for allowed_dir in allowed_dirs:
        try:
            resolved.relative_to(allowed_dir.resolve())
            is_allowed = True
            break
        except ValueError:
            continue

    if not is_allowed:
        allowed_list = ", ".join(str(d) for d in allowed_dirs)
        raise ValueError(
            f"Export path '{resolved}' is not within allowed directories. "
            f"Allowed: {allowed_list}"
        )

    return resolved


class ExportUtility:
    """
    Memory export utility.

    Features:
    - Export episodes to JSON/CSV
    - Export entities to JSON/CSV
    - Export skills to JSON/CSV
    - Export knowledge graph to GraphML
    - Backup entire session
    """

    def __init__(
        self,
        session_id: str | None = None,
        console: Optional["Console"] = None,
    ):
        """
        Initialize export utility.

        Args:
            session_id: Session to export
            console: Rich console instance
        """
        self.session_id = session_id or "default"
        self.console = console or (Console() if RICH_AVAILABLE else None)

        self.episodic = EpisodicMemory(session_id=session_id)
        self.semantic = SemanticMemory(session_id=session_id)
        self.procedural = ProceduralMemory(session_id=session_id)
        self.vector_store = get_qdrant_store(session_id)

        self._initialized = False

    async def initialize(self) -> None:
        """Initialize storage backends."""
        if self._initialized:
            return

        if self.console and RICH_AVAILABLE:
            with Progress(
                SpinnerColumn(),
                TextColumn("[progress.description]{task.description}"),
                console=self.console,
            ) as progress:
                task = progress.add_task("Initializing export utility...", total=None)
                await self.episodic.initialize()
                await self.semantic.initialize()
                await self.procedural.initialize()
                progress.update(task, completed=True)
        else:
            await self.episodic.initialize()
            await self.semantic.initialize()
            await self.procedural.initialize()

        self._initialized = True

    async def export_episodes_json(
        self,
        output_path: str,
        limit: int | None = None,
        include_embeddings: bool = False,
    ) -> int:
        """
        Export episodes to JSON.

        Args:
            output_path: Output file path
            limit: Maximum episodes to export
            include_embeddings: Include embedding vectors

        Returns:
            Number of episodes exported
        """
        await self.initialize()

        if self.console:
            self.console.print("\n[bold cyan]Exporting Episodes to JSON...[/bold cyan]")

        # Collect episodes
        episodes = []
        offset = None

        while True:
            batch, next_offset = await self.vector_store.scroll(
                collection=self.vector_store.episodes_collection,
                limit=100,
                offset=offset,
                with_payload=True,
                with_vectors=include_embeddings,
            )

            for id_str, payload, vector in batch:
                episode_data = {
                    "id": id_str,
                    **payload,
                }
                if include_embeddings and vector:
                    episode_data["embedding"] = vector

                episodes.append(episode_data)

            if not next_offset or (limit and len(episodes) >= limit):
                break

            offset = next_offset

        # Trim to limit
        if limit:
            episodes = episodes[:limit]

        # SEC-003 FIX: Validate path before writing
        output_file = _validate_export_path(output_path)
        output_file.parent.mkdir(parents=True, exist_ok=True)

        with output_file.open("w", encoding="utf-8") as f:
            json.dump(
                {
                    "session_id": self.session_id,
                    "exported_at": datetime.now().isoformat(),
                    "count": len(episodes),
                    "episodes": episodes,
                },
                f,
                indent=2,
                ensure_ascii=False,
            )

        if self.console:
            self.console.print(f"[green]✓[/green] Exported {len(episodes)} episodes to {output_path}")

        return len(episodes)

    async def export_episodes_csv(
        self,
        output_path: str,
        limit: int | None = None,
    ) -> int:
        """
        Export episodes to CSV.

        Args:
            output_path: Output file path
            limit: Maximum episodes to export

        Returns:
            Number of episodes exported
        """
        await self.initialize()

        if self.console:
            self.console.print("\n[bold cyan]Exporting Episodes to CSV...[/bold cyan]")

        # Collect episodes
        episodes = []
        offset = None

        while True:
            batch, next_offset = await self.vector_store.scroll(
                collection=self.vector_store.episodes_collection,
                limit=100,
                offset=offset,
                with_payload=True,
                with_vectors=False,
            )

            for id_str, payload, _ in batch:
                episodes.append({
                    "id": id_str,
                    "session_id": payload.get("session_id", ""),
                    "content": payload.get("content", ""),
                    "timestamp": payload.get("timestamp", ""),
                    "outcome": payload.get("outcome", ""),
                    "emotional_valence": payload.get("emotional_valence", 0.5),
                    "access_count": payload.get("access_count", 1),
                    "stability": payload.get("stability", 1.0),
                    "last_accessed": payload.get("last_accessed", ""),
                })

            if not next_offset or (limit and len(episodes) >= limit):
                break

            offset = next_offset

        # Trim to limit
        if limit:
            episodes = episodes[:limit]

        # SEC-003 FIX: Validate path before writing
        output_file = _validate_export_path(output_path)
        output_file.parent.mkdir(parents=True, exist_ok=True)

        with output_file.open("w", encoding="utf-8", newline="") as f:
            if episodes:
                writer = csv.DictWriter(f, fieldnames=episodes[0].keys())
                writer.writeheader()
                writer.writerows(episodes)

        if self.console:
            self.console.print(f"[green]✓[/green] Exported {len(episodes)} episodes to {output_path}")

        return len(episodes)

    async def export_entities_json(
        self,
        output_path: str,
        limit: int | None = None,
        include_embeddings: bool = False,
    ) -> int:
        """
        Export entities to JSON.

        Args:
            output_path: Output file path
            limit: Maximum entities to export
            include_embeddings: Include embedding vectors

        Returns:
            Number of entities exported
        """
        await self.initialize()

        if self.console:
            self.console.print("\n[bold cyan]Exporting Entities to JSON...[/bold cyan]")

        # Collect entities
        entities = []
        offset = None

        while True:
            batch, next_offset = await self.vector_store.scroll(
                collection=self.vector_store.entities_collection,
                limit=100,
                offset=offset,
                with_payload=True,
                with_vectors=include_embeddings,
            )

            for id_str, payload, vector in batch:
                entity_data = {
                    "id": id_str,
                    **payload,
                }
                if include_embeddings and vector:
                    entity_data["embedding"] = vector

                entities.append(entity_data)

            if not next_offset or (limit and len(entities) >= limit):
                break

            offset = next_offset

        # Trim to limit
        if limit:
            entities = entities[:limit]

        # SEC-003 FIX: Validate path before writing
        output_file = _validate_export_path(output_path)
        output_file.parent.mkdir(parents=True, exist_ok=True)

        with output_file.open("w", encoding="utf-8") as f:
            json.dump(
                {
                    "session_id": self.session_id,
                    "exported_at": datetime.now().isoformat(),
                    "count": len(entities),
                    "entities": entities,
                },
                f,
                indent=2,
                ensure_ascii=False,
            )

        if self.console:
            self.console.print(f"[green]✓[/green] Exported {len(entities)} entities to {output_path}")

        return len(entities)

    async def export_entities_csv(
        self,
        output_path: str,
        limit: int | None = None,
    ) -> int:
        """
        Export entities to CSV.

        Args:
            output_path: Output file path
            limit: Maximum entities to export

        Returns:
            Number of entities exported
        """
        await self.initialize()

        if self.console:
            self.console.print("\n[bold cyan]Exporting Entities to CSV...[/bold cyan]")

        # Collect entities
        entities = []
        offset = None

        while True:
            batch, next_offset = await self.vector_store.scroll(
                collection=self.vector_store.entities_collection,
                limit=100,
                offset=offset,
                with_payload=True,
                with_vectors=False,
            )

            for id_str, payload, _ in batch:
                entities.append({
                    "id": id_str,
                    "session_id": payload.get("session_id", ""),
                    "name": payload.get("name", ""),
                    "entity_type": payload.get("entity_type", ""),
                    "summary": payload.get("summary", ""),
                    "details": payload.get("details", ""),
                    "source": payload.get("source", ""),
                    "stability": payload.get("stability", 1.0),
                    "access_count": payload.get("access_count", 1),
                    "last_accessed": payload.get("last_accessed", ""),
                    "created_at": payload.get("created_at", ""),
                })

            if not next_offset or (limit and len(entities) >= limit):
                break

            offset = next_offset

        # Trim to limit
        if limit:
            entities = entities[:limit]

        # SEC-003 FIX: Validate path before writing
        output_file = _validate_export_path(output_path)
        output_file.parent.mkdir(parents=True, exist_ok=True)

        with output_file.open("w", encoding="utf-8", newline="") as f:
            if entities:
                writer = csv.DictWriter(f, fieldnames=entities[0].keys())
                writer.writeheader()
                writer.writerows(entities)

        if self.console:
            self.console.print(f"[green]✓[/green] Exported {len(entities)} entities to {output_path}")

        return len(entities)

    async def export_skills_json(
        self,
        output_path: str,
        limit: int | None = None,
        include_embeddings: bool = False,
    ) -> int:
        """
        Export skills to JSON.

        Args:
            output_path: Output file path
            limit: Maximum skills to export
            include_embeddings: Include embedding vectors

        Returns:
            Number of skills exported
        """
        await self.initialize()

        if self.console:
            self.console.print("\n[bold cyan]Exporting Skills to JSON...[/bold cyan]")

        # Collect skills
        skills = []
        offset = None

        while True:
            batch, next_offset = await self.vector_store.scroll(
                collection=self.vector_store.procedures_collection,
                limit=100,
                offset=offset,
                with_payload=True,
                with_vectors=include_embeddings,
            )

            for id_str, payload, vector in batch:
                skill_data = {
                    "id": id_str,
                    **payload,
                }
                if include_embeddings and vector:
                    skill_data["embedding"] = vector

                skills.append(skill_data)

            if not next_offset or (limit and len(skills) >= limit):
                break

            offset = next_offset

        # Trim to limit
        if limit:
            skills = skills[:limit]

        # SEC-003 FIX: Validate path before writing
        output_file = _validate_export_path(output_path)
        output_file.parent.mkdir(parents=True, exist_ok=True)

        with output_file.open("w", encoding="utf-8") as f:
            json.dump(
                {
                    "session_id": self.session_id,
                    "exported_at": datetime.now().isoformat(),
                    "count": len(skills),
                    "skills": skills,
                },
                f,
                indent=2,
                ensure_ascii=False,
            )

        if self.console:
            self.console.print(f"[green]✓[/green] Exported {len(skills)} skills to {output_path}")

        return len(skills)

    async def export_graph_graphml(
        self,
        output_path: str,
        limit: int | None = None,
    ) -> int:
        """
        Export knowledge graph to GraphML format.

        Args:
            output_path: Output file path
            limit: Maximum entities to include

        Returns:
            Number of nodes exported
        """
        await self.initialize()

        if self.console:
            self.console.print("\n[bold cyan]Exporting Knowledge Graph to GraphML...[/bold cyan]")

        # Collect entities (nodes)
        entities = []
        offset = None

        while True:
            batch, next_offset = await self.vector_store.scroll(
                collection=self.vector_store.entities_collection,
                limit=100,
                offset=offset,
                with_payload=True,
                with_vectors=False,
            )

            for id_str, payload, _ in batch:
                entities.append({
                    "id": id_str,
                    "name": payload.get("name", ""),
                    "type": payload.get("entity_type", ""),
                    "summary": payload.get("summary", ""),
                })

            if not next_offset or (limit and len(entities) >= limit):
                break

            offset = next_offset

        # Trim to limit
        if limit:
            entities = entities[:limit]

        # Collect relationships (edges)
        relationships = []
        for entity in entities:
            rels = await self.semantic.graph_store.get_relationships(
                node_id=entity["id"],
                direction="out",
            )
            for rel in rels:
                relationships.append({
                    "source": entity["id"],
                    "target": rel["other_id"],
                    "type": rel.get("type", "RELATED"),
                    "weight": rel.get("properties", {}).get("weight", 0.0),
                })

        # SEC-003 FIX: Validate path before writing
        output_file = _validate_export_path(output_path)
        output_file.parent.mkdir(parents=True, exist_ok=True)

        with output_file.open("w", encoding="utf-8") as f:
            f.write('<?xml version="1.0" encoding="UTF-8"?>\n')
            f.write('<graphml xmlns="http://graphml.graphdrawing.org/xmlns" ')
            f.write('xmlns:xsi="http://www.w3.org/2001/XMLSchema-instance" ')
            f.write('xsi:schemaLocation="http://graphml.graphdrawing.org/xmlns ')
            f.write('http://graphml.graphdrawing.org/xmlns/1.0/graphml.xsd">\n')

            # Define keys
            f.write('  <key id="name" for="node" attr.name="name" attr.type="string"/>\n')
            f.write('  <key id="type" for="node" attr.name="type" attr.type="string"/>\n')
            f.write('  <key id="summary" for="node" attr.name="summary" attr.type="string"/>\n')
            f.write('  <key id="weight" for="edge" attr.name="weight" attr.type="double"/>\n')
            f.write('  <key id="reltype" for="edge" attr.name="reltype" attr.type="string"/>\n')

            f.write('  <graph id="WorldWeaver" edgedefault="directed">\n')

            # Write nodes
            for entity in entities:
                f.write(f'    <node id="{entity["id"]}">\n')
                f.write(f'      <data key="name">{self._escape_xml(entity["name"])}</data>\n')
                f.write(f'      <data key="type">{self._escape_xml(entity["type"])}</data>\n')
                f.write(f'      <data key="summary">{self._escape_xml(entity["summary"])}</data>\n')
                f.write("    </node>\n")

            # Write edges
            for i, rel in enumerate(relationships):
                f.write(f'    <edge id="e{i}" source="{rel["source"]}" target="{rel["target"]}">\n')
                f.write(f'      <data key="weight">{rel["weight"]}</data>\n')
                f.write(f'      <data key="reltype">{self._escape_xml(rel["type"])}</data>\n')
                f.write("    </edge>\n")

            f.write("  </graph>\n")
            f.write("</graphml>\n")

        if self.console:
            self.console.print(
                f"[green]✓[/green] Exported {len(entities)} nodes, "
                f"{len(relationships)} edges to {output_path}"
            )

        return len(entities)

    def _escape_xml(self, text: str) -> str:
        """Escape XML special characters."""
        return (
            text.replace("&", "&amp;")
            .replace("<", "&lt;")
            .replace(">", "&gt;")
            .replace('"', "&quot;")
            .replace("'", "&apos;")
        )

    async def backup_session(
        self,
        output_dir: str,
    ) -> dict[str, int]:
        """
        Backup entire session to directory.

        Creates:
        - episodes.json
        - entities.json
        - skills.json
        - graph.graphml
        - metadata.json

        Args:
            output_dir: Output directory

        Returns:
            Dict with counts of exported items
        """
        await self.initialize()

        # SEC-003 FIX: Validate directory path before writing
        output_path = _validate_export_path(output_dir)
        output_path.mkdir(parents=True, exist_ok=True)

        if self.console:
            self.console.print(f"\n[bold cyan]Backing Up Session: {self.session_id}[/bold cyan]")
            self.console.print(f"[white]Output: {output_dir}[/white]\n")

        results = {}

        # Export episodes
        results["episodes"] = await self.export_episodes_json(
            output_path / "episodes.json",
            include_embeddings=False,
        )

        # Export entities
        results["entities"] = await self.export_entities_json(
            output_path / "entities.json",
            include_embeddings=False,
        )

        # Export skills
        results["skills"] = await self.export_skills_json(
            output_path / "skills.json",
            include_embeddings=False,
        )

        # Export graph
        results["graph_nodes"] = await self.export_graph_graphml(
            output_path / "graph.graphml",
        )

        # Write metadata
        metadata = {
            "session_id": self.session_id,
            "exported_at": datetime.now().isoformat(),
            "counts": results,
        }

        with (output_path / "metadata.json").open("w") as f:
            json.dump(metadata, f, indent=2)

        if self.console:
            self.console.print("\n[bold green]✓ Backup Complete[/bold green]")
            self.console.print(f"  Episodes: {results['episodes']}")
            self.console.print(f"  Entities: {results['entities']}")
            self.console.print(f"  Skills: {results['skills']}")
            self.console.print(f"  Graph Nodes: {results['graph_nodes']}")

        return results


async def main():
    """CLI demo for export utility."""
    import sys

    session_id = sys.argv[1] if len(sys.argv) > 1 else "default"
    output_dir = sys.argv[2] if len(sys.argv) > 2 else f"./backups/{session_id}"

    exporter = ExportUtility(session_id=session_id)
    await exporter.backup_session(output_dir)


if __name__ == "__main__":
    asyncio.run(main())
