#!/usr/bin/env python3
"""
World Weaver Interfaces Demo

Demonstrates all interface modules with sample usage.
"""

import asyncio
from datetime import datetime, timedelta

from ww.interfaces import (
    MemoryExplorer,
    TraceViewer,
    CRUDManager,
    ExportUtility,
    SystemDashboard,
)


async def demo_memory_explorer():
    """Demonstrate MemoryExplorer functionality."""
    print("\n" + "="*60)
    print("MEMORY EXPLORER DEMO")
    print("="*60 + "\n")

    explorer = MemoryExplorer(session_id="demo")
    await explorer.initialize()

    # List episodes
    print("Listing recent episodes...")
    await explorer.list_episodes(limit=10)

    # Search
    print("\nSearching memories...")
    await explorer.search("test", limit=5)


async def demo_trace_viewer():
    """Demonstrate TraceViewer functionality."""
    print("\n" + "="*60)
    print("TRACE VIEWER DEMO")
    print("="*60 + "\n")

    viewer = TraceViewer(session_id="demo")
    await viewer.initialize()

    # Show access timeline
    print("Access timeline (last 24 hours)...")
    await viewer.show_access_timeline(hours=24)

    # Show decay curves
    print("\nMemory decay curves...")
    await viewer.show_decay_curves(sample_size=10)

    # Show consolidation events
    print("\nConsolidation events...")
    await viewer.show_consolidation_events(limit=10)


async def demo_crud_manager():
    """Demonstrate CRUDManager functionality."""
    print("\n" + "="*60)
    print("CRUD MANAGER DEMO")
    print("="*60 + "\n")

    manager = CRUDManager(session_id="demo")
    await manager.initialize()

    # Create episode
    print("Creating episode...")
    episode = await manager.create_episode(
        content="Testing CRUD interface with demo episode",
        outcome="success",
        emotional_valence=0.8,
        context={"project": "world-weaver", "tool": "interfaces"},
    )
    print(f"Created episode: {episode.id}\n")

    # Create entity
    print("Creating entity...")
    entity = await manager.create_entity(
        name="World Weaver Interfaces",
        entity_type="TOOL",
        summary="Rich terminal UI for memory exploration",
        details="Provides memory explorer, trace viewer, CRUD manager, export utility, and dashboard",
    )
    print(f"Created entity: {entity.id}\n")

    # Create skill
    print("Creating skill...")
    skill = await manager.create_skill(
        name="Explore Memories",
        domain="coding",
        steps=[
            {
                "order": 1,
                "action": "Initialize memory explorer",
                "tool": "ww-explore",
            },
            {
                "order": 2,
                "action": "Search or browse memories",
                "tool": "rich",
            },
            {
                "order": 3,
                "action": "View detailed information",
                "tool": "rich",
            },
        ],
        trigger_pattern="When user wants to explore memory contents",
    )
    print(f"Created skill: {skill.id}\n")

    # Batch create episodes
    print("Batch creating episodes...")
    episodes = await manager.batch_create_episodes([
        {
            "content": f"Demo episode {i}",
            "outcome": "success",
            "emotional_valence": 0.5 + (i * 0.1),
        }
        for i in range(5)
    ])
    print(f"Created {len(episodes)} episodes in batch\n")


async def demo_export_utility():
    """Demonstrate ExportUtility functionality."""
    print("\n" + "="*60)
    print("EXPORT UTILITY DEMO")
    print("="*60 + "\n")

    exporter = ExportUtility(session_id="demo")
    await exporter.initialize()

    # Export episodes to JSON
    print("Exporting episodes to JSON...")
    count = await exporter.export_episodes_json(
        "/tmp/ww_demo_episodes.json",
        limit=100,
        include_embeddings=False,
    )
    print(f"Exported {count} episodes\n")

    # Export entities to CSV
    print("Exporting entities to CSV...")
    count = await exporter.export_entities_csv(
        "/tmp/ww_demo_entities.csv",
        limit=100,
    )
    print(f"Exported {count} entities\n")

    # Full session backup
    print("Creating full session backup...")
    results = await exporter.backup_session("/tmp/ww_demo_backup")
    print(f"Backup complete:")
    print(f"  Episodes: {results['episodes']}")
    print(f"  Entities: {results['entities']}")
    print(f"  Skills: {results['skills']}")
    print(f"  Graph nodes: {results['graph_nodes']}")
    print()


async def demo_system_dashboard():
    """Demonstrate SystemDashboard functionality."""
    print("\n" + "="*60)
    print("SYSTEM DASHBOARD DEMO")
    print("="*60 + "\n")

    dashboard = SystemDashboard(session_id="demo")
    await dashboard.initialize()

    # Show static dashboard
    print("Displaying system dashboard...")
    await dashboard.show()

    # Show detailed health
    print("\n\nDetailed health report...")
    await dashboard.show_detailed_health()


async def main():
    """Run all demos."""
    print("""
    ╔══════════════════════════════════════════════════════════╗
    ║                                                          ║
    ║         World Weaver Interfaces Demo                    ║
    ║                                                          ║
    ║  Comprehensive demonstration of all interface modules   ║
    ║                                                          ║
    ╚══════════════════════════════════════════════════════════╝
    """)

    # Run demos sequentially
    try:
        await demo_crud_manager()  # Create some data first
        await demo_memory_explorer()
        await demo_trace_viewer()
        await demo_export_utility()
        await demo_system_dashboard()

        print("\n" + "="*60)
        print("DEMO COMPLETE")
        print("="*60)
        print("\nAll interface modules demonstrated successfully!")
        print("\nNext steps:")
        print("  - Try interactive mode: ww-explore")
        print("  - View traces: ww-trace demo")
        print("  - Monitor system: ww-dashboard demo live")
        print("  - Backup session: ww-export demo /tmp/backup")
        print()

    except Exception as e:
        print(f"\n[ERROR] Demo failed: {e}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    asyncio.run(main())
