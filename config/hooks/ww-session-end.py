#!/usr/bin/env python3
"""
World Weaver Session End Hook.

Triggers light memory consolidation at end of session.
"""

import asyncio
import os
import sys
from datetime import datetime


async def consolidate():
    """Run light consolidation on session end."""
    try:
        # Add project to path
        sys.path.insert(0, "/mnt/projects/ww/src")

        from ww.consolidation.service import get_consolidation_service

        session_id = os.environ.get("WW_SESSION_ID", "unknown")
        print()
        print("=" * 50)
        print("World Weaver Session End - Memory Consolidation")
        print("=" * 50)
        print(f"Session ID: {session_id}")
        print(f"Timestamp:  {datetime.now().isoformat()}")
        print()

        consolidation = get_consolidation_service()
        result = await consolidation.consolidate(
            consolidation_type="light",
            session_filter=session_id if session_id != "unknown" else None,
        )

        print(f"Status:     {result['status']}")
        print(f"Duration:   {result.get('duration_seconds', 0):.2f}s")

        if "results" in result and "light" in result["results"]:
            light = result["results"]["light"]
            print(f"Episodes:   {light.get('episodes_scanned', 0)} scanned")
            print(f"Duplicates: {light.get('duplicates_found', 0)} found, {light.get('cleaned', 0)} cleaned")

        print()
        print("=" * 50)

    except Exception as e:
        print(f"Consolidation skipped: {e}")


def main():
    """Entry point."""
    asyncio.run(consolidate())


if __name__ == "__main__":
    main()
