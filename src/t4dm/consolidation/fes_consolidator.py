"""
FES Consolidator - Fast Episodic Store to Episodic/Semantic consolidation.

Biological inspiration: Sleep-dependent consolidation, sharp-wave ripples.

This module transfers high-value memories from the fast episodic store
to the permanent episodic and semantic stores, implementing memory
consolidation similar to hippocampal-cortical transfer during sleep.
"""

import asyncio
import logging
import re
from datetime import datetime
from typing import Any

from t4dm.core.types import EntityType, Episode
from t4dm.memory.fast_episodic import FastEpisodicStore

logger = logging.getLogger(__name__)


# Security limits
MAX_CONSOLIDATION_BATCH = 100
MAX_ENTITY_EXTRACTION = 50


def _score_entity(entity: str, entity_type: str) -> float:
    """
    Score entity extraction confidence.

    ATOM-P3-16: Adds confidence scoring for regex-based entity extraction.

    Args:
        entity: Extracted entity name
        entity_type: Type of entity (file, function, name)

    Returns:
        Confidence score [0, 1]
    """
    if entity_type == "file" and "/" not in entity and "\\" not in entity:
        return 0.3  # Low confidence without path separator
    if entity_type == "function" and not entity[0].isalpha():
        return 0.2  # Functions should start with letter
    if entity_type == "name" and len(entity.split()) < 2:
        return 0.3  # Names usually have 2+ words
    return 0.8  # Default confidence


def extract_entities_simple(content: str) -> list[dict[str, Any]]:
    """
    Simple entity extraction from episode content.

    For production, this would use an LLM or NER model.
    This implementation extracts basic patterns.

    ATOM-P3-16: Added confidence scoring and filtering.

    Args:
        content: Episode content text

    Returns:
        List of entity dicts with name, type, and description
    """
    entities = []

    # Extract code-like entities (files, functions, classes)
    # Files: anything.ext
    file_pattern = r"\b[\w\-/]+\.\w{1,10}\b"
    files = re.findall(file_pattern, content)
    for f in files[:MAX_ENTITY_EXTRACTION]:
        confidence = _score_entity(f, "file")
        if confidence >= 0.5:
            entities.append({
                "name": f,
                "type": EntityType.CONCEPT,
                "description": f"File referenced: {f}",
                "confidence": confidence,
            })

    # Function/method calls
    func_pattern = r"\b(\w+)\s*\("
    funcs = re.findall(func_pattern, content)
    for func in list(set(funcs))[:MAX_ENTITY_EXTRACTION]:
        if len(func) > 2 and func not in ["if", "for", "while", "print"]:
            confidence = _score_entity(func, "function")
            if confidence >= 0.5:
                entities.append({
                    "name": func,
                    "type": EntityType.CONCEPT,
                    "description": f"Function/method: {func}",
                    "confidence": confidence,
                })

    # Capitalized phrases (potential named entities)
    name_pattern = r"\b([A-Z][a-z]+(?:\s+[A-Z][a-z]+)*)\b"
    names = re.findall(name_pattern, content)
    for name in list(set(names))[:MAX_ENTITY_EXTRACTION]:
        if len(name) > 2:
            confidence = _score_entity(name, "name")
            if confidence >= 0.5:
                entities.append({
                    "name": name,
                    "type": EntityType.PERSON,
                    "description": f"Named entity: {name}",
                    "confidence": confidence,
                })

    return entities[:MAX_ENTITY_EXTRACTION]


class FESConsolidator:
    """
    Consolidates fast episodic memories to episodic and semantic stores.

    Biological inspiration: Sleep-dependent consolidation, sharp-wave ripples.

    Features:
    - Selects high-value memories (high replay count, high salience)
    - Transfers to permanent episodic store
    - Extracts entities/relationships for semantic store
    - Supports background consolidation

    Integration:
    - Works with existing EpisodicMemory and SemanticMemory services
    - Can run as background task or on-demand
    """

    def __init__(
        self,
        fast_store: FastEpisodicStore,
        episodic_store: Any | None = None,
        semantic_store: Any | None = None,
        consolidation_rate: float = 0.1,  # 10% of candidates per cycle
        min_consolidation_score: float = 0.5
    ):
        """
        Initialize FES consolidator.

        Args:
            fast_store: FastEpisodicStore to consolidate from
            episodic_store: Optional EpisodicMemory to consolidate to
            semantic_store: Optional SemanticMemory for entity extraction
            consolidation_rate: Fraction of candidates to process per cycle
            min_consolidation_score: Minimum score for consolidation
        """
        self.fast_store = fast_store
        self.episodic_store = episodic_store
        self.semantic_store = semantic_store
        self.consolidation_rate = consolidation_rate
        self.min_consolidation_score = min_consolidation_score

        # Statistics
        self._cycles_run = 0
        self._episodes_consolidated = 0
        self._entities_extracted = 0

        # Background task reference
        self._background_task: asyncio.Task | None = None

    async def consolidate_cycle(
        self,
        max_episodes: int = 10
    ) -> list[dict[str, Any]]:
        """
        Run one consolidation cycle.

        Process:
        1. Select candidates from FES (high replay, salience)
        2. Transfer to standard episodic store (if available)
        3. Extract entities/relationships → semantic store (if available)
        4. Mark as consolidated in FES

        Args:
            max_episodes: Maximum episodes to consolidate this cycle

        Returns:
            List of consolidation results
        """
        max_episodes = min(max_episodes, MAX_CONSOLIDATION_BATCH)

        # Get candidates
        candidates = self.fast_store.get_consolidation_candidates(max_episodes)

        if not candidates:
            logger.debug("No consolidation candidates found")
            return []

        results = []

        for eid, episode, score in candidates:
            if score < self.min_consolidation_score:
                continue

            result = {
                "episode_id": eid,
                "consolidation_score": score,
                "episodic_stored": False,
                "entities_extracted": 0,
                "timestamp": datetime.now().isoformat()
            }

            # Transfer to episodic store
            if self.episodic_store is not None:
                try:
                    episodic_result = await self._store_to_episodic(episode)
                    result["episodic_stored"] = True
                    result["episodic_id"] = str(episodic_result)
                except Exception as e:
                    logger.error(f"Failed to store to episodic: {e}")
                    result["episodic_error"] = str(e)

            # Extract entities for semantic store
            if self.semantic_store is not None:
                try:
                    entities = await self._extract_and_store_entities(episode)
                    result["entities_extracted"] = len(entities)
                    self._entities_extracted += len(entities)
                except Exception as e:
                    logger.error(f"Failed to extract entities: {e}")
                    result["semantic_error"] = str(e)

            # Mark as consolidated in FES
            self.fast_store.mark_consolidated(eid)

            results.append(result)
            self._episodes_consolidated += 1

        self._cycles_run += 1

        logger.info(
            f"Consolidation cycle {self._cycles_run}: "
            f"{len(results)} episodes consolidated"
        )

        return results

    async def _store_to_episodic(self, episode: Episode) -> Any:
        """
        Store episode to permanent episodic memory.

        Args:
            episode: Episode to store

        Returns:
            Storage result (ID or result object)
        """
        # Handle different episodic store interfaces
        if hasattr(self.episodic_store, "store"):
            # Async store method
            return await self.episodic_store.store(episode)
        if hasattr(self.episodic_store, "write"):
            # Sync write method
            return self.episodic_store.write(episode)
        logger.warning("Episodic store has no recognized storage method")
        return episode.id

    async def _extract_and_store_entities(
        self,
        episode: Episode
    ) -> list[dict[str, Any]]:
        """
        Extract entities from episode and store to semantic memory.

        Args:
            episode: Episode to extract from

        Returns:
            List of extracted entities
        """
        # Extract entities
        entities = extract_entities_simple(episode.content)

        # Store to semantic
        for entity in entities:
            try:
                if hasattr(self.semantic_store, "add_entity"):
                    await self.semantic_store.add_entity(entity)
                elif hasattr(self.semantic_store, "store_entity"):
                    self.semantic_store.store_entity(entity)
            except Exception as e:
                logger.debug(f"Entity storage skipped: {e}")

        return entities

    async def consolidate_all(self) -> dict[str, Any]:
        """
        Consolidate all eligible episodes.

        Runs multiple cycles until no more candidates.

        Returns:
            Summary of all consolidation
        """
        total_results = []
        cycle_count = 0
        max_cycles = 100  # Safety limit

        while cycle_count < max_cycles:
            results = await self.consolidate_cycle(max_episodes=50)
            if not results:
                break
            total_results.extend(results)
            cycle_count += 1

        return {
            "cycles": cycle_count,
            "episodes_consolidated": len(total_results),
            "results": total_results
        }

    async def start_background_consolidation(
        self,
        interval_seconds: int = 3600
    ):
        """
        Start background consolidation task.

        Runs consolidation at specified interval.

        Args:
            interval_seconds: Seconds between consolidation cycles
        """
        if self._background_task is not None:
            logger.warning("Background consolidation already running")
            return

        async def background_loop():
            while True:
                try:
                    await asyncio.sleep(interval_seconds)
                    await self.consolidate_cycle()
                except asyncio.CancelledError:
                    logger.info("Background consolidation stopped")
                    break
                except Exception as e:
                    logger.error(f"Background consolidation error: {e}")

        self._background_task = asyncio.create_task(background_loop())
        logger.info(
            f"Started background consolidation (interval: {interval_seconds}s)"
        )

    async def stop_background_consolidation(self):
        """Stop background consolidation task."""
        if self._background_task is not None:
            self._background_task.cancel()
            try:
                await self._background_task
            except asyncio.CancelledError:
                pass
            self._background_task = None
            logger.info("Background consolidation stopped")

    def get_stats(self) -> dict[str, Any]:
        """Get consolidator statistics."""
        return {
            "cycles_run": self._cycles_run,
            "episodes_consolidated": self._episodes_consolidated,
            "entities_extracted": self._entities_extracted,
            "fes_count": self.fast_store.count,
            "fes_capacity_usage": self.fast_store.capacity_usage,
            "background_running": self._background_task is not None
        }


class ReplayConsolidator(FESConsolidator):
    """
    Consolidator with replay-based selection.

    Prioritizes memories with high replay frequency,
    mimicking hippocampal replay during sleep.
    """

    def __init__(self, *args, replay_weight: float = 0.7, **kwargs):
        """
        Initialize replay-focused consolidator.

        Args:
            replay_weight: Weight for replay count in scoring (0-1)
        """
        super().__init__(*args, **kwargs)
        self.replay_weight = replay_weight

    async def consolidate_cycle(
        self,
        max_episodes: int = 10
    ) -> list[dict[str, Any]]:
        """
        Consolidation cycle with replay-weighted selection.

        High replay count episodes are prioritized, similar to
        how frequently replayed memories during sleep consolidation
        become more stable.
        """
        max_episodes = min(max_episodes, MAX_CONSOLIDATION_BATCH)

        # Custom candidate scoring with replay weighting
        candidates = []
        for eid, entry in self.fast_store.entries.items():
            if entry.consolidated:
                continue

            # Replay-weighted score
            replay_score = self.fast_store.access_counts[eid]
            base_score = entry.salience

            weighted_score = (
                self.replay_weight * min(replay_score / 10, 1.0) +
                (1 - self.replay_weight) * base_score
            )

            candidates.append((eid, entry.episode, weighted_score))

        # Sort by weighted score
        candidates = sorted(
            candidates, key=lambda x: x[2], reverse=True
        )[:max_episodes]

        if not candidates:
            return []

        results = []
        for eid, episode, score in candidates:
            result = {
                "episode_id": eid,
                "consolidation_score": score,
                "replay_count": self.fast_store.access_counts[eid],
                "timestamp": datetime.now().isoformat()
            }

            # Store to episodic
            if self.episodic_store is not None:
                try:
                    await self._store_to_episodic(episode)
                    result["episodic_stored"] = True
                except Exception as e:
                    result["episodic_error"] = str(e)

            # Mark consolidated
            self.fast_store.mark_consolidated(eid)
            results.append(result)
            self._episodes_consolidated += 1

        self._cycles_run += 1
        return results
