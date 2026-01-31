"""P4-03: REM phase — clustering + prototype creation + T4DX compaction."""

from __future__ import annotations

import logging
import uuid
import time
from dataclasses import dataclass
from typing import Any

import numpy as np

from t4dm.storage.t4dx.engine import T4DXEngine
from t4dm.storage.t4dx.types import EdgeRecord, ItemRecord

logger = logging.getLogger(__name__)


@dataclass
class REMConfig:
    """REM consolidation configuration."""

    kappa_min: float = 0.3
    kappa_max: float = 0.7
    prototype_kappa: float = 0.85
    min_cluster_size: int = 3
    similarity_threshold: float = 0.7


@dataclass
class REMResult:
    """Results from REM consolidation."""

    candidates_scanned: int = 0
    clusters_formed: int = 0
    prototypes_created: int = 0
    segment_id: int | None = None


class REMPhase:
    """REM consolidation: cluster transitional items, create semantic prototypes.

    1. SCAN(0.3 ≤ κ ≤ 0.7) from T4DX
    2. Cluster embeddings by cosine similarity
    3. Create prototype Items (centroid, κ=0.85, type=semantic)
    4. INSERT prototypes with MERGED_FROM edges
    5. Trigger Compactor.rem_compact()
    """

    def __init__(
        self,
        engine: T4DXEngine,
        cfg: REMConfig | None = None,
    ) -> None:
        self.engine = engine
        self.cfg = cfg or REMConfig()

    def run(self) -> REMResult:
        """Execute REM consolidation phase."""
        result = REMResult()

        candidates = self.engine.scan(
            kappa_min=self.cfg.kappa_min,
            kappa_max=self.cfg.kappa_max,
        )
        result.candidates_scanned = len(candidates)

        if len(candidates) < self.cfg.min_cluster_size:
            logger.info("REM: too few candidates (%d)", len(candidates))
            return result

        # Cluster by cosine similarity (simple greedy approach)
        clusters = self._cluster(candidates)
        result.clusters_formed = len(clusters)

        # Create prototypes
        for cluster in clusters:
            if len(cluster) < self.cfg.min_cluster_size:
                continue
            self._create_prototype(cluster, result)

        # Trigger REM compaction
        new_sid = self.engine.rem_compact(
            kappa_threshold=self.cfg.kappa_min,
            prototype_kappa=self.cfg.prototype_kappa,
        )
        result.segment_id = new_sid

        logger.info(
            "REM complete: scanned=%d, clusters=%d, prototypes=%d",
            result.candidates_scanned, result.clusters_formed, result.prototypes_created,
        )
        return result

    def _cluster(self, items: list[ItemRecord]) -> list[list[ItemRecord]]:
        """Simple greedy cosine clustering."""
        if not items:
            return []

        vecs = np.array([r.vector for r in items if r.vector], dtype=np.float32)
        if len(vecs) < 2:
            return []

        norms = np.linalg.norm(vecs, axis=1, keepdims=True)
        norms = np.where(norms == 0, 1.0, norms)
        normed = vecs / norms

        assigned = [False] * len(items)
        clusters: list[list[ItemRecord]] = []

        for i in range(len(items)):
            if assigned[i] or not items[i].vector:
                continue
            cluster = [items[i]]
            assigned[i] = True

            for j in range(i + 1, len(items)):
                if assigned[j] or not items[j].vector:
                    continue
                sim = float(normed[i] @ normed[j])
                if sim >= self.cfg.similarity_threshold:
                    cluster.append(items[j])
                    assigned[j] = True

            clusters.append(cluster)

        return clusters

    def _create_prototype(
        self, cluster: list[ItemRecord], result: REMResult,
    ) -> None:
        """Create a semantic prototype from a cluster."""
        vecs = np.array([r.vector for r in cluster], dtype=np.float32)
        centroid = vecs.mean(axis=0).tolist()

        contents = [r.content for r in cluster]
        proto_content = f"[prototype] {contents[0][:80]}... (+{len(cluster)-1} more)"

        now = time.time()
        proto = ItemRecord(
            id=uuid.uuid4().bytes,
            vector=centroid,
            kappa=self.cfg.prototype_kappa,
            importance=max(r.importance for r in cluster),
            event_time=max(r.event_time for r in cluster),
            record_time=now,
            valid_from=min(r.valid_from for r in cluster),
            valid_until=None,
            item_type="semantic",
            content=proto_content,
            access_count=sum(r.access_count for r in cluster),
            session_id=None,
            metadata={
                "source_count": len(cluster),
                "source_ids": [r.id.hex() for r in cluster],
            },
        )
        self.engine.insert(proto)

        # Create MERGED_FROM edges
        for src in cluster:
            edge = EdgeRecord(
                source_id=src.id,
                target_id=proto.id,
                edge_type="CONSOLIDATED_INTO",
                weight=0.8,
            )
            self.engine.insert_edge(edge)

        result.prototypes_created += 1
