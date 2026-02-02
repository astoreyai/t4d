"""
World Weaver REST API System Routes.

Health checks, statistics, system operations, and documentation.

API-CRITICAL-002/003 FIX: Consolidation and doc endpoints require admin auth.
"""

import logging
from datetime import datetime
from pathlib import Path

from fastapi import APIRouter, HTTPException, status
from fastapi.responses import PlainTextResponse
from pydantic import BaseModel

from t4dm.api.deps import AdminAuth, MemoryServices, SessionId
from t4dm.consolidation.service import get_consolidation_service
from t4dm.core.config import get_settings

logger = logging.getLogger(__name__)

# Documentation directory (relative to package root)
DOCS_DIR = Path(__file__).parent.parent.parent.parent.parent / "docs"

router = APIRouter()


class HealthResponse(BaseModel):
    """Health check response."""

    status: str
    timestamp: str
    version: str
    session_id: str | None = None


class StatsResponse(BaseModel):
    """Memory statistics response."""

    session_id: str
    episodic: dict
    semantic: dict
    procedural: dict


class ConsolidationRequest(BaseModel):
    """Consolidation request parameters."""

    deep: bool = False


class ConsolidationResponse(BaseModel):
    """Consolidation response."""

    success: bool
    type: str
    results: dict


@router.get("/health", response_model=HealthResponse)
async def health_check(session_id: SessionId = None):
    """
    Check API health status.

    Returns service status and version information.
    """
    return HealthResponse(
        status="healthy",
        timestamp=datetime.utcnow().isoformat(),
        version="0.5.0",
        session_id=session_id,
    )


@router.get("/stats", response_model=StatsResponse)
async def get_stats(services: MemoryServices):
    """
    Get memory statistics for current session.

    Returns counts and metrics for all memory subsystems.
    """
    episodic = services["episodic"]
    semantic = services["semantic"]
    procedural = services["procedural"]

    return StatsResponse(
        session_id=services["session_id"],
        episodic={
            "total_episodes": await episodic.count() if hasattr(episodic, "count") else 0,
        },
        semantic={
            "total_entities": await semantic.count_entities() if hasattr(semantic, "count_entities") else 0,
            "total_relations": await semantic.count_relations() if hasattr(semantic, "count_relations") else 0,
        },
        procedural={
            "total_skills": await procedural.count() if hasattr(procedural, "count") else 0,
        },
    )


@router.post("/consolidate", response_model=ConsolidationResponse)
async def consolidate_memory(
    request: ConsolidationRequest,
    services: MemoryServices,
    _: AdminAuth,
):
    """
    Trigger memory consolidation.

    Requires admin authentication via X-Admin-Key header.

    Light consolidation: Duplicate detection, decay updates
    Deep consolidation: Clustering, entity extraction, skill building
    """
    try:
        consolidation = get_consolidation_service()
        consolidation._session_id = services["session_id"]

        if request.deep:
            results = await consolidation.deep_consolidate()
            consolidation_type = "deep"
        else:
            results = await consolidation.light_consolidate()
            consolidation_type = "light"

        return ConsolidationResponse(
            success=True,
            type=consolidation_type,
            results=results,
        )
    except Exception as e:
        logger.error(f"Consolidation failed: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Consolidation failed: {e!s}",
        )


def _sanitize_uri(uri: str) -> str:
    """
    Sanitize database URI to only show host (no credentials, port, or path).

    SEC-001: Prevents leaking sensitive connection details.
    """
    from urllib.parse import urlparse
    try:
        parsed = urlparse(uri)
        # Only return scheme and hostname, redact everything else
        if parsed.hostname:
            return f"{parsed.scheme}://{parsed.hostname}/***"
        return "***"
    except Exception:
        return "***"


@router.get("/session")
async def get_session_info(session_id: SessionId):
    """
    Get current session information.

    Returns configured session ID and storage backend.
    """
    settings = get_settings()
    return {
        "session_id": session_id,
        "configured_session": settings.session_id,
        "storage_backend": "t4dx_embedded",
    }


# Documentation endpoints
class DocInfo(BaseModel):
    """Documentation file info."""

    name: str
    path: str
    size_bytes: int
    category: str


class DocsListResponse(BaseModel):
    """List of available documentation."""

    docs: list[DocInfo]
    total: int


def _categorize_doc(name: str) -> str:
    """Categorize documentation by filename."""
    name_lower = name.lower()
    if name_lower in ("readme.md", "api.md", "sdk.md"):
        return "core"
    if "architecture" in name_lower or "design" in name_lower:
        return "architecture"
    if "bio" in name_lower or "neural" in name_lower or "hinton" in name_lower:
        return "bioinspired"
    if "hook" in name_lower:
        return "hooks"
    if "citation" in name_lower or "paper" in name_lower or "ieee" in name_lower:
        return "academic"
    if "deploy" in name_lower or "self_hosted" in name_lower:
        return "deployment"
    if "test" in name_lower or "qa" in name_lower:
        return "testing"
    if "plan" in name_lower or "roadmap" in name_lower:
        return "planning"
    return "other"


@router.get("/docs", response_model=DocsListResponse)
async def list_docs(_: AdminAuth, category: str | None = None):
    """
    List available documentation files.

    Requires admin authentication via X-Admin-Key header.

    Optionally filter by category:
    - core: README, API, SDK
    - architecture: System design documents
    - bioinspired: Neural/biological memory research
    - hooks: Hook system documentation
    - academic: Citation and paper-related
    - deployment: Installation and deployment
    - testing: QA and testing protocols
    - planning: Roadmaps and implementation plans
    - other: Miscellaneous
    """
    if not DOCS_DIR.exists():
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail="Documentation directory not found",
        )

    docs = []
    for doc_file in sorted(DOCS_DIR.glob("*.md")):
        doc_category = _categorize_doc(doc_file.name)
        if category and doc_category != category:
            continue

        docs.append(DocInfo(
            name=doc_file.stem,
            path=f"/api/v1/docs/{doc_file.stem}",
            size_bytes=doc_file.stat().st_size,
            category=doc_category,
        ))

    return DocsListResponse(docs=docs, total=len(docs))


@router.get("/docs/{doc_name}", response_class=PlainTextResponse)
async def get_doc(doc_name: str, _: AdminAuth):
    """
    Get documentation file content.

    Requires admin authentication via X-Admin-Key header.

    Returns raw markdown content for the specified document.
    Use the /docs endpoint to list available documents.
    """
    import re

    # SEC-002: Validate doc_name BEFORE any path operations to prevent traversal
    # Only allow alphanumeric, underscore, hyphen, and dot (for extension)
    if not re.match(r"^[a-zA-Z0-9_-]+(?:\.md)?$", doc_name):
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="Invalid document name. Only alphanumeric characters, underscores, and hyphens are allowed.",
        )

    # Reject any path separators or parent directory references
    if "/" in doc_name or "\\" in doc_name or ".." in doc_name:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="Invalid document name. Path separators not allowed.",
        )

    # Support with or without .md extension
    if not doc_name.endswith(".md"):
        doc_name = f"{doc_name}.md"

    doc_path = DOCS_DIR / doc_name

    if not doc_path.exists():
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"Document '{doc_name}' not found. Use GET /api/v1/docs to list available documents.",
        )

    # Defense in depth: ensure we're not traversing outside docs dir
    try:
        doc_path.resolve().relative_to(DOCS_DIR.resolve())
    except ValueError:
        raise HTTPException(
            status_code=status.HTTP_403_FORBIDDEN,
            detail="Access denied",
        )

    return doc_path.read_text(encoding="utf-8")
