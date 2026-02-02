"""
Configuration management for T4DM.

Uses pydantic-settings for environment variable support.
"""

import logging
import os
import re
from functools import lru_cache
from pathlib import Path
from typing import Any

from pydantic import BaseModel, Field, field_validator, model_validator
from pydantic_settings import BaseSettings, SettingsConfigDict

logger = logging.getLogger(__name__)


# Weak passwords to reject
WEAK_PASSWORDS: frozenset[str] = frozenset([
    "password", "Password", "PASSWORD", "passw0rd",
    "neo4j", "admin", "root", "test", "guest",
    "123456", "12345678", "123456789",
    "qwerty", "abc123", "letmein",
    "password123", "admin123", "welcome",
    "monkey", "dragon", "master",
    "default", "changeme", "secret",
])

# P2-SEC-M1: Common weak patterns to reject (case-insensitive)
# These patterns match specific weak password formats, not general strings
WEAK_PATTERNS: frozenset[str] = frozenset([
    r"^password\d*$",     # password, password1, password123
    r"^welcome\d*$",      # welcome, welcome1
    r"^admin\d*$",        # admin, admin123
    r"^qwerty\d*$",       # qwerty, qwerty123
    r"^letmein\d*$",      # letmein, letmein1
    r"^\d+$",             # all digits
    r"^[a-z]{1,6}\d{1,6}$",  # short lowercase+digits (abc123, test1)
    r"^test\d*$",         # test, test1, test123
    r"^user\d*$",         # user, user1, user123
    r"^guest\d*$",        # guest, guest1
    r"^default\d*$",      # default, default1
    r"^login\d*$",        # login, login1
    r"^changeme\d*$",     # changeme, changeme1
])


def validate_password_strength(password: str, field_name: str = "password") -> str:
    """
    Validate password meets security requirements.

    P2-SEC-M1: Enhanced password validation with:
    - 12 character minimum (was 8)
    - Pattern-based weak password detection
    - Requires 3 of 4 character classes (was 2)

    Args:
        password: Password to validate
        field_name: Name for error messages

    Returns:
        The validated password

    Raises:
        ValueError: If password is too weak
    """
    if not password:
        raise ValueError(
            f"{field_name} is required. "
            f"Set T4DM_{field_name.upper()} environment variable."
        )

    # P2-SEC-M1: Increased minimum length from 8 to 12
    if len(password) < 12:
        raise ValueError(
            f"{field_name} must be at least 12 characters (got {len(password)})"
        )

    # Check exact match against known weak passwords (case-insensitive)
    if password.lower() in WEAK_PASSWORDS:
        raise ValueError(
            f"{field_name} is too weak. Choose a stronger password. "
            f"Rejected: common/default passwords"
        )

    # P2-SEC-M1: Check against weak patterns (case-insensitive)
    password_lower = password.lower()
    for pattern in WEAK_PATTERNS:
        if re.match(pattern, password_lower):
            raise ValueError(
                f"{field_name} is too weak. Choose a stronger password. "
                f"Rejected: matches common weak pattern"
            )

    # P2-SEC-M1: Require 3 of 4 character classes (was 2)
    complexity = 0
    if re.search(r"[a-z]", password):
        complexity += 1
    if re.search(r"[A-Z]", password):
        complexity += 1
    if re.search(r"[0-9]", password):
        complexity += 1
    if re.search(r"[^a-zA-Z0-9]", password):
        complexity += 1

    if complexity < 3:
        raise ValueError(
            f"{field_name} needs more complexity. "
            f"Include at least 3 of: uppercase, lowercase, digits, special characters"
        )

    return password


def mask_secret(value: str, visible_chars: int = 4) -> str:
    """
    Mask a secret value for logging.

    Args:
        value: The secret value to mask
        visible_chars: Number of characters to show at start

    Returns:
        Masked string (e.g., "pass***")
    """
    if not value or len(value) <= visible_chars:
        return "***"
    return value[:visible_chars] + "*" * (len(value) - visible_chars)


def check_file_permissions(
    path: Path,
    enforce: bool = False,
    auto_fix: bool = False,
) -> None:
    """
    Check and optionally enforce file permissions.

    P3-SEC-L1: Enhanced to optionally enforce secure permissions.

    Args:
        path: Path to the file to check
        enforce: If True, raise PermissionError if permissions are too loose
        auto_fix: If True, automatically fix permissions to 0o600

    Raises:
        PermissionError: If enforce=True and permissions are too permissive

    Warns if file has group or world permissions.
    """
    if not path.exists():
        return

    mode = path.stat().st_mode
    # Check if group or others have any permissions (0o077 = rwxrwxrwx with user bits masked)
    if mode & 0o077:
        msg = (
            f"Config file '{path}' has permissive permissions: {oct(mode)}. "
            f"Should be 600 or more restrictive."
        )

        if auto_fix:
            try:
                path.chmod(0o600)
                logger.info(f"Auto-fixed permissions for '{path}' to 0o600")
                return
            except OSError as e:
                logger.error(f"Failed to fix permissions for '{path}': {e}")
                if enforce:
                    raise PermissionError(msg) from e

        if enforce:
            raise PermissionError(msg)

        logger.warning(f"{msg} Consider running: chmod 600 {path}")


def load_secret_from_env(key: str, default: str | None = None) -> str | None:
    """
    Load a secret from environment, logging masked value.

    Args:
        key: Environment variable name
        default: Default value if not set

    Returns:
        Secret value or default
    """
    value = os.getenv(key, default)
    if value:
        logger.debug(f"Loaded {key}: {mask_secret(value)}")
    else:
        logger.debug(f"{key} not set")
    return value


def validate_weights(weights: dict[str, float], tolerance: float = 0.001) -> dict[str, float]:
    """
    Validate retrieval weights.

    Args:
        weights: Dict of weight names to values
        tolerance: Allowed deviation from 1.0 for sum

    Returns:
        Validated weights dict

    Raises:
        ValueError: If weights invalid
    """
    if not weights:
        return weights

    # Check all values in [0, 1]
    for name, val in weights.items():
        if not isinstance(val, (int, float)):
            raise ValueError(f"Weight '{name}' must be numeric, got {type(val).__name__}")
        if val < 0 or val > 1:
            raise ValueError(f"Weight '{name}' must be in [0, 1], got {val}")

    # Check sum equals 1.0
    total = sum(weights.values())
    if abs(total - 1.0) > tolerance:
        raise ValueError(
            f"Weights must sum to 1.0 (±{tolerance}), got {total}. "
            f"Weights: {weights}"
        )

    return weights


# ===================
# Bioinspired Configuration
# ===================

class DendriticConfig(BaseModel):
    """Dendritic neuron configuration."""
    input_dim: int = 1024
    hidden_dim: int = 512
    context_dim: int = 512
    coupling_strength: float = Field(default=0.5, ge=0.0, le=1.0)
    tau_dendrite: float = Field(default=10.0, ge=1.0, le=100.0)
    tau_soma: float = Field(default=15.0, ge=1.0, le=100.0)


class SparseEncoderConfig(BaseModel):
    """Sparse encoder configuration."""
    input_dim: int = 1024
    hidden_dim: int = 8192
    sparsity: float = Field(default=0.02, ge=0.01, le=0.10)
    use_kwta: bool = True
    lateral_inhibition: float = Field(default=0.2, ge=0.0, le=1.0)


class AttractorConfig(BaseModel):
    """Attractor network configuration."""
    dim: int = 8192
    settling_steps: int = Field(default=10, ge=1, le=100)
    step_size: float = Field(default=0.1, ge=0.01, le=1.0)
    noise_std: float = Field(default=0.01, ge=0.0, le=0.5)
    capacity_ratio: float = Field(default=0.138, ge=0.01, le=0.5)


class FastEpisodicConfig(BaseModel):
    """Fast episodic store configuration."""
    capacity: int = Field(default=10000, ge=100, le=100000)
    learning_rate: float = Field(default=0.1, ge=0.01, le=1.0)
    consolidation_threshold: float = Field(default=0.7, ge=0.0, le=1.0)
    eviction_strategy: str = "lru_salience"


class NeuromodGainsConfig(BaseModel):
    """Neuromodulator gain parameters for learning rate modulation."""
    rho_da: float = Field(default=1.0, ge=0.1, le=5.0, description="Dopamine gain")
    rho_ne: float = Field(default=1.0, ge=0.1, le=5.0, description="Norepinephrine gain")
    rho_ach_fast: float = Field(default=2.0, ge=0.1, le=10.0, description="ACh fast gain")
    rho_ach_slow: float = Field(default=0.2, ge=0.01, le=1.0, description="ACh slow gain")
    alpha_ne: float = Field(default=0.5, ge=0.0, le=1.0, description="NE sigmoid steepness")


class EligibilityConfig(BaseModel):
    """Eligibility trace configuration."""
    decay: float = Field(default=0.95, ge=0.5, le=0.999)
    tau_trace: float = Field(default=20.0, ge=1.0, le=100.0)
    a_plus: float = Field(default=0.005, ge=0.001, le=0.1)
    a_minus: float = Field(default=0.00525, ge=0.001, le=0.1)


class BioinspiredConfig(BaseModel):
    """
    Bioinspired neural memory configuration.

    Controls the biologically-inspired encoding and memory components:
    - Dendritic neurons with two-compartment model
    - Sparse encoding with k-WTA activation
    - Attractor network for pattern completion
    - Fast episodic store with one-shot learning
    - Neuromodulator-gated learning rates
    - Eligibility traces for credit assignment
    """
    enabled: bool = Field(
        default=False,
        description="Enable bioinspired encoding (experimental)"
    )
    dendritic: DendriticConfig = Field(default_factory=DendriticConfig)
    sparse_encoder: SparseEncoderConfig = Field(default_factory=SparseEncoderConfig)
    attractor: AttractorConfig = Field(default_factory=AttractorConfig)
    fast_episodic: FastEpisodicConfig = Field(default_factory=FastEpisodicConfig)
    neuromod_gains: NeuromodGainsConfig = Field(default_factory=NeuromodGainsConfig)
    eligibility: EligibilityConfig = Field(default_factory=EligibilityConfig)


class Settings(BaseSettings):
    """T4DM configuration with environment variable support."""

    model_config = SettingsConfigDict(
        env_prefix="T4DM_",
        env_file=".env",
        env_file_encoding="utf-8",
        extra="ignore",
    )

    # Environment Configuration
    environment: str = Field(
        default="development",
        description="Environment: development, staging, production, test",
    )

    # Session
    session_id: str = Field(
        default="default",
        description="Unique identifier for this Claude Code instance",
    )

    # P2-SEC-M4: API key authentication for all endpoints
    api_key: str | None = Field(
        default=None,
        description="API key required for all endpoints in production. "
                    "Generate with: openssl rand -hex 32. "
                    "Pass via X-API-Key header.",
    )
    api_key_required: bool = Field(
        default=False,
        description="Whether API key is required. Auto-enabled in production if api_key is set.",
    )

    # API-CRITICAL-001/002/003 FIX: Admin authentication
    admin_api_key: str | None = Field(
        default=None,
        description="API key for admin endpoints (config, consolidation, docs). "
                    "Required for production. Generate with: openssl rand -hex 32",
    )

    # Legacy configuration fields (removed - T4DX embedded storage used)
    # These fields are kept for backwards compatibility but are no longer used

    # Embedding Configuration
    embedding_model: str = Field(
        default="BAAI/bge-m3",
        description="Embedding model name (HuggingFace)",
    )
    embedding_dimension: int = Field(
        default=1024,
        description="Embedding vector dimension",
    )
    embedding_hybrid_enabled: bool = Field(
        default=True,
        description="Enable hybrid search (dense + sparse vectors)",
    )
    embedding_device: str = Field(
        default="cuda:0",
        description="Device for embedding inference (cuda:0 or cpu)",
    )
    embedding_use_fp16: bool = Field(
        default=True,
        description="Use FP16 for memory efficiency",
    )
    embedding_batch_size: int = Field(
        default=32,
        description="Batch size for embedding generation",
    )
    embedding_max_length: int = Field(
        default=512,
        description="Maximum token length for embedding",
    )
    embedding_cache_dir: str = Field(
        default="./models",
        description="Directory for model caching",
    )
    embedding_cache_size: int = Field(
        default=1000,
        ge=100,
        le=100000,
        description="Maximum embedding cache entries",
    )
    embedding_cache_ttl: int = Field(
        default=3600,
        ge=60,
        le=86400,
        description="Embedding cache TTL in seconds (1 hour default)",
    )

    # ===================
    # FSRS Parameters
    # ===================
    fsrs_decay_factor: float = Field(
        default=0.9,
        ge=0.1,
        le=1.0,
        description="Power law decay scaling (0.9 = slower decay than standard FSRS)",
    )
    fsrs_default_stability: float = Field(
        default=1.0,
        ge=0.1,
        le=10.0,
        description="Initial stability for new episodes (days)",
    )
    fsrs_retention_target: float = Field(
        default=0.9,
        ge=0.5,
        le=1.0,
        description="Target retention rate for FSRS",
    )
    fsrs_recency_decay: float = Field(
        default=0.1,
        ge=0.01,
        le=1.0,
        description="Exponential decay factor for episodic recency scoring",
    )

    # ===================
    # ACT-R Parameters
    # ===================
    actr_spreading_strength: float = Field(
        default=1.6,
        ge=0.1,
        le=5.0,
        description="Spreading activation strength parameter (S)",
    )
    actr_decay: float = Field(
        default=0.5,
        ge=0.1,
        le=1.0,
        description="Base-level activation decay parameter (d)",
    )
    actr_threshold: float = Field(
        default=0.0,
        description="ACT-R retrieval threshold",
    )
    actr_noise: float = Field(
        default=0.0,
        ge=0.0,
        le=1.0,
        description="Activation noise (0 = deterministic)",
    )

    # ===================
    # Spreading Activation Settings
    # ===================
    spreading_max_nodes: int = Field(
        default=1000,
        ge=10,
        le=10000,
        description="Maximum nodes in spreading activation (prevents graph explosion)",
    )
    spreading_max_neighbors: int = Field(
        default=50,
        ge=1,
        le=200,
        description="Maximum neighbors per node in spreading activation",
    )
    spreading_default_steps: int = Field(
        default=3,
        ge=1,
        le=5,
        description="Default propagation steps for spreading activation",
    )

    # ===================
    # Hebbian Parameters
    # ===================
    hebbian_learning_rate: float = Field(
        default=0.1,
        ge=0.01,
        le=0.5,
        description="Rate of weight increase on co-retrieval",
    )
    hebbian_decay_rate: float = Field(
        default=0.01,
        ge=0.001,
        le=0.1,
        description="Rate of weight decay for stale relationships",
    )
    hebbian_initial_weight: float = Field(
        default=0.1,
        ge=0.01,
        le=1.0,
        description="Initial weight for new relationships",
    )
    hebbian_min_weight: float = Field(
        default=0.01,
        ge=0.001,
        le=0.1,
        description="Minimum weight before relationship is pruned",
    )
    hebbian_stale_days: int = Field(
        default=30,
        ge=1,
        le=365,
        description="Days without access before relationship is considered stale",
    )

    # ===================
    # HDBSCAN Parameters
    # ===================
    hdbscan_min_cluster_size: int = Field(
        default=3,
        ge=2,
        le=100,
        description="Minimum episodes to form a cluster",
    )
    hdbscan_min_samples: int | None = Field(
        default=None,
        description="Core point threshold (None = min_cluster_size)",
    )
    hdbscan_metric: str = Field(
        default="cosine",
        description="Distance metric for clustering (cosine, euclidean, manhattan)",
    )
    hdbscan_max_samples: int = Field(
        default=5000,
        ge=100,
        le=50000,
        description="Maximum samples for HDBSCAN clustering (memory limit)",
    )

    # ===================
    # Consolidation Parameters
    # ===================
    consolidation_min_similarity: float = Field(
        default=0.75,
        ge=0.5,
        le=1.0,
        description="Minimum similarity for episode clustering",
    )
    consolidation_min_occurrences: int = Field(
        default=3,
        ge=2,
        le=10,
        description="Minimum occurrences for consolidation",
    )
    consolidation_skill_similarity: float = Field(
        default=0.85,
        ge=0.5,
        le=1.0,
        description="Minimum similarity for procedure merging",
    )

    # ===================
    # Automatic Consolidation (P3.3)
    # ===================
    auto_consolidation_enabled: bool = Field(
        default=True,
        description="Enable automatic consolidation triggering",
    )
    auto_consolidation_interval_hours: float = Field(
        default=8.0,
        ge=1.0,
        le=168.0,
        description="Hours between automatic consolidation runs (time-based trigger)",
    )
    auto_consolidation_memory_threshold: int = Field(
        default=100,
        ge=10,
        le=10000,
        description="New memory count to trigger consolidation (load-based trigger)",
    )
    auto_consolidation_check_interval_seconds: float = Field(
        default=300.0,
        ge=10.0,
        le=3600.0,
        description="Seconds between scheduler checks (default: 5 minutes)",
    )
    auto_consolidation_type: str = Field(
        default="light",
        description="Type of consolidation to run automatically (light, deep, skill, all)",
    )

    # ===================
    # Interleaved Replay (P3.4 - CLS Theory)
    # ===================
    replay_recent_ratio: float = Field(
        default=0.6,
        ge=0.0,
        le=1.0,
        description="Ratio of recent vs older memories in replay batch (0.6 = 60% recent)",
    )
    replay_batch_size: int = Field(
        default=100,
        ge=10,
        le=1000,
        description="Total batch size for interleaved replay",
    )
    replay_recent_hours: int = Field(
        default=24,
        ge=1,
        le=168,
        description="Hours to consider for 'recent' memories in replay",
    )
    replay_interleave_enabled: bool = Field(
        default=True,
        description="Enable CLS-style interleaved replay (mix recent and older memories)",
    )

    # ===================
    # Sleep Consolidation Timing (P2.5)
    # ===================
    replay_delay_ms: int = Field(
        default=500,
        ge=10,
        le=2000,
        description="Delay between memory replays in milliseconds (500ms = ~2Hz, biologically accurate)",
    )

    # ===================
    # EWC (P3.5 - Elastic Weight Consolidation)
    # ===================
    ewc_enabled: bool = Field(
        default=False,
        description="Enable EWC regularization for continual learning",
    )
    ewc_lambda: float = Field(
        default=1000.0,
        ge=1.0,
        le=100000.0,
        description="EWC regularization strength (higher = more protection of old knowledge)",
    )
    ewc_online: bool = Field(
        default=True,
        description="Use online EWC (exponential moving average of Fisher information)",
    )
    ewc_gamma: float = Field(
        default=0.95,
        ge=0.0,
        le=1.0,
        description="Decay factor for online EWC (0.95 = slow decay, 0.5 = fast decay)",
    )
    ewc_consolidation_interval: int = Field(
        default=50,
        ge=1,
        le=1000,
        description="Training steps between EWC consolidations",
    )
    ewc_fisher_samples: int = Field(
        default=200,
        ge=10,
        le=1000,
        description="Number of samples for Fisher information estimation",
    )

    # ===================
    # Episodic Retrieval Weights
    # ===================
    episodic_weight_semantic: float = Field(
        default=0.4,
        ge=0.0,
        le=1.0,
        description="Weight for semantic similarity in episodic retrieval",
    )
    episodic_weight_recency: float = Field(
        default=0.25,
        ge=0.0,
        le=1.0,
        description="Weight for recency in episodic retrieval",
    )
    episodic_weight_outcome: float = Field(
        default=0.2,
        ge=0.0,
        le=1.0,
        description="Weight for outcome in episodic retrieval",
    )
    episodic_weight_importance: float = Field(
        default=0.15,
        ge=0.0,
        le=1.0,
        description="Weight for importance in episodic retrieval",
    )

    # ===================
    # Semantic Retrieval Weights
    # ===================
    semantic_weight_similarity: float = Field(
        default=0.4,
        ge=0.0,
        le=1.0,
        description="Weight for embedding similarity in semantic retrieval",
    )
    semantic_weight_activation: float = Field(
        default=0.35,
        ge=0.0,
        le=1.0,
        description="Weight for ACT-R activation in semantic retrieval",
    )
    semantic_weight_retrievability: float = Field(
        default=0.25,
        ge=0.0,
        le=1.0,
        description="Weight for FSRS retrievability in semantic retrieval",
    )

    # ===================
    # Procedural Retrieval Weights
    # ===================
    procedural_weight_similarity: float = Field(
        default=0.6,
        ge=0.0,
        le=1.0,
        description="Weight for task similarity in procedural retrieval",
    )
    procedural_weight_success: float = Field(
        default=0.3,
        ge=0.0,
        le=1.0,
        description="Weight for success rate in procedural retrieval",
    )
    procedural_weight_experience: float = Field(
        default=0.1,
        ge=0.0,
        le=1.0,
        description="Weight for execution experience in procedural retrieval",
    )
    procedural_dopamine_enabled: bool = Field(
        default=False,
        description="Enable dopamine RPE modulation for procedural memory updates",
    )

    # Deprecated - kept for backward compatibility
    retrieval_semantic_weight: float = Field(
        default=0.4,
        description="DEPRECATED: Use episodic_weight_semantic",
    )
    retrieval_recency_weight: float = Field(
        default=0.25,
        description="DEPRECATED: Use episodic_weight_recency",
    )
    retrieval_outcome_weight: float = Field(
        default=0.2,
        description="DEPRECATED: Use episodic_weight_outcome",
    )
    retrieval_importance_weight: float = Field(
        default=0.15,
        description="DEPRECATED: Use episodic_weight_importance",
    )

    # OpenTelemetry Tracing
    otel_enabled: bool = Field(
        default=False,
        description="Enable OpenTelemetry distributed tracing",
    )
    otel_endpoint: str = Field(
        default="http://localhost:4317",
        description="OTLP gRPC endpoint for trace export",
    )
    otel_insecure: bool = Field(
        default=True,
        description="Use insecure gRPC for OTLP (set False for production TLS)",
    )
    otel_cert_file: str | None = Field(
        default=None,
        description="Path to TLS certificate file for OTLP (PEM format)",
    )
    otel_headers: dict[str, str] | None = Field(
        default=None,
        description="Additional headers for OTLP exporter (e.g., auth tokens)",
    )
    otel_service_name: str = Field(
        default="world-weaver",
        description="Service name for distributed traces",
    )
    otel_batch_delay_ms: int = Field(
        default=5000,
        ge=100,
        le=60000,
        description="Batch export delay in milliseconds",
    )
    otel_max_export_batch_size: int = Field(
        default=512,
        ge=1,
        le=10000,
        description="Maximum batch size for export",
    )
    otel_console: bool = Field(
        default=False,
        description="Export traces to console for debugging",
    )

    # Entity Extraction Settings
    auto_extraction_enabled: bool = Field(
        default=True,
        description="Automatically extract entities from new episodes",
    )
    extraction_confidence_threshold: float = Field(
        default=0.7,
        ge=0.0,
        le=1.0,
        description="Minimum confidence to create entity (0-1)",
    )
    extraction_use_llm: bool = Field(
        default=False,
        description="Use LLM for entity extraction (default: regex only)",
    )
    extraction_llm_model: str = Field(
        default="gpt-4o-mini",
        description="OpenAI model for entity extraction",
    )
    extraction_llm_timeout: float = Field(
        default=30.0,
        ge=5.0,
        le=120.0,
        description="LLM API timeout in seconds",
    )
    extraction_llm_max_text: int = Field(
        default=4000,
        ge=100,
        le=16000,
        description="Max text length to send to LLM",
    )
    extraction_batch_size: int = Field(
        default=10,
        ge=1,
        le=100,
        description="Batch size for background extraction jobs",
    )

    # Batch operation parameters
    batch_max_concurrency: int = Field(
        default=10,
        ge=1,
        le=100,
        description="Maximum concurrent batch operations",
    )

    # ===================
    # Bioinspired Configuration
    # ===================
    bioinspired: BioinspiredConfig = Field(
        default_factory=BioinspiredConfig,
        description="Bioinspired neural memory configuration (experimental)",
    )

    # API Server Configuration
    api_host: str = Field(
        default="0.0.0.0",
        description="API server host address",
    )
    api_port: int = Field(
        default=8765,
        ge=1,
        le=65535,
        description="API server port",
    )
    api_workers: int = Field(
        default=1,
        ge=1,
        le=32,
        description="Number of API worker processes",
    )
    cors_allowed_origins: list[str] = Field(
        default_factory=lambda: ["http://localhost:3000", "http://localhost:8080"],
        description="Allowed CORS origins (use specific origins in production)",
    )
    cors_allowed_headers: list[str] = Field(
        default_factory=lambda: [
            "Authorization",
            "Content-Type",
            "X-Session-ID",
            "X-Request-ID",
        ],
        description="Allowed CORS headers (restrict in production)",
    )

    @field_validator("cors_allowed_origins")
    @classmethod
    def validate_cors_origins(cls, v: list[str]) -> list[str]:
        """
        Validate CORS origins.

        API-HIGH-001 FIX: Reject wildcards in production
        API-HIGH-002 FIX: Validate origin schemes (require HTTPS in production)
        """
        from urllib.parse import urlparse

        env = os.getenv("T4DM_ENVIRONMENT", "development")

        validated_origins = []
        for origin in v:
            # API-HIGH-001: Reject wildcards in production
            if origin == "*":
                if env == "production":
                    raise ValueError(
                        "Wildcard CORS origin '*' not allowed in production. "
                        "Specify explicit origins."
                    )
                validated_origins.append(origin)
                continue

            if "*" in origin:
                logger.warning(
                    f"CORS origin '{origin}' contains wildcard - verify this is intended"
                )
                validated_origins.append(origin)
                continue

            # API-HIGH-002 FIX: Validate URL scheme
            try:
                parsed = urlparse(origin)
                if not parsed.scheme:
                    raise ValueError(f"CORS origin '{origin}' missing scheme (http/https)")

                if not parsed.netloc:
                    raise ValueError(f"CORS origin '{origin}' missing host")

                # Require HTTPS in production (except localhost)
                if env == "production":
                    is_localhost = parsed.netloc.startswith(("localhost", "127.0.0.1", "[::1]"))
                    if parsed.scheme != "https" and not is_localhost:
                        raise ValueError(
                            f"CORS origin '{origin}' must use HTTPS in production. "
                            "Only localhost exempted for local development."
                        )

                validated_origins.append(origin)
            except ValueError:
                raise
            except Exception as e:
                raise ValueError(f"Invalid CORS origin '{origin}': {e}")

        return validated_origins

    @field_validator("environment")
    @classmethod
    def validate_environment(cls, v: str) -> str:
        """Validate environment value."""
        valid = {"development", "staging", "production", "test"}
        if v.lower() not in valid:
            raise ValueError(f"environment must be one of: {valid}")
        return v.lower()


    @model_validator(mode="after")
    def validate_all_weights_sum(self) -> "Settings":
        """Validate that all retrieval weight groups sum to 1.0."""
        # Episodic weights
        episodic_weights = {
            "semantic": self.episodic_weight_semantic,
            "recency": self.episodic_weight_recency,
            "outcome": self.episodic_weight_outcome,
            "importance": self.episodic_weight_importance,
        }
        try:
            validate_weights(episodic_weights)
        except ValueError as e:
            raise ValueError(f"Episodic weights validation failed: {e}")

        # Semantic weights
        semantic_weights = {
            "similarity": self.semantic_weight_similarity,
            "activation": self.semantic_weight_activation,
            "retrievability": self.semantic_weight_retrievability,
        }
        try:
            validate_weights(semantic_weights)
        except ValueError as e:
            raise ValueError(f"Semantic weights validation failed: {e}")

        # Procedural weights
        procedural_weights = {
            "similarity": self.procedural_weight_similarity,
            "success": self.procedural_weight_success,
            "experience": self.procedural_weight_experience,
        }
        try:
            validate_weights(procedural_weights)
        except ValueError as e:
            raise ValueError(f"Procedural weights validation failed: {e}")

        return self

    @model_validator(mode="after")
    def validate_production_settings(self) -> "Settings":
        """Additional validation for production environment."""
        env = os.getenv("T4DM_ENVIRONMENT", "development")

        if env == "production":
            issues = []

            if self.otel_insecure:
                issues.append("OTEL insecure mode enabled in production")

            if issues:
                logger.warning(
                    f"Production security concerns: {'; '.join(issues)}"
                )

        return self

    @classmethod
    def validate_permissions(
        cls,
        env_file: Path | None = None,
        enforce: bool | None = None,
        auto_fix: bool = False,
    ) -> None:
        """
        Validate file permissions for config files.

        P3-SEC-L1: Enhanced to enforce secure permissions in production.

        Args:
            env_file: Path to .env file (defaults to .env in current directory)
            enforce: If True, raise PermissionError on insecure permissions.
                     Defaults to True in production, False otherwise.
            auto_fix: If True, automatically fix permissions to 0o600

        Raises:
            PermissionError: If enforce=True and permissions are too permissive
        """
        if env_file is None:
            env_file = Path(".env")

        # Default: enforce in production
        if enforce is None:
            env = os.getenv("T4DM_ENVIRONMENT", "development")
            enforce = (env == "production")

        check_file_permissions(env_file, enforce=enforce, auto_fix=auto_fix)

    def _load_with_masking(self, field_name: str) -> str:
        """
        Load a secret field and log masked value.

        Args:
            field_name: Name of the field to load

        Returns:
            Field value with masked logging
        """
        value = getattr(self, field_name)
        logger.debug(f"{field_name}: {mask_secret(value)}")
        return value

    def log_safe_config(self) -> dict[str, Any]:
        """Return config dict with secrets masked for logging."""
        config = self.model_dump()

        # Mask all password/secret fields
        secret_fields = [
            "api_key",
            "admin_api_key",
        ]

        for field in secret_fields:
            if config.get(field):
                config[field] = mask_secret(config[field])

        return config

    def log_config_info(self) -> None:
        """Log configuration with secrets masked."""
        logger.info("T4DM Configuration:")
        logger.info(f"  Environment: {self.environment}")
        logger.info(f"  Session ID: {self.session_id}")
        logger.info(f"  Embedding Model: {self.embedding_model}")
        logger.info(f"  Embedding Device: {self.embedding_device}")
        logger.info("  Retrieval Weights:")
        logger.info(f"    Semantic: {self.retrieval_semantic_weight}")
        logger.info(f"    Recency: {self.retrieval_recency_weight}")
        logger.info(f"    Outcome: {self.retrieval_outcome_weight}")
        logger.info(f"    Importance: {self.retrieval_importance_weight}")


def _find_config_file() -> Path | None:
    """
    Find YAML config file in standard locations.

    Search order:
    1. T4DM_CONFIG_FILE environment variable
    2. ./ww.yaml (current directory)
    3. ~/.ww/config.yaml (user home)
    4. /etc/ww/config.yaml (system-wide)

    Returns:
        Path to config file if found, None otherwise
    """
    # Check T4DM_CONFIG_FILE environment variable
    env_config = os.getenv("T4DM_CONFIG_FILE")
    if env_config:
        path = Path(env_config).expanduser()
        if path.exists():
            return path
        logger.warning(f"Config file from T4DM_CONFIG_FILE not found: {path}")

    # Check standard locations
    search_paths = [
        Path("t4dm.yaml"),
        Path("t4dm.yml"),
        Path(".ww.yaml"),
        Path.home() / ".ww" / "config.yaml",
        Path.home() / ".ww" / "config.yml",
        Path("/etc/ww/config.yaml"),
    ]

    for path in search_paths:
        if path.exists():
            logger.debug(f"Found config file: {path}")
            return path

    return None


def _load_yaml_config(path: Path) -> dict:
    """
    Load configuration from YAML file.

    Args:
        path: Path to YAML config file

    Returns:
        Dictionary of configuration values

    Raises:
        ValueError: If YAML file is invalid
    """
    try:
        import yaml
    except ImportError:
        logger.warning("pyyaml not installed, cannot load YAML config")
        return {}

    try:
        with open(path) as f:
            config = yaml.safe_load(f) or {}

        if not isinstance(config, dict):
            raise ValueError(f"Config file must contain a dictionary, got {type(config).__name__}")

        logger.info(f"Loaded configuration from: {path}")
        return config
    except yaml.YAMLError as e:
        raise ValueError(f"Invalid YAML in config file {path}: {e}")


def _merge_config_with_env(yaml_config: dict) -> dict:
    """
    Merge YAML config with environment variables.

    Environment variables take precedence over YAML values.
    Keys are converted from snake_case to T4DM_UPPER_CASE format.

    Args:
        yaml_config: Configuration dictionary from YAML

    Returns:
        Merged configuration with env vars taking precedence
    """
    result = dict(yaml_config)

    # Check for environment variable overrides
    for key in yaml_config:
        env_key = f"T4DM_{key.upper()}"
        env_value = os.getenv(env_key)
        if env_value is not None:
            result[key] = env_value
            logger.debug(f"Override {key} from environment ({env_key})")

    return result


def load_settings_from_yaml(config_path: Path | str | None = None) -> Settings:
    """
    Load Settings from YAML file with environment variable overrides.

    This function allows loading configuration from a YAML file while
    still supporting environment variable overrides. Env vars always
    take precedence over YAML values.

    Args:
        config_path: Path to YAML config file. If None, searches standard locations.

    Returns:
        Settings instance with values from YAML and env vars

    Example YAML config:
        ```yaml
        # ~/.ww/config.yaml
        session_id: my-session
        environment: development

        # Embedding
        embedding_model: bge-m3
        embedding_device: cuda
        ```
    """
    # Find config file
    if config_path:
        path = Path(config_path).expanduser()
        if not path.exists():
            raise FileNotFoundError(f"Config file not found: {path}")
    else:
        path = _find_config_file()

    # Load YAML config
    yaml_config = {}
    if path:
        yaml_config = _load_yaml_config(path)

    # Filter out YAML values that have env var overrides
    # This ensures env vars take precedence over YAML
    filtered_config = {}
    for key, value in yaml_config.items():
        env_key = f"T4DM_{key.upper()}"
        if os.getenv(env_key) is None:
            filtered_config[key] = value
        else:
            logger.debug(f"Skipping YAML key '{key}' - overridden by {env_key}")

    # Create settings with filtered YAML values
    # pydantic-settings will read env vars for the remaining fields
    return Settings(**filtered_config)


@lru_cache
def get_settings() -> Settings:
    """
    Get cached settings instance.

    Settings are loaded from (in order of precedence):
    1. Environment variables (T4DM_* prefix)
    2. YAML config file (if found)
    3. Default values

    Config file search order:
    - T4DM_CONFIG_FILE environment variable
    - ./ww.yaml (current directory)
    - ~/.ww/config.yaml (user home)
    - /etc/ww/config.yaml (system-wide)

    Returns:
        Cached Settings instance
    """
    config_path = _find_config_file()

    if config_path:
        yaml_config = _load_yaml_config(config_path)

        # Filter out YAML values that have env var overrides
        filtered_config = {}
        for key, value in yaml_config.items():
            env_key = f"T4DM_{key.upper()}"
            if os.getenv(env_key) is None:
                filtered_config[key] = value

        return Settings(**filtered_config)

    return Settings()


def reset_settings() -> None:
    """
    Clear the cached settings, forcing reload on next get_settings() call.

    Useful for:
    - Testing with different configurations
    - Hot-reloading configuration changes
    - Resetting to default settings
    """
    get_settings.cache_clear()
