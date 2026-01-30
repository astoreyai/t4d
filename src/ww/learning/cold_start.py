"""
Cold Start Priming System for World Weaver.

This module addresses the chicken-and-egg problem of learned memory systems:
- The gate needs examples to learn what to store
- But without storing, we have no examples to learn from

Solution: Multi-strategy cold start priming that gradually transitions
from heuristic-driven to learning-driven behavior.

Strategies:
1. Population Priors: Initialize weights from "average user" preferences
2. Context Loading: Extract priors from CLAUDE.md, README, project structure
3. Heuristic Blending: Linearly interpolate heuristic -> learned
4. Optimistic Initialization: Assume memories are useful, update from negatives
5. Exploration Bonus: Boost Thompson sampling variance during cold start

Key insight from Hinton: The brain doesn't start from scratch - it has
genetic priors (population knowledge) and developmental period where
heuristics dominate before learned patterns take over.

Usage:
    manager = ColdStartManager(gate, orchestra, persister)

    # At session start
    await manager.initialize_session(
        working_dir="/home/user/project",
        claude_md_path="~/.claude/CLAUDE.md"
    )

    # Periodically during session
    manager.checkpoint()

    # At session end
    manager.finalize_session(session_outcome=0.8)
"""

from __future__ import annotations

import logging
import re
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from typing import TYPE_CHECKING, Any

import numpy as np

if TYPE_CHECKING:
    from ww.core.learned_gate import LearnedMemoryGate
    from ww.learning.neuromodulators import NeuromodulatorOrchestra
    from ww.learning.persistence import StatePersister

logger = logging.getLogger(__name__)


@dataclass
class ContextSignals:
    """Signals extracted from project context for priming."""

    # Project identification
    project_name: str | None = None
    project_type: str | None = None  # "python", "typescript", "rust", etc.

    # Importance signals
    is_test_file: bool = False
    is_main_file: bool = False
    is_config_file: bool = False
    is_documentation: bool = False

    # Project keywords for content relevance
    keywords: list[str] = field(default_factory=list)

    # Priority signals from CLAUDE.md
    priority_patterns: list[str] = field(default_factory=list)

    # Recent activity patterns
    active_files: list[str] = field(default_factory=list)

    def to_feature_bias(self, feature_dim: int) -> np.ndarray:
        """
        Convert context signals to feature bias vector.

        This biases the learned gate's predictions toward storing
        content that matches the project context.
        """
        bias = np.zeros(feature_dim, dtype=np.float32)

        # Project type affects which content is likely valuable
        type_boosts = {
            "python": 0.1,  # Python projects - bias toward code
            "typescript": 0.1,
            "rust": 0.1,
            "latex": 0.15,  # Academic - bias toward documentation
            "markdown": 0.05,
        }

        if self.project_type and self.project_type in type_boosts:
            # Small global bias (affects all features equally)
            bias += type_boosts[self.project_type]

        return bias


class ContextLoader:
    """
    Loads context from project files to prime cold start.

    Extracts signals from:
    - CLAUDE.md (user preferences and patterns)
    - README.md (project overview)
    - Directory structure (project type, active areas)
    - Recent git activity (what's being worked on)
    """

    # File patterns for project type detection
    TYPE_INDICATORS = {
        "python": ["*.py", "setup.py", "pyproject.toml", "requirements.txt"],
        "typescript": ["*.ts", "*.tsx", "package.json", "tsconfig.json"],
        "javascript": ["*.js", "*.jsx", "package.json"],
        "rust": ["*.rs", "Cargo.toml"],
        "go": ["*.go", "go.mod"],
        "latex": ["*.tex", "*.bib"],
        "markdown": ["*.md"],
    }

    def __init__(self, working_dir: str | None = None):
        """
        Initialize context loader.

        Args:
            working_dir: Base directory for project (default: cwd)
        """
        self.working_dir = Path(working_dir or ".").resolve()

    def load_context(
        self,
        claude_md_path: str | None = None,
        include_git: bool = True
    ) -> ContextSignals:
        """
        Load context signals from available sources.

        Args:
            claude_md_path: Path to CLAUDE.md (default: search)
            include_git: Whether to include git activity

        Returns:
            Extracted context signals
        """
        signals = ContextSignals()

        # Extract project name
        signals.project_name = self.working_dir.name

        # Detect project type
        signals.project_type = self._detect_project_type()

        # Load CLAUDE.md if available
        if claude_md_path:
            claude_path = Path(claude_md_path).expanduser()
        else:
            claude_path = self._find_claude_md()

        if claude_path and claude_path.exists():
            self._extract_claude_md_signals(claude_path, signals)

        # Load README for keywords
        readme_path = self._find_readme()
        if readme_path and readme_path.exists():
            self._extract_readme_signals(readme_path, signals)

        # Load git activity
        if include_git:
            self._extract_git_signals(signals)

        logger.info(
            f"Loaded context: project={signals.project_name}, "
            f"type={signals.project_type}, "
            f"keywords={len(signals.keywords)}"
        )

        return signals

    def _detect_project_type(self) -> str | None:
        """Detect project type from file patterns."""
        for proj_type, patterns in self.TYPE_INDICATORS.items():
            for pattern in patterns:
                if list(self.working_dir.glob(pattern)):
                    return proj_type
                # Also check src/ subdirectory
                if list(self.working_dir.glob(f"src/{pattern}")):
                    return proj_type
        return None

    def _find_claude_md(self) -> Path | None:
        """Find CLAUDE.md in standard locations."""
        candidates = [
            self.working_dir / "CLAUDE.md",
            self.working_dir / ".claude" / "CLAUDE.md",
            Path.home() / ".claude" / "CLAUDE.md",
        ]
        for path in candidates:
            if path.exists():
                return path
        return None

    def _find_readme(self) -> Path | None:
        """Find README file."""
        candidates = ["README.md", "README.rst", "README.txt", "README"]
        for name in candidates:
            path = self.working_dir / name
            if path.exists():
                return path
        return None

    def _extract_claude_md_signals(
        self,
        path: Path,
        signals: ContextSignals
    ) -> None:
        """Extract signals from CLAUDE.md."""
        try:
            content = path.read_text(encoding="utf-8")

            # Extract priority patterns (lines with "IMPORTANT" or "CRITICAL")
            for line in content.split("\n"):
                line_lower = line.lower()
                if "important" in line_lower or "critical" in line_lower:
                    signals.priority_patterns.append(line.strip())

            # Extract keywords from headers
            headers = re.findall(r"^#+\s+(.+)$", content, re.MULTILINE)
            signals.keywords.extend(headers[:20])  # Limit

            # Extract project references
            projects = re.findall(r"\*\*Path\*\*:\s*`([^`]+)`", content)
            for proj in projects:
                proj_name = Path(proj).name
                if proj_name not in signals.keywords:
                    signals.keywords.append(proj_name)

        except Exception as e:
            logger.debug(f"Failed to parse CLAUDE.md: {e}")

    def _extract_readme_signals(
        self,
        path: Path,
        signals: ContextSignals
    ) -> None:
        """Extract signals from README."""
        try:
            content = path.read_text(encoding="utf-8")

            # Extract headers as keywords
            headers = re.findall(r"^#+\s+(.+)$", content, re.MULTILINE)
            signals.keywords.extend(headers[:10])

            # Extract code block languages
            languages = re.findall(r"```(\w+)", content)
            signals.keywords.extend(set(languages))

        except Exception as e:
            logger.debug(f"Failed to parse README: {e}")

    def _extract_git_signals(self, signals: ContextSignals) -> None:
        """Extract signals from git activity."""
        try:
            import subprocess

            # Get recently modified files
            result = subprocess.run(
                ["git", "diff", "--name-only", "HEAD~10..HEAD"],
                check=False, cwd=self.working_dir,
                capture_output=True,
                text=True,
                timeout=5,
            )

            if result.returncode == 0:
                files = result.stdout.strip().split("\n")
                signals.active_files = [f for f in files if f][:20]

        except Exception as e:
            logger.debug(f"Failed to get git activity: {e}")


@dataclass
class PopulationPrior:
    """
    Population-level prior for cold start.

    These represent "average user" preferences learned from
    aggregate patterns across many users/sessions.

    In practice, these are hand-tuned based on observed patterns:
    - Code files are more often useful than configs
    - Main logic files more useful than tests (usually)
    - Recent files more useful than old
    - Explicitly marked "important" is valuable
    """

    # Weight adjustments by content type
    content_type_weights: dict[str, float] = field(default_factory=lambda: {
        "code": 0.3,        # Source code - often valuable
        "test": 0.1,        # Tests - sometimes valuable
        "config": 0.05,     # Configs - rarely need to remember
        "docs": 0.2,        # Documentation - moderately valuable
        "data": 0.15,       # Data files - context-dependent
    })

    # Temporal decay parameters
    recency_half_life_hours: float = 24.0  # How fast old memories lose value

    # Neuromodulator priors (baseline levels)
    dopamine_baseline: float = 0.5   # Neutral expectation
    ne_baseline_gain: float = 1.0    # Normal arousal
    ach_encoding_bias: float = 0.55  # Slight encoding preference

    def get_prior_weights(self, feature_dim: int) -> np.ndarray:
        """
        Get population prior as weight initialization.

        This biases the learned gate toward storing patterns that
        have historically been valuable across users.
        """
        weights = np.zeros(feature_dim, dtype=np.float32)

        # Content embedding section (first 1024 dims)
        # Small positive bias - assume content is useful until proven otherwise
        weights[:1024] = 0.001

        # Context section (next 64 dims)
        # These encode project/task - neutral prior
        weights[1024:1088] = 0.0

        # Neuromodulator section (next 7 dims)
        neuro_start = 1088
        weights[neuro_start + 0] = 0.3   # DA RPE: positive surprise = store
        weights[neuro_start + 1] = 0.2   # NE gain: high arousal = store
        weights[neuro_start + 2] = 0.1   # 5-HT mood: positive mood = store
        weights[neuro_start + 3] = 0.4   # ACh encoding: encoding mode = store
        weights[neuro_start + 4] = 0.0   # ACh balanced: neutral
        weights[neuro_start + 5] = -0.3  # ACh retrieval: retrieval mode = skip
        weights[neuro_start + 6] = 0.05  # Inhibition: slight sparsity preference

        return weights


class ColdStartManager:
    """
    Orchestrates cold start priming for learned memory systems.

    Lifecycle:
    1. initialize_session(): Load persisted state, apply context priors
    2. (normal operation with learned gate)
    3. checkpoint(): Periodically save state
    4. finalize_session(): Save final state with session outcome

    The manager handles:
    - Loading persisted state from previous sessions
    - Applying population and context priors when no state exists
    - Gradual transition from heuristic to learned behavior
    - Periodic checkpointing for crash recovery
    """

    def __init__(
        self,
        gate: LearnedMemoryGate,
        orchestra: NeuromodulatorOrchestra | None = None,
        persister: StatePersister | None = None,
        population_prior: PopulationPrior | None = None
    ):
        """
        Initialize cold start manager.

        Args:
            gate: The learned memory gate to manage
            orchestra: Neuromodulator orchestra (optional)
            persister: State persister (created if None)
            population_prior: Population-level priors (default if None)
        """
        from ww.learning.persistence import StatePersister

        self.gate = gate
        self.orchestra = orchestra
        self.persister = persister or StatePersister()
        self.population_prior = population_prior or PopulationPrior()

        self._context_loader: ContextLoader | None = None
        self._context_signals: ContextSignals | None = None
        self._session_start: datetime | None = None
        self._initialized = False

    def initialize_session(
        self,
        working_dir: str | None = None,
        claude_md_path: str | None = None,
        force_cold_start: bool = False
    ) -> dict[str, Any]:
        """
        Initialize session with appropriate priming.

        Priority order:
        1. Persisted state (if available and not force_cold_start)
        2. Context-informed priors (from CLAUDE.md, project)
        3. Population priors (fallback)

        Args:
            working_dir: Working directory for context loading
            claude_md_path: Path to CLAUDE.md
            force_cold_start: If True, ignore persisted state

        Returns:
            Dict with initialization info
        """
        self._session_start = datetime.now()
        result = {
            "strategy": None,
            "n_observations": 0,
            "context_loaded": False,
            "persisted_state_loaded": False,
        }

        # Try to load persisted state
        if not force_cold_start:
            restored = self.persister.restore_gate(self.gate)
            if restored:
                result["strategy"] = "persisted"
                result["persisted_state_loaded"] = True
                result["n_observations"] = self.gate.n_observations
                logger.info(
                    f"Restored from persisted state: "
                    f"n_obs={self.gate.n_observations}"
                )
                self._initialized = True
                return result

        # Load context for priming
        self._context_loader = ContextLoader(working_dir)
        self._context_signals = self._context_loader.load_context(claude_md_path)
        result["context_loaded"] = True

        # Apply priors to gate
        if self.gate.n_observations == 0:
            self._apply_population_priors()
            self._apply_context_priors()
            result["strategy"] = "cold_start_priors"
            logger.info("Applied cold start priors (population + context)")
        else:
            result["strategy"] = "warm_start"
            logger.info(f"Warm start with n_obs={self.gate.n_observations}")

        result["n_observations"] = self.gate.n_observations
        self._initialized = True
        return result

    def _apply_population_priors(self) -> None:
        """Apply population-level priors to gate weights."""
        prior_weights = self.population_prior.get_prior_weights(
            self.gate.feature_dim
        )

        # Blend with current initialization (don't completely override)
        blend_factor = 0.3  # 30% population prior, 70% default init
        self.gate.μ = (1 - blend_factor) * self.gate.μ + blend_factor * prior_weights

        logger.debug("Applied population priors to gate weights")

    def _apply_context_priors(self) -> None:
        """Apply context-specific priors from project."""
        if self._context_signals is None:
            return

        context_bias = self._context_signals.to_feature_bias(self.gate.feature_dim)

        # Small additive bias (doesn't change weight magnitudes much)
        self.gate.μ += context_bias * 0.1

        logger.debug(
            f"Applied context priors: project={self._context_signals.project_name}"
        )

    def checkpoint(self) -> Path:
        """
        Save current state for crash recovery.

        Returns:
            Path where state was saved
        """
        return self.persister.save_gate_state(self.gate)

    def finalize_session(
        self,
        session_outcome: float | None = None
    ) -> dict[str, Any]:
        """
        Finalize session and save state.

        Args:
            session_outcome: Overall session outcome [0, 1]

        Returns:
            Summary of session learning
        """
        result = {
            "session_duration_minutes": None,
            "observations_gained": 0,
            "decisions": dict(self.gate.decisions),
            "saved_path": None,
        }

        if self._session_start:
            duration = datetime.now() - self._session_start
            result["session_duration_minutes"] = duration.total_seconds() / 60

        # Save state
        saved_path = self.persister.save_gate_state(self.gate)
        result["saved_path"] = str(saved_path)

        # Also save neuromodulator state if available
        if self.orchestra:
            self.persister.save_neuromodulator_state(self.orchestra)

        logger.info(
            f"Session finalized: decisions={result['decisions']}, "
            f"saved to {saved_path}"
        )

        return result

    def get_cold_start_progress(self) -> float:
        """
        Get progress toward escaping cold start.

        Returns:
            Progress [0, 1] where 1 = fully learned
        """
        return min(1.0, self.gate.n_observations / self.gate.cold_start_threshold)

    def get_current_blend_weights(self) -> dict[str, float]:
        """
        Get current heuristic vs learned blend weights.

        Returns:
            Dict with "heuristic" and "learned" weights
        """
        progress = self.get_cold_start_progress()
        return {
            "heuristic": 1.0 - progress,
            "learned": progress,
        }


__all__ = [
    "ColdStartManager",
    "ContextLoader",
    "ContextSignals",
    "PopulationPrior",
]
