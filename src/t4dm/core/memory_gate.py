"""
Memory Gate - Decides what gets stored in T4DM.

Implements signal detection to filter noise from voice/text interactions,
ensuring only meaningful episodes reach long-term memory.

P1-02: Integrates τ(t) temporal control signal for neural-gated memory writes.
The temporal gate modulates storage decisions based on:
- Prediction error
- Novelty
- Reward signals
- Dopamine modulation
- Theta phase (encoding vs retrieval)
"""

from __future__ import annotations

import logging
import re
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from enum import Enum
from hashlib import md5
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from t4dm.core.temporal_control import TemporalControlSignal, TemporalControlState
    from t4dm.learning.neuromodulators import NeuromodulatorState

logger = logging.getLogger(__name__)


class StorageDecision(Enum):
    """Result of memory gate evaluation."""
    STORE = "store"
    SKIP = "skip"
    BUFFER = "buffer"  # Hold for batching
    REDACT_THEN_STORE = "redact_then_store"


@dataclass
class GateContext:
    """Context for memory gate decision."""
    session_id: str
    project: str | None = None
    cwd: str | None = None
    recent_entities: list[str] = field(default_factory=list)
    last_store_time: datetime | None = None
    message_count_since_store: int = 0
    current_task: str | None = None
    is_voice: bool = False

    # P1-02: Temporal control context
    prediction_error: float = 0.0  # |δ| from dopamine system
    novelty_signal: float = 0.0  # External novelty score
    reward_signal: float = 0.0  # Reward from outcome
    theta_phase: float = 0.0  # Current theta oscillation phase [0, 2π]


@dataclass
class GateResult:
    """Result from memory gate evaluation."""
    decision: StorageDecision
    score: float
    reasons: list[str]
    suggested_importance: float
    batch_key: str | None = None  # For temporal batching

    # P1-02: Temporal control outputs
    tau_value: float = 0.5  # τ(t) gate value [0, 1]
    plasticity_gain: float = 1.0  # Multiplier for learning rate


class MemoryGate:
    """
    Decides what interactions should be stored as episodes.

    Uses a scoring model based on:
    - Novelty (is this new information?)
    - Outcome presence (did something complete?)
    - Entity density (mentions of known/new entities)
    - Action significance (file writes, deployments, etc.)
    - Time since last store (prevent over-storing)
    - Explicit triggers ("remember", "important")
    """

    # Explicit triggers - always store
    ALWAYS_STORE_PATTERNS = [
        r"\bremember\s+(that|this)\b",
        r"\bdon\'?t\s+forget\b",
        r"\bimportant\b",
        r"\bnote\s+(that|this)\b",
        r"\bprefer(ence)?\b.*\b(always|never|use|don\'t)\b",
        r"\bdeployed?\b",
        r"\breleased?\b",
        r"\bfixed\b.*\b(bug|error|issue)\b",
        r"\bcompleted?\b",
        r"\bfinished\b",
        r"\blearned?\b",
        r"\bdiscovered?\b",
    ]

    # Never store - noise
    NEVER_STORE_PATTERNS = [
        r"^(hi|hello|hey|thanks|thank you|ok|okay|sure|yes|no|yep|nope)[\.\!\?]?$",
        r"^(um+|uh+|hmm+|ah+|er+)[\.\!\?]?$",
        r"^(got it|sounds good|makes sense|i see|right)[\.\!\?]?$",
        r"^\s*$",  # Empty
    ]

    # High-value action patterns
    ACTION_PATTERNS = [
        (r"\b(created?|wrote|added)\s+\w+\.(py|ts|js|go|rs|md)\b", 0.8),
        (r"\b(edited?|modified?|updated?|changed?)\s+", 0.6),
        (r"\b(deleted?|removed?)\s+", 0.7),
        (r"\bgit\s+(commit|push|merge)\b", 0.8),
        (r"\b(test|tests)\s+(pass|passed|fail|failed)\b", 0.9),
        (r"\b(error|exception|bug|issue)\b", 0.7),
        (r"\b(resolved?|fixed|solved)\b", 0.8),
        (r"\bdeployed?\s+to\s+(prod|production|staging)\b", 1.0),
    ]

    # Entity patterns (will be enhanced with known entities)
    ENTITY_PATTERNS = [
        r"\b[A-Z][a-z]+(?:\s+[A-Z][a-z]+)+\b",  # Proper nouns
        r"\b\w+\.(py|ts|js|go|rs|md|json|yaml|toml)\b",  # Files
        r"\b(?:class|function|def|const|let|var)\s+(\w+)\b",  # Code entities
    ]

    def __init__(
        self,
        store_threshold: float = 0.4,
        buffer_threshold: float = 0.2,
        min_store_interval: timedelta = timedelta(seconds=30),
        max_messages_without_store: int = 20,
        voice_mode_adjustments: bool = True,
        use_temporal_control: bool = True,
        tau_weight: float = 0.3,
    ):
        """
        Initialize memory gate.

        Args:
            store_threshold: Score above which to store immediately
            buffer_threshold: Score above which to buffer for batching
            min_store_interval: Minimum time between stores
            max_messages_without_store: Force store after N messages
            voice_mode_adjustments: Apply voice-specific heuristics
            use_temporal_control: P1-02: Enable τ(t) neural gating
            tau_weight: Weight of τ(t) in final score (0-1)
        """
        self.store_threshold = store_threshold
        self.buffer_threshold = buffer_threshold
        self.min_store_interval = min_store_interval
        self.max_messages_without_store = max_messages_without_store
        self.voice_mode_adjustments = voice_mode_adjustments
        self.use_temporal_control = use_temporal_control
        self.tau_weight = tau_weight

        # Compile patterns
        self._always_store = [re.compile(p, re.IGNORECASE) for p in self.ALWAYS_STORE_PATTERNS]
        self._never_store = [re.compile(p, re.IGNORECASE) for p in self.NEVER_STORE_PATTERNS]
        self._actions = [(re.compile(p, re.IGNORECASE), w) for p, w in self.ACTION_PATTERNS]
        self._entities = [re.compile(p) for p in self.ENTITY_PATTERNS]

        # Recent content for novelty detection
        self._recent_hashes: set[str] = set()
        self._recent_hash_limit = 100

        # P1-02: Initialize temporal control signal
        self._temporal_control: TemporalControlSignal | None = None
        if use_temporal_control:
            self._init_temporal_control()

        logger.info(f"MemoryGate initialized (threshold={store_threshold}, tau={use_temporal_control})")

    def _init_temporal_control(self) -> None:
        """Initialize the τ(t) temporal control signal."""
        try:
            from t4dm.core.temporal_control import TemporalControlSignal
            self._temporal_control = TemporalControlSignal()
            logger.info("Temporal control signal initialized")
        except ImportError:
            logger.warning("TemporalControlSignal not available, disabling temporal control")
            self.use_temporal_control = False

    def set_temporal_control(self, control: TemporalControlSignal) -> None:
        """Set an external temporal control signal instance."""
        self._temporal_control = control
        self.use_temporal_control = True

    def evaluate(self, content: str, context: GateContext) -> GateResult:
        """
        Evaluate whether content should be stored.

        Args:
            content: Text content to evaluate
            context: Current context

        Returns:
            GateResult with decision and reasoning
        """
        reasons = []

        # Quick exits - explicit triggers
        for pattern in self._always_store:
            if pattern.search(content):
                reasons.append(f"Explicit trigger: {pattern.pattern}")
                return GateResult(
                    decision=StorageDecision.STORE,
                    score=1.0,
                    reasons=reasons,
                    suggested_importance=0.8,
                )

        # Quick exits - noise
        for pattern in self._never_store:
            if pattern.match(content.strip()):
                reasons.append(f"Noise pattern: {pattern.pattern}")
                return GateResult(
                    decision=StorageDecision.SKIP,
                    score=0.0,
                    reasons=reasons,
                    suggested_importance=0.0,
                )

        # Compute component scores
        novelty = self._novelty_score(content)
        outcome = self._outcome_score(content)
        entity_density = self._entity_score(content, context)
        action_sig = self._action_score(content)
        time_pressure = self._time_pressure_score(context)

        # Weighted combination
        weights = {
            "novelty": 0.25,
            "outcome": 0.25,
            "entity": 0.20,
            "action": 0.15,
            "time": 0.15,
        }

        # Voice mode adjustments
        if context.is_voice and self.voice_mode_adjustments:
            # Voice is noisier - raise threshold for non-action content
            if action_sig < 0.3:
                weights["novelty"] = 0.15
                weights["time"] = 0.25

        heuristic_score = (
            weights["novelty"] * novelty +
            weights["outcome"] * outcome +
            weights["entity"] * entity_density +
            weights["action"] * action_sig +
            weights["time"] * time_pressure
        )

        # P1-02: Compute τ(t) temporal control signal
        tau_value = 0.5  # Default neutral
        plasticity_gain = 1.0
        if self.use_temporal_control and self._temporal_control is not None:
            tau_state = self._compute_tau(context, novelty)
            tau_value = tau_state.tau
            plasticity_gain = tau_state.plasticity_gain

            # Blend heuristic score with τ(t)
            score = (1.0 - self.tau_weight) * heuristic_score + self.tau_weight * tau_value

            reasons.append(f"τ(t)={tau_value:.2f}, plasticity={plasticity_gain:.2f}")
        else:
            score = heuristic_score

        reasons.append(f"novelty={novelty:.2f}, outcome={outcome:.2f}, "
                      f"entity={entity_density:.2f}, action={action_sig:.2f}, "
                      f"time={time_pressure:.2f}")

        # Determine decision
        if score >= self.store_threshold:
            decision = StorageDecision.STORE
            self._record_content(content)
        elif score >= self.buffer_threshold:
            decision = StorageDecision.BUFFER
            batch_key = self._compute_batch_key(content, context)
        else:
            decision = StorageDecision.SKIP
            batch_key = None

        # Suggested importance for stored content
        importance = min(1.0, score * 1.5)  # Scale up slightly

        return GateResult(
            decision=decision,
            score=score,
            reasons=reasons,
            suggested_importance=importance,
            batch_key=batch_key if decision == StorageDecision.BUFFER else None,
            tau_value=tau_value,
            plasticity_gain=plasticity_gain,
        )

    def _compute_tau(self, context: GateContext, novelty_heuristic: float) -> TemporalControlState:
        """
        P1-02: Compute τ(t) temporal control state.

        Combines context signals with the temporal control neural network
        to produce a biologically-plausible gating signal.

        Args:
            context: Gate context with neural signals
            novelty_heuristic: Novelty score from heuristic analysis

        Returns:
            TemporalControlState with tau value and derived signals
        """
        import torch
        from t4dm.core.temporal_control import TemporalControlMode

        # Use context signals if available, else derive from heuristics
        prediction_error = context.prediction_error if context.prediction_error > 0 else novelty_heuristic * 0.5
        novelty = context.novelty_signal if context.novelty_signal > 0 else novelty_heuristic
        reward = context.reward_signal  # 0 if not set

        # Determine mode based on message count
        # More messages = more likely in encoding mode
        if context.message_count_since_store < 3:
            mode = TemporalControlMode.RETRIEVAL
        elif context.message_count_since_store > 10:
            mode = TemporalControlMode.ENCODING
        else:
            mode = TemporalControlMode.MAINTENANCE

        return self._temporal_control.compute_state(
            prediction_error=torch.tensor(prediction_error),
            novelty=torch.tensor(novelty),
            reward=torch.tensor(reward),
            dopamine=torch.tensor(context.prediction_error),  # Use PE as dopamine proxy
            theta_phase=torch.tensor(context.theta_phase),
            mode=mode,
        )

    def _novelty_score(self, content: str) -> float:
        """Score based on how novel this content is."""
        content_hash = md5(content.lower().encode(), usedforsecurity=False).hexdigest()[:16]

        if content_hash in self._recent_hashes:
            return 0.0  # Duplicate

        # Check for similar content (simple word overlap)
        words = set(content.lower().split())
        if len(words) < 3:
            return 0.3  # Too short to judge

        return 0.8  # Assume novel

    def _outcome_score(self, content: str) -> float:
        """Score based on presence of outcome indicators."""
        outcome_patterns = [
            (r"\b(success|succeeded|passed|worked|done|complete)\b", 0.9),
            (r"\b(fail|failed|error|broken|issue)\b", 0.8),
            (r"\b(fixed|resolved|solved)\b", 1.0),
            (r"\b(started|beginning|trying)\b", 0.3),
        ]

        max_score = 0.0
        for pattern, weight in outcome_patterns:
            if re.search(pattern, content, re.IGNORECASE):
                max_score = max(max_score, weight)

        return max_score

    def _entity_score(self, content: str, context: GateContext) -> float:
        """Score based on entity mentions."""
        entities_found = 0

        # Check for known entities
        for entity in context.recent_entities:
            if entity.lower() in content.lower():
                entities_found += 1

        # Check for new entity patterns
        for pattern in self._entities:
            matches = pattern.findall(content)
            entities_found += len(matches)

        # Normalize: 0 entities = 0, 3+ entities = 1.0
        return min(1.0, entities_found / 3.0)

    def _action_score(self, content: str) -> float:
        """Score based on significant actions."""
        max_score = 0.0
        for pattern, weight in self._actions:
            if pattern.search(content):
                max_score = max(max_score, weight)
        return max_score

    def _time_pressure_score(self, context: GateContext) -> float:
        """Score based on time/message count since last store."""
        score = 0.0

        # Time-based pressure
        if context.last_store_time:
            elapsed = datetime.now() - context.last_store_time
            if elapsed > timedelta(minutes=10):
                score += 0.5
            elif elapsed > timedelta(minutes=5):
                score += 0.3
        else:
            score += 0.3  # No previous store

        # Message count pressure
        msg_ratio = context.message_count_since_store / self.max_messages_without_store
        score += min(0.5, msg_ratio * 0.5)

        return min(1.0, score)

    def _record_content(self, content: str) -> None:
        """Record content hash for novelty detection."""
        content_hash = md5(content.lower().encode(), usedforsecurity=False).hexdigest()[:16]
        self._recent_hashes.add(content_hash)

        # Limit size
        if len(self._recent_hashes) > self._recent_hash_limit:
            # Remove oldest (convert to list, remove first half)
            hashes = list(self._recent_hashes)
            self._recent_hashes = set(hashes[len(hashes)//2:])

    def _compute_batch_key(self, content: str, context: GateContext) -> str:
        """Compute key for batching similar content."""
        # Batch by project + task + time window
        parts = [
            context.project or "default",
            context.current_task or "general",
        ]
        return ":".join(parts)

    def force_store_check(self, context: GateContext) -> bool:
        """Check if we should force a store regardless of content."""
        # Force store after N messages
        if context.message_count_since_store >= self.max_messages_without_store:
            return True

        # Force store after long silence then activity
        if context.last_store_time:
            elapsed = datetime.now() - context.last_store_time
            if elapsed > timedelta(minutes=30):
                return True

        return False


class TemporalBatcher:
    """
    Batches buffered content into coherent episodes.

    For voice interactions, aggregates multiple utterances
    into single meaningful episodes.
    """

    def __init__(
        self,
        batch_window: timedelta = timedelta(minutes=2),
        max_batch_size: int = 10,
    ):
        self.batch_window = batch_window
        self.max_batch_size = max_batch_size
        self._batches: dict[str, list[tuple[datetime, str]]] = {}

    def add(self, batch_key: str, content: str) -> str | None:
        """
        Add content to batch, return completed batch if ready.

        Args:
            batch_key: Key for grouping related content
            content: Content to add

        Returns:
            Batched content if batch is complete, None otherwise
        """
        now = datetime.now()

        if batch_key not in self._batches:
            self._batches[batch_key] = []

        batch = self._batches[batch_key]

        # Check if batch should be flushed first
        if batch and (
            now - batch[0][0] > self.batch_window or
            len(batch) >= self.max_batch_size
        ):
            result = self._flush_batch(batch_key)
            self._batches[batch_key] = [(now, content)]
            return result

        batch.append((now, content))
        return None

    def _flush_batch(self, batch_key: str) -> str:
        """Combine batch into single content string."""
        batch = self._batches.get(batch_key, [])
        if not batch:
            return ""

        # Combine content, removing duplicates
        seen = set()
        unique_content = []
        for _, content in batch:
            if content not in seen:
                seen.add(content)
                unique_content.append(content)

        return " | ".join(unique_content)

    def flush_all(self) -> list[tuple[str, str]]:
        """Flush all batches, return (key, content) pairs."""
        results = []
        for key in list(self._batches.keys()):
            content = self._flush_batch(key)
            if content:
                results.append((key, content))
        self._batches.clear()
        return results
