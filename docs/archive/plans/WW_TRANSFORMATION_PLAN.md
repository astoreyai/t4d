# World Weaver: Adaptive Learning System Transformation Plan

**Vision**: Transform WW from "a system that gets fuller" to "a system that gets smarter"
**Architect**: Geoffrey Hinton design principles
**Target**: Fully integrated Claude Code CLI adaptive memory

---

## Current State Assessment

| Metric | Current | Target |
|--------|---------|--------|
| Learning | None (storage only) | Continuous adaptation |
| Credit Assignment | None | Temporal traces + attention + DAG |
| Retrieval Scoring | Fixed weights | Learned parameters |
| Consolidation | Heuristic thresholds | Adaptive per-domain |
| Decay | Fixed FSRS | Per-type/domain learned |
| CLI Integration | Partial hooks | Full lifecycle |
| Test Coverage | 60% (21 failing) | 90%+ |

### Critical Blockers
1. **Vector dimension mismatch**: Qdrant has 384-dim, code uses 1024-dim BGE-M3
2. **No feedback collection**: No infrastructure to track retrieval→outcome
3. **Static weights**: All scoring is hardcoded

---

## Phase 0: Infrastructure Stabilization
**Duration**: 1-2 days | **Complexity**: Low | **Prerequisite**: None

### Objectives
- Fix all 21 failing tests
- Stabilize dual-store infrastructure
- Establish clean baseline

### Tasks

#### 0.1 Fix Vector Dimensions
```bash
# Delete old collections
curl -X DELETE http://localhost:6333/collections/ww_episodes
curl -X DELETE http://localhost:6333/collections/ww_entities
curl -X DELETE http://localhost:6333/collections/ww_procedures
curl -X DELETE http://localhost:6333/collections/ww_episodes_hybrid

# Collections will be recreated with 1024-dim on first use
```

#### 0.2 Verify Infrastructure
- [ ] Neo4j connection healthy
- [ ] Qdrant collections recreated with 1024-dim
- [ ] All 1,259 tests passing
- [ ] Coverage back to 79%+

#### 0.3 Baseline Metrics
- [ ] Document current retrieval accuracy
- [ ] Benchmark query latency
- [ ] Record storage sizes

### Exit Criteria
- 0 test failures
- Infrastructure health checks pass
- Baseline metrics documented

---

## Phase 1: Feedback Collection Infrastructure
**Duration**: 3-5 days | **Complexity**: Low | **Prerequisite**: Phase 0
**Hinton Principle**: "You can't learn without contrastive signals"

### Objectives
- Instrument all memory operations
- Collect retrieval→outcome pairs
- Build experience storage (no ML yet)

### Tasks

#### 1.1 Event Data Structures
```python
# src/t4dm/learning/events.py

@dataclass
class RetrievalEvent:
    retrieval_id: UUID
    query: str
    query_embedding: list[float]
    retrieved_ids: list[UUID]
    retrieval_scores: dict[UUID, float]
    score_components: dict[UUID, dict[str, float]]  # semantic, recency, etc.
    timestamp: datetime
    context_hash: str  # Hash of cwd + project + tool
    session_id: str

@dataclass
class OutcomeEvent:
    outcome_id: UUID
    success_score: float  # [0, 1]
    context_hash: str
    timestamp: datetime
    task_description: str
    error: Optional[str]
    explicit_citations: list[UUID]  # Memories LLM explicitly referenced

@dataclass
class FeedbackEvent:
    feedback_id: UUID
    feedback_type: str  # accept, reject, modify, explicit_positive, explicit_negative
    context_hash: str
    timestamp: datetime
    implied_reward: float
```

#### 1.2 Event Collection Hooks
- [ ] Create `src/t4dm/learning/collector.py`
- [ ] Hook into `episodic.recall()` → emit RetrievalEvent
- [ ] Hook into `semantic.recall()` → emit RetrievalEvent
- [ ] Hook into `procedural.match_trigger()` → emit RetrievalEvent
- [ ] Create outcome detection from tool results
- [ ] Implement implicit feedback detection (accept/reject patterns)

#### 1.3 Event Storage
- [ ] Create SQLite-based event store (lightweight, no dependencies)
- [ ] Schema: events table with type, payload JSON, timestamp, session_id
- [ ] Retention policy: 90 days rolling window
- [ ] Index on context_hash for fast matching

#### 1.4 Context Hashing
```python
def compute_context_hash(cwd: str, project: str, tool: str, query: str) -> str:
    """Stable hash for matching retrievals to outcomes."""
    content = f"{cwd}|{project}|{tool}|{query[:100]}"
    return hashlib.sha256(content.encode()).hexdigest()[:16]
```

### Exit Criteria
- All memory operations instrumented
- Events persisting to SQLite
- Can query: "What retrievals preceded this outcome?"

---

## Phase 2: Reward Computation & Credit Assignment
**Duration**: 5-7 days | **Complexity**: Medium | **Prerequisite**: Phase 1
**Hinton Principle**: "Credit assignment is the fundamental problem"

### Objectives
- Compute rewards from outcomes
- Implement temporal credit assignment
- Connect rewards to memories

### Tasks

#### 2.1 Baseline Tracker
```python
# src/t4dm/learning/baseline.py

class TaskBaselineTracker:
    """Track expected success rate per task type."""

    def __init__(self, window_days: int = 30):
        self.baselines: dict[str, RunningMean] = {}

    def get_baseline(self, context_hash: str) -> float:
        """Return expected success for this context."""
        return self.baselines.get(context_hash, RunningMean(0.5)).mean

    def update(self, context_hash: str, outcome: float):
        if context_hash not in self.baselines:
            self.baselines[context_hash] = RunningMean(0.5)
        self.baselines[context_hash].update(outcome)
```

#### 2.2 Retrieval Reward Computation
```python
# src/t4dm/learning/rewards.py

def compute_retrieval_reward(
    retrieval: RetrievalEvent,
    outcome: OutcomeEvent,
    baseline: float,
) -> dict[UUID, float]:
    """
    R = (outcome - baseline) × time_discount × attention_weight
    """
    advantage = outcome.success_score - baseline

    hours_delay = (outcome.timestamp - retrieval.timestamp).total_seconds() / 3600
    time_discount = 1.0 / (1.0 + 0.1 * hours_delay)  # Hyperbolic

    total_score = sum(retrieval.retrieval_scores.values())

    rewards = {}
    for mem_id, score in retrieval.retrieval_scores.items():
        attention_weight = score / total_score if total_score > 0 else 0

        # Boost for explicit citations
        citation_bonus = 1.5 if mem_id in outcome.explicit_citations else 1.0

        rewards[mem_id] = advantage * time_discount * attention_weight * citation_bonus

    return rewards
```

#### 2.3 Eligibility Traces (TD-λ)
```python
# src/t4dm/learning/traces.py

class EligibilityTraceManager:
    """
    Temporal credit assignment via eligibility traces.

    trace[mem] += score     # on retrieval
    trace[mem] *= (γλ)^Δt   # decay over time
    credit = reward × trace/Σtraces  # on outcome
    """

    def __init__(self, gamma: float = 0.99, lambda_: float = 0.9):
        self.gamma = gamma
        self.lambda_ = lambda_
        self.traces: dict[UUID, float] = {}
        self.last_update: dict[UUID, datetime] = {}

    def on_retrieval(self, memory_id: UUID, score: float, timestamp: datetime):
        self._decay_all(timestamp)
        self.traces[memory_id] = self.traces.get(memory_id, 0) + score
        self.last_update[memory_id] = timestamp

    def on_outcome(self, reward: float, timestamp: datetime) -> dict[UUID, float]:
        self._decay_all(timestamp)
        total = sum(self.traces.values())
        if total == 0:
            return {}
        return {mem_id: reward * trace / total
                for mem_id, trace in self.traces.items()}

    def _decay_all(self, current: datetime):
        for mem_id in list(self.traces.keys()):
            last = self.last_update.get(mem_id, current)
            hours = (current - last).total_seconds() / 3600
            decay = (self.gamma * self.lambda_) ** hours
            self.traces[mem_id] *= decay
            if self.traces[mem_id] < 0.01:
                del self.traces[mem_id]
```

#### 2.4 Reward Storage
- [ ] Create reward ledger table: memory_id, reward, timestamp, source
- [ ] Aggregate rewards per memory (running sum)
- [ ] Expose via API: `GET /memories/{id}/rewards`

#### 2.5 Reward-Weighted Hebbian Update
```python
# Modify existing Hebbian strengthening

async def strengthen_with_reward(
    mem1: UUID, mem2: UUID,
    reward1: float, reward2: float,
    base_lr: float = 0.05
):
    """Only strengthen if both have positive reward."""
    if reward1 > 0 and reward2 > 0:
        combined = reward1 + reward2
        lr = min(base_lr * combined, 0.2)  # Cap learning rate
        await graph_store.strengthen_relationship(mem1, mem2, lr)
    elif reward1 < 0 or reward2 < 0:
        # Weaken connection for negative rewards
        lr = min(base_lr * abs(min(reward1, reward2)), 0.1)
        await graph_store.weaken_relationship(mem1, mem2, lr)
```

### Exit Criteria
- Rewards computed for all retrieval→outcome pairs
- Eligibility traces updating correctly
- Hebbian updates weighted by reward
- Can query: "What is this memory's cumulative reward?"

---

## Phase 3: Learnable Components
**Duration**: 7-10 days | **Complexity**: High | **Prerequisite**: Phase 2
**Hinton Principle**: "Replace hand-tuned constants with learned parameters"

### Objectives
- Add PyTorch for gradient-based learning
- Make retrieval weights learnable
- Implement experience replay buffer

### Tasks

#### 3.1 Dependencies
```toml
# Add to pyproject.toml
[project.optional-dependencies]
learning = [
    "torch>=2.0",
    "numpy>=1.24",
]
```

#### 3.2 Learnable Retrieval Scorer
```python
# src/t4dm/learning/scorer.py

import torch
import torch.nn as nn
import torch.nn.functional as F

class LearnedRetrievalScorer(nn.Module):
    """
    Replaces fixed weights with learnable parameters.

    Current: score = 0.4*semantic + 0.25*recency + 0.2*outcome + 0.15*importance
    Target:  score = softmax(θ) · [semantic, recency, outcome, importance]
    """

    def __init__(self):
        super().__init__()
        # Initialize near current hand-tuned values
        self.raw_weights = nn.Parameter(torch.tensor([
            0.4, 0.25, 0.2, 0.15  # semantic, recency, outcome, importance
        ]))

        # Per-memory-type scaling
        self.type_scales = nn.ParameterDict({
            'episodic': nn.Parameter(torch.tensor(1.0)),
            'semantic': nn.Parameter(torch.tensor(1.0)),
            'procedural': nn.Parameter(torch.tensor(1.0)),
        })

        # Temperature for ranking sharpness
        self.temperature = nn.Parameter(torch.tensor(1.0))

    def forward(self, components: torch.Tensor, memory_type: str) -> torch.Tensor:
        """
        components: [batch, 4] - semantic, recency, outcome, importance
        Returns: [batch] scores
        """
        weights = F.softmax(self.raw_weights, dim=0)
        base_score = (components * weights).sum(dim=-1)
        type_scale = self.type_scales[memory_type]
        return base_score * type_scale / self.temperature

    def get_weights(self) -> dict[str, float]:
        """Return current normalized weights for inspection."""
        weights = F.softmax(self.raw_weights, dim=0).detach().numpy()
        return {
            'semantic': float(weights[0]),
            'recency': float(weights[1]),
            'outcome': float(weights[2]),
            'importance': float(weights[3]),
        }
```

#### 3.3 Experience Replay Buffer
```python
# src/t4dm/learning/replay.py

@dataclass
class Experience:
    experience_id: UUID
    timestamp: datetime
    query: str
    query_embedding: list[float]
    retrieved_ids: list[UUID]
    retrieval_scores: list[float]
    component_vectors: list[list[float]]  # [n_retrieved, 4]
    outcome_score: float
    per_memory_rewards: dict[str, float]
    session_id: str
    memory_type: str

class PrioritizedReplayBuffer:
    """
    Prioritized experience replay for offline learning.
    Priority = |TD error| + ε
    """

    def __init__(self, capacity: int = 100_000, alpha: float = 0.6):
        self.capacity = capacity
        self.alpha = alpha  # Priority exponent
        self.buffer: list[Experience] = []
        self.priorities: list[float] = []
        self.position = 0

    def add(self, experience: Experience, td_error: float):
        priority = (abs(td_error) + 1e-6) ** self.alpha

        if len(self.buffer) < self.capacity:
            self.buffer.append(experience)
            self.priorities.append(priority)
        else:
            self.buffer[self.position] = experience
            self.priorities[self.position] = priority

        self.position = (self.position + 1) % self.capacity

    def sample(self, batch_size: int, beta: float = 0.4) -> tuple[list[Experience], list[float]]:
        """Sample with importance weights."""
        probs = np.array(self.priorities) / sum(self.priorities)
        indices = np.random.choice(len(self.buffer), min(batch_size, len(self.buffer)),
                                   p=probs, replace=False)

        # Importance sampling weights
        weights = (len(self.buffer) * probs[indices]) ** -beta
        weights /= weights.max()

        return [self.buffer[i] for i in indices], weights.tolist()
```

#### 3.4 Ranking Loss (ListMLE)
```python
# src/t4dm/learning/losses.py

def list_mle_loss(scores: torch.Tensor, rewards: torch.Tensor) -> torch.Tensor:
    """
    ListMLE: Learn to rank memories by eventual utility.

    scores: [n] predicted retrieval scores
    rewards: [n] actual rewards from outcomes
    """
    # Sort by reward to get target ranking
    target_order = torch.argsort(rewards, descending=True)

    loss = torch.tensor(0.0)
    remaining_mask = torch.ones(len(scores), dtype=torch.bool)

    for target_idx in target_order:
        if not remaining_mask[target_idx]:
            continue

        remaining_scores = scores[remaining_mask]
        target_score = scores[target_idx]

        # Log probability of selecting target from remaining
        log_prob = target_score - torch.logsumexp(remaining_scores, dim=0)
        loss = loss - log_prob

        remaining_mask[target_idx] = False

    return loss / len(target_order)
```

#### 3.5 Training Loop
```python
# src/t4dm/learning/trainer.py

class RetrievalScorerTrainer:
    def __init__(self, scorer: LearnedRetrievalScorer, lr: float = 1e-4):
        self.scorer = scorer
        self.optimizer = torch.optim.Adam(scorer.parameters(), lr=lr)
        self.replay_buffer = PrioritizedReplayBuffer()

    def train_step(self, batch_size: int = 32) -> float:
        experiences, weights = self.replay_buffer.sample(batch_size)
        if not experiences:
            return 0.0

        self.optimizer.zero_grad()
        total_loss = torch.tensor(0.0)

        for exp, weight in zip(experiences, weights):
            components = torch.tensor(exp.component_vectors)
            scores = self.scorer(components, exp.memory_type)
            rewards = torch.tensor([exp.per_memory_rewards.get(str(m), 0)
                                   for m in exp.retrieved_ids])

            loss = list_mle_loss(scores, rewards)
            total_loss = total_loss + weight * loss

        total_loss = total_loss / len(experiences)
        total_loss.backward()
        self.optimizer.step()

        return total_loss.item()
```

#### 3.6 Integration with Retrieval
- [ ] Modify `episodic.py` to use `LearnedRetrievalScorer`
- [ ] Modify `semantic.py` to use `LearnedRetrievalScorer`
- [ ] Modify `procedural.py` to use `LearnedRetrievalScorer`
- [ ] Add fallback to fixed weights if model not loaded
- [ ] Checkpoint saving/loading

### Exit Criteria
- Retrieval weights are PyTorch parameters
- Experience replay buffer collecting data
- Training loop can run and update weights
- Weights persist across restarts

---

## Phase 4: Claude Code CLI Integration
**Duration**: 5-7 days | **Complexity**: Medium | **Prerequisite**: Phase 2
**Goal**: Seamless lifecycle integration with Claude Code

### Tasks

#### 4.1 Session Hooks

**SessionStart Hook** (`~/.claude/skills/ww-learning/session_start.py`):
```python
#!/usr/bin/env python3
"""Load context and initialize learning state for new session."""

import sys
import json
from ww.sdk import WorldWeaverClient
from ww.learning import LearningSession

def main():
    session_id = os.environ.get("CLAUDE_SESSION_ID", str(uuid4()))
    project = os.path.basename(os.getcwd())

    with WorldWeaverClient() as ww:
        # Initialize learning session
        learning = LearningSession(ww, session_id)
        learning.start()

        # Load relevant context
        context = ww.get_context(
            query=f"working on {project}",
            include_episodes=True,
            include_entities=True,
            include_skills=True,
            limit=20,
        )

        # Prime eligibility traces with loaded memories
        for item in context.episodes + context.entities:
            learning.traces.on_retrieval(item.id, 0.5, datetime.now())

        # Return context for Claude
        output = {
            "session_id": session_id,
            "project": project,
            "context": {
                "episodes": len(context.episodes),
                "entities": len(context.entities),
                "skills": len(context.skills),
            },
            "learning_status": "active",
        }

        print(json.dumps(output))

if __name__ == "__main__":
    main()
```

**SessionEnd Hook** (`~/.claude/skills/ww-learning/session_end.py`):
```python
#!/usr/bin/env python3
"""Persist learning state and trigger offline training."""

import sys
import json
from ww.sdk import WorldWeaverClient
from ww.learning import LearningSession, OfflineTrainer

def main():
    session_id = os.environ.get("CLAUDE_SESSION_ID")

    with WorldWeaverClient() as ww:
        learning = LearningSession.load(session_id)

        # Compute final rewards for session
        rewards = learning.finalize()

        # Persist to replay buffer
        learning.save_experiences()

        # Maybe trigger offline training
        if learning.experience_count > 10:
            trainer = OfflineTrainer(ww)
            trainer.train_step(batch_size=32)

        output = {
            "session_id": session_id,
            "experiences_collected": learning.experience_count,
            "memories_rewarded": len(rewards),
            "offline_training": learning.experience_count > 10,
        }

        print(json.dumps(output))

if __name__ == "__main__":
    main()
```

#### 4.2 MCP Tool Wrappers

Add learning-aware wrappers to MCP tools:

```python
# src/t4dm/mcp/tools/learning_wrappers.py

class LearningAwareRecall:
    """Wraps recall to emit retrieval events."""

    def __init__(self, base_recall, collector: EventCollector):
        self.base_recall = base_recall
        self.collector = collector

    async def __call__(self, query: str, **kwargs) -> list[ScoredResult]:
        results = await self.base_recall(query, **kwargs)

        # Emit retrieval event
        await self.collector.on_retrieval(
            query=query,
            results=results,
            context=kwargs.get("context", {}),
        )

        return results
```

#### 4.3 Implicit Feedback Detection

```python
# src/t4dm/learning/feedback.py

class ImplicitFeedbackDetector:
    """Detect feedback from user behavior patterns."""

    REJECTION_PATTERNS = [
        r"no,?\s+(actually|instead|let'?s)",
        r"that'?s not (right|what I)",
        r"try again",
        r"^(no|nope)$",
    ]

    ACCEPTANCE_PATTERNS = [
        r"(thanks|thank you|perfect|great|good)",
        r"^(yes|yep|ok|okay)$",
    ]

    def detect(self, claude_output: str, user_response: str) -> Optional[FeedbackEvent]:
        user_lower = user_response.lower().strip()

        for pattern in self.REJECTION_PATTERNS:
            if re.search(pattern, user_lower):
                return FeedbackEvent(
                    feedback_type="reject",
                    implied_reward=-0.5,
                    timestamp=datetime.now(),
                )

        for pattern in self.ACCEPTANCE_PATTERNS:
            if re.search(pattern, user_lower):
                return FeedbackEvent(
                    feedback_type="accept",
                    implied_reward=0.3,
                    timestamp=datetime.now(),
                )

        return None
```

#### 4.4 Tool Outcome Tracking

```python
# Track outcomes from Bash, Edit, Write tools

class ToolOutcomeTracker:
    """Infer success/failure from tool execution."""

    def on_bash_result(self, command: str, exit_code: int, output: str) -> OutcomeEvent:
        if exit_code == 0:
            success_score = 1.0
        elif "error" in output.lower() or "failed" in output.lower():
            success_score = 0.0
        else:
            success_score = 0.3  # Non-zero exit but no obvious error

        return OutcomeEvent(
            success_score=success_score,
            task_description=command[:100],
            error=output if exit_code != 0 else None,
            timestamp=datetime.now(),
        )

    def on_edit_result(self, file_path: str, success: bool, error: str = None) -> OutcomeEvent:
        return OutcomeEvent(
            success_score=1.0 if success else 0.0,
            task_description=f"edit {file_path}",
            error=error,
            timestamp=datetime.now(),
        )
```

#### 4.5 Settings Configuration

Update `~/.claude/settings.json`:
```json
{
  "hooks": {
    "SessionStart": [{
      "matcher": "*",
      "hooks": [{
        "type": "command",
        "command": "python3 ~/.claude/skills/ww-learning/session_start.py",
        "timeout": 15
      }]
    }],
    "SessionEnd": [{
      "matcher": "*",
      "hooks": [{
        "type": "command",
        "command": "python3 ~/.claude/skills/ww-learning/session_end.py",
        "timeout": 180
      }]
    }]
  }
}
```

### Exit Criteria
- Session hooks loading/persisting learning state
- All MCP tools emitting events
- Implicit feedback detection working
- Offline training triggered on session end

---

## Phase 5: Offline Learning Pipeline
**Duration**: 5-7 days | **Complexity**: High | **Prerequisite**: Phase 3, 4
**Hinton Principle**: "Offline replay enables credit assignment across long delays"

### Tasks

#### 5.1 Scheduled Training
```python
# src/t4dm/learning/scheduler.py

class LearningScheduler:
    """Schedule offline training runs."""

    def __init__(self, trainer: RetrievalScorerTrainer):
        self.trainer = trainer
        self.last_training = datetime.now()
        self.training_interval = timedelta(hours=1)

    async def maybe_train(self):
        """Check if training is due and run if so."""
        if datetime.now() - self.last_training < self.training_interval:
            return None

        # Check if enough new experiences
        new_count = self.trainer.replay_buffer.count_since(self.last_training)
        if new_count < 50:
            return None

        # Run training
        metrics = await self.train_epoch(batches=10)
        self.last_training = datetime.now()

        return metrics

    async def train_epoch(self, batches: int = 10) -> dict:
        """Run multiple training batches."""
        losses = []
        for _ in range(batches):
            loss = self.trainer.train_step()
            losses.append(loss)

        return {
            "mean_loss": sum(losses) / len(losses),
            "batches": batches,
            "timestamp": datetime.now().isoformat(),
        }
```

#### 5.2 Consolidation Quality Evaluation
```python
# src/t4dm/learning/consolidation_eval.py

class ConsolidationEvaluator:
    """Evaluate quality of consolidated entities/skills."""

    async def evaluate_entity(self, entity_id: UUID, window_days: int = 7) -> float:
        """
        Q = 0.3×usage + 0.4×utility + 0.15×centrality + 0.15×specificity
        """
        since = datetime.now() - timedelta(days=window_days)

        # Usage frequency
        retrievals = await self._count_retrievals(entity_id, since)
        usage_score = 1 - math.exp(-retrievals / 5)

        # Outcome-weighted utility
        outcomes = await self._get_retrieval_outcomes(entity_id, since)
        utility_score = sum(o.success_score * o.weight for o in outcomes) / max(len(outcomes), 1)

        # Graph centrality (PageRank contribution)
        centrality = await self._compute_pagerank(entity_id)
        centrality_score = min(centrality * 10, 1.0)

        # Specificity (inverse of fan-out)
        fan_out = await self._get_fan_out(entity_id)
        specificity_score = 1.0 / (1 + math.log(1 + fan_out))

        return (0.3 * usage_score + 0.4 * utility_score +
                0.15 * centrality_score + 0.15 * specificity_score)
```

#### 5.3 Forgetting Regret Detection
```python
# src/t4dm/learning/forgetting.py

class ForgettingRegretDetector:
    """Detect when we forgot something we shouldn't have."""

    def __init__(self, embedding_model):
        self.embedder = embedding_model
        self.forgotten_signatures: list[ForgottenRecord] = []

    async def on_memory_forgotten(self, memory: Episode | Entity | Procedure):
        """Record signature of forgotten content."""
        signature = await self.embedder.embed(memory.content)
        self.forgotten_signatures.append(ForgottenRecord(
            memory_id=memory.id,
            signature=signature,
            forgotten_at=datetime.now(),
            memory_type=type(memory).__name__,
        ))

    async def check_regret(self, new_content: str) -> Optional[ForgottenRecord]:
        """Check if new content matches something we forgot."""
        new_sig = await self.embedder.embed(new_content)

        for record in self.forgotten_signatures:
            similarity = cosine_similarity(new_sig, record.signature)
            if similarity > 0.85:
                # We're re-learning something we forgot!
                days_since = (datetime.now() - record.forgotten_at).days
                regret_severity = 1.0 / (1 + days_since / 7)

                await self._record_regret(record, regret_severity)
                return record

        return None
```

#### 5.4 Adaptive Decay Training
```python
# src/t4dm/learning/decay.py

class AdaptiveDecayTrainer:
    """Learn optimal decay rates per memory type/domain."""

    def __init__(self):
        self.decay_params = nn.ParameterDict({
            'episodic': nn.Parameter(torch.tensor(math.log(0.9))),
            'semantic': nn.Parameter(torch.tensor(math.log(0.95))),
            'procedural': nn.Parameter(torch.tensor(math.log(0.92))),
        })
        self.optimizer = torch.optim.Adam(self.decay_params.values(), lr=1e-5)

    def train_step(self, forgotten_with_outcomes: list[tuple]) -> float:
        """
        Train on (decay_rate, time_to_next_access, was_accessed_again) tuples.
        Minimize: keeping useless + forgetting useful
        """
        self.optimizer.zero_grad()
        loss = torch.tensor(0.0)

        for memory_type, time_to_access, was_accessed in forgotten_with_outcomes:
            decay = torch.exp(self.decay_params[memory_type])
            predicted_retrievability = (1 + decay * time_to_access) ** -0.5

            if was_accessed:
                # Should have kept it - want high retrievability
                loss = loss - torch.log(predicted_retrievability + 1e-6)
            else:
                # Good we forgot it - want low retrievability
                loss = loss - 0.1 * torch.log(1 - predicted_retrievability + 1e-6)

        loss = loss / len(forgotten_with_outcomes)
        loss.backward()
        self.optimizer.step()

        return loss.item()
```

### Exit Criteria
- Scheduled training running hourly
- Entity/skill quality being evaluated
- Forgetting regret being tracked
- Decay rates adapting to access patterns

---

## Phase 6: Advanced Learning Features
**Duration**: 10-14 days | **Complexity**: Very High | **Prerequisite**: Phase 5
**Hinton Principle**: "Meta-learning and catastrophic forgetting prevention"

### Tasks

#### 6.1 Elastic Weight Consolidation (EWC)
```python
# src/t4dm/learning/ewc.py

class EWCRegularizer:
    """Prevent catastrophic forgetting of old tasks."""

    def __init__(self, lambda_: float = 1000):
        self.lambda_ = lambda_
        self.saved_params: dict[str, torch.Tensor] = {}
        self.fisher_diag: dict[str, torch.Tensor] = {}

    def compute_fisher(self, model: nn.Module, dataloader):
        """Compute Fisher information diagonal after task completion."""
        self.saved_params = {n: p.clone() for n, p in model.named_parameters()}
        self.fisher_diag = {n: torch.zeros_like(p) for n, p in model.named_parameters()}

        model.eval()
        for batch in dataloader:
            model.zero_grad()
            loss = model.compute_loss(batch)
            loss.backward()

            for name, param in model.named_parameters():
                if param.grad is not None:
                    self.fisher_diag[name] += param.grad ** 2

        for name in self.fisher_diag:
            self.fisher_diag[name] /= len(dataloader)

    def penalty(self, model: nn.Module) -> torch.Tensor:
        """EWC penalty: λ × Σᵢ Fᵢ × (θᵢ - θᵢ*)²"""
        penalty = torch.tensor(0.0)
        for name, param in model.named_parameters():
            if name in self.saved_params:
                penalty += (self.fisher_diag[name] *
                           (param - self.saved_params[name]) ** 2).sum()
        return self.lambda_ * penalty
```

#### 6.2 MAML for Domain Adaptation
```python
# src/t4dm/learning/maml.py

class MAMLAdapter:
    """
    Model-Agnostic Meta-Learning for fast domain adaptation.

    Outer loop: Learn good initialization across domains
    Inner loop: Fast adapt to specific domain with few examples
    """

    def __init__(self, model: nn.Module, inner_lr: float = 0.01, outer_lr: float = 0.001):
        self.model = model
        self.inner_lr = inner_lr
        self.meta_optimizer = torch.optim.Adam(model.parameters(), lr=outer_lr)

    def adapt(self, support_set: list[Experience], query_set: list[Experience],
              inner_steps: int = 5) -> float:
        """Adapt to task and return query loss."""
        # Clone params for inner loop
        adapted = {n: p.clone() for n, p in self.model.named_parameters()}

        # Inner loop: adapt on support set
        for _ in range(inner_steps):
            loss = self._compute_loss(support_set, adapted)
            grads = torch.autograd.grad(loss, adapted.values(), create_graph=True)
            adapted = {n: p - self.inner_lr * g
                      for (n, p), g in zip(adapted.items(), grads)}

        # Evaluate on query set
        return self._compute_loss(query_set, adapted)

    def meta_train_step(self, tasks: list[tuple[list, list]]) -> float:
        """Meta-training across multiple tasks."""
        self.meta_optimizer.zero_grad()

        total_loss = sum(self.adapt(support, query) for support, query in tasks)
        total_loss /= len(tasks)

        total_loss.backward()
        self.meta_optimizer.step()

        return total_loss.item()
```

#### 6.3 Differentiable Memory Retrieval
```python
# src/t4dm/learning/differentiable.py

class DifferentiableRetrieval(nn.Module):
    """
    Soft attention over memory for end-to-end learning.
    Gradients flow through retrieval decisions.
    """

    def __init__(self, embed_dim: int = 1024, num_heads: int = 8):
        super().__init__()
        self.query_proj = nn.Linear(embed_dim, embed_dim)
        self.key_proj = nn.Linear(embed_dim, embed_dim)
        self.value_proj = nn.Linear(embed_dim, embed_dim)
        self.attention = nn.MultiheadAttention(embed_dim, num_heads, batch_first=True)
        self.gate = nn.Linear(embed_dim, 1)

    def forward(self, query: torch.Tensor, memories: torch.Tensor,
                mask: torch.Tensor = None) -> tuple[torch.Tensor, torch.Tensor]:
        """
        query: [batch, embed_dim]
        memories: [batch, n_memories, embed_dim]
        Returns: (retrieved, attention_weights)
        """
        q = self.query_proj(query).unsqueeze(1)
        k = self.key_proj(memories)
        v = self.value_proj(memories)

        retrieved, weights = self.attention(q, k, v, key_padding_mask=mask)

        # Gate: should we use memory at all?
        gate = torch.sigmoid(self.gate(query))
        retrieved = gate * retrieved.squeeze(1)

        return retrieved, weights.squeeze(1)
```

#### 6.4 Multi-Step Credit Graph
```python
# src/t4dm/learning/credit_graph.py

class MultiStepCreditGraph:
    """
    DAG for multi-step reasoning credit assignment.

    Nodes: memories, reasoning steps, outcomes
    Edges: causal influence
    Credit flows backward from outcomes.
    """

    def __init__(self):
        self.graph = nx.DiGraph()

    def add_retrieval(self, memory_id: UUID, step_id: str, weight: float):
        self.graph.add_node(f"mem_{memory_id}", type="memory", memory_id=memory_id)
        self.graph.add_node(step_id, type="step")
        self.graph.add_edge(f"mem_{memory_id}", step_id, weight=weight)

    def add_step_dependency(self, from_step: str, to_step: str):
        self.graph.add_edge(from_step, to_step, weight=1.0)

    def distribute_reward(self, final_reward: float) -> dict[UUID, float]:
        """Propagate reward backward through graph."""
        # Add outcome node connected to all leaf steps
        self.graph.add_node("outcome", type="outcome", reward=final_reward)
        for node in self.graph.nodes():
            if self.graph.out_degree(node) == 0 and node != "outcome":
                self.graph.add_edge(node, "outcome")

        # Backward pass
        credits = {"outcome": final_reward}
        for node in reversed(list(nx.topological_sort(self.graph))):
            if node == "outcome":
                continue
            for child in self.graph.successors(node):
                edge_weight = self.graph.edges[node, child].get("weight", 1.0)
                in_degree = self.graph.in_degree(child)
                credits[node] = credits.get(node, 0) + credits[child] * edge_weight / in_degree

        # Extract memory credits
        return {self.graph.nodes[n]["memory_id"]: credits[n]
                for n in self.graph.nodes()
                if self.graph.nodes[n].get("type") == "memory"}
```

### Exit Criteria
- EWC preventing forgetting of old domains
- MAML enabling fast adaptation to new projects
- Differentiable retrieval gradients flowing
- Multi-step credit properly attributed

---

## Phase 7: Testing & Validation
**Duration**: 5-7 days | **Complexity**: Medium | **Prerequisite**: Phase 5

### Tasks

#### 7.1 Unit Tests for Learning Components
- [ ] Test reward computation formulas
- [ ] Test eligibility trace decay
- [ ] Test replay buffer sampling
- [ ] Test ListMLE loss gradients
- [ ] Test EWC penalty computation

#### 7.2 Integration Tests
- [ ] Test full retrieval→outcome→reward flow
- [ ] Test session start/end hooks
- [ ] Test offline training pipeline
- [ ] Test checkpoint save/restore

#### 7.3 Learning Validation
- [ ] Verify weights change with training
- [ ] Verify ranking improves on held-out data
- [ ] Verify no regression on old tasks (EWC)
- [ ] A/B test: learned vs fixed weights

#### 7.4 Performance Benchmarks
- [ ] Retrieval latency with learned scorer
- [ ] Training step throughput
- [ ] Memory footprint of replay buffer
- [ ] End-to-end session overhead

### Exit Criteria
- 90%+ coverage on learning module
- All integration tests passing
- Demonstrated learning (weights change, ranking improves)
- No performance regression

---

## Phase 8: Documentation & Release
**Duration**: 3-5 days | **Complexity**: Low | **Prerequisite**: Phase 7

### Tasks

#### 8.1 Documentation Updates
- [ ] Update README with learning system description
- [ ] Update ARCHITECTURE.md to reflect current state
- [ ] Write LEARNING.md explaining the learning system
- [ ] Document configuration options
- [ ] Add troubleshooting guide

#### 8.2 API Documentation
- [ ] Document learning-related MCP tools
- [ ] Document SDK learning extensions
- [ ] Document hook interfaces

#### 8.3 Release Preparation
- [ ] Version bump to 0.2.0
- [ ] Create GitHub release
- [ ] Generate DOI via Zenodo
- [ ] Update claude-skills plugin

### Exit Criteria
- All docs current
- GitHub release created
- Skills updated

---

## Master Timeline

```
Week 1:  Phase 0 (Infrastructure) + Phase 1 (Data Collection)
Week 2:  Phase 2 (Rewards & Credit)
Week 3:  Phase 3 (Learnable Components)
Week 4:  Phase 4 (CLI Integration) + Phase 5 Start
Week 5:  Phase 5 (Offline Learning)
Week 6:  Phase 6 (Advanced - EWC, MAML)
Week 7:  Phase 7 (Testing)
Week 8:  Phase 8 (Documentation) + Release
```

---

## Risk Mitigation

| Risk | Mitigation |
|------|------------|
| Learning instability | Start with low learning rate, monitor gradients |
| Catastrophic forgetting | EWC from Phase 6, experience replay |
| Latency regression | Profile, use caching, async training |
| Complexity explosion | Each phase has clear exit criteria |
| Reward hacking | Multiple reward signals, sanity checks |

---

## Success Metrics

| Metric | Current | Target |
|--------|---------|--------|
| Test Pass Rate | 96.8% | 100% |
| Code Coverage | 60% | 90%+ |
| Retrieval MRR | Unknown | +10% vs. fixed |
| User Satisfaction | N/A | Positive feedback |
| Adaptation Speed | N/A | <100 examples to new domain |

---

## Key Files to Create

```
src/t4dm/learning/
├── __init__.py
├── events.py          # Event data structures
├── collector.py       # Event collection hooks
├── baseline.py        # Task baseline tracking
├── rewards.py         # Reward computation
├── traces.py          # Eligibility traces
├── scorer.py          # Learned retrieval scorer
├── replay.py          # Experience replay buffer
├── losses.py          # Training losses
├── trainer.py         # Training loop
├── scheduler.py       # Training scheduler
├── decay.py           # Adaptive decay
├── forgetting.py      # Forgetting regret
├── ewc.py             # Elastic weight consolidation
├── maml.py            # Meta-learning
├── differentiable.py  # Differentiable retrieval
└── credit_graph.py    # Multi-step credit

~/.claude/skills/ww-learning/
├── session_start.py   # SessionStart hook
├── session_end.py     # SessionEnd hook
└── config.yaml        # Learning config
```
