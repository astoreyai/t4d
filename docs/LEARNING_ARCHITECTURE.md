# T4DM Learning Architecture
## A Complete Design for Memory That Improves With Experience

**Author**: Geoffrey Hinton (design perspective)
**Date**: 2025-12-05 | **Updated**: 2026-01-05
**Status**: Phase 2 Implementation Complete

---

## Phase 2 Implementation Status

The following components are now **implemented and wired**:

| Component | File | Status |
|-----------|------|--------|
| Three-Factor Learning Rule | `src/t4dm/learning/three_factor.py` | ✅ Wired |
| Eligibility Traces | `src/t4dm/learning/eligibility.py` | ✅ Marked on recall |
| Reconsolidation Engine | `src/t4dm/learning/reconsolidation.py` | ✅ Uses three-factor |
| Episode Feedback API | `src/t4dm/api/routes/episodes.py` | ✅ `/feedback` endpoint |
| Entity Feedback API | `src/t4dm/api/routes/entities.py` | ✅ `/feedback` endpoint |
| Learning Stats API | `src/t4dm/api/routes/episodes.py` | ✅ `/learning-stats` |
| Hebbian Semantic Updates | `src/t4dm/memory/semantic.py` | ✅ `learn_from_outcome()` |

### Learning Signal Flow (Implemented)

```
┌─────────────────────────────────────────────────────────────────┐
│                    LEARNING SIGNAL FLOW                         │
├─────────────────────────────────────────────────────────────────┤
│                                                                 │
│  Recall Query ──┬──▶ mark_active() ──▶ Eligibility Traces       │
│                 └──▶ trigger_lability() ──▶ Lability Window     │
│                                                                 │
│  ... time passes ...                                            │
│                                                                 │
│  POST /feedback ──▶ learn_from_outcome() ──┬──▶ Reconsolidate   │
│  {episode_ids,    │                        └──▶ Hebbian Update  │
│   outcome_score}  │                                             │
│                   ▼                                             │
│              Three-Factor: eligibility × neuromod × dopamine    │
│                                                                 │
└─────────────────────────────────────────────────────────────────┘
```

### API Usage Example

```bash
# 1. Recall memories (eligibility traces auto-marked)
curl -X POST /api/v1/episodes/recall \
  -d '{"query": "how to fix authentication bug"}'
# Returns: episode_ids, scores

# 2. Later, after task outcome is known
curl -X POST /api/v1/episodes/feedback \
  -d '{
    "episode_ids": ["uuid1", "uuid2"],
    "query": "how to fix authentication bug",
    "outcome_score": 0.9
  }'
# Returns: reconsolidated_count, avg_rpe, three_factor_stats

# 3. Check learning statistics
curl -X GET /api/v1/episodes/learning-stats
# Returns: reconsolidation stats, dopamine stats, three_factor stats
```

---

## Original Design Specification (Hinton)

---

## Diagnosis: Why "Fuller, Not Smarter"

Your system has the *infrastructure* for learning but not the *objectives*. Let me be precise about what's missing:

1. **No contrastive signal**: You track success rates, but success at *what*? A skill that succeeds at trivial tasks isn't valuable. You need to know if a retrieval *improved outcomes compared to alternatives*.

2. **No credit assignment**: When a task succeeds, which retrieved memories contributed? You strengthen co-retrieved entities (Hebbian), but this is *correlation*, not *causation*.

3. **No delayed reward handling**: A memory retrieved now may prove useful hours later. Your current system has no mechanism to trace this.

4. **Static scoring weights**: Your retrieval weights (semantic=0.4, recency=0.25, etc.) are hand-tuned constants. These should be *learned parameters*.

5. **Passive decay, not active forgetting**: FSRS decays memories based on access patterns, but this doesn't *actively* learn what to forget.

---

## Part 1: Reward Functions

### 1.1 Retrieval Utility Reward (R_retrieval)

The key insight: **Utility is counterfactual**. A retrieval is useful if the outcome *with* that memory was better than the expected outcome *without* it.

```python
@dataclass
class RetrievalEvent:
    """Track a single retrieval for later reward attribution."""
    retrieval_id: UUID
    query: str
    retrieved_memory_ids: list[UUID]
    retrieval_scores: dict[UUID, float]  # Original retrieval scores
    timestamp: datetime
    context_hash: str  # Hash of task context for matching

@dataclass
class OutcomeEvent:
    """Outcome that may be attributable to prior retrievals."""
    outcome_id: UUID
    success_score: float  # [0, 1]
    task_context_hash: str
    timestamp: datetime
    explicit_memory_refs: list[UUID]  # Memories explicitly cited by LLM

class RetrievalRewardComputer:
    """
    Compute rewards for retrievals based on subsequent outcomes.

    Key insight: We can't know the counterfactual (what would have happened
    without this memory), but we can estimate it using:
    1. Baseline success rate for similar tasks
    2. Leave-one-out analysis on multi-memory retrievals
    3. Explicit citations from the LLM
    """

    def __init__(self, baseline_window_days: int = 30):
        self.baseline_window = baseline_window_days
        self.task_baselines: dict[str, RunningMean] = {}
        self.pending_retrievals: dict[str, list[RetrievalEvent]] = {}

    async def compute_reward(
        self,
        retrieval: RetrievalEvent,
        outcome: OutcomeEvent,
        time_delay_hours: float,
    ) -> dict[UUID, float]:
        """
        Compute per-memory reward from an outcome.

        Returns: {memory_id: reward} where reward in [-1, 1]
        """
        rewards = {}

        # Get baseline for this task type
        baseline = self._get_task_baseline(outcome.task_context_hash)

        # Counterfactual advantage: how much better than expected?
        advantage = outcome.success_score - baseline

        # Time discount: delayed outcomes give weaker signal
        # Using hyperbolic discounting (more cognitively realistic than exponential)
        time_discount = 1.0 / (1.0 + 0.1 * time_delay_hours)

        for memory_id in retrieval.retrieved_memory_ids:
            # Base reward from advantage
            reward = advantage * time_discount

            # Boost if explicitly cited by LLM
            if memory_id in outcome.explicit_memory_refs:
                reward *= 1.5

            # Normalize by number of retrieved memories (credit splitting)
            # Use retrieval score as attention weight
            total_score = sum(retrieval.retrieval_scores.values())
            attention_weight = retrieval.retrieval_scores[memory_id] / total_score

            rewards[memory_id] = reward * attention_weight

        # Update baseline
        self._update_task_baseline(outcome.task_context_hash, outcome.success_score)

        return rewards

    def _get_task_baseline(self, context_hash: str) -> float:
        """Get expected success rate for similar tasks."""
        if context_hash in self.task_baselines:
            return self.task_baselines[context_hash].mean
        # Default baseline: 0.5 (no prior expectation)
        return 0.5
```

### 1.2 Consolidation Reward (R_consolidation)

Entities and skills extracted from episodes should be rewarded based on *downstream utility*, not just extraction confidence.

```python
@dataclass
class ConsolidationEvent:
    """Track entity/skill extraction for reward attribution."""
    source_episode_ids: list[UUID]
    extracted_id: UUID  # Entity or Procedure ID
    extraction_type: str  # "entity" or "skill"
    extraction_confidence: float
    timestamp: datetime

class ConsolidationRewardComputer:
    """
    Compute rewards for consolidation decisions.

    Key insight: An extracted entity is valuable if:
    1. It gets retrieved in future queries (used knowledge)
    2. Those retrievals lead to positive outcomes (useful knowledge)
    3. It participates in productive spreading activation
    """

    def __init__(self, evaluation_window_days: int = 7):
        self.window = evaluation_window_days

    async def evaluate_entity(
        self,
        entity_id: UUID,
        since_creation: datetime,
    ) -> float:
        """
        Evaluate entity quality based on usage patterns.

        Returns: Quality score in [0, 1]
        """
        # Metric 1: Retrieval frequency (is it being used?)
        retrieval_count = await self._count_retrievals(entity_id, since_creation)
        retrieval_score = 1 - math.exp(-retrieval_count / 5)  # Saturates around 5

        # Metric 2: Outcome-weighted retrieval (is it useful?)
        weighted_outcomes = await self._get_outcome_weighted_retrievals(
            entity_id, since_creation
        )
        utility_score = weighted_outcomes  # Already in [0, 1]

        # Metric 3: Graph centrality (is it connected to useful things?)
        centrality = await self._compute_pagerank_contribution(entity_id)
        centrality_score = min(centrality * 10, 1.0)  # Normalize

        # Metric 4: Specificity (is it precise, not generic?)
        # Inverse of fan-out: entities with fewer, stronger connections are better
        fan_out = await self._get_fan_out(entity_id)
        specificity_score = 1.0 / (1.0 + math.log(1 + fan_out))

        # Combined score with learned weights (initially equal)
        return (
            0.3 * retrieval_score +
            0.4 * utility_score +
            0.15 * centrality_score +
            0.15 * specificity_score
        )

    async def evaluate_skill(
        self,
        procedure_id: UUID,
        since_creation: datetime,
    ) -> float:
        """
        Evaluate skill quality beyond simple success rate.

        Key insight: A skill isn't just good if it succeeds, but if it:
        1. Succeeds at *difficult* tasks (not just easy ones)
        2. Generalizes across contexts
        3. Improves over baseline (what would happen without it)
        """
        procedure = await self._get_procedure(procedure_id)

        # Metric 1: Difficulty-adjusted success rate
        # Weight successes by task difficulty
        executions = await self._get_executions(procedure_id, since_creation)
        if not executions:
            return 0.0

        difficulty_weighted_success = sum(
            (1 if e.success else 0) * e.estimated_difficulty
            for e in executions
        ) / sum(e.estimated_difficulty for e in executions)

        # Metric 2: Context diversity (does it work across projects/tools?)
        unique_contexts = len(set(e.context_hash for e in executions))
        diversity_score = 1 - math.exp(-unique_contexts / 3)

        # Metric 3: Improvement over time (is it getting better?)
        if len(executions) >= 5:
            first_half = executions[:len(executions)//2]
            second_half = executions[len(executions)//2:]
            improvement = (
                mean(e.success for e in second_half) -
                mean(e.success for e in first_half)
            )
            improvement_score = (improvement + 1) / 2  # Normalize to [0, 1]
        else:
            improvement_score = 0.5  # Neutral

        return (
            0.5 * difficulty_weighted_success +
            0.3 * diversity_score +
            0.2 * improvement_score
        )
```

### 1.3 Forgetting Reward (R_forget)

This is the hardest: how do we know forgetting was right?

```python
class ForgettingRewardComputer:
    """
    Evaluate whether forgetting decisions were beneficial.

    Key insight: We can't know if we should have kept a forgotten memory,
    but we CAN detect when we're missing something we used to know.

    Signals:
    1. User re-provides information we previously forgot
    2. Tasks fail with queries that would have matched forgotten content
    3. System asks questions whose answers we previously knew
    """

    def __init__(self):
        # Keep a "forgetting ledger" - lightweight records of what was forgotten
        self.forgotten_records: list[ForgottenRecord] = []

    @dataclass
    class ForgottenRecord:
        memory_id: UUID
        memory_type: str
        forgotten_at: datetime
        content_signature: str  # Hash/embedding of content
        retrieval_count_at_death: int
        last_successful_use: Optional[datetime]

    async def detect_regret(
        self,
        new_content: str,
        context: str,
    ) -> Optional[ForgottenRecord]:
        """
        Check if new content matches something we forgot.

        Returns: The forgotten record if we're re-learning something, None otherwise.
        """
        new_signature = await self._compute_signature(new_content)

        for record in self.forgotten_records:
            similarity = self._signature_similarity(new_signature, record.content_signature)
            if similarity > 0.85:  # High match
                # We forgot something we're now re-learning
                time_since_forgotten = datetime.now() - record.forgotten_at

                # Record this as a forgetting mistake
                await self._record_forgetting_mistake(
                    record=record,
                    regret_delay=time_since_forgotten,
                    similarity=similarity,
                )

                return record

        return None

    def compute_forgetting_reward(
        self,
        memory_id: UUID,
        time_since_forgotten: timedelta,
        was_regretted: bool,
    ) -> float:
        """
        Reward for a forgetting decision.

        Returns: Reward in [-1, 1]
        - Positive if we correctly forgot (didn't need it)
        - Negative if we shouldn't have forgotten (needed it later)
        """
        if was_regretted:
            # We forgot something useful
            # Penalty scales with how quickly we regretted it
            days = time_since_forgotten.total_seconds() / 86400
            # Faster regret = worse decision
            return -1.0 / (1.0 + days / 7)  # Worst is -1 if regretted same day
        else:
            # We forgot something and didn't need it
            # Small positive reward that grows with time
            days = time_since_forgotten.total_seconds() / 86400
            return min(0.1 + 0.01 * days, 0.5)  # Max +0.5 after 40 days
```

---

## Part 2: Learning Objectives

### 2.1 Retrieval Scoring Weights

**What to learn**: The weights combining semantic similarity, recency, outcome, importance, activation, retrievability.

**Parameterization**:
```python
class LearnedRetrievalScorer:
    """
    Learn optimal retrieval weights from experience.

    Current: Fixed weights (semantic=0.4, recency=0.25, outcome=0.2, importance=0.15)
    Target: Learned weights that maximize downstream task success
    """

    def __init__(self):
        # Learnable parameters (initialized to current hand-tuned values)
        self.weights = nn.Parameter(torch.tensor([
            0.40,  # semantic
            0.25,  # recency
            0.20,  # outcome
            0.15,  # importance
        ]))

        # Temperature for softmax over candidates
        self.temperature = nn.Parameter(torch.tensor(1.0))

        # Per-memory-type weight adjustments
        self.type_scales = nn.ParameterDict({
            'episodic': nn.Parameter(torch.tensor(1.0)),
            'semantic': nn.Parameter(torch.tensor(1.0)),
            'procedural': nn.Parameter(torch.tensor(1.0)),
        })

    def score(
        self,
        components: dict[str, float],
        memory_type: str,
    ) -> float:
        """Compute learned retrieval score."""
        # Softmax to ensure weights sum to 1 and are positive
        normalized_weights = F.softmax(self.weights, dim=0)

        component_tensor = torch.tensor([
            components.get('semantic', 0),
            components.get('recency', 0),
            components.get('outcome', 0),
            components.get('importance', 0),
        ])

        base_score = torch.dot(normalized_weights, component_tensor)
        type_scale = self.type_scales[memory_type]

        return (base_score * type_scale).item()
```

**Loss Function**:
```python
def retrieval_ranking_loss(
    retrieved_scores: list[float],
    memory_rewards: list[float],
) -> torch.Tensor:
    """
    ListMLE loss: learn to rank memories by their eventual utility.

    Idea: Memories that led to better outcomes should have been ranked higher.
    This is a listwise learning-to-rank objective.
    """
    # Sort by reward to get target ranking
    target_order = sorted(range(len(memory_rewards)),
                          key=lambda i: memory_rewards[i], reverse=True)

    scores = torch.tensor(retrieved_scores)

    # ListMLE loss
    loss = 0.0
    remaining = set(range(len(scores)))

    for target_idx in target_order:
        if target_idx not in remaining:
            continue

        # Log probability that target_idx is selected from remaining
        remaining_scores = torch.tensor([scores[i] for i in remaining])
        target_score = scores[target_idx]

        log_prob = target_score - torch.logsumexp(remaining_scores, dim=0)
        loss -= log_prob
        remaining.remove(target_idx)

    return loss / len(target_order)
```

**Training Signal**: Retrieval rewards computed as above.

**Update Frequency**: Mini-batch every 100 retrieval-outcome pairs, or daily.

### 2.2 Consolidation Thresholds

**What to learn**: When to consolidate episodic memories into semantic entities.

```python
class LearnedConsolidationPolicy:
    """
    Learn when to extract entities and create skills.

    Current: Fixed thresholds (min_episode_count=3, pattern_strength>=2)
    Target: Adaptive thresholds based on entity/skill quality feedback
    """

    def __init__(self):
        # Pattern detection threshold (log scale for numerical stability)
        self.log_min_occurrences = nn.Parameter(torch.tensor(math.log(3)))

        # Confidence threshold for extraction
        self.extraction_threshold = nn.Parameter(torch.tensor(0.7))

        # Skill success threshold
        self.skill_success_threshold = nn.Parameter(torch.tensor(0.7))

        # Context-dependent adjustments
        self.domain_adjustments = nn.ParameterDict({
            domain.value: nn.Parameter(torch.tensor(0.0))
            for domain in Domain
        })

    def should_consolidate(
        self,
        occurrence_count: int,
        extraction_confidence: float,
        domain: str,
        context_diversity: float,
    ) -> tuple[bool, float]:
        """
        Decide whether to consolidate.

        Returns: (should_consolidate, confidence)
        """
        min_occurrences = math.exp(self.log_min_occurrences.item())
        threshold = self.extraction_threshold.item()
        domain_adj = self.domain_adjustments.get(domain, torch.tensor(0.0)).item()

        # Adjust threshold based on domain
        adjusted_threshold = threshold + domain_adj

        # Bonus for diverse contexts (generalization signal)
        diversity_bonus = 0.1 * context_diversity
        effective_confidence = extraction_confidence + diversity_bonus

        should = (
            occurrence_count >= min_occurrences and
            effective_confidence >= adjusted_threshold
        )

        return should, effective_confidence
```

**Loss Function**:
```python
def consolidation_policy_loss(
    consolidation_decisions: list[tuple[bool, float, float]],
    # (was_consolidated, decision_confidence, eventual_quality)
) -> torch.Tensor:
    """
    Learn to consolidate when it leads to high-quality entities.

    Binary cross-entropy where the target is whether the entity
    turned out to be high-quality.
    """
    losses = []

    for was_consolidated, confidence, quality in consolidation_decisions:
        if was_consolidated:
            # We consolidated. Was it good?
            # High quality entity = good decision
            target = 1.0 if quality > 0.5 else 0.0
            pred = torch.sigmoid(torch.tensor(confidence))
            loss = F.binary_cross_entropy(pred, torch.tensor(target))
        else:
            # We didn't consolidate. Would it have been valuable?
            # We don't know quality of unconsolidated entities,
            # so use confidence as proxy
            # Low confidence decisions we didn't take = small positive reward
            loss = -0.1 * (1 - confidence)  # Reward for conservative skipping

        losses.append(loss)

    return torch.stack(losses).mean()
```

### 2.3 Decay Rate Adaptation

**What to learn**: Different memory types should decay at different rates.

```python
class AdaptiveDecay:
    """
    Learn optimal decay rates per memory type and domain.

    Current: Fixed FSRS decay factor (0.9)
    Target: Adaptive decay based on memory utility lifetime
    """

    def __init__(self):
        # Base decay rate (log scale)
        self.log_base_decay = nn.Parameter(torch.tensor(math.log(0.9)))

        # Per-type adjustments
        self.type_adjustments = nn.ParameterDict({
            'episodic': nn.Parameter(torch.tensor(0.0)),
            'semantic': nn.Parameter(torch.tensor(0.2)),   # Slower decay
            'procedural': nn.Parameter(torch.tensor(0.1)), # Medium
        })

        # Per-domain adjustments (some domains need longer memory)
        self.domain_adjustments = nn.ParameterDict({
            'coding': nn.Parameter(torch.tensor(0.0)),
            'research': nn.Parameter(torch.tensor(0.1)),  # Research needs longer memory
            'trading': nn.Parameter(torch.tensor(-0.1)), # Trading needs freshness
        })

    def get_decay_rate(
        self,
        memory_type: str,
        domain: Optional[str] = None,
    ) -> float:
        """Get adaptive decay rate."""
        base = math.exp(self.log_base_decay.item())
        type_adj = self.type_adjustments.get(memory_type, torch.tensor(0.0)).item()
        domain_adj = self.domain_adjustments.get(domain, torch.tensor(0.0)).item() if domain else 0.0

        # Combine multiplicatively
        return base * math.exp(type_adj + domain_adj)
```

**Loss Function**:
```python
def decay_optimization_loss(
    forgotten_memories: list[tuple[float, float, bool]],
    # (decay_rate_used, time_to_next_access, was_accessed_again)
) -> torch.Tensor:
    """
    Optimize decay rate to minimize:
    1. Keeping useless memories (wasted storage)
    2. Forgetting useful memories (regret)

    This is a survival analysis problem.
    """
    losses = []

    for decay_rate, time_to_access, was_accessed in forgotten_memories:
        if was_accessed:
            # Memory was accessed again - should have decayed slower
            # Loss proportional to how much we decayed it
            predicted_retrievability = (1 + decay_rate * time_to_access) ** -0.5
            # We want high retrievability for accessed memories
            loss = -torch.log(torch.tensor(predicted_retrievability + 1e-6))
        else:
            # Memory was never accessed again - decay was appropriate
            # Small reward for efficient forgetting
            predicted_retrievability = (1 + decay_rate * time_to_access) ** -0.5
            # Low retrievability for unaccessed = good
            loss = -0.1 * torch.log(1 - torch.tensor(predicted_retrievability) + 1e-6)

        losses.append(loss)

    return torch.stack(losses).mean()
```

---

## Part 3: Credit Assignment

The fundamental challenge: **When a task succeeds, which memories contributed?**

### 3.1 Attention-Based Credit

Track how the LLM "attends" to retrieved memories.

```python
class MemoryAttentionTracker:
    """
    Track which retrieved memories the LLM actually uses.

    Method: Analyze the LLM's response for references to retrieved content.
    """

    async def compute_attention_weights(
        self,
        retrieved_memories: list[ScoredResult],
        llm_response: str,
        task_context: str,
    ) -> dict[UUID, float]:
        """
        Estimate attention weights for each retrieved memory.

        Uses:
        1. Text overlap between memory and response
        2. Semantic similarity to response
        3. Explicit citations/references
        """
        weights = {}

        for result in retrieved_memories:
            memory = result.item

            # Method 1: N-gram overlap
            memory_ngrams = self._extract_ngrams(memory.content, n=3)
            response_ngrams = self._extract_ngrams(llm_response, n=3)
            overlap = len(memory_ngrams & response_ngrams) / len(memory_ngrams)

            # Method 2: Semantic similarity
            memory_emb = await self._embed(memory.content)
            response_emb = await self._embed(llm_response)
            semantic_sim = cosine_similarity(memory_emb, response_emb)

            # Method 3: Explicit citation detection
            # Look for phrases like "as mentioned in..." or quotes from memory
            citation_score = self._detect_citations(memory.content, llm_response)

            # Combine
            weights[memory.id] = (
                0.3 * overlap +
                0.4 * semantic_sim +
                0.3 * citation_score
            )

        # Normalize to sum to 1
        total = sum(weights.values())
        if total > 0:
            weights = {k: v/total for k, v in weights.items()}

        return weights
```

### 3.2 Temporal Credit Assignment

Handle the delay between retrieval and outcome.

```python
class TemporalCreditAssigner:
    """
    Assign credit across time using eligibility traces.

    Key insight: Use TD(lambda) style traces where:
    - Recent retrievals get more credit
    - Traces decay exponentially
    - Outcomes refresh traces for still-active memories
    """

    def __init__(self, gamma: float = 0.99, lambda_: float = 0.9):
        self.gamma = gamma  # Discount factor
        self.lambda_ = lambda_  # Trace decay

        # Eligibility traces: {memory_id: trace_value}
        self.traces: dict[UUID, float] = defaultdict(float)

        # Last retrieval time per memory
        self.last_retrieved: dict[UUID, datetime] = {}

    def on_retrieval(
        self,
        memory_id: UUID,
        retrieval_score: float,
        timestamp: datetime,
    ):
        """Update trace when memory is retrieved."""
        # Decay all existing traces
        self._decay_traces(timestamp)

        # Boost trace for retrieved memory
        # Higher retrieval score = stronger trace
        self.traces[memory_id] += retrieval_score
        self.last_retrieved[memory_id] = timestamp

    def on_outcome(
        self,
        reward: float,
        timestamp: datetime,
    ) -> dict[UUID, float]:
        """
        Distribute reward based on eligibility traces.

        Returns: Credit assigned to each memory
        """
        # Decay traces to current time
        self._decay_traces(timestamp)

        # Distribute reward proportional to traces
        total_trace = sum(self.traces.values())
        if total_trace == 0:
            return {}

        credits = {}
        for memory_id, trace in self.traces.items():
            credits[memory_id] = reward * (trace / total_trace)

        return credits

    def _decay_traces(self, current_time: datetime):
        """Decay all traces based on elapsed time."""
        decayed = {}

        for memory_id, trace in self.traces.items():
            last_time = self.last_retrieved.get(memory_id, current_time)
            elapsed_hours = (current_time - last_time).total_seconds() / 3600

            # Exponential decay
            decay_factor = (self.gamma * self.lambda_) ** elapsed_hours
            new_trace = trace * decay_factor

            if new_trace > 0.01:  # Threshold
                decayed[memory_id] = new_trace

        self.traces = decayed
```

### 3.3 Multi-Step Reasoning Credit

For complex tasks involving multiple retrieval rounds:

```python
class MultiStepCreditGraph:
    """
    Build a DAG of memory usage for multi-step reasoning.

    Nodes: Retrieved memories and intermediate reasoning steps
    Edges: Causal influence (memory X informed step Y)

    Credit flows backward from outcome through the graph.
    """

    def __init__(self):
        self.graph = nx.DiGraph()
        self.current_session_id: Optional[str] = None

    def start_session(self, session_id: str):
        """Start tracking a new reasoning session."""
        self.current_session_id = session_id
        self.graph.add_node(
            f"session_{session_id}",
            type="session",
            timestamp=datetime.now(),
        )

    def add_retrieval(
        self,
        memory_id: UUID,
        retrieval_context: str,
        attention_weight: float,
    ):
        """Record a retrieval during reasoning."""
        node_id = f"mem_{memory_id}_{datetime.now().timestamp()}"
        self.graph.add_node(
            node_id,
            type="memory",
            memory_id=memory_id,
            weight=attention_weight,
            timestamp=datetime.now(),
        )

        # Connect to current session
        self.graph.add_edge(
            node_id,
            f"session_{self.current_session_id}",
            influence=attention_weight,
        )

    def add_intermediate_step(
        self,
        step_id: str,
        step_type: str,  # "reasoning", "tool_use", "generation"
        contributing_memories: list[UUID],
    ):
        """Record an intermediate reasoning step."""
        self.graph.add_node(
            step_id,
            type="step",
            step_type=step_type,
            timestamp=datetime.now(),
        )

        # Connect contributing memories
        for mem_id in contributing_memories:
            # Find most recent retrieval of this memory
            mem_nodes = [
                n for n in self.graph.nodes()
                if self.graph.nodes[n].get("memory_id") == mem_id
            ]
            if mem_nodes:
                latest = max(mem_nodes,
                            key=lambda n: self.graph.nodes[n]["timestamp"])
                self.graph.add_edge(latest, step_id)

    def distribute_reward(self, final_reward: float) -> dict[UUID, float]:
        """
        Propagate reward backward through the graph.

        Uses PageRank-style propagation where:
        - Reward flows backward from outcome
        - Each node keeps some, passes some to parents
        - Memory nodes accumulate final credit
        """
        # Add outcome node
        outcome_node = f"outcome_{self.current_session_id}"
        self.graph.add_node(outcome_node, type="outcome", reward=final_reward)

        # Connect to all leaf nodes (steps/memories with no children)
        for node in self.graph.nodes():
            if self.graph.out_degree(node) == 0 and node != outcome_node:
                self.graph.add_edge(node, outcome_node)

        # Backward propagation
        credit = {outcome_node: final_reward}

        # Reverse topological order
        for node in reversed(list(nx.topological_sort(self.graph))):
            if node == outcome_node:
                continue

            # Receive credit from children
            for child in self.graph.successors(node):
                edge_weight = self.graph.edges[node, child].get("influence", 1.0)
                num_parents = self.graph.in_degree(child)
                credit[node] = credit.get(node, 0) + credit[child] * edge_weight / num_parents

        # Aggregate credit by memory_id
        memory_credits = defaultdict(float)
        for node in self.graph.nodes():
            if self.graph.nodes[node].get("type") == "memory":
                mem_id = self.graph.nodes[node]["memory_id"]
                memory_credits[mem_id] += credit.get(node, 0)

        return dict(memory_credits)
```

---

## Part 4: Architecture Changes

### 4.1 Differentiable Memory Operations

Make retrieval differentiable for end-to-end training:

```python
class DifferentiableMemoryRetrieval(nn.Module):
    """
    Soft attention over memory store for gradient-based learning.

    Instead of hard k-NN retrieval, use soft attention that allows
    gradients to flow through retrieval decisions.
    """

    def __init__(self, embed_dim: int = 1024, num_heads: int = 8):
        super().__init__()

        # Query transformation
        self.query_proj = nn.Linear(embed_dim, embed_dim)

        # Memory key/value projections
        self.key_proj = nn.Linear(embed_dim, embed_dim)
        self.value_proj = nn.Linear(embed_dim, embed_dim)

        # Multi-head attention
        self.attention = nn.MultiheadAttention(embed_dim, num_heads, batch_first=True)

        # Gating mechanism for retrieval vs. no-retrieval
        self.retrieval_gate = nn.Linear(embed_dim, 1)

    def forward(
        self,
        query_embedding: torch.Tensor,  # [batch, embed_dim]
        memory_embeddings: torch.Tensor,  # [batch, num_memories, embed_dim]
        memory_mask: Optional[torch.Tensor] = None,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """
        Soft retrieval with attention.

        Returns:
            retrieved: Weighted combination of memories [batch, embed_dim]
            attention_weights: Retrieval weights [batch, num_memories]
        """
        # Project query and memories
        query = self.query_proj(query_embedding).unsqueeze(1)  # [batch, 1, embed_dim]
        keys = self.key_proj(memory_embeddings)
        values = self.value_proj(memory_embeddings)

        # Attend over memories
        retrieved, attention_weights = self.attention(
            query, keys, values,
            key_padding_mask=memory_mask,
        )

        # Compute retrieval gate (should we use memory at all?)
        gate = torch.sigmoid(self.retrieval_gate(query_embedding))

        # Gated output
        retrieved = gate * retrieved.squeeze(1)

        return retrieved, attention_weights.squeeze(1)
```

### 4.2 Experience Replay Buffer

Store and replay experiences for offline learning:

```python
@dataclass
class Experience:
    """A single learning experience."""
    experience_id: UUID
    timestamp: datetime

    # Retrieval context
    query: str
    query_embedding: list[float]
    retrieved_memory_ids: list[UUID]
    retrieval_scores: list[float]
    attention_weights: list[float]

    # Outcome
    outcome_score: float
    outcome_timestamp: datetime

    # Credit assignment
    per_memory_credit: dict[str, float]  # UUID string -> credit

    # Meta
    session_id: str
    task_type: str

class ExperienceReplayBuffer:
    """
    Prioritized experience replay for offline learning.

    Key features:
    1. Priority based on TD error (surprise)
    2. Importance sampling for unbiased updates
    3. Memory-aware sampling (include related experiences)
    """

    def __init__(
        self,
        capacity: int = 100000,
        alpha: float = 0.6,  # Priority exponent
        beta: float = 0.4,   # Importance sampling
    ):
        self.buffer = []
        self.priorities = []
        self.capacity = capacity
        self.alpha = alpha
        self.beta = beta
        self.position = 0

    def add(self, experience: Experience, td_error: float):
        """Add experience with priority based on TD error."""
        priority = (abs(td_error) + 1e-6) ** self.alpha

        if len(self.buffer) < self.capacity:
            self.buffer.append(experience)
            self.priorities.append(priority)
        else:
            self.buffer[self.position] = experience
            self.priorities[self.position] = priority

        self.position = (self.position + 1) % self.capacity

    def sample(self, batch_size: int) -> tuple[list[Experience], list[float]]:
        """
        Sample batch with prioritized probabilities.

        Returns: (experiences, importance_weights)
        """
        if len(self.buffer) == 0:
            return [], []

        priorities = np.array(self.priorities)
        probabilities = priorities / priorities.sum()

        indices = np.random.choice(
            len(self.buffer),
            size=min(batch_size, len(self.buffer)),
            p=probabilities,
            replace=False,
        )

        # Importance sampling weights
        N = len(self.buffer)
        weights = (N * probabilities[indices]) ** -self.beta
        weights = weights / weights.max()  # Normalize

        experiences = [self.buffer[i] for i in indices]

        return experiences, weights.tolist()

    def sample_memory_related(
        self,
        memory_id: UUID,
        batch_size: int,
    ) -> list[Experience]:
        """Sample experiences involving a specific memory."""
        related = [
            exp for exp in self.buffer
            if memory_id in exp.retrieved_memory_ids
        ]

        if len(related) <= batch_size:
            return related

        return random.sample(related, batch_size)
```

### 4.3 Meta-Learning for Rapid Adaptation

```python
class MAMLMemoryAdapter:
    """
    Model-Agnostic Meta-Learning for memory system adaptation.

    Goal: Learn to adapt retrieval weights quickly to new domains/users.

    Structure:
    - Outer loop: Learn good initialization across many tasks
    - Inner loop: Fast adaptation to specific task with few examples
    """

    def __init__(
        self,
        base_scorer: LearnedRetrievalScorer,
        inner_lr: float = 0.01,
        outer_lr: float = 0.001,
        inner_steps: int = 5,
    ):
        self.scorer = base_scorer
        self.inner_lr = inner_lr
        self.outer_lr = outer_lr
        self.inner_steps = inner_steps

        # Meta-optimizer for outer loop
        self.meta_optimizer = torch.optim.Adam(
            self.scorer.parameters(),
            lr=outer_lr,
        )

    def adapt(
        self,
        support_set: list[Experience],
        query_set: list[Experience],
    ) -> float:
        """
        Adapt to a new task and evaluate.

        Args:
            support_set: Few examples for adaptation
            query_set: Examples for evaluation

        Returns:
            Loss on query set after adaptation
        """
        # Clone parameters for inner loop
        adapted_params = {
            name: param.clone()
            for name, param in self.scorer.named_parameters()
        }

        # Inner loop: adapt on support set
        for _ in range(self.inner_steps):
            support_loss = self._compute_loss(support_set, adapted_params)

            # Compute gradients
            grads = torch.autograd.grad(
                support_loss,
                adapted_params.values(),
                create_graph=True,  # For second-order gradients
            )

            # Update adapted parameters
            adapted_params = {
                name: param - self.inner_lr * grad
                for (name, param), grad in zip(adapted_params.items(), grads)
            }

        # Evaluate on query set with adapted parameters
        query_loss = self._compute_loss(query_set, adapted_params)

        return query_loss

    def meta_train_step(
        self,
        task_batch: list[tuple[list[Experience], list[Experience]]],
    ):
        """
        Meta-training step across multiple tasks.

        Each task is (support_set, query_set).
        """
        self.meta_optimizer.zero_grad()

        total_loss = 0.0
        for support_set, query_set in task_batch:
            query_loss = self.adapt(support_set, query_set)
            total_loss += query_loss

        total_loss /= len(task_batch)
        total_loss.backward()
        self.meta_optimizer.step()

        return total_loss.item()
```

---

## Part 5: Integration Points for Claude Code CLI

### 5.1 Automatic Data Collection

```python
class WWLearningDataCollector:
    """
    Collect learning data from Claude Code CLI sessions.

    Integrates via hooks that fire on:
    1. Memory retrieval
    2. Task completion
    3. User feedback (explicit and implicit)
    """

    # Event types to collect
    RETRIEVAL_EVENT = "retrieval"
    OUTCOME_EVENT = "outcome"
    FEEDBACK_EVENT = "feedback"
    CONSOLIDATION_EVENT = "consolidation"

    def __init__(self, storage_path: Path):
        self.storage = storage_path
        self.current_session: Optional[str] = None
        self.session_events: list[dict] = []

    async def on_retrieval(
        self,
        query: str,
        results: list[ScoredResult],
        context: dict,
    ):
        """Hook: Called when memories are retrieved."""
        event = {
            "type": self.RETRIEVAL_EVENT,
            "timestamp": datetime.now().isoformat(),
            "query": query,
            "retrieved": [
                {
                    "memory_id": str(r.item.id),
                    "memory_type": type(r.item).__name__,
                    "score": r.score,
                    "components": r.components,
                }
                for r in results
            ],
            "context": {
                "cwd": context.get("cwd"),
                "project": context.get("project"),
                "tool": context.get("tool"),
            },
        }

        self.session_events.append(event)

    async def on_task_complete(
        self,
        task_description: str,
        success: bool,
        error: Optional[str] = None,
        duration_seconds: float = 0,
    ):
        """Hook: Called when a task completes."""
        # Infer success score
        if success:
            success_score = 1.0
        elif error:
            success_score = 0.0
        else:
            success_score = 0.5  # Partial/unknown

        event = {
            "type": self.OUTCOME_EVENT,
            "timestamp": datetime.now().isoformat(),
            "task": task_description,
            "success": success,
            "success_score": success_score,
            "error": error,
            "duration_seconds": duration_seconds,
        }

        self.session_events.append(event)

        # Trigger reward computation asynchronously
        await self._compute_and_store_rewards()

    async def on_user_feedback(
        self,
        feedback_type: str,  # "accept", "reject", "modify", "explicit"
        content: Optional[str] = None,
    ):
        """
        Hook: Called on user feedback (explicit or inferred).

        Implicit feedback types:
        - accept: User proceeds with suggested action
        - reject: User cancels or changes approach
        - modify: User edits Claude's output

        Explicit feedback:
        - User says "that was helpful" or similar
        """
        # Convert feedback to reward signal
        feedback_rewards = {
            "accept": 0.3,
            "reject": -0.5,
            "modify": 0.0,  # Neutral - partially useful
            "explicit_positive": 0.8,
            "explicit_negative": -0.8,
        }

        event = {
            "type": self.FEEDBACK_EVENT,
            "timestamp": datetime.now().isoformat(),
            "feedback_type": feedback_type,
            "content": content,
            "implied_reward": feedback_rewards.get(feedback_type, 0.0),
        }

        self.session_events.append(event)
```

### 5.2 Implicit Feedback Detection

```python
class ImplicitFeedbackDetector:
    """
    Detect implicit feedback from user behavior.

    Signals:
    1. User accepts/rejects suggestions
    2. User edits Claude's output
    3. User repeats similar requests (memory failure?)
    4. Time between request and acceptance
    5. Follow-up questions (confusion signal)
    """

    def __init__(self):
        self.recent_requests: list[dict] = []
        self.recent_outputs: list[dict] = []

    def detect_rejection(
        self,
        claude_output: str,
        user_next_action: str,
    ) -> Optional[float]:
        """
        Detect if user rejected Claude's suggestion.

        Returns: Rejection confidence [0, 1] or None if unclear
        """
        rejection_patterns = [
            r"no,?\s+(actually|instead|let'?s)",
            r"that'?s not (right|what I)",
            r"(wrong|incorrect)",
            r"try again",
            r"forget that",
            r"^(no|nope|nah)$",
        ]

        for pattern in rejection_patterns:
            if re.search(pattern, user_next_action.lower()):
                return 0.8  # High confidence rejection

        # Check if user completely ignored output
        if self._is_topic_change(claude_output, user_next_action):
            return 0.5  # Medium confidence rejection

        return None

    def detect_repetition(
        self,
        current_request: str,
    ) -> Optional[dict]:
        """
        Detect if user is repeating a previous request.

        This suggests the memory system failed to help adequately.
        """
        current_embedding = self._embed(current_request)

        for prev in self.recent_requests:
            similarity = cosine_similarity(current_embedding, prev["embedding"])

            if similarity > 0.85:
                time_diff = datetime.now() - prev["timestamp"]

                if time_diff < timedelta(hours=24):
                    return {
                        "repeated_request_id": prev["id"],
                        "similarity": similarity,
                        "time_since_original": time_diff.total_seconds(),
                        "implied_feedback": "memory_failure",
                    }

        return None
```

### 5.3 Online vs. Offline Learning

```python
class LearningScheduler:
    """
    Coordinate online and offline learning.

    Online (during session):
    - Update eligibility traces
    - Collect experiences
    - Light Hebbian updates

    Offline (between sessions):
    - Full gradient updates
    - Experience replay
    - Decay optimization
    - Consolidation evaluation
    """

    def __init__(self):
        self.online_updates_per_session = 0
        self.max_online_updates = 10  # Limit online compute

    async def online_update(
        self,
        experience: Experience,
        credit_assigner: TemporalCreditAssigner,
    ):
        """
        Lightweight online learning.

        - Updates eligibility traces
        - Stores experience for replay
        - Hebbian co-retrieval strengthening
        """
        if self.online_updates_per_session >= self.max_online_updates:
            return

        # Update eligibility traces
        for mem_id, score in zip(
            experience.retrieved_memory_ids,
            experience.retrieval_scores,
        ):
            credit_assigner.on_retrieval(mem_id, score, experience.timestamp)

        # Store for offline learning
        await self.experience_buffer.add(experience, td_error=0.5)

        # Hebbian update for co-retrieved memories
        await self._hebbian_strengthen(experience.retrieved_memory_ids)

        self.online_updates_per_session += 1

    async def offline_training_loop(
        self,
        scorer: LearnedRetrievalScorer,
        consolidation_policy: LearnedConsolidationPolicy,
        decay_policy: AdaptiveDecay,
    ):
        """
        Full offline training between sessions.

        Runs during idle time or scheduled maintenance.
        """
        # 1. Sample from experience replay
        experiences, weights = self.experience_buffer.sample(batch_size=64)

        if not experiences:
            return

        # 2. Train retrieval scorer
        retrieval_loss = self._train_retrieval_scorer(scorer, experiences, weights)

        # 3. Train consolidation policy
        consolidation_events = await self._get_recent_consolidations()
        if consolidation_events:
            consolidation_loss = self._train_consolidation_policy(
                consolidation_policy, consolidation_events
            )

        # 4. Optimize decay rates
        forgotten = await self._get_forgotten_memories_with_outcomes()
        if forgotten:
            decay_loss = self._train_decay_policy(decay_policy, forgotten)

        # 5. Checkpoint
        await self._checkpoint(scorer, consolidation_policy, decay_policy)
```

### 5.4 Catastrophic Forgetting Prevention

```python
class EWCRegularizer:
    """
    Elastic Weight Consolidation to prevent catastrophic forgetting.

    When learning new patterns, we don't want to destroy what we've
    learned about old patterns.
    """

    def __init__(self, lambda_: float = 1000):
        self.lambda_ = lambda_
        self.saved_params: dict[str, torch.Tensor] = {}
        self.fisher_diagonal: dict[str, torch.Tensor] = {}

    async def compute_fisher(
        self,
        model: nn.Module,
        dataloader: DataLoader,
    ):
        """
        Compute Fisher information diagonal.

        Fisher diagonal approximates importance of each parameter
        for the current task.
        """
        self.saved_params = {
            name: param.clone()
            for name, param in model.named_parameters()
        }

        self.fisher_diagonal = {
            name: torch.zeros_like(param)
            for name, param in model.named_parameters()
        }

        model.eval()
        for batch in dataloader:
            model.zero_grad()
            loss = model.compute_loss(batch)
            loss.backward()

            for name, param in model.named_parameters():
                if param.grad is not None:
                    self.fisher_diagonal[name] += param.grad ** 2

        # Normalize
        for name in self.fisher_diagonal:
            self.fisher_diagonal[name] /= len(dataloader)

    def penalty(self, model: nn.Module) -> torch.Tensor:
        """
        Compute EWC penalty for current parameters.

        Penalizes deviation from saved parameters, weighted by
        Fisher diagonal (importance).
        """
        penalty = 0.0

        for name, param in model.named_parameters():
            if name in self.saved_params:
                penalty += (
                    self.fisher_diagonal[name] *
                    (param - self.saved_params[name]) ** 2
                ).sum()

        return self.lambda_ * penalty
```

---

## Part 6: Concrete Implementation

### 6.1 Core Learning Loop

```python
class T4DMLearner:
    """
    Main learning system for T4DM.

    Coordinates:
    1. Data collection from sessions
    2. Reward computation
    3. Credit assignment
    4. Model updates (online + offline)
    5. Checkpointing
    """

    def __init__(
        self,
        memory_service: UnifiedMemoryService,
        config: LearningConfig,
    ):
        self.memory = memory_service
        self.config = config

        # Learnable components
        self.retrieval_scorer = LearnedRetrievalScorer()
        self.consolidation_policy = LearnedConsolidationPolicy()
        self.decay_policy = AdaptiveDecay()

        # Credit assignment
        self.credit_assigner = TemporalCreditAssigner(
            gamma=config.gamma,
            lambda_=config.lambda_,
        )
        self.multi_step_graph = MultiStepCreditGraph()

        # Reward computation
        self.retrieval_reward = RetrievalRewardComputer()
        self.consolidation_reward = ConsolidationRewardComputer()
        self.forgetting_reward = ForgettingRewardComputer()

        # Experience storage
        self.replay_buffer = ExperienceReplayBuffer(
            capacity=config.buffer_capacity,
        )

        # Regularization
        self.ewc = EWCRegularizer(lambda_=config.ewc_lambda)

        # Optimizers
        self.optimizers = {
            'retrieval': torch.optim.Adam(
                self.retrieval_scorer.parameters(),
                lr=config.retrieval_lr,
            ),
            'consolidation': torch.optim.Adam(
                self.consolidation_policy.parameters(),
                lr=config.consolidation_lr,
            ),
            'decay': torch.optim.Adam(
                self.decay_policy.parameters(),
                lr=config.decay_lr,
            ),
        }

        # Metrics
        self.metrics = LearningMetrics()

    async def on_session_start(self, session_id: str):
        """Initialize learning state for new session."""
        self.multi_step_graph.start_session(session_id)
        self.credit_assigner = TemporalCreditAssigner(
            gamma=self.config.gamma,
            lambda_=self.config.lambda_,
        )

    async def on_retrieval(
        self,
        query: str,
        results: list[ScoredResult],
        context: dict,
    ):
        """
        Called when memories are retrieved.

        1. Update eligibility traces
        2. Record retrieval event for reward attribution
        3. Track in multi-step graph
        """
        now = datetime.now()

        # Update eligibility traces
        for result in results:
            self.credit_assigner.on_retrieval(
                memory_id=result.item.id,
                retrieval_score=result.score,
                timestamp=now,
            )

            # Track in multi-step graph
            self.multi_step_graph.add_retrieval(
                memory_id=result.item.id,
                retrieval_context=query,
                attention_weight=result.score,
            )

        # Store retrieval event
        retrieval_event = RetrievalEvent(
            retrieval_id=uuid4(),
            query=query,
            retrieved_memory_ids=[r.item.id for r in results],
            retrieval_scores={r.item.id: r.score for r in results},
            timestamp=now,
            context_hash=self._hash_context(context),
        )

        self.retrieval_reward.pending_retrievals.setdefault(
            retrieval_event.context_hash, []
        ).append(retrieval_event)

    async def on_outcome(
        self,
        success_score: float,
        context: dict,
        explicit_citations: list[UUID] = None,
    ):
        """
        Called when a task completes.

        1. Compute rewards for pending retrievals
        2. Assign credit using traces and graph
        3. Store experience for replay
        4. Maybe do online update
        """
        now = datetime.now()
        context_hash = self._hash_context(context)

        # Get pending retrievals for this context
        pending = self.retrieval_reward.pending_retrievals.get(context_hash, [])

        if not pending:
            return

        # Compute per-memory rewards
        outcome_event = OutcomeEvent(
            outcome_id=uuid4(),
            success_score=success_score,
            task_context_hash=context_hash,
            timestamp=now,
            explicit_memory_refs=explicit_citations or [],
        )

        all_rewards = {}
        for retrieval in pending:
            time_delay = (now - retrieval.timestamp).total_seconds() / 3600
            rewards = await self.retrieval_reward.compute_reward(
                retrieval, outcome_event, time_delay
            )

            for mem_id, reward in rewards.items():
                all_rewards[mem_id] = all_rewards.get(mem_id, 0) + reward

        # Also get credit from eligibility traces
        trace_credits = self.credit_assigner.on_outcome(
            reward=success_score - 0.5,  # Center around 0
            timestamp=now,
        )

        # Combine rewards
        for mem_id, credit in trace_credits.items():
            all_rewards[mem_id] = all_rewards.get(mem_id, 0) + credit * 0.5

        # Graph-based credit (for multi-step reasoning)
        graph_credits = self.multi_step_graph.distribute_reward(success_score)
        for mem_id, credit in graph_credits.items():
            all_rewards[mem_id] = all_rewards.get(mem_id, 0) + credit * 0.3

        # Create experience
        for retrieval in pending:
            experience = Experience(
                experience_id=uuid4(),
                timestamp=retrieval.timestamp,
                query=retrieval.query,
                query_embedding=await self._embed(retrieval.query),
                retrieved_memory_ids=retrieval.retrieved_memory_ids,
                retrieval_scores=list(retrieval.retrieval_scores.values()),
                attention_weights=[],  # Would be populated by attention tracker
                outcome_score=success_score,
                outcome_timestamp=now,
                per_memory_credit={str(k): v for k, v in all_rewards.items()},
                session_id=str(self.multi_step_graph.current_session_id),
                task_type=self._infer_task_type(context),
            )

            # Compute TD error for prioritization
            expected_value = self._predict_value(experience)
            td_error = success_score - expected_value

            await self.replay_buffer.add(experience, td_error)

        # Clear pending
        self.retrieval_reward.pending_retrievals[context_hash] = []

        # Maybe online update
        if random.random() < self.config.online_update_prob:
            await self._online_update(all_rewards)

    async def _online_update(self, rewards: dict[UUID, float]):
        """
        Lightweight online learning step.

        Only updates Hebbian weights, not gradient-based parameters.
        """
        # Strengthen connections between positively-rewarded memories
        positive_mems = [m for m, r in rewards.items() if r > 0]

        for i, mem1 in enumerate(positive_mems):
            for mem2 in positive_mems[i+1:]:
                # Weight by combined reward
                combined_reward = rewards[mem1] + rewards[mem2]
                learning_rate = 0.05 * combined_reward

                await self.memory.semantic.graph_store.strengthen_relationship(
                    source_id=str(mem1),
                    target_id=str(mem2),
                    learning_rate=min(learning_rate, 0.2),
                )

    async def offline_training_step(self):
        """
        Full offline training step.

        Run during idle time or scheduled maintenance.
        """
        # Sample batch from replay buffer
        experiences, importance_weights = self.replay_buffer.sample(
            batch_size=self.config.batch_size
        )

        if not experiences:
            return {"status": "no_data"}

        # 1. Train retrieval scorer
        retrieval_loss = await self._train_retrieval(experiences, importance_weights)

        # 2. Train consolidation policy
        consolidation_loss = await self._train_consolidation()

        # 3. Train decay policy
        decay_loss = await self._train_decay()

        # 4. Apply EWC regularization if needed
        if self.config.use_ewc:
            ewc_penalty = self.ewc.penalty(self.retrieval_scorer)
            retrieval_loss = retrieval_loss + ewc_penalty

        # Record metrics
        self.metrics.record({
            "retrieval_loss": retrieval_loss.item(),
            "consolidation_loss": consolidation_loss.item() if consolidation_loss else 0,
            "decay_loss": decay_loss.item() if decay_loss else 0,
        })

        return self.metrics.get_summary()

    async def _train_retrieval(
        self,
        experiences: list[Experience],
        importance_weights: list[float],
    ) -> torch.Tensor:
        """Train retrieval scorer on batch."""
        self.optimizers['retrieval'].zero_grad()

        total_loss = torch.tensor(0.0)

        for exp, weight in zip(experiences, importance_weights):
            # Get current scores
            current_scores = []
            for mem_id in exp.retrieved_memory_ids:
                # Reconstruct components (would be stored in practice)
                components = {"semantic": 0.5, "recency": 0.5, "outcome": 0.5, "importance": 0.5}
                score = self.retrieval_scorer.score(components, "episodic")
                current_scores.append(score)

            # Get rewards
            rewards = [exp.per_memory_credit.get(str(m), 0) for m in exp.retrieved_memory_ids]

            # Compute ranking loss
            loss = retrieval_ranking_loss(current_scores, rewards)
            total_loss = total_loss + weight * loss

        total_loss = total_loss / len(experiences)
        total_loss.backward()
        self.optimizers['retrieval'].step()

        return total_loss

    async def checkpoint(self, path: Path):
        """Save all learnable parameters."""
        torch.save({
            'retrieval_scorer': self.retrieval_scorer.state_dict(),
            'consolidation_policy': self.consolidation_policy.state_dict(),
            'decay_policy': self.decay_policy.state_dict(),
            'ewc_params': self.ewc.saved_params,
            'ewc_fisher': self.ewc.fisher_diagonal,
            'metrics': self.metrics.history,
        }, path)

    async def restore(self, path: Path):
        """Restore from checkpoint."""
        checkpoint = torch.load(path)
        self.retrieval_scorer.load_state_dict(checkpoint['retrieval_scorer'])
        self.consolidation_policy.load_state_dict(checkpoint['consolidation_policy'])
        self.decay_policy.load_state_dict(checkpoint['decay_policy'])
        self.ewc.saved_params = checkpoint['ewc_params']
        self.ewc.fisher_diagonal = checkpoint['ewc_fisher']
        self.metrics.history = checkpoint['metrics']
```

### 6.2 Configuration

```python
@dataclass
class LearningConfig:
    """Configuration for learning system."""

    # Credit assignment
    gamma: float = 0.99  # Discount factor
    lambda_: float = 0.9  # Eligibility trace decay

    # Experience replay
    buffer_capacity: int = 100000
    batch_size: int = 64

    # Learning rates
    retrieval_lr: float = 1e-4
    consolidation_lr: float = 1e-4
    decay_lr: float = 1e-5

    # Online learning
    online_update_prob: float = 0.1  # Probability of online update

    # Regularization
    use_ewc: bool = True
    ewc_lambda: float = 1000

    # Checkpointing
    checkpoint_interval: int = 1000  # Steps
    checkpoint_path: Path = Path("~/.t4dm/checkpoints")

    # Training schedule
    offline_training_interval: int = 3600  # Seconds
    offline_batch_count: int = 10  # Batches per offline session

    @classmethod
    def from_file(cls, path: Path) -> "LearningConfig":
        """Load config from YAML file."""
        import yaml
        with open(path) as f:
            data = yaml.safe_load(f)
        return cls(**data)
```

### 6.3 Integration Hook

```python
# Add to t4dm/hooks/learning.py

class LearningHook(Hook):
    """
    Hook that integrates learning system with memory operations.

    Automatically collects data and triggers learning.
    """

    def __init__(self, learner: T4DMLearner):
        super().__init__(name="learning_hook", priority=HookPriority.LOW)
        self.learner = learner

class PostRetrievalLearningHook(LearningHook):
    """Collect retrieval data for learning."""

    async def execute(self, context: HookContext) -> HookContext:
        results = context.output_data.get("results", [])
        query = context.input_data.get("query", "")

        await self.learner.on_retrieval(
            query=query,
            results=results,
            context={
                "cwd": context.metadata.get("cwd"),
                "project": context.metadata.get("project"),
                "tool": context.metadata.get("tool"),
            },
        )

        return context

class OutcomeLearningHook(LearningHook):
    """Process task outcomes for learning."""

    async def execute(self, context: HookContext) -> HookContext:
        success = context.output_data.get("success", False)
        error = context.output_data.get("error")

        success_score = 1.0 if success else (0.0 if error else 0.5)

        await self.learner.on_outcome(
            success_score=success_score,
            context=context.metadata,
            explicit_citations=context.metadata.get("cited_memories"),
        )

        return context
```

---

## Summary: What Changes

### Immediate (Can implement now):

1. **Add retrieval tracking**: Store retrieval events with context hashes for later reward attribution.

2. **Add outcome tracking**: Capture task success/failure with timestamps.

3. **Implement temporal credit assignment**: Use eligibility traces to distribute rewards.

4. **Add experience buffer**: Store (retrieval, outcome) pairs for offline learning.

5. **Hebbian updates based on reward**: Currently you strengthen co-retrieved memories; instead, strengthen based on *outcome-weighted* co-retrieval.

### Short-term (Requires some infrastructure):

6. **Learnable retrieval weights**: Replace fixed weights with PyTorch parameters.

7. **Offline training loop**: Batch training from experience buffer.

8. **Forgetting detection**: Track forgotten content and detect regret.

### Long-term (Research-grade):

9. **Meta-learning for domain adaptation**: MAML-style fast adaptation.

10. **Differentiable memory retrieval**: End-to-end learning through retrieval.

11. **Multi-step credit graphs**: Full DAG-based credit assignment.

---

## Key Formulas Reference

### Retrieval Reward
```
R(memory, outcome) = (outcome_score - baseline) * time_discount * attention_weight

time_discount = 1 / (1 + 0.1 * hours_delay)  # Hyperbolic
attention_weight = retrieval_score / sum(retrieval_scores)
```

### Eligibility Trace Update
```
trace[memory] += retrieval_score  (on retrieval)
trace[memory] *= (gamma * lambda)^elapsed_hours  (decay)
credit[memory] = reward * trace[memory] / sum(traces)  (on outcome)
```

### Retrieval Ranking Loss (ListMLE)
```
L = -sum_{i in target_order} log(exp(score[i]) / sum_{j in remaining} exp(score[j]))
```

### EWC Penalty
```
L_EWC = lambda * sum_i F_i * (theta_i - theta_i^*)^2

where F_i = Fisher diagonal, theta* = saved params
```

### Entity Quality
```
Q(entity) = 0.3 * usage_frequency + 0.4 * outcome_weighted_utility +
            0.15 * graph_centrality + 0.15 * specificity
```

---

This architecture transforms T4DM from a storage system into a learning system. The key insight is that **learning requires contrastive signals** - we must know not just what happened, but what *would have* happened differently. The reward functions, credit assignment mechanisms, and learning objectives I've specified provide exactly these contrastive signals.

Start with the immediate changes. They require minimal infrastructure and will immediately make your system smarter, not just fuller.
