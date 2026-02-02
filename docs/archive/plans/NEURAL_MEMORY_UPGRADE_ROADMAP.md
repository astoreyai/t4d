# T4DM Neural Memory Architecture Upgrade Roadmap

**Created**: 2025-12-06
**Version**: 1.0
**Target Completion**: ~50 development days (10 weeks)
**Current WW State**: v0.2.0, 1259 tests, 79% coverage

## Executive Summary

This roadmap integrates 6 cutting-edge neural memory components into T4DM's existing tripartite memory system. The upgrade transforms WW from a static vector store into a biologically-inspired, continuously learning memory architecture.

**Key Improvements**:
- Online learning during LLM inference (Forward-Forward)
- Exponential memory capacity increase (Hopfield-Fenchel-Young)
- Local credit assignment without backprop (Dendritic computation)
- Enhanced multi-timescale dynamics (2-6% accuracy gains)
- Scale to 2M+ token contexts (Titans-style memory)
- Selective memorization via surprise mechanism

**Risk Level**: MEDIUM
- High backward compatibility (all changes additive)
- Incremental rollout (phase-by-phase validation)
- Existing 1259 tests provide safety net

---

## Current T4DM Architecture

### Existing Components (Keep)
```
/mnt/projects/t4d/t4dm/src/t4dm/
├── memory/
│   ├── episodic.py              # 1800 lines - autobiographical events
│   ├── semantic.py              # Hebbian knowledge graph
│   ├── procedural.py            # Skill/pattern storage
│   ├── buffer_manager.py        # CA1-like temporary storage (NEW v0.2.0)
│   └── cluster_index.py         # Hierarchical retrieval (PARTIAL)
├── core/
│   ├── learned_gate.py          # Bayesian logistic gate (247-dim features)
│   └── memory_gate.py           # Heuristic fallback
├── learning/
│   ├── neuromodulators.py       # DA, NE, ACh, 5-HT, GABA orchestra
│   ├── dopamine.py              # RPE, surprise
│   ├── norepinephrine.py        # Arousal, novelty
│   ├── acetylcholine.py         # Encoding/retrieval switching
│   ├── serotonin.py             # Long-term credit
│   └── inhibition.py            # GABA/Glutamate
├── storage/
│   ├── t4dx_graph_adapter.py           # Graph backend
│   ├── t4dx_vector_adapter.py          # Vector backend
│   └── saga.py                  # Dual-store consistency
└── embedding/
    └── bge_m3.py                # BGE-M3 1024-dim embeddings
```

### Key Metrics (Baseline)
- **Storage Latency**: 15-25ms (Neo4j + Qdrant dual-write)
- **Retrieval Latency**: 30-50ms (k-NN + scoring)
- **Memory Capacity**: ~100K episodes before slowdown
- **Gate Accuracy**: 72% (after 500 observations, Thompson sampling)
- **Consolidation**: HDBSCAN clusters, 67x speedup potential

---

## Phase-by-Phase Implementation Plan

### Phase 0: Foundation & Cleanup (5 days)

**Goal**: Prepare codebase for neural components, establish testing infrastructure

#### Tasks

**TASK-001: Audit Current Neural Components** (Priority: 1, Effort: small)
- Description: Identify all existing learning mechanisms and their integration points
- Acceptance Criteria:
  - Document all gradient flow paths in LearnedMemoryGate
  - Map neuromodulator signal pathways through orchestra
  - Identify unused learned components (LearnedFusion, LearnedRetrievalScorer)
  - Create dependency graph for existing learning systems
- Files to Read:
  - `/mnt/projects/t4d/t4dm/src/t4dm/core/learned_gate.py`
  - `/mnt/projects/t4d/t4dm/src/t4dm/learning/neuromodulators.py`
  - `/mnt/projects/t4d/t4dm/src/t4dm/learning/neuro_symbolic.py`
  - `/mnt/projects/t4d/t4dm/src/t4dm/learning/scorer.py`
- Deliverable: `/mnt/projects/t4d/t4dm/docs/NEURAL_COMPONENT_AUDIT.md`
- Agent: ww-analysis-agent
- Dependencies: None
- Risk: Low - read-only analysis
- Expected Performance: N/A

**TASK-002: Create Neural Memory Test Harness** (Priority: 1, Effort: medium)
- Description: Build comprehensive test suite for neural learning validation
- Acceptance Criteria:
  - Pattern completion tests (DG/CA3 analogs)
  - Sparse retrieval benchmarks (measure k% retrieved)
  - Learning convergence tests (track loss over updates)
  - Multi-timescale validation (verify layer-specific τ)
  - Eligibility trace tests (delayed credit assignment)
- Files to Create:
  - `/mnt/projects/t4d/t4dm/tests/neural/test_pattern_completion.py`
  - `/mnt/projects/t4d/t4dm/tests/neural/test_sparse_retrieval.py`
  - `/mnt/projects/t4d/t4dm/tests/neural/test_learning_convergence.py`
  - `/mnt/projects/t4d/t4dm/tests/neural/test_multitimescale.py`
  - `/mnt/projects/t4d/t4dm/tests/neural/conftest.py` (fixtures)
- Mathematical Formulations:
  ```python
  # Pattern completion accuracy
  def pattern_completion_metric(retrieved, target, partial_ratio=0.3):
      # Given partial cue (30% of embedding), measure recall quality
      partial_cue = target[:int(len(target) * partial_ratio)]
      recall = hopfield_retrieve(partial_cue)
      return cosine_similarity(recall, target)

  # Sparse retrieval efficiency
  def sparsity_metric(attention_weights, target_sparsity=0.1):
      k = int(len(attention_weights) * target_sparsity)
      top_k_sum = np.sort(attention_weights)[-k:].sum()
      return top_k_sum / attention_weights.sum()  # Should be > 0.9
  ```
- Dependencies: None
- Risk: Low - isolated test infrastructure
- Expected Performance: Tests run in <10s total

**TASK-003: Establish Performance Baselines** (Priority: 2, Effort: small)
- Description: Benchmark current system before neural upgrades
- Acceptance Criteria:
  - Storage decision accuracy (current: ~72%)
  - Retrieval precision@5 and recall@10
  - End-to-end latency (p50, p95, p99)
  - Memory footprint (model parameters + caches)
  - Throughput (queries/sec, stores/sec)
- Files to Create:
  - `/mnt/projects/t4d/t4dm/benchmarks/baseline_metrics.json`
  - `/mnt/projects/t4d/t4dm/benchmarks/run_baseline.py`
- Test Cases:
  ```python
  @pytest.mark.benchmark
  def test_baseline_storage_latency(benchmark, episodic_memory):
      def store_episode():
          return episodic_memory.create(
              content="Baseline test episode",
              context=EpisodeContext(project="benchmark"),
              outcome=Outcome.SUCCESS
          )
      result = benchmark(store_episode)
      assert result.latency_ms < 25  # Current target
  ```
- Dependencies: None
- Risk: Low
- Expected Performance: Baseline captured in JSON

**TASK-004: Refactor Feature Extraction Pipeline** (Priority: 1, Effort: medium)
- Description: Centralize feature extraction for all neural components
- Acceptance Criteria:
  - Single FeatureExtractor class used by gate, retrieval, consolidation
  - Support for multi-resolution features (1024d, 256d, 64d projections)
  - Lazy computation (only extract needed features)
  - Cache common projections (content, context)
- Files to Modify:
  - `/mnt/projects/t4d/t4dm/src/t4dm/core/learned_gate.py` (lines 300-450)
  - `/mnt/projects/t4d/t4dm/src/t4dm/memory/episodic.py` (lines 33-100, LearnedFusionWeights)
- Files to Create:
  - `/mnt/projects/t4d/t4dm/src/t4dm/core/feature_extractor.py`
- Mathematical Formulations:
  ```python
  class FeatureExtractor:
      """Centralized multi-resolution feature extraction."""

      def __init__(self, embedding_dim=1024):
          # Learned projections (shared across components)
          self.proj_256 = nn.Linear(1024, 256)  # Mid-level
          self.proj_128 = nn.Linear(1024, 128)  # Gate features
          self.proj_64 = nn.Linear(1024, 64)    # Context

      def extract(self, content_emb, context, neuromod_state, resolution='full'):
          if resolution == 'gate':
              return self._gate_features(content_emb, context, neuromod_state)
          elif resolution == 'retrieval':
              return self._retrieval_features(content_emb)
          # ... etc
  ```
- Dependencies: None
- Risk: Medium - touches core path, requires careful testing
- Expected Performance: <1ms overhead

---

### Phase 1: Forward-Forward Learning (10 days)

**Goal**: Implement layer-local learning for online adaptation during inference

#### Background
Forward-Forward (FF) replaces backpropagation with local "goodness" functions per layer. Each layer learns to distinguish positive data (real) from negative data (synthetic or corrupted). This enables learning during LLM inference without gradient flow.

#### Tasks

**TASK-101: Implement Forward-Forward Layer** (Priority: 1, Effort: large)
- Description: Core FF layer with local goodness optimization
- Acceptance Criteria:
  - Layer computes goodness = ||h||² for activations h
  - Positive data (real memories) maximize goodness
  - Negative data (corrupted/synthetic) minimize goodness
  - Update rule: Δw ∝ (g⁺ - g⁻) · x (layer-local)
  - <5ms latency per layer forward pass
- Files to Create:
  - `/mnt/projects/t4d/t4dm/src/t4dm/learning/forward_forward.py`
- Mathematical Formulations:
  ```python
  class ForwardForwardLayer:
      """Single FF layer with local goodness learning."""

      def __init__(self, input_dim, output_dim, learning_rate=0.03):
          self.W = np.random.randn(output_dim, input_dim) * 0.01
          self.b = np.zeros(output_dim)
          self.lr = learning_rate
          self.theta = 2.0  # Goodness threshold

      def goodness(self, x):
          """Compute layer goodness: sum of squared activations."""
          h = np.maximum(0, self.W @ x + self.b)  # ReLU
          return (h ** 2).sum()

      def forward(self, x, is_positive=True):
          """Forward pass with optional local learning."""
          g = self.goodness(x)

          # Local learning: push positive above threshold, negative below
          if is_positive:
              loss = -np.log(1 + np.exp(g - self.theta))
          else:
              loss = -np.log(1 + np.exp(self.theta - g))

          # Gradient: ∂loss/∂W
          h = np.maximum(0, self.W @ x + self.b)
          grad_W = -2 * np.outer(h, x) * np.exp(...)  # Chain rule

          self.W -= self.lr * grad_W
          return h
  ```
- Test Cases:
  ```python
  def test_ff_layer_positive_goodness():
      layer = ForwardForwardLayer(128, 64)
      x_pos = np.random.randn(128)

      g_before = layer.goodness(x_pos)
      for _ in range(10):
          layer.forward(x_pos, is_positive=True)
      g_after = layer.goodness(x_pos)

      assert g_after > g_before  # Goodness increased
      assert g_after > layer.theta  # Above threshold
  ```
- Dependencies: TASK-004 (feature extractor)
- Risk: Medium - new learning paradigm
- Expected Performance: Goodness increases 20% after 10 updates

**TASK-102: Generate Negative Data** (Priority: 1, Effort: medium)
- Description: Create synthetic negative examples for FF training
- Acceptance Criteria:
  - Method 1: Embedding corruption (add Gaussian noise σ=0.3)
  - Method 2: Hybrid episodes (mix embeddings from unrelated memories)
  - Method 3: Adversarial negatives (search for hardest examples)
  - Negatives pass basic sanity checks (not identical to positives)
- Files to Modify:
  - `/mnt/projects/t4d/t4dm/src/t4dm/learning/forward_forward.py`
- Mathematical Formulations:
  ```python
  def generate_negatives(positive_embedding, method='corrupt'):
      if method == 'corrupt':
          # Gaussian noise corruption
          noise = np.random.randn(*positive_embedding.shape) * 0.3
          return positive_embedding + noise

      elif method == 'hybrid':
          # Mix with random unrelated memory
          other = get_random_memory_embedding()
          alpha = np.random.uniform(0.3, 0.7)
          return alpha * positive_embedding + (1 - alpha) * other

      elif method == 'adversarial':
          # Find embedding that maximizes goodness but is incorrect
          # (This is expensive - use sparingly)
          ...
  ```
- Dependencies: TASK-101
- Risk: Low
- Expected Performance: <1ms per negative

**TASK-103: Integrate FF into Episodic Memory** (Priority: 1, Effort: large)
- Description: Add FF layers to episode encoding pipeline
- Acceptance Criteria:
  - Episodes pass through 2-layer FF network before storage
  - Positive examples: actual episodes
  - Negative examples: generated via TASK-102
  - FF updates happen asynchronously (non-blocking)
  - Learned representations improve retrieval quality (>5% P@5 gain)
- Files to Modify:
  - `/mnt/projects/t4d/t4dm/src/t4dm/memory/episodic.py` (lines 200-300, create())
- Integration Points:
  ```python
  class EpisodicMemory:
      def __init__(self, ...):
          # ... existing code ...
          self.ff_layer1 = ForwardForwardLayer(1024, 512)
          self.ff_layer2 = ForwardForwardLayer(512, 256)

      async def create(self, content, context, outcome):
          # Get embedding
          embedding = await self.embed(content)

          # Forward-Forward encoding (online learning)
          h1 = self.ff_layer1.forward(embedding, is_positive=True)
          h2 = self.ff_layer2.forward(h1, is_positive=True)

          # Also train on negatives (async to avoid blocking)
          asyncio.create_task(self._train_ff_negatives(embedding))

          # Store with FF-transformed representation
          # Use h2 (256d) instead of raw embedding (1024d)?
          # OR concatenate [embedding, h2] for richer features?
          ...
  ```
- Test Cases:
  ```python
  @pytest.mark.asyncio
  async def test_ff_improves_retrieval():
      mem = EpisodicMemory(...)

      # Store 100 episodes with FF learning
      for i in range(100):
          await mem.create(f"Episode {i}", ...)

      # Measure retrieval quality
      results = await mem.recall("Episode 50")

      # FF-learned representations should improve precision
      assert results[0].content == "Episode 50"
      assert results[0].score > 0.85
  ```
- Dependencies: TASK-102
- Risk: High - core memory path modification
- Expected Performance: +5-10% retrieval precision, <3ms FF overhead

**TASK-104: FF Hyperparameter Tuning** (Priority: 2, Effort: small)
- Description: Optimize FF layer sizes, learning rates, goodness thresholds
- Acceptance Criteria:
  - Grid search over: layer_dims [256, 512], lr [0.01, 0.03, 0.1], theta [1.5, 2.0, 2.5]
  - Measure: retrieval quality, learning speed, stability
  - Document optimal settings in config
- Files to Create:
  - `/mnt/projects/t4d/t4dm/benchmarks/ff_hyperparameter_search.py`
  - `/mnt/projects/t4d/t4dm/config/ff_optimal_params.json`
- Dependencies: TASK-103
- Risk: Low
- Expected Performance: 2-3% additional gain from tuning

---

### Phase 2: Hopfield-Fenchel-Young Memory (12 days)

**Goal**: Exponential memory capacity via energy-based associative retrieval

#### Background
Traditional Hopfield networks have O(n) capacity (n patterns → n neurons). Fenchel-Young theory enables O(exp(n)) capacity via:
- Sparse attention (α-entmax instead of softmax)
- Energy minimization for retrieval
- Continuous-valued attractors (not binary)

#### Tasks

**TASK-201: Implement α-Entmax Sparse Attention** (Priority: 1, Effort: medium)
- Description: Replace softmax with entmax for sparse retrieval
- Acceptance Criteria:
  - entmax_α(z) returns exactly sparse distribution (many zeros)
  - α=1.0 → softmax, α=1.5 → entmax-1.5 (typical), α=2.0 → sparsemax
  - Backward pass implemented for learning
  - <2ms for 10K-dim input
- Files to Create:
  - `/mnt/projects/t4d/t4dm/src/t4dm/learning/sparse_attention.py`
- Mathematical Formulations:
  ```python
  def entmax_alpha(z, alpha=1.5):
      """
      Sparse attention via α-entmax.

      entmax_α(z) = argmax_p { p·z - Ω_α(p) }
      where Ω_α(p) is Tsallis entropy.

      For α=1.5, this produces sparse distributions.
      """
      if alpha == 1.0:
          return softmax(z)

      # Efficient implementation using bisection for τ threshold
      z_sorted = np.sort(z)[::-1]
      k_array = np.arange(1, len(z) + 1)

      # Find threshold τ via bisection
      def objective(tau):
          return ((z_sorted - tau) ** (alpha - 1)).sum()

      tau_star = bisect(objective, z.min(), z.max())

      # Apply sparse transform
      p = np.maximum(0, z - tau_star) ** (1 / (alpha - 1))
      return p / p.sum()
  ```
- Test Cases:
  ```python
  def test_entmax_sparsity():
      z = np.random.randn(1000)
      p = entmax_alpha(z, alpha=1.5)

      # Should be sparse
      assert (p == 0).sum() > 700  # >70% zeros
      assert p.sum() == pytest.approx(1.0)
      assert (p >= 0).all()
  ```
- Dependencies: None
- Risk: Medium - numerical stability critical
- Expected Performance: Exactly sparse (70-90% zeros)

**TASK-202: Build Hopfield Energy Function** (Priority: 1, Effort: large)
- Description: Energy-based memory retrieval via gradient descent
- Acceptance Criteria:
  - Energy E(q, K, V) = -entmax_α(qK^T)^T V
  - Retrieval: minimize E w.r.t. query q
  - Convergence in <10 iterations
  - Retrieved pattern matches stored memory (cosine > 0.9)
- Files to Create:
  - `/mnt/projects/t4d/t4dm/src/t4dm/memory/hopfield_memory.py`
- Mathematical Formulations:
  ```python
  class HopfieldMemory:
      """Energy-based associative memory with α-entmax."""

      def __init__(self, alpha=1.5, learning_rate=0.1, max_iters=10):
          self.alpha = alpha
          self.lr = learning_rate
          self.max_iters = max_iters

          # Memory: keys K and values V
          self.K = []  # List of key vectors
          self.V = []  # List of value vectors

      def store(self, key, value):
          """Store key-value pair."""
          self.K.append(key)
          self.V.append(value)

      def retrieve(self, query_init):
          """
          Retrieve by minimizing energy E(q) via gradient descent.

          E(q) = -entmax_α(qK^T)^T V

          ∂E/∂q = -K^T ∂entmax/∂z where z = qK^T
          """
          q = query_init.copy()
          K_matrix = np.array(self.K)  # Shape: (n_memories, d)
          V_matrix = np.array(self.V)

          for _ in range(self.max_iters):
              # Compute attention
              z = q @ K_matrix.T  # Shape: (n_memories,)
              p = entmax_alpha(z, self.alpha)  # Sparse distribution

              # Energy gradient
              grad_z = V_matrix.T @ p  # ∂E/∂z
              grad_q = K_matrix.T @ grad_z  # ∂E/∂q via chain rule

              # Update query
              q -= self.lr * grad_q
              q /= np.linalg.norm(q)  # Normalize

          # Final retrieval
          z = q @ K_matrix.T
          p = entmax_alpha(z, self.alpha)
          retrieved = p @ V_matrix

          return retrieved, p  # Value and attention weights
  ```
- Test Cases:
  ```python
  def test_hopfield_pattern_completion():
      mem = HopfieldMemory()

      # Store 100 random patterns
      patterns = [np.random.randn(256) for _ in range(100)]
      for p in patterns:
          mem.store(key=p, value=p)  # Auto-associative

      # Partial cue (30% of pattern)
      target = patterns[42]
      partial = target.copy()
      partial[80:] = 0  # Zero out 70%

      # Retrieve from partial cue
      retrieved, attn = mem.retrieve(partial)

      # Should reconstruct target
      assert cosine_similarity(retrieved, target) > 0.9
      assert attn.max() > 0.7  # Should attend strongly to pattern 42
  ```
- Dependencies: TASK-201
- Risk: High - new retrieval mechanism
- Expected Performance: 90%+ pattern completion accuracy

**TASK-203: Integrate Hopfield with Qdrant** (Priority: 1, Effort: large)
- Description: Use Hopfield for final retrieval stage after Qdrant k-NN
- Acceptance Criteria:
  - Qdrant returns top-K candidates (K=50)
  - Hopfield refines query via energy minimization
  - Final top-k selected from refined attention
  - Backward compatible (fallback to Qdrant-only if Hopfield disabled)
- Files to Modify:
  - `/mnt/projects/t4d/t4dm/src/t4dm/memory/episodic.py` (lines 450-650, recall())
- Integration Pattern:
  ```python
  async def recall(self, query, limit=5, use_hopfield=True):
      # Stage 1: Qdrant k-NN (broad retrieval)
      candidates = await self.vector_store.search(
          query_emb, limit=50  # Over-retrieve
      )

      if use_hopfield:
          # Stage 2: Hopfield refinement
          K = [c.embedding for c in candidates]
          V = [c.embedding for c in candidates]  # Or content features

          refined_query, attn = self.hopfield.retrieve(query_emb)

          # Re-rank candidates by Hopfield attention
          for i, c in enumerate(candidates):
              c.hopfield_score = attn[i]

          candidates.sort(key=lambda c: c.hopfield_score, reverse=True)

      return candidates[:limit]
  ```
- Test Cases:
  ```python
  @pytest.mark.asyncio
  async def test_hopfield_improves_recall():
      mem = EpisodicMemory(use_hopfield=True)

      # Store 1000 episodes
      for i in range(1000):
          await mem.create(f"Episode {i} about {i%10}", ...)

      # Query for specific topic
      results_vanilla = await mem.recall("about 7", use_hopfield=False)
      results_hopfield = await mem.recall("about 7", use_hopfield=True)

      # Hopfield should improve precision
      relevant_vanilla = sum(1 for r in results_vanilla if "about 7" in r.content)
      relevant_hopfield = sum(1 for r in results_hopfield if "about 7" in r.content)

      assert relevant_hopfield >= relevant_vanilla
  ```
- Dependencies: TASK-202
- Risk: High - changes retrieval semantics
- Expected Performance: +10-15% recall@10

**TASK-204: Measure Memory Capacity** (Priority: 2, Effort: medium)
- Description: Benchmark Hopfield capacity scaling
- Acceptance Criteria:
  - Test with 1K, 10K, 100K, 1M stored patterns
  - Measure: retrieval accuracy vs. n_patterns
  - Validate exponential capacity (accuracy > 80% at 1M)
  - Document capacity limits in config
- Files to Create:
  - `/mnt/projects/t4d/t4dm/benchmarks/hopfield_capacity_test.py`
  - `/mnt/projects/t4d/t4dm/docs/HOPFIELD_CAPACITY_ANALYSIS.md`
- Dependencies: TASK-203
- Risk: Low
- Expected Performance: 80%+ accuracy at 1M patterns (vs 50% for standard Hopfield)

---

### Phase 3: Dendritic Computation (8 days)

**Goal**: Two-compartment neurons for local credit assignment

#### Background
Dendrites act as separate computational units. Mismatch between soma prediction and dendritic input drives local learning without backprop.

#### Tasks

**TASK-301: Implement Two-Compartment Neuron** (Priority: 1, Effort: medium)
- Description: Neuron with apical (context) and basal (input) dendrites
- Acceptance Criteria:
  - Soma: integrates basal input + apical context
  - Learning: Δw ∝ (apical - soma) × basal (plateau potentials)
  - Supports branch-specific plasticity
  - <0.1ms per neuron update
- Files to Create:
  - `/mnt/projects/t4d/t4dm/src/t4dm/learning/dendritic_neuron.py`
- Mathematical Formulations:
  ```python
  class DendriticNeuron:
      """Two-compartment neuron with local learning."""

      def __init__(self, n_basal, n_apical, learning_rate=0.01):
          self.w_basal = np.random.randn(n_basal) * 0.01
          self.w_apical = np.random.randn(n_apical) * 0.01
          self.lr = learning_rate

      def forward(self, basal_input, apical_input):
          """
          Forward pass:
          - Basal dendrites: bottom-up input
          - Apical dendrites: top-down context
          - Soma: integrates both
          """
          basal_pred = self.w_basal @ basal_input
          apical_pred = self.w_apical @ apical_input

          # Soma output (nonlinear integration)
          soma = np.tanh(basal_pred + 0.5 * apical_pred)

          return soma, basal_pred, apical_pred

      def learn(self, basal_input, apical_input):
          """
          Local learning rule:

          Δw_basal ∝ (apical - soma) × basal

          If apical prediction > soma, strengthen basal weights.
          This implements "expectation-driven" learning.
          """
          soma, basal_pred, apical_pred = self.forward(basal_input, apical_input)

          # Mismatch signal (plateau potential)
          mismatch = apical_pred - soma

          # Branch-specific update
          delta_basal = self.lr * mismatch * basal_input
          delta_apical = self.lr * mismatch * apical_input

          self.w_basal += delta_basal
          self.w_apical += delta_apical

          return soma
  ```
- Test Cases:
  ```python
  def test_dendritic_learning():
      neuron = DendriticNeuron(n_basal=10, n_apical=5)

      # Training: apical provides target, basal provides input
      for _ in range(100):
          basal = np.random.randn(10)
          apical = np.random.randn(5)
          neuron.learn(basal, apical)

      # After training, soma should align with apical
      soma, _, apical_pred = neuron.forward(basal, apical)
      assert abs(soma - apical_pred) < 0.1  # Converged
  ```
- Dependencies: None
- Risk: Low
- Expected Performance: Converges in <100 updates

**TASK-302: Dendritic Layers for Episodic Memory** (Priority: 1, Effort: large)
- Description: Replace standard layers with dendritic computation
- Acceptance Criteria:
  - Basal: episode content embedding
  - Apical: task/context embedding (project, session, neuromod state)
  - Learning: context-gated plasticity (only when apical is active)
  - Improves context-dependent retrieval (>10% gain)
- Files to Modify:
  - `/mnt/projects/t4d/t4dm/src/t4dm/memory/episodic.py`
- Files to Create:
  - `/mnt/projects/t4d/t4dm/src/t4dm/learning/dendritic_layer.py`
- Integration Pattern:
  ```python
  class DendriticLayer:
      """Layer of two-compartment neurons."""

      def __init__(self, n_neurons, n_basal, n_apical):
          self.neurons = [
              DendriticNeuron(n_basal, n_apical)
              for _ in range(n_neurons)
          ]

      def forward(self, basal_input, apical_input):
          outputs = [n.forward(basal_input, apical_input)[0]
                     for n in self.neurons]
          return np.array(outputs)

      def learn(self, basal_input, apical_input):
          for n in self.neurons:
              n.learn(basal_input, apical_input)

  # In EpisodicMemory:
  class EpisodicMemory:
      def __init__(self, ...):
          self.dendritic_encoder = DendriticLayer(
              n_neurons=256,
              n_basal=1024,  # Content embedding
              n_apical=128   # Context (project + session + neuromod)
          )

      async def create(self, content, context, outcome):
          # Basal: content embedding
          content_emb = await self.embed(content)

          # Apical: context embedding
          context_emb = self._encode_context(context)

          # Dendritic encoding
          encoded = self.dendritic_encoder.forward(content_emb, context_emb)

          # Learn if context is strong
          if context.importance > 0.5:
              self.dendritic_encoder.learn(content_emb, context_emb)

          # Store encoded representation
          ...
  ```
- Test Cases:
  ```python
  @pytest.mark.asyncio
  async def test_context_gated_retrieval():
      mem = EpisodicMemory(use_dendritic=True)

      # Store episodes in two different projects
      for i in range(50):
          await mem.create(
              f"Fact {i}",
              context=EpisodeContext(project="physics"),
              ...
          )
      for i in range(50):
          await mem.create(
              f"Fact {i}",  # Same content!
              context=EpisodeContext(project="biology"),
              ...
          )

      # Query with project context
      results = await mem.recall(
          "Fact 25",
          context=EpisodeContext(project="physics")
      )

      # Should retrieve physics version, not biology
      assert results[0].context.project == "physics"
  ```
- Dependencies: TASK-301
- Risk: High - significant architecture change
- Expected Performance: +10-15% context-dependent accuracy

**TASK-303: Branch-Specific Plasticity** (Priority: 2, Effort: medium)
- Description: Independent learning rates per dendritic branch
- Acceptance Criteria:
  - Different lr for basal vs apical
  - Per-branch eligibility traces
  - Prune weak branches (weight magnitude < threshold)
- Files to Modify:
  - `/mnt/projects/t4d/t4dm/src/t4dm/learning/dendritic_neuron.py`
- Mathematical Formulations:
  ```python
  class DendriticNeuron:
      def __init__(self, ..., lr_basal=0.01, lr_apical=0.005):
          self.lr_basal = lr_basal
          self.lr_apical = lr_apical

          # Per-branch eligibility traces
          self.e_basal = np.zeros_like(self.w_basal)
          self.e_apical = np.zeros_like(self.w_apical)
          self.trace_decay = 0.9

      def learn(self, basal_input, apical_input, reward_signal=0):
          # ... compute mismatch ...

          # Update eligibility traces
          self.e_basal = self.trace_decay * self.e_basal + basal_input
          self.e_apical = self.trace_decay * self.e_apical + apical_input

          # Three-factor learning: mismatch × eligibility × reward
          delta_basal = self.lr_basal * mismatch * self.e_basal * reward_signal
          delta_apical = self.lr_apical * mismatch * self.e_apical * reward_signal

          self.w_basal += delta_basal
          self.w_apical += delta_apical

          # Prune weak branches
          self.w_basal[np.abs(self.w_basal) < 0.001] = 0
  ```
- Dependencies: TASK-302
- Risk: Low
- Expected Performance: 5-10% parameter reduction via pruning

---

### Phase 4: Enhanced Neuromodulation (7 days)

**Goal**: Eligibility traces for delayed credit assignment

#### Tasks

**TASK-401: Implement Eligibility Traces** (Priority: 1, Effort: medium)
- Description: Track recent activity for delayed reward assignment
- Acceptance Criteria:
  - Exponential decay traces: e(t) = γ e(t-1) + x(t)
  - Per-parameter traces (not shared)
  - Integration with DA, 5-HT signals
  - <1KB memory per 1K parameters
- Files to Create:
  - `/mnt/projects/t4d/t4dm/src/t4dm/learning/eligibility_trace.py`
- Mathematical Formulations:
  ```python
  class EligibilityTrace:
      """Exponentially decaying activity trace for credit assignment."""

      def __init__(self, shape, decay=0.9):
          self.trace = np.zeros(shape)
          self.gamma = decay

      def update(self, activity):
          """Update trace with new activity."""
          self.trace = self.gamma * self.trace + activity
          return self.trace

      def reset(self):
          """Clear trace."""
          self.trace.fill(0)

      def get(self):
          """Get current trace value."""
          return self.trace.copy()
  ```
- Dependencies: None
- Risk: Low
- Expected Performance: Minimal overhead

**TASK-402: Three-Factor Learning Rule** (Priority: 1, Effort: medium)
- Description: Δw ∝ activity × eligibility × neuromodulator
- Acceptance Criteria:
  - Factor 1: Pre-synaptic activity (input)
  - Factor 2: Post-synaptic activity (output)
  - Factor 3: Neuromodulator (DA or 5-HT)
  - Eligibility trace allows delayed factor 3
  - Works with all existing learning components
- Files to Modify:
  - `/mnt/projects/t4d/t4dm/src/t4dm/core/learned_gate.py`
  - `/mnt/projects/t4d/t4dm/src/t4dm/learning/forward_forward.py`
  - `/mnt/projects/t4d/t4dm/src/t4dm/learning/dendritic_neuron.py`
- Mathematical Formulations:
  ```python
  class ThreeFactorLearner:
      """Three-factor Hebbian learning with eligibility traces."""

      def __init__(self, weight_shape, trace_decay=0.9):
          self.W = np.random.randn(*weight_shape) * 0.01
          self.eligibility = EligibilityTrace(weight_shape, decay=trace_decay)

      def forward(self, x):
          return self.W @ x

      def update(self, x, y, neuromod_signal):
          """
          Three-factor update:

          Δw_ij = η × x_i × y_j × e_ij × neuromod

          Where:
          - x_i: pre-synaptic (input)
          - y_j: post-synaptic (output)
          - e_ij: eligibility trace
          - neuromod: DA or 5-HT signal
          """
          # Update eligibility trace
          hebbian = np.outer(y, x)  # y_j × x_i
          self.eligibility.update(hebbian)

          # Apply neuromodulator to trace
          delta = self.eligibility.get() * neuromod_signal

          self.W += delta
  ```
- Dependencies: TASK-401
- Risk: Medium
- Expected Performance: +5% learning speed

**TASK-403: Cascading Traces for Long Delays** (Priority: 2, Effort: medium)
- Description: Multiple timescales for credit assignment (fast + slow)
- Acceptance Criteria:
  - Fast trace: γ=0.7, captures recent events (seconds)
  - Medium trace: γ=0.9, captures episode (minutes)
  - Slow trace: γ=0.98, captures session (hours)
  - Combined: weighted sum based on delay
- Files to Modify:
  - `/mnt/projects/t4d/t4dm/src/t4dm/learning/eligibility_trace.py`
- Mathematical Formulations:
  ```python
  class CascadingEligibility:
      """Multi-timescale eligibility traces."""

      def __init__(self, shape):
          self.fast = EligibilityTrace(shape, decay=0.7)
          self.medium = EligibilityTrace(shape, decay=0.9)
          self.slow = EligibilityTrace(shape, decay=0.98)

      def update(self, activity):
          self.fast.update(activity)
          self.medium.update(activity)
          self.slow.update(activity)

      def get(self, delay_seconds):
          """Get trace weighted by delay."""
          if delay_seconds < 10:
              # Recent: mostly fast trace
              return 0.7 * self.fast.get() + 0.3 * self.medium.get()
          elif delay_seconds < 300:
              # Medium delay: balanced
              return 0.5 * self.medium.get() + 0.5 * self.slow.get()
          else:
              # Long delay: mostly slow trace
              return 0.9 * self.slow.get() + 0.1 * self.medium.get()
  ```
- Dependencies: TASK-402
- Risk: Low
- Expected Performance: +10% accuracy on delayed rewards

**TASK-404: Integrate with Neuromodulator Orchestra** (Priority: 1, Effort: medium)
- Description: Connect eligibility traces to DA/5-HT systems
- Acceptance Criteria:
  - DA (surprise) → fast trace updates
  - 5-HT (mood) → slow trace updates
  - Combined signal for three-factor learning
  - Backward compatible with existing orchestra
- Files to Modify:
  - `/mnt/projects/t4d/t4dm/src/t4dm/learning/neuromodulators.py`
  - `/mnt/projects/t4d/t4dm/src/t4dm/core/learned_gate.py`
- Integration Pattern:
  ```python
  class LearnedMemoryGate:
      def __init__(self, ...):
          # ... existing code ...
          self.eligibility = CascadingEligibility(shape=(self.feature_dim,))

      def update(self, memory_id, utility, delay_seconds=0):
          # Get neuromod signals
          neuromod_state = self.neuromod.get_state()

          # Compute combined signal
          da_signal = neuromod_state.dopamine_rpe
          serotonin_signal = neuromod_state.serotonin_mood - 0.5

          combined = 0.7 * da_signal + 0.3 * serotonin_signal

          # Get features and trace
          features = self.pending_labels[memory_id][0]
          trace = self.eligibility.get(delay_seconds)

          # Three-factor update
          delta = trace * combined * (utility - self.μ @ features)
          self.μ += self.η * delta

          # Update trace
          self.eligibility.update(features)
  ```
- Dependencies: TASK-403
- Risk: Medium
- Expected Performance: +8% utility prediction accuracy

---

### Phase 5: Multi-Timescale Dynamics (6 days)

**Goal**: Hierarchical temporal constants for 2-6% accuracy gains

#### Tasks

**TASK-501: Implement Temporal Constant Layers** (Priority: 1, Effort: medium)
- Description: Different time constants τ per processing layer
- Acceptance Criteria:
  - Layer 1 (input): τ=10ms (fast, reactive)
  - Layer 2 (intermediate): τ=100ms (integration)
  - Layer 3 (memory): τ=1s (stable representations)
  - Continuous-time update: dx/dt = (input - x) / τ
  - Discrete approximation: x(t+Δt) = x(t) + Δt/τ · (input - x)
- Files to Create:
  - `/mnt/projects/t4d/t4dm/src/t4dm/learning/temporal_layer.py`
- Mathematical Formulations:
  ```python
  class TemporalLayer:
      """Layer with temporal dynamics."""

      def __init__(self, input_dim, output_dim, tau_ms=100):
          self.W = np.random.randn(output_dim, input_dim) * 0.01
          self.tau = tau_ms / 1000.0  # Convert to seconds
          self.state = np.zeros(output_dim)  # Current activation
          self.dt = 0.01  # 10ms timestep

      def forward(self, x):
          """
          Temporal integration:

          dh/dt = (Wx - h) / τ

          Discrete: h(t+Δt) = h(t) + (Δt/τ) · (Wx - h)
          """
          target = self.W @ x

          # Exponential approach to target
          self.state += (self.dt / self.tau) * (target - self.state)

          return self.state.copy()

      def reset(self):
          """Reset state (e.g., between episodes)."""
          self.state.fill(0)
  ```
- Test Cases:
  ```python
  def test_temporal_integration():
      layer = TemporalLayer(10, 5, tau_ms=100)

      # Step input
      x = np.ones(10)

      # Should approach steady state exponentially
      states = []
      for _ in range(100):  # 1 second
          h = layer.forward(x)
          states.append(h.copy())

      # Should reach ~63% of final value after 1τ
      assert np.linalg.norm(states[10]) > 0.6 * np.linalg.norm(states[-1])
  ```
- Dependencies: None
- Risk: Low
- Expected Performance: Smooth temporal integration

**TASK-502: Hierarchical Temporal Processing** (Priority: 1, Effort: medium)
- Description: Stack temporal layers with increasing τ
- Acceptance Criteria:
  - 3-layer hierarchy: τ=[10ms, 100ms, 1s]
  - Fast layers track transients, slow layers track trends
  - Output: multi-timescale representation
  - <5ms latency for 3 layers
- Files to Create:
  - `/mnt/projects/t4d/t4dm/src/t4dm/learning/temporal_hierarchy.py`
- Mathematical Formulations:
  ```python
  class TemporalHierarchy:
      """Stack of temporal layers with increasing timescales."""

      def __init__(self, layer_dims, taus):
          """
          Args:
              layer_dims: [input_dim, hidden1, hidden2, output_dim]
              taus: [τ1, τ2, τ3] in milliseconds
          """
          self.layers = [
              TemporalLayer(layer_dims[i], layer_dims[i+1], tau_ms=taus[i])
              for i in range(len(taus))
          ]

      def forward(self, x):
          """Forward pass through all layers."""
          h = x
          for layer in self.layers:
              h = layer.forward(h)
          return h

      def get_all_states(self):
          """Return states from all timescales."""
          return [layer.state.copy() for layer in self.layers]
  ```
- Dependencies: TASK-501
- Risk: Low
- Expected Performance: Multi-resolution temporal features

**TASK-503: Integrate into Episodic Encoding** (Priority: 1, Effort: medium)
- Description: Use temporal hierarchy for episode encoding
- Acceptance Criteria:
  - Replace static encoding with temporal integration
  - Fast layer: immediate content features
  - Medium layer: contextual integration
  - Slow layer: stable memory representation
  - Improves sequential memory (>5% accuracy)
- Files to Modify:
  - `/mnt/projects/t4d/t4dm/src/t4dm/memory/episodic.py`
- Integration Pattern:
  ```python
  class EpisodicMemory:
      def __init__(self, ...):
          self.temporal_encoder = TemporalHierarchy(
              layer_dims=[1024, 512, 256, 128],
              taus=[10, 100, 1000]  # 10ms, 100ms, 1s
          )

      async def create(self, content, context, outcome):
          # Temporal encoding
          content_emb = await self.embed(content)
          temporal_repr = self.temporal_encoder.forward(content_emb)

          # Store multi-timescale representation
          # Can also store intermediate states for richer retrieval
          all_states = self.temporal_encoder.get_all_states()
          ...
  ```
- Dependencies: TASK-502
- Risk: Medium
- Expected Performance: +2-6% accuracy (per literature)

**TASK-504: Adaptive Timescale Selection** (Priority: 2, Effort: small)
- Description: Learn optimal τ per layer via meta-learning
- Acceptance Criteria:
  - Treat τ as learnable parameter
  - Gradient: ∂L/∂τ via implicit differentiation
  - Constrain: 10ms < τ < 10s
  - Converges to task-appropriate timescales
- Files to Modify:
  - `/mnt/projects/t4d/t4dm/src/t4dm/learning/temporal_layer.py`
- Mathematical Formulations:
  ```python
  class AdaptiveTemporalLayer(TemporalLayer):
      def __init__(self, ..., tau_init=100, tau_lr=0.001):
          super().__init__(..., tau_ms=tau_init)
          self.tau_lr = tau_lr
          self.log_tau = np.log(self.tau)  # Learn in log space

      def update_tau(self, loss_gradient):
          """
          Update timescale based on loss.

          ∂L/∂τ computed via implicit differentiation.
          """
          # Gradient through temporal dynamics
          grad_tau = loss_gradient * self.dt / (self.tau ** 2)

          # Update in log space (ensures τ > 0)
          self.log_tau -= self.tau_lr * grad_tau
          self.tau = np.exp(np.clip(self.log_tau, np.log(0.01), np.log(10)))
  ```
- Dependencies: TASK-503
- Risk: Low
- Expected Performance: 1-2% additional gain from optimization

---

### Phase 6: Titans-Style Neural Memory (12 days)

**Goal**: MLP-based memory with surprise-driven selective storage

#### Tasks

**TASK-601: Implement Memory MLP** (Priority: 1, Effort: large)
- Description: MLP that memorizes inputs via gradient descent
- Acceptance Criteria:
  - 3-layer MLP: [input_dim, 1024, 512, input_dim]
  - Perfect reconstruction after 10-20 gradient steps
  - Memory persists (no forgetting without decay)
  - <10ms forward pass
- Files to Create:
  - `/mnt/projects/t4d/t4dm/src/t4dm/memory/memory_mlp.py`
- Mathematical Formulations:
  ```python
  class MemoryMLP:
      """MLP that memorizes inputs via SGD."""

      def __init__(self, dim, hidden_dims=[1024, 512]):
          self.layers = []
          prev_dim = dim
          for h in hidden_dims:
              self.layers.append({
                  'W': np.random.randn(h, prev_dim) * 0.01,
                  'b': np.zeros(h)
              })
              prev_dim = h

          # Output layer
          self.layers.append({
              'W': np.random.randn(dim, prev_dim) * 0.01,
              'b': np.zeros(dim)
          })

          self.lr = 0.01

      def forward(self, x):
          """Forward pass through MLP."""
          h = x
          for layer in self.layers[:-1]:
              h = np.tanh(layer['W'] @ h + layer['b'])

          # Output layer (linear)
          output = self.layers[-1]['W'] @ h + self.layers[-1]['b']
          return output

      def memorize(self, x, n_steps=20):
          """
          Memorize input x by training to reconstruct it.

          Loss: ||MLP(x) - x||²
          """
          for _ in range(n_steps):
              # Forward pass
              h_list = [x]
              for layer in self.layers[:-1]:
                  h = np.tanh(layer['W'] @ h_list[-1] + layer['b'])
                  h_list.append(h)

              output = self.layers[-1]['W'] @ h_list[-1] + self.layers[-1]['b']

              # Backprop
              loss = output - x
              delta = loss

              # Output layer
              grad_W = np.outer(delta, h_list[-1])
              self.layers[-1]['W'] -= self.lr * grad_W
              self.layers[-1]['b'] -= self.lr * delta

              # Hidden layers (simplified backprop)
              for i in range(len(self.layers) - 2, -1, -1):
                  delta = self.layers[i+1]['W'].T @ delta
                  delta = delta * (1 - h_list[i+1]**2)  # tanh derivative

                  grad_W = np.outer(delta, h_list[i])
                  self.layers[i]['W'] -= self.lr * grad_W
                  self.layers[i]['b'] -= self.lr * delta

      def recall(self, query, n_iters=5):
          """
          Recall memorized pattern via gradient descent on query.

          Start with query, optimize to minimize reconstruction error.
          """
          x = query.copy()
          for _ in range(n_iters):
              # Gradient of ||MLP(x) - x||² w.r.t. x
              output = self.forward(x)
              grad = 2 * (output - x)
              x -= 0.1 * grad

          return x
  ```
- Test Cases:
  ```python
  def test_memory_mlp_memorization():
      mlp = MemoryMLP(dim=256)

      # Random input
      x = np.random.randn(256)

      # Memorize
      mlp.memorize(x, n_steps=20)

      # Recall
      output = mlp.forward(x)

      # Should perfectly reconstruct
      assert np.linalg.norm(output - x) < 0.01
  ```
- Dependencies: None
- Risk: Low
- Expected Performance: <0.01 reconstruction error

**TASK-602: Surprise Mechanism** (Priority: 1, Effort: medium)
- Description: Selective memorization based on prediction error
- Acceptance Criteria:
  - Surprise = ||MLP(x) - x||² (reconstruction error)
  - Memorize if surprise > threshold θ
  - θ adapts to maintain target storage rate
  - Prevents memory overflow
- Files to Modify:
  - `/mnt/projects/t4d/t4dm/src/t4dm/memory/memory_mlp.py`
- Mathematical Formulations:
  ```python
  class SurpriseMemory(MemoryMLP):
      """MLP with surprise-driven selective memorization."""

      def __init__(self, dim, surprise_threshold=0.5, target_rate=0.1):
          super().__init__(dim)
          self.theta = surprise_threshold
          self.target_rate = target_rate
          self.recent_surprises = deque(maxlen=100)

      def compute_surprise(self, x):
          """Compute prediction error (surprise)."""
          output = self.forward(x)
          surprise = np.linalg.norm(output - x) ** 2
          return surprise

      def should_memorize(self, x):
          """Decide whether to memorize based on surprise."""
          surprise = self.compute_surprise(x)
          self.recent_surprises.append(surprise)

          # Adaptive threshold
          if len(self.recent_surprises) >= 50:
              # Adjust θ to maintain target storage rate
              sorted_surprises = sorted(self.recent_surprises)
              percentile_idx = int(len(sorted_surprises) * (1 - self.target_rate))
              self.theta = sorted_surprises[percentile_idx]

          return surprise > self.theta

      def process(self, x):
          """Process input: memorize if surprising."""
          if self.should_memorize(x):
              self.memorize(x)
              return True, self.compute_surprise(x)
          return False, self.compute_surprise(x)
  ```
- Dependencies: TASK-601
- Risk: Low
- Expected Performance: 10% storage rate (90% filtered)

**TASK-603: Weight Decay Forgetting** (Priority: 1, Effort: small)
- Description: Gradual forgetting via L2 weight decay
- Acceptance Criteria:
  - Weights decay: W(t+1) = (1-λ)W(t)
  - λ=0.0001 per update (slow decay)
  - Old memories fade unless refreshed
  - Configurable decay rate
- Files to Modify:
  - `/mnt/projects/t4d/t4dm/src/t4dm/memory/memory_mlp.py`
- Mathematical Formulations:
  ```python
  class MemoryMLP:
      def __init__(self, ..., weight_decay=0.0001):
          # ... existing code ...
          self.weight_decay = weight_decay

      def apply_decay(self):
          """Apply weight decay (forgetting)."""
          for layer in self.layers:
              layer['W'] *= (1 - self.weight_decay)
              # Don't decay biases

      def memorize(self, x, n_steps=20):
          # ... existing code ...

          # Apply decay after memorization
          self.apply_decay()
  ```
- Dependencies: TASK-602
- Risk: Low
- Expected Performance: 50% memory strength after 5000 updates

**TASK-604: Integrate Titans Memory with Episodes** (Priority: 1, Effort: large)
- Description: Use Titans memory for long-context retrieval
- Acceptance Criteria:
  - Store episode embeddings in Titans MLP
  - Surprise threshold from LearnedMemoryGate
  - Recall via gradient descent on query
  - Scale to 2M+ tokens (test with 10K episodes)
- Files to Modify:
  - `/mnt/projects/t4d/t4dm/src/t4dm/memory/episodic.py`
- Integration Pattern:
  ```python
  class EpisodicMemory:
      def __init__(self, ..., use_titans=True):
          self.titans_memory = SurpriseMemory(
              dim=1024,
              surprise_threshold=0.5,
              target_rate=0.1
          )
          self.use_titans = use_titans

      async def create(self, content, context, outcome):
          # ... existing code ...

          if self.use_titans:
              # Check surprise
              memorized, surprise = self.titans_memory.process(embedding)

              if memorized:
                  logger.info(f"Titans memorized episode (surprise={surprise:.3f})")

          # Still store in Qdrant for hybrid retrieval
          ...

      async def recall(self, query, limit=5):
          # Stage 1: Qdrant k-NN
          qdrant_results = await self._qdrant_search(query, limit=50)

          if self.use_titans:
              # Stage 2: Titans recall
              query_emb = await self.embed(query)
              titans_result = self.titans_memory.recall(query_emb, n_iters=5)

              # Find closest Qdrant result to Titans recall
              # (Titans provides attractor, Qdrant provides specific episode)
              for r in qdrant_results:
                  r.titans_score = cosine_similarity(r.embedding, titans_result)

              # Re-rank by Titans alignment
              qdrant_results.sort(key=lambda r: r.titans_score, reverse=True)

          return qdrant_results[:limit]
  ```
- Test Cases:
  ```python
  @pytest.mark.asyncio
  async def test_titans_long_context():
      mem = EpisodicMemory(use_titans=True)

      # Store 10K episodes
      for i in range(10000):
          await mem.create(f"Episode {i} content", ...)

      # Query
      results = await mem.recall("Episode 5000")

      # Should recall despite massive context
      assert "Episode 5000" in results[0].content
  ```
- Dependencies: TASK-603
- Risk: High - novel integration pattern
- Expected Performance: 80%+ recall@5 at 10K episodes

---

## Summary of Files to Create/Modify

### New Files (34 total)

#### Phase 0: Foundation
1. `/mnt/projects/t4d/t4dm/docs/NEURAL_COMPONENT_AUDIT.md`
2. `/mnt/projects/t4d/t4dm/tests/neural/test_pattern_completion.py`
3. `/mnt/projects/t4d/t4dm/tests/neural/test_sparse_retrieval.py`
4. `/mnt/projects/t4d/t4dm/tests/neural/test_learning_convergence.py`
5. `/mnt/projects/t4d/t4dm/tests/neural/test_multitimescale.py`
6. `/mnt/projects/t4d/t4dm/tests/neural/conftest.py`
7. `/mnt/projects/t4d/t4dm/benchmarks/baseline_metrics.json`
8. `/mnt/projects/t4d/t4dm/benchmarks/run_baseline.py`
9. `/mnt/projects/t4d/t4dm/src/t4dm/core/feature_extractor.py`

#### Phase 1: Forward-Forward
10. `/mnt/projects/t4d/t4dm/src/t4dm/learning/forward_forward.py`
11. `/mnt/projects/t4d/t4dm/benchmarks/ff_hyperparameter_search.py`
12. `/mnt/projects/t4d/t4dm/config/ff_optimal_params.json`

#### Phase 2: Hopfield Memory
13. `/mnt/projects/t4d/t4dm/src/t4dm/learning/sparse_attention.py`
14. `/mnt/projects/t4d/t4dm/src/t4dm/memory/hopfield_memory.py`
15. `/mnt/projects/t4d/t4dm/benchmarks/hopfield_capacity_test.py`
16. `/mnt/projects/t4d/t4dm/docs/HOPFIELD_CAPACITY_ANALYSIS.md`

#### Phase 3: Dendritic Computation
17. `/mnt/projects/t4d/t4dm/src/t4dm/learning/dendritic_neuron.py`
18. `/mnt/projects/t4d/t4dm/src/t4dm/learning/dendritic_layer.py`

#### Phase 4: Enhanced Neuromodulation
19. `/mnt/projects/t4d/t4dm/src/t4dm/learning/eligibility_trace.py`

#### Phase 5: Multi-Timescale
20. `/mnt/projects/t4d/t4dm/src/t4dm/learning/temporal_layer.py`
21. `/mnt/projects/t4d/t4dm/src/t4dm/learning/temporal_hierarchy.py`

#### Phase 6: Titans Memory
22. `/mnt/projects/t4d/t4dm/src/t4dm/memory/memory_mlp.py`

### Modified Files (8 total)

#### Core Learning
1. `/mnt/projects/t4d/t4dm/src/t4dm/core/learned_gate.py` (Phases 0, 4)
2. `/mnt/projects/t4d/t4dm/src/t4dm/learning/neuromodulators.py` (Phase 4)

#### Memory Systems
3. `/mnt/projects/t4d/t4dm/src/t4dm/memory/episodic.py` (Phases 1, 2, 3, 5, 6)

#### Learning Components
4. `/mnt/projects/t4d/t4dm/src/t4dm/learning/forward_forward.py` (Phase 4)
5. `/mnt/projects/t4d/t4dm/src/t4dm/learning/dendritic_neuron.py` (Phases 3, 4)
6. `/mnt/projects/t4d/t4dm/src/t4dm/learning/temporal_layer.py` (Phase 5)
7. `/mnt/projects/t4d/t4dm/src/t4dm/memory/memory_mlp.py` (Phase 6)

---

## Dependency Graph

```
Phase 0 (Foundation)
├─→ Phase 1 (Forward-Forward)
│   └─→ Phase 4 (Enhanced Neuromod)
├─→ Phase 2 (Hopfield Memory)
│   └─→ Phase 4 (Enhanced Neuromod)
├─→ Phase 3 (Dendritic Computation)
│   └─→ Phase 4 (Enhanced Neuromod)
├─→ Phase 5 (Multi-Timescale)
│   └─→ Phase 4 (Enhanced Neuromod)
└─→ Phase 6 (Titans Memory)
    └─→ Phase 4 (Enhanced Neuromod)

Phase 4 (Enhanced Neuromod) is the convergence point
```

**Critical Path**: Phase 0 → Phase 2 → Phase 4 → Phase 6 (longest sequence)

---

## Risk Assessment

### High-Risk Changes
1. **TASK-103**: FF integration into episodic.py (core path modification)
2. **TASK-203**: Hopfield integration (changes retrieval semantics)
3. **TASK-302**: Dendritic layers (significant architecture change)
4. **TASK-604**: Titans integration (novel pattern)

**Mitigation**:
- Feature flags for each component (easy rollback)
- A/B testing framework
- Comprehensive unit + integration tests
- Gradual rollout (enable per-session)

### Medium-Risk Changes
1. **TASK-004**: Feature extractor refactor (touches many files)
2. **TASK-404**: Neuromod integration (complex state management)
3. **TASK-503**: Temporal encoding (performance concerns)

**Mitigation**:
- Extensive benchmarking before/after
- Backward compatibility via adapters
- Performance budgets (<5ms overhead)

### Low-Risk Changes
- All test infrastructure (Phase 0)
- Isolated component implementations
- Documentation updates

---

## Performance Expectations

### Latency Budget
- **Storage**: Currently 15-25ms
  - FF overhead: +3ms → 18-28ms total (acceptable)
  - Titans surprise: +2ms → 20-30ms total

- **Retrieval**: Currently 30-50ms
  - Hopfield refinement: +10ms → 40-60ms total
  - Dendritic context: +2ms → 42-62ms total
  - **Target**: <70ms p95

### Accuracy Improvements
- **Forward-Forward**: +5-10% retrieval precision
- **Hopfield Memory**: +10-15% recall@10
- **Dendritic Computation**: +10-15% context-dependent accuracy
- **Multi-Timescale**: +2-6% overall accuracy
- **Titans Memory**: Enables 10K+ episode contexts

**Cumulative Expected Gain**: +25-40% across benchmarks

### Memory Footprint
- **Current**: ~50MB (model parameters + caches)
- **After Upgrade**: ~200MB
  - FF layers: +30MB
  - Hopfield memory: +50MB (10K patterns × 5KB)
  - Dendritic layers: +20MB
  - Temporal layers: +10MB
  - Titans MLP: +80MB (1024×512×1024 parameters)

**Acceptable**: Modern machines have >8GB RAM

---

## Testing Strategy

### Unit Tests (per phase)
- Isolated component tests (math correctness)
- Property-based tests (hypothesis)
- Edge cases (empty input, extreme values)
- **Target**: 100% coverage for new code

### Integration Tests
- End-to-end storage + retrieval
- Multi-component interaction
- Neuromodulator coordination
- **Target**: 90%+ coverage

### Performance Tests
- Latency benchmarks (p50, p95, p99)
- Throughput tests (queries/sec)
- Memory profiling
- **Target**: No regression vs baseline

### Biological Validation
- Pattern completion (DG/CA3 analog)
- Sparse coding (10-20% active)
- Temporal integration (τ matching)
- **Target**: Matches neuroscience literature

---

## Rollout Plan

### Week 1-2: Phase 0 (Foundation)
- Audit existing components
- Build test harness
- Establish baselines
- Refactor feature extraction

### Week 3-4: Phase 1 (Forward-Forward)
- Implement FF layers
- Negative data generation
- Integration with episodic
- Hyperparameter tuning

### Week 5-6: Phase 2 (Hopfield Memory)
- α-entmax implementation
- Energy-based retrieval
- Qdrant integration
- Capacity testing

### Week 7-8: Phase 3 (Dendritic Computation)
- Two-compartment neurons
- Dendritic layers
- Branch-specific plasticity
- Context-gated learning

### Week 9: Phase 4 (Enhanced Neuromodulation)
- Eligibility traces
- Three-factor learning
- Cascading traces
- Orchestra integration

### Week 10: Phase 5 (Multi-Timescale) + Phase 6 (Titans Memory)
- Temporal layers (days 1-3)
- Hierarchical processing (days 4-5)
- Titans MLP + surprise (days 6-8)
- Final integration + testing (days 9-10)

---

## Backward Compatibility

### Feature Flags
```python
# config/neural_features.py
class NeuralConfig:
    enable_forward_forward: bool = False
    enable_hopfield: bool = False
    enable_dendritic: bool = False
    enable_eligibility: bool = True  # Safe to enable by default
    enable_multitimescale: bool = False
    enable_titans: bool = False
```

### Graceful Degradation
- All components have fallbacks to existing behavior
- Missing dependencies → log warning + continue
- Failed initialization → use heuristic alternatives

### Migration Path
1. Deploy with all flags OFF
2. Enable eligibility traces (lowest risk)
3. Enable per-session opt-in for other features
4. Monitor metrics (latency, accuracy, errors)
5. Gradual rollout to 100% traffic

---

## Success Metrics

### Primary KPIs
1. **Retrieval Precision@5**: 65% → 85%+ (target: +20%)
2. **Storage Decision Accuracy**: 72% → 82%+ (target: +10%)
3. **p95 Latency**: <70ms (acceptable degradation from 50ms)
4. **Memory Capacity**: 100K → 1M+ episodes

### Secondary KPIs
1. **Context-Dependent Accuracy**: +10-15%
2. **Pattern Completion**: 90%+ from 30% cues
3. **Sparse Retrieval**: 70-90% sparsity
4. **Long-Context Recall**: 80%+ at 10K episodes

### Deployment Criteria
- All primary KPIs met
- No P0/P1 bugs in production
- <5% error rate in neural components
- Positive user feedback (if applicable)

---

## Documentation Deliverables

1. **NEURAL_COMPONENT_AUDIT.md** - Existing system analysis
2. **HOPFIELD_CAPACITY_ANALYSIS.md** - Memory capacity study
3. **NEURAL_ARCHITECTURE.md** - Updated system design
4. **NEURAL_API.md** - Developer API reference
5. **MIGRATION_GUIDE.md** - Upgrade instructions
6. **BENCHMARKS.md** - Performance comparisons

---

## Open Questions / Future Work

### Phase 7+ (Future Enhancements)
1. **Attention Mechanisms**: Integrate with Transformers for query refinement
2. **Meta-Learning**: Learn learning rates, timescales, thresholds
3. **Continual Learning**: Prevent catastrophic forgetting at scale
4. **Neural Architecture Search**: Optimize layer sizes, depths
5. **Distributed Memory**: Shard across multiple nodes for >10M episodes

### Research Questions
1. Can Forward-Forward match backprop performance in practice?
2. What is the empirical capacity limit of Hopfield-FY memory?
3. How do dendritic layers interact with attention mechanisms?
4. Optimal timescale hierarchy for episodic vs semantic memory?
5. Titans MLP vs Transformer for long-context retrieval?

---

## Estimated Effort Summary

| Phase | Description | Tasks | Effort (days) | Risk |
|-------|-------------|-------|---------------|------|
| 0 | Foundation | 4 | 5 | Low |
| 1 | Forward-Forward | 4 | 10 | Medium |
| 2 | Hopfield Memory | 4 | 12 | High |
| 3 | Dendritic Computation | 3 | 8 | High |
| 4 | Enhanced Neuromodulation | 4 | 7 | Medium |
| 5 | Multi-Timescale | 4 | 6 | Low |
| 6 | Titans Memory | 4 | 12 | High |
| **Total** | | **27** | **60** | **Medium-High** |

**Note**: 60 development days ≈ 12 calendar weeks with 1 developer

---

## Contacts & Resources

### Papers
1. **Forward-Forward**: Hinton (2022) - arxiv:2212.13345
2. **Hopfield-FY**: Ramsauer et al. (2021) - arxiv:2008.02217
3. **Dendritic Computation**: Sacramento et al. (2018) - arxiv:1801.05439
4. **Eligibility Traces**: Sutton & Barto, Ch. 12
5. **Multi-Timescale RNNs**: Yamashita & Tani (2008)
6. **Titans Memory**: Granger et al. (2024) - arxiv:2501.00663

### Code References
- T4DM: `/mnt/projects/t4d/t4dm/`
- Existing tests: `/mnt/projects/t4d/t4dm/tests/`
- HSA implementation plan: `/mnt/projects/t4d/t4dm/docs/IMPLEMENTATION_PLAN_HSA.md`

---

**END OF ROADMAP**

*This is a living document. Update as implementation progresses.*
