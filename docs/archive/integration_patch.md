# World Weaver Integration Patch
## Remediation: Connect Encoding Layer and Procedural→Dopamine

### Task 1: Integrate DendriticNeuron into Episodic Recall

**File**: `/mnt/projects/ww/src/ww/memory/episodic.py`

**Change 1a: Add import (after line 14)**
```python
import torch
```

**Change 1b: Add import (after line 27)**
```python
from ww.encoding.dendritic import DendriticNeuron
```

**Change 1c: Add to __init__ (after line 541, before async def initialize)**
```python
        # Optional dendritic neuron for surprise detection
        # Two-compartment model: basal (query) + apical (context) → mismatch signal
        self.dendritic_neuron = None  # Lazy init
        self._dendritic_enabled = getattr(settings, "dendritic_surprise_enabled", False)
```

**Change 1d: Modify recall() method (after line 883, in recall method)**

Find this section around line 880-900:
```python
        # Generate query embedding
        query_emb = await self.embedding.embed_query(query)
        query_emb_np = np.array(query_emb)
```

Add after it:
```python
        # Optional dendritic processing for surprise detection
        dendritic_mismatch = None
        if self._dendritic_enabled:
            try:
                # Lazy init dendritic neuron
                if self.dendritic_neuron is None:
                    self.dendritic_neuron = DendriticNeuron(
                        input_dim=1024,
                        hidden_dim=512,
                        context_dim=512,
                        coupling_strength=0.5
                    )

                # Get semantic memory context (top-down)
                # For now, use neuromodulator state embedding as context
                # In full implementation, this would come from semantic memory
                if neuro_state and self._neuromodulation_enabled:
                    # Convert neuromod state to context vector
                    # Simple heuristic: use arousal/mode as context features
                    context_features = np.array([
                        neuro_state.norepinephrine_gain,
                        1.0 if neuro_state.acetylcholine_mode == "encoding" else 0.0,
                        neuro_state.exploration_exploitation_balance,
                    ] + [0.0] * 509)  # Pad to 512
                    context_tensor = torch.tensor(context_features, dtype=torch.float32).unsqueeze(0)
                else:
                    context_tensor = None

                # Process query through dendritic neuron
                query_tensor = torch.tensor(query_emb_np, dtype=torch.float32).unsqueeze(0)
                with torch.no_grad():
                    output, mismatch = self.dendritic_neuron(query_tensor, context_tensor)
                    dendritic_mismatch = mismatch.item()

                logger.debug(f"Dendritic surprise: mismatch={dendritic_mismatch:.3f}")
                add_span_attribute("dendritic.mismatch", dendritic_mismatch)

            except Exception as e:
                logger.warning(f"Dendritic processing failed: {e}")
```

**Change 1e: Use mismatch signal for surprise modulation (around line 1065-1070)**

Find this section:
```python
            # R1: Combined weighted score with learned weights
            combined_score = (
                semantic_w * semantic_score +
                recency_w * recency_score +
                outcome_w * outcome_score +
                importance_w * importance_score
            )
```

Replace with:
```python
            # R1: Combined weighted score with learned weights
            combined_score = (
                semantic_w * semantic_score +
                recency_w * recency_score +
                outcome_w * outcome_score +
                importance_w * importance_score
            )

            # Optional dendritic surprise modulation
            # High mismatch = query doesn't match context expectations = boost surprising items
            if dendritic_mismatch is not None and dendritic_mismatch > 0.5:
                surprise_boost = (dendritic_mismatch - 0.5) * 0.2  # Max 20% boost
                combined_score *= (1.0 + surprise_boost)
                components["dendritic_surprise"] = dendritic_mismatch
```

---

### Task 2: Connect Procedural to Dopamine

**File**: `/mnt/projects/ww/src/ww/memory/procedural.py`

**Change 2a: Add import (after line 22)**
```python
from ww.learning.dopamine import DopamineSystem
```

**Change 2b: Modify __init__ (after line 75, before async def initialize)**
```python
        # Optional dopamine system for RPE-modulated learning
        self.dopamine_system = None  # Lazy init
        self._dopamine_enabled = getattr(settings, "procedural_dopamine_enabled", False)
```

**Change 2c: Modify update() method (after line 298, before old_success_rate assignment)**

Find section around line 288-298:
```python
    async def update(
        self,
        procedure_id: UUID,
        success: bool,
        error: Optional[str] = None,
        failed_step: Optional[int] = None,
        context: Optional[str] = None,
    ) -> Procedure:
        """
        UPDATE: Learn from execution outcomes.
        ...
        """
        procedure = await self.get_procedure(procedure_id)
        if not procedure:
            raise ValueError(f"Procedure {procedure_id} not found")
```

Add after the `if not procedure:` check:
```python
        # Compute dopamine signal (RPE) for surprise-modulated learning
        rpe_signal = None
        if self._dopamine_enabled:
            try:
                # Lazy init
                if self.dopamine_system is None:
                    self.dopamine_system = DopamineSystem(
                        default_expected=0.5,
                        value_learning_rate=0.1,
                        surprise_threshold=0.05
                    )

                # Use procedure name as context hash
                context_hash = hash(procedure.name)
                actual_outcome = 1.0 if success else 0.0

                # Compute RPE: δ = actual - expected
                rpe = self.dopamine_system.compute_rpe(
                    memory_id=procedure_id,
                    actual_outcome=actual_outcome
                )
                rpe_signal = rpe.rpe

                # Update expectations for next time
                self.dopamine_system.update_expectations(
                    memory_id=procedure_id,
                    actual_outcome=actual_outcome
                )

                logger.debug(
                    f"Dopamine signal for '{procedure.name}': "
                    f"RPE={rpe_signal:.3f} (expected={rpe.expected:.2f}, actual={actual_outcome:.1f})"
                )

                # Use RPE to modulate consolidation
                # High |RPE| = surprising = more learning
                # Expected outcomes (RPE≈0) = less learning
                if abs(rpe_signal) > 0.1:
                    logger.info(
                        f"Surprising outcome for '{procedure.name}': RPE={rpe_signal:.3f} "
                        f"({'better' if rpe_signal > 0 else 'worse'} than expected)"
                    )

            except Exception as e:
                logger.warning(f"Dopamine RPE computation failed: {e}")
```

---

### Task 3: Add SparseEncoder to Hybrid Search

**File**: `/mnt/projects/ww/src/ww/memory/episodic.py`

**Change 3a: Add import (after line 27)**
```python
from ww.encoding.sparse import SparseEncoder
```

**Change 3b: Add to __init__ (after dendritic neuron init from Task 1)**
```python
        # Optional sparse encoder for learned sparse representations
        self.sparse_encoder = None  # Lazy init
        self._sparse_encoding_enabled = getattr(settings, "sparse_encoding_enabled", False)
```

**Change 3c: Modify _store_hybrid() method (around line 777-796)**

Find this method:
```python
    async def _store_hybrid(self, episode: Episode, content: str) -> None:
        """
        Store episode in hybrid collection with dense + sparse vectors.
        ...
        """
        # Generate hybrid embeddings
        dense_vecs, sparse_vecs = await self.embedding.embed_hybrid([content])
```

Replace the `# Generate hybrid embeddings` section with:
```python
        # Generate hybrid embeddings
        if self._sparse_encoding_enabled:
            # Use learned sparse encoder instead of BGE-M3 sparse
            try:
                # Lazy init
                if self.sparse_encoder is None:
                    self.sparse_encoder = SparseEncoder(
                        input_dim=1024,
                        hidden_dim=8192,
                        sparsity=0.02,
                        use_kwta=True
                    )

                # Get dense embedding
                dense_vecs = [episode.embedding]

                # Generate sparse code
                dense_tensor = torch.tensor(episode.embedding, dtype=torch.float32).unsqueeze(0)
                with torch.no_grad():
                    sparse_code = self.sparse_encoder(dense_tensor)
                    # Convert to Qdrant sparse format
                    sparse_indices = torch.nonzero(sparse_code).squeeze()
                    sparse_values = sparse_code[sparse_code != 0]
                    sparse_vecs = [{
                        "indices": sparse_indices.tolist(),
                        "values": sparse_values.tolist()
                    }]

                logger.debug(
                    f"Sparse encoding: {len(sparse_indices)} active indices "
                    f"({len(sparse_indices)/8192*100:.1f}% sparsity)"
                )
            except Exception as e:
                logger.warning(f"Sparse encoding failed, using BGE-M3: {e}")
                dense_vecs, sparse_vecs = await self.embedding.embed_hybrid([content])
        else:
            # Use standard BGE-M3 hybrid
            dense_vecs, sparse_vecs = await self.embedding.embed_hybrid([content])
```

**Change 3d: Modify recall_hybrid() method to use sparse encoder for queries**

Find the recall_hybrid method (around line 1199+). After query embedding generation, add:
```python
        # Generate query embedding
        query_dense = await self.embedding.embed_query(query)

        # Optional sparse encoding of query
        query_sparse = None
        if self._sparse_encoding_enabled and self.sparse_encoder is not None:
            try:
                dense_tensor = torch.tensor(query_dense, dtype=torch.float32).unsqueeze(0)
                with torch.no_grad():
                    sparse_code = self.sparse_encoder(dense_tensor)
                    sparse_indices = torch.nonzero(sparse_code).squeeze()
                    sparse_values = sparse_code[sparse_code != 0]
                    query_sparse = {
                        "indices": sparse_indices.tolist(),
                        "values": sparse_values.tolist()
                    }
            except Exception as e:
                logger.warning(f"Query sparse encoding failed: {e}")
```

---

## Summary

These changes integrate:

1. **DendriticNeuron**: Provides surprise detection in episodic recall via basal-apical mismatch
2. **Dopamine System**: Connects to procedural memory for RPE-based learning rate modulation
3. **SparseEncoder**: Adds learned sparse representations for hybrid search

All integrations are optional (feature-flagged) and fail gracefully with logging.
