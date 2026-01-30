#!/usr/bin/env python3
"""
Apply dendritic neuron and sparse encoder integration patch to episodic.py
"""

import re

# Read the file
with open('/mnt/projects/ww/src/ww/memory/episodic.py', 'r') as f:
    lines = f.readlines()

# Track modifications
modified = []
i = 0

while i < len(lines):
    line = lines[i]

    # Change 1a: Add torch import after numpy import
    if 'import numpy as np' in line:
        modified.append(line)
        modified.append('import torch\n')
        i += 1
        continue

    # Change 1b & 3a: Add dendritic and sparse imports after FeatureAligner import
    if 'from ww.memory.feature_aligner import FeatureAligner' in line:
        modified.append(line)
        modified.append('from ww.encoding.dendritic import DendriticNeuron\n')
        modified.append('from ww.encoding.sparse import SparseEncoder\n')
        i += 1
        continue

    # Change 1c & 3b: Add dendritic neuron and sparse encoder to __init__
    # Find the line with joint_optimization_enabled
    if 'self._joint_optimization_enabled = getattr(settings, "joint_optimization_enabled", True)' in line:
        modified.append(line)
        modified.append('\n')
        modified.append('        # Optional dendritic neuron for surprise detection\n')
        modified.append('        # Two-compartment model: basal (query) + apical (context) -> mismatch signal\n')
        modified.append('        self.dendritic_neuron = None  # Lazy init\n')
        modified.append('        self._dendritic_enabled = getattr(settings, "dendritic_surprise_enabled", False)\n')
        modified.append('\n')
        modified.append('        # Optional sparse encoder for learned sparse representations\n')
        modified.append('        self.sparse_encoder = None  # Lazy init\n')
        modified.append('        self._sparse_encoding_enabled = getattr(settings, "sparse_encoding_enabled", False)\n')
        i += 1
        continue

    # Change 1d: Add dendritic processing in recall method
    # Find the line "query_emb_np = np.array(query_emb)"
    if 'query_emb_np = np.array(query_emb)' in line and 'async def recall(' in ''.join(lines[max(0, i-20):i]):
        modified.append(line)
        modified.append('\n')
        modified.append('        # Optional dendritic processing for surprise detection\n')
        modified.append('        dendritic_mismatch = None\n')
        modified.append('        if self._dendritic_enabled:\n')
        modified.append('            try:\n')
        modified.append('                # Lazy init dendritic neuron\n')
        modified.append('                if self.dendritic_neuron is None:\n')
        modified.append('                    self.dendritic_neuron = DendriticNeuron(\n')
        modified.append('                        input_dim=1024,\n')
        modified.append('                        hidden_dim=512,\n')
        modified.append('                        context_dim=512,\n')
        modified.append('                        coupling_strength=0.5\n')
        modified.append('                    )\n')
        modified.append('\n')
        modified.append('                # Get semantic memory context (top-down)\n')
        modified.append('                # For now, use neuromodulator state as context\n')
        modified.append('                if neuro_state and self._neuromodulation_enabled:\n')
        modified.append('                    # Convert neuromod state to context vector\n')
        modified.append('                    context_features = np.array([\n')
        modified.append('                        neuro_state.norepinephrine_gain,\n')
        modified.append('                        1.0 if neuro_state.acetylcholine_mode == "encoding" else 0.0,\n')
        modified.append('                        neuro_state.exploration_exploitation_balance,\n')
        modified.append('                    ] + [0.0] * 509)  # Pad to 512\n')
        modified.append('                    context_tensor = torch.tensor(context_features, dtype=torch.float32).unsqueeze(0)\n')
        modified.append('                else:\n')
        modified.append('                    context_tensor = None\n')
        modified.append('\n')
        modified.append('                # Process query through dendritic neuron\n')
        modified.append('                query_tensor = torch.tensor(query_emb_np, dtype=torch.float32).unsqueeze(0)\n')
        modified.append('                with torch.no_grad():\n')
        modified.append('                    output, mismatch = self.dendritic_neuron(query_tensor, context_tensor)\n')
        modified.append('                    dendritic_mismatch = mismatch.item()\n')
        modified.append('\n')
        modified.append('                logger.debug(f"Dendritic surprise: mismatch={dendritic_mismatch:.3f}")\n')
        modified.append('                add_span_attribute("dendritic.mismatch", dendritic_mismatch)\n')
        modified.append('\n')
        modified.append('            except Exception as e:\n')
        modified.append('                logger.warning(f"Dendritic processing failed: {e}")\n')
        modified.append('\n')
        i += 1
        continue

    # Change 1e: Add dendritic surprise modulation to scoring
    # Find the components dict creation
    if '            components = {' in line and '"semantic": semantic_score,' in lines[i+1]:
        # Copy the components block
        modified.append(line)
        j = i + 1
        while j < len(lines) and '}' not in lines[j]:
            modified.append(lines[j])
            j += 1
        modified.append(lines[j])  # Add the closing brace line
        modified.append('\n')
        modified.append('            # Optional dendritic surprise modulation\n')
        modified.append('            # High mismatch = query doesn\'t match context expectations = boost surprising items\n')
        modified.append('            if dendritic_mismatch is not None and dendritic_mismatch > 0.5:\n')
        modified.append('                surprise_boost = (dendritic_mismatch - 0.5) * 0.2  # Max 20% boost\n')
        modified.append('                combined_score *= (1.0 + surprise_boost)\n')
        modified.append('                components["dendritic_surprise"] = dendritic_mismatch\n')
        i = j + 1
        continue

    # Change 3c: Modify _store_hybrid to use sparse encoder
    if 'async def _store_hybrid(self, episode: Episode, content: str) -> None:' in line:
        # Keep the method definition
        modified.append(line)
        # Copy docstring
        i += 1
        while i < len(lines) and ('"""' in lines[i] or lines[i].strip().startswith('#')):
            modified.append(lines[i])
            i += 1

        # Replace the embedding generation logic
        modified.append('        # Generate hybrid embeddings\n')
        modified.append('        if self._sparse_encoding_enabled:\n')
        modified.append('            # Use learned sparse encoder instead of BGE-M3 sparse\n')
        modified.append('            try:\n')
        modified.append('                # Lazy init\n')
        modified.append('                if self.sparse_encoder is None:\n')
        modified.append('                    self.sparse_encoder = SparseEncoder(\n')
        modified.append('                        input_dim=1024,\n')
        modified.append('                        hidden_dim=8192,\n')
        modified.append('                        sparsity=0.02,\n')
        modified.append('                        use_kwta=True\n')
        modified.append('                    )\n')
        modified.append('\n')
        modified.append('                # Get dense embedding\n')
        modified.append('                dense_vecs = [episode.embedding]\n')
        modified.append('\n')
        modified.append('                # Generate sparse code\n')
        modified.append('                dense_tensor = torch.tensor(episode.embedding, dtype=torch.float32).unsqueeze(0)\n')
        modified.append('                with torch.no_grad():\n')
        modified.append('                    sparse_code = self.sparse_encoder(dense_tensor)\n')
        modified.append('                    # Convert to Qdrant sparse format\n')
        modified.append('                    sparse_indices = torch.nonzero(sparse_code).squeeze()\n')
        modified.append('                    sparse_values = sparse_code[sparse_code != 0]\n')
        modified.append('                    if sparse_indices.dim() == 0:  # Single index case\n')
        modified.append('                        sparse_indices = sparse_indices.unsqueeze(0)\n')
        modified.append('                        sparse_values = sparse_values.unsqueeze(0)\n')
        modified.append('                    sparse_vecs = [{\n')
        modified.append('                        "indices": sparse_indices.tolist() if isinstance(sparse_indices, torch.Tensor) else [int(sparse_indices)],\n')
        modified.append('                        "values": sparse_values.tolist() if isinstance(sparse_values, torch.Tensor) else [float(sparse_values)]\n')
        modified.append('                    }]\n')
        modified.append('\n')
        modified.append('                logger.debug(\n')
        modified.append('                    f"Sparse encoding: {len(sparse_indices)} active indices "\n')
        modified.append('                    f"({len(sparse_indices)/8192*100:.1f}% sparsity)"\n')
        modified.append('                )\n')
        modified.append('            except Exception as e:\n')
        modified.append('                logger.warning(f"Sparse encoding failed, using BGE-M3: {e}")\n')
        modified.append('                dense_vecs, sparse_vecs = await self.embedding.embed_hybrid([content])\n')
        modified.append('        else:\n')
        modified.append('            # Use standard BGE-M3 hybrid\n')
        modified.append('            dense_vecs, sparse_vecs = await self.embedding.embed_hybrid([content])\n')
        modified.append('\n')

        # Skip original embedding generation lines
        while i < len(lines) and 'dense_vecs, sparse_vecs = await self.embedding.embed_hybrid' in lines[i]:
            i += 1

        # Continue with rest of method
        continue

    modified.append(line)
    i += 1

# Write the modified content
with open('/mnt/projects/ww/src/ww/memory/episodic.py', 'w') as f:
    f.writelines(modified)

print("✓ Episodic memory dendritic and sparse encoder integration applied")
