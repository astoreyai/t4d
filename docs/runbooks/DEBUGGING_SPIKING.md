# T4DM Spiking Dynamics Debugging Runbook

**Location**: `/mnt/projects/t4d/t4dm/docs/runbooks/DEBUGGING_SPIKING.md`

Troubleshoot spiking neural network blocks, neuromodulator dynamics, STDP learning, and spike timing issues.

---

## Symptoms

### Spiking Block Not Firing
- Spike output all zeros despite non-zero input
- Membrane potential stuck at reset
- Logs show "No spikes detected" warnings

**Likelihood**: Threshold too high, input scaling wrong, or neuron state not initializing

### Extreme Sparsity (No Learning)
- Spike rate < 1% despite high inputs
- Gradient flow blocked during training
- Model not converging on memory tasks

**Likelihood**: LIF parameters (α, β, v_thresh) miscalibrated, input current scaling off

### Runaway Spiking (Epilepsy)
- Spike rate > 50% across neurons
- Exploding memory potentials
- System becomes unresponsive

**Likelihood**: Positive feedback loop, recurrent weights not clipped, neuromodulators gone haywire

### STDP Learning Not Engaging
- Weight deltas all zeros despite spikes
- Synaptic potentiation visible in logs but weights don't change
- Learning rate appears stuck

**Likelihood**: STDP enabled in config but gated off, causal timing window too narrow, or learning rate 0

### Neuromodulator Levels Frozen
- Dopamine, acetylcholine, norepinephrine stuck at initial values
- No modulation despite reward/novelty signals
- "NT injection" log messages absent

**Likelihood**: Neuromod bus not connected to learning signals, update function not called

### Membrane Potentials Diverging
- Membrane values > 100 or < -100
- Numerical instability in solver
- "Membrane overflow" warnings in logs

**Likelihood**: Missing clipping, learning rate too high for discrete time, α parameter unstable

---

## Diagnostic Commands

### Spiking Health & Status

```bash
# Check spiking block health
curl -H "X-API-Key: $T4DM_API_KEY" \
  "http://localhost:8000/spiking/health?session_id=$SESSION_ID"

# Get spike statistics (rate, firing patterns)
curl -H "X-API-Key: $T4DM_API_KEY" \
  "http://localhost:8000/spiking/stats?session_id=$SESSION_ID"

# Check LIF neuron parameters
curl -H "X-API-Key: $T4DM_API_KEY" \
  "http://localhost:8000/spiking/lif-params?session_id=$SESSION_ID"

# Get neuromodulator levels
curl -H "X-API-Key: $T4DM_API_KEY" \
  "http://localhost:8000/spiking/neuromodulators?session_id=$SESSION_ID"

# Check STDP learning metrics
curl -H "X-API-Key: $T4DM_API_KEY" \
  "http://localhost:8000/learning/stdp-metrics?session_id=$SESSION_ID"

# Get synaptic weight distributions
curl -H "X-API-Key: $T4DM_API_KEY" \
  "http://localhost:8000/spiking/weights?session_id=$SESSION_ID"

# Monitor spike raster in real-time
curl -H "X-API-Key: $T4DM_API_KEY" \
  "http://localhost:8000/visualization/spike-raster?session_id=$SESSION_ID&neurons=100"
```

### Trace Spiking Blocks

```bash
# Get activation trace for a single forward pass
curl -H "X-API-Key: $T4DM_API_KEY" \
  -X POST "http://localhost:8000/spiking/trace" \
  -H "Content-Type: application/json" \
  -d "{
    \"input\": [0.1, 0.2, ...],
    \"session_id\": \"$SESSION_ID\",
    \"verbose\": true
  }"

# Inspect membrane potentials over timesteps
curl -H "X-API-Key: $T4DM_API_KEY" \
  -X POST "http://localhost:8000/spiking/trace-membrane" \
  -H "Content-Type: application/json" \
  -d "{
    \"input\": [...],
    \"steps\": 10,
    \"session_id\": \"$SESSION_ID\"
  }"

# Get spike times for specific neurons
curl -H "X-API-Key: $T4DM_API_KEY" \
  "http://localhost:8000/spiking/spike-times?neurons=0,1,2&session_id=$SESSION_ID"

# Visualize thalamic gate activity
curl -H "X-API-Key: $T4DM_API_KEY" \
  "http://localhost:8000/visualization/thalamic-gate?session_id=$SESSION_ID"

# Check spike attention weights
curl -H "X-API-Key: $T4DM_API_KEY" \
  "http://localhost:8000/spiking/attention-weights?session_id=$SESSION_ID"
```

### Neuromodulator Inspection

```bash
# Get dopamine trace
curl -H "X-API-Key: $T4DM_API_KEY" \
  "http://localhost:8000/spiking/dopamine-trace?session_id=$SESSION_ID"

# Get acetylcholine levels
curl -H "X-API-Key: $T4DM_API_KEY" \
  "http://localhost:8000/spiking/ach-levels?session_id=$SESSION_ID"

# Get norepinephrine and serotonin
curl -H "X-API-Key: $T4DM_API_KEY" \
  "http://localhost:8000/spiking/monoamine-trace?session_id=$SESSION_ID"

# Check neuromod injection points
curl -H "X-API-Key: $T4DM_API_KEY" \
  "http://localhost:8000/spiking/nt-injection-schedule?session_id=$SESSION_ID"

# Visualize NT modulation
curl -H "X-API-Key: $T4DM_API_KEY" \
  "http://localhost:8000/visualization/neuromod-heatmap?session_id=$SESSION_ID"
```

### Learning & Plasticity

```bash
# Get STDP weight changes
curl -H "X-API-Key: $T4DM_API_KEY" \
  "http://localhost:8000/learning/weight-deltas?session_id=$SESSION_ID&layer=2"

# Check causal timing windows
curl -H "X-API-Key: $T4DM_API_KEY" \
  "http://localhost:8000/learning/stdp-windows?session_id=$SESSION_ID"

# Get Hebbian potentiation rate
curl -H "X-API-Key: $T4DM_API_KEY" \
  "http://localhost:8000/learning/hebbian-trace?session_id=$SESSION_ID"

# Check homeostatic plasticity
curl -H "X-API-Key: $T4DM_API_KEY" \
  "http://localhost:8000/learning/homeostatic-state?session_id=$SESSION_ID"

# Visualize weight histograms
curl -H "X-API-Key: $T4DM_API_KEY" \
  "http://localhost:8000/visualization/weight-distribution?session_id=$SESSION_ID"
```

### Logs & Performance

```bash
# Tail spiking-related logs
docker logs t4dm-service | grep -i "spike\|lif\|stdp\|neuromod" | tail -50

# Monitor spike rate per timestep
watch -n 1 'curl -s -H "X-API-Key: $T4DM_API_KEY" \
  http://localhost:8000/spiking/stats | grep spike_rate'

# Check for gradient flow
docker logs t4dm-service | grep -i "gradient\|backprop" | tail -20

# Monitor learning rate scheduling
docker logs t4dm-service | grep -i "learning_rate\|lr_schedule" | tail -20
```

---

## Code-Based Debugging

### Trace Spiking Block Outputs

```python
from t4dm.qwen.unified_model import UnifiedModel
import torch

model = UnifiedModel(model_name="Qwen/Qwen2.5-3B-Instruct")

# Get a sample input
input_ids = torch.tensor([[101, 2054, 2003, 102]])  # "What is..."

# Forward pass with spike tracing
with torch.no_grad():
    outputs = model(
        input_ids=input_ids,
        return_spike_trace=True,  # Enable tracing
    )

# Inspect spike traces
spike_trace = outputs.spike_trace
for block_idx, block_trace in enumerate(spike_trace.items()):
    spikes = block_trace['spikes']
    spike_rate = spikes.float().mean().item()
    print(f"Block {block_idx}: {spike_rate:.1%} spike rate")

    # Show per-neuron rates
    per_neuron = spikes.float().mean(dim=0)
    print(f"  Min rate: {per_neuron.min():.3f}")
    print(f"  Max rate: {per_neuron.max():.3f}")
    print(f"  Median rate: {per_neuron.median():.3f}")
```

### Inspect Membrane Potentials

```python
from t4dm.spiking.lif import LIFNeuron
import torch

lif = LIFNeuron(size=256, alpha=0.9, v_thresh=1.0)

# Run multiple timesteps with trace
input_seq = torch.randn(10, 256)  # 10 timesteps
membrane_trace = []

u = torch.zeros(256)
for t, input_current in enumerate(input_seq):
    spikes, u = lif(input_current, u)
    membrane_trace.append(u.clone().detach())
    print(f"t={t}: mean_u={u.mean():.3f}, max_u={u.max():.3f}, spikes={spikes.sum():.0f}")

# Analyze stability
import numpy as np
membrane_array = torch.stack(membrane_trace).numpy()
print(f"\nMembrane stability:")
print(f"  Mean: {membrane_array.mean():.3f}")
print(f"  Std: {membrane_array.std():.3f}")
print(f"  Min: {membrane_array.min():.3f}")
print(f"  Max: {membrane_array.max():.3f}")

# Check for saturation
saturated = (membrane_array > 10.0).sum()
print(f"  Saturated timesteps: {saturated}")
```

### Check Neuromodulator State

```python
from t4dm.spiking.neuromod_bus import NeuromodBus

neuromod = NeuromodBus()

# Get current NT levels
state = neuromod.get_state()
print("Neuromodulator Levels:")
print(f"  Dopamine (DA): {state['DA']:.3f}")
print(f"  Acetylcholine (ACh): {state['ACh']:.3f}")
print(f"  Norepinephrine (NE): {state['NE']:.3f}")
print(f"  Serotonin (5-HT): {state['5HT']:.3f}")

# Simulate reward signal (increases dopamine)
neuromod.inject_reward(reward=1.0)
print("\nAfter reward injection:")
state = neuromod.get_state()
print(f"  Dopamine (DA): {state['DA']:.3f}")

# Get NT injection schedule
schedule = neuromod.get_injection_schedule()
print(f"\nNT injection schedule ({len(schedule)} events):")
for event in schedule[:5]:
    print(f"  {event}")
```

### Debug STDP Learning

```python
from t4dm.learning.stdp import STDPLearner
import torch

stdp = STDPLearner(
    window_pre_to_post=20,  # Pre→post: Δt < 20 ms
    window_post_to_pre=20,  # Post→pre: Δt < 20 ms
    tau_plus=20.0,
    tau_minus=20.0,
)

# Simulate pre and post spike times
pre_spike_times = torch.tensor([10, 50, 100])
post_spike_times = torch.tensor([15, 55, 105])

# Calculate weight change
dw = stdp.compute_weight_change(
    pre_spike_times=pre_spike_times,
    post_spike_times=post_spike_times,
    current_weight=0.5,
)

print(f"Weight change from STDP: {dw:.6f}")
print(f"  Pre→post (causal): strengthens weight")
print(f"  Post→pre (anticausal): weakens weight")

# Check causal window
print("\nCausal window analysis:")
for pre_t in pre_spike_times:
    for post_t in post_spike_times:
        delta_t = post_t - pre_t
        weight_change = stdp.window_function(delta_t)
        print(f"  Δt={delta_t:3d} ms: dw={weight_change:.6f}")
```

### Monitor Homeostatic Plasticity

```python
from t4dm.learning.homeostasis import HomeostasisControl
import torch

homeostasis = HomeostasisControl(
    target_rate=0.1,  # 10% spike rate
    learning_rate=0.001,
)

# Simulate firing rate history
spike_rates = [0.02, 0.05, 0.08, 0.12, 0.15, 0.11, 0.09, 0.10, 0.10]

print("Homeostatic adjustment:")
for t, rate in enumerate(spike_rates):
    adjustment = homeostasis.compute_adjustment(current_rate=rate)
    print(f"  t={t}: rate={rate:.2%}, adjustment={adjustment:.4f}")

# Check if converging to target
final_error = abs(spike_rates[-1] - homeostasis.target_rate)
print(f"\nFinal error to target: {final_error:.3%}")
```

### Analyze Attention Weights

```python
from t4dm.spiking.spike_attention import SpikeAttention
import torch

attn = SpikeAttention(dim=256, num_heads=8)

# Simulate spike-based keys and queries
spikes_t = torch.randint(0, 2, (8, 256)).float()  # Spike raster
spikes_tm1 = torch.randint(0, 2, (8, 256)).float()  # Previous spikes

# Compute STDP-weighted attention
attention_weights = attn(
    spikes=spikes_t,
    spikes_prev=spikes_tm1,
)

print("Attention weight statistics:")
print(f"  Mean: {attention_weights.mean():.3f}")
print(f"  Std: {attention_weights.std():.3f}")
print(f"  Min: {attention_weights.min():.3f}")
print(f"  Max: {attention_weights.max():.3f}")

# Check for attention collapse
if attention_weights.std() < 0.01:
    print("  WARNING: Attention weights have very low variance (collapse)")

# Check for attention spikes
if attention_weights.max() > 0.9:
    print("  WARNING: Some attention weights > 0.9 (winner-take-all)")
```

---

## Common Issues & Solutions

### Issue: No Spikes (Spike Rate = 0%)

**Symptoms**:
- All spike outputs are 0
- Membrane potentials at threshold or below
- Model cannot learn

**Diagnosis**:
```python
from t4dm.spiking.lif import LIFNeuron
import torch

lif = LIFNeuron(size=256)
u = torch.zeros(256)

# Try strong input
input_current = torch.ones(256) * 2.0  # Strong input
spikes, u = lif(input_current, u)

print(f"Spikes with strong input: {spikes.sum():.0f}")
print(f"Membrane potential: mean={u.mean():.3f}, max={u.max():.3f}")

# Check threshold
print(f"Threshold: {lif.v_thresh}")
print(f"Membrane > threshold: {(u > lif.v_thresh).sum():.0f} neurons")
```

**Solutions**:

1. **Lower threshold**:
   ```python
   lif = LIFNeuron(size=256, v_thresh=0.5)  # was 1.0
   ```

2. **Increase input scaling**:
   ```python
   input_current = input_current * 5.0  # Scale inputs
   ```

3. **Increase leak decay α** (retain more membrane):
   ```python
   lif = LIFNeuron(size=256, alpha=0.95)  # was 0.9
   ```

4. **Check input distribution**:
   ```python
   print(f"Input mean: {input_current.mean():.3f}")
   print(f"Input std: {input_current.std():.3f}")
   # Inputs should be ~N(0, 0.1) or scale appropriately
   ```

### Issue: Runaway Spiking (> 50% Spike Rate)

**Symptoms**:
- Spike rate climbs over time
- Model becomes unresponsive
- "Spike explosion" warnings in logs

**Diagnosis**:
```python
# Monitor spike rate over iterations
spike_rates = []
for step in range(100):
    spikes, u = lif(input_seq[step], u)
    rate = spikes.float().mean().item()
    spike_rates.append(rate)
    if rate > 0.5:
        print(f"Step {step}: RUNAWAY at {rate:.1%}")

# Check for positive feedback
print(f"Rate trend: {spike_rates[-10:]}")
```

**Solutions**:

1. **Clip recurrent weights**:
   ```python
   # In training loop
   with torch.no_grad():
       model.spiking_blocks[0].recurrent.weight.clamp_(-1.0, 1.0)
   ```

2. **Increase soft reset strength β**:
   ```python
   lif = LIFNeuron(size=256, beta=2.0)  # was 1.0
   ```

3. **Add firing rate regularization**:
   ```python
   spike_loss = (spikes.mean() - 0.1) ** 2  # Target 10% rate
   total_loss = main_loss + 0.01 * spike_loss
   ```

4. **Check for positive feedback loops**:
   ```python
   # Verify recurrent weights are not strongly positive
   recurrent_weights = model.spiking_blocks[0].recurrent.weight
   print(f"Recurrent weight range: {recurrent_weights.min():.3f} to {recurrent_weights.max():.3f}")
   if recurrent_weights.max() > 0.5:
       print("WARNING: Strong positive recurrence detected")
   ```

### Issue: STDP Not Updating Weights

**Symptoms**:
- Weight deltas computed but weights unchanged
- Loss not decreasing despite learning signal
- "STDP update skipped" in logs

**Diagnosis**:
```python
from t4dm.learning.stdp import STDPLearner

stdp = STDPLearner()

# Check if STDP is enabled
print(f"STDP enabled: {stdp.enabled}")
print(f"Learning rate: {stdp.learning_rate}")

# Test weight update
old_weight = 0.5
pre_spikes = torch.tensor([10, 50])
post_spikes = torch.tensor([15, 55])

new_weight = stdp.update_weight(
    old_weight, pre_spikes, post_spikes
)

print(f"Weight change: {old_weight} → {new_weight}")
if new_weight == old_weight:
    print("ERROR: Weight not changing!")
```

**Solutions**:

1. **Enable STDP in config**:
   ```bash
   export T4DM_STDP_ENABLED=true
   export T4DM_STDP_LEARNING_RATE=0.001
   ```

2. **Verify spike timing is causal**:
   ```python
   # Post should fire shortly after pre
   # Check actual spike times
   pre_times = [...]
   post_times = [...]
   for pre_t in pre_times:
       causal_posts = [p for p in post_times if 0 < p - pre_t < 20]
       print(f"Pre spike at {pre_t}: {len(causal_posts)} causal posts")
   ```

3. **Increase learning rate**:
   ```bash
   export T4DM_STDP_LEARNING_RATE=0.01  # was 0.001
   ```

4. **Widen causal window**:
   ```python
   stdp = STDPLearner(
       window_pre_to_post=50,  # was 20 ms
       tau_plus=50.0,
   )
   ```

### Issue: Neuromodulator Levels Frozen

**Symptoms**:
- DA, ACh, NE all stuck at initial values
- No modulation despite reward signals
- Learning metrics show zero neuromod effect

**Diagnosis**:
```python
from t4dm.spiking.neuromod_bus import NeuromodBus

neuromod = NeuromodBus()

# Check if updating
state1 = neuromod.get_state()
neuromod.inject_reward(reward=1.0)  # Increases dopamine
state2 = neuromod.get_state()

if state1 == state2:
    print("ERROR: Neuromod not updating")
else:
    print(f"DA before: {state1['DA']}, after: {state2['DA']}")
```

**Solutions**:

1. **Check neuromod bus is initialized**:
   ```python
   # Verify injection points are connected
   print(f"NT injection schedule: {neuromod.get_injection_schedule()}")
   if not neuromod.get_injection_schedule():
       print("ERROR: No injection points configured")
   ```

2. **Manually trigger neuromod updates**:
   ```bash
   curl -H "X-API-Key: $T4DM_API_KEY" \
     -X POST "http://localhost:8000/spiking/inject-neuromod" \
     -d "{\"nt\": \"DA\", \"level\": 0.8, \"session_id\": \"$SESSION_ID\"}"
   ```

3. **Enable neuromod in training**:
   ```bash
   export T4DM_NEUROMOD_ENABLED=true
   export T4DM_NEUROMOD_UPDATE_FREQ=every_step
   ```

4. **Check reward signal is being computed**:
   ```python
   # Verify reward calculation
   from t4dm.learning.reward import compute_reward

   reward = compute_reward(
       outcome="success",
       prediction_error=0.1,
   )
   print(f"Computed reward: {reward}")
   ```

### Issue: Membrane Potentials Diverging

**Symptoms**:
- Membrane values grow unbounded (> 100)
- NaN values in gradients
- "Numerical instability" warnings

**Diagnosis**:
```python
import torch

# Check membrane range
u_trace = [...]  # Membrane values over time
u_array = torch.stack(u_trace)

print(f"Membrane range: {u_array.min():.1f} to {u_array.max():.1f}")
print(f"Contains NaN: {torch.isnan(u_array).any()}")
print(f"Contains Inf: {torch.isinf(u_array).any()}")

# Check gradients
if torch.autograd.is_grad_enabled():
    u_array.sum().backward()
    print(f"Gradient norms: {[p.grad.norm().item() for p in params]}")
```

**Solutions**:

1. **Clip membrane potentials**:
   ```python
   # In LIF forward pass
   u = torch.clamp(u, min=-10, max=10)
   ```

2. **Reduce learning rate**:
   ```bash
   export T4DM_LEARNING_RATE=0.0001  # was 0.001
   ```

3. **Reduce α (leak decay)**:
   ```python
   lif = LIFNeuron(size=256, alpha=0.8)  # was 0.9
   ```

4. **Add gradient clipping**:
   ```python
   torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
   ```

---

## Checklist for Spiking Debugging

- [ ] Check spike rate: `GET /spiking/stats`
- [ ] Verify LIF parameters: `GET /spiking/lif-params`
- [ ] Check neuromod levels: `GET /spiking/neuromodulators`
- [ ] Verify STDP is enabled: Check logs for "STDP enabled"
- [ ] Test forward pass with tracing: `POST /spiking/trace`
- [ ] Inspect membrane potentials: `POST /spiking/trace-membrane`
- [ ] Check causal timing: `GET /learning/stdp-windows`
- [ ] Verify attention weights: `GET /spiking/attention-weights`
- [ ] Monitor homeostatic plasticity: `GET /learning/homeostatic-state`
- [ ] Watch spike raster: `GET /visualization/spike-raster`
- [ ] Verify input distribution: Print sample inputs
- [ ] Check gradient flow: Monitor backprop logs
- [ ] Verify learning rate schedule: Check logs for adjustments
