#!/bin/bash
# E2E test for World Weaver learning pipeline
# Tests: Retrieval → Outcome → Credit Assignment → Scorer Training

set -e

echo "=== World Weaver E2E Learning Pipeline Test ==="
cd "$(dirname "$0")/../.."
source ../projects/venv/bin/activate 2>/dev/null || source ~/projects/venv/bin/activate

# Test 1: Learning events creation
echo -e "\n[1/6] Testing learning event creation..."
python -c "
from ww.learning.events import (
    RetrievalEvent, OutcomeEvent, Experience,
    MemoryType, OutcomeType, ToonJSON
)
from uuid import uuid4

# Create retrieval event
mem_ids = [uuid4() for _ in range(3)]
ret = RetrievalEvent(
    query='test query for E2E',
    memory_type=MemoryType.EPISODIC,
    retrieved_ids=mem_ids,
    retrieval_scores={str(m)[:8]: 0.9-i*0.1 for i, m in enumerate(mem_ids)},
    session_id='e2e-test',
)
ret.compute_context_hash('E2E test context')
assert len(ret.context_hash) == 16, 'Context hash failed'

# Create outcome
outcome = OutcomeEvent(
    outcome_type=OutcomeType.SUCCESS,
    success_score=0.85,
    session_id='e2e-test',
)
outcome.compute_context_hash('E2E test context')
assert outcome.context_hash == ret.context_hash, 'Hash mismatch'

# Create experience
exp = Experience(
    query='test',
    memory_type=MemoryType.EPISODIC,
    component_vectors=[[0.9, 0.5, 0.7, 0.3] for _ in range(3)],
    outcome_score=0.85,
)

# Test ToonJSON
toon = ToonJSON()
ret_dict = ret.to_dict()
encoded = toon.encode(ret_dict)
assert len(encoded) < len(str(ret_dict)), 'ToonJSON not compressing'

print('✓ Learning events OK')
"

# Test 2: Neural scorer
echo -e "\n[2/6] Testing neural scorer..."
python -c "
import torch
from ww.learning.scorer import (
    LearnedRetrievalScorer,
    PrioritizedReplayBuffer,
    ListMLELoss,
    create_scorer,
)

# Create scorer on CPU for testing
scorer = create_scorer(device='cpu')
assert sum(p.numel() for p in scorer.parameters()) > 1000, 'Too few params'

# Forward pass
x = torch.randn(2, 5, 4)
scores = scorer(x)
assert scores.shape == (2, 5), f'Wrong shape: {scores.shape}'

# Score memories
component_vectors = [[0.9, 0.5, 0.7, 0.3], [0.7, 0.4, 0.6, 0.2]]
mem_scores = scorer.score_memories(component_vectors)
assert len(mem_scores) == 2, 'Score count mismatch'

# ListMLE loss
loss_fn = ListMLELoss()
scores = torch.randn(2, 5)
relevance = torch.rand(2, 5)
loss = loss_fn(scores, relevance)
assert loss >= 0, 'Negative loss'

# Replay buffer
buffer = PrioritizedReplayBuffer(capacity=100)
from ww.learning.events import Experience, MemoryType
for i in range(20):
    exp = Experience(
        query=f'q{i}',
        memory_type=MemoryType.EPISODIC,
        component_vectors=[[0.5]*4],
        per_memory_rewards={'m': 0.6},
    )
    buffer.add_from_experience(exp)
items, indices, weights = buffer.sample(5)
assert len(items) == 5, 'Sample failed'

print('✓ Neural scorer OK')
"

# Test 3: Collector store operations
echo -e "\n[3/6] Testing collector store..."
python -c "
import tempfile
from pathlib import Path
from ww.learning.collector import EventStore

# Create temp store
with tempfile.TemporaryDirectory() as tmpdir:
    db_path = str(Path(tmpdir) / 'test.db')
    store = EventStore(db_path)

    # Store retrieval
    from ww.learning.events import RetrievalEvent, OutcomeEvent, MemoryType, OutcomeType
    from uuid import uuid4

    ret = RetrievalEvent(
        query='store test',
        memory_type=MemoryType.EPISODIC,
        retrieved_ids=[uuid4()],
        session_id='e2e',
    )
    ret.compute_context_hash('context')
    store.store_retrieval(ret)

    # Store outcome
    out = OutcomeEvent(
        outcome_type=OutcomeType.SUCCESS,
        success_score=0.9,
    )
    out.compute_context_hash('context')
    store.store_outcome(out)

    # Retrieve
    retrievals = store.get_retrievals_by_context(ret.context_hash)
    assert len(retrievals) == 1, 'Retrieval not stored'

    # Baseline
    store.update_baseline('test', 0.8)
    baseline = store.get_baseline('test')
    assert abs(baseline - 0.8) < 0.01, 'Baseline failed'

    print('✓ Collector store OK')
"

# Test 4: Full collector flow
echo -e "\n[4/6] Testing full collector flow..."
python -c "
import tempfile
from pathlib import Path
from ww.learning.collector import EventCollector, CollectorConfig
from ww.learning.events import MemoryType, OutcomeType
from uuid import uuid4

with tempfile.TemporaryDirectory() as tmpdir:
    config = CollectorConfig(
        db_path=Path(tmpdir) / 'test.db',
        auto_match=True,
    )
    collector = EventCollector(config)

    # Record retrieval
    mem_ids = [uuid4(), uuid4()]
    ret = collector.record_retrieval(
        query='collector test',
        memory_type=MemoryType.EPISODIC,
        retrieved_ids=mem_ids,
        retrieval_scores={str(m)[:8]: 0.9 for m in mem_ids},
        context='test context',
        session_id='e2e',
    )

    # Record outcome
    out = collector.record_outcome(
        outcome_type=OutcomeType.SUCCESS,
        success_score=0.85,
        context='test context',
        session_id='e2e',
    )

    # Should have matched
    assert out.context_hash == ret.context_hash, 'Context hash mismatch'

    print('✓ Full collector flow OK')
"

# Test 5: ccapi integration
echo -e "\n[5/6] Testing ccapi integration..."
python -c "
from ww.integration import (
    WWMemory, Message, create_ww_memory,
    WWObserver, Event, create_ww_observer,
    create_ww_router,
)

# Memory adapter
memory = create_ww_memory(session_id='e2e-test')
memory.add(Message(role='user', content='E2E test message'))
memory.add(Message(role='assistant', content='E2E response'))
assert len(memory) == 2, 'Memory add failed'

results = memory.search('E2E test')
assert len(results) == 1, 'Search failed'

# Observer
observer = create_ww_observer(session_id='e2e-test')
observer.on_event(Event(type='agent.start', name='e2e-agent'))
observer.on_event(Event(type='tool.end', name='shell', data={'result': 'ok'}, duration_ms=50))
observer.on_event(Event(type='agent.end', name='e2e-agent', data={'status': 'ok'}))
observer.flush()

stats = observer.get_session_stats()
assert stats['initialized'], 'Observer not initialized'
observer.close()

# Router
router = create_ww_router()
assert len(router.routes) == 6, f'Expected 6 routes, got {len(router.routes)}'

print('✓ ccapi integration OK')
"

# Test 6: Trainer training loop
echo -e "\n[6/6] Testing trainer training loop..."
python -c "
import tempfile
from pathlib import Path
from ww.learning.scorer import create_trainer, TrainerConfig
from ww.learning.events import Experience, MemoryType

config = TrainerConfig(batch_size=4, device='cpu')
trainer = create_trainer(config=config)

# Add experiences
for i in range(20):
    exp = Experience(
        query=f'training query {i}',
        memory_type=MemoryType.EPISODIC,
        component_vectors=[[0.5 + i*0.01]*4 for _ in range(3)],
        outcome_score=0.7 + (i % 3) * 0.1,
        per_memory_rewards={f'm{j}': 0.6 + j*0.1 for j in range(3)},
    )
    trainer.add_experience(exp)

# Train epoch
initial_loss = trainer.train_step()
assert initial_loss is not None, 'No training occurred'
assert initial_loss >= 0, 'Negative loss'

# Multiple steps
for _ in range(5):
    trainer.train_step()

assert trainer.step == 6, f'Wrong step count: {trainer.step}'

# Save checkpoint
with tempfile.TemporaryDirectory() as tmpdir:
    ckpt = Path(tmpdir) / 'test.pt'
    trainer.save_checkpoint(ckpt)
    assert ckpt.exists(), 'Checkpoint not saved'

print('✓ Trainer training loop OK')
"

echo -e "\n=== All E2E tests passed! ==="
