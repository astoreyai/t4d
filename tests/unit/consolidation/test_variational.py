"""
Unit Tests for Variational Consolidation Framing (W4-01).

Verifies consolidation as variational EM following Friston's
free energy principle.

Evidence Base: Friston (2010) "The free-energy principle: a unified brain theory?"
"""

import pytest
import numpy as np


class TestVariationalStep:
    """Test VariationalStep enum."""

    def test_step_values(self):
        """Should have E_STEP, M_STEP, REGULARIZATION."""
        from t4dm.consolidation.variational import VariationalStep

        assert VariationalStep.E_STEP.value == "e_step"
        assert VariationalStep.M_STEP.value == "m_step"
        assert VariationalStep.REGULARIZATION.value == "regularization"


class TestVariationalState:
    """Test VariationalState dataclass."""

    def test_state_fields(self):
        """State should have free_energy, elbo, kl_divergence, log_likelihood."""
        from t4dm.consolidation.variational import VariationalState

        state = VariationalState(
            free_energy=0.5,
            elbo=-0.5,
            kl_divergence=0.3,
            log_likelihood=-0.2,
        )

        assert state.free_energy == 0.5
        assert state.elbo == -0.5
        assert state.kl_divergence == 0.3
        assert state.log_likelihood == -0.2

    def test_converged_property(self):
        """Converged when free_energy < 0.01."""
        from t4dm.consolidation.variational import VariationalState

        converged = VariationalState(0.001, -0.001, 0.0, 0.0)
        not_converged = VariationalState(0.5, -0.5, 0.3, -0.2)

        assert converged.converged
        assert not not_converged.converged


class TestClusterAssignment:
    """Test ClusterAssignment dataclass."""

    def test_from_similarities(self):
        """Should create soft assignment from similarities."""
        from t4dm.consolidation.variational import ClusterAssignment

        similarities = np.array([1.0, 0.5, 0.1])
        assignment = ClusterAssignment.from_similarities(
            memory_id="test",
            similarities=similarities,
            temperature=1.0,
        )

        assert assignment.memory_id == "test"
        assert len(assignment.cluster_posteriors) == 3
        assert np.isclose(assignment.cluster_posteriors.sum(), 1.0)
        assert assignment.most_likely_cluster == 0  # Highest similarity

    def test_low_temperature_sharper(self):
        """Lower temperature should give sharper assignments."""
        from t4dm.consolidation.variational import ClusterAssignment

        similarities = np.array([1.0, 0.9, 0.8])

        high_temp = ClusterAssignment.from_similarities("x", similarities, temperature=2.0)
        low_temp = ClusterAssignment.from_similarities("x", similarities, temperature=0.1)

        # Lower temperature = lower entropy (sharper)
        assert low_temp.assignment_entropy < high_temp.assignment_entropy


class TestClusterPrototype:
    """Test ClusterPrototype dataclass."""

    def test_prototype_fields(self):
        """Prototype should have mean, variance, member_count."""
        from t4dm.consolidation.variational import ClusterPrototype

        prototype = ClusterPrototype(
            cluster_id=0,
            mean=np.array([1.0, 2.0]),
            variance=np.array([0.1, 0.2]),
            member_count=10,
            total_responsibility=9.5,
        )

        assert prototype.cluster_id == 0
        assert np.array_equal(prototype.mean, np.array([1.0, 2.0]))
        assert prototype.member_count == 10


class TestVariationalConsolidation:
    """Test VariationalConsolidation class."""

    def test_creation(self):
        """Should create with temperature and threshold."""
        from t4dm.consolidation.variational import VariationalConsolidation

        vc = VariationalConsolidation(
            temperature=0.5,
            posterior_threshold=0.2,
        )

        assert vc.temperature == 0.5
        assert vc.posterior_threshold == 0.2

    def test_e_step_assigns_all_memories(self):
        """E-step should assign every memory to clusters."""
        from t4dm.consolidation.variational import (
            VariationalConsolidation,
            ClusterPrototype,
        )

        np.random.seed(42)
        vc = VariationalConsolidation()

        # Create memories and clusters
        memories = [np.random.randn(16) for _ in range(20)]
        clusters = [
            ClusterPrototype(i, np.random.randn(16), np.ones(16), 0, 0.0)
            for i in range(3)
        ]

        assignments = vc.e_step(memories, clusters)

        assert len(assignments) == 20
        for a in assignments:
            assert len(a.cluster_posteriors) == 3
            assert np.isclose(a.cluster_posteriors.sum(), 1.0)

    def test_m_step_updates_prototypes(self):
        """M-step should update cluster means and variances."""
        from t4dm.consolidation.variational import (
            VariationalConsolidation,
            ClusterAssignment,
        )

        np.random.seed(42)
        vc = VariationalConsolidation()

        # Create memories in two distinct groups
        group1 = [np.array([1.0] * 8) + np.random.randn(8) * 0.1 for _ in range(10)]
        group2 = [np.array([-1.0] * 8) + np.random.randn(8) * 0.1 for _ in range(10)]
        memories = group1 + group2

        # Create assignments (group1 -> cluster0, group2 -> cluster1)
        assignments = []
        for i, m in enumerate(memories):
            if i < 10:
                posteriors = np.array([0.9, 0.1])
            else:
                posteriors = np.array([0.1, 0.9])
            assignments.append(ClusterAssignment(
                memory_id=str(i),
                cluster_posteriors=posteriors,
                most_likely_cluster=0 if i < 10 else 1,
                assignment_entropy=0.3,
            ))

        clusters = vc.m_step(memories, assignments, n_clusters=2)

        assert len(clusters) == 2
        # Cluster 0 should have mean near [1, 1, ...]
        assert clusters[0].mean.mean() > 0.5
        # Cluster 1 should have mean near [-1, -1, ...]
        assert clusters[1].mean.mean() < -0.5

    def test_regularization_prunes_uncertain(self):
        """Regularization should identify low-posterior items."""
        from t4dm.consolidation.variational import (
            VariationalConsolidation,
            ClusterAssignment,
        )

        vc = VariationalConsolidation(posterior_threshold=0.5)

        assignments = [
            ClusterAssignment("a", np.array([0.9, 0.1]), 0, 0.3),  # Keep
            ClusterAssignment("b", np.array([0.3, 0.3, 0.4]), 2, 1.0),  # Prune (max=0.4)
            ClusterAssignment("c", np.array([0.6, 0.4]), 0, 0.6),  # Keep
        ]

        to_prune = vc.regularization_step(assignments)

        assert "b" in to_prune
        assert "a" not in to_prune
        assert "c" not in to_prune

    def test_compute_state_returns_metrics(self):
        """compute_state should return free energy decomposition."""
        from t4dm.consolidation.variational import (
            VariationalConsolidation,
            ClusterPrototype,
            ClusterAssignment,
        )

        np.random.seed(42)
        vc = VariationalConsolidation()

        memories = [np.random.randn(8) for _ in range(5)]
        clusters = [ClusterPrototype(0, np.zeros(8), np.ones(8), 5, 5.0)]
        assignments = [
            ClusterAssignment(str(i), np.array([1.0]), 0, 0.0)
            for i in range(5)
        ]

        state = vc.compute_state(memories, clusters, assignments)

        assert hasattr(state, "free_energy")
        assert hasattr(state, "elbo")
        assert hasattr(state, "kl_divergence")
        assert hasattr(state, "log_likelihood")


class TestVariationalEMConvergence:
    """Test EM algorithm convergence."""

    def test_em_decreases_free_energy(self):
        """EM iterations should decrease free energy."""
        from t4dm.consolidation.variational import (
            VariationalConsolidation,
            ClusterPrototype,
        )

        np.random.seed(42)
        vc = VariationalConsolidation(temperature=0.5)

        # Create clustered data
        group1 = [np.array([2.0] * 8) + np.random.randn(8) * 0.3 for _ in range(15)]
        group2 = [np.array([-2.0] * 8) + np.random.randn(8) * 0.3 for _ in range(15)]
        memories = group1 + group2

        # Initialize clusters randomly
        clusters = [
            ClusterPrototype(i, np.random.randn(8), np.ones(8), 0, 0.0)
            for i in range(2)
        ]

        free_energies = []

        # Run EM iterations
        for _ in range(5):
            assignments = vc.e_step(memories, clusters)
            clusters = vc.m_step(memories, assignments, n_clusters=2)
            state = vc.compute_state(memories, clusters, assignments)
            free_energies.append(state.free_energy)

        # Free energy should generally decrease (allowing some noise)
        # Check that final is less than initial
        assert free_energies[-1] < free_energies[0], \
            f"Free energy should decrease: {free_energies}"
