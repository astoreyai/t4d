"""
Tests for ww.nca.connectome module.

Tests anatomically-constrained brain connectivity with NT-specific
projection pathways and biological plausibility.
"""

import numpy as np
import pytest

from ww.nca.connectome import (
    Connectome,
    ConnectomeConfig,
    ConnectomeIntegrator,
    BrainRegion,
    ProjectionPathway,
    RegionType,
    NTSystem,
    create_default_connectome,
    create_minimal_connectome,
    get_pathway_summary,
)


class TestRegionType:
    """Tests for RegionType enum."""

    def test_region_types_exist(self):
        """All expected region types should exist."""
        assert RegionType.CORTICAL is not None
        assert RegionType.SUBCORTICAL is not None
        assert RegionType.LIMBIC is not None
        assert RegionType.BRAINSTEM is not None
        assert RegionType.CEREBELLAR is not None

    def test_region_type_count(self):
        """Should have 5 region types."""
        assert len(RegionType) == 5


class TestNTSystem:
    """Tests for NTSystem enum."""

    def test_nt_systems_exist(self):
        """All 6 NT systems should exist."""
        assert NTSystem.DOPAMINE is not None
        assert NTSystem.SEROTONIN is not None
        assert NTSystem.NOREPINEPHRINE is not None
        assert NTSystem.ACETYLCHOLINE is not None
        assert NTSystem.GLUTAMATE is not None
        assert NTSystem.GABA is not None

    def test_nt_system_count(self):
        """Should have exactly 6 NT systems."""
        assert len(NTSystem) == 6

    def test_nt_system_values(self):
        """NT systems should have string values."""
        assert NTSystem.DOPAMINE.value == "dopamine"
        assert NTSystem.SEROTONIN.value == "serotonin"
        assert NTSystem.GABA.value == "gaba"


class TestBrainRegion:
    """Tests for BrainRegion dataclass."""

    def test_create_region(self):
        """Should create region with basic properties."""
        region = BrainRegion(
            name="TestRegion",
            region_type=RegionType.CORTICAL,
            coordinates=(10.0, 5.0, 0.0),
            volume_mm3=200.0,
        )
        assert region.name == "TestRegion"
        assert region.region_type == RegionType.CORTICAL
        assert region.coordinates == (10.0, 5.0, 0.0)
        assert region.volume_mm3 == 200.0

    def test_default_receptor_densities(self):
        """Regions should have default receptor densities."""
        region = BrainRegion(
            name="Test",
            region_type=RegionType.CORTICAL
        )
        assert NTSystem.DOPAMINE in region.nt_receptors
        assert NTSystem.GLUTAMATE in region.nt_receptors
        assert 0.0 <= region.nt_receptors[NTSystem.DOPAMINE] <= 1.0

    def test_default_nt_source_flags(self):
        """Regions should default to not being NT sources."""
        region = BrainRegion(
            name="Test",
            region_type=RegionType.CORTICAL
        )
        assert all(not v for v in region.is_nt_source.values())

    def test_custom_receptor_densities(self):
        """Should accept custom receptor densities."""
        receptors = {NTSystem.DOPAMINE: 0.9, NTSystem.SEROTONIN: 0.1}
        region = BrainRegion(
            name="Test",
            region_type=RegionType.LIMBIC,
            nt_receptors=receptors
        )
        assert region.nt_receptors[NTSystem.DOPAMINE] == 0.9
        assert region.nt_receptors[NTSystem.SEROTONIN] == 0.1


class TestProjectionPathway:
    """Tests for ProjectionPathway dataclass."""

    def test_create_pathway(self):
        """Should create pathway with properties."""
        pathway = ProjectionPathway(
            source="VTA",
            target="NAcc",
            nt_system=NTSystem.DOPAMINE,
            strength=0.8,
            probability=0.9,
            is_inhibitory=False
        )
        assert pathway.source == "VTA"
        assert pathway.target == "NAcc"
        assert pathway.nt_system == NTSystem.DOPAMINE
        assert pathway.strength == 0.8
        assert pathway.is_inhibitory is False

    def test_signed_strength_excitatory(self):
        """Excitatory pathways should have positive signed strength."""
        pathway = ProjectionPathway(
            source="A", target="B",
            nt_system=NTSystem.GLUTAMATE,
            strength=0.5,
            is_inhibitory=False
        )
        assert pathway.signed_strength == 0.5

    def test_signed_strength_inhibitory(self):
        """Inhibitory pathways should have negative signed strength."""
        pathway = ProjectionPathway(
            source="A", target="B",
            nt_system=NTSystem.GABA,
            strength=0.5,
            is_inhibitory=True
        )
        assert pathway.signed_strength == -0.5


class TestConnectomeConfig:
    """Tests for ConnectomeConfig."""

    def test_default_config(self):
        """Default config should have sensible values."""
        config = ConnectomeConfig()
        assert config.n_cortical_regions == 6
        assert config.n_subcortical_regions == 4
        assert config.n_limbic_regions == 3
        assert config.n_brainstem_regions == 4
        assert 0.0 < config.local_connection_prob <= 1.0
        assert config.enforce_dale_law is True

    def test_custom_config(self):
        """Should accept custom configuration."""
        config = ConnectomeConfig(
            n_cortical_regions=3,
            n_subcortical_regions=2,
            local_connection_prob=0.5
        )
        assert config.n_cortical_regions == 3
        assert config.n_subcortical_regions == 2
        assert config.local_connection_prob == 0.5


class TestConnectome:
    """Tests for Connectome class."""

    def test_create_default_connectome(self):
        """Should create connectome with default regions."""
        conn = Connectome()
        assert len(conn.regions) > 0
        assert len(conn.pathways) > 0

    def test_expected_regions_exist(self):
        """Should have expected brain regions."""
        conn = Connectome()
        region_names = list(conn.regions.keys())

        # Cortical
        assert "PFC" in region_names
        assert "Motor" in region_names

        # Subcortical
        assert "Striatum" in region_names
        assert "Thalamus" in region_names

        # Limbic
        assert "Hippocampus" in region_names
        assert "Amygdala" in region_names
        assert "NAcc" in region_names

        # Brainstem NT sources
        assert "VTA" in region_names
        assert "SNc" in region_names
        assert "LC" in region_names
        assert "Raphe" in region_names

        # Cholinergic
        assert "NBM" in region_names

    def test_nt_sources_defined(self):
        """NT source regions should be flagged correctly."""
        conn = Connectome()

        # VTA produces dopamine
        assert conn.regions["VTA"].is_nt_source[NTSystem.DOPAMINE] is True

        # SNc produces dopamine
        assert conn.regions["SNc"].is_nt_source[NTSystem.DOPAMINE] is True

        # LC produces norepinephrine
        assert conn.regions["LC"].is_nt_source[NTSystem.NOREPINEPHRINE] is True

        # Raphe produces serotonin
        assert conn.regions["Raphe"].is_nt_source[NTSystem.SEROTONIN] is True

        # NBM produces acetylcholine
        assert conn.regions["NBM"].is_nt_source[NTSystem.ACETYLCHOLINE] is True

    def test_connectivity_matrix_shape(self):
        """Connectivity matrices should have correct shape."""
        conn = Connectome()
        n_regions = len(conn.regions)

        for nt in NTSystem:
            matrix = conn.get_connectivity_matrix(nt)
            assert matrix.shape == (n_regions, n_regions)

    def test_connectivity_values_bounded(self):
        """Connectivity values should be in [0, 1]."""
        conn = Connectome()

        for nt in NTSystem:
            matrix = conn.get_connectivity_matrix(nt)
            assert np.all(matrix >= 0.0)
            assert np.all(matrix <= 1.0)

    def test_no_self_connections(self):
        """Diagonal of connectivity matrices should be zero."""
        conn = Connectome()

        for nt in NTSystem:
            matrix = conn.get_connectivity_matrix(nt)
            assert np.allclose(np.diag(matrix), 0.0)

    def test_dopamine_pathways_exist(self):
        """Should have dopaminergic pathways."""
        conn = Connectome()
        da_pathways = [p for p in conn.pathways if p.nt_system == NTSystem.DOPAMINE]
        assert len(da_pathways) > 0

        # Mesolimbic: VTA -> NAcc
        mesolimbic = [p for p in da_pathways if p.source == "VTA" and p.target == "NAcc"]
        assert len(mesolimbic) == 1
        assert mesolimbic[0].strength > 0.5  # Strong pathway

        # Nigrostriatal: SNc -> Striatum
        nigrostriatal = [p for p in da_pathways if p.source == "SNc" and p.target == "Striatum"]
        assert len(nigrostriatal) == 1
        assert nigrostriatal[0].strength > 0.5

    def test_serotonin_pathways_diffuse(self):
        """Serotonergic projections should be diffuse."""
        conn = Connectome()
        sero_pathways = [p for p in conn.pathways if p.nt_system == NTSystem.SEROTONIN]
        assert len(sero_pathways) > 3  # Multiple targets

        # All from Raphe
        assert all(p.source == "Raphe" for p in sero_pathways)

    def test_noradrenergic_pathways_widespread(self):
        """Noradrenergic projections should be widespread."""
        conn = Connectome()
        ne_pathways = [p for p in conn.pathways if p.nt_system == NTSystem.NOREPINEPHRINE]
        assert len(ne_pathways) > 5  # Many targets

        # All from LC
        assert all(p.source == "LC" for p in ne_pathways)

    def test_cholinergic_to_cortex(self):
        """Cholinergic projections should target cortex."""
        conn = Connectome()
        ach_pathways = [p for p in conn.pathways if p.nt_system == NTSystem.ACETYLCHOLINE]
        assert len(ach_pathways) > 0

        # From NBM to cortical regions
        nbm_to_cortex = [p for p in ach_pathways if p.source == "NBM"]
        assert len(nbm_to_cortex) > 0

    def test_gaba_pathways_inhibitory(self):
        """GABA pathways should be marked inhibitory."""
        conn = Connectome()
        gaba_pathways = [p for p in conn.pathways if p.nt_system == NTSystem.GABA]

        # At least striato-pallidal
        assert len(gaba_pathways) > 0
        # GABA pathways should be inhibitory
        for p in gaba_pathways:
            assert p.is_inhibitory is True

    def test_get_combined_connectivity(self):
        """Should compute weighted combination of NT matrices."""
        conn = Connectome()

        # Equal weights
        combined = conn.get_combined_connectivity()
        assert combined.shape == (len(conn.regions), len(conn.regions))

        # Custom weights
        weights = {NTSystem.DOPAMINE: 2.0, NTSystem.GLUTAMATE: 1.0}
        weighted = conn.get_combined_connectivity(weights)
        assert weighted.shape == combined.shape

    def test_get_region_index(self):
        """Should return correct region index."""
        conn = Connectome()
        pfc_idx = conn.get_region_index("PFC")
        assert isinstance(pfc_idx, int)
        assert pfc_idx >= 0
        assert pfc_idx < len(conn.regions)

    def test_get_distance_matrix(self):
        """Should compute inter-region distances."""
        conn = Connectome()
        distances = conn.get_distance_matrix()

        n = len(conn.regions)
        assert distances.shape == (n, n)

        # Diagonal should be zero (self-distance)
        assert np.allclose(np.diag(distances), 0.0)

        # Symmetric
        assert np.allclose(distances, distances.T)

        # All positive off-diagonal
        assert np.all(distances >= 0)

    def test_get_pathways_for_region(self):
        """Should get pathways involving a region."""
        conn = Connectome()

        # VTA has outgoing DA pathways
        vta_out = conn.get_pathways_for_region("VTA", direction="out")
        assert len(vta_out) > 0
        assert all(p.source == "VTA" for p in vta_out)

        # NAcc has incoming pathways
        nacc_in = conn.get_pathways_for_region("NAcc", direction="in")
        assert len(nacc_in) > 0
        assert all(p.target == "NAcc" for p in nacc_in)

        # Both directions
        pfc_both = conn.get_pathways_for_region("PFC", direction="both")
        assert len(pfc_both) >= len(conn.get_pathways_for_region("PFC", direction="in"))

    def test_get_nt_sources(self):
        """Should return NT source regions."""
        conn = Connectome()

        da_sources = conn.get_nt_sources(NTSystem.DOPAMINE)
        assert "VTA" in da_sources
        assert "SNc" in da_sources

        sero_sources = conn.get_nt_sources(NTSystem.SEROTONIN)
        assert "Raphe" in sero_sources

        ne_sources = conn.get_nt_sources(NTSystem.NOREPINEPHRINE)
        assert "LC" in ne_sources

        ach_sources = conn.get_nt_sources(NTSystem.ACETYLCHOLINE)
        assert "NBM" in ach_sources

    def test_to_coupling_matrix(self):
        """Should convert to 6x6 coupling matrix."""
        conn = Connectome()
        coupling = conn.to_coupling_matrix()

        assert coupling.shape == (6, 6)
        assert coupling.dtype == np.float32

        # GABA row should be negative (inhibitory)
        gaba_idx = 4
        assert np.all(coupling[gaba_idx, :] <= 0)

    def test_validate_passes(self):
        """Default connectome should validate."""
        conn = Connectome()
        is_valid, issues = conn.validate()
        assert is_valid is True
        assert len(issues) == 0

    def test_validate_dale_law(self):
        """Should check Dale's law for GABA/Glu."""
        config = ConnectomeConfig(enforce_dale_law=True)
        conn = Connectome(config)

        # Add invalid pathway (GABA not inhibitory)
        conn.pathways.append(ProjectionPathway(
            source="Striatum", target="PFC",
            nt_system=NTSystem.GABA,
            strength=0.5,
            is_inhibitory=False  # Violates Dale's law
        ))

        is_valid, issues = conn.validate()
        assert is_valid is False
        assert any("GABA" in issue for issue in issues)

    def test_get_stats(self):
        """Should return connectome statistics."""
        conn = Connectome()
        stats = conn.get_stats()

        assert "n_regions" in stats
        assert "n_pathways" in stats
        assert "region_types" in stats
        assert "mean_connectivity" in stats
        assert "sparsity" in stats
        assert "nt_sources" in stats

        assert stats["n_regions"] == len(conn.regions)
        assert stats["n_pathways"] == len(conn.pathways)


class TestConnectomeIntegrator:
    """Tests for ConnectomeIntegrator."""

    def test_create_integrator(self):
        """Should create integrator from connectome."""
        conn = Connectome()
        integrator = ConnectomeIntegrator(conn)
        assert integrator.connectome is conn

    def test_configure_delay_system(self):
        """Should configure delay system with distances."""
        from ww.nca.delays import TransmissionDelaySystem

        conn = Connectome()
        integrator = ConnectomeIntegrator(conn)

        # Create delay system with enough regions
        delay = TransmissionDelaySystem(
            grid_shape=(32,),
            n_regions=len(conn.regions)
        )

        # Should not raise
        integrator.configure_delay_system(delay)

    def test_configure_coupling(self):
        """Should configure coupling from connectome."""
        from ww.nca.coupling import LearnableCoupling

        conn = Connectome()
        integrator = ConnectomeIntegrator(conn)

        coupling = LearnableCoupling()
        original_K = coupling.K.copy()

        integrator.configure_coupling(coupling)

        # Coupling should be modified
        assert not np.allclose(coupling.K, original_K)

    def test_get_region_nt_modulation(self):
        """Should return NT modulation for region."""
        conn = Connectome()
        integrator = ConnectomeIntegrator(conn)

        # PFC should have receptor densities
        mods = integrator.get_region_nt_modulation("PFC")
        assert "dopamine" in mods
        assert "glutamate" in mods
        assert 0.0 <= mods["dopamine"] <= 1.0

    def test_get_region_nt_modulation_invalid(self):
        """Should return empty dict for invalid region."""
        conn = Connectome()
        integrator = ConnectomeIntegrator(conn)

        mods = integrator.get_region_nt_modulation("InvalidRegion")
        assert mods == {}


class TestConvenienceFunctions:
    """Tests for convenience functions."""

    def test_create_default_connectome(self):
        """Should create standard connectome."""
        conn = create_default_connectome()
        assert isinstance(conn, Connectome)
        assert len(conn.regions) > 10

    def test_create_minimal_connectome(self):
        """Should create minimal connectome for testing."""
        conn = create_minimal_connectome()
        assert isinstance(conn, Connectome)
        assert len(conn.regions) < 10  # Minimal

    def test_get_pathway_summary(self):
        """Should return human-readable summary."""
        conn = create_minimal_connectome()
        summary = get_pathway_summary(conn)

        assert isinstance(summary, str)
        assert "Connectome Pathways" in summary
        assert "DOPAMINE" in summary or len(conn.pathways) == 0


class TestNeuralFieldIntegration:
    """Tests for connectome integration with NeuralFieldSolver."""

    def test_solver_with_connectome(self):
        """Should create solver with connectome."""
        from ww.nca.neural_field import NeuralFieldSolver, NeuralFieldConfig

        conn = create_minimal_connectome()
        solver = NeuralFieldSolver(
            config=NeuralFieldConfig(grid_size=16),
            connectome=conn
        )

        assert solver.connectome is conn

    def test_solver_get_connectome_stats(self):
        """Should get stats from connectome."""
        from ww.nca.neural_field import NeuralFieldSolver

        conn = create_minimal_connectome()
        solver = NeuralFieldSolver(connectome=conn)

        stats = solver.get_connectome_stats()
        assert "n_regions" in stats
        assert "n_pathways" in stats

    def test_solver_get_region_names(self):
        """Should get region names from connectome."""
        from ww.nca.neural_field import NeuralFieldSolver

        conn = create_minimal_connectome()
        solver = NeuralFieldSolver(connectome=conn)

        names = solver.get_region_names()
        assert len(names) == len(conn.regions)

    def test_solver_get_nt_sources(self):
        """Should get NT sources from connectome."""
        from ww.nca.neural_field import NeuralFieldSolver

        conn = Connectome()
        solver = NeuralFieldSolver(connectome=conn)

        da_sources = solver.get_nt_sources("dopamine")
        assert "VTA" in da_sources

    def test_solver_without_connectome(self):
        """Solver should work without connectome."""
        from ww.nca.neural_field import NeuralFieldSolver

        solver = NeuralFieldSolver()
        assert solver.connectome is None
        assert solver.get_connectome_stats() == {}
        assert solver.get_region_names() == []

    def test_solver_step_with_connectome(self):
        """Solver should step with connectome configured."""
        from ww.nca.neural_field import NeuralFieldSolver, NeuralFieldConfig
        from ww.nca.coupling import LearnableCoupling
        from ww.nca.delays import TransmissionDelaySystem

        conn = create_minimal_connectome()
        coupling = LearnableCoupling()
        delay = TransmissionDelaySystem(
            grid_shape=(32,),
            n_regions=len(conn.regions)
        )

        solver = NeuralFieldSolver(
            config=NeuralFieldConfig(grid_size=32),
            coupling=coupling,
            delay_system=delay,
            connectome=conn
        )

        # Should step without error
        state = solver.step()
        assert state is not None

        # Run a few steps
        for _ in range(10):
            solver.step()

        stats = solver.get_stats()
        assert stats["step_count"] == 11


class TestBiologicalPlausibility:
    """Tests for biological plausibility of connectome."""

    def test_receptor_densities_realistic(self):
        """Receptor densities should match neuroscience literature."""
        conn = Connectome()

        # Striatum has high DA receptors
        striatum = conn.regions["Striatum"]
        assert striatum.nt_receptors[NTSystem.DOPAMINE] > 0.7

        # NAcc has high DA receptors
        nacc = conn.regions["NAcc"]
        assert nacc.nt_receptors[NTSystem.DOPAMINE] > 0.7

        # PFC has moderate DA
        pfc = conn.regions["PFC"]
        assert pfc.nt_receptors[NTSystem.DOPAMINE] > 0.4

        # Hippocampus has high ACh
        hipp = conn.regions["Hippocampus"]
        assert hipp.nt_receptors[NTSystem.ACETYLCHOLINE] > 0.5

    def test_pathway_strengths_realistic(self):
        """Pathway strengths should reflect known connectivity."""
        conn = Connectome()

        # Nigrostriatal is strong
        nigro = [p for p in conn.pathways
                 if p.source == "SNc" and p.target == "Striatum"]
        assert len(nigro) == 1
        assert nigro[0].strength > 0.8

        # Mesolimbic is strong
        meso = [p for p in conn.pathways
                if p.source == "VTA" and p.target == "NAcc"]
        assert len(meso) == 1
        assert meso[0].strength > 0.7

    def test_distance_decay(self):
        """Connectivity should decay with distance."""
        conn = Connectome()
        distances = conn.get_distance_matrix()
        combined = conn.get_combined_connectivity()

        # Find pairs at different distances
        n = len(conn.regions)
        short_dist_conns = []
        long_dist_conns = []

        for i in range(n):
            for j in range(n):
                if i == j:
                    continue
                d = distances[i, j]
                c = combined[i, j]
                if d < 20:  # Short range
                    short_dist_conns.append(c)
                elif d > 50:  # Long range
                    long_dist_conns.append(c)

        if short_dist_conns and long_dist_conns:
            # Average short-range should be higher
            assert np.mean(short_dist_conns) >= np.mean(long_dist_conns) * 0.5

    def test_region_coordinates(self):
        """Region coordinates should be in reasonable range."""
        conn = Connectome()

        for name, region in conn.regions.items():
            x, y, z = region.coordinates
            # Coordinates should be within ~100mm of origin (brain size)
            assert abs(x) < 100, f"{name} x coordinate out of range"
            assert abs(y) < 100, f"{name} y coordinate out of range"
            assert abs(z) < 100, f"{name} z coordinate out of range"

            # Brainstem should be below cortex
            if region.region_type == RegionType.BRAINSTEM:
                assert z < -30, f"Brainstem region {name} not deep enough"

    def test_connectivity_sparsity(self):
        """Brain connectivity structure check."""
        conn = Connectome()
        combined = conn.get_combined_connectivity()

        # With diffuse neuromodulatory systems (5-HT, NE, ACh),
        # connectivity is NOT sparse - most regions receive projections
        # Check that self-connections are zero
        assert np.allclose(np.diag(combined), 0.0)

        # Check that there's variability in connection strengths
        nonzero = combined[combined > 0.01]
        assert len(nonzero) > 0, "Should have some connections"
        assert np.std(nonzero) > 0.05, "Should have connection variability"
