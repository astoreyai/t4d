"""
Connectome Structure for World Weaver NCA.

Implements anatomically-constrained connectivity based on real brain
architecture, integrating with coupling and delay systems.

Biological Basis:
- Brain regions have specific NT projection patterns
- Connectivity is NOT uniform - follows anatomical pathways
- Long-range projections have specific source nuclei:
  - DA: VTA (mesolimbic/mesocortical), SNc (nigrostriatal)
  - 5-HT: Dorsal/Median Raphe nuclei
  - NE: Locus Coeruleus (LC)
  - ACh: Nucleus Basalis of Meynert (NBM), pedunculopontine nucleus

Connectivity Types:
1. Local circuits: Within-region E/I balance (Glu/GABA)
2. Cortico-cortical: Long-range cortical connections
3. Subcortical loops: Basal ganglia, thalamic circuits
4. Neuromodulatory projections: Diffuse ascending systems

Implementation:
- BrainRegion: Anatomical region with properties
- Connectome: Full connectivity specification
- ProjectionPathway: NT-specific pathways
- ConnectomeConstraints: Biological validity checks

References:
- Allen Brain Atlas connectivity data
- Mesulam (2000) - Cholinergic systems
- Haber & Knutson (2010) - DA circuits
- Jacobs & Azmitia (1992) - 5-HT projections
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from enum import Enum, auto

import numpy as np

logger = logging.getLogger(__name__)


class RegionType(Enum):
    """Types of brain regions."""
    CORTICAL = auto()           # Neocortex
    SUBCORTICAL = auto()        # Basal ganglia, thalamus, etc.
    LIMBIC = auto()             # Hippocampus, amygdala, etc.
    BRAINSTEM = auto()          # Midbrain, pons, medulla
    CEREBELLAR = auto()         # Cerebellum


class NTSystem(Enum):
    """Neurotransmitter systems with their source nuclei."""
    DOPAMINE = "dopamine"
    SEROTONIN = "serotonin"
    NOREPINEPHRINE = "norepinephrine"
    ACETYLCHOLINE = "acetylcholine"
    GLUTAMATE = "glutamate"
    GABA = "gaba"


@dataclass
class BrainRegion:
    """
    Anatomical brain region with properties.

    Attributes:
        name: Region identifier
        region_type: Cortical, subcortical, etc.
        coordinates: 3D position (mm from bregma, MNI-like)
        volume_mm3: Region volume
        nt_receptors: Receptor densities for each NT
        is_nt_source: Whether region is source for NT projections
    """
    name: str
    region_type: RegionType
    coordinates: tuple[float, float, float] = (0.0, 0.0, 0.0)
    volume_mm3: float = 100.0
    nt_receptors: dict[NTSystem, float] = field(default_factory=dict)
    is_nt_source: dict[NTSystem, bool] = field(default_factory=dict)

    def __post_init__(self):
        # Default receptor densities (normalized 0-1)
        if not self.nt_receptors:
            self.nt_receptors = {
                NTSystem.DOPAMINE: 0.5,
                NTSystem.SEROTONIN: 0.5,
                NTSystem.NOREPINEPHRINE: 0.5,
                NTSystem.ACETYLCHOLINE: 0.5,
                NTSystem.GLUTAMATE: 0.8,
                NTSystem.GABA: 0.8,
            }

        if not self.is_nt_source:
            self.is_nt_source = {nt: False for nt in NTSystem}


@dataclass
class ProjectionPathway:
    """
    NT-specific projection pathway between regions.

    Attributes:
        source: Source region name
        target: Target region name
        nt_system: Neurotransmitter of this projection
        strength: Connection strength (0-1)
        probability: Connection probability
        is_inhibitory: Whether projection is inhibitory
    """
    source: str
    target: str
    nt_system: NTSystem
    strength: float = 0.5
    probability: float = 1.0
    is_inhibitory: bool = False

    @property
    def signed_strength(self) -> float:
        """Get signed connection strength."""
        return -self.strength if self.is_inhibitory else self.strength


@dataclass
class ConnectomeConfig:
    """Configuration for connectome generation."""
    # Region parameters
    n_cortical_regions: int = 6      # PFC, motor, sensory, parietal, temporal, occipital
    n_subcortical_regions: int = 4   # Striatum, thalamus, GP, STN
    n_limbic_regions: int = 3        # Hippocampus, amygdala, NAcc
    n_brainstem_regions: int = 4     # VTA, SNc, LC, Raphe

    # Connectivity parameters
    local_connection_prob: float = 0.8     # Within-region
    cortico_cortical_prob: float = 0.3     # Between cortical
    subcortical_loop_prob: float = 0.6     # Basal ganglia loops
    neuromodulatory_prob: float = 0.9      # Diffuse projections

    # Biological constraints
    max_long_range_strength: float = 0.3   # Long-range weaker
    local_ei_ratio: float = 0.8            # E/I balance ~4:1
    enforce_dale_law: bool = True          # Neurons are E or I, not both

    # Distance parameters (mm)
    cortical_spacing: float = 15.0         # Between cortical regions
    subcortical_depth: float = 40.0        # Below cortex


class Connectome:
    """
    Complete brain connectome with anatomical constraints.

    Provides:
    1. Region definitions with properties
    2. Connectivity matrices per NT system
    3. Pathway specifications
    4. Integration with coupling/delays
    """

    def __init__(self, config: ConnectomeConfig | None = None):
        """
        Initialize connectome.

        Args:
            config: Connectome configuration
        """
        self.config = config or ConnectomeConfig()

        # Brain regions
        self.regions: dict[str, BrainRegion] = {}

        # Connectivity matrices per NT (region x region)
        self._connectivity: dict[NTSystem, np.ndarray] = {}

        # Named pathways
        self.pathways: list[ProjectionPathway] = []

        # Initialize default structure
        self._init_regions()
        self._init_connectivity()
        self._init_pathways()

        logger.info(
            f"Connectome initialized: {len(self.regions)} regions, "
            f"{len(self.pathways)} pathways"
        )

    def _init_regions(self) -> None:
        """Initialize brain regions with anatomical properties."""
        # Cortical regions
        cortical_names = ["PFC", "Motor", "Sensory", "Parietal", "Temporal", "Occipital"]
        for i, name in enumerate(cortical_names[:self.config.n_cortical_regions]):
            angle = 2 * np.pi * i / len(cortical_names)
            x = self.config.cortical_spacing * np.cos(angle)
            y = self.config.cortical_spacing * np.sin(angle)

            self.regions[name] = BrainRegion(
                name=name,
                region_type=RegionType.CORTICAL,
                coordinates=(x, y, 0.0),
                volume_mm3=500.0,
                nt_receptors={
                    NTSystem.DOPAMINE: 0.6 if name == "PFC" else 0.4,
                    NTSystem.SEROTONIN: 0.5,
                    NTSystem.NOREPINEPHRINE: 0.5,
                    NTSystem.ACETYLCHOLINE: 0.6,
                    NTSystem.GLUTAMATE: 0.9,
                    NTSystem.GABA: 0.9,
                }
            )

        # Subcortical regions
        subcortical = [
            ("Striatum", (0, 5, -self.config.subcortical_depth)),
            ("Thalamus", (0, -5, -self.config.subcortical_depth)),
            ("GP", (3, 0, -self.config.subcortical_depth - 5)),  # Globus Pallidus
            ("STN", (-3, 0, -self.config.subcortical_depth - 5)),  # Subthalamic
        ]
        for name, coords in subcortical[:self.config.n_subcortical_regions]:
            self.regions[name] = BrainRegion(
                name=name,
                region_type=RegionType.SUBCORTICAL,
                coordinates=coords,
                volume_mm3=200.0,
                nt_receptors={
                    NTSystem.DOPAMINE: 0.9 if name == "Striatum" else 0.4,
                    NTSystem.ACETYLCHOLINE: 0.7 if name == "Striatum" else 0.3,
                    NTSystem.GLUTAMATE: 0.7,
                    NTSystem.GABA: 0.8,
                }
            )

        # Limbic regions
        limbic = [
            ("Hippocampus", (8, -10, -self.config.subcortical_depth)),
            ("Amygdala", (12, 5, -self.config.subcortical_depth)),
            ("NAcc", (0, 10, -self.config.subcortical_depth + 5)),  # Nucleus Accumbens
        ]
        for name, coords in limbic[:self.config.n_limbic_regions]:
            is_nacc = name == "NAcc"
            self.regions[name] = BrainRegion(
                name=name,
                region_type=RegionType.LIMBIC,
                coordinates=coords,
                volume_mm3=150.0 if not is_nacc else 50.0,
                nt_receptors={
                    NTSystem.DOPAMINE: 0.9 if is_nacc else 0.5,
                    NTSystem.SEROTONIN: 0.6,
                    NTSystem.ACETYLCHOLINE: 0.7 if name == "Hippocampus" else 0.4,
                    NTSystem.GLUTAMATE: 0.8,
                    NTSystem.GABA: 0.8,
                }
            )

        # Brainstem NT source nuclei
        brainstem = [
            ("VTA", (0, 0, -60), NTSystem.DOPAMINE),      # Ventral Tegmental Area
            ("SNc", (3, 0, -58), NTSystem.DOPAMINE),      # Substantia Nigra pars compacta
            ("LC", (0, -5, -55), NTSystem.NOREPINEPHRINE),  # Locus Coeruleus
            ("Raphe", (0, -3, -50), NTSystem.SEROTONIN),  # Raphe Nuclei
        ]
        for name, coords, nt in brainstem[:self.config.n_brainstem_regions]:
            region = BrainRegion(
                name=name,
                region_type=RegionType.BRAINSTEM,
                coordinates=coords,
                volume_mm3=20.0,
            )
            region.is_nt_source[nt] = True
            self.regions[name] = region

        # Add cholinergic source
        self.regions["NBM"] = BrainRegion(
            name="NBM",  # Nucleus Basalis of Meynert
            region_type=RegionType.SUBCORTICAL,
            coordinates=(5, 8, -self.config.subcortical_depth - 10),
            volume_mm3=15.0,
        )
        self.regions["NBM"].is_nt_source[NTSystem.ACETYLCHOLINE] = True

    def _init_connectivity(self) -> None:
        """Initialize connectivity matrices for each NT system."""
        n_regions = len(self.regions)
        region_names = list(self.regions.keys())

        for nt in NTSystem:
            self._connectivity[nt] = np.zeros((n_regions, n_regions), dtype=np.float32)

        # Build connectivity based on region types and NT sources
        for i, source_name in enumerate(region_names):
            source = self.regions[source_name]

            for j, target_name in enumerate(region_names):
                if i == j:
                    continue  # Skip self-connections here

                target = self.regions[target_name]
                distance = self._compute_distance(source, target)

                # Determine connectivity for each NT
                for nt in NTSystem:
                    strength = self._compute_connection_strength(
                        source, target, nt, distance
                    )
                    self._connectivity[nt][i, j] = strength

    def _init_pathways(self) -> None:
        """Initialize named projection pathways."""
        # Dopaminergic pathways
        if "VTA" in self.regions:
            # Mesolimbic: VTA → NAcc
            if "NAcc" in self.regions:
                self.pathways.append(ProjectionPathway(
                    source="VTA", target="NAcc",
                    nt_system=NTSystem.DOPAMINE,
                    strength=0.8, probability=0.9
                ))
            # Mesocortical: VTA → PFC
            if "PFC" in self.regions:
                self.pathways.append(ProjectionPathway(
                    source="VTA", target="PFC",
                    nt_system=NTSystem.DOPAMINE,
                    strength=0.6, probability=0.8
                ))

        if "SNc" in self.regions and "Striatum" in self.regions:
            # Nigrostriatal: SNc → Striatum
            self.pathways.append(ProjectionPathway(
                source="SNc", target="Striatum",
                nt_system=NTSystem.DOPAMINE,
                strength=0.9, probability=0.95
            ))

        # Serotonergic pathways (diffuse)
        if "Raphe" in self.regions:
            for target_name, target in self.regions.items():
                if target_name != "Raphe" and target.region_type in [
                    RegionType.CORTICAL, RegionType.LIMBIC
                ]:
                    self.pathways.append(ProjectionPathway(
                        source="Raphe", target=target_name,
                        nt_system=NTSystem.SEROTONIN,
                        strength=0.4, probability=self.config.neuromodulatory_prob
                    ))

        # Noradrenergic pathways (widespread)
        if "LC" in self.regions:
            for target_name, target in self.regions.items():
                if target_name != "LC" and target.region_type != RegionType.BRAINSTEM:
                    self.pathways.append(ProjectionPathway(
                        source="LC", target=target_name,
                        nt_system=NTSystem.NOREPINEPHRINE,
                        strength=0.3, probability=self.config.neuromodulatory_prob
                    ))

        # Cholinergic pathways
        if "NBM" in self.regions:
            for target_name, target in self.regions.items():
                if target_name != "NBM" and target.region_type == RegionType.CORTICAL:
                    self.pathways.append(ProjectionPathway(
                        source="NBM", target=target_name,
                        nt_system=NTSystem.ACETYLCHOLINE,
                        strength=0.5, probability=self.config.neuromodulatory_prob
                    ))

        # Cortico-striatal (glutamatergic)
        if "Striatum" in self.regions:
            for name, region in self.regions.items():
                if region.region_type == RegionType.CORTICAL:
                    self.pathways.append(ProjectionPathway(
                        source=name, target="Striatum",
                        nt_system=NTSystem.GLUTAMATE,
                        strength=0.6, probability=self.config.subcortical_loop_prob
                    ))

        # Striato-pallidal (GABAergic)
        if "Striatum" in self.regions and "GP" in self.regions:
            self.pathways.append(ProjectionPathway(
                source="Striatum", target="GP",
                nt_system=NTSystem.GABA,
                strength=0.8, probability=0.9,
                is_inhibitory=True
            ))

        # PFC backprojections to neuromodulatory nuclei
        # These implement top-down cortical control
        if "PFC" in self.regions:
            # PFC → VTA (goal-directed motivation)
            if "VTA" in self.regions:
                self.pathways.append(ProjectionPathway(
                    source="PFC", target="VTA",
                    nt_system=NTSystem.GLUTAMATE,
                    strength=0.4, probability=0.8
                ))

            # PFC → LC (attentional control)
            if "LC" in self.regions:
                self.pathways.append(ProjectionPathway(
                    source="PFC", target="LC",
                    nt_system=NTSystem.GLUTAMATE,
                    strength=0.4, probability=0.8
                ))

            # mPFC → DRN (emotional regulation)
            if "Raphe" in self.regions:
                self.pathways.append(ProjectionPathway(
                    source="PFC", target="Raphe",
                    nt_system=NTSystem.GLUTAMATE,
                    strength=0.3, probability=0.7
                ))

    def _compute_distance(self, source: BrainRegion, target: BrainRegion) -> float:
        """Compute Euclidean distance between regions."""
        s = np.array(source.coordinates)
        t = np.array(target.coordinates)
        return float(np.linalg.norm(s - t))

    def _compute_connection_strength(
        self,
        source: BrainRegion,
        target: BrainRegion,
        nt: NTSystem,
        distance: float
    ) -> float:
        """
        Compute connection strength based on anatomy.

        Args:
            source: Source region
            target: Target region
            nt: Neurotransmitter system
            distance: Distance in mm

        Returns:
            Connection strength (0-1)
        """
        # Check if source produces this NT
        if nt in [NTSystem.DOPAMINE, NTSystem.SEROTONIN,
                  NTSystem.NOREPINEPHRINE, NTSystem.ACETYLCHOLINE]:
            if not source.is_nt_source.get(nt, False):
                # Non-source regions don't project this NT long-range
                if distance > 10:
                    return 0.0

        # Distance decay
        distance_factor = np.exp(-distance / 50.0)  # 50mm length constant

        # Receptor density in target
        receptor_density = target.nt_receptors.get(nt, 0.5)

        # Region type compatibility
        type_factor = 1.0
        if source.region_type == RegionType.BRAINSTEM:
            # Brainstem projects widely
            type_factor = 0.8
        elif source.region_type == target.region_type == RegionType.CORTICAL:
            # Cortico-cortical
            type_factor = self.config.cortico_cortical_prob

        # Compute strength
        base_strength = distance_factor * receptor_density * type_factor

        # Cap long-range connections
        if distance > 30:
            base_strength = min(base_strength, self.config.max_long_range_strength)

        return float(np.clip(base_strength, 0.0, 1.0))

    def get_connectivity_matrix(self, nt: NTSystem) -> np.ndarray:
        """
        Get connectivity matrix for a specific NT.

        Args:
            nt: Neurotransmitter system

        Returns:
            Connectivity matrix [n_regions, n_regions]
        """
        return self._connectivity[nt].copy()

    def get_combined_connectivity(
        self,
        weights: dict[NTSystem, float] | None = None
    ) -> np.ndarray:
        """
        Get weighted combination of all NT connectivity.

        Args:
            weights: Per-NT weights (default: equal)

        Returns:
            Combined connectivity matrix
        """
        if weights is None:
            weights = {nt: 1.0 for nt in NTSystem}

        total_weight = sum(weights.values())
        combined = np.zeros_like(self._connectivity[NTSystem.GLUTAMATE])

        for nt, w in weights.items():
            combined += (w / total_weight) * self._connectivity[nt]

        return combined

    def get_region_index(self, name: str) -> int:
        """Get index of region by name."""
        region_names = list(self.regions.keys())
        return region_names.index(name)

    def get_region_names(self) -> list[str]:
        """Get list of region names."""
        return list(self.regions.keys())

    def get_distance_matrix(self) -> np.ndarray:
        """Get matrix of inter-region distances."""
        n = len(self.regions)
        distances = np.zeros((n, n), dtype=np.float32)
        region_list = list(self.regions.values())

        for i in range(n):
            for j in range(n):
                distances[i, j] = self._compute_distance(region_list[i], region_list[j])

        return distances

    def get_pathways_for_region(
        self,
        region_name: str,
        direction: str = "both"
    ) -> list[ProjectionPathway]:
        """
        Get pathways involving a region.

        Args:
            region_name: Region name
            direction: "in", "out", or "both"

        Returns:
            List of pathways
        """
        result = []
        for pathway in self.pathways:
            if direction in ["out", "both"] and pathway.source == region_name:
                result.append(pathway)
            if direction in ["in", "both"] and pathway.target == region_name:
                result.append(pathway)
        return result

    def get_nt_sources(self, nt: NTSystem) -> list[str]:
        """Get regions that are sources for an NT."""
        return [
            name for name, region in self.regions.items()
            if region.is_nt_source.get(nt, False)
        ]

    def to_coupling_matrix(self) -> np.ndarray:
        """
        Convert to 6x6 NT coupling matrix format.

        Aggregates regional connectivity into NT-NT coupling
        suitable for LearnableCoupling.

        Returns:
            6x6 coupling matrix [source_NT, target_NT]
        """
        # NT order: DA, 5HT, ACh, NE, GABA, Glu
        nt_order = [
            NTSystem.DOPAMINE, NTSystem.SEROTONIN, NTSystem.ACETYLCHOLINE,
            NTSystem.NOREPINEPHRINE, NTSystem.GABA, NTSystem.GLUTAMATE
        ]

        coupling = np.zeros((6, 6), dtype=np.float32)

        # Compute average connectivity between NT systems
        for i, source_nt in enumerate(nt_order):
            for j, target_nt in enumerate(nt_order):
                # Get regions that produce source_nt
                source_regions = self.get_nt_sources(source_nt)
                if not source_regions:
                    # For Glu/GABA, all regions produce locally
                    if source_nt in [NTSystem.GLUTAMATE, NTSystem.GABA]:
                        source_regions = self.get_region_names()
                    else:
                        continue

                # Average connectivity from source NT regions
                # weighted by target NT receptor density
                total = 0.0
                count = 0

                for source_name in source_regions:
                    source_idx = self.get_region_index(source_name)
                    for target_name, target_region in self.regions.items():
                        target_idx = self.get_region_index(target_name)
                        if source_idx == target_idx:
                            continue

                        conn = self._connectivity[source_nt][source_idx, target_idx]
                        receptor = target_region.nt_receptors.get(target_nt, 0.5)
                        total += conn * receptor
                        count += 1

                if count > 0:
                    coupling[i, j] = total / count

        # Apply biological signs
        # GABA is inhibitory
        coupling[4, :] = -np.abs(coupling[4, :])
        # DA inhibits 5-HT
        coupling[0, 1] = -np.abs(coupling[0, 1]) * 0.5

        return coupling

    def validate(self) -> tuple[bool, list[str]]:
        """
        Validate connectome for biological plausibility.

        Returns:
            (is_valid, list of issues)
        """
        issues = []

        # Check Dale's law if enforced
        if self.config.enforce_dale_law:
            for pathway in self.pathways:
                if pathway.nt_system == NTSystem.GABA and not pathway.is_inhibitory:
                    issues.append(f"GABA pathway {pathway.source}→{pathway.target} not inhibitory")
                if pathway.nt_system == NTSystem.GLUTAMATE and pathway.is_inhibitory:
                    issues.append(f"Glu pathway {pathway.source}→{pathway.target} is inhibitory")

        # Check for isolated regions
        combined = self.get_combined_connectivity()
        for i, name in enumerate(self.get_region_names()):
            if np.sum(combined[i, :]) + np.sum(combined[:, i]) < 0.01:
                issues.append(f"Region {name} is isolated")

        # Check NT sources exist
        for nt in [NTSystem.DOPAMINE, NTSystem.SEROTONIN,
                   NTSystem.NOREPINEPHRINE, NTSystem.ACETYLCHOLINE]:
            sources = self.get_nt_sources(nt)
            if not sources:
                issues.append(f"No source region for {nt.value}")

        return len(issues) == 0, issues

    def get_stats(self) -> dict:
        """Get connectome statistics."""
        combined = self.get_combined_connectivity()
        distances = self.get_distance_matrix()

        return {
            "n_regions": len(self.regions),
            "n_pathways": len(self.pathways),
            "region_types": {
                rt.name: sum(1 for r in self.regions.values() if r.region_type == rt)
                for rt in RegionType
            },
            "mean_connectivity": float(np.mean(combined)),
            "max_connectivity": float(np.max(combined)),
            "sparsity": float(np.mean(combined < 0.01)),
            "mean_distance_mm": float(np.mean(distances[distances > 0])),
            "nt_sources": {
                nt.value: self.get_nt_sources(nt)
                for nt in NTSystem
            },
        }


class ConnectomeIntegrator:
    """
    Integrates connectome with NCA delay and coupling systems.

    Provides unified interface for:
    1. Setting up delay system with anatomical distances
    2. Configuring coupling with connectome constraints
    3. Validating biological plausibility
    """

    def __init__(self, connectome: Connectome):
        """
        Initialize integrator.

        Args:
            connectome: Connectome instance
        """
        self.connectome = connectome

    def configure_delay_system(
        self,
        delay_system: TransmissionDelaySystem  # noqa: F821
    ) -> None:
        """
        Configure delay system with connectome distances.

        Args:
            delay_system: TransmissionDelaySystem to configure
        """
        distances = self.connectome.get_distance_matrix()
        n_regions = len(self.connectome.regions)

        # Ensure delay system has enough regions
        if delay_system.n_regions < n_regions:
            logger.warning(
                f"Delay system has {delay_system.n_regions} regions, "
                f"connectome has {n_regions}. Using minimum."
            )
            n_regions = min(n_regions, delay_system.n_regions)

        # Set distances
        for i in range(n_regions):
            for j in range(n_regions):
                if i != j:
                    delay_system.set_region_distance(i, j, float(distances[i, j]))

    def configure_coupling(
        self,
        coupling: LearnableCoupling  # noqa: F821
    ) -> None:
        """
        Configure learnable coupling with connectome constraints.

        Args:
            coupling: LearnableCoupling to configure
        """
        # Get NT-level coupling from connectome
        connectome_coupling = self.connectome.to_coupling_matrix()

        # Update coupling's K matrix
        # Blend with existing biological bounds
        coupling.K = 0.5 * coupling.K + 0.5 * connectome_coupling

        # Ensure within bounds
        if coupling.bounds is not None:
            coupling.K = coupling.bounds.clamp(coupling.K)

    def get_region_nt_modulation(
        self,
        region_name: str
    ) -> dict[str, float]:
        """
        Get NT modulation factors for a region.

        Based on receptor densities.

        Args:
            region_name: Region name

        Returns:
            Dict of NT name → modulation factor
        """
        if region_name not in self.connectome.regions:
            return {}

        region = self.connectome.regions[region_name]
        return {
            nt.value: density
            for nt, density in region.nt_receptors.items()
        }


# Convenience functions
def create_default_connectome() -> Connectome:
    """Create a default connectome with standard regions."""
    return Connectome(ConnectomeConfig())


def create_minimal_connectome() -> Connectome:
    """Create minimal connectome for testing."""
    config = ConnectomeConfig(
        n_cortical_regions=2,
        n_subcortical_regions=1,
        n_limbic_regions=1,
        n_brainstem_regions=2
    )
    return Connectome(config)


def get_pathway_summary(connectome: Connectome) -> str:
    """Get human-readable pathway summary."""
    lines = ["Connectome Pathways:"]

    for nt in NTSystem:
        pathways = [p for p in connectome.pathways if p.nt_system == nt]
        if pathways:
            lines.append(f"\n{nt.value.upper()}:")
            for p in pathways:
                sign = "⊖" if p.is_inhibitory else "⊕"
                lines.append(f"  {p.source} → {p.target} {sign} ({p.strength:.2f})")

    return "\n".join(lines)
