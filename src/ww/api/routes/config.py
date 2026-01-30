"""
World Weaver REST API Configuration Routes.

GET/PUT system configuration parameters for all learning subsystems.

API-CRITICAL-001 FIX: Config modification endpoints require admin auth.
RACE-CONDITION-FIX: Added asyncio.Lock for _runtime_config access.
"""

import asyncio
import logging

from fastapi import APIRouter, HTTPException, status
from pydantic import BaseModel, Field, model_validator

from ww.api.deps import AdminAuth
from ww.core.config import get_settings

logger = logging.getLogger(__name__)


# ============================================================================
# Configuration Models
# ============================================================================

class FSRSConfig(BaseModel):
    """FSRS (Free Spaced Repetition Scheduler) parameters."""

    defaultStability: float = Field(ge=0.1, le=10.0)
    retentionTarget: float = Field(ge=0.5, le=1.0)
    decayFactor: float = Field(ge=0.1, le=1.0)
    recencyDecay: float = Field(ge=0.01, le=1.0)


class ACTRConfig(BaseModel):
    """ACT-R activation model parameters."""

    decay: float = Field(ge=0.1, le=1.0)
    noise: float = Field(ge=0.0, le=1.0)
    threshold: float = Field(ge=-5.0, le=5.0)
    spreadingWeight: float = Field(ge=0.1, le=5.0)


class HebbianConfig(BaseModel):
    """Hebbian learning parameters."""

    learningRate: float = Field(ge=0.01, le=0.5)
    initialWeight: float = Field(ge=0.01, le=1.0)
    minWeight: float = Field(ge=0.001, le=0.1)
    decayRate: float = Field(ge=0.001, le=0.1)
    staleDays: int = Field(ge=1, le=365)


class AcetylcholineConfig(BaseModel):
    """Acetylcholine mode-switching parameters.

    Biologically: ACh high = encoding mode (DG/CA3 active), ACh low = retrieval mode (CA1).
    Critical constraint: encoding_threshold > retrieval_threshold for proper mode switching.
    """
    encodingThreshold: float = Field(
        default=0.7,
        ge=0.5,
        le=0.95,
        description="ACh level above which encoding mode is activated"
    )
    retrievalThreshold: float = Field(
        default=0.3,
        ge=0.1,
        le=0.5,
        description="ACh level below which retrieval mode is activated"
    )
    adaptationRate: float = Field(
        default=0.1,
        ge=0.01,
        le=1.0,
        description="Rate of ACh level adaptation to context"
    )
    hysteresis: float = Field(
        default=0.05,
        ge=0.0,
        le=0.2,
        description="Prevents rapid mode oscillation near thresholds"
    )

    @model_validator(mode='after')
    def validate_threshold_order(self):
        """Ensure encoding_threshold > retrieval_threshold (biological constraint)."""
        if self.encodingThreshold <= self.retrievalThreshold:
            raise ValueError(
                f"Encoding threshold ({self.encodingThreshold}) must be greater than "
                f"retrieval threshold ({self.retrievalThreshold}) for proper mode switching"
            )
        # Ensure sufficient gap for balanced mode
        gap = self.encodingThreshold - self.retrievalThreshold
        if gap < 0.2:
            raise ValueError(
                f"Threshold gap ({gap:.2f}) too small; need >= 0.2 for balanced mode region"
            )
        return self


class NeuromodConfig(BaseModel):
    """Neuromodulation system parameters."""

    dopamineBaseline: float = Field(ge=0.0, le=1.0)
    norepinephrineGain: float = Field(ge=0.1, le=5.0)
    serotoninDiscount: float = Field(ge=0.0, le=1.0)
    acetylcholineThreshold: float = Field(ge=0.0, le=1.0, description="Deprecated: use acetylcholine config")
    gabaInhibition: float = Field(ge=0.0, le=1.0)
    acetylcholine: AcetylcholineConfig | None = Field(default=None, description="Detailed ACh config")


class PatternSepConfig(BaseModel):
    """Pattern separation (Dentate Gyrus) parameters."""

    targetSparsity: float = Field(ge=0.01, le=0.2)
    maxNeighbors: int = Field(ge=1, le=200)
    maxNodes: int = Field(ge=10, le=10000)


class MemoryGateConfig(BaseModel):
    """Memory gate parameters."""

    baseThreshold: float = Field(ge=0.0, le=1.0)
    noveltyWeight: float = Field(ge=0.0, le=1.0)
    importanceWeight: float = Field(ge=0.0, le=1.0)
    contextWeight: float = Field(ge=0.0, le=1.0)


class ConsolidationConfig(BaseModel):
    """Memory consolidation parameters."""

    minSimilarity: float = Field(ge=0.5, le=1.0)
    minOccurrences: int = Field(ge=2, le=10)
    skillSimilarity: float = Field(ge=0.5, le=1.0)
    clusterSize: int = Field(ge=2, le=100)


class EpisodicWeightsConfig(BaseModel):
    """Episodic retrieval scoring weights."""

    semanticWeight: float = Field(ge=0.0, le=1.0)
    recencyWeight: float = Field(ge=0.0, le=1.0)
    outcomeWeight: float = Field(ge=0.0, le=1.0)
    importanceWeight: float = Field(ge=0.0, le=1.0)


class SemanticWeightsConfig(BaseModel):
    """Semantic retrieval scoring weights."""

    similarityWeight: float = Field(ge=0.0, le=1.0)
    activationWeight: float = Field(ge=0.0, le=1.0)
    retrievabilityWeight: float = Field(ge=0.0, le=1.0)


class ProceduralWeightsConfig(BaseModel):
    """Procedural retrieval scoring weights."""

    similarityWeight: float = Field(ge=0.0, le=1.0)
    successWeight: float = Field(ge=0.0, le=1.0)
    experienceWeight: float = Field(ge=0.0, le=1.0)


# Bioinspired Configuration Models
class DendriticConfig(BaseModel):
    """Dendritic processing parameters."""
    hiddenDim: int = Field(default=512, ge=64, le=4096)
    contextDim: int = Field(default=512, ge=64, le=4096)
    couplingStrength: float = Field(default=0.5, ge=0.0, le=1.0)
    tauDendrite: float = Field(default=10.0, ge=1.0, le=100.0)
    tauSoma: float = Field(default=15.0, ge=1.0, le=100.0)


class SparseEncoderConfig(BaseModel):
    """Sparse encoding (k-WTA) parameters."""
    hiddenDim: int = Field(default=8192, ge=512, le=65536)
    sparsity: float = Field(default=0.02, ge=0.001, le=0.2)
    useKwta: bool = Field(default=True)
    lateralInhibition: float = Field(default=0.2, ge=0.0, le=1.0)


class AttractorConfig(BaseModel):
    """Attractor network parameters."""
    settlingSteps: int = Field(default=10, ge=1, le=100)
    noiseStd: float = Field(default=0.01, ge=0.0, le=0.5)
    adaptationTau: float = Field(default=5.0, ge=0.1, le=50.0)
    stepSize: float = Field(default=0.1, ge=0.01, le=1.0)


class FastEpisodicConfig(BaseModel):
    """Fast Episodic Store parameters."""
    capacity: int = Field(default=10000, ge=100, le=1000000)
    learningRate: float = Field(default=0.1, ge=0.001, le=1.0)
    consolidationThreshold: float = Field(default=0.7, ge=0.0, le=1.0)


class NeuromodGainsConfig(BaseModel):
    """Neuromodulator gain parameters."""
    rhoDa: float = Field(default=1.0, ge=0.0, le=5.0)
    rhoNe: float = Field(default=1.0, ge=0.0, le=5.0)
    rhoAchFast: float = Field(default=1.0, ge=0.0, le=5.0)
    rhoAchSlow: float = Field(default=0.5, ge=0.0, le=5.0)
    alphaNe: float = Field(default=0.1, ge=0.0, le=1.0)


class EligibilityConfig(BaseModel):
    """Eligibility trace parameters."""
    decay: float = Field(default=0.95, ge=0.5, le=1.0)
    tauTrace: float = Field(default=20.0, ge=1.0, le=100.0)


class ThreeFactorConfig(BaseModel):
    """Three-factor learning rule parameters.

    Combined signal = effective_lr × eligibility × surprise × patience + bootstrap
    """
    achWeight: float = Field(default=0.4, ge=0.0, le=1.0, description="ACh contribution to neuromod gate")
    neWeight: float = Field(default=0.35, ge=0.0, le=1.0, description="NE contribution to neuromod gate")
    serotoninWeight: float = Field(default=0.25, ge=0.0, le=1.0, description="5-HT contribution")
    minEffectiveLr: float = Field(default=0.1, ge=0.01, le=0.5, description="Learning rate floor")
    maxEffectiveLr: float = Field(default=3.0, ge=1.0, le=10.0, description="Learning rate ceiling")
    bootstrapRate: float = Field(default=0.01, ge=0.001, le=0.1, description="Prevents zero-learning deadlock")

    @model_validator(mode='after')
    def validate_weight_sum(self):
        """Ensure neuromodulator weights sum to 1.0 for proper normalization."""
        total = self.achWeight + self.neWeight + self.serotoninWeight
        if abs(total - 1.0) > 0.01:
            raise ValueError(f"Three-factor weights must sum to 1.0, got {total:.3f}")
        return self


class LearnedGateConfig(BaseModel):
    """Learned memory gate (Thompson sampling) parameters.

    Controls the 'learning what to remember' system.
    """
    storeThreshold: float = Field(default=0.6, ge=0.3, le=0.9, description="Min score to store")
    bufferThreshold: float = Field(default=0.3, ge=0.1, le=0.5, description="Min score for buffer")
    learningRateMean: float = Field(default=0.1, ge=0.01, le=0.5, description="Thompson sampling mean update")
    learningRateVar: float = Field(default=0.05, ge=0.01, le=0.2, description="Thompson sampling var update")
    coldStartThreshold: int = Field(default=100, ge=10, le=1000, description="Samples before adaptive")
    thompsonTemperature: float = Field(default=1.0, ge=0.1, le=5.0, description="Exploration temperature")


class BioinspiredConfig(BaseModel):
    """Bioinspired architecture configuration."""
    enabled: bool = Field(default=True)
    dendritic: DendriticConfig = Field(default_factory=DendriticConfig)
    sparseEncoder: SparseEncoderConfig = Field(default_factory=SparseEncoderConfig)
    attractor: AttractorConfig = Field(default_factory=AttractorConfig)
    fastEpisodic: FastEpisodicConfig = Field(default_factory=FastEpisodicConfig)
    neuromodGains: NeuromodGainsConfig = Field(default_factory=NeuromodGainsConfig)
    eligibility: EligibilityConfig = Field(default_factory=EligibilityConfig)
    threeFactor: ThreeFactorConfig = Field(default_factory=ThreeFactorConfig)
    learnedGate: LearnedGateConfig = Field(default_factory=LearnedGateConfig)


class SystemConfigResponse(BaseModel):
    """Complete system configuration response."""

    fsrs: FSRSConfig
    actr: ACTRConfig
    hebbian: HebbianConfig
    neuromod: NeuromodConfig
    patternSep: PatternSepConfig
    memoryGate: MemoryGateConfig
    consolidation: ConsolidationConfig
    episodicWeights: EpisodicWeightsConfig
    semanticWeights: SemanticWeightsConfig
    proceduralWeights: ProceduralWeightsConfig
    bioinspired: BioinspiredConfig = Field(default_factory=BioinspiredConfig)


class SystemConfigUpdate(BaseModel):
    """Partial system configuration update."""

    fsrs: FSRSConfig | None = None
    actr: ACTRConfig | None = None
    hebbian: HebbianConfig | None = None
    neuromod: NeuromodConfig | None = None
    patternSep: PatternSepConfig | None = None
    memoryGate: MemoryGateConfig | None = None
    consolidation: ConsolidationConfig | None = None
    episodicWeights: EpisodicWeightsConfig | None = None
    semanticWeights: SemanticWeightsConfig | None = None
    proceduralWeights: ProceduralWeightsConfig | None = None
    bioinspired: BioinspiredConfig | None = None


# ============================================================================
# Runtime Configuration Storage
# ============================================================================

# Runtime configuration storage (allows runtime updates without env vars)
_runtime_config: dict = {}
_runtime_config_lock = asyncio.Lock()


async def _get_merged_config() -> SystemConfigResponse:
    """Get configuration with runtime overrides applied."""
    # Copy runtime config under lock to prevent race conditions
    async with _runtime_config_lock:
        runtime_copy = _runtime_config.copy()

    settings = get_settings()

    # Build base config from settings using the snapshot
    config = SystemConfigResponse(
        fsrs=FSRSConfig(
            defaultStability=runtime_copy.get(
                "fsrs_default_stability", settings.fsrs_default_stability
            ),
            retentionTarget=runtime_copy.get(
                "fsrs_retention_target", settings.fsrs_retention_target
            ),
            decayFactor=runtime_copy.get(
                "fsrs_decay_factor", settings.fsrs_decay_factor
            ),
            recencyDecay=runtime_copy.get(
                "fsrs_recency_decay", settings.fsrs_recency_decay
            ),
        ),
        actr=ACTRConfig(
            decay=runtime_copy.get("actr_decay", settings.actr_decay),
            noise=runtime_copy.get("actr_noise", settings.actr_noise),
            threshold=runtime_copy.get("actr_threshold", settings.actr_threshold),
            spreadingWeight=runtime_copy.get(
                "actr_spreading_strength", settings.actr_spreading_strength
            ),
        ),
        hebbian=HebbianConfig(
            learningRate=runtime_copy.get(
                "hebbian_learning_rate", settings.hebbian_learning_rate
            ),
            initialWeight=runtime_copy.get(
                "hebbian_initial_weight", settings.hebbian_initial_weight
            ),
            minWeight=runtime_copy.get(
                "hebbian_min_weight", settings.hebbian_min_weight
            ),
            decayRate=runtime_copy.get(
                "hebbian_decay_rate", settings.hebbian_decay_rate
            ),
            staleDays=runtime_copy.get(
                "hebbian_stale_days", settings.hebbian_stale_days
            ),
        ),
        neuromod=NeuromodConfig(
            dopamineBaseline=runtime_copy.get("dopamine_baseline", 0.5),
            norepinephrineGain=runtime_copy.get("norepinephrine_gain", 1.0),
            serotoninDiscount=runtime_copy.get("serotonin_discount", 0.5),
            acetylcholineThreshold=runtime_copy.get("acetylcholine_threshold", 0.5),
            gabaInhibition=runtime_copy.get("gaba_inhibition", 0.3),
            acetylcholine=AcetylcholineConfig(
                encodingThreshold=runtime_copy.get("ach_encoding_threshold", 0.7),
                retrievalThreshold=runtime_copy.get("ach_retrieval_threshold", 0.3),
                adaptationRate=runtime_copy.get("ach_adaptation_rate", 0.1),
                hysteresis=runtime_copy.get("ach_hysteresis", 0.05),
            ),
        ),
        patternSep=PatternSepConfig(
            targetSparsity=runtime_copy.get("pattern_sep_sparsity", 0.01),
            maxNeighbors=runtime_copy.get(
                "spreading_max_neighbors", settings.spreading_max_neighbors
            ),
            maxNodes=runtime_copy.get(
                "spreading_max_nodes", settings.spreading_max_nodes
            ),
        ),
        memoryGate=MemoryGateConfig(
            baseThreshold=runtime_copy.get("memory_gate_threshold", 0.3),
            noveltyWeight=runtime_copy.get("memory_gate_novelty", 0.3),
            importanceWeight=runtime_copy.get("memory_gate_importance", 0.4),
            contextWeight=runtime_copy.get("memory_gate_context", 0.3),
        ),
        consolidation=ConsolidationConfig(
            minSimilarity=runtime_copy.get(
                "consolidation_min_similarity", settings.consolidation_min_similarity
            ),
            minOccurrences=runtime_copy.get(
                "consolidation_min_occurrences", settings.consolidation_min_occurrences
            ),
            skillSimilarity=runtime_copy.get(
                "consolidation_skill_similarity", settings.consolidation_skill_similarity
            ),
            clusterSize=runtime_copy.get(
                "hdbscan_min_cluster_size", settings.hdbscan_min_cluster_size
            ),
        ),
        episodicWeights=EpisodicWeightsConfig(
            semanticWeight=runtime_copy.get(
                "episodic_weight_semantic", settings.episodic_weight_semantic
            ),
            recencyWeight=runtime_copy.get(
                "episodic_weight_recency", settings.episodic_weight_recency
            ),
            outcomeWeight=runtime_copy.get(
                "episodic_weight_outcome", settings.episodic_weight_outcome
            ),
            importanceWeight=runtime_copy.get(
                "episodic_weight_importance", settings.episodic_weight_importance
            ),
        ),
        semanticWeights=SemanticWeightsConfig(
            similarityWeight=runtime_copy.get(
                "semantic_weight_similarity", settings.semantic_weight_similarity
            ),
            activationWeight=runtime_copy.get(
                "semantic_weight_activation", settings.semantic_weight_activation
            ),
            retrievabilityWeight=runtime_copy.get(
                "semantic_weight_retrievability", settings.semantic_weight_retrievability
            ),
        ),
        proceduralWeights=ProceduralWeightsConfig(
            similarityWeight=runtime_copy.get(
                "procedural_weight_similarity", settings.procedural_weight_similarity
            ),
            successWeight=runtime_copy.get(
                "procedural_weight_success", settings.procedural_weight_success
            ),
            experienceWeight=runtime_copy.get(
                "procedural_weight_experience", settings.procedural_weight_experience
            ),
        ),
        bioinspired=BioinspiredConfig(
            enabled=runtime_copy.get("bioinspired_enabled", True),
            dendritic=DendriticConfig(
                hiddenDim=runtime_copy.get("dendritic_hidden_dim", 512),
                contextDim=runtime_copy.get("dendritic_context_dim", 512),
                couplingStrength=runtime_copy.get("dendritic_coupling", 0.5),
                tauDendrite=runtime_copy.get("dendritic_tau_dendrite", 10.0),
                tauSoma=runtime_copy.get("dendritic_tau_soma", 15.0),
            ),
            sparseEncoder=SparseEncoderConfig(
                hiddenDim=runtime_copy.get("sparse_hidden_dim", 8192),
                sparsity=runtime_copy.get("sparse_sparsity", 0.02),
                useKwta=runtime_copy.get("sparse_use_kwta", True),
                lateralInhibition=runtime_copy.get("sparse_lateral_inhibition", 0.2),
            ),
            attractor=AttractorConfig(
                settlingSteps=runtime_copy.get("attractor_settling_steps", 10),
                noiseStd=runtime_copy.get("attractor_noise_std", 0.01),
                adaptationTau=runtime_copy.get("attractor_adaptation_tau", 5.0),
                stepSize=runtime_copy.get("attractor_step_size", 0.1),
            ),
            fastEpisodic=FastEpisodicConfig(
                capacity=runtime_copy.get("fes_capacity", 10000),
                learningRate=runtime_copy.get("fes_learning_rate", 0.1),
                consolidationThreshold=runtime_copy.get("fes_consolidation_threshold", 0.7),
            ),
            neuromodGains=NeuromodGainsConfig(
                rhoDa=runtime_copy.get("neuromod_rho_da", 1.0),
                rhoNe=runtime_copy.get("neuromod_rho_ne", 1.0),
                rhoAchFast=runtime_copy.get("neuromod_rho_ach_fast", 1.0),
                rhoAchSlow=runtime_copy.get("neuromod_rho_ach_slow", 0.5),
                alphaNe=runtime_copy.get("neuromod_alpha_ne", 0.1),
            ),
            eligibility=EligibilityConfig(
                decay=runtime_copy.get("eligibility_decay", 0.95),
                tauTrace=runtime_copy.get("eligibility_tau_trace", 20.0),
            ),
            threeFactor=ThreeFactorConfig(
                achWeight=runtime_copy.get("three_factor_ach_weight", 0.4),
                neWeight=runtime_copy.get("three_factor_ne_weight", 0.35),
                serotoninWeight=runtime_copy.get("three_factor_serotonin_weight", 0.25),
                minEffectiveLr=runtime_copy.get("three_factor_min_lr", 0.1),
                maxEffectiveLr=runtime_copy.get("three_factor_max_lr", 3.0),
                bootstrapRate=runtime_copy.get("three_factor_bootstrap", 0.01),
            ),
            learnedGate=LearnedGateConfig(
                storeThreshold=runtime_copy.get("learned_gate_store_threshold", 0.6),
                bufferThreshold=runtime_copy.get("learned_gate_buffer_threshold", 0.3),
                learningRateMean=runtime_copy.get("learned_gate_lr_mean", 0.1),
                learningRateVar=runtime_copy.get("learned_gate_lr_var", 0.05),
                coldStartThreshold=runtime_copy.get("learned_gate_cold_start", 100),
                thompsonTemperature=runtime_copy.get("learned_gate_temperature", 1.0),
            ),
        ),
    )

    return config


def _apply_config_update(update: SystemConfigUpdate) -> dict:
    """
    Build updates dictionary from SystemConfigUpdate.

    Returns dict of runtime config keys to update.
    Raises HTTPException for validation errors.
    """
    updates: dict = {}

    # Apply FSRS updates
    if update.fsrs:
        updates["fsrs_default_stability"] = update.fsrs.defaultStability
        updates["fsrs_retention_target"] = update.fsrs.retentionTarget
        updates["fsrs_decay_factor"] = update.fsrs.decayFactor
        updates["fsrs_recency_decay"] = update.fsrs.recencyDecay

    # Apply ACT-R updates
    if update.actr:
        updates["actr_decay"] = update.actr.decay
        updates["actr_noise"] = update.actr.noise
        updates["actr_threshold"] = update.actr.threshold
        updates["actr_spreading_strength"] = update.actr.spreadingWeight

    # Apply Hebbian updates
    if update.hebbian:
        updates["hebbian_learning_rate"] = update.hebbian.learningRate
        updates["hebbian_initial_weight"] = update.hebbian.initialWeight
        updates["hebbian_min_weight"] = update.hebbian.minWeight
        updates["hebbian_decay_rate"] = update.hebbian.decayRate
        updates["hebbian_stale_days"] = update.hebbian.staleDays

    # Apply Neuromodulation updates
    if update.neuromod:
        updates["dopamine_baseline"] = update.neuromod.dopamineBaseline
        updates["norepinephrine_gain"] = update.neuromod.norepinephrineGain
        updates["serotonin_discount"] = update.neuromod.serotoninDiscount
        updates["acetylcholine_threshold"] = update.neuromod.acetylcholineThreshold
        updates["gaba_inhibition"] = update.neuromod.gabaInhibition
        # Apply ACh config if provided
        if update.neuromod.acetylcholine:
            updates["ach_encoding_threshold"] = update.neuromod.acetylcholine.encodingThreshold
            updates["ach_retrieval_threshold"] = update.neuromod.acetylcholine.retrievalThreshold
            updates["ach_adaptation_rate"] = update.neuromod.acetylcholine.adaptationRate
            updates["ach_hysteresis"] = update.neuromod.acetylcholine.hysteresis

    # Apply Pattern Separation updates
    if update.patternSep:
        updates["pattern_sep_sparsity"] = update.patternSep.targetSparsity
        updates["spreading_max_neighbors"] = update.patternSep.maxNeighbors
        updates["spreading_max_nodes"] = update.patternSep.maxNodes

    # Apply Memory Gate updates
    if update.memoryGate:
        updates["memory_gate_threshold"] = update.memoryGate.baseThreshold
        updates["memory_gate_novelty"] = update.memoryGate.noveltyWeight
        updates["memory_gate_importance"] = update.memoryGate.importanceWeight
        updates["memory_gate_context"] = update.memoryGate.contextWeight

    # Apply Consolidation updates
    if update.consolidation:
        updates["consolidation_min_similarity"] = update.consolidation.minSimilarity
        updates["consolidation_min_occurrences"] = update.consolidation.minOccurrences
        updates["consolidation_skill_similarity"] = update.consolidation.skillSimilarity
        updates["hdbscan_min_cluster_size"] = update.consolidation.clusterSize

    # Apply Episodic Weights updates
    if update.episodicWeights:
        # Validate weights sum to 1.0
        total = (
            update.episodicWeights.semanticWeight
            + update.episodicWeights.recencyWeight
            + update.episodicWeights.outcomeWeight
            + update.episodicWeights.importanceWeight
        )
        if abs(total - 1.0) > 0.001:
            raise HTTPException(
                status_code=status.HTTP_422_UNPROCESSABLE_ENTITY,
                detail=f"Episodic weights must sum to 1.0, got {total}",
            )
        updates["episodic_weight_semantic"] = update.episodicWeights.semanticWeight
        updates["episodic_weight_recency"] = update.episodicWeights.recencyWeight
        updates["episodic_weight_outcome"] = update.episodicWeights.outcomeWeight
        updates["episodic_weight_importance"] = update.episodicWeights.importanceWeight

    # Apply Semantic Weights updates
    if update.semanticWeights:
        total = (
            update.semanticWeights.similarityWeight
            + update.semanticWeights.activationWeight
            + update.semanticWeights.retrievabilityWeight
        )
        if abs(total - 1.0) > 0.001:
            raise HTTPException(
                status_code=status.HTTP_422_UNPROCESSABLE_ENTITY,
                detail=f"Semantic weights must sum to 1.0, got {total}",
            )
        updates["semantic_weight_similarity"] = update.semanticWeights.similarityWeight
        updates["semantic_weight_activation"] = update.semanticWeights.activationWeight
        updates["semantic_weight_retrievability"] = update.semanticWeights.retrievabilityWeight

    # Apply Procedural Weights updates
    if update.proceduralWeights:
        total = (
            update.proceduralWeights.similarityWeight
            + update.proceduralWeights.successWeight
            + update.proceduralWeights.experienceWeight
        )
        if abs(total - 1.0) > 0.001:
            raise HTTPException(
                status_code=status.HTTP_422_UNPROCESSABLE_ENTITY,
                detail=f"Procedural weights must sum to 1.0, got {total}",
            )
        updates["procedural_weight_similarity"] = update.proceduralWeights.similarityWeight
        updates["procedural_weight_success"] = update.proceduralWeights.successWeight
        updates["procedural_weight_experience"] = update.proceduralWeights.experienceWeight

    # Apply Bioinspired updates
    if update.bioinspired:
        updates["bioinspired_enabled"] = update.bioinspired.enabled
        # Dendritic
        updates["dendritic_hidden_dim"] = update.bioinspired.dendritic.hiddenDim
        updates["dendritic_context_dim"] = update.bioinspired.dendritic.contextDim
        updates["dendritic_coupling"] = update.bioinspired.dendritic.couplingStrength
        updates["dendritic_tau_dendrite"] = update.bioinspired.dendritic.tauDendrite
        updates["dendritic_tau_soma"] = update.bioinspired.dendritic.tauSoma
        # Sparse Encoder
        updates["sparse_hidden_dim"] = update.bioinspired.sparseEncoder.hiddenDim
        updates["sparse_sparsity"] = update.bioinspired.sparseEncoder.sparsity
        updates["sparse_use_kwta"] = update.bioinspired.sparseEncoder.useKwta
        updates["sparse_lateral_inhibition"] = update.bioinspired.sparseEncoder.lateralInhibition
        # Attractor
        updates["attractor_settling_steps"] = update.bioinspired.attractor.settlingSteps
        updates["attractor_noise_std"] = update.bioinspired.attractor.noiseStd
        updates["attractor_adaptation_tau"] = update.bioinspired.attractor.adaptationTau
        updates["attractor_step_size"] = update.bioinspired.attractor.stepSize
        # Fast Episodic Store
        updates["fes_capacity"] = update.bioinspired.fastEpisodic.capacity
        updates["fes_learning_rate"] = update.bioinspired.fastEpisodic.learningRate
        updates["fes_consolidation_threshold"] = update.bioinspired.fastEpisodic.consolidationThreshold
        # Neuromodulator Gains
        updates["neuromod_rho_da"] = update.bioinspired.neuromodGains.rhoDa
        updates["neuromod_rho_ne"] = update.bioinspired.neuromodGains.rhoNe
        updates["neuromod_rho_ach_fast"] = update.bioinspired.neuromodGains.rhoAchFast
        updates["neuromod_rho_ach_slow"] = update.bioinspired.neuromodGains.rhoAchSlow
        updates["neuromod_alpha_ne"] = update.bioinspired.neuromodGains.alphaNe
        # Eligibility
        updates["eligibility_decay"] = update.bioinspired.eligibility.decay
        updates["eligibility_tau_trace"] = update.bioinspired.eligibility.tauTrace
        # Three-Factor Learning Rule
        updates["three_factor_ach_weight"] = update.bioinspired.threeFactor.achWeight
        updates["three_factor_ne_weight"] = update.bioinspired.threeFactor.neWeight
        updates["three_factor_serotonin_weight"] = update.bioinspired.threeFactor.serotoninWeight
        updates["three_factor_min_lr"] = update.bioinspired.threeFactor.minEffectiveLr
        updates["three_factor_max_lr"] = update.bioinspired.threeFactor.maxEffectiveLr
        updates["three_factor_bootstrap"] = update.bioinspired.threeFactor.bootstrapRate
        # Learned Memory Gate (Thompson sampling)
        updates["learned_gate_store_threshold"] = update.bioinspired.learnedGate.storeThreshold
        updates["learned_gate_buffer_threshold"] = update.bioinspired.learnedGate.bufferThreshold
        updates["learned_gate_lr_mean"] = update.bioinspired.learnedGate.learningRateMean
        updates["learned_gate_lr_var"] = update.bioinspired.learnedGate.learningRateVar
        updates["learned_gate_cold_start"] = update.bioinspired.learnedGate.coldStartThreshold
        updates["learned_gate_temperature"] = update.bioinspired.learnedGate.thompsonTemperature

    return updates


# ============================================================================
# Configuration Routes
# ============================================================================

def create_config_routes() -> APIRouter:
    """Config-specific routes (GET/PUT/reset)."""
    router = APIRouter(tags=["config"])

    @router.get("/", response_model=SystemConfigResponse)
    async def get_config():
        """
        Get current system configuration.

        Returns all tunable parameters for FSRS, ACT-R, Hebbian learning,
        neuromodulation, pattern separation, and retrieval scoring.
        """
        return await _get_merged_config()

    @router.put("/", response_model=SystemConfigResponse)
    async def update_config(update: SystemConfigUpdate, _: AdminAuth):
        """
        Update system configuration at runtime.

        Requires admin authentication via X-Admin-Key header.

        Partial updates are supported - only specify sections you want to change.
        Changes take effect immediately but do not persist across restarts.
        To persist changes, update environment variables or .env file.
        """
        try:
            # Build updates dict
            updates = _apply_config_update(update)

            # Apply all updates atomically under lock
            async with _runtime_config_lock:
                _runtime_config.update(updates)

            logger.info(f"Configuration updated: {list(updates.keys())}")
            return await _get_merged_config()

        except HTTPException:
            raise
        except Exception as e:
            logger.error(f"Config update failed: {e}")
            raise HTTPException(
                status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
                detail=f"Failed to update configuration: {e!s}",
            )

    @router.post("/reset")
    async def reset_config(_: AdminAuth):
        """
        Reset all runtime configuration to defaults.

        Requires admin authentication via X-Admin-Key header.

        Clears all runtime overrides and returns to environment variable values.
        """
        async with _runtime_config_lock:
            _runtime_config.clear()
        logger.info("Configuration reset to defaults")
        return {"status": "reset", "message": "Configuration reset to defaults"}

    return router


# ============================================================================
# Preset Routes
# ============================================================================

# Bio-Plausible Presets (CompBio Agent recommendations)
PRESETS: dict[str, dict] = {
    "bio-plausible": {
        "description": "CompBio-recommended values for maximum biological plausibility",
        "changes": {
            # E/I ratio closer to cortical (4:1)
            "gaba_inhibition": 0.75,
            "sparse_lateral_inhibition": 0.75,
            # Hippocampal DG ~2-5% active
            "sparse_sparsity": 0.05,
            "pattern_sep_sparsity": 0.05,
            # Faster LC-NE burst decay
            "neuromod_alpha_ne": 0.3,
            # Balanced LTP/LTD (a_minus ≈ 0.02 vs default 0.00525)
            "eligibility_decay": 0.98,
            # More biologically realistic time constants
            "dendritic_tau_dendrite": 15.0,
            "dendritic_tau_soma": 20.0,
            # Stronger attractor dynamics
            "attractor_settling_steps": 20,
            # Three-factor learning with serotonin patience emphasis
            "three_factor_serotonin_weight": 0.35,
            "three_factor_ach_weight": 0.35,
            "three_factor_ne_weight": 0.30,
        },
    },
    "performance": {
        "description": "Optimized for computational efficiency over bio-fidelity",
        "changes": {
            # Sparser for speed
            "sparse_sparsity": 0.01,
            "sparse_hidden_dim": 4096,
            # Fewer attractor iterations
            "attractor_settling_steps": 5,
            # Faster eligibility decay
            "eligibility_decay": 0.90,
            # Aggressive memory gating
            "learned_gate_store_threshold": 0.7,
            "memory_gate_threshold": 0.5,
        },
    },
    "conservative": {
        "description": "Prioritizes memory retention and stability",
        "changes": {
            # Higher retention target
            "fsrs_retention_target": 0.95,
            "fsrs_default_stability": 2.0,
            # Lower memory gate threshold (store more)
            "learned_gate_store_threshold": 0.4,
            "memory_gate_threshold": 0.2,
            # Slower learning rates
            "hebbian_learning_rate": 0.05,
            "fes_learning_rate": 0.05,
            # Longer eligibility traces
            "eligibility_tau_trace": 30.0,
        },
    },
    "exploration": {
        "description": "Biased toward novelty-seeking and exploration",
        "changes": {
            # Higher arousal baseline
            "norepinephrine_gain": 1.5,
            # Higher dopamine surprise sensitivity
            "dopamine_baseline": 0.3,
            # More exploration in Thompson sampling
            "learned_gate_temperature": 2.0,
            # ACh biased toward encoding
            "acetylcholine_threshold": 0.7,
            # Higher attractor noise for exploration
            "attractor_noise_std": 0.05,
        },
    },
}


class PresetInfo(BaseModel):
    """Information about a configuration preset."""
    name: str
    description: str
    changes: dict


class PresetListResponse(BaseModel):
    """List of available presets."""
    presets: list[PresetInfo]


class PresetApplyResponse(BaseModel):
    """Response from applying a preset."""
    preset: str
    applied_changes: dict
    config: SystemConfigResponse


def create_preset_routes() -> APIRouter:
    """Preset management routes."""
    router = APIRouter(prefix="/presets", tags=["presets"])

    @router.get("", response_model=PresetListResponse)
    async def list_presets():
        """
        List available configuration presets.

        Presets provide curated parameter combinations for different use cases:
        - bio-plausible: CompBio-recommended values for biological fidelity
        - performance: Optimized for speed over bio-accuracy
        - conservative: Prioritizes memory retention
        - exploration: Biased toward novelty-seeking
        """
        return PresetListResponse(
            presets=[
                PresetInfo(name=name, description=info["description"], changes=info["changes"])
                for name, info in PRESETS.items()
            ]
        )

    @router.post("/{preset_name}", response_model=PresetApplyResponse)
    async def apply_preset(preset_name: str, _: AdminAuth):
        """
        Apply a configuration preset.

        Requires admin authentication via X-Admin-Key header.

        Available presets:
        - bio-plausible: CompBio-recommended biological fidelity
        - performance: Computational efficiency
        - conservative: Memory retention priority
        - exploration: Novelty-seeking bias
        """
        if preset_name not in PRESETS:
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND,
                detail=f"Preset '{preset_name}' not found. Available: {list(PRESETS.keys())}",
            )

        preset = PRESETS[preset_name]
        applied = {}

        # Apply all changes from preset under lock
        async with _runtime_config_lock:
            for key, value in preset["changes"].items():
                _runtime_config[key] = value
                applied[key] = value

        logger.info(f"Applied preset '{preset_name}': {list(applied.keys())}")

        return PresetApplyResponse(
            preset=preset_name,
            applied_changes=applied,
            config=await _get_merged_config(),
        )

    return router


# ============================================================================
# Main Router Factory
# ============================================================================

def create_ww_router() -> APIRouter:
    """
    Create main World Weaver configuration router.

    Delegates to focused sub-routers:
    - Config routes: GET/PUT/reset configuration
    - Preset routes: List and apply presets
    """
    router = APIRouter()

    # Include config routes (no prefix, mounted at /config by parent)
    router.include_router(create_config_routes())

    # Include preset routes (has /presets prefix)
    router.include_router(create_preset_routes())

    return router


# Legacy compatibility: expose router as module-level variable
router = create_ww_router()
