/**
 * System Configuration Panel
 * Full CRUD and parameter editing for all World Weaver internals
 */

import React, { useEffect, useState } from 'react';
import './ConfigPanel.scss';

interface SystemConfig {
  // FSRS Parameters
  fsrs: {
    defaultStability: number;
    retentionTarget: number;
    decayFactor: number;
    recencyDecay: number;
  };
  // ACT-R Parameters
  actr: {
    decay: number;
    noise: number;
    threshold: number;
    spreadingWeight: number;
  };
  // Hebbian Learning
  hebbian: {
    learningRate: number;
    initialWeight: number;
    minWeight: number;
    decayRate: number;
    staleDays: number;
  };
  // Neuromodulation
  neuromod: {
    dopamineBaseline: number;
    norepinephrineGain: number;
    serotoninDiscount: number;
    acetylcholineThreshold: number;
    gabaInhibition: number;
  };
  // Pattern Separation
  patternSep: {
    targetSparsity: number;
    maxNeighbors: number;
    maxNodes: number;
  };
  // Memory Gate
  memoryGate: {
    baseThreshold: number;
    noveltyWeight: number;
    importanceWeight: number;
    contextWeight: number;
  };
  // Consolidation
  consolidation: {
    minSimilarity: number;
    minOccurrences: number;
    skillSimilarity: number;
    clusterSize: number;
  };
  // Episodic Retrieval Weights
  episodicWeights: {
    semanticWeight: number;
    recencyWeight: number;
    outcomeWeight: number;
    importanceWeight: number;
  };
  // Semantic Retrieval Weights
  semanticWeights: {
    similarityWeight: number;
    activationWeight: number;
    retrievabilityWeight: number;
  };
  // Procedural Retrieval Weights
  proceduralWeights: {
    similarityWeight: number;
    successWeight: number;
    experienceWeight: number;
  };
  // Bioinspired Configuration (NEW - CompBio Integration)
  bioinspired: {
    enabled: boolean;
    // Dendritic Processing
    dendritic: {
      hiddenDim: number;
      contextDim: number;
      couplingStrength: number;
      tauDendrite: number;
      tauSoma: number;
    };
    // Sparse Encoding
    sparseEncoder: {
      hiddenDim: number;
      sparsity: number;
      useKwta: boolean;
      lateralInhibition: number;
    };
    // Attractor Network
    attractor: {
      settlingSteps: number;
      noiseStd: number;
      adaptationTau: number;
      stepSize: number;
    };
    // Fast Episodic Store
    fastEpisodic: {
      capacity: number;
      learningRate: number;
      consolidationThreshold: number;
    };
    // Enhanced Neuromodulator Gains
    neuromodGains: {
      rhoDa: number;
      rhoNe: number;
      rhoAchFast: number;
      rhoAchSlow: number;
      alphaNe: number;
    };
    // Eligibility Traces
    eligibility: {
      decay: number;
      tauTrace: number;
    };
  };
}

const defaultConfig: SystemConfig = {
  fsrs: {
    defaultStability: 1.0,
    retentionTarget: 0.9,
    decayFactor: 0.9,
    recencyDecay: 0.1,
  },
  actr: {
    decay: 0.5,
    noise: 0.0,
    threshold: 0.0,
    spreadingWeight: 1.6,
  },
  hebbian: {
    learningRate: 0.1,
    initialWeight: 0.1,
    minWeight: 0.01,
    decayRate: 0.01,
    staleDays: 30,
  },
  neuromod: {
    dopamineBaseline: 0.5,
    norepinephrineGain: 1.0,
    serotoninDiscount: 0.5,
    acetylcholineThreshold: 0.5,
    gabaInhibition: 0.3,
  },
  patternSep: {
    targetSparsity: 0.04,
    maxNeighbors: 50,
    maxNodes: 1000,
  },
  memoryGate: {
    baseThreshold: 0.3,
    noveltyWeight: 0.3,
    importanceWeight: 0.4,
    contextWeight: 0.3,
  },
  consolidation: {
    minSimilarity: 0.75,
    minOccurrences: 3,
    skillSimilarity: 0.85,
    clusterSize: 3,
  },
  episodicWeights: {
    semanticWeight: 0.4,
    recencyWeight: 0.25,
    outcomeWeight: 0.2,
    importanceWeight: 0.15,
  },
  semanticWeights: {
    similarityWeight: 0.4,
    activationWeight: 0.35,
    retrievabilityWeight: 0.25,
  },
  proceduralWeights: {
    similarityWeight: 0.6,
    successWeight: 0.3,
    experienceWeight: 0.1,
  },
  // Bioinspired defaults (from CompBio architecture)
  bioinspired: {
    enabled: true,
    dendritic: {
      hiddenDim: 512,
      contextDim: 512,
      couplingStrength: 0.5,
      tauDendrite: 10.0,
      tauSoma: 15.0,
    },
    sparseEncoder: {
      hiddenDim: 8192,
      sparsity: 0.02,
      useKwta: true,
      lateralInhibition: 0.2,
    },
    attractor: {
      settlingSteps: 10,
      noiseStd: 0.01,
      adaptationTau: 200.0,
      stepSize: 0.1,
    },
    fastEpisodic: {
      capacity: 10000,
      learningRate: 0.1,
      consolidationThreshold: 0.7,
    },
    neuromodGains: {
      rhoDa: 2.0,
      rhoNe: 1.5,
      rhoAchFast: 3.0,
      rhoAchSlow: 0.5,
      alphaNe: 0.3,
    },
    eligibility: {
      decay: 0.95,
      tauTrace: 20.0,
    },
  },
};

interface ConfigSectionProps {
  title: string;
  icon: string;
  children: React.ReactNode;
  isOpen: boolean;
  onToggle: () => void;
}

const ConfigSection: React.FC<ConfigSectionProps> = ({
  title,
  icon,
  children,
  isOpen,
  onToggle,
}) => (
  <div className={`config-section ${isOpen ? 'open' : ''}`}>
    <button className="section-header" onClick={onToggle}>
      <i className={`fa ${icon}`} />
      <span>{title}</span>
      <i className={`fa fa-chevron-${isOpen ? 'up' : 'down'}`} />
    </button>
    {isOpen && <div className="section-content">{children}</div>}
  </div>
);

interface ToggleInputProps {
  label: string;
  value: boolean;
  description?: string;
  onChange: (value: boolean) => void;
}

const ToggleInput: React.FC<ToggleInputProps> = ({
  label,
  value,
  description,
  onChange,
}) => (
  <div className="toggle-input">
    <div className="toggle-header">
      <label>{label}</label>
      <button
        className={`toggle-btn ${value ? 'active' : ''}`}
        onClick={() => onChange(!value)}
      >
        <span className="toggle-track">
          <span className="toggle-thumb" />
        </span>
        <span className="toggle-label">{value ? 'ON' : 'OFF'}</span>
      </button>
    </div>
    {description && <p className="description">{description}</p>}
  </div>
);

interface SliderInputProps {
  label: string;
  value: number;
  min: number;
  max: number;
  step: number;
  unit?: string;
  description?: string;
  onChange: (value: number) => void;
}

const SliderInput: React.FC<SliderInputProps> = ({
  label,
  value,
  min,
  max,
  step,
  unit = '',
  description,
  onChange,
}) => (
  <div className="slider-input">
    <div className="slider-header">
      <label>{label}</label>
      <div className="value-display">
        <input
          type="number"
          value={value}
          min={min}
          max={max}
          step={step}
          onChange={(e) => onChange(parseFloat(e.target.value))}
        />
        {unit && <span className="unit">{unit}</span>}
      </div>
    </div>
    <input
      type="range"
      min={min}
      max={max}
      step={step}
      value={value}
      onChange={(e) => onChange(parseFloat(e.target.value))}
    />
    {description && <p className="description">{description}</p>}
  </div>
);

export const ConfigPanel: React.FC = () => {
  const [config, setConfig] = useState<SystemConfig>(defaultConfig);
  const [openSections, setOpenSections] = useState<Set<string>>(new Set(['fsrs']));
  const [hasChanges, setHasChanges] = useState(false);
  const [saving, setSaving] = useState(false);
  const [error, setError] = useState<string | null>(null);

  useEffect(() => {
    loadConfig();
  }, []);

  const loadConfig = async () => {
    try {
      const response = await fetch('/api/v1/config');
      if (response.ok) {
        const data = await response.json();
        // Merge with defaults to handle missing sections (e.g., bioinspired)
        setConfig({
          ...defaultConfig,
          ...data,
          // Deep merge for nested objects that may be missing
          bioinspired: {
            ...defaultConfig.bioinspired,
            ...(data.bioinspired || {}),
            dendritic: { ...defaultConfig.bioinspired.dendritic, ...(data.bioinspired?.dendritic || {}) },
            sparseEncoder: { ...defaultConfig.bioinspired.sparseEncoder, ...(data.bioinspired?.sparseEncoder || {}) },
            attractor: { ...defaultConfig.bioinspired.attractor, ...(data.bioinspired?.attractor || {}) },
            fastEpisodic: { ...defaultConfig.bioinspired.fastEpisodic, ...(data.bioinspired?.fastEpisodic || {}) },
            neuromodGains: { ...defaultConfig.bioinspired.neuromodGains, ...(data.bioinspired?.neuromodGains || {}) },
            eligibility: { ...defaultConfig.bioinspired.eligibility, ...(data.bioinspired?.eligibility || {}) },
          },
        });
      }
    } catch (err) {
      // Use defaults if API not available
    }
  };

  const saveConfig = async () => {
    setSaving(true);
    setError(null);
    try {
      const response = await fetch('/api/v1/config', {
        method: 'PUT',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify(config),
      });
      if (!response.ok) {
        throw new Error('Failed to save configuration');
      }
      setHasChanges(false);
    } catch (err) {
      setError(err instanceof Error ? err.message : 'Save failed');
    } finally {
      setSaving(false);
    }
  };

  const resetToDefaults = () => {
    setConfig(defaultConfig);
    setHasChanges(true);
  };

  const toggleSection = (section: string) => {
    setOpenSections((prev) => {
      const next = new Set(prev);
      if (next.has(section)) {
        next.delete(section);
      } else {
        next.add(section);
      }
      return next;
    });
  };

  const updateConfig = <K extends keyof SystemConfig>(
    section: K,
    key: keyof SystemConfig[K],
    value: number
  ) => {
    setConfig((prev) => ({
      ...prev,
      [section]: {
        ...prev[section],
        [key]: value,
      },
    }));
    setHasChanges(true);
  };

  // Helper for nested bioinspired config updates
  const updateBioConfig = <
    S extends Exclude<keyof SystemConfig['bioinspired'], 'enabled'>,
    K extends keyof SystemConfig['bioinspired'][S]
  >(
    subSection: S,
    key: K,
    value: SystemConfig['bioinspired'][S][K]
  ) => {
    setConfig((prev) => {
      const currentSubSection = prev.bioinspired[subSection] as Record<string, unknown>;
      return {
        ...prev,
        bioinspired: {
          ...prev.bioinspired,
          [subSection]: {
            ...currentSubSection,
            [key]: value,
          },
        },
      };
    });
    setHasChanges(true);
  };

  const toggleBioEnabled = () => {
    setConfig((prev) => ({
      ...prev,
      bioinspired: {
        ...prev.bioinspired,
        enabled: !prev.bioinspired.enabled,
      },
    }));
    setHasChanges(true);
  };

  return (
    <div className="config-panel">
      {/* Header */}
      <div className="panel-header">
        <h1>
          <i className="fa fa-sliders" />
          System Configuration
        </h1>
        <div className="header-actions">
          <button className="btn-secondary" onClick={resetToDefaults}>
            <i className="fa fa-undo" />
            Reset Defaults
          </button>
          <button
            className="btn-primary"
            onClick={saveConfig}
            disabled={!hasChanges || saving}
          >
            {saving ? (
              <i className="fa fa-spinner fa-spin" />
            ) : (
              <i className="fa fa-save" />
            )}
            Save Changes
          </button>
        </div>
      </div>

      {error && (
        <div className="error-banner">
          <i className="fa fa-exclamation-triangle" />
          {error}
        </div>
      )}

      {hasChanges && (
        <div className="changes-banner">
          <i className="fa fa-info-circle" />
          You have unsaved changes
        </div>
      )}

      <div className="config-sections">
        {/* FSRS Memory Decay */}
        <ConfigSection
          title="FSRS Memory Decay"
          icon="fa-clock"
          isOpen={openSections.has('fsrs')}
          onToggle={() => toggleSection('fsrs')}
        >
          <SliderInput
            label="Default Stability"
            value={config.fsrs.defaultStability}
            min={0.1}
            max={10}
            step={0.1}
            unit="days"
            description="Initial stability for new memories"
            onChange={(v) => updateConfig('fsrs', 'defaultStability', v)}
          />
          <SliderInput
            label="Retention Target"
            value={config.fsrs.retentionTarget}
            min={0.5}
            max={0.99}
            step={0.01}
            description="Target recall probability"
            onChange={(v) => updateConfig('fsrs', 'retentionTarget', v)}
          />
          <SliderInput
            label="Decay Factor"
            value={config.fsrs.decayFactor}
            min={0.1}
            max={1.0}
            step={0.05}
            description="Power-law decay exponent"
            onChange={(v) => updateConfig('fsrs', 'decayFactor', v)}
          />
          <SliderInput
            label="Recency Decay"
            value={config.fsrs.recencyDecay}
            min={0.01}
            max={1.0}
            step={0.01}
            description="Exponential decay for recency scoring"
            onChange={(v) => updateConfig('fsrs', 'recencyDecay', v)}
          />
        </ConfigSection>

        {/* ACT-R Activation */}
        <ConfigSection
          title="ACT-R Activation"
          icon="fa-brain"
          isOpen={openSections.has('actr')}
          onToggle={() => toggleSection('actr')}
        >
          <SliderInput
            label="Base Decay"
            value={config.actr.decay}
            min={0.1}
            max={1.0}
            step={0.05}
            description="Logarithmic decay rate"
            onChange={(v) => updateConfig('actr', 'decay', v)}
          />
          <SliderInput
            label="Noise"
            value={config.actr.noise}
            min={0}
            max={0.5}
            step={0.01}
            description="Gaussian noise for stochastic retrieval"
            onChange={(v) => updateConfig('actr', 'noise', v)}
          />
          <SliderInput
            label="Activation Threshold"
            value={config.actr.threshold}
            min={0}
            max={1.0}
            step={0.05}
            description="Minimum activation for retrieval"
            onChange={(v) => updateConfig('actr', 'threshold', v)}
          />
          <SliderInput
            label="Spreading Weight"
            value={config.actr.spreadingWeight}
            min={0}
            max={2.0}
            step={0.1}
            description="Weight for spreading activation"
            onChange={(v) => updateConfig('actr', 'spreadingWeight', v)}
          />
        </ConfigSection>

        {/* Hebbian Learning */}
        <ConfigSection
          title="Hebbian Learning"
          icon="fa-link"
          isOpen={openSections.has('hebbian')}
          onToggle={() => toggleSection('hebbian')}
        >
          <SliderInput
            label="Learning Rate"
            value={config.hebbian.learningRate}
            min={0.01}
            max={0.5}
            step={0.01}
            description="Weight update magnitude (eta)"
            onChange={(v) => updateConfig('hebbian', 'learningRate', v)}
          />
          <SliderInput
            label="Initial Weight"
            value={config.hebbian.initialWeight}
            min={0.01}
            max={0.5}
            step={0.01}
            description="Starting weight for new connections"
            onChange={(v) => updateConfig('hebbian', 'initialWeight', v)}
          />
          <SliderInput
            label="Min Weight"
            value={config.hebbian.minWeight}
            min={0.001}
            max={0.1}
            step={0.001}
            description="Minimum weight before pruning"
            onChange={(v) => updateConfig('hebbian', 'minWeight', v)}
          />
          <SliderInput
            label="Decay Rate"
            value={config.hebbian.decayRate}
            min={0.001}
            max={0.1}
            step={0.005}
            description="Weight decay per time unit"
            onChange={(v) => updateConfig('hebbian', 'decayRate', v)}
          />
          <SliderInput
            label="Stale Days"
            value={config.hebbian.staleDays}
            min={1}
            max={365}
            step={1}
            unit="days"
            description="Days without access before decay"
            onChange={(v) => updateConfig('hebbian', 'staleDays', v)}
          />
        </ConfigSection>

        {/* Neuromodulation */}
        <ConfigSection
          title="Neuromodulation"
          icon="fa-flask"
          isOpen={openSections.has('neuromod')}
          onToggle={() => toggleSection('neuromod')}
        >
          <SliderInput
            label="Dopamine Baseline"
            value={config.neuromod.dopamineBaseline}
            min={0}
            max={1.0}
            step={0.05}
            description="Tonic DA level"
            onChange={(v) => updateConfig('neuromod', 'dopamineBaseline', v)}
          />
          <SliderInput
            label="Norepinephrine Gain"
            value={config.neuromod.norepinephrineGain}
            min={1.0}
            max={3.0}
            step={0.1}
            description="NE signal amplification"
            onChange={(v) => updateConfig('neuromod', 'norepinephrineGain', v)}
          />
          <SliderInput
            label="Serotonin Discount"
            value={config.neuromod.serotoninDiscount}
            min={0}
            max={0.5}
            step={0.01}
            description="5-HT temporal discount rate"
            onChange={(v) => updateConfig('neuromod', 'serotoninDiscount', v)}
          />
          <SliderInput
            label="ACh Threshold"
            value={config.neuromod.acetylcholineThreshold}
            min={0.2}
            max={0.8}
            step={0.05}
            description="Encoding/retrieval mode switch"
            onChange={(v) => updateConfig('neuromod', 'acetylcholineThreshold', v)}
          />
          <SliderInput
            label="GABA Inhibition"
            value={config.neuromod.gabaInhibition}
            min={0}
            max={1.0}
            step={0.05}
            description="Inhibitory control strength"
            onChange={(v) => updateConfig('neuromod', 'gabaInhibition', v)}
          />
        </ConfigSection>

        {/* Pattern Separation */}
        <ConfigSection
          title="Pattern Separation"
          icon="fa-random"
          isOpen={openSections.has('patternSep')}
          onToggle={() => toggleSection('patternSep')}
        >
          <SliderInput
            label="Target Sparsity"
            value={config.patternSep.targetSparsity}
            min={0.01}
            max={0.2}
            step={0.01}
            description="DG sparsity (biological ~4%)"
            onChange={(v) => updateConfig('patternSep', 'targetSparsity', v)}
          />
          <SliderInput
            label="Max Neighbors"
            value={config.patternSep.maxNeighbors}
            min={1}
            max={200}
            step={1}
            description="Maximum neighbors per node in spreading activation"
            onChange={(v) => updateConfig('patternSep', 'maxNeighbors', v)}
          />
          <SliderInput
            label="Max Nodes"
            value={config.patternSep.maxNodes}
            min={10}
            max={10000}
            step={10}
            description="Maximum nodes in spreading activation"
            onChange={(v) => updateConfig('patternSep', 'maxNodes', v)}
          />
        </ConfigSection>

        {/* Memory Gate */}
        <ConfigSection
          title="Memory Gate"
          icon="fa-filter"
          isOpen={openSections.has('memoryGate')}
          onToggle={() => toggleSection('memoryGate')}
        >
          <SliderInput
            label="Base Threshold"
            value={config.memoryGate.baseThreshold}
            min={0.1}
            max={0.9}
            step={0.05}
            description="Minimum storage probability"
            onChange={(v) => updateConfig('memoryGate', 'baseThreshold', v)}
          />
          <SliderInput
            label="Novelty Weight"
            value={config.memoryGate.noveltyWeight}
            min={0}
            max={1.0}
            step={0.05}
            description="Importance of novelty"
            onChange={(v) => updateConfig('memoryGate', 'noveltyWeight', v)}
          />
          <SliderInput
            label="Importance Weight"
            value={config.memoryGate.importanceWeight}
            min={0}
            max={1.0}
            step={0.05}
            description="Weight for valence signal"
            onChange={(v) => updateConfig('memoryGate', 'importanceWeight', v)}
          />
          <SliderInput
            label="Context Weight"
            value={config.memoryGate.contextWeight}
            min={0}
            max={1.0}
            step={0.05}
            description="Relevance to current context"
            onChange={(v) => updateConfig('memoryGate', 'contextWeight', v)}
          />
        </ConfigSection>

        {/* Consolidation */}
        <ConfigSection
          title="Consolidation"
          icon="fa-compress"
          isOpen={openSections.has('consolidation')}
          onToggle={() => toggleSection('consolidation')}
        >
          <SliderInput
            label="Min Similarity"
            value={config.consolidation.minSimilarity}
            min={0.5}
            max={0.99}
            step={0.01}
            description="Threshold for deduplication"
            onChange={(v) => updateConfig('consolidation', 'minSimilarity', v)}
          />
          <SliderInput
            label="Min Occurrences"
            value={config.consolidation.minOccurrences}
            min={2}
            max={10}
            step={1}
            description="Cluster size for abstraction"
            onChange={(v) => updateConfig('consolidation', 'minOccurrences', v)}
          />
          <SliderInput
            label="Skill Similarity"
            value={config.consolidation.skillSimilarity}
            min={0.5}
            max={0.99}
            step={0.01}
            description="Procedural merge threshold"
            onChange={(v) => updateConfig('consolidation', 'skillSimilarity', v)}
          />
          <SliderInput
            label="Cluster Size"
            value={config.consolidation.clusterSize}
            min={2}
            max={100}
            step={1}
            description="Minimum cluster size for HDBSCAN"
            onChange={(v) => updateConfig('consolidation', 'clusterSize', v)}
          />
        </ConfigSection>

        {/* Episodic Retrieval Weights */}
        <ConfigSection
          title="Episodic Retrieval Weights"
          icon="fa-history"
          isOpen={openSections.has('episodicWeights')}
          onToggle={() => toggleSection('episodicWeights')}
        >
          <SliderInput
            label="Semantic Weight"
            value={config.episodicWeights.semanticWeight}
            min={0}
            max={1.0}
            step={0.05}
            description="Vector similarity importance"
            onChange={(v) => updateConfig('episodicWeights', 'semanticWeight', v)}
          />
          <SliderInput
            label="Recency Weight"
            value={config.episodicWeights.recencyWeight}
            min={0}
            max={1.0}
            step={0.05}
            description="Time-based decay factor"
            onChange={(v) => updateConfig('episodicWeights', 'recencyWeight', v)}
          />
          <SliderInput
            label="Outcome Weight"
            value={config.episodicWeights.outcomeWeight}
            min={0}
            max={1.0}
            step={0.05}
            description="Success/failure preference"
            onChange={(v) => updateConfig('episodicWeights', 'outcomeWeight', v)}
          />
          <SliderInput
            label="Importance Weight"
            value={config.episodicWeights.importanceWeight}
            min={0}
            max={1.0}
            step={0.05}
            description="Emotional valence factor"
            onChange={(v) => updateConfig('episodicWeights', 'importanceWeight', v)}
          />
          <p className="weight-sum-note">
            Weights must sum to 1.0 (current:{' '}
            {(
              config.episodicWeights.semanticWeight +
              config.episodicWeights.recencyWeight +
              config.episodicWeights.outcomeWeight +
              config.episodicWeights.importanceWeight
            ).toFixed(2)}
            )
          </p>
        </ConfigSection>

        {/* Semantic Retrieval Weights */}
        <ConfigSection
          title="Semantic Retrieval Weights"
          icon="fa-project-diagram"
          isOpen={openSections.has('semanticWeights')}
          onToggle={() => toggleSection('semanticWeights')}
        >
          <SliderInput
            label="Similarity Weight"
            value={config.semanticWeights.similarityWeight}
            min={0}
            max={1.0}
            step={0.05}
            description="Embedding similarity importance"
            onChange={(v) => updateConfig('semanticWeights', 'similarityWeight', v)}
          />
          <SliderInput
            label="Activation Weight"
            value={config.semanticWeights.activationWeight}
            min={0}
            max={1.0}
            step={0.05}
            description="ACT-R activation importance"
            onChange={(v) => updateConfig('semanticWeights', 'activationWeight', v)}
          />
          <SliderInput
            label="Retrievability Weight"
            value={config.semanticWeights.retrievabilityWeight}
            min={0}
            max={1.0}
            step={0.05}
            description="FSRS retrievability importance"
            onChange={(v) => updateConfig('semanticWeights', 'retrievabilityWeight', v)}
          />
          <p className="weight-sum-note">
            Weights must sum to 1.0 (current:{' '}
            {(
              config.semanticWeights.similarityWeight +
              config.semanticWeights.activationWeight +
              config.semanticWeights.retrievabilityWeight
            ).toFixed(2)}
            )
          </p>
        </ConfigSection>

        {/* Procedural Retrieval Weights */}
        <ConfigSection
          title="Procedural Retrieval Weights"
          icon="fa-cogs"
          isOpen={openSections.has('proceduralWeights')}
          onToggle={() => toggleSection('proceduralWeights')}
        >
          <SliderInput
            label="Similarity Weight"
            value={config.proceduralWeights.similarityWeight}
            min={0}
            max={1.0}
            step={0.05}
            description="Task similarity importance"
            onChange={(v) => updateConfig('proceduralWeights', 'similarityWeight', v)}
          />
          <SliderInput
            label="Success Weight"
            value={config.proceduralWeights.successWeight}
            min={0}
            max={1.0}
            step={0.05}
            description="Success rate importance"
            onChange={(v) => updateConfig('proceduralWeights', 'successWeight', v)}
          />
          <SliderInput
            label="Experience Weight"
            value={config.proceduralWeights.experienceWeight}
            min={0}
            max={1.0}
            step={0.05}
            description="Execution count importance"
            onChange={(v) => updateConfig('proceduralWeights', 'experienceWeight', v)}
          />
          <p className="weight-sum-note">
            Weights must sum to 1.0 (current:{' '}
            {(
              config.proceduralWeights.similarityWeight +
              config.proceduralWeights.successWeight +
              config.proceduralWeights.experienceWeight
            ).toFixed(2)}
            )
          </p>
        </ConfigSection>

        {/* Bioinspired Section Divider */}
        <div className="section-divider">
          <span>Bioinspired Architecture (CompBio)</span>
        </div>

        {/* Bioinspired Master Toggle */}
        <ConfigSection
          title="Bioinspired Components"
          icon="fa-dna"
          isOpen={openSections.has('bioEnabled')}
          onToggle={() => toggleSection('bioEnabled')}
        >
          <ToggleInput
            label="Enable Bioinspired Components"
            value={config.bioinspired.enabled}
            description="Master toggle for all biologically-inspired neural memory features"
            onChange={toggleBioEnabled}
          />
          {config.bioinspired.enabled && (
            <div className="bio-status">
              <i className="fa fa-check-circle" />
              <span>CompBio features active: Dendritic processing, sparse encoding, attractor dynamics, FES</span>
            </div>
          )}
        </ConfigSection>

        {/* Dendritic Processing */}
        <ConfigSection
          title="Dendritic Processing"
          icon="fa-code-branch"
          isOpen={openSections.has('dendritic')}
          onToggle={() => toggleSection('dendritic')}
        >
          <SliderInput
            label="Hidden Dimension"
            value={config.bioinspired.dendritic.hiddenDim}
            min={128}
            max={1024}
            step={64}
            description="Dendritic/somatic hidden layer size"
            onChange={(v) => updateBioConfig('dendritic', 'hiddenDim', v)}
          />
          <SliderInput
            label="Context Dimension"
            value={config.bioinspired.dendritic.contextDim}
            min={128}
            max={1024}
            step={64}
            description="Working memory context vector size"
            onChange={(v) => updateBioConfig('dendritic', 'contextDim', v)}
          />
          <SliderInput
            label="Coupling Strength"
            value={config.bioinspired.dendritic.couplingStrength}
            min={0.1}
            max={1.0}
            step={0.1}
            description="Dendrite-to-soma coupling (g_ds)"
            onChange={(v) => updateBioConfig('dendritic', 'couplingStrength', v)}
          />
          <SliderInput
            label="Tau Dendrite"
            value={config.bioinspired.dendritic.tauDendrite}
            min={1}
            max={50}
            step={1}
            unit="ms"
            description="Dendritic membrane time constant"
            onChange={(v) => updateBioConfig('dendritic', 'tauDendrite', v)}
          />
          <SliderInput
            label="Tau Soma"
            value={config.bioinspired.dendritic.tauSoma}
            min={1}
            max={50}
            step={1}
            unit="ms"
            description="Somatic membrane time constant"
            onChange={(v) => updateBioConfig('dendritic', 'tauSoma', v)}
          />
        </ConfigSection>

        {/* Sparse Encoding */}
        <ConfigSection
          title="Sparse Encoding"
          icon="fa-th"
          isOpen={openSections.has('sparseEncoder')}
          onToggle={() => toggleSection('sparseEncoder')}
        >
          <SliderInput
            label="Hidden Dimension"
            value={config.bioinspired.sparseEncoder.hiddenDim}
            min={1024}
            max={16384}
            step={1024}
            description="Overcomplete sparse code dimension (8-16x expansion)"
            onChange={(v) => updateBioConfig('sparseEncoder', 'hiddenDim', v)}
          />
          <SliderInput
            label="Target Sparsity"
            value={config.bioinspired.sparseEncoder.sparsity}
            min={0.01}
            max={0.1}
            step={0.01}
            description="k-WTA sparsity (biological: 1-5%)"
            onChange={(v) => updateBioConfig('sparseEncoder', 'sparsity', v)}
          />
          <ToggleInput
            label="Use k-WTA"
            value={config.bioinspired.sparseEncoder.useKwta}
            description="Enable k-Winners-Take-All activation"
            onChange={(v) => updateBioConfig('sparseEncoder', 'useKwta', v)}
          />
          <SliderInput
            label="Lateral Inhibition"
            value={config.bioinspired.sparseEncoder.lateralInhibition}
            min={0}
            max={1.0}
            step={0.05}
            description="Competitive inhibition strength"
            onChange={(v) => updateBioConfig('sparseEncoder', 'lateralInhibition', v)}
          />
        </ConfigSection>

        {/* Attractor Network */}
        <ConfigSection
          title="Attractor Network"
          icon="fa-magnet"
          isOpen={openSections.has('attractor')}
          onToggle={() => toggleSection('attractor')}
        >
          <SliderInput
            label="Settling Steps"
            value={config.bioinspired.attractor.settlingSteps}
            min={1}
            max={30}
            step={1}
            description="Iterations for attractor convergence"
            onChange={(v) => updateBioConfig('attractor', 'settlingSteps', v)}
          />
          <SliderInput
            label="Noise Std"
            value={config.bioinspired.attractor.noiseStd}
            min={0}
            max={0.1}
            step={0.005}
            description="Stochastic dynamics noise"
            onChange={(v) => updateBioConfig('attractor', 'noiseStd', v)}
          />
          <SliderInput
            label="Adaptation Tau"
            value={config.bioinspired.attractor.adaptationTau}
            min={50}
            max={500}
            step={25}
            unit="ms"
            description="Synaptic depression recovery time"
            onChange={(v) => updateBioConfig('attractor', 'adaptationTau', v)}
          />
          <SliderInput
            label="Step Size"
            value={config.bioinspired.attractor.stepSize}
            min={0.01}
            max={0.5}
            step={0.01}
            description="Integration step size (dt)"
            onChange={(v) => updateBioConfig('attractor', 'stepSize', v)}
          />
        </ConfigSection>

        {/* Fast Episodic Store */}
        <ConfigSection
          title="Fast Episodic Store"
          icon="fa-bolt"
          isOpen={openSections.has('fastEpisodic')}
          onToggle={() => toggleSection('fastEpisodic')}
        >
          <SliderInput
            label="Capacity"
            value={config.bioinspired.fastEpisodic.capacity}
            min={1000}
            max={100000}
            step={1000}
            description="Maximum memories before eviction"
            onChange={(v) => updateBioConfig('fastEpisodic', 'capacity', v)}
          />
          <SliderInput
            label="Learning Rate"
            value={config.bioinspired.fastEpisodic.learningRate}
            min={0.01}
            max={0.5}
            step={0.01}
            description="FES learning rate (100x standard)"
            onChange={(v) => updateBioConfig('fastEpisodic', 'learningRate', v)}
          />
          <SliderInput
            label="Consolidation Threshold"
            value={config.bioinspired.fastEpisodic.consolidationThreshold}
            min={0.3}
            max={0.95}
            step={0.05}
            description="Salience threshold for transfer to episodic"
            onChange={(v) => updateBioConfig('fastEpisodic', 'consolidationThreshold', v)}
          />
          <p className="learning-rate-note">
            LR Ratio (FES/Standard): {(config.bioinspired.fastEpisodic.learningRate / 0.001).toFixed(0)}x
            {config.bioinspired.fastEpisodic.learningRate / 0.001 >= 50 &&
             config.bioinspired.fastEpisodic.learningRate / 0.001 <= 200 ? (
              <span className="valid"> (Valid biological range: 50-200x)</span>
            ) : (
              <span className="invalid"> (Outside biological range: 50-200x)</span>
            )}
          </p>
        </ConfigSection>

        {/* Enhanced Neuromodulator Gains */}
        <ConfigSection
          title="Neuromodulator Gains (CompBio)"
          icon="fa-wave-square"
          isOpen={openSections.has('neuromodGains')}
          onToggle={() => toggleSection('neuromodGains')}
        >
          <SliderInput
            label="Rho DA (Dopamine Gain)"
            value={config.bioinspired.neuromodGains.rhoDa}
            min={0.5}
            max={5.0}
            step={0.25}
            description="TD error scaling factor"
            onChange={(v) => updateBioConfig('neuromodGains', 'rhoDa', v)}
          />
          <SliderInput
            label="Rho NE (Norepinephrine Gain)"
            value={config.bioinspired.neuromodGains.rhoNe}
            min={0.5}
            max={3.0}
            step={0.25}
            description="Novelty signal amplification"
            onChange={(v) => updateBioConfig('neuromodGains', 'rhoNe', v)}
          />
          <SliderInput
            label="Rho ACh Fast (FES)"
            value={config.bioinspired.neuromodGains.rhoAchFast}
            min={1.0}
            max={5.0}
            step={0.25}
            description="ACh gain for Fast Episodic Store"
            onChange={(v) => updateBioConfig('neuromodGains', 'rhoAchFast', v)}
          />
          <SliderInput
            label="Rho ACh Slow (Standard)"
            value={config.bioinspired.neuromodGains.rhoAchSlow}
            min={0.1}
            max={1.0}
            step={0.1}
            description="ACh gain for standard memory"
            onChange={(v) => updateBioConfig('neuromodGains', 'rhoAchSlow', v)}
          />
          <SliderInput
            label="Alpha NE"
            value={config.bioinspired.neuromodGains.alphaNe}
            min={0.1}
            max={0.9}
            step={0.1}
            description="NE running average coefficient"
            onChange={(v) => updateBioConfig('neuromodGains', 'alphaNe', v)}
          />
        </ConfigSection>

        {/* Eligibility Traces */}
        <ConfigSection
          title="Eligibility Traces"
          icon="fa-clock-rotate-left"
          isOpen={openSections.has('eligibility')}
          onToggle={() => toggleSection('eligibility')}
        >
          <SliderInput
            label="Decay (Lambda)"
            value={config.bioinspired.eligibility.decay}
            min={0.8}
            max={0.99}
            step={0.01}
            description="Trace decay factor (TD-lambda)"
            onChange={(v) => updateBioConfig('eligibility', 'decay', v)}
          />
          <SliderInput
            label="Tau Trace"
            value={config.bioinspired.eligibility.tauTrace}
            min={5}
            max={100}
            step={5}
            unit="ms"
            description="STDP window (biological: 10-20ms)"
            onChange={(v) => updateBioConfig('eligibility', 'tauTrace', v)}
          />
        </ConfigSection>
      </div>
    </div>
  );
};

export default ConfigPanel;
