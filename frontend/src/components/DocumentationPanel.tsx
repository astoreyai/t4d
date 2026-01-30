/**
 * Interactive Documentation Panel
 * Browse and search system documentation with live examples
 */

import React, { useState } from 'react';
import './DocumentationPanel.scss';

interface DocSection {
  id: string;
  title: string;
  icon: string;
  content: React.ReactNode;
}

export const DocumentationPanel: React.FC = () => {
  const [activeSection, setActiveSection] = useState('overview');
  const [searchQuery, setSearchQuery] = useState('');

  const sections: DocSection[] = [
    {
      id: 'overview',
      title: 'Overview',
      icon: 'fa-home',
      content: <OverviewDoc />,
    },
    {
      id: 'memory-types',
      title: 'Memory Types',
      icon: 'fa-database',
      content: <MemoryTypesDoc />,
    },
    {
      id: 'neuromodulation',
      title: 'Neuromodulation',
      icon: 'fa-flask',
      content: <NeuromodulationDoc />,
    },
    {
      id: 'algorithms',
      title: 'Algorithms',
      icon: 'fa-cogs',
      content: <AlgorithmsDoc />,
    },
    {
      id: 'api',
      title: 'API Reference',
      icon: 'fa-code',
      content: <ApiDoc />,
    },
  ];

  const filteredSections = sections.filter(
    (s) =>
      s.title.toLowerCase().includes(searchQuery.toLowerCase()) ||
      s.id.toLowerCase().includes(searchQuery.toLowerCase())
  );

  return (
    <div className="documentation-panel">
      {/* Sidebar */}
      <nav className="doc-sidebar">
        <div className="search-box">
          <i className="fa fa-search" />
          <input
            type="text"
            placeholder="Search docs..."
            value={searchQuery}
            onChange={(e) => setSearchQuery(e.target.value)}
          />
        </div>
        <ul className="nav-list">
          {filteredSections.map((section) => (
            <li key={section.id}>
              <button
                className={activeSection === section.id ? 'active' : ''}
                onClick={() => setActiveSection(section.id)}
              >
                <i className={`fa ${section.icon}`} />
                {section.title}
              </button>
            </li>
          ))}
        </ul>
      </nav>

      {/* Content */}
      <main className="doc-content">
        {sections.find((s) => s.id === activeSection)?.content}
      </main>
    </div>
  );
};

const OverviewDoc: React.FC = () => (
  <article>
    <h1>World Weaver Documentation</h1>
    <p className="lead">
      A biologically-inspired neural memory system implementing tripartite memory architecture
      with cognitive neuroscience principles.
    </p>

    <section>
      <h2>Architecture</h2>
      <div className="diagram">
        <pre>{`
┌─────────────────────────────────────────────────────────┐
│                    INTERFACE LAYER                       │
│  MCP Server │ REST API │ Claude Code │ Kymera Voice     │
└─────────────────────────┬───────────────────────────────┘
                          │
┌─────────────────────────┴───────────────────────────────┐
│                     MEMORY LAYER                         │
│  Episodic │ Semantic │ Procedural │ Working Memory      │
└─────────────────────────┬───────────────────────────────┘
                          │
┌─────────────────────────┴───────────────────────────────┐
│                    LEARNING LAYER                        │
│  Neuromodulators │ Hebbian │ FSRS │ ACT-R │ Plasticity │
└─────────────────────────┬───────────────────────────────┘
                          │
┌─────────────────────────┴───────────────────────────────┐
│                   PERSISTENCE LAYER                      │
│           Qdrant (Vectors) │ Neo4j (Graph)              │
└─────────────────────────────────────────────────────────┘
        `}</pre>
      </div>
    </section>

    <section>
      <h2>Key Concepts</h2>
      <div className="concept-grid">
        <div className="concept">
          <h3>Episodic Memory</h3>
          <p>Autobiographical experiences with temporal context and FSRS decay.</p>
        </div>
        <div className="concept">
          <h3>Semantic Memory</h3>
          <p>Abstracted knowledge entities with ACT-R activation and Hebbian learning.</p>
        </div>
        <div className="concept">
          <h3>Procedural Memory</h3>
          <p>Learned skills with success tracking and version control.</p>
        </div>
        <div className="concept">
          <h3>Neuromodulation</h3>
          <p>DA/NE/5-HT/ACh orchestra controlling learning and retrieval.</p>
        </div>
      </div>
    </section>
  </article>
);

const MemoryTypesDoc: React.FC = () => (
  <article>
    <h1>Memory Types</h1>

    <section>
      <h2>Episodic Memory</h2>
      <p>Stores autobiographical experiences with rich temporal-spatial context.</p>

      <h3>Key Properties</h3>
      <ul>
        <li><strong>stability</strong>: FSRS stability parameter (days)</li>
        <li><strong>retrievability</strong>: Current recall probability</li>
        <li><strong>emotional_valence</strong>: Importance (0-1)</li>
        <li><strong>outcome</strong>: success | failure | partial | neutral</li>
      </ul>

      <h3>FSRS Decay Formula</h3>
      <div className="formula">
        <code>R(t) = (1 + t / (9 * S))^(-0.5)</code>
      </div>
    </section>

    <section>
      <h2>Semantic Memory</h2>
      <p>Stores abstracted knowledge as entities with relationships.</p>

      <h3>Entity Types</h3>
      <ul>
        <li><code>CONCEPT</code> - Abstract ideas</li>
        <li><code>PERSON</code> - Individual actors</li>
        <li><code>PROJECT</code> - Named projects</li>
        <li><code>TOOL</code> - Software/hardware tools</li>
        <li><code>TECHNIQUE</code> - Methods/approaches</li>
        <li><code>FACT</code> - Factual statements</li>
      </ul>

      <h3>ACT-R Activation</h3>
      <div className="formula">
        <code>A = B + sum(W * S)</code>
        <p>Where B = base-level, W = weight, S = spreading activation</p>
      </div>
    </section>

    <section>
      <h2>Procedural Memory</h2>
      <p>Stores learned skills and procedures with execution tracking.</p>

      <h3>Domains</h3>
      <ul>
        <li><code>CODING</code> - Programming tasks</li>
        <li><code>RESEARCH</code> - Academic work</li>
        <li><code>TRADING</code> - Financial analysis</li>
        <li><code>DEVOPS</code> - Infrastructure</li>
        <li><code>WRITING</code> - Content creation</li>
      </ul>
    </section>
  </article>
);

const NeuromodulationDoc: React.FC = () => (
  <article>
    <h1>Neuromodulation System</h1>

    <section>
      <h2>Dopamine (DA)</h2>
      <p>Reward prediction error and value learning.</p>
      <div className="formula">
        <code>RPE = R - V(s)</code>
        <p>Positive RPE strengthens, negative weakens associations.</p>
      </div>
    </section>

    <section>
      <h2>Norepinephrine (NE)</h2>
      <p>Arousal and attention modulation.</p>
      <div className="formula">
        <code>gain = 1 + NE * multiplier</code>
        <p>High NE increases signal sensitivity.</p>
      </div>
    </section>

    <section>
      <h2>Serotonin (5-HT)</h2>
      <p>Temporal discounting and patience.</p>
      <div className="formula">
        <code>discount = e^(-5HT * time)</code>
        <p>Higher 5-HT = more patient decisions.</p>
      </div>
    </section>

    <section>
      <h2>Acetylcholine (ACh)</h2>
      <p>Encoding vs retrieval mode switching.</p>
      <div className="formula">
        <code>{'mode = ENCODING if ACh > 0.5 else RETRIEVAL'}</code>
      </div>
    </section>

    <section>
      <h2>GABA</h2>
      <p>Inhibitory control and pattern separation.</p>
      <div className="formula">
        <code>inhibition = GABA * activity</code>
      </div>
    </section>
  </article>
);

const AlgorithmsDoc: React.FC = () => (
  <article>
    <h1>Algorithms</h1>

    <section>
      <h2>Hebbian Learning</h2>
      <div className="formula">
        <code>Δw = η * pre * post * DA</code>
      </div>
      <p>Connections strengthen when pre and post fire together with reward.</p>
    </section>

    <section>
      <h2>Pattern Separation (DG)</h2>
      <p>Orthogonalizes similar inputs to reduce interference.</p>
      <ul>
        <li>Target sparsity: ~4% (biological)</li>
        <li>Uses k-WTA (k-winners-take-all)</li>
        <li>Decorrelates overlapping patterns</li>
      </ul>
    </section>

    <section>
      <h2>Pattern Completion (CA3)</h2>
      <p>Completes partial cues using attractor dynamics.</p>
      <ul>
        <li>Hopfield-like attractor network</li>
        <li>Retrieves full pattern from partial input</li>
        <li>Converges to nearest stored pattern</li>
      </ul>
    </section>

    <section>
      <h2>Sleep Consolidation</h2>
      <ul>
        <li><strong>NREM</strong>: Hippocampal replay, episodic consolidation</li>
        <li><strong>REM</strong>: Cortical integration, semantic abstraction</li>
      </ul>
    </section>
  </article>
);

const ApiDoc: React.FC = () => (
  <article>
    <h1>API Reference</h1>

    <section>
      <h2>Base URL</h2>
      <code>http://localhost:8765/api/v1</code>
    </section>

    <section>
      <h2>Episodes</h2>
      <div className="endpoint">
        <span className="method post">POST</span>
        <code>/episodes</code>
        <p>Create new episodic memory</p>
      </div>
      <div className="endpoint">
        <span className="method get">GET</span>
        <code>/episodes/{'{id}'}</code>
        <p>Get episode by ID</p>
      </div>
      <div className="endpoint">
        <span className="method post">POST</span>
        <code>/episodes/recall</code>
        <p>Semantic search for episodes</p>
      </div>
    </section>

    <section>
      <h2>Entities</h2>
      <div className="endpoint">
        <span className="method post">POST</span>
        <code>/entities</code>
        <p>Create semantic entity</p>
      </div>
      <div className="endpoint">
        <span className="method post">POST</span>
        <code>/entities/recall</code>
        <p>Search entities</p>
      </div>
      <div className="endpoint">
        <span className="method post">POST</span>
        <code>/entities/spread-activation</code>
        <p>Spreading activation from entity</p>
      </div>
    </section>

    <section>
      <h2>Skills</h2>
      <div className="endpoint">
        <span className="method post">POST</span>
        <code>/skills</code>
        <p>Create procedural skill</p>
      </div>
      <div className="endpoint">
        <span className="method get">GET</span>
        <code>/skills/how-to/{'{query}'}</code>
        <p>Natural language skill query</p>
      </div>
    </section>

    <section>
      <h2>Visualization</h2>
      <div className="endpoint">
        <span className="method get">GET</span>
        <code>/viz/graph</code>
        <p>Full memory graph with positions</p>
      </div>
      <div className="endpoint">
        <span className="method get">GET</span>
        <code>/viz/bio/neuromodulators</code>
        <p>Current neuromodulator state</p>
      </div>
    </section>
  </article>
);

export default DocumentationPanel;
