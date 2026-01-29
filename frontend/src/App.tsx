import React, { useState } from 'react';
import { MemoryGraphPage } from './components/MemoryGraphPage';
import { MemoryListPanel } from './components/MemoryListPanel';
import { BioDashboard } from './components/BioDashboard';
import { TracingDashboard } from './components/TracingDashboard';
import { ThreeFactorDashboard } from './components/ThreeFactorDashboard';
import { ReconsolidationTimeline } from './components/ReconsolidationTimeline';
import { DocumentationPanel } from './components/DocumentationPanel';
import { ConfigPanel } from './components/ConfigPanel';
import { DiagramGraphPage } from './components/DiagramGraphPage';
import './styles/app.scss';

type TabId = 'graph' | 'memories' | 'learning' | 'bio' | 'tracing' | 'architecture' | 'config' | 'docs';

interface Tab {
  id: TabId;
  label: string;
  icon: string;
}

const tabs: Tab[] = [
  { id: 'graph', label: 'Memory Graph', icon: 'fa-project-diagram' },
  { id: 'memories', label: 'Memories', icon: 'fa-database' },
  { id: 'learning', label: 'Learning', icon: 'fa-graduation-cap' },
  { id: 'bio', label: 'Bio Systems', icon: 'fa-brain' },
  { id: 'tracing', label: 'Tracing', icon: 'fa-route' },
  { id: 'architecture', label: 'Architecture', icon: 'fa-sitemap' },
  { id: 'config', label: 'Config', icon: 'fa-sliders-h' },
  { id: 'docs', label: 'Docs', icon: 'fa-book' },
];

const App: React.FC = () => {
  const [activeTab, setActiveTab] = useState<TabId>('graph');

  return (
    <div className="ww-app">
      {/* Header */}
      <header className="ww-header">
        <div className="logo">
          <i className="fa fa-brain" />
          <span>World Weaver</span>
        </div>
        <nav className="tabs">
          {tabs.map((tab) => (
            <button
              key={tab.id}
              className={`tab ${activeTab === tab.id ? 'active' : ''}`}
              onClick={() => setActiveTab(tab.id)}
            >
              <i className={`fa ${tab.icon}`} />
              <span>{tab.label}</span>
            </button>
          ))}
        </nav>
        <div className="status">
          <span className="status-dot online" />
          <span>Connected</span>
        </div>
      </header>

      {/* Main Content */}
      <main className="ww-main">
        {activeTab === 'graph' && <MemoryGraphPage />}
        {activeTab === 'memories' && <MemoryListPanel />}
        {activeTab === 'learning' && (
          <div className="learning-layout">
            <div className="learning-main">
              <ThreeFactorDashboard />
            </div>
            <div className="learning-sidebar">
              <ReconsolidationTimeline />
            </div>
          </div>
        )}
        {activeTab === 'bio' && <BioDashboard />}
        {activeTab === 'tracing' && <TracingDashboard />}
        {activeTab === 'architecture' && <DiagramGraphPage />}
        {activeTab === 'config' && <ConfigPanel />}
        {activeTab === 'docs' && <DocumentationPanel />}
      </main>
    </div>
  );
};

export default App;
