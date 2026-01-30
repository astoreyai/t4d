/**
 * Memory List Panel - CRUD Operations for Episodic and Semantic Memories
 *
 * Provides list view with search, delete, and memory surgery controls.
 */

import React, { useState, useEffect, useCallback } from 'react';
import './MemoryListPanel.scss';

// ============================================================================
// Types
// ============================================================================

type MemoryTab = 'episodes' | 'entities';

interface EpisodeContext {
  project?: string;
  file?: string;
  tool?: string;
}

interface Episode {
  id: string;
  session_id: string;
  content: string;
  timestamp: string;
  outcome: string;
  emotional_valence: number;
  context: EpisodeContext;
  access_count: number;
  stability: number;
  retrievability: number | null;
}

interface Entity {
  id: string;
  name: string;
  entity_type: string;
  summary: string;
  details: string | null;
  source: string | null;
  stability: number;
  access_count: number;
  created_at: string;
  valid_from: string;
  valid_to: string | null;
}

interface EpisodeListResponse {
  episodes: Episode[];
  total: number;
  page: number;
  page_size: number;
}

interface EntityListResponse {
  entities: Entity[];
  total: number;
}

// ============================================================================
// Memory Surgery Modal
// ============================================================================

interface SurgeryModalProps {
  item: Episode | Entity | null;
  type: 'episode' | 'entity';
  onClose: () => void;
  onSave: (id: string, updates: Record<string, unknown>) => Promise<void>;
}

const SurgeryModal: React.FC<SurgeryModalProps> = ({ item, type, onClose, onSave }) => {
  const [saving, setSaving] = useState(false);
  const [importance, setImportance] = useState(0.5);
  const [content, setContent] = useState('');
  const [summary, setSummary] = useState('');

  useEffect(() => {
    if (item) {
      if (type === 'episode') {
        const ep = item as Episode;
        setImportance(ep.emotional_valence);
        setContent(ep.content);
      } else {
        const ent = item as Entity;
        setSummary(ent.summary);
      }
    }
  }, [item, type]);

  if (!item) return null;

  const handleSave = async () => {
    setSaving(true);
    try {
      if (type === 'episode') {
        await onSave(item.id, {
          content,
          emotional_valence: importance,
        });
      } else {
        await onSave(item.id, {
          summary,
        });
      }
      onClose();
    } catch (err) {
      console.error('Failed to save:', err);
    } finally {
      setSaving(false);
    }
  };

  return (
    <div className="surgery-modal-overlay" onClick={onClose}>
      <div className="surgery-modal" onClick={(e) => e.stopPropagation()}>
        <div className="modal-header">
          <h3>
            <i className="fa fa-tools" />
            Memory Surgery
          </h3>
          <button className="close-btn" onClick={onClose}>
            <i className="fa fa-times" />
          </button>
        </div>

        <div className="modal-body">
          {type === 'episode' ? (
            <>
              <div className="form-group">
                <label>Content</label>
                <textarea
                  value={content}
                  onChange={(e) => setContent(e.target.value)}
                  rows={5}
                  placeholder="Episode content..."
                />
              </div>

              <div className="form-group">
                <label>Importance (Emotional Valence)</label>
                <div className="slider-wrapper">
                  <input
                    type="range"
                    min="0"
                    max="1"
                    step="0.01"
                    value={importance}
                    onChange={(e) => setImportance(parseFloat(e.target.value))}
                  />
                  <span className="value">{(importance * 100).toFixed(0)}%</span>
                </div>
                <p className="help-text">
                  Higher importance increases stability and reduces decay rate.
                </p>
              </div>

              <div className="info-grid">
                <div className="info-item">
                  <span className="label">Stability</span>
                  <span className="value">{(item as Episode).stability.toFixed(2)}</span>
                </div>
                <div className="info-item">
                  <span className="label">Retrievability</span>
                  <span className="value">
                    {(item as Episode).retrievability !== null
                      ? ((item as Episode).retrievability! * 100).toFixed(1) + '%'
                      : 'N/A'}
                  </span>
                </div>
                <div className="info-item">
                  <span className="label">Access Count</span>
                  <span className="value">{item.access_count}</span>
                </div>
              </div>
            </>
          ) : (
            <>
              <div className="form-group">
                <label>Name</label>
                <input
                  type="text"
                  value={(item as Entity).name}
                  disabled
                  className="disabled"
                />
              </div>

              <div className="form-group">
                <label>Summary</label>
                <textarea
                  value={summary}
                  onChange={(e) => setSummary(e.target.value)}
                  rows={4}
                  placeholder="Entity summary..."
                />
              </div>

              <div className="info-grid">
                <div className="info-item">
                  <span className="label">Type</span>
                  <span className="value">{(item as Entity).entity_type}</span>
                </div>
                <div className="info-item">
                  <span className="label">Stability</span>
                  <span className="value">{(item as Entity).stability.toFixed(2)}</span>
                </div>
                <div className="info-item">
                  <span className="label">Access Count</span>
                  <span className="value">{item.access_count}</span>
                </div>
              </div>
            </>
          )}
        </div>

        <div className="modal-footer">
          <button className="btn-secondary" onClick={onClose}>
            Cancel
          </button>
          <button className="btn-primary" onClick={handleSave} disabled={saving}>
            {saving ? 'Saving...' : 'Save Changes'}
          </button>
        </div>
      </div>
    </div>
  );
};

// ============================================================================
// Delete Confirmation Modal
// ============================================================================

interface DeleteModalProps {
  item: Episode | Entity | null;
  type: 'episode' | 'entity';
  onClose: () => void;
  onConfirm: (id: string) => Promise<void>;
}

const DeleteModal: React.FC<DeleteModalProps> = ({ item, type, onClose, onConfirm }) => {
  const [deleting, setDeleting] = useState(false);

  if (!item) return null;

  const handleDelete = async () => {
    setDeleting(true);
    try {
      await onConfirm(item.id);
      onClose();
    } catch (err) {
      console.error('Failed to delete:', err);
    } finally {
      setDeleting(false);
    }
  };

  const displayName = type === 'episode'
    ? (item as Episode).content.slice(0, 50) + '...'
    : (item as Entity).name;

  return (
    <div className="delete-modal-overlay" onClick={onClose}>
      <div className="delete-modal" onClick={(e) => e.stopPropagation()}>
        <div className="modal-header danger">
          <h3>
            <i className="fa fa-exclamation-triangle" />
            Confirm Deletion
          </h3>
        </div>

        <div className="modal-body">
          <p>Are you sure you want to delete this {type}?</p>
          <p className="item-preview">{displayName}</p>
          <p className="warning">This action cannot be undone.</p>
        </div>

        <div className="modal-footer">
          <button className="btn-secondary" onClick={onClose}>
            Cancel
          </button>
          <button className="btn-danger" onClick={handleDelete} disabled={deleting}>
            {deleting ? 'Deleting...' : 'Delete'}
          </button>
        </div>
      </div>
    </div>
  );
};

// ============================================================================
// Memory Item Row
// ============================================================================

interface MemoryRowProps {
  item: Episode | Entity;
  type: 'episode' | 'entity';
  onEdit: () => void;
  onDelete: () => void;
}

const MemoryRow: React.FC<MemoryRowProps> = ({ item, type, onEdit, onDelete }) => {
  const isEpisode = type === 'episode';
  const ep = isEpisode ? (item as Episode) : null;
  const ent = !isEpisode ? (item as Entity) : null;

  const formatDate = (dateStr: string) => {
    const date = new Date(dateStr);
    return date.toLocaleDateString() + ' ' + date.toLocaleTimeString([], { hour: '2-digit', minute: '2-digit' });
  };

  return (
    <div className="memory-row">
      <div className="row-icon">
        <i className={`fa ${isEpisode ? 'fa-history' : 'fa-cube'}`} />
      </div>

      <div className="row-content">
        {isEpisode ? (
          <>
            <p className="content-text">{ep!.content.slice(0, 150)}{ep!.content.length > 150 ? '...' : ''}</p>
            <div className="meta">
              <span className={`outcome ${ep!.outcome.toLowerCase()}`}>{ep!.outcome}</span>
              <span className="timestamp">{formatDate(ep!.timestamp)}</span>
              {ep!.context.project && <span className="project">{ep!.context.project}</span>}
            </div>
          </>
        ) : (
          <>
            <p className="entity-name">{ent!.name}</p>
            <p className="entity-summary">{ent!.summary.slice(0, 100)}{ent!.summary.length > 100 ? '...' : ''}</p>
            <div className="meta">
              <span className="entity-type">{ent!.entity_type}</span>
              <span className="timestamp">{formatDate(ent!.created_at)}</span>
            </div>
          </>
        )}
      </div>

      <div className="row-stats">
        <div className="stat">
          <span className="value">
            {isEpisode
              ? (ep!.retrievability !== null ? (ep!.retrievability * 100).toFixed(0) + '%' : '-')
              : ep?.stability.toFixed(1) ?? ent?.stability.toFixed(1)}
          </span>
          <span className="label">{isEpisode ? 'Recall' : 'Stability'}</span>
        </div>
        <div className="stat">
          <span className="value">{item.access_count}</span>
          <span className="label">Access</span>
        </div>
      </div>

      <div className="row-actions">
        <button className="action-btn edit" onClick={onEdit} title="Edit / Surgery">
          <i className="fa fa-edit" />
        </button>
        <button className="action-btn delete" onClick={onDelete} title="Delete">
          <i className="fa fa-trash" />
        </button>
      </div>
    </div>
  );
};

// ============================================================================
// Main Component
// ============================================================================

export const MemoryListPanel: React.FC = () => {
  const [activeTab, setActiveTab] = useState<MemoryTab>('episodes');
  const [episodes, setEpisodes] = useState<Episode[]>([]);
  const [entities, setEntities] = useState<Entity[]>([]);
  const [loading, setLoading] = useState(true);
  const [error, setError] = useState<string | null>(null);
  const [searchQuery, setSearchQuery] = useState('');
  const [page, setPage] = useState(1);
  const [totalEpisodes, setTotalEpisodes] = useState(0);
  const [totalEntities, setTotalEntities] = useState(0);

  // Modal state
  const [surgeryItem, setSurgeryItem] = useState<Episode | Entity | null>(null);
  const [deleteItem, setDeleteItem] = useState<Episode | Entity | null>(null);
  const [modalType, setModalType] = useState<'episode' | 'entity'>('episode');

  const PAGE_SIZE = 20;

  // Fetch episodes
  const fetchEpisodes = useCallback(async () => {
    try {
      setLoading(true);
      setError(null);

      const url = searchQuery
        ? `/api/v1/episodes/recall`
        : `/api/v1/episodes?page=${page}&page_size=${PAGE_SIZE}`;

      let response;
      if (searchQuery) {
        response = await fetch(url, {
          method: 'POST',
          headers: { 'Content-Type': 'application/json' },
          body: JSON.stringify({ query: searchQuery, limit: PAGE_SIZE }),
        });
      } else {
        response = await fetch(url);
      }

      if (!response.ok) {
        throw new Error(`HTTP ${response.status}: ${response.statusText}`);
      }

      const data = await response.json();

      if (searchQuery) {
        setEpisodes(data.episodes || []);
        setTotalEpisodes(data.episodes?.length || 0);
      } else {
        const listData = data as EpisodeListResponse;
        setEpisodes(listData.episodes);
        setTotalEpisodes(listData.total);
      }
    } catch (err) {
      setError(err instanceof Error ? err.message : 'Failed to fetch episodes');
    } finally {
      setLoading(false);
    }
  }, [page, searchQuery]);

  // Fetch entities
  const fetchEntities = useCallback(async () => {
    try {
      setLoading(true);
      setError(null);

      const url = searchQuery
        ? `/api/v1/entities/recall`
        : `/api/v1/entities?limit=100`;

      let response;
      if (searchQuery) {
        response = await fetch(url, {
          method: 'POST',
          headers: { 'Content-Type': 'application/json' },
          body: JSON.stringify({ query: searchQuery, limit: 50 }),
        });
      } else {
        response = await fetch(url);
      }

      if (!response.ok) {
        throw new Error(`HTTP ${response.status}: ${response.statusText}`);
      }

      const data = await response.json() as EntityListResponse;
      setEntities(data.entities);
      setTotalEntities(data.total);
    } catch (err) {
      setError(err instanceof Error ? err.message : 'Failed to fetch entities');
    } finally {
      setLoading(false);
    }
  }, [searchQuery]);

  // Load data on tab/page change
  useEffect(() => {
    if (activeTab === 'episodes') {
      fetchEpisodes();
    } else {
      fetchEntities();
    }
  }, [activeTab, fetchEpisodes, fetchEntities]);

  // Update episode
  const updateEpisode = async (id: string, updates: Record<string, unknown>) => {
    const response = await fetch(`/api/v1/episodes/${id}`, {
      method: 'PUT',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify(updates),
    });

    if (!response.ok) {
      throw new Error(`HTTP ${response.status}: ${response.statusText}`);
    }

    await fetchEpisodes();
  };

  // Update entity
  const updateEntity = async (id: string, updates: Record<string, unknown>) => {
    const response = await fetch(`/api/v1/entities/${id}`, {
      method: 'PUT',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify(updates),
    });

    if (!response.ok) {
      throw new Error(`HTTP ${response.status}: ${response.statusText}`);
    }

    await fetchEntities();
  };

  // Delete episode
  const deleteEpisode = async (id: string) => {
    const response = await fetch(`/api/v1/episodes/${id}`, {
      method: 'DELETE',
    });

    if (!response.ok && response.status !== 204) {
      throw new Error(`HTTP ${response.status}: ${response.statusText}`);
    }

    await fetchEpisodes();
  };

  // Delete entity
  const deleteEntity = async (id: string) => {
    const response = await fetch(`/api/v1/entities/${id}`, {
      method: 'DELETE',
    });

    if (!response.ok && response.status !== 204) {
      throw new Error(`HTTP ${response.status}: ${response.statusText}`);
    }

    await fetchEntities();
  };

  // Handle search
  const handleSearch = (e: React.FormEvent) => {
    e.preventDefault();
    setPage(1);
    if (activeTab === 'episodes') {
      fetchEpisodes();
    } else {
      fetchEntities();
    }
  };

  const openSurgery = (item: Episode | Entity, type: 'episode' | 'entity') => {
    setSurgeryItem(item);
    setModalType(type);
  };

  const openDelete = (item: Episode | Entity, type: 'episode' | 'entity') => {
    setDeleteItem(item);
    setModalType(type);
  };

  const totalPages = activeTab === 'episodes'
    ? Math.ceil(totalEpisodes / PAGE_SIZE)
    : 1;

  return (
    <div className="memory-list-panel">
      {/* Header */}
      <div className="panel-header">
        <h1>
          <i className="fa fa-database" />
          Memory Management
        </h1>
        <p className="subtitle">View, edit, and delete memories</p>
      </div>

      {/* Tabs */}
      <div className="panel-tabs">
        <button
          className={`tab ${activeTab === 'episodes' ? 'active' : ''}`}
          onClick={() => { setActiveTab('episodes'); setPage(1); }}
        >
          <i className="fa fa-history" />
          Episodes ({totalEpisodes})
        </button>
        <button
          className={`tab ${activeTab === 'entities' ? 'active' : ''}`}
          onClick={() => { setActiveTab('entities'); setPage(1); }}
        >
          <i className="fa fa-cube" />
          Entities ({totalEntities})
        </button>
      </div>

      {/* Search */}
      <form className="search-form" onSubmit={handleSearch}>
        <div className="search-input-wrapper">
          <i className="fa fa-search" />
          <input
            type="text"
            placeholder={`Search ${activeTab}...`}
            value={searchQuery}
            onChange={(e) => setSearchQuery(e.target.value)}
          />
          {searchQuery && (
            <button
              type="button"
              className="clear-btn"
              onClick={() => { setSearchQuery(''); setPage(1); }}
            >
              <i className="fa fa-times" />
            </button>
          )}
        </div>
        <button type="submit" className="search-btn">
          Search
        </button>
      </form>

      {/* Content */}
      <div className="panel-content">
        {error ? (
          <div className="error-state">
            <i className="fa fa-exclamation-circle" />
            <p>{error}</p>
            <button onClick={() => activeTab === 'episodes' ? fetchEpisodes() : fetchEntities()}>
              Retry
            </button>
          </div>
        ) : loading ? (
          <div className="loading-state">
            <i className="fa fa-spinner fa-spin" />
            <p>Loading memories...</p>
          </div>
        ) : (
          <>
            {activeTab === 'episodes' ? (
              episodes.length > 0 ? (
                <div className="memory-list">
                  {episodes.map((ep) => (
                    <MemoryRow
                      key={ep.id}
                      item={ep}
                      type="episode"
                      onEdit={() => openSurgery(ep, 'episode')}
                      onDelete={() => openDelete(ep, 'episode')}
                    />
                  ))}
                </div>
              ) : (
                <div className="empty-state">
                  <i className="fa fa-inbox" />
                  <p>No episodes found</p>
                </div>
              )
            ) : (
              entities.length > 0 ? (
                <div className="memory-list">
                  {entities.map((ent) => (
                    <MemoryRow
                      key={ent.id}
                      item={ent}
                      type="entity"
                      onEdit={() => openSurgery(ent, 'entity')}
                      onDelete={() => openDelete(ent, 'entity')}
                    />
                  ))}
                </div>
              ) : (
                <div className="empty-state">
                  <i className="fa fa-inbox" />
                  <p>No entities found</p>
                </div>
              )
            )}

            {/* Pagination for episodes */}
            {activeTab === 'episodes' && totalPages > 1 && (
              <div className="pagination">
                <button
                  disabled={page === 1}
                  onClick={() => setPage(page - 1)}
                >
                  <i className="fa fa-chevron-left" />
                </button>
                <span className="page-info">
                  Page {page} of {totalPages}
                </span>
                <button
                  disabled={page >= totalPages}
                  onClick={() => setPage(page + 1)}
                >
                  <i className="fa fa-chevron-right" />
                </button>
              </div>
            )}
          </>
        )}
      </div>

      {/* Modals */}
      {surgeryItem && (
        <SurgeryModal
          item={surgeryItem}
          type={modalType}
          onClose={() => setSurgeryItem(null)}
          onSave={modalType === 'episode' ? updateEpisode : updateEntity}
        />
      )}

      {deleteItem && (
        <DeleteModal
          item={deleteItem}
          type={modalType}
          onClose={() => setDeleteItem(null)}
          onConfirm={modalType === 'episode' ? deleteEpisode : deleteEntity}
        />
      )}
    </div>
  );
};

export default MemoryListPanel;
