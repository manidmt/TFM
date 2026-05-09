import { useEffect, useState } from 'react';
import { Link, useNavigate } from 'react-router-dom';
import { api } from '../../api/client';
import type { PortfolioSummaryOut } from '../../api/types';
import SignalBadge from '../../components/SignalBadge';
import './PortfolioList.css';

function timeAgo(iso: string): string {
  const diff = Date.now() - new Date(iso).getTime();
  const mins = Math.floor(diff / 60000);
  if (mins < 1) return 'just now';
  if (mins < 60) return `${mins} min ago`;
  const hrs = Math.floor(mins / 60);
  if (hrs < 24) return `${hrs}h ago`;
  const days = Math.floor(hrs / 24);
  return `${days}d ago`;
}

export default function PortfolioList() {
  const [portfolios, setPortfolios] = useState<PortfolioSummaryOut[]>([]);
  const [loading, setLoading] = useState(true);
  const [newName, setNewName] = useState('');
  const [creating, setCreating] = useState(false);
  const [error, setError] = useState<string | null>(null);
  const navigate = useNavigate();

  function load() {
    return api
      .get<PortfolioSummaryOut[]>('/api/private/portfolios')
      .then(setPortfolios)
      .finally(() => setLoading(false));
  }

  useEffect(() => { load(); }, []);

  async function handleCreate(e: React.FormEvent) {
    e.preventDefault();
    setError(null);
    setCreating(true);
    try {
      const created = await api.post<{ portfolio_id: string }>('/api/private/portfolios', {
        name: newName,
        positions: [],
      });
      navigate(`/app/portfolios/${created.portfolio_id}`);
    } catch {
      setError('Could not create portfolio.');
      setCreating(false);
    }
  }

  return (
    <div className="portfolio-list container">
      <header className="page-header">
        <h2 className="page-title">My portfolios</h2>
      </header>

      {loading && <p className="loading-text">Loading…</p>}

      {!loading && portfolios.length === 0 && (
        <p className="empty-text">No portfolios yet. Create one below.</p>
      )}

      <ul className="portfolio-items">
        {portfolios.map((p) => (
          <li key={p.portfolio_id} className="portfolio-item">
            <Link to={`/app/portfolios/${p.portfolio_id}`} className="portfolio-item-link">
              <div className="portfolio-item-info">
                <span className="portfolio-name">{p.name}</span>
                <span className="portfolio-meta">
                  {p.position_count} position{p.position_count !== 1 ? 's' : ''}
                  {p.position_count > 0 && <> · {p.total_weight_pct.toFixed(0)}%</>}
                </span>
                {p.last_analysis_at && (
                  <span className="portfolio-analysed">Analysed {timeAgo(p.last_analysis_at)}</span>
                )}
              </div>
              <div className="portfolio-item-right">
                {p.last_signal ? (
                  <SignalBadge signal={p.last_signal} size="sm" />
                ) : (
                  <span className="portfolio-no-analysis">No analysis</span>
                )}
                <span className="portfolio-arrow">→</span>
              </div>
            </Link>
          </li>
        ))}
      </ul>

      <div className="create-section">
        <h3 className="section-title">New portfolio</h3>
        {error && <p className="error-text">{error}</p>}
        <form onSubmit={handleCreate} className="create-form">
          <input
            type="text"
            className="field-input"
            placeholder="Portfolio name"
            value={newName}
            onChange={(e) => setNewName(e.target.value)}
            required
          />
          <button type="submit" className="btn-primary" disabled={creating}>
            {creating ? 'Creating…' : 'Create'}
          </button>
        </form>
      </div>
    </div>
  );
}
