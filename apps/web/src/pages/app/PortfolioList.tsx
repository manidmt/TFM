import { useEffect, useState } from 'react';
import { Link, useNavigate } from 'react-router-dom';
import { api } from '../../api/client';
import type { PortfolioSummaryOut } from '../../api/types';
import './PortfolioList.css';

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
              <span className="portfolio-name">{p.name}</span>
              <span className="portfolio-arrow">→</span>
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
