import { useEffect, useState } from 'react';
import { Link } from 'react-router-dom';
import { api } from '../../api/client';
import type { OpsSummaryOut } from '../../api/types';
import './OpsOverview.css';

export default function OpsOverview() {
  const [summary, setSummary] = useState<OpsSummaryOut | null>(null);
  const [loading, setLoading] = useState(true);
  const [error, setError] = useState<string | null>(null);

  useEffect(() => {
    api
      .get<OpsSummaryOut>('/api/admin/ops/summary')
      .then(setSummary)
      .catch(() => setError('Failed to load ops summary.'))
      .finally(() => setLoading(false));
  }, []);

  return (
    <div className="ops-overview container">
      <header className="page-header">
        <h2 className="page-title ops">Ops overview</h2>
        <div className="ops-nav-links">
          <Link to="/ops/assets" className="ops-link">Asset status →</Link>
          <Link to="/ops/promotions" className="ops-link">Promotions →</Link>
        </div>
      </header>

      {loading && <p className="loading-text">Loading…</p>}
      {error && <p className="error-text">{error}</p>}

      {summary && (
        <>
          <div className="stats-row">
            <div className="stat-card">
              <p className="stat-value">{summary.asset_count}</p>
              <p className="stat-label">Total assets</p>
            </div>
            <div className="stat-card fresh">
              <p className="stat-value">{summary.fresh_count}</p>
              <p className="stat-label">Fresh</p>
            </div>
            <div className="stat-card stale">
              <p className="stat-value">{summary.stale_count}</p>
              <p className="stat-label">Stale</p>
            </div>
            <div className="stat-card failed">
              <p className="stat-value">{summary.failed_count}</p>
              <p className="stat-label">Failed</p>
            </div>
          </div>

          <h3 className="section-title">Asset status</h3>
          <div className="ops-table-wrapper">
            <table className="ops-table">
              <thead>
                <tr>
                  <th>Asset</th>
                  <th>Status</th>
                  <th>Last forecast</th>
                  <th>Last run</th>
                  <th>Failures</th>
                </tr>
              </thead>
              <tbody>
                {summary.assets.map((a) => (
                  <tr key={a.asset_id}>
                    <td className="mono">{a.asset_id}</td>
                    <td>
                      <span className={`status-chip status-${a.run_status ?? 'unknown'}`}>
                        {a.run_status ?? '—'}
                      </span>
                    </td>
                    <td className="mono">{a.last_forecast_date ?? '—'}</td>
                    <td className="mono faint">{a.last_run_at ?? '—'}</td>
                    <td className="mono">{a.consecutive_failures ?? 0}</td>
                  </tr>
                ))}
              </tbody>
            </table>
          </div>
        </>
      )}
    </div>
  );
}
