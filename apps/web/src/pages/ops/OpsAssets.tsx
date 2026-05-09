import { useEffect, useState } from 'react';
import { Link } from 'react-router-dom';
import { api } from '../../api/client';
import type { AssetStatusOut } from '../../api/types';
import './OpsAssets.css';

export default function OpsAssets() {
  const [assets, setAssets] = useState<AssetStatusOut[]>([]);
  const [loading, setLoading] = useState(true);
  const [error, setError] = useState<string | null>(null);

  useEffect(() => {
    api
      .get<AssetStatusOut[]>('/api/admin/ops/assets')
      .then(setAssets)
      .catch(() => setError('Failed to load asset status.'))
      .finally(() => setLoading(false));
  }, []);

  return (
    <div className="ops-assets container">
      <div className="ops-back">
        <Link to="/ops" className="back-link">← Ops overview</Link>
      </div>

      <header className="page-header">
        <h2 className="page-title ops">Asset status</h2>
      </header>

      {loading && <p className="loading-text">Loading…</p>}
      {error && <p className="error-text">{error}</p>}

      {assets.length > 0 && (
        <div className="ops-table-wrapper">
          <table className="ops-table">
            <thead>
              <tr>
                <th>Asset ID</th>
                <th>Run status</th>
                <th>Last forecast</th>
                <th>Last run at</th>
                <th>Consecutive failures</th>
              </tr>
            </thead>
            <tbody>
              {assets.map((a) => (
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
      )}
    </div>
  );
}
