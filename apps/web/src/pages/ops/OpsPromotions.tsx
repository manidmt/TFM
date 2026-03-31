import { useEffect, useState } from 'react';
import { Link } from 'react-router-dom';
import { api } from '../../api/client';
import type { PromotionEventOut } from '../../api/types';
import './OpsPromotions.css';

export default function OpsPromotions() {
  const [events, setEvents] = useState<PromotionEventOut[]>([]);
  const [loading, setLoading] = useState(true);
  const [error, setError] = useState<string | null>(null);

  useEffect(() => {
    api
      .get<PromotionEventOut[]>('/api/admin/ops/promotions?limit=40')
      .then(setEvents)
      .catch(() => setError('Failed to load promotions.'))
      .finally(() => setLoading(false));
  }, []);

  return (
    <div className="ops-promotions container">
      <div className="ops-back">
        <Link to="/ops" className="back-link">← Ops overview</Link>
      </div>

      <header className="page-header">
        <h2 className="page-title ops">Promotion history</h2>
        <p className="page-subtitle">Most recent 40 bundle promotion events.</p>
      </header>

      {loading && <p className="loading-text">Loading…</p>}
      {error && <p className="error-text">{error}</p>}
      {!loading && events.length === 0 && !error && (
        <p className="loading-text">No promotion events found.</p>
      )}

      {events.length > 0 && (
        <div className="ops-table-wrapper">
          <table className="ops-table">
            <thead>
              <tr>
                <th>Promoted at</th>
                <th>Asset</th>
                <th>Bundle version</th>
                <th>Status</th>
                <th>Previous</th>
                <th>Validation errors</th>
              </tr>
            </thead>
            <tbody>
              {events.map((e) => (
                <tr key={e.event_id}>
                  <td className="mono faint">{e.promoted_at}</td>
                  <td className="mono">{e.asset_id}</td>
                  <td className="mono">{e.bundle_version}</td>
                  <td>
                    <span className={`status-chip status-${e.status}`}>{e.status}</span>
                  </td>
                  <td className="mono faint">{e.previous_version ?? '—'}</td>
                  <td className="error-cell">{e.validation_errors ?? '—'}</td>
                </tr>
              ))}
            </tbody>
          </table>
        </div>
      )}
    </div>
  );
}
