import { useEffect, useState } from 'react';
import { useParams, Link, useNavigate } from 'react-router-dom';
import { api, ApiError } from '../../api/client';
import type { PortfolioDetailOut, AssetOut, PortfolioAnalysisOut } from '../../api/types';
import SignalBadge from '../../components/SignalBadge';
import ProbBar from '../../components/ProbBar';
import './PortfolioDetail.css';

interface EditablePosition {
  label: string;
  weight_pct: number;
  proxy_asset_id: string;
}

function emptyPosition(): EditablePosition {
  return { label: '', weight_pct: 0, proxy_asset_id: '' };
}

export default function PortfolioDetail() {
  const { portfolioId } = useParams<{ portfolioId: string }>();
  const navigate = useNavigate();

  const [portfolio, setPortfolio] = useState<PortfolioDetailOut | null>(null);
  const [assets, setAssets] = useState<AssetOut[]>([]);
  const [positions, setPositions] = useState<EditablePosition[]>([]);
  const [editName, setEditName] = useState('');
  const [editMode, setEditMode] = useState(false);
  const [analysis, setAnalysis] = useState<PortfolioAnalysisOut | null>(null);
  const [whatIfPositions, setWhatIfPositions] = useState<EditablePosition[]>([]);
  const [whatIfMode, setWhatIfMode] = useState(false);
  const [loading, setLoading] = useState(true);
  const [saving, setSaving] = useState(false);
  const [analysing, setAnalysing] = useState(false);
  const [error, setError] = useState<string | null>(null);

  useEffect(() => {
    Promise.all([
      api.get<PortfolioDetailOut>(`/api/private/portfolios/${portfolioId}`),
      api.get<AssetOut[]>('/api/public/assets'),
    ]).then(([p, a]) => {
      setPortfolio(p);
      setPositions(p.positions.map((pos) => ({ ...pos })));
      setEditName(p.name);
      setAssets(a);
    }).finally(() => setLoading(false));
  }, [portfolioId]);

  function handlePositionChange(idx: number, field: keyof EditablePosition, value: string | number) {
    setPositions((prev) => prev.map((p, i) => i === idx ? { ...p, [field]: value } : p));
  }

  function addPosition() {
    setPositions((prev) => [...prev, emptyPosition()]);
  }

  function removePosition(idx: number) {
    setPositions((prev) => prev.filter((_, i) => i !== idx));
  }

  async function handleSave() {
    setSaving(true);
    setError(null);
    try {
      const updated = await api.put<PortfolioDetailOut>(`/api/private/portfolios/${portfolioId}`, {
        name: editName !== portfolio?.name ? editName : undefined,
        positions,
      });
      setPortfolio(updated);
      setPositions(updated.positions.map((p) => ({ ...p })));
      setEditMode(false);
    } catch (err) {
      setError(err instanceof ApiError ? err.message : 'Save failed.');
    } finally {
      setSaving(false);
    }
  }

  async function handleDelete() {
    if (!confirm('Delete this portfolio?')) return;
    await api.delete(`/api/private/portfolios/${portfolioId}`);
    navigate('/app');
  }

  async function handleAnalyse(overridePositions?: EditablePosition[]) {
    setAnalysing(true);
    setError(null);
    setAnalysis(null);
    try {
      const result = await api.post<PortfolioAnalysisOut>(
        `/api/private/portfolios/${portfolioId}/analyze`,
        overridePositions ? { positions: overridePositions } : {},
      );
      setAnalysis(result);
    } catch (err) {
      setError(err instanceof ApiError ? err.message : 'Analysis failed.');
    } finally {
      setAnalysing(false);
    }
  }

  if (loading) return <div className="container page-loading">Loading…</div>;
  if (!portfolio) return <div className="container page-loading">Portfolio not found.</div>;

  const isEditing = editMode;

  return (
    <div className="portfolio-detail container">
      <div className="detail-nav">
        <Link to="/app" className="back-link">← My portfolios</Link>
      </div>

      <header className="detail-header">
        {isEditing ? (
          <input
            className="field-input name-input"
            value={editName}
            onChange={(e) => setEditName(e.target.value)}
          />
        ) : (
          <h2 className="page-title">{portfolio.name}</h2>
        )}
        <div className="detail-actions">
          {!isEditing && (
            <>
              <button className="btn-ghost" onClick={() => setEditMode(true)}>Edit</button>
              <button className="btn-ghost danger" onClick={handleDelete}>Delete</button>
            </>
          )}
        </div>
      </header>

      {error && <p className="error-text">{error}</p>}

      {/* Positions editor */}
      <section className="positions-section">
        <div style={{ display: 'flex', alignItems: 'center', justifyContent: 'space-between', marginBottom: '12px' }}>
          <h3 className="section-title" style={{ margin: 0 }}>Positions</h3>
          {isEditing && (() => {
            const total = positions.reduce((s, p) => s + (p.weight_pct || 0), 0);
            const ok = Math.abs(total - 100) < 0.01;
            return (
              <span style={{ fontSize: '12px', padding: '3px 10px', borderRadius: '12px', border: '1px solid #444', background: '#1a2340', color: ok ? '#4ade80' : '#f59e0b' }}>
                Total: {total.toFixed(1)}%{!ok && ` — ${(100 - total).toFixed(1)}% remaining`}
              </span>
            );
          })()}
        </div>

        {positions.length === 0 && !isEditing && (
          <p className="empty-text">No positions. Click Edit to add some.</p>
        )}

        <div className="positions-list">
          {positions.map((pos, idx) => (
            <div key={idx} className="position-row">
              {isEditing ? (
                <>
                  <input
                    className="field-input pos-label"
                    placeholder="Label"
                    value={pos.label}
                    onChange={(e) => handlePositionChange(idx, 'label', e.target.value)}
                  />
                  <div style={{ position: 'relative' }}>
                    <input
                      className="field-input pos-weight"
                      type="number"
                      placeholder="Weight"
                      value={pos.weight_pct}
                      onChange={(e) => handlePositionChange(idx, 'weight_pct', parseFloat(e.target.value))}
                      min={0}
                      step={0.01}
                      style={{ paddingRight: '24px' }}
                    />
                    <span style={{ position: 'absolute', right: '8px', top: '50%', transform: 'translateY(-50%)', color: '#64748b', fontSize: '12px', pointerEvents: 'none' }}>%</span>
                  </div>
                  <select
                    className="field-input pos-proxy"
                    value={pos.proxy_asset_id}
                    onChange={(e) => handlePositionChange(idx, 'proxy_asset_id', e.target.value)}
                  >
                    <option value="">— proxy asset —</option>
                    {assets.map((a) => (
                      <option key={a.asset_id} value={a.asset_id}>{a.display_name} — {a.source_ticker}</option>
                    ))}
                  </select>
                  <button className="btn-remove" onClick={() => removePosition(idx)}>✕</button>
                </>
              ) : (
                <>
                  <span className="pos-label-text">{pos.label}</span>
                  <span className="pos-weight-text">{pos.weight_pct}%</span>
                  <span className="pos-proxy-text">
                    {(() => {
                      const asset = assets.find((a) => a.asset_id === pos.proxy_asset_id);
                      return asset
                        ? <>{asset.display_name}<span style={{ fontFamily: 'monospace', fontSize: '11px', color: '#475569', marginLeft: '4px' }}>{asset.source_ticker}</span></>
                        : pos.proxy_asset_id;
                    })()}
                  </span>
                </>
              )}
            </div>
          ))}
        </div>

        {isEditing && (
          <div className="edit-controls">
            <button className="btn-ghost" onClick={addPosition}>+ Add position</button>
            <div className="edit-save-row">
              <button className="btn-ghost" onClick={() => { setEditMode(false); setPositions(portfolio.positions.map(p => ({...p}))); }}>
                Cancel
              </button>
              <button className="btn-primary" onClick={handleSave} disabled={saving}>
                {saving ? 'Saving…' : 'Save'}
              </button>
            </div>
          </div>
        )}
      </section>

      {/* Analyse section */}
      {!isEditing && (
        <section className="analyse-section">
          <h3 className="section-title">Portfolio analysis</h3>

          <div className="analyse-controls">
            <button
              className="btn-primary"
              onClick={() => handleAnalyse()}
              disabled={analysing}
            >
              {analysing ? 'Analysing…' : 'Analyse portfolio'}
            </button>
            <button
              className="btn-ghost"
              onClick={() => {
                setWhatIfPositions(positions.map(p => ({...p})));
                setWhatIfMode(true);
              }}
            >
              What-if analysis
            </button>
          </div>

          {whatIfMode && (
            <div className="whatif-section">
              <h4 className="section-title">What-if positions</h4>
              <div className="positions-list">
                {whatIfPositions.map((pos, idx) => (
                  <div key={idx} className="position-row">
                    <input
                      className="field-input pos-label"
                      placeholder="Label"
                      value={pos.label}
                      onChange={(e) => setWhatIfPositions(prev => prev.map((p, i) => i === idx ? {...p, label: e.target.value} : p))}
                    />
                    <div style={{ position: 'relative' }}>
                      <input
                        className="field-input pos-weight"
                        type="number"
                        placeholder="Weight"
                        value={pos.weight_pct}
                        onChange={(e) => setWhatIfPositions(prev => prev.map((p, i) => i === idx ? {...p, weight_pct: parseFloat(e.target.value)} : p))}
                        min={0}
                        step={0.01}
                        style={{ paddingRight: '24px' }}
                      />
                      <span style={{ position: 'absolute', right: '8px', top: '50%', transform: 'translateY(-50%)', color: '#64748b', fontSize: '12px', pointerEvents: 'none' }}>%</span>
                    </div>
                    <select
                      className="field-input pos-proxy"
                      value={pos.proxy_asset_id}
                      onChange={(e) => setWhatIfPositions(prev => prev.map((p, i) => i === idx ? {...p, proxy_asset_id: e.target.value} : p))}
                    >
                      <option value="">— proxy —</option>
                      {assets.map((a) => <option key={a.asset_id} value={a.asset_id}>{a.display_name} — {a.source_ticker}</option>)}
                    </select>
                    <button className="btn-remove" onClick={() => setWhatIfPositions(prev => prev.filter((_, i) => i !== idx))}>✕</button>
                  </div>
                ))}
              </div>
              <div className="edit-controls">
                <button className="btn-ghost" onClick={() => setWhatIfPositions(prev => [...prev, emptyPosition()])}>+ Add</button>
                <div className="edit-save-row">
                  <button className="btn-ghost" onClick={() => setWhatIfMode(false)}>Cancel</button>
                  <button className="btn-primary" onClick={() => handleAnalyse(whatIfPositions)} disabled={analysing}>
                    Run what-if
                  </button>
                </div>
              </div>
            </div>
          )}

          {analysis && <AnalysisResult result={analysis} />}
        </section>
      )}
    </div>
  );
}

function AnalysisResult({ result }: { result: PortfolioAnalysisOut }) {
  return (
    <div className="analysis-result">
      <div className="analysis-summary">
        <div className="summary-signal">
          <p className="summary-label">Portfolio signal</p>
          <SignalBadge signal={result.portfolio_signal} />
        </div>
        <div className="summary-probs">
          <p className="summary-label">Weighted probabilities</p>
          <ProbBar p_low={result.portfolio_p_low} p_medium={result.portfolio_p_medium} p_high={result.portfolio_p_high} />
        </div>
        <div className="summary-weight">
          <p className="summary-label">Total weight</p>
          <p className="summary-value mono">{result.total_weight_pct.toFixed(1)}%</p>
        </div>
      </div>

      {result.missing_predictions.length > 0 && (
        <p className="missing-warning">
          No predictions for: {result.missing_predictions.join(', ')}
        </p>
      )}

      <h4 className="section-title">By position</h4>
      <div className="positions-table-wrapper">
        <table className="positions-table">
          <thead>
            <tr>
              <th>Label</th>
              <th>Weight</th>
              <th>Proxy</th>
              <th>Signal</th>
              <th className="th-wide">Probabilities</th>
            </tr>
          </thead>
          <tbody>
            {result.positions.map((p) => (
              <tr key={p.label}>
                <td>{p.label}</td>
                <td className="mono">{p.weight_pct.toFixed(1)}%</td>
                <td className="mono">{p.proxy_asset_id}</td>
                <td><SignalBadge signal={p.predicted_class} size="sm" /></td>
                <td><ProbBar p_low={p.p_low} p_medium={p.p_medium} p_high={p.p_high} /></td>
              </tr>
            ))}
          </tbody>
        </table>
      </div>
    </div>
  );
}
