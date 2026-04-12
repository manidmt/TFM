import { useEffect, useState } from 'react';
import { useParams, Link, useNavigate } from 'react-router-dom';
import { api, ApiError } from '../../api/client';
import type { PortfolioDetailOut, AssetOut, PortfolioAnalysisOut, TickerOut, VolatilityClass } from '../../api/types';
import SignalBadge from '../../components/SignalBadge';
import TickerCombobox from '../../components/TickerCombobox';
import './PortfolioDetail.css';

// ---------------------------------------------------------------------------
// Constants
// ---------------------------------------------------------------------------

const PROXY_LABEL: Record<string, string> = {
  us_equities: 'US',
  euro_equities: 'EU',
  bitcoin: 'Crypto',
  long_us_treasuries: 'Bond L',
  short_us_treasuries: 'Bond S',
  gold: 'Cmdty',
};

const PROXY_NAME: Record<string, string> = {
  us_equities: 'US Equities',
  euro_equities: 'Euro Equities',
  bitcoin: 'Bitcoin',
  long_us_treasuries: 'Long Treasuries',
  short_us_treasuries: 'Short Treasuries',
  gold: 'Gold',
};

const PROXY_COLOR: Record<string, string> = {
  us_equities: '#6366f1',
  euro_equities: '#8b5cf6',
  bitcoin: '#f59e0b',
  long_us_treasuries: '#06b6d4',
  short_us_treasuries: '#14b8a6',
  gold: '#eab308',
};

const SIGNAL_COLOR: Record<string, string> = {
  low: '#4ade80',
  medium: '#f59e0b',
  high: '#ef4444',
};

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

// ---------------------------------------------------------------------------
// Types
// ---------------------------------------------------------------------------

interface EditablePosition {
  label: string;
  weight_pct: number;
  proxy_asset_id: string;
}

function emptyPosition(): EditablePosition {
  return { label: '', weight_pct: 0, proxy_asset_id: '' };
}

// ---------------------------------------------------------------------------
// Main component
// ---------------------------------------------------------------------------

export default function PortfolioDetail() {
  const { portfolioId } = useParams<{ portfolioId: string }>();
  const navigate = useNavigate();

  const [portfolio, setPortfolio] = useState<PortfolioDetailOut | null>(null);
  const [assets, setAssets] = useState<AssetOut[]>([]);
  const [tickers, setTickers] = useState<TickerOut[]>([]);
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
      api.get<TickerOut[]>('/api/public/tickers'),
    ]).then(([p, a, t]) => {
      setPortfolio(p);
      setPositions(p.positions.map((pos) => ({ ...pos })));
      setEditName(p.name);
      setAssets(a);
      setTickers(t);
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
      // Update portfolio metadata if it was a real analysis (not what-if)
      if (!overridePositions && portfolio) {
        setPortfolio({
          ...portfolio,
          last_signal: result.portfolio_signal as VolatilityClass | null,
          last_analysis_at: new Date().toISOString(),
        });
      }
    } catch (err) {
      setError(err instanceof ApiError ? err.message : 'Analysis failed.');
    } finally {
      setAnalysing(false);
    }
  }

  if (loading) return <div className="container page-loading">Loading…</div>;
  if (!portfolio) return <div className="container page-loading">Portfolio not found.</div>;

  const isEditing = editMode;
  const totalWeight = positions.reduce((s, p) => s + (p.weight_pct || 0), 0);

  return (
    <div className="portfolio-detail container">
      <div className="detail-nav">
        <Link to="/app" className="back-link">← My portfolios</Link>
      </div>

      {/* Header */}
      <header className="detail-header">
        <div className="detail-header-info">
          {isEditing ? (
            <input
              className="field-input name-input"
              value={editName}
              onChange={(e) => setEditName(e.target.value)}
            />
          ) : (
            <>
              <h2 className="page-title">{portfolio.name}</h2>
              <p className="detail-subtitle">
                {portfolio.positions.length} position{portfolio.positions.length !== 1 ? 's' : ''}
                {portfolio.last_analysis_at
                  ? ` · Last analysed ${timeAgo(portfolio.last_analysis_at)}`
                  : ' · Never analysed'}
              </p>
            </>
          )}
        </div>
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

      {/* Positions */}
      <section className="positions-section">
        <div className="positions-header">
          <h3 className="section-title" style={{ margin: 0 }}>Positions</h3>
        </div>

        {/* Weight progress bar (edit mode) */}
        {isEditing && (
          <div className="weight-progress">
            <div className="weight-progress-header">
              <span className="weight-progress-label">Allocation</span>
              <span
                className="weight-progress-value"
                style={{ color: Math.abs(totalWeight - 100) < 0.01 ? '#4ade80' : totalWeight > 100 ? '#ef4444' : '#f59e0b' }}
              >
                {totalWeight.toFixed(1)}% / 100%
              </span>
            </div>
            <div className="weight-progress-track">
              <div
                className="weight-progress-fill"
                style={{
                  width: `${Math.min(totalWeight, 100)}%`,
                  background: Math.abs(totalWeight - 100) < 0.01 ? '#4ade80' : totalWeight > 100 ? '#ef4444' : '#f59e0b',
                }}
              />
            </div>
            {Math.abs(totalWeight - 100) >= 0.01 && (
              <span className="weight-progress-remaining">
                {totalWeight < 100 ? `${(100 - totalWeight).toFixed(1)}% remaining` : `${(totalWeight - 100).toFixed(1)}% over`}
              </span>
            )}
          </div>
        )}

        {positions.length === 0 && !isEditing && (
          <p className="empty-text">No positions. Click Edit to add some.</p>
        )}

        <div className="positions-list">
          {positions.map((pos, idx) => (
            <div key={idx} className={isEditing ? 'position-row' : 'position-card'}>
              {isEditing ? (
                <>
                  <TickerCombobox
                    tickers={tickers}
                    value={pos.label}
                    onChange={(ticker, proxyId) => {
                      setPositions((prev) => prev.map((p, i) =>
                        i === idx
                          ? { ...p, label: ticker, ...(proxyId ? { proxy_asset_id: proxyId } : {}) }
                          : p
                      ));
                    }}
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
                  <div className="pos-card-header">
                    <div className="pos-card-title">
                      <span className="pos-card-label">{pos.label}</span>
                      <span className="pos-card-class">{PROXY_LABEL[pos.proxy_asset_id] ?? '?'}</span>
                    </div>
                    <span className="pos-card-weight">{pos.weight_pct}%</span>
                  </div>
                  <div className="pos-card-bar-track">
                    <div
                      className="pos-card-bar-fill"
                      style={{
                        width: `${pos.weight_pct}%`,
                        background: PROXY_COLOR[pos.proxy_asset_id] ?? '#64748b',
                      }}
                    />
                  </div>
                  <div className="pos-card-proxy">→ {PROXY_NAME[pos.proxy_asset_id] ?? pos.proxy_asset_id}</div>
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
                    <TickerCombobox
                      tickers={tickers}
                      value={pos.label}
                      onChange={(ticker, proxyId) => {
                        setWhatIfPositions((prev) => prev.map((p, i) =>
                          i === idx
                            ? { ...p, label: ticker, ...(proxyId ? { proxy_asset_id: proxyId } : {}) }
                            : p
                        ));
                      }}
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

// ---------------------------------------------------------------------------
// AnalysisResult
// ---------------------------------------------------------------------------

function DonutChart({ groups }: { groups: PortfolioAnalysisOut['asset_groups'] }) {
  const total = groups.reduce((s, g) => s + g.aggregate_weight, 0) || 1;
  const radius = 50;
  const circumference = 2 * Math.PI * radius;
  let offset = 0;

  const segments = groups.map((g) => {
    const pct = g.aggregate_weight / total;
    const dash = pct * circumference;
    const seg = { ...g, dash, gap: circumference - dash, offset, color: PROXY_COLOR[g.asset_id] ?? '#64748b' };
    offset += dash;
    return seg;
  });

  return (
    <div className="donut-container">
      <svg width="120" height="120" viewBox="0 0 120 120" className="donut-svg">
        <circle cx="60" cy="60" r={radius} fill="none" stroke="var(--line)" strokeWidth="16" />
        {segments.map((seg) => (
          <circle
            key={seg.asset_id}
            cx="60"
            cy="60"
            r={radius}
            fill="none"
            stroke={seg.color}
            strokeWidth="16"
            strokeDasharray={`${seg.dash} ${seg.gap}`}
            strokeDashoffset={-seg.offset}
            transform="rotate(-90 60 60)"
          />
        ))}
        <text x="60" y="56" textAnchor="middle" fill="var(--text)" fontSize="14" fontWeight="600">{groups.length}</text>
        <text x="60" y="72" textAnchor="middle" fill="var(--text-soft)" fontSize="10">group{groups.length !== 1 ? 's' : ''}</text>
      </svg>
      <div className="donut-legend">
        {groups.map((g) => (
          <div key={g.asset_id} className="donut-legend-item">
            <span className="donut-legend-swatch" style={{ background: PROXY_COLOR[g.asset_id] ?? '#64748b' }} />
            <span className="donut-legend-name">{PROXY_NAME[g.asset_id] ?? g.asset_id}</span>
            <span className="donut-legend-pct">{(g.aggregate_weight * 100).toFixed(0)}%</span>
            <div className="donut-legend-labels">{g.position_labels.join(', ')}</div>
          </div>
        ))}
      </div>
    </div>
  );
}

function ProbBars({ p_low, p_medium, p_high }: { p_low: number; p_medium: number; p_high: number }) {
  const bars = [
    { label: 'P(low)', value: p_low, color: '#4ade80' },
    { label: 'P(medium)', value: p_medium, color: '#f59e0b' },
    { label: 'P(high)', value: p_high, color: '#ef4444' },
  ];
  return (
    <div className="prob-bars">
      {bars.map((b) => (
        <div key={b.label} className="prob-bar-row">
          <div className="prob-bar-header">
            <span style={{ color: b.color }}>{b.label}</span>
            <span className="prob-bar-value" style={{ color: b.color }}>{(b.value * 100).toFixed(1)}%</span>
          </div>
          <div className="prob-bar-track">
            <div className="prob-bar-fill" style={{ width: `${b.value * 100}%`, background: b.color }} />
          </div>
        </div>
      ))}
    </div>
  );
}

function AnalysisResult({ result }: { result: PortfolioAnalysisOut }) {
  const uniqueProxies = new Set(result.asset_groups.map((g) => g.asset_id)).size;

  return (
    <div className="analysis-result">
      {/* Summary cards */}
      <div className="analysis-summary-cards">
        <div className="summary-card">
          <p className="summary-card-label">Portfolio signal</p>
          <SignalBadge signal={result.portfolio_signal as VolatilityClass | null} />
        </div>
        <div className="summary-card">
          <p className="summary-card-label">Risk concentration</p>
          <p className="summary-card-value">{result.positions.length} position{result.positions.length !== 1 ? 's' : ''}, {uniqueProxies} proxy group{uniqueProxies !== 1 ? 's' : ''}</p>
        </div>
        <div className="summary-card">
          <p className="summary-card-label">Total weight</p>
          <p className="summary-card-value mono">{result.total_weight_pct.toFixed(1)}%</p>
        </div>
      </div>

      {/* Two-column: donut + prob bars */}
      <div className="analysis-two-col">
        <div className="analysis-panel">
          <p className="analysis-panel-title">Allocation by proxy asset</p>
          <DonutChart groups={result.asset_groups} />
        </div>
        <div className="analysis-panel">
          <p className="analysis-panel-title">Weighted probability breakdown</p>
          <ProbBars p_low={result.portfolio_p_low} p_medium={result.portfolio_p_medium} p_high={result.portfolio_p_high} />
        </div>
      </div>

      {/* Missing predictions */}
      {result.missing_predictions.length > 0 && (
        <div className="missing-warning">
          No predictions available for: {result.missing_predictions.join(', ')}
        </div>
      )}

      {/* Position cards */}
      <h4 className="section-title">By position</h4>
      <div className="analysis-position-cards">
        {result.positions.map((p) => (
          <div
            key={p.label}
            className="analysis-pos-card"
            style={{ borderLeftColor: SIGNAL_COLOR[p.predicted_class ?? ''] ?? 'var(--line)' }}
          >
            <div className="analysis-pos-header">
              <span className="analysis-pos-label">{p.label}</span>
              <SignalBadge signal={p.predicted_class as VolatilityClass | null} size="sm" />
            </div>
            <div className="analysis-pos-meta">
              {p.weight_pct.toFixed(1)}% · {PROXY_NAME[p.proxy_asset_id] ?? p.proxy_asset_id}
            </div>
            {p.p_low != null && p.p_medium != null && p.p_high != null && (
              <div className="analysis-pos-minibar">
                <div style={{ width: `${p.p_low * 100}%`, background: '#4ade80' }} />
                <div style={{ width: `${p.p_medium * 100}%`, background: '#f59e0b' }} />
                <div style={{ width: `${p.p_high * 100}%`, background: '#ef4444' }} />
              </div>
            )}
            {p.p_low != null && (
              <div className="analysis-pos-pcts">
                <span>{(p.p_low! * 100).toFixed(0)}%</span>
                <span>{(p.p_medium! * 100).toFixed(0)}%</span>
                <span>{(p.p_high! * 100).toFixed(0)}%</span>
              </div>
            )}
          </div>
        ))}
      </div>
    </div>
  );
}
