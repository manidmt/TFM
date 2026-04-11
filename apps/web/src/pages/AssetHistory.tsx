// apps/web/src/pages/AssetHistory.tsx
import { useEffect, useMemo, useState } from 'react';
import { useParams, Link } from 'react-router-dom';
import { api } from '../api/client';
import type { PredictionOut, PricePoint, RegimePoint, VolatilityClass, VolProfile } from '../api/types';
import SignalBadge from '../components/SignalBadge';
import ProbPills from '../components/ProbPills';
import PriceRegimeChart, { type Range, RANGE_DAYS } from '../components/PriceRegimeChart';
import './AssetHistory.css';

// Fallback 5-day vol values (conservative); overridden by per-asset vol-profile from API.
const FALLBACK_VOL_5D: Record<VolatilityClass, number> = { low: 0.005, medium: 0.008, high: 0.014 };

export default function AssetHistory() {
  const { assetId } = useParams<{ assetId: string }>();
  const [prices, setPrices] = useState<PricePoint[]>([]);
  const [history, setHistory] = useState<PredictionOut[]>([]);
  const [realRegimes, setRealRegimes] = useState<RegimePoint[]>([]);
  const [volProfile, setVolProfile] = useState<VolProfile | null>(null);
  const [error, setError] = useState<string | null>(null);
  const [loading, setLoading] = useState(true);
  const [range, setRange] = useState<Range>('6M');

  useEffect(() => {
    if (!assetId) return;
    let cancelled = false;

    Promise.all([
      api.get<PricePoint[]>(`/api/public/prices/history?asset_id=${assetId}&days=365`),
      api.get<PredictionOut[]>(`/api/public/predictions/history?asset_id=${assetId}&limit=365`),
      api.get<RegimePoint[]>(`/api/public/regimes/history?asset_id=${assetId}&days=365`),
      api.get<VolProfile>(`/api/public/vol-profile?asset_id=${assetId}`),
    ])
      .then(([p, h, r, v]) => {
        if (cancelled) return;
        setPrices(p);
        setHistory(h);
        setRealRegimes(r);
        setVolProfile(v);
      })
      .catch(() => {
        if (!cancelled) setError('Failed to load asset data.');
      })
      .finally(() => {
        if (!cancelled) setLoading(false);
      });

    return () => { cancelled = true; };
  }, [assetId]);

  const latestPred = useMemo(() => {
    if (history.length === 0) return null;
    return [...history].sort((a, b) => b.forecast_date.localeCompare(a.forecast_date))[0];
  }, [history]);

  const dist = useMemo(() => {
    const cutoff = new Date();
    cutoff.setDate(cutoff.getDate() - RANGE_DAYS[range]);
    const filtered = realRegimes.filter((r) => new Date(r.date) >= cutoff);
    const counts = { low: 0, medium: 0, high: 0 };
    filtered.forEach((r) => {
      counts[r.regime as VolatilityClass]++;
    });
    const total = filtered.length || 1;
    return {
      low: Math.round((counts.low / total) * 100),
      medium: Math.round((counts.medium / total) * 100),
      high: 100 - Math.round((counts.low / total) * 100) - Math.round((counts.medium / total) * 100),
    };
  }, [realRegimes, range]);

  const streak = useMemo(() => {
    if (history.length === 0) return { days: 0, regime: 'medium' as VolatilityClass };
    const sorted = [...history].sort((a, b) => b.forecast_date.localeCompare(a.forecast_date));
    const cur = sorted[0].predicted_class as VolatilityClass;
    let count = 0;
    for (const h of sorted) {
      if (h.predicted_class !== cur) break;
      count++;
    }
    return { days: count, regime: cur };
  }, [history]);

  const vol5d = useMemo((): Record<VolatilityClass, number> => {
    if (!volProfile) return FALLBACK_VOL_5D;
    return { low: volProfile.vol_5d_low, medium: volProfile.vol_5d_medium, high: volProfile.vol_5d_high };
  }, [volProfile]);

  const forecastBand = useMemo(() => {
    if (!latestPred || prices.length === 0) return null;
    const lastClose = prices[prices.length - 1].close;
    const sig5 =
      latestPred.p_low * vol5d.low +
      latestPred.p_medium * vol5d.medium +
      latestPred.p_high * vol5d.high;
    const hi = lastClose * Math.exp(1.65 * sig5);
    const lo = lastClose * Math.exp(-1.65 * sig5);
    const pct = ((hi / lastClose - 1) * 100).toFixed(1);
    const fmt = (v: number) => v >= 1000 ? `$${(v / 1000).toFixed(1)}k` : `$${v.toFixed(2)}`;
    return { pct, range: `${fmt(lo)} – ${fmt(hi)}` };
  }, [latestPred, prices, vol5d]);

  const periodReturn = useMemo(() => {
    if (prices.length < 2) return null;
    const cutoff = new Date();
    cutoff.setDate(cutoff.getDate() - RANGE_DAYS[range]);
    const old = prices.find((p) => new Date(p.date) >= cutoff);
    if (!old) return null;
    const last = prices[prices.length - 1];
    const ret = ((last.close / old.close - 1) * 100).toFixed(1);
    return { value: parseFloat(ret), label: ret };
  }, [prices, range]);

  const lastPrice = prices.length > 0 ? prices[prices.length - 1] : null;
  const regimeClass = latestPred ? `regime-${latestPred.predicted_class}` : '';

  return (
    <div className="asset-history container">
      <div className="history-nav">
        <Link to="/predictions" className="back-link">← All predictions</Link>
      </div>

      <div className="page-header-row">
        <div>
          <span className="asset-id-inline">{assetId}</span>
          <h2 className="page-title">{assetId?.replace(/_/g, ' ')}</h2>
          <p className="page-subtitle">Price history with volatility regime overlay</p>
        </div>
        {latestPred && <SignalBadge signal={latestPred.predicted_class} />}
      </div>

      {loading && <p className="loading-text">Loading…</p>}
      {error && <p className="error-text">{error}</p>}

      {!loading && !error && prices.length > 0 && (
        <>
          {/* Chart */}
          <div className="chart-card">
            <PriceRegimeChart
              prices={prices}
              realRegimes={realRegimes}
              latestPred={latestPred}
              vol5d={vol5d}
              range={range}
              onRangeChange={setRange}
            />
          </div>

          {/* Stats row */}
          <div className="stats-row">
            <div className="stat-card">
              <div className="stat-label">Current streak</div>
              <div className="stat-value" style={{ color: `var(--${streak.regime})` }}>
                {streak.days} days
              </div>
              <div className="stat-sub">in {streak.regime.toUpperCase()} regime</div>
            </div>
            <div className="stat-card">
              <div className="stat-label">Forecast band</div>
              <div className="stat-value">{forecastBand ? `±${forecastBand.pct}%` : '—'}</div>
              <div className="stat-sub">{forecastBand?.range ?? '5-day ±1.65σ'}</div>
            </div>
            <div className="stat-card">
              <div className="stat-label">Last price</div>
              <div className="stat-value">
                {lastPrice
                  ? lastPrice.close >= 1000
                    ? `$${(lastPrice.close / 1000).toFixed(1)}k`
                    : `$${lastPrice.close.toFixed(2)}`
                  : '—'}
              </div>
              <div className="stat-sub">{lastPrice?.date ?? ''}</div>
            </div>
            <div className="stat-card">
              <div className="stat-label">{range} return</div>
              <div
                className="stat-value"
                style={{ color: periodReturn && periodReturn.value >= 0 ? 'var(--low)' : 'var(--high)' }}
              >
                {periodReturn ? `${periodReturn.value >= 0 ? '+' : ''}${periodReturn.label}%` : '—'}
              </div>
              <div className="stat-sub">vs {RANGE_DAYS[range]} days ago</div>
            </div>
          </div>

          {/* Bottom panels */}
          <div className="bottom-panels">
            <div className={`panel-card ${regimeClass}`}>
              <div className="panel-title">
                Latest prediction · {latestPred?.forecast_date ?? '—'}
              </div>
              {latestPred && (
                <>
                  <SignalBadge signal={latestPred.predicted_class} />
                  <ProbPills
                    p_low={latestPred.p_low}
                    p_medium={latestPred.p_medium}
                    p_high={latestPred.p_high}
                    predicted_class={latestPred.predicted_class as VolatilityClass}
                  />
                </>
              )}
            </div>
            <div className="panel-card">
              <div className="panel-title">Regime distribution · history</div>
              <div className="regime-dist">
                {(['low', 'medium', 'high'] as VolatilityClass[]).map((r) => (
                  <div className="dist-row" key={r}>
                    <span className={`dist-label ${r}`}>{r}</span>
                    <div className="dist-track">
                      <div className={`dist-fill ${r}`} style={{ width: `${dist[r]}%` }} />
                    </div>
                    <span className="dist-pct">{dist[r]}%</span>
                  </div>
                ))}
              </div>
            </div>
          </div>
        </>
      )}

      {!loading && !error && prices.length === 0 && (
        <p className="empty-text">No price data available for this asset.</p>
      )}
    </div>
  );
}
