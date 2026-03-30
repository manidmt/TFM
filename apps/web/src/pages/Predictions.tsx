// apps/web/src/pages/Predictions.tsx
import { useEffect, useState } from 'react';
import { Link } from 'react-router-dom';
import { api } from '../api/client';
import type { AssetOut, PredictionOut, VolatilityClass } from '../api/types';
import SignalBadge from '../components/SignalBadge';
import ProbPills from '../components/ProbPills';
import RegimeSparkline from '../components/RegimeSparkline';
import ChartCanvas from '../components/ChartCanvas';
import './Predictions.css';

export default function Predictions() {
  const [assets, setAssets] = useState<AssetOut[]>([]);
  const [predictions, setPredictions] = useState<PredictionOut[]>([]);
  const [sparklines, setSparklines] = useState<Record<string, VolatilityClass[]>>({});
  const [error, setError] = useState<string | null>(null);

  useEffect(() => {
    let cancelled = false;

    Promise.all([
      api.get<AssetOut[]>('/api/public/assets'),
      api.get<PredictionOut[]>('/api/public/predictions/latest'),
    ])
      .then(([a, p]) => {
        if (cancelled) return;
        setAssets(a);
        setPredictions(p);

        Promise.all(
          a.map((asset) =>
            api
              .get<PredictionOut[]>(
                `/api/public/predictions/history?asset_id=${asset.asset_id}&limit=30`,
              )
              .then((rows) => ({
                asset_id: asset.asset_id,
                regimes: rows.map((r) => r.predicted_class),
              }))
              .catch(() => ({ asset_id: asset.asset_id, regimes: [] as VolatilityClass[] })),
          ),
        )
          .then((results) => {
            if (cancelled) return;
            const map: Record<string, VolatilityClass[]> = {};
            results.forEach(({ asset_id, regimes }) => {
              map[asset_id] = regimes;
            });
            setSparklines(map);
          })
          .catch(() => {});
      })
      .catch(() => setError('Failed to load predictions.'));

    return () => {
      cancelled = true;
    };
  }, []);

  const predMap = Object.fromEntries(predictions.map((p) => [p.asset_id, p]));

  return (
    <div className="predictions">
      <ChartCanvas />
      <div className="predictions-inner container">
        <header className="page-header">
          <h2 className="page-title">Latest predictions</h2>
          <p className="page-subtitle">
            5-day volatility regime forecasts updated daily.
          </p>
        </header>

        {error && <p className="error-text">{error}</p>}

        <div className="asset-grid">
          {assets.map((asset) => {
            const pred = predMap[asset.asset_id];
            const regimes = sparklines[asset.asset_id] ?? [];
            const regimeClass = pred ? `regime-${pred.predicted_class}` : '';

            return (
              <div key={asset.asset_id} className={`asset-card ${regimeClass}`}>
                <div className="asset-card-header">
                  <div>
                    <p className="asset-id">{asset.asset_id}</p>
                    <p className="asset-label">{asset.label}</p>
                  </div>
                  <SignalBadge signal={pred?.predicted_class ?? null} />
                </div>
                {regimes.length > 0 && (
                  <>
                    <p className="sparkline-label">30-day regime history →</p>
                    <RegimeSparkline regimes={regimes} />
                  </>
                )}
                {pred ? (
                  <ProbPills
                    p_low={pred.p_low}
                    p_medium={pred.p_medium}
                    p_high={pred.p_high}
                    predicted_class={pred.predicted_class}
                  />
                ) : (
                  <p className="no-pred">No prediction available</p>
                )}
                {pred && (
                  <p className="forecast-date">
                    Forecast date: <span>{pred.forecast_date}</span>
                  </p>
                )}
                <Link to={`/predictions/${asset.asset_id}`} className="card-link">
                  View history →
                </Link>
              </div>
            );
          })}
        </div>
      </div>
    </div>
  );
}
