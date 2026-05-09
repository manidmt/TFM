# Predictions Enhanced Cards Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Improve the Predictions page with colored card borders, 30-day regime sparklines, and pill-chip probability display.

**Architecture:** Three self-contained changes to the Predictions page: (1) a new `ProbPills` component replaces `ProbBar`; (2) a new `RegimeSparkline` component renders a canvas-based 30-bar history strip; (3) `Predictions.tsx` is updated to fetch per-asset history, derive a CSS class from the predicted regime, and wire both new components into each card.

**Tech Stack:** React 18, TypeScript, Vite, CSS custom properties (no chart library — Canvas API for sparkline).

---

## File Map

| File | Action | Responsibility |
|---|---|---|
| `apps/web/src/components/ProbPills.tsx` | Create | Pill-chip probability display |
| `apps/web/src/components/ProbPills.css` | Create | Pill chip styles |
| `apps/web/src/components/RegimeSparkline.tsx` | Create | Canvas-based 30-bar sparkline |
| `apps/web/src/pages/Predictions.tsx` | Modify | Fetch history, add border class, wire components |
| `apps/web/src/pages/Predictions.css` | Modify | Card border-left color classes + sparkline label |

`ProbBar.tsx` and `ProbBar.css` are left in place (used nowhere after this change, but not deleted to avoid breaking anything accidentally).

---

## Task 1: ProbPills component

**Files:**
- Create: `apps/web/src/components/ProbPills.tsx`
- Create: `apps/web/src/components/ProbPills.css`

- [ ] **Step 1: Create `ProbPills.css`**

```css
/* apps/web/src/components/ProbPills.css */
.prob-pills {
  display: flex;
  gap: 6px;
}

.prob-pill {
  flex: 1;
  border-radius: 8px;
  padding: 8px 6px 7px;
  display: flex;
  flex-direction: column;
  align-items: center;
  gap: 3px;
  border: 1.5px solid transparent;
}

.prob-pill.low {
  background: color-mix(in srgb, var(--low) 10%, transparent);
  border-color: color-mix(in srgb, var(--low) 30%, transparent);
}

.prob-pill.medium {
  background: color-mix(in srgb, var(--medium) 10%, transparent);
  border-color: color-mix(in srgb, var(--medium) 30%, transparent);
}

.prob-pill.high {
  background: color-mix(in srgb, var(--high) 10%, transparent);
  border-color: color-mix(in srgb, var(--high) 30%, transparent);
}

.prob-pill.active {
  border-width: 2px;
}

.prob-pill.active.low    { border-color: var(--low);    background: color-mix(in srgb, var(--low)    16%, transparent); }
.prob-pill.active.medium { border-color: var(--medium); background: color-mix(in srgb, var(--medium) 16%, transparent); }
.prob-pill.active.high   { border-color: var(--high);   background: color-mix(in srgb, var(--high)   16%, transparent); }

.prob-pill-label {
  font-size: 0.58rem;
  font-weight: 600;
  text-transform: uppercase;
  letter-spacing: 0.05em;
}

.prob-pill.low    .prob-pill-label { color: var(--low); }
.prob-pill.medium .prob-pill-label { color: var(--medium); }
.prob-pill.high   .prob-pill-label { color: var(--high); }

.prob-pill-value {
  font-family: 'IBM Plex Mono', monospace;
  font-size: 0.85rem;
  font-weight: 500;
}

.prob-pill.low    .prob-pill-value { color: var(--low); }
.prob-pill.medium .prob-pill-value { color: var(--medium); }
.prob-pill.high   .prob-pill-value { color: var(--high); }
```

- [ ] **Step 2: Create `ProbPills.tsx`**

```tsx
// apps/web/src/components/ProbPills.tsx
import type { VolatilityClass } from '../api/types';
import './ProbPills.css';

interface Props {
  p_low: number | null;
  p_medium: number | null;
  p_high: number | null;
  predicted_class: VolatilityClass | null;
}

function pct(v: number | null): string {
  if (v == null) return '—';
  return (v * 100).toFixed(1) + '%';
}

export default function ProbPills({ p_low, p_medium, p_high, predicted_class }: Props) {
  return (
    <div className="prob-pills">
      <div className={`prob-pill low${predicted_class === 'low' ? ' active' : ''}`}>
        <span className="prob-pill-label">Low</span>
        <span className="prob-pill-value">{pct(p_low)}</span>
      </div>
      <div className={`prob-pill medium${predicted_class === 'medium' ? ' active' : ''}`}>
        <span className="prob-pill-label">Med</span>
        <span className="prob-pill-value">{pct(p_medium)}</span>
      </div>
      <div className={`prob-pill high${predicted_class === 'high' ? ' active' : ''}`}>
        <span className="prob-pill-label">High</span>
        <span className="prob-pill-value">{pct(p_high)}</span>
      </div>
    </div>
  );
}
```

- [ ] **Step 3: Verify TypeScript compiles**

```bash
cd apps/web && npm run build 2>&1 | tail -5
```

Expected: build succeeds (no TS errors).

- [ ] **Step 4: Commit**

```bash
git add apps/web/src/components/ProbPills.tsx apps/web/src/components/ProbPills.css
git commit -m "feat(web): add ProbPills component — pill-chip probability display"
```

---

## Task 2: RegimeSparkline component

**Files:**
- Create: `apps/web/src/components/RegimeSparkline.tsx`

- [ ] **Step 1: Create `RegimeSparkline.tsx`**

```tsx
// apps/web/src/components/RegimeSparkline.tsx
import { useEffect, useRef } from 'react';
import type { VolatilityClass } from '../api/types';

interface Props {
  regimes: VolatilityClass[];  // ordered oldest → newest, up to 30 entries
}

const COLOR: Record<VolatilityClass, string> = {
  low: '#4f7a64',
  medium: '#a57a2a',
  high: '#9a5246',
};

export default function RegimeSparkline({ regimes }: Props) {
  const canvasRef = useRef<HTMLCanvasElement>(null);

  useEffect(() => {
    const canvas = canvasRef.current;
    if (!canvas || regimes.length === 0) return;

    const ctx = canvas.getContext('2d');
    if (!ctx) return;

    const W = canvas.offsetWidth;
    const H = canvas.height;
    canvas.width = W;

    const bw = W / regimes.length;
    ctx.clearRect(0, 0, W, H);

    regimes.forEach((regime, i) => {
      ctx.fillStyle = COLOR[regime];
      ctx.globalAlpha = 0.75;
      ctx.beginPath();
      // roundRect may not exist in older browsers — use rect as fallback
      if (ctx.roundRect) {
        ctx.roundRect(i * bw + 0.5, 0, bw - 1, H, 1);
      } else {
        ctx.rect(i * bw + 0.5, 0, bw - 1, H);
      }
      ctx.fill();
    });
    ctx.globalAlpha = 1;
  }, [regimes]);

  if (regimes.length === 0) return null;

  return (
    <canvas
      ref={canvasRef}
      height={20}
      style={{ display: 'block', width: '100%', borderRadius: '3px' }}
    />
  );
}
```

- [ ] **Step 2: Verify TypeScript compiles**

```bash
cd apps/web && npm run build 2>&1 | tail -5
```

Expected: build succeeds.

- [ ] **Step 3: Commit**

```bash
git add apps/web/src/components/RegimeSparkline.tsx
git commit -m "feat(web): add RegimeSparkline component — canvas 30-bar regime history"
```

---

## Task 3: Wire everything into Predictions page

**Files:**
- Modify: `apps/web/src/pages/Predictions.tsx`
- Modify: `apps/web/src/pages/Predictions.css`

The `Predictions` page currently fetches `assets` and `predictions/latest`.
We add a third fetch: for each asset, `GET /api/public/predictions/history?asset_id={id}&limit=30`.

- [ ] **Step 1: Add CSS card border classes to `Predictions.css`**

Open `apps/web/src/pages/Predictions.css` and add at the end:

```css
/* Regime-colored left border */
.asset-card.regime-low    { border-left: 4px solid var(--low); }
.asset-card.regime-medium { border-left: 4px solid var(--medium); }
.asset-card.regime-high   { border-left: 4px solid var(--high); }

/* Sparkline strip label */
.sparkline-label {
  font-family: 'IBM Plex Mono', monospace;
  font-size: 0.58rem;
  color: var(--text-faint);
  text-transform: uppercase;
  letter-spacing: 0.06em;
  margin-bottom: 4px;
}
```

- [ ] **Step 2: Rewrite `Predictions.tsx`**

Replace the entire file with:

```tsx
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
    Promise.all([
      api.get<AssetOut[]>('/api/public/assets'),
      api.get<PredictionOut[]>('/api/public/predictions/latest'),
    ])
      .then(([a, p]) => {
        setAssets(a);
        setPredictions(p);

        // Fetch sparkline history for each asset in parallel
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
        ).then((results) => {
          const map: Record<string, VolatilityClass[]> = {};
          results.forEach(({ asset_id, regimes }) => {
            map[asset_id] = regimes;
          });
          setSparklines(map);
        });
      })
      .catch(() => setError('Failed to load predictions.'));
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
```

- [ ] **Step 3: Verify TypeScript compiles**

```bash
cd apps/web && npm run build 2>&1 | tail -10
```

Expected: build succeeds with no TypeScript errors.

- [ ] **Step 4: Visual check**

Start the dev server (backend must be running with `QUANT_RISK_SERVING_DB_PATH=data/db/serving.duckdb`):

```bash
cd apps/web && npm run dev
```

Open http://localhost:3000/predictions. Verify:
- Each card has a colored left border (green/amber/red matching the badge)
- Each card shows the 30-bar sparkline strip
- Probability section shows 3 pill chips; the predicted class pill has a thicker border

- [ ] **Step 5: Commit**

```bash
git add apps/web/src/pages/Predictions.tsx apps/web/src/pages/Predictions.css
git commit -m "feat(web): enhance Predictions cards — colored border, sparkline, ProbPills"
```
