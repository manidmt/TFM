# Asset History Chart Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Replace the AssetHistory prediction table with a professional price chart: daily close prices with regime-colored backgrounds, a probability-weighted 5-day forecast cone, range selector (1M/3M/6M/1Y), and a stats panel.

**Architecture:** A new `GET /api/public/prices/history` endpoint reads daily close prices from `financial_data.duckdb` (the research DB) and maps `asset_id` → `source_ticker` via the asset catalog. The frontend fetches prices + prediction history in parallel, then `PriceRegimeChart` renders everything on a single `<canvas>` with a client-side range filter.

**Tech Stack:** FastAPI + DuckDB (backend); React 18 + TypeScript + Vite + Canvas API (frontend). Tests: pytest (backend), `npm run build` (frontend TypeScript check).

---

## File Map

| File | Action | Responsibility |
|---|---|---|
| `src/quant_risk/prod/api/config.py` | Modify | Add `research_db_path` from env `QUANT_RISK_DB_PATH` |
| `src/quant_risk/prod/api/deps.py` | Modify | Add `get_research_db` generator dependency |
| `src/quant_risk/prod/api/routers/public.py` | Modify | Add `PricePoint` schema + `GET /api/public/prices/history` route |
| `tests/prod/test_prices_endpoint.py` | Create | Tests for the new endpoint |
| `apps/web/src/api/types.ts` | Modify | Add `PricePoint` interface |
| `apps/web/src/components/PriceRegimeChart.tsx` | Create | Canvas chart: price line, regime backgrounds, forecast cone, range selector |
| `apps/web/src/pages/AssetHistory.tsx` | Modify | Replace table with chart + stats row + forecast card + regime distribution |
| `apps/web/src/pages/AssetHistory.css` | Modify | New layout styles (replace table styles) |

---

## Task 1: Backend — prices/history endpoint

**Files:**
- Modify: `src/quant_risk/prod/api/config.py`
- Modify: `src/quant_risk/prod/api/deps.py`
- Modify: `src/quant_risk/prod/api/routers/public.py`
- Create: `tests/prod/test_prices_endpoint.py`

- [ ] **Step 1: Write failing tests**

```python
# tests/prod/test_prices_endpoint.py
'''Tests for GET /api/public/prices/history endpoint.'''
from __future__ import annotations

import duckdb
import pytest
from fastapi.testclient import TestClient

from quant_risk.prod.api.app import create_app
from quant_risk.prod.api.config import AppConfig
from quant_risk.prod.api.deps import get_auth_db, get_config, get_research_db, get_serving_db
from quant_risk.prod.auth.db import AuthDB
from quant_risk.prod.serving.duckdb import ServingDB


@pytest.fixture
def research_db():
    """In-memory DuckDB with raw_prices rows for bitcoin (BTC-USD) and gold (GLD)."""
    conn = duckdb.connect(":memory:")
    conn.execute("""
        CREATE TABLE raw_prices (
            ticker  VARCHAR NOT NULL,
            date    DATE    NOT NULL,
            open    DOUBLE,
            high    DOUBLE,
            low     DOUBLE,
            close   DOUBLE,
            volume  BIGINT,
            PRIMARY KEY (ticker, date)
        )
    """)
    conn.execute("""
        INSERT INTO raw_prices VALUES
        ('BTC-USD', '2025-10-01', 40000, 41000, 39000, 40500, 1000),
        ('BTC-USD', '2025-10-02', 40500, 42000, 40000, 41200, 1100),
        ('BTC-USD', '2025-10-03', 41200, 43000, 41000, 42800, 1200),
        ('GLD',     '2025-10-01', 180,   182,   179,   181,   5000),
        ('GLD',     '2025-10-02', 181,   183,   180,   182,   5100)
    """)
    yield conn
    conn.close()


@pytest.fixture
def client(research_db, tmp_path):
    """TestClient with research_db and minimal auth/serving stubs."""
    app = create_app()
    auth_db = AuthDB(f"sqlite:///{tmp_path}/auth.db")
    auth_db.create_tables()
    serving_db = ServingDB(":memory:")

    test_config = AppConfig()
    test_config.postgres_url = f"sqlite:///{tmp_path}/auth.db"
    test_config.serving_db_path = ":memory:"
    test_config.research_db_path = ":memory:"  # unused — overridden below

    def _auth_db_override():
        with auth_db.session() as s:
            try:
                yield s
            except Exception:
                s.rollback()
                raise
            finally:
                s.close()

    app.dependency_overrides[get_auth_db] = _auth_db_override
    app.dependency_overrides[get_serving_db] = lambda: (yield serving_db)
    app.dependency_overrides[get_research_db] = lambda: (yield research_db)
    app.dependency_overrides[get_config] = lambda: test_config

    with TestClient(app, raise_server_exceptions=True) as c:
        yield c

    app.dependency_overrides.clear()
    serving_db.close()


def test_prices_history_returns_list(client):
    resp = client.get("/api/public/prices/history?asset_id=bitcoin&days=180")
    assert resp.status_code == 200
    data = resp.json()
    assert isinstance(data, list)
    assert len(data) == 3


def test_prices_history_fields(client):
    resp = client.get("/api/public/prices/history?asset_id=bitcoin&days=180")
    row = resp.json()[0]
    assert set(row.keys()) == {"date", "close"}
    assert isinstance(row["date"], str)
    assert isinstance(row["close"], float)


def test_prices_history_ordered_asc(client):
    resp = client.get("/api/public/prices/history?asset_id=bitcoin&days=180")
    dates = [r["date"] for r in resp.json()]
    assert dates == sorted(dates)


def test_prices_history_unknown_asset_404(client):
    resp = client.get("/api/public/prices/history?asset_id=nonexistent&days=180")
    assert resp.status_code == 404


def test_prices_history_gold(client):
    resp = client.get("/api/public/prices/history?asset_id=gold&days=180")
    assert resp.status_code == 200
    assert len(resp.json()) == 2
```

- [ ] **Step 2: Run tests to verify they fail**

```bash
cd /home/manidmt/TFM/quant-risk-tfm
poetry run pytest tests/prod/test_prices_endpoint.py -v 2>&1 | tail -20
```

Expected: FAIL — `ImportError` or `AttributeError` (get_research_db not defined yet).

- [ ] **Step 3: Add `research_db_path` to `AppConfig`**

Open `src/quant_risk/prod/api/config.py`. After the line `self.serving_db_path: str | None = os.environ.get("QUANT_RISK_SERVING_DB_PATH")`, add:

```python
        self.research_db_path: str = os.environ.get(
            "QUANT_RISK_DB_PATH", "data/db/financial_data.duckdb"
        )
```

- [ ] **Step 4: Add `get_research_db` to `deps.py`**

At the top of `src/quant_risk/prod/api/deps.py`, add `import duckdb` to the existing imports. Then after the `get_serving_db` function, add:

```python
def get_research_db(
    config: AppConfig = Depends(get_config),
) -> Generator[duckdb.DuckDBPyConnection, None, None]:
    """Yield a read-only DuckDB connection to the research database (financial_data.duckdb)."""
    conn = duckdb.connect(config.research_db_path, read_only=True)
    try:
        yield conn
    finally:
        conn.close()
```

- [ ] **Step 5: Add `PricePoint` schema and route to `public.py`**

Open `src/quant_risk/prod/api/routers/public.py`. Add `import duckdb` and `from datetime import date, timedelta` to the existing imports at the top. Add `get_research_db` to the deps import line. Then add `PricePoint` schema after `PredictionOut` and the new route at the end of the file:

```python
# After PredictionOut class:
class PricePoint(BaseModel):
    date: str
    close: float
```

```python
# At the end of the file, after predictions_history():
@router.get("/prices/history", response_model=list[PricePoint])
def prices_history(
    asset_id: str = Query(..., description="Production asset ID"),
    days: int = Query(180, ge=30, le=730, description="Lookback window in calendar days"),
    research_db: duckdb.DuckDBPyConnection = Depends(get_research_db),
):
    """Return daily close prices for one asset from the research DB."""
    try:
        catalog = load_asset_catalog(_ASSETS_CONFIG)
    except FileNotFoundError:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"Asset '{asset_id}' not found.",
        )

    asset = next((a for a in catalog if a.asset_id == asset_id), None)
    if asset is None:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"Asset '{asset_id}' not found.",
        )

    since = date.today() - timedelta(days=days)
    rows = research_db.execute(
        """
        SELECT CAST(date AS VARCHAR) AS date, close
        FROM raw_prices
        WHERE ticker = ? AND date >= ?
        ORDER BY date ASC
        """,
        [asset.source_ticker, since],
    ).fetchall()

    return [PricePoint(date=r[0], close=r[1]) for r in rows]
```

- [ ] **Step 6: Run tests to verify they pass**

```bash
cd /home/manidmt/TFM/quant-risk-tfm
poetry run pytest tests/prod/test_prices_endpoint.py -v 2>&1 | tail -20
```

Expected: 5 tests PASS.

- [ ] **Step 7: Run full test suite to check no regressions**

```bash
poetry run pytest tests/ -q 2>&1 | tail -10
```

Expected: all tests pass (or same failures as before this task).

- [ ] **Step 8: Commit**

```bash
git add src/quant_risk/prod/api/config.py \
        src/quant_risk/prod/api/deps.py \
        src/quant_risk/prod/api/routers/public.py \
        tests/prod/test_prices_endpoint.py
git commit -m "feat(api): add GET /api/public/prices/history endpoint"
```

---

## Task 2: PriceRegimeChart component

**Files:**
- Modify: `apps/web/src/api/types.ts`
- Create: `apps/web/src/components/PriceRegimeChart.tsx`

- [ ] **Step 1: Add `PricePoint` type to `types.ts`**

Open `apps/web/src/api/types.ts`. Add after the `PredictionOut` interface:

```typescript
export interface PricePoint {
  date: string;   // ISO date string e.g. "2025-10-01"
  close: number;
}
```

- [ ] **Step 2: Create `PriceRegimeChart.tsx`**

Create `apps/web/src/components/PriceRegimeChart.tsx`:

```tsx
// apps/web/src/components/PriceRegimeChart.tsx
import { useEffect, useMemo, useRef, useState } from 'react';
import type { PredictionOut, PricePoint, VolatilityClass } from '../api/types';

interface Props {
  prices: PricePoint[];
  regimes: PredictionOut[];         // prediction history, any order — component sorts by date
  latestPred: PredictionOut | null;
}

type Range = '1M' | '3M' | '6M' | '1Y';
const RANGE_DAYS: Record<Range, number> = { '1M': 30, '3M': 90, '6M': 180, '1Y': 365 };

const COLOR: Record<VolatilityClass, string> = {
  low: '#4f7a64',
  medium: '#a57a2a',
  high: '#9a5246',
};
const ACCENT = '#21384d';
// Annualised vol per regime (used for forecast band)
const VOL_ANN: Record<VolatilityClass, number> = { low: 0.12, medium: 0.22, high: 0.40 };

function sigmaWeighted(pred: PredictionOut): number {
  return pred.p_low * VOL_ANN.low + pred.p_medium * VOL_ANN.medium + pred.p_high * VOL_ANN.high;
}

export default function PriceRegimeChart({ prices, regimes, latestPred }: Props) {
  const canvasRef = useRef<HTMLCanvasElement>(null);
  const [range, setRange] = useState<Range>('6M');

  // Filter prices to selected range
  const filtered = useMemo(() => {
    if (prices.length === 0) return [];
    const cutoff = new Date();
    cutoff.setDate(cutoff.getDate() - RANGE_DAYS[range]);
    return prices.filter((p) => new Date(p.date) >= cutoff);
  }, [prices, range]);

  // Build regime map: date string → VolatilityClass (carry-forward for missing days)
  const regimeByDate = useMemo(() => {
    const sorted = [...regimes].sort((a, b) => a.forecast_date.localeCompare(b.forecast_date));
    const map = new Map<string, VolatilityClass>();
    sorted.forEach((r) => map.set(r.forecast_date, r.predicted_class as VolatilityClass));
    return map;
  }, [regimes]);

  // Assign a regime to each price point using carry-forward
  const points = useMemo((): Array<{ date: string; close: number; regime: VolatilityClass }> => {
    let last: VolatilityClass = 'medium';
    return filtered.map((p) => {
      const r = regimeByDate.get(p.date);
      if (r) last = r;
      return { ...p, regime: last };
    });
  }, [filtered, regimeByDate]);

  useEffect(() => {
    const canvas = canvasRef.current;
    if (!canvas || points.length === 0) return;

    function render() {
      if (!canvas) return;
      const ctx = canvas.getContext('2d');
      if (!ctx) return;

      const dpr = window.devicePixelRatio || 1;
      const W = canvas.offsetWidth;
      const H = 260;
      canvas.width = W * dpr;
      canvas.height = H * dpr;
      ctx.scale(dpr, dpr);

      const HORIZON = 5;
      const PAD = { top: 14, right: 22, bottom: 28, left: 60 };
      const cW = W - PAD.left - PAD.right;
      const cH = H - PAD.top - PAD.bottom;
      const N = points.length;

      // Forecast band
      let bandHi = 0, bandLo = 0;
      if (latestPred) {
        const sigAnn = sigmaWeighted(latestPred);
        const sig5 = (sigAnn / Math.sqrt(252)) * Math.sqrt(HORIZON);
        const last = points[N - 1].close;
        bandHi = last * Math.exp(1.65 * sig5);
        bandLo = last * Math.exp(-1.65 * sig5);
      }

      const allY = points.map((p) => p.close);
      if (latestPred) { allY.push(bandHi, bandLo); }
      const minP = Math.min(...allY) * 0.991;
      const maxP = Math.max(...allY) * 1.009;

      const total = latestPred ? N + HORIZON : N;
      function px(i: number) { return PAD.left + (i / (total - 1)) * cW; }
      function py(v: number) { return PAD.top + (1 - (v - minP) / (maxP - minP)) * cH; }

      // Grid
      ctx.strokeStyle = '#d9d3c830';
      ctx.lineWidth = 1;
      for (let i = 0; i <= 4; i++) {
        const v = minP + ((maxP - minP) * i) / 4;
        ctx.beginPath();
        ctx.moveTo(PAD.left, py(v));
        ctx.lineTo(PAD.left + cW, py(v));
        ctx.stroke();
      }

      // Regime backgrounds (batch by consecutive same regime)
      let segStart = 0;
      for (let i = 1; i <= N; i++) {
        if (i === N || points[i].regime !== points[segStart].regime) {
          ctx.fillStyle = COLOR[points[segStart].regime] + '26';
          ctx.fillRect(px(segStart), PAD.top, px(Math.min(i, N - 1)) - px(segStart) + 1, cH);
          segStart = i;
        }
      }

      // Forecast cone
      if (latestPred && bandHi > 0) {
        const xNow = px(N - 1);
        const xEnd = px(N + HORIZON - 1);
        const grad = ctx.createLinearGradient(xNow, 0, xEnd, 0);
        grad.addColorStop(0, ACCENT + '00');
        grad.addColorStop(1, ACCENT + '1a');
        ctx.beginPath();
        ctx.moveTo(xNow, py(points[N - 1].close));
        ctx.lineTo(xEnd, py(bandHi));
        ctx.lineTo(xEnd, py(bandLo));
        ctx.closePath();
        ctx.fillStyle = grad;
        ctx.fill();

        ctx.strokeStyle = ACCENT + '60';
        ctx.lineWidth = 1;
        ctx.setLineDash([3, 4]);
        ctx.beginPath();
        ctx.moveTo(xNow, py(points[N - 1].close));
        ctx.lineTo(xEnd, py(bandHi));
        ctx.stroke();
        ctx.beginPath();
        ctx.moveTo(xNow, py(points[N - 1].close));
        ctx.lineTo(xEnd, py(bandLo));
        ctx.stroke();
        ctx.setLineDash([]);

        // Today vertical
        ctx.strokeStyle = ACCENT + '28';
        ctx.lineWidth = 1;
        ctx.setLineDash([4, 4]);
        ctx.beginPath();
        ctx.moveTo(xNow, PAD.top);
        ctx.lineTo(xNow, PAD.top + cH);
        ctx.stroke();
        ctx.setLineDash([]);

        ctx.fillStyle = ACCENT + '70';
        ctx.font = '9px IBM Plex Mono, monospace';
        ctx.textAlign = 'center';
        ctx.fillText('Today', xNow, PAD.top + cH + 14);
      }

      // Price line
      ctx.strokeStyle = ACCENT;
      ctx.lineWidth = 2;
      ctx.lineJoin = 'round';
      ctx.beginPath();
      points.forEach((p, i) =>
        i === 0 ? ctx.moveTo(px(i), py(p.close)) : ctx.lineTo(px(i), py(p.close)),
      );
      ctx.stroke();

      // Y-axis labels
      ctx.fillStyle = '#8a938f';
      ctx.font = '10px IBM Plex Mono, monospace';
      ctx.textAlign = 'right';
      for (let i = 0; i <= 4; i++) {
        const v = minP + ((maxP - minP) * i) / 4;
        const label = v >= 1000 ? `$${(v / 1000).toFixed(0)}k` : `$${v.toFixed(0)}`;
        ctx.fillText(label, PAD.left - 6, py(v) + 3.5);
      }

      // X-axis month labels from actual dates
      if (points.length > 1) {
        ctx.textAlign = 'center';
        const step = Math.floor(N / 5);
        for (let i = 0; i < N; i += step) {
          const d = new Date(points[i].date);
          const label = d.toLocaleDateString('en-US', { month: 'short' });
          ctx.fillText(label, px(i), PAD.top + cH + 14);
        }
      }
    }

    render();
    const ro = new ResizeObserver(render);
    ro.observe(canvas);
    return () => ro.disconnect();
  }, [points, latestPred]);

  return (
    <div>
      {/* Range selector */}
      <div style={{ display: 'flex', justifyContent: 'space-between', alignItems: 'center', marginBottom: '12px' }}>
        <span className="chart-label">Price · USD</span>
        <div className="range-pills">
          {(['1M', '3M', '6M', '1Y'] as Range[]).map((r) => (
            <button
              key={r}
              className={`range-pill${range === r ? ' active' : ''}`}
              onClick={() => setRange(r)}
            >
              {r}
            </button>
          ))}
        </div>
      </div>
      <canvas
        ref={canvasRef}
        aria-label="Price chart with volatility regime overlay"
        role="img"
        style={{ display: 'block', width: '100%', height: '260px' }}
      />
      {/* Legend */}
      <div className="chart-legend">
        <span className="legend-item">
          <span className="legend-sq" style={{ background: '#4f7a6426', border: '1px solid #4f7a6460' }} />
          Low regime
        </span>
        <span className="legend-item">
          <span className="legend-sq" style={{ background: '#a57a2a26', border: '1px solid #a57a2a60' }} />
          Medium regime
        </span>
        <span className="legend-item">
          <span className="legend-sq" style={{ background: '#9a524626', border: '1px solid #9a524660' }} />
          High regime
        </span>
        <span className="legend-item">
          <span className="legend-line" />
          Close price
        </span>
        {latestPred && (
          <span className="legend-item">
            <span className="legend-sq" style={{ background: '#21384d18', border: '1px dashed #21384d70' }} />
            5-day forecast band
          </span>
        )}
      </div>
    </div>
  );
}
```

- [ ] **Step 3: Verify TypeScript compiles**

```bash
cd /home/manidmt/TFM/quant-risk-tfm/apps/web && npm run build 2>&1 | tail -8
```

Expected: build succeeds.

- [ ] **Step 4: Commit**

```bash
cd /home/manidmt/TFM/quant-risk-tfm
git add apps/web/src/api/types.ts apps/web/src/components/PriceRegimeChart.tsx
git commit -m "feat(web): add PricePoint type and PriceRegimeChart canvas component"
```

---

## Task 3: AssetHistory page rewrite

**Files:**
- Modify: `apps/web/src/pages/AssetHistory.tsx`
- Modify: `apps/web/src/pages/AssetHistory.css`

- [ ] **Step 1: Replace `AssetHistory.css` entirely**

```css
/* apps/web/src/pages/AssetHistory.css */
.asset-history {
  padding-top: var(--space-7);
  padding-bottom: var(--space-8);
}

.history-nav {
  margin-bottom: var(--space-5);
}

.back-link {
  font-family: 'IBM Plex Sans', system-ui, sans-serif;
  font-size: 0.875rem;
  color: var(--text-soft);
  text-decoration: none;
}

.back-link:hover {
  color: var(--text);
}

/* Header */
.asset-id-inline {
  font-family: 'IBM Plex Mono', monospace;
  font-size: 0.875rem;
  color: var(--text-faint);
  display: block;
  margin-bottom: 2px;
}

.page-header-row {
  display: flex;
  justify-content: space-between;
  align-items: flex-start;
  margin-bottom: var(--space-6);
}

/* Chart card */
.chart-card {
  background: var(--surface);
  border: 1px solid var(--line);
  border-radius: var(--radius-md);
  padding: var(--space-4) var(--space-4) var(--space-3);
  margin-bottom: var(--space-4);
}

.chart-label {
  font-family: 'IBM Plex Mono', monospace;
  font-size: 0.62rem;
  font-weight: 600;
  text-transform: uppercase;
  letter-spacing: 0.06em;
  color: var(--text-faint);
}

.range-pills {
  display: flex;
  gap: 4px;
}

.range-pill {
  font-family: 'IBM Plex Mono', monospace;
  font-size: 0.62rem;
  font-weight: 600;
  padding: 3px 9px;
  border-radius: 5px;
  border: 1px solid var(--line);
  color: var(--text-soft);
  background: transparent;
  cursor: pointer;
  transition: background 0.1s, color 0.1s;
}

.range-pill.active {
  background: var(--accent);
  color: white;
  border-color: var(--accent);
}

.range-pill:hover:not(.active) {
  background: var(--bg-soft);
}

.chart-legend {
  display: flex;
  gap: 14px;
  margin-top: var(--space-3);
  flex-wrap: wrap;
}

.legend-item {
  display: flex;
  align-items: center;
  gap: 5px;
  font-size: 0.6rem;
  color: var(--text-soft);
}

.legend-sq {
  width: 10px;
  height: 10px;
  border-radius: 2px;
  flex-shrink: 0;
}

.legend-line {
  width: 16px;
  height: 2px;
  background: var(--accent);
  border-radius: 1px;
  flex-shrink: 0;
}

/* Stats row */
.stats-row {
  display: grid;
  grid-template-columns: repeat(4, 1fr);
  gap: var(--space-3);
  margin-bottom: var(--space-4);
}

.stat-card {
  background: var(--surface);
  border: 1px solid var(--line);
  border-radius: var(--radius-md);
  padding: var(--space-3) var(--space-3);
}

.stat-label {
  font-size: 0.58rem;
  font-weight: 600;
  text-transform: uppercase;
  letter-spacing: 0.06em;
  color: var(--text-faint);
  margin-bottom: var(--space-1);
}

.stat-value {
  font-family: 'IBM Plex Mono', monospace;
  font-size: 1.05rem;
  font-weight: 500;
  color: var(--text);
}

.stat-sub {
  font-size: 0.62rem;
  color: var(--text-soft);
  margin-top: 2px;
}

/* Bottom panels */
.bottom-panels {
  display: grid;
  grid-template-columns: 1fr 1fr;
  gap: var(--space-3);
}

.panel-card {
  background: var(--surface);
  border: 1px solid var(--line);
  border-radius: var(--radius-md);
  padding: var(--space-3) var(--space-4);
}

.panel-card.regime-high   { border-left: 4px solid var(--high); }
.panel-card.regime-medium { border-left: 4px solid var(--medium); }
.panel-card.regime-low    { border-left: 4px solid var(--low); }

.panel-title {
  font-size: 0.62rem;
  font-weight: 600;
  text-transform: uppercase;
  letter-spacing: 0.06em;
  color: var(--text-faint);
  margin-bottom: var(--space-2);
}

/* Regime distribution bars */
.regime-dist {
  display: flex;
  flex-direction: column;
  gap: var(--space-2);
  margin-top: var(--space-2);
}

.dist-row {
  display: flex;
  align-items: center;
  gap: var(--space-2);
}

.dist-label {
  font-size: 0.62rem;
  font-weight: 600;
  text-transform: uppercase;
  width: 48px;
  flex-shrink: 0;
}

.dist-label.low    { color: var(--low); }
.dist-label.medium { color: var(--medium); }
.dist-label.high   { color: var(--high); }

.dist-track {
  flex: 1;
  height: 7px;
  background: var(--line);
  border-radius: 4px;
  overflow: hidden;
}

.dist-fill {
  height: 100%;
  border-radius: 4px;
}

.dist-fill.low    { background: var(--low); }
.dist-fill.medium { background: var(--medium); }
.dist-fill.high   { background: var(--high); }

.dist-pct {
  font-family: 'IBM Plex Mono', monospace;
  font-size: 0.65rem;
  color: var(--text-soft);
  width: 32px;
  text-align: right;
}

.loading-text,
.empty-text,
.error-text {
  font-size: 0.9rem;
  color: var(--text-faint);
  font-style: italic;
}

.error-text { color: var(--high); font-style: normal; }
```

- [ ] **Step 2: Replace `AssetHistory.tsx` entirely**

```tsx
// apps/web/src/pages/AssetHistory.tsx
import { useEffect, useMemo, useState } from 'react';
import { useParams, Link } from 'react-router-dom';
import { api } from '../api/client';
import type { PredictionOut, PricePoint, VolatilityClass } from '../api/types';
import SignalBadge from '../components/SignalBadge';
import ProbPills from '../components/ProbPills';
import PriceRegimeChart from '../components/PriceRegimeChart';
import './AssetHistory.css';

const VOL_ANN: Record<VolatilityClass, number> = { low: 0.12, medium: 0.22, high: 0.40 };

export default function AssetHistory() {
  const { assetId } = useParams<{ assetId: string }>();
  const [prices, setPrices] = useState<PricePoint[]>([]);
  const [history, setHistory] = useState<PredictionOut[]>([]);
  const [error, setError] = useState<string | null>(null);
  const [loading, setLoading] = useState(true);

  useEffect(() => {
    if (!assetId) return;
    let cancelled = false;

    Promise.all([
      api.get<PricePoint[]>(`/api/public/prices/history?asset_id=${assetId}&days=365`),
      api.get<PredictionOut[]>(`/api/public/predictions/history?asset_id=${assetId}&limit=365`),
    ])
      .then(([p, h]) => {
        if (cancelled) return;
        setPrices(p);
        setHistory(h);
      })
      .catch(() => {
        if (!cancelled) setError('Failed to load asset data.');
      })
      .finally(() => {
        if (!cancelled) setLoading(false);
      });

    return () => { cancelled = true; };
  }, [assetId]);

  // Latest prediction = newest forecast_date in history
  const latestPred = useMemo(() => {
    if (history.length === 0) return null;
    return [...history].sort((a, b) => b.forecast_date.localeCompare(a.forecast_date))[0];
  }, [history]);

  // Regime distribution (all history)
  const dist = useMemo(() => {
    const counts = { low: 0, medium: 0, high: 0 };
    history.forEach((h) => {
      counts[h.predicted_class as VolatilityClass]++;
    });
    const total = history.length || 1;
    return {
      low: Math.round((counts.low / total) * 100),
      medium: Math.round((counts.medium / total) * 100),
      high: 100 - Math.round((counts.low / total) * 100) - Math.round((counts.medium / total) * 100),
    };
  }, [history]);

  // Current streak
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

  // Forecast band width % and range
  const forecastBand = useMemo(() => {
    if (!latestPred || prices.length === 0) return null;
    const lastClose = prices[prices.length - 1].close;
    const sigAnn =
      latestPred.p_low * VOL_ANN.low +
      latestPred.p_medium * VOL_ANN.medium +
      latestPred.p_high * VOL_ANN.high;
    const sig5 = (sigAnn / Math.sqrt(252)) * Math.sqrt(5);
    const hi = lastClose * Math.exp(1.65 * sig5);
    const lo = lastClose * Math.exp(-1.65 * sig5);
    const pct = ((hi / lastClose - 1) * 100).toFixed(1);
    const fmt = (v: number) => v >= 1000 ? `$${(v / 1000).toFixed(1)}k` : `$${v.toFixed(2)}`;
    return { pct, range: `${fmt(lo)} – ${fmt(hi)}` };
  }, [latestPred, prices]);

  // 6M return
  const periodReturn = useMemo(() => {
    if (prices.length < 2) return null;
    const cutoff = new Date();
    cutoff.setDate(cutoff.getDate() - 180);
    const old = prices.find((p) => new Date(p.date) >= cutoff);
    if (!old) return null;
    const last = prices[prices.length - 1];
    const ret = ((last.close / old.close - 1) * 100).toFixed(1);
    return { value: parseFloat(ret), label: ret };
  }, [prices]);

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
              regimes={history}
              latestPred={latestPred}
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
              <div className="stat-label">6M return</div>
              <div
                className="stat-value"
                style={{ color: periodReturn && periodReturn.value >= 0 ? 'var(--low)' : 'var(--high)' }}
              >
                {periodReturn ? `${periodReturn.value >= 0 ? '+' : ''}${periodReturn.label}%` : '—'}
              </div>
              <div className="stat-sub">vs 6 months ago</div>
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
```

- [ ] **Step 3: Verify TypeScript compiles**

```bash
cd /home/manidmt/TFM/quant-risk-tfm/apps/web && npm run build 2>&1 | tail -8
```

Expected: build succeeds with no TypeScript errors.

- [ ] **Step 4: Commit**

```bash
cd /home/manidmt/TFM/quant-risk-tfm
git add apps/web/src/pages/AssetHistory.tsx apps/web/src/pages/AssetHistory.css
git commit -m "feat(web): redesign AssetHistory — price chart, forecast band, stats panel"
```
