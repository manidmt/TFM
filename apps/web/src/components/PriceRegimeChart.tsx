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

      // X-axis labels — adaptive format based on range
      if (points.length > 1) {
        ctx.textAlign = 'center';
        const step = Math.max(1, Math.floor(N / 5));
        for (let i = 0; i < N; i += step) {
          // Add T12:00:00 to avoid UTC midnight shifting day by timezone offset
          const d = new Date(points[i].date + 'T12:00:00');
          const label =
            range === '1M' || range === '3M'
              ? d.toLocaleDateString('en-US', { month: 'short', day: 'numeric' })
              : d.toLocaleDateString('en-US', { month: 'short' });
          ctx.fillText(label, px(i), PAD.top + cH + 14);
        }
      }
    }

    render();
    const ro = new ResizeObserver(render);
    ro.observe(canvas);
    return () => ro.disconnect();
  }, [points, latestPred, range]);

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
