import { useEffect, useRef } from 'react';

// Seeded LCG random — reproducible per line so it looks the same on every mount
function makeLCG(seed: number) {
  let s = seed >>> 0;
  return () => {
    s = (Math.imul(s, 1664525) + 1013904223) >>> 0;
    return s / 0xffffffff;
  };
}

// Box-Muller transform for normal-distributed returns
function makeNormal(rand: () => number) {
  let spare: number | null = null;
  return () => {
    if (spare !== null) { const v = spare; spare = null; return v; }
    let u: number, v: number, s: number;
    do { u = rand() * 2 - 1; v = rand() * 2 - 1; s = u * u + v * v; } while (s >= 1 || s === 0);
    const f = Math.sqrt(-2 * Math.log(s) / s);
    spare = v * f;
    return u * f;
  };
}

interface Series {
  pts: { x: number; y: number }[];
  color: string;
  yRatio: number;
  seed: number;
}

const LINE_DEFS = [
  { color: '#4f7a64', seed: 1001, drift:  0.00012, vol: 0.007 },  // S&P 500
  { color: '#6366f1', seed: 1729, drift:  0.00008, vol: 0.009 },  // Euro equities
  { color: '#a57a2a', seed: 2777, drift:  0.00020, vol: 0.016 },  // BTC
  { color: '#d4a34a', seed: 3301, drift:  0.00005, vol: 0.012 },  // Gold
  { color: '#9a5246', seed: 4999, drift: -0.00004, vol: 0.008 },  // Treasuries
  { color: '#7c8a6e', seed: 6173, drift:  0.00010, vol: 0.010 },  // Broad market
];

const OPACITY      = 0.18;
const SCROLL_SPEED = 0.55;
const STEP         = 4;
const EXTRA_FACTOR = 2.8;

function buildSeries(W: number, H: number): Series[] {
  const n = LINE_DEFS.length;
  return LINE_DEFS.map((def, i) => {
    const rand = makeLCG(def.seed);
    const normal = makeNormal(rand);
    const totalW = W * EXTRA_FACTOR;

    // Distribute base lines evenly with padding
    const yRatio = 0.15 + (0.70 * i) / (n - 1);
    const baseY  = yRatio * H;

    // Boundaries: lines stay within 8%-92% of canvas height
    const yMin = H * 0.08;
    const yMax = H * 0.92;

    let price = 100;
    let localVol = def.vol;

    const pts: { x: number; y: number }[] = [];

    for (let x = 0; x <= totalW; x += STEP) {
      const z = normal();

      // GARCH-like volatility clustering
      localVol = def.vol * 0.7 + localVol * 0.3 + Math.abs(z) * def.vol * 0.08;

      // Elastic mean reversion: stronger pull as y approaches boundaries
      const logDev = Math.log(price / 100);
      const y = baseY - logDev * H * 0.9;
      let reversion = 0;
      if (y < yMin + H * 0.10) {
        // Approaching top — push price down (y down = price down)
        const urgency = 1 - (y - yMin) / (H * 0.10);
        reversion = -Math.abs(localVol) * urgency * 1.5;
      } else if (y > yMax - H * 0.10) {
        // Approaching bottom — push price up
        const urgency = 1 - (yMax - y) / (H * 0.10);
        reversion = Math.abs(localVol) * urgency * 1.5;
      }
      // Gentle pull toward baseY at all times
      reversion += (baseY - y) / H * 0.003;

      const ret = def.drift + reversion + localVol * z;
      price *= Math.exp(ret);

      const finalY = baseY - Math.log(price / 100) * H * 0.9;
      pts.push({ x, y: finalY });
    }
    return { pts, color: def.color, yRatio, seed: def.seed };
  });
}

export default function ChartCanvas() {
  const canvasRef = useRef<HTMLCanvasElement>(null);

  useEffect(() => {
    const canvas = canvasRef.current;
    if (!canvas) return;

    const ctx = canvas.getContext('2d')!;
    let rafId: number;
    let running  = true;
    let offset   = 0;
    let series: Series[] = [];

    function resize() {
      const parent = canvas!.parentElement!;
      canvas!.width  = parent.offsetWidth;
      canvas!.height = parent.offsetHeight;
      series = buildSeries(canvas!.width, canvas!.height);
      offset = 0;
    }

    resize();
    const ro = new ResizeObserver(resize);
    ro.observe(canvas.parentElement!);

    function draw() {
      if (!running) return;

      const W = canvas!.width;
      const H = canvas!.height;
      ctx.clearRect(0, 0, W, H);

      series.forEach(s => {
        ctx.beginPath();
        ctx.strokeStyle = s.color;
        ctx.lineWidth   = 1.5;
        ctx.globalAlpha = OPACITY;

        let started = false;
        for (const p of s.pts) {
          const dx = p.x - offset;
          if (dx < -STEP || dx > W + STEP) continue;
          if (!started) { ctx.moveTo(dx, p.y); started = true; }
          else ctx.lineTo(dx, p.y);
        }
        ctx.stroke();

        // Live dot at rightmost visible point
        const visible = s.pts.filter(p => {
          const dx = p.x - offset;
          return dx >= 0 && dx <= W;
        });
        if (visible.length) {
          const last = visible[visible.length - 1];
          ctx.beginPath();
          ctx.arc(last.x - offset, last.y, 3, 0, Math.PI * 2);
          ctx.fillStyle   = s.color;
          ctx.globalAlpha = OPACITY * 2.5;
          ctx.fill();
        }
      });

      ctx.globalAlpha = 1;

      // Fade mask — top and bottom 18% blend into the background colour
      const FADE = H * 0.18;
      const BG   = '#f7f4ee';

      const topGrad = ctx.createLinearGradient(0, 0, 0, FADE);
      topGrad.addColorStop(0, BG);
      topGrad.addColorStop(1, 'rgba(247,244,238,0)');
      ctx.fillStyle = topGrad;
      ctx.fillRect(0, 0, W, FADE);

      const botGrad = ctx.createLinearGradient(0, H - FADE, 0, H);
      botGrad.addColorStop(0, 'rgba(247,244,238,0)');
      botGrad.addColorStop(1, BG);
      ctx.fillStyle = botGrad;
      ctx.fillRect(0, H - FADE, W, FADE);

      // Advance scroll; rebuild when we've consumed the series
      offset += SCROLL_SPEED;
      const maxX = series[0]?.pts.at(-1)?.x ?? 0;
      if (offset >= maxX - W) {
        series = buildSeries(W, H);
        offset = 0;
      }

      rafId = requestAnimationFrame(draw);
    }

    draw();

    return () => {
      running = false;
      cancelAnimationFrame(rafId);
      ro.disconnect();
    };
  }, []);

  return (
    <canvas
      ref={canvasRef}
      aria-hidden="true"
      style={{
        position: 'absolute',
        inset: 0,
        width: '100%',
        height: '100%',
        pointerEvents: 'none',
      }}
    />
  );
}
