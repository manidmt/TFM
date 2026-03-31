import { useEffect, useRef } from 'react';

// Seeded LCG random — reproducible per line so it looks the same on every mount
function makeLCG(seed: number) {
  let s = seed >>> 0;
  return () => {
    s = (Math.imul(s, 1664525) + 1013904223) >>> 0;
    return s / 0xffffffff;
  };
}

interface Series {
  pts: { x: number; y: number }[];
  color: string;
  yRatio: number;
  seed: number;
}

const LINE_DEFS = [
  { color: '#4f7a64', yRatio: 0.28, seed: 1001 }, // S&P 500 — low/green
  { color: '#a57a2a', yRatio: 0.54, seed: 2777 }, // BTC    — medium/amber
  { color: '#9a5246', yRatio: 0.76, seed: 4999 }, // TLT    — high/red
];

const OPACITY      = 0.20;
const SCROLL_SPEED = 0.55;   // px per frame — slow drift
const STEP         = 4;      // x distance between points
const EXTRA_FACTOR = 2.8;    // how many screen-widths to pre-generate

function buildSeries(W: number, H: number): Series[] {
  return LINE_DEFS.map(def => {
    const rand = makeLCG(def.seed);
    const totalW = W * EXTRA_FACTOR;
    const baseY  = def.yRatio * H;
    const amp    = H * 0.11;        // amplitude scales with canvas height

    const pts: { x: number; y: number }[] = [];
    let y = baseY;
    let vel = 0;

    for (let x = 0; x <= totalW; x += STEP) {
      vel = vel * 0.90 + (rand() - 0.5) * amp * 0.36;
      // occasional spike — simulates vol event
      if (rand() < 0.035) vel += (rand() - 0.5) * amp * 1.8;
      y += vel;
      // soft mean reversion
      y += (baseY - y) * 0.035;
      pts.push({ x, y });
    }
    return { pts, color: def.color, yRatio: def.yRatio, seed: def.seed };
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
