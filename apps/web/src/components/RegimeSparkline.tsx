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

function draw(canvas: HTMLCanvasElement, regimes: VolatilityClass[]) {
  const ctx = canvas.getContext('2d');
  if (!ctx) return;

  const dpr = window.devicePixelRatio || 1;
  const W = canvas.offsetWidth;
  const H = 20;

  canvas.width = W * dpr;
  canvas.height = H * dpr;
  ctx.scale(dpr, dpr);

  const bw = W / regimes.length;
  ctx.clearRect(0, 0, W, H);
  ctx.globalAlpha = 0.75;

  regimes.forEach((regime, i) => {
    ctx.fillStyle = COLOR[regime];
    ctx.beginPath();
    if (ctx.roundRect) {
      ctx.roundRect(i * bw + 0.5, 0, bw - 1, H, 1);
    } else {
      ctx.rect(i * bw + 0.5, 0, bw - 1, H);
    }
    ctx.fill();
  });

  ctx.globalAlpha = 1;
}

export default function RegimeSparkline({ regimes }: Props) {
  const canvasRef = useRef<HTMLCanvasElement>(null);

  useEffect(() => {
    const canvas = canvasRef.current;
    if (!canvas) return;

    // Draw immediately in case layout is already complete
    draw(canvas, regimes);

    // Redraw on resize (handles first-render zero-width case)
    const ro = new ResizeObserver(() => draw(canvas, regimes));
    ro.observe(canvas);
    return () => ro.disconnect();
  }, [regimes]);

  if (regimes.length === 0) return null;

  return (
    <canvas
      ref={canvasRef}
      aria-label="Volatility regime history"
      role="img"
      style={{ display: 'block', width: '100%', height: '20px', borderRadius: '3px' }}
    />
  );
}
