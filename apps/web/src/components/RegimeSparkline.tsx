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
