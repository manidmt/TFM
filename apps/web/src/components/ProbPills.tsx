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
