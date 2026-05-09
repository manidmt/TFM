import './ProbBar.css';

interface Props {
  p_low: number | null;
  p_medium: number | null;
  p_high: number | null;
}

function pct(v: number | null) {
  if (v == null) return '—';
  return (v * 100).toFixed(1) + '%';
}

export default function ProbBar({ p_low, p_medium, p_high }: Props) {
  const hasData = p_low != null && p_medium != null && p_high != null;
  return (
    <div className="prob-bar-wrapper">
      {hasData && (
        <div className="prob-bar-track">
          <div className="prob-bar-segment seg-low" style={{ width: `${(p_low! * 100).toFixed(1)}%` }} />
          <div className="prob-bar-segment seg-medium" style={{ width: `${(p_medium! * 100).toFixed(1)}%` }} />
          <div className="prob-bar-segment seg-high" style={{ width: `${(p_high! * 100).toFixed(1)}%` }} />
        </div>
      )}
      <div className="prob-bar-labels">
        <span className="prob-label low">{pct(p_low)}</span>
        <span className="prob-label medium">{pct(p_medium)}</span>
        <span className="prob-label high">{pct(p_high)}</span>
      </div>
    </div>
  );
}
