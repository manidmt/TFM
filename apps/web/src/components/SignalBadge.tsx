import type { VolatilityClass } from '../api/types';
import './SignalBadge.css';

const LABEL: Record<VolatilityClass, string> = {
  low: 'Low',
  medium: 'Medium',
  high: 'High',
};

interface Props {
  signal: VolatilityClass | null;
  size?: 'sm' | 'md';
}

export default function SignalBadge({ signal, size = 'md' }: Props) {
  if (!signal) return <span className="signal-badge signal-none">—</span>;
  return (
    <span className={`signal-badge signal-${signal} signal-${size}`}>
      {LABEL[signal]}
    </span>
  );
}
