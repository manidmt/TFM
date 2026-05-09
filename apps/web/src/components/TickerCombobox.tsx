import { useEffect, useRef, useState } from 'react';
import type { TickerOut } from '../api/types';
import './TickerCombobox.css';

interface Props {
  tickers: TickerOut[];
  value: string;
  onChange: (ticker: string, proxyAssetId: string) => void;
}

const ASSET_CLASS_LABEL: Record<string, string> = {
  us_equity: 'US',
  euro_equity: 'EU',
  crypto: 'Crypto',
  bond_long: 'Bond L',
  bond_short: 'Bond S',
  commodity: 'Cmdty',
};

export default function TickerCombobox({ tickers, value, onChange }: Props) {
  const [query, setQuery] = useState(value);
  const [open, setOpen] = useState(false);
  const [activeIdx, setActiveIdx] = useState(-1);
  const wrapperRef = useRef<HTMLDivElement>(null);
  const listRef = useRef<HTMLUListElement>(null);

  // Sync external value changes
  useEffect(() => { setQuery(value); }, [value]);

  const filtered = query.length === 0
    ? []
    : tickers.filter((t) => {
        const q = query.toLowerCase();
        return t.ticker.toLowerCase().includes(q) || t.display_name.toLowerCase().includes(q);
      }).slice(0, 10);

  function select(t: TickerOut) {
    setQuery(t.ticker);
    onChange(t.ticker, t.proxy_asset_id);
    setOpen(false);
    setActiveIdx(-1);
  }

  function handleBlur(e: React.FocusEvent) {
    // Don't close if focus moves within the wrapper
    if (wrapperRef.current?.contains(e.relatedTarget as Node)) return;
    setOpen(false);
    // If typed text doesn't match current value, emit as custom label
    if (query !== value) {
      onChange(query, '');
    }
  }

  function handleKeyDown(e: React.KeyboardEvent) {
    if (!open || filtered.length === 0) {
      if (e.key === 'ArrowDown' && filtered.length > 0) {
        setOpen(true);
        setActiveIdx(0);
        e.preventDefault();
      }
      return;
    }
    if (e.key === 'ArrowDown') {
      e.preventDefault();
      setActiveIdx((i) => Math.min(i + 1, filtered.length - 1));
    } else if (e.key === 'ArrowUp') {
      e.preventDefault();
      setActiveIdx((i) => Math.max(i - 1, 0));
    } else if (e.key === 'Enter' && activeIdx >= 0) {
      e.preventDefault();
      select(filtered[activeIdx]);
    } else if (e.key === 'Escape') {
      setOpen(false);
    }
  }

  // Scroll active item into view
  useEffect(() => {
    if (activeIdx >= 0 && listRef.current) {
      const item = listRef.current.children[activeIdx] as HTMLElement | undefined;
      item?.scrollIntoView({ block: 'nearest' });
    }
  }, [activeIdx]);

  return (
    <div className="ticker-combobox" ref={wrapperRef}>
      <input
        className="field-input pos-label"
        placeholder="Search ticker…"
        value={query}
        onChange={(e) => {
          setQuery(e.target.value);
          setOpen(true);
          setActiveIdx(-1);
        }}
        onFocus={() => { if (query.length > 0) setOpen(true); }}
        onBlur={handleBlur}
        onKeyDown={handleKeyDown}
        role="combobox"
        aria-expanded={open && filtered.length > 0}
        autoComplete="off"
      />
      {open && filtered.length > 0 && (
        <ul className="ticker-dropdown" ref={listRef} role="listbox">
          {filtered.map((t, i) => (
            <li
              key={t.ticker}
              className={`ticker-option${i === activeIdx ? ' active' : ''}`}
              onMouseDown={(e) => { e.preventDefault(); select(t); }}
              role="option"
              aria-selected={i === activeIdx}
            >
              <span className="ticker-symbol">{t.ticker}</span>
              <span className="ticker-name">{t.display_name}</span>
              <span className="ticker-class">{ASSET_CLASS_LABEL[t.asset_class] ?? t.asset_class}</span>
            </li>
          ))}
        </ul>
      )}
    </div>
  );
}
