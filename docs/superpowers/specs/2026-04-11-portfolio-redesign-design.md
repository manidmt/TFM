# Portfolio Redesign — Design Spec

## Summary

Redesign the portfolio UI from a plain text-based layout to a visually rich, card-based interface. Enrich the backend API to support summary metadata (position count, total weight, last analysis signal) so the portfolio list provides useful context without requiring navigation.

## Backend Changes

### DB Schema — `portfolios` table

Add two nullable columns (no Alembic — direct ALTER TABLE):

```sql
ALTER TABLE portfolios ADD COLUMN last_analysis_signal VARCHAR(20) DEFAULT NULL;
ALTER TABLE portfolios ADD COLUMN last_analysis_at TIMESTAMP WITH TIME ZONE DEFAULT NULL;
```

### ORM Model — `Portfolio` (models.py)

Add two mapped columns:
- `last_analysis_signal: Mapped[str | None]` — one of "low", "medium", "high", or null
- `last_analysis_at: Mapped[datetime | None]` — UTC timestamp

### API — `PortfolioSummaryOut` (private.py)

Enrich from:
```python
class PortfolioSummaryOut(BaseModel):
    portfolio_id: str
    name: str
```

To:
```python
class PortfolioSummaryOut(BaseModel):
    portfolio_id: str
    name: str
    position_count: int
    total_weight_pct: float
    last_signal: str | None
    last_analysis_at: str | None
```

`list_portfolios` endpoint computes `position_count` and `total_weight_pct` from the already-loaded `positions` relationship. `last_signal` and `last_analysis_at` come from the new columns.

### API — `analyze_portfolio_endpoint` (private.py)

After computing the analysis result, persist the signal back:
```python
portfolio.last_analysis_signal = result.portfolio_signal
portfolio.last_analysis_at = datetime.now(timezone.utc)
db.commit()
```

## Frontend Changes

### Types — `types.ts`

Update `PortfolioSummaryOut`:
```ts
interface PortfolioSummaryOut {
  portfolio_id: string;
  name: string;
  position_count: number;
  total_weight_pct: number;
  last_signal: VolatilityClass | null;
  last_analysis_at: string | null;
}
```

### Constants (in PortfolioDetail.tsx)

```ts
const PROXY_LABEL: Record<string, string> = {
  us_equities: 'US', euro_equities: 'EU', bitcoin: 'Crypto',
  long_us_treasuries: 'Bond L', short_us_treasuries: 'Bond S', gold: 'Cmdty',
};

const PROXY_COLOR: Record<string, string> = {
  us_equities: '#6366f1', euro_equities: '#8b5cf6', bitcoin: '#f59e0b',
  long_us_treasuries: '#06b6d4', short_us_treasuries: '#14b8a6', gold: '#eab308',
};
```

### Utility — `timeAgo(isoString: string): string`

~10-line function returning "just now", "5 min ago", "2 hours ago", "3 days ago". No external library.

### PortfolioList.tsx

Each portfolio item becomes a card showing:
- Name + "{N} positions · {weight}%"
- SignalBadge with last analysis signal (or "No analysis" if null)
- Subtitle "Analysed {timeAgo}" if available

### PortfolioDetail.tsx — Header

Add subtitle below portfolio name:
- "{N} positions · Last analysed {timeAgo}" (or "Never analysed" if null)

Uses `last_analysis_at` from the portfolio detail response. This requires adding `last_analysis_signal` and `last_analysis_at` to `PortfolioDetailOut` as well.

### PortfolioDetail.tsx — Positions (read mode)

Replace plain text rows with cards:
- Ticker name + asset class badge derived from `proxy_asset_id` via `PROXY_LABEL`
- Weight percentage with proportional bar (width = weight_pct%)
- Proxy asset name in subtle subtitle

### PortfolioDetail.tsx — Weight indicator (edit mode)

Replace the small pill with a full-width progress bar:
- Bar fills proportionally (green at 100%, amber below, red above)
- Text: "{total}% / 100%" + "{remaining}% remaining"

### PortfolioDetail.tsx — AnalysisResult component

Complete redesign:

**Row 1 — Summary cards (3-col grid):**
- Portfolio signal (SignalBadge)
- Risk concentration: "{N} assets, {M} proxy groups"
- Total weight (monospace)

**Row 2 — Two-column grid:**
- Left: SVG donut chart of allocation grouped by proxy_asset_id (using `asset_groups` from API). Each segment colored by `PROXY_COLOR`. Legend with proxy name, aggregate weight, and position labels.
- Right: Three individual probability bars (P(low), P(medium), P(high)) with colored fills and percentages.

**Row 3 — Position cards:**
- Horizontal grid of cards (one per position)
- Left border colored by predicted_class signal
- Ticker + signal badge + weight/proxy subtitle
- Mini horizontal prob bar (3-segment)

**Row 4 — Missing predictions:**
- Warning box listing positions without predictions, including the proxy they mapped to.

The donut is pure SVG using `stroke-dasharray` on `<circle>` elements — no charting library.

## Files Modified

| File | Action |
|------|--------|
| `src/quant_risk/prod/auth/models.py` | Add 2 columns to Portfolio |
| `src/quant_risk/prod/api/routers/private.py` | Enrich PortfolioSummaryOut + PortfolioDetailOut, persist analysis signal |
| `apps/web/src/api/types.ts` | Update PortfolioSummaryOut |
| `apps/web/src/pages/app/PortfolioList.tsx` | Redesign list items |
| `apps/web/src/pages/app/PortfolioList.css` | New card styles |
| `apps/web/src/pages/app/PortfolioDetail.tsx` | Position cards, header, weight bar, AnalysisResult redesign |
| `apps/web/src/pages/app/PortfolioDetail.css` | New styles |

## Out of Scope

- PDF export
- Analysis history / versioning
- Regime change alerts
- Chatbot integration (next phase)
- Mobile-specific responsive breakpoints (existing flex/grid is sufficient)
