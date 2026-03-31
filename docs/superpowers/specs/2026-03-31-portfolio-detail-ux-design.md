# Portfolio Detail UX Improvements — Design Spec

## Goal

Fix two bugs in `PortfolioDetail.tsx` (broken proxy dropdown, wrong `AssetOut` type) and improve UX clarity for the weight field and read-only proxy display.

## Scope

Two files only:
- `apps/web/src/api/types.ts` — fix `AssetOut` interface
- `apps/web/src/pages/app/PortfolioDetail.tsx` — fix dropdown, weight UX, read-only display

No backend changes. No CSS structural changes.

## Changes

### 1. Fix `AssetOut` type (`types.ts`)

Current (wrong — does not match backend response):
```ts
interface AssetOut {
  asset_id: string;
  label: string;
  description: string;
}
```

Correct (matches `GET /api/public/assets` response):
```ts
interface AssetOut {
  asset_id: string;
  source_ticker: string;
  display_name: string;
}
```

### 2. Proxy dropdown options

Both dropdowns (edit mode and what-if mode) must render options as:

```
display_name — source_ticker
```

Example: `S&P 500 — GSPC`, `Bitcoin — BTC-USD`, `US Treasuries — TLT`

```tsx
{assets.map((a) => (
  <option key={a.asset_id} value={a.asset_id}>
    {a.display_name} — {a.source_ticker}
  </option>
))}
```

### 3. Weight input with `%` suffix

Wrap the `<input type="number">` in a relative-positioned div and overlay a `%` label:

```tsx
<div style={{ position: 'relative' }}>
  <input
    className="field-input pos-weight"
    type="number"
    ...
  />
  <span className="weight-suffix">%</span>
</div>
```

CSS for `.weight-suffix`: `position: absolute; right: 8px; top: 50%; transform: translateY(-50%); color: #64748b; font-size: 12px; pointer-events: none;`

Applied in both edit mode and what-if mode rows.

### 4. Weight sum indicator

Computed from active positions in state:

```ts
const totalWeight = positions.reduce((s, p) => s + (p.weight_pct || 0), 0);
const weightOk = Math.abs(totalWeight - 100) < 0.01;
```

Shown next to the "Positions" heading when in edit mode:

```tsx
<span className={weightOk ? 'weight-sum ok' : 'weight-sum warn'}>
  Total: {totalWeight.toFixed(1)}%{!weightOk && ` — ${(100 - totalWeight).toFixed(1)}% remaining`}
</span>
```

CSS: `.weight-sum.ok { color: #4ade80; } .weight-sum.warn { color: #f59e0b; }` (inline styles acceptable to avoid CSS file churn).

Same indicator for what-if mode, computed from `whatIfPositions`.

### 5. Read-only proxy display

In read-only position rows, resolve `proxy_asset_id` to its asset entry and display `display_name` + `source_ticker`:

```tsx
const asset = assets.find((a) => a.asset_id === pos.proxy_asset_id);
// ...
<span className="pos-proxy-text">
  {asset ? `${asset.display_name}` : pos.proxy_asset_id}
  {asset && <span className="proxy-ticker"> {asset.source_ticker}</span>}
</span>
```

`.proxy-ticker`: `font-family: monospace; font-size: 11px; color: #475569; margin-left: 4px;`

If the asset is not found in the loaded list, fall back to the raw `proxy_asset_id`.

## What does NOT change

- Backend endpoints and models
- CSS file structure (only inline styles added for new elements)
- Analysis result table (already shows `proxy_asset_id` which is acceptable there)
- Portfolio CRUD logic

## Testing

- `cd apps/web && npm run build` — must pass with zero TypeScript errors
- Manual: open a portfolio in edit mode → proxy dropdown shows `display_name — source_ticker` options → weight field shows `%` suffix → sum indicator appears → read-only mode shows human-readable proxy name
