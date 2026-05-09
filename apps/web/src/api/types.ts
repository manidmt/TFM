// Shared API response types mirroring the FastAPI response schemas.

export type VolatilityClass = 'low' | 'medium' | 'high';

export interface AssetOut {
  asset_id: string;
  source_ticker: string;
  display_name: string;
}

export interface PredictionOut {
  asset_id: string;
  forecast_date: string;
  predicted_class: VolatilityClass;
  p_low: number;
  p_medium: number;
  p_high: number;
}

export interface PricePoint {
  date: string;   // ISO date string e.g. "2025-10-01"
  close: number;
}

export interface RegimePoint {
  date: string;
  regime: VolatilityClass;
}

export interface VolProfile {
  asset_id: string;
  vol_5d_low: number;
  vol_5d_medium: number;
  vol_5d_high: number;
}

export interface TickerOut {
  ticker: string;
  display_name: string;
  asset_class: string;
  proxy_asset_id: string;
}

export interface UserOut {
  user_id: string;
  email: string;
  role: string;
  must_change_password: boolean;
}

export interface PositionIn {
  label: string;
  weight_pct: number;
  proxy_asset_id: string;
}

export interface PositionOut {
  label: string;
  weight_pct: number;
  proxy_asset_id: string;
}

export interface PortfolioSummaryOut {
  portfolio_id: string;
  name: string;
  position_count: number;
  total_weight_pct: number;
  last_signal: VolatilityClass | null;
  last_analysis_at: string | null;
}

export interface PortfolioDetailOut {
  portfolio_id: string;
  name: string;
  positions: PositionOut[];
  last_signal: VolatilityClass | null;
  last_analysis_at: string | null;
}

export interface PositionAnalysisOut {
  label: string;
  weight_pct: number;
  normalized_weight: number;
  proxy_asset_id: string;
  predicted_class: VolatilityClass | null;
  p_low: number | null;
  p_medium: number | null;
  p_high: number | null;
  forecast_date: string | null;
  adj_p_low: number | null;
  adj_p_medium: number | null;
  adj_p_high: number | null;
  adj_predicted_class: VolatilityClass | null;
  vol_ratio: number | null;
  has_risk_adjustment: boolean;
}

export interface AssetGroupOut {
  asset_id: string;
  aggregate_weight: number;
  predicted_class: VolatilityClass | null;
  p_low: number | null;
  p_medium: number | null;
  p_high: number | null;
  forecast_date: string | null;
  position_labels: string[];
}

export interface PortfolioAnalysisOut {
  portfolio_id: string;
  portfolio_name: string;
  total_weight_pct: number;
  positions: PositionAnalysisOut[];
  asset_groups: AssetGroupOut[];
  portfolio_p_low: number;
  portfolio_p_medium: number;
  portfolio_p_high: number;
  portfolio_signal: VolatilityClass | null;
  missing_predictions: string[];
}

export interface PortfolioVaROut {
  var_95: number;
  var_99: number;
  cvar_95: number;
  regime_filter: string;
  scenario_count: number;
  low_sample_warning: boolean;
  histogram: number[];
}

// Admin user management
export interface UserAdminOut {
  user_id: string;
  email: string;
  role: string;
  is_active: boolean;
  is_approved: boolean;
  created_at: string;
}

// Admin ops types
export interface OpsSummaryOut {
  asset_count: number;
  fresh_count: number;
  stale_count: number;
  failed_count: number;
  assets: AssetStatusOut[];
}

export interface AssetStatusOut {
  asset_id: string;
  run_status: string | null;
  last_forecast_date: string | null;
  last_run_at: string | null;
  consecutive_failures: number | null;
}

export interface PromotionEventOut {
  event_id: string;
  asset_id: string;
  bundle_version: string;
  status: string;
  promoted_at: string;
  previous_version: string | null;
  validation_errors: string | null;
}

// Chat
export interface ChatMessage {
  role: 'user' | 'assistant';
  content: string;
}

export interface ChatToolEvent {
  name: string;
  status: 'start' | 'end';
}

export interface ChatSSEEvent {
  type: 'token' | 'tool' | 'done';
  content?: string;
  name?: string;
  status?: string;
}
