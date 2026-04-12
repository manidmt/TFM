import { Link } from 'react-router-dom';
import './Methodology.css';

const ASSETS = [
  { ticker: '^GSPC', name: 'US Equities', cls: 'Equities', why: 'Broad US equity market benchmark (S&P 500)' },
  { ticker: '^STOXX50E', name: 'Euro Area Equities', cls: 'Equities', why: 'Major European equity exposure (EURO STOXX 50)' },
  { ticker: 'BTC-USD', name: 'Bitcoin', cls: 'Crypto', why: 'High-volatility digital asset with distinct regime dynamics' },
  { ticker: 'GLD', name: 'Gold', cls: 'Commodities', why: 'Traditional safe-haven and inflation hedge' },
  { ticker: 'TLT', name: 'Long US Treasuries', cls: 'Fixed Income', why: 'Duration-sensitive government bond proxy' },
  { ticker: 'SHY', name: 'Short US Treasuries', cls: 'Fixed Income', why: 'Low-duration safe asset, near-cash benchmark' },
];

const PIPELINE_STEPS = [
  { label: 'Raw Data', desc: 'Daily prices, macro indicators, news signals' },
  { label: 'Feature Engineering', desc: 'Realised vol, rolling stats, shock detection, news tone' },
  { label: 'Econometric Chain', desc: 'SARIMAX mean + GARCH/EGARCH/GJR/HAR variance' },
  { label: 'Tabular ML', desc: 'XGBoost or TabPFN classifier on merged features' },
  { label: 'Calibration & Gating', desc: 'Isotonic calibration, persistence gating, confidence thresholds' },
  { label: 'Prediction', desc: 'Calibrated probabilities for low / medium / high' },
];

const DATA_SOURCES = [
  {
    name: 'Market Prices',
    source: 'yfinance',
    desc: 'Daily close prices for all six assets. Realised volatility computed over rolling windows (5, 10, 21 days). Source of the core target variable.',
  },
  {
    name: 'Macroeconomic Indicators',
    source: 'FRED',
    desc: 'VIX, yield curve spreads, interest rates, net liquidity. Provides broader market context that price data alone cannot capture.',
  },
  {
    name: 'News Signals',
    source: 'GDELT GKG',
    desc: 'Global event tone, thematic volume, information shock detection. An alternative signal source independent of price action.',
  },
];

export default function Methodology() {
  return (
    <div className="methodology">
      {/* 1. Overview */}
      <section className="meth-hero">
        <div className="container">
          <h1 className="meth-title">Methodology</h1>
          <p className="meth-subtitle">
            This platform predicts 5-day forward volatility regimes for six financial assets
            using a chained econometric-to-machine-learning pipeline. It combines classical
            time-series models, macroeconomic indicators, and news signals to classify upcoming
            volatility as <em>low</em>, <em>medium</em>, or <em>high</em>.
          </p>
          <Link to="/" className="meth-back">← Back to home</Link>
        </div>
      </section>

      {/* 2. Asset Universe */}
      <section className="meth-section container">
        <h2 className="meth-section-title">Asset Universe</h2>
        <div className="asset-universe-grid">
          {ASSETS.map((a) => (
            <div key={a.ticker} className="asset-universe-card">
              <span className="auc-ticker">{a.ticker}</span>
              <span className="auc-name">{a.name}</span>
              <span className="auc-class">{a.cls}</span>
              <p className="auc-why">{a.why}</p>
            </div>
          ))}
        </div>
      </section>

      {/* 3. Pipeline */}
      <section className="meth-section container">
        <h2 className="meth-section-title">Pipeline</h2>
        <div className="pipeline-flow">
          {PIPELINE_STEPS.map((step, i) => (
            <div key={step.label} className="pipeline-step-wrapper">
              {i > 0 && <span className="pipeline-arrow">→</span>}
              <div className="pipeline-step">
                <span className="pipeline-step-label">{step.label}</span>
                <span className="pipeline-step-desc">{step.desc}</span>
              </div>
            </div>
          ))}
        </div>
      </section>

      {/* 4. Models */}
      <section className="meth-section container">
        <h2 className="meth-section-title">Models</h2>

        <div className="model-subsections">
          <div className="model-block">
            <h3 className="model-block-title">Econometric Layer</h3>
            <p>
              First stage of the pipeline. For each asset, multiple chained configurations
              are evaluated: <strong>SARIMAX</strong> for the conditional mean (varying orders
              and optional macro exogenous variables), paired with one of four volatility
              models — <strong>GARCH</strong>, <strong>EGARCH</strong> (exponential),{' '}
              <strong>GJR-GARCH</strong> (threshold/asymmetric), or <strong>HAR-RV</strong>{' '}
              (heterogeneous autoregressive realised volatility). Each is fitted with different
              distributional assumptions (Student's t, skewed-t) and aggregation methods.
              The "promising" profile evaluates 8 structural configurations per asset; the
              "full" profile evaluates 32.
            </p>
          </div>

          <div className="model-block">
            <h3 className="model-block-title">Tabular ML Layer</h3>
            <p>
              Econometric features — conditional mean, variance, standardised residuals — are
              merged with macro and news features, then fed into a supervised classifier.{' '}
              <strong>XGBoost</strong> provides a robust, well-understood gradient-boosting
              baseline. <strong>TabPFN</strong>, a prior-data fitted network, is a neural
              few-shot learner that generalises well with limited training windows — more novel,
              and frequently selected by the sweep runner.
            </p>
          </div>

          <div className="model-block">
            <h3 className="model-block-title">Sweep Runner &amp; Model Selection</h3>
            <p>
              The system does not hand-pick a model. A systematic sweep explores the full
              combination space using <strong>Optuna</strong> (TPE sampler) with independent
              studies per asset, tracked in <strong>MLflow</strong>. The search space — econometric
              variant × tabular model × hyperparameters — yields <strong>100+ combinations</strong>{' '}
              per asset. Each trial runs a complete walk-forward evaluation across multiple temporal
              folds. The optimisation target is the <strong>robust score</strong>: mean delta
              macro F1 vs. the persistence baseline, penalised for instability across folds and
              rewarded for correctly predicting regime transitions.
            </p>
          </div>

          <div className="model-block">
            <h3 className="model-block-title">Blending, Calibration &amp; Gating</h3>
            <p>
              The raw prediction passes through three post-processing layers.{' '}
              <strong>Blending</strong> combines the econometric chain and tabular model
              predictions with a confidence-adaptive weight.{' '}
              <strong>Isotonic calibration</strong> maps raw probabilities to well-calibrated
              ones. <strong>Persistence gating</strong> falls back to the persistence baseline
              when model confidence is low, optionally modulated by a{' '}
              <strong>GKG change detector</strong> — a binary classifier predicting regime
              transitions from news signals.
            </p>
          </div>
        </div>
      </section>

      {/* 5. Data Sources */}
      <section className="meth-section container">
        <h2 className="meth-section-title">Data Sources</h2>
        <div className="data-sources-grid">
          {DATA_SOURCES.map((ds) => (
            <div key={ds.name} className="data-source-card">
              <span className="ds-name">{ds.name}</span>
              <span className="ds-source">{ds.source}</span>
              <p className="ds-desc">{ds.desc}</p>
            </div>
          ))}
        </div>
      </section>

      {/* 6. Regime Classification */}
      <section className="meth-section container">
        <h2 className="meth-section-title">Regime Classification</h2>
        <p className="meth-body">
          Realised volatility values over the training window are sorted and split at the
          33rd and 66th percentiles, producing three classes:
        </p>
        <div className="regime-bars">
          <div className="regime-bar">
            <div className="regime-bar-fill regime-low" style={{ width: '33.3%' }} />
            <div className="regime-bar-fill regime-medium" style={{ width: '33.4%' }} />
            <div className="regime-bar-fill regime-high" style={{ width: '33.3%' }} />
          </div>
          <div className="regime-labels">
            <span className="regime-label low">Low (0–33%)</span>
            <span className="regime-label medium">Medium (33–66%)</span>
            <span className="regime-label high">High (66–100%)</span>
          </div>
        </div>
        <p className="meth-body meth-body-muted">
          Thresholds are recalculated per walk-forward fold to avoid lookahead bias.
        </p>
      </section>

      {/* 7. Evaluation */}
      <section className="meth-section container">
        <h2 className="meth-section-title">Evaluation</h2>
        <p className="meth-body">
          A model must outperform the <strong>persistence baseline</strong> — always predicting
          the most recently observed regime. Key metrics:
        </p>
        <ul className="eval-list">
          <li>
            <strong>Delta macro F1 vs. persistence</strong> — primary metric. How much better
            is the model than simply repeating the last regime?
          </li>
          <li>
            <strong>High-vol recall</strong> — does the model catch high-volatility regimes?
            Missing these is costly for risk management.
          </li>
          <li>
            <strong>Robust score</strong> — cross-fold aggregate penalised for instability and
            rewarded for correct transition predictions.
          </li>
          <li>
            <strong>Walk-forward validation</strong> — train and evaluate on advancing temporal
            windows to simulate real deployment with no data leakage.
          </li>
        </ul>
      </section>

      {/* 8. Infrastructure */}
      <section className="meth-section container">
        <h2 className="meth-section-title">Infrastructure</h2>
        <div className="infra-diagram">
          <div className="infra-node">
            <span className="infra-node-title">Workstation</span>
            <ul className="infra-node-list">
              <li>Sweep runner</li>
              <li>Model training</li>
              <li>MLflow tracking</li>
              <li>Heavy compute</li>
            </ul>
          </div>
          <div className="infra-arrow">
            <span className="infra-arrow-line" />
            <span className="infra-arrow-label">push predictions</span>
          </div>
          <div className="infra-node">
            <span className="infra-node-title">Raspberry Pi 5</span>
            <ul className="infra-node-list">
              <li>FastAPI + Postgres</li>
              <li>Nginx + Cloudflare</li>
              <li>DuckDB (serving)</li>
              <li>24/7 low-power</li>
            </ul>
          </div>
        </div>
        <p className="meth-body">
          The <strong>workstation</strong> runs sweeps, trains models, and evaluates walk-forward
          folds. The best model configuration is promoted weekly. The{' '}
          <strong>Raspberry Pi 5</strong> serves the web application and API around the clock —
          it does not run inference, only serves pre-computed predictions from DuckDB. Daily
          predictions are pushed via an authenticated internal endpoint, and a systemd timer
          runs data ingestion (prices, macro, news).
        </p>
      </section>

      {/* 9. Limitations & Disclaimer */}
      <section className="meth-section container">
        <div className="meth-disclaimer-box">
          <h2 className="meth-section-title">Limitations &amp; Disclaimer</h2>
          <ul className="disclaimer-list">
            <li>This is an academic research tool, not financial advice.</li>
            <li>Predictions are probabilistic — they express tendency, not certainty.</li>
            <li>Past performance does not guarantee future results.</li>
            <li>
              Models are trained on data from 2015 onwards and may not capture unprecedented
              regime dynamics.
            </li>
          </ul>
        </div>
      </section>
    </div>
  );
}
