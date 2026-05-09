import { Link } from 'react-router-dom';
import { useAuth } from '../context/AuthContext';
import ChartCanvas from '../components/ChartCanvas';
import './Landing.css';

function greeting(): string {
  const h = new Date().getHours();
  if (h < 12) return 'Good morning';
  if (h < 18) return 'Good afternoon';
  return 'Good evening';
}

export default function Landing() {
  const { user } = useAuth();

  return (
    <div className="landing">
      <section className="hero">
        <ChartCanvas />
        <div className="hero-content container">
          {user ? (
            <>
              <p className="hero-greeting">{greeting()}</p>
              <p className="hero-user">{user.email}</p>
            </>
          ) : null}
          <h1 className="hero-title">Volatility Regimes</h1>
          <p className="hero-thesis">Master's Thesis · AI &amp; Analytics</p>
          <p className="hero-desc">
            Volatility regime predictions for S&amp;P 500, Euro Stocks, Bitcoin, Gold, and US Treasuries —
            combining econometric models, news signals, and tabular machine learning.
          </p>

          <div className="hero-actions">
            <Link to="/predictions" className="action-block">
              <span className="action-label">View predictions</span>
              <span className="action-arrow">→</span>
            </Link>
            <Link to="/app" className="action-block">
              <span className="action-label">Analyze portfolio</span>
              <span className="action-arrow">→</span>
            </Link>
          </div>
        </div>
      </section>

      <section className="about container">
        <div className="about-inner">
          <p className="about-text">
            This tool is the live interface of a Master's thesis in AI &amp; Analytics.
            It produces daily short-term volatility regime forecasts — <em>low</em>,{' '}
            <em>medium</em>, or <em>high</em> — for five major asset classes (S&amp;P 500,
            Euro Stocks, Bitcoin, Gold, and US Treasuries), and allows registered users
            to analyse their own portfolios against those signals.
          </p>
          <Link to="/methodology" className="about-methodology-link">
            Learn about the methodology →
          </Link>
        </div>
      </section>
    </div>
  );
}
