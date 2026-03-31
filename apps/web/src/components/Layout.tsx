import { Link, NavLink, useNavigate } from 'react-router-dom';
import { useAuth } from '../context/AuthContext';
import './Layout.css';

export default function Layout({ children }: { children: React.ReactNode }) {
  const { user, logout } = useAuth();
  const navigate = useNavigate();

  async function handleLogout() {
    await logout();
    navigate('/');
  }

  return (
    <div className="layout">
      <header className="layout-header">
        <nav className="layout-nav container">
          <Link to="/" className="layout-wordmark">Volatility Regimes</Link>
          <div className="layout-nav-links">
            <NavLink to="/predictions" className={({ isActive }) => isActive ? 'nav-link active' : 'nav-link'}>
              Predictions
            </NavLink>
            <NavLink to="/app" className={({ isActive }) => isActive ? 'nav-link active' : 'nav-link'}>
              Portfolio
            </NavLink>
            {user?.role === 'admin' && (
              <NavLink to="/ops" className={({ isActive }) => isActive ? 'nav-link active' : 'nav-link'}>
                Ops
              </NavLink>
            )}
          </div>
          <div className="layout-nav-auth">
            {user ? (
              <button className="btn-ghost" onClick={handleLogout}>Sign out</button>
            ) : (
              <Link to="/login" className="btn-ghost">Log in</Link>
            )}
          </div>
        </nav>
      </header>

      <main className="layout-main">
        {children}
      </main>

      <footer className="layout-footer">
        <div className="container footer-inner">
          <span className="footer-name">Manuel Díaz-Meco Terrés</span>
          <span className="footer-sep">·</span>
          <span className="footer-desc">Master's Thesis in AI &amp; Analytics</span>
        </div>
      </footer>
    </div>
  );
}
