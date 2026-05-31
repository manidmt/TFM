import { Link, NavLink, useNavigate } from 'react-router-dom';
import { useAuth } from '../context/AuthContext';
import { useEffect, useState } from 'react';
import { api } from '../api/client';
import type { UserAdminOut } from '../api/types';
import './Layout.css';
import ChatDrawer from './ChatDrawer';

export default function Layout({ children }: { children: React.ReactNode }) {
  const { user, logout } = useAuth();
  const navigate = useNavigate();
  const [pendingCount, setPendingCount] = useState(0);

  useEffect(() => {
    if (user?.role !== 'admin') return;
    api.get<UserAdminOut[]>('/api/admin/users/pending')
      .then(data => setPendingCount(data.length))
      .catch(() => {});
  }, [user]);

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
                {pendingCount > 0 && (
                  <span className="nav-pending-badge">{pendingCount}</span>
                )}
              </NavLink>
            )}
          </div>
          <div className="layout-nav-auth">
            {user ? (
              <button className="btn-ghost" onClick={handleLogout}>Sign out</button>
            ) : (
              <Link to="/login" className="btn-ghost">Sign in</Link>
            )}
          </div>
        </nav>
      </header>

      <main className="layout-main">
        {children}
      </main>

      <footer className="layout-footer">
        <div className="container footer-inner">
          <a
            className="footer-name"
            href="https://www.linkedin.com/in/manuel-diaz-meco-terres/"
            target="_blank"
            rel="noopener noreferrer"
          >Manuel Díaz-Meco Terrés</a>
          <span className="footer-sep">·</span>
          <a
            className="footer-desc"
            href="https://github.com/manidmt/TFM"
            target="_blank"
            rel="noopener noreferrer"
          >Master's Thesis in AI &amp; Analytics</a>
        </div>
      </footer>
      <ChatDrawer />
    </div>
  );
}
