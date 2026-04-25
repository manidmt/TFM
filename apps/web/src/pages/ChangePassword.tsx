import { FormEvent, useState } from 'react';
import { useNavigate } from 'react-router-dom';
import { api, ApiError } from '../api/client';
import { useAuth } from '../context/AuthContext';
import ChartCanvas from '../components/ChartCanvas';
import './Login.css';

export default function ChangePassword() {
  const { refresh } = useAuth();
  const navigate = useNavigate();

  const [current, setCurrent] = useState('');
  const [next, setNext] = useState('');
  const [confirm, setConfirm] = useState('');
  const [error, setError] = useState<string | null>(null);
  const [loading, setLoading] = useState(false);

  async function handleSubmit(e: FormEvent) {
    e.preventDefault();
    setError(null);

    if (next.length < 8) {
      setError('New password must be at least 8 characters.');
      return;
    }
    if (next !== confirm) {
      setError('New passwords do not match.');
      return;
    }

    setLoading(true);
    try {
      await api.post('/api/auth/change-password', {
        current_password: current,
        new_password: next,
      });
      await refresh();
      navigate('/app', { replace: true });
    } catch (err) {
      if (err instanceof ApiError && err.status === 400) {
        setError('Current password is incorrect.');
      } else if (err instanceof ApiError && err.status === 422) {
        setError(err.message || 'Password does not meet requirements.');
      } else {
        setError('Something went wrong. Please try again.');
      }
    } finally {
      setLoading(false);
    }
  }

  return (
    <div className="login-page">
      <ChartCanvas />
      <div className="login-card">
        <h2 className="login-title">Change password</h2>
        <p className="login-subtitle">You must set a new password before continuing.</p>

        {error && <p className="login-error">{error}</p>}

        <form onSubmit={handleSubmit} className="login-form">
          <label className="field-label" htmlFor="current">Current password</label>
          <input
            id="current"
            type="password"
            className="field-input"
            value={current}
            onChange={(e) => setCurrent(e.target.value)}
            required
            autoFocus
            autoComplete="current-password"
          />

          <label className="field-label" htmlFor="next">New password</label>
          <input
            id="next"
            type="password"
            className="field-input"
            value={next}
            onChange={(e) => setNext(e.target.value)}
            required
            autoComplete="new-password"
            minLength={8}
          />

          <label className="field-label" htmlFor="confirm">Confirm new password</label>
          <input
            id="confirm"
            type="password"
            className="field-input"
            value={confirm}
            onChange={(e) => setConfirm(e.target.value)}
            required
            autoComplete="new-password"
          />

          <button type="submit" className="btn-primary" disabled={loading}>
            {loading ? 'Saving…' : 'Save new password'}
          </button>
        </form>
      </div>
    </div>
  );
}
