import { FormEvent, useState } from 'react';
import { Link, Navigate } from 'react-router-dom';
import { useAuth } from '../context/AuthContext';
import { api, ApiError } from '../api/client';
import ChartCanvas from '../components/ChartCanvas';
import './Signup.css';

export default function Signup() {
  const { user } = useAuth();

  const [email, setEmail] = useState('');
  const [password, setPassword] = useState('');
  const [confirm, setConfirm] = useState('');
  const [error, setError] = useState<string | null>(null);
  const [success, setSuccess] = useState(false);
  const [loading, setLoading] = useState(false);

  if (user) return <Navigate to="/app" replace />;

  if (success) {
    return (
      <div className="login-page">
        <ChartCanvas />
        <div className="login-card">
          <h2 className="login-title">Request sent</h2>
          <p className="login-subtitle signup-pending">
            Your account is pending admin approval. You will be able to log in once approved.
          </p>
          <Link to="/login" className="btn-primary signup-back-btn">Back to sign in</Link>
        </div>
      </div>
    );
  }

  async function handleSubmit(e: FormEvent) {
    e.preventDefault();
    setError(null);

    if (password.length < 8) {
      setError('Password must be at least 8 characters.');
      return;
    }
    if (password !== confirm) {
      setError('Passwords do not match.');
      return;
    }

    setLoading(true);
    try {
      await api.post('/api/auth/signup', { email, password });
      setSuccess(true);
    } catch (err) {
      if (err instanceof ApiError && err.status === 409) {
        setError('An account with this email already exists.');
      } else if (err instanceof ApiError && err.status === 422) {
        setError(err.message || 'Invalid email or password format.');
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
        <h2 className="login-title">Create account</h2>
        <p className="login-subtitle">Request access to Volatility Regimes</p>

        {error && <p className="login-error">{error}</p>}

        <form onSubmit={handleSubmit} className="login-form">
          <label className="field-label" htmlFor="email">Email</label>
          <input
            id="email"
            type="email"
            className="field-input"
            value={email}
            onChange={(e) => setEmail(e.target.value)}
            required
            autoFocus
            autoComplete="email"
          />

          <label className="field-label" htmlFor="password">Password</label>
          <input
            id="password"
            type="password"
            className="field-input"
            value={password}
            onChange={(e) => setPassword(e.target.value)}
            required
            autoComplete="new-password"
            minLength={8}
          />

          <label className="field-label" htmlFor="confirm">Confirm password</label>
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
            {loading ? 'Requesting…' : 'Request access'}
          </button>
        </form>

        <p className="signup-switch">
          Already have an account?{' '}
          <Link to="/login" className="signup-link">Sign in</Link>
        </p>
      </div>
    </div>
  );
}
