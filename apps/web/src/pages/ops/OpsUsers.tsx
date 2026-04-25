import { useEffect, useState } from 'react';
import { api, ApiError } from '../../api/client';
import type { UserAdminOut } from '../../api/types';
import './OpsUsers.css';

type Action = 'approve' | 'reject' | 'activate' | 'deactivate';

function statusLabel(u: UserAdminOut): string {
  if (!u.is_approved) return 'Pending';
  if (!u.is_active) return 'Inactive';
  return 'Active';
}

function statusClass(u: UserAdminOut): string {
  if (!u.is_approved) return 'status-pending';
  if (!u.is_active) return 'status-inactive';
  return 'status-active';
}

export default function OpsUsers() {
  const [users, setUsers] = useState<UserAdminOut[]>([]);
  const [loading, setLoading] = useState(true);
  const [error, setError] = useState<string | null>(null);
  const [acting, setActing] = useState<string | null>(null);

  async function load() {
    setLoading(true);
    setError(null);
    try {
      const data = await api.get<UserAdminOut[]>('/api/admin/users');
      setUsers(data);
    } catch (err) {
      setError(err instanceof ApiError ? err.message : 'Failed to load users.');
    } finally {
      setLoading(false);
    }
  }

  useEffect(() => { load(); }, []);

  async function act(userId: string, action: Action) {
    setActing(userId + action);
    try {
      const updated = await api.post<UserAdminOut>(`/api/admin/users/${userId}/${action}`);
      setUsers(prev => prev.map(u => u.user_id === userId ? updated : u));
    } catch (err) {
      setError(err instanceof ApiError ? err.message : 'Action failed.');
    } finally {
      setActing(null);
    }
  }

  const pending = users.filter(u => !u.is_approved);
  const rest = users.filter(u => u.is_approved);

  return (
    <div className="ops-users">
      <header className="page-header">
        <h2 className="page-title">User management</h2>
        <p className="page-subtitle">Approve new signups and manage existing accounts.</p>
      </header>

      {error && <p className="ops-error">{error}</p>}

      {loading ? (
        <p className="ops-loading">Loading…</p>
      ) : (
        <>
          <section className="users-section">
            <h3 className="users-section-title">
              Pending approval
              {pending.length > 0 && (
                <span className="pending-badge">{pending.length}</span>
              )}
            </h3>
            {pending.length === 0 ? (
              <p className="users-empty">No pending signups.</p>
            ) : (
              <table className="users-table">
                <thead>
                  <tr>
                    <th>Email</th>
                    <th>Requested</th>
                    <th>Actions</th>
                  </tr>
                </thead>
                <tbody>
                  {pending.map(u => (
                    <tr key={u.user_id}>
                      <td>{u.email}</td>
                      <td>{new Date(u.created_at).toLocaleDateString()}</td>
                      <td className="actions-cell">
                        <button
                          className="btn-approve"
                          disabled={acting !== null}
                          onClick={() => act(u.user_id, 'approve')}
                        >
                          {acting === u.user_id + 'approve' ? '…' : 'Approve'}
                        </button>
                        <button
                          className="btn-reject"
                          disabled={acting !== null}
                          onClick={() => act(u.user_id, 'reject')}
                        >
                          {acting === u.user_id + 'reject' ? '…' : 'Reject'}
                        </button>
                      </td>
                    </tr>
                  ))}
                </tbody>
              </table>
            )}
          </section>

          <section className="users-section">
            <h3 className="users-section-title">All accounts</h3>
            {rest.length === 0 ? (
              <p className="users-empty">No accounts yet.</p>
            ) : (
              <table className="users-table">
                <thead>
                  <tr>
                    <th>Email</th>
                    <th>Role</th>
                    <th>Status</th>
                    <th>Created</th>
                    <th>Actions</th>
                  </tr>
                </thead>
                <tbody>
                  {rest.map(u => (
                    <tr key={u.user_id}>
                      <td>{u.email}</td>
                      <td>{u.role}</td>
                      <td>
                        <span className={`status-chip ${statusClass(u)}`}>
                          {statusLabel(u)}
                        </span>
                      </td>
                      <td>{new Date(u.created_at).toLocaleDateString()}</td>
                      <td className="actions-cell">
                        {u.is_active ? (
                          <button
                            className="btn-deactivate"
                            disabled={acting !== null}
                            onClick={() => act(u.user_id, 'deactivate')}
                          >
                            {acting === u.user_id + 'deactivate' ? '…' : 'Deactivate'}
                          </button>
                        ) : (
                          <button
                            className="btn-activate"
                            disabled={acting !== null}
                            onClick={() => act(u.user_id, 'activate')}
                          >
                            {acting === u.user_id + 'activate' ? '…' : 'Activate'}
                          </button>
                        )}
                      </td>
                    </tr>
                  ))}
                </tbody>
              </table>
            )}
          </section>
        </>
      )}
    </div>
  );
}
