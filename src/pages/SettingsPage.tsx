import { UI } from '@/config/uiConfig';
import { APP_CONFIG } from '@/config/appConfig';
import { resetRuns } from '@/services/runStore';

export function SettingsPage() {
  function handleReset() {
    resetRuns();
    // Reload so every page re-reads the freshly seeded run list.
    window.location.href = '/';
  }

  return (
    <div>
      <div className="page-header">
        <h1 className="page-heading">{UI.sidebar.settings}</h1>
        <p className="page-subheading">Local prototype settings.</p>
      </div>

      <div className="card">
        <div className="card-title">Solver API</div>
        <p style={{ fontSize: 'var(--fs-sm)', color: 'var(--text-sec)' }}>
          Base URL: <code>{APP_CONFIG.apiBaseUrl || '(not set — running in mock mode)'}</code><br />
          Set <code>VITE_API_BASE_URL</code> in <code>webapp/.env</code> to point at a live solver endpoint.
        </p>
      </div>

      <div className="card">
        <div className="card-title">Run store (mock)</div>
        <p style={{ fontSize: 'var(--fs-sm)', color: 'var(--text-sec)', marginBottom: 12 }}>
          Runs are stored in your browser's localStorage. Resetting restores the demo seed runs.
        </p>
        <button className="btn btn-secondary btn-sm" onClick={handleReset}>Reset run store</button>
      </div>
    </div>
  );
}
