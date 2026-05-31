import { UI } from '@/config/uiConfig';
import { APP_CONFIG } from '@/config/appConfig';
import { resetRuns } from '@/services/runStore';

export function SettingsPage() {
  async function handleReset() {
    if (!window.confirm('Clear runs.json? Folders under public/local/ are NOT deleted.')) return;
    try {
      await resetRuns();
      window.location.href = '/';
    } catch (e) {
      window.alert(`Failed to reset: ${String((e as Error).message || e)}`);
    }
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
        <div className="card-title">Run database</div>
        <p style={{ fontSize: 'var(--fs-sm)', color: 'var(--text-sec)', marginBottom: 12 }}>
          The run log is read from <code>public/local/runs.json</code>. Resetting empties the JSON
          database — folders under <code>public/local/</code> are left on disk and can be removed manually.
        </p>
        <button className="btn btn-secondary btn-sm" onClick={handleReset}>Reset runs.json</button>
      </div>
    </div>
  );
}
