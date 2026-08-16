import { UI } from '@/config/uiConfig';
import { APP_CONFIG } from '@/config/appConfig';
import { resetRuns } from '@/services/runStore';

export function SettingsPage() {
  async function handleReset() {
    if (!window.confirm(UI.settings.resetConfirm)) return;
    try {
      await resetRuns();
      window.location.href = '/';
    } catch (e) {
      window.alert(`${UI.settings.resetFailedPrefix}${String((e as Error).message || e)}`);
    }
  }

  return (
    <div>
      <div className="page-header">
        <h1 className="page-heading">{UI.sidebar.settings}</h1>
        <p className="page-subheading">{UI.settings.subheading}</p>
      </div>

      <div className="card">
        <div className="card-title">{UI.settings.solverApiTitle}</div>
        <p style={{ fontSize: 'var(--fs-sm)', color: 'var(--text-sec)' }}>
          {UI.settings.baseUrlLabel} <code>{APP_CONFIG.apiBaseUrl || UI.settings.baseUrlUnset}</code><br />
          {UI.settings.envVarHint}
        </p>
      </div>

      <div className="card">
        <div className="card-title">{UI.settings.runDbTitle}</div>
        <p style={{ fontSize: 'var(--fs-sm)', color: 'var(--text-sec)', marginBottom: 12 }}>
          {UI.settings.runDbDescription}
        </p>
        <button className="btn btn-secondary btn-sm" onClick={handleReset}>{UI.settings.resetBtn}</button>
      </div>
    </div>
  );
}
