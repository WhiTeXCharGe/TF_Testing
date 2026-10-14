import { UI } from '@/config/uiConfig';
import { useEffect, useState } from 'react';
import { resetRuns } from '@/services/runStore';
import { getSolverConfig, type SolverConfig } from '@/services/solverConfig';

export function SettingsPage() {
  const [cfg, setCfg] = useState<SolverConfig | null>(null);
  useEffect(() => { void getSolverConfig().then(setCfg); }, []);

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
          {cfg?.error && (
            <span style={{ color: 'var(--red, #c0392b)' }}>{UI.settings.configErrorPrefix}{cfg.error}<br /></span>
          )}
          {UI.settings.modeLabel} <code>{cfg?.mode ?? '-'}</code><br />
          {UI.settings.baseUrlLabel} <code>{cfg?.apiBaseUrl || UI.settings.baseUrlUnset}</code><br />
          {cfg?.configPath && <>{UI.settings.configPathLabel} <code>{cfg.configPath}</code><br /></>}
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
