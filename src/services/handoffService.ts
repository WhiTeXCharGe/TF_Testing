import { ScheduleData } from '../types/schedule';
import { EnvConfig } from '../types/envConfig';
import { stringifyScheduleYaml, stringifyEnvConfigYaml } from './yamlService';
import { resolveScheduleColors } from './fileService';
import { UI } from '../config/uiText';

interface HandoffCreateResponse {
  ok: boolean;
  url?: string;
  error?: string;
}

// Send the current EnvConfig + Schedule YAML (same format as Save) to the
// Scheduler Webapp, launching it in the background if it isn't already running.
//
// Desktop mode returns { url: null } — the transfer URL was already delivered
// straight to SchedulerWeb's one window via IPC (see electron/main.cts's
// single-instance-lock handling), so the caller must NOT window.open() it too
// (that used to open a second, blank SchedulerWeb window racing the real one
// for the one-time token — the actual cause of "opens twice / empty data").
// Web mode returns { url } for the caller to window.open() as before.
export async function sendToScheduler(envConfig: EnvConfig, schedule: ScheduleData): Promise<{ url: string | null }> {
  const envYaml = stringifyEnvConfigYaml(envConfig);
  const scheduleYaml = stringifyScheduleYaml(resolveScheduleColors(schedule));

  if (window.electronAPI) {
    // Make sure SchedulerWeb is up before minting a token — /api/handoff/create's
    // own dev-only "spawn npm run dev:all" fallback won't work in a packaged app.
    const ensureUp = await window.electronAPI.launchScheduler();
    if (!ensureUp.ok) throw new Error(ensureUp.error ?? UI.schedulerLaunchFailedError);
  }

  const res = await fetch('/api/handoff/create', {
    method: 'POST',
    headers: { 'Content-Type': 'application/json' },
    body: JSON.stringify({ envYaml, scheduleYaml }),
  });

  const data: HandoffCreateResponse = await res.json().catch(() => ({ ok: false }));
  if (!res.ok || !data.ok || !data.url) {
    throw new Error(data.error ?? UI.sendFailedError);
  }

  if (window.electronAPI) {
    // Deliver the token straight to SchedulerWeb's window instead of window.open().
    const deliver = await window.electronAPI.launchScheduler(data.url);
    if (!deliver.ok) throw new Error(deliver.error ?? UI.schedulerDeliveryFailedError);
    return { url: null };
  }
  return { url: data.url };
}