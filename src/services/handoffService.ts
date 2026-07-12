import { ScheduleData } from '../types/schedule';
import { EnvConfig } from '../types/envConfig';
import { stringifyScheduleYaml, stringifyEnvConfigYaml } from './yamlService';
import { resolveScheduleColors } from './fileService';

interface HandoffCreateResponse {
  ok: boolean;
  url?: string;
  error?: string;
}

// Send the current EnvConfig + Schedule YAML (same format as Save) to the
// Scheduler Webapp, launching it in the background if it isn't already running.
// Resolves with the URL to open once the Scheduler Webapp is confirmed ready.
export async function sendToScheduler(envConfig: EnvConfig, schedule: ScheduleData): Promise<{ url: string }> {
  const envYaml = stringifyEnvConfigYaml(envConfig);
  const scheduleYaml = stringifyScheduleYaml(resolveScheduleColors(schedule));

  const res = await fetch('/api/handoff/create', {
    method: 'POST',
    headers: { 'Content-Type': 'application/json' },
    body: JSON.stringify({ envYaml, scheduleYaml }),
  });

  const data: HandoffCreateResponse = await res.json().catch(() => ({ ok: false }));
  if (!res.ok || !data.ok || !data.url) {
    throw new Error(data.error ?? '送信に失敗しました');
  }
  return { url: data.url };
}
