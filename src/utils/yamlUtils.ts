import yaml from 'js-yaml';
import { APP_CONFIG } from '@/config/appConfig';
import type { RawEnvConfig, RawSchedule } from '@/types';

/** Fetch and parse a YAML file from a public URL. */
async function fetchYaml<T>(url: string): Promise<T> {
  const res = await fetch(url);
  if (!res.ok) throw new Error(`Failed to fetch ${url}: ${res.status}`);
  const text = await res.text();
  return yaml.load(text) as T;
}

/**
 * Load EnvConfig.yaml from a folder URL under /public,
 * e.g. dirUrl = "/local/20260521/input".
 */
export async function loadEnvConfig(dirUrl: string): Promise<RawEnvConfig> {
  return fetchYaml<RawEnvConfig>(`${dirUrl}/${APP_CONFIG.envConfigFile}`);
}

/** Load Schedule.yaml from a folder URL under /public. */
export async function loadSchedule(dirUrl: string): Promise<RawSchedule> {
  return fetchYaml<RawSchedule>(`${dirUrl}/${APP_CONFIG.scheduleFile}`);
}
