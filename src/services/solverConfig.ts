/**
 * Runtime solver-API config — which solver backend this copy of the app talks to.
 *
 * Read from config.txt (next to the exe, or the project root in dev) via the
 * local API's GET /api/app-config, so switching local ↔ online or changing
 * the Azure URL needs no rebuild:
 *   mode=local  → http://localhost:3001 (or local_url)
 *   mode=online → azure_url
 * When no local API / config.txt is available (plain static web deploy), it
 * falls back to the build-time VITE_API_BASE_URL.
 */
import { APP_CONFIG } from '@/config/appConfig';

export interface SolverConfig {
  /** null = solver disabled (no config and no VITE_API_BASE_URL). */
  mode: 'local' | 'online' | null;
  apiBaseUrl: string;
  /** Set when config.txt exists but is unreadable/invalid — submitting is blocked. */
  error: string | null;
  configPath: string | null;
}

let cached: Promise<SolverConfig> | null = null;

function buildTimeFallback(): SolverConfig {
  const url = APP_CONFIG.apiBaseUrl;
  return { mode: url ? 'online' : null, apiBaseUrl: url, error: null, configPath: null };
}

async function fetchSolverConfig(): Promise<SolverConfig> {
  try {
    const res = await fetch('/api/app-config', { cache: 'no-store' });
    if (!res.ok) return buildTimeFallback();
    const data = await res.json() as
      | { ok: true; mode: 'local' | 'online' | null; apiBaseUrl: string | null; configPath: string | null }
      | { ok: false; error: string; configPath: string };
    if (!data.ok) return { mode: null, apiBaseUrl: '', error: data.error, configPath: data.configPath };
    if (!data.mode || !data.apiBaseUrl) return buildTimeFallback();
    return { mode: data.mode, apiBaseUrl: data.apiBaseUrl, error: null, configPath: data.configPath };
  } catch {
    return buildTimeFallback();
  }
}

/** Loaded once per page load — edit config.txt, then reload/restart to apply. */
export function getSolverConfig(): Promise<SolverConfig> {
  if (!cached) cached = fetchSolverConfig();
  return cached;
}

/** Solver API base URL, or throws if config.txt is invalid. */
export async function getSolverBaseUrl(): Promise<string> {
  const cfg = await getSolverConfig();
  if (cfg.error) throw new Error(cfg.error);
  return cfg.apiBaseUrl;
}
