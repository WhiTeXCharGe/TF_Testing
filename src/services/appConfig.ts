// Runtime connection settings, read from config.txt next to the installed exe
// (served by the local server at /api/app-config — see server routes/
// appConfig.ts). One build serves customers (mode=online, the real Azure) and
// developers (mode=local, or a different Azure URL) — nothing is baked in at
// build time and there is no UI for it.

export type RuntimeConfig =
  | { mode: 'local' }
  | { mode: 'online'; azureUrl: string }
  // config.txt exists but is invalid. Deliberately NOT treated as local: a typo
  // must show an error, not silently point customers at the wrong server.
  | { mode: 'error'; message: string };

let cached: RuntimeConfig | null = null;
let inFlight: Promise<RuntimeConfig> | null = null;

function interpret(data: unknown): RuntimeConfig {
  const d = (data ?? {}) as { ok?: unknown; mode?: unknown; azureUrl?: unknown; error?: unknown };
  if (d.ok === false) {
    return { mode: 'error', message: typeof d.error === 'string' ? d.error : 'config.txt の内容が正しくありません。' };
  }
  if (d.mode === 'online' && typeof d.azureUrl === 'string' && d.azureUrl) {
    return { mode: 'online', azureUrl: d.azureUrl.replace(/\/+$/, '') };
  }
  // Anything else (mode=local, an older server without the route, a dev proxy
  // with nothing behind it) is the plain local behavior the app always had.
  return { mode: 'local' };
}

/**
 * Loads the config once per page load (a changed config.txt applies when the
 * app is restarted or a new window is opened). A failed request is not
 * cached, so the next call retries.
 */
export function loadAppConfig(): Promise<RuntimeConfig> {
  if (cached) return Promise.resolve(cached);
  if (inFlight) return inFlight;
  inFlight = (async () => {
    try {
      const res = await fetch('/api/app-config');
      const config = interpret(await res.json());
      cached = config;
      return config;
    } catch {
      return { mode: 'local' } as RuntimeConfig;
    } finally {
      inFlight = null;
    }
  })();
  return inFlight;
}

/** The already-loaded config, or null before the first loadAppConfig() resolves. */
export function getLoadedAppConfig(): RuntimeConfig | null {
  return cached;
}

/** True once config.txt says mode=online (and it is valid). */
export function isOnlineMode(): boolean {
  return cached?.mode === 'online';
}

/** Test hook: forget the cached config. */
export function resetAppConfigForTests(): void {
  cached = null;
  inFlight = null;
}
