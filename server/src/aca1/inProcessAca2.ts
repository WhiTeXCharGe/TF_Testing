import type { Aca2Client } from './aca2Client.js';
import type { SessionStore } from '../collab/sessionStore.js';
import type { AppConfig } from '../config.js';

// ROLE=local runs ACA1 and ACA2 in one process (the desktop app / LAN host),
// so the session API calls the store directly instead of doing HTTP to
// itself. Behaviour mirrors routes/internal.ts.
export function createInProcessAca2Client(
  store: SessionStore,
  config: Pick<AppConfig, 'instanceId' | 'publicRelayUrl'>,
): Aca2Client {
  return {
    async activate(id) {
      const ok = await store.activateFromStorage(id);
      if (!ok) return { notFound: true };
      const status = await store.markActivated(id, {
        relayInstance: config.instanceId,
        relayUrl: config.publicRelayUrl,
      });
      return { relayUrl: config.publicRelayUrl, status };
    },
    async live(id) {
      return store.getLive(id);
    },
    async evict(id) {
      await store.evict(id);
      await store.markClosed(id);
    },
  };
}
