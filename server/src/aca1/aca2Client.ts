import type { AppConfig } from '../config.js';
import type { SessionStatus } from '../collab/types.js';

// Thin typed wrapper over ACA2's /internal/* control plane. ACA2 scales to
// zero, so `live` treats any transport failure or non-2xx as "not live"
// rather than an error the caller must handle.
export interface Aca2Client {
  activate(id: string): Promise<{ relayUrl: string; status: SessionStatus } | { notFound: true }>;
  live(id: string): Promise<
    | { active: boolean; participantCount: number; status: SessionStatus }
    | { unreachable: true }
  >;
  evict(id: string): Promise<void>;
}

export function createAca2Client(
  config: Pick<AppConfig, 'aca2Url' | 'internalKey'>,
  fetchImpl: typeof fetch = fetch,
): Aca2Client {
  const url = (id: string, suffix: string): string =>
    `${config.aca2Url.replace(/\/$/, '')}/internal/sessions/${encodeURIComponent(id)}/${suffix}`;
  const headers = { 'x-internal-key': config.internalKey, 'content-type': 'application/json' };

  return {
    async activate(id) {
      const res = await fetchImpl(url(id, 'activate'), { method: 'POST', headers });
      if (res.status === 404) return { notFound: true };
      const body = (await res.json()) as { relayUrl: string; status: SessionStatus };
      return { relayUrl: body.relayUrl, status: body.status };
    },

    async live(id) {
      try {
        const res = await fetchImpl(url(id, 'live'), { method: 'GET', headers });
        if (!res.ok) return { unreachable: true };
        const body = (await res.json()) as {
          active: boolean; participantCount: number; status: SessionStatus;
        };
        return { active: body.active, participantCount: body.participantCount, status: body.status };
      } catch {
        return { unreachable: true };
      }
    },

    async evict(id) {
      try {
        await fetchImpl(url(id, 'evict'), { method: 'POST', headers });
      } catch {
        // ACA2 already asleep / gone — the storage record is deleted by ACA1 anyway.
      }
    },
  };
}
