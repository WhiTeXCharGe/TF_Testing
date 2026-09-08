import type { StorageClient } from '../collab/storage/storageClient.js';
import type { AppConfig } from '../config.js';
import type { SessionMeta, SessionStatusRecord } from '../collab/types.js';
import {
  deleteSessionRecord, listSessionIds, metaKey, statusKey, writeStatus,
} from '../collab/persistence.js';
import type { Aca2Client } from './aca2Client.js';

export interface SweepDeps {
  storage: StorageClient;
  aca2: Aca2Client;
  config: Pick<AppConfig, 'absoluteSessionMaxMs' | 'idleSessionTimeoutMs' | 'idleSweepMs'>;
  now?: number;
}

export interface SweepResult {
  deleted: string[];
  closed: string[];
}

// One pass:
//  - a `close` session older than the absolute cap → delete outright
//  - an `open`/`lock` session whose relay is gone (ACA2 says not-active or is
//    unreachable) and that has been idle past the timeout → force to `close`
//    so the next open re-activates it cleanly
export async function runSweepOnce(deps: SweepDeps): Promise<SweepResult> {
  const { storage, aca2, config } = deps;
  const now = deps.now ?? Date.now();
  const deleted: string[] = [];
  const closed: string[] = [];

  for (const id of await listSessionIds(storage)) {
    const [meta, status] = await Promise.all([
      storage.getJson<SessionMeta>(metaKey(id)),
      storage.getJson<SessionStatusRecord>(statusKey(id)),
    ]);
    if (!meta || !status) continue;
    const age = now - status.lastActivityAt;

    if (status.status === 'close') {
      if (age > config.absoluteSessionMaxMs) {
        await deleteSessionRecord(storage, id);
        deleted.push(id);
      }
      continue;
    }

    // status is 'open' or 'lock'
    if (age <= config.idleSessionTimeoutMs) continue;
    const live = await aca2.live(id);
    const gone = 'unreachable' in live || live.active === false;
    if (gone) {
      await writeStatus(storage, id, { status: 'close', relayInstance: null, relayUrl: null });
      closed.push(id);
    }
  }

  return { deleted, closed };
}

export function startSweep(deps: SweepDeps): () => void {
  const timer = setInterval(() => {
    void runSweepOnce(deps);
  }, deps.config.idleSweepMs);
  timer.unref();
  return () => clearInterval(timer);
}
