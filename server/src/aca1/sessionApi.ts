import { Router } from 'express';
import multer from 'multer';
import { randomBytes } from 'node:crypto';
import type { StorageClient } from '../collab/storage/storageClient.js';
import type { AppConfig } from '../config.js';
import type { SessionMeta, SessionStatusRecord, SessionSummary } from '../collab/types.js';
import {
  createSessionRecord, deleteSessionRecord, hashOwnerToken, listSessionIds,
  metaKey, ownerTokenMatches, statusKey,
} from '../collab/persistence.js';
import { intake, IntakeError } from './yamlIntake.js';
import type { Aca2Client } from './aca2Client.js';

export interface SessionApiDeps {
  storage: StorageClient;
  aca2: Aca2Client;
  config: AppConfig;
}

function fileText(
  files: Record<string, Express.Multer.File[]> | undefined,
  field: string,
): string | undefined {
  return files?.[field]?.[0]?.buffer.toString('utf-8');
}

export function createSessionApiRouter(deps: SessionApiDeps): Router {
  const { storage, aca2, config } = deps;
  const router = Router();

  const upload = multer({
    storage: multer.memoryStorage(),
    limits: { fileSize: config.limits.maxUploadBytes, files: 2 },
  }).fields([
    { name: 'schedule', maxCount: 1 },
    { name: 'envConfig', maxCount: 1 },
  ]);

  async function buildSummary(id: string): Promise<SessionSummary | null> {
    const [meta, status] = await Promise.all([
      storage.getJson<SessionMeta>(metaKey(id)),
      storage.getJson<SessionStatusRecord>(statusKey(id)),
    ]);
    if (!meta) return null;
    const st = status ?? { status: 'close' as const, relayInstance: null, relayUrl: null, lastActivityAt: meta.createdAt, lastJoinAt: null };
    let participantCount: number | null = null;
    if (st.status !== 'close') {
      const live = await aca2.live(id);
      participantCount = 'unreachable' in live ? null : live.participantCount;
    }
    return {
      id: meta.id,
      name: meta.name,
      status: st.status,
      createdAt: meta.createdAt,
      lastActivityAt: st.lastActivityAt,
      lastJoinAt: st.lastJoinAt ?? null,
      participantCount,
    };
  }

  // Create — JSON body (desktop) or multipart 2-YAML upload (browser).
  router.post('/sessions', upload, async (req, res) => {
    const files = req.files as Record<string, Express.Multer.File[]> | undefined;
    const isMultipart = req.is('multipart/form-data');
    try {
      const liveCount = (await listSessionIds(storage)).length;
      if (liveCount >= config.limits.maxConcurrentSessions) {
        res.status(429).json({ ok: false, error: 'too many active sessions; try again later' });
        return;
      }
      const result = intake(
        isMultipart
          ? {
              name: req.body?.name,
              scheduleYaml: fileText(files, 'schedule'),
              envConfigYaml: fileText(files, 'envConfig'),
              currentView: req.body?.currentView,
            }
          : {
              name: req.body?.name,
              schedule: req.body?.schedule,
              envConfig: req.body?.envConfig,
              currentView: req.body?.currentView,
            },
      );
      const ownerToken = randomBytes(24).toString('hex');
      const sessionId = await createSessionRecord(storage, {
        name: result.name,
        baseline: result.baseline,
        ownerTokenHash: hashOwnerToken(ownerToken),
      });
      res.json({ ok: true, sessionId, ownerToken });
    } catch (err) {
      if (err instanceof IntakeError) {
        res.status(400).json({ ok: false, error: err.message });
        return;
      }
      throw err;
    }
  });

  router.get('/sessions', async (_req, res) => {
    const ids = await listSessionIds(storage);
    const summaries = (await Promise.all(ids.map(buildSummary))).filter((s): s is SessionSummary => s !== null);
    // Most recently joined first; never-joined (null) fall back to createdAt.
    summaries.sort((a, b) => (b.lastJoinAt ?? b.createdAt) - (a.lastJoinAt ?? a.createdAt));
    res.json({ ok: true, sessions: summaries });
  });

  router.get('/sessions/:id', async (req, res) => {
    const summary = await buildSummary(req.params.id);
    if (!summary) {
      res.status(404).json({ ok: false, error: 'no such session' });
      return;
    }
    res.json({ ok: true, session: summary });
  });

  // Open — always ask ACA2 to activate (idempotent: no-op if already live,
  // re-wakes a cold/dead replica, first-loads a `close` session).
  router.post('/sessions/:id/open', async (req, res) => {
    const { id } = req.params;
    const activated = await aca2.activate(id);
    if ('notFound' in activated) {
      res.status(404).json({ ok: false, error: 'no such session' });
      return;
    }
    res.json({ ok: true, sessionId: id, relayUrl: activated.relayUrl, status: activated.status });
  });

  router.delete('/sessions/:id', async (req, res) => {
    const { id } = req.params;
    const meta = await storage.getJson<SessionMeta>(metaKey(id));
    if (!meta) {
      res.status(404).json({ ok: false, error: 'no such session' });
      return;
    }
    const token = req.get('x-owner-token') ?? '';
    if (!token || !ownerTokenMatches(token, meta.ownerTokenHash)) {
      res.status(403).json({ ok: false, error: 'owner token required' });
      return;
    }
    await aca2.evict(id);
    await deleteSessionRecord(storage, id);
    res.json({ ok: true });
  });

  return router;
}
