import { Router } from 'express';
import multer from 'multer';
import { randomBytes } from 'node:crypto';
import type { StorageClient } from '../collab/storage/storageClient.js';
import type { AppConfig } from '../config.js';
import type { SessionMeta, SessionStatusRecord, SessionSummary } from '../collab/types.js';
import {
  createSessionRecord, deleteSessionRecord, hashOwnerToken, listSessionIds,
  metaKey, statusKey,
} from '../collab/persistence.js';
import { intake, intakeBaseline, IntakeError } from './yamlIntake.js';
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

  // Overwrite an EXISTING session's whole data — used by the client when
  // creating with a name that already matches one (after the user confirms
  // the overwrite warning), or when explicitly picking a session to
  // overwrite from the create dialog's list. Same id, no owner-token gate
  // (consistent with lock/unlock and the in-session update feature — this
  // app doesn't gate collab actions on ownership). Only allowed while the
  // target is locked — same rule as the in-session update feature, so
  // nobody's mid-edit when their session's data gets replaced out from
  // under them. Broadcasts a resync to anyone currently connected to it.
  router.post('/sessions/:id/replace', async (req, res) => {
    const { id } = req.params;
    try {
      const baseline = intakeBaseline({
        schedule: req.body?.schedule, envConfig: req.body?.envConfig, currentView: req.body?.currentView,
      });
      const meta = await storage.getJson<SessionMeta>(metaKey(id));
      if (!meta) {
        res.status(404).json({ ok: false, error: 'no such session' });
        return;
      }
      const live = await aca2.live(id);
      const status = 'unreachable' in live ? null : live.status;
      if (status !== 'lock') {
        res.status(409).json({ ok: false, error: 'ロックされているセッションのみ上書きできます' });
        return;
      }
      const result = await aca2.replaceBaseline(id, baseline);
      if ('notFound' in result) {
        res.status(404).json({ ok: false, error: 'no such session' });
        return;
      }
      res.json({ ok: true, sessionId: id });
    } catch (err) {
      if (err instanceof IntakeError) {
        res.status(400).json({ ok: false, error: err.message });
        return;
      }
      throw err;
    }
  });

  // Delete — no owner-token gate: this is a shared admin action reachable
  // from 編集 > オンラインセッションを削除 for ANY session in the list, not
  // just ones this client created (matching this app's general stance of not
  // gating collab actions on ownership — see lock/unlock and /replace above).
  // Only allowed with zero participants currently connected — deleting a
  // session out from under someone actively in it would just orphan their
  // socket (see collabSocket.ts: a session removed from the live store
  // silently stops accepting further actions from anyone still connected to
  // it, with no error surfaced to them).
  router.delete('/sessions/:id', async (req, res) => {
    const { id } = req.params;
    const meta = await storage.getJson<SessionMeta>(metaKey(id));
    if (!meta) {
      res.status(404).json({ ok: false, error: 'no such session' });
      return;
    }
    const live = await aca2.live(id);
    const participantCount = 'unreachable' in live ? 0 : live.participantCount;
    if (participantCount > 0) {
      res.status(409).json({ ok: false, error: '参加者が0人のセッションのみ削除できます' });
      return;
    }
    await aca2.evict(id);
    await deleteSessionRecord(storage, id);
    res.json({ ok: true });
  });

  return router;
}
