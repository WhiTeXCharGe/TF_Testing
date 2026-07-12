import { Router } from 'express';
import { randomUUID } from 'node:crypto';
import { spawn } from 'node:child_process';
import path from 'node:path';
import { fileURLToPath } from 'node:url';

export const handoffRouter = Router();

// ── Transfer token store (in-memory, single-use, short TTL) ─────────────────

interface TransferPayload {
  envYaml: string;
  scheduleYaml: string;
  expiresAt: number;
}

const TTL_MS = 5 * 60 * 1000; // 5 minutes
const pending = new Map<string, TransferPayload>();

function purgeExpired(): void {
  const now = Date.now();
  for (const [token, entry] of pending) {
    if (entry.expiresAt < now) pending.delete(token);
  }
}

// ── Scheduler Webapp launch/health-check ────────────────────────────────────

const SCHEDULER_WEB_URL = 'http://localhost:5174';
const SCHEDULER_SERVICE_URL = 'http://localhost:3001';

// server/src/routes (dev, tsx) or server/dist/routes (build) → src|dist → server → GanttChartEditor → GanttChart → web → SchedulerWeb
const __dirname = path.dirname(fileURLToPath(import.meta.url));
const SCHEDULER_WEB_DIR = path.resolve(__dirname, '../../../../../SchedulerWeb');
console.log(`[handoff] SchedulerWeb dir resolved to: ${SCHEDULER_WEB_DIR}`);

async function isReachable(url: string, timeoutMs = 1500): Promise<boolean> {
  try {
    const ctrl = new AbortController();
    const timer = setTimeout(() => ctrl.abort(), timeoutMs);
    const res = await fetch(url, { signal: ctrl.signal });
    clearTimeout(timer);
    return res.status < 500;
  } catch {
    return false;
  }
}

function launchSchedulerWebDevAll(): void {
  const npmCmd = process.platform === 'win32' ? 'npm.cmd' : 'npm';
  const child = spawn(npmCmd, ['run', 'dev:all'], {
    cwd: SCHEDULER_WEB_DIR,
    detached: true,
    stdio: ['ignore', 'ignore', 'pipe'],
  });
  child.stderr?.on('data', (d: Buffer) => {
    console.error(`[scheduler-web] ${d.toString()}`);
  });
  child.on('error', (err) => {
    console.error('[handoff] failed to spawn SchedulerWeb dev:all:', err);
  });
  child.unref();
}

async function waitUntilUp(url: string, timeoutMs: number, intervalMs = 1000): Promise<boolean> {
  const deadline = Date.now() + timeoutMs;
  while (Date.now() < deadline) {
    if (await isReachable(url)) return true;
    await new Promise(r => setTimeout(r, intervalMs));
  }
  return false;
}

// ── Routes ────────────────────────────────────────────────────────────────

handoffRouter.post('/handoff/create', async (req, res) => {
  const { envYaml, scheduleYaml } = req.body as { envYaml?: string; scheduleYaml?: string };
  if (!envYaml || !scheduleYaml) {
    res.status(400).json({ ok: false, error: 'envYaml / scheduleYaml が必要です' });
    return;
  }

  purgeExpired();
  const token = randomUUID();
  pending.set(token, { envYaml, scheduleYaml, expiresAt: Date.now() + TTL_MS });

  try {
    const [webUp, serviceUp] = await Promise.all([
      isReachable(SCHEDULER_WEB_URL),
      isReachable(`${SCHEDULER_SERVICE_URL}/queue`),
    ]);

    if (!webUp || !serviceUp) {
      launchSchedulerWebDevAll();
      const [webOk, serviceOk] = await Promise.all([
        waitUntilUp(SCHEDULER_WEB_URL, 28000),
        waitUntilUp(`${SCHEDULER_SERVICE_URL}/queue`, 28000),
      ]);
      if (!webOk || !serviceOk) {
        pending.delete(token);
        res.status(504).json({
          ok: false,
          error: `計画管理ツールの起動待ちがタイムアウトしました（web:${webOk ? 'OK' : 'NG'}, service:${serviceOk ? 'OK' : 'NG'}）。手動で起動して再試行してください。`,
        });
        return;
      }
    }

    res.json({ ok: true, url: `${SCHEDULER_WEB_URL}/?incomingTransfer=${token}` });
  } catch (err) {
    pending.delete(token);
    res.status(500).json({ ok: false, error: String(err) });
  }
});

handoffRouter.get('/handoff/consume/:token', (req, res) => {
  purgeExpired();
  const { token } = req.params;
  const entry = pending.get(token);
  if (!entry) {
    res.status(404).json({ ok: false, error: 'トークンが無効か期限切れです' });
    return;
  }
  pending.delete(token);
  res.json({ ok: true, envYaml: entry.envYaml, scheduleYaml: entry.scheduleYaml });
});
