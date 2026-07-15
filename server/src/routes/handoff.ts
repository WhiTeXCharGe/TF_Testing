import { Router } from 'express';
import { randomUUID } from 'node:crypto';
import { spawn } from 'node:child_process';
import { existsSync } from 'node:fs';
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

// server/src/routes (dev, tsx) or server/dist/routes (build) → src|dist → server → GanttChartEditor → SchedulerWeb (container)
const __dirname = path.dirname(fileURLToPath(import.meta.url));
const SCHEDULER_CONTAINER_DIR = path.resolve(__dirname, '../../../..');
// The Vite frontend (port 5174) lives in the nested SchedulerWeb project…
const SCHEDULER_WEB_DIR = path.join(SCHEDULER_CONTAINER_DIR, 'SchedulerWeb');
// …and the queue/API service (port 3001) lives in a sibling `server` folder.
const SCHEDULER_SERVICE_DIR = path.join(SCHEDULER_CONTAINER_DIR, 'server');
console.log(`[handoff] SchedulerWeb web dir:     ${SCHEDULER_WEB_DIR}`);
console.log(`[handoff] SchedulerWeb service dir: ${SCHEDULER_SERVICE_DIR}`);

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

// Launch a `npm run <script>` process detached in the given directory.
// Returns true if the spawn was attempted, false if the directory is missing.
function launchNpmScript(cwd: string, script: string, label: string): boolean {
  if (!existsSync(path.join(cwd, 'package.json'))) {
    console.error(`[handoff] cannot launch ${label}: no package.json in ${cwd}`);
    return false;
  }
  const isWin = process.platform === 'win32';
  const npmCmd = isWin ? 'npm.cmd' : 'npm';
  const child = spawn(npmCmd, ['run', script], {
    cwd,
    detached: true,
    stdio: ['ignore', 'ignore', 'pipe'],
    // On Windows, Node >=18.20.2/20.12.2 throws `spawn EINVAL` when launching
    // .cmd/.bat files without a shell (CVE-2024-27980 mitigation). Using the
    // shell lets cmd.exe resolve and run npm.cmd correctly.
    shell: isWin,
  });
  child.stderr?.on('data', (d: Buffer) => {
    console.error(`[${label}] ${d.toString()}`);
  });
  child.on('error', (err) => {
    console.error(`[handoff] failed to spawn ${label} (${script}):`, err);
  });
  child.unref();
  return true;
}

// Launch the Scheduler Webapp: the Vite frontend and the queue/API service.
// They are started as two independent processes with verified paths rather
// than delegating to SchedulerWeb's `dev:all`, whose `dev:service` script
// assumes a `../service` folder that does not exist in this checkout.
//
// The service is started with `npm start` (plain `node server.js`), NOT
// `npm run dev` (nodemon). Nodemon watches the working directory and restarts
// on any `.json` change — but the service writes `data/status/<runId>.json`
// on every status update, which would restart it mid-run, kill the running
// `docker run`, and strand the run at "Submitted". A non-watching process
// stays alive for the full (multi-hour) solve.
function launchSchedulerWeb(): void {
  launchNpmScript(SCHEDULER_WEB_DIR, 'dev', 'scheduler-web');
  launchNpmScript(SCHEDULER_SERVICE_DIR, 'start', 'scheduler-service');
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
      launchSchedulerWeb();
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