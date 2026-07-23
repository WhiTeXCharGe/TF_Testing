/**
 * Standalone local API server for SchedulerWeb.
 *
 * Extracted from vite.config.ts's dev-only `localApiPlugin` (which only ran
 * inside `vite dev` and would not exist at all in a production/static build)
 * so the same routes can run for real inside the packaged Electron app.
 * vite.config.ts now just mounts `createLocalApiApp()` as Vite middleware,
 * so dev behavior is unchanged.
 *
 * Routes (unchanged from the original plugin unless noted):
 *   GET    /api/runs
 *   DELETE /api/runs
 *   POST   /api/upload
 *   DELETE /api/run/:id
 *   GET    /api/run/:id/output
 *   POST   /api/run/:id/output   — NEW: persists a downloaded solver-output
 *                                  blob to local/<id>/output/<filename>, the
 *                                  actual fix for the Azure-download bug
 *                                  (previously the blob only ever reached the
 *                                  browser's Downloads folder via saveAs()).
 *   POST   /api/handoff/create
 *   GET    /api/handoff/consume/:token
 */
import express from 'express';
import fs from 'fs';
import path from 'path';
import { spawn } from 'child_process';
import { randomUUID } from 'crypto';
import type { IncomingMessage, ServerResponse } from 'http';

export interface LocalApiOptions {
  /** Directory that holds runs.json and <runId>/{input,output}/ — e.g. public/local in dev. */
  publicLocalDir: string;
  /** GanttChartEditor's frontend + server origin, for the handoff health-check/spawn fallback. */
  ganttEditorUrl?: string;
  ganttEditorServerUrl?: string;
  /** GanttChartEditor's project dir, used only for the dev-mode `npm run dev:all` fallback spawn. */
  ganttEditorDir?: string;
}

interface RunRow {
  id: string;
  solveDate: string;
  label: string;
  folderPath: string;
  inputEnvName: string;
  inputSchedName: string;
  inputDir: string | null;
  output: 'none' | 'fetching' | 'ready';
  outputHasYaml: boolean;
  originalEnvPath?: string;
  originalSchedPath?: string;
  savedEnvPath?: string;
  savedSchedPath?: string;
  savedOutputPath?: string;
}
interface RunDB { runs: RunRow[] }

interface GanttTransferPayload {
  envYaml: string;
  scheduleYaml: string;
  expiresAt: number;
}

export function createLocalApiApp(opts: LocalApiOptions): express.Express {
  const publicLocal = opts.publicLocalDir;
  const dbPath = path.join(publicLocal, 'runs.json');
  const GANTT_EDITOR_URL = opts.ganttEditorUrl ?? 'http://localhost:5173';
  const GANTT_EDITOR_SERVER_URL = opts.ganttEditorServerUrl ?? 'http://localhost:3010';
  const GANTT_EDITOR_DIR = opts.ganttEditorDir;

  const ganttTransferTTLMs = 5 * 60 * 1000;
  const ganttTransferPending = new Map<string, GanttTransferPayload>();

  function purgeExpiredGanttTransfers(): void {
    const now = Date.now();
    for (const [token, entry] of ganttTransferPending) {
      if (entry.expiresAt < now) ganttTransferPending.delete(token);
    }
  }

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

  function launchGanttChartEditor(): void {
    if (!GANTT_EDITOR_DIR || !fs.existsSync(path.join(GANTT_EDITOR_DIR, 'package.json'))) {
      console.error(`[gantt-handoff] cannot launch: no package.json in ${GANTT_EDITOR_DIR}`);
      return;
    }
    const npmCmd = process.platform === 'win32' ? 'npm.cmd' : 'npm';
    const child = spawn(npmCmd, ['run', 'dev:all'], {
      cwd: GANTT_EDITOR_DIR,
      detached: true,
      stdio: ['ignore', 'ignore', 'pipe'],
      shell: process.platform === 'win32',
    });
    child.stderr?.on('data', (d: Buffer) => console.error(`[gantt-chart-editor] ${d.toString()}`));
    child.on('error', err => console.error('[gantt-handoff] failed to spawn GanttChartEditor:', err));
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

  function readDB(): RunDB {
    try {
      if (!fs.existsSync(dbPath)) return { runs: [] };
      const raw = fs.readFileSync(dbPath, 'utf8').trim();
      if (!raw) return { runs: [] };
      const parsed = JSON.parse(raw);
      if (!parsed || !Array.isArray(parsed.runs)) return { runs: [] };
      return parsed as RunDB;
    } catch {
      return { runs: [] };
    }
  }

  function writeDB(db: RunDB): void {
    fs.mkdirSync(publicLocal, { recursive: true });
    fs.writeFileSync(dbPath, JSON.stringify(db, null, 2), 'utf8');
  }

  function send(res: ServerResponse, status: number, body: unknown): void {
    res.statusCode = status;
    res.setHeader('Content-Type', 'application/json');
    res.setHeader('Cache-Control', 'no-store');
    res.end(JSON.stringify(body));
  }

  function readBody(req: IncomingMessage): Promise<Buffer> {
    return new Promise((resolve, reject) => {
      const chunks: Buffer[] = [];
      req.on('data', (c: Buffer) => chunks.push(c));
      req.on('end', () => resolve(Buffer.concat(chunks)));
      req.on('error', reject);
    });
  }

  /** Minimal multipart/form-data parser — handles text fields + binary files. */
  function parseMultipart(buf: Buffer, boundary: string): {
    fields: Record<string, string>;
    files: { fieldName: string; filename: string; data: Buffer }[];
  } {
    const delim = Buffer.from(`--${boundary}`);
    const fields: Record<string, string> = {};
    const files: { fieldName: string; filename: string; data: Buffer }[] = [];

    let start = buf.indexOf(delim);
    if (start === -1) return { fields, files };
    start += delim.length;

    while (start < buf.length) {
      if (buf[start] === 0x2d && buf[start + 1] === 0x2d) break; // "--" → end
      if (buf[start] === 0x0d) start += 2;

      const headerEnd = buf.indexOf('\r\n\r\n', start);
      if (headerEnd === -1) break;
      const headers = buf.slice(start, headerEnd).toString('utf8');
      const bodyStart = headerEnd + 4;
      const nextDelim = buf.indexOf(delim, bodyStart);
      if (nextDelim === -1) break;
      const body = buf.slice(bodyStart, nextDelim - 2);

      const dispMatch = /Content-Disposition: form-data; name="([^"]+)"(?:; filename="([^"]*)")?/i.exec(headers);
      if (dispMatch) {
        const name = dispMatch[1];
        const filename = dispMatch[2];
        if (filename) {
          files.push({ fieldName: name, filename, data: body });
        } else {
          fields[name] = body.toString('utf8');
        }
      }

      start = nextDelim + delim.length;
    }

    return { fields, files };
  }

  function safeRunId(id: string): boolean {
    return /^[A-Za-z0-9_-]+$/.test(id);
  }

  const app = express();

  app.use(async (req, res, next) => {
    if (!req.url) return next();

    // ── GET /api/runs ────────────────────────────────────────
    if (req.method === 'GET' && req.url === '/api/runs') {
      return send(res, 200, readDB());
    }

    // ── DELETE /api/runs ─ clear the JSON database (folders untouched) ──
    if (req.method === 'DELETE' && req.url === '/api/runs') {
      try {
        writeDB({ runs: [] });
        return send(res, 200, { ok: true });
      } catch (e) {
        return send(res, 500, { error: String(e) });
      }
    }

    // ── POST /api/upload ─────────────────────────────────────
    if (req.method === 'POST' && req.url === '/api/upload') {
      try {
        const ct = req.headers['content-type'] || '';
        const m = /boundary=(.+)$/.exec(ct);
        if (!m) return send(res, 400, { error: 'missing multipart boundary' });
        const body = await readBody(req);
        const { fields, files } = parseMultipart(body, m[1].trim());

        const runId = fields.runId;
        if (!runId || !safeRunId(runId)) {
          return send(res, 400, { error: 'invalid runId' });
        }

        const inputDir = path.join(publicLocal, runId, 'input');
        fs.mkdirSync(inputDir, { recursive: true });

        const saved: Record<string, string> = {};
        const filenames: Record<string, string> = {};
        for (const f of files) {
          const safeName = path.basename(f.filename || 'upload.yaml');
          const dst = path.join(inputDir, safeName);
          fs.writeFileSync(dst, f.data);
          saved[f.fieldName] = `/local/${runId}/input/${safeName}`;
          filenames[f.fieldName] = safeName;
        }

        const row: RunRow = {
          id: runId,
          solveDate: new Date().toISOString(),
          label: (fields.label?.trim() || 'New run'),
          folderPath: `./local/${runId}/`,
          inputEnvName: filenames.env ?? 'EnvConfig.yaml',
          inputSchedName: filenames.sched ?? 'Schedule.yaml',
          inputDir: `/local/${runId}/input`,
          output: 'none',
          outputHasYaml: false,
          originalEnvPath: fields.originalEnvPath?.trim() || undefined,
          originalSchedPath: fields.originalSchedPath?.trim() || undefined,
          savedEnvPath: saved.env,
          savedSchedPath: saved.sched,
        };
        const db = readDB();
        db.runs = [row, ...db.runs.filter(r => r.id !== runId)];
        writeDB(db);

        return send(res, 200, { run: row });
      } catch (e) {
        return send(res, 500, { error: String(e) });
      }
    }

    // ── DELETE /api/run/:id ──────────────────────────────────
    if (req.method === 'DELETE' && req.url.startsWith('/api/run/')) {
      const id = req.url.slice('/api/run/'.length);
      if (!safeRunId(id)) return send(res, 400, { error: 'invalid id' });
      const dir = path.join(publicLocal, id);
      try {
        if (fs.existsSync(dir)) fs.rmSync(dir, { recursive: true, force: true });
        const db = readDB();
        const before = db.runs.length;
        db.runs = db.runs.filter(r => r.id !== id);
        if (db.runs.length !== before) writeDB(db);
        return send(res, 200, { ok: true });
      } catch (e) {
        return send(res, 500, { error: String(e) });
      }
    }

    // ── GET /api/run/:id/output ──────────────────────────────
    if (req.method === 'GET' && /^\/api\/run\/[^/]+\/output$/.test(req.url)) {
      const id = req.url.split('/')[3];
      if (!safeRunId(id)) return send(res, 400, { error: 'invalid id' });
      const outDir = path.join(publicLocal, id, 'output');
      try {
        let hasYaml = false;
        let yamlPath: string | null = null;
        if (fs.existsSync(outDir)) {
          const yaml = fs.readdirSync(outDir).find(f => f.endsWith('.yaml') || f.endsWith('.yml'));
          if (yaml) {
            hasYaml = true;
            yamlPath = `/local/${id}/output/${yaml}`;
          }
        }
        const db = readDB();
        const row = db.runs.find(r => r.id === id);
        if (row) {
          const changed =
            row.outputHasYaml !== hasYaml ||
            (row.savedOutputPath ?? null) !== yamlPath ||
            (hasYaml && row.output !== 'ready');
          if (changed) {
            row.outputHasYaml = hasYaml;
            row.savedOutputPath = yamlPath ?? undefined;
            if (hasYaml) row.output = 'ready';
            writeDB(db);
          }
        }
        return send(res, 200, { hasYaml, yamlPath });
      } catch (e) {
        return send(res, 500, { error: String(e) });
      }
    }

    // ── POST /api/run/:id/output?filename=... ────────────────
    // Persists a solver-output blob (downloaded from the Azure API) to
    // local/<id>/output/<filename> — the fix for the Azure-download bug.
    if (req.method === 'POST' && /^\/api\/run\/[^/]+\/output$/.test(req.url.split('?')[0])) {
      const id = req.url.split('?')[0].split('/')[3];
      if (!safeRunId(id)) return send(res, 400, { error: 'invalid id' });
      try {
        const url = new URL(req.url, 'http://localhost');
        const filename = path.basename(url.searchParams.get('filename') || 'result_Schedule.yaml');
        const outDir = path.join(publicLocal, id, 'output');
        fs.mkdirSync(outDir, { recursive: true });
        const body = await readBody(req);
        fs.writeFileSync(path.join(outDir, filename), body);

        const yamlPath = `/local/${id}/output/${filename}`;
        const db = readDB();
        const row = db.runs.find(r => r.id === id);
        if (row) {
          row.outputHasYaml = true;
          row.savedOutputPath = yamlPath;
          row.output = 'ready';
          writeDB(db);
        }
        return send(res, 200, { ok: true, yamlPath });
      } catch (e) {
        return send(res, 500, { error: String(e) });
      }
    }

    // ── POST /api/handoff/create ─ send a run's YAMLs to GanttChartEditor ──
    if (req.method === 'POST' && req.url === '/api/handoff/create') {
      try {
        const body = await readBody(req);
        const { envYaml, scheduleYaml } = JSON.parse(body.toString('utf8') || '{}') as {
          envYaml?: string; scheduleYaml?: string;
        };
        if (!envYaml || !scheduleYaml) {
          return send(res, 400, { ok: false, error: 'envYaml / scheduleYaml が必要です' });
        }

        purgeExpiredGanttTransfers();
        const token = randomUUID();
        ganttTransferPending.set(token, { envYaml, scheduleYaml, expiresAt: Date.now() + ganttTransferTTLMs });

        const [webUp, serverUp] = await Promise.all([
          isReachable(GANTT_EDITOR_URL),
          isReachable(`${GANTT_EDITOR_SERVER_URL}/api/health`),
        ]);

        if (!webUp || !serverUp) {
          launchGanttChartEditor();
          const [webOk, serverOk] = await Promise.all([
            waitUntilUp(GANTT_EDITOR_URL, 28000),
            waitUntilUp(`${GANTT_EDITOR_SERVER_URL}/api/health`, 28000),
          ]);
          if (!webOk || !serverOk) {
            ganttTransferPending.delete(token);
            return send(res, 504, {
              ok: false,
              error: `GanttChartEditorの起動待ちがタイムアウトしました（frontend:${webOk ? 'OK' : 'NG'}, server:${serverOk ? 'OK' : 'NG'}）。手動で起動して再試行してください。`,
            });
          }
        }

        return send(res, 200, { ok: true, url: `${GANTT_EDITOR_URL}/?incomingTransfer=${token}` });
      } catch (e) {
        return send(res, 500, { ok: false, error: String(e) });
      }
    }

    // ── GET /api/handoff/consume/:token ─────────────────────────────────
    if (req.method === 'GET' && req.url.startsWith('/api/handoff/consume/')) {
      res.setHeader('Access-Control-Allow-Origin', GANTT_EDITOR_URL);
      purgeExpiredGanttTransfers();
      const token = req.url.slice('/api/handoff/consume/'.length);
      const entry = ganttTransferPending.get(token);
      if (!entry) {
        return send(res, 404, { ok: false, error: 'トークンが無効か期限切れです' });
      }
      ganttTransferPending.delete(token);
      return send(res, 200, { ok: true, envYaml: entry.envYaml, scheduleYaml: entry.scheduleYaml });
    }

    next();
  });

  // Serve the run data files themselves (input/output YAMLs) as static content
  // at /local/<runId>/... — the same relative paths stored in runs.json
  // (savedEnvPath, savedSchedPath, savedOutputPath) and fetched via plain
  // fetch() by fetchText() in ganttHandoffService.ts / RunLogPage.tsx (コピー
  // ファイル表示・結果を表示). In dev this duplicates what Vite's own public/
  // static serving already does (harmless); in a packaged build there is no
  // Vite dev server, so without this line those fetches fell through to the
  // SPA catch-all below and silently got index.html back instead of the real
  // YAML — GanttChartEditor would open with no usable data.
  app.use('/local', express.static(publicLocal));

  // Packaged Electron builds serve the built frontend from this same origin so
  // relative fetch('/api/...') calls in the renderer keep working. Unset in
  // dev — Vite serves the frontend itself and just mounts this app as middleware.
  const staticDir = process.env.SERVE_STATIC_DIR;
  if (staticDir) {
    app.use(express.static(staticDir));
    app.get('*', (_req, res) => {
      res.sendFile(path.join(staticDir, 'index.html'));
    });
  }

  return app;
}
