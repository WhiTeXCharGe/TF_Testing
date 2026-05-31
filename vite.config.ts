import { defineConfig, type Plugin } from 'vite';
import react from '@vitejs/plugin-react';
import path from 'path';
import fs from 'fs';
import type { IncomingMessage, ServerResponse } from 'http';

/**
 * Dev-only API plugin.
 *
 *   GET    /api/runs               → returns { runs: Run[] } from public/local/runs.json
 *   POST   /api/upload             → multipart { runId, env, sched, originalEnvPath?, originalSchedPath?, label? }
 *                                    Writes the two uploaded files to public/local/<runId>/input/ and
 *                                    appends a row to runs.json. originalEnvPath / originalSchedPath
 *                                    are persisted verbatim (the user typed them).
 *   DELETE /api/run/:id            → removes public/local/<id>/ recursively AND removes the row from runs.json
 *   GET    /api/run/:id/output     → checks public/local/<id>/output for a yaml. If found, also updates the
 *                                    run's savedOutputPath in runs.json so the row reflects the new state.
 *
 * runs.json is the single source of truth for the Run Log. Folders dropped
 * manually under public/local/ are NOT shown — only entries in runs.json.
 *
 * In production you'd swap this whole plugin for a real backend (Express,
 * Cloud Run, etc) exposing the same routes.
 */
function localApiPlugin(): Plugin {
  const publicLocal = path.resolve(__dirname, 'public/local');
  const dbPath      = path.join(publicLocal, 'runs.json');

  // ── Run shape (kept loose; the frontend's @/types is the strict version) ──
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

  return {
    name: 'local-api',
    apply: 'serve',
    configureServer(server) {
      server.middlewares.use(async (req, res, next) => {
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
        // multipart fields: runId, env (file), sched (file),
        //                   originalEnvPath?, originalSchedPath?, label?
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
              inputEnvName:   filenames.env   ?? 'EnvConfig.yaml',
              inputSchedName: filenames.sched ?? 'Schedule.yaml',
              inputDir: `/local/${runId}/input`,
              output: 'none',
              outputHasYaml: false,
              originalEnvPath:   fields.originalEnvPath?.trim()   || undefined,
              originalSchedPath: fields.originalSchedPath?.trim() || undefined,
              savedEnvPath:   saved.env,
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
            // Sync JSON db so the row reflects the latest output state.
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

        next();
      });
    },
  };
}

export default defineConfig({
  plugins: [react(), localApiPlugin()],
  resolve: {
    alias: {
      '@': path.resolve(__dirname, './src'),
    },
  },
  optimizeDeps: {
    // Pre-bundle exceljs so Vite applies CJS→ESM interop (its dist is a UMD
    // bundle with no real ESM default export). Excluding it breaks the import.
    include: ['exceljs'],
  },
  build: {
    commonjsOptions: {
      transformMixedEsModules: true,
    },
  },
});
