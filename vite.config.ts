import { defineConfig, type Plugin } from 'vite';
import react from '@vitejs/plugin-react';
import path from 'path';
import fs from 'fs';
import type { IncomingMessage, ServerResponse } from 'http';

/**
 * Dev-only API plugin.
 *
 *   POST   /api/upload         multipart upload of { runId, file: env, file: sched }
 *                              → writes to public/local/<runId>/input/<filename>
 *   DELETE /api/run/:id        → removes public/local/<id>/ recursively
 *   GET    /api/run/:id/output → returns { hasYaml: boolean, yamlPath: string|null }
 *
 * Files are written to disk so the rest of the app can keep using
 * import.meta.glob('/public/local/...') to discover runs.
 *
 * In production this whole plugin is a no-op — you'd swap it for a real
 * backend (Express, Cloud Run, etc) that accepts the same endpoints.
 */
function localApiPlugin(): Plugin {
  const publicLocal = path.resolve(__dirname, 'public/local');

  function send(res: ServerResponse, status: number, body: unknown): void {
    res.statusCode = status;
    res.setHeader('Content-Type', 'application/json');
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
      // After the boundary marker comes \r\n then headers then \r\n\r\n then body, then \r\n--boundary
      if (buf[start] === 0x2d && buf[start + 1] === 0x2d) break; // "--" → end
      // skip leading \r\n
      if (buf[start] === 0x0d) start += 2;

      const headerEnd = buf.indexOf('\r\n\r\n', start);
      if (headerEnd === -1) break;
      const headers = buf.slice(start, headerEnd).toString('utf8');
      const bodyStart = headerEnd + 4;
      const nextDelim = buf.indexOf(delim, bodyStart);
      if (nextDelim === -1) break;
      // body excludes trailing \r\n
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
            for (const f of files) {
              const safeName = path.basename(f.filename || 'upload.yaml');
              const dst = path.join(inputDir, safeName);
              fs.writeFileSync(dst, f.data);
              saved[f.fieldName] = `/local/${runId}/input/${safeName}`;
            }

            return send(res, 200, { runId, saved });
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
            if (!fs.existsSync(outDir)) return send(res, 200, { hasYaml: false, yamlPath: null });
            const yaml = fs.readdirSync(outDir).find(f => f.endsWith('.yaml') || f.endsWith('.yml'));
            return send(res, 200, {
              hasYaml: !!yaml,
              yamlPath: yaml ? `/local/${id}/output/${yaml}` : null,
            });
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
