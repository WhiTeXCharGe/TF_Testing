import { defineConfig, type Plugin } from 'vite';
import react from '@vitejs/plugin-react';
import path from 'path';
import { createLocalApiApp } from './localApi/server';

// ── Cross-app handoff to GanttChartEditor (結果を表示 / コピーファイル表示) ──
//
// SchedulerWeb is a sibling folder of GanttChartEditor under web/. When the
// user asks to view a run's files in the Gantt editor, the local API server
// (see localApi/server.ts):
//   1. stores the two YAML strings under a short-lived, single-use token
//   2. health-checks GanttChartEditor's frontend (5173) and its own server (3010)
//   3. if either is down, spawns `npm run dev:all` in GanttChartEditor's folder
//      (dev only — the desktop app launches GanttChartEditor.exe directly instead,
//      via window.electronAPI.launchGanttEditor(), before this route is even hit)
//   4. returns a URL GanttChartEditor's frontend can open itself with:
//        http://localhost:5173/?incomingTransfer=<token>
// GanttChartEditor then fetches GET /api/handoff/consume/:token from this
// server (cross-origin — hence the CORS header set in localApi/server.ts).
const GANTT_EDITOR_URL = 'http://localhost:5173';
const GANTT_EDITOR_SERVER_URL = 'http://localhost:3010';
const GANTT_EDITOR_DIR = path.resolve(__dirname, '../GanttChartEditor');

/**
 * Dev-only Vite plugin: mounts the shared local API app (localApi/server.ts)
 * as Vite middleware so `/api/runs`, `/api/upload`, `/api/run/:id/output`,
 * `/api/handoff/*` etc. work exactly as before during `vite dev`. The same
 * routes run for real (not dev-only) inside the packaged Electron app via
 * localApi/standalone.ts — see electron/main.cts.
 */
function localApiPlugin(): Plugin {
  return {
    name: 'local-api',
    apply: 'serve',
    configureServer(server) {
      const app = createLocalApiApp({
        publicLocalDir: path.resolve(__dirname, 'public/local'),
        ganttEditorUrl: GANTT_EDITOR_URL,
        ganttEditorServerUrl: GANTT_EDITOR_SERVER_URL,
        ganttEditorDir: GANTT_EDITOR_DIR,
      });
      server.middlewares.use(app);
    },
  };
}

export default defineConfig({
  plugins: [react(), localApiPlugin()],
  server: {
    port: 5174,
    strictPort: true,
  },
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
