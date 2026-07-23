// Standalone entrypoint for the local API server — used only in packaged
// Electron builds (electron/main.cts spawns this compiled file). In dev,
// vite.config.ts mounts createLocalApiApp() directly as Vite middleware instead.
import path from 'path';
import { createLocalApiApp } from './server.js';

const PORT = Number(process.env.PORT ?? 5174);
const publicLocalDir = process.env.LOCAL_DATA_DIR ?? path.join(process.cwd(), 'local');

const app = createLocalApiApp({
  publicLocalDir,
  ganttEditorUrl: process.env.GANTT_EDITOR_URL ?? 'http://localhost:3010',
  ganttEditorServerUrl: process.env.GANTT_EDITOR_SERVER_URL ?? 'http://localhost:3010',
});

app.listen(PORT, () => {
  console.log(`[localApi] running on http://localhost:${PORT}`);
  console.log(`[localApi] data dir: ${publicLocalDir}`);
});
