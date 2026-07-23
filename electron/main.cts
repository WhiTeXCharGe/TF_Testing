import { app, BrowserWindow, dialog, ipcMain } from 'electron';
import { ChildProcess, spawn } from 'node:child_process';
import { promises as fs, existsSync, readdirSync } from 'node:fs';
import path from 'node:path';

const APP_PORT = 5174;
const APP_URL = `http://localhost:${APP_PORT}`;
const GANTT_EDITOR_URL = 'http://localhost:3010';
const GANTT_EDITOR_EXE_NAME = 'GanttChartEditor.exe';

// ── Sibling auto-discovery ───────────────────────────────────────────────
// Recommended distribution layout: both apps' unpacked folders sit side by
// side under one common parent (however that parent is named), e.g.
//   SomeFolder/GanttChartEditor/GanttChartEditor.exe
//   SomeFolder/SchedulerWeb/Timefold Scheduler.exe
// This scans that common parent for a sibling folder containing the known
// exe name, so the two apps find each other with zero setup — no manual
// "locate the file" dialog needed for the common case.
function findSiblingExe(exeName: string): string | null {
  if (!app.isPackaged) return null;
  try {
    const ownDir = path.dirname(app.getPath('exe'));
    const parentDir = path.dirname(ownDir);
    const entries = readdirSync(parentDir, { withFileTypes: true });
    for (const entry of entries) {
      if (!entry.isDirectory()) continue;
      const candidate = path.join(parentDir, entry.name, exeName);
      if (existsSync(candidate)) return candidate;
    }
  } catch {
    // parent dir unreadable — fall through to config/manual pick
  }
  return null;
}

let serverProcess: ChildProcess | null = null;
let mainWindow: BrowserWindow | null = null;

// A cross-app handoff passes the target URL (with its one-time ?incomingTransfer=
// token) as a plain argv entry when spawning/re-spawning the sibling app.
function extractTransferUrl(argv: string[]): string | null {
  return argv.find(a => a.startsWith('http://localhost')) ?? null;
}

// Cold-start handoff race: when this process itself IS the fresh instance a
// handoff just spawned (no window yet), createWindow() below has to wait
// ~10+s for the embedded server before its first loadURL call. If a second
// handoff spawn arrives during that wait (its own "ensure running" launch
// resolves as soon as the port answers, then immediately re-spawns with the
// real URL — both easily inside that window), the 'second-instance' handler
// fires and navigates early, but createWindow()'s own pending loadURL would
// then overwrite it with a blank load right after. Tracking the latest
// transfer URL in this mutable, module-level variable — read fresh at the
// end of the wait rather than captured at its start — avoids that clobber.
let pendingTransferUrl: string | null = extractTransferUrl(process.argv);

// ── Sibling app (GanttChartEditor) install path, remembered after the first pick ──

interface DesktopConfig {
  ganttEditorExePath?: string;
}

function configPath(): string {
  return path.join(app.getPath('userData'), 'desktop-config.json');
}

async function readConfig(): Promise<DesktopConfig> {
  try {
    return JSON.parse(await fs.readFile(configPath(), 'utf-8'));
  } catch {
    return {};
  }
}

async function writeConfig(cfg: DesktopConfig): Promise<void> {
  await fs.writeFile(configPath(), JSON.stringify(cfg, null, 2), 'utf-8');
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

async function waitUntilUp(url: string, timeoutMs: number, intervalMs = 1000): Promise<boolean> {
  const deadline = Date.now() + timeoutMs;
  while (Date.now() < deadline) {
    if (await isReachable(url)) return true;
    await new Promise(r => setTimeout(r, intervalMs));
  }
  return false;
}

// ── Embedded local API server (dist build, run via Electron's own Node) ─────

function startEmbeddedServer(): void {
  if (!app.isPackaged) return; // dev mode: Vite's localApiPlugin already serves these routes on 5174

  const serverEntry = path.join(process.resourcesPath, 'localApi', 'dist', 'standalone.js');
  const staticDir = path.join(process.resourcesPath, 'app-dist');
  // Data lives beside the installed exe (per-user NSIS install dir is writable),
  // keeping the existing local/<runId>/... relative convention the frontend expects.
  const localDataDir = path.join(path.dirname(app.getPath('exe')), 'local');

  serverProcess = spawn(process.execPath, [serverEntry], {
    env: {
      ...process.env,
      ELECTRON_RUN_AS_NODE: '1',
      SERVE_STATIC_DIR: staticDir,
      LOCAL_DATA_DIR: localDataDir,
      GANTT_EDITOR_URL,
      GANTT_EDITOR_SERVER_URL: GANTT_EDITOR_URL,
      PORT: String(APP_PORT),
    },
    stdio: 'inherit',
  });
  serverProcess.on('error', err => console.error('[embedded-server] failed to start:', err));
}

function stopEmbeddedServer(): void {
  serverProcess?.kill();
  serverProcess = null;
}

// ── Window ────────────────────────────────────────────────────────────────

async function createWindow(): Promise<void> {
  mainWindow = new BrowserWindow({
    width: 1400,
    height: 900,
    webPreferences: {
      preload: path.join(__dirname, 'preload.cjs'),
      contextIsolation: true,
      nodeIntegration: false,
    },
  });

  if (app.isPackaged) {
    await waitUntilUp(`${APP_URL}/api/runs`, 15000);
  }

  // Re-read pendingTransferUrl now, not before the wait above — a handoff may
  // have arrived (and already navigated the window) while we were waiting.
  await mainWindow.loadURL(pendingTransferUrl ?? APP_URL);
}

// ── IPC ───────────────────────────────────────────────────────────────────

ipcMain.handle('dialog:pickOpenFile', async () => {
  if (!mainWindow) return null;
  const res = await dialog.showOpenDialog(mainWindow, {
    title: 'YAML ファイルを選択',
    filters: [{ name: 'YAML', extensions: ['yaml', 'yml'] }],
    properties: ['openFile'],
  });
  if (res.canceled || res.filePaths.length === 0) return null;
  const filePath = res.filePaths[0];
  const content = await fs.readFile(filePath, 'utf-8');
  return { path: filePath, content };
});

// transferUrl: when set, this is a handoff — GanttChartEditor should navigate
// to this exact URL (which carries the one-time token) rather than just being
// "reachable". Re-spawning an already-running instance with a URL argv entry
// is intentional: GanttChartEditor's own single-instance lock catches it as a
// 'second-instance' event and forwards the URL to its one real window instead
// of opening a second one — see GanttChartEditor/electron/main.cts.
ipcMain.handle('sibling:launchGanttEditor', async (_evt, transferUrl?: string) => {
  if (!transferUrl && await isReachable(GANTT_EDITOR_URL)) return { ok: true };

  const cfg = await readConfig();
  let exePath = cfg.ganttEditorExePath;

  if (!exePath || !existsSync(exePath)) {
    exePath = findSiblingExe(GANTT_EDITOR_EXE_NAME) ?? undefined;
    if (exePath) await writeConfig({ ...cfg, ganttEditorExePath: exePath });
  }

  if (!exePath || !existsSync(exePath)) {
    if (!mainWindow) return { ok: false, error: 'ウィンドウが見つかりません' };
    const res = await dialog.showOpenDialog(mainWindow, {
      title: 'GanttChartEditor.exe の場所を選択してください',
      filters: [{ name: 'Executable', extensions: ['exe'] }],
      properties: ['openFile'],
    });
    if (res.canceled || res.filePaths.length === 0) {
      return { ok: false, error: 'GanttChartEditorの場所が指定されませんでした' };
    }
    exePath = res.filePaths[0];
    await writeConfig({ ...cfg, ganttEditorExePath: exePath });
  }

  const child = spawn(exePath, transferUrl ? [transferUrl] : [], { detached: true, stdio: 'ignore' });
  child.unref();

  const up = await waitUntilUp(GANTT_EDITOR_URL, 30000);
  if (!up) return { ok: false, error: 'GanttChartEditorの起動待ちがタイムアウトしました' };
  return { ok: true };
});

// ── Lifecycle ────────────────────────────────────────────────────────────

// Single-instance lock: a handoff re-spawns this exe with a transfer URL as
// an argv entry even when an instance is already running. Without this lock
// that would open a second, unrelated window — with it, the second launch
// attempt is caught below and forwarded to the one real window instead.
const gotSingleInstanceLock = app.requestSingleInstanceLock();
if (!gotSingleInstanceLock) {
  app.quit();
} else {
  app.on('second-instance', (_event, argv) => {
    const transferUrl = extractTransferUrl(argv);
    if (transferUrl) pendingTransferUrl = transferUrl;
    if (mainWindow) {
      if (transferUrl) void mainWindow.loadURL(transferUrl);
      if (mainWindow.isMinimized()) mainWindow.restore();
      mainWindow.focus();
    }
  });

  app.whenReady().then(() => {
    startEmbeddedServer();
    void createWindow();

    app.on('activate', () => {
      if (BrowserWindow.getAllWindows().length === 0) void createWindow();
    });
  });

  app.on('window-all-closed', () => {
    stopEmbeddedServer();
    if (process.platform !== 'darwin') app.quit();
  });

  app.on('before-quit', stopEmbeddedServer);
}
