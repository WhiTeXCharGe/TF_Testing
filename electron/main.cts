import { app, BrowserWindow, dialog, ipcMain, Tray, Menu, nativeImage } from 'electron';
import { ChildProcess, spawn } from 'node:child_process';
import { promises as fs, existsSync, readdirSync } from 'node:fs';
import path from 'node:path';

const SERVER_PORT = 3010;
const SERVER_URL = `http://localhost:${SERVER_PORT}`;
const SCHEDULER_URL = 'http://localhost:5174';
const SCHEDULER_EXE_NAME = 'Timefold Scheduler.exe';

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
let tray: Tray | null = null;
let sessionActive = false;
// Set by the 'before-quit' handler at the bottom of this file. The window's
// 'close' handler needs it to tell a *window* close (hide to tray, keeping
// the session alive) apart from an actual application quit — Ctrl/Cmd+Q, the
// app menu, app.quit() from anywhere, and OS shutdown/logoff all fire 'close'
// on every window first, and without this flag the session's preventDefault()
// swallowed all of them.
let isQuitting = false;

// 16x16 solid blue (#1976d2, the app accent) circle — inlined so the tray
// entry is genuinely visible without shipping an asset file. Swap for a real
// branded icon later; the point of this one is that, unlike the transparent
// placeholder it replaced, a user can actually see and click it — and the
// tray menu is the recovery path when the window is hidden mid-session.
const TRAY_ICON_DATA_URL =
  'data:image/png;base64,iVBORw0KGgoAAAANSUhEUgAAABAAAAAQCAYAAAAf8/9hAAAAaklEQVR42mOQLLvEgAVbAHEuEDdAaQsc6hjQBbyA+BQQ/8eCT0HlcRqQi0MjOs7FZoAXkZph2AvdgFMkGnAK2QALEjXDsAXMgFwyDciFGdBApgENVHMBxWFAcSxQJR1QnBKpkheokhtJxgDO8vsDCYQo1QAAAABJRU5ErkJggg==';

function ensureTray(): void {
  if (tray) return;
  tray = new Tray(nativeImage.createFromDataURL(TRAY_ICON_DATA_URL));
  tray.setToolTip('GanttChartEditor — 共同編集セッション実行中');
  tray.setContextMenu(Menu.buildFromTemplate([
    {
      label: 'ウィンドウを開く',
      click: () => {
        if (mainWindow) { mainWindow.show(); mainWindow.focus(); } else { void createWindow(); }
      },
    },
    {
      label: '終了（セッションも終了します）',
      click: () => { sessionActive = false; mainWindow?.destroy(); app.quit(); },
    },
  ]));
}

function destroyTray(): void {
  tray?.destroy();
  tray = null;
}

ipcMain.on('collab:session-active-changed', (_evt, active: boolean) => {
  sessionActive = active;
  if (active) ensureTray(); else destroyTray();
});

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

// ── Sibling app (SchedulerWeb) install path, remembered after the first pick ──

interface DesktopConfig {
  schedulerExePath?: string;
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

// ── Embedded local Express server (dist build, run via Electron's own Node) ──

function startEmbeddedServer(): void {
  if (!app.isPackaged) return; // dev mode: `npm run dev:server` already runs this on 3010

  const serverEntry = path.join(process.resourcesPath, 'server', 'dist', 'index.js');
  const staticDir = path.join(process.resourcesPath, 'app-dist');

  serverProcess = spawn(process.execPath, [serverEntry], {
    env: {
      ...process.env,
      ELECTRON_RUN_AS_NODE: '1',
      SERVE_STATIC_DIR: staticDir,
      PORT: String(SERVER_PORT),
      DESKTOP_MODE: '1',
    },
    stdio: 'inherit',
    detached: true,
  });
  serverProcess.unref();
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

  // Closing the window mid-session hides to tray instead of tearing the
  // collab session down — but only for a genuine window close. Once the app
  // is actually quitting (isQuitting, set in 'before-quit'), the close must
  // go through or the quit is silently swallowed and the app is unkillable
  // except through the tray.
  mainWindow.on('close', (event) => {
    if (sessionActive && !isQuitting) {
      event.preventDefault();
      mainWindow?.hide();
    }
  });

  if (app.isPackaged) {
    // Give the embedded server a moment to bind before loading it.
    await waitUntilUp(`${SERVER_URL}/api/health`, 15000);
  }

  // Re-read pendingTransferUrl now, not before the wait above — a handoff may
  // have arrived (and already navigated the window) while we were waiting.
  await mainWindow.loadURL(pendingTransferUrl ?? (app.isPackaged ? SERVER_URL : 'http://localhost:5173'));
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

ipcMain.handle('dialog:pickSaveTarget', async (_evt, defaultName: string) => {
  if (!mainWindow) return null;
  const res = await dialog.showSaveDialog(mainWindow, {
    title: '名前を付けて保存',
    defaultPath: defaultName,
    filters: [{ name: 'YAML', extensions: ['yaml', 'yml'] }],
  });
  if (res.canceled || !res.filePath) return null;
  return res.filePath;
});

ipcMain.handle('fs:writeTextFile', async (_evt, filePath: string, content: string) => {
  await fs.writeFile(filePath, content, 'utf-8');
});

// transferUrl: when set, this is a handoff — SchedulerWeb should navigate to
// this exact URL (which carries the one-time token) rather than just being
// "reachable". Re-spawning an already-running instance with a URL argv entry
// is intentional: SchedulerWeb's own single-instance lock catches it as a
// 'second-instance' event and forwards the URL to its one real window instead
// of opening a second one — see SchedulerWeb/electron/main.cts.
ipcMain.handle('sibling:launchScheduler', async (_evt, transferUrl?: string) => {
  if (!transferUrl && await isReachable(SCHEDULER_URL)) return { ok: true };

  const cfg = await readConfig();
  let exePath = cfg.schedulerExePath;

  if (!exePath || !existsSync(exePath)) {
    exePath = findSiblingExe(SCHEDULER_EXE_NAME) ?? undefined;
    if (exePath) await writeConfig({ ...cfg, schedulerExePath: exePath });
  }

  if (!exePath || !existsSync(exePath)) {
    if (!mainWindow) return { ok: false, error: 'ウィンドウが見つかりません' };
    const res = await dialog.showOpenDialog(mainWindow, {
      title: 'Timefold Scheduler (SchedulerWeb.exe) の場所を選択してください',
      filters: [{ name: 'Executable', extensions: ['exe'] }],
      properties: ['openFile'],
    });
    if (res.canceled || res.filePaths.length === 0) {
      return { ok: false, error: 'SchedulerWebの場所が指定されませんでした' };
    }
    exePath = res.filePaths[0];
    await writeConfig({ ...cfg, schedulerExePath: exePath });
  }

  const child = spawn(exePath, transferUrl ? [transferUrl] : [], { detached: true, stdio: 'ignore' });
  child.unref();

  const up = await waitUntilUp(SCHEDULER_URL, 30000);
  if (!up) return { ok: false, error: 'SchedulerWebの起動待ちがタイムアウトしました' };
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

  app.on('before-quit', () => {
    // Fires before any window's 'close', so the session-active close intercept
    // above sees this and lets the quit through.
    isQuitting = true;
    stopEmbeddedServer();
  });
}
