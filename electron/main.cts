import { app, BrowserWindow, dialog, ipcMain } from 'electron';
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
// Multiple windows are allowed — e.g. comparing two Gantt files, or one
// local file next to an online session — all sharing the one embedded
// server process (a second server on the same port would just fail to
// bind, so "multi-window" here means multiple BrowserWindows in this one
// process, not multiple app processes). Every IPC handler below resolves
// the calling window from the event itself rather than assuming a single
// fixed window.
const windows: BrowserWindow[] = [];
// Per-window close-intercept ack callback (see createWindow's 'close'
// listener), keyed by BrowserWindow.id so each window's checkpoint-then-
// close handshake resolves independently of any other open window.
const windowReadyToClose = new Map<number, () => void>();

ipcMain.on('app:ready-to-close', (event) => {
  const win = BrowserWindow.fromWebContents(event.sender);
  if (win) windowReadyToClose.get(win.id)?.();
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
  });
  serverProcess.on('error', err => console.error('[embedded-server] failed to start:', err));
}

function stopEmbeddedServer(): void {
  serverProcess?.kill();
  serverProcess = null;
}

// ── Window ────────────────────────────────────────────────────────────────

// opts.url: an explicit URL to load (used for a warm handoff — see
// 'second-instance' below — which always opens a NEW window at that URL
// rather than disturbing whatever's already open elsewhere). opts.isInitial:
// this is the very first window at app launch, so it's the one that
// consumes a cold-start pendingTransferUrl if one is waiting. Every other
// caller (a plain "new window" from the menu) just loads the app's own root
// and lets that window's own File > Open pick whatever it should show,
// independently of any other window.
async function createWindow(opts: { isInitial?: boolean; url?: string } = {}): Promise<BrowserWindow> {
  const win = new BrowserWindow({
    width: 1400,
    height: 900,
    webPreferences: {
      preload: path.join(__dirname, 'preload.cjs'),
      contextIsolation: true,
      nodeIntegration: false,
    },
  });
  windows.push(win);

  if (app.isPackaged) {
    // Give the embedded server a moment to bind before loading it — a no-op
    // wait for the second/third window onward, since it's already up by then.
    await waitUntilUp(`${SERVER_URL}/api/health`, 15000);
  }

  // Re-read pendingTransferUrl now, not before the wait above — a handoff may
  // have arrived while we were waiting. Only the very first window created at
  // launch ever consumes a pending handoff this way; a window opened later
  // via the in-app "new window" action, or via opts.url, ignores it.
  const url = opts.url
    ?? (opts.isInitial && pendingTransferUrl ? pendingTransferUrl : (app.isPackaged ? SERVER_URL : 'http://localhost:5173'));
  await win.loadURL(url);
  if (opts.isInitial) pendingTransferUrl = null;

  // Clicking the window's X (or Alt+F4) would otherwise tear the renderer
  // down immediately, killing its socket before an in-progress collab
  // session's last-editor checkpoint (see AppContext.leaveCollabSession) has
  // a chance to reach the server — silently losing edits since the last
  // explicit "leave session"/lock-update. Intercept the close, ask the
  // renderer to checkpoint-and-ack, then actually close. A short timeout
  // guards against a wedged/crashed renderer never acking. Scoped to this
  // window alone — another open window's session is unaffected.
  let readyToClose = false;
  win.on('close', (event) => {
    if (readyToClose) return;
    event.preventDefault();
    win.webContents.send('app:before-close');
    setTimeout(() => { readyToClose = true; win.close(); }, 3000).unref();
  });
  windowReadyToClose.set(win.id, () => { readyToClose = true; win.close(); });
  win.on('closed', () => {
    windowReadyToClose.delete(win.id);
    const idx = windows.indexOf(win);
    if (idx !== -1) windows.splice(idx, 1);
  });

  return win;
}

// ── IPC ───────────────────────────────────────────────────────────────────

ipcMain.handle('dialog:pickOpenFile', async (event) => {
  const win = BrowserWindow.fromWebContents(event.sender);
  if (!win) return null;
  const res = await dialog.showOpenDialog(win, {
    title: 'YAML ファイルを選択',
    filters: [{ name: 'YAML', extensions: ['yaml', 'yml'] }],
    properties: ['openFile'],
  });
  if (res.canceled || res.filePaths.length === 0) return null;
  const filePath = res.filePaths[0];
  const content = await fs.readFile(filePath, 'utf-8');
  return { path: filePath, content };
});

ipcMain.handle('dialog:pickSaveTarget', async (event, defaultName: string) => {
  const win = BrowserWindow.fromWebContents(event.sender);
  if (!win) return null;
  const res = await dialog.showSaveDialog(win, {
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

// New window (compare two Gantt files, or a local file next to an online
// session, side by side) — shares the one embedded server process; each
// window's renderer is an independent React app / AppContext, so opening a
// different file or joining a different session in the new window has no
// effect on any other open window.
ipcMain.handle('window:new', async () => {
  await createWindow();
});

// transferUrl: when set, this is a handoff — SchedulerWeb should navigate to
// this exact URL (which carries the one-time token) rather than just being
// "reachable". Re-spawning an already-running instance with a URL argv entry
// is intentional: SchedulerWeb's own single-instance lock catches it as a
// 'second-instance' event and forwards the URL to its one real window instead
// of opening a second one — see SchedulerWeb/electron/main.cts.
ipcMain.handle('sibling:launchScheduler', async (event, transferUrl?: string) => {
  if (!transferUrl && await isReachable(SCHEDULER_URL)) return { ok: true };

  const cfg = await readConfig();
  let exePath = cfg.schedulerExePath;

  if (!exePath || !existsSync(exePath)) {
    exePath = findSiblingExe(SCHEDULER_EXE_NAME) ?? undefined;
    if (exePath) await writeConfig({ ...cfg, schedulerExePath: exePath });
  }

  if (!exePath || !existsSync(exePath)) {
    const win = BrowserWindow.fromWebContents(event.sender);
    if (!win) return { ok: false, error: 'ウィンドウが見つかりません' };
    const res = await dialog.showOpenDialog(win, {
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
// that would start a second, wholly separate process — one that would just
// fail to bind the embedded server's port (see the windows[] comment up
// top). With it, the second launch attempt is caught below and turned into
// a new window in THIS process instead.
const gotSingleInstanceLock = app.requestSingleInstanceLock();
if (!gotSingleInstanceLock) {
  app.quit();
} else {
  app.on('second-instance', (_event, argv) => {
    const transferUrl = extractTransferUrl(argv);
    if (windows.length === 0) {
      // Cold-start race: this process IS the fresh primary instance, still
      // waiting on its own first createWindow() (see that function's own
      // comment) — there's no window yet to open a new one "instead of", so
      // just let the in-flight initial window pick this up when it's ready
      // (a plain relaunch with no transferUrl needs nothing here at all —
      // the initial window already covers it).
      if (transferUrl) pendingTransferUrl = transferUrl;
      return;
    }
    // Launching the exe again — the Start Menu/desktop icon, not the
    // taskbar (which just focuses the running app) — opens a NEW window,
    // same as VS Code, rather than silently doing nothing or replacing
    // whatever's already open. A handoff (transferUrl set) loads straight
    // into that new window instead of the app's own root.
    void createWindow(transferUrl ? { url: transferUrl } : {}).then((win) => {
      if (win.isMinimized()) win.restore();
      win.focus();
    });
  });

  app.whenReady().then(() => {
    startEmbeddedServer();
    void createWindow({ isInitial: true });

    app.on('activate', () => {
      if (BrowserWindow.getAllWindows().length === 0) void createWindow({ isInitial: true });
    });
  });

  app.on('window-all-closed', () => {
    stopEmbeddedServer();
    if (process.platform !== 'darwin') app.quit();
  });

  app.on('before-quit', stopEmbeddedServer);
}