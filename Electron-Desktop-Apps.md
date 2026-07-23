# GanttChartEditor & SchedulerWeb — Electron Desktop Apps

How the desktop build works, why it's structured this way, and what actually
lives where. Companion to each app's own docs (`GanttChartEditor/` and
`SchedulerWeb/` under this `documents/` folder).

---

## 1. Why this exists

Both apps' "save" features only worked as browser workarounds:
- GanttChartEditor's 上書き保存 could never get a real file path from the
  browser, so it silently wrote next to the server code instead of the file
  the user opened.
- SchedulerWeb's Azure-mode result download only ever reached the browser's
  Downloads folder — there was no way for a web page to redirect it to
  `local/<runId>/output/`.

Wrapping both apps in Electron gives the app real OS-level file access
(native Open/Save dialogs, direct `fs.writeFile`) instead of routing around
what a browser sandbox allows.

## 2. How an Electron app is put together

An Electron app bundles Chromium + Node.js into one executable, split into
two process types:

- **Main process** (`electron/main.cts` → compiled to `electron-dist/main.cjs`)
  — plain Node.js, full OS access. Creates the window, owns native dialogs,
  spawns the embedded backend, handles cross-app launching.
- **Renderer process** — the actual React app, running inside a sandboxed
  Chromium window. No direct file/OS access, by design (security).
- **Preload script** (`electron/preload.cts` → `preload.cjs`) — the only
  bridge between the two, via `contextBridge`. It exposes a narrow
  `window.electronAPI` surface (open/save dialogs, write file, launch
  sibling app) to the renderer; each call is really an IPC message forwarded
  to the main process, which does the actual work and replies.

Renderer code checks `if (window.electronAPI)` to decide whether it's
running as the desktop app or a plain browser tab — every desktop-only code
path has a browser fallback alongside it, so the same source still builds
and deploys as a normal web app (Azure Static Web Apps) unchanged.

## 3. The embedded backend

Beyond the Electron norm, the main process also spawns a **second** Node
child process — the app's own Express server — using Electron's own bundled
Node runtime (`ELECTRON_RUN_AS_NODE=1`, no separate Node.js install needed
on the target machine):

- **GanttChartEditor**: `server/dist/index.js` on port 3010 — constraint
  checking, save-files, and the cross-app handoff routes.
- **SchedulerWeb**: `localApi/dist/standalone.js` on port 5174 — the run
  database (`runs.json`), file uploads, and the handoff routes. This used to
  be *dev-only* Vite middleware (`vite.config.ts`); it's now a real
  standalone Express app (`localApi/server.ts`) so it works in a packaged
  build too, not just `vite dev`.

This embedded server also serves the built frontend as static files on the
same origin/port — so the renderer's `fetch('/api/...')` calls keep working
without CORS, and so `local/<runId>/...` (the actual saved files) are
reachable as plain static files, not just via the JSON API.

## 4. Build pipeline (what happens when you run a build)

```
npm run electron:build
  ├─ vite build                    → dist/            (built React app)
  ├─ tsc (server / localApi)       → server/dist/      or localApi/dist/
  ├─ tsc -p electron/tsconfig.json → electron-dist/    (main.cjs, preload.cjs)
  └─ electron-builder              → release/win-unpacked/<AppName>.exe
                                      (everything above, bundled together)
```

`release/` is gitignored — it's a build artifact, regenerated from source
every time, never edited by hand.

## 5. Distribution layout — the two apps finding each other

Both apps launch the other's `.exe` for cross-app handoffs (計画管理ツール
へ送信 / 結果を表示・コピーファイル表示). To do that without asking the user
to manually locate a file every time, each app scans **its own parent
folder** for a sibling folder containing the other app's known `.exe` name.
That only works if both apps are extracted as siblings under one common
folder:

```
SomeFolder/
  GanttChartEditor/GanttChartEditor.exe
  SchedulerWeb/Timefold Scheduler.exe
```

`web/dist/` (also outside any git repo) holds exactly this layout for local
testing — it's a copy of each `release/win-unpacked/`, refreshed by hand
after each rebuild. If the two apps aren't laid out this way, each falls
back to asking once via a native file picker and remembers the answer.

## 6. Cross-app handoff mechanics

A handoff (e.g. 結果を表示) does three things, in order:
1. **Ensure the target app is running** — IPC call with no URL; spawns the
   sibling `.exe` if it isn't reachable yet.
2. **Mint a one-time token** — POST to the sending app's own
   `/api/handoff/create`, which stores the YAML payload in memory and
   returns a URL like `http://localhost:5174/?incomingTransfer=<token>`.
3. **Deliver it** — IPC call again, this time *with* that URL. The target
   app's `.exe` gets spawned again with the URL as a plain command-line
   argument — even if it's already running.

That last point is deliberate: both apps use
`app.requestSingleInstanceLock()`, so a second launch attempt against an
already-running instance doesn't open a second window — it's caught as a
`'second-instance'` event on the *real* window, which just navigates itself
to the delivered URL and gets focused. Without this, `window.open()` used
to create a second, unrelated window race-ing the real one for the
one-time token — the original "opens twice / empty data" bug.

## 7. Known environment quirk on this machine

Producing the final single-file NSIS installer (not just
`release/win-unpacked/`) currently fails here because **Windows Developer
Mode is off**, which blocks a symlink step `electron-builder` needs for an
unrelated macOS code-signing tool it fetches regardless of target platform.
`release/win-unpacked/*.exe` is already a fully working, self-contained app
— enable Developer Mode (Settings → Privacy & security → For developers) or
run the build elevated to also get the installer.
