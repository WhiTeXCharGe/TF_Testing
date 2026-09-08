# GanttChartEditor Online Collaboration — Local Test Guide

> **Date:** 2026-09-08 · **Branch:** `online-collab-aca`
> Covers the local "mock Azure" stack (ACA1 + ACA2 + folder-as-Blob). No Azure account, no Docker needed.
> Companion docs: `..._ACA_ImplementationPlan20260908.md` (backend), `..._ACA_ClientPlan20260908.md` (web client), `GanttChartEditor/server/MOCK_AZURE.md` (quick reference).

---

## 0. What you're testing

| Piece | Local | What it does |
|---|---|---|
| **ACA1** – session API | `http://localhost:4000` | lists sessions, creates them, opens them, deletes them |
| **ACA2** – live relay | `http://localhost:4010` | real-time co-editing (Socket.IO) + `/internal/*` control plane |
| **Storage** | `GanttChartEditor/server/mock-blob/` | `sessions/<id>/{meta,status,baseline,log}.json` — stand-in for Azure Blob |
| **Web client** | `http://localhost:5173` (Vite) | the app; session list, create, join, lock |

One codebase, one image; the `ROLE` env var picks ACA1 vs ACA2 vs the old `local` (Electron/LAN) mode.

---

## 1. One-time setup

```bash
cd C:/Users/PC_USER/OneDrive/Desktop/work/Timefold/web/GanttChartEditor
git checkout online-collab-aca
npm install
npm install --prefix server
```

Node **20+** required (`node -v`).

---

## 2. Automated tests (run these first)

### 2.1 Server — 120 tests incl. the full end-to-end flow

```bash
cd C:/Users/PC_USER/OneDrive/Desktop/work/Timefold/web/GanttChartEditor/server
npx vitest run
```

Expect: `Test Files 16 passed`, `Tests 120 passed`.

The one that matters most is `src/__integration__/localFlow.test.ts` — it boots a real ACA1 + ACA2 on ephemeral ports sharing a temp folder and drives the whole lifecycle:

> create (JSON) → list (status `close`) → open (ACA1 wakes ACA2) → 2 socket clients join → client A edits, client B receives it → owner **locks** → both get `session-status: lock`, ACA1 list shows `lock` → B's edit is **dropped** while locked → **unlock** → edits flow again → both disconnect → ACA2 flushes the log + sets `close` → re-open → new client's `sync-init` **replays** `SET_SCHEDULE` + `UPDATE_PLAN_RANGE` from storage → `DELETE` without the owner token → **403** → `DELETE` with it → **200** → list is empty.

Run just that one:
```bash
npx vitest run src/__integration__/localFlow.test.ts
```

Other notable suites: `blobStorage.test.ts` runs the **Azure Blob** implementation against Azurite (an npm emulator — no Docker) to prove the same `StorageClient` contract; `collabSocket.test.ts` covers lock/unlock, drop-on-lock, participant cap, flush-on-last-leave; `config.test.ts` covers the abuse-limit config and the `STORAGE=blob` production gates.

### 2.2 Client — 174 jest tests

```bash
cd C:/Users/PC_USER/OneDrive/Desktop/work/Timefold/web/GanttChartEditor
npx jest --config jest.config.cjs
```

Expect: `Test Suites: 13 passed`, `Tests: 174 passed`.
Covers `collabService` (new ACA1 transport, loopback→hostname rewrite, YAML parsed client-side), `AppContext` (create/open/lock wiring, edits blocked while locked), the session-list dialog, and read-only gating (viewer **and** locked).

### 2.3 Build check

```bash
cd C:/Users/PC_USER/OneDrive/Desktop/work/Timefold/web/GanttChartEditor
npx vite build          # client — must succeed
cd server && npm run build   # server — tsc, must succeed
```

> Note: `npm run build` at the repo root also runs `tsc -b`, which reports **two pre-existing type errors** in `yamlService.ts` and `WorkerViewGantt.tsx` that exist on `GanttChartEditor` too — not from this branch. `vite build` (the actual bundler) succeeds.

---

## 3. Run the stack — `npm run dev:mock`

```bash
cd C:/Users/PC_USER/OneDrive/Desktop/work/Timefold/web/GanttChartEditor
npm run dev:mock
```

Starts three processes, prefixed `[aca1]` / `[aca2]` / `[web]`:

```
[dev:mock] ACA1 http://localhost:4000  ·  ACA2 http://localhost:4010  ·  web http://localhost:5173
[aca1] [server] role=aca1 listening on http://localhost:4000
[aca2] [server] role=aca2 listening on http://localhost:4010
[web]  VITE v5 ready ... Local: http://localhost:5173/
```

Ctrl-C stops all three. State is written to `server/mock-blob/` and **persists** between restarts — delete that folder to start clean:

```bash
rm -rf server/mock-blob
```

Quick health check (new terminal):
```bash
curl -s localhost:4000/api/health   # {"ok":true,"role":"aca1",...}
curl -s localhost:4010/api/health   # {"ok":true,"role":"aca2","instance":"...",...}
```

---

## 4. Manual test — single machine, two browser tabs

Open **two** browser tabs (or two windows) at `http://localhost:5173`.

### 4.1 Create a session (from two uploaded YAML files)

1. Tab A → menu **共同編集 → 新規作成**.
2. **表示名** = `Alice`, **セッション名** = `Test Plan`.
3. **スケジュール YAML** → pick `GanttChartEditor/Test_data/Schedule.yaml`.
4. **EnvConfig YAML** → pick `GanttChartEditor/Test_data/EnvConfig.yaml`.
5. **作成して開始** → the editor opens with the Gantt fully rendered, top bar shows **1人が参加中 共同編集中**.

> The two YAML files are parsed by the app's own `yamlService` (same as ファイル → 開く) *before* upload — ACA1 only ever stores a normalised baseline.

**Alternative — create from what's already open:** if you've loaded a schedule via ファイル → 開く, the 新規作成 tab also shows **現在のスケジュールから作成** (no files needed).

### 4.2 Join from the second tab

1. Tab B → menu **共同編集 → セッション一覧**.
2. The list shows **Test Plan** with a green **開催中** chip and a participant count.
3. **表示名** = `Bob`, leave role on **編集**, click **開く**.
4. Tab B's editor opens; both tabs' top bar now shows **2人が参加中**.

### 4.3 Live co-edit

1. Tab A → drag a bar / edit a date / change flexibility.
2. Tab B reflects the change within ~1 second (and vice-versa).
3. Undo/redo in either tab converges both.

### 4.4 Lock / unlock (owner only)

The **creator** (Alice, the tab that made the session) holds the owner token.

1. Tab A → menu **共同編集 → セッション情報** → **ロックする**.
2. Both tabs: the top bar indicator switches to **ロック中**; the session-info dialog shows a locked banner.
3. Try to edit in **either** tab (including Alice's) → the change is refused with *"このセッションはロックされているため編集できません。"*; toolbar edit buttons, undo/redo, side-panel fields are all disabled.
4. Viewing, scrolling, filtering, 制約チェック still work — it's a frozen live view, not a dead one.
5. Tab A → **ロック解除** → editing resumes in both tabs.

Tab B (Bob, no owner token) never sees a ロックする button.

### 4.5 Join as viewer

1. New tab / Tab B leaves and re-opens the session from the list with role **閲覧のみ**.
2. Editor opens read-only (same gating as lock), indicator shows **閲覧のみ**.

### 4.6 Leave, re-open, replay

1. Both tabs → menu **共同編集 → セッションを終了** (or just close the tabs).
2. ACA2 flushes to storage and the session goes **停止中** (`close`) in the list (refresh with 更新).
3. Re-open it from the list → the editor comes back with **every edit from before** (replayed from `server/mock-blob/sessions/<id>/log.json`).

### 4.7 Delete

Only the owner-token holder (Alice's tab, still in memory as `state.session.ownerToken`) can delete. There's no delete button in the UI yet — use Postman or curl (see §6). A non-owner `DELETE` returns 403.

---

## 5. Manual test — two machines on the LAN

This is the real "participants no longer need to be on the same relay process" test.

1. On the **host** machine: `npm run dev:mock`. Note its LAN IP (`ipconfig` → IPv4, e.g. `192.168.11.2`).
2. On a **second machine** on the same network, open `http://192.168.11.2:5173`.
3. It loads the app, the session list is populated from the host's ACA1, and 開く connects the socket to the host's ACA2.

**Why it works across machines:** ACA1 hands back `relayUrl = http://localhost:4010`; the client rewrites the loopback host to whatever host loaded the page (`192.168.11.2`), so the second machine connects to the host, not itself. A real Azure FQDN is left untouched.

If the second machine can't reach it: Windows Firewall on the host may be blocking inbound `5173` / `4000` / `4010` — allow Node through the private-network firewall, or temporarily disable it for the test.

---

## 6. Poke the API directly — Postman / curl

Import `GanttChartEditor/server/postman/OnlineCollabAzure.postman_collection.json` (variable `baseUrl` = `http://localhost:4000`). Requests: Health, Create (JSON), Create (2 YAML), List, Get, Open, Delete. The two Create requests stash `sessionId` + `ownerToken` into collection variables for the later calls.

curl equivalents:

```bash
# create (JSON)
SID=$(curl -s -X POST localhost:4000/api/sessions -H 'content-type: application/json' \
  -d '{"name":"curl test","schedule":{"planRange":{"startDate":"2026-01-01","endDate":"2026-01-31"},"workflowTaskList":[],"assignmentList":[]},"envConfig":{"workerList":[]},"currentView":"worker"}' \
  | node -pe 'JSON.parse(require("fs").readFileSync(0)).sessionId')

# list
curl -s localhost:4000/api/sessions | node -pe 'JSON.stringify(JSON.parse(require("fs").readFileSync(0)),null,2)'

# open (ACA1 -> ACA2 activate)
curl -s -X POST localhost:4000/api/sessions/$SID/open

# delete needs the ownerToken from the create response
curl -s -X DELETE localhost:4000/api/sessions/$SID -H "x-owner-token: <ownerToken>"
```

Lock / unlock are **Socket.IO events**, not HTTP — exercised by `localFlow.test.ts`, not Postman.

---

## 7. Inspect the "Blob" on disk

```bash
find GanttChartEditor/server/mock-blob -type f
cat GanttChartEditor/server/mock-blob/sessions/<id>/status.json   # {"status":"open|lock|close","relayInstance":...,"relayUrl":...,"lastActivityAt":...}
cat GanttChartEditor/server/mock-blob/sessions/<id>/log.json      # the ordered edit log
```

`status.json` transitions: `close` → (open) → `open` ⇄ `lock` → (last participant leaves) → `close`. On Azure this is the exact same layout, one blob per file.

---

## 8. Abuse limits (Phase 3)

Defaults (override with env vars in front of `dev:mock`, or in `docker-compose.mock.yml`):

| Limit | Default | Env var |
|---|---|---|
| Session creates / hour / IP | 10 | `CREATE_RATE_PER_HOUR` |
| Concurrent sessions | 100 | `MAX_SESSIONS` |
| Participants per session | 25 | `MAX_PARTICIPANTS` |
| YAML upload size | 5 MiB | `MAX_UPLOAD_MB` |
| Socket message size | 10 MiB | `MAX_SOCKET_MB` |

Quick check — set a tiny cap and watch it bite:

```bash
# in one terminal
cd GanttChartEditor
MAX_SESSIONS=1 CREATE_RATE_PER_HOUR=2 npm run dev:mock
# in another: 2nd create -> HTTP 429
curl -s -o /dev/null -w "%{http_code}\n" -X POST localhost:4000/api/sessions -H 'content-type: application/json' -d '{"name":"a","schedule":{},"envConfig":{},"currentView":"worker"}'
curl -s -o /dev/null -w "%{http_code}\n" -X POST localhost:4000/api/sessions -H 'content-type: application/json' -d '{"name":"b","schedule":{},"envConfig":{},"currentView":"worker"}'
```

Participant cap: with `MAX_PARTICIPANTS=1`, a 2nd non-owner `join` gets `sync-init {ok:false, error:"session is full"}`; the owner-token holder is always let in.

---

## 9. Docker path (optional — currently blocked on this machine)

```bash
cd GanttChartEditor/server
docker compose -f docker-compose.mock.yml up --build
```

This runs ACA1 + ACA2 from the **actual image** that deploys to Azure (one image, `ROLE` env picks the role), sharing `./mock-blob`. It is the only way to sanity-check the image locally before deploy.

**It does not work on this dev machine** — Docker Desktop reports "Virtualization support not detected" because *App Control for Business* policy blocks the hypervisor even though CPU virtualization is on. Whoever runs the Azure deploy will have Docker and can do this check there. Everything else in this guide runs without Docker.

---

## 10. Troubleshooting

| Symptom | Cause / fix |
|---|---|
| `dev:mock` — `EADDRINUSE :::4000` (or 4010 / 5173) | A previous run didn't exit. Kill stray node: PowerShell `Get-CimInstance Win32_Process -Filter "Name='node.exe'" \| Where-Object { $_.CommandLine -match 'dev-mock\|src.index.ts\|vite' } \| ForEach-Object { Stop-Process -Id $_.ProcessId -Force }` |
| Editor opens blank after joining | You created the session with **raw** YAML (curl `-F` multipart against ACA1) — that stores an un-normalised shape. Create through the web UI (parses client-side) or send normalised JSON. Fixed for the UI path in commit `fix(client): parse uploaded YAML …`. |
| Second machine can't load `:5173` | Host firewall blocking inbound. Allow Node on the private network, or the ports 4000/4010/5173. |
| Session list empty after a restart | Expected only if you `rm -rf server/mock-blob`. Otherwise the list reads from disk — check `find server/mock-blob -type f`. |
| `open` returns 404 | The `sessionId` has no `meta.json` in storage (deleted, or wrong id). |
| Menu dropdown seems not to open when scripted | Cosmetic timing quirk of automated clicks; it opens fine for a human. |
| jest: `import.meta` syntax error | Only if you re-add `import.meta.env` to client code — use the `__ACA1_URL__` Vite `define` instead (see `collabService.ts`). |

---

## 11. What "pass" looks like

- [ ] `npx vitest run` in `server/` — 120 passing, incl. `localFlow.test.ts`
- [ ] `npx jest` in `GanttChartEditor/` — 174 passing
- [ ] `npx vite build` succeeds
- [ ] `npm run dev:mock` boots all three; both `/api/health` respond
- [ ] Two browser tabs: create-from-YAML → join → live co-edit → lock (edits refused both sides) → unlock → leave → re-open replays the edits
- [ ] Two machines on the LAN: second machine loads the app, sees the list, joins, co-edits
- [ ] `server/mock-blob/sessions/<id>/status.json` shows `open` while live, `close` after everyone leaves
- [ ] `MAX_SESSIONS=1` → second create returns 429
