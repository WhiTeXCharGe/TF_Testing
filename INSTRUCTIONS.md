# Project Instructions — Timefold Scheduler Webapp

> Written for GitHub Copilot (or any AI assistant) picking this up on a new machine.
> Read this fully before touching any file.

---

## What this project is

A **React + TypeScript frontend** for the Timefold employee schedule optimizer.

Users upload two YAML files → a solver runs (on Azure Batch) → a result YAML comes back → 
the webapp displays a Gantt chart of the optimized schedule.

**Current situation:** Azure infrastructure is not provisioned yet (waiting for company
permission). All backend work is being tested locally using a Postman mock server.

---

## Folder structure

```
web/
  webapp/          ← THIS project (React frontend, Vite)
  service/         ← Local Express backend (replaces Azure when testing without Postman mock)
  Timefold/        ← (other team's solver code, not this project)
```

This file lives in `web/webapp/`. All paths below are relative to `web/webapp/`.

---

## How to run

```bash
npm install
npm run dev       # starts at http://localhost:5173
```

If you want the solver API buttons to work, set the backend URL in `.env`:
```
VITE_API_BASE_URL=https://your-postman-mock-url.mock.pstmn.io
```
Leave it blank to run in local-only mode (file upload/list works, solver buttons do nothing).

---

## Architecture — two separate layers

The webapp has two completely separate API layers. Do not confuse them.

### Layer 1 — Local file storage (always active)
Handled by the **Vite dev server plugin** in `vite.config.ts`.

| Endpoint | Purpose |
|---|---|
| `GET /api/runs` | Load run list from `public/local/runs.json` |
| `POST /api/upload` | Save uploaded YAMLs to `public/local/<runId>/input/` |
| `DELETE /api/run/:id` | Delete run folder + remove from runs.json |
| `GET /api/run/:id/output` | Check if output YAML exists in `public/local/<runId>/output/` |
| `DELETE /api/runs` | Reset runs.json (Settings page) |

This layer makes the run list work entirely locally — no server needed.
In production this would be replaced by a real backend or removed.

### Layer 2 — Solver API (active only when `VITE_API_BASE_URL` is set)
Calls the real backend (Azure Container Apps in production, Postman mock in development).

| Endpoint | Purpose |
|---|---|
| `POST /runSolver` | Send YAMLs to the solver, returns `{ runId, status }` |
| `GET /status/:runId` | Poll solve progress, returns `{ status, stage, progress, error }` |
| `GET /download/:runId` | Download the output YAML as a file |

---

## File-by-file reference

### Entry points
| File | Purpose |
|---|---|
| `index.html` | Single HTML entry point |
| `src/main.tsx` | React root mount |
| `src/App.tsx` | Router — 2 routes: `/` (RunLogPage) and `/settings` (SettingsPage) |
| `vite.config.ts` | Vite config + **entire local API plugin** (Layer 1 above) |
| `.env` | `VITE_API_BASE_URL` — points webapp at real/mock backend |

### Pages
| File | Purpose |
|---|---|
| `src/pages/RunLogPage.tsx` | Main page — run list table, New Run modal, Show Result button |
| `src/pages/SettingsPage.tsx` | Settings — shows API URL, reset runs.json button |

### Services (API clients)
| File | Purpose |
|---|---|
| `src/services/runStore.ts` | Layer 1 calls — `/api/runs`, `/api/upload`, etc. (local Vite middleware) |
| `src/services/runService.ts` | Layer 2 calls — `/runSolver`, `/status/:runId`, `/download/:runId` (real/mock backend) |
| `src/services/databaseService.ts` | Reads `public/data/database.xlsx` for dataset/run log data. Falls back to mockData if file is missing. Uses localStorage cache with 5-minute TTL. |
| `src/services/ganttService.ts` | TypeScript port of `yaml_to_excel.py` — parses EnvConfig.yaml + Schedule.yaml into GanttData grid. This is pure logic, no API calls. |

### Hooks
| File | Purpose |
|---|---|
| `src/hooks/useRuns.ts` | Main hook used by RunLogPage. Combines runStore (local) and runService (solver). Exposes: `runs`, `submitNewRun`, `checkOutput`, `removeRun`, `submitToSolver`, `checkRunStatus`, `triggerDownload`, `solverEnabled` |
| `src/hooks/useGantt.ts` | Fetches and parses YAML pair into GanttData for rendering |
| `src/hooks/useTimer.ts` | Live elapsed-time counter (seconds since a timestamp). Not currently wired to the UI. |

### Config
| File | Purpose |
|---|---|
| `src/config/appConfig.ts` | App constants: API base URL, file paths, sheet names, feature flags, cache TTL |
| `src/config/uiConfig.ts` | Language selector — change `uiConfig.ja` ↔ `uiConfig.en` to switch UI language |
| `src/config/uiConfig.ja.ts` | All Japanese UI strings (currently active) |
| `src/config/uiConfig.en.ts` | All English UI strings |

**To switch language:** edit `src/config/uiConfig.ts`, change the import from `.ja` to `.en`.

### Components
| File | Purpose |
|---|---|
| `src/components/layout/Layout.tsx` | App shell — wraps all pages, renders Sidebar + Topbar |
| `src/components/layout/Sidebar.tsx` | Left nav with Run Log and Settings links |
| `src/components/layout/Topbar.tsx` | Top bar with app title |
| `src/components/common/Dialog.tsx` | Reusable modal dialog component |
| `src/components/common/Icon.tsx` | SVG icon component (inline icons, no library) |
| `src/components/common/Badge.tsx` | Status badge (Completed/Failed/Executing) |
| `src/components/common/StatCard.tsx` | Metric card (used in dashboards, currently unused) |

### Types
| File | Purpose |
|---|---|
| `src/types/index.ts` | All TypeScript types: `Run`, `RunStatus`, `GanttData`, `GanttCell`, `RawEnvConfig`, `RawSchedule`, etc. |

### Styles
| File | Purpose |
|---|---|
| `src/styles/globals.css` | Global resets, base styles |
| `src/styles/variables.css` | CSS custom properties (colors, spacing, fonts) |
| `src/styles/components.css` | All component styles (table, buttons, modal, dropzone, chips, etc.) |
| `src/styles/gantt.css` | Gantt chart grid styles |

### Utils
| File | Purpose |
|---|---|
| `src/utils/dateUtils.ts` | Date parsing/formatting, `nowLabel()`, `dateRange()`, timer formatter |
| `src/utils/colorUtils.ts` | Module color assignment (stable colors by first-occurrence order), company colors |
| `src/utils/yamlUtils.ts` | YAML parse/stringify helpers (wraps js-yaml) |
| `src/utils/excelExport.ts` | Exports GanttData to an Excel file using exceljs |

### Data
| File | Purpose |
|---|---|
| `src/data/mockData.ts` | Fallback datasets/run logs/comments used when `database.xlsx` is not found |

### Postman mock
| File | Purpose |
|---|---|
| `postman/timefold-mock.postman_collection.json` | Import this into Postman to create a mock server |
| `postman/README.md` | Step-by-step guide for setting up Postman mock |

---

## Current state of each feature

### ✅ Works
- Upload 2 YAML files via "New Run" modal (saves locally + sends to backend if configured)
- Run list table with date, input files, result, delete/cancel
- File path tooltip popup on hover
- Delete / cancel a run (removes row + local folder)
- Show Result button:
  - **Solver mode** (`VITE_API_BASE_URL` set): calls `/status/:runId` → if Completed downloads YAML, if Running/Failed shows dialog
  - **Local mode** (no URL): checks `public/local/<id>/output/` folder
- Status dialogs: "Solve in progress", "Solver failed" with error detail, network error
- Settings page with runs.json reset button
- Bilingual UI (Japanese default, English available)

### 🔶 Stub / placeholder
- **Gantt chart viewer** — dialog opens but says "not connected yet". `ganttService.ts` is fully implemented (parses YAML → GanttData), but the rendering component hasn't been built.
- **Status polling** — `useTimer.ts` exists, but there is NO automatic polling loop. The user must manually click "Show Result" each time to re-check status.
- **Copy File button** — opens a dialog showing run ID and input dir, but no actual copy action.

### ❌ Not built yet
- Actual Gantt chart grid renderer (React component that takes GanttData and renders cells)
- Automatic polling (poll `GET /status/:runId` every N seconds while status is Running/Submitted)
- Azure connection (waiting for company Azure permissions — see `azure/Azure-Company-Permission-Request.md`)
- Docker trigger in local backend (`web/service/server.js` has a `// TODO: triggerDocker` stub)

---

## The local backend (`web/service/`)

> **Context:** This folder was created as a local Express server to replace Azure during
> development. It is separate from the webapp (`web/webapp/`). The user initially wanted
> to use only Postman mock, but the Express server exists as a more capable alternative.

`web/service/server.js` runs on `http://localhost:3001` and exposes:
- `POST /runSolver` — saves YAMLs to `data/input/{runId}/`
- `GET /status/:runId` — reads `data/status/{runId}.json`
- `GET /download/:runId` — sends `data/output/{runId}/result_Schedule.yaml`
- `PUT /status/:runId` — **test helper**: manually set status (simulates Docker writing status.json)
- `POST /output/:runId` — **test helper**: upload a fake output YAML (simulates Docker output)

To use it: `cd web/service && npm install && npm run dev`
Then set `VITE_API_BASE_URL=http://localhost:3001` in `webapp/.env`.

---

## Azure architecture (target state)

When Azure permissions are granted, the full system is:

```
[Webapp browser]
     ↓ POST /runSolver (2 YAMLs)
[Azure Container Apps — api-controller]
     ↓ upload to                  ↓ create task
[Azure Blob Storage]         [Azure Batch pool]
  input/{runId}/                   ↓ docker run timefold:v1
  status/{runId}.json         reads input/ → writes output/ + status/
  output/{runId}/

[Webapp] polls GET /status/:runId every 10s
         when Completed → GET /download/:runId → SAS URL → downloads YAML
```

Key files for Azure setup: `azure/Azure-Company-Permission-Request.md`

---

## Key decisions and constraints

- **runs.json is the single source of truth** for the run list. The Vite middleware manages it. Folders manually dropped into `public/local/` do NOT show up — only entries in runs.json appear.
- **runId format** is `YYYYMMDD_HHMMSSmmm` (generated in `runStore.ts`). The same ID is passed to the solver backend so both sides stay in sync.
- **Language switch** is a compile-time config change (edit one line in `uiConfig.ts`), not a runtime toggle.
- **`solverEnabled`** (`useRuns.ts`) is the guard for all solver API calls. It is `true` only when `VITE_API_BASE_URL` is non-empty. If false, the app runs fully locally.
- **`file-saver`** (already in dependencies) is used for programmatic YAML file downloads. Import: `import { saveAs } from 'file-saver'`.
- **ganttService.ts** is a faithful TypeScript port of a Python script (`yaml_to_excel.py`). Do not refactor its logic without comparing to the Python original.

---

## Things NOT to change without understanding context

| What | Why |
|---|---|
| `vite.config.ts` plugin logic | This IS the entire local API backend. Changing it breaks the run list. |
| `runs.json` schema (in `RunRow` type in `vite.config.ts`) | Must match `Run` type in `src/types/index.ts` |
| `runStore.ts` runId generation | Must stay in sync with what the solver backend expects |
| `ganttService.ts` cell assignment logic | Mirrors Python logic exactly — "first assignment per day wins", weekday calculation, etc. |
| `uiConfig.ja.ts` / `uiConfig.en.ts` key names | Must be identical between both files or TypeScript will error |

---

## Postman mock — quick summary

The file `postman/timefold-mock.postman_collection.json` is a Postman collection
that mocks all 3 solver API endpoints. Import it into Postman → create a mock server
from it → set the mock URL as `VITE_API_BASE_URL`.

**Two test scenarios for "Show Result" button:**
1. **Running** (first example in list) → webapp shows "Solve in progress" dialog
2. **Completed** (drag to top of examples list) → webapp downloads `result_Schedule.yaml`

Full guide: `postman/README.md`

---

## Dependencies worth knowing

| Package | Used for |
|---|---|
| `react-router-dom` v6 | Routing (BrowserRouter, Routes, Route) |
| `axios` | HTTP calls to solver backend (Layer 2) |
| `file-saver` | Programmatic file downloads (output YAML) |
| `js-yaml` | Parse/stringify YAML files |
| `exceljs` | Export Gantt data to Excel |
| `xlsx` | Read `database.xlsx` in databaseService |
| `vite` + `@vitejs/plugin-react` | Build tool + React HMR |
