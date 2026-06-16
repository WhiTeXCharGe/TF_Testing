# GanttChart Editor — Investigation Summary

> Created: 2026-06-17  
> Scope: Both 2025 and 2026 versions under `web/GanttChart/`

---

## Table of Contents

0. [Overview & Features](#0-overview--features)
1. [Tech Stack & Tools](#1-tech-stack--tools)
2. [2025 Version vs 2026 Version](#2-2025-vs-2026-version)
3. [Features Not Yet Implemented](#3-features-not-yet-implemented)
4. [How to Run the App](#4-how-to-run-the-app)
5. [Known Bugs & Code Issues](#5-known-bugs--code-issues)
6. [Integration Implementation Plan (Webapp ↔ GanttChart)](#6-integration-implementation-plan)
7. [Additional Notes](#7-additional-notes)

---

## 0. Overview & Features

### What does this tool do?

The **GanttChart Editor** is a desktop application for manually adjusting the schedule output (Schedule.yaml) produced by the Timefold solver. After the Scheduler Webapp runs the solver, a human can review the optimized result, make manual adjustments, and resubmit it back to the solver. The GanttChart Editor acts as the middle editing layer in that workflow.

### Currently Implemented Features (2026 version)

| Category | What's implemented |
|---|---|
| **File operations** | Load EnvConfig.yaml + Schedule.yaml (2 sequential dialogs), overwrite save, save as |
| **Gantt display** | Worker View (rows = workers, bars = their assignments) and Device View (rows = device phases, bars = assigned operations) — 2 modes |
| **Bar drag** | Full bar horizontal drag (moves start date + end date by the same number of days) |
| **Bar color coding** | 20-color palette, one color per workflow_task (device) |
| **Holiday display** | Holiday columns highlighted with light red background (`#ffe0e0`) |
| **Device View** | Phase expand/collapse toggle (`expandedPhaseTaskIds`) |
| **Side Panel** | Shows details of selected bar, `plan_flexibility` dropdown editing (Flexible/Reluctant/Fixed), bar delete |
| **Task add dialog** | Add new assignment: worker, operation task, start date, end date, hours per day, plan_flexibility |
| **Undo/Redo** | Up to 100 operations (`undoStack` / `redoStack`) |
| **Bulk plan_flexibility change** | Set flexibility for all assignments before a given date at once |
| **Search** | Search query stored in state (SearchBar component exists) |
| **Display period** | User-specifiable start/end date for the Gantt view |
| **Constraint check** | Rust-side check: worker unavailable day overrun, bar overlaps, phase date overrun (`check_constraints` IPC) |
| **Error display** | ErrorDialog component for error messages |
| **Logging** | Rust-side `write_log` IPC for debug logs |

---

## 1. Tech Stack & Tools

### 2026 Version (current)

| Layer | Technology |
|---|---|
| **Desktop framework** | [Tauri v2](https://tauri.app/) — WebView + Rust native backend |
| **Frontend** | React 19 + TypeScript + Vite |
| **State management** | React Context + useReducer (no Redux, no Zustand) |
| **Backend (native layer)** | Rust (`serde_yaml` for YAML parsing, `tauri::command` for IPC) |
| **IPC communication** | Tauri invoke API (`@tauri-apps/api` v2) |
| **CSS / styling** | None (inline styles only) |
| **Testing** | Jest (unit) + Cypress (component tests) |
| **Package manager** | npm |
| **Build output** | `npm run tauri:build` → Windows `.exe` / `.msi` |
| **Note: unused package** | `gantt-task-react ^0.3.9` is installed but not used — Gantt rendering is custom HTML/CSS |

### 2025 Version (previous)

| Layer | Technology |
|---|---|
| **Backend** | Python + FastAPI (`app.py`) |
| **Frontend** | React (served as static files via FastAPI) |
| **Data models** | Python dataclass (`models.py`) |
| **YAML handling** | Python `yaml_loader.py` |
| **Distribution** | PyInstaller-packaged `.exe` (`GanttEditor.exe`) |
| **Entry point** | `main.py` → `gantt_editor_launcher.py` |

---

## 2. 2025 vs 2026 Version

| Aspect | 2025 Version | 2026 Version |
|---|---|---|
| **Architecture** | Python FastAPI server + React SPA (browser) | Tauri desktop app (Rust + React WebView) |
| **Distribution** | PyInstaller `.exe` (portable) | Tauri `.exe` / `.msi` installer |
| **Backend language** | Python | Rust |
| **Constraint checking** | Python-side | Rust-side (`constraint_checker.rs`) |
| **State management** | Python `AppState` class (server-side) | React useReducer (client-side) |
| **Undo/Redo** | Python `undo_stack` / `redo_stack` on server | React state `undoStack` / `redoStack` |
| **Device View** | `expand_state` dict for phase expansion | `expandedPhaseTaskIds: Set<string>` |
| **Model definitions** | Python `dataclass` | TypeScript `interface` + Rust `struct` |
| **Testing** | Unknown (test files not checked) | Jest + Cypress component tests |
| **Gantt rendering** | React custom implementation | Custom HTML/CSS (gantt-task-react not used) |
| **Required environment** | Python runtime (or exe) | Node.js + Rust/Cargo (dev), built exe (prod) |

---

## 3. Features Not Yet Implemented

> Source: `要求定義書.md` (2026 requirements) and `外部設計書.md` (external design spec)

### 3-1. Gantt Chart Interaction

| Feature ID | Requirement | Current Status |
|---|---|---|
| F-03 | **Left/right edge drag** to change start date or end date independently | **Not implemented.** Only full-bar horizontal drag (move) exists |
| F-04 | **Row-to-row drag** to reassign a bar to a different worker (Worker View) | **Not implemented** |
| F-05 | Visual highlight (e.g. red border) on overlapping bars | Constraint check exists but no visual highlight on bars confirmed |
| F-06 | Color change on bars that have constraint violations | **Not implemented.** Violations shown in SidePanel but bars don't change color |

### 3-2. Side Panel

| Feature ID | Requirement | Current Status |
|---|---|---|
| F-07 | Edit `work_date_list[].hour` (daily work hours) in the Side Panel | **Not implemented.** Side Panel only allows `plan_flexibility` editing |
| F-08 | Device View Side Panel — phase date editing, add/remove workers per operation | **Not implemented.** No Device View-specific Side Panel designed |

### 3-3. Task Add Dialog

| Feature ID | Requirement | Current Status |
|---|---|---|
| F-09 | **Cascading dropdowns**: Device → Phase → Operation selection | **Not implemented.** Currently shows a flat list of all operation_tasks |
| F-10 | Respect `workload_hours` when adding a task (hours-based calculation) | **Unclear.** Dialog has `hoursPerDay` field but no link to `workload_hours` |
| F-11 | Inline validation error display | **Not implemented** |

### 3-4. File Operations

| Feature ID | Requirement | Current Status |
|---|---|---|
| UI-IN-01 | "Select both files in a single operation" (EnvConfig + Schedule simultaneously) | **Not implemented.** Currently 2 sequential dialogs (`FileButtons.tsx:11-29`) |

### 3-5. Integration Features (Most Critical)

| Feature | Requirement | Current Status |
|---|---|---|
| **Submit → Webapp** | In GanttChart editor, press "Submit" → Scheduler Webapp's "New Run" dialog opens with the edited YAMLs pre-loaded | **The Submit button doesn't exist at all.** Not added to Toolbar |
| **Show Result → Gantt** | In Scheduler Webapp, press "Show Result" → GanttChart editor launches and auto-loads the solver output YAMLs | **Not implemented.** `RunLogPage.tsx` `handleShowResult` only opens a simple dialog — no Tauri app launch |

---

## 4. How to Run the App

### 2026 Version — Development Mode

```bash
# Navigate to the source directory
cd "GanttChart/2026年度/ガントチャートエディターソースコード/今期ガントチャートエディターソースコード"

# Install dependencies (first time only)
npm install

# Start Tauri dev server (React frontend + Rust backend together)
npm run tauri:dev
```

> **Important:** `npm run dev` only starts the Vite frontend. Tauri IPC features (file dialogs, file I/O, etc.) only work with `tauri:dev`.

### 2026 Version — Production Build

```bash
npm run tauri:build
# → Outputs .exe / .msi under src-tauri/target/release/bundle/
```

### 2025 Version (Python)

```bash
cd "GanttChart/2025年度/前期ガントチャートエディターソースコード"

# Install Python dependencies (first time only)
pip install fastapi uvicorn pyyaml

# Run
python main.py
# → Open browser at http://localhost:<port>
```

Or use the pre-built executable directly:

```
GanttChart/2025年度/前期ガントチャートエディターソースコード/GanttEditor_1224/GanttEditor.exe
```

---

## 5. Known Bugs & Code Issues

### 5-1. [CRITICAL] YAML Schema Mismatch

The Timefold solver's YAML output does not match the GanttChart 2026 type definitions.

**Timefold output (actual YAML):**
```yaml
schedule:
  workflow_task_list:
    - id: e1p1o1
      workload_hours: 240    # ← hours (integer)
```

**GanttChart 2026 — Rust model (`schedule.rs:119`):**
```rust
pub struct OperationTask {
    pub workload_days: u32,  // ← expects "workload_days" key — will not read "workload_hours"
}
```

**GanttChart 2026 — TypeScript type (`src/types/schedule.ts:25`):**
```typescript
export interface OperationTask {
  workloadDays: number;  // ← same mismatch
}
```

**Impact:** When a Timefold-output YAML is loaded, the `workload_hours` field is silently ignored during deserialization. `workloadDays` becomes `undefined` / 0. Any workload-based calculations in the task add dialog will be wrong.

**Fix:**
- Rust: rename `workload_days` → `workload_hours`, or add `#[serde(alias = "workload_hours")]`
- TypeScript: rename `workloadDays` → `workloadHours`
- Update task add dialog calculation logic to match

---

### 5-2. [IMPORTANT] `schedule:` Root Key Wrapper

Timefold outputs YAML wrapped under a `schedule:` root key:
```yaml
schedule:
  plan_range: ...
  workflow_task_list: ...
  assignment_list: ...
```

The Rust side has a `ScheduleWrapper` struct (`schedule.rs:6`) that appears to handle this on load. However, it needs to be confirmed that the **save path** (`save_schedule_yaml`) also re-wraps with `schedule:` when writing back. If not, the saved file won't parse correctly on the next load.

---

### 5-3. "Submit" Button Does Not Exist

**Expected behavior:** After editing in the GanttChart editor, click "Submit" → Scheduler Webapp's "New Run" dialog opens with the 2 edited YAML files pre-loaded.

**Actual code (`Toolbar.tsx`):**
- Only these buttons exist: "ファイル読込" (Load), "上書保存" (Save), "名前を付けて保存" (Save As), "タスク追加" (Add Task)
- No Submit button
- No IPC command that calls or launches Timefold exists in the current codebase

> Note: The user mentioned it "seems to run Timefold when Submit is pressed," but no such code path was found in the current source. If it appeared to run, it may have been a leftover from a previous version or a different environment.

---

### 5-4. Webapp "Show Result" Does Not Launch the GanttChart Editor

**`webapp/src/pages/RunLogPage.tsx` — `handleShowResult()`:**
```tsx
// Current behavior: opens a simple dialog showing run ID and path info only
setGanttDialog({ runId, outputDir });
```

A plain `Dialog` component shows the run ID and output directory path. There is zero code for launching the Tauri GanttChart application.

---

### 5-5. File Selection UX Doesn't Match Spec

**Spec (外部設計書 UI-IN-01):** "Select both files in a single operation"  
**Current code (`FileButtons.tsx:11-29`):** Two sequential file dialogs — first for EnvConfig, second for Schedule.

This means if the user accidentally selects them in the wrong order, the data will be cross-loaded with no warning.

---

## 6. Integration Implementation Plan

### Integration A: GanttChart "Submit" → Webapp "New Run" Dialog

#### Full flow

```
[User] Finishes editing in GanttChart editor
    ↓ clicks "Submit" button
[GanttChart] Saves current Schedule.yaml to a temp/specified path
    ↓ opens browser pointing to Webapp's New Run dialog
[Webapp] "New Run" dialog opens with EnvConfig + Schedule pre-filled
    ↓ User clicks "Run"
[Timefold] Solver executes
```

#### Implementation options

**Option A (Recommended): Pass file paths via URL parameters**

1. GanttChart (Tauri side):
   - On Submit → save current Schedule.yaml
   - Launch browser with `shell::open()` or `tauri-plugin-opener`:
     ```
     http://localhost:3000/newrun?env=<EnvConfigPath>&schedule=<SchedulePath>
     ```
2. Webapp (React side):
   - Add a `/newrun` route or handle `?env=&schedule=` query params in existing page
   - Auto-open `NewRunModal` with the specified YAML paths pre-loaded into the drop zones

**Files to change:**

| File | Change |
|---|---|
| `GanttChart/src/components/Toolbar/FileButtons.tsx` | Add "Submit" button |
| `GanttChart/src/api/tauriCommands.ts` | Add `openBrowserToNewRun(envPath, schedPath)` |
| `GanttChart/src-tauri/src/commands/` | Add `open_browser_command` IPC handler |
| `Webapp/src/App.tsx` | Add new route or query param handling |
| `Webapp/src/pages/RunLogPage.tsx` | Read URL params, auto-open `NewRunModal` |

**Option B: HTTP notification via service layer**
1. Tauri sends HTTP POST to `web/service/` local API (with file paths)
2. Service notifies Webapp via WebSocket / SSE
3. Webapp opens New Run dialog

---

### Integration B: Webapp "Show Result" → Launch GanttChart Editor

#### Full flow

```
[User] Clicks "Show Result" in Webapp
    ↓ Webapp identifies the EnvConfig + Schedule paths for that run
[Webapp] Calls service API to launch GanttChart with those file paths
    ↓
[GanttChart Tauri app] Starts up and auto-loads the specified YAMLs → displays Gantt
```

#### Implementation

**How to launch a Tauri app from a browser:**
The Webapp is a browser-based React app — it cannot directly launch a desktop `.exe`. The `web/service/` Node.js layer acts as the bridge.

```javascript
// Webapp side (browser React)
await fetch('http://localhost:<servicePort>/api/open-gantt', {
  method: 'POST',
  headers: { 'Content-Type': 'application/json' },
  body: JSON.stringify({ envPath: '...', schedulePath: '...' })
});
```

**`web/service/` side (Node.js):**
```javascript
const { spawn } = require('child_process');

app.post('/api/open-gantt', (req, res) => {
  const { envPath, schedulePath } = req.body;
  const ganttExePath = process.env.GANTT_EXE_PATH; // from config/env
  spawn(ganttExePath, ['--env', envPath, '--schedule', schedulePath], {
    detached: true,
    stdio: 'ignore'
  });
  res.json({ ok: true });
});
```

**GanttChart Tauri side — add CLI argument support:**
- `src-tauri/src/main.rs`: parse `--env` and `--schedule` startup arguments
- On startup, fire a `LOAD_FILES` action with those paths automatically

**Files to change:**

| File | Change |
|---|---|
| `Webapp/src/pages/RunLogPage.tsx` | Change `handleShowResult` to call service `/api/open-gantt` |
| `web/service/` (Node.js) | Add `POST /api/open-gantt` endpoint; spawn Tauri exe with args |
| `GanttChart/src-tauri/src/main.rs` | Parse CLI args (`--env`, `--schedule`) on startup |
| `GanttChart/src/api/tauriCommands.ts` | Add `getStartupFiles()` IPC command if needed |

---

## 7. Additional Notes

### 7-1. The service layer's current role

`web/service/` is a Node.js server with its own git repo and data folder. It is not currently designed as a bridge between the Webapp and the GanttChart editor, but it is the **best place to implement that bridge** for the integrations described above.

### 7-2. Webapp's `ganttService.ts` vs the desktop GanttChart editor

- `webapp/src/services/ganttService.ts` builds a `GanttData` object for **in-browser Gantt preview** inside the Webapp — it is a completely separate thing from the desktop GanttChart editor
- `webapp/src/hooks/useGantt.ts` fetches YAMLs from a public directory to render a read-only Gantt preview in the browser
- Actual **editing and saving** is only available in the desktop Tauri GanttChart app

### 7-3. The `gantt-task-react` package

`package.json` lists `gantt-task-react ^0.3.9` as a dependency, but the actual Gantt rendering is done with custom HTML/CSS. Either adopt it (standardize rendering) or remove it (reduce bundle size). The current half-state is confusing.

### 7-4. Cypress test setup

`AppContext.tsx:33-36` exposes `window.__APP_CONTEXT__` (state + dispatch) when `window.Cypress` is detected. This allows Cypress tests to directly manipulate app state. It is guarded so it only activates during Cypress test runs and is safe for production.

### 7-5. Date format normalization

The Rust `normalize_dates()` function (`schedule.rs:20`) converts `YYYY/MM/DD` and `YYYY/M/D` to `YYYY-MM-DD` on load. Timefold outputs dates in `2025/09/01` format, so this conversion is necessary and currently in place.

### 7-6. Recommended priority order

| Priority | Action |
|---|---|
| 🔴 Critical | Fix `workload_hours` schema mismatch (Rust + TypeScript) |
| 🔴 Critical | Verify that `schedule:` wrapper is preserved on save |
| 🟠 High | Add Submit button + wire up to Webapp New Run dialog |
| 🟠 High | Implement Show Result → GanttChart launch (via service layer) |
| 🟡 Medium | Bar left/right edge drag (independent start/end date resize) |
| 🟡 Medium | Cascading dropdowns in Task Add dialog (Device → Phase → Operation) |
| 🟡 Medium | Side Panel daily work hours editing (`work_date_list`) |
| 🟢 Low | Change file selection from 2 sequential dialogs to 1 simultaneous selection |
| 🟢 Low | Decide what to do with `gantt-task-react` (use it or remove it) |

---

*This document reflects the state of the source code as of 2026-06-17. If the Timefold data model has been further updated, re-verify the YAML schema before implementing any changes.*
