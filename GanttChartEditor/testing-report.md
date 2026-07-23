# GanttChartEditor — Testing Report

**Date:** 2026-06-26  
**Project:** GanttChart Editor (React 19 + TypeScript + Vite)  
**Testing Frameworks:** Jest 29 (unit / integration) · Cypress 13 (end-to-end)

---

## 1. Overview

This project uses two complementary testing tools:

| Tool | Purpose | What it tests |
|------|---------|---------------|
| **Jest** | Unit & integration tests | Pure functions, state reducers, data-model builders |
| **Cypress** | End-to-end (E2E) tests | Real browser interactions — clicking, typing, file loading |

Think of them like this: Jest checks that each engine part works correctly in isolation; Cypress checks that the whole car drives properly on the road.

---

## 2. Setup on a New PC

Follow these steps exactly on any new machine to get tests running.

### Step 1 — Requirements

| Tool | Required version | Check command |
|------|-----------------|---------------|
| Node.js | **v18.x** (not v20+) | `node -v` |
| npm | v9 or later | `npm -v` |

> **Why Node 18?** Cypress 13 requires Node 18. Cypress 15 requires Node 20+ but has SSL certificate issues in some corporate networks. Stick with Node 18 + Cypress 13.

### Step 2 — Clone or copy the project

Make sure all these config files are present (they are already in the repository):

```
jest.config.cjs
tsconfig.jest.json
cypress.config.cjs
src/setupTests.ts
src/__mocks__/fileMock.cjs
cypress/support/commands.ts
cypress/support/e2e.ts
cypress/fixtures/envConfig.yaml
cypress/fixtures/schedule.yaml
```

### Step 3 — Install dependencies

```powershell
npm install --legacy-peer-deps
```

The `--legacy-peer-deps` flag is required because `gantt-task-react` declares a React 18 peer dependency but this project uses React 19.

### Step 4 — Verify Jest works

```powershell
npm run test:jest
```

Expected output:
```
PASS  src/__tests__/utils/colorUtils.test.ts
PASS  src/__tests__/utils/dateUtils.test.ts
PASS  src/__tests__/gantt/workerViewModel.test.ts
PASS  src/__tests__/context/reducer.test.ts
PASS  src/__tests__/services/yamlService.test.ts

Test Suites: 5 passed, 5 total
Tests:       94 passed, 94 total
Time:        ~9 seconds
```

### Step 5 — Verify Cypress works

Open **two terminals**:

```powershell
# Terminal 1 — start the app
npm run dev
```

```powershell
# Terminal 2 — open Cypress
npm run test:cypress
```

A Cypress window will open. Click any `.cy.ts` file to run that test suite.

---

## 3. Known Issues & Version Rules

| Issue | Cause | Fix |
|-------|-------|-----|
| Jest crashes with out-of-memory after 4 minutes | Jest 30 + ts-jest 29 are incompatible | `package.json` must have `jest@^29.7.0`, not `^30` |
| `npm install` fails with peer conflict | `gantt-task-react` needs React 18, project uses React 19 | Always use `npm install --legacy-peer-deps` |
| Cypress shows blank white window | Normal — takes 10–20 seconds to load on first run | Just wait |
| Cypress shows `exports is not defined in ES module scope` | Config file is `.ts` instead of `.cjs` | Use `cypress.config.cjs` (already fixed in repo) |
| Cypress 15 shows `Invalid regular expression flags` | Node 18 is too old for Cypress 15 | Use Cypress 13 (`cypress@^13`) |
| `moduleViewModel.test.ts` crashes with OOM | ts-jest memory issue specific to this file on Windows | Excluded from default run; 94 other tests still run |

---

## 4. How to Run the Tests

### Run Jest (unit tests)
```powershell
npm run test:jest
```

Watch mode (re-runs when you save a file):
```powershell
npm run test:jest -- --watch
```

### Run Cypress (E2E tests — interactive)
```powershell
# Terminal 1
npm run dev

# Terminal 2
npm run test:cypress
```

### Run Cypress (headless — no browser window)
```powershell
# Terminal 1
npm run dev

# Terminal 2
npm run test:cypress:run
```

---

## 5. Jest Test Results

**Result: 94 tests passed across 5 test suites — completed in ~9 seconds.**

```
PASS  src/__tests__/utils/colorUtils.test.ts
PASS  src/__tests__/utils/dateUtils.test.ts
PASS  src/__tests__/gantt/workerViewModel.test.ts
PASS  src/__tests__/context/reducer.test.ts
PASS  src/__tests__/services/yamlService.test.ts

Test Suites: 5 passed, 5 total
Tests:       94 passed, 94 total
Time:        8.002 s
```

> **Note on `moduleViewModel.test.ts`:** Excluded from the default run due to a ts-jest memory issue on Windows where the worker process crashes after ~4 minutes. All other areas are covered by the 94 passing tests.

### 5.1 `utils/dateUtils.test.ts` — 34 tests ✅
| Function | What is tested |
|----------|----------------|
| `normalizeDate` | Strips time, handles null/undefined, YYYY/MM/DD → YYYY-MM-DD |
| `parseDate` | Parses YYYY-MM-DD strings into Date objects |
| `formatDate` | Outputs YYYY-MM-DD from Date |
| `formatDateShort` | Short format MM/DD |
| `addDays` | Adds/subtracts days across month/year boundaries |
| `diffDays` | Difference in days between two dates |
| `generateDateRange` | Generates all dates between two bounds (inclusive) |
| `getDayOfWeek` | Returns 0 (Sun) – 6 (Sat) |
| `isWeekend` | Returns true for Saturday and Sunday |
| `isDateInList` | Checks if a date string appears in an array |

### 5.2 `utils/colorUtils.test.ts` — 12 tests ✅
| Function | What is tested |
|----------|----------------|
| `getColorForDevice` | Consistent hex color per device ID |
| `getColorForPhaseIndex` | Different colors for different phase indices |
| `lightenColor` | Hex color lightening by percentage |

### 5.3 `services/yamlService.test.ts` — 18 tests ✅
| Test area | What is tested |
|-----------|----------------|
| `parseScheduleYaml` | Parses workers, modules, assignments from YAML |
| `parseEnvConfigYaml` | Parses workflows, phases, operations, workers, fab info |
| `workload_hours` | `op.workloadHours` is correctly read from YAML field |
| `minWorkerNum / maxWorkerNum` | Worker count defaults are read correctly |
| Round-trip | `stringifyScheduleYaml(parseScheduleYaml(yaml))` produces valid re-parseable YAML |

### 5.4 `context/reducer.test.ts` — 22 tests ✅
| Action | What is tested |
|--------|----------------|
| `SWITCH_VIEW` | Switches between worker/device views |
| `SELECT_ASSIGNMENT` | Sets selected assignment ID |
| `UPDATE_ASSIGNMENT` | Updates an existing assignment's fields |
| `DELETE_ASSIGNMENT` | Removes assignment; clears selection |
| `UNDO / REDO` | Reverts and re-applies changes |
| `SET_WORKER_VIEW_FILTER` | Partial update merges into existing filter state |
| `SET_MODULE_VIEW_FILTER` | Partial update merges into existing filter state |
| `OPEN/CLOSE dialogs` | Dialog open/close state flags |
| `SET_ERROR` | Sets and clears error message |
| `ADD_WORKFLOW_TASKS` | Adds new workflow tasks to schedule |
| `LOAD_FILES` | Resets filters to defaults when files are loaded |

### 5.5 `gantt/workerViewModel.test.ts` — 8 tests ✅
| Scenario | What is tested |
|----------|----------------|
| Worker inclusion | Workers with assignments appear in the model |
| Worker exclusion | Workers with no assignments and no off-dates are excluded |
| Unavailable dates | Workers with off-dates appear even without assignments |
| Segments | Assignment segments have correct start/end date indices |
| Worker metadata | Company name and worker name are correctly populated |

---

## 6. Cypress E2E Test Suites

All spec files are in `cypress/e2e/`. Fixtures are in `cypress/fixtures/`.

> Requires `npm run dev` running in a separate terminal before starting Cypress.

### 6.1 `01_empty_state.cy.ts` — 9 tests ✅
- App title is visible
- File menu opens correctly
- Action buttons (割付追加, 新規製番追加) are disabled before loading
- Empty state message is shown
- File dialog can be opened and cancelled

### 6.2 `02_file_loading.cy.ts` — 4 tests ✅
- Dialog shows EnvConfig and Schedule file inputs
- Loading fixture YAML files succeeds and renders worker names
- Status bar shows success message after loading
- Action buttons become enabled after loading

### 6.3 `03_worker_view_filter.cy.ts` — 10 tests ✅
- Filter chips (装置, 工程, Fab, Region) are visible in Worker View
- Bar name text input works and accepts text
- Typing a bar name triggers クリア button
- クリア button resets the filter
- 装置 chip is visible and clickable
- 装置 dropdown shows SU-1001 and SU-1002
- Fab dropdown shows Osaka Fab
- Date range inputs are visible
- Module View shows 作業者 / Fab / Region chips
- Misc tasks (wf_misc) do not appear as module rows

### 6.4 `04_dialogs.cy.ts` — 9 tests ✅
- **割付追加:** opens dialog, shows radio options, shows 装置 label, can cancel
- **新規製番追加:** opens dialog, shows tabs, wf_misc not in dropdown, can cancel
- **Undo/Redo:** 元に戻す and やり直し buttons visible in toolbar

---

## 7. Test Data (Fixtures)

### `cypress/fixtures/envConfig.yaml`
- 1 standard workflow (`wf_standard`) with 2 phases and 3 operations
- 1 misc workflow (`wf_misc` — "Other Work") for filter testing
- 1 fab location ("Osaka Fab", Region: "Kansai")
- 3 workers: Alice Tanaka, Bob Yamada, Carol Sato

### `cypress/fixtures/schedule.yaml`
- Module `SU-1001` — 2 phases
- Module `SU-1002` — 1 phase
- `misc001` — a misc task (filtered out of Device View)
- 4 assignments linking workers to tasks

---

## 8. Reading Test Results

### Jest
```
PASS  src/__tests__/utils/dateUtils.test.ts
  ✓ normalizeDate handles null (3 ms)
  ✓ parseDate parses YYYY-MM-DD string (1 ms)
```
- `✓` = passed · `✗` = failed (shows expected vs actual diff below)

### Cypress
In interactive mode: live browser with steps checked off in green/red.  
In headless mode: summary table per spec file.  
Screenshots of failures → `cypress/screenshots/`

---

## 9. Adding New Tests

### New Jest test
Create `src/__tests__/<folder>/myFeature.test.ts`:
```typescript
import { myFunction } from '../../utils/myUtils';

describe('myFunction', () => {
  it('returns expected result', () => {
    expect(myFunction('input')).toBe('expected output');
  });
});
```
Run: `npm run test:jest`

### New Cypress test
Create `cypress/e2e/05_myFeature.cy.ts`:
```typescript
describe('My Feature', () => {
  beforeEach(() => cy.visit('/'));

  it('does something', () => {
    cy.contains('button', 'Click Me').click();
    cy.contains('Result').should('be.visible');
  });
});
```
Run: `npm run dev` in one terminal, then `npm run test:cypress` in another.
