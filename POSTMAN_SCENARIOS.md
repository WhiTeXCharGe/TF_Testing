# Postman Test Scenarios — Show Result Button

## Setup (do once)

### 1. Start the backend service
```bash
cd web/service
npm install
npm run dev        # runs on http://localhost:3001
```

### 2. Start the webapp
```bash
cd web/webapp
# create .env if it doesn't exist
echo VITE_API_BASE_URL=http://localhost:3001 > .env
npm run dev        # runs on http://localhost:5173
```

### 3. Postman environment
Create environment **"Timefold Local"** with:

| Variable   | Value                   |
|------------|-------------------------|
| `base_url` | `http://localhost:3001` |
| `run_id`   | *(filled by Test script)* |

---

## Step 0 — Upload YAML files (New Run button test)

This verifies the **New Run** button sends files to the backend.

**Request: POST /runSolver**

| Field  | Value                    |
|--------|--------------------------|
| Method | POST                     |
| URL    | `{{base_url}}/runSolver` |
| Body   | form-data                |

Body form-data fields:

| Key     | Type | Value                 |
|---------|------|-----------------------|
| `env`   | File | EnvConfig.yaml        |
| `sched` | File | Schedule.yaml         |

**Tests tab** (auto-saves runId for next requests):
```js
const res = pm.response.json();
pm.environment.set("run_id", res.runId);
console.log("Saved run_id:", res.runId);
```

**Expected response — 202:**
```json
{
  "runId": "20260610_143022500",
  "status": "Submitted"
}
```

**What to check in webapp:**
- Click "New Run", select the same YAMLs, click Submit Run
- The run appears in the run list
- On the backend, `data/input/{runId}/` now contains both YAML files

---

## Scenario 1 — Show Result: NOT ready (solver still running)

**Goal:** Click "Show Result" → webapp calls `/status/:runId` → backend returns
`Running` → webapp shows "Solver is still running." dialog.

### Step 1-A — Ensure status is "Running"

**Request: PUT /status/:runId**

| Field  | Value                              |
|--------|------------------------------------|
| Method | PUT                                |
| URL    | `{{base_url}}/status/{{run_id}}`   |
| Body   | raw → JSON                         |

Body:
```json
{
  "status": "Running",
  "stage": 1,
  "progress": 0.35,
  "error": null
}
```

Expected response — 200:
```json
{ "ok": true, "runId": "...", "status": "Running" }
```

### Step 1-B — Click "Show Result" in the webapp

1. Open http://localhost:5173
2. Find the run row (same runId)
3. Click **Show Result**

**Expected result in web:**
> Dialog opens with title **"ソルブ実行中"** (or "Solve in progress")
> Body: "Solver is still running. (Stage 1, 35%)"

### Step 1-C — Verify via Postman (optional)

**Request: GET /status/:runId**

| Field  | Value                              |
|--------|------------------------------------|
| Method | GET                                |
| URL    | `{{base_url}}/status/{{run_id}}`   |

Expected response — 200:
```json
{
  "status": "Running",
  "stage": 1,
  "progress": 0.35,
  "error": null
}
```

---

## Scenario 2 — Show Result: READY (solver completed, download YAML)

**Goal:** Click "Show Result" → webapp calls `/status/:runId` → backend returns
`Completed` → webapp calls `/download/:runId` → YAML file downloads to disk.

### Step 2-A — Upload a fake output YAML

**Request: POST /output/:runId**

| Field  | Value                              |
|--------|------------------------------------|
| Method | POST                               |
| URL    | `{{base_url}}/output/{{run_id}}`   |
| Body   | form-data                          |

Body form-data:

| Key      | Type | Value                           |
|----------|------|---------------------------------|
| `result` | File | result_Schedule.yaml (any yaml) |

Expected response — 200:
```json
{
  "ok": true,
  "runId": "...",
  "file": "result_Schedule.yaml"
}
```

### Step 2-B — Set status to Completed

**Request: PUT /status/:runId**

| Field  | Value                              |
|--------|------------------------------------|
| Method | PUT                                |
| URL    | `{{base_url}}/status/{{run_id}}`   |
| Body   | raw → JSON                         |

Body:
```json
{
  "status": "Completed",
  "stage": 2,
  "progress": 1,
  "error": null
}
```

### Step 2-C — Click "Show Result" in the webapp

1. Open http://localhost:5173
2. Find the run row
3. Click **Show Result**

**Expected result in web:**
> No dialog — the browser immediately downloads **result_Schedule.yaml** to your Downloads folder.

### Step 2-D — Verify download via Postman (optional)

**Request: GET /download/:runId**

| Field  | Value                               |
|--------|-------------------------------------|
| Method | GET                                 |
| URL    | `{{base_url}}/download/{{run_id}}`  |

Click **Send** → then **Save Response → Save to a file** to download the YAML.

Expected: file downloads as `result_Schedule.yaml`

---

## Scenario 3 — Solver failed (bonus error case)

**Goal:** Click "Show Result" → webapp shows "Solver failed" dialog with error detail.

### Step 3-A — Set status to Failed

**Request: PUT /status/:runId** — body:
```json
{
  "status": "Failed",
  "stage": 1,
  "progress": 0.2,
  "error": "Java heap space exceeded at Stage 1 after 8 minutes"
}
```

### Step 3-B — Click "Show Result" in the webapp

**Expected result in web:**
> Dialog opens with title **"ソルバが失敗しました"** (or "Solver failed")
> Error line: "Java heap space exceeded at Stage 1 after 8 minutes"

---

## Summary of all Postman requests used

| # | Method | URL                          | Purpose                        |
|---|--------|------------------------------|--------------------------------|
| 0 | POST   | `/runSolver`                 | Upload YAMLs (New Run button)  |
| 1 | PUT    | `/status/{{run_id}}`         | Set status (simulate Docker)   |
| 2 | POST   | `/output/{{run_id}}`         | Upload fake output YAML        |
| 3 | GET    | `/status/{{run_id}}`         | Verify current status          |
| 4 | GET    | `/download/{{run_id}}`       | Download output YAML           |
