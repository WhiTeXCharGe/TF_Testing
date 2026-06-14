# Postman Setup — Timefold API Service

## 1. Start the server

```bash
cd web/service
npm install
npm run dev      # or: npm start
```

Server runs at **http://localhost:3001**

---

## 2. Create an Environment in Postman

**Environments → Add → Name: "Timefold Local"**

| Variable   | Initial value               | Current value               |
|------------|-----------------------------|-----------------------------|
| `base_url` | `http://localhost:3001`     | `http://localhost:3001`     |
| `run_id`   | *(leave blank)*             | *(filled automatically)*    |

Set this environment as active before running any request.

---

## 3. Create a Collection: "Timefold API"

Add 5 requests in this order:

---

### Request 1 — Upload YAMLs (POST /runSolver)

| Field  | Value                        |
|--------|------------------------------|
| Method | `POST`                       |
| URL    | `{{base_url}}/runSolver`     |

**Body tab → form-data:**

| Key     | Type | Value                  |
|---------|------|------------------------|
| `env`   | File | *(select EnvConfig.yaml)*  |
| `sched` | File | *(select Schedule.yaml)*   |

**Tests tab** — paste this to auto-save the runId:
```js
const res = pm.response.json();
pm.environment.set("run_id", res.runId);
console.log("run_id saved:", res.runId);
```

**Expected response:**
```json
{
  "runId": "20260610_143022500",
  "status": "Submitted"
}
```

**Error cases the server returns:**
- `400` — missing env or sched file, or not a .yaml file
- `413` — file over 50 MB
- `500` — disk write failure

---

### Request 2 — Check Status (GET /status/:runId)

| Field  | Value                            |
|--------|----------------------------------|
| Method | `GET`                            |
| URL    | `{{base_url}}/status/{{run_id}}` |

**Expected response (after upload, before solve):**
```json
{
  "status": "Submitted",
  "stage": null,
  "progress": 0,
  "error": null
}
```

**After Docker runs (or you manually set it):**
```json
{ "status": "Running",   "stage": 1, "progress": 0.4, "error": null }
{ "status": "Running",   "stage": 2, "progress": 0.8, "error": null }
{ "status": "Completed", "stage": 2, "progress": 1,   "error": null }
{ "status": "Failed",    "stage": 1, "progress": 0.2, "error": "OOM at Stage 2" }
```

**Error cases:**
- `400` — runId has invalid characters
- `404` — runId doesn't exist
- `500` — can't read status file

---

### Request 3 — Download Output (GET /download/:runId)

| Field  | Value                              |
|--------|------------------------------------|
| Method | `GET`                              |
| URL    | `{{base_url}}/download/{{run_id}}` |

**To save the file to disk in Postman:**
Send → then click **Save Response → Save to a file**
The YAML file will download to wherever you choose.

**Expected:** file download of `result_Schedule.yaml`

**Error cases:**
- `404` — run not found, or output folder empty
- `409` — run is not Completed yet, or run Failed
- `500` — unexpected error

---

### Request 4 — Set Status Manually (PUT /status/:runId) [TEST HELPER]

*Simulates what Docker writes to the status file — use this without Docker.*

| Field  | Value                            |
|--------|----------------------------------|
| Method | `PUT`                            |
| URL    | `{{base_url}}/status/{{run_id}}` |

**Body → raw → JSON:**
```json
{
  "status": "Completed",
  "stage": 2,
  "progress": 1,
  "error": null
}
```

Other values you can test:
```json
{ "status": "Running",  "stage": 1, "progress": 0.4, "error": null }
{ "status": "Failed",   "stage": 1, "progress": 0.3, "error": "Java heap space exceeded" }
```

---

### Request 5 — Upload Fake Output (POST /output/:runId) [TEST HELPER]

*Simulates what Docker writes to the output folder — use this without Docker.*

| Field  | Value                              |
|--------|------------------------------------|
| Method | `POST`                             |
| URL    | `{{base_url}}/output/{{run_id}}`   |

**Body → form-data:**

| Key      | Type | Value                           |
|----------|------|---------------------------------|
| `result` | File | *(select your result_Schedule.yaml)* |

**Expected:**
```json
{
  "ok": true,
  "runId": "20260610_143022500",
  "file": "result_Schedule.yaml"
}
```

---

## 4. Full test flow (no Docker)

Run these requests in order:

```
1. POST /runSolver          → upload 2 YAMLs → run_id is auto-saved
2. GET  /status/{{run_id}}  → see "Submitted"
3. PUT  /status/{{run_id}}  → body: { "status": "Running", "stage": 1, "progress": 0.3 }
4. GET  /status/{{run_id}}  → see "Running"
5. POST /output/{{run_id}}  → upload your result_Schedule.yaml (fake Docker output)
6. PUT  /status/{{run_id}}  → body: { "status": "Completed", "stage": 2, "progress": 1 }
7. GET  /status/{{run_id}}  → see "Completed"
8. GET  /download/{{run_id}}→ download the YAML → Save Response → Save to a file
```

---

## 5. Test error cases

| Scenario                      | How to trigger                                          | Expected        |
|-------------------------------|---------------------------------------------------------|-----------------|
| Upload non-YAML               | Send a .txt file as `env`                               | `400`           |
| Download before Completed     | Run step 8 before step 6                                | `409` not ready |
| Download a failed run         | PUT status `Failed`, then GET /download                 | `409` failed    |
| Wrong runId                   | GET /status/../../etc/passwd                            | `400` invalid   |
| runId not found               | GET /status/does_not_exist                              | `404`           |

---

## 6. Data folder layout (for reference)

After running the full flow, you will see:

```
web/service/data/
  input/
    20260610_143022500/
      EnvConfig.yaml
      Schedule.yaml
  output/
    20260610_143022500/
      result_Schedule.yaml
  status/
    20260610_143022500.json
```
