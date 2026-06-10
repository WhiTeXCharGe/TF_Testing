# Postman Mock Setup — Timefold API

> **Note:** Postman's UI changes between versions. If a step doesn't match what you
> see, check the section **"If the UI looks different"** at the bottom of this file,
> or update this doc with what actually worked.
>
> Last verified against: **Postman v11 (2025)**

---

## What this does

Postman acts as a fake backend server so you can test the webapp buttons
without Azure or Docker running. The webapp calls the mock URL exactly
like it would call the real API.

---

## Step 1 — Import the collection

1. Open Postman
2. Click **Import** (top-left area, near the sidebar)
3. Drag and drop `timefold-mock.postman_collection.json` (this folder)  
   — OR — click **files** and browse to it
4. Click **Import**

You should see **"Timefold API Mock"** appear in your **Collections** sidebar.

---

## Step 2 — Create the Mock Server

> In the current Postman version (v10/v11), Mock Server is created FROM a collection,
> not from the New menu.

1. In the left sidebar, hover over **"Timefold API Mock"**
2. Click the **`...`** (three dots) that appear on the right
3. Click **"Mock collection"**  
   _(it may also say "Add mock" or "Create mock server" depending on version)_
4. Give it any name, e.g. `timefold-local`
5. Leave environment as **No Environment**
6. Click **Create Mock Server**
7. **Copy the mock URL** — it looks like:  
   `https://xxxxxxxx-xxxx-xxxx-xxxx-xxxxxxxxxxxx.mock.pstmn.io`

> **Requirement:** You must be logged into a Postman account.  
> Free tier = 1,000 mock calls/month (more than enough for testing).

---

## Step 3 — Point the webapp at the mock

Open `webapp/.env` and replace the URL:

```
VITE_API_BASE_URL=https://xxxxxxxx-xxxx-xxxx-xxxx-xxxxxxxxxxxx.mock.pstmn.io
```

Restart the webapp:
```bash
cd web/webapp
npm run dev
```

---

## Step 4 — Run the test scenarios

### Scenario 1 — "Show Result" button → not ready dialog

**What happens:** webapp calls `GET /status/:runId` → mock returns `Running`
→ webapp shows *"Solver is still running"* dialog

**Setup (default — no changes needed):**
The collection already has **"Scenario 1 — Running"** as the first example,
so the mock returns `Running` by default.

**Test:**
1. Open `http://localhost:5173`
2. Click **New Run** → upload any 2 YAML files → Submit
3. Click **Show Result** on the new row
4. ✅ Dialog appears: *"ソルブ実行中 / Solve in progress — Solver is still running. (Stage 1, 35%)"*

---

### Scenario 2 — "Show Result" button → YAML downloads

**What happens:** webapp calls `GET /status/:runId` → mock returns `Completed`
→ webapp immediately calls `GET /download/:runId` → mock returns YAML text
→ file saves to your Downloads folder

**Setup — switch the active example:**
1. In the sidebar, expand **"Timefold API Mock"** collection
2. Click **"GET /status/:runId — Check Status"**
3. In the right panel, click the **Examples** tab (top right of the request panel)
4. Drag **"Scenario 2 — Completed"** to the **top** of the examples list
   _(the mock always uses the first example)_

**Test:**
1. Back in the webapp, click **Show Result** on any run
2. ✅ `result_Schedule.yaml` downloads automatically to your Downloads folder

**To switch back to Scenario 1:** drag "Scenario 1 — Running" back to the top.

---

### Scenario 3 — Solver failed (bonus)

Drag **"Scenario 3 — Failed"** to the top of the examples list.

**Test:** Click Show Result → dialog shows:
*"Solver failed — Java heap space exceeded at Stage 1 after 8 minutes"*

---

## Step 5 — Test New Run button upload

1. In the webapp, click **New Run**
2. Drop `EnvConfig.yaml` and `Schedule.yaml`
3. Click **Submit Run**

✅ The modal closes and the run appears in the list.

> **Note:** The mock receives the request and returns `{"runId":"test-run-001","status":"Submitted"}`.
> The YAML files are **not stored** — Postman mock ignores the file contents and always
> returns the fixed response. This is a Postman limitation. To actually store files,
> the real backend (`web/service/server.js`) is needed.

---

## Updating the mock responses

To change what the mock returns (e.g. different progress %, different error message):

1. In the collection, click the request
2. Click the **Examples** tab
3. Click the example you want to edit
4. Change the JSON in the response body
5. Save (Ctrl+S)

The mock server updates immediately — no restart needed.

---

## If the UI looks different

Postman updates its UI regularly. If the steps above don't match:

| What you're trying to do | Where to look |
|--------------------------|---------------|
| Create a mock server | Collection `...` menu → look for "Mock", "Add mock", "Mock collection" |
| Find existing mock servers | Left sidebar → look for a server/cloud icon, or search "mock" |
| Switch active example | Request → Examples tab (may be in a dropdown or panel) |
| Find mock URL | Left sidebar → Mock Servers → click mock → copy URL |

**If you find the correct path for your Postman version, update this file here:**

```
Last verified against: Postman vXX (YYYY-MM)
Step 2 — actual path: [write what worked]
```

---

## Files in this folder

| File | Purpose |
|------|---------|
| `timefold-mock.postman_collection.json` | Import this into Postman |
| `README.md` | This guide |
