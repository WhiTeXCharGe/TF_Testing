# Error Cases — Test Checklist

This document lists every place the system can show an error to the user,
what causes it, and how to trigger it manually for testing.

Use the checkboxes below to track which cases have been verified.

> **Who this is for:** anyone testing or reviewing the webapp and mock server
> setup. Each case is described in plain language — what the user sees, what
> caused it, and how to reproduce it.

---

## How errors are grouped

There are two kinds of errors in this system:

**Solver errors** — the Docker container ran, but something went wrong
*inside* the solve itself. These are reported via the `status.json` file
and appear in the webapp's "Solver Failed" dialog.

**System errors** — something went wrong before or around the solve:
the server was unreachable, the file upload failed, the local database
could not be read, etc. These appear as different dialogs or inline
messages depending on where they happened.

---

## Group A — Solver errors (status = "Failed")

These are triggered by the container writing `"status": "Failed"` to
`status.json`. The webapp shows a **Solver Failed** dialog with the
error message.

To test any of these without running Docker, use Postman's
`PUT /status/{{runId}}` request and paste the JSON body shown below.
Then click **Show Result** in the webapp.

---

- [ ] **A1 — Invalid input data**

  **What the user sees:** "Solver Failed" dialog with a message about
  a bad or missing field in one of the YAML files.

  **What caused it:** The container checked the input files before
  starting the solve and found a problem — for example, `worker_list`
  is empty, `plan_range.start_date` is missing, or the YAML is not
  valid syntax.

  **When it happens:** The solve never starts. `stage` will be `null`
  in the status.

  **How to trigger it (Postman):**
  ```json
  {
    "status": "Failed",
    "stage": null,
    "error": {
      "type": "InvalidInputData",
      "message": "worker_list is empty in EnvConfig.yaml"
    }
  }
  ```

---

- [ ] **A2 — Solver internal error (Stage 1)**

  **What the user sees:** "Solver Failed" dialog with a message about
  an internal Java error or out-of-memory during Stage 1.

  **What caused it:** The Java solver process crashed or ran out of
  memory while running the first solve stage. The JVM exited with a
  non-zero code before finishing.

  **When it happens:** Stage 1 was in progress. `stage` will be `1`.

  **How to trigger it (Postman):**
  ```json
  {
    "status": "Failed",
    "stage": 1,
    "progress": 0.3,
    "error": {
      "type": "SolverError",
      "message": "Java heap space exceeded — increase container memory limit"
    }
  }
  ```

---

- [ ] **A3 — Solver internal error (Stage 2)**

  **What the user sees:** Same "Solver Failed" dialog as A2, but the
  stage shown is Stage 2.

  **What caused it:** Same as A2 but the crash happened during the
  second solve stage. Stage 1 completed successfully; Stage 2 did not.

  **When it happens:** Stage 2 was in progress. `stage` will be `2`.

  **How to trigger it (Postman):**
  ```json
  {
    "status": "Failed",
    "stage": 2,
    "progress": 0.6,
    "error": {
      "type": "SolverError",
      "message": "Uncaught exception in Stage 2 solver thread"
    }
  }
  ```

---

- [ ] **A4 — Output write failure**

  **What the user sees:** "Solver Failed" dialog with a message about
  the result file not being saved.

  **What caused it:** Both solve stages finished successfully, but
  writing `result_Schedule.yaml` to the output folder failed — for
  example, the disk is full or the output volume is read-only.

  **When it happens:** After Stage 2 finishes. `stage` will be `2`.
  There is no output file even though the solve completed.

  **How to trigger it (Postman):**
  ```json
  {
    "status": "Failed",
    "stage": 2,
    "progress": 1.0,
    "error": {
      "type": "OutputError",
      "message": "Failed to write result_Schedule.yaml: No space left on device"
    }
  }
  ```

---

- [ ] **A5 — Unknown error**

  **What the user sees:** "Solver Failed" dialog with a generic or
  unexpected error message.

  **What caused it:** An error happened that does not fit any of the
  categories above. Used as a catch-all.

  **How to trigger it (Postman):**
  ```json
  {
    "status": "Failed",
    "stage": 1,
    "error": {
      "type": "UnknownError",
      "message": "Unexpected signal received — container may have been killed"
    }
  }
  ```

---

## Group B — Network / API call errors

These happen when the webapp makes an HTTP request to the server
(`localhost:3001`) and the request itself fails — the server is down,
returned an unexpected error code, or the connection timed out.

The webapp shows a **Solver Error** dialog (different from the
"Solver Failed" dialog in Group A) with the raw error message.

---

- [ ] **B1 — Upload to solver fails**

  **What the user sees:** The New Run modal closes (the local copy
  succeeded), but a "Solver Error" dialog appears immediately
  afterward saying the upload to the solver failed.

  **What caused it:** The webapp saved the files locally first, then
  tried to `POST /runSolver` to the backend. That second request
  failed — typically because the server is not running.

  **How to trigger it:** Stop the Express server (`Ctrl+C` in
  Terminal 1), then click **New Run** and submit files in the webapp.

  **Note:** The run row still appears in the list because the local
  save succeeded. Only the solver submission failed.

---

- [ ] **B2 — Status check fails**

  **What the user sees:** A "Solver Error" dialog when clicking
  **Show Result** on a run.

  **What caused it:** The webapp tried to `GET /status/:runId` but
  the server did not respond. The webapp never learned the current
  status.

  **How to trigger it:** Stop the Express server, then click
  **Show Result** on any run in the webapp.

---

- [ ] **B3 — Download fails**

  **What the user sees:** A "Solver Error" dialog when the webapp
  tries to download the result file after the user confirms in the
  Completed dialog.

  **What caused it:** The webapp tried to `GET /download/:runId`
  but the server returned an error or was unreachable. This can also
  happen if the output file was manually deleted from the server
  after the status showed Completed.

  **How to trigger it:** Using Postman, first set status to
  `Completed`. Then stop the Express server. Then click **Show
  Result** → **Download** in the webapp.

---

## Group C — "Not done yet" states

These are not errors — the solve is still in progress or was stopped.
The webapp shows a **Solver Status** dialog describing the current
state. They are included here because they need to be tested the same
way as errors (via `PUT /status`).

---

- [ ] **C1 — Status is Submitted**

  **What the user sees:** A status dialog saying the run has been
  submitted and the container is starting up.

  **What it means:** The files were uploaded successfully. Docker has
  not yet written the first `Running` status to `status.json`.

  **How to trigger it (Postman):**
  ```json
  { "status": "Submitted" }
  ```

---

- [ ] **C2 — Status is Running**

  **What the user sees:** A status dialog showing the current stage
  and progress percentage.

  **What it means:** The container is actively solving. Progress is
  between 0% and 100% within the current stage.

  **How to trigger it (Postman):**
  ```json
  { "status": "Running", "stage": 1, "progress": 0.45 }
  ```

---

- [ ] **C3 — Status is Cancelled**

  **What the user sees:** A status dialog saying the run was
  cancelled.

  **What it means:** The Docker container received a stop signal
  (`docker stop tf-solver` or Ctrl+C) while it was running and exited
  cleanly. There is no output file.

  **How to trigger it (Postman):**
  ```json
  { "status": "Cancelled", "stage": 1 }
  ```

---

## Group D — Local mode errors

These only happen when `VITE_API_BASE_URL` is **not** set — i.e., the
webapp is running in local-only mode without a backend solver connection.

---

- [ ] **D1 — Output YAML not found (local mode)**

  **What the user sees:** A "Not Ready" dialog saying the result is
  not available yet.

  **What caused it:** The webapp looked for a `result_Schedule.yaml`
  file inside `public/local/<runId>/output/` and found nothing there.
  The solve either has not finished or no solver was ever connected.

  **How to trigger it:** Make sure `VITE_API_BASE_URL` is not set in
  `webapp/.env`. Submit a new run. Click **Show Result** before
  placing any output YAML in the output folder.

---

## Group E — Startup and data errors

These happen when the webapp first loads, before any action is taken.

---

- [ ] **E1 — Run list fails to load**

  **What the user sees:** A red error message inside the run table
  that says "Failed to load runs.json — " followed by the error
  detail. The table is otherwise empty.

  **What caused it:** The webapp calls `GET /api/runs` on startup to
  load the list of past runs. If that request fails (dev server not
  running, `runs.json` is corrupt, or the middleware crashed), the
  entire run list cannot be displayed.

  **How to trigger it:** Stop the Vite dev server while the browser
  tab is open, then refresh the page.

---

- [ ] **E2 — File upload rejected by local API**

  **What the user sees:** An inline red error message inside the
  New Run modal, below the file pickers.

  **What caused it:** The user selected files and clicked Submit, but
  the local `POST /api/upload` call returned a non-OK response. This
  can happen if the server-side middleware rejected the file (wrong
  type, file too large, disk write failed, etc.).

  **How to trigger it:** Modify the Vite dev server middleware to
  return a 500 on `/api/upload`, then try to submit a new run through
  the modal.

---

## Quick reference

| Code | Dialog shown | Group |
|---|---|---|
| A1 | Solver Failed — InvalidInputData | Solver error |
| A2 | Solver Failed — SolverError (Stage 1) | Solver error |
| A3 | Solver Failed — SolverError (Stage 2) | Solver error |
| A4 | Solver Failed — OutputError | Solver error |
| A5 | Solver Failed — UnknownError | Solver error |
| B1 | Solver Error — upload failed | Network error |
| B2 | Solver Error — status check failed | Network error |
| B3 | Solver Error — download failed | Network error |
| C1 | Solver Status — Submitted | Not done yet |
| C2 | Solver Status — Running | Not done yet |
| C3 | Solver Status — Cancelled | Not done yet |
| D1 | Not Ready (local mode) | Local mode |
| E1 | Inline table error — runs.json | Startup |
| E2 | Inline modal error — upload rejected | Startup |
