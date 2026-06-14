# How to Run This System — Beginner Guide

> **Who this is for:** Someone who is new to this project and wants to understand
> what each piece does and how to get it running. No coding experience required.

---

## What does this system do?

This system is an **employee scheduling tool**.

You give it two files:
- A list of your employees and their information (`EnvConfig.yaml`)
- A list of what shifts need to be filled (`Schedule.yaml`)

The system figures out the best way to assign employees to shifts and gives you
back a finished schedule as a file (`result_Schedule.yaml`).

---

## The 3 parts

Think of it like a restaurant with a kitchen, a waiter, and a customer.

```
┌─────────────────────┐     ┌─────────────────────┐     ┌─────────────────────┐
│   1. Webapp          │     │   2. Server          │     │   3. Docker Solver   │
│   (the customer)     │────>│   (the waiter)       │────>│   (the kitchen)      │
│                      │     │                      │     │                      │
│ A website you open   │     │ A program running    │     │ A container that     │
│ in your browser.     │     │ in the background.   │     │ runs the Java solver │
│ You click buttons    │     │ It receives your     │     │ and does the actual  │
│ and upload files.    │     │ files, saves them,   │     │ math to build the    │
│                      │     │ and starts Docker.   │     │ schedule.            │
└─────────────────────┘     └─────────────────────┘     └─────────────────────┘
         │                           │                           │
    web/webapp/                web/service/               web/Timefold/
    (React website)            (server.js)                (Dockerfile + Java)
    runs on port 5173          runs on port 3001          runs inside Docker
```

---

## What you need installed (one-time setup)

| Tool | What it is | How to get it |
|---|---|---|
| **Node.js** | Runs the server and webapp | nodejs.org — download and install |
| **Docker Desktop** | Runs the solver container | docs.docker.com/desktop/install/windows-install/ |
| **Git Bash** (or any terminal) | Where you type commands | Comes with Git for Windows |
| **Postman** | A tool to send test requests to the server | postman.com/downloads |

> After installing Docker Desktop, open it from the Start menu and wait for the
> whale icon in the taskbar to stop animating. It must be running every time you
> use Docker commands.

---

## How to run — step by step

You need **three terminal windows** open at the same time (one per part).

---

### Step 1 — Build the Docker image (first time only, ~5 minutes)

Open a terminal and go to the `Timefold` folder:

```bash
cd C:/Users/Seiya/Desktop/work/Timefold/web/Timefold
docker compose build solver
```

This downloads Java, the solver library, and compiles the code into a container.
**You only need to do this once.** Future runs skip all the downloading.

When it finishes you will see: `timefold-scheduler    local    ...`

---

### Step 2 — Start the Server (Terminal 1)

```bash
cd C:/Users/Seiya/Desktop/work/Timefold/web/service
npm install        # first time only — downloads the server's dependencies
npm run dev        # starts the server
```

You will see:
```
Timefold API service running at http://localhost:3001
```

Leave this terminal open. This is the "waiter" — it handles all communication
between the webapp and Docker.

---

### Step 3 — Start the Webapp (Terminal 2)

```bash
cd C:/Users/Seiya/Desktop/work/Timefold/web/webapp
npm install        # first time only
npm run dev        # starts the website
```

You will see something like:
```
Local: http://localhost:5173/
```

Open that address in your browser. You will see the scheduling website.

---

### Step 4 — Use the website

1. Open `http://localhost:5173` in your browser
2. Upload your `EnvConfig.yaml` and `Schedule.yaml` files
3. Click the button to start the solve
4. Wait — the page will show you the progress (Stage 1 → Stage 2 → Completed)
5. Download the result when it is done

The Docker container starts automatically in the background when you click submit.
You can watch it running in **Terminal 1** (the server terminal).

---

## What happens behind the scenes (the full flow)

```
You click Submit
      │
      ▼
Webapp sends the 2 YAML files to http://localhost:3001/runSolver
      │
      ▼
Server saves the files to: web/service/data/input/<runId>/
Server starts a Docker container with those files mounted in
      │
      ▼
Docker container (the solver):
  - Writes status = "Running" to data/status/<runId>.json
  - Runs the Java solver (Stage 1 then Stage 2)
  - Writes the result to data/output/<runId>/result_Schedule.yaml
  - Writes status = "Completed"
      │
      ▼
Webapp keeps asking GET /status/<runId> every few seconds
      │
      ▼
When status = "Completed", webapp shows Download button
      │
      ▼
You click Download → GET /download/<runId> → file arrives in your browser
```

---

## What is Postman and why do we use it?

**Postman** is a free tool that lets you send HTTP requests (like clicking a
button on a website) directly to the server — without needing to open the webapp.

Think of it as a "remote control" for the server.

**Why is this useful?**

- You can test the server on its own, without the webapp being ready
- You can fake different situations (like pretending the solver failed) to check
  that the webapp handles errors correctly
- You can see the exact data the server sends back, which helps with debugging

### What is an HTTP request?

When your browser loads a page, it sends a "request" to a server and gets a
"response" back. Postman lets you manually send these requests and see the response.

- **GET** — "Give me some information" (like checking status)
- **POST** — "Here is some data, please process it" (like uploading files)
- **PUT** — "Replace this with new data" (used for test helpers)

---

## Using Postman to test the server

The project includes a ready-made Postman collection:
`document/Timefold-Postman-Collection.json`

### Import the collection

1. Open Postman
2. Click **Import** (top left)
3. Drag in the file `Timefold-Postman-Collection.json`
4. The collection named **"Timefold Solver — Local Test Suite"** will appear

### Set the file paths (one-time setup)

In the collection, click the name → **Variables** tab → update these two lines
to point to your actual YAML files:

| Variable | Value |
|---|---|
| `envFilePath` | `C:/Users/Seiya/Desktop/work/Timefold/web/Timefold/work/input/EnvConfig.yaml` |
| `schedFilePath` | `C:/Users/Seiya/Desktop/work/Timefold/web/Timefold/work/input/Schedule.yaml` |

---

### The 3 main requests (the normal flow)

Run them in this order:

**Request 1 — POST /runSolver**

Uploads your two YAML files to the server. The server saves them and immediately
starts a Docker container to solve the schedule.

Response you get back:
```json
{ "runId": "20260615_143022500", "status": "Submitted" }
```

The `runId` is automatically saved so requests 2 and 3 use it automatically.

---

**Request 2 — GET /status/{{runId}}**

Asks the server "how is the solve going?"

Keep clicking **Send** to watch it change:
```
Submitted → Running (stage 1) → Running (stage 2) → Completed
```

When you see `"status": "Completed"` — move to request 3.

---

**Request 3 — GET /download/{{runId}}**

Downloads the finished schedule file.

In Postman: after clicking Send, click **Save Response → Save to a file**
to save `result_Schedule.yaml` to your computer.

---

### The error injection requests (for testing only)

These let you pretend the solver had a problem, so you can check that the webapp
shows error messages correctly — without actually running Docker.

**PUT /status/{{runId}}**

Manually sets the status to anything you want. For example:
```json
{ "status": "Failed", "stage": 1, "error": { "type": "SolverError", "message": "Out of memory" } }
```

After you do this, open the webapp and click "Show Result" — it should display
the error message you wrote.

---

## Status values explained

| Status | What it means |
|---|---|
| `Submitted` | Files uploaded, Docker container is starting up |
| `Running` | Solver is actively working (Stage 1 or Stage 2) |
| `Completed` | Done! Result file is ready to download |
| `Failed` | Something went wrong — see the `error` field for details |
| `Cancelled` | You stopped it (Ctrl+C or docker stop) |

---

## Cancel a running solve

If you started the solver and want to stop it:

```bash
docker stop tf-solver
```

The status file will show `"status": "Cancelled"`. There will be no output file
for a cancelled run — partial results are discarded.

---

## Files that the system creates

While the system is running, it creates these files automatically:

```
web/service/data/
  input/
    <runId>/
      EnvConfig.yaml      ← your uploaded employee file
      Schedule.yaml       ← your uploaded schedule file
  output/
    <runId>/
      result_Schedule.yaml ← the finished schedule (appears when Completed)
  status/
    <runId>.json          ← current status, updated by the Docker container
```

You can open `status/<runId>.json` in any text editor to see exactly what
the solver is reporting.

---

## Quick reference — daily use

After the first-time setup, daily use is just:

```bash
# Terminal 1 — start server
cd web/service && npm run dev

# Terminal 2 — start webapp
cd web/webapp && npm run dev

# Then open http://localhost:5173 in your browser
```

The Docker container starts automatically when you upload files through the webapp.

---

## Troubleshooting

| Problem | Likely cause | Fix |
|---|---|---|
| `Cannot connect to Docker daemon` | Docker Desktop not open | Open Docker Desktop from Start menu, wait for whale icon |
| Server does not start | Port 3001 already in use | Close any other programs using port 3001 |
| `run_id not found` in Postman | Request 1 was not run first | Run request 1 first to create a run |
| Download returns 409 | Status is not Completed yet | Keep polling request 2 until status = Completed |
| Solver shows Failed | Check the `error` field in status | See the error type — InvalidInputData means your YAML has a problem |
| `permission denied: entrypoint.sh` | Windows line endings in the script | See DockerGuide.md section 8 |
