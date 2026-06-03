# Timefold Scheduler — Docker Guide

How to run the Timefold solver inside a Docker container. No Java install
needed on your machine — Docker bundles everything.

> **Why Docker?** The container is a portable, self-contained box: OS + JDK 17
> + the solver jar + all dependencies. It runs the same way on any machine
> (your laptop, a colleague's PC, an Azure server). You only need Docker
> Desktop installed; nothing else.

---

## 0. Install Docker Desktop (one-time per machine)

1. Download **Docker Desktop for Windows**: https://docs.docker.com/desktop/install/windows-install/
2. Run the installer with default options. When asked, **enable WSL 2 backend**.
3. Restart your computer when it asks.
4. Open **Docker Desktop** from the Start menu. The whale icon in the system
   tray (bottom-right) should be **steady** (not animating) when it's ready —
   takes ~30 seconds the first time.
5. Open Git Bash (or any terminal) and verify:
   ```bash
   docker --version
   docker compose version
   ```
   Both should print version numbers.

> **Docker Desktop must be running every time you use `docker` commands.** If
> you close it or restart the PC, open it again from the Start menu first.

---

## 1. What's in this folder

```
Timefold/
├── Dockerfile                  ← the recipe for the container image
├── docker-compose.yml          ← one-command run with the right mounts
├── .dockerignore               ← files Docker should skip while building
├── docker/
│   ├── pom-standalone.xml      ← Maven POM used inside the Docker build (see note below)
│   └── entrypoint.sh           ← runs inside the container: status.json + solver + cancel handling
├── pom.xml                     ← the real Maven POM for the project
└── src/
    ├── main/java/com/yourorg/scheduler/
    │   ├── EmployeeSchedule.java
    │   └── ExportSchedule.java
    └── main/resource/
        ├── EnvConfig.yaml      ← sample input
        └── Schedule.yaml       ← sample input
```

### A note on `docker/pom-standalone.xml`
The real `pom.xml` has a `<parent>` element pointing at `../../pom.xml` — a
parent project that isn't present inside the Docker build context. The
standalone pom is the same project with the parent removed and `groupId` +
`version` added explicitly, so Maven can build the project on its own inside
the container. The Dockerfile uses this one instead of the real pom.

If you ever change the real `pom.xml` (new dependency, Timefold version bump),
update `docker/pom-standalone.xml` to match.

---

## 2. Set up the working folders

The container expects three folders to be mounted at `/work/{input,output,status}`.
Create them on your machine first:

```bash
cd /c/Users/YourName/path/to/Timefold     # adjust to your actual path
mkdir -p work/input work/output work/status
```

You'll end up with:
```
Timefold/
└── work/
    ├── input/      ← you put your YAML files here
    ├── output/     ← solver writes result_Schedule.yaml here
    └── status/     ← <RUN_ID>.json appears here showing progress
```

> The `work/` folder is git-ignored — it stays local to your machine.

### Drop your input YAMLs in
For a first run, just copy the samples that come with the project:
```bash
cp src/main/resource/EnvConfig.yaml work/input/
cp src/main/resource/Schedule.yaml work/input/
```
Check:
```bash
ls work/input/
# EnvConfig.yaml  Schedule.yaml
```

For real runs, just replace those two files with your own (keep the names
`EnvConfig.yaml` and `Schedule.yaml`).

---

## 3. Build the image (first time — takes ~5 minutes)

```bash
docker compose build solver
```

What happens:
- Docker downloads a Maven + JDK 17 image (~500 MB the first time).
- Maven downloads Timefold and its dependencies (~150 MB).
- It compiles the Java source and copies the jar + dependency jars into a
  smaller JRE-only runtime image (~250 MB).

**First build is slow because of the downloads. Future builds are seconds**
because Docker caches every step.

When it finishes, verify the image exists:
```bash
docker images timefold-scheduler
# REPOSITORY            TAG     IMAGE ID       CREATED         SIZE
# timefold-scheduler    local   <hex>          1 minute ago    ~280MB
```

---

## 4. Run a solve

```bash
docker compose up solver
```

You'll see live logs from the solver:
- `[entrypoint] ...` lines from the wrapper script
- Maven / startup output
- Timefold solver progress (best score updates)
- Finally `[entrypoint] done. result at /work/output/result_Schedule.yaml`

While it runs you can open a **second** Git Bash window and peek at the
status file:
```bash
cat work/status/local-run-001.json
```
You'll see `"status": "Running"` → eventually `"status": "Completed"`.

When the solve finishes:
```bash
ls work/output/
# result_Schedule.yaml
```
That file is the solver's output — the schedule with all assignments filled in.

### Background mode (run + close terminal)
```bash
docker compose up -d solver        # -d = detached
docker compose logs -f solver      # follow the logs anytime
```

To stop following logs press **Ctrl+C** — that only stops the log tail, not
the container. The solver keeps running.

---

## 5. Cancel a running solve

The same operation that "Cancel button" will do in the future Azure deployment.

### If it's running in the foreground
Press **Ctrl+C** in the terminal that's running it. Docker sends `SIGTERM`,
the entrypoint catches it and:
- Writes `status.json` with `"status": "Cancelled"`
- Exits cleanly

### If it's running in the background
```bash
docker stop tf-solver
# wait ~10 seconds; if still running:
docker kill tf-solver
```

Verify:
```bash
cat work/status/local-run-001.json
# "status": "Cancelled"
```

A cancelled run **has no output yaml** — partial work is discarded.

---

## 6. Run multiple solves in parallel

ACA Jobs (in Azure) run each execution in its own container, isolated. You
can do the same locally. Each one needs its own `RUN_ID` and its own work
folders (otherwise they overwrite each other):

```bash
# Run #1
docker run --rm \
  -e RUN_ID=run-001 \
  -v "$(pwd)/work/input:/work/input:ro" \
  -v "$(pwd)/work/output:/work/output" \
  -v "$(pwd)/work/status:/work/status" \
  --cpus 4 --memory 8g \
  --name tf-run-001 \
  timefold-scheduler:local

# Run #2 in another terminal — separate work-002/ folder with different YAMLs
mkdir -p work-002/input work-002/output work-002/status
cp other_envconfig.yaml work-002/input/EnvConfig.yaml
cp other_schedule.yaml  work-002/input/Schedule.yaml
docker run --rm \
  -e RUN_ID=run-002 \
  -v "$(pwd)/work-002/input:/work/input:ro" \
  -v "$(pwd)/work-002/output:/work/output" \
  -v "$(pwd)/work-002/status:/work/status" \
  --cpus 4 --memory 8g \
  --name tf-run-002 \
  timefold-scheduler:local
```

Each container has its own RAM, its own CPU quota, its own files. This is
what a real batch node looks like.

---

## 7. Useful commands cheat-sheet

```bash
# Build the image
docker compose build solver

# Build from scratch (ignore cache)
docker compose build --no-cache solver

# Run (foreground, Ctrl+C to cancel)
docker compose up solver

# Run (background)
docker compose up -d solver

# What containers are running right now?
docker ps

# What containers ran recently (including exited)?
docker ps -a

# Tail logs of a running container
docker logs -f tf-solver

# Open a shell INSIDE a running container (for debugging)
docker exec -it tf-solver /bin/bash

# Stop a container gracefully
docker stop tf-solver

# Kill it instantly
docker kill tf-solver

# Remove stopped containers + dangling images (free disk)
docker container prune
docker image prune
docker system prune -a              # heavier cleanup
```

---

## 8. Troubleshooting

| You see this                                                         | What's wrong                                          | Fix                                                                  |
| -------------------------------------------------------------------- | ----------------------------------------------------- | -------------------------------------------------------------------- |
| `docker: command not found`                                          | Docker not installed                                  | Install Docker Desktop (section 0)                                   |
| `Cannot connect to the Docker daemon`                                | Docker Desktop not running                            | Open it from Start menu, wait for whale icon to be steady            |
| Build hangs at `[INFO] Downloading from central`                     | First build, just downloading deps                    | Wait — 5–10 minutes is normal                                        |
| `Non-resolvable parent POM`                                          | Docker is using the real pom instead of standalone    | The Dockerfile should fix this; rebuild with `--no-cache`            |
| `EnvConfig.yaml not found at /work/input/EnvConfig.yaml`             | Input file isn't where the container expects          | Check `ls work/input/` shows both YAML files                         |
| `permission denied` on `entrypoint.sh`                               | Line endings (CRLF) from Windows broke the script     | Open `docker/entrypoint.sh` in VS Code, bottom-right change to `LF`, save, rebuild |
| Volume mount can't see files (`Cannot read /work/input/...`)         | Path issue on Windows                                 | Confirm the path you ran `docker compose` from — `pwd` should show this `Timefold/` folder |
| `no space left on device`                                            | Old Docker images / containers piling up              | `docker system prune -a --volumes` to clean up                       |

---

## 9. How this works under the hood (optional reading)

When you run `docker compose up solver`:

1. Docker reads `docker-compose.yml`, sees the `solver` service.
2. It mounts your local `work/input` → container's `/work/input`, same for
   output and status. These are real folders on your disk that the container
   can see.
3. It starts the container, which runs `/app/entrypoint.sh` (set in `Dockerfile`).
4. The entrypoint:
   - Writes `status.json` = `Submitted`
   - Copies `Schedule.yaml` to `result_Schedule.yaml` (solver mutates its
     argument; we work on the copy so the input stays pristine)
   - Sets a SIGTERM trap (so Cancel works)
   - Updates `status.json` = `Running`
   - Runs `java -cp /app/lib/* com.yourorg.scheduler.EmployeeSchedule <env> <result>`
   - When solver exits successfully: `status.json` = `Completed`
   - If solver exits with error: `status.json` = `Failed`
   - If interrupted by SIGTERM: `status.json` = `Cancelled`

The whole thing runs inside the container's isolated Linux environment but
reads/writes through the volume mounts, so the result file appears on your
real disk in `work/output/`.

---

That's everything. After the first successful run, day-to-day usage is just:

```bash
docker compose up solver
```

If you hit something not covered here, copy the error and ask.
