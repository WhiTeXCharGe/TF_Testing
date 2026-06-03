# Dockerising Timefold — local "batch node" guide

Goal: package the Timefold Java solver as a Docker container so you can run it
the same way you'll eventually run it on Azure Container Apps Jobs. No Azure
account needed yet — everything works on your laptop.

> **Why this matters now:** the container you build here is the **exact same
> artefact** that will run as the compute layer in [Azure.md](./Azure.md). When
> Azure is ready, you push this image to a registry and ACA pulls it. The web
> app doesn't need to change at all — the API Controller just talks to a
> container instead of a `java -jar` process.

---

## 0. What Docker is, in three lines

- **Image** — a frozen template: OS + JDK + your jar + a startup script.
- **Container** — a running copy of an image (a process, basically).
- **Volume mount** — a folder on your laptop that the container can read/write,
  so your input YAMLs go in and the result YAML comes out without "uploading"
  anything.

That's it. You don't need to learn Kubernetes, Compose internals, or anything
else to get value here.

---

## 1. Install Docker Desktop (one-time)

1. Download **Docker Desktop for Windows**: <https://docs.docker.com/desktop/install/windows-install/>
2. Run the installer. When asked, **enable WSL2 backend** (the default — it's
   faster and uses less RAM).
3. After install, open **Docker Desktop** once and let it finish setting up.
   The whale icon in the system tray should be steady (not animating) when
   it's ready.
4. Check it works — open Git Bash or PowerShell and run:
   ```bash
   docker --version
   docker compose version
   ```
   Both should print version numbers.

That's the entire setup. Docker Desktop runs a tiny Linux VM in the background;
you don't have to manage it.

---

## 2. Files we just created (already in the repo)

In `web/Timefold/`:

| File                          | What it does                                                                                   |
| ----------------------------- | ---------------------------------------------------------------------------------------------- |
| `Dockerfile`                  | The recipe for the image. Two stages: build the jar with Maven, then copy into a tiny JRE image. |
| `.dockerignore`               | Tells Docker which files to skip when packaging the build context (faster, smaller image).     |
| `docker/entrypoint.sh`        | Shell script that runs inside the container: writes status.json, runs the solver, handles cancel (SIGTERM). |
| `docker/pom-standalone.xml`   | A self-contained Maven POM used **only inside the Docker build** — see note below.            |
| `docker-compose.yml`          | Convenience config so `docker compose up` Just Works with the right mounts.                    |

### About `docker/pom-standalone.xml`

The real `web/Timefold/pom.xml` has a `<parent>` element pointing at
`../../pom.xml`. When Maven runs **inside the container**, that parent file is
not in the build context, so the build would fail with *"Non-resolvable parent
POM"*.

The standalone pom is the same project with the `<parent>` block removed and
`<groupId>` + `<version>` added explicitly so it can stand on its own. It's a
~10-line edit and the Dockerfile uses it instead of the real pom.

**If `web/Timefold/pom.xml` ever changes** (e.g. new dependency, Timefold
version bump), update `docker/pom-standalone.xml` to match.

---

## 3. The folder layout you'll use locally

Create this on your laptop (anywhere; we'll use the v924 folder):

```
web/Timefold/
├── Dockerfile              (already created)
├── docker-compose.yml      (already created)
├── docker/entrypoint.sh    (already created)
├── pom.xml
├── src/...                 (the Java source)
└── work/                   ← YOU create this folder
    ├── input/              ← drop your EnvConfig.yaml + Schedule.yaml here
    ├── output/             ← solver writes result_Schedule.yaml here
    └── status/             ← status.json appears here while/after solving
```

Create the `work/` folders:
```bash
cd /c/Users/Seiya/Desktop/work/Timefold/web/Timefold
mkdir -p work/input work/output work/status
```

Then **copy your two YAML files** into `work/input/`:
- `work/input/EnvConfig.yaml`
- `work/input/Schedule.yaml`

The container will mount `work/` as `/work/` inside, so it can see your files.

---

## 4. Build the image (first time, ~5 min — pulls Maven + JDK)

From the v924 folder:
```bash
docker compose build solver
```

What this does:
1. Reads `Dockerfile`.
2. Stage 1 — spins up a Maven image, copies your `pom.xml` + `src/`, runs
   `mvn package`. The jar + all dependency jars end up in `target/`.
3. Stage 2 — copies those jars into a small JRE-only image, copies
   `entrypoint.sh`, sets the user to non-root.

First build downloads ~500 MB of Maven dependencies. **Subsequent builds reuse
the cache and finish in seconds** (unless you change `pom.xml`).

When it's done:
```bash
docker images timefold-scheduler
```
Should print a `timefold-scheduler:local` image, around 250–300 MB.

---

## 5. Run a solve

```bash
docker compose up solver
```

You'll see:
- Maven logs from the build (if anything changed)
- The entrypoint banner
- Timefold's solver output streaming live

While it runs:
- `work/status/local-run-001.json` shows `{ "status": "Running", ... }`
- After it finishes, it shows `{ "status": "Completed", "output": "/work/output/result_Schedule.yaml" }`
- The result yaml appears in `work/output/result_Schedule.yaml`

To run it in the background instead:
```bash
docker compose up -d solver         # detached
docker compose logs -f solver       # follow the logs
```

---

## 6. Cancel a running solve

This is the "Cancel button" from the web app, locally.

### From `docker compose`
Press **Ctrl+C** in the terminal that's running it. Docker sends `SIGTERM` to
the container; our entrypoint catches it, writes `status = Cancelled` to the
status file, and exits.

### From a separate terminal
```bash
docker stop tf-solver               # graceful stop (sends SIGTERM)
# wait 10s; if still running:
docker kill tf-solver               # forceful (SIGKILL)
```

Check the status file afterwards — it should say `"status": "Cancelled"`.

> **Note:** the current Java code doesn't yet call `solver.terminateEarly()` on
> SIGTERM, so cancel discards the partial result. We can add a small shutdown
> hook later so cancel returns "best partial solution found so far" if you want
> that.

---

## 7. Run multiple solves in parallel (simulating a batch node)

ACA Jobs run each execution in its own container, isolated from the others.
You can do exactly the same thing locally:

```bash
# Pretend run #1
docker run --rm \
  -e RUN_ID=run-001 \
  -v "$(pwd)/work/input:/work/input:ro" \
  -v "$(pwd)/work/output:/work/output" \
  -v "$(pwd)/work/status:/work/status" \
  --cpus 4 --memory 8g \
  --name tf-run-001 \
  timefold-scheduler:local

# In another terminal, pretend run #2 with different inputs
docker run --rm \
  -e RUN_ID=run-002 \
  -v "$(pwd)/work-002/input:/work/input:ro" \
  -v "$(pwd)/work-002/output:/work/output" \
  -v "$(pwd)/work-002/status:/work/status" \
  --cpus 4 --memory 8g \
  --name tf-run-002 \
  timefold-scheduler:local
```

Each container is independent — its own RAM, its own CPU quota, its own files.
This is exactly how ACA Job executions behave in Azure: spin one up per `runId`,
let it run, scale it down when done.

---

## 8. Useful commands cheat-sheet

```bash
# What's running right now?
docker ps

# What ran recently (including ones that exited)?
docker ps -a

# Tail the logs of a running container
docker logs -f tf-solver

# Look inside a running container (shell)
docker exec -it tf-solver /bin/bash

# Remove all stopped containers
docker container prune

# Remove all unused images (free disk)
docker image prune -a

# Rebuild from scratch (no cache)
docker compose build --no-cache solver
```

---

## 9. How this maps to Azure

| What you do locally                  | What Azure does                                                       |
| ------------------------------------ | --------------------------------------------------------------------- |
| `docker compose up solver`           | API Controller calls `az containerapp job start --image tf:tag`       |
| `work/input/` bind mount             | ACA Job reads `input/{runId}/` from Blob via Managed Identity         |
| `work/output/result_Schedule.yaml`   | ACA Job writes to `output/{runId}/result_Schedule.yaml` in Blob        |
| `work/status/<RUN_ID>.json`          | ACA Job writes to `status/{runId}.json` in Blob                       |
| `docker stop tf-solver`              | API Controller calls `az containerapp job execution stop`             |
| Image lives on your laptop           | Image lives in **Azure Container Registry**, pulled by ACA at run time |

When you're ready to ship, you'll:
1. Tag the image: `docker tag timefold-scheduler:local <acr>.azurecr.io/timefold:1.0`
2. Push it: `docker push <acr>.azurecr.io/timefold:1.0`
3. Point the ACA Job at it.

Everything else stays the same.

---

## 10. Troubleshooting

### "Cannot connect to the Docker daemon"
Docker Desktop isn't running. Open it from the Start menu; wait for the whale
icon to go steady; retry.

### Build hangs on `mvn package`
First build is slow (downloading deps). Give it 5 minutes. If still hung after
10, check Docker Desktop → Settings → Resources and make sure it has at least
4 CPU and 4 GB RAM allocated.

### Solver exits immediately with "EnvConfig.yaml not found"
The input mount isn't pointing at the right folder. Confirm:
```bash
ls work/input/                       # should show EnvConfig.yaml + Schedule.yaml
```

### `permission denied` on `entrypoint.sh`
On Windows + WSL2, line endings can get mangled. From Git Bash:
```bash
dos2unix docker/entrypoint.sh        # or open in VS Code and save with LF
```

### "no space left on device"
Old images/containers piling up. Free disk:
```bash
docker system prune -a --volumes
```

### Running container can't see my files
Volume mount paths must be **absolute** on Windows. `docker compose` handles
this for you; raw `docker run` needs full paths like
`-v "C:/Users/Seiya/.../work/input:/work/input:ro"`.

---

## What's next

Once this is working on your laptop:

1. **Wire the web app to call the container** — the Vite middleware can shell
   out to `docker compose up -d solver` when the user clicks New Run, and read
   `work/status/<RUN_ID>.json` for the Show Result poll. This makes the local
   dev experience end-to-end without Azure.
2. **Add `solver.terminateEarly()` on SIGTERM** in the Java code so Cancel
   returns a partial result instead of nothing.
3. **Push the image to Docker Hub or Azure Container Registry** when you're
   ready for the cloud step.

Tell me which one you want to tackle next and I'll guide you through it.
