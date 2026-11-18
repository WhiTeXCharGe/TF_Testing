# Weekly Planning — Testing the Flow Without (or With) Timefold

How to test `01_StartSchedule` and `02_UpdateSchedule` (tools 2-6 and 3-1) when the real solver is not there, how to prepare `run_info.json` by hand, and what to write in the `config.txt` of 2-6 and 3-1.

---

## 1. Short answer

**2-6 and 3-1 do not need Timefold itself. They need a server that speaks the solver API.** They only send files to it and ask it for the state. Three servers can answer:

| Server | Solves? | Needs | Use it for |
|---|---|---|---|
| **Stub solver** `common/testing/stub_solver.py` | No. A run "finishes" after N seconds and hands back a file | Python only | Testing the flow: sending, waiting, receiving, file names, errors |
| **MockServer** (`web/MockServer`, `node server.js`) | Yes, with the Docker image `timefold-scheduler:local` | Node.js, `npm install`, Docker running | A real solve on the PC |
| **Azure API controller** (`mode = online`) | Yes | Network, `azure_url` | The real thing |

All three have the same four calls (this is the contract in `common/timefold_api.py`):

```
POST /runSolver        multipart: env, sched, params (optional), runId (optional)   ->  202 {runId, status}
GET  /status/<runId>   ->  {status: Submitted | Running | Completed | Failed | Cancelled, stage, progress, error}
GET  /download/<runId> ->  the result Schedule (only when Completed)
GET  /health           ->  {ok: true}      (online mode asks it first)
```

So the same tools and the same programs run against any of them; only `mode` and the address in the settings change.

**What counts as success for `01_StartSchedule`:** the run is accepted by the server (answer 202), `steps/run_info.json` exists, `week_status.json` says `stage: submitted` with a run id, and the server's `/status/<runId>` knows that run. With the stub that means the whole program is proven. "The solver really finds a plan" is a separate test (MockServer + Docker, or Azure).

**What counts as success for `02_UpdateSchedule` (3-1 part):** with a finished run, `04_Schedule.yaml` and `04_EnvConfig.yaml` appear next to the run record.

---

## 2. The stub solver

```
python common/testing/stub_solver.py [--port 3001] [--seconds 20] [--result <file>] [--fail] [--data-dir <folder>]
python common/testing/stub_solver.py --make-run <runId> --result <file> [--data-dir <folder>]
```

| Option | Meaning |
|---|---|
| `--port` | Default 3001, the port of the local MockServer, so `local_url = http://localhost:3001` works for both |
| `--seconds` | How long a run takes until `Completed`. It shows `Submitted`, then `Running` (stage 1, then stage 2 for the last quarter), then `Completed` |
| `--result` | The file handed out by `/download`. Default: the `Schedule.yaml` that was sent (so the "result" is the input, **not a solved plan**) |
| `--fail` | Every run ends as `Failed` (to see how the tools report it) |
| `--data-dir` | Where the runs are kept (default `stub_solver_data` in the current folder). A restart keeps old runs |
| `--make-run <runId>` | Does not start the server. Puts a run that is already `Completed` into the data folder (needs `--result`). This is how a hand-written `run_info.json` gets a run to ask for (section 5) |

Stop it with Ctrl+C. It listens on `127.0.0.1` only. It checks nothing about the content of the YAML files, so a wrong plan is also "accepted".

**Limit to remember:** the file you get from the stub is not a solver result. `3-2` and `3-3` can run on it, but the output means nothing. For a meaningful part-2 test give `--result` a real solved schedule (for example the `04_Schedule.yaml` of an earlier real week).

---

## 3. `config.txt` of tool 2-6 (`2-6submit_to_timefold`)

Plain `key = value` lines, `#` starts a comment, an empty value means "default", a relative path is relative to the folder of the file. A command line option replaces the same key for one run.

| Key | Test with the stub / MockServer | Real (Azure) |
|---|---|---|
| `mode` | `local` | `online` |
| `local_url` | `http://localhost:3001` | (not used) |
| `azure_url` | (not used) | the address of the API controller |
| `schedule` | `03_Schedule.yaml` of the week | same |
| `env` | `03_EnvConfig.yaml` of the week | same |
| `out_dir` | folder for `run_info.json` and `SolverParams.yaml` (empty = folder of the schedule file) | same |
| `run_id` | empty (date and time), or any name of letters, digits, `_`, `-`, up to 64 characters | empty, or a name that was never used (**Azure accepts a name only once**) |
| `params_file` | empty | empty, or your own SolverParams.yaml (the keys below still win) |
| `stage1_minutes` / `stage2_minutes` | with MockServer + Docker a short test: `2` / `1`. With the stub: any, it is not used | `45` / `15` is the default of the tool |
| `stage2_enabled` | `true` | `true` |
| `stage1_unimproved_seconds`, `stage1_stop_when_feasible`, `stage2_unimproved_seconds`, `random_seed` | empty | empty, or set to taste |
| `allow_cut_fixed_trip` | `false` | `false` |

Example for a test against the stub or MockServer:

```
mode = local
local_url = http://localhost:3001
schedule = <week folder>/steps/03_Schedule.yaml
env = <week folder>/steps/03_EnvConfig.yaml
out_dir = <week folder>/steps
run_id =
stage1_minutes = 2
stage2_enabled = true
stage2_minutes = 1
allow_cut_fixed_trip = false
```

Check without sending: `python submit_to_timefold.py --dry-run`.

**Result:** `run_info.json` and `SolverParams.yaml` in `out_dir`. Exit code 0 = sent, 1 = error (message on screen).

---

## 4. `config.txt` of tool 3-1 (`3-1receive_from_timefold`)

| Key | Meaning |
|---|---|
| `mode`, `local_url`, `azure_url` | **Must be the same server as 2-6 used.** If they differ the tool prints a note and the run is "not found" |
| `run_info` | The path of the `run_info.json` that 2-6 wrote. Empty: `run_info.json` in the folder the tool is started from |
| `run_id` | Used only when there is no run record (then `04_EnvConfig.yaml` cannot be made, see below) |
| `result_dir` | Folder for `04_Schedule.yaml` and `04_EnvConfig.yaml`. Empty = the folder of the run record |
| `watch` | Seconds between two checks until the run ends. Empty or `0` = check once and stop |
| `download` | `ask` (asks Y/N), `yes` (no question), `no` (only show the state) |

Example:

```
mode = local
local_url = http://localhost:3001
run_info = <week folder>/steps/run_info.json
result_dir = <week folder>/result
watch = 30
download = yes
```

Command line: `python receive_from_timefold.py --run-info <file> --yes` (or `--no`, `--watch 30`, `--mode`, `--api-url`).

**Exit codes:** `0` finished (downloaded or not) · `1` not finished yet · `2` failed · `3` cancelled · `4` cannot reach the server (or download failed) · `5` bad settings or no run given.

**What it writes:** `04_Schedule.yaml` = the result from the server. `04_EnvConfig.yaml` = a copy of the EnvConfig that was sent (the solver does not change it); the copy is taken from the file named by `envFile` in the run record.

---

## 5. `run_info.json`: what it is and how to prepare it by hand

2-6 writes it after a successful send. 3-1 reads it. Normally you never touch it. By hand you need it when part 2 must be tested although part 1 did not run on this PC.

```json
{
  "runId": "20261201_090000",
  "mode": "local",
  "apiUrl": "http://localhost:3001",
  "submittedAt": "2026-12-01T09:00:00",
  "scheduleFile": "<path>/03_Schedule.yaml",
  "envFile": "<path>/03_EnvConfig.yaml",
  "params": {"stage1_minutes": 45, "stage2_enabled": true, "stage2_minutes": 15, "allow_cut_fixed_trip": false}
}
```

| Field | Needed by 3-1? | Notes |
|---|---|---|
| `runId` | **yes** | The run to ask the server about. Must exist on that server |
| `mode`, `apiUrl` | no (config wins), but 3-1 compares them with the settings and prints a note when they differ; with no address in the settings it is the last resort | |
| `envFile` | **yes for `04_EnvConfig.yaml`** | Full path of the EnvConfig that was sent. If the file is missing 3-1 still saves `04_Schedule.yaml` and tells you to copy the EnvConfig yourself |
| `submittedAt`, `scheduleFile`, `params` | no | Information only |

**Recipe: a run record for part 2 without part 1**

1. Take any `03_EnvConfig.yaml` (the EnvConfig the solved schedule belongs to) and a Schedule file to serve as the result.
2. Start the stub (`python stub_solver.py`) in the folder where it should keep its runs.
3. Make a finished run: `python stub_solver.py --make-run hand_test_1 --result <the result Schedule.yaml>` (with the same `--data-dir` as the running stub).
4. Write `run_info.json` as above with `"runId": "hand_test_1"`, `"mode": "local"`, `"apiUrl": "http://localhost:3001"` and the full path of the EnvConfig in `"envFile"`. Only `runId` and `envFile` are really needed.
5. Run 3-1: `python receive_from_timefold.py --mode local --run-info <that file> --yes`. Expect `04_Schedule.yaml` and `04_EnvConfig.yaml` next to the record.

**For `02_UpdateSchedule` inside `plan_assist`:** the run record of a week is `Work/<week>/steps/run_info.json` and the week status has the run id. For a hand-made test week: create `Work/<week>/steps/` with `run_info.json` and the EnvConfig, plus `Work/<week>/week_status.json` as the program writes it (stage `submitted`, `run_id` the same as in the record), and make the run on the stub as in step 3. The part-2 program then finds everything the same way as after a real part 1. After a successful part 2 the week folder must hold `FlexibleSchedule.yaml`, `EnvConfig.yaml` and `SU_Others_rewritten.xlsm`, because that is how next week's part 1 finds its previous week.

---

## 6. Test recipes

### A. `01_StartSchedule` against the stub (no Timefold, nothing real is sent)

1. Start the stub: `python common/testing/stub_solver.py --seconds 30`.
2. In `plan_assist.config.txt` set `mode = local`, `local_url = http://localhost:3001`, `submit = true`. Keep `stage1_minutes` / `stage2_minutes` small. For a first week also have the skill workbook and `EnvConfig_start.yaml` in `Master/`.
3. Run the program (window, or `python start_schedule.py --su-others <file> --seiban-list <file> --no-ui`).
4. It is a success when the last lines say `Sent to the solver as run <id>`, `Work/<week>/steps/run_info.json` exists and `week_status.json` has `"stage": "submitted"`.
5. Ask the stub: `GET http://localhost:3001/status/<id>` (a browser is enough). It shows `Running`, then `Completed`.

Without a server at all: `submit = false` (or `--no-submit`) stops after the 03 files (`stage: ready`). That proves 2-2-3 to 2-5 only.

### B. 3-1 on the run of A

1. Write the 3-1 settings as in section 4 with `run_info = Work/<week>/steps/run_info.json`.
2. Start it early: it prints `Still running: Running (stage 1, ..%)`, exit code 1. Start it again after the time of `--seconds`: `Done`, and it saves the two 04 files.
3. `04_EnvConfig.yaml` must be identical to `steps/03_EnvConfig.yaml`.

### C. Failure paths

- `--fail` on the stub: 3-1 prints `Failed: StubError: ...`, exit code 2.
- Stop the stub: 2-6 and 3-1 print `cannot reach http://localhost:3001/... Is the local server running?`, exit 1 / 4.
- Start a run again with the same `run_id` on Azure: refused (name used). Leave `run_id` empty.

### D. A real solve on the PC (MockServer + Docker)

1. Docker Desktop must be **running** (the `docker` command alone is not enough; if the Docker engine is not up, MockServer answers `Cannot start Docker` and the run is `Failed` at once).
2. The image `timefold-scheduler:local` must exist (`docker images`; built from `web/TFDocker`, see `documents/SchedulerWeb/Docker.md`).
3. `cd web/MockServer`, `npm install` (first time), `node server.js`. It uses port 3001 and keeps its runs in `MockServer/data/`.
4. Use the same settings as for the stub (`mode = local`, `local_url = http://localhost:3001`) and a short time (`stage1_minutes = 2`, `stage2_minutes = 1`) for a quick check; the real default is 45 + 15 minutes.
5. The result of 3-1 is then a real solved schedule.

The MockServer also has test helpers that need no Docker: `PUT /status/<runId>` (body `{"status":"Completed","progress":1}`) and `POST /output/<runId>` (multipart, field `result`, a `.yaml` file). With them a run can be faked on the real MockServer the same way `--make-run` does on the stub.

---

## 7. What was tested (2026-11-19)

Done with the stub, on real input files, with the real tools (not fakes):

- `01_StartSchedule` from files to the send: 2-2-3, 2-4, 2-5 and 2-6 ran, the run was accepted, `run_info.json` was written with the run id and the paths.
- 3-1 on that run: while it ran `Still running: Running (stage 1, 42%)` (exit 1); when finished it saved `04_Schedule.yaml` and `04_EnvConfig.yaml`, and the EnvConfig was identical to the one sent.
- 3-1 on a hand-written `run_info.json` (only `runId`, `mode`, `apiUrl`, `envFile`) and a run made with `--make-run`: worked.
- The stub itself has unit tests (`common/tests/test_stub_solver.py`). One bug was found and fixed during this test: a result of several MB was cut short on Windows when the server closed the connection straight after writing; the stub now closes the sending side first.

Not tested: a real solve (Docker was not running on the test PC, MockServer was not started), Azure, and `02_UpdateSchedule` as a program (it does not exist yet; 3-1 alone was tested).

---

## 8. Quick problem table

| You see | Reason | Do |
|---|---|---|
| `cannot reach http://localhost:3001/...` | No server running | Start the stub or MockServer; check `local_url` |
| `run "<id>" was not found on <url>` | 3-1 asks another server than 2-6 used, or the server was restarted without its data | Same `mode` and address in both configs; with the stub use the same `--data-dir` |
| `no run to look at` | No `run_info.json` found | Set `run_info` in the 3-1 config, or `--run-info`, or `--run-id` |
| `04_EnvConfig.yaml not written` | `envFile` missing or wrong in the record | Fix the path in `run_info.json`, or copy `03_EnvConfig.yaml` yourself |
| `run_id ... cannot be used` | Characters other than letters, digits, `_`, `-`, or longer than 64 | Change the name or leave it empty |
| `Failed: UnknownError: Cannot start Docker` | MockServer cannot start Docker | Start Docker Desktop, check the image |
| A relative path in the config is not found | Relative to the config file's folder, not to where you started the tool | Use full paths |
