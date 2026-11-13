# 2-6 submit_to_timefold

Sends the solver input to Timefold and starts the run. Progress and the result are handled by tool 3-1.

**In** `03_Schedule.yaml`, `03_EnvConfig.yaml`. **Out** `run_info.json` (run id, mode, address, where the input came from) and `SolverParams.yaml` (a copy of what was sent), in `out_dir` or next to the schedule.

## Run
Fill in the `config.txt` in this folder (see below), then:
```
python submit_to_timefold.py
```
`--config <file>` uses another settings file. Any option replaces the same setting of `config.txt` for that run; `--help` lists them, `--dry-run` shows what would be sent and sends nothing.

## config.txt
The settings file of this tool (tool 3-1 has its own `config.txt` in its folder), `key = value` lines, `#` starts a comment, an empty value means "not set", relative paths are relative to the folder of the file. Priority: command line, then `config.txt`, then the built-in default.

| Setting | Meaning |
|---|---|
| `mode` | `local` or `online` |
| `azure_url` | address of the API controller on Azure (used when `mode = online`) |
| `local_url` | address of the local server (used when `mode = local`, default `http://localhost:3001`) |
| `schedule`, `env` | the two input files |
| `out_dir` | folder for `run_info.json` and `SolverParams.yaml` |
| `run_id` | name of the run (the job id): letters, digits, `_`, `-`, up to 64. Empty: date and time |
| `params_file`, `stage1_minutes`, `stage2_enabled`, `stage2_minutes`, `stage1_unimproved_seconds`, `stage1_stop_when_feasible`, `stage2_unimproved_seconds`, `random_seed`, `allow_cut_fixed_trip` | solver parameters |

Tool 3-1 reads `run_info.json` from `out_dir` (or next to the schedule); its own `config.txt` must use the same `mode` and address.

## Modes
- `local`: the local server (MockServer, which runs the solver in Docker). Needs Docker running and the image `timefold-scheduler:local`; start the server with `node server.js` in the MockServer folder.
- `online`: the API controller on Azure, the same server and the same requests as SchedulerWeb. The tool first asks `/health` (this also wakes the app up when it was idle), then sends the run. Short server errors while it wakes up are retried. You must be on a network that can reach `azure_url`.

## What it does
- Sends three solver inputs: `EnvConfig.yaml`, `Schedule.yaml` and `SolverParams.yaml` (always under these names).
- `SolverParams.yaml` is made from defaults: **stage 2 on, stage 1 at most 45 min + stage 2 at most 15 min = 1 hour**, and `allow_cut_fixed_trip: false`. Everything else keeps the solver default (the solver alone would allow 300 + 60 minutes). Settings from `config.txt` and options change them.
- Checks that the two files are the right kind of file and that the run id is usable before sending anything.
- On Azure a run id can be used only once; leave `run_id` empty to get a new one every time.

## API
The API controller and MockServer must accept the optional file field `params` (`SolverParams.yaml`) next to `env` and `sched`, and store the files under the fixed names above. This is done in the repository versions; the deployed controller has to be updated to match.