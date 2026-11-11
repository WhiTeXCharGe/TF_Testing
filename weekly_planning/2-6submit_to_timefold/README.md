# 2-6 submit_to_timefold

Sends the solver input to Timefold and starts the run. Progress and the result are handled by tool 3-1.

**In** `03_Schedule.yaml`, `03_EnvConfig.yaml`. **Out** `run_info.json` (run id, where the input came from) and `SolverParams.yaml` (a copy of what was sent), next to the schedule or in `--out-dir`.

## What it does
- Sends three solver inputs: `EnvConfig.yaml`, `Schedule.yaml` and `SolverParams.yaml` (always under these names).
- `SolverParams.yaml` is made from defaults: **stage 2 on, stage 1 at most 45 min + stage 2 at most 15 min = 1 hour**, and `allow_cut_fixed_trip: false`. Everything else keeps the solver default (the solver alone would allow 300 + 60 minutes).
- Change them with options, or give your own file with `--params-file` (the options still win).
- Checks that the two files are the right kind of file before sending anything. `--dry-run` shows what would be sent.
- Prints the run id; the next step is tool 3-1.

## Modes
- `--mode local`: sends to the local server (MockServer, which runs the solver in Docker), default address `http://localhost:3001` (`--api-url` or `TF_API_URL_LOCAL`). Needs Docker running and the image `timefold-scheduler:local`; start the server with `node server.js` in the MockServer folder.
- `--mode real` (the default): the Azure API controller. **Not available yet**; the tool says so and stops.

## Run
```
python submit_to_timefold.py --schedule <03_Schedule.yaml> --env <03_EnvConfig.yaml> --mode local [--out-dir <folder>]
```
Options: `--stage1-minutes`, `--stage2-minutes`, `--no-stage2`, `--stage1-unimproved-seconds`, `--stage2-unimproved-seconds`, `--seed`, `--allow-cut-fixed-trip`, `--params-file`, `--run-id`, `--api-url`, `--dry-run`.

## API
The API controller and MockServer must accept the optional file field `params` (`SolverParams.yaml`) next to `env` and `sched`, and store the files under the fixed names above. This is done in the repository versions; the deployed controller has to be updated to match.
