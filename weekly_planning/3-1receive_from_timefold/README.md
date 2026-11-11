# 3-1 receive_from_timefold

Asks Timefold how the run started by tool 2-6 is going, and downloads the result.

**In** `run_info.json` (from 2-6) or `--run-id`. **Out** `04_Schedule.yaml` and `04_EnvConfig.yaml`, only if you answer Y.

## What it does
- Prints the state in the terminal.
  - Not finished: prints that it is waiting or running (stage and progress) and ends. Run it again later, or use `--watch <seconds>` to keep checking.
  - Finished: prints that it is done and asks `Download the result? [Y/N]`. Y downloads, N ends. `--yes` / `--no` answer for you.
  - Failed or cancelled: prints the reason; nothing to download.
- `04_Schedule.yaml` is the solver result. The solver does not change the EnvConfig, so `04_EnvConfig.yaml` is a copy of the `03_EnvConfig.yaml` that was sent.
- Exit codes: 0 done, 1 not finished, 2 failed, 3 cancelled, 4 cannot reach the server or mode not available, 5 bad arguments.

## Modes
Same as tool 2-6: `--mode local` (MockServer on `http://localhost:3001`) works; `--mode real` (default) is not available yet. The mode and address are taken from the run record unless you give them.

## Run
```
python receive_from_timefold.py --run-info <run_info.json> [--out-dir <folder>] [--watch 30] [--yes | --no]
```
