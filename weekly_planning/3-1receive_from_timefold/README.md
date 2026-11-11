# 3-1 receive_from_timefold

Asks Timefold how the run started by tool 2-6 is going, and downloads the result.

**In** `run_info.json` (from 2-6) or a run id. **Out** `04_Schedule.yaml` and `04_EnvConfig.yaml`, only if you answer Y (or `download = yes`).

## Run
Fill in the `config.txt` in this folder (see below). Its `mode`, `azure_url` and `local_url` must match the `config.txt` of tool 2-6. Then:
```
python receive_from_timefold.py
```
`--config <file>` uses another settings file. Any option replaces the same setting of `config.txt` for that run; `--help` lists them.

## config.txt
`key = value` lines, `#` starts a comment, an empty value means "not set", relative paths are relative to the folder of the file. Priority: command line, then `config.txt`, then the built-in default.

| Setting | Meaning |
|---|---|
| `mode`, `azure_url`, `local_url` | which server to ask (as in 2-6) |
| `run_info` | the run record (`run_info.json`) that 2-6 wrote, in its `out_dir` or next to its schedule. Empty: `run_info.json` in the current folder |
| `run_id` | used when there is no run record |
| `result_dir` | folder for the downloaded files (empty: the folder of the run record) |
| `watch` | seconds between two checks until the run has ended (empty or 0: check once) |
| `download` | `ask` (default), `yes` or `no` |

The server comes from `config.txt` (or `--mode` / `--api-url`); only when neither says, the one in the run record is used. If the record says the run was sent somewhere else than the settings point to, the tool says so.

## What it does
- Prints the state in the terminal.
  - Not finished: prints that it is waiting or running (stage and progress) and ends. Run it again later, or set `watch` / use `--watch <seconds>` to keep checking.
  - Finished: prints that it is done and asks `Download the result? [Y/N]` (unless `download` says otherwise). Y downloads, N ends.
  - Failed or cancelled: prints the reason; nothing to download.
- In online mode, short server errors while the app wakes up are retried.
- `04_Schedule.yaml` is the solver result. The solver does not change the EnvConfig, so `04_EnvConfig.yaml` is a copy of the `03_EnvConfig.yaml` that was sent.
- Exit codes: 0 done, 1 not finished, 2 failed, 3 cancelled, 4 cannot reach the server, 5 bad arguments or settings.
