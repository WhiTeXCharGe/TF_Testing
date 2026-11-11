#!/usr/bin/env python3
"""
receive_from_timefold.py  --  weekly planning tool "3-1": ask Timefold how the run is going and fetch the result.

Run it with no options: the server and the run are taken from the config.txt next to this script and from the run record that tool 2-6 wrote.
Any command line option replaces the same setting of config.txt for that run.

Input   run_info.json (written by tool 2-6) or a run id
Output  04_Schedule.yaml, 04_EnvConfig.yaml   (only if you answer Y, or download=yes in config.txt)

  * not finished  -> prints that it is still running (stage and progress) and ends; run it again later, or use --watch
  * finished      -> prints that it is done and asks whether to download:  Y downloads, N ends
  * failed / cancelled -> prints the reason, nothing to download
04_Schedule.yaml is the result of the solver. The solver does not change the EnvConfig, so 04_EnvConfig.yaml is the 03_EnvConfig.yaml
that was sent (taken from the run record).

Exit codes: 0 done (downloaded or not), 1 not finished yet, 2 failed, 3 cancelled, 4 cannot reach the server, 5 bad arguments or settings.
"""
from __future__ import annotations

import argparse
import shutil
import sys
import time
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent / "2-6submit_to_timefold"))
import timefold_api as api   # noqa: E402
import tf_config as tfc      # noqa: E402

CONFIG_FILE = HERE / tfc.CONFIG_NAME        # the settings of this tool

EXIT_DONE, EXIT_RUNNING, EXIT_FAILED, EXIT_CANCELLED, EXIT_UNREACHABLE, EXIT_ARGS = 0, 1, 2, 3, 4, 5
TERMINAL = ("Completed", "Failed", "Cancelled")


def describe(status):
    """One line for the terminal: 'Running (stage 1, 35%)'."""
    text = str(status.get("status"))
    extra = []
    if status.get("stage") is not None:
        extra.append(f"stage {status['stage']}")
    if isinstance(status.get("progress"), (int, float)) and text in ("Running", "Submitted"):
        extra.append(f"{round(status['progress'] * 100)}%")
    return text + (f" ({', '.join(extra)})" if extra else "")


def ask_download():
    """True for Y / yes, False for anything else (also when there is no terminal to answer)."""
    try:
        return input("Download the result? [Y/N] ").strip().lower() in ("y", "yes")
    except EOFError:
        return False


def find_run_record(arg, cfg):
    """The run record to use: --run-info, else run_info of config.txt, else run_info.json in the current folder.
    None when there is none."""
    if arg:
        return Path(arg)
    for c in (cfg.file("run_info"), api.RUN_INFO_NAME):
        if c and Path(c).is_file():
            return Path(c)
    return None


def main(argv=None):
    ap = argparse.ArgumentParser(description="Weekly planning 3-1: show the state of a Timefold run and download the result. "
                                             "Settings come from config.txt; the options below replace them for one run.")
    ap.add_argument("--config", default=None, help=f"settings file (default: {tfc.CONFIG_NAME} next to this script)")
    ap.add_argument("--run-info", default=None, help=f"the {api.RUN_INFO_NAME} written by tool 2-6 (config: run_info; default: ./run_info.json)")
    ap.add_argument("--run-id", default=None, help="the run id (instead of the run record; config: run_id)")
    ap.add_argument("--mode", choices=api.MODES, default=None, help="local or online (config: mode)")
    ap.add_argument("--api-url", default=None, help="address of the server, replaces local_url / azure_url")
    ap.add_argument("--out-dir", default=None, help="folder for 04_Schedule.yaml and 04_EnvConfig.yaml (config: result_dir; default: the folder of the run record)")
    ap.add_argument("--yes", action="store_true", help="download without asking (config: download=yes)")
    ap.add_argument("--no", action="store_true", help="never download, only show the state (config: download=no)")
    ap.add_argument("--watch", type=int, default=None, metavar="SECONDS", help="check again every SECONDS until the run has ended (config: watch)")
    args = ap.parse_args(argv)
    if args.yes and args.no:
        ap.error("--yes and --no cannot be used together")

    try:
        cfg = tfc.load(args.config, CONFIG_FILE, tfc.RECEIVE_KEYS)
        for key in cfg.unknown:
            print(f"Note: {cfg.path.name}: unknown setting '{key}' is ignored.", file=sys.stderr)
        # an explicit run id is not mixed with a record that was found by itself
        record_path = None if (args.run_id and not args.run_info) else find_run_record(args.run_info, cfg)
        info = api.read_run_info(record_path) if record_path else {}
        run_id = args.run_id or info.get("runId") or cfg.text("run_id")
        if not run_id:
            print("Error: no run to look at. Run tool 2-6 first (it writes run_info.json), or give --run-id / --run-info.", file=sys.stderr)
            return EXIT_ARGS
        api.check_run_id(run_id)
        conn = tfc.connection(cfg, args.mode, args.api_url, record=info)
        watch = args.watch if args.watch is not None else (cfg.int("watch") or 0)
        choice = (cfg.text("download", "ask") or "ask").lower()
        if choice not in ("ask", "yes", "no"):
            raise tfc.ConfigError(f"download must be ask, yes or no (got {choice!r})")
        want = "yes" if args.yes else "no" if args.no else choice
        result_arg = args.out_dir or cfg.file("result_dir")
        out_dir = Path(result_arg) if result_arg else (record_path.resolve().parent if record_path else Path.cwd())

        print(f"Settings: {cfg.path or 'no config file, command line only'}")
        if record_path:
            print(f"Run record: {record_path}")
        print(f"Run {run_id} on {conn.url} ({conn.mode} mode)")
        if info.get("apiUrl") and (info.get("mode"), info["apiUrl"].rstrip("/")) != (conn.mode, conn.url):
            print(f"Note: the run record says it was sent to {info['apiUrl']} ({info.get('mode')} mode), "
                  f"but the settings point to {conn.url}. If the run is not found, check mode and the address in config.txt.")
        while True:
            status = api.get_status(conn, run_id)
            state = status.get("status")
            if state in TERMINAL or watch <= 0:
                break
            print(f"  {describe(status)} ... checking again in {watch} s")
            time.sleep(watch)
    except tfc.ConfigError as e:
        print(f"Error: {e}", file=sys.stderr)
        return EXIT_ARGS
    except api.ApiError as e:
        print(f"Error: {e}", file=sys.stderr)
        return EXIT_UNREACHABLE

    if state == "Failed":
        err = status.get("error") or {}
        print(f"Failed: {err.get('type', 'error')}: {err.get('message', 'no message')}")
        return EXIT_FAILED
    if state == "Cancelled":
        print("Cancelled. Nothing to download.")
        return EXIT_CANCELLED
    if state != "Completed":
        label = "Waiting to start" if state == "Submitted" else "Still running"
        print(f"{label}: {describe(status)}. Run this tool again later (or use --watch 30).")
        return EXIT_RUNNING

    print("Done. The run has finished.")
    if want == "no" or (want == "ask" and not ask_download()):
        print("Not downloaded.")
        return EXIT_DONE
    try:
        result = api.download(conn, run_id)
    except api.ApiError as e:
        print(f"Error: {e}", file=sys.stderr)
        return EXIT_UNREACHABLE
    out_dir.mkdir(parents=True, exist_ok=True)
    schedule_out = out_dir / "04_Schedule.yaml"
    schedule_out.write_bytes(result)
    print(f"Saved {schedule_out}")
    env_src = info.get("envFile")
    if env_src and Path(env_src).is_file():
        shutil.copyfile(env_src, out_dir / "04_EnvConfig.yaml")
        print(f"Saved {out_dir / '04_EnvConfig.yaml'} (the EnvConfig that was sent; the solver does not change it)")
    else:
        print("04_EnvConfig.yaml not written: the run record does not point to the EnvConfig that was sent. "
              "Copy 03_EnvConfig.yaml to 04_EnvConfig.yaml.")
    return EXIT_DONE


if __name__ == "__main__":
    sys.exit(main())
