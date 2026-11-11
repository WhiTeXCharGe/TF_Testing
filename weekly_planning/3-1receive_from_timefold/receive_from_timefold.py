#!/usr/bin/env python3
"""
receive_from_timefold.py  --  weekly planning tool "3-1": ask Timefold how the run is going and fetch the result.

Input   run_info.json (written by tool 2-6) or --run-id
Output  N/04_Schedule.yaml, N/04_EnvConfig.yaml   (only if you answer Y)

  * not finished  -> prints that it is still running (stage and progress) and ends; run it again later, or use --watch
  * finished      -> prints that it is done and asks whether to download:  Y downloads, N ends
  * failed / cancelled -> prints the reason, nothing to download
04_Schedule.yaml is the result of the solver. The solver does not change the EnvConfig, so 04_EnvConfig.yaml is the 03_EnvConfig.yaml
that was sent (taken from the run record).

Exit codes: 0 done (downloaded or not), 1 not finished yet, 2 failed, 3 cancelled, 4 cannot reach the server / not available, 5 bad arguments.
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


def main(argv=None):
    ap = argparse.ArgumentParser(description="Weekly planning 3-1: show the state of a Timefold run and download the result.")
    ap.add_argument("--run-info", default=None, help=f"the {api.RUN_INFO_NAME} written by tool 2-6 (default: ./{api.RUN_INFO_NAME} if it exists)")
    ap.add_argument("--run-id", default=None, help="the run id (instead of --run-info)")
    ap.add_argument("--mode", choices=api.MODES, default=None, help="real (default, not available yet) or local (default: the mode of the run record)")
    ap.add_argument("--api-url", default=None, help="address of the local server (default: the one of the run record)")
    ap.add_argument("--out-dir", default=None, help="folder for 04_Schedule.yaml and 04_EnvConfig.yaml (default: the folder of the run record)")
    ap.add_argument("--yes", action="store_true", help="download without asking")
    ap.add_argument("--no", action="store_true", help="never download, only show the state")
    ap.add_argument("--watch", type=int, default=0, metavar="SECONDS", help="check again every SECONDS until the run has ended")
    args = ap.parse_args(argv)
    if args.yes and args.no:
        ap.error("--yes and --no cannot be used together")

    try:
        info, info_path = {}, None
        run_info_arg = args.run_info or (api.RUN_INFO_NAME if not args.run_id and Path(api.RUN_INFO_NAME).is_file() else None)
        if run_info_arg:
            info_path = Path(run_info_arg)
            info = api.read_run_info(info_path)
        run_id = args.run_id or info.get("runId")
        if not run_id:
            print("Error: give --run-info (written by tool 2-6) or --run-id.", file=sys.stderr)
            return EXIT_ARGS
        mode = args.mode or info.get("mode") or api.DEFAULT_MODE
        base_url = api.resolve_base_url(mode, args.api_url or (info.get("apiUrl") if mode == "local" else None))
        out_dir = Path(args.out_dir) if args.out_dir else (info_path.resolve().parent if info_path else Path.cwd())

        print(f"Run {run_id} on {base_url} ({mode} mode)")
        while True:
            status = api.get_status(base_url, run_id)
            state = status.get("status")
            if state in TERMINAL or args.watch <= 0:
                break
            print(f"  {describe(status)} ... checking again in {args.watch} s")
            time.sleep(args.watch)
    except api.NotAvailable as e:
        print(f"Not available: {e}", file=sys.stderr)
        return EXIT_UNREACHABLE
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
    if args.no or not (args.yes or ask_download()):
        print("Not downloaded.")
        return EXIT_DONE
    try:
        result = api.download(base_url, run_id)
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
