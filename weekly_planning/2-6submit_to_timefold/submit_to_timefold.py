#!/usr/bin/env python3
"""
submit_to_timefold.py  --  weekly planning tool "2-6": send the solver input to Timefold and start the run.

Input   N/03_Schedule.yaml, N/03_EnvConfig.yaml   (from tool 2-5)
Output  run_info.json (the run id and where the input came from; tool 3-1 reads it)  and  SolverParams.yaml (a copy of what was sent)

The two yaml files are sent as the solver inputs EnvConfig.yaml and Schedule.yaml, plus a third input, SolverParams.yaml,
made from these defaults (everything else keeps the solver default):
    stage 2 on, stage 1 at most 45 minutes + stage 2 at most 15 minutes = 1 hour in total, Fixed trips are not cut away.
Change them with the options below, or give a SolverParams.yaml of your own with --params-file (the options still win).

--mode local  sends to the local server (MockServer, runs the solver in Docker), --api-url to change its address.
--mode real   (the default) is the Azure API controller; it is not available yet.
"""
from __future__ import annotations

import argparse
import sys
from datetime import datetime
from pathlib import Path

import yaml

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import timefold_api as api   # noqa: E402


def check_inputs(schedule_path, env_path):
    """Cheap sanity check so that swapped or wrong files are caught before anything is sent."""
    for label, path, key in (("schedule", schedule_path, "schedule"), ("EnvConfig", env_path, "environment")):
        if not Path(path).is_file():
            raise api.ApiError(f"{label} file not found: {path}")
        try:
            doc = yaml.safe_load(Path(path).read_text(encoding="utf-8"))
        except yaml.YAMLError as e:
            raise api.ApiError(f"{label} file is not valid YAML: {path} ({e})") from None
        if not isinstance(doc, dict) or key not in doc:
            raise api.ApiError(f"{path} does not look like a {label} file (no top-level '{key}'). Are --schedule and --env swapped?")


def main(argv=None):
    ap = argparse.ArgumentParser(description="Weekly planning 2-6: send the solver input to Timefold and start the run.")
    ap.add_argument("--schedule", required=True, help="03_Schedule.yaml")
    ap.add_argument("--env", required=True, help="03_EnvConfig.yaml")
    ap.add_argument("--out-dir", default=None, help="folder for run_info.json and SolverParams.yaml (default: the folder of the schedule)")
    ap.add_argument("--mode", choices=api.MODES, default=api.DEFAULT_MODE, help="real (default, not available yet) or local")
    ap.add_argument("--api-url", default=None, help=f"address of the local server (default {api.DEFAULT_LOCAL_URL})")
    ap.add_argument("--run-id", default=None, help="name of the run (default: date and time)")
    ap.add_argument("--params-file", default=None, help="a SolverParams.yaml of your own (the options below still win)")
    ap.add_argument("--stage1-minutes", type=int, default=None, help="stage 1 time limit (default 45)")
    ap.add_argument("--stage2-minutes", type=int, default=None, help="stage 2 time limit (default 15)")
    ap.add_argument("--no-stage2", action="store_true", help="skip stage 2")
    ap.add_argument("--stage1-unimproved-seconds", type=int, default=None, help="stage 1 stops after this long without improvement (solver default 600)")
    ap.add_argument("--stage2-unimproved-seconds", type=int, default=None, help="stage 2 stops after this long without improvement (solver default 60)")
    ap.add_argument("--seed", type=int, default=None, help="random seed")
    ap.add_argument("--allow-cut-fixed-trip", action="store_true", help="let the solver cut Fixed trips without work inside (off by default)")
    ap.add_argument("--dry-run", action="store_true", help="show what would be sent, send nothing")
    args = ap.parse_args(argv)

    try:
        check_inputs(args.schedule, args.env)
        params = api.build_params(
            args.params_file, stage1_minutes=args.stage1_minutes, stage2_minutes=args.stage2_minutes,
            stage2_enabled=False if args.no_stage2 else None,
            stage1_unimproved_seconds=args.stage1_unimproved_seconds, stage2_unimproved_seconds=args.stage2_unimproved_seconds,
            random_seed=args.seed, allow_cut_fixed_trip=True if args.allow_cut_fixed_trip else None)
        base_url = api.resolve_base_url(args.mode, args.api_url)
        run_id = args.run_id or api.new_run_id()
        out_dir = Path(args.out_dir) if args.out_dir else Path(args.schedule).resolve().parent

        print(f"Solver parameters (SolverParams.yaml), up to {api.time_budget_minutes(params)} minutes in total:")
        for line in api.params_yaml(params).splitlines():
            print(f"  {line}")
        if args.dry_run:
            print(f"Dry run: would send {args.schedule} and {args.env} to {base_url} as run {run_id}. Nothing sent.")
            return 0

        print(f"Sending to {base_url} ({args.mode} mode) ...")
        answer = api.submit(base_url, args.env, args.schedule, api.params_yaml(params), run_id)
    except api.NotAvailable as e:
        print(f"Not available: {e}", file=sys.stderr)
        return 2
    except api.ApiError as e:
        print(f"Error: {e}", file=sys.stderr)
        return 1

    run_id = answer.get("runId", run_id)
    out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / "SolverParams.yaml").write_text(api.params_yaml(params), encoding="utf-8")
    info_path = api.write_run_info(out_dir, {
        "runId": run_id, "mode": args.mode, "apiUrl": base_url,
        "submittedAt": datetime.now().isoformat(timespec="seconds"),
        "scheduleFile": str(Path(args.schedule).resolve()), "envFile": str(Path(args.env).resolve()),
        "params": params})
    print(f"Submitted: run {run_id} ({answer.get('status', 'Submitted')}).")
    print(f"Run record: {info_path}")
    print(f"Check it later with tool 3-1:  --run-info {info_path}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
