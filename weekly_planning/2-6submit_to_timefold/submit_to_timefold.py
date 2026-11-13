#!/usr/bin/env python3
"""
submit_to_timefold.py  --  weekly planning tool "2-6": send the solver input to Timefold and start the run.

Run it with no options: everything is read from the config.txt next to this script (mode, address, input files, run id, solver parameters).
Any command line option replaces the same setting of config.txt for that run.

Input   03_Schedule.yaml, 03_EnvConfig.yaml   (from tool 2-5)
Output  run_info.json (the run id and where the input came from; tool 3-1 reads it)  and  SolverParams.yaml (a copy of what was sent)

The two yaml files are sent as the solver inputs EnvConfig.yaml and Schedule.yaml, plus a third input, SolverParams.yaml,
made from these defaults (everything else keeps the solver default):
    stage 2 on, stage 1 at most 45 minutes + stage 2 at most 15 minutes = 1 hour in total, Fixed trips are not cut away.

mode=local   sends to the local server (MockServer, runs the solver in Docker).
mode=online  sends to the API controller on Azure (azure_url), the same server SchedulerWeb uses.
"""
from __future__ import annotations

import argparse
import sys
from datetime import datetime
from pathlib import Path

import yaml

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent))      # weekly_planning/ (the folder of common)
from common import tf_config as tfc, timefold_api as api   # noqa: E402

CONFIG_FILE = HERE / tfc.CONFIG_NAME        # the settings of this tool


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
            raise api.ApiError(f"{path} does not look like a {label} file (no top-level '{key}'). Are schedule and env swapped?")


def first(*values):
    """The first value that is not None (command line, then config.txt)."""
    for v in values:
        if v is not None:
            return v
    return None


def main(argv=None):
    ap = argparse.ArgumentParser(description="Weekly planning 2-6: send the solver input to Timefold and start the run. "
                                             "Settings come from config.txt; the options below replace them for one run.")
    ap.add_argument("--config", default=None, help=f"settings file (default: {tfc.CONFIG_NAME} next to this script)")
    ap.add_argument("--schedule", default=None, help="03_Schedule.yaml (config: schedule)")
    ap.add_argument("--env", default=None, help="03_EnvConfig.yaml (config: env)")
    ap.add_argument("--out-dir", default=None, help="folder for run_info.json and SolverParams.yaml (config: out_dir; default: the folder of the schedule)")
    ap.add_argument("--mode", choices=api.MODES, default=None, help="local or online (config: mode)")
    ap.add_argument("--api-url", default=None, help="address of the server, replaces local_url / azure_url")
    ap.add_argument("--run-id", default=None, help="name of the run (config: run_id; default: date and time)")
    ap.add_argument("--params-file", default=None, help="a SolverParams.yaml of your own (config: params_file; the values below still win)")
    ap.add_argument("--stage1-minutes", type=int, default=None, help="stage 1 time limit (default 45)")
    ap.add_argument("--stage2-minutes", type=int, default=None, help="stage 2 time limit (default 15)")
    ap.add_argument("--no-stage2", action="store_true", help="skip stage 2")
    ap.add_argument("--stage1-unimproved-seconds", type=int, default=None, help="stage 1 stops after this long without improvement (solver default 600)")
    ap.add_argument("--stage2-unimproved-seconds", type=int, default=None, help="stage 2 stops after this long without improvement (solver default 60)")
    ap.add_argument("--seed", type=int, default=None, help="random seed (config: random_seed)")
    ap.add_argument("--allow-cut-fixed-trip", action="store_true", help="let the solver cut Fixed trips without work inside (off by default)")
    ap.add_argument("--dry-run", action="store_true", help="show what would be sent, send nothing")
    args = ap.parse_args(argv)

    try:
        cfg = tfc.load(args.config, CONFIG_FILE, tfc.SUBMIT_KEYS)
        for key in cfg.unknown:
            print(f"Note: {cfg.path.name}: unknown setting '{key}' is ignored.", file=sys.stderr)
        schedule = first(args.schedule, cfg.file("schedule"))
        env = first(args.env, cfg.file("env"))
        for label, value, key in (("schedule", schedule, "schedule"), ("env", env, "env")):
            if not value:
                raise api.ApiError(f"no {label} file: give --{label} or set {key} in {cfg.path or tfc.CONFIG_NAME}")
        check_inputs(schedule, env)
        params_file = first(args.params_file, cfg.file("params_file"))
        params = api.build_params(
            params_file,
            stage1_minutes=first(args.stage1_minutes, cfg.int("stage1_minutes")),
            stage2_minutes=first(args.stage2_minutes, cfg.int("stage2_minutes")),
            stage2_enabled=False if args.no_stage2 else cfg.bool("stage2_enabled"),
            stage1_unimproved_seconds=first(args.stage1_unimproved_seconds, cfg.int("stage1_unimproved_seconds")),
            stage1_stop_when_feasible=cfg.bool("stage1_stop_when_feasible"),
            stage2_unimproved_seconds=first(args.stage2_unimproved_seconds, cfg.int("stage2_unimproved_seconds")),
            random_seed=first(args.seed, cfg.int("random_seed")),
            allow_cut_fixed_trip=True if args.allow_cut_fixed_trip else cfg.bool("allow_cut_fixed_trip"))
        conn = tfc.connection(cfg, args.mode, args.api_url)
        run_id = api.check_run_id(first(args.run_id, cfg.text("run_id")) or api.new_run_id())
        out_arg = first(args.out_dir, cfg.file("out_dir"))
        out_dir = Path(out_arg) if out_arg else Path(schedule).resolve().parent

        print(f"Settings: {cfg.path or 'no config file, command line only'}")
        print(f"Server:   {conn.url} ({conn.mode} mode)")
        print(f"Solver parameters (SolverParams.yaml), up to {api.time_budget_minutes(params)} minutes in total:")
        for line in api.params_yaml(params).splitlines():
            print(f"  {line}")
        if args.dry_run:
            print(f"Dry run: would send {schedule} and {env} as run {run_id}. Nothing sent.")
            return 0

        if conn.mode == "online":
            print("Checking the server ...")
            api.health(conn)
        print(f"Sending run {run_id} ...")
        answer = api.submit(conn, env, schedule, api.params_yaml(params), run_id)
    except api.ApiError as e:
        print(f"Error: {e}", file=sys.stderr)
        return 1

    run_id = answer.get("runId", run_id)
    out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / "SolverParams.yaml").write_text(api.params_yaml(params), encoding="utf-8")
    info_path = api.write_run_info(out_dir, {
        "runId": run_id, "mode": conn.mode, "apiUrl": conn.url,
        "submittedAt": datetime.now().isoformat(timespec="seconds"),
        "scheduleFile": str(Path(schedule).resolve()), "envFile": str(Path(env).resolve()),
        "params": params})
    print(f"Submitted: run {run_id} ({answer.get('status', 'Submitted')}).")
    print(f"Run record: {info_path}")
    print("Check it later with tool 3-1 (python receive_from_timefold.py).")
    return 0


if __name__ == "__main__":
    sys.exit(main())