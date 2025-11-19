#!/usr/bin/env python3
# generate_schedule.py
# Usage examples:
#   python generate_schedule.py --env EnvConfig.yaml --out Schedule.yaml
#
# New usage example:
#   python generate_schedule.py --env EnvConfig.yaml --out Schedule.yaml \
#     --plan-start 2025/09/01 \
#     --modules 30 \
#     --modules-per-day 0.5 \
#     --max-concurrent-modules 20 \
#     --shift-end-days 60 \
#     --workflow-id workflow \
#     --name-base 1000 --name-prefix "SU " --name-suffix "A" \
#     --phase-offsets "{p1:[20,25], p2:[30,35], p3:[40,42], p4:[51,55]}" \
#     --workload-default 10 \
#     --workload-map "{p1o1:9, p1o2:15, p1o3:15, p2o1:10, p2o2:13, p2o3:9, p3o1:11, p3o2:10, p3o3:12, p4o1:8, p4o2:12, p4o3:14}" \
#     --seed 42
#
# Notes (new spec):
# - You specify only plan_start. plan_range.end_date is computed as:
#       last_module_start_date + shift_end_days
# - modules_per_day is per *working* day (Mon–Fri).
# - max_concurrent_modules is the maximum number of modules that can be
#   "active" at the same time (from module start until its last phase ends).
# - If max_concurrent_modules is too small for the given modules_per_day and
#   phase durations, the script exits with an error instead of generating
#   a huge impossible horizon.

import argparse
import random
import math
from datetime import datetime, timedelta
from pathlib import Path
from typing import Dict, List, Tuple, Any, Optional

import yaml

# ---------------- Default knobs ----------------
DEFAULT_PLAN_START = "2025/09/01"
# plan_end is now computed automatically; DEFAULT_PLAN_END kept only for docs
DEFAULT_PLAN_END   = "2025/12/20"

DEFAULT_NUM_MODULES = 30
DEFAULT_WORKFLOW_ID = None       # None -> auto-pick first workflow; or set to "workflow"

# Module name pattern: "SU {num}A", where num = name_base + index (starting at 1)
DEFAULT_NAME_BASE   = 1000
DEFAULT_NAME_PREFIX = "SU "
DEFAULT_NAME_SUFFIX = "A"

# max modules that can be active at the same time
DEFAULT_MAX_CONCURRENT_MODULES = 20

# average modules starting per working day (Mon–Fri)
DEFAULT_MODULES_PER_DAY = 0.5

# plan_range.end_date = last_start + SHIFT_END_DAYS
DEFAULT_SHIFT_END_DAYS = 60

# Phase end-date offset ranges (days from module start). Keys = phase id (e.g., p1, p2, ...).
# Values = [min,max] inclusive.
DEFAULT_PHASE_OFFSETS = {
    "p1": [20, 25],
    "p2": [30, 35],
    "p3": [40, 42],
    "p4": [51, 55],
}

# workload_days: default if not overridden per operation id (e.g., p1o1).
DEFAULT_WORKLOAD_DAYS_DEFAULT = 20
DEFAULT_WORKLOAD_DAYS_MAP: Dict[str, int] = {}

DEFAULT_RANDOM_SEED = 42
# ------------------------------------------------


def parse_inline_yaml_dict(s: str) -> Dict[str, Any]:
    """
    Parse a simple inline YAML/JSON-ish dict e.g. "{p1:[20,25], p2:[30,35]}"
    Returns {} if empty/None.
    """
    if not s:
        return {}
    try:
        return yaml.safe_load(s) or {}
    except Exception as e:
        raise SystemExit(f"Failed to parse inline YAML dict: {s}\nError: {e}")


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Generate Schedule.yaml from EnvConfig.yaml")

    p.add_argument("--env", required=True, help="Path to EnvConfig.yaml")
    p.add_argument("--out", required=True, help="Path to write Schedule.yaml")
    p.add_argument("--modules", type=int, default=DEFAULT_NUM_MODULES,
                   help="Number of modules (e1..eN)")

    p.add_argument("--workflow-id", type=str, default=DEFAULT_WORKFLOW_ID,
                   help="Workflow id to use from EnvConfig (default: first workflow)")

    p.add_argument("--plan-start", type=str, default=DEFAULT_PLAN_START,
                   help="Plan range start date (YYYY/MM/DD).")

    # Deprecated: kept only so old command lines don't crash.
    # This value is IGNORED; plan_range.end_date is derived automatically.
    p.add_argument("--plan-end", type=str, default=None,
                   help="(Deprecated, ignored) plan_range.end_date is computed from last module start + shift-end-days.")

    p.add_argument("--name-base", type=int, default=DEFAULT_NAME_BASE)
    p.add_argument("--name-prefix", type=str, default=DEFAULT_NAME_PREFIX)
    p.add_argument("--name-suffix", type=str, default=DEFAULT_NAME_SUFFIX)

    # module start density and concurrency
    p.add_argument("--modules-per-day", type=float, default=DEFAULT_MODULES_PER_DAY,
                   help="Average number of modules starting per working day (Mon–Fri).")
    p.add_argument("--max-concurrent-modules", type=int, default=DEFAULT_MAX_CONCURRENT_MODULES,
                   help="Maximum number of modules that may be active at the same time.")
    p.add_argument("--shift-end-days", type=int, default=DEFAULT_SHIFT_END_DAYS,
                   help="Plan end = last module start + shift-end-days.")

    p.add_argument("--phase-offsets", type=str, default="",
                   help='Inline YAML: e.g. "{p1:[20,25], p2:[30,35]}"')

    p.add_argument("--workload-default", type=int, default=DEFAULT_WORKLOAD_DAYS_DEFAULT)
    p.add_argument("--workload-map", type=str, default="",
                   help='Inline YAML: e.g. "{p1o1:9, p1o2:15}"')

    p.add_argument("--seed", type=int, default=DEFAULT_RANDOM_SEED)
    return p.parse_args()


def load_yaml(path: str) -> Dict[str, Any]:
    with open(path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f) or {}


def ymd(d: datetime) -> str:
    return d.strftime("%Y/%m/%d")


def parse_date(s: str) -> datetime:
    return datetime.strptime(s.replace("-", "/"), "%Y/%m/%d")


def pick_workflow(env: Dict[str, Any], workflow_id: Optional[str]) -> Dict[str, Any]:
    wf_list = (env.get("environment") or env).get("workflow_list", [])
    if not wf_list:
        raise SystemExit("EnvConfig has no environment.workflow_list")

    if workflow_id:
        for wf in wf_list:
            if str(wf.get("id")) == str(workflow_id):
                return wf
        raise SystemExit(f'Workflow id "{workflow_id}" not found in EnvConfig.')
    else:
        return wf_list[0]


def collect_fabs(env: Dict[str, Any]) -> List[str]:
    lst = (env.get("environment") or env).get("fab_list", [])
    ids = [str(x.get("id")) for x in lst if x.get("id")]
    if not ids:
        raise SystemExit("EnvConfig has no fab_list ids.")
    return ids


def collect_phase_ops(workflow: Dict[str, Any]) -> List[Tuple[Dict[str, Any], List[Dict[str, Any]]]]:
    """
    Returns list of (phase_dict, op_list) preserving Env order.
    phase_dict must have id, name. op_list entries must have id, name.
    """
    out = []
    for ph in workflow.get("phase_list", []):
        ph_id = str(ph.get("id") or "").strip()
        if not ph_id:
            continue
        ops = []
        for op in ph.get("operation_list", []):
            op_id = str(op.get("id") or "").strip()
            if op_id:
                ops.append(op)
        if ops:
            out.append((ph, ops))
    if not out:
        raise SystemExit("Selected workflow has no phases with operations.")
    return out


def is_workday(d: datetime) -> bool:
    # Monday=0 .. Sunday=6 → workdays = 0..4
    return d.weekday() < 5


def main():
    args = parse_args()

    if args.modules <= 0:
        raise SystemExit("--modules must be > 0")

    if args.modules_per_day <= 0:
        raise SystemExit("--modules-per-day must be > 0")

    if args.max_concurrent_modules <= 0:
        raise SystemExit("--max-concurrent-modules must be > 0")

    random.seed(args.seed)

    env_root = load_yaml(args.env)
    env = env_root.get("environment", env_root)

    workflow = pick_workflow(env_root, args.workflow_id)
    fab_ids = collect_fabs(env_root)
    phase_ops = collect_phase_ops(workflow)

    # plan start
    plan_start_dt = parse_date(args.plan_start)

    # phase offset ranges: start from defaults, then apply CLI overrides (if any)
    phase_offsets = {k: list(v) for k, v in DEFAULT_PHASE_OFFSETS.items()}
    if args.phase_offsets:
        user_po = parse_inline_yaml_dict(args.phase_offsets)
        for k, v in user_po.items():
            if not isinstance(v, list) or len(v) != 2:
                raise SystemExit(f"--phase-offsets value for {k} must be [min,max]")
            phase_offsets[str(k)] = [int(v[0]), int(v[1])]

    # workload map: start from defaults, then apply CLI overrides (if any)
    wl_default = int(args.workload_default)
    if wl_default <= 0:
        raise SystemExit("--workload-default must be > 0")

    workload_map = dict(DEFAULT_WORKLOAD_DAYS_MAP)
    if args.workload_map:
        user_wm = parse_inline_yaml_dict(args.workload_map)
        for k, v in user_wm.items():
            workload_map[str(k)] = int(v)

    # module name helper
    def module_name(idx1: int) -> str:
        num = args.name_base + idx1
        return f"{args.name_prefix}{num}{args.name_suffix}"

    num_modules = int(args.modules)

    # ----------------------------------------------------
    # 1) Pre-sample phase offsets per module (deterministic)
    # ----------------------------------------------------
    # module_phase_offsets[mi][ph_id] = offset_days
    module_phase_offsets: Dict[int, Dict[str, int]] = {}
    module_duration_days: Dict[int, int] = {}
    global_max_duration = 0

    phase_ids = [str(ph["id"]) for (ph, _ops) in phase_ops]

    for mi in range(1, num_modules + 1):
        rng = random.Random(args.seed + mi)
        offsets_for_module: Dict[str, int] = {}
        max_off = 0
        for ph_id in phase_ids:
            if ph_id not in phase_offsets:
                raise SystemExit(
                    f"Phase '{ph_id}' missing offset range. "
                    f"Add it via --phase-offsets or DEFAULT_PHASE_OFFSETS."
                )
            min_off, max_off_cfg = phase_offsets[ph_id]
            if min_off > max_off_cfg:
                raise SystemExit(f"Invalid phase offset for {ph_id}: [{min_off},{max_off_cfg}]")
            off = rng.randint(min_off, max_off_cfg)
            offsets_for_module[ph_id] = off
            if off > max_off:
                max_off = off
        module_phase_offsets[mi] = offsets_for_module
        module_duration_days[mi] = max_off
        if max_off > global_max_duration:
            global_max_duration = max_off

    # ----------------------------------------------------
    # 2) Feasibility check: rate vs concurrency
    # ----------------------------------------------------
    # modules_per_day is per *working* day → approx modules_per_calendar_day:
    modules_per_calendar_day = args.modules_per_day * (5.0 / 7.0)
    # worst-case active modules if pipeline is "full":
    estimated_peak = modules_per_calendar_day * global_max_duration

    if estimated_peak - 1e-9 > args.max_concurrent_modules:
        needed = math.ceil(estimated_peak)
        raise SystemExit(
            "Infeasible parameters:\n"
            f"- modules_per_day (working days)  = {args.modules_per_day}\n"
            f"- approx modules_per_calendar_day = {modules_per_calendar_day:.3f}\n"
            f"- max phase duration (days)       = {global_max_duration}\n"
            f"→ estimated peak concurrent modules ≈ {estimated_peak:.2f}, "
            f"so you need at least {needed} max-concurrent-modules.\n"
            f"But you specified max-concurrent-modules = {args.max_concurrent_modules}.\n"
            "Please increase --max-concurrent-modules or decrease --modules-per-day "
            "or shorten phase offset ranges."
        )

    # ----------------------------------------------------
    # 3) Schedule module start dates with:
    #       - modules_per_day on Mon–Fri
    #       - concurrency <= max_concurrent_modules
    # ----------------------------------------------------
    module_start_day_index: Dict[int, int] = {}
    active_intervals: List[Tuple[int, int]] = []  # (start_idx, end_idx) inclusive

    day_idx = 0
    cur_date = plan_start_dt
    carry = 0.0
    next_module = 1

    # safety guard (very generous)
    max_days_limit = 365 * 5  # 5 years

    while next_module <= num_modules:
        if day_idx > max_days_limit:
            raise SystemExit(
                "Internal error: scheduling exceeded 5 years. "
                "Check parameters; maybe modules_per_day is too small for the requested modules."
            )

        # drop finished modules (end_idx < current day_idx)
        active_intervals = [(s, e) for (s, e) in active_intervals if e >= day_idx]

        if is_workday(cur_date):
            carry += args.modules_per_day

            # Try to start as many modules as allowed by carry and concurrency
            while (
                carry >= 1.0
                and next_module <= num_modules
                and len(active_intervals) < args.max_concurrent_modules
            ):
                mi = next_module
                dur = module_duration_days[mi]
                start_idx = day_idx
                end_idx = day_idx + dur

                active_intervals.append((start_idx, end_idx))
                module_start_day_index[mi] = start_idx

                next_module += 1
                carry -= 1.0

        # Move to next day
        day_idx += 1
        cur_date += timedelta(days=1)

    # all modules scheduled
    last_start_idx = max(module_start_day_index.values())
    last_start_date = plan_start_dt + timedelta(days=last_start_idx)
    plan_end_dt = last_start_date + timedelta(days=int(args.shift_end_days))

    # ----------------------------------------------------
    # 4) Build schedule YAML structure
    # ----------------------------------------------------
    schedule: Dict[str, Any] = {
        "schedule": {
            "plan_range": {
                "start_date": ymd(plan_start_dt),
                "end_date": ymd(plan_end_dt),  # computed
            },
            "workflow_task_list": [],
            "assignment_list": [],  # no edits
        }
    }

    wf_id_in_env = str(workflow.get("id"))
    warnings: List[str] = []

    for mi in range(1, num_modules + 1):
        fab = random.choice(fab_ids)
        mod_id = f"e{mi}"
        mod_name = module_name(mi)

        start_idx = module_start_day_index[mi]
        module_start = plan_start_dt + timedelta(days=start_idx)
        offsets_for_module = module_phase_offsets[mi]

        phase_task_list = []

        for (ph, ops) in phase_ops:
            ph_id = str(ph.get("id"))
            ph_name = str(ph.get("name") or ph_id)

            offset_days = offsets_for_module[ph_id]
            phase_start = module_start
            phase_end = module_start + timedelta(days=offset_days)

            op_task_list = []
            for op in ops:
                op_id = str(op.get("id"))
                op_name = str(op.get("name") or op_id)

                wl = workload_map.get(op_id, wl_default)
                if wl <= 0:
                    raise SystemExit(
                        f"workload_days for op '{op_id}' is <= 0 (computed {wl}). "
                        f"Fix --workload-default/--workload-map or DEFAULT_WORKLOAD_DAYS_*."
                    )

                op_suffix = op_id.split("o", 1)[1] if "o" in op_id else op_id
                op_task_list.append({
                    "id": f"{mod_id}{ph_id}o{op_suffix}",
                    "name": op_name,
                    "operation": op_id,
                    "workload_days": int(wl),
                })

            phase_task_list.append({
                "id": f"{mod_id}{ph_id}",
                "name": ph_name,
                "phase": ph_id,
                "start_date": ymd(phase_start),
                "end_date": ymd(phase_end),
                "operation_task_list": op_task_list,
            })

            # warn if end beyond computed plan_end
            if phase_end > plan_end_dt:
                warnings.append(
                    f"[WARN] {mod_id} {ph_id} end_date {ymd(phase_end)} exceeds "
                    f"plan_range.end_date {ymd(plan_end_dt)}"
                )

        schedule["schedule"]["workflow_task_list"].append({
            "id": mod_id,
            "name": mod_name,
            "workflow": wf_id_in_env,
            "fab": fab,
            "phase_task_list": phase_task_list,
        })

    # ----------------------------------------------------
    # 5) Write Schedule.yaml
    # ----------------------------------------------------
    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, "w", encoding="utf-8") as f:
        yaml.dump(schedule, f, allow_unicode=True, sort_keys=False)

    print(f"Wrote {args.out}")
    print(f"Plan range: start={ymd(plan_start_dt)} end={ymd(plan_end_dt)}")
    print(f"Last module start: {ymd(last_start_date)} (shift_end_days={args.shift_end_days})")

    if warnings:
        print("\n".join(warnings))


if __name__ == "__main__":
    main()
