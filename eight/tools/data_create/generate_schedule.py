#!/usr/bin/env python3
# generate_schedule.py
# Usage examples:
#   python generate_schedule.py --env EnvConfig.yaml --out Schedule.yaml
#   python generate_schedule.py --env EnvConfig.yaml --out Schedule.yaml \
#     --modules 2 \
#     --plan-start 2025/09/01 --plan-end 2025/12/10 \
#     --workflow-id workflow \
#     --name-base 1000 --name-prefix "SU " --name-suffix "A" \
#     --start-shift-min 0 --start-shift-max 4 \
#     --phase-offsets "{p1:[20,25], p2:[30,35], p3:[40,42], p4:[51,55]}" \
#     --workload-default 10 \
#     --workload-map "{p1o1:9, p1o2:15, p1o3:15, p2o1:10, p2o2:13, p2o3:9, p3o1:11, p3o2:10, p3o3:12, p4o1:8, p4o2:12, p4o3:14}" \
#     --seed 42
#
# Notes:
# - Phases/operations are taken from EnvConfig.environment.workflow_list[].phase_list[].operation_list[].
# - Each module gets the same start_date across all its phases. End dates per phase are randomized within
#   a phase-specific offset range (configurable).
# - workload_days are constant per operation (same across modules) via --workload-default plus overrides
#   in --workload-map.
# - If any phase end_date exceeds plan_range.end_date, it's still written, and a warning is printed.

import argparse
import random
from datetime import datetime, timedelta
from pathlib import Path
from typing import Dict, List, Tuple, Any, Optional

import yaml

# ---------------- Default knobs (you can edit these instead of passing CLI flags) ----------------
DEFAULT_PLAN_START = "2025/09/01"
DEFAULT_PLAN_END   = "2025/12/20"

DEFAULT_NUM_MODULES = 30
DEFAULT_WORKFLOW_ID = None       # None -> auto-pick first workflow; or set to "workflow"

# Module name pattern: "SU {num}A", where num = name_base + index (starting at 1)
DEFAULT_NAME_BASE   = 1000
DEFAULT_NAME_PREFIX = "SU "
DEFAULT_NAME_SUFFIX = "A"

# Inter-module start-date shift range (days)
DEFAULT_START_SHIFT_MIN = 0
DEFAULT_START_SHIFT_MAX = 4

# Phase end-date offset ranges (days from module start). Keys = phase id (e.g., p1, p2, ...).
# Values = [min,max] inclusive.
DEFAULT_PHASE_OFFSETS = {
    "p1": [20, 25],
    "p2": [30, 35],
    "p3": [40, 42],
    "p4": [51, 55],
}

# workload_days: default if not overridden per operation id (e.g., p1o1).
DEFAULT_WORKLOAD_DAYS_DEFAULT = 10
# Per-op overrides. Example:
# {"p1o1": 9, "p1o2": 15, "p1o3": 15, "p2o1": 10, "p2o2": 13, "p2o3": 9,
#  "p3o1": 11, "p3o2": 10, "p3o3": 12, "p4o1": 8, "p4o2": 12, "p4o3": 14}
DEFAULT_WORKLOAD_DAYS_MAP: Dict[str, int] = {}

DEFAULT_RANDOM_SEED = 42
# -------------------------------------------------------------------------------------------------


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
    p.add_argument("--modules", type=int, default=DEFAULT_NUM_MODULES, help="Number of modules (e1..eN)")
    p.add_argument("--workflow-id", type=str, default=DEFAULT_WORKFLOW_ID,
                   help="Workflow id to use from EnvConfig (default: first workflow)")

    p.add_argument("--plan-start", type=str, default=DEFAULT_PLAN_START)
    p.add_argument("--plan-end", type=str, default=DEFAULT_PLAN_END)

    p.add_argument("--name-base", type=int, default=DEFAULT_NAME_BASE)
    p.add_argument("--name-prefix", type=str, default=DEFAULT_NAME_PREFIX)
    p.add_argument("--name-suffix", type=str, default=DEFAULT_NAME_SUFFIX)

    p.add_argument("--start-shift-min", type=int, default=DEFAULT_START_SHIFT_MIN)
    p.add_argument("--start-shift-max", type=int, default=DEFAULT_START_SHIFT_MAX)

    p.add_argument("--phase-offsets", type=str, default="", help='Inline YAML: e.g. "{p1:[20,25], p2:[30,35]}"')

    p.add_argument("--workload-default", type=int, default=DEFAULT_WORKLOAD_DAYS_DEFAULT)
    p.add_argument("--workload-map", type=str, default="", help='Inline YAML: e.g. "{p1o1:9, p1o2:15}"')

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


def main():
    args = parse_args()
    random.seed(args.seed)

    env_root = load_yaml(args.env)
    env = env_root.get("environment", env_root)

    workflow = pick_workflow(env_root, args.workflow_id)
    fab_ids = collect_fabs(env_root)
    phase_ops = collect_phase_ops(workflow)

    # plan range
    plan_start_dt = parse_date(args.plan_start)
    plan_end_dt   = parse_date(args.plan_end)

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
    workload_map = dict(DEFAULT_WORKLOAD_DAYS_MAP)
    if args.workload_map:
        user_wm = parse_inline_yaml_dict(args.workload_map)
        for k, v in user_wm.items():
            workload_map[str(k)] = int(v)

    # module name helper
    def module_name(idx1: int) -> str:
        # idx1 starts from 1
        num = args.name_base + idx1
        return f"{args.name_prefix}{num}{args.name_suffix}"

    # Start building schedule
    schedule: Dict[str, Any] = {
        "schedule": {
            "plan_range": {
                "start_date": ymd(plan_start_dt),
                "end_date": ymd(plan_end_dt),
            },
            "workflow_task_list": [],
            # keep assignment_list present but empty (no edits)
            "assignment_list": [],
        }
    }

    # inter-module start shifting
    shift_min = int(args.start_shift_min)
    shift_max = int(args.start_shift_max)
    if shift_min > shift_max:
        raise SystemExit("--start-shift-min cannot be greater than --start-shift-max")

    # workflow id used in schedule rows
    wf_id_in_env = str(workflow.get("id"))

    # generate modules e1..eN
    warnings: List[str] = []
    cursor_start = plan_start_dt  # first module starts at plan_start, others shift from previous
    for mi in range(1, args.modules + 1):
        fab = random.choice(fab_ids)

        # shift start date for modules after the first
        if mi > 1:
            shift_days = random.randint(shift_min, shift_max)
            cursor_start = cursor_start + timedelta(days=shift_days)

        mod_id = f"e{mi}"
        mod_name = module_name(mi)

        # one phase_task_list per phase in Env order
        phase_task_list = []
        for (ph, ops) in phase_ops:
            ph_id = str(ph.get("id"))
            ph_name = str(ph.get("name") or ph_id)

            # determine end-date offset range for this phase
            if ph_id not in phase_offsets:
                raise SystemExit(f"Phase '{ph_id}' missing offset range. Add it via --phase-offsets or DEFAULT_PHASE_OFFSETS.")
            min_off, max_off = phase_offsets[ph_id]
            if min_off > max_off:
                raise SystemExit(f"Invalid phase offset for {ph_id}: [{min_off},{max_off}]")

            # random end date in range
            phase_end = cursor_start + timedelta(days=random.randint(min_off, max_off))

            # build operation_task_list
            op_task_list = []
            for op in ops:
                op_id = str(op.get("id"))
                op_name = str(op.get("name") or op_id)

                # workload_days for this operation (constant across modules)
                wl = workload_map.get(op_id, wl_default)
                if wl <= 0:
                    raise SystemExit(
                        f"workload_days for op '{op_id}' is <= 0 (computed {wl}). "
                        f"Fix --workload-default/--workload-map or DEFAULT_WORKLOAD_DAYS_*."
                    )

                # make a stable per-module op task id, e.g., e1p2o1 (use suffix after 'o' from op_id)
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
                "start_date": ymd(cursor_start),
                "end_date": ymd(phase_end),
                "operation_task_list": op_task_list
            })

            # warn if end beyond plan_end
            if phase_end > plan_end_dt:
                warnings.append(
                    f"[WARN] {mod_id} {ph_id} end_date {ymd(phase_end)} exceeds plan_range.end_date {ymd(plan_end_dt)}"
                )

        schedule["schedule"]["workflow_task_list"].append({
            "id": mod_id,
            "name": mod_name,
            "workflow": wf_id_in_env,
            "fab": fab,
            "phase_task_list": phase_task_list
        })

    # write Schedule.yaml
    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, "w", encoding="utf-8") as f:
        yaml.dump(schedule, f, allow_unicode=True, sort_keys=False)

    print(f"Wrote {args.out}")
    if warnings:
        print("\n".join(warnings))


if __name__ == "__main__":
    main()
