# update_schedule.py
# Usage:
#   python update_schedule.py 2025/10/01 --env EnvConfig.yaml --in Schedule.yaml --out Schedule.yaml
#
# This version does two things:
#   1) Updates schedule.assignment_list[*].plan_flexibility:
#        - start_date < cutoff  -> "Fixed"
#        - start_date >= cutoff -> "Flexible"
#   2) Optionally extends workflow_task_list with more modules (e(N+1)..eM)
#      according to:
#        - modules-per-day (average starts per working day)
#        - max-concurrent-modules (max modules running at once)
#        - shift-end-days (plan_range.end_date = last_module_start + shift_end_days)
#      using phase/operation definitions from EnvConfig.yaml
#
# Notes:
# - Dates are calendar dates; working days are Monday–Friday (no weekend starts).
# - Existing modules are kept as-is; new modules are appended.
# - You never lose modules: if --modules < existing count, the existing count wins.

import argparse
import datetime as dt
import shutil
import sys
from pathlib import Path
from typing import Any, Dict, List, Tuple, Optional

import yaml
import random


# ---------------- Helpers ----------------

def parse_date(s: str) -> dt.date:
    s = str(s).strip().replace("-", "/")
    return dt.datetime.strptime(s, "%Y/%m/%d").date()


def ymd(d: dt.date) -> str:
    return d.strftime("%Y/%m/%d")


def load_yaml(path: str) -> Dict[str, Any]:
    with open(path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f) or {}


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


def pick_workflow(env_root: Dict[str, Any], workflow_id: Optional[str]) -> Dict[str, Any]:
    wf_list = (env_root.get("environment") or env_root).get("workflow_list", [])
    if not wf_list:
        raise SystemExit("EnvConfig has no environment.workflow_list")

    if workflow_id:
        for wf in wf_list:
            if str(wf.get("id")) == str(workflow_id):
                return wf
        raise SystemExit(f'Workflow id "{workflow_id}" not found in EnvConfig.')
    else:
        return wf_list[0]


def collect_fabs(env_root: Dict[str, Any]) -> List[str]:
    lst = (env_root.get("environment") or env_root).get("fab_list", [])
    ids = [str(x.get("id")) for x in lst if x.get("id")]
    if not ids:
        raise SystemExit("EnvConfig has no fab_list ids.")
    return ids


def collect_phase_ops(workflow: Dict[str, Any]) -> List[Tuple[Dict[str, Any], List[Dict[str, Any]]]]:
    """
    Returns list of (phase_dict, op_list) preserving Env order.
    phase_dict must have id, name. op_list entries must have id, name.
    """
    out: List[Tuple[Dict[str, Any], List[Dict[str, Any]]]] = []
    for ph in workflow.get("phase_list", []):
        ph_id = str(ph.get("id") or "").strip()
        if not ph_id:
            continue
        ops: List[Dict[str, Any]] = []
        for op in ph.get("operation_list", []):
            op_id = str(op.get("id") or "").strip()
            if op_id:
                ops.append(op)
        if ops:
            out.append((ph, ops))
    if not out:
        raise SystemExit("Selected workflow has no phases with operations.")
    return out


def is_workday(d: dt.date) -> bool:
    # Monday=0, Sunday=6
    return d.weekday() < 5


# ---------------- Defaults (can be overridden by CLI) ----------------

DEFAULT_NUM_MODULES = 30
DEFAULT_MODULES_PER_DAY = 0.5
DEFAULT_MAX_CONCURRENT = 20
DEFAULT_SHIFT_END_DAYS = 60

DEFAULT_WORKFLOW_ID = None  # None -> first workflow in EnvConfig

DEFAULT_NAME_BASE = 1000
DEFAULT_NAME_PREFIX = "SU "
DEFAULT_NAME_SUFFIX = "A"

DEFAULT_PHASE_OFFSETS = {
    "p1": [20, 25],
    "p2": [30, 35],
    "p3": [40, 42],
    "p4": [51, 55],
}

DEFAULT_WORKLOAD_DAYS_DEFAULT = 20
DEFAULT_WORKLOAD_DAYS_MAP: Dict[str, int] = {}

DEFAULT_RANDOM_SEED = 42


# ---------------- Main ----------------

def main():
    ap = argparse.ArgumentParser(
        description=(
            "Update Schedule.yaml: "
            "1) mark assignments before cutoff as Fixed/after as Flexible, "
            "2) optionally append more modules using EnvConfig.yaml."
        )
    )
    ap.add_argument(
        "cutoff",
        help="Cutoff date (YYYY/MM/DD or YYYY-MM-DD). "
             "Assignments with start_date < cutoff become Fixed; others Flexible. "
             "plan_range.start_date is also set to this date.",
    )
    ap.add_argument("--env", required=True, help="Path to EnvConfig.yaml")
    ap.add_argument("--in", dest="inp", default="Schedule.yaml", help="Input Schedule.yaml path")
    ap.add_argument("--out", dest="out", default=None, help="Output path (default: overwrite input)")

    # Module-generation knobs (similar to generate_schedule.py)
    ap.add_argument("--modules", type=int, default=DEFAULT_NUM_MODULES,
                    help="Desired TOTAL modules after update (e1..eN). Existing modules are kept; "
                         "if this value is smaller than current count, the current count wins.")
    ap.add_argument("--modules-per-day", type=float, default=DEFAULT_MODULES_PER_DAY,
                    help="Average number of module starts per working day (Mon–Fri). "
                         "Example: 0.5 -> about 1 module every 2 working days.")
    ap.add_argument("--max-concurrent-modules", type=int, default=DEFAULT_MAX_CONCURRENT,
                    help="Maximum number of modules allowed to be active (any phase) on the same day.")
    ap.add_argument("--shift-end-days", type=int, default=DEFAULT_SHIFT_END_DAYS,
                    help="plan_range.end_date = (last module start date) + shift_end_days")

    ap.add_argument("--workflow-id", type=str, default=DEFAULT_WORKFLOW_ID,
                    help="Workflow id to use from EnvConfig (default: first workflow)")

    ap.add_argument("--name-base", type=int, default=DEFAULT_NAME_BASE)
    ap.add_argument("--name-prefix", type=str, default=DEFAULT_NAME_PREFIX)
    ap.add_argument("--name-suffix", type=str, default=DEFAULT_NAME_SUFFIX)

    ap.add_argument("--phase-offsets", type=str, default="",
                    help='Inline YAML: phase end offset in days from module start, '
                         'e.g. "{p1:[20,25], p2:[30,35]}". If omitted, defaults are used.')

    ap.add_argument("--workload-default", type=int, default=DEFAULT_WORKLOAD_DAYS_DEFAULT)
    ap.add_argument("--workload-map", type=str, default="",
                    help='Inline YAML: workload_days per operation id, '
                         'e.g. "{p1o1:9, p1o2:15}".')

    ap.add_argument("--seed", type=int, default=DEFAULT_RANDOM_SEED)

    args = ap.parse_args()

    cutoff = parse_date(args.cutoff)
    in_path = Path(args.inp)
    out_path = Path(args.out) if args.out else in_path

    if not in_path.exists():
        print(f"Input file not found: {in_path}", file=sys.stderr)
        sys.exit(1)

    # Load Schedule
    with in_path.open("r", encoding="utf-8") as f:
        data = yaml.safe_load(f) or {}

    # root may be nested under "schedule"
    root = data.get("schedule", data)

    # ---- 1) Update assignments Fixed/Flexible ----
    assignments = root.get("assignment_list", []) or []
    changed_fixed = 0
    total_assign = 0

    for a in assignments:
        total_assign += 1
        sd_raw = a.get("start_date")
        if not sd_raw:
            # if missing, try infer from first work_date_list entry
            wd_key = "work_date_lsit" if "work_date_lsit" in a else "work_date_list"
            if a.get(wd_key):
                sd_raw = a[wd_key][0].get("date")

        try:
            if sd_raw:
                sd = parse_date(sd_raw)
                if sd < cutoff:
                    if a.get("plan_flexibility") != "Fixed":
                        a["plan_flexibility"] = "Fixed"
                        changed_fixed += 1
                else:
                    if a.get("plan_flexibility") != "Flexible":
                        a["plan_flexibility"] = "Flexible"
            else:
                # No start date info; leave as-is
                pass
        except Exception:
            # If date parsing fails, leave entry unchanged
            pass

    # ---- 2) Extend workflow_task_list with more modules (optional) ----

    env_root = load_yaml(args.env)
    workflow = pick_workflow(env_root, args.workflow_id)
    fab_ids = collect_fabs(env_root)
    phase_ops = collect_phase_ops(workflow)

    # Phase offset ranges
    phase_offsets = {k: list(v) for k, v in DEFAULT_PHASE_OFFSETS.items()}
    if args.phase_offsets:
        user_po = parse_inline_yaml_dict(args.phase_offsets)
        for k, v in user_po.items():
            if not isinstance(v, list) or len(v) != 2:
                raise SystemExit(f"--phase-offsets value for {k} must be [min,max]")
            phase_offsets[str(k)] = [int(v[0]), int(v[1])]

    # Workload-days map
    wl_default = int(args.workload_default)
    workload_map: Dict[str, int] = dict(DEFAULT_WORKLOAD_DAYS_MAP)
    if args.workload_map:
        user_wm = parse_inline_yaml_dict(args.workload_map)
        for k, v in user_wm.items():
            workload_map[str(k)] = int(v)

    # Helper to build module display-name (for NEW modules only)
    def module_name(idx1: int) -> str:
        # idx1 is module index (1-based, from e{idx})
        num = args.name_base + idx1
        return f"{args.name_prefix}{num}{args.name_suffix}"

    wf_list = root.get("workflow_task_list", []) or []

    # Collect existing modules: id = eN, start/end from phases
    existing_modules: List[Tuple[int, Dict[str, Any], dt.date, dt.date]] = []
    last_mod_idx = 0
    last_existing_start: Optional[dt.date] = None
    existing_intervals: List[Tuple[dt.date, dt.date]] = []

    for mod in wf_list:
        mod_id = str(mod.get("id", "")).strip()
        if not (mod_id.startswith("e") and mod_id[1:].isdigit()):
            continue
        idx = int(mod_id[1:])
        phases = mod.get("phase_task_list", []) or []
        if not phases:
            continue

        start_dates: List[dt.date] = []
        end_dates: List[dt.date] = []
        for ph in phases:
            try:
                sd = parse_date(ph.get("start_date"))
                ed = parse_date(ph.get("end_date"))
                start_dates.append(sd)
                end_dates.append(ed)
            except Exception:
                continue
        if not start_dates or not end_dates:
            continue

        mstart = min(start_dates)
        mend = max(end_dates)
        existing_modules.append((idx, mod, mstart, mend))
        existing_intervals.append((mstart, mend))
        if idx > last_mod_idx:
            last_mod_idx = idx
        if last_existing_start is None or mstart > last_existing_start:
            last_existing_start = mstart

    existing_modules.sort(key=lambda t: t[0])

    # Decide how many modules we want in total
    desired_total = max(int(args.modules), last_mod_idx)  # never drop existing ones
    new_modules_needed = max(0, desired_total - last_mod_idx)

    # Base date for new module start scheduling:
    # - cannot be before cutoff (we only care about future plan)
    # - but must be at least the start of the last existing module,
    #   so that e(N+1) starts at/after last module's start date.
    if last_existing_start is not None:
        base_date = max(cutoff, last_existing_start)
    else:
        base_date = cutoff

    # Build deterministic fab choices as if all modules 1..desired_total were generated in one run
    fab_rng = random.Random(args.seed)
    fab_choice: Dict[int, str] = {}
    for mi in range(1, desired_total + 1):
        fab_choice[mi] = fab_rng.choice(fab_ids)

    # For NEW modules only: decide phase end offsets & duration in days
    module_phase_offsets: Dict[int, Dict[str, int]] = {}
    module_duration: Dict[int, int] = {}
    if new_modules_needed > 0:
        for mi in range(last_mod_idx + 1, desired_total + 1):
            rng = random.Random(args.seed + mi)
            per_phase: Dict[str, int] = {}
            max_off = 0
            for (ph, _ops) in phase_ops:
                ph_id = str(ph.get("id"))
                if ph_id not in phase_offsets:
                    raise SystemExit(
                        f"Phase '{ph_id}' missing offset range. "
                        f"Add it via --phase-offsets or DEFAULT_PHASE_OFFSETS."
                    )
                min_off, max_off_val = phase_offsets[ph_id]
                if min_off > max_off_val:
                    raise SystemExit(f"Invalid phase offset for {ph_id}: [{min_off},{max_off_val}]")
                off = rng.randint(min_off, max_off_val)
                per_phase[ph_id] = off
                if off > max_off:
                    max_off = off
            module_phase_offsets[mi] = per_phase
            module_duration[mi] = max_off

    # Build existing intervals relative to base_date for concurrency tracking
    existing_initial_active: List[Tuple[int, int]] = []
    existing_by_start: Dict[int, List[Tuple[int, int]]] = {}

    for (s_date, e_date) in existing_intervals:
        start_idx = (s_date - base_date).days
        end_idx = (e_date - base_date).days
        if end_idx < 0:
            continue  # finished before base_date
        interval = (start_idx, end_idx)
        if start_idx <= 0:
            existing_initial_active.append(interval)
        else:
            existing_by_start.setdefault(start_idx, []).append(interval)

    # Schedule NEW modules (if any) obeying modules-per-day & max-concurrent
    module_start_idx: Dict[int, int] = {}
    if new_modules_needed > 0:
        active_intervals: List[Tuple[int, int]] = list(existing_initial_active)
        day_idx = 0
        current_date = base_date
        carry = 0.0
        next_module = last_mod_idx + 1
        max_days_limit = 365 * 5  # safety upper bound

        while next_module <= desired_total:
            if day_idx > max_days_limit:
                raise SystemExit(
                    f"Cannot place all new modules within {max_days_limit} days "
                    f"from {ymd(base_date)} under modules-per-day={args.modules_per_day} "
                    f"and max-concurrent-modules={args.max_concurrent_modules}."
                )

            # Drop finished modules from active set
            active_intervals = [(s, e) for (s, e) in active_intervals if e >= day_idx]

            # Activate existing intervals starting today
            for itv in existing_by_start.get(day_idx, []):
                active_intervals.append(itv)

            if is_workday(current_date):
                carry += float(args.modules_per_day)

                # Start as many modules as allowed today
                while (
                    carry >= 1.0
                    and next_module <= desired_total
                    and len(active_intervals) < int(args.max_concurrent_modules)
                ):
                    mi = next_module
                    dur = module_duration.get(mi)
                    if dur is None or dur <= 0:
                        raise SystemExit(
                            f"Internal error: duration for module {mi} not set correctly."
                        )
                    start_idx = day_idx
                    end_idx = day_idx + dur
                    active_intervals.append((start_idx, end_idx))
                    module_start_idx[mi] = start_idx
                    next_module += 1
                    carry -= 1.0

            day_idx += 1
            current_date += dt.timedelta(days=1)

    # Compute last module start date (existing + new)
    last_start_date_overall: Optional[dt.date] = last_existing_start
    if module_start_idx:
        last_new_idx = max(module_start_idx.values())
        last_new_start = base_date + dt.timedelta(days=last_new_idx)
        if last_start_date_overall is None or last_new_start > last_start_date_overall:
            last_start_date_overall = last_new_start

    if last_start_date_overall is None:
        # No modules at all; base on cutoff
        last_start_date_overall = cutoff

    plan_end = last_start_date_overall + dt.timedelta(days=int(args.shift_end_days))

    # Update plan_range start/end
    plan_range = root.get("plan_range") or {}
    plan_range["start_date"] = ymd(cutoff)
    plan_range["end_date"] = ymd(plan_end)
    root["plan_range"] = plan_range

    # Append newly built modules to workflow_task_list
    wf_id_in_env = str(workflow.get("id"))
    warnings: List[str] = []

    if new_modules_needed > 0:
        for mi in range(last_mod_idx + 1, desired_total + 1):
            mod_id = f"e{mi}"
            mod_name = module_name(mi)
            fab = fab_choice.get(mi, fab_ids[0])

            start_idx = module_start_idx[mi]
            module_start_date = base_date + dt.timedelta(days=start_idx)

            per_phase = module_phase_offsets[mi]
            phase_task_list: List[Dict[str, Any]] = []

            for (ph, ops) in phase_ops:
                ph_id = str(ph.get("id"))
                ph_name = str(ph.get("name") or ph_id)

                if ph_id not in per_phase:
                    raise SystemExit(f"Internal error: missing phase offset for {ph_id} in module {mi}.")

                offset_days = per_phase[ph_id]
                phase_start = module_start_date
                phase_end = module_start_date + dt.timedelta(days=offset_days)

                op_task_list: List[Dict[str, Any]] = []
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

                if phase_end > plan_end:
                    warnings.append(
                        f"[WARN] {mod_id} {ph_id} end_date {ymd(phase_end)} exceeds plan_range.end_date {ymd(plan_end)}"
                    )

            wf_list.append({
                "id": mod_id,
                "name": mod_name,
                "workflow": wf_id_in_env,
                "fab": fab,
                "phase_task_list": phase_task_list,
            })

        root["workflow_task_list"] = wf_list

    # Put root back into data if nested under "schedule"
    if "schedule" in data:
        data["schedule"] = root
    else:
        data = root

    # Backup if overwriting
    if out_path == in_path:
        bak = in_path.with_suffix(in_path.suffix + ".bak")
        try:
            shutil.copy2(in_path, bak)
            print(f"Backup written: {bak}")
        except Exception as e:
            print(f"Warning: failed to write backup: {e}", file=sys.stderr)

    # Dump YAML back
    with out_path.open("w", encoding="utf-8") as f:
        yaml.safe_dump(data, f, sort_keys=False, allow_unicode=True)

    print(f"Processed {total_assign} assignments. Updated to Fixed: {changed_fixed}.")
    if new_modules_needed > 0:
        print(f"Existing modules: {last_mod_idx}. Total after update: {desired_total}. "
              f"Added {new_modules_needed} modules.")
    else:
        print(f"No new modules added. Existing modules: {last_mod_idx} (>= desired {args.modules}).")

    print(f"New plan_range: start_date={ymd(cutoff)}, end_date={ymd(plan_end)} -> written: {out_path}")
    if warnings:
        print("\n".join(warnings))


if __name__ == "__main__":
    main()
