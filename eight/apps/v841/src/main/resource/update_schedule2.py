# update_schedule2.py
# Incremental schedule updater used by daily_run.py
#
# Behavior:
#   1) EnvConfig.yaml:
#        - If worker_list length < WORKER_NUM (from update_config.py),
#          add more workers using same logic as data_generation.py.
#
#   2) Schedule.yaml:
#        - assignment_list[*].plan_flexibility:
#             start_date < CUTOFF_DATE_STR  -> "Fixed"
#             start_date >= CUTOFF_DATE_STR -> "Flexible"
#        - Incrementally append new modules e(N+1).. as needed
#          until EQ_NUM is reached, but only a *few per run*
#          using EQ_PER_DAYS / EQ_PER_DAYS_SIGMA.
#        - plan_range:
#             start_date = CUTOFF_DATE_STR
#             end_date   = max(last module end, old end) + PLAN_RANGE_EXTRA_DAYS
#
# Notes:
# - Uses config_base.py for workload & weekend logic.
# - Uses update_config.py for cutoff date, paths, target worker/EQ numbers.
# - daily_run.py sets:
#       cfg.CUTOFF_DATE_STR        (cutoff for Fixed/Flexible)
#       cfg.CURRENT_SIM_DAY_STR    (the "today" of this simulation step)
#       cfg.MODULE_SEED_OFFSET     (per-day offset for randomness)

import datetime as dt
from pathlib import Path
from typing import Any, Dict, List, Tuple, Optional
import random
import shutil

import yaml

import config_base as base
import update_config as cfg


# ---------------- helpers ----------------

class InlineList(list):
    pass

class InlineDict(dict):
    pass

def represent_inline_list(dumper, data):
    return dumper.represent_sequence("tag:yaml.org,2002:seq", data, flow_style=True)

def represent_inline_dict(dumper, data):
    return dumper.represent_mapping("tag:yaml.org,2002:map", data, flow_style=True)

# register representers (works with safe_dump too)
def set_up_yaml_inline():
    # for default Dumper (yaml.dump)
    yaml.add_representer(InlineList, represent_inline_list)
    yaml.add_representer(InlineDict, represent_inline_dict)

    # for SafeDumper (yaml.safe_dump)
    yaml.add_representer(InlineList, represent_inline_list, Dumper=yaml.SafeDumper)
    yaml.add_representer(InlineDict, represent_inline_dict, Dumper=yaml.SafeDumper)


def parse_date(s: str) -> dt.date:
    s = str(s).strip().replace("-", "/")
    return dt.datetime.strptime(s, "%Y/%m/%d").date()


def ymd(d: dt.date) -> str:
    return d.strftime("%Y/%m/%d")


def is_holiday(day: dt.date) -> bool:
    if base.is_skip_weekend and day.weekday() in (5, 6):
        return True
    return False


def create_worklength_list() -> Tuple[List[List[Tuple[int, List[int]]]], List[float]]:
    # Uses your normal_worklength / vip_worklength from config_base.py
    return (
        [base.normal_worklength, base.vip_worklength],
        [0.8, 0.2],
    )


def load_yaml(path: Path) -> Dict[str, Any]:
    if not path.exists():
        return {}
    with path.open("r", encoding="utf-8") as f:
        return yaml.safe_load(f) or {}


def backup_file(path: Path) -> None:
    if not path.exists():
        return
    bak = path.with_suffix(path.suffix + ".bak")
    try:
        shutil.copy2(path, bak)
        print(f"Backup written: {bak}")
    except Exception as e:
        print(f"Warning: failed to write backup for {path}: {e}")


def add_working_days(start_date: dt.date, days: int) -> dt.date:
    """Add 'days' working days to start_date, skipping weekends and holidays."""
    current = start_date
    remaining = days
    while remaining > 0:
        current += dt.timedelta(days=1)
        if current.weekday() < 5 and not is_holiday(current):
            remaining -= 1
    return current


def enforce_flow_style_in_env(env_root: dict) -> None:
    """Wraps lists/maps so PyYAML prints them in inline flow style."""
    env = env_root.get("environment") or env_root

    # workflow_list -> phases -> operations: work_hours -> InlineList
    for wf in env.get("workflow_list") or []:
        for ph in wf.get("phase_list") or []:
            for op in ph.get("operation_list") or []:
                wh = op.get("work_hours")
                if isinstance(wh, list) and not isinstance(wh, InlineList):
                    op["work_hours"] = InlineList(wh)

    # worker_list: skill_map -> InlineDict
    for w in env.get("worker_list") or []:
        sm = w.get("skill_map")
        if isinstance(sm, dict) and not isinstance(sm, InlineDict):
            w["skill_map"] = InlineDict(sm)


# ---------------- Env: workers extension ----------------

def idx_to_worker_name(idx: int) -> str:
    # same as data_generation: "AA", "AB", ...
    return chr(ord("A") + idx // 26) + chr(ord("A") + idx % 26)


def extend_workers_if_needed(env_root: Dict[str, Any]) -> Tuple[int, int]:
    """
    Make sure environment.worker_list has at least WORKER_NUM workers.
    Returns (before_count, added_count).
    """
    env = env_root.get("environment") or env_root
    worker_list = env.get("worker_list") or []
    before_n = len(worker_list)
    target_n = int(cfg.WORKER_NUM)

    if before_n >= target_n:
        return before_n, 0

    # All operations from workflow_list[0]
    wf_list = env.get("workflow_list") or []
    if not wf_list:
        raise SystemExit("EnvConfig has no workflow_list")
    workflow = wf_list[0]

    ops: List[str] = []
    for ph in workflow.get("phase_list") or []:
        for op in ph.get("operation_list") or []:
            op_id = str(op.get("id"))
            if op_id:
                ops.append(op_id)
    if not ops:
        raise SystemExit("EnvConfig workflow has no operations")

    # worker companies
    company_list = env.get("worker_company_list") or []
    if not company_list:
        raise SystemExit("EnvConfig has no worker_company_list")

    added = 0
    skill_min, skill_max = 3, 6

    # A separate RNG for workers is fine; no need to be per-day.
    rng_env = random.Random(cfg.ENV_SEED)

    for i in range(before_n, target_n):
        wid = f"w{i + 1}"
        wname = idx_to_worker_name(i)

        # choose skills
        skill_num = rng_env.randint(skill_min, skill_max)
        skill_indices = sorted(rng_env.sample(range(len(ops)), skill_num))
        skill_ids = [ops[k] for k in skill_indices]
        skill_levels = rng_env.choices(
            base.skill_level_list,
            base.skill_level_weights,
            k=len(skill_ids),
        )

        company = rng_env.choice(company_list)
        is_manager = rng_env.choices(
            [True, False],
            [base.manager_rate, 1.0 - base.manager_rate],
            k=1,
        )[0]

        skill_map = {op_id: lvl for op_id, lvl in zip(skill_ids, skill_levels)}

        worker_list.append(
            {
                "id": wid,
                "name": wname,
                "worker_company": company.get("id"),
                "is_manager": is_manager,
                "skill_map": InlineDict(skill_map),
                "fab_suitability_map": [],
                "unavailable_dates": [],
            }
        )
        added += 1

    env["worker_list"] = worker_list
    return before_n, added


# ---------------- Schedule: modules extension ----------------

def collect_fab_ids(env_root: Dict[str, Any]) -> List[str]:
    env = env_root.get("environment") or env_root
    fab_list = env.get("fab_list") or []
    ids = [str(f.get("id")) for f in fab_list if f.get("id")]
    if not ids:
        raise SystemExit("EnvConfig has no fab_list ids")
    return ids


def pick_workflow(env_root: Dict[str, Any]) -> Dict[str, Any]:
    env = env_root.get("environment") or env_root
    wf_list = env.get("workflow_list") or []
    if not wf_list:
        raise SystemExit("EnvConfig has no workflow_list")
    return wf_list[0]


def to_operation_task_dict(eq_id: str, op_dict: Dict[str, Any], workload_days: int) -> Dict[str, Any]:
    op_id = str(op_dict.get("id"))
    op_name = str(op_dict.get("name") or op_id)
    return {
        "id": f"{eq_id}{op_id}",
        "name": op_name,
        "operation": op_id,
        "workload_days": int(workload_days),
    }


def build_phase_task(
    eq_id: str,
    phase_dict: Dict[str, Any],
    op_worklengths: List[int],
    start_day: dt.date,
    end_day: dt.date,
) -> Dict[str, Any]:
    phase_id = str(phase_dict.get("id"))
    phase_name = str(phase_dict.get("name") or phase_id)
    ops = phase_dict.get("operation_list") or []

    op_task_list: List[Dict[str, Any]] = []
    for op, wl in zip(ops, op_worklengths):
        op_task_list.append(to_operation_task_dict(eq_id, op, wl))

    return {
        "id": f"{eq_id}{phase_id}",
        "name": phase_name,
        "phase": phase_id,
        "start_date": ymd(start_day),
        "end_date": ymd(end_day),
        "operation_task_list": op_task_list,
    }


def build_one_module(
    eq_index: int,
    workflow: Dict[str, Any],
    fab_id: str,
    start_day: dt.date,
    worklength: List[Tuple[int, List[int]]],
) -> Tuple[Dict[str, Any], dt.date]:
    """
    Build one equipment module e{eq_index+1}.

    NEW PATTERN:
      - All phases share the SAME start_date (module_start).
      - Each phase has its own end_date.
      - end_date is cumulative in WORKING DAYS from module_start.
    """
    eq_id = f"e{eq_index + 1}"
    name = f"SU {1000 + eq_index + 1}A"

    phase_task_list: List[Dict[str, Any]] = []

    phase_list = workflow.get("phase_list") or []
    if len(phase_list) != len(worklength):
        raise SystemExit(
            f"worklength length {len(worklength)} does not match number of phases {len(phase_list)}"
        )

    # module_start: shift to next non-holiday if needed
    module_start = start_day
    while is_holiday(module_start):
        module_start += dt.timedelta(days=1)

    cumulative_days = 0
    final_end = module_start

    for phase_dict, (phase_days, op_wls) in zip(phase_list, worklength):
        phase_days = int(phase_days)

        # cumulative working days from module_start
        cumulative_days += phase_days
        phase_end = add_working_days(module_start, cumulative_days)

        phase_task = build_phase_task(
            eq_id=eq_id,
            phase_dict=phase_dict,
            op_worklengths=op_wls,
            start_day=module_start,  # <<< all phases same start
            end_day=phase_end,
        )
        phase_task_list.append(phase_task)

        final_end = phase_end

    eq_dict = {
        "id": eq_id,
        "name": name,
        "workflow": str(workflow.get("id") or "workflow"),
        "fab": fab_id,
        "phase_task_list": phase_task_list,
    }

    return eq_dict, final_end



def create_new_modules(
    workflow: Dict[str, Any],
    fab_ids: List[str],
    start_index: int,
    num_to_add: int,
    start_day: dt.date,
) -> Tuple[List[Dict[str, Any]], dt.date]:
    """
    Build extra modules for *this* update only.

    The caller already decided how many modules to add (num_to_add).
    Here we just:
      - place them at or after start_day,
      - skip weekends/holidays,
      - if multiple modules are added in one run, they are spaced
        1 working day apart: D, D+1w, D+2w, ...
    """
    if num_to_add <= 0:
        return [], start_day

    (worklength_list, worklength_weights) = create_worklength_list()

    rng_mod = random.Random(
        int(getattr(cfg, "MODULE_SEED", 0))
        + int(getattr(cfg, "MODULE_SEED_OFFSET", 0))
    )

    modules: List[Dict[str, Any]] = []
    current_start = start_day
    last_end = start_day

    for offset in range(num_to_add):
        worklength = rng_mod.choices(worklength_list, worklength_weights, k=1)[0]
        fab_id = rng_mod.choice(fab_ids)

        eq_idx = start_index + offset
        module_dict, eq_end = build_one_module(
            eq_index=eq_idx,
            workflow=workflow,
            fab_id=fab_id,
            start_day=current_start,
            worklength=worklength,
        )
        modules.append(module_dict)
        if eq_end > last_end:
            last_end = eq_end

        # next module (if any) starts next working day
        current_start = add_working_days(current_start, 1)


    return modules, last_end



# ---------------- assignments + schedule helpers ----------------

def update_assignments(schedule_root: Dict[str, Any], cutoff: dt.date) -> Tuple[int, int]:
    """
    Set plan_flexibility = Fixed/Flexible based on cutoff date.
    """
    sched = schedule_root.get("schedule") or schedule_root
    assignments = sched.get("assignment_list") or []
    changed_fixed = 0
    total = 0

    for a in assignments:
        total += 1
        sd_raw = a.get("start_date")
        if not sd_raw:
            # fallback from work_date_list if missing
            wd_key = "work_date_lsit" if "work_date_lsit" in a else "work_date_list"
            if a.get(wd_key):
                sd_raw = a[wd_key][0].get("date")

        if not sd_raw:
            continue

        try:
            sd = parse_date(sd_raw)
        except Exception:
            continue

        if sd < cutoff:
            if a.get("plan_flexibility") != "Fixed":
                a["plan_flexibility"] = "Fixed"
                changed_fixed += 1
        else:
            if a.get("plan_flexibility") != "Flexible":
                a["plan_flexibility"] = "Flexible"

    sched["assignment_list"] = assignments
    if "schedule" in schedule_root:
        schedule_root["schedule"] = sched
    return total, changed_fixed






def collect_existing_modules(schedule_root: Dict[str, Any]) -> Tuple[List[Dict[str, Any]], Optional[dt.date], Optional[dt.date]]:
    """
    Returns (modules_list_sorted, last_module_start, last_module_end).

    last_module_start = max over modules of MIN(phase.start_date)  (p1 start)
    last_module_end   = max over modules of MAX(phase.end_date)    (last phase end)
    """
    sched = schedule_root.get("schedule") or schedule_root
    wf_list = sched.get("workflow_task_list") or []
    modules: List[Dict[str, Any]] = []
    last_start: Optional[dt.date] = None
    last_end: Optional[dt.date] = None

    for mod in wf_list:
        mid = str(mod.get("id") or "")
        if not (mid.startswith("e") and mid[1:].isdigit()):
            continue
        modules.append(mod)

        mod_start: Optional[dt.date] = None
        mod_end: Optional[dt.date] = None

        for ph in mod.get("phase_task_list") or []:
            try:
                s = parse_date(ph.get("start_date"))
                e = parse_date(ph.get("end_date"))
            except Exception:
                continue
            if mod_start is None or s < mod_start:
                mod_start = s
            if mod_end is None or e > mod_end:
                mod_end = e

        if mod_start is not None:
            if last_start is None or mod_start > last_start:
                last_start = mod_start
        if mod_end is not None:
            if last_end is None or mod_end > last_end:
                last_end = mod_end

    modules.sort(key=lambda m: int(str(m.get("id"))[1:] if str(m.get("id")).startswith("e") else 0))
    return modules, last_start, last_end


# ---------------- main ----------------

def main():
    cutoff = parse_date(cfg.CUTOFF_DATE_STR)

    env_path = Path(cfg.ENV_PATH)
    sched_in = Path(cfg.SCHEDULE_IN_PATH)
    sched_out = Path(cfg.SCHEDULE_OUT_PATH)

    # Load env & schedule
    env_root = load_yaml(env_path)
    sched_root = load_yaml(sched_in)

    sched = sched_root.get("schedule") or sched_root
    assignments = sched.get("assignment_list") or []
    initial_schedule = (len(assignments) == 0)

    set_up_yaml_inline()

    # 1) extend workers (only if needed)
    before_workers, added_workers = extend_workers_if_needed(env_root)
    after_workers = before_workers + added_workers

    # 2) update assignments
    incremental_mode = (not initial_schedule)  # we already solved once if assignments exist
    total_assign, changed_fixed = update_assignments(sched_root, cutoff)

    # 2.5) decide if this is the "initial day"
    sched_for_assign = sched_root.get("schedule") or sched_root
    assignment_list_now = sched_for_assign.get("assignment_list") or []
    has_assignments = len(assignment_list_now) > 0

    # 3) extend modules *incrementally*
    modules, last_start, last_end = collect_existing_modules(sched_root)
    current_n = len(modules)
    target_n = int(cfg.EQ_NUM)
    remaining = max(0, target_n - current_n)

    workflow = pick_workflow(env_root)
    fab_ids = collect_fab_ids(env_root)

    new_modules: List[Dict[str, Any]] = []
    new_last_end = last_end

    # --- per-day simulation info from daily_run.py ---
    current_sim_day_str = getattr(cfg, "CURRENT_SIM_DAY_STR", None)
    if current_sim_day_str:
        current_sim_day = parse_date(current_sim_day_str)
    else:
        current_sim_day = cutoff  # fallback

    # Initial-day rule:
    # - If Schedule.yaml has NO assignments (fresh initial run)
    # - AND current day == cutoff (plan_start),
    #   then do NOT add new modules yet (just solve the initial 10, etc.)
    if (not has_assignments) and (current_sim_day == cutoff):
        allow_extend_today = False
    else:
        allow_extend_today = True

    # How many modules to add on this *evaluation* (block of X working days)
    modules_today = 0
    if remaining > 0 and allow_extend_today:
        # per-working-day rate from config_base
        eq_per_day = float(getattr(base, "EQ_PER_DAYS", 0.0))
        eq_sigma_day = float(getattr(base, "EQ_PER_DAYS_SIGMA", 0.0))

        # X = how many working days we bundle into one evaluation
        eval_days = max(1, int(getattr(cfg, "EQ_EVAL_DAYS", 1)))

        # Block rate ≈ per-day rate × X days
        eq_per = eq_per_day * eval_days
        eq_sigma = eq_sigma_day * eval_days

        base_seed = int(getattr(cfg, "MODULE_SEED", 0))
        offset = int(getattr(cfg, "MODULE_SEED_OFFSET", 0))
        rng_day = random.Random(base_seed * 10007 + offset)

        # Sample "demand" for this whole block of eval_days
        demand = max(0.0, rng_day.gauss(eq_per, eq_sigma))

        if demand <= 0.0:
            modules_today = 0
        elif demand < 1.0:
            # For fractional demand, add 1 module with probability = demand
            if rng_day.random() < demand:
                modules_today = 1
        else:
            # Integer part + Bernoulli on fractional part
            whole = int(demand)
            frac = demand - whole
            modules_today = whole
            if rng_day.random() < frac:
                modules_today += 1

        # Do not exceed total remaining
        if modules_today > remaining:
            modules_today = remaining


    # Actually create this block's modules (0,1,2,...)
    if modules_today > 0:
        # 11th module start = (last module's start) + 1 working day
        if last_start is not None:
            start_day = add_working_days(last_start, 1)
        else:
            # If somehow no modules exist yet, fall back to cutoff
            start_day = cutoff

        new_modules, new_last_end = create_new_modules(
            workflow=workflow,
            fab_ids=fab_ids,
            start_index=current_n,
            num_to_add=modules_today,
            start_day=start_day,
        )

        # Append to workflow_task_list
        sched = sched_root.get("schedule") or sched_root
        wf_list = sched.get("workflow_task_list") or []
        wf_list.extend(new_modules)
        sched["workflow_task_list"] = wf_list
        if "schedule" in sched_root:
            sched_root["schedule"] = sched





    # 4) update plan_range
    sched = sched_root.get("schedule") or sched_root
    plan_range = sched.get("plan_range") or {}

    # start_date: cutoff (daily_run keeps cutoff = plan_start when lookback disabled)
    # start_date:
    if initial_schedule:
        # first-ever run → keep whatever is already there, or fallback to cutoff
        if not plan_range.get("start_date"):
            plan_range["start_date"] = ymd(cutoff)
    else:
        # incremental: start_date = "today" (CURRENT_SIM_DAY_STR)
        plan_range["start_date"] = ymd(current_sim_day)

    # end_date: max(existing end, new end, old end) + buffer
    current_end = None
    if plan_range.get("end_date"):
        try:
            current_end = parse_date(plan_range["end_date"])
        except Exception:
            current_end = None

    # last_end = latest end of existing modules before this run
    # new_last_end = latest end considering modules we just added (if any)
    end_candidates = [d for d in (current_end, last_end, new_last_end) if d is not None]
    if end_candidates:
        end_base = max(end_candidates)
    else:
        end_base = cutoff

    end_final = end_base + dt.timedelta(days=int(getattr(cfg, "PLAN_RANGE_EXTRA_DAYS", 0)))
    plan_range["end_date"] = ymd(end_final)
    sched["plan_range"] = plan_range
    if "schedule" in sched_root:
        sched_root["schedule"] = sched

    # write env & schedule back (with backup)
    enforce_flow_style_in_env(env_root)
    backup_file(env_path)
    with env_path.open("w", encoding="utf-8") as f:
        yaml.safe_dump(env_root, f, sort_keys=False, allow_unicode=True)

    if sched_out == sched_in:
        backup_file(sched_in)
    with sched_out.open("w", encoding="utf-8") as f:
        yaml.safe_dump(sched_root, f, sort_keys=False, allow_unicode=True)

    # logs
    modules_added = len(new_modules)
    print(f"Assignments: {total_assign} processed, {changed_fixed} set to Fixed.")
    print(f"Workers: before={before_workers}, added={added_workers}, after={after_workers}, target={cfg.WORKER_NUM}.")
    print(f"Modules: existing={current_n}, added={modules_added}, target={target_n}.")
    print(f"New plan_range: {plan_range['start_date']} .. {plan_range['end_date']}")



if __name__ == "__main__":
    main()
