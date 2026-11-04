# fix_edited_modules.py
# ---------------------------------------------------------------------
# Step 2(★) of the weekly loop: "人が編集した製番をFixedにする".
#
# Compares last week's Timefold result (EnvConfig_(N-1)_output.yaml +
# Schedule_(N-1)_output.yaml) with this week's conversion of the hand-edited
# SU_Others.xlsx (EnvConfig_N.yaml + Schedule_N.yaml, from decoder7.py).
# Every 製番 (workflow task) whose assignments differ is treated as edited
# by a person, and the WHOLE module is frozen in the new files:
#   - every assignment on that module's operation tasks -> plan_flexibility: Fixed
#   - each of its operation tasks' workload_hours -> the hours already assigned
#     to it, so the solver has nothing left to add.
# Output: EnvConfig_N_input.yaml + Schedule_N_input.yaml (+ a log).
#
# Matching is by NAME, not id: decoder7 renumbers e{n} / w{nnn} on every run,
# so ids are not stable between weeks. Module = workflow task `name` (製番),
# worker = EnvConfig worker `name`.
#
# What counts as "changed" (default): the set of (worker name, date) pairs on
# the module differs. That is all SU_Others can express — a cell is a worker,
# a day and a 製番 — so hours and Mech/Elec/QC operation are ignored by
# default (decoder7 derives those itself, always 10h, op from role).
# --strict also compares operation (p2o1/p2o2/p3o1/p4o1) and hours.
#
# Only the plan_flexibility / workload_hours lines of the new Schedule are
# rewritten; everything else is copied byte-for-byte. EnvConfig is copied
# unchanged (it is only read for worker id -> name).
#
# Usage:
#   python fix_edited_modules.py \
#       --prev-env EnvConfig_7_output.yaml --prev-schedule Schedule_7_output.yaml \
#       --new-env  EnvConfig_8.yaml        --new-schedule  Schedule_8.yaml
#   (writes EnvConfig_8_input.yaml, Schedule_8_input.yaml, FixLog_8.txt next to
#    the new files; override with --out-env / --out-schedule / --log-out)
#
# python fix_edited_modules.py --prev-env "EnvConfig_saved_before_backup.yaml" --prev-schedule "Schedule_saved_before_backup.yaml" --new-env "EnvConfig_saved_before_backup2.yaml" --new-schedule "Schedule_saved_before_backup2.yaml"
# ---------------------------------------------------------------------

import argparse
import os
import re
import shutil
import sys
from collections import defaultdict

import yaml


def load_yaml(path):
    with open(path, encoding="utf-8") as f:
        return yaml.safe_load(f) or {}


def norm_name(s):
    # Full-width spaces / doubled spaces differ between Excel edits.
    return re.sub(r"\s+", " ", str(s or "").replace("　", " ")).strip()


def norm_date(d):
    return str(d).strip().replace("-", "/")


def worker_names(env):
    workers = (env.get("environment") or {}).get("worker_list") or []
    return {w["id"]: norm_name(w.get("name", w["id"])) for w in workers}


def module_index(sched):
    """op_task_id -> (module name, operation), plus module name -> [op_task ids]."""
    op_to_module = {}
    module_ops = defaultdict(list)
    for wt in (sched.get("schedule") or {}).get("workflow_task_list") or []:
        mname = norm_name(wt.get("name", wt.get("id")))
        for pt in wt.get("phase_task_list") or []:
            for ot in pt.get("operation_task_list") or []:
                op_to_module[ot["id"]] = (mname, ot.get("operation", ""))
                module_ops[mname].append(ot["id"])
    return op_to_module, module_ops


def module_signatures(env, sched, strict):
    """module name -> set of comparable assignment facts."""
    names = worker_names(env)
    op_to_module, module_ops = module_index(sched)
    sigs = {m: set() for m in module_ops}
    for a in (sched.get("schedule") or {}).get("assignment_list") or []:
        hit = op_to_module.get(a.get("operation_task"))
        if not hit:
            continue  # misc / personal-business task
        mname, operation = hit
        wname = names.get(a.get("worker"), norm_name(a.get("worker")))
        for wd in a.get("work_date_list") or []:
            if strict:
                sigs[mname].add((wname, norm_date(wd["date"]), operation, float(wd.get("hour", 0) or 0)))
            else:
                sigs[mname].add((wname, norm_date(wd["date"])))
    return sigs


def describe_diff(prev, new, limit=8):
    lines = []
    for label, items in (("-", sorted(prev - new)), ("+", sorted(new - prev))):
        for it in items[:limit]:
            lines.append(f"      {label} " + " ".join(str(x) for x in it))
        if len(items) > limit:
            lines.append(f"      {label} ... ({len(items) - limit} more)")
    return lines


def rewrite_schedule(text, sched, fixed_ops, new_workload):
    """Rewrite plan_flexibility of assignments on fixed_ops and workload_hours of
    ops in new_workload, touching only those lines."""
    lines = text.splitlines(keepends=True)
    assignments = (sched.get("schedule") or {}).get("assignment_list") or []

    # workload_hours: follows the operation task's "- id:" line.
    cur_id = None
    in_assignments = False
    block_starts = []
    for i, ln in enumerate(lines):
        if re.match(r"^\s*assignment_list:\s*$", ln):
            in_assignments = True
            continue
        if in_assignments:
            if re.match(r"^\s*- worker:", ln):
                block_starts.append(i)
            elif re.match(r"^\S", ln) or re.match(r"^  \S", ln):
                in_assignments = False  # left the list (next top-level key)
            continue
        m = re.match(r"^\s*- id:\s*(\S+)\s*$", ln)
        if m:
            cur_id = m.group(1)
            continue
        m = re.match(r"^(\s*)workload_hours:", ln)
        if m and cur_id in new_workload:
            eol = "\r\n" if ln.endswith("\r\n") else "\n"
            lines[i] = f"{m.group(1)}workload_hours: {new_workload[cur_id]}{eol}"

    if len(block_starts) != len(assignments):
        sys.exit(f"ERROR: found {len(block_starts)} assignment blocks in text but "
                 f"{len(assignments)} parsed — unexpected Schedule layout, nothing written.")

    inserts = []
    changed = 0
    for n, (start, a) in enumerate(zip(block_starts, assignments)):
        if a.get("operation_task") not in fixed_ops:
            continue
        end = block_starts[n + 1] if n + 1 < len(block_starts) else len(lines)
        for i in range(start, end):
            m = re.match(r"^(\s*)plan_flexibility:\s*(\S*)", lines[i])
            if m:
                if m.group(2) != "Fixed":
                    eol = "\r\n" if lines[i].endswith("\r\n") else "\n"
                    lines[i] = f"{m.group(1)}plan_flexibility: Fixed{eol}"
                    changed += 1
                break
        else:
            inserts.append(start + 1)  # no plan_flexibility line: add one
    for i in reversed(inserts):
        lines.insert(i, "    plan_flexibility: Fixed\n")
        changed += 1
    return "".join(lines), changed


def default_out(path, suffix):
    root, ext = os.path.splitext(path)
    return f"{root}{suffix}{ext or '.yaml'}"


def main():
    ap = argparse.ArgumentParser(description="Freeze every 製番 a person edited between two weekly schedules.")
    ap.add_argument("--prev-env", required=True, help="EnvConfig_(N-1)_output.yaml")
    ap.add_argument("--prev-schedule", required=True, help="Schedule_(N-1)_output.yaml")
    ap.add_argument("--new-env", required=True, help="EnvConfig_N.yaml (converted from edited SU_Others)")
    ap.add_argument("--new-schedule", required=True, help="Schedule_N.yaml (converted from edited SU_Others)")
    ap.add_argument("--out-env", help="default: <new-env>_input.yaml")
    ap.add_argument("--out-schedule", help="default: <new-schedule>_input.yaml")
    ap.add_argument("--log-out", help="default: FixLog.txt next to --out-schedule")
    ap.add_argument("--strict", action="store_true",
                    help="also treat operation / hour differences as edits (default: worker+date only)")
    ap.add_argument("--keep-workload", action="store_true",
                    help="do not change workload_hours of fixed modules")
    args = ap.parse_args()

    out_env = args.out_env or default_out(args.new_env, "_input")
    out_sched = args.out_schedule or default_out(args.new_schedule, "_input")
    log_out = args.log_out or os.path.join(os.path.dirname(os.path.abspath(out_sched)), "FixLog.txt")

    prev_env, prev_sched = load_yaml(args.prev_env), load_yaml(args.prev_schedule)
    new_env, new_sched = load_yaml(args.new_env), load_yaml(args.new_schedule)
    with open(args.new_schedule, encoding="utf-8", newline="") as f:
        new_text = f.read()

    prev_sigs = module_signatures(prev_env, prev_sched, args.strict)
    new_sigs = module_signatures(new_env, new_sched, args.strict)
    _, new_module_ops = module_index(new_sched)

    log = [f"prev: {args.prev_env} / {args.prev_schedule}",
           f"new : {args.new_env} / {args.new_schedule}",
           f"mode: {'strict (worker, date, operation, hour)' if args.strict else '(worker, date)'}",
           ""]

    changed, only_new, only_prev, unchanged = [], [], [], []
    for m in new_sigs:
        if m not in prev_sigs:
            (only_new if new_sigs[m] else unchanged).append(m)
        elif new_sigs[m] != prev_sigs[m]:
            changed.append(m)
        else:
            unchanged.append(m)
    only_prev = [m for m in prev_sigs if m not in new_sigs]

    # A module not in last week's output but already carrying assignments
    # can only have come from a person typing it into SU_Others -> freeze it.
    to_fix = changed + only_new

    # Hours already assigned per operation task in the new schedule.
    assigned = defaultdict(float)
    for a in (new_sched.get("schedule") or {}).get("assignment_list") or []:
        for wd in a.get("work_date_list") or []:
            assigned[a.get("operation_task")] += float(wd.get("hour", 0) or 0)

    old_workload = {}
    for wt in (new_sched.get("schedule") or {}).get("workflow_task_list") or []:
        for pt in wt.get("phase_task_list") or []:
            for ot in pt.get("operation_task_list") or []:
                old_workload[ot["id"]] = ot.get("workload_hours")

    fixed_ops, new_workload = set(), {}
    for m in to_fix:
        for op in new_module_ops[m]:
            fixed_ops.add(op)
            if not args.keep_workload:
                h = assigned.get(op, 0.0)
                new_workload[op] = int(h) if h == int(h) else h

    out_text, n_flex = rewrite_schedule(new_text, new_sched, fixed_ops, new_workload)

    log.append(f"EDITED -> Fixed ({len(changed)}):")
    for m in changed:
        log.append(f"  {m}")
        log.extend(describe_diff(prev_sigs[m], new_sigs[m]))
    log.append(f"\nNOT IN PREVIOUS OUTPUT, has assignments -> Fixed ({len(only_new)}):")
    log.extend(f"  {m} ({len(new_sigs[m])} entries)" for m in only_new)
    log.append(f"\nWORKLOAD (fixed modules, workload_hours before -> after):")
    for m in to_fix:
        for op in new_module_ops[m]:
            if op in new_workload:
                warn = "   WARNING: no assignments, solver will not schedule it" \
                    if new_workload[op] == 0 and (old_workload.get(op) or 0) > 0 else ""
                log.append(f"  {m} {op}: {old_workload.get(op)} -> {new_workload[op]}{warn}")
    log.append(f"\nIN PREVIOUS OUTPUT, missing now (info only) ({len(only_prev)}):")
    log.extend(f"  {m}" for m in only_prev)
    log.append(f"\nunchanged: {len(unchanged)} modules")
    log.append(f"assignments switched to Fixed: {n_flex}")

    with open(out_sched, "w", encoding="utf-8", newline="") as f:
        f.write(out_text)
    if os.path.abspath(out_env) != os.path.abspath(args.new_env):
        shutil.copyfile(args.new_env, out_env)
    with open(log_out, "w", encoding="utf-8") as f:
        f.write("\n".join(log) + "\n")

    print(f"edited: {len(changed)}, new-with-assignments: {len(only_new)}, "
          f"unchanged: {len(unchanged)}, assignments -> Fixed: {n_flex}")
    print(f"wrote {out_env}\nwrote {out_sched}\nwrote {log_out}")


if __name__ == "__main__":
    main()
