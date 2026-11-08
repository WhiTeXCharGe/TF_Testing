#!/usr/bin/env python3
"""
extract_flexible_schedule.py  --  weekly planning tool "3-2": extract the optimised part of the solver result.

Input   N/04_Schedule.yaml, N/04_EnvConfig.yaml   (the two yaml files of the solver, downloaded in step 3-1)
Output  N/FlexibleSchedule.yaml

FlexibleSchedule.yaml keeps what the solver planned and drops everything else:
  * workflow_task_list    unchanged (the modules that were optimised)
  * assignment_list       only the assignments of those modules (normal tasks), unchanged; misc rows are left out.
                          Every row gets one more key, `cell_name`: the text of that bar in SU_Others.  It is taken from the
                          row (`cell_name`, else `description`, which is where the solver carries the text of the previous
                          week) and, for a module that has none (a new module), made from its fab and customer
                          ('{customer}{fab}_{code}_新規').  3-3 writes this text into the cells, 2-2 checks it next week.
  * trip_list             only the Flexible trips that have work of one of those modules inside; the solver also writes
                          Flexible trips without any module work (and the Fixed trips of the sheet): those are dropped
  * plan_range            unchanged
It has the same format as Schedule.yaml, so the next week's tools (prepare_schedule, write_su_others) read it like one.
04_EnvConfig.yaml is only used to check the worker ids; the next week still needs it as the "previous EnvConfig".
"""
from __future__ import annotations

import argparse
import copy
import sys
from collections import Counter
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent / "2-2-3prepare_schedule"))
import prepare_schedule as ps   # noqa: E402  (ymd, load_yaml, dump_yaml, flatten_trips)


def extract(schedule_doc, env_doc):
    """Returns (FlexibleSchedule doc, report [str], stats dict).  The input is not modified."""
    sch = schedule_doc.get("schedule", schedule_doc)
    env = env_doc.get("environment", env_doc) if isinstance(env_doc, dict) else {}
    report = []
    tasks = list(sch.get("workflow_task_list") or [])
    ops = {op["id"] for t in tasks for ph in t.get("phase_task_list") or [] for op in ph.get("operation_task_list") or []}
    module_of = {op["id"]: t.get("name") or t.get("id") for t in tasks for ph in t.get("phase_task_list") or []
                 for op in ph.get("operation_task_list") or []}

    rows = [copy.deepcopy(a) for a in sch.get("assignment_list") or [] if a.get("operation_task") in ops]
    dropped_rows = len(sch.get("assignment_list") or []) - len(rows)
    task_of = {op["id"]: t for t in tasks for ph in t.get("phase_task_list") or [] for op in ph.get("operation_task_list") or []}
    infos = {}
    named = Counter()
    for a in rows:
        task = task_of[a["operation_task"]]
        code = ps.tool_code(task.get("name")) or str(task.get("name") or task.get("id"))
        name = next((str(v).strip() for v in (a.get("cell_name"), a.get("description"))
                     if isinstance(v, str) and v.strip()), None)
        if name:
            named["from the result"] += 1
        else:
            infos.setdefault(task["id"], ps.module_name_info(task, env))
            name = ps.module_cell_name(code, infos[task["id"]])
            named["made from fab and customer"] += 1
        a["cell_name"] = name
    worker_ids = {w["id"] for w in env.get("worker_list") or []}
    unknown = sorted({a.get("worker") for a in rows if worker_ids and a.get("worker") not in worker_ids})
    if unknown:
        report.append(f"WARN: assignments of workers that are not in 04_EnvConfig: {unknown}")
    with_rows = {module_of[a["operation_task"]] for a in rows}
    for t in tasks:
        name = t.get("name") or t.get("id")
        if name not in with_rows:
            report.append(f"WARN: module {name} has no assignment in the result")

    days_by_worker = {}
    for a in rows:
        for it in a.get("work_date_list") or []:
            d = ps.ymd(it.get("date"))
            if d:
                days_by_worker.setdefault(a.get("worker"), set()).add(d)

    kept_trips, dropped = {}, Counter()
    for entry in ps.flatten_trips(sch.get("trip_list")):
        wid, trip = entry["worker"], entry["trip"]
        s, e = ps.ymd(trip.get("start_date")), ps.ymd(trip.get("end_date"))
        flex = str(trip.get("plan_flexibility") or "Flexible")
        if flex == "Fixed":
            dropped["fixed"] += 1
        elif not (s and e):
            dropped["unreadable"] += 1
            report.append(f"WARN: trip of {wid} with an unreadable date dropped: {trip}")
        elif not any(s <= d <= e for d in days_by_worker.get(wid, ())):
            dropped["no module work inside"] += 1
        else:
            kept_trips.setdefault(wid, []).append({k: v for k, v in trip.items() if k != "worker"})
    trip_list = [{"worker": w, "trips": sorted(ts, key=lambda t: (t["start_date"], t["end_date"]))}
                 for w, ts in sorted(kept_trips.items())]

    out = {"schedule": {"plan_range": sch.get("plan_range"), "workflow_task_list": tasks,
                        "misc_task_list": [], "assignment_list": rows, "trip_list": trip_list}}
    stats = {"modules": len(tasks), "assignment_rows": len(rows), "misc_or_other_rows_dropped": dropped_rows,
             "trips_kept": sum(len(e["trips"]) for e in trip_list), "trips_dropped": dict(dropped),
             "cell_name": dict(named)}
    report.insert(0, f"stats: {stats}")
    return out, report, stats


def main(argv=None):
    ap = argparse.ArgumentParser(description="Weekly planning 3-2: extract the optimised modules, assignments and trips of the result.")
    ap.add_argument("--schedule", required=True, help="N/04_Schedule.yaml (solver result)")
    ap.add_argument("--env", required=True, help="N/04_EnvConfig.yaml (solver result)")
    ap.add_argument("--out-dir", required=True, help="folder for FlexibleSchedule.yaml")
    ap.add_argument("--report", action="store_true", help="also write FlexibleSchedule_report.txt")
    args = ap.parse_args(argv)
    doc, report, _stats = extract(ps.load_yaml(args.schedule), ps.load_yaml(args.env))
    od = Path(args.out_dir)
    ps.dump_yaml(od / "FlexibleSchedule.yaml", doc)
    print("\n".join(report))
    names = ["FlexibleSchedule.yaml"]
    if args.report:
        (od / "FlexibleSchedule_report.txt").write_text("\n".join(report) + "\n", encoding="utf-8")
        names.append("FlexibleSchedule_report.txt")
    print(f"written to {od}: " + ", ".join(names))
    return 0


if __name__ == "__main__":
    sys.exit(main())
