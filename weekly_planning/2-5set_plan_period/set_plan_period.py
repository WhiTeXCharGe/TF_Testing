#!/usr/bin/env python3
"""
set_plan_period.py  --  weekly planning tool "2-5": decide the plan period of the solver input.

Input   N/02_Schedule.yaml, N/02_EnvConfig.yaml   (after the planner added the new modules, step 2-4)
Output  N/03_Schedule.yaml, N/03_EnvConfig.yaml

  start_date = the later of (a) the earliest day any module to be optimised may start (earliest phase start_date) and
                            (b) today + K days
  end_date   = the latest due date of the modules to be optimised (latest phase end_date) + L days
  K and L are constants (--k, --l) to be tuned for the operation.
  Assignments earlier than start_date (only those) become Fixed; a row that crosses start_date is split in two rows.

The modules to be optimised are the workflow tasks of the schedule: in the weekly flow the Schedule of step 2-3 only
contains the flexible modules (everything else is a misc task), plus the modules added in step 2-4.
03_EnvConfig.yaml is a byte-for-byte copy of 02_EnvConfig.yaml (the period does not change it).
"""
from __future__ import annotations

import argparse
import copy
import shutil
import sys
from datetime import date, datetime, timedelta
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent / "2-2-3prepare_schedule"))
import prepare_schedule as ps   # noqa: E402  (ymd, load_yaml, dump_yaml)

DEFAULT_K = 7     # days of the plan that are frozen: the run is on the weekend, the planner reviews the result in the week
DEFAULT_L = 30    # days of slack after the latest due date


def _day(s):
    return datetime.strptime(s, "%Y/%m/%d").date()


def _plus(s, n):
    return (_day(s) + timedelta(days=n)).strftime("%Y/%m/%d")


def plan_period(schedule_doc, today, k=DEFAULT_K, l=DEFAULT_L):
    """
    schedule_doc: parsed 02_Schedule.yaml ({"schedule": {...}}).  today: 'YYYY/MM/DD'.
    Returns (new schedule doc, report [str], info dict).  The input is not modified.
    """
    doc = copy.deepcopy(schedule_doc)
    sch = doc["schedule"]
    report = []
    starts, ends = [], []
    for task in sch.get("workflow_task_list") or []:
        for ph in task.get("phase_task_list") or []:
            s, e = ps.ymd(ph.get("start_date")), ps.ymd(ph.get("end_date"))
            if s:
                starts.append(s)
            if e:
                ends.append(e)
    if not starts or not ends:
        raise ValueError("the schedule has no workflow task with phase start_date / end_date: nothing to optimise")
    earliest, latest = min(starts), max(ends)
    start = max(earliest, _plus(today, k))
    end = _plus(latest, l)
    if end < start:
        report.append(f"WARN: latest due date + L ({end}) is before start_date ({start}): end_date set to start_date + L")
        end = _plus(start, l)
    old = sch.get("plan_range") or {}
    report.append(f"earliest start of the modules {earliest}, today + K ({k}) {_plus(today, k)} -> start_date {start}")
    report.append(f"latest due date of the modules {latest}, + L ({l}) -> end_date {end}")
    report.append(f"plan_range {old.get('start_date')}..{old.get('end_date')} -> {start}..{end}")
    sch["plan_range"] = {"start_date": start, "end_date": end}

    # assignments before start_date are Fixed (only those days)
    rows, fixed_days, split_rows, whole_rows = [], 0, 0, 0
    for a in sch.get("assignment_list") or []:
        flex = str(a.get("plan_flexibility") or "Flexible")
        items = a.get("work_date_list") or []
        before = [it for it in items if (ps.ymd(it.get("date")) or "9999/99/99") < start]
        if flex == "Fixed" or not before:
            rows.append(a)
            continue
        after = [it for it in items if it not in before]
        fixed_row = copy.deepcopy(a)
        fixed_row["plan_flexibility"] = "Fixed"
        fixed_row["work_date_list"] = before
        fixed_row["start_date"] = min(ps.ymd(it["date"]) for it in before)
        fixed_row["end_date"] = max(ps.ymd(it["date"]) for it in before)
        rows.append(fixed_row)
        fixed_days += len(before)
        if after:
            rest = copy.deepcopy(a)
            rest["work_date_list"] = after
            rest["start_date"] = min(ps.ymd(it["date"]) for it in after)
            rest["end_date"] = max(ps.ymd(it["date"]) for it in after)
            rows.append(rest)
            split_rows += 1
        else:
            whole_rows += 1
    sch["assignment_list"] = rows
    report.append(f"ASSIGNMENTS before {start}: {fixed_days} worker-days made Fixed "
                  f"({whole_rows} rows completely, {split_rows} rows split at the start date)")

    beyond = sum(1 for a in rows for it in a.get("work_date_list") or [] if (ps.ymd(it.get("date")) or "") > end)
    if beyond:
        report.append(f"WARN: {beyond} worker-days of existing assignments lie after end_date {end}")
    trips_beyond = sum(1 for e in sch.get("trip_list") or [] for t in e.get("trips") or [] if (ps.ymd(t.get("end_date")) or "") > end)
    if trips_beyond:
        report.append(f"WARN: {trips_beyond} trips end after end_date {end}")
    return doc, report, {"start_date": start, "end_date": end, "fixed_days": fixed_days, "split_rows": split_rows}


def main(argv=None):
    ap = argparse.ArgumentParser(description="Weekly planning 2-5: decide the plan period of the solver input.")
    ap.add_argument("--schedule", required=True, help="N/02_Schedule.yaml")
    ap.add_argument("--env", required=True, help="N/02_EnvConfig.yaml")
    ap.add_argument("--out-dir", required=True, help="folder for 03_Schedule.yaml and 03_EnvConfig.yaml")
    ap.add_argument("--today", default=None, help="YYYY/MM/DD (default: today)")
    ap.add_argument("--k", type=int, default=DEFAULT_K, help=f"start_date is at least today + K days (default {DEFAULT_K})")
    ap.add_argument("--l", type=int, default=DEFAULT_L, help=f"end_date = latest due date + L days (default {DEFAULT_L})")
    ap.add_argument("--report", action="store_true", help="also write 03_plan_period_report.txt")
    args = ap.parse_args(argv)

    today = ps.ymd(args.today) if args.today else date.today().strftime("%Y/%m/%d")
    if today is None:
        ap.error("--today must look like 2026/10/01")
    if args.k < 0 or args.l < 0:
        ap.error("--k and --l must not be negative")
    doc, report, info = plan_period(ps.load_yaml(args.schedule), today, args.k, args.l)
    od = Path(args.out_dir)
    od.mkdir(parents=True, exist_ok=True)
    ps.dump_yaml(od / "03_Schedule.yaml", doc)
    shutil.copyfile(args.env, od / "03_EnvConfig.yaml")
    print(f"set_plan_period (today = {today}, K = {args.k}, L = {args.l})")
    print("\n".join(report))
    names = ["03_Schedule.yaml", "03_EnvConfig.yaml"]
    if args.report:
        (od / "03_plan_period_report.txt").write_text(f"today = {today}, K = {args.k}, L = {args.l}\n" + "\n".join(report) + "\n",
                                                      encoding="utf-8")
        names.append("03_plan_period_report.txt")
    print(f"written to {od}: " + ", ".join(names))
    return 0


if __name__ == "__main__":
    sys.exit(main())