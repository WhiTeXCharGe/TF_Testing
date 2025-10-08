# python make_schedule_mock.py --in Schedule.yaml --out Schedule_mock.yaml
import yaml
from argparse import ArgumentParser
from datetime import date, datetime, timedelta

def _d(s):
    if isinstance(s, date):
        return s
    s = str(s).replace("-", "/")
    return datetime.strptime(s, "%Y/%m/%d").date()

def _fmt(d: date) -> str:
    return d.strftime("%Y/%m/%d")

def _idx(plan_start: date, d: date) -> int:
    """1-based index of d within the (original) plan range."""
    return (d - plan_start).days + 1

def _shift_index(n: int) -> int:
    """Business-day mapping: n -> n + 2*floor((n-1)/5)."""
    return n + 2 * ((n - 1) // 5)

def _shift_date(plan_start: date, d: date) -> date:
    """Shift any date to business-day timeline anchored at plan_start."""
    n = _idx(plan_start, d)
    n2 = _shift_index(n)
    return plan_start + timedelta(days=n2 - 1)

def shift_schedule(in_path: str, out_path: str):
    with open(in_path, "r", encoding="utf-8") as f:
        root = yaml.safe_load(f)

    s = root.get("schedule", root)
    plan_start = _d(s["plan_range"]["start_date"])
    plan_end   = _d(s["plan_range"]["end_date"])

    # shift plan range
    s["plan_range"]["start_date"] = _fmt(plan_start)  # day 1 stays day 1
    s["plan_range"]["end_date"]   = _fmt(_shift_date(plan_start, plan_end))

    # shift all workflow phase windows and their operation tasks (only dates)
    for wf in s.get("workflow_task_list", []):
        for ph in wf.get("phase_task_list", []):
            ph["start_date"] = _fmt(_shift_date(plan_start, _d(ph["start_date"])))
            ph["end_date"]   = _fmt(_shift_date(plan_start, _d(ph["end_date"])))
            # (workload_days is unchanged)

    # shift assignments (block dates + per-day dates)
    for a in s.get("assignment_list", []):
        if "start_date" in a and a["start_date"]:
            a["start_date"] = _fmt(_shift_date(plan_start, _d(a["start_date"])))
        if "end_date" in a and a["end_date"]:
            a["end_date"] = _fmt(_shift_date(plan_start, _d(a["end_date"])))

        wd_key = "work_date_lsit" if "work_date_lsit" in a else "work_date_list"
        new_wds = []
        for di in a.get(wd_key, []):
            sd = _shift_date(plan_start, _d(di["date"]))
            # mapping guarantees Mon–Fri only, but be defensive:
            if sd.weekday() < 5:  # 0..4 = Mon..Fri
                new_wds.append({"date": _fmt(sd), "hour": int(di["hour"])})
        a[wd_key] = new_wds

    # write
    out = {"schedule": s} if "schedule" in root else s
    with open(out_path, "w", encoding="utf-8") as f:
        yaml.safe_dump(out, f, sort_keys=False, allow_unicode=True)
    print(f"Wrote mock: {out_path}")

def main():
    ap = ArgumentParser()
    ap.add_argument("--in",  dest="inp",  default="Schedule.yaml")
    ap.add_argument("--out", dest="outp", default="Schedule_mock.yaml")
    args = ap.parse_args()
    shift_schedule(args.inp, args.outp)

if __name__ == "__main__":
    main()
