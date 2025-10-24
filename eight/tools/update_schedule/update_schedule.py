# update_schedule.py
# Usage:
#   python update_schedule.py 2025/10/01 --in Schedule.yaml --out Schedule.yaml
#
# Notes:
# - Accepts dates as YYYY/MM/DD or YYYY-MM-DD.
# - Only updates schedule.assignment_list[*].plan_flexibility.
# - Makes a .bak backup if writing in-place.

import argparse
import datetime as dt
import shutil
import sys
import yaml
from pathlib import Path

def parse_date(s: str) -> dt.date:
    s = str(s).strip().replace("-", "/")
    return dt.datetime.strptime(s, "%Y/%m/%d").date()

def main():
    ap = argparse.ArgumentParser(description="Set plan_flexibility=Fixed for assignments that start before cutoff date.")
    ap.add_argument("cutoff", help="Cutoff date (YYYY/MM/DD or YYYY-MM-DD). Assignments with start_date < cutoff become Fixed.")
    ap.add_argument("--in", dest="inp", default="Schedule.yaml", help="Input Schedule.yaml path")
    ap.add_argument("--out", dest="out", default=None, help="Output path (default: overwrite input)")
    args = ap.parse_args()

    cutoff = parse_date(args.cutoff)
    in_path = Path(args.inp)
    out_path = Path(args.out) if args.out else in_path

    if not in_path.exists():
        print(f"Input file not found: {in_path}", file=sys.stderr)
        sys.exit(1)

    with in_path.open("r", encoding="utf-8") as f:
        data = yaml.safe_load(f) or {}

    # Support both top-level and nested 'schedule'
    root = data.get("schedule", data)

    changed = 0
    total = 0

    assignments = root.get("assignment_list", [])
    for a in assignments:
        total += 1
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
                        changed += 1
                else:
                    # keep or normalize to Flexible for not-started items
                    if a.get("plan_flexibility") != "Flexible":
                        a["plan_flexibility"] = "Flexible"
            else:
                # No start date info; leave as-is
                pass
        except Exception:
            # If date parsing fails, leave entry unchanged
            pass

    # If we were working on nested 'schedule', put it back
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

    # Dump back
    with out_path.open("w", encoding="utf-8") as f:
        yaml.safe_dump(data, f, sort_keys=False, allow_unicode=True)

    print(f"Processed {total} assignments. Updated to Fixed: {changed}. Wrote: {out_path}")

if __name__ == "__main__":
    main()
