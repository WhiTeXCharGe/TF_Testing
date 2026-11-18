#!/usr/bin/env python3
"""
start_schedule.py  --  01_StartSchedule: part 1 of the weekly flow (before the solver) in one run.

Runs tools 2-2-3, 2-4, 2-5 and 2-6 of weekly_planning for the week of today, from two files: the SU_Others workbook and the seiban list
workbook.  Without options a window opens where the two files are chosen.  With both files on the command line it runs without a
window:

    python start_schedule.py --su-others <SU_Others.xlsm> --seiban-list <seiban list.xlsm> [--no-submit] [--yes]

Settings (folders, solver address, sheet names) are in plan_assist.config.txt next to this program.
"""
from __future__ import annotations

import argparse
import sys
from datetime import date, datetime
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
from pa import settings as st   # noqa: E402
from pa.start import run_start  # noqa: E402


def parse_day(text):
    for fmt in ("%Y%m%d", "%Y/%m/%d", "%Y-%m-%d"):
        try:
            return datetime.strptime(text, fmt).date()
        except ValueError:
            continue
    raise argparse.ArgumentTypeError("must be a date such as 20261117 or 2026/11/17")


def main(argv=None):
    ap = argparse.ArgumentParser(description="01_StartSchedule: run 2-2-3, 2-4, 2-5 and 2-6 for this week.")
    ap.add_argument("--su-others", default=None, help="the SU_Others workbook of the week (.xlsm / .xlsx)")
    ap.add_argument("--seiban-list", default=None, help="the seiban list workbook of the week (.xlsm / .xlsx)")
    ap.add_argument("--date", type=parse_day, default=None, help="the run date, YYYYMMDD (default: today); it names the week folder")
    ap.add_argument("--config", default=None, help=f"settings file (default: {st.CONFIG_NAME} next to this program)")
    ap.add_argument("--no-submit", action="store_true", help="stop after the 03 files are made; nothing is sent to the solver")
    ap.add_argument("--yes", action="store_true", help="run a week again even if it was already sent to the solver")
    ap.add_argument("--no-ui", action="store_true", help="never open the window (an error when a file is missing)")
    args = ap.parse_args(argv)

    try:
        settings = st.load(args.config)
    except st.SettingsError as e:
        print(f"Error: {e}", file=sys.stderr)
        return 1
    day = args.date or date.today()
    submit = False if args.no_submit else None

    if args.su_others and args.seiban_list or args.no_ui:
        console = sys.stdout           # the tools run with sys.stdout redirected, so the log must keep the real stream
        if hasattr(console, "reconfigure"):
            console.reconfigure(errors="replace")      # a Windows console cannot show every character of a worker name

        def show(line):
            console.write(line + "\n")
            console.flush()

        result = run_start(settings, args.su_others, args.seiban_list, day, log=show, submit=submit, overwrite=args.yes)
        return 0 if result.ok else 1
    from pa.ui import run_window     # tkinter is only needed for the window
    run_window(settings, day, args.su_others or "", args.seiban_list or "", submit=submit)
    return 0


if __name__ == "__main__":
    sys.exit(main())
