# export_csv.py
import os
from argparse import ArgumentParser
from collections import defaultdict
from datetime import timedelta
from typing import Dict, Tuple, List

from openpyxl import Workbook
from openpyxl.styles import PatternFill, Alignment, Font
from openpyxl.utils import get_column_letter

# <<< match the minimal mock solver you’re using >>>
from employee_scheduler_mock_tf import solve_from_config

WEEKEND_FILL = PatternFill(start_color="FFEFEF", end_color="FFEFEF", fill_type="solid")
HEADER_BOLD  = Font(bold=True)
CENTER       = Alignment(horizontal="center", vertical="center", wrap_text=True)
LEFT         = Alignment(horizontal="left",   vertical="center", wrap_text=True)

def write_excel(solution, start_day, out_path: str = "schedule_mock.xlsx"):
    """
    Minimal 2-sheet export (no skills, no process deadlines, no org calendars).

    Sheet 1: "Tasks x Dates"
      - Col A: task_id (e.g., T1)
      - Col B: country (e.g., A / B)
      - Columns C..: YYYY-MM-DD dates across the horizon
      - Cell: "AA, AB (2)" -> list of employees, and headcount in parentheses
      - Weekend columns shaded

    Sheet 2: "Employees x Dates"
      - Col A: employee name
      - Columns B..: YYYY-MM-DD
      - Cell: task_id assigned that day (or empty)
      - Weekend columns shaded
    """
    wb = Workbook()

    # Collect dates from the solved solution
    dates = [d.d for d in solution.days]  # list[date]
    day_id_to_date = {i: d for i, d in enumerate(dates)}

    # Build quick maps from the solved assignments
    # (task_id, day_id) -> [employee_name, ...]
    task_day_to_emps: Dict[Tuple[str, int], List[str]] = defaultdict(list)
    # (employee_name, day_id) -> task_id
    emp_day_to_task: Dict[Tuple[str, int], str] = {}

    # We also want to know each task's country so we can display it in Sheet 1.
    # Because the entity carries task_id & country, we can gather from any assignment.
    task_country: Dict[str, str] = {}

    for r in solution.reqs:
        if r.employee is None or r.day is None:
            continue
        tid = r.task_id
        did = r.day.id
        ename = r.employee.name
        task_day_to_emps[(tid, did)].append(ename)
        emp_day_to_task[(ename, did)] = tid
        # remember country
        if tid not in task_country:
            task_country[tid] = r.country

    # ---------------- Sheet 1: Tasks x Dates ----------------
    ws1 = wb.active
    ws1.title = "Tasks x Dates"

    # Headers
    ws1.cell(row=1, column=1, value="task_id").font = HEADER_BOLD
    ws1.cell(row=1, column=2, value="country").font = HEADER_BOLD

    for j, d in enumerate(dates, start=3):
        c = ws1.cell(row=1, column=j, value=d.strftime("%Y-%m-%d"))
        c.font = HEADER_BOLD
        ws1.column_dimensions[get_column_letter(j)].width = 18
        if d.weekday() >= 5:
            # weekend shading in header too
            c.fill = WEEKEND_FILL

    ws1.column_dimensions["A"].width = 12
    ws1.column_dimensions["B"].width = 10
    ws1.freeze_panes = "C2"

    # Stable order of tasks: by task_id lexicographically
    all_task_ids = sorted({r.task_id for r in solution.reqs})

    for i, tid in enumerate(all_task_ids, start=2):
        ws1.cell(row=i, column=1, value=tid).font = HEADER_BOLD
        ws1.cell(row=i, column=1).alignment = LEFT

        ws1.cell(row=i, column=2, value=task_country.get(tid, "")).alignment = LEFT

        for j, d in enumerate(dates, start=3):
            did = j - 3  # because days start at 0; j starts at 3
            emps = sorted(task_day_to_emps.get((tid, did), []))
            txt = ""
            if emps:
                txt = f"{', '.join(emps)} ({len(emps)})"
            cell = ws1.cell(row=i, column=j, value=txt)
            cell.alignment = CENTER
            if d.weekday() >= 5:
                cell.fill = WEEKEND_FILL

    # ---------------- Sheet 2: Employees x Dates ----------------
    ws2 = wb.create_sheet("Employees x Dates")
    ws2.cell(row=1, column=1, value="employee").font = HEADER_BOLD
    ws2.column_dimensions["A"].width = 16

    for j, d in enumerate(dates, start=2):
        c = ws2.cell(row=1, column=j, value=d.strftime("%Y-%m-%d"))
        c.font = HEADER_BOLD
        ws2.column_dimensions[get_column_letter(j)].width = 16
        if d.weekday() >= 5:
            c.fill = WEEKEND_FILL

    ws2.freeze_panes = "B2"

    employees_sorted = sorted([e.name for e in solution.employees])

    for i, ename in enumerate(employees_sorted, start=2):
        ws2.cell(row=i, column=1, value=ename).font = HEADER_BOLD
        ws2.cell(row=i, column=1).alignment = LEFT

        for j, d in enumerate(dates, start=2):
            did = j - 2
            tid = emp_day_to_task.get((ename, did), "")
            c = ws2.cell(row=i, column=j, value=tid)
            c.alignment = CENTER
            if d.weekday() >= 5:
                c.fill = WEEKEND_FILL

    wb.save(out_path)
    print(f"Wrote Excel: {os.path.abspath(out_path)}")


def main():
    ap = ArgumentParser(description="Export mock Timefold schedule to a simple Excel workbook")
    ap.add_argument("--config", default="config_mock.yaml", help="Path to mock YAML config")
    ap.add_argument("--out", default="schedule_mock.xlsx", help="Output .xlsx path")
    args = ap.parse_args()

    # Solve using your mock Timefold solver
    solution, start_day = solve_from_config(args.config)
    print(f"Best score: {solution.score}")

    # Write a simple, readable workbook
    write_excel(solution, start_day, out_path=args.out)


if __name__ == "__main__":
    main()
