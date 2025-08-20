# export_matrix.py
import os
from argparse import ArgumentParser
from collections import defaultdict
from datetime import timedelta
from openpyxl import Workbook
from openpyxl.styles import PatternFill, Alignment, Font
from openpyxl.utils import get_column_letter

from employee_scheduler_second import solve_from_config  # uses your YAML-driven solver

LIGHT_BLUE = "ADD8E6"  # deadline highlight color

def write_excel_matrix(solution, start_day, deadline_by_task, out_path: str = "schedule_matrix.xlsx"):
    """Create an Excel matrix:
       - Rows: task_code (P1-A, P1-B, …)
       - Columns: dates across the horizon
       - Cells: comma-joined employee names assigned that day
       - Deadline cell per task highlighted light blue
    """
    # ordered dates from the planning horizon
    dates = [d.d for d in solution.days]  # list[date]

    # map (task_code, date) -> list of employee names
    cell = defaultdict(list)
    for r in solution.reqs:
        if r.employee and r.bucket and r.bucket.day:
            cell[(r.skill.code(), r.bucket.day.d)].append(r.employee.name)

    # compute deadline date per task_code (from day index)
    dl_date_for = {}
    for task_code, day_idx in deadline_by_task.items():
        dl_date_for[task_code] = start_day + timedelta(days=day_idx)

    # task list ordered by process then letter
    tasks = sorted(
        {r.skill.code() for r in solution.reqs},
        key=lambda code: (int(code.split("-")[0][1:]), code.split("-")[1])
    )

    # workbook & styles
    wb = Workbook()
    ws = wb.active
    ws.title = "Schedule"

    deadline_fill = PatternFill(start_color=LIGHT_BLUE, end_color=LIGHT_BLUE, fill_type="solid")
    center = Alignment(horizontal="center", vertical="center", wrap_text=True)
    bold = Font(bold=True)

    # layout settings
    LABEL_W = 10        # width of first column (task)
    DATE_W  = 22        # width of each date column (to fit names)
    ROW_H   = 36        # row height (for wrapped names)

    # header
    ws.cell(row=1, column=1, value="task \\ date").font = bold
    for j, d in enumerate(dates, start=2):
        c = ws.cell(row=1, column=j, value=d.strftime("%Y-%m-%d"))
        c.font = bold
        ws.column_dimensions[get_column_letter(j)].width = DATE_W
    ws.column_dimensions["A"].width = LABEL_W
    ws.freeze_panes = "B2"

    # body
    for i, code in enumerate(tasks, start=2):
        # task label
        label_cell = ws.cell(row=i, column=1, value=code)
        label_cell.font = bold
        ws.row_dimensions[i].height = ROW_H

        dl_date = dl_date_for.get(code, None)

        # one cell per date
        for j, d in enumerate(dates, start=2):
            names = cell.get((code, d), [])
            text = ", ".join(sorted(names))
            c = ws.cell(row=i, column=j, value=text)
            c.alignment = center
            if dl_date is not None and d == dl_date:
                c.fill = deadline_fill

    wb.save(out_path)
    print(f"Wrote Excel: {os.path.abspath(out_path)}")


def main():
    ap = ArgumentParser(description="Export schedule matrix (tasks x dates) with deadline highlight")
    ap.add_argument("--config", default="config.yaml", help="Path to YAML config")
    ap.add_argument("--out", default="schedule_matrix.xlsx", help="Output .xlsx path")
    args = ap.parse_args()

    solution, start_day, deadline_by_task = solve_from_config(args.config)
    print(f"Best score: {solution.score}")
    write_excel_matrix(solution, start_day, deadline_by_task, out_path=args.out)

if __name__ == "__main__":
    main()

