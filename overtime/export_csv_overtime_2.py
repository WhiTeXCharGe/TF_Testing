# export_csv_overtime_option_b.py
import os
from argparse import ArgumentParser
from collections import defaultdict
from datetime import timedelta, datetime, date
from typing import Dict, Tuple, List

from openpyxl import Workbook
from openpyxl.styles import PatternFill, Alignment, Font
from openpyxl.utils import get_column_letter

from employee_scheduler_overtime_2 import solve_from_config

LIGHT_BLUE = "ADD8E6"   # module start highlight
RED        = "FF9999"   # task end highlight

def _parse_modules_config(config_path: str):
    with open(config_path, "r", encoding="utf-8") as f:
        import yaml
        cfg = yaml.safe_load(f)

    start_day = datetime.strptime(str(cfg["start_day"]), "%Y-%m-%d").date()
    horizon_days = int(cfg.get("horizon_days", 30))

    # Index helpers
    def to_idx(datestr: str) -> int:
        d = datetime.strptime(str(datestr), "%Y-%m-%d").date()
        return max(0, min(horizon_days - 1, (d - start_day).days))

    module_start_idx: Dict[str, int] = {}
    task_end_idx: Dict[str, int] = {}

    for m in cfg["modules"]:
        mcode = str(m["code"]).strip()
        m_start = to_idx(m.get("start_date", cfg["start_day"]))
        module_start_idx[mcode] = m_start
        for proc in m["processes"]:
            pid = int(proc["id"])
            p_end = to_idx(proc["end_date"])
            for t in proc["tasks"]:
                full_code = str(t["code"]).strip().upper()
                task_end_idx[full_code] = p_end

    return start_day, horizon_days, module_start_idx, task_end_idx


def write_excel(solution, start_day, config_path: str, out_path: str = "schedule_matrix_overtime.xlsx"):
    dates = [d.d for d in solution.days]  # list[date]
    # Sheet 1 aggregation: (task_code, date, employee) -> total hours
    tde_hours = defaultdict(int)
    # Sheet 2 aggregation: (employee, date, task_code) -> total hours
    edt_hours = defaultdict(int)

    for a in solution.assigns:
        if a.employee is None or a.hours <= 0:
            continue
        code = f"{a.module}-P{a.process_id}-{a.task_letter}"
        d = a.day.d
        tde_hours[(code, d, a.employee.name)] += a.hours
        edt_hours[(a.employee.name, d, code)] += a.hours

    # Read meta for starts/ends
    cfg_start_day, horizon_days, module_start_idx, task_end_idx = _parse_modules_config(config_path)

    wb = Workbook()
    bold = Font(bold=True)
    center = Alignment(horizontal="center", vertical="center", wrap_text=True)
    left   = Alignment(horizontal="left",   vertical="center", wrap_text=True)
    fill_deadline = PatternFill(start_color=RED,  end_color=RED,  fill_type="solid")
    fill_start    = PatternFill(start_color=LIGHT_BLUE, end_color=LIGHT_BLUE, fill_type="solid")

    # ---------------- Sheet 1: Tasks x Dates ----------------
    ws1 = wb.active
    ws1.title = "Tasks x Dates"

    ws1.cell(row=1, column=1, value="module").font = bold
    ws1.cell(row=1, column=2, value="task").font = bold
    for j, d in enumerate(dates, start=3):
        c = ws1.cell(row=1, column=j, value=d.strftime("%Y-%m-%d"))
        c.font = bold
        ws1.column_dimensions[get_column_letter(j)].width = 26
    ws1.column_dimensions["A"].width = 10
    ws1.column_dimensions["B"].width = 12
    ws1.freeze_panes = "C2"

    # All task codes that appear or are in config
    all_task_codes = sorted(set([k[0] for k in tde_hours.keys()] + list(task_end_idx.keys())))

    def task_sort_key(code: str):
        s, p, z = code.split("-")
        try:
            sN = int(s[1:]) if s.startswith("S") else 999
            pN = int(p[1:]) if p.startswith("P") else 999
        except Exception:
            sN, pN = 999, 999
        return (sN, pN, z)

    all_task_codes.sort(key=task_sort_key)

    for i, code in enumerate(all_task_codes, start=2):
        mcode, ppart, letter = code.split("-")
        ws1.cell(row=i, column=1, value=mcode).font = bold
        ws1.cell(row=i, column=2, value=f"{ppart}-{letter}").font = bold

        # start/end highlights
        start_idx = module_start_idx.get(mcode, None)
        end_idx = task_end_idx.get(code, None)

        for j, d in enumerate(dates, start=3):
            # build "AA(3,8H) | AB(2,6H)"
            entries = []
            # aggregate per employee for this task/date
            per_emp = {emp: h for (tcode, day, emp), h in tde_hours.items() if tcode == code and day == d}
            for emp, h in sorted(per_emp.items()):
                # find level
                # lookup by scanning solution.employees
                lvl = ""
                # find employee object
                for e in solution.employees:
                    if e.name == emp:
                        lvl = e.skills.get(f"{ppart}-{letter}", "")
                        break
                entries.append(f"{emp}({lvl},{h}H)")
            text = " | ".join(entries)
            c = ws1.cell(row=i, column=j, value=text)
            c.alignment = center

            if start_idx is not None and (d == (start_day + timedelta(days=start_idx))):
                c.fill = fill_start
            if end_idx is not None and (d == (start_day + timedelta(days=end_idx))):
                c.fill = fill_deadline

    # ---------------- Sheet 2: Employees x Dates ----------------
    ws2 = wb.create_sheet("Employees x Dates")
    ws2.cell(row=1, column=1, value="employee").font = bold
    ws2.cell(row=1, column=2, value="skills").font = bold
    ws2.cell(row=1, column=3, value="capacity").font = bold

    for j, d in enumerate(dates, start=4):
        c = ws2.cell(row=1, column=j, value=d.strftime("%Y-%m-%d"))
        c.font = bold
        ws2.column_dimensions[get_column_letter(j)].width = 30

    ws2.column_dimensions["A"].width = 16
    ws2.column_dimensions["B"].width = 48
    ws2.column_dimensions["C"].width = 12
    ws2.freeze_panes = "D2"

    # Sort employees by name
    employees_sorted = sorted(solution.employees, key=lambda e: e.name)

    # Build quick lookup for skills string
    def skills_text(e) -> str:
        parts = []
        for k, v in sorted(e.skills.items(), key=lambda kv: (kv[0][1:].split('-')[0] if kv[0].startswith('P') else kv[0], kv[0])):
            parts.append(f"{k}:{v}")
        return ", ".join(parts)

    # totals
    workday_counts = defaultdict(int)
    workhour_counts = defaultdict(int)

    for i, e in enumerate(employees_sorted, start=2):
        ws2.row_dimensions[i].height = 36
        ws2.cell(row=i, column=1, value=e.name).font = bold
        ws2.cell(row=i, column=1).alignment = left

        ws2.cell(row=i, column=2, value=skills_text(e)).alignment = left
        cap = int(e.capacity_hours_per_day) + int(e.overtime_hours_per_day)
        ws2.cell(row=i, column=3, value=cap).alignment = center

        for j, d in enumerate(dates, start=4):
            # concatenate tasks with hours for this employee/day
            tasks_here = [(t, hrs) for (emp, day, t), hrs in edt_hours.items() if emp == e.name and day == d]
            tasks_here.sort(key=lambda kv: kv[0])
            if tasks_here:
                workday_counts[e.name] += 1
                total_h = sum(h for _, h in tasks_here)
                workhour_counts[e.name] += total_h
                text = " | ".join(f"{t} ({h}H)" for t, h in tasks_here)
            else:
                text = ""
            ws2.cell(row=i, column=j, value=text).alignment = center

    # Totals columns
    workdays_col = len(dates) + 4
    workhours_col = len(dates) + 5
    ws2.cell(row=1, column=workdays_col, value="Workdays").font = bold
    ws2.cell(row=1, column=workhours_col, value="WorkHours").font = bold
    ws2.column_dimensions[get_column_letter(workdays_col)].width = 12
    ws2.column_dimensions[get_column_letter(workhours_col)].width = 12

    for i, e in enumerate(employees_sorted, start=2):
        ws2.cell(row=i, column=workdays_col, value=workday_counts[e.name]).alignment = center
        ws2.cell(row=i, column=workhours_col, value=workhour_counts[e.name]).alignment = center

    wb.save(out_path)
    print(f"Wrote Excel: {os.path.abspath(out_path)}")


def main():
    ap = ArgumentParser(description="Export Option-B overtime schedule to Excel")
    ap.add_argument("--config", default="config_modules.yaml", help="Path to modules YAML")
    ap.add_argument("--out", default="schedule_matrix_overtime.xlsx", help="Output .xlsx path")
    args = ap.parse_args()

    solution, start_day = solve_from_config(args.config)
    print(f"Best score: {solution.score}")
    write_excel(solution, start_day, config_path=args.config, out_path=args.out)


if __name__ == "__main__":
    main()
