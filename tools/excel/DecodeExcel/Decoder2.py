# Decoder2_with_options.py
# ---------------------------------------------------------------------
# Generates EnvConfig.yaml + Schedule.yaml for the Timefold scheduler from Excels.
#
# Inputs (normal mode):
#   1) 20260105 SU_Others.xlsm
#       - Worker list (name/company/manager flag)
#       - Worker unavailable dates (RED cells)
#       - Worker daily "what they did" text matrix (normal cells)
#       - Personal business (GREY cells) -> fixed assignments
#   2) SU_Others_予定表_2025_新規製番リスト_*.xlsx (sheet "CSV")
#       - Defines tool-install tasks (modules) + per-phase start dates (P1..P4) + P4 end
#       - Also provides Customer company / Country / Fab name for EnvConfig
#   3) スキル集計_*.xlsx (sheet "Sheet1")
#       - Main source of worker skill levels (p1,p2,p3,p4) using 合計/Level and 最小/Level
#
# Output format:
#   - wf_tool operations are generic "p1,p2,p3,p4"
#   - Task IDs are "e{n}" and phase IDs are "e{n}_p{phase}"
#   - Worker skill_map keys are "p1,p2,p3,p4" (plus other_op, personal_business_op)
#
# ---------------------------------------------------------------------
# CONFIG (edit here)
# ---------------------------------------------------------------------
# 1) READ_ALL_DATA:
#    - True  : normal mode (read SU_Others + skill + tasks, build assignments)
#    - False : "preview" mode (NO SU_Others)
#              - workers come from スキル集計 only
#              - tasks come from 新規製番リスト only
#              - assignment_list is empty
READ_ALL_DATA = False
#
# 2) DATE_RANGE:
#    - None: include all tasks and all SU_Others dates
#    - "YYYY/MM/DD-YYYY/MM/DD": include tasks that overlap this range
#      and (when READ_ALL_DATA=True) only read SU_Others date-cells within the range
#      (unavailable/personal/misc/assignments are generated only within the range).
#
# Example:
#   DATE_RANGE = "2026/01/01-2026/03/31"
DATE_RANGE = "2026/01/01-2026/06/30"
# ---------------------------------------------------------------------

import re
from collections import defaultdict
from datetime import datetime
from pathlib import Path

import pandas as pd
import yaml
from openpyxl import load_workbook


# ============================================================
# Helpers
# ============================================================

def _to_ymd(dt) -> str:
    """Convert pandas/datetime to 'YYYY/MM/DD'."""
    if isinstance(dt, pd.Timestamp):
        return dt.strftime("%Y/%m/%d")
    if isinstance(dt, datetime):
        return dt.strftime("%Y/%m/%d")
    return str(dt)


def _norm(s: str) -> str:
    """Normalize string for matching (trim, collapse spaces, lower)."""
    if not isinstance(s, str):
        return ""
    s = s.replace("　", " ")
    s = s.replace("（", "(").replace("）", ")")
    return " ".join(s.split()).lower()


def _as_timestamp(v):
    """Best-effort convert excel cell value into pd.Timestamp or None."""
    if v is None or (isinstance(v, float) and pd.isna(v)):
        return None
    if isinstance(v, pd.Timestamp):
        return v.normalize()
    if isinstance(v, datetime):
        return pd.Timestamp(v).normalize()
    if isinstance(v, str) and v.strip():
        dt = pd.to_datetime(v.strip(), errors="coerce")
        if isinstance(dt, pd.Timestamp) and not pd.isna(dt):
            return dt.normalize()
    return None


def _parse_date_range(date_range_str):
    """Parse DATE_RANGE like '2026/01/01-2026/03/31' -> (start_ts, end_ts)."""
    if not date_range_str:
        return None, None
    s = str(date_range_str).strip()
    if not s:
        return None, None
    parts = [p.strip() for p in s.replace("〜", "-").split("-") if p.strip()]
    if len(parts) != 2:
        raise ValueError(f"DATE_RANGE must be like 'YYYY/MM/DD-YYYY/MM/DD'. Got: {date_range_str}")
    a = pd.to_datetime(parts[0], errors="coerce")
    b = pd.to_datetime(parts[1], errors="coerce")
    if not isinstance(a, pd.Timestamp) or pd.isna(a) or not isinstance(b, pd.Timestamp) or pd.isna(b):
        raise ValueError(f"DATE_RANGE could not be parsed. Got: {date_range_str}")
    if b < a:
        a, b = b, a
    return a.normalize(), b.normalize()


def _overlaps(a_start, a_end, b_start, b_end) -> bool:
    """Inclusive overlap check."""
    return (a_start <= b_end) and (a_end >= b_start)


def load_sheet_as_df(path: str, sheet_name: str) -> pd.DataFrame:
    """Load a sheet via openpyxl and convert to DataFrame."""
    wb = load_workbook(path, data_only=True, read_only=False)
    if sheet_name not in wb.sheetnames:
        raise ValueError(f"Sheet '{sheet_name}' not found in {path}. Available: {wb.sheetnames}")
    ws = wb[sheet_name]
    rows = [list(r) for r in ws.iter_rows(values_only=True)]
    return pd.DataFrame(rows)


# ============================================================
# Code extraction from SU_Others cell text
# ============================================================

TOOLCODE_RE = re.compile(r"\d{3}[A-Z0-9]\d{5}A")


def extract_tool_code(s: str):
    """Extract FIRST tool code match from text."""
    if not isinstance(s, str):
        return None
    m = TOOLCODE_RE.search(s)
    return m.group(0) if m else None

def load_sheet_as_df_with_header(path: str, sheet_name: str, header_row: int = 0) -> pd.DataFrame:
    """
    Read excel sheet via openpyxl (read_only=False) and return a DataFrame.
    header_row is 0-based index in the sheet.
    """
    df_raw = load_sheet_as_df(path, sheet_name)  # your existing function (openpyxl)
    if df_raw.empty:
        return df_raw

    # Make sure header_row exists
    if header_row < 0 or header_row >= len(df_raw):
        raise ValueError(f"header_row={header_row} out of range for sheet {sheet_name} in {path}")

    headers = df_raw.iloc[header_row].tolist()
    df = df_raw.iloc[header_row + 1:].copy()
    df.columns = headers
    df = df.reset_index(drop=True)
    return df
# ============================================================
# SU_Others: worker info + raw matrix for assignments
# ============================================================

IGNORE_WHITE_TEXT = {"FI", "FO"}

GREY_RGB_LAST6 = {"A6A6A6", "BFBFBF", "D9D9D9", "808080"}
RED_RGB_LAST6 = {"FF0000"}


def _cell_rgb_last6(cell):
    """Return last 6 hex chars of an rgb fill (upper), or None."""
    fill = getattr(cell, "fill", None)
    if fill is None:
        return None
    fg = getattr(fill, "fgColor", None)
    if fg is None:
        return None
    if getattr(fg, "type", None) != "rgb":
        return None
    rgb = getattr(fg, "rgb", None)
    if not rgb:
        return None
    return str(rgb).upper()[-6:]


def _find_date_header_ws(ws, max_scan_rows=12):
    for r in range(1, max_scan_rows + 1):
        row_vals = [cell.value for cell in ws[r]]
        if any(isinstance(v, (pd.Timestamp, datetime)) for v in row_vals):
            date_cols = [c for c, v in enumerate(row_vals, start=1)
                         if isinstance(v, (pd.Timestamp, datetime))]
            if not date_cols:
                continue
            dt_by_col = {c: pd.Timestamp(row_vals[c - 1]).normalize() for c in date_cols}
            return r, date_cols, dt_by_col
    raise RuntimeError("Could not find date header row in SU_Others.")


def parse_su_others(path: str, sheet_names=("予定表_2024", "予定表_2025"), date_filter=None):
    """Parse SU_Others. If date_filter is set, only reads date columns within the range."""
    wb = load_workbook(path, data_only=True, read_only=False)
    used_sheets = [s for s in sheet_names if s in wb.sheetnames]
    if not used_sheets:
        raise ValueError(f"None of {sheet_names} found in {path}. Available: {wb.sheetnames}")

    f_start, f_end = date_filter if date_filter else (None, None)

    worker_company_map = {}
    worker_company_list = []

    def get_worker_company_id(company_name: str) -> str:
        company_name = str(company_name).strip()
        if company_name not in worker_company_map:
            cid = f"wc{len(worker_company_map) + 1}"
            worker_company_map[company_name] = cid
            worker_company_list.append({
                "id": cid,
                "name": company_name,
                "annual_overtime_limit": 360,
                "monthly_overtime_limit": 40,
                "unavailable_dates": [],
            })
        return worker_company_map[company_name]

    worker_key_to_id = {}
    worker_acc = {}
    worker_date_map = {}
    worker_personal_map = {}

    plan_start = None
    plan_end = None

    for sheet_name in used_sheets:
        ws = wb[sheet_name]
        date_row_idx, date_cols, dt_by_col = _find_date_header_ws(ws)

        if f_start is not None and f_end is not None:
            date_cols = [c for c in date_cols if f_start <= dt_by_col[c] <= f_end]
        if not date_cols:
            continue

        worker_start_row = date_row_idx + 2

        s = min(dt_by_col[c] for c in date_cols)
        e = max(dt_by_col[c] for c in date_cols)
        plan_start = s if plan_start is None else min(plan_start, s)
        plan_end = e if plan_end is None else max(plan_end, e)

        blank_streak = 0
        max_col = max(date_cols)

        for r, row_cells in enumerate(ws.iter_rows(min_row=worker_start_row, min_col=1, max_col=max_col), start=worker_start_row):
            company = row_cells[0].value
            name = row_cells[1].value

            if name is None or str(name).strip() == "":
                blank_streak += 1
                if blank_streak >= 30:
                    break
                continue
            blank_streak = 0

            company_str = "" if company is None else str(company).strip()
            name_str = str(name).strip()

            free_slot = row_cells[4].value if len(row_cells) >= 5 else None
            is_manager = bool(isinstance(free_slot, str) and "責" in free_slot)

            key = (company_str, name_str)
            if key not in worker_key_to_id:
                wid = f"w{len(worker_key_to_id) + 1:03d}"
                worker_key_to_id[key] = wid
                company_id = get_worker_company_id(company_str)
                worker_acc[key] = {
                    "id": wid,
                    "name": name_str,
                    "worker_company": company_id,
                    "is_manager": is_manager,
                    "unavailable_set": set(),
                }
            else:
                if is_manager:
                    worker_acc[key]["is_manager"] = True

            wid = worker_key_to_id[key]

            for c in date_cols:
                dt = dt_by_col[c]
                cell = row_cells[c - 1]
                rgb6 = _cell_rgb_last6(cell)

                if rgb6 in RED_RGB_LAST6:
                    worker_acc[key]["unavailable_set"].add(_to_ymd(dt))
                    continue

                val = cell.value
                if not (isinstance(val, str) and val.strip()):
                    continue
                text = val.strip()

                if text.upper() in IGNORE_WHITE_TEXT:
                    continue

                if rgb6 in GREY_RGB_LAST6:
                    worker_personal_map[(wid, dt)] = text
                else:
                    worker_date_map[(wid, dt)] = text

    worker_list = []
    for acc in worker_acc.values():
        worker_list.append({
            "id": acc["id"],
            "name": acc["name"],
            "worker_company": acc["worker_company"],
            "is_manager": acc["is_manager"],
            "skill_map": {},
            "fab_suitability_map": [],
            "unavailable_dates": [{"date": d} for d in sorted(acc["unavailable_set"])],
        })

    plan_range = {
        "start_date": _to_ymd(plan_start) if plan_start is not None else "2025/01/01",
        "end_date": _to_ymd(plan_end) if plan_end is not None else "2025/01/01",
    }

    return {
        "worker_company_list": worker_company_list,
        "worker_company_map": worker_company_map,
        "worker_list": worker_list,
        "plan_range": plan_range,
        "worker_date_map": worker_date_map,
        "worker_personal_map": worker_personal_map,
    }


# ============================================================
# Task list from 新規製番リスト (sheet "CSV")
# ============================================================

def parse_tasks_from_csv(path: str, sheet_name: str = "CSV", date_filter=None):
    df = load_sheet_as_df_with_header(path, sheet_name, header_row=0)
    colmap = {c: str(c).replace("\n", "").strip() for c in df.columns}
    df = df.rename(columns=colmap)

    code_col = [c for c in df.columns if "新規製番" in c]
    if not code_col:
        raise ValueError("Could not find 新規製番 column in CSV sheet.")
    code_col = code_col[0]

    cust_col = "ユーザー名"
    country_col = "国"
    fab_col = "ファブ名"

    p1s_col = "工程１開始可能日"
    p2s_col = "工程２開始可能日"
    p3s_col = "工程３開始可能日"
    p4s_col = "工程４開始可能日"
    p4e_col = "工程４終了予定日"

    f_start, f_end = date_filter if date_filter else (None, None)

    tool_tasks = []
    code_to_phases = defaultdict(list)
    all_dates = []
    task_counter = 1

    for _, row in df.iterrows():
        code_raw = row.get(code_col)
        if not isinstance(code_raw, str) or not code_raw.strip():
            continue
        code = code_raw.strip()

        customer = row.get(cust_col)
        customer = str(customer).strip() if isinstance(customer, str) and str(customer).strip() else "OTHER"

        country = row.get(country_col)
        country = str(country).strip() if isinstance(country, str) and str(country).strip() else "Other"

        fab_name = row.get(fab_col)
        fab_name = str(fab_name).strip() if isinstance(fab_name, str) and str(fab_name).strip() else "Other"

        p1s = _as_timestamp(row.get(p1s_col))
        p2s = _as_timestamp(row.get(p2s_col))
        p3s = _as_timestamp(row.get(p3s_col))
        p4s = _as_timestamp(row.get(p4s_col))
        p4e = _as_timestamp(row.get(p4e_col))

        if not any([p1s, p2s, p3s, p4s, p4e]):
            continue

        starts = {1: p1s, 2: p2s, 3: p3s, 4: p4s}
        ends = {1: None, 2: None, 3: None, 4: p4e or p4s}

        # end = next phase start - 1 day
        for ph in (1, 2, 3):
            nxt = starts.get(ph + 1)
            if nxt is not None:
                ends[ph] = (nxt - pd.Timedelta(days=1)).normalize()

        # fill missing start/end
        for ph in (1, 2, 3, 4):
            if starts[ph] is None:
                # inherit from later start
                for j in range(ph + 1, 5):
                    if starts.get(j) is not None:
                        starts[ph] = starts[j]
                        break
            if starts[ph] is None:
                starts[ph] = ends.get(ph) or pd.Timestamp("2025-01-01")
            if ends[ph] is None:
                ends[ph] = starts[ph]
            if ends[ph] < starts[ph]:
                ends[ph] = starts[ph]
            starts[ph] = starts[ph].normalize()
            ends[ph] = ends[ph].normalize()

        # filter by overlap
        if f_start is not None and f_end is not None:
            overall_start = min(starts.values())
            overall_end = max(ends.values())
            if not _overlaps(overall_start, overall_end, f_start, f_end):
                continue

        task_id = f"e{task_counter}"
        task_counter += 1

        phase_task_list = []
        for ph in (1, 2, 3, 4):
            phase_id = f"{task_id}_p{ph}"
            start = starts[ph]
            end = ends[ph]
            workload_days = int((end - start).days) + 1
            nm = {1: "Module Setup", 2: "Hardware Setup", 3: "Function Setup", 4: "Utility"}.get(ph, f"P{ph}")

            phase_task_list.append({
                "id": phase_id,
                "name": nm,
                "phase": f"tool_p{ph}",
                "start_date": _to_ymd(start),
                "end_date": _to_ymd(end),
                "operation_task_list": [{
                    "id": phase_id,
                    "name": nm,
                    "operation": f"p{ph}",
                    "workload_days": workload_days,
                }],
            })

            code_to_phases[code].append({
                "phase_index": ph,
                "phase_id": phase_id,
                "start": start,
                "end": end,
                "operation": f"p{ph}",
            })

            all_dates.append(start)
            all_dates.append(end)

        tool_tasks.append({
            "id": task_id,
            "name": code,
            "workflow": "wf_tool",
            "fab": None,  # filled later
            "phase_task_list": phase_task_list,
            "module_code": code,   # internal
            "customer": customer,  # internal
            "country": country,    # internal
            "fab_name": fab_name,  # internal
        })

    return {
        "tool_tasks": tool_tasks,
        "code_to_phases": code_to_phases,
        "date_list": all_dates,
    }


# ============================================================
# Skill aggregation (スキル集計_*.xlsx)
# ============================================================

def _skill_level(total_level, min_level):
    try:
        total = 0 if total_level is None or (isinstance(total_level, float) and pd.isna(total_level)) else int(total_level)
    except Exception:
        total = 0
    try:
        mn = 0 if min_level is None or (isinstance(min_level, float) and pd.isna(min_level)) else int(min_level)
    except Exception:
        mn = 0

    bucket = 0 if total <= 20 else (total // 20)
    lvl = max(bucket, mn)
    return max(0, min(5, lvl))


def parse_skill_excel(path: str, sheet_name: str = "Sheet1"):
    """Returns (skill_levels, people_meta)."""
    df = load_sheet_as_df(path, sheet_name)

    # Find header row that contains '氏名' and '所属'
    header_row = None
    for r in range(0, min(20, len(df))):
        row = df.iloc[r].astype(str).tolist()
        if any("氏名" in c for c in row) and any("所属" in c for c in row):
            header_row = r
            break
    if header_row is None:
        # fallback: original format assumed row 6 (1-based 7)
        header_row = 6

    headers = df.iloc[header_row].tolist()
    col_index = {str(h).strip(): i for i, h in enumerate(headers) if str(h).strip() != "nan"}

    # Expected columns (some files vary, so use contains-match)
    def find_col_contains(keyword):
        for k, i in col_index.items():
            if keyword in k:
                return i
        return None

    name_c = find_col_contains("氏名")
    comp_c = find_col_contains("所属")

    # group labels are on another row in some files; easiest is to locate by text in the sheet
    # We'll search for exact labels in the whole DataFrame.
    def find_first_cell(text):
        for r in range(len(df)):
            for c in range(df.shape[1]):
                v = df.iat[r, c]
                if isinstance(v, str) and v.strip() == text:
                    return r, c
        return None, None

    g1_r, g1_c = find_first_cell("1:Module Setup")
    g2_r, g2_c = find_first_cell("2:Hardware Setup")
    g3_r, g3_c = find_first_cell("3:Function Setup")
    if g1_c is None or g2_c is None or g3_c is None:
        raise RuntimeError("Could not find group labels (1/2/3) in skill sheet.")

    p1_total, p1_min = g1_c, g1_c + 1
    p2_total, p2_min = g2_c, g2_c + 1
    p3_total, p3_min = g3_c, g3_c + 1

    # Start scanning from below the label row
    start_row = max(g1_r, g2_r, g3_r) + 2

    skill_levels = {}
    people_meta = {}

    for r in range(start_row, len(df)):
        name = df.iat[r, name_c] if name_c is not None else None
        if not (isinstance(name, str) and name.strip()):
            continue
        comp = df.iat[r, comp_c] if comp_c is not None else ""
        name_s = name.strip()
        comp_s = str(comp).strip() if isinstance(comp, str) and str(comp).strip() else ""

        p1_lvl = _skill_level(df.iat[r, p1_total], df.iat[r, p1_min])
        p2_lvl = _skill_level(df.iat[r, p2_total], df.iat[r, p2_min])
        p3_lvl = _skill_level(df.iat[r, p3_total], df.iat[r, p3_min])

        key = (_norm(comp_s), _norm(name_s))
        skill_levels[key] = {"p1": p1_lvl, "p2": p2_lvl, "p3": p3_lvl, "p4": p3_lvl}
        if key not in people_meta:
            people_meta[key] = {"company": comp_s, "name": name_s}

    return skill_levels, people_meta


# ============================================================
# Build assignments (SU_Others -> task list via code)
# ============================================================

def build_assignments(su_data: dict, task_data: dict, date_filter=None):
    code_to_phases = task_data["code_to_phases"]
    worker_date_map = su_data["worker_date_map"]
    worker_personal_map = su_data["worker_personal_map"]

    f_start, f_end = date_filter if date_filter else (None, None)

    known_assign_map = defaultdict(list)
    misc_label_dates = defaultdict(set)
    misc_worker_label_dates = defaultdict(list)

    personal_label_dates = defaultdict(set)
    personal_worker_label_dates = defaultdict(list)

    inferred_worker_phase = defaultdict(set)

    for (wid, dt), text in worker_date_map.items():
        if f_start is not None and f_end is not None and (dt < f_start or dt > f_end):
            continue

        code = extract_tool_code(text)
        if code and code in code_to_phases:
            for phase_meta in code_to_phases[code]:
                ps = phase_meta["start"]
                pe = phase_meta["end"]
                if ps <= dt <= pe:
                    known_assign_map[(wid, phase_meta["phase_id"])].append(dt)
                    inferred_worker_phase[wid].add(phase_meta["operation"])
                    break
        else:
            label = text.strip()
            misc_label_dates[label].add(dt)
            misc_worker_label_dates[(wid, label)].append(dt)

    for (wid, dt), text in worker_personal_map.items():
        if f_start is not None and f_end is not None and (dt < f_start or dt > f_end):
            continue
        label = text.strip() if isinstance(text, str) and text.strip() else "personal_business"
        personal_label_dates[label].add(dt)
        personal_worker_label_dates[(wid, label)].append(dt)

    assignments = []

    for (wid, phase_id), dates in known_assign_map.items():
        uniq_dates = sorted(set(dates))
        if not uniq_dates:
            continue
        work_date_list = [{"hour": 12, "date": _to_ymd(d)} for d in uniq_dates]
        assignments.append({
            "worker": wid,
            "operation_task": phase_id,
            "start_date": _to_ymd(uniq_dates[0]),
            "end_date": _to_ymd(uniq_dates[-1]),
            "work_date_list": work_date_list,
            "plan_flexibility": "Flexible",
        })

    # In this Decoder2, misc/personal tasks exist only when we read SU_Others
    # (kept same behavior as before)
    misc_tasks = []
    personal_tasks = []
    return assignments, misc_tasks, personal_tasks, inferred_worker_phase


# ============================================================
# Build EnvConfig + Schedule and dump YAML
# ============================================================

def build_env_and_schedule_v2(
    su_others_path: str,
    task_csv_path: str,
    skill_excel_path: str,
    envconfig_out: str = "EnvConfig_from_excel_decoder2.yaml",
    schedule_out: str = "Schedule_from_excel_decoder2.yaml",
    read_all_data: bool = True,
    date_range: str | None = None,
):
    date_filter = _parse_date_range(date_range) if date_range else (None, None)
    f_start, f_end = date_filter

    task_data = parse_tasks_from_csv(task_csv_path, date_filter=date_filter if f_start is not None else None)
    skill_levels, skill_people = parse_skill_excel(skill_excel_path)

    # ---------- workers ----------
    if read_all_data:
        su_data = parse_su_others(su_others_path, date_filter=date_filter if f_start is not None else None)
        worker_company_list = su_data["worker_company_list"]
        worker_list = su_data["worker_list"]
    else:
        su_data = None
        # worker list from skill excel only
        comp_to_id = {}
        worker_company_list = []

        def get_wc_id(cn: str):
            cn = str(cn).strip() if isinstance(cn, str) else ""
            if cn not in comp_to_id:
                cid = f"wc{len(comp_to_id) + 1}"
                comp_to_id[cn] = cid
                worker_company_list.append({
                    "id": cid,
                    "name": cn,
                    "annual_overtime_limit": 360,
                    "monthly_overtime_limit": 40,
                    "unavailable_dates": [],
                })
            return comp_to_id[cn]

        keys = sorted(skill_people.keys())
        worker_list = []
        for i, k in enumerate(keys, start=1):
            meta = skill_people[k]
            worker_list.append({
                "id": f"w{i:03d}",
                "name": meta["name"],
                "worker_company": get_wc_id(meta["company"]),
                "is_manager": False,
                "skill_map": {},
                "fab_suitability_map": [],
                "unavailable_dates": [],
            })

    # ---------- customer/region/fab lists from task csv ----------
    customer_name_to_id = {"OTHER": "c_other"}
    region_name_to_id = {"Other": "r_other"}
    fab_name_to_id = {"Other": "f_other"}

    customer_company_list = [{"id": "c_other", "name": "OTHER", "unavailable_dates": []}]
    region_list = [{
        "id": "r_other",
        "name": "Other",
        "max_stay_on": 90,
        "max_annual_stay": 240,
        "stay_off_interval": 3,
        "unavailable_dates": [{"weekly": {"weekdays": ["sat", "sun"]}}],
    }]
    fab_list = [{"id": "f_other", "name": "Other", "region": "r_other", "customer_company": "c_other", "unavailable_dates": []}]

    def get_customer_id(name: str) -> str:
        nm = name.strip() if isinstance(name, str) and name.strip() else "OTHER"
        if nm not in customer_name_to_id:
            cid = f"c{len(customer_name_to_id)}"
            customer_name_to_id[nm] = cid
            customer_company_list.append({"id": cid, "name": nm, "unavailable_dates": []})
        return customer_name_to_id[nm]

    def get_region_id(country: str) -> str:
        nm = country.strip() if isinstance(country, str) and country.strip() else "Other"
        if nm not in region_name_to_id:
            rid = f"r{len(region_name_to_id)}"
            region_name_to_id[nm] = rid
            region_list.append({
                "id": rid,
                "name": nm,
                "max_stay_on": 90,
                "max_annual_stay": 240,
                "stay_off_interval": 3,
                "unavailable_dates": [{"weekly": {"weekdays": ["sat", "sun"]}}],
            })
        return region_name_to_id[nm]

    def get_fab_id(fab_name: str, country: str, customer: str) -> str:
        nm = fab_name.strip() if isinstance(fab_name, str) and fab_name.strip() else "Other"
        if nm not in fab_name_to_id:
            fid = f"f{len(fab_name_to_id)}"
            fab_name_to_id[nm] = fid
            fab_list.append({
                "id": fid,
                "name": nm,
                "region": get_region_id(country),
                "customer_company": get_customer_id(customer),
                "unavailable_dates": [],
            })
        return fab_name_to_id[nm]

    for t in task_data["tool_tasks"]:
        fid = get_fab_id(t.get("fab_name"), t.get("country"), t.get("customer"))
        t["fab"] = fid

    # ---------- environment ----------
    environment = {
        "workflow_list": [
            {
                "id": "wf_tool",
                "name": "Tool Install",
                "phase_list": [
                    {"id": "tool_p1", "name": "Module Setup", "operation_list": [{"id": "p1", "name": "Module Setup", "work_hours": [8], "min_worker_num": 1, "max_worker_num": 3}]},
                    {"id": "tool_p2", "name": "Hardware Setup", "operation_list": [{"id": "p2", "name": "Hardware Setup", "work_hours": [8], "min_worker_num": 1, "max_worker_num": 3}]},
                    {"id": "tool_p3", "name": "Function Setup", "operation_list": [{"id": "p3", "name": "Function Setup", "work_hours": [8], "min_worker_num": 1, "max_worker_num": 3}]},
                    {"id": "tool_p4", "name": "Utility", "operation_list": [{"id": "p4", "name": "Utility", "work_hours": [8], "min_worker_num": 1, "max_worker_num": 3}]},
                ],
            },
            {
                "id": "wf_other",
                "name": "Other work (from SU_Others)",
                "phase_list": [{
                    "id": "other_p1",
                    "name": "Other work",
                    "operation_list": [{"id": "other_op", "name": "Other work", "work_hours": [8], "min_worker_num": 1, "max_worker_num": 3}],
                }],
            },
            {
                "id": "wf_personal_business",
                "name": "Personal Business (from SU_Others grey cells)",
                "phase_list": [{
                    "id": "pb_p1",
                    "name": "Personal Business",
                    "operation_list": [{"id": "personal_business_op", "name": "Personal Business", "work_hours": [8], "min_worker_num": 1, "max_worker_num": 1}],
                }],
            },
        ],
        "fab_list": fab_list,
        "region_list": region_list,
        "customer_company_list": customer_company_list,
        "worker_company_list": worker_company_list,
        "worker_list": worker_list,
        "transite_day_map": [],
    }

    # ---------- assignments ----------
    if read_all_data and su_data is not None:
        assignments, misc_tasks, personal_tasks, inferred_worker_phase = build_assignments(
            su_data, task_data, date_filter=date_filter if f_start is not None else None
        )
    else:
        assignments, misc_tasks, personal_tasks, inferred_worker_phase = [], [], [], defaultdict(set)

    # ---------- fill skills ----------
    wc_id_to_name = {wc["id"]: wc["name"] for wc in worker_company_list}

    workers_with_skill_excel = set()
    for w in environment["worker_list"]:
        w["skill_map"] = {"p1": 0, "p2": 0, "p3": 0, "p4": 0, "other_op": 1, "personal_business_op": 1}
        key = (_norm(wc_id_to_name.get(w["worker_company"], "")), _norm(w["name"]))
        if key in skill_levels:
            w["skill_map"].update(skill_levels[key])
            workers_with_skill_excel.add(w["id"])

    for wid, ops in inferred_worker_phase.items():
        if wid in workers_with_skill_excel:
            continue
        for op in ops:
            if op in ("p1", "p2", "p3", "p4"):
                # minimum inferred level is 1
                w = next((x for x in environment["worker_list"] if x["id"] == wid), None)
                if w:
                    w["skill_map"][op] = max(w["skill_map"].get(op, 0), 1)

    # ---------- plan_range ----------
    all_dates = [d for d in task_data["date_list"] if isinstance(d, pd.Timestamp)]
    if read_all_data and su_data is not None:
        try:
            all_dates.append(pd.to_datetime(su_data["plan_range"]["start_date"]))
            all_dates.append(pd.to_datetime(su_data["plan_range"]["end_date"]))
        except Exception:
            pass
    all_dates.sort()
    plan_range = {"start_date": _to_ymd(all_dates[0]), "end_date": _to_ymd(all_dates[-1])} if all_dates else {"start_date": "2025/01/01", "end_date": "2025/01/01"}

    # strip internal keys
    tool_tasks_for_yaml = []
    for t in task_data["tool_tasks"]:
        t2 = dict(t)
        for k in ("module_code", "customer", "country", "fab_name"):
            t2.pop(k, None)
        tool_tasks_for_yaml.append(t2)

    schedule = {
        "plan_range": plan_range,
        "workflow_task_list": tool_tasks_for_yaml + misc_tasks + personal_tasks,
        "assignment_list": assignments,
    }

    env_root = {"environment": environment}
    sch_root = {"schedule": schedule}

    _BaseDumper = getattr(yaml, "CSafeDumper", yaml.SafeDumper)

    class NoAliasDumper(_BaseDumper):
        def ignore_aliases(self, data):
            return True

    with open(envconfig_out, "w", encoding="utf-8") as f:
        yaml.dump(env_root, f, Dumper=NoAliasDumper, sort_keys=False, allow_unicode=True, width=4096)

    with open(schedule_out, "w", encoding="utf-8") as f:
        yaml.dump(sch_root, f, Dumper=NoAliasDumper, sort_keys=False, allow_unicode=True, width=4096)

    return env_root, sch_root


# ============================================================
# Entrypoint
# ============================================================

if __name__ == "__main__":
    su_file = "20260105 SU_Others.xlsm"
    task_file = "SU_Others_予定表_2025_新規製番リスト_20260127.xlsx"
    skill_file = "スキル集計_20260127.xlsx"

    su_path = Path(su_file)
    task_path = Path(task_file)
    skill_path = Path(skill_file)

    if (not READ_ALL_DATA or su_path.exists()) and task_path.exists() and skill_path.exists():
        build_env_and_schedule_v2(
            str(su_path),
            str(task_path),
            str(skill_path),
            envconfig_out="EnvConfig.yaml",
            schedule_out="Schedule.yaml",
            read_all_data=READ_ALL_DATA,
            date_range=DATE_RANGE,
        )
        print("EnvConfig_from_excel_decoder2.yaml and Schedule_from_excel_decoder2.yaml have been written.")
    else:
        print("Please fix input file paths at the bottom of Decoder2_with_options.py.")
