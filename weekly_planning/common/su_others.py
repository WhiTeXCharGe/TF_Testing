"""
The SU_Others workbook (plan sheet of workers and ongoing tasks): header / column detection, the skill reference sheet and
parse_su_others(), which turns the sheet into workers, companies and per-day cells.
"""
from __future__ import annotations

import sys
from datetime import datetime

import pandas as pd
from openpyxl import load_workbook

from .dates import to_ymd
from .excel_colors import IGNORE_WHITE_TEXT, cell_color_hex, is_grey_cell, is_red_cell, load_theme_palette
from .text import clean_text, extract_tool_code, norm_name, normalize_tool_code_in_text
from .worker_skills import split_slash


def find_date_header_ws(ws, max_scan_rows=12):
    for r in range(1, max_scan_rows + 1):
        row_vals = [cell.value for cell in ws[r]]
        if any(isinstance(v, (pd.Timestamp, datetime)) for v in row_vals):
            date_cols = [c for c, v in enumerate(row_vals, start=1) if isinstance(v, (pd.Timestamp, datetime))]
            if not date_cols:
                continue
            dt_by_col = {c: pd.Timestamp(row_vals[c - 1]).normalize() for c in date_cols}

            # Guard against stray cells that carry a date number-format but hold
            # a leftover/blank numeric value (e.g. a "1900-01-09" glitch from a
            # near-zero serial number) — such outliers would otherwise poison
            # min()/max() plan-range computation. Drop columns far (>3y) from
            # the median date of this header row.
            if len(dt_by_col) >= 3:
                sorted_dates = sorted(dt_by_col.values())
                median_dt = sorted_dates[len(sorted_dates) // 2]
                good_cols = [c for c, d in dt_by_col.items() if abs((d - median_dt).days) <= 1095]
                if len(good_cols) < len(date_cols):
                    dropped = sorted(set(date_cols) - set(good_cols))
                    print(f"WARNING: dropping {len(dropped)} SU_Others date-header column(s) far from median "
                          f"(likely formatting artifacts, not real dates): columns={dropped}", file=sys.stderr)
                    date_cols = sorted(good_cols)
                    dt_by_col = {c: dt_by_col[c] for c in good_cols}

            return r, date_cols, dt_by_col
    raise RuntimeError("Could not find date header row in SU_Others.")


def detect_su_columns(ws, label_row_idx, stamp_row_idx=None):
    """
    Detect SU_Others columns from the label row by their Japanese header,
    so column moves between file versions don't matter. Returns 0-indexed
    columns (None when a column is absent).

    予定表_2026 in the 20260915 file has TWO スキルレベル columns (H, I), with
    a yyyymmdd stamp (20260915 / 20260401) in the row above; the newest stamp
    wins. Without stamps, the first スキルレベル column is used.
    """
    row_vals = [c.value for c in ws[label_row_idx]]
    labels = [str(v).strip() if v is not None else "" for v in row_vals]

    def first_index(label):
        for i, l in enumerate(labels):
            if l == label:
                return i
        return None

    company_col = first_index("企業名")
    company_col = 0 if company_col is None else company_col

    name_col = first_index("姓名")
    name_col = 1 if name_col is None else name_col

    role_col = first_index("担当職種")
    if role_col is None:
        role_col = 6

    skill_cols = [i for i, l in enumerate(labels) if l == "スキルレベル"]
    skill_col = skill_cols[0] if skill_cols else None
    if len(skill_cols) > 1 and stamp_row_idx is not None:
        stamps = [c.value for c in ws[stamp_row_idx]]

        def _stamp(i):
            v = stamps[i] if i < len(stamps) else None
            try:
                return int(v)
            except Exception:
                return -1

        skill_col = max(skill_cols, key=lambda i: (_stamp(i), -i))

    return {
        "company": company_col, "name": name_col, "role": role_col,
        "id": first_index("ID"), "skill": skill_col, "status": first_index("Status"),
        "manager": first_index("作業責任者"),
        "gyoumu": first_index("業務形態"), "visa1": first_index("VISA1"), "visa2": first_index("VISA2"),
        "kaigai": first_index("海外運転"), "ojt": first_index("OJT"),
    }


def _cell_str(v):
    if v is None:
        return ""
    if isinstance(v, float) and v.is_integer():
        v = int(v)
    return clean_text(str(v)).strip()


def read_skill_reference(wb, sheet_name):
    """
    {("id", ID) | ("name", norm_name): {"role", "skill", "status"}} from the
    skill reference sheet (default "予定表_2026 (2)": 担当職種 / スキルレベル /
    Status). The first row wins when an ID/name repeats.
    """
    if not sheet_name or sheet_name not in wb.sheetnames:
        return {}
    ws = wb[sheet_name]
    try:
        date_row_idx, _c, _d = find_date_header_ws(ws)
    except RuntimeError:
        return {}
    label_row_idx = date_row_idx + 1
    cols = detect_su_columns(ws, label_row_idx, stamp_row_idx=date_row_idx)
    used = [c for c in (cols["id"], cols["name"], cols["role"], cols["skill"], cols["status"]) if c is not None]
    last_col = max(used) + 1

    def _get(row, c):
        return _cell_str(row[c]) if c is not None and c < len(row) else ""

    out = {}
    for row in ws.iter_rows(min_row=label_row_idx + 1, max_col=last_col, values_only=True):
        name = _get(row, cols["name"])
        if not name:
            continue
        ref = {
            "role": "/".join("QC" if t.upper() == "Q" else t for t in split_slash(_get(row, cols["role"]))),
            "skill": _get(row, cols["skill"]),
            "status": _get(row, cols["status"]),
        }
        out.setdefault(("name", norm_name(name)), ref)
        ext_id = _get(row, cols["id"])
        if ext_id and ext_id.lower() != "null":
            out.setdefault(("id", ext_id), ref)
    return out


def parse_su_others(path: str, sheet_names=("予定表_2026",), date_filter=None, skill_sheet=None,
                    workbook=None, skill_reference=None):
    """workbook: an already loaded openpyxl workbook of `path` (default: load it).
    skill_reference: the result of read_skill_reference() of another workbook, when the skills are kept in a separate file
    (default: read it from `skill_sheet` of this workbook)."""
    wb = workbook if workbook is not None else load_workbook(path, data_only=True, read_only=False)
    used_sheets = [s for s in sheet_names if s in wb.sheetnames]
    if not used_sheets:
        raise ValueError(f"None of {sheet_names} found in {path}. Available: {wb.sheetnames}")

    theme_palette = load_theme_palette(wb)
    skill_ref = skill_reference if skill_reference is not None else read_skill_reference(wb, skill_sheet)
    f_start, f_end = date_filter if date_filter else (None, None)

    worker_company_map = {}
    worker_company_list = []

    def get_worker_company_id(company_name: str) -> str:
        company_name = str(company_name).strip()
        if company_name not in worker_company_map:
            cid = f"wc{len(worker_company_map) + 1}"
            worker_company_map[company_name] = cid
            worker_company_list.append({
                "id": cid, "name": company_name,
                "annual_overtime_limit": 10000, "monthly_overtime_limit": 10000,
                "unavailable_dates": [],
            })
        return worker_company_map[company_name]

    worker_key_to_id = {}
    worker_acc = {}
    worker_date_map = {}
    worker_personal_map = {}
    worker_color_map = {}      # (wid, dt) -> "RRGGBB" fill colour of that cell
    worker_roles = {}          # wid -> role used by the phase heuristics (skill sheet, else main sheet)
    worker_skill_role = {}     # wid -> 担当職種 from the skill sheet ("" if absent)
    worker_skill_raw = {}      # wid -> raw スキルレベル text ("3", "2/3", "0", ...)
    worker_status = {}         # wid -> "Primary"/"Secondary"/""
    worker_in_skill_sheet = {} # wid -> True if matched in the skill sheet
    worker_description = {}

    plan_start = None
    plan_end = None

    for sheet_name in used_sheets:
        ws = wb[sheet_name]
        date_row_idx, date_cols, dt_by_col = find_date_header_ws(ws)
        label_row_idx = date_row_idx + 1
        cols = detect_su_columns(ws, label_row_idx, stamp_row_idx=date_row_idx)
        company_c, name_c, role_c = cols["company"], cols["name"], cols["role"]

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
        min_needed_col = max([c for c in cols.values() if c is not None] + [0]) + 1

        for r, row_cells in enumerate(
            ws.iter_rows(min_row=worker_start_row, min_col=1, max_col=max(max_col, min_needed_col)),
            start=worker_start_row,
        ):
            def _col_text(col_idx):
                if col_idx is None or col_idx >= len(row_cells):
                    return ""
                return _cell_str(row_cells[col_idx].value)

            company = row_cells[company_c].value if company_c < len(row_cells) else None
            name = row_cells[name_c].value if name_c < len(row_cells) else None

            if name is None or str(name).strip() == "":
                blank_streak += 1
                if blank_streak >= 30:
                    break
                continue
            blank_streak = 0

            company_str = "" if company is None else str(company).strip()
            name_str = str(name).strip()

            # Non-worker note rows (e.g. a "特殊検診" section marker) have no company.
            if not company_str:
                continue

            # a lone "Q" token is a typo of QC (seen once); normalise it so the
            # QC-based phase-3 trigger and no-QC cut rule see it too
            role_text = "/".join("QC" if t.upper() == "Q" else t for t in split_slash(_col_text(role_c)))
            is_manager = _col_text(cols["manager"]) != ""

            key = norm_name(name_str)
            if key not in worker_key_to_id:
                wid = f"w{len(worker_key_to_id) + 1:03d}"
                worker_key_to_id[key] = wid
                company_id = get_worker_company_id(company_str)
                worker_acc[key] = {
                    "id": wid, "name": name_str, "worker_company": company_id,
                    "is_manager": is_manager, "unavailable_set": set(),
                }
            else:
                if is_manager:
                    worker_acc[key]["is_manager"] = True

            wid = worker_key_to_id[key]
            if wid not in worker_in_skill_sheet:
                ext_id = _col_text(cols["id"])
                ref = (skill_ref.get(("id", ext_id)) if ext_id and ext_id.lower() != "null" else None) \
                    or skill_ref.get(("name", key))
                worker_in_skill_sheet[wid] = ref is not None
                ref = ref or {}
                worker_skill_role[wid] = ref.get("role", "")
                worker_skill_raw[wid] = ref.get("skill", "")
                worker_status[wid] = ref.get("status", "")
                if worker_skill_role[wid]:
                    worker_roles[wid] = worker_skill_role[wid]
            # main-sheet role only as a fallback for the phase heuristics
            if role_text and not worker_skill_role.get(wid):
                if len(role_text) >= len(worker_roles.get(wid, "")):
                    worker_roles[wid] = role_text

            if wid not in worker_description:
                gyoumu = _col_text(cols["gyoumu"])
                visa = " ".join(v for v in (_col_text(cols["visa1"]), _col_text(cols["visa2"])) if v).strip()
                kaigai = _col_text(cols["kaigai"])
                ojt_val = _col_text(cols["ojt"])

                if gyoumu or visa or kaigai or ojt_val:
                    desc = {"業務形態": gyoumu, "VISA": visa, "海外運転": kaigai}
                    if ojt_val:
                        desc["備考"] = "OJT"
                    worker_description[wid] = desc

            for c in date_cols:
                dt = dt_by_col[c]
                cell = row_cells[c - 1]

                if is_red_cell(cell):
                    worker_acc[key]["unavailable_set"].add(to_ymd(dt))
                    continue

                val = cell.value
                text = normalize_tool_code_in_text(val.strip()) if isinstance(val, str) else ""

                if text.upper() in IGNORE_WHITE_TEXT:
                    continue

                if is_grey_cell(cell):
                    if text and extract_tool_code(text):
                        worker_date_map[(wid, dt)] = text
                    else:
                        worker_personal_map[(wid, dt)] = text
                    color = cell_color_hex(cell, theme_palette)
                    if color:
                        worker_color_map[(wid, dt)] = color
                    continue

                if text == "":
                    continue

                worker_date_map[(wid, dt)] = text
                color = cell_color_hex(cell, theme_palette)
                if color:
                    worker_color_map[(wid, dt)] = color

    worker_list = []
    for acc in worker_acc.values():
        wid = acc["id"]
        worker_list.append({
            "id": wid, "name": acc["name"], "worker_company": acc["worker_company"],
            "is_manager": acc["is_manager"], "role": worker_roles.get(wid, ""),
            "skill_map": {}, "worker_type_by_operation": {}, "fab_suitability_map": [],
            "description": worker_description.get(wid),
            "unavailable_dates": [{"date": d} for d in sorted(acc["unavailable_set"])],
        })

    plan_range = {
        "start_date": to_ymd(plan_start) if plan_start is not None else "2025/01/01",
        "end_date": to_ymd(plan_end) if plan_end is not None else "2025/01/01",
    }

    return {
        "worker_company_list": worker_company_list, "worker_company_map": worker_company_map,
        "worker_list": worker_list, "plan_range": plan_range,
        "worker_date_map": worker_date_map, "worker_personal_map": worker_personal_map,
        "worker_color_map": worker_color_map,
        "worker_roles": worker_roles, "worker_skill_role": worker_skill_role,
        "worker_skill_raw": worker_skill_raw, "worker_status": worker_status,
        "worker_in_skill_sheet": worker_in_skill_sheet,
        "worker_description": worker_description,
    }
