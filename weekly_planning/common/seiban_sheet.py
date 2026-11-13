"""
The seiban info workbook (sheet 製番): module code -> customer / fab / region ...
"""
from __future__ import annotations

import unicodedata

import pandas as pd
from openpyxl import load_workbook

from .dates import as_timestamp


_SEIBAN_COL = {
    "code": 1, "customer": 2, "fab": 3, "region": 4,
    "p2_start": 5, "p2m_headcount": 8,
    "p2e_headcount": 11,
    "p3_start": 12, "p3_headcount": 15,
    "p4_start": 16, "p4_kousu": 18, "p4_headcount": 19,
    "delivery": 20,
}


def _num_or_none(v):
    if v is None:
        return None
    if isinstance(v, (int, float)) and not (isinstance(v, float) and pd.isna(v)):
        return v
    if isinstance(v, str) and v.strip():
        try:
            return float(v.strip())
        except Exception:
            return None
    return None


def read_seiban_sheet(path, start_row=3):
    if not path:
        return {}
    wb = load_workbook(path, data_only=True)
    if "製番" not in wb.sheetnames:
        return {}
    ws = wb["製番"]
    rows = {}
    for r in range(max(3, int(start_row)), ws.max_row + 1):
        code_raw = ws.cell(row=r, column=_SEIBAN_COL["code"]).value
        if not isinstance(code_raw, str) or not code_raw.strip():
            continue
        code = unicodedata.normalize("NFKC", code_raw).strip()

        def cell(key):
            return ws.cell(row=r, column=_SEIBAN_COL[key]).value

        rows[code] = {
            "customer": cell("customer"),
            "fab": cell("fab"),
            "region": cell("region"),
            "p2_start": as_timestamp(cell("p2_start")),
            "p2m_headcount": _num_or_none(cell("p2m_headcount")),
            "p2e_headcount": _num_or_none(cell("p2e_headcount")),
            "p3_start": as_timestamp(cell("p3_start")),
            "p3_headcount": _num_or_none(cell("p3_headcount")),
            "p4_start": as_timestamp(cell("p4_start")),
            "p4_headcount": _num_or_none(cell("p4_headcount")),
            "p4_kousu": _num_or_none(cell("p4_kousu")),
            "delivery": as_timestamp(cell("delivery")),
            "delivery_note": cell("delivery") if isinstance(cell("delivery"), str) else None,
        }
    return rows
