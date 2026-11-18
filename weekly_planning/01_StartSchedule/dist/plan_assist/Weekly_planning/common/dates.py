"""
Date helpers.  ymd() turns anything date-like into the 'YYYY/MM/DD' text used in every yaml file; to_ymd() / as_timestamp()
are the pandas flavoured ones that the Excel readers use.
"""
from __future__ import annotations

from datetime import datetime

import pandas as pd


EXCEL_EPOCH = datetime(1899, 12, 30)


def ymd(v):
    """'YYYY/MM/DD' for a date-like value (str with - or /, date, datetime, Timestamp); None if unreadable."""
    if v is None:
        return None
    if hasattr(v, "strftime"):
        return v.strftime("%Y/%m/%d")
    s = str(v).strip().replace("-", "/")[:10]
    try:
        return datetime.strptime(s, "%Y/%m/%d").strftime("%Y/%m/%d")
    except ValueError:
        return None


def to_ymd(dt) -> str:
    if isinstance(dt, pd.Timestamp):
        return dt.strftime("%Y/%m/%d")
    if isinstance(dt, datetime):
        return pd.Timestamp(dt).strftime("%Y/%m/%d")
    return str(dt)


def as_timestamp(v):
    if v is None or (isinstance(v, float) and pd.isna(v)):
        return None
    if isinstance(v, pd.Timestamp):
        return v.normalize()
    if isinstance(v, datetime):
        return pd.Timestamp(v).normalize()
    if isinstance(v, (int, float)):
        try:
            return pd.Timestamp(EXCEL_EPOCH + pd.Timedelta(days=float(v))).normalize()
        except Exception:
            return None
    if isinstance(v, str):
        s = v.strip()
        if not s or s.upper() == "N/A":
            return None
        dt = pd.to_datetime(s, errors="coerce")
        if isinstance(dt, pd.Timestamp) and not pd.isna(dt):
            return dt.normalize()
    return None
