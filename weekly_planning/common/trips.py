"""
Trips (FI / FO): the shapes of trip_list, reading FI / FO marks from sheet text, and pairing them (with the repair of
marks that have no partner).
"""
from __future__ import annotations

import re
import unicodedata
from datetime import datetime


def flatten_trips(trip_list):
    """[{worker, trip}] from the nested form ({worker, trips:[...]}) and the flat form ({worker, start_date, ...})."""
    out = []
    for item in trip_list or []:
        if not isinstance(item, dict) or not item.get("worker"):
            continue
        if isinstance(item.get("trips"), list):
            for t in item["trips"]:
                if isinstance(t, dict):
                    out.append({"worker": item["worker"], "trip": t})
        elif "start_date" in item:
            out.append({"worker": item["worker"], "trip": item})
    return out


def pair_fifo(marks):
    """
    marks: {date: 'FI' | 'FO'} of one worker.  Pairs every FI with the next FO.
    Returns ([(start, end)], [warning]).  Orphans (FI without FO, FO without FI, FI FI, FO FO) are reported, not guessed.
    """
    pairs, warnings, open_fi = [], [], None
    for d in sorted(marks):
        kind = marks[d]
        if kind == "FI":
            if open_fi is not None:
                warnings.append(f"FI {open_fi} has no FO before the next FI {d}")
            open_fi = d
        else:
            if open_fi is None:
                warnings.append(f"FO {d} has no FI before it")
            else:
                pairs.append((open_fi, d))
                open_fi = None
    if open_fi is not None:
        warnings.append(f"FI {open_fi} has no FO after it")
    return pairs, warnings


def read_mark(text):
    """'FI' / 'FO' / None for one cell text.  Tolerates case, full-width letters, spaces, the typos F1 / Fl / F0
    and a trailing note such as FI(仮).  Anything else (including 移動) is not a mark."""
    if not isinstance(text, str):
        return None
    t = re.sub(r"[\(\[【].*$", "", unicodedata.normalize("NFKC", text).strip().upper().replace(" ", "")).strip()
    if t in ("FI", "F1", "FL", "F|", "F!"):
        return "FI"
    if t in ("FO", "F0"):
        return "FO"
    return None


def _day(s):
    return datetime.strptime(s, "%Y/%m/%d").date()


def _run_end(start, upto, activity, max_gap, window_end):
    """Last day of the run of busy days that continues after `start` (gaps up to max_gap days), before `upto`."""
    cur = start
    for x in activity:
        if x <= start or (upto is not None and x >= upto):
            continue
        if (_day(x) - _day(cur)).days > max_gap:
            break
        cur = x
    return cur


def _run_start(end, lower, activity, max_gap, window_start):
    """First day of the run of busy days that leads up to `end` (gaps up to max_gap days), after `lower`."""
    cur = end
    for x in reversed(activity):
        if x >= end or (lower is not None and x <= lower):
            continue
        if (_day(cur) - _day(x)).days > max_gap:
            break
        cur = x
    if lower is None and window_start and (_day(cur) - _day(window_start)).days <= max_gap:
        cur = window_start        # the sheet starts while the worker is already away: the trip began before it
    return cur


def pair_fifo_repaired(marks, activity, window=(None, None), max_gap=14, dup_days=2, chain_days=3):
    """
    Like pair_fifo, but every FI / FO that has no partner is repaired from the days the worker is busy in the sheet.
    marks: {date: 'FI'|'FO'} of one worker; activity: that worker's busy dates (any text in the sheet), window: first and
    last date of the sheet.  Returns ([(start, end)], [note]).
      FI right after FI or FO right after FO (<= dup_days apart, nothing busy between) .. one mark: the first FI, the last FO
      FI without FO (next FI or end of the sheet) .. ends on the last day of the busy run after the FI
      FO without FI (previous FO or start of the sheet) .. starts on the first day of the busy run before the FO
      FI, work, FI / FO, work, FO in a chain: a trip may end and the next one start on the SAME date (FI = FO date).
        FO after an FO: if the busy run starts within chain_days of the previous FO, the trip starts on the previous FO's date
        FI before an FI: if the next FI is within chain_days of the end of the busy run, the trip ends on that FI's date
    A busy run may have gaps of up to max_gap days.  Repaired trips are listed in the notes as INFERRED.
    """
    activity = sorted(set(activity))
    w_start, w_end = window
    seq = []
    notes = []
    for d in sorted(marks):
        k = marks[d]
        if (seq and seq[-1][1] == k and (_day(d) - _day(seq[-1][0])).days <= dup_days
                and not any(seq[-1][0] < x < d for x in activity)):
            if k == "FO":
                notes.append(f"duplicate FO {seq[-1][0]} / {d}: trip ends {d}")
                seq[-1] = (d, k)
            else:
                notes.append(f"duplicate FI {seq[-1][0]} / {d}: trip starts {seq[-1][0]}")
            continue
        seq.append((d, k))
    trips, open_fi, boundary, boundary_is_fo = [], None, None, False
    for d, k in seq:
        if k == "FI":
            if open_fi is not None:
                end = _run_end(open_fi, d, activity, max_gap, w_end)
                if (_day(d) - _day(end)).days <= chain_days:
                    end = d       # the next trip starts right after the busy run: same date for FO and FI
                    notes.append(f"INFERRED end {end} of the trip from FI {open_fi} (the next FI {d} is the same date)")
                else:
                    notes.append(f"INFERRED end {end} of the trip from FI {open_fi} (no FO before the next FI {d})")
                trips.append((open_fi, end))
                boundary, boundary_is_fo = end, False
            open_fi = d
        elif open_fi is None:
            start = _run_start(d, boundary, activity, max_gap, w_start)
            if boundary_is_fo and boundary is not None and (_day(start) - _day(boundary)).days <= chain_days:
                start = boundary  # FO, work, FO: the second trip starts on the date of the first FO
                notes.append(f"INFERRED start {start} of the trip to FO {d} (same date as the previous FO)")
            else:
                notes.append(f"INFERRED start {start} of the trip to FO {d} (no FI before it)")
            trips.append((start, d))
            boundary, boundary_is_fo = d, True
        else:
            trips.append((open_fi, d))
            open_fi, boundary, boundary_is_fo = None, d, True
    if open_fi is not None:
        end = _run_end(open_fi, None, activity, max_gap, w_end)
        trips.append((open_fi, end))
        notes.append(f"INFERRED end {end} of the trip from FI {open_fi} (no FO after it)")
    return trips, notes
