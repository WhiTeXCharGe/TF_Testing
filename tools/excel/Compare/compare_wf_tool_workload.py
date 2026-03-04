#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Compare wf_tool only in schedule.workflow_task_list between two Schedule.yaml files.

Compares:
- workload_days
- start/end dates
- recommend min/max (rec_min/rec_max variants)

Output:
- Always writes a text report (markdown-like) to compare_schedule.txt by default.
- Terminal prints only 1 line: where the file was written.

Usage (PowerShell):
  python .\compare_wf_tool_workload.py .\Schedule1.yaml .\Schedule2.yaml
  python .\compare_wf_tool_workload.py .\Schedule1.yaml .\Schedule2.yaml --out compare_schedule.txt
"""

import argparse
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Tuple, Optional, List, Iterable
import yaml


# ----------------------------
# Generic extraction helpers
# ----------------------------

def _to_str(x: Any) -> str:
    if x is None:
        return ""
    return str(x)

def _to_int(x: Any) -> Optional[int]:
    if x is None:
        return None
    xs = str(x).strip()
    if xs == "":
        return None
    try:
        return int(float(xs))
    except Exception:
        return None

def _norm_date(s: Any) -> str:
    if s is None:
        return ""
    s = str(s).strip()
    if len(s) == 10 and s[4] == "-" and s[7] == "-":
        s = s.replace("-", "/")
    return s

def _iter_strings(v: Any) -> Iterable[str]:
    """Yield all string-like values inside nested dict/list structures."""
    if v is None:
        return
    if isinstance(v, str):
        yield v
        return
    if isinstance(v, (int, float, bool)):
        yield str(v)
        return
    if isinstance(v, dict):
        for kk, vv in v.items():
            # include both keys and values sometimes helpful
            if isinstance(kk, str):
                yield kk
            yield from _iter_strings(vv)
        return
    if isinstance(v, (list, tuple)):
        for x in v:
            yield from _iter_strings(x)
        return
    # fallback
    yield str(v)

def _find_first_key(d: Dict[str, Any], key_pred) -> Optional[Any]:
    """Return d[k] for first key where key_pred(k) is True."""
    for k, v in d.items():
        if isinstance(k, str) and key_pred(k):
            return v
    return None


# ----------------------------
# Record model
# ----------------------------

@dataclass(frozen=True)
class PhaseRec:
    module_id: str
    phase: str
    workload_days: Optional[int]
    rec_min: Optional[int]
    rec_max: Optional[int]
    start_date: str
    end_date: str

    @property
    def date_span(self) -> str:
        if self.start_date or self.end_date:
            return f"{self.start_date} → {self.end_date}"
        return ""


# ----------------------------
# Field extractors (robust)
# ----------------------------

def extract_workflow(item: Dict[str, Any]) -> str:
    """
    Try hard to extract workflow string.
    We look at any key containing 'workflow' (case-insensitive), and search nested values.
    """
    candidates: List[str] = []

    for k, v in item.items():
        if isinstance(k, str) and "workflow" in k.lower():
            for s in _iter_strings(v):
                ss = s.strip()
                if ss:
                    candidates.append(ss)

    # If nothing found via workflow keys, fall back to searching whole object (last resort)
    if not candidates:
        for s in _iter_strings(item):
            ss = s.strip()
            if ss:
                candidates.append(ss)

    # pick the best match:
    # exact wf_tool first
    for c in candidates:
        if c.lower() == "wf_tool":
            return "wf_tool"

    # sometimes stored like "workflow:wf_tool" or "wf_tool_xxx"
    for c in candidates:
        cl = c.lower()
        if "wf_tool" in cl:
            return "wf_tool"

    return ""


def extract_module_id(item: Dict[str, Any]) -> str:
    # Prefer explicit module_id-like keys
    v = _find_first_key(item, lambda k: k.lower() in ("module_id", "moduleid", "m_id"))
    if v is None:
        v = _find_first_key(item, lambda k: "module" in k.lower() and "id" in k.lower())
    if v is None:
        v = _find_first_key(item, lambda k: k.lower() == "module")
    if v is None:
        return ""

    # if nested, find something like "e72"
    for s in _iter_strings(v):
        ss = s.strip()
        if ss:
            return ss
    return ""


def extract_phase(item: Dict[str, Any]) -> str:
    # Prefer explicit phase/phase_id/phase_name keys
    v = _find_first_key(item, lambda k: k.lower() in ("phase", "phase_id", "phaseid", "phase_name"))
    if v is None:
        v = _find_first_key(item, lambda k: "phase" in k.lower())

    if v is None:
        return ""

    for s in _iter_strings(v):
        ss = s.strip()
        if ss:
            return ss
    return ""


def extract_workload_days(item: Dict[str, Any]) -> Optional[int]:
    v = _find_first_key(item, lambda k: k.lower() in ("workload_days", "workloaddays", "workloadday", "workload"))
    if v is None:
        v = _find_first_key(item, lambda k: "workload" in k.lower() and "day" in k.lower())
    return _to_int(v)


def extract_rec_min_max(item: Dict[str, Any]) -> Tuple[Optional[int], Optional[int]]:
    """
    Look for common recommend min/max variants and also pattern-like keys.
    """
    # direct common keys
    rmin = _find_first_key(item, lambda k: k.lower() in (
        "rec_min", "recommend_min", "recommended_min", "recommended_staff_min",
        "recommended_head_min", "recommend_staff_min"
    ))
    rmax = _find_first_key(item, lambda k: k.lower() in (
        "rec_max", "recommend_max", "recommended_max", "recommended_staff_max",
        "recommended_head_max", "recommend_staff_max"
    ))

    # fuzzy fallback: keys containing "rec"/"recommend" + "min/max"
    if rmin is None:
        rmin = _find_first_key(item, lambda k: ("rec" in k.lower() or "recommend" in k.lower()) and "min" in k.lower())
    if rmax is None:
        rmax = _find_first_key(item, lambda k: ("rec" in k.lower() or "recommend" in k.lower()) and "max" in k.lower())

    return _to_int(rmin), _to_int(rmax)


def extract_dates(item: Dict[str, Any]) -> Tuple[str, str]:
    s = _find_first_key(item, lambda k: k.lower() in ("start_date", "start", "phase_start", "planned_start", "date_start"))
    e = _find_first_key(item, lambda k: k.lower() in ("end_date", "end", "phase_end", "planned_end", "date_end"))
    return _norm_date(s), _norm_date(e)


# ----------------------------
# Load map
# ----------------------------

def load_wf_tool_map(path: Path) -> Dict[Tuple[str, str], PhaseRec]:
    data = yaml.safe_load(path.read_text(encoding="utf-8"))
    if not isinstance(data, dict):
        raise ValueError(f"{path} top-level YAML must be a dict")

    sched = data.get("schedule", data)
    wtl = sched.get("workflow_task_list")
    if not isinstance(wtl, list):
        raise ValueError(f"{path}: cannot find schedule.workflow_task_list (list)")

    out: Dict[Tuple[str, str], PhaseRec] = {}

    for item in wtl:
        if not isinstance(item, dict):
            continue

        wf = extract_workflow(item)
        if wf != "wf_tool":
            continue

        module_id = extract_module_id(item)
        phase = extract_phase(item)
        if not module_id or not phase:
            # skip if we can't key it
            continue

        workload_days = extract_workload_days(item)
        rec_min, rec_max = extract_rec_min_max(item)
        start_date, end_date = extract_dates(item)

        out[(module_id, phase)] = PhaseRec(
            module_id=module_id,
            phase=phase,
            workload_days=workload_days,
            rec_min=rec_min,
            rec_max=rec_max,
            start_date=start_date,
            end_date=end_date,
        )

    return out


# ----------------------------
# Report
# ----------------------------

def fmt_int(x: Optional[int]) -> str:
    return "" if x is None else str(x)


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("schedule1", type=Path)
    ap.add_argument("schedule2", type=Path)
    ap.add_argument("--out", type=Path, default=Path("compare_schedule.txt"))
    ap.add_argument("--big", type=int, default=30, help="threshold for big workload diff (abs(delta) >= this)")
    args = ap.parse_args()

    m1 = load_wf_tool_map(args.schedule1)
    m2 = load_wf_tool_map(args.schedule2)

    keys1 = set(m1.keys())
    keys2 = set(m2.keys())
    all_keys = sorted(keys1 | keys2)

    modules1 = {k[0] for k in keys1}
    modules2 = {k[0] for k in keys2}

    diffs: List[Tuple[Tuple[str, str], Optional[PhaseRec], Optional[PhaseRec]]] = []
    date_only: List[Tuple[Tuple[str, str], PhaseRec, PhaseRec]] = []
    rec_only: List[Tuple[Tuple[str, str], PhaseRec, PhaseRec]] = []

    for key in all_keys:
        r1 = m1.get(key)
        r2 = m2.get(key)

        if r1 is None or r2 is None:
            diffs.append((key, r1, r2))
            continue

        workload_same = (r1.workload_days == r2.workload_days)
        dates_same = (r1.start_date == r2.start_date and r1.end_date == r2.end_date)
        rec_same = (r1.rec_min == r2.rec_min and r1.rec_max == r2.rec_max)

        if workload_same and not dates_same and rec_same:
            date_only.append((key, r1, r2))
        if workload_same and dates_same and not rec_same:
            rec_only.append((key, r1, r2))

        if (not workload_same) or (not dates_same) or (not rec_same):
            diffs.append((key, r1, r2))

    # module total abs workload deltas
    abs_by_module: Dict[str, int] = {}
    for (mid, _ph), r1, r2 in diffs:
        w1 = (r1.workload_days if r1 else 0) or 0
        w2 = (r2.workload_days if r2 else 0) or 0
        abs_by_module[mid] = abs_by_module.get(mid, 0) + abs(int(w2) - int(w1))
    top_modules = sorted(abs_by_module.items(), key=lambda x: x[1], reverse=True)

    # buckets by workload delta
    big: List[Tuple[Tuple[str, str], Optional[PhaseRec], Optional[PhaseRec], int]] = []
    medium: List[Tuple[Tuple[str, str], Optional[PhaseRec], Optional[PhaseRec], int]] = []
    small: List[Tuple[Tuple[str, str], Optional[PhaseRec], Optional[PhaseRec], int]] = []
    minor: List[Tuple[Tuple[str, str], Optional[PhaseRec], Optional[PhaseRec], int]] = []

    for key, r1, r2 in diffs:
        w1 = (r1.workload_days if r1 else None)
        w2 = (r2.workload_days if r2 else None)
        d = ((w2 or 0) - (w1 or 0)) if not (w1 is None and w2 is None) else 0
        ad = abs(d)
        if ad >= args.big:
            big.append((key, r1, r2, d))
        elif 20 <= ad <= args.big - 1:
            medium.append((key, r1, r2, d))
        elif 10 <= ad <= 19:
            small.append((key, r1, r2, d))
        else:
            minor.append((key, r1, r2, d))

    big.sort(key=lambda x: abs(x[3]), reverse=True)
    medium.sort(key=lambda x: abs(x[3]), reverse=True)
    small.sort(key=lambda x: abs(x[3]), reverse=True)
    minor.sort(key=lambda x: abs(x[3]), reverse=True)

    # recommend diffs list
    rec_diffs: List[Tuple[Tuple[str, str], Optional[PhaseRec], Optional[PhaseRec]]] = []
    for key, r1, r2 in diffs:
        if r1 is None or r2 is None:
            if (r1 and (r1.rec_min is not None or r1.rec_max is not None)) or (r2 and (r2.rec_min is not None or r2.rec_max is not None)):
                rec_diffs.append((key, r1, r2))
            continue
        if r1.rec_min != r2.rec_min or r1.rec_max != r2.rec_max:
            rec_diffs.append((key, r1, r2))

    def _w(x: Optional[PhaseRec]) -> str:
        return "" if x is None else fmt_int(x.workload_days)

    def _rmin(x: Optional[PhaseRec]) -> str:
        return "" if x is None else fmt_int(x.rec_min)

    def _rmax(x: Optional[PhaseRec]) -> str:
        return "" if x is None else fmt_int(x.rec_max)

    def _dates(x: Optional[PhaseRec]) -> str:
        return "" if x is None else x.date_span

    out_lines: List[str] = []
    def line(s=""):
        out_lines.append(s)

    line("wf_tool workload + recommend(min/max) differences (Schedule1 vs Schedule2)")
    line("")
    line("Files:")
    line(f"  1) {args.schedule1}")
    line(f"  2) {args.schedule2}")
    line("")
    line("Quick summary")
    line(f"- Modules (wf_tool): {len(modules1)} in Schedule1, {len(modules2)} in Schedule2")
    line(f"- Phase entries compared (union): {len(all_keys)}")
    line(f"- Phase entries with any differences: {len(diffs)}")
    line(f"- Date-only changes (same workload & same rec): {len(date_only)}")
    line(f"- Recommend-only changes (same workload & same dates): {len(rec_only)}")
    line("")

    line("Modules with largest total workload change (sum of |delta| across phases)")
    if top_modules:
        for mid, tot in top_modules[:10]:
            line(f"- {mid}: {tot}")
    else:
        line("- (none)")
    line("")

    def print_table(rows, title):
        line(title)
        line("| module_id | phase | workload_1 | workload_2 | delta | rec_min_1 | rec_min_2 | rec_max_1 | rec_max_2 | dates_1 | dates_2 |")
        line("|---|---|---:|---:|---:|---:|---:|---:|---:|---|---|")
        for (mid, ph), r1, r2, d in rows:
            line(
                f"| {mid} | {ph} | {_w(r1)} | {_w(r2)} | {d:+d} | "
                f"{_rmin(r1)} | {_rmin(r2)} | {_rmax(r1)} | {_rmax(r2)} | "
                f"{_dates(r1)} | {_dates(r2)} |"
            )
        line("")

    if big:
        print_table(big, f"Biggest workload differences (|delta| ≥ {args.big})")
    else:
        line(f"Biggest workload differences (|delta| ≥ {args.big})")
        line("(none)")
        line("")

    def print_list(rows, title):
        line(title)
        if not rows:
            line("(none)")
            line("")
            return
        for (mid, ph), r1, r2, d in rows:
            line(f"- {mid} {ph}: workload {_w(r1)} → {_w(r2)} ({d:+d})")
            if (r1 and r2) and (r1.rec_min != r2.rec_min or r1.rec_max != r2.rec_max):
                line(f"  - recommend: min {_rmin(r1)} → {_rmin(r2)}, max {_rmax(r1)} → {_rmax(r2)}")
            if _dates(r1) or _dates(r2):
                line(f"  - dates: {_dates(r1)} → {_dates(r2)}")
        line("")

    print_list(medium, f"Medium workload differences (20 ≤ |delta| ≤ {args.big - 1})")
    print_list(small, "Small workload differences (10 ≤ |delta| ≤ 19)")
    print_list(minor, "Minor workload differences (|delta| < 10)")

    line("Recommend min/max differences (any change)")
    if not rec_diffs:
        line("(none)")
        line("")
    else:
        line("| module_id | phase | rec_min_1 | rec_min_2 | rec_max_1 | rec_max_2 | workload_1 | workload_2 | dates_1 | dates_2 |")
        line("|---|---|---:|---:|---:|---:|---:|---:|---|---|")
        for (mid, ph), r1, r2 in sorted(rec_diffs, key=lambda x: (x[0][0], x[0][1])):
            line(
                f"| {mid} | {ph} | {_rmin(r1)} | {_rmin(r2)} | {_rmax(r1)} | {_rmax(r2)} | "
                f"{_w(r1)} | {_w(r2)} | {_dates(r1)} | {_dates(r2)} |"
            )
        line("")

    line("Date-only changes (workload & recommend unchanged)")
    if not date_only:
        line("(none)")
    else:
        for (mid, ph), r1, r2 in sorted(date_only, key=lambda x: (x[0][0], x[0][1])):
            line(f"- {mid} {ph}: workload {fmt_int(r1.workload_days)} (same), recommend {fmt_int(r1.rec_min)}/{fmt_int(r1.rec_max)} (same)")
            line(f"  - dates: {r1.date_span} → {r2.date_span}")
    line("")

    args.out.write_text("\n".join(out_lines), encoding="utf-8")

    # only 1-line output (no report in terminal)
    print(f"Wrote: {args.out.resolve()}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())