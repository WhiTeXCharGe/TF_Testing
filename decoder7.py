
# Decoder7.py
# ---------------------------------------------------------------------
# Generates EnvConfig.yaml + Schedule.yaml (+ TransformationLog.txt) for the
# Timefold scheduler from a SINGLE SU_Others workbook
# (e.g. "20260915 SU_Others_skill level有配布禁止_.xlsm").
#
# Decoder7 vs Decoder6 — what changed and why:
#
# 1) INPUTS (two files):
#    --su-others   SU_Others .xlsm (e.g. "20260915 SU_Others_skill level有配布禁止_.xlsm")
#                    - 予定表_2026      : MAIN TASK sheet (day-by-day assignments,
#                                         red/grey cells, company, 作業責任者, description)
#                    - 予定表_2026 (2)  : SKILL REFERENCE sheet (担当職種, スキルレベル,
#                                         Status = Primary/Secondary)
#    --seiban-info 初期データ追加情報 (e.g. "初期データ追加情報 _20260930.rev1.xlsx"),
#                  sheet 製番 only: the TARGET modules = rows from
#                  --seiban-start-row (default 65) down. A tool code that is
#                  not in that part of 製番 becomes an "other" misc task.
#                  Rows whose 希望納期 says 製番間違い/製番まちがい ("wrong 製番")
#                  are not targets either. There is no 作業者 sheet anymore.
#
# 2) SKILL LEVELS, from the skill sheet's 担当職種 (G) + スキルレベル (H),
#    matched to the main-sheet worker by ID, then by name:
#    - 担当職種 blank                -> not a target of (auto) planning
#    - スキルレベル == 0 / blank      -> not a target of (auto) planning
#    - worker not in the skill sheet -> not a target of (auto) planning
#    - "M/QC" + "2/3"               -> M=2, QC=3 (paired by position)
#    - "M/QC" + "3" (single number) -> M=3, QC=3 (same level for every role)
#    Role -> operation: M -> p2o1, E -> p2o2, QC (or a lone "Q") -> p3o1+p4o1.
#    Other roles (搬送/溶接/通訳/安全/...) map to no operation.
#    Workers that are not planning targets keep their real SU_Others
#    assignments, but always as plan_flexibility: Fixed.
#    (The skill sheet's 担当職種 is also the role used by the phase-split /
#    no-QC heuristics; the main sheet's 担当職種 is only a fallback there.)
#
# 3) STATUS (skill sheet): Primary -> "regular"; Secondary / blank -> "spot",
#    applied to every operation in the worker's skill_map.
#
# 4) MISC TASKS live in Schedule.yaml only: EnvConfig no longer has
#    other_op / personal_business_op in skill_map. "Other work" misc tasks
#    have no workflow line; grey cells use workflow "wf_personal_business".
#
# 5) COLOR CODE + DESCRIPTION: every operation task and misc task gets
#    color_code = the most common SU_Others cell fill colour (6-digit hex,
#    theme colours + tint resolved to RGB) of the cells assigned to it.
#    description is written empty everywhere.
#
# 6) AFFINITY: EnvConfig gets an affinity_tag list with one tag per worker
#    company (wct{n}, weight 2), and each worker gets affinity: [wct{n}] for
#    its own company, so workers of the same company prefer working together.
#
# 8) WORKLOAD PREDICTION (see PRED_* config): 新規製番 have too little
#    SU_Others data for actual-only workloads, so every target module gets
#    workload = max(actual, 推奨人数 x 30 worker-days) (工数 60 / 推奨人数 2),
#    in hours = worker-days x 10 like decoder6. Targets with a blank 推奨人数
#    (and 製番間違い rows) become misc tasks.
#    Target modules the actual-data cut rules would drop (no QC, zero
#    workload, few cells) are kept when 製番 has p2/p3/p4 start dates: phase
#    windows from 製番 (p4 end predicted), their SU_Others cells kept.
#
# 7) Tool codes are NFKC-normalised before matching, so full-width digits
#    (e.g. "５３０N03567A") are the same module as "530N03567A".
#
# Unknown EnvConfig values keep their fields instead of being dropped:
# unavailable_dates: [], and 10000 for max_stay_on / max_annual_stay /
# overtime limits.
#
# The SU_Others cell rules (red = unavailable, grey = personal business,
# FI/FO ignored), the 製番 dummy/rescue rules and the whole cut/outlier/shift
# pipeline are carried over from decoder6 unchanged.
#
#python decoder7.py --su-others "20260915 SU_Others_skill level有配布禁止_.xlsm" --seiban-info "初期データ追加情報 _20260930.rev1.xlsx" --plan-start 2026/09/15 --plan-end 2027/07/31 --seiban-start-row 65 --su-sheets "予定表_2026" --skill-sheet "予定表_2026 (2)" --envconfig-out decoder7_out/EnvConfig.yaml --schedule-out decoder7_out/Schedule.yaml --log-out decoder7_out/TransformationLog.txt
# ---------------------------------------------------------------------

import argparse
import json
import re
import sys
from collections import Counter, defaultdict
from datetime import datetime
from pathlib import Path
import unicodedata
import math

import pandas as pd
import yaml
from openpyxl import load_workbook
from openpyxl.styles.colors import COLOR_INDEX

# ---------------------------------------------------------------------
# CONFIG (edit here; the input files + plan range are CLI args, see main())
# ---------------------------------------------------------------------
DEFAULT_MAX_WORKER = 8
HOURS_PER_WORKDAY = 10  # matches the "hour: 10" used per assignment work day

CUT_DISTANCE_DAYS = 365
SHIFT_USE_WORKED_DAYS = True
# Decoder5 default was True. On the new data this naive proportional
# calendar pre-split (as opposed to cut_final_zero_workload_modules_to_dummy,
# which checks the real *shifted* result) was wrongly discarding good real
# assignment data for several modules whose actual work isn't evenly spread
# across the calendar span (see TransformationLog "phase zero workload"
# entries). Disabled by default since decoder6; the final post-shift check
# still guards against genuinely empty phases.
CUT_MODULE_IF_PHASE_ZERO_WORKLOAD = False

# Decoder5 default was True (skip modules with no SU_Others match entirely).
# Decoder6/7 default is False: modules with no actual work found still show up
# in Schedule.yaml, using their planned 製番 dates (as long as those dates are
# complete — see parse_seiban_merged).
SKIP_MODULE_IF_NO_SU_MATCH = False

MIN_WORKED_DAYS_FOR_TOOL = 4
MIN_LEFT_DATE_SPAN_RATIO = 0.20

# Disabled: decoder5's "module appears within the first N days of the
# SU_Others plan range -> treat entire module as dummy" rule assumed a
# single-year sheet, where day-1 entries were likely carryover/phantom data.
# Now that the plan range can span years, "first 10 days of the range" no
# longer reliably means that; it wrongly zapped a real, heavily-worked module
# in testing. Set back to True to re-enable.
ENABLE_HEAD_OF_RANGE_CUT = False
DUMMY_HEAD_DAYS_FROM_PLAN_START = 10
ONGOING_TAIL_KEEP_GAP_DAYS = 30

TRANSFORMATION_LOG = "TransformationLog.txt"

# ------------------------------------------------------------------
# Workload PREDICTION for target (新規) modules.
# 新規製番 have little or no SU_Others data yet, so actual-only workloads
# (decoder6) leave most of them empty. Rule agreed for now: 製番's
# 第4工程 is 工数 60 for 推奨人数 2, i.e. 30 workload (worker-days) per
# recommended person, applied to every operation:
#     predicted worker-days = 推奨人数 x PRED_WD_PER_PERSON
# converted to hours like decoder6 (1 worker-day = HOURS_PER_WORKDAY h).
# Final workload = max(actual SU_Others worker-days, predicted).
# Target modules with any blank 推奨人数 can't be predicted -> misc tasks.
# ------------------------------------------------------------------
PREDICT_TARGET_WORKLOAD = True
PRED_WD_PER_PERSON = 60 / 2  # 製番 第4工程: 工数 60 / 推奨人数 2

# 製番 target list starts at this Excel row (rows 3..64 of
# "初期データ追加情報 _20260930.rev1" are an older block; the target list is
# 65 onward). Codes in rows above it are treated like codes not in 製番 (misc).
SEIBAN_START_ROW = 65

EXCEL_EPOCH = datetime(1899, 12, 30)

# ============================================================
# Generic helpers (unchanged from decoder5)
# ============================================================

def _to_ymd(dt) -> str:
    if isinstance(dt, pd.Timestamp):
        return dt.strftime("%Y/%m/%d")
    if isinstance(dt, datetime):
        return pd.Timestamp(dt).strftime("%Y/%m/%d")
    return str(dt)


def _as_timestamp(v):
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


def _parse_simple_date(s):
    """Parse a CLI plan-range date string like 2026/01/01 or 2026-01-01."""
    if not s:
        return None
    dt = pd.to_datetime(str(s).strip(), errors="coerce")
    if isinstance(dt, pd.Timestamp) and not pd.isna(dt):
        return dt.normalize()
    raise ValueError(f"Could not parse date: {s!r}")


def _overlaps(a_start, a_end, b_start, b_end) -> bool:
    return (a_start <= b_end) and (a_end >= b_start)


_WS_RE = re.compile(r"\s+")
_ZERO_WIDTH = {"​", "‌", "‍", "﻿"}


def _clean_text(s: str) -> str:
    if not isinstance(s, str):
        return ""
    s = unicodedata.normalize("NFKC", s)
    s = s.replace("　", " ")
    for z in _ZERO_WIDTH:
        s = s.replace(z, "")
    return s


def _norm_name(s: str) -> str:
    s = _clean_text(s)
    s = _WS_RE.sub("", s).strip()
    return s


def _planned_actual_gap_days(planned_start, planned_end, actual_start, actual_end) -> int:
    if planned_start is None or planned_end is None or actual_start is None or actual_end is None:
        return 0
    if actual_end < planned_start:
        return int((planned_start - actual_end).days)
    if actual_start > planned_end:
        return int((actual_start - planned_end).days)
    return 0


def _remember_original_text(su_data: dict, wid: str, dt: pd.Timestamp, old_text: str, new_text: str):
    m = su_data.setdefault("su_outlier_original_text", {})
    k = (wid, _to_ymd(dt))
    if k not in m:
        m[k] = {"old": old_text, "new": new_text}


# ============================================================
# Tool-code extraction (unchanged pattern; still \d{3}[A-Z0-9]\d{5}A)
# ============================================================

TOOLCODE_RE = re.compile(r"\d{3}[A-Z0-9]\d{5}A")


def extract_tool_code(s: str):
    if not isinstance(s, str):
        return None
    m = TOOLCODE_RE.search(s)
    return m.group(0) if m else None


def _normalize_tool_code_in_text(s: str) -> str:
    r"""Rewrite a full-width tool code (Python's \d also matches "５") to its
    half-width form in place, so "５３０N03567A_..." and "530N03567A_..." are
    the same module everywhere downstream (cut rules replace the code inside
    the text, so the text itself has to carry the normalised code)."""
    m = TOOLCODE_RE.search(s) if isinstance(s, str) else None
    if not m:
        return s
    norm = unicodedata.normalize("NFKC", m.group(0))
    return s if norm == m.group(0) else s[:m.start()] + norm + s[m.end():]


# ============================================================
# SU_Others cut/outlier pipeline (unchanged from decoder5)
# ============================================================

def cut_su_short_span_modules_to_dummy(su_data, min_unique_worked_days=4, planned_meta=None):
    if not su_data:
        return []
    worker_date_map = su_data.get("worker_date_map", {})
    if not worker_date_map:
        return []
    code_to_occ = defaultdict(list)
    for (wid, dt), text in list(worker_date_map.items()):
        code = extract_tool_code(text)
        if not code:
            continue
        if planned_meta is not None and code not in planned_meta:
            continue
        code_to_occ[code].append((dt, wid, text))
    corrections = []
    for code, occ in code_to_occ.items():
        uniq_days = sorted(set(dt for dt, _, _ in occ))
        if len(uniq_days) >= int(min_unique_worked_days):
            continue
        broken = _break_tool_code(code)
        for dt, wid, text in occ:
            if not isinstance(text, str):
                continue
            old = text
            new = old.replace(code, broken, 1)
            if new == old:
                continue
            _remember_original_text(su_data, wid, dt, old, new)
            worker_date_map[(wid, dt)] = new
            corrections.append({
                "wid": wid, "date": _to_ymd(dt), "code": code, "text": old,
                "reason": f"short span: {len(uniq_days)} worked days < {min_unique_worked_days}",
            })
    su_data["su_short_span_corrections"] = corrections
    return corrections


def cut_module_if_remaining_dates_too_small_vs_planned(su_data, planned_meta, min_left_date_span_ratio=0.20):
    if not su_data:
        return []
    worker_date_map = su_data.get("worker_date_map", {})
    if not worker_date_map:
        return []
    code_to_occ = defaultdict(list)
    for (wid, dt), text in worker_date_map.items():
        code = extract_tool_code(text)
        if not code:
            continue
        if code not in planned_meta:
            continue
        code_to_occ[code].append((dt, wid, text))
    corrections = []
    for code, occ in code_to_occ.items():
        plan = planned_meta.get(code)
        if not plan:
            continue
        planned_total_days = int(plan.get("total_len", 0))
        if planned_total_days <= 0:
            continue
        remaining_unique_days = sorted(set(dt for dt, _, _ in occ))
        remaining_count = len(remaining_unique_days)
        threshold = int(math.ceil(planned_total_days * float(min_left_date_span_ratio)))
        if remaining_count >= threshold:
            continue
        broken = _break_tool_code(code)
        for dt, wid, text in occ:
            if not isinstance(text, str):
                continue
            old = text
            new = old.replace(code, broken, 1)
            if new == old:
                continue
            _remember_original_text(su_data, wid, dt, old, new)
            worker_date_map[(wid, dt)] = new
            corrections.append({
                "wid": wid, "date": _to_ymd(dt), "code": code, "text": old,
                "reason": f"remaining {remaining_count} days < {threshold} threshold ({int(min_left_date_span_ratio*100)}% of planned {planned_total_days}d)",
            })
    su_data["su_remaining_ratio_corrections"] = corrections
    return corrections


def cut_module_if_phase_zero_workload(su_data, planned_meta):
    if not su_data:
        return []
    worker_date_map = su_data.get("worker_date_map", {})
    if not worker_date_map:
        return []
    code_to_occ = defaultdict(list)
    for (wid, dt), text in worker_date_map.items():
        code = extract_tool_code(text)
        if not code:
            continue
        if code not in planned_meta:
            continue
        code_to_occ[code].append((dt, wid, text))
    corrections = []
    for code, occ in code_to_occ.items():
        worked_days = sorted(set(dt for dt, _, _ in occ))
        if not worked_days:
            continue
        actual_start = worked_days[0]
        actual_end = worked_days[-1]
        actual_total_span = int((actual_end - actual_start).days) + 1
        if actual_total_span <= 0:
            continue
        span_days = [actual_start + pd.Timedelta(days=i) for i in range(actual_total_span)]
        alloc = _allocate_phase_lengths_v5(
            actual_total_span, planned_meta[code]["phase_len"], phase_ids=(2, 3, 4),
            min_one=(actual_total_span >= 3),
        )
        phase_span_days = {}
        idx = 0
        for ph in (2, 3, 4):
            ln = int(alloc.get(ph, 0))
            phase_span_days[ph] = span_days[idx: idx + ln] if ln > 0 else []
            idx += ln
        worked_set = set(worked_days)
        phase_worked_counts = {ph: sum(1 for d in phase_span_days[ph] if d in worked_set) for ph in (2, 3, 4)}
        if any(phase_worked_counts[ph] == 0 for ph in (2, 3, 4)):
            broken = _break_tool_code(code)
            for dt, wid, text in occ:
                if not isinstance(text, str):
                    continue
                old = text
                new = old.replace(code, broken, 1)
                if new == old:
                    continue
                _remember_original_text(su_data, wid, dt, old, new)
                worker_date_map[(wid, dt)] = new
                corrections.append({
                    "wid": wid, "date": _to_ymd(dt), "code": code, "text": old,
                    "reason": f"phase-zero pre-check: {phase_worked_counts}",
                })
    su_data["su_phase_zero_corrections"] = corrections
    return corrections


def cut_final_zero_workload_modules_to_dummy(su_data, shifted_meta):
    if not su_data:
        return []
    worker_date_map = su_data.get("worker_date_map", {})
    if not worker_date_map:
        return []
    zero_codes = set()
    for code, meta in shifted_meta.items():
        if not meta.get("had_su_match"):
            continue
        alloc = meta.get("alloc_worked_days") or {}
        if any(int(alloc.get(ph, 0)) <= 0 for ph in (2, 3, 4)):
            zero_codes.add(code)
    if not zero_codes:
        return []
    corrections = []
    for (wid, dt), text in list(worker_date_map.items()):
        code = extract_tool_code(text)
        if code not in zero_codes:
            continue
        if not isinstance(text, str):
            continue
        old = text
        new = old.replace(code, _break_tool_code(code), 1)
        if new == old:
            continue
        _remember_original_text(su_data, wid, dt, old, new)
        worker_date_map[(wid, dt)] = new
        corrections.append({
            "wid": wid, "date": _to_ymd(dt), "code": code, "text": old,
            "reason": "a phase has zero worked days after shifting",
        })
    su_data["su_final_zero_phase_corrections"] = corrections
    return corrections


def cut_modules_with_no_qc_to_dummy(su_data, shifted_meta):
    if not su_data:
        return []
    worker_date_map = su_data.get("worker_date_map", {})
    worker_roles = su_data.get("worker_roles", {})
    if not worker_date_map:
        return []
    code_to_wids = defaultdict(set)
    for (wid, dt), text in worker_date_map.items():
        code = extract_tool_code(text)
        if code:
            code_to_wids[code].add(wid)
    no_qc_codes = set()
    for code, meta in shifted_meta.items():
        if not meta.get("had_su_match"):
            continue
        has_qc = False
        for wid in code_to_wids.get(code, set()):
            role_text = _clean_text(worker_roles.get(wid, "")).upper()
            if "QC" in role_text:
                has_qc = True
                break
        if not has_qc:
            no_qc_codes.add(code)
    if not no_qc_codes:
        return []
    corrections = []
    for (wid, dt), text in list(worker_date_map.items()):
        code = extract_tool_code(text)
        if code not in no_qc_codes:
            continue
        if not isinstance(text, str):
            continue
        old = text
        new = old.replace(code, _break_tool_code(code), 1)
        if new == old:
            continue
        _remember_original_text(su_data, wid, dt, old, new)
        worker_date_map[(wid, dt)] = new
        corrections.append({
            "wid": wid, "date": _to_ymd(dt), "code": code, "text": old,
            "reason": "no QC worker assigned",
        })
    su_data["su_no_qc_corrections"] = corrections
    return corrections


def _break_tool_code(code: str) -> str:
    if not isinstance(code, str) or len(code) < 2:
        return "OUTLIER"
    return code[:-1] + "X"


def _cluster_by_date_gap(sorted_dates, gap_days: int):
    clusters = []
    cur = []
    for d in sorted_dates:
        if not cur:
            cur = [d]
            continue
        if (d - cur[-1]).days <= gap_days:
            cur.append(d)
        else:
            clusters.append(cur)
            cur = [d]
    if cur:
        clusters.append(cur)
    return clusters


def cut_su_outlier_cells(
    su_data, cluster_gap_days=7, far_gap_days=60, small_cluster_max_unique_days=7,
    cut_module_if_total_cells_lt=4, cut_module_if_unique_days_lt=4, planned_meta=None,
    cut_if_far_from_planned_days=90,
):
    if not su_data:
        return []
    plan_range = su_data.get("plan_range", {}) if su_data else {}
    plan_start = _as_timestamp(plan_range.get("start_date"))
    dummy_head_end = None
    if ENABLE_HEAD_OF_RANGE_CUT and plan_start is not None:
        dummy_head_end = plan_start + pd.Timedelta(days=max(0, DUMMY_HEAD_DAYS_FROM_PLAN_START - 1))
    worker_date_map = su_data.get("worker_date_map", {})
    if not worker_date_map:
        return []
    code_to_occ = defaultdict(list)
    for (wid, dt), text in list(worker_date_map.items()):
        code = extract_tool_code(text)
        if not code:
            continue
        code_to_occ[code].append((dt, wid, text))

    def _longest_consecutive_run(dts_sorted):
        if not dts_sorted:
            return 0
        best = 1
        cur = 1
        for i in range(1, len(dts_sorted)):
            if dts_sorted[i] == dts_sorted[i - 1] + pd.Timedelta(days=1):
                cur += 1
                if cur > best:
                    best = cur
            else:
                cur = 1
        return best

    corrections = []

    def _cut_all_occurrences(code, occ_list, reason):
        broken = _break_tool_code(code)
        for dt, wid, text in occ_list:
            if not isinstance(text, str):
                continue
            old = text
            new = old.replace(code, broken, 1)
            if new == old:
                continue
            _remember_original_text(su_data, wid, dt, old, new)
            worker_date_map[(wid, dt)] = new
            corrections.append({"wid": wid, "date": _to_ymd(dt), "code": code, "text": old, "reason": reason})

    def _rebuild_occ_for_code(code):
        out = []
        for (wid, dt), text in worker_date_map.items():
            if extract_tool_code(text) == code:
                out.append((dt, wid, text))
        return sorted(out, key=lambda x: (x[0], x[1]))

    for code, occ in code_to_occ.items():
        if dummy_head_end is not None:
            has_head_hit = any(plan_start <= dt <= dummy_head_end for dt, _, _ in occ)
            if has_head_hit:
                _cut_all_occurrences(code, occ, f"within first {DUMMY_HEAD_DAYS_FROM_PLAN_START} days of plan range")
                continue

        uniq_days_pre = sorted(set(dt for dt, _, _ in occ))
        if len(uniq_days_pre) < int(cut_module_if_unique_days_lt):
            _cut_all_occurrences(code, occ, f"{len(uniq_days_pre)} worked days < {cut_module_if_unique_days_lt}")
            continue
        if len(occ) < int(cut_module_if_total_cells_lt):
            _cut_all_occurrences(code, occ, f"{len(occ)} cells < {cut_module_if_total_cells_lt}")
            continue
        if len(occ) <= 1:
            _cut_all_occurrences(code, occ, "only 1 cell total")
            continue

        occ_by_wid = defaultdict(list)
        for dt, wid, text in occ:
            occ_by_wid[wid].append((dt, text))
        broken = _break_tool_code(code)

        for wid, wid_occ in occ_by_wid.items():
            wid_dates = sorted(set(dt for dt, _ in wid_occ))
            if len(wid_dates) <= 1:
                for dt, text in wid_occ:
                    if not isinstance(text, str):
                        continue
                    old = text
                    new = old.replace(code, broken, 1)
                    if new == old:
                        continue
                    _remember_original_text(su_data, wid, dt, old, new)
                    worker_date_map[(wid, dt)] = new
                    corrections.append({"wid": wid, "date": _to_ymd(dt), "code": code, "text": old,
                                         "reason": "only 1 day for this worker"})
                continue
            wid_clusters = _cluster_by_date_gap(wid_dates, gap_days=cluster_gap_days)
            keep_dates = set()
            for cl in wid_clusters:
                if _longest_consecutive_run(cl) >= 2:
                    keep_dates.update(cl)
            if not keep_dates:
                for dt, text in wid_occ:
                    if not isinstance(text, str):
                        continue
                    old = text
                    new = old.replace(code, broken, 1)
                    if new == old:
                        continue
                    _remember_original_text(su_data, wid, dt, old, new)
                    worker_date_map[(wid, dt)] = new
                    corrections.append({"wid": wid, "date": _to_ymd(dt), "code": code, "text": old,
                                         "reason": "no run of 2+ consecutive days"})
                continue
            for dt, text in wid_occ:
                if dt in keep_dates:
                    continue
                if not isinstance(text, str):
                    continue
                old = text
                new = old.replace(code, broken, 1)
                if new == old:
                    continue
                _remember_original_text(su_data, wid, dt, old, new)
                worker_date_map[(wid, dt)] = new
                corrections.append({"wid": wid, "date": _to_ymd(dt), "code": code, "text": old,
                                     "reason": "isolated, non-consecutive cluster"})

        clean_occ = _rebuild_occ_for_code(code)
        if not clean_occ:
            continue
        clean_uniq_days = sorted(set(dt for dt, _, _ in clean_occ))
        if len(clean_uniq_days) < int(cut_module_if_unique_days_lt) or len(clean_occ) < int(cut_module_if_total_cells_lt):
            _cut_all_occurrences(code, clean_occ, "too small after per-worker cleanup")
            continue

        if planned_meta is not None and code in planned_meta:
            pstart = planned_meta[code].get("overall_start")
            pend = planned_meta[code].get("overall_end")
            if pstart is not None and pend is not None:
                cutoff_day = pend + pd.Timedelta(days=int(cut_if_far_from_planned_days))
                uniq_dates = sorted(set(dt for dt, _, _ in clean_occ))
                clusters = _cluster_by_date_gap(uniq_dates, gap_days=1)
                kept_dates = set()
                latest_kept_day = None
                stop_keeping = False
                for cl in clusters:
                    cl_start = cl[0]
                    cl_end = cl[-1]
                    if stop_keeping:
                        continue
                    if cl_start <= cutoff_day:
                        kept_dates.update(cl)
                        latest_kept_day = cl_end
                        continue
                    if latest_kept_day is not None and (cl_start - latest_kept_day).days <= int(ONGOING_TAIL_KEEP_GAP_DAYS):
                        kept_dates.update(cl)
                        latest_kept_day = cl_end
                        continue
                    stop_keeping = True
                broken = _break_tool_code(code)
                for dt, wid, text in clean_occ:
                    if dt in kept_dates:
                        continue
                    if not isinstance(text, str):
                        continue
                    old = text
                    new = old.replace(code, broken, 1)
                    if new == old:
                        continue
                    _remember_original_text(su_data, wid, dt, old, new)
                    worker_date_map[(wid, dt)] = new
                    corrections.append({"wid": wid, "date": _to_ymd(dt), "code": code, "text": old,
                                         "reason": f"more than {cut_if_far_from_planned_days}d past planned end, {ONGOING_TAIL_KEEP_GAP_DAYS}d+ gap from last kept cluster"})

    def _final_cut_isolated_worker_module_pairs(su_data):
        if not su_data:
            return []
        worker_date_map = su_data.get("worker_date_map", {})
        if not worker_date_map:
            return []
        pair_to_occ = defaultdict(list)
        for (wid, dt), text in list(worker_date_map.items()):
            code = extract_tool_code(text)
            if not code:
                continue
            pair_to_occ[(wid, code)].append((dt, text))
        corrections = []

        def _has_adjacent_pair(sorted_dates):
            for i in range(1, len(sorted_dates)):
                if sorted_dates[i] == sorted_dates[i - 1] + pd.Timedelta(days=1):
                    return True
            return False

        for (wid, code), occ in pair_to_occ.items():
            uniq_dates = sorted(set(dt for dt, _ in occ))
            cut_flag = False
            reason = None
            if len(uniq_dates) <= 1:
                cut_flag = True
                reason = "only 1 day for this worker"
            elif not _has_adjacent_pair(uniq_dates):
                cut_flag = True
                reason = "no adjacent-day pair for this worker"
            if not cut_flag:
                continue
            broken = _break_tool_code(code)
            for dt, text in occ:
                if not isinstance(text, str):
                    continue
                old = text
                new = old.replace(code, broken, 1)
                if new == old:
                    continue
                _remember_original_text(su_data, wid, dt, old, new)
                worker_date_map[(wid, dt)] = new
                corrections.append({"wid": wid, "date": _to_ymd(dt), "code": code, "text": old, "reason": reason})
        su_data["su_final_isolated_pair_corrections"] = corrections
        return corrections

    final_pair_corrections = _final_cut_isolated_worker_module_pairs(su_data)
    corrections.extend(final_pair_corrections)
    su_data["su_outlier_corrections"] = corrections
    return corrections


# ============================================================
# SU_Others parsing — column layout is now auto-detected from the header
# row, since 予定表_2025 and 予定表_2026 use different layouts.
# ============================================================

GREY_RGB_LAST6 = {"A6A6A6", "BFBFBF", "D9D9D9", "808080"}
RED_RGB_LAST6 = {"FF0000"}
GREY_INDEXED = {15, 22, 23, 24, 25, 26, 27, 28, 29}
RED_INDEXED = {10}
IGNORE_WHITE_TEXT = {"FI", "FO"}


def _color_to_rgb6(color):
    if color is None:
        return None
    ctype = getattr(color, "type", None)
    if ctype == "rgb":
        rgb = getattr(color, "rgb", None)
        return str(rgb).upper()[-6:] if rgb else None
    if ctype == "indexed":
        idx = getattr(color, "indexed", None)
        if idx is None:
            return None
        try:
            rgb = COLOR_INDEX[idx]
            return str(rgb).upper()[-6:] if rgb else None
        except Exception:
            return None
    return None


def _cell_fill_rgb6(cell):
    fill = getattr(cell, "fill", None)
    if fill is None:
        return None
    candidates = [getattr(fill, "fgColor", None), getattr(fill, "start_color", None),
                  getattr(fill, "bgColor", None), getattr(fill, "end_color", None)]
    for col in candidates:
        rgb6 = _color_to_rgb6(col)
        if rgb6:
            return rgb6
    return None


def _cell_fill_indexed(cell):
    fill = getattr(cell, "fill", None)
    if fill is None:
        return None
    for col in [getattr(fill, "fgColor", None), getattr(fill, "start_color", None),
                getattr(fill, "bgColor", None), getattr(fill, "end_color", None)]:
        if col is None:
            continue
        if getattr(col, "type", None) == "indexed":
            return getattr(col, "indexed", None)
    return None


def _is_theme_grey(color) -> bool:
    if color is None:
        return False
    if getattr(color, "type", None) != "theme":
        return False
    tint = getattr(color, "tint", None)
    theme = getattr(color, "theme", None)
    if tint is not None:
        try:
            t = float(tint)
            if -0.6 <= t <= 0.6:
                return True
        except Exception:
            pass
    if theme is not None:
        return True
    return False


def _is_red_cell(cell) -> bool:
    rgb6 = _cell_fill_rgb6(cell)
    if rgb6 and rgb6 in RED_RGB_LAST6:
        return True
    idx = _cell_fill_indexed(cell)
    if idx is not None and idx in RED_INDEXED:
        return True
    return False


def _is_grey_cell(cell) -> bool:
    rgb6 = _cell_fill_rgb6(cell)
    if rgb6 and rgb6 in GREY_RGB_LAST6:
        return True
    idx = _cell_fill_indexed(cell)
    if idx is not None and idx in GREY_INDEXED:
        return True
    fill = getattr(cell, "fill", None)
    if fill is not None:
        for col in [getattr(fill, "fgColor", None), getattr(fill, "start_color", None),
                    getattr(fill, "bgColor", None), getattr(fill, "end_color", None)]:
            if _is_theme_grey(col):
                return True
    return False


def _cell_rgb_last6(cell):
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


# ------------------------------------------------------------
# Cell colour -> 6-digit hex (for color_code). Theme colours are resolved
# against the workbook's own theme palette and then tinted the way Excel does.
# ------------------------------------------------------------

# Excel theme index -> clrScheme element (lt/dk pairs are swapped vs. XML order)
_THEME_INDEX_TO_SLOT = ["lt1", "dk1", "lt2", "dk2", "accent1", "accent2", "accent3",
                        "accent4", "accent5", "accent6", "hlink", "folHlink"]


def _load_theme_palette(wb):
    """{theme_index: "RRGGBB"} from the workbook's theme1.xml (empty if unreadable)."""
    raw = getattr(wb, "loaded_theme", None)
    if not raw:
        return {}
    try:
        import xml.etree.ElementTree as ET
        root = ET.fromstring(raw)
        ns = {"a": "http://schemas.openxmlformats.org/drawingml/2006/main"}
        scheme = root.find(".//a:clrScheme", ns)
        if scheme is None:
            return {}
        slot_rgb = {}
        for child in scheme:
            slot = child.tag.split("}")[-1]
            srgb = child.find("a:srgbClr", ns)
            sys_c = child.find("a:sysClr", ns)
            if srgb is not None:
                slot_rgb[slot] = srgb.get("val", "").upper()
            elif sys_c is not None:
                slot_rgb[slot] = sys_c.get("lastClr", "").upper()
        return {i: slot_rgb[s] for i, s in enumerate(_THEME_INDEX_TO_SLOT) if slot_rgb.get(s)}
    except Exception:
        return {}


def _apply_tint(rgb6: str, tint) -> str:
    try:
        t = float(tint or 0.0)
    except Exception:
        t = 0.0
    if not t:
        return rgb6
    import colorsys
    r, g, b = (int(rgb6[i:i + 2], 16) / 255.0 for i in (0, 2, 4))
    h, l, s = colorsys.rgb_to_hls(r, g, b)
    l = l * (1.0 + t) if t < 0 else l * (1.0 - t) + t
    r, g, b = colorsys.hls_to_rgb(h, max(0.0, min(1.0, l)), s)
    return "".join(f"{round(c * 255):02X}" for c in (r, g, b))


def _cell_color_hex(cell, theme_palette):
    """Solid fill colour of a cell as "RRGGBB", or None when it has no fill."""
    fill = getattr(cell, "fill", None)
    if fill is None or getattr(fill, "fill_type", None) in (None, "none"):
        return None
    col = getattr(fill, "fgColor", None)
    ctype = getattr(col, "type", None)
    if ctype == "rgb":
        rgb = str(getattr(col, "rgb", "") or "").upper()
        return rgb[-6:] if len(rgb) >= 6 and all(ch in "0123456789ABCDEF" for ch in rgb[-6:]) else None
    if ctype == "indexed":
        return _color_to_rgb6(col)
    if ctype == "theme":
        base = theme_palette.get(getattr(col, "theme", None))
        return _apply_tint(base, getattr(col, "tint", 0.0)) if base else None
    return None


def _detect_su_columns(ws, label_row_idx, stamp_row_idx=None):
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


def _read_skill_reference(wb, sheet_name):
    """
    {("id", ID) | ("name", norm_name): {"role", "skill", "status"}} from the
    skill reference sheet (default "予定表_2026 (2)": 担当職種 / スキルレベル /
    Status). The first row wins when an ID/name repeats.
    """
    if not sheet_name or sheet_name not in wb.sheetnames:
        return {}
    ws = wb[sheet_name]
    try:
        date_row_idx, _c, _d = _find_date_header_ws(ws)
    except RuntimeError:
        return {}
    label_row_idx = date_row_idx + 1
    cols = _detect_su_columns(ws, label_row_idx, stamp_row_idx=date_row_idx)
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
            "role": "/".join("QC" if t.upper() == "Q" else t for t in _split_slash(_get(row, cols["role"]))),
            "skill": _get(row, cols["skill"]),
            "status": _get(row, cols["status"]),
        }
        out.setdefault(("name", _norm_name(name)), ref)
        ext_id = _get(row, cols["id"])
        if ext_id and ext_id.lower() != "null":
            out.setdefault(("id", ext_id), ref)
    return out


def _cell_str(v):
    if v is None:
        return ""
    if isinstance(v, float) and v.is_integer():
        v = int(v)
    return _clean_text(str(v)).strip()


def parse_su_others(path: str, sheet_names=("予定表_2026",), date_filter=None, skill_sheet=None):
    wb = load_workbook(path, data_only=True, read_only=False)
    used_sheets = [s for s in sheet_names if s in wb.sheetnames]
    if not used_sheets:
        raise ValueError(f"None of {sheet_names} found in {path}. Available: {wb.sheetnames}")

    theme_palette = _load_theme_palette(wb)
    skill_ref = _read_skill_reference(wb, skill_sheet)
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
        date_row_idx, date_cols, dt_by_col = _find_date_header_ws(ws)
        label_row_idx = date_row_idx + 1
        cols = _detect_su_columns(ws, label_row_idx, stamp_row_idx=date_row_idx)
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
            role_text = "/".join("QC" if t.upper() == "Q" else t for t in _split_slash(_col_text(role_c)))
            is_manager = _col_text(cols["manager"]) != ""

            key = _norm_name(name_str)
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

                if _is_red_cell(cell):
                    worker_acc[key]["unavailable_set"].add(_to_ymd(dt))
                    continue

                val = cell.value
                text = _normalize_tool_code_in_text(val.strip()) if isinstance(val, str) else ""

                if text.upper() in IGNORE_WHITE_TEXT:
                    continue

                if _is_grey_cell(cell):
                    if text and extract_tool_code(text):
                        worker_date_map[(wid, dt)] = text
                    else:
                        worker_personal_map[(wid, dt)] = text
                    color = _cell_color_hex(cell, theme_palette)
                    if color:
                        worker_color_map[(wid, dt)] = color
                    continue

                if text == "":
                    continue

                worker_date_map[(wid, dt)] = text
                color = _cell_color_hex(cell, theme_palette)
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
        "start_date": _to_ymd(plan_start) if plan_start is not None else "2025/01/01",
        "end_date": _to_ymd(plan_end) if plan_end is not None else "2025/01/01",
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


# ============================================================
# Skill levels / planning target (担当職種 G + スキルレベル H + Status)
# ============================================================

# role token -> operations it qualifies for ("Q" is a lone-Q typo of QC seen once)
ROLE_TOKEN_OPS = {"M": ("p2o1",), "E": ("p2o2",), "QC": ("p3o1", "p4o1"), "Q": ("p3o1", "p4o1")}
TASK_OPS = ("p2o1", "p2o2", "p3o1", "p4o1")
STATUS_TO_TYPE = {"PRIMARY": "regular"}  # everything else (Secondary / blank) -> spot


def _split_slash(s: str):
    return [t.strip() for t in re.split(r"[/／]", s) if t.strip()]


def parse_worker_skills(role_text: str, skill_raw: str):
    """
    Returns (skill_map, excluded_reason). skill_map is {op: level} with only
    nonzero levels; excluded_reason is None for planning targets.

      - 担当職種 blank                       -> excluded
      - スキルレベル 0 (or blank/#N/A)        -> excluded
      - "M/QC" + "2/3"                      -> M=2, QC=3
      - "M/QC" + "3"                        -> M=3, QC=3
    """
    roles = [r.upper() for r in _split_slash(_clean_text(role_text or ""))]
    if not roles:
        return {}, "担当職種 blank"

    levels = []
    for tok in _split_slash(skill_raw or ""):
        try:
            levels.append(int(float(tok)))
        except ValueError:
            levels.append(None)
    if not levels:
        return {}, "スキルレベル blank"
    if all(lv is None for lv in levels):
        return {}, f"スキルレベル unreadable ({skill_raw!r})"
    if all((lv or 0) == 0 for lv in levels):
        return {}, "スキルレベル 0"

    if len(levels) == 1:
        per_role = [levels[0]] * len(roles)
    elif len(levels) == len(roles):
        per_role = levels
    else:
        # count mismatch (not seen in the data so far): pair by position,
        # reuse the last level for any extra roles
        per_role = [levels[min(i, len(levels) - 1)] for i in range(len(roles))]

    skill_map = {}
    for role, lv in zip(roles, per_role):
        if not lv or lv <= 0:
            continue
        for op in ROLE_TOKEN_OPS.get(role, ()):
            skill_map[op] = max(skill_map.get(op, 0), lv)
    skill_map = {op: skill_map[op] for op in TASK_OPS if op in skill_map}
    return skill_map, None


# ============================================================
# 製番 sheet parsing (replaces 新規製番リスト "CSV" sheet)
# ============================================================
# Columns (1-based), confirmed against both new Excel files:
#  1 製番 | 2 ユーザー | 3 Fab | 4 地域
#  5 第2工程 開始日 | 6 作業種別(M) | 7 工数(M) | 8 推奨人数(M)
#  9 作業種別(E) | 10 工数(E) | 11 推奨人数(E)
# 12 第3工程 開始日 | 13 作業種別(QC) | 14 工数 | 15 推奨人数
# 16 第4工程 開始日 | 17 作業種別(QC) | 18 工数 | 19 推奨人数
# 20 希望納期

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


def _read_seiban_sheet(path, start_row=3):
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
            "p2_start": _as_timestamp(cell("p2_start")),
            "p2m_headcount": _num_or_none(cell("p2m_headcount")),
            "p2e_headcount": _num_or_none(cell("p2e_headcount")),
            "p3_start": _as_timestamp(cell("p3_start")),
            "p3_headcount": _num_or_none(cell("p3_headcount")),
            "p4_start": _as_timestamp(cell("p4_start")),
            "p4_headcount": _num_or_none(cell("p4_headcount")),
            "p4_kousu": _num_or_none(cell("p4_kousu")),
            "delivery": _as_timestamp(cell("delivery")),
            "delivery_note": cell("delivery") if isinstance(cell("delivery"), str) else None,
        }
    return rows


def parse_seiban_merged(base_path, r_path, plan_start: pd.Timestamp, plan_end: pd.Timestamp, su_code_span: dict | None = None,
                        start_row=3):
    """
    Returns the same shape as decoder5's parse_tasks_from_csv_v5():
    {"valid_codes", "planned_meta", "cut_rows", "date_list"}, so the rest of
    the pipeline (build_shifted_meta etc.) is unchanged.

    SU_Others is the main source of truth, same philosophy as decoder5's
    "SU_Others provides actual execution span". 製番's p2/p3/p4 start dates
    are only used as a *planned reference* (proportions, ratio checks) — if
    they're missing or out of order, but the module code has real occurrences
    in SU_Others (`su_code_span`), the module is NOT dropped: a nominal
    evenly-split "planned" window is built from SU_Others' own actual span
    instead, and the rest of the pipeline (which shifts onto the real worked
    days regardless) takes it from there. A module is only dropped entirely
    (treated as dummy/not-listed) when 製番 has nothing usable AND SU_Others
    has no occurrences of it either — genuinely no data anywhere.

    There is no plan-range containment check here anymore: a module starting
    before plan_start is not dummied, it's kept and its already-happened
    portion is marked plan_flexibility="Fixed" downstream (build_assignments_v6),
    with only the portion at/after plan_start left "Flexible" for the
    scheduler.

    希望納期 (delivery) is handled separately from p2/p3/p4: in practice it is
    blank in essentially every row of both files (0/60 in the sample data),
    unlike p2/p3/p4 start (26/60 filled) — so treating a missing delivery the
    same as missing start dates would dummy out every module. When missing
    (or earlier than p4_start), it defaults to plan_end.
    """
    su_code_span = su_code_span or {}
    base_rows = _read_seiban_sheet(base_path, start_row)
    r_rows = _read_seiban_sheet(r_path, start_row)
    all_codes = sorted(set(base_rows) | set(r_rows))

    planned_meta = {}
    cut_rows = []
    all_dates = []

    for code in all_codes:
        b = base_rows.get(code, {})
        rr = r_rows.get(code, {})

        def pick(key):
            v = rr.get(key)
            if v is None or (isinstance(v, str) and not v.strip()):
                v = b.get(key)
            return v

        customer = pick("customer")
        customer = str(customer).strip() if isinstance(customer, str) and str(customer).strip() else "OTHER"
        fab_name = pick("fab")
        fab_name = str(fab_name).strip() if isinstance(fab_name, str) and str(fab_name).strip() else "Other"
        country = pick("region")
        country = str(country).strip() if isinstance(country, str) and str(country).strip() else "Other"

        note = pick("delivery_note")
        if isinstance(note, str) and ("間違" in note or "まちがい" in note):
            cut_rows.append((code, f"DUMMY: 希望納期 says {note.strip()!r} (wrong 製番) -> not a target"))
            continue

        p2s, p3s, p4s, deliv = pick("p2_start"), pick("p3_start"), pick("p4_start"), pick("delivery")

        complete = all(x is not None for x in (p2s, p3s, p4s))
        ordered = complete and (p2s <= p3s <= p4s)

        if not ordered:
            span = su_code_span.get(code)
            reason = "missing p2/p3/p4" if not complete else "p2/p3/p4 out of order"
            if span is None:
                cut_rows.append((code, f"DUMMY: {reason}, no SU_Others data either"))
                continue
            actual_start, actual_end = span
            total_days = int((actual_end - actual_start).days) + 1
            alloc = _allocate_phase_lengths_v5(total_days, {2: 1, 3: 1, 4: 1}, phase_ids=(2, 3, 4), min_one=(total_days >= 3))
            p2s = actual_start
            p3s = p2s + pd.Timedelta(days=int(alloc[2]))
            p4s = p3s + pd.Timedelta(days=int(alloc[3]))
            deliv = actual_end
            cut_rows.append((code, f"NOTE: {reason}; using SU_Others span {_to_ymd(actual_start)}-{_to_ymd(actual_end)} as planned reference"))

        if deliv is None or deliv < p4s:
            reason = "missing" if deliv is None else "before p4_start"
            deliv = max(p4s, plan_end)
            cut_rows.append((code, f"NOTE: delivery {reason}; defaulted to {_to_ymd(deliv)}"))

        overall_start = p2s
        overall_end = deliv

        starts = {2: p2s, 3: p3s, 4: p4s}
        ends = {
            2: (p3s - pd.Timedelta(days=1)).normalize(),
            3: (p4s - pd.Timedelta(days=1)).normalize(),
            4: deliv,
        }
        for ph in (2, 3, 4):
            if ends[ph] < starts[ph]:
                ends[ph] = starts[ph]

        phase_len = {ph: int((ends[ph] - starts[ph]).days) + 1 for ph in (2, 3, 4)}

        # 希望納期 (delivery) is a customer deadline, not a tight "phase-4 end"
        # like the old sheet's p4終了予定日 was — it can sit hundreds of days
        # past p4's start. Used raw, that blows up phase4's weight and skews
        # every ratio-based heuristic downstream (e.g. a module with a fine,
        # short real worked span gets wrongly judged "too short vs. plan").
        # overall_end / ends[4] still show the true delivery date; only the
        # *proportion* used for splitting/ratio checks is capped.
        cap4 = phase_len[2] + phase_len[3]
        if cap4 <= 0:
            cap4 = phase_len[4]
        if phase_len[4] > cap4:
            phase_len[4] = max(cap4, 1)

        total_len = sum(phase_len.values())
        phase_pct = {ph: (phase_len[ph] / total_len) for ph in (2, 3, 4)}

        planned_meta[code] = {
            "customer": customer, "country": country, "fab_name": fab_name,
            "starts": starts, "ends": ends, "phase_len": phase_len, "phase_pct": phase_pct,
            "overall_start": overall_start, "overall_end": overall_end, "total_len": total_len,
            "p2m_headcount": pick("p2m_headcount"), "p2e_headcount": pick("p2e_headcount"),
            "p3_headcount": pick("p3_headcount"), "p4_headcount": pick("p4_headcount"),
            "p4_kousu": pick("p4_kousu"),
            "seiban_dates": bool(ordered),  # p2/p3/p4 start really come from 製番
        }
        all_dates.extend([overall_start, overall_end])

    return {
        "valid_codes": sorted(planned_meta.keys()), "planned_meta": planned_meta,
        "cut_rows": cut_rows, "date_list": all_dates,
    }


# ============================================================
# Shifting / rescaling plan onto SU_Others actual span (unchanged)
# ============================================================

def _allocate_phase_lengths_v5(actual_total_days, planned_phase_len, phase_ids=(2, 3, 4), min_one=True):
    phs = list(phase_ids)
    if actual_total_days <= 0:
        return {ph: 0 for ph in phs}
    total_planned = sum(max(0, int(planned_phase_len.get(ph, 0))) for ph in phs)
    if total_planned <= 0:
        base = actual_total_days // len(phs)
        rem = actual_total_days - base * len(phs)
        out = {ph: base for ph in phs}
        for i in range(rem):
            out[phs[i % len(phs)]] += 1
    else:
        raw = {ph: (actual_total_days * (planned_phase_len.get(ph, 0) / total_planned)) for ph in phs}
        flo = {ph: int(math.floor(raw[ph])) for ph in phs}
        rem = actual_total_days - sum(flo.values())
        frac = sorted(phs, key=lambda ph: (raw[ph] - flo[ph]), reverse=True)
        out = dict(flo)
        for i in range(rem):
            out[frac[i % len(frac)]] += 1
    if min_one and actual_total_days >= len(phs):
        zeros = [ph for ph in phs if out.get(ph, 0) <= 0]
        for z in zeros:
            donors = sorted([ph for ph in phs if out.get(ph, 0) > 1], key=lambda ph: out[ph], reverse=True)
            if not donors:
                break
            d = donors[0]
            out[d] -= 1
            out[z] = 1
    drift = actual_total_days - sum(out.values())
    if phs:
        out[phs[-1]] = out.get(phs[-1], 0) + drift
    return out


def _find_qc_phase3_start_day(code, worked_days, code_occ, su_data, phase34_cap_days):
    worker_roles = su_data.get("worker_roles", {}) if su_data else {}
    worked_index = {d: i for i, d in enumerate(worked_days)}
    PURE_ME_STOP_GAP_TOLERANCE = 3

    def _rt(role_text):
        return _clean_text(role_text).upper()

    def _role_is_pure_qc(role_text):
        rt = _rt(role_text)
        return ("QC" in rt) and ("M" not in rt) and ("E" not in rt)

    def _role_is_pure_m(role_text):
        rt = _rt(role_text)
        return ("M" in rt) and ("QC" not in rt) and ("E" not in rt)

    def _role_is_pure_e(role_text):
        rt = _rt(role_text)
        return ("E" in rt) and ("QC" not in rt) and ("M" not in rt)

    by_wid = defaultdict(list)
    for dt, wid, _disp in sorted(code_occ.get(code, []), key=lambda x: (x[0], x[1])):
        if dt in worked_index:
            by_wid[wid].append(worked_index[dt])

    total_days = len(worked_days)
    latest_p2_end_idx = max(0, total_days - int(phase34_cap_days) - 1) if phase34_cap_days is not None else 0

    pure_qc_first_idx = None
    for wid, idxs in by_wid.items():
        role_text = worker_roles.get(wid, "")
        if not _role_is_pure_qc(role_text):
            continue
        first_idx = min(idxs)
        pure_qc_first_idx = first_idx if pure_qc_first_idx is None else min(pure_qc_first_idx, first_idx)

    pure_me_stop_idx = None
    for wid, idxs in by_wid.items():
        role_text = worker_roles.get(wid, "")
        if not (_role_is_pure_m(role_text) or _role_is_pure_e(role_text)):
            continue
        idxs = sorted(set(idxs))
        if not idxs:
            continue
        cluster_end = idxs[0]
        for i in range(1, len(idxs)):
            gap = idxs[i] - idxs[i - 1]
            if gap <= (PURE_ME_STOP_GAP_TOLERANCE + 1):
                cluster_end = idxs[i]
            else:
                break
        candidate = cluster_end + 1
        if candidate >= total_days:
            continue
        pure_me_stop_idx = candidate if pure_me_stop_idx is None else min(pure_me_stop_idx, candidate)

    if pure_qc_first_idx is not None:
        if pure_qc_first_idx == 0:
            if pure_me_stop_idx is not None:
                earliest_from_cap = max(1, latest_p2_end_idx + 1) if total_days >= 2 else 0
                phase3_start_idx = max(pure_me_stop_idx, earliest_from_cap)
                if total_days >= 2:
                    phase3_start_idx = min(phase3_start_idx, total_days - 1)
                return phase3_start_idx, f"pure QC from start, pure M/E stopped before {_to_ymd(worked_days[phase3_start_idx])}", phase3_start_idx
            return 0, "pure QC exists from first worked day and no pure M/E stop found", pure_qc_first_idx
        else:
            earliest_from_cap = max(1, latest_p2_end_idx + 1) if total_days >= 2 else 0
            phase3_start_idx = max(pure_qc_first_idx, earliest_from_cap)
            if total_days >= 2:
                phase3_start_idx = min(phase3_start_idx, total_days - 1)
            return phase3_start_idx, f"pure QC first joined on {_to_ymd(worked_days[pure_qc_first_idx])}", pure_qc_first_idx

    if pure_me_stop_idx is not None:
        earliest_from_cap = max(1, latest_p2_end_idx + 1) if total_days >= 2 else 0
        phase3_start_idx = max(pure_me_stop_idx, earliest_from_cap)
        if total_days >= 2:
            phase3_start_idx = min(phase3_start_idx, total_days - 1)
        return phase3_start_idx, f"pure M/E stopped before {_to_ymd(worked_days[phase3_start_idx])}", phase3_start_idx

    phase3_start_idx = latest_p2_end_idx + 1 if total_days >= 1 else 0
    if phase3_start_idx < 0:
        phase3_start_idx = 0
    if total_days >= 1:
        phase3_start_idx = min(phase3_start_idx, total_days)
    return phase3_start_idx, "no pure QC and no pure M/E stop found; fallback to cap split", None


def build_shifted_meta(planned_meta: dict, su_data, phase34_cap_days=None):
    orig_map = su_data.get("su_outlier_original_text", {}) if su_data else {}
    shifted_meta = {}
    code_to_shifted_phases = defaultdict(list)

    code_occ = defaultdict(list)
    if su_data is not None:
        for (wid, dt), text in su_data["worker_date_map"].items():
            code = extract_tool_code(text)
            if code:
                disp = text
                k = (wid, _to_ymd(dt))
                if k in orig_map:
                    disp = orig_map[k]["old"]
                code_occ[code].append((dt, wid, disp))

    def _split_tail_equal(tail_days, meta):
        if tail_days <= 0:
            return {3: 0, 4: 0}
        if tail_days == 1:
            return {3: 1, 4: 0}
        half = tail_days // 2
        rem = tail_days % 2
        p3, p4 = half, half
        if rem == 1:
            p3_len = int(meta.get("phase_len", {}).get(3, 0))
            p4_len = int(meta.get("phase_len", {}).get(4, 0))
            if p3_len >= p4_len:
                p3 += 1
            else:
                p4 += 1
        if tail_days >= 2:
            if p3 <= 0:
                p3, p4 = 1, tail_days - 1
            elif p4 <= 0:
                p4, p3 = 1, tail_days - 1
        return {3: p3, 4: p4}

    for code, meta in planned_meta.items():
        occ = code_occ.get(code, [])
        if occ:
            worked_days = sorted(set(x[0] for x in occ))
            actual_start = worked_days[0]
            actual_end = worked_days[-1]
            total_days = len(worked_days)

            phase3_start_idx, trigger_reason, qc_first_idx = _find_qc_phase3_start_day(
                code, worked_days, code_occ, su_data, phase34_cap_days
            )

            p2_days = phase3_start_idx if total_days >= 2 else total_days
            p2_days = max(0, min(p2_days, total_days))
            tail_days = max(0, total_days - p2_days)
            if phase34_cap_days is not None:
                tail_days = min(tail_days, int(phase34_cap_days))
            tail_alloc = _split_tail_equal(tail_days, meta)

            phase_days = {
                2: worked_days[:p2_days],
                3: worked_days[p2_days:p2_days + tail_alloc.get(3, 0)],
                4: worked_days[p2_days + tail_alloc.get(3, 0):p2_days + tail_alloc.get(3, 0) + tail_alloc.get(4, 0)],
            }

            shifted_starts, shifted_ends = {}, {}
            for ph in (2, 3, 4):
                ds = phase_days.get(ph, [])
                if ds:
                    shifted_starts[ph] = ds[0]
                    shifted_ends[ph] = ds[-1]
                else:
                    if ph == 2:
                        shifted_starts[ph] = actual_start
                        shifted_ends[ph] = actual_start
                    else:
                        anchor = shifted_ends.get(ph - 1, actual_start)
                        shifted_starts[ph] = anchor
                        shifted_ends[ph] = anchor

            alloc_worked = {ph: len(phase_days.get(ph, [])) for ph in (2, 3, 4)}

            shifted_meta[code] = {
                "plan": meta, "had_su_match": True,
                "actual_first": actual_start, "actual_last": actual_end, "actual_total": total_days,
                "alloc_span_days": dict(alloc_worked), "alloc_worked_days": alloc_worked,
                "phase_days": phase_days, "shifted_starts": shifted_starts, "shifted_ends": shifted_ends,
                "occ_sample": sorted(occ, key=lambda x: x[0])[:3],
                "occ_last_sample": sorted(occ, key=lambda x: x[0])[-3:],
                "phase3_trigger_reason": trigger_reason,
                "qc_first_join": worked_days[qc_first_idx] if qc_first_idx is not None else None,
                "phase34_cap_days": phase34_cap_days,
            }
        else:
            shifted_meta[code] = {
                "plan": meta, "had_su_match": False,
                "actual_first": None, "actual_last": None, "actual_total": None,
                "alloc_span_days": None, "alloc_worked_days": None, "phase_days": None,
                "shifted_starts": {2: meta["starts"][2], 3: meta["starts"][3], 4: meta["starts"][4]},
                "shifted_ends": {2: meta["ends"][2], 3: meta["ends"][3], 4: meta["ends"][4]},
                "occ_sample": [], "occ_last_sample": [],
                "phase3_trigger_reason": "no SU_Others match", "qc_first_join": None,
                "phase34_cap_days": phase34_cap_days,
            }

        for ph in (2, 3, 4):
            ds = None
            if shifted_meta[code].get("had_su_match") and shifted_meta[code].get("phase_days"):
                ds = set(shifted_meta[code]["phase_days"][ph])
            code_to_shifted_phases[code].append({
                "phase_index": ph, "start": shifted_meta[code]["shifted_starts"][ph],
                "end": shifted_meta[code]["shifted_ends"][ph], "operation": f"p{ph}", "date_set": ds,
            })

    return shifted_meta, code_to_shifted_phases, code_occ


# ============================================================
# Build tool tasks (Schedule.yaml workflow_task_list) + code_to_phases
# ============================================================

def _role_flags(role_text: str):
    rt = _clean_text(role_text).upper()
    return ("M" in rt), ("E" in rt), ("QC" in rt)


def build_tool_tasks(task_meta, shifted_meta):
    """
    New schema: single workflow "wf_tool" per module, phases p2/p3/p4 with
    ids e{n}p2/e{n}p3/e{n}p4; p2 has TWO operations (Mech=o1, Elec=o2), p3/p4
    have one (QC=o1). IDs follow e{n}p{ph}o{k} (no underscore), matching
    GanttChartEditor's current convention.
    """
    tool_tasks = []
    code_to_phases = defaultdict(list)
    all_dates = []
    task_counter = 1

    OP_NAME = {(2, 1): "Mech", (2, 2): "Elec", (3, 1): "QC", (4, 1): "QC"}
    PHASE_NAME = {2: "Hardware Setup", 3: "Function Setup", 4: "Acceptance Inspection"}

    for code in task_meta["valid_codes"]:
        meta = shifted_meta.get(code)
        if not meta:
            continue
        plan = meta["plan"]
        customer, country, fab_name = plan["customer"], plan["country"], plan["fab_name"]

        task_id = f"e{task_counter}"
        task_counter += 1

        phase_task_list = []
        for ph in (2, 3, 4):
            phase_id = f"{task_id}p{ph}"
            start = meta["shifted_starts"][ph]
            end = meta["shifted_ends"][ph]
            if end < start:
                end = start

            op_ks = (1, 2) if ph == 2 else (1,)
            operation_task_list = []
            for k in op_ks:
                operation_task_list.append({
                    "id": f"{phase_id}o{k}", "name": OP_NAME[(ph, k)], "operation": f"p{ph}o{k}",
                    "workload_hours": 0,  # overwritten later from real assignment data
                })

            phase_task_list.append({
                "id": phase_id, "name": PHASE_NAME[ph], "phase": f"p{ph}",
                "start_date": _to_ymd(start), "end_date": _to_ymd(end),
                "operation_task_list": operation_task_list,
            })

            ds = None
            if meta.get("had_su_match") and meta.get("phase_days"):
                ds = set(meta["phase_days"].get(ph, []))
            code_to_phases[code].append({
                "phase_index": ph, "phase_id": phase_id, "start": start, "end": end, "date_set": ds,
            })
            all_dates.extend([start, end])

        tool_tasks.append({
            "id": task_id, "name": code, "workflow": "wf_tool", "fab": None,
            "phase_task_list": phase_task_list,
            "module_code": code, "customer": customer, "country": country, "fab_name": fab_name,
        })

    return tool_tasks, code_to_phases, all_dates


# ============================================================
# Build assignments (Schedule.yaml assignment_list + misc_task_list)
# ============================================================

def build_assignments_v7(su_data, code_to_phases, valid_code_set, plan_start=None, date_filter=None,
                         fixed_wids=frozenset()):
    """
    decoder6's build_assignments_v6, plus (decoder7):
      - workers in `fixed_wids` (not a target of auto-planning: 担当職種 blank
        or スキルレベル 0) keep their real tool-task assignments, but always
        as plan_flexibility="Fixed".
      - every operation task / misc task gets a color_code: the most common
        SU_Others cell fill colour among the cells assigned to it.
      - "other work" misc tasks have no workflow; grey cells use
        "wf_personal_business".

    decoder6 notes — same "known tool task vs everything else" split as
    decoder5's build_assignments_v5, but:
      - phase 2 attributes each worker's days to Mech (o1) and/or Elec (o2)
        operation based on that worker's role text (M -> o1, E -> o2; if
        neither is present, default to Mech so the day isn't silently lost);
        phase 3/4 always use o1 (QC).
      - "other" SU_Others labels and grey-cell "personal business" both become
        flat misc_task_list entries (no phase/operation wrapper) per the new
        schema; assignments reference the misc task's own id directly.
      - a phase whose real (shifted) start date is before plan_start is
        already underway/complete relative to the plan range: every
        assignment in that phase is marked plan_flexibility="Fixed" (already
        happened, not up for the scheduler to move) instead of "Flexible".
        Phases at/after plan_start stay "Flexible".
    """
    orig_map = su_data.get("su_outlier_original_text", {})
    worker_date_map = su_data["worker_date_map"]
    worker_personal_map = su_data["worker_personal_map"]
    worker_roles = su_data.get("worker_roles", {})
    worker_color_map = su_data.get("worker_color_map", {})
    f_start, f_end = date_filter if date_filter else (None, None)

    # task id / label -> Counter of cell colours, resolved to color_code below
    op_colors = defaultdict(Counter)
    misc_colors = defaultdict(Counter)
    pb_colors = defaultdict(Counter)

    def _count_color(counter_map, key, wid, dt):
        color = worker_color_map.get((wid, dt))
        if color:
            counter_map[key][color] += 1

    # Per-operation Fixed/Flexible: an operation's phase already started
    # before the plan range -> Fixed for every worker on it.
    op_is_fixed = {}
    for phase_list in code_to_phases.values():
        for phase_meta in phase_list:
            phase_id = phase_meta["phase_id"]
            phase_start = phase_meta.get("start")
            is_fixed = bool(plan_start is not None and phase_start is not None and phase_start < plan_start)
            op_ks = (1, 2) if phase_meta["phase_index"] == 2 else (1,)
            for k in op_ks:
                op_is_fixed[f"{phase_id}o{k}"] = is_fixed

    known_assign_map = defaultdict(list)   # (wid, operation_task_id) -> [dt,...]
    misc_label_dates = defaultdict(set)
    misc_worker_label_dates = defaultdict(list)
    personal_label_dates = defaultdict(set)
    personal_worker_label_dates = defaultdict(list)
    pb_norm_to_display = {}
    dummy_tool_labels = defaultdict(set)

    for (wid, dt), raw_text in worker_date_map.items():
        if f_start is not None and f_end is not None and (dt < f_start or dt > f_end):
            continue

        internal_text = raw_text
        display_text = raw_text
        k = (wid, _to_ymd(dt))
        if k in orig_map:
            display_text = orig_map[k]["old"]
            internal_text = orig_map[k]["new"]

        code = extract_tool_code(internal_text)

        if code and (code in valid_code_set) and (code in code_to_phases):
            matched = False
            for phase_meta in code_to_phases[code]:
                ds = phase_meta.get("date_set")
                in_window = (dt in ds) if ds is not None else (
                    phase_meta["start"] is not None and phase_meta["end"] is not None
                    and phase_meta["start"] <= dt <= phase_meta["end"]
                )
                if in_window:
                    phase_id = phase_meta["phase_id"]
                    ph = phase_meta["phase_index"]
                    if ph == 2:
                        has_m, has_e, _ = _role_flags(worker_roles.get(wid, ""))
                        ops = []
                        if has_m:
                            ops.append(f"{phase_id}o1")
                        if has_e:
                            ops.append(f"{phase_id}o2")
                        if not ops:
                            ops = [f"{phase_id}o1"]  # unclassified role -> default to Mech
                    else:
                        ops = [f"{phase_id}o1"]
                    for op_id in ops:
                        known_assign_map[(wid, op_id)].append(dt)
                        _count_color(op_colors, op_id, wid, dt)
                    matched = True
                    break
            if matched:
                continue

        label = display_text.strip() if isinstance(display_text, str) else ""
        if not label:
            label = "other"
        misc_label_dates[label].add(dt)
        misc_worker_label_dates[(wid, label)].append(dt)
        _count_color(misc_colors, label, wid, dt)

        code_disp = extract_tool_code(display_text)
        if code_disp and code_disp not in valid_code_set:
            dummy_tool_labels[code_disp].add(label)

    for (wid, dt), text in worker_personal_map.items():
        if f_start is not None and f_end is not None and (dt < f_start or dt > f_end):
            continue
        display_label = text if isinstance(text, str) else ""
        norm_label = _clean_text(display_label).strip()
        if norm_label not in pb_norm_to_display:
            pb_norm_to_display[norm_label] = display_label
        personal_label_dates[norm_label].add(dt)
        personal_worker_label_dates[(wid, norm_label)].append(dt)
        _count_color(pb_colors, norm_label, wid, dt)

    assignments = []
    op_workerday_count = defaultdict(int)
    op_date_worker_set = defaultdict(lambda: defaultdict(set))
    tmp_op_to_worker_dates = defaultdict(set)
    for (wid, op_id), dates in known_assign_map.items():
        for d in dates:
            tmp_op_to_worker_dates[(op_id, wid)].add(d)
    for (op_id, wid), dset in tmp_op_to_worker_dates.items():
        op_workerday_count[op_id] += len(dset)
        for d in dset:
            op_date_worker_set[op_id][d].add(wid)

    op_worker_count = {}
    op_assigned_date_count = {}
    for op_id, date_map in op_date_worker_set.items():
        if not date_map:
            op_worker_count[op_id] = 0
            op_assigned_date_count[op_id] = 0
        else:
            op_worker_count[op_id] = max(len(wset) for wset in date_map.values())
            op_assigned_date_count[op_id] = len(date_map)

    for (wid, op_id), dates in known_assign_map.items():
        uniq_dates = sorted(set(dates))
        if not uniq_dates:
            continue
        work_date_list = [{"hour": HOURS_PER_WORKDAY, "date": _to_ymd(d)} for d in uniq_dates]
        assignments.append({
            "worker": wid, "operation_task": op_id,
            "start_date": _to_ymd(uniq_dates[0]), "end_date": _to_ymd(uniq_dates[-1]),
            "work_date_list": work_date_list,
            "plan_flexibility": "Fixed" if (op_is_fixed.get(op_id) or wid in fixed_wids) else "Flexible",
        })

    # misc: "other" SU_Others labels (flat, no phases)
    misc_label_workerday = defaultdict(int)
    tmp_label_wid_dates = defaultdict(set)
    for (wid, label), dates in misc_worker_label_dates.items():
        for d in set(dates):
            tmp_label_wid_dates[label].add((wid, d))
    for label, s in tmp_label_wid_dates.items():
        misc_label_workerday[label] = len(s)

    misc_tasks = []
    misc_label_to_id = {}
    misc_counter = 1
    for label in sorted(misc_label_dates.keys()):
        dates = misc_label_dates[label]
        if not dates:
            continue
        task_id = f"misc_other_{misc_counter}"
        misc_counter += 1
        misc_label_to_id[label] = task_id
        misc_tasks.append({
            "id": task_id, "name": label, "description": None,
            "workflow": None, "region": "r_other", "color_code": _top_color(misc_colors.get(label)),
        })

    for (wid, label), dates in misc_worker_label_dates.items():
        uniq_dates = sorted(set(dates))
        if not uniq_dates:
            continue
        task_id = misc_label_to_id.get(label)
        if not task_id:
            continue
        work_date_list = [{"hour": HOURS_PER_WORKDAY, "date": _to_ymd(d)} for d in uniq_dates]
        assignments.append({
            "worker": wid, "operation_task": task_id,
            "start_date": _to_ymd(uniq_dates[0]), "end_date": _to_ymd(uniq_dates[-1]),
            "work_date_list": work_date_list, "plan_flexibility": "Fixed",
        })

    # misc: personal business (grey empty cells) — same flat shape
    pb_label_to_id = {}
    pb_counter = 1
    for norm_label in sorted(personal_label_dates.keys()):
        dates = personal_label_dates[norm_label]
        if not dates:
            continue
        task_id = f"misc_pb_{pb_counter}"
        pb_counter += 1
        pb_label_to_id[norm_label] = task_id
        display_label = pb_norm_to_display.get(norm_label, "") or "Personal Business"
        misc_tasks.append({
            "id": task_id, "name": display_label, "description": None,
            "workflow": "wf_personal_business", "region": "r_other", "color_code": _top_color(pb_colors.get(norm_label)),
        })

    pb_worker_dates = defaultdict(list)
    for (wid, dt), _ in worker_personal_map.items():
        if f_start is not None and f_end is not None and (dt < f_start or dt > f_end):
            continue
        pb_worker_dates[wid].append(dt)
    for wid in list(pb_worker_dates.keys()):
        pb_worker_dates[wid] = sorted(set(pb_worker_dates[wid]))

    for (wid, norm_label), dates in personal_worker_label_dates.items():
        uniq_dates = sorted(set(dates))
        if not uniq_dates:
            continue
        task_id = pb_label_to_id.get(norm_label)
        if not task_id:
            continue
        work_date_list = [{"hour": HOURS_PER_WORKDAY, "date": _to_ymd(d)} for d in uniq_dates]
        assignments.append({
            "worker": wid, "operation_task": task_id,
            "start_date": _to_ymd(uniq_dates[0]), "end_date": _to_ymd(uniq_dates[-1]),
            "work_date_list": work_date_list, "plan_flexibility": "Fixed",
        })

    op_color_code = {op_id: _top_color(c) for op_id, c in op_colors.items()}

    return (assignments, misc_tasks, op_workerday_count, op_worker_count,
            op_assigned_date_count, pb_worker_dates, dummy_tool_labels, op_color_code)


def _top_color(counter):
    """Most common colour in a Counter (ties -> alphabetically first), or None."""
    if not counter:
        return None
    return min(counter.items(), key=lambda kv: (-kv[1], kv[0]))[0]


# ============================================================
# Transformation log
# ============================================================

def _format_phase_line(ph, start, end, extra=""):
    s = _to_ymd(start) if start is not None else "N/A"
    e = _to_ymd(end) if end is not None else "N/A"
    return f"  - P{ph}: {s} - {e} {extra}" if extra else f"  - P{ph}: {s} - {e}"


SEIBAN_HEADCOUNT_KEYS = {"p2o1": "p2m_headcount", "p2o2": "p2e_headcount",
                         "p3o1": "p3_headcount", "p4o1": "p4_headcount"}


def predict_operation_workdays(meta):
    """Predicted worker-days per operation: 推奨人数 x PRED_WD_PER_PERSON."""
    return {op: float(meta.get(key) or 0) * PRED_WD_PER_PERSON for op, key in SEIBAN_HEADCOUNT_KEYS.items()}


def _prediction_log_header(shifted_meta, predicted_codes, seiban_planned):
    lines = [f"method: worker-days = 推奨人数 x {PRED_WD_PER_PERSON:g} (= 工数 60 / 推奨人数 2), "
             f"hours = worker-days x {HOURS_PER_WORKDAY}; workload = max(actual SU_Others, predicted)"]
    for code in sorted(shifted_meta):
        m = seiban_planned.get(code, {})
        kind = "製番 dates + prediction (cut by actual-data rules)" if code in predicted_codes else "SU_Others actual + prediction"
        k4 = m.get("p4_kousu")
        lines.append(f"module {code}: {kind}; 推奨人数 M/E/P3/P4 = {m.get('p2m_headcount')}/{m.get('p2e_headcount')}/"
                     f"{m.get('p3_headcount')}/{m.get('p4_headcount')}; 製番 第4工程 工数 = {k4 if k4 is not None else '-'}")
    return lines


def _worker_skill_log_lines(worker_list, su_data, excluded_workers):
    roles = su_data.get("worker_skill_role", {})
    skill_raw = su_data.get("worker_skill_raw", {})
    status = su_data.get("worker_status", {})
    lines = []
    for w in worker_list:
        wid = w["id"]
        head = (f"- {wid}({w['name']}) 担当職種={roles.get(wid, '') or '-'} "
                f"スキルレベル={skill_raw.get(wid, '') or '-'} "
                f"Status={status.get(wid, '') or '-'}")
        if wid in excluded_workers:
            lines.append(f"{head} -> NOT A PLANNING TARGET ({excluded_workers[wid]}); assignments kept as Fixed")
        else:
            sm = ", ".join(f"{k}:{v}" for k, v in w["skill_map"].items()) or "(no task operation)"
            types = sorted(set(w["worker_type_by_operation"].values())) or ["-"]
            lines.append(f"{head} -> {sm} [{'/'.join(types)}]")
    return lines


def write_transformation_log(out_path, cut_rows, shifted_meta, worker_id_to_name, workload_zero_ops,
                              dummy_tool_labels, su_outlier_corrections, outlier_cut_summary, pb_worker_dates,
                              worker_skill_lines=(), prediction_lines=()):
    lines = ["Decoder7 Transformation Log", ""]

    lines.append("---------------------- WORKLOAD PREDICTION (target 新規製番) ----------------------")
    lines.extend(prediction_lines or ["(none)"])
    lines.append("")

    lines.append("---------------------- WORKER SKILL / STATUS (skill sheet: 担当職種 + スキルレベル + Status) ----------------------")
    lines.extend(worker_skill_lines or ["(none)"])
    lines.append("")

    lines.append("---------------------- CUT / DEFAULTED (from 製番) ----------------------")
    lines.append("(none)" if not cut_rows else "")
    for code, reason in cut_rows:
        lines.append(f"- {code}: {reason}")
    lines.append("")

    lines.append("---------------------- SU_OTHERS OUTLIER CELLS CUT ----------------------")
    if not su_outlier_corrections:
        lines.append("(none)")
    else:
        by_code = defaultdict(list)
        for rec in su_outlier_corrections:
            by_code[rec.get("code", "?")].append(rec)
        for code in sorted(by_code.keys()):
            lines.append(f"module: {code}")
            recs = sorted(by_code[code], key=lambda r: (r.get("date", ""), r.get("wid", "")))
            for rec in recs[:2000]:
                wid = rec.get("wid", "?")
                nm = worker_id_to_name.get(wid, wid)
                lines.append(f"  - {rec.get('date')} / {wid}({nm})")
                lines.append(f"      {rec.get('text')}")
                lines.append(f"      reason: {rec.get('reason')}")
            if len(recs) > 2000:
                lines.append(f"  ... ({len(recs)-2000} more)")
            lines.append("")
    lines.append("")

    lines.append("---------------------- SHIFTING DATE (plan -> actual) ----------------------")
    for code in sorted(shifted_meta.keys()):
        m = shifted_meta[code]
        plan = m["plan"]
        lines.append(f"module: {code}")
        lines.append("planned (製番):")
        total_len = plan["total_len"]
        for ph in (2, 3, 4):
            pl = plan["phase_len"][ph]
            pct = plan["phase_pct"][ph] * 100.0
            lines.append(_format_phase_line(ph, plan["starts"][ph], plan["ends"][ph], extra=f"(len={pl}d, {pct:.1f}%)"))
        lines.append(f"  planned overall: {_to_ymd(plan['overall_start'])} - {_to_ymd(plan['overall_end'])} (total={total_len}d)")
        if m["had_su_match"]:
            lines.append("shifted result (used in Schedule.yaml):")
            alloc_worked = m.get("alloc_worked_days")
            for ph in (2, 3, 4):
                extra = f"worked_in_phase={alloc_worked.get(ph, 0)}d" if alloc_worked else ""
                lines.append(_format_phase_line(ph, m["shifted_starts"][ph], m["shifted_ends"][ph], extra=extra))
            trig = m.get("phase3_trigger_reason")
            if trig:
                lines.append(f"  phase3 trigger: {trig}")
            lines.append(f"  shifted overall: {_to_ymd(m['actual_first'])} - {_to_ymd(m['actual_last'])} (worked_total={m.get('actual_total')}d)")
        else:
            lines.append("actual (SU_Others): NOT FOUND -> no shift (kept planned/defaulted dates)")
        lines.append("")

    lines.append("---------------------- WORKLOAD WARNING (worker-days == 0) ----------------------")
    if not workload_zero_ops:
        lines.append("(none)")
    else:
        for op_id, mod in workload_zero_ops:
            lines.append(f"- operation_task: {op_id} / module: {mod} (no assigned worker-days in SU_Others after shifting)")
    lines.append("")

    lines.append("---------------------- DUMMY MODULES (SU_Others tool-code not in 製番 -> misc) ----------------------")
    if not dummy_tool_labels:
        lines.append("(none)")
    else:
        for code in sorted(dummy_tool_labels.keys()):
            labels = sorted(dummy_tool_labels[code])
            lines.append(f"- {code}:")
            for lb in labels[:50]:
                lines.append(f"    - {lb}")
            if len(labels) > 50:
                lines.append(f"    ... ({len(labels)-50} more)")
    lines.append("")

    lines.append("---------------------- SU_Others outlier-cut modules ----------------------")
    if not outlier_cut_summary:
        lines.append("(none)")
    else:
        for code in sorted(outlier_cut_summary.keys()):
            lines.append(f"- {code}:")
            for txt in outlier_cut_summary[code][:20]:
                lines.append(f"    - {txt}")
    lines.append("")

    lines.append("---------------------- PERSONAL BUSINESS (grey empty cells) ----------------------")
    if not pb_worker_dates:
        lines.append("(none)")
    else:
        for wid in sorted(pb_worker_dates.keys()):
            nm = worker_id_to_name.get(wid, wid)
            dts = pb_worker_dates[wid]
            lines.append(f"- {wid}({nm}) : {len(dts)} days")
    lines.append("")

    Path(out_path).write_text("\n".join(lines), encoding="utf-8")


# ============================================================
# Main build
# ============================================================

def build_env_and_schedule_decoder7(
    su_others_path, seiban_info_path,
    envconfig_out="EnvConfig.yaml", schedule_out="Schedule.yaml", log_out=TRANSFORMATION_LOG,
    plan_start=None, plan_end=None, su_sheet_names=("予定表_2026",),
    skill_sheet="予定表_2026 (2)", seiban_start_row=SEIBAN_START_ROW, phase34_cap_days=None,
):
    # 1) SU_Others actual work data (need this first to get a natural plan range fallback)
    su_data = parse_su_others(su_others_path, sheet_names=su_sheet_names, skill_sheet=skill_sheet)

    natural_start = _as_timestamp(su_data["plan_range"]["start_date"])
    natural_end = _as_timestamp(su_data["plan_range"]["end_date"])
    resolved_plan_start = plan_start or natural_start
    resolved_plan_end = plan_end or natural_end
    if resolved_plan_start is None or resolved_plan_end is None:
        raise RuntimeError("Could not resolve a plan range: pass --plan-start/--plan-end explicitly.")
    if resolved_plan_end < resolved_plan_start:
        resolved_plan_start, resolved_plan_end = resolved_plan_end, resolved_plan_start

    # 1.5) raw code -> (earliest, latest) occurrence in SU_Others, BEFORE any
    # outlier cleanup — used so parse_seiban_merged can recognize "this
    # module has no usable 製番 dates, but SU_Others is the main source and
    # it does have real data for it" instead of dummying it.
    su_code_dates = defaultdict(list)
    for (_wid, _dt), _text in su_data["worker_date_map"].items():
        _code = extract_tool_code(_text)
        if _code:
            su_code_dates[_code].append(_dt)
    su_code_span = {code: (min(dts), max(dts)) for code, dts in su_code_dates.items()}

    # 2) 製番 = the target modules; SU_Others fills in for modules missing 製番 dates.
    #    Single file in decoder7 (no base/_r pair), so it is passed as the "_r" file.
    #    Only 製番 rows >= seiban_start_row are targets; codes above it fall
    #    through to "other" misc tasks like any code not in 製番.
    task_meta = parse_seiban_merged(None, seiban_info_path, resolved_plan_start, resolved_plan_end,
                                    su_code_span=su_code_span, start_row=seiban_start_row)
    planned_meta = task_meta["planned_meta"]
    cut_rows = list(task_meta["cut_rows"])
    valid_code_set = set(planned_meta.keys())
    if PREDICT_TARGET_WORKLOAD:
        for code in sorted(planned_meta):
            blank = [op for op, key in SEIBAN_HEADCOUNT_KEYS.items() if not planned_meta[code].get(key)]
            if blank:
                cut_rows.append((code, f"DUMMY: 推奨人数 blank for {', '.join(blank)} -> misc task"))
                planned_meta.pop(code)
        valid_code_set = set(planned_meta.keys())
        task_meta["valid_codes"] = [c for c in task_meta["valid_codes"] if c in valid_code_set]
    seiban_planned = dict(planned_meta)  # every target module, before any cut

    # 3) SU_Others outlier cleanup pipeline (unchanged from decoder5/6)
    su_outlier_corrections = cut_su_outlier_cells(
        su_data, cluster_gap_days=7, far_gap_days=60, small_cluster_max_unique_days=7,
        cut_module_if_unique_days_lt=4, cut_module_if_total_cells_lt=4,
        planned_meta=planned_meta, cut_if_far_from_planned_days=90,
    )
    su_short_span_corrections = cut_su_short_span_modules_to_dummy(
        su_data, min_unique_worked_days=MIN_WORKED_DAYS_FOR_TOOL, planned_meta=planned_meta,
    )
    su_remaining_ratio_corrections = cut_module_if_remaining_dates_too_small_vs_planned(
        su_data, planned_meta=planned_meta, min_left_date_span_ratio=MIN_LEFT_DATE_SPAN_RATIO,
    )
    su_phase_zero_corrections = []
    if CUT_MODULE_IF_PHASE_ZERO_WORKLOAD:
        su_phase_zero_corrections = cut_module_if_phase_zero_workload(su_data, planned_meta=planned_meta)

    shifted_meta_placeholder = {}
    su_final_zero_phase_corrections = cut_final_zero_workload_modules_to_dummy(su_data, shifted_meta_placeholder)

    all_su_corrections = (
        su_outlier_corrections + su_short_span_corrections + su_remaining_ratio_corrections
        + su_phase_zero_corrections + su_final_zero_phase_corrections
    )
    outlier_cut_summary = defaultdict(list)
    for rec in all_su_corrections:
        code = rec.get("code")
        txt = rec.get("text")
        if code and txt and len(outlier_cut_summary[code]) < 20:
            outlier_cut_summary[code].append(txt)
    for code in sorted(outlier_cut_summary.keys()):
        cut_rows.append((code, "outlier cells converted to 'other'"))

    short_span_codes = {rec.get("code") for rec in su_short_span_corrections if rec.get("code")}
    if short_span_codes:
        for code in sorted(short_span_codes):
            cut_rows.append((code, f"DUMMY: worked days < {MIN_WORKED_DAYS_FOR_TOOL}"))
            planned_meta.pop(code, None)
        valid_code_set = set(planned_meta.keys())
        task_meta["valid_codes"] = [c for c in task_meta["valid_codes"] if c in valid_code_set]

    worker_company_list = su_data["worker_company_list"]
    worker_list = su_data["worker_list"]

    worker_id_to_name = {w["id"]: w["name"] for w in worker_list}

    # snapshot of the cleaned cells, before the module-level cuts below break
    # tool codes (no QC / zero phase); predicted modules get their cells back
    cells_before_module_cuts = dict(su_data["worker_date_map"])
    orig_keys_before_module_cuts = set(su_data.get("su_outlier_original_text", {}))

    # 6) shifted meta (real work -> phase windows) using SU_Others
    shifted_meta, code_to_shifted_phases, code_occ = build_shifted_meta(
        planned_meta, su_data, phase34_cap_days=phase34_cap_days,
    )

    su_no_qc_corrections = cut_modules_with_no_qc_to_dummy(su_data, shifted_meta)
    if su_no_qc_corrections:
        no_qc_codes = {rec.get("code") for rec in su_no_qc_corrections if rec.get("code")}
        for code in sorted(no_qc_codes):
            cut_rows.append((code, "DUMMY: no QC worker assigned"))
            planned_meta.pop(code, None)
            shifted_meta.pop(code, None)
        valid_code_set = set(planned_meta.keys())
        task_meta["valid_codes"] = [c for c in task_meta["valid_codes"] if c in valid_code_set]

    su_final_zero_phase_corrections = cut_final_zero_workload_modules_to_dummy(su_data, shifted_meta)
    if su_final_zero_phase_corrections:
        zero_codes = {rec.get("code") for rec in su_final_zero_phase_corrections if rec.get("code")}
        for code in sorted(zero_codes):
            cut_rows.append((code, "DUMMY: a phase has zero workload"))
            planned_meta.pop(code, None)
            shifted_meta.pop(code, None)
        valid_code_set = set(planned_meta.keys())
        task_meta["valid_codes"] = [c for c in task_meta["valid_codes"] if c in valid_code_set]

    if SKIP_MODULE_IF_NO_SU_MATCH:
        no_su_codes = [code for code, m in shifted_meta.items() if not m.get("had_su_match")]
        for code in no_su_codes:
            cut_rows.append((code, "SKIPPED: no SU_Others match"))
            planned_meta.pop(code, None)
            shifted_meta.pop(code, None)
        valid_code_set = set(planned_meta.keys())
        task_meta["valid_codes"] = [c for c in task_meta["valid_codes"] if c in valid_code_set]

    cut_due_to_distance = []
    for code, m in list(shifted_meta.items()):
        if not m.get("had_su_match"):
            continue
        plan = m["plan"]
        gap = _planned_actual_gap_days(plan["overall_start"], plan["overall_end"], m["actual_first"], m["actual_last"])
        if gap > CUT_DISTANCE_DAYS:
            cut_due_to_distance.append((code, gap))
    if cut_due_to_distance:
        for code, gap in cut_due_to_distance:
            cut_rows.append((code, f"DUMMY: actual span {gap}d outside planned window (> {CUT_DISTANCE_DAYS})"))
            planned_meta.pop(code, None)
            shifted_meta.pop(code, None)
        valid_code_set = set(planned_meta.keys())
        task_meta["valid_codes"] = [c for c in task_meta["valid_codes"] if c in valid_code_set]

    # 6.9) decoder7: a module is only kept if EVERY operation (p2o1, p2o2,
    #     p3o1, p4o1) gets real worker-days. Dry-run the assignment step; any
    #     module with a zero-workload operation is dropped, so its cells fall
    #     through to "other" misc tasks. Repeat until stable.
    while True:
        _tt, _c2p, _ = build_tool_tasks(task_meta, shifted_meta)
        _op_workerdays = build_assignments_v7(su_data, _c2p, valid_code_set, plan_start=resolved_plan_start)[2]
        zero_codes = {}
        for t in _tt:
            zero_ops = [ot["operation"] for pt in t["phase_task_list"] for ot in pt["operation_task_list"]
                        if not _op_workerdays.get(ot["id"])]
            if zero_ops:
                zero_codes[t["module_code"]] = zero_ops
        if not zero_codes:
            break
        for code in sorted(zero_codes):
            cut_rows.append((code, f"DUMMY: zero workload in {', '.join(zero_codes[code])} -> misc task"))
            planned_meta.pop(code, None)
            shifted_meta.pop(code, None)
        valid_code_set = set(planned_meta.keys())
        task_meta["valid_codes"] = [c for c in task_meta["valid_codes"] if c in valid_code_set]

    # 6.95) decoder7 prediction: target modules that were cut above (no QC,
    #      zero workload, too little SU_Others data, ...) are kept anyway when
    #      製番 has real p2/p3/p4 start dates — that's the normal case for a
    #      新規製番 that hasn't been staffed in SU_Others yet. Their windows come
    #      from 製番 (stretched to cover their remaining SU_Others cells), and
    #      their workload comes from the prediction (step 13).
    predicted_codes = set()
    if PREDICT_TARGET_WORKLOAD:
        restore_codes = sorted(c for c, m in seiban_planned.items()
                               if c not in planned_meta and m.get("seiban_dates"))
        orig_map = su_data.setdefault("su_outlier_original_text", {})
        for code in restore_codes:
            meta = seiban_planned[code]
            # give back the cells the module-level cuts broke
            cell_dates = []
            for (wid, dt), text in cells_before_module_cuts.items():
                if extract_tool_code(text) == code:
                    su_data["worker_date_map"][(wid, dt)] = text
                    k = (wid, _to_ymd(dt))
                    if k in orig_map and k not in orig_keys_before_module_cuts:
                        del orig_map[k]
                    cell_dates.append(dt)
            pred = predict_operation_workdays(meta)
            starts = dict(meta["starts"])
            p4_days = math.ceil(pred["p4o1"] / meta["p4_headcount"])
            ends = {2: meta["ends"][2], 3: meta["ends"][3],
                    4: starts[4] + pd.Timedelta(days=max(1, int(p4_days)) - 1)}
            if cell_dates:  # stretch the 製番 windows to cover existing SU_Others assignments
                starts[2] = min(starts[2], min(cell_dates))
                ends[4] = max(ends[4], max(cell_dates))
            planned_meta[code] = meta
            shifted_meta[code] = {
                "plan": meta, "had_su_match": False, "predicted": True,
                "actual_first": min(cell_dates) if cell_dates else None,
                "actual_last": max(cell_dates) if cell_dates else None, "actual_total": len(set(cell_dates)),
                "alloc_span_days": None, "alloc_worked_days": None, "phase_days": None,
                "shifted_starts": starts, "shifted_ends": ends,
                "occ_sample": [], "occ_last_sample": [],
                "phase3_trigger_reason": "predicted (製番 dates)", "qc_first_join": None,
                "phase34_cap_days": phase34_cap_days,
            }
            predicted_codes.add(code)
            cut_rows.append((code, "PREDICTED: kept anyway — 製番 dates + predicted workload "
                                   f"({len(set(cell_dates))} SU_Others day(s) kept as assignments)"))
        valid_code_set = set(planned_meta.keys())
        task_meta["valid_codes"] = [c for c in sorted(seiban_planned) if c in valid_code_set]

    # 7) build tool tasks (Schedule.yaml workflow_task_list) with new schema
    tool_tasks, code_to_phases, all_dates = build_tool_tasks(task_meta, shifted_meta)

    # 8) customer/region/fab lists
    customer_name_to_id = {"OTHER": "c_other"}
    region_name_to_id = {"Other": "r_other"}
    fab_name_to_id = {"Other": "f_other"}
    customer_company_list = [{"id": "c_other", "name": "OTHER", "unavailable_dates": []}]
    region_list = [{"id": "r_other", "name": "Other", "max_stay_on": 10000, "max_annual_stay": 10000,
                     "stay_off_interval": 3, "unavailable_dates": []}]
    fab_list = [{"id": "f_other", "name": "Other", "region": "r_other", "customer_company": "c_other", "unavailable_dates": []}]

    def get_customer_id(name):
        nm = name.strip() if isinstance(name, str) and name.strip() else "OTHER"
        if nm not in customer_name_to_id:
            cid = f"c{len(customer_name_to_id)}"
            customer_name_to_id[nm] = cid
            customer_company_list.append({"id": cid, "name": nm, "unavailable_dates": []})
        return customer_name_to_id[nm]

    def get_region_id(country):
        nm = country.strip() if isinstance(country, str) and country.strip() else "Other"
        if nm not in region_name_to_id:
            rid = f"r{len(region_name_to_id)}"
            region_name_to_id[nm] = rid
            region_list.append({"id": rid, "name": nm, "max_stay_on": 10000, "max_annual_stay": 10000,
                                 "stay_off_interval": 3, "unavailable_dates": []})
        return region_name_to_id[nm]

    def get_fab_id(fab_name, country, customer):
        nm = fab_name.strip() if isinstance(fab_name, str) and fab_name.strip() else "Other"
        if nm not in fab_name_to_id:
            fid = f"f{len(fab_name_to_id)}"
            fab_name_to_id[nm] = fid
            fab_list.append({"id": fid, "name": nm, "region": get_region_id(country),
                              "customer_company": get_customer_id(customer), "unavailable_dates": []})
        return fab_name_to_id[nm]

    for t in tool_tasks:
        t["fab"] = get_fab_id(t.get("fab_name"), t.get("country"), t.get("customer"))

    def build_transite_day_map(region_list, days_default=1):
        region_ids = [r["id"] for r in region_list if r.get("id")]
        out = []
        for fr in region_ids:
            for to in region_ids:
                if fr == to:
                    continue
                d = 0 if (fr == "r_other" and to == "r_other") else days_default
                out.append({"from": fr, "to": to, "days": d})
        return out

    transite_day_map = build_transite_day_map(region_list, days_default=1)

    # 9) environment
    environment = {
        "workflow_list": [
            {
                "id": "wf_tool", "name": "Tool Install",
                "phase_list": [
                    {"id": "p2", "name": "Hardware Setup", "operation_list": [
                        {"id": "p2o1", "name": "Mech", "work_hours": [HOURS_PER_WORKDAY], "min_worker_num": 1, "max_worker_num": DEFAULT_MAX_WORKER},
                        {"id": "p2o2", "name": "Elec", "work_hours": [HOURS_PER_WORKDAY], "min_worker_num": 1, "max_worker_num": DEFAULT_MAX_WORKER},
                    ]},
                    {"id": "p3", "name": "Function Setup", "operation_list": [
                        {"id": "p3o1", "name": "QC", "work_hours": [HOURS_PER_WORKDAY], "min_worker_num": 1, "max_worker_num": DEFAULT_MAX_WORKER},
                    ]},
                    {"id": "p4", "name": "Acceptance Inspection", "operation_list": [
                        {"id": "p4o1", "name": "QC", "work_hours": [HOURS_PER_WORKDAY], "min_worker_num": 1, "max_worker_num": DEFAULT_MAX_WORKER},
                    ]},
                ],
            },
        ],
        "fab_list": fab_list, "region_list": region_list, "customer_company_list": customer_company_list,
        "worker_company_list": worker_company_list, "transite_day_map": transite_day_map,
        # one affinity tag per worker company: same-company workers attract (weight 2)
        "affinity_tag": [{"id": "wct" + wc["id"][2:], "weight": 2} for wc in worker_company_list],
        "worker_list": worker_list,
    }

    # 10) skills (担当職種 + スキルレベル) and regular/spot (Status), per worker.
    #     Workers that are not planning targets get an empty skill_map, and
    #     their real assignments are kept but forced to Fixed (see step 11).
    worker_skill_role = su_data.get("worker_skill_role", {})
    worker_skill_raw = su_data.get("worker_skill_raw", {})
    worker_status = su_data.get("worker_status", {})
    in_skill_sheet = su_data.get("worker_in_skill_sheet", {})
    excluded_workers = {}  # wid -> reason
    for w in environment["worker_list"]:
        wid = w["id"]
        if in_skill_sheet.get(wid):
            skill_map, excluded_reason = parse_worker_skills(worker_skill_role.get(wid, ""), worker_skill_raw.get(wid, ""))
        else:
            skill_map, excluded_reason = {}, f"not in skill sheet {skill_sheet!r}"
        if excluded_reason:
            excluded_workers[wid] = excluded_reason
        w["skill_map"] = skill_map
        wtype = STATUS_TO_TYPE.get(str(worker_status.get(wid, "")).strip().upper(), "spot")
        w["worker_type_by_operation"] = {op: wtype for op in skill_map}
        w["affinity"] = ["wct" + w["worker_company"][2:]]

    # 11) assignments
    (assignments, misc_tasks, op_workerday_count, op_worker_count, op_assigned_date_count,
     pb_worker_dates, dummy_tool_labels, op_color_code) = build_assignments_v7(
        su_data, code_to_phases, valid_code_set, plan_start=resolved_plan_start,
        fixed_wids=frozenset(excluded_workers),
    )

    # 12) plan_range (explicit / resolved above)
    plan_range = {"start_date": _to_ymd(resolved_plan_start), "end_date": _to_ymd(resolved_plan_end)}

    tool_tasks_for_yaml = []
    for t in tool_tasks:
        t2 = dict(t)
        for k in ("module_code", "customer", "country", "fab_name"):
            t2.pop(k, None)
        tool_tasks_for_yaml.append(t2)

    op_id_to_module_code = {}
    for t in tool_tasks_for_yaml:
        mod = t.get("name", "")
        for pt in t.get("phase_task_list", []):
            for ot in pt.get("operation_task_list", []):
                op_id_to_module_code[ot["id"]] = mod

    # 13) workload_hours + recommends_worker_min/max, from real assignment data
    #     (falling back to 製番's 推奨人数 headcount where SU_Others gave nothing)
    headcount_by_op = {}
    for code, meta in shifted_meta.items():
        plan = meta.get("plan", {})
        task = next((t for t in tool_tasks if t.get("module_code") == code), None)
        if not task:
            continue
        eid = task["id"]
        headcount_by_op[f"{eid}p2o1"] = plan.get("p2m_headcount")
        headcount_by_op[f"{eid}p2o2"] = plan.get("p2e_headcount")
        headcount_by_op[f"{eid}p3o1"] = plan.get("p3_headcount")
        headcount_by_op[f"{eid}p4o1"] = plan.get("p4_headcount")

    pred_by_op = {}  # op_id -> (predicted worker-days, actual worker-days) for the log
    if PREDICT_TARGET_WORKLOAD:
        for t in tool_tasks:
            pred = predict_operation_workdays(seiban_planned.get(t["module_code"], {}))
            for pt in t["phase_task_list"]:
                for ot in pt["operation_task_list"]:
                    pred_by_op[ot["id"]] = pred[ot["operation"]]

    workload_zero_ops = []
    prediction_lines = []
    for t in tool_tasks_for_yaml:
        for pt in t.get("phase_task_list", []):
            for ot in pt.get("operation_task_list", []):
                op_id = ot["id"]
                actual_days = int(op_workerday_count.get(op_id, 0))
                worked_days = actual_days
                if op_id in pred_by_op:
                    worked_days = max(actual_days, int(round(pred_by_op[op_id])))
                    prediction_lines.append(
                        f"- {op_id_to_module_code.get(op_id, '')} {ot['operation']}: actual={actual_days}d "
                        f"predicted={pred_by_op[op_id]:.0f}d -> workload {worked_days}d ({worked_days * HOURS_PER_WORKDAY}h)")
                if worked_days == 0:
                    workload_zero_ops.append((op_id, op_id_to_module_code.get(op_id, "")))
                ot["workload_hours"] = worked_days * HOURS_PER_WORKDAY

                peak_w = int(op_worker_count.get(op_id, 0))
                assigned_date_count = int(op_assigned_date_count.get(op_id, 0))
                excel_headcount = headcount_by_op.get(op_id)

                if assigned_date_count > 0:
                    recommend_avg = actual_days / assigned_date_count
                    rec_min = int(math.floor(recommend_avg))
                    rec_max = int(math.ceil(recommend_avg))
                else:
                    rec_min, rec_max = 0, 0
                if worked_days > 0:
                    rec_min = max(1, rec_min)
                    rec_max = max(1, rec_max)
                if excel_headcount:
                    rec_max = max(rec_max, int(excel_headcount))
                    rec_min = max(min(rec_min, rec_max), 1 if rec_max > 0 else 0)

                ot["recommends_worker_min"] = rec_min
                ot["recommends_worker_max"] = rec_max
                ot["description"] = None
                ot["color_code"] = op_color_code.get(op_id)

                if peak_w > DEFAULT_MAX_WORKER or (excel_headcount and excel_headcount > DEFAULT_MAX_WORKER):
                    pass  # environment max_worker_num stays a generic global default; per-task caps live in recommends_*

    for t in tool_tasks_for_yaml:
        t["description"] = None
        for pt in t["phase_task_list"]:
            pt["description"] = None

    schedule = {
        "plan_range": plan_range,
        "workflow_task_list": tool_tasks_for_yaml,
        "misc_task_list": misc_tasks,
        "assignment_list": assignments,
    }

    env_root = {"environment": environment}
    sch_root = {"schedule": schedule}

    _write_env_config_yaml(envconfig_out, environment)
    _write_schedule_yaml(schedule_out, schedule)

    write_transformation_log(
        log_out, cut_rows=cut_rows, shifted_meta=shifted_meta, worker_id_to_name=worker_id_to_name,
        workload_zero_ops=sorted(set(workload_zero_ops)), dummy_tool_labels=dummy_tool_labels,
        su_outlier_corrections=all_su_corrections, outlier_cut_summary=outlier_cut_summary,
        pb_worker_dates=pb_worker_dates,
        worker_skill_lines=_worker_skill_log_lines(environment["worker_list"], su_data, excluded_workers),
        prediction_lines=_prediction_log_header(shifted_meta, predicted_codes, seiban_planned) + prediction_lines,
    )

    return env_root, sch_root, shifted_meta


# ============================================================
# YAML writers, matching GanttChartEditor/src/services/yamlService.ts exactly
# (snake_case keys; workload_hours; e{n}p{ph}o{k} ids; misc_task_list flat).
# ============================================================

def _yd(s):
    return s  # dates are already "YYYY/MM/DD" strings from _to_ymd


_YAML_RISKY_CHARS_RE = re.compile(r'[:#\[\]{}&*!|>\'"%@`,]|^[\-?]')
_YAML_KEYWORD_RE = re.compile(r'^(true|false|null|yes|no|on|off|~)$', re.IGNORECASE)
_YAML_NUMLIKE_RE = re.compile(r'^[+-]?(\d+\.?\d*|\.\d+)([eE][+-]?\d+)?$')


def _ys(v):
    """
    YAML-safe scalar string. Free-text labels straight from SU_Others cells
    (misc task names etc.) can be almost anything — a bare "," or "-" alone,
    something containing ": ", a lone number-looking token — all of which
    break unquoted plain-scalar syntax. Quote (with proper JSON/YAML-safe
    escaping) whenever the value isn't unambiguously plain; otherwise leave
    it bare for readability.
    """
    if v is None:
        return ""
    s = str(v)
    if s == "":
        return '""'
    if (
        _YAML_RISKY_CHARS_RE.search(s)
        or s != s.strip()
        or "\n" in s
        or _YAML_KEYWORD_RE.match(s)
        or _YAML_NUMLIKE_RE.match(s)
    ):
        return json.dumps(s, ensure_ascii=False)
    return s


def _flow_arr(items):
    return "[" + ", ".join(str(i) for i in items) + "]"


def _flow_num_map(d):
    return "{" + ", ".join(f"{k}: {v}" for k, v in d.items()) + "}"


def _emit_unavail_dates(lines, dates, indent):
    if not dates:
        lines.append(f"{indent}unavailable_dates: []")
        return
    lines.append(f"{indent}unavailable_dates:")
    # decoder-internal shape {"date": "YYYY/MM/DD"}: all grouped under ONE
    # single.days list, like the GanttChartEditor sample EnvConfig
    single_days = [ud["date"] for ud in dates if "date" in ud]
    if single_days:
        lines.append(f"{indent}- single:")
        lines.append(f"{indent}    days:")
        for d in single_days:
            lines.append(f"{indent}    - {d}")
    for ud in dates:
        if "date" in ud:
            continue
        elif "weekly" in ud:
            lines.append(f"{indent}- weekly:")
            lines.append(f"{indent}    weekdays: {_flow_arr(ud['weekly']['weekdays'])}")
        elif "single" in ud:
            lines.append(f"{indent}- single:")
            lines.append(f"{indent}    days:")
            for d in ud["single"]["days"]:
                lines.append(f"{indent}    - {d}")


def _write_env_config_yaml(path, env):
    L = []
    p = L.append
    p("environment:")

    p("  workflow_list:")
    for wf in env["workflow_list"]:
        p(f"  - id: {wf['id']}")
        p(f"    name: {_ys(wf.get('name'))}")
        p("    phase_list:")
        for ph in wf.get("phase_list", []):
            p(f"    - id: {ph['id']}")
            p(f"      name: {_ys(ph.get('name'))}")
            p("      operation_list:")
            for op in ph.get("operation_list", []):
                p(f"      - id: {op['id']}")
                p(f"        name: {_ys(op.get('name'))}")
                p(f"        work_hours: {_flow_arr(op.get('work_hours', []))}")
                p(f"        min_worker_num: {op.get('min_worker_num', 0)}")
                p(f"        max_worker_num: {op.get('max_worker_num', 0)}")

    p("  fab_list:")
    for f in env["fab_list"]:
        p(f"  - id: {f['id']}")
        p(f"    name: {_ys(f.get('name'))}")
        p(f"    region: {f.get('region')}")
        p(f"    customer_company: {f.get('customer_company')}")
        _emit_unavail_dates(L, f.get("unavailable_dates", []), "    ")

    p("  region_list:")
    for r in env["region_list"]:
        p(f"  - id: {r['id']}")
        p(f"    name: {_ys(r.get('name'))}")
        p(f"    max_stay_on: {r.get('max_stay_on', 0)}")
        p(f"    max_annual_stay: {r.get('max_annual_stay', 0)}")
        p(f"    stay_off_interval: {r.get('stay_off_interval', 0)}")
        _emit_unavail_dates(L, r.get("unavailable_dates", []), "    ")

    p("  customer_company_list:")
    for c in env["customer_company_list"]:
        p(f"  - id: {c['id']}")
        p(f"    name: {_ys(c.get('name'))}")
        _emit_unavail_dates(L, c.get("unavailable_dates", []), "    ")

    p("  worker_company_list:")
    for wc in env["worker_company_list"]:
        p(f"  - id: {wc['id']}")
        p(f"    name: {_ys(wc.get('name'))}")
        p(f"    annual_overtime_limit: {wc.get('annual_overtime_limit', 0)}")
        p(f"    monthly_overtime_limit: {wc.get('monthly_overtime_limit', 0)}")
        _emit_unavail_dates(L, wc.get("unavailable_dates", []), "    ")

    p("  transite_day_map:")
    for t in env["transite_day_map"]:
        p(f"  - from: {t['from']}")
        p(f"    to: {t['to']}")
        p(f"    days: {t['days']}")

    p("  affinity_tag:")
    for tag in env.get("affinity_tag", []):
        p(f"  - id: {tag['id']}")
        p(f"    weight: {tag['weight']}")

    p("  worker_list:")
    for w in env["worker_list"]:
        p(f"  - id: {w['id']}")
        p(f"    name: {_ys(w.get('name'))}")
        p(f"    worker_company: {w.get('worker_company')}")
        p(f"    is_manager: {'true' if w.get('is_manager') else 'false'}")
        sm = w.get("skill_map") or {}
        p(f"    skill_map: {_flow_num_map(sm) if sm else '{}'}")
        wto = w.get("worker_type_by_operation") or {}
        if wto:
            p(f"    worker_type_by_operation: {{{', '.join(f'{k}: {v}' for k, v in wto.items())}}}")
        else:
            p("    worker_type_by_operation: {}")
        fsm = w.get("fab_suitability_map") or []
        if fsm:
            p("    fab_suitability_map:")
            for entry in fsm:
                p(f"    - kind: {entry['kind']}")
                p(f"      suitability: {_flow_num_map(entry['suitability'])}")
        else:
            p("    fab_suitability_map: []")
        _emit_unavail_dates(L, w.get("unavailable_dates", []), "    ")
        p(f"    affinity: {_flow_arr(w.get('affinity') or [])}")
        desc = w.get("description")
        if desc is not None:
            p("    description:")
            p(f"      業務形態: {_ys(desc.get('業務形態', ''))}")
            p(f"      VISA: {_ys(desc.get('VISA', ''))}")
            p(f"      海外運転: {_ys(desc.get('海外運転', ''))}")
            if "備考" in desc:
                p(f"      備考: {_ys(desc['備考'])}")

    Path(path).write_text("\n".join(L) + "\n", encoding="utf-8")


def _write_schedule_yaml(path, sch):
    L = []
    p = L.append
    p("schedule:")
    p("  plan_range:")
    p(f"    start_date: {sch['plan_range']['start_date']}")
    p(f"    end_date: {sch['plan_range']['end_date']}")

    p("  workflow_task_list:")
    for wt in sch["workflow_task_list"]:
        p(f"  - id: {wt['id']}")
        p(f"    name: {_ys(wt.get('name'))}")
        p(f"    description: {_ys(wt.get('description'))}")
        p(f"    workflow: {wt['workflow']}")
        if wt.get("fab") is not None:
            p(f"    fab: {wt['fab']}")
        p("    phase_task_list:")
        for pt in wt["phase_task_list"]:
            p(f"    - id: {pt['id']}")
            p(f"      name: {_ys(pt.get('name'))}")
            p(f"      description: {_ys(pt.get('description'))}")
            p(f"      phase: {pt['phase']}")
            p(f"      start_date: {pt['start_date']}")
            p(f"      end_date: {pt['end_date']}")
            p("      operation_task_list:")
            for ot in pt["operation_task_list"]:
                p(f"      - id: {ot['id']}")
                p(f"        name: {_ys(ot.get('name'))}")
                p(f"        description: {_ys(ot.get('description'))}")
                p(f"        operation: {ot['operation']}")
                p(f"        workload_hours: {ot.get('workload_hours', 0)}")
                if "recommends_worker_min" in ot:
                    p(f"        recommends_worker_min: {ot['recommends_worker_min']}")
                    p(f"        recommends_worker_max: {ot['recommends_worker_max']}")
                p(f"        color_code: {_ys(ot.get('color_code')) if ot.get('color_code') else ''}")

    p("  misc_task_list:")
    for mt in sch["misc_task_list"]:
        p(f"  - id: {mt['id']}")
        p(f"    name: {_ys(mt.get('name'))}")
        p(f"    description: {_ys(mt.get('description'))}")
        if mt.get("workflow"):
            p(f"    workflow: {mt['workflow']}")
        if mt.get("region") is not None:
            p(f"    region: {mt['region']}")
        if mt.get("color_code"):
            p(f"    color_code: {_ys(mt['color_code'])}")

    p("  assignment_list:")
    for a in sch["assignment_list"]:
        p(f"  - worker: {a['worker']}")
        p(f"    operation_task: {a['operation_task']}")
        p(f"    start_date: {a['start_date']}")
        p(f"    end_date: {a['end_date']}")
        p("    work_date_list:")
        for w in a["work_date_list"]:
            p(f"    - date: {w['date']}")
            p(f"      hour: {w['hour']}")
        p(f"    plan_flexibility: {a['plan_flexibility']}")
        p(f"    description: {_ys(a.get('description'))}")

    Path(path).write_text("\n".join(L) + "\n", encoding="utf-8")


# ============================================================
# CLI
# ============================================================

def main():
    ap = argparse.ArgumentParser(description="Decoder7: SU_Others + 初期データ追加情報 (製番) -> EnvConfig.yaml + Schedule.yaml")
    ap.add_argument("--su-others", required=True, help="Path to the SU_Others .xlsm file (e.g. '20260915 SU_Others_skill level有配布禁止_.xlsm')")
    ap.add_argument("--seiban-info", required=True, help="Path to 初期データ追加情報 (e.g. '初期データ追加情報 _20260930.rev1.xlsx'); its 製番 sheet lists the target modules")
    ap.add_argument("--plan-start", default=None, help="Plan range start date, e.g. 2026/09/15. Default: earliest date found in SU_Others.")
    ap.add_argument("--plan-end", default=None, help="Plan range end date, e.g. 2027/03/31. Default: latest date found in SU_Others.")
    ap.add_argument("--su-sheets", default="予定表_2026", help="Comma-separated SU_Others main task sheet name(s).")
    ap.add_argument("--seiban-start-row", type=int, default=SEIBAN_START_ROW,
                    help=f"First 製番 row (Excel row number) that is a target module; rows above it become misc tasks. Default {SEIBAN_START_ROW}. Use 3 for the whole list.")
    ap.add_argument("--skill-sheet", default="予定表_2026 (2)",
                    help="SU_Others sheet with 担当職種 / スキルレベル / Status (skill + regular/spot reference).")
    ap.add_argument("--envconfig-out", default="EnvConfig.yaml")
    ap.add_argument("--schedule-out", default="Schedule.yaml")
    ap.add_argument("--log-out", default=TRANSFORMATION_LOG)
    args = ap.parse_args()

    plan_start = _parse_simple_date(args.plan_start) if args.plan_start else None
    plan_end = _parse_simple_date(args.plan_end) if args.plan_end else None
    su_sheets = tuple(s.strip() for s in args.su_sheets.split(",") if s.strip())

    for label, path in [("--su-others", args.su_others), ("--seiban-info", args.seiban_info)]:
        if not Path(path).exists():
            print(f"ERROR: {label} file not found: {path}", file=sys.stderr)
            sys.exit(1)

    build_env_and_schedule_decoder7(
        args.su_others, args.seiban_info,
        envconfig_out=args.envconfig_out, schedule_out=args.schedule_out, log_out=args.log_out,
        plan_start=plan_start, plan_end=plan_end, su_sheet_names=su_sheets,
        skill_sheet=args.skill_sheet, seiban_start_row=args.seiban_start_row,
    )
    print(f"{args.envconfig_out}, {args.schedule_out}, and {args.log_out} have been written.")


if __name__ == "__main__":
    main()
