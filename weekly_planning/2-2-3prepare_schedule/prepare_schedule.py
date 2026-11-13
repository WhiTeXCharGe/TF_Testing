#!/usr/bin/env python3
"""
prepare_schedule.py  --  weekly planning tool "2-2 + 2-3" (one tool).

  2-2  find the modules / FI-FO that a planner changed in the SU_Others workbook (they become Fixed)
  2-3  build the Timefold input (01_Schedule.yaml, 01_EnvConfig.yaml) where only the
       still-flexible modules and trips are optimisable and everything else is misc / Fixed trip

Inputs
  --prev-flexible   last week's FlexibleSchedule.yaml.  A Timefold result Schedule.yaml also works
                    (everything in it is treated as flexible; misc tasks in it are ignored).
  --prev-env        last week's EnvConfig.yaml (workflows, fabs, regions ... are carried over)
  --su-others       the dated SU_Others workbook (.xlsm): ongoing tasks and the worker list
  --skill-file      the skill-level workbook (.xlsm): skills / status only (optional; without it the
                    skill sheet is looked up inside --su-others)
  --seiban-info     the seiban info workbook (.xlsx, sheet 製番: module -> customer / fab / region)
Output (in --out-dir)
  <today>_FlexibleWork.yaml   modules and trips that stay auto-planned (file name starts with the --today date)
  01_EnvConfig.yaml           EnvConfig for the run
  01_Schedule.yaml            Schedule for the run (flexible modules + misc tasks + trip_list)
  <today>_prepare_report.txt  only with --report: what was decided and why, and everything that could not be resolved

The decisions (changed / unchanged) are plain functions on plain data, so they can be tested
without Excel files (see tests/).  Reading the workbooks and the small shared functions come from the `common` folder of weekly_planning.
"""
from __future__ import annotations

import argparse
import copy
import re
import sys
import unicodedata
from collections import Counter, defaultdict
from datetime import date
from pathlib import Path

from openpyxl import load_workbook

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent))      # weekly_planning/ (the folder of common)
from common import seiban_sheet, su_others   # noqa: E402
from common.cell_names import cell_name_key   # noqa: E402
from common.dates import ymd   # noqa: E402
from common.flexible_schedule import read_prev_flexible   # noqa: E402
from common.text import name_key, tool_code   # noqa: E402
from common.trips import pair_fifo, pair_fifo_repaired, read_mark   # noqa: E402
from common.worker_skills import STATUS_TO_TYPE, parse_worker_skills   # noqa: E402
from common.yaml_io import dump_yaml, load_yaml   # noqa: E402

HOURS_PER_WORKDAY = 10          # hours of every work day (the same value as in the SU_Others parser)
NEUTRAL_REGION_ID = "r_other"   # region of misc tasks that have no place (Timefold treats it as "no region")


# ----------------------------------------------------------------------------- small helpers

def top_color(counter):
    return min(counter.items(), key=lambda kv: (-kv[1], kv[0]))[0] if counter else None


# ----------------------------------------------------------------------------- cell names of the flexible modules

def put_cell_name(row, worker_key, sheet_texts):
    """Gives an assignment row of a flexible module its `cell_name`: the one it already has, else the most common text of the
    worker's cells on the row's days in the sheet.  The same text goes into `description` when that is empty, because the
    solver keeps `description` of a (worker, task) in its result but drops other new keys of the rows it rebuilds."""
    name = row.get("cell_name")
    if not (isinstance(name, str) and name.strip()):
        texts = Counter(sheet_texts[(worker_key, ymd(i.get("date")))] for i in row.get("work_date_list") or []
                        if (worker_key, ymd(i.get("date"))) in sheet_texts)
        name = texts.most_common(1)[0][0] if texts else None
    if name:
        row["cell_name"] = name
        if row.get("description") in (None, ""):
            row["description"] = name


# ----------------------------------------------------------------------------- 2-2: decisions

def decide_modules(prev_modules, sheet_module_cells, today, misc_cover=None, sheet_texts=None):
    """
    prev_modules: {code: {"assign": {(worker_key, 'YYYY/MM/DD')}, "last": 'YYYY/MM/DD' or None}}
    sheet_module_cells: {code: {(worker_key, date)}}  cells of SU_Others that carry the module code
    today: 'YYYY/MM/DD'
    misc_cover: {code: {(worker_key, date)}}  cells the previous schedule already holds as a misc task of that module
                (the SU_Others parser turns outlier cells into such tasks); a sheet cell found there is not a change.
    sheet_texts: {(worker_key, date): text}  the text of the cells; with it the text of a bar is checked too: a module whose
                cells now carry another text than the `cell_name` of the previous rows is changed (rows without a cell_name are
                not checked).  The text of a bar is its most common text, so one odd cell is not a change.
    Returns (flexible [codes], fixed {code: reason}).
    """
    flexible, fixed = [], {}
    misc_cover = misc_cover or {}
    for code in sorted(prev_modules):
        m = prev_modules[code]
        sheet = sheet_module_cells.get(code, set())
        sheet_only = sheet - m["assign"] - misc_cover.get(code, set())
        planned_only = m["assign"] - sheet
        text_change = changed_text(m, sheet, sheet_texts) if sheet_texts else None
        if sheet_only or planned_only:
            fixed[code] = (f"assignment changed in SU_Others (planned only: {len(planned_only)} worker-days, "
                           f"sheet only: {len(sheet_only)} worker-days)")
        elif text_change:
            fixed[code] = text_change
        elif m["last"] is not None and m["last"] < today:
            fixed[code] = f"all work finished before today ({m['last']})"
        else:
            flexible.append(code)
    return flexible, fixed


def changed_text(module, sheet_cells, sheet_texts):
    """Reason (str) if the text of a bar of the module in the sheet differs from its cell_name, else None.
    Per (worker, row of the previous plan): the most common text of that worker's cells of the module on the row's days."""
    by_name = {}
    for key, name in module.get("names", {}).items():
        by_name.setdefault((key[0], name), []).append(key)
    for (wk, name), keys in sorted(by_name.items()):
        texts = Counter(cell_name_key(sheet_texts[k]) for k in keys if k in sheet_cells and k in sheet_texts)
        if not texts:
            continue                                   # the cells are not there any more: the assignment check reports that
        top, _n = texts.most_common(1)[0]
        if top != cell_name_key(name):
            now = next(sheet_texts[k] for k in sorted(keys) if k in sheet_texts and cell_name_key(sheet_texts[k]) == top)
            return f"cell text changed in SU_Others ({wk}: {name!r} -> {now!r})"
    return None


def decide_trips(prev_trips, sheet_pairs):
    """
    prev_trips: list from read_prev_flexible (worker_key, start, end, region, flex)
    sheet_pairs: {worker_key: [(start, end)]} FI/FO pairs found in SU_Others
    Returns dict: kept (prev Flexible trips whose FI and FO are still where they were),
                  changed (prev Flexible trips whose FI/FO moved or were removed),
                  fixed ([(worker_key, start, end, prev_region_or_None)]: every other FI/FO pair of the sheet).
    """
    sheet = {(wk, s, e) for wk, pairs in sheet_pairs.items() for s, e in pairs}
    prev_region = {(t["worker_key"], t["start"], t["end"]): t.get("region") for t in prev_trips}
    kept, changed, used = [], [], set()
    for t in prev_trips:
        key = (t["worker_key"], t["start"], t["end"])
        if t["flex"].lower() == "flexible":
            if key in sheet:
                kept.append(t)
                used.add(key)
            else:
                changed.append(t)
    fixed = [(wk, s, e, prev_region.get((wk, s, e))) for wk, s, e in sorted(sheet - used)]
    return {"kept": kept, "changed": changed, "fixed": fixed}


def infer_trip_region(worker_key, start, end, worker_code_days, code_region, worker_text_days=None, fab_region=None):
    """Region of a trip from what the worker does strictly between FI and FO.
    1. module codes of known regions (through 製番);  2. otherwise a fab name inside a bar label
    (fab_region: {NFKC upper fab name: region id}, names of 3+ characters, one region per name).
    Returns (region_id or None, note or None).  Several regions by label are not guessed."""
    seen = Counter()
    for d, code in worker_code_days.get(worker_key, {}).items():
        if start < d < end and code in code_region:
            seen[code_region[code]] += 1
    if seen:
        if len(seen) == 1:
            return next(iter(seen)), None
        best = seen.most_common(1)[0][0]
        return best, f"several regions between FI and FO {dict(seen)}; used {best}"
    by_label = Counter()
    for d, text in (worker_text_days or {}).get(worker_key, {}).items():
        if not (start < d < end):
            continue
        t = unicodedata.normalize("NFKC", str(text)).upper()
        hits = sorted((f for f in (fab_region or {}) if f in t), key=len, reverse=True)
        if hits:
            by_label[fab_region[hits[0]]] += 1
    if len(by_label) == 1:
        region = next(iter(by_label))
        return region, f"region {region} taken from a fab name in the bar label (no module code of a known region)"
    if len(by_label) > 1:
        return None, f"fab names of several regions in the labels {dict(by_label)}"
    return None, "no module of a known region and no fab name in the labels between FI and FO"


# ----------------------------------------------------------------------------- master data (EnvConfig)

def build_master_lists(prev_env, seiban_rows, sheet_company_names):
    """
    prev_env: parsed previous EnvConfig 'environment' dict.
    seiban_rows: {code: {"customer","fab","region"}} (names as in the seiban info workbook)
    sheet_company_names: worker company names found in SU_Others (in order)
    Returns dict with the merged lists and name->id maps; ids of existing entries never change.
    """
    env = copy.deepcopy(prev_env)
    regions = env.setdefault("region_list", [])
    fabs = env.setdefault("fab_list", [])
    customers = env.setdefault("customer_company_list", [])
    companies = env.setdefault("worker_company_list", [])

    def by_name(lst):
        return {str(x.get("name")): x["id"] for x in lst}

    reg, fab, cus, wco = by_name(regions), by_name(fabs), by_name(customers), by_name(companies)
    if NEUTRAL_REGION_ID not in {r["id"] for r in regions}:
        regions.append({"id": NEUTRAL_REGION_ID, "name": "Other", "max_stay_on": 10000, "max_annual_stay": 10000,
                        "stay_off_interval": 3, "unavailable_dates": []})
        reg["Other"] = NEUTRAL_REGION_ID

    def new_id(prefix, lst):
        used = {x["id"] for x in lst}
        n = 1
        while f"{prefix}{n}" in used:
            n += 1
        return f"{prefix}{n}"

    def clean(v):
        return str(v).strip() if isinstance(v, str) and v.strip() else None

    for code, row in sorted(seiban_rows.items()):
        rname, cname, fname = clean(row.get("region")), clean(row.get("customer")), clean(row.get("fab"))
        if rname and rname not in reg:
            rid = new_id("r", regions)
            regions.append({"id": rid, "name": rname, "max_stay_on": 10000, "max_annual_stay": 10000,
                            "stay_off_interval": 3, "unavailable_dates": []})
            reg[rname] = rid
        if cname and cname not in cus:
            cid = new_id("c", customers)
            customers.append({"id": cid, "name": cname, "unavailable_dates": []})
            cus[cname] = cid
        if fname and fname not in fab:
            fid = new_id("f", fabs)
            fabs.append({"id": fid, "name": fname, "region": reg.get(rname, NEUTRAL_REGION_ID),
                         "customer_company": cus.get(cname, (customers[0]["id"] if customers else None)),
                         "unavailable_dates": []})
            fab[fname] = fid
    for cname in sheet_company_names:
        if cname not in wco:
            wid = new_id("wc", companies)
            companies.append({"id": wid, "name": cname, "annual_overtime_limit": 10000,
                              "monthly_overtime_limit": 10000, "unavailable_dates": []})
            wco[cname] = wid

    # one transit entry per ordered region pair, existing values kept
    existing = {(t["from"], t["to"]): t.get("days", 1) for t in env.get("transite_day_map") or []}
    ids = [r["id"] for r in regions]
    env["transite_day_map"] = [{"from": a, "to": b, "days": existing.get((a, b), 1)}
                               for a in ids for b in ids if a != b]
    # one affinity tag per worker company, existing tags kept
    tags = {t["id"]: t for t in env.get("affinity_tag") or []}
    for c in companies:
        tid = "wct" + c["id"][2:]
        tags.setdefault(tid, {"id": tid, "weight": 2})
    env["affinity_tag"] = list(tags.values())

    # module code -> region id (through the 製番 row)
    code_region = {}
    for code, row in seiban_rows.items():
        rname = clean(row.get("region"))
        if rname and rname in reg:
            code_region[code] = reg[rname]
    return {"env": env, "region_by_name": reg, "company_by_name": wco, "code_region": code_region}


def fab_name_regions(env):
    """{NFKC upper fab name (3+ chars): region id} for names that belong to exactly one real region."""
    found = defaultdict(set)
    for f in env.get("fab_list") or []:
        nm = unicodedata.normalize("NFKC", str(f.get("name") or "")).strip().upper()
        if len(nm) >= 3 and nm != "OTHER" and f.get("region") and f["region"] != NEUTRAL_REGION_ID:
            found[nm].add(f["region"])
    return {nm: next(iter(r)) for nm, r in found.items() if len(r) == 1}


# ----------------------------------------------------------------------------- sheet data -> misc tasks, workers

def final_worker_ids(sheet_workers, prev_env_workers):
    """name key -> final worker id: ids of known workers are kept, new workers get the next free w### id."""
    ids = {name_key(w["name"]): w["id"] for w in prev_env_workers}
    used = set(ids.values())
    nxt = 1
    out = {}
    for w in sheet_workers:
        k = name_key(w["name"])
        if k in ids:
            out[k] = ids[k]
        else:
            while f"w{nxt:03d}" in used:
                nxt += 1
            out[k] = f"w{nxt:03d}"
            used.add(out[k])
    return out


def build_misc(sheet, flexible_codes, code_region, worker_id_by_key):
    """
    sheet["date_map"]: {(worker_key, 'YYYY/MM/DD'): text} (red / FI / FO cells are not in it),
    sheet["personal_map"]: grey cells without a module code, sheet["color_map"]: {(worker_key, date): 'RRGGBB'}.
    Every bar that is not part of a flexible module becomes a misc task with Fixed rows.
    Returns (misc_task_list, assignment_rows).
    """
    colors = defaultdict(Counter)
    label_days = defaultdict(lambda: defaultdict(set))       # label -> worker_key -> dates
    for (wk, d), text in sheet["date_map"].items():
        code = tool_code(text)
        if code and code in flexible_codes:
            continue
        label = str(text).strip() or "other"
        label_days[label][wk].add(d)
        c = sheet["color_map"].get((wk, d))
        if c:
            colors[label][c] += 1
    pb_days = defaultdict(lambda: defaultdict(set))
    pb_colors = defaultdict(Counter)
    for (wk, d), text in sheet["personal_map"].items():
        label = re.sub(r"\s+", " ", str(text or "")).strip()
        pb_days[label][wk].add(d)
        c = sheet["color_map"].get((wk, d))
        if c:
            pb_colors[label][c] += 1

    misc, rows = [], []

    def add(task_id, name, workflow, region, color, per_worker):
        t = {"id": task_id, "name": name, "description": None, "workflow": workflow, "region": region}
        if color:
            t["color_code"] = color
        misc.append(t)
        for wk in sorted(per_worker, key=lambda k: worker_id_by_key.get(k, k)):
            days = sorted(per_worker[wk])
            rows.append({"worker": worker_id_by_key[wk], "operation_task": task_id,
                         "start_date": days[0], "end_date": days[-1],
                         "work_date_list": [{"date": d, "hour": HOURS_PER_WORKDAY} for d in days],
                         "plan_flexibility": "Fixed", "description": None})

    for n, label in enumerate(sorted(label_days), start=1):
        code = tool_code(label)
        region = code_region.get(code, NEUTRAL_REGION_ID) if code else NEUTRAL_REGION_ID
        add(f"misc_other_{n}", label, None, region, top_color(colors[label]), label_days[label])
    for n, label in enumerate(sorted(pb_days), start=1):
        add(f"misc_pb_{n}", label or "Personal Business", "wf_personal_business", NEUTRAL_REGION_ID,
            top_color(pb_colors[label]), pb_days[label])
    return misc, rows


def build_worker_list(sheet_workers, worker_id_by_key, company_by_name, prev_env_workers, valid_ops):
    """EnvConfig worker_list from the SU_Others workers (skills etc. are computed by the reader).
    Skills for operations that do not exist in the workflow are dropped (and reported by the caller)."""
    prev_by_key = {name_key(w["name"]): w for w in prev_env_workers}
    out, dropped, kept_prev = [], Counter(), []
    for w in sheet_workers:
        k = name_key(w["name"])
        skill_map = w.get("skill_map") or {}
        type_map = w.get("worker_type_by_operation") or {}
        prev_w = prev_by_key.get(k) or {}
        if not skill_map and not w.get("has_skill_ref", True) and prev_w.get("skill_map"):
            # not in the skill file: keep the skills of last week instead of erasing them
            skill_map, type_map = prev_w["skill_map"], prev_w.get("worker_type_by_operation") or {}
            kept_prev.append(w["name"])
        skill = {op: lv for op, lv in skill_map.items() if op in valid_ops}
        for op in skill_map:
            if op not in valid_ops:
                dropped[op] += 1
        wtype = {op: t for op, t in type_map.items() if op in skill}
        item = {
            "id": worker_id_by_key[k], "name": w["name"],
            "worker_company": company_by_name[w["company_name"]],
            "is_manager": bool(w.get("is_manager")),
            "skill_map": skill, "worker_type_by_operation": wtype,
            "fab_suitability_map": (prev_by_key.get(k) or {}).get("fab_suitability_map") or [],
            "unavailable_dates": [{"single": {"days": sorted(w.get("unavailable") or [])}}] if w.get("unavailable") else [],
            "affinity": ["wct" + company_by_name[w["company_name"]][2:]],
        }
        if w.get("description"):
            item["description"] = w["description"]
        out.append(item)
    return out, dropped, kept_prev


# ----------------------------------------------------------------------------- the whole preparation (pure)

def prepare(prev_flex_doc, prev_env_doc, sheet, seiban_rows, today, default_trip_region=None, plan_range=None,
            repair_fifo=True, max_gap_days=14, chain_days=3):
    """
    sheet: {"workers": [{name, company_name, is_manager, skill_map, worker_type_by_operation, unavailable, description}],
            "date_map", "personal_map", "color_map", "fifo": {worker_key: {date: 'FI'|'FO'}},
            "fifo_variants": {raw text: (mark, count)}, "plan_range": {...}}
    repair_fifo: FI/FO without a partner are repaired from the busy days (pair_fifo_repaired); False reports and drops them.
    Returns dict(flexible_work, env, schedule, report [lines], stats).
    """
    report = []
    prev_env = (prev_env_doc.get("environment") if isinstance(prev_env_doc, dict) and "environment" in prev_env_doc
                else prev_env_doc) or {}
    prev_workers = prev_env.get("worker_list") or []
    prev_names = {w["id"]: w["name"] for w in prev_workers}
    prev = read_prev_flexible(prev_flex_doc, prev_names)
    report += [f"WARN: {w}" for w in prev["warnings"]]

    # --- what the sheet says
    module_cells = defaultdict(set)
    worker_code_days = defaultdict(dict)
    worker_text_days = defaultdict(dict)
    display = {name_key(w["name"]): w["name"] for w in sheet["workers"]}
    for (wk, d), text in sheet["date_map"].items():
        worker_text_days[wk][d] = text
        c = tool_code(text)
        if c:
            module_cells[c].add((wk, d))
            worker_code_days[wk][d] = c
    for raw, (mark, n) in sorted((sheet.get("fifo_variants") or {}).items()):
        report.append(f"INFO: {n} cells with the text {raw!r} were read as {mark}")
    activity = defaultdict(set)     # worker -> days with anything written in the sheet
    for (wk, d) in list(sheet["date_map"]) + list(sheet.get("personal_map") or {}):
        activity[wk].add(d)
    pr0 = sheet.get("plan_range") or {}
    window = (ymd(pr0.get("start_date")), ymd(pr0.get("end_date")))
    sheet_pairs, repaired = {}, 0
    for wk, marks in sheet["fifo"].items():
        if repair_fifo:
            pairs, notes = pair_fifo_repaired(marks, activity[wk], window, max_gap_days, chain_days=chain_days)
            repaired += sum(1 for n in notes if n.startswith("INFERRED"))
            report += [f"{'NOTE' if n.startswith('INFERRED') else 'INFO'}: FI/FO of {display.get(wk, wk)}: {n}" for n in notes]
        else:
            pairs, warns = pair_fifo(marks)
            report += [f"WARN: FI/FO of {display.get(wk, wk)}: {w}" for w in warns]
        sheet_pairs[wk] = pairs

    # --- 2-2: decisions
    flexible, fixed_modules = decide_modules(prev["modules"], module_cells, today, prev.get("misc_cover"),
                                             sheet_texts=sheet["date_map"])
    trip_dec = decide_trips(prev["trips"], sheet_pairs)
    report.append(f"MODULES: {len(flexible)} stay flexible, {len(fixed_modules)} become Fixed (misc)")
    for code, why in sorted(fixed_modules.items()):
        report.append(f"  FIXED module {code}: {why}")
    report.append(f"TRIPS: {len(trip_dec['kept'])} flexible trips unchanged, {len(trip_dec['changed'])} changed or removed "
                  f"in the sheet, {len(trip_dec['fixed'])} become Fixed trips")
    for t in trip_dec["changed"]:
        report.append(f"  CHANGED flexible trip {display.get(t['worker_key'], t['worker_key'])} {t['start']}..{t['end']}: FI/FO no longer there")

    # --- 2-3: environment
    master = build_master_lists(prev_env, seiban_rows, [w["company_name"] for w in sheet["workers"]])
    env = master["env"]
    worker_id_by_key = final_worker_ids(sheet["workers"], prev_workers)
    valid_ops = {op["id"] for wf in env.get("workflow_list") or [] for ph in wf.get("phase_list") or []
                 for op in ph.get("operation_list") or []}
    env["worker_list"], dropped_ops, kept_prev = build_worker_list(sheet["workers"], worker_id_by_key,
                                                                   master["company_by_name"], prev_workers, valid_ops)
    if kept_prev:
        report.append(f"INFO: {len(kept_prev)} workers are not in the skill file; skills of the previous EnvConfig kept: "
                      + ", ".join(kept_prev[:20]) + (" ..." if len(kept_prev) > 20 else ""))
    for op, n in sorted(dropped_ops.items()):
        report.append(f"WARN: skill for operation {op} of {n} workers dropped (not in the workflow of the previous EnvConfig)")

    # --- 2-3: schedule
    flexible_set = set(flexible)
    tasks, rows = [], []
    for code in flexible:
        m = prev["modules"][code]
        tasks.append(copy.deepcopy(m["task"]))
        for r in m["rows"]:
            r2 = copy.deepcopy(r)
            wname = prev_names.get(r2.get("worker"), str(r2.get("worker")))
            r2["worker"] = worker_id_by_key.get(name_key(wname), r2.get("worker"))
            put_cell_name(r2, name_key(wname), sheet["date_map"])
            rows.append(r2)
    misc, misc_rows = build_misc(sheet, flexible_set, master["code_region"], worker_id_by_key)
    rows += misc_rows

    trips = defaultdict(list)    # worker id -> [trip]
    for t in trip_dec["kept"]:
        trips[worker_id_by_key.get(t["worker_key"], t["worker_id"])].append(
            {"start_date": t["start"], "end_date": t["end"], "region": t["region"],
             "plan_flexibility": "Flexible", "description": None})
    unresolved = 0
    fab_region = fab_name_regions(env)
    for wk, s, e, prev_reg in trip_dec["fixed"]:
        region, note = ((prev_reg, None) if prev_reg else
                        infer_trip_region(wk, s, e, worker_code_days, master["code_region"], worker_text_days, fab_region))
        if note:
            report.append(f"  NOTE trip {display.get(wk, wk)} {s}..{e}: {note}")
        if region is None:
            region = default_trip_region
        if region is None:
            unresolved += 1
            report.append(f"WARN: Fixed trip {display.get(wk, wk)} {s}..{e} skipped: region could not be determined "
                          f"(--default-trip-region keeps such trips)")
            continue
        if wk not in worker_id_by_key:
            report.append(f"WARN: Fixed trip {display.get(wk, wk)} {s}..{e} skipped: the worker is not in the SU_Others worker rows")
            unresolved += 1
            continue
        trips[worker_id_by_key[wk]].append({"start_date": s, "end_date": e, "region": region,
                                            "plan_flexibility": "Fixed", "description": None})
    trip_list = [{"worker": w, "trips": sorted(ts, key=lambda t: (t["start_date"], t["end_date"]))}
                 for w, ts in sorted(trips.items())]

    pr = plan_range or sheet.get("plan_range") or prev.get("plan_range") or {}
    schedule = {"plan_range": {"start_date": ymd(pr.get("start_date")), "end_date": ymd(pr.get("end_date"))},
                "workflow_task_list": tasks, "misc_task_list": misc, "assignment_list": rows, "trip_list": trip_list}
    flexible_work = {"flexible_work": {
        "modules": flexible,
        "trips": [{"worker": worker_id_by_key.get(t["worker_key"], t["worker_id"]),
                   "start_date": t["start"], "end_date": t["end"], "region": t["region"]} for t in trip_dec["kept"]],
    }}
    stats = {"flexible": len(flexible), "fixed_modules": len(fixed_modules), "misc_tasks": len(misc),
             "rows": len(rows), "trips_flexible": len(trip_dec["kept"]), "trips_fixed": len(trip_dec["fixed"]) - unresolved,
             "trips_skipped": unresolved, "trips_repaired": repaired}
    return {"flexible_work": flexible_work, "env": {"environment": env}, "schedule": {"schedule": schedule},
            "report": report, "stats": stats}


# ----------------------------------------------------------------------------- Excel reading


def read_sheet(su_path, sheets, skill_sheet, skill_path=None):
    """The SU_Others workbook -> the `sheet` dict used by prepare().  One workbook load serves the SU_Others parser and the
    FI/FO scan (the parser skips FI/FO cells, so they are read here).
    skill_path: the separate "skill level" workbook; skills / status are taken from its skill_sheet. Without it the
    skill sheet is looked up inside su_path."""
    wb = load_workbook(su_path, data_only=True, read_only=False)
    skill_ref = None
    if skill_path:
        skill_wb = load_workbook(skill_path, data_only=True, read_only=False)
        if skill_sheet not in skill_wb.sheetnames:
            raise ValueError(f"sheet {skill_sheet!r} not found in {skill_path}; available: {skill_wb.sheetnames} (use --skill-sheet)")
        skill_ref = su_others.read_skill_reference(skill_wb, skill_sheet)
        if not skill_ref:
            raise ValueError(f"no skill rows found in sheet {skill_sheet!r} of {skill_path}")
    su = su_others.parse_su_others(su_path, sheet_names=sheets, skill_sheet=skill_sheet, workbook=wb, skill_reference=skill_ref)

    wid_to_key = {w["id"]: name_key(w["name"]) for w in su["worker_list"]}
    company_name = {c["id"]: c["name"] for c in su["worker_company_list"]}
    workers = []
    for w in su["worker_list"]:
        wid = w["id"]
        if su["worker_in_skill_sheet"].get(wid):
            skill_map, _why = parse_worker_skills(su["worker_skill_role"].get(wid, ""), su["worker_skill_raw"].get(wid, ""))
        else:
            skill_map = {}
        wtype = STATUS_TO_TYPE.get(str(su["worker_status"].get(wid, "")).strip().upper(), "spot")
        workers.append({
            "name": w["name"], "company_name": company_name[w["worker_company"]], "is_manager": w["is_manager"],
            "has_skill_ref": bool(su["worker_in_skill_sheet"].get(wid)),
            "skill_map": skill_map, "worker_type_by_operation": {op: wtype for op in skill_map},
            "unavailable": [u["date"] for u in w["unavailable_dates"]], "description": w.get("description"),
        })

    def keyed(m):
        return {(wid_to_key[wid], ymd(dt)): v for (wid, dt), v in m.items()}

    fifo = defaultdict(dict)
    variants = {}
    for sheet_name in [s for s in sheets if s in wb.sheetnames]:
        ws = wb[sheet_name]
        date_row_idx, date_cols, dt_by_col = su_others.find_date_header_ws(ws)
        cols = su_others.detect_su_columns(ws, date_row_idx + 1, stamp_row_idx=date_row_idx)
        start = date_row_idx + 2
        for row in ws.iter_rows(min_row=start, min_col=1, max_col=max(date_cols)):
            name = row[cols["name"]].value if cols["name"] < len(row) else None
            company = row[cols["company"]].value if cols["company"] < len(row) else None
            if name is None or not str(name).strip() or not str(company or "").strip():
                continue
            key = name_key(name)
            for c in date_cols:
                v = row[c - 1].value
                mark = read_mark(v)
                if mark:
                    fifo[key][ymd(dt_by_col[c])] = mark
                    raw = unicodedata.normalize("NFKC", v).strip()
                    if raw != mark:
                        variants[raw] = (mark, variants.get(raw, (mark, 0))[1] + 1)
    return {
        "workers": workers, "date_map": keyed(su["worker_date_map"]), "personal_map": keyed(su["worker_personal_map"]),
        "color_map": keyed(su["worker_color_map"]), "fifo": dict(fifo), "fifo_variants": variants,
        "plan_range": su["plan_range"],
    }


def read_seiban(path, start_row):
    return seiban_sheet.read_seiban_sheet(path, start_row)


# ----------------------------------------------------------------------------- CLI

def main(argv=None):
    ap = argparse.ArgumentParser(description="Weekly planning 2-2 + 2-3: detect Fixed modules / FI-FO and build the Timefold input.")
    ap.add_argument("--prev-flexible", required=True, help="last week's FlexibleSchedule.yaml (or a Timefold result Schedule.yaml)")
    ap.add_argument("--prev-env", required=True, help="last week's EnvConfig.yaml")
    ap.add_argument("--su-others", required=True, help="the dated SU_Others workbook (.xlsm): ongoing tasks and the worker list")
    ap.add_argument("--skill-file", default=None,
                    help="the skill-level workbook (.xlsm): skills / status only (default: look for the skill sheet inside --su-others)")
    ap.add_argument("--seiban-info", required=True, help="the seiban info workbook (.xlsx, sheet 製番)")
    ap.add_argument("--out-dir", required=True, help="folder for <today>_FlexibleWork.yaml, 01_EnvConfig.yaml, 01_Schedule.yaml")
    ap.add_argument("--report", action="store_true", help="also write <today>_prepare_report.txt (every decision and unresolved item)")
    ap.add_argument("--today", default=None, help="YYYY/MM/DD; modules whose work all ended before it become Fixed (default: today)")
    ap.add_argument("--su-sheets", default="予定表_2026")
    ap.add_argument("--skill-sheet", default="予定表_2026 (2)")
    ap.add_argument("--seiban-start-row", type=int, default=3, help="first 製番 row to read (default 3: the whole list)")
    ap.add_argument("--default-trip-region", default=NEUTRAL_REGION_ID,
                    help="region id for Fixed trips whose region cannot be found (default r_other = away, place unknown); "
                         "'none' skips and reports such trips instead")
    ap.add_argument("--no-fifo-repair", action="store_true",
                    help="do not repair FI/FO without a partner (they are only reported and dropped)")
    ap.add_argument("--chain-days", type=int, default=3,
                    help="FI/FO repair: FO, work, FO (or FI, work, FI) that follow each other within this many days share the "
                         "date of the mark between them, i.e. one trip ends and the next starts on the same date (default 3: covers a weekend)")
    ap.add_argument("--max-gap-days", type=int, default=14,
                    help="FI/FO repair: a busy run in the sheet may have gaps of up to this many days (default 14)")
    args = ap.parse_args(argv)

    today = ymd(args.today) if args.today else date.today().strftime("%Y/%m/%d")
    if today is None:
        ap.error("--today must look like 2026/10/01")
    sheets = tuple(s.strip() for s in args.su_sheets.split(",") if s.strip())
    print("reading SU_Others (this takes a while) ...", flush=True)
    sheet = read_sheet(args.su_others, sheets, args.skill_sheet, args.skill_file)
    seiban = read_seiban(args.seiban_info, args.seiban_start_row)

    out = prepare(load_yaml(args.prev_flexible), load_yaml(args.prev_env), sheet, seiban, today,
                  default_trip_region=None if args.default_trip_region.lower() == "none" else args.default_trip_region,
                  repair_fifo=not args.no_fifo_repair, max_gap_days=args.max_gap_days, chain_days=args.chain_days)
    od = Path(args.out_dir)
    stamp = today.replace("/", "")
    names = [f"{stamp}_FlexibleWork.yaml", "01_EnvConfig.yaml", "01_Schedule.yaml"]
    dump_yaml(od / names[0], out["flexible_work"])
    dump_yaml(od / names[1], out["env"])
    dump_yaml(od / names[2], out["schedule"])
    lines = [f"prepare_schedule report (today = {today})", f"stats: {out['stats']}", ""] + out["report"]
    print("\n".join(lines[:2]))
    if args.report:
        names.append(f"{stamp}_prepare_report.txt")
        (od / names[3]).write_text("\n".join(lines) + "\n", encoding="utf-8")
    else:
        print(f"{sum(1 for r in out['report'] if r.startswith('WARN'))} warnings (run again with --report to write them to a file)")
    print(f"written to {od}: " + ", ".join(names))
    return 0


if __name__ == "__main__":
    sys.exit(main())
