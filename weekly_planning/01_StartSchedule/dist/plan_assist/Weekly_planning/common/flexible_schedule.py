"""
Reading the previous week's FlexibleSchedule.yaml (or a Timefold result Schedule.yaml): modules, their assignments,
cell names and trips.
"""
from __future__ import annotations

from collections import defaultdict

from .dates import ymd
from .text import name_key, tool_code
from .trips import flatten_trips


def read_prev_flexible(doc, env_worker_names):
    """
    doc: parsed FlexibleSchedule.yaml / Timefold result Schedule.yaml.
    env_worker_names: prev worker id -> name (from the previous EnvConfig).
    Returns dict: plan_range, modules {code: {...}}, trips [{worker_key, worker_id, start, end, region, flex, raw}], warnings.
    A module is identified by its workflow task `name` (the module code, e.g. 530N00009A).
    modules[code]["names"] = {(worker_key, date): cell_name} from the `cell_name` of the assignment rows (empty if the file has none).
    """
    sch = doc.get("schedule", doc) if isinstance(doc, dict) else {}
    warnings = []
    tasks = sch.get("workflow_task_list") or []
    modules, op_to_code = {}, {}
    for t in tasks:
        code = tool_code(t.get("name")) or str(t.get("name") or t.get("id"))
        ops = set()
        for ph in t.get("phase_task_list") or []:
            for op in ph.get("operation_task_list") or []:
                ops.add(op["id"])
        if code in modules:
            warnings.append(f"duplicate module name {code} in the previous schedule: the later one is used")
        modules[code] = {"task": t, "ops": ops, "assign": set(), "rows": [], "last": None, "names": {}}
        for o in ops:
            op_to_code[o] = code
    misc_code = {}   # misc task id -> module code written in its name (the SU_Others parser names converted cells after the module)
    for mt in sch.get("misc_task_list") or []:
        c = tool_code(mt.get("name"))
        if c:
            misc_code[mt.get("id")] = c
    misc_cover = defaultdict(set)   # module code -> {(worker_key, date)} already held by a misc task of that module
    for a in sch.get("assignment_list") or []:
        code = op_to_code.get(a.get("operation_task"))
        if code is None:
            c = misc_code.get(a.get("operation_task"))
            if c:
                wk = name_key(env_worker_names.get(a.get("worker"), a.get("worker")))
                for item in a.get("work_date_list") or []:
                    d = ymd(item.get("date"))
                    if d:
                        misc_cover[c].add((wk, d))
            continue   # misc row (or unknown task): not part of a flexible module
        wid = a.get("worker")
        wname = env_worker_names.get(wid)
        if wname is None:
            warnings.append(f"assignment of module {code}: worker id {wid!r} is not in the previous EnvConfig")
            wname = str(wid)
        m = modules[code]
        m["rows"].append(a)
        cn = a.get("cell_name")
        cn = cn.strip() if isinstance(cn, str) else ""
        for item in a.get("work_date_list") or []:
            d = ymd(item.get("date"))
            if d:
                m["assign"].add((name_key(wname), d))
                m["last"] = d if m["last"] is None or d > m["last"] else m["last"]
                if cn:
                    m["names"][(name_key(wname), d)] = cn      # the text this cell had when the plan was made
    trips = []
    for entry in flatten_trips(sch.get("trip_list")):
        wid = entry["worker"]
        wname = env_worker_names.get(wid, wid)
        s, e = ymd(entry["trip"].get("start_date")), ymd(entry["trip"].get("end_date"))
        if not (s and e):
            warnings.append(f"previous trip of {wid} has an unreadable date and is ignored: {entry['trip']}")
            continue
        flex = str(entry["trip"].get("plan_flexibility") or "Flexible")
        trips.append({"worker_key": name_key(wname), "worker_id": wid, "start": s, "end": e,
                      "region": entry["trip"].get("region"), "flex": flex})
    pr = sch.get("plan_range") or {}
    return {"plan_range": pr, "modules": modules, "trips": trips, "warnings": warnings, "misc_cover": dict(misc_cover)}
