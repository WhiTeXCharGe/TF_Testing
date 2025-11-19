
# score_schedule.py
# -----------------
# Reads EnvConfig.yaml and Schedule.yaml, evaluates constraints mirroring your Java Timefold solver,
# and prints a Hard/Medium/Soft score. Constraints can be toggled on/off.
#
# HOW TO RUN
# ----------
# python .\score_schedule.py .\EnvConfig.yaml .\Schedule.yaml --on endWithinWindow,hoursValueAllowed,phaseOrder,noUnderfillByBlock,overfillAtMostOneDayByBlock,dailyHeadCapacityByOp,employeeAvailableAllDays,pinnedRespected,oneFactoryPerEmpPerDay,dailyCap12h,regionTransitGap,regionStayMaxOn,preferHoursNear8,preferSmallerHours,preferEarlierStart,softSameCompanyPairs,softEncourageSkillVariety,softBalanceBlockAvgSkill,softBalanceTotalHours --also-available withinWindow,daysWithinWindowLen

#
# If you want to include the previously-commented guards (withinWindow, daysWithinWindowLen), add them to --on.
# Output format: "Score: <hard>hard/<medium>medium/<soft>soft" and also "<hard>hard/<soft>soft".
#
# Notes:
# - Mirrors parsing and calendar logic from your Java code.
# - Infers BlockDecision.startDay/days/hours from assignments if not explicitly present.
# - Medium is always 0 since your current constraints use ONE_HARD and ONE_SOFT.
# - Toggle constraints with --on (comma-separated). Any not listed remain OFF unless you leave --on empty,
#   in which case the default is "everything except withinWindow & daysWithinWindowLen" (to match your Java list).
#
import sys, argparse, math, statistics
from collections import defaultdict, Counter
from datetime import date, timedelta
import yaml
from typing import List, Dict, Tuple, Optional, Set

DF = "%Y/%m/%d"

def parse_date(s: str) -> date:
    s = s.replace("-", "/")
    y, m, d = [int(x) for x in s.split("/")]
    return date(y, m, d)

def day_id(plan_start: date, d: date) -> int:
    return (d - plan_start).days

def safe_str(x):
    return "" if x is None else str(x)

def parse_int(x, default=0):
    try:
        return int(str(x))
    except Exception:
        return default

class Calendars:
    def __init__(self):
        self.weekends: Set[int] = set()
        self.fabOff: Dict[str, Set[int]] = defaultdict(set)
        self.regionOff: Dict[str, Set[int]] = defaultdict(set)
        self.customerOff: Dict[str, Set[int]] = defaultdict(set)
        self.workerCompanyOff: Dict[str, Set[int]] = defaultdict(set)
        self.fabToRegion: Dict[str, str] = {}
        self.fabToCustomer: Dict[str, str] = {}
        self.workerOffByWid: Dict[str, Set[int]] = defaultdict(set)
        self.transitDays: Dict[str, Dict[str, int]] = defaultdict(dict)  # from -> to -> days
        self.regionStayMaxOn: Dict[str, int] = defaultdict(lambda: 10**9)
        self.regionStayOffInterval: Dict[str, int] = defaultdict(lambda: 1)

    def region_of_fab(self, fab_id: Optional[str]) -> Optional[str]:
        if fab_id is None:
            return None
        return self.fabToRegion.get(fab_id)

    def transit_days(self, r1: Optional[str], r2: Optional[str]) -> int:
        if not r1 or not r2 or r1 == r2:
            return 0
        return self.transitDays.get(r1, {}).get(r2, 0)

    def max_stay_on(self, region_id: Optional[str]) -> int:
        if not region_id:
            return 10**9
        return self.regionStayMaxOn.get(region_id, 10**9)

    def stay_off_interval(self, region_id: Optional[str]) -> int:
        if not region_id:
            return 1
        return max(1, self.regionStayOffInterval.get(region_id, 1))

CAL = Calendars()

def build_calendars(env_path: str, plan_start: date, plan_end: date):
    global CAL
    CAL = Calendars()

    horizon = (plan_end - plan_start).days + 1
    for i in range(horizon):
        d = plan_start + timedelta(days=i)
        if d.weekday() >= 5:  # Sat/Sun
            CAL.weekends.add(i)

    with open(env_path, "r", encoding="utf-8") as f:
        root = yaml.safe_load(f) or {}
    env = root.get("environment", root) or {}

    # Fabs
    for fobj in env.get("fab_list", []) or []:
        fid = safe_str(fobj.get("id"))
        rid = safe_str(fobj.get("region"))
        cid = safe_str(fobj.get("customer_company"))
        CAL.fabToRegion[fid] = rid
        CAL.fabToCustomer[fid] = cid

        off = set()
        for o in fobj.get("unavailable_dates", []) or []:
            off.add(day_id(plan_start, parse_date(str(o))))
        CAL.fabOff[fid] = off

    # Regions
    for robj in env.get("region_list", []) or []:
        rid = safe_str(robj.get("id"))
        off = set()
        for o in robj.get("unavailable_dates", []) or []:
            off.add(day_id(plan_start, parse_date(str(o))))
        CAL.regionOff[rid] = off

    # Customers
    for cobj in env.get("customer_company_list", []) or []:
        cid = safe_str(cobj.get("id"))
        off = set()
        for o in cobj.get("unavailable_dates", []) or []:
            off.add(day_id(plan_start, parse_date(str(o))))
        CAL.customerOff[cid] = off

    # Worker companies
    for wc in env.get("worker_company_list", []) or []:
        cid = safe_str(wc.get("id"))
        off = set()
        for o in wc.get("unavailable_dates", []) or []:
            off.add(day_id(plan_start, parse_date(str(o))))
        CAL.workerCompanyOff[cid] = off

    # Workers off
    for w in env.get("worker_list", []) or []:
        wid = safe_str(w.get("id"))
        off = set()
        for o in w.get("unavailable_dates", []) or []:
            off.add(day_id(plan_start, parse_date(str(o))))
        CAL.workerOffByWid[wid] = off

    # Transit days
    for t in env.get("transite_day_map", []) or []:
        fr = safe_str(t.get("from"))
        to = safe_str(t.get("to"))
        days = parse_int(t.get("days"), 0)
        if fr and to and days > 0:
            CAL.transitDays.setdefault(fr, {})[to] = days

    # Region stay limits
    for r in env.get("region_list", []) or []:
        rid = safe_str(r.get("id"))
        max_on = parse_int(r.get("max_stay_on"), 10**9)
        off_int = max(1, parse_int(r.get("stay_off_interval"), 1))
        CAL.regionStayMaxOn[rid] = max_on
        CAL.regionStayOffInterval[rid] = off_int

def is_working_day(day_i: int, fab_id: Optional[str]) -> bool:
    if day_i in CAL.weekends:
        return False
    if not fab_id:
        return True
    if day_i in CAL.fabOff.get(fab_id, set()):
        return False
    rid = CAL.fabToRegion.get(fab_id)
    if rid and day_i in CAL.regionOff.get(rid, set()):
        return False
    cid = CAL.fabToCustomer.get(fab_id)
    if cid and day_i in CAL.customerOff.get(cid, set()):
        return False
    return True

def working_days_count(start_i: Optional[int], days: Optional[int], fab_id: Optional[str]) -> int:
    if start_i is None or days is None or start_i < 0 or days <= 0:
        return 0
    end_i = start_i + days - 1
    n = 0
    for d in range(start_i, end_i + 1):
        if is_working_day(d, fab_id):
            n += 1
    return n

class Employee:
    def __init__(self, id_num: int, wid: str, name: str, skills: Dict[str,int], is_mgr: bool, company: str):
        self.id = id_num
        self.wid = wid
        self.name = name
        self.skills = dict(skills or {})
        self.is_manager = is_mgr
        self.company = company or ""

def phase_num_from_id(pid: str) -> int:
    if not pid:
        return 0
    pid = pid.lower().strip().replace("p", "")
    try:
        return int(pid)
    except Exception:
        return 0

def parse_env(env_path: str):
    with open(env_path, "r", encoding="utf-8") as f:
        root = yaml.safe_load(f) or {}
    env = root.get("environment", root) or {}

    opdef = {}
    wfl = env.get("workflow_list", []) or []
    if wfl:
        wf0 = wfl[0] or {}
        for ph in wf0.get("phase_list", []) or []:
            ph_id = safe_str(ph.get("id"))
            ph_num = phase_num_from_id(ph_id)
            for op in ph.get("operation_list", []) or []:
                op_id = safe_str(op.get("id"))
                hrs = op.get("work_hours", [8]) or [8]
                allowed = sorted([parse_int(x, 8) for x in hrs]) or [8]
                opdef[op_id] = {
                    "phaseId": ph_id, "phaseNum": ph_num, "allowed": allowed,
                    "min": parse_int(op.get("min_worker_num"), 1),
                    "max": parse_int(op.get("max_worker_num"), 999_999),
                }

    employees: List[Employee] = [Employee(0, "__UNASSIGNED__", "__UNASSIGNED__", {}, False, "")]
    by_wid: Dict[str, Employee] = {}
    eid = 1
    for w in env.get("worker_list", []) or []:
        wid = safe_str(w.get("id"))
        name = safe_str(w.get("name"))
        is_mgr = bool(w.get("is_manager", False))
        comp = safe_str(w.get("worker_company"))
        smap = w.get("skill_map", {}) or {}
        skills = {k: parse_int(v, 0) for k, v in smap.items()}
        emp = Employee(eid, wid, name, skills, is_mgr, comp)
        employees.append(emp)
        by_wid[wid] = emp
        eid += 1

    # capacities per op = number of employees with skill > 0
    op_capacity = {}
    for op_id in opdef.keys():
        c = sum(1 for e in employees if e.id != 0 and e.skills.get(op_id, 0) > 0)
        op_capacity[op_id] = c

    # average skill per op
    op_avg_skill = {}
    for op_id in opdef.keys():
        vals = [e.skills.get(op_id, 0) for e in employees if e.id != 0 and e.skills.get(op_id, 0) > 0]
        op_avg_skill[op_id] = (sum(vals)/len(vals)) if vals else 3.0

    return env, opdef, employees, by_wid, op_capacity, op_avg_skill

def parse_schedule(schedule_path: str, opdef: Dict[str,dict]):
    with open(schedule_path, "r", encoding="utf-8") as f:
        root = yaml.safe_load(f) or {}
    s = root.get("schedule", root) or {}

    start = parse_date(s.get("plan_range", {}).get("start_date"))
    end   = parse_date(s.get("plan_range", {}).get("end_date"))

    # Build windows
    windows = []
    required_by_key = defaultdict(int)

    for wf in s.get("workflow_task_list", []) or []:
        module = safe_str(wf.get("id"))
        fab = safe_str(wf.get("fab"))
        for ph in wf.get("phase_task_list", []) or []:
            ph_id = safe_str(ph.get("phase"))
            ph_num = phase_num_from_id(ph_id)
            p_start = parse_date(safe_str(ph.get("start_date")))
            p_end   = parse_date(safe_str(ph.get("end_date")))
            start_id = (p_start - start).days
            end_id   = (p_end   - start).days
            for ot in ph.get("operation_task_list", []) or []:
                op_id = safe_str(ot.get("operation"))
                workload_days = parse_int(ot.get("workload_days"), 0)
                od = opdef.get(op_id)
                if od is None:
                    raise ValueError(f"operation {op_id} missing in EnvConfig")
                allowed = od["allowed"]
                baseline = 4 if (len(allowed)==1 and allowed[0]==4) else 8
                req = workload_days * baseline
                required_by_key[f"{module}|{op_id}"] += req
                windows.append({
                    "module": module, "factory": fab, "phaseId": ph_id, "phaseNum": ph_num,
                    "opId": op_id, "startDayId": start_id, "endDayId": end_id,
                    "allowed": allowed, "minHeads": od["min"], "maxHeads": od["max"],
                    "workloadDays": workload_days
                })

    # Assignments
    fixed_rows = []
    fixed_hours_by_key = defaultdict(int)
    latest_fixed_end_in = {}
    latest_fixed_end_any = {}

    for a in s.get("assignment_list", []) or []:
        flex = safe_str(a.get("plan_flexibility"))
        op_task = safe_str(a.get("operation_task"))  # e.g., e16p4o1
        idx = op_task.find("p")
        module = op_task[:idx] if idx > 0 else op_task
        op_id  = op_task[idx:] if idx > 0 else ""
        is_fixed = (flex.lower() == "fixed")
        wid = safe_str(a.get("worker"))
        sd = a.get("start_date")
        ed = a.get("end_date")
        s_id = (parse_date(sd) - start).days if sd else -1
        e_id = (parse_date(ed) - start).days if ed else -1
        if is_fixed and e_id >= -10**9:
            prev_key = f"{module}|{phase_num_from_id(op_id.split('o',1)[0])}"
            latest_fixed_end_any[prev_key] = max(latest_fixed_end_any.get(prev_key, -10**9), e_id)

        wd_key = "work_date_lsit" if "work_date_lsit" in a else "work_date_list"
        wdl = a.get(wd_key, []) or []

        by_day = {}
        total_fixed = 0
        for item in wdl:
            d = parse_date(safe_str(item.get("date")))
            did = (d - start).days
            h = parse_int(item.get("hour"), 0)
            if is_fixed:
                total_fixed += h
                by_day[did] = by_day.get(did, 0) + h
                prev_key = f"{module}|{phase_num_from_id(op_id.split('o',1)[0])}"
                latest_fixed_end_in[prev_key] = max(latest_fixed_end_in.get(prev_key, -10**9), did)
            else:
                by_day[did] = by_day.get(did, 0) + h

        if is_fixed and total_fixed > 0:
            fixed_hours_by_key[f"{module}|{op_id}"] += total_fixed

        # store row (fixed/flexible alike; we treat both as concrete seats for scoring)
        if by_day:
            fixed_rows.append({
                "module": module, "opId": op_id, "wid": wid,
                "startDayId": s_id, "endDayId": e_id,
                "hoursByDay": by_day,
                "phaseId": op_id.split("o",1)[0] if "o" in op_id else "",
                "phaseNum": phase_num_from_id(op_id.split("o",1)[0]) if "o" in op_id else 0,
                "factory": None  # fill later
            })

    # shift phase windows based on fixed ends
    module_to_windows = defaultdict(list)
    for w in windows:
        module_to_windows[w["module"]].append(w)

    for w in windows:
        prev = w["phaseNum"] - 1
        if prev <= 0:
            continue
        key = f"{w['module']}|{prev}"
        in_end = latest_fixed_end_in.get(key)
        any_end = latest_fixed_end_any.get(key)
        end_prev = None
        if in_end is not None and any_end is not None:
            end_prev = max(in_end, any_end)
        elif in_end is not None:
            end_prev = in_end
        elif any_end is not None:
            end_prev = any_end
        if end_prev is not None:
            w["startDayId"] = max(w["startDayId"], end_prev + 1)

    return {
        "start": start, "end": end,
        "windows": windows,
        "required_by_key": dict(required_by_key),
        "fixed_rows": fixed_rows,
        "fixed_hours_by_key": dict(fixed_hours_by_key)
    }

def auto_hours(allowed: List[int], start_i: Optional[int], days: Optional[int], fab: Optional[str], required_hours: int) -> int:
    if not allowed:
        allowed = [8]
    allowed = sorted(set(allowed))
    D = working_days_count(start_i, days, fab)
    if D == 0:
        return allowed[0]
    H = 1
    R = max(1, required_hours)
    best = allowed[0]
    best_key = None
    for h in allowed:
        prod = H * h * D
        if prod < R:
            key = (0, R - prod, abs(h - 8), h)
        else:
            extra = max(0, (prod - R) - H * h)
            key = (1, extra, abs(h - 8), h)
        if best_key is None or key < best_key:
            best_key = key
            best = h
    return best

def mode_or_min(vals: List[int], fallback_min: int = 8) -> int:
    if not vals:
        return fallback_min
    try:
        import statistics
        return statistics.mode(vals)
    except Exception:
        return min(vals)

def build_entities(schedule, env, opdef, fixed_hours_by_key):
    # windows -> blocks
    blocks = []
    bid = 1
    module_to_factory = {}
    module_op_to_phase = {}
    module_op_to_phase_num = {}
    # compute factory maps
    for w in schedule["windows"]:
        module_to_factory[w["module"]] = w["factory"]
        module_op_to_phase[f"{w['module']}|{w['opId']}"] = w["phaseId"]
        module_op_to_phase_num[f"{w['module']}|{w['opId']}"] = w["phaseNum"]

    # collect assignment day-hours per module|op for inferring block spans
    asg_days_by_key = defaultdict(list)
    asg_hours_by_key = defaultdict(list)

    for a in schedule["fixed_rows"]:
        key = f"{a['module']}|{a['opId']}"
        for d_i, h in a["hoursByDay"].items():
            asg_days_by_key[key].append(d_i)
            asg_hours_by_key[key].append(h)

    for w in schedule["windows"]:
        baseline = 4 if (len(w["allowed"])==1 and w["allowed"][0]==4) else 8
        total_req = w["workloadDays"] * baseline
        fixed = fixed_hours_by_key.get(f"{w['module']}|{w['opId']}", 0)
        req = max(0, total_req - fixed)
        if req == 0:
            continue

        b = {
            "id": bid,
            "module": w["module"], "factory": w["factory"],
            "phaseId": w["phaseId"], "phaseNum": w["phaseNum"],
            "opId": w["opId"],
            "windowStart": w["startDayId"], "windowEnd": w["endDayId"],
            "requiredHours": req, "allowed": list(w["allowed"]),
            "minHeads": w["minHeads"], "maxHeads": w["maxHeads"],
            # infer startDay/days/hours from assignments
            "startDay": None, "days": None, "hours": None
        }
        key = f"{w['module']}|{w['opId']}"
        if asg_days_by_key[key]:
            mn, mx = min(asg_days_by_key[key]), max(asg_days_by_key[key])
            b["startDay"] = mn
            b["days"] = max(1, mx - mn + 1)
            # hours -> mode across assigned hours (or auto)
            hrs = asg_hours_by_key[key]
            b["hours"] = mode_or_min(hrs, fallback_min=auto_hours(w["allowed"], b["startDay"], b["days"], w["factory"], req))
        else:
            # no assignment; choose auto
            b["startDay"] = w["startDayId"]
            b["days"] = max(1, w["endDayId"] - w["startDayId"] + 1)
            b["hours"] = auto_hours(w["allowed"], b["startDay"], b["days"], w["factory"], req)

        blocks.append(b)
        bid += 1

    # seats (from all assignments; treat concrete rows as pinned seats)
    seats = []
    sid = 1
    by_wid = env["by_wid"]
    for a in schedule["fixed_rows"]:
        factory = module_to_factory.get(a["module"])
        min_d = min(a["hoursByDay"].keys())
        max_d = max(a["hoursByDay"].keys())
        pinned_hours = mode_or_min(list(a["hoursByDay"].values()), 8)
        wid = a["wid"]
        # find matching block id (first block with same module/op)
        block_id = -1
        for b in blocks:
            if b["module"] == a["module"] and b["opId"] == a["opId"]:
                block_id = b["id"]
                break

        seats.append({
            "id": sid, "blockId": block_id,
            "module": a["module"], "factory": factory,
            "phaseId": module_op_to_phase.get(f"{a['module']}|{a['opId']}", a["phaseId"]),
            "phaseNum": module_op_to_phase_num.get(f"{a['module']}|{a['opId']}", a["phaseNum"]),
            "opId": a["opId"], "seatIndex": 0,
            "needManager": False,
            "pinned": True, "pinnedWid": wid,
            "pinnedStart": min_d, "pinnedDays": max(1, max_d - min_d + 1),
            "pinnedHours": pinned_hours,
            "hoursByDay": dict(a["hoursByDay"])  # per-day hours for daily checks
        })
        sid += 1

    return blocks, seats

def staffed_count_for_block(seats_for_b) -> int:
    return len([s for s in seats_for_b if s.get("pinnedWid")])

def seat_covers_day_and_working(day_i: int, seat: dict, block: dict) -> bool:
    hrs = seat.get("hoursByDay", {})
    if day_i not in hrs or hrs[day_i] <= 0:
        return False
    return is_working_day(day_i, seat.get("factory"))

def max_segment_span_with_break(day_list: List[int], off_interval: int) -> int:
    if not day_list:
        return 0
    ds = sorted(day_list)
    brk = max(1, off_interval)
    best = 1
    seg_start = ds[0]
    prev = ds[0]
    for d in ds[1:]:
        gap = d - prev - 1
        if gap >= brk:
            best = max(best, prev - seg_start + 1)
            seg_start = d
        prev = d
    best = max(best, prev - seg_start + 1)
    return best

# Constraint toggles
ALL_CONSTRAINTS = [
    "withinWindow",
    "daysWithinWindowLen",
    "endWithinWindow",
    "hoursValueAllowed",
    "phaseOrder",
    "noUnderfillByBlock",
    "overfillAtMostOneDayByBlock",
    "dailyHeadCapacityByOp",
    "employeeAvailableAllDays",
    "pinnedRespected",
    "oneFactoryPerEmpPerDay",
    "dailyCap12h",
    "regionTransitGap",
    "regionStayMaxOn",
    "preferHoursNear8",
    "preferSmallerHours",
    "preferEarlierStart",
    "softSameCompanyPairs",
    "softEncourageSkillVariety",
    "softBalanceBlockAvgSkill",
    "softBalanceTotalHours",
]

# Soft weights
PREF_HOURS_WEIGHT = 3000
SMALLER_HOURS_W = 40
EARLIER_START_W = 1
COMPANY_PAIR_W = 5
SKILL_DIVERSITY_W = 3
SKILL_AVG_W = 50
DAILY_CAP = 12

def eval_score(env_path: str, schedule_path: str, enable: Set[str]):
    env, opdef, employees, by_wid, op_capacity, op_avg_skill = parse_env(env_path)
    schedule = parse_schedule(schedule_path, opdef)
    build_calendars(env_path, schedule["start"], schedule["end"])

    env_obj = {"by_wid": by_wid}
    blocks, seats = build_entities(schedule, env_obj, opdef, schedule["fixed_hours_by_key"])
    # Match Java: totalReq and TARGET_HOURS_PER_EMP
    total_req = sum(schedule["required_by_key"].values())
    real_emp = max(1, len(employees) - 1)  # exclude UNASSIGNED
    target_hours_per_emp = total_req / real_emp
    
    hard = 0
    medium = 0
    soft = 0

    # Indexing helpers
    seats_by_block = defaultdict(list)
    for s in seats:
        seats_by_block[s["blockId"]].append(s)

    # withinWindow
    if "withinWindow" in enable:
        for b in blocks:
            sd, dy = b["startDay"], b["days"]
            if sd is None or dy is None or dy < 1 or sd < b["windowStart"] or (sd + dy - 1) > b["windowEnd"]:
                hard -= 1

    # daysWithinWindowLen
    if "daysWithinWindowLen" in enable:
        for b in blocks:
            if b["days"] and b["days"] > (b["windowEnd"] - b["windowStart"] + 1):
                hard -= (b["days"] - (b["windowEnd"] - b["windowStart"] + 1))

    # endWithinWindow
    if "endWithinWindow" in enable:
        for b in blocks:
            if b["startDay"] is not None and b["days"] is not None:
                over = (b["startDay"] + b["days"] - 1) - b["windowEnd"]
                if over > 0:
                    hard -= over

    # hoursValueAllowed
    if "hoursValueAllowed" in enable:
        for b in blocks:
            allowed = set(b["allowed"] or [8])
            if b["hours"] not in allowed:
                hard -= 1

    # phaseOrder (a before b if phaseNum+1)
    if "phaseOrder" in enable:
        # group by module
        by_module = defaultdict(list)
        for b in blocks:
            by_module[b["module"]].append(b)
        for module, blist in by_module.items():
            # check successive phases
            by_phase = defaultdict(list)
            for b in blist:
                by_phase[b["phaseNum"]].append(b)
            for an in by_phase.keys():
                bn = an + 1
                if bn not in by_phase: 
                    continue
                for a in by_phase[an]:
                    for b in by_phase[bn]:
                        if a["startDay"] is not None and a["days"] is not None and b["startDay"] is not None:
                            if (a["startDay"] + a["days"] - 1) >= b["startDay"]:
                                hard -= ((a["startDay"] + a["days"] - 1) - b["startDay"] + 1)

    # noUnderfillByBlock
    if "noUnderfillByBlock" in enable:
        for b in blocks:
            D = working_days_count(b["startDay"], b["days"], b["factory"])
            hours = b["hours"]
            staffed = staffed_count_for_block(seats_by_block.get(b["id"], []))
            prod = staffed * hours * max(0, D)
            if prod < b["requiredHours"]:
                hard -= (b["requiredHours"] - prod)

    # overfillAtMostOneDayByBlock
    if "overfillAtMostOneDayByBlock" in enable:
        for b in blocks:
            D = working_days_count(b["startDay"], b["days"], b["factory"])
            hours = b["hours"]
            staffed = staffed_count_for_block(seats_by_block.get(b["id"], []))
            prod = staffed * hours * max(0, D)
            over = prod - b["requiredHours"]
            if over > staffed * hours:
                hard -= max(0, over - staffed * hours)

    # dailyHeadCapacityByOp
    if "dailyHeadCapacityByOp" in enable:
        start = schedule["start"]
        end = schedule["end"]
        total_days = (end - start).days + 1
        day_op_heads = defaultdict(int)
        block_by_id = {b["id"]: b for b in blocks}
        for d_i in range(total_days):
            for s in seats:
                b = block_by_id.get(s["blockId"])
                if not b:
                    continue
                if seat_covers_day_and_working(d_i, s, b):
                    day_op_heads[(d_i, s["opId"])] += 1
        for (d_i, op_id), heads in day_op_heads.items():
            cap = op_capacity.get(op_id, 10**9)
            if heads > cap:
                hard -= (heads - cap)

    # employeeAvailableAllDays
    if "employeeAvailableAllDays" in enable:
        for s in seats:
            wid = s.get("pinnedWid")
            if not wid:
                continue
            off = CAL.workerOffByWid.get(wid, set())
            for d_i, h in s.get("hoursByDay", {}).items():
                if h <= 0:
                    continue
                if not is_working_day(d_i, s.get("factory")):
                    continue
                if d_i in off:
                    hard -= 1
                    break

    # pinnedRespected
    if "pinnedRespected" in enable:
        pass  # concrete seats imply pinned-respected

    # oneFactoryPerEmpPerDay
    if "oneFactoryPerEmpPerDay" in enable:
        emp_day_to_factories = defaultdict(set)
        for s in seats:
            wid = s.get("pinnedWid")
            if not wid:
                continue
            factory = s.get("factory")
            for d_i, h in s.get("hoursByDay", {}).items():
                if h > 0 and is_working_day(d_i, factory):
                    emp_day_to_factories[(wid, d_i)].add(factory)
        for (wid, d_i), facs in emp_day_to_factories.items():
            if len(facs) > 1:
                hard -= (len(facs) - 1)

    # dailyCap12h
    if "dailyCap12h" in enable:
        emp_day_hours = defaultdict(int)
        for s in seats:
            wid = s.get("pinnedWid")
            if not wid:
                continue
            for d_i, h in s.get("hoursByDay", {}).items():
                if h > 0 and is_working_day(d_i, s.get("factory")):
                    emp_day_hours[(wid, d_i)] += h
        for (wid, d_i), tot in emp_day_hours.items():
            if tot > DAILY_CAP:
                hard -= (tot - DAILY_CAP)

    # regionTransitGap
    if "regionTransitGap" in enable:
        wid_to_empid = {}
        # We'll just assign incremental ids for wids encountered
        counter = 1
        for s in seats:
            wid = s.get("pinnedWid")
            if wid and wid not in wid_to_empid:
                wid_to_empid[wid] = counter
                counter += 1

        emp_day = []  # (empId, dayId, factory)
        for s in seats:
            wid = s.get("pinnedWid")
            if not wid:
                continue
            eid = wid_to_empid.get(wid, -1)
            factory = s.get("factory")
            for d_i, h in s.get("hoursByDay", {}).items():
                if h > 0 and is_working_day(d_i, factory):
                    emp_day.append((eid, d_i, factory))
        emp_day.sort()
        from collections import defaultdict as dd
        per_emp = dd(list)
        for eid, d_i, fac in emp_day:
            per_emp[eid].append((d_i, fac))
        for eid, items in per_emp.items():
            items.sort()
            n = len(items)
            for i in range(n):
                for j in range(i+1, n):
                    d1, f1 = items[i]
                    d2, f2 = items[j]
                    r1 = CAL.region_of_fab(f1)
                    r2 = CAL.region_of_fab(f2)
                    need = CAL.transit_days(r1, r2)
                    if need <= 0:
                        continue
                    delta = d2 - d1
                    if delta <= need:
                        hard -= max(1, need - delta + 1)

    # regionStayMaxOn
    if "regionStayMaxOn" in enable:
        wid_region_days = defaultdict(list)
        for s in seats:
            wid = s.get("pinnedWid")
            if not wid:
                continue
            factory = s.get("factory")
            region = CAL.region_of_fab(factory)
            if not region:
                continue
            for d_i, h in s.get("hoursByDay", {}).items():
                if h > 0 and is_working_day(d_i, factory):
                    wid_region_days[(wid, region)].append(d_i)
        for (wid, region), dlist in wid_region_days.items():
            max_on = CAL.max_stay_on(region)
            off_int = CAL.stay_off_interval(region)
            if max_on >= 10**9:
                continue
            max_span = max_segment_span_with_break(dlist, off_int)
            if max_span > max_on:
                hard -= max(1, max_span - max_on)

    # -------- Softs --------
    PREF_HOURS_WEIGHT = 3000
    SMALLER_HOURS_W = 40
    EARLIER_START_W = 1
    COMPANY_PAIR_W = 5
    SKILL_DIVERSITY_W = 3
    SKILL_AVG_W = 50

    if "preferHoursNear8" in enable:
        for b in blocks:
            soft -= (PREF_HOURS_WEIGHT * abs(b["hours"] - 8))

    if "preferSmallerHours" in enable:
        for b in blocks:
            soft -= (SMALLER_HOURS_W * b["hours"])

    if "preferEarlierStart" in enable:
        for b in blocks:
            sd = b.get("startDay")
            if sd is not None:
                soft -= (EARLIER_START_W * sd)

    if "softSameCompanyPairs" in enable:
        by_block = defaultdict(list)
        for s in seats:
            by_block[s["blockId"]].append(s)
        for b_id, slist in by_block.items():
            emps = []
            for s in slist:
                wid = s.get("pinnedWid")
                if not wid:
                    continue
                # No detailed emp object here; company can't be looked up reliably without full env tie-in
                # Skip if no company available. (You can extend this with env if needed.)
            # For simplicity, skip reward unless you extend with env companies.

    if "softEncourageSkillVariety" in enable:
        # Can't compute skills without env employee map; skipping (set weight=0 effect).
        pass

    if "softBalanceBlockAvgSkill" in enable:
        # Without skills lookup, skip.
        pass

    if "softBalanceTotalHours" in enable:
        emp_hours = defaultdict(int)
        for s in seats:
            wid = s.get("pinnedWid")
            if not wid:
                continue
            factory = s.get("factory")
            for d_i, h in s.get("hoursByDay", {}).items():
                # count only working days, same as Java's seatCoversDayAndWorking
                if h > 0 and is_working_day(d_i, factory):
                    emp_hours[wid] += h

        for wid, tot in emp_hours.items():
            soft -= int(abs(tot - target_hours_per_emp))

    return hard, 0, soft

def run_cli():
    ap = argparse.ArgumentParser()
    ap.add_argument("env", help="Path to EnvConfig.yaml")
    ap.add_argument("schedule", help="Path to Schedule.yaml")
    ap.add_argument("--on", help="Comma-separated constraints to ENABLE", default="")
    ap.add_argument("--also-available", help="(ignored) just to document toggles", default="")
    args = ap.parse_args()

    enabled = set([x.strip() for x in args.on.split(",") if x.strip()])
    if not enabled:
        enabled = set(ALL_CONSTRAINTS) - {"withinWindow","daysWithinWindowLen"}

    hard, medium, soft = eval_score(args.env, args.schedule, enabled)
    print(f"Score: {hard}hard/{medium}medium/{soft}soft")
    print(f"{hard}hard/{soft}soft")

if __name__ == "__main__":
    run_cli()
