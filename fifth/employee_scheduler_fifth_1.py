from __future__ import annotations
from dataclasses import dataclass, field
from datetime import date, datetime, timedelta
from typing import Annotated, Optional, Dict, List, Tuple, Set
import yaml
from collections import defaultdict
import time

from timefold.solver.domain import (
    planning_entity, planning_solution,
    PlanningId, PlanningVariable,
    PlanningEntityCollectionProperty, ProblemFactCollectionProperty,
    PlanningScore, ValueRangeProvider
)
from timefold.solver import SolverFactory
    # If your package version differs, adjust these imports accordingly.
from timefold.solver.config import SolverConfig, TerminationConfig, ScoreDirectorFactoryConfig, Duration
from timefold.solver.score import HardSoftScore, ConstraintFactory, Constraint, Joiners, constraint_provider, ConstraintCollectors

# -------------------- Domain --------------------

@dataclass(frozen=True)
class DaySlot:
    id: int
    d: date  # calendar date

@dataclass(frozen=True)
class Employee:
    id: int
    name: str
    # key: "P{process}-{task}" -> level 1..5 (must exist to be eligible)
    skills: Dict[str, int]
    unavailable_day_ids: Set[int] = field(default_factory=set)  # personal blackout (by day.id)

@dataclass(frozen=True)
class TaskWindow:
    """Task window within a module/process: allowed day range for that task."""
    module: str
    process_id: int
    task_letter: str
    start_day_id: int    # inclusive
    end_day_id: int      # inclusive
    workload: int        # total slots to schedule (1 slot = 1 day for now)

    def pcode(self) -> str: return f"P{self.process_id}-{self.task_letter}"
    def tcode(self) -> str: return f"{self.module}-P{self.process_id}-{self.task_letter}"

@planning_entity
@dataclass
class RequirementSlot:
    """One person-day of work for a specific (module, process, task)."""
    id: Annotated[int, PlanningId]
    module: str
    process_id: int
    task_letter: str
    start_day_id: int
    end_day_id: int
    # Planning vars:
    employee: Annotated[Optional[Employee], PlanningVariable] = field(default=None)
    day:      Annotated[Optional[DaySlot],  PlanningVariable] = field(default=None)

    # helpers
    def pcode(self) -> str: return f"P{self.process_id}-{self.task_letter}"
    def tcode(self) -> str: return f"{self.module}-P{self.process_id}-{self.task_letter}"

@planning_solution
@dataclass
class Schedule:
    days:      Annotated[List[DaySlot], ProblemFactCollectionProperty, ValueRangeProvider]
    employees: Annotated[List[Employee], ProblemFactCollectionProperty, ValueRangeProvider]
    reqs:      Annotated[List[RequirementSlot], PlanningEntityCollectionProperty]
    score:     Annotated[HardSoftScore, PlanningScore] = field(default=None)

# -------------------- Config knobs / globals --------------------

# Working week
_BUSINESS_DAYS: Set[int] = {0,1,2,3,4}  # Mon..Fri by default

# Multi-level calendars (day.id sets)
_COMPANY_UNAVAIL: Dict[str, Set[int]] = {}
_COUNTRY_UNAVAIL: Dict[str, Set[int]] = {}
_MODULE_UNAVAIL:  Dict[str, Set[int]] = {}

# Module metadata
_MODULE_COMPANY: Dict[str, str] = {}
_MODULE_COUNTRY: Dict[str, str] = {}
_MODULE_CHANGEOVER: Dict[str, Dict[str, int]] = {}  # from -> {to: gapDays}

# Quantum (for future: per-day resolution) + global monthly cap (quanta)
_QUANTA_PER_DAY: int = 4
_MONTHLY_CAP_QUANTA: Optional[int] = None  # if provided in YAML

# Day id -> (year, month) for monthly reporting
_DAYID_YYYYMM: Dict[int, Tuple[int,int]] = {}

_VISA_LIMITS: Dict[str, int] = {}              # country -> max calendar days per stay
_VISA_GAP_BREAK_DAYS: int = 0                  # reset threshold between stays

_ANNUAL_LIMITS: Dict[str, int] = {}            # country -> max calendar days per segment
_ANNUAL_BREAK_DAYS: int = 0                    # reset threshold for annual segmentation

_COUNTRY_CHANGEOVER: Dict[str, Dict[str, int]] = {}  # from country -> {to country: gap}

# Staffing cap per task per day (read from YAML)
_STAFF_MIN_PER_DAY: int = 1
_STAFF_MAX_PER_DAY: int = 10**9
# -------------------- Constraints --------------------

@constraint_provider
def define_constraints(cf: ConstraintFactory) -> List[Constraint]:
    return [
        # HARD
        require_employee_assigned(cf),
        require_day_assigned(cf),
        day_within_window(cf),
        skill_must_exist(cf),
        employee_not_double_booked_same_day(cf),
        process_precedence_within_module(cf),
        employee_not_on_unavailable_day(cf),
        no_weekend_work(cf),
        module_company_country_unavailable(cf),

        # HARD: staffing min/max (exact via collectors)
        staffing_minmax_combined(cf),

        # HARD: travel + presence
        country_changeover_gap(cf),
        visa_presence_limit(cf),
        annual_presence_limit(cf),

        # SOFT: pack early and tight
        finish_asap(cf),
        minimize_task_makespan(cf),  # NEW
        pack_to_daily_max(cf),       # NEW

        # keep your existing softs
        gently_prefer_levels_near_3(cf),
        prefer_continuity(cf),
        balance_workload(cf),
    ]


# ---- HARD ----

def require_employee_assigned(cf: ConstraintFactory) -> Constraint:
    return (cf.for_each(RequirementSlot)
            .filter(lambda r: r.employee is None)
            .penalize(HardSoftScore.ONE_HARD)
            .as_constraint("Employee must be assigned"))

def require_day_assigned(cf: ConstraintFactory) -> Constraint:
    return (cf.for_each(RequirementSlot)
            .filter(lambda r: r.day is None)
            .penalize(HardSoftScore.ONE_HARD)
            .as_constraint("Day must be assigned"))

def day_within_window(cf: ConstraintFactory) -> Constraint:
    def outside(r: RequirementSlot) -> bool:
        if r.day is None:
            return True
        return not (r.start_day_id <= r.day.id <= r.end_day_id)
    return (cf.for_each(RequirementSlot)
            .filter(outside)
            .penalize(HardSoftScore.ONE_HARD)
            .as_constraint("Day outside task window (deadline or start)"))

def skill_must_exist(cf: ConstraintFactory) -> Constraint:
    def lacks(r: RequirementSlot) -> bool:
        if r.employee is None: return True
        return r.employee.skills.get(r.pcode(), 0) < 1
    return (cf.for_each(RequirementSlot)
            .filter(lacks)
            .penalize(HardSoftScore.ONE_HARD)
            .as_constraint("Missing skill for task"))

def employee_not_double_booked_same_day(cf: ConstraintFactory) -> Constraint:
    return (
        cf.for_each_unique_pair(
            RequirementSlot,
            Joiners.equal(lambda r: r.employee),
            Joiners.equal(lambda r: r.day.id if r.day is not None else None)
        )
        .filter(lambda a, b:
            a.employee is not None and b.employee is not None and
            a.day is not None and b.day is not None
        )
        .penalize(HardSoftScore.ONE_HARD)
        .as_constraint("Employee double-booked same day")
    )

def process_precedence_within_module(cf: ConstraintFactory) -> Constraint:
    """Inside SAME module: any work of process p+1 must be strictly AFTER work of process p."""
    def violates(a: RequirementSlot, b: RequirementSlot) -> bool:
        if a.day is None or b.day is None:
            return True
        return not (b.day.id > a.day.id)
    return (
        cf.for_each_unique_pair(
            RequirementSlot,
            Joiners.equal(lambda r: r.module),
            Joiners.less_than(lambda r: r.process_id)
        )
        .filter(violates)
        .penalize(HardSoftScore.ONE_HARD)
        .as_constraint("Module process precedence")
    )

def employee_not_on_unavailable_day(cf: ConstraintFactory) -> Constraint:
    return (
        cf.for_each(RequirementSlot)
        .filter(lambda r:
            (r.employee is not None) and
            (r.day is not None) and
            (r.day.id in r.employee.unavailable_day_ids)
        )
        .penalize(HardSoftScore.ONE_HARD)
        .as_constraint("Employee unavailable day")
    )

def no_weekend_work(cf: ConstraintFactory) -> Constraint:
    return (
        cf.for_each(RequirementSlot)
        .filter(lambda r:
            r.day is not None and r.day.d.weekday() not in _BUSINESS_DAYS
        )
        .penalize(HardSoftScore.ONE_HARD)
        .as_constraint("No weekend work")
    )

# NEW: Calendar closure by module/company/country
def module_company_country_unavailable(cf: ConstraintFactory) -> Constraint:
    def closed(r: RequirementSlot) -> bool:
        if r.day is None:
            return True  # force assignment to real days
        did = r.day.id
        m   = r.module

        # module-level
        s_mod = _MODULE_UNAVAIL.get(m)
        if (s_mod is not None) and (did in s_mod):
            return True

        # company-level
        comp = _MODULE_COMPANY.get(m)
        if comp is not None:
            s_comp = _COMPANY_UNAVAIL.get(comp)
            if (s_comp is not None) and (did in s_comp):
                return True

        # country-level
        ctry = _MODULE_COUNTRY.get(m)
        if ctry is not None:
            s_ctry = _COUNTRY_UNAVAIL.get(ctry)
            if (s_ctry is not None) and (did in s_ctry):
                return True

        return False

    return (
        cf.for_each(RequirementSlot)
        .filter(closed)
        .penalize(HardSoftScore.ONE_HARD)
        .as_constraint("Closed by module or company or country calendar")
    )

def country_changeover_gap(cf: ConstraintFactory) -> Constraint:
    def violates(a: RequirementSlot, b: RequirementSlot) -> bool:
        # a.day < b.day from Joiners.less_than
        if a.employee is None or b.employee is None or a.day is None or b.day is None:
            return False  # no violation if incomplete

        # SAFE defaults: use "" instead of the type 'str'
        c1 = _MODULE_COUNTRY.get(a.module, "")
        c2 = _MODULE_COUNTRY.get(b.module, "")

        # If either country unknown/blank, don't enforce
        if c1 == "" or c2 == "":
            return False

        # Same-country? no changeover gap needed
        if c1 == c2:
            return False

        gap = _COUNTRY_CHANGEOVER.get(c1, {}).get(c2, 0)
        # require at least `gap` empty days BETWEEN a and b:
        # b.day.id >= a.day.id + gap + 1
        return not (b.day.id >= a.day.id + gap + 1)

    return (
        cf.for_each_unique_pair(
            RequirementSlot,
            Joiners.equal(lambda r: r.employee),
            Joiners.less_than(lambda r: r.day.id if r.day is not None else -10)
        )
        .filter(violates)
        .penalize(HardSoftScore.ONE_HARD)
        .as_constraint("Country changeover gap")
    )



def staffing_minmax_combined(cf: ConstraintFactory) -> Constraint:
    """
    For each (module, process, task, day), enforce MIN and MAX headcount.
    Single group + single penalty function keeps it compact.
    """
    def penalty(count: int) -> int:
        if count < _STAFF_MIN_PER_DAY:
            return (_STAFF_MIN_PER_DAY - count)  # shortfall
        if count > _STAFF_MAX_PER_DAY:
            return (count - _STAFF_MAX_PER_DAY)  # overflow
        return 0

    return (
        cf.for_each(RequirementSlot)
        .filter(lambda r: r.day is not None)
        .group_by(lambda r: (r.module, r.process_id, r.task_letter, r.day.id),
                  ConstraintCollectors.count())
        .filter(lambda key, cnt: cnt < _STAFF_MIN_PER_DAY or cnt > _STAFF_MAX_PER_DAY)
        .penalize(HardSoftScore.ONE_HARD, lambda key, cnt: penalty(cnt))
        .as_constraint("Daily staffing min or max")
    )


def _segments_from_days(day_ids: List[int], break_days: int) -> List[Tuple[int, int]]:
    """
    Build contiguous segments from sorted day IDs.
    Avoid Python truthiness on Java-backed lists (JPyInterpreter):
    use explicit len() checks instead of "if not day_ids".
    """
    if day_ids is None or len(day_ids) == 0:
        return []
    s = sorted(set(int(x) for x in day_ids))
    segs: List[Tuple[int,int]] = []
    cur_start = s[0]
    prev = s[0]
    gap = max(1, int(break_days))
    for d in s[1:]:
        if (d - prev) >= gap:
            segs.append((cur_start, prev))
            cur_start = d
        prev = d
    segs.append((cur_start, prev))
    return segs


def visa_presence_limit(cf: ConstraintFactory) -> Constraint:
    def key_emp_country(r: RequirementSlot):
        if r.employee is None or r.day is None:
            return None
        ctry = _MODULE_COUNTRY.get(r.module, None)
        if not ctry:
            return None
        return (r.employee.id, ctry)

    def overstay_penalty(days, ctry: str) -> int:
        if days is None or len(days) == 0:
            return 0
        limit = _VISA_LIMITS.get(ctry, None)
        gap = max(1, int(_VISA_GAP_BREAK_DAYS))
        if limit is None:
            return 0
        pen = 0
        for a, b in _segments_from_days(days, gap):
            span = (b - a) + 1
            if span > limit:
                pen += (span - limit)
        return int(pen)

    return (
        cf.for_each(RequirementSlot)
        .filter(lambda r: r.employee is not None and r.day is not None)
        .group_by(key_emp_country, ConstraintCollectors.to_list(lambda r: r.day.id))
        .filter(lambda key, day_list: key is not None)
        .penalize(HardSoftScore.ONE_HARD, lambda key, day_list: overstay_penalty(day_list, key[1]))
        .as_constraint("Visa stay limit (per country)")
    )


def annual_presence_limit(cf: ConstraintFactory) -> Constraint:
    def key_emp_country(r: RequirementSlot):
        if r.employee is None or r.day is None:
            return None
        ctry = _MODULE_COUNTRY.get(r.module, None)
        if not ctry:
            return None
        return (r.employee.id, ctry)

    def over_annual_penalty(days, ctry: str) -> int:
        if days is None or len(days) == 0:
            return 0
        limit = _ANNUAL_LIMITS.get(ctry, None)
        br = max(1, int(_ANNUAL_BREAK_DAYS))
        if limit is None:
            return 0
        total = 0
        for a, b in _segments_from_days(days, br):
            total += (b - a) + 1
        return int(max(0, total - limit))

    return (
        cf.for_each(RequirementSlot)
        .filter(lambda r: r.employee is not None and r.day is not None)
        .group_by(key_emp_country, ConstraintCollectors.to_list(lambda r: r.day.id))
        .filter(lambda key, day_list: key is not None)
        .penalize(HardSoftScore.ONE_HARD, lambda key, day_list: over_annual_penalty(day_list, key[1]))
        .as_constraint("Annual presence limit (per country)")
    )






# ---- SOFT ----

def finish_asap(cf: ConstraintFactory) -> Constraint:
    def day_cost(r: RequirementSlot) -> int:
        return r.day.id if r.day is not None else 1000
    return (
        cf.for_each(RequirementSlot)
        .penalize(HardSoftScore.ONE_SOFT, lambda r: 5 * day_cost(r))
        .as_constraint("Finish ASAP")
    )

def minimize_task_makespan(cf: ConstraintFactory) -> Constraint:
    """
    SOFT: compact each task window by penalizing (max_day - min_day) actually used.
    Works on assigned days only.
    """
    WEIGHT = 50  # tune up to compress more
    def span(day_ids):
        if day_ids is None:
            return 0
        n = len(day_ids)
        if n == 0:
            return 0
        days = sorted(day_ids)
        return int(days[-1] - days[0])
    return (
        cf.for_each(RequirementSlot)
        .filter(lambda r: r.day is not None)
        .group_by(lambda r: (r.module, r.process_id, r.task_letter),
                  ConstraintCollectors.to_list(lambda r: r.day.id))
        .penalize(HardSoftScore.of_soft(WEIGHT), lambda key, dlist: span(dlist))
        .as_constraint("Minimize task makespan")
    )


def pack_to_daily_max(cf: ConstraintFactory) -> Constraint:
    """
    SOFT: prefer filling each (module,process,task,day) up to the max.
    Encourages early days to hit max headcount and finish sooner.
    """
    SLACK_WEIGHT = 5  # tune
    return (
        cf.for_each(RequirementSlot)
        .filter(lambda r: r.day is not None)
        .group_by(lambda r: (r.module, r.process_id, r.task_letter, r.day.id),
                  ConstraintCollectors.count())
        .penalize(HardSoftScore.of_soft(SLACK_WEIGHT),
                  lambda key, cnt: max(0, _STAFF_MAX_PER_DAY - cnt))
        .as_constraint("Pack toward daily max")
    )


def gently_prefer_levels_near_3(cf: ConstraintFactory) -> Constraint:
    def lvl_cost(r: RequirementSlot) -> int:
        if r.employee is None: return 0
        lvl = r.employee.skills.get(r.pcode(), 1)
        return max(0, abs(lvl - 3) // 3)  # 0 or 1 only
    return (
        cf.for_each(RequirementSlot)
        .penalize(HardSoftScore.ONE_SOFT, lvl_cost)
        .as_constraint("Tiny level-centering")
    )

def prefer_continuity(cf: ConstraintFactory) -> Constraint:
    return (
        cf.for_each_unique_pair(
            RequirementSlot,
            Joiners.equal(lambda r: r.employee),
            Joiners.equal(lambda r: r.module),
            Joiners.equal(lambda r: r.process_id),
            Joiners.equal(lambda r: r.task_letter),
            Joiners.less_than(lambda r: r.day.id if (r.day is not None) else -1)
        )
        .filter(lambda a, b:
            a.employee is not None and b.employee is not None and
            a.day is not None and b.day is not None and
            b.day.id == a.day.id + 1
        )
        .reward(HardSoftScore.ONE_SOFT)
        .as_constraint("Prefer continuity")
    )

# NEW: balance workload (encourage even spread)
def balance_workload(cf: ConstraintFactory) -> Constraint:
    # Penalize each unique pair of assignments for the same employee (on different days).
    # With fixed total workload, minimizing sum of pairs ≈ equalizing load.
    return (
        cf.for_each_unique_pair(
            RequirementSlot,
            Joiners.equal(lambda r: r.employee)
        )
        .filter(lambda a, b: a.employee is not None and b.employee is not None and a != b)
        .penalize(HardSoftScore.ONE_SOFT)
        .as_constraint("Balance workload across employees")
    )

# -------------------- YAML loading & problem build --------------------

@dataclass
class ModuleSpec:
    code: str
    start_day_id: int

def load_config_modules(path: str):
    """
    Expected YAML (v5-lite fields used here):
      start_day: 2025-09-01
      horizon_days: 80
      business_days: ["Mon","Tue","Wed","Thu","Fri"]
      module_changeover_days: { S1: {S2: 1, ...}, ... }
      calendars:
        company: { A: {unavailable: [..]}, ... }
        country: { A: {unavailable: [..]}, ... }
        module:  { S1: {unavailable: [..]}, ... }
      quantum:
        per_day: 4
        capacity_quanta_per_month: 88  # global cap (optional)
      modules:
        - code: S1
          company: A
          country: A
          start_date: 2025-09-01
          processes:
            - id: 1
              end_date: 2025-09-15
              tasks:
                - code: S1-P1-A
                  workload_days: 12
                ...
      employees:
        - name: AA
          skills: { P1-A: 3, ... }
          unavailable: [2025-09-10]
    """
    with open(path, "r", encoding="utf-8") as f:
        cfg = yaml.safe_load(f)

    # Working week
    global _BUSINESS_DAYS
    wd_map = {"Mon":0,"Tue":1,"Wed":2,"Thu":3,"Fri":4,"Sat":5,"Sun":6}
    _BUSINESS_DAYS = set(wd_map[d] for d in cfg.get("business_days", ["Mon","Tue","Wed","Thu","Fri"]))

    # Horizon & calendar index
    start_day = datetime.strptime(str(cfg["start_day"]), "%Y-%m-%d").date()
    horizon_days = int(cfg.get("horizon_days", 30))
    days: List[DaySlot] = []
    for i in range(horizon_days):
        d = start_day + timedelta(days=i)
        days.append(DaySlot(i, d))
        _DAYID_YYYYMM[i] = (d.year, d.month)

    def align_to_business(d: date) -> date:
        while d.weekday() not in _BUSINESS_DAYS:
            d += timedelta(days=1)
        return d

    def day_to_idx(datestr: str) -> int:
        d = datetime.strptime(str(datestr), "%Y-%m-%d").date()
        d = align_to_business(d)
        idx = (d - start_day).days
        return max(0, min(horizon_days - 1, idx))

    # Quantum config
    global _QUANTA_PER_DAY, _MONTHLY_CAP_QUANTA
    q = cfg.get("quantum", {}) or {}
    _QUANTA_PER_DAY = int(q.get("per_day", 4))
    _MONTHLY_CAP_QUANTA = q.get("capacity_quanta_per_month", None)
    if _MONTHLY_CAP_QUANTA is not None:
        _MONTHLY_CAP_QUANTA = int(_MONTHLY_CAP_QUANTA)

    # Calendars (org-level)
    global _COMPANY_UNAVAIL, _COUNTRY_UNAVAIL, _MODULE_UNAVAIL
    _COMPANY_UNAVAIL = {}
    _COUNTRY_UNAVAIL = {}
    _MODULE_UNAVAIL = {}

    global _STAFF_MIN_PER_DAY, _STAFF_MAX_PER_DAY
    staff_cfg = cfg.get("staffing_per_task_per_day", {}) or {}
    _STAFF_MIN_PER_DAY = int(staff_cfg.get("min", 1))
    _STAFF_MAX_PER_DAY = int(staff_cfg.get("max", 10**9))
    print(f"[STAFF LIMITS] min={_STAFF_MIN_PER_DAY} max={_STAFF_MAX_PER_DAY}")
    # --- NEW: visa limits ---
    global _VISA_LIMITS, _VISA_GAP_BREAK_DAYS
    _VISA_LIMITS = {str(k): int(v) for k, v in (cfg.get("visa_limits", {}).get("countries", {}) or {}).items()}
    _VISA_GAP_BREAK_DAYS = int((cfg.get("visa_limits", {}) or {}).get("presence_gap_break_days", 0))

    # --- NEW: annual presence limits ---
    global _ANNUAL_LIMITS, _ANNUAL_BREAK_DAYS
    _ANNUAL_LIMITS = {str(k): int(v) for k, v in (cfg.get("annual_limits", {}).get("countries", {}) or {}).items()}
    _ANNUAL_BREAK_DAYS = int((cfg.get("annual_limits", {}) or {}).get("break_days", 0))

    # --- NEW: country changeover gaps ---
    global _COUNTRY_CHANGEOVER
    _COUNTRY_CHANGEOVER = {src: {dst: int(g) for dst, g in (inner or {}).items()}
                           for src, inner in (cfg.get("country_changeover_days", {}) or {}).items()}
    
    cal = cfg.get("calendars", {}) or {}
    for comp, obj in (cal.get("company", {}) or {}).items():
        s = set()
        for ds in obj.get("unavailable", []) or []:
            s.add(day_to_idx(ds))
        _COMPANY_UNAVAIL[comp] = s
    for ctry, obj in (cal.get("country", {}) or {}).items():
        s = set()
        for ds in obj.get("unavailable", []) or []:
            s.add(day_to_idx(ds))
        _COUNTRY_UNAVAIL[ctry] = s
    for mod, obj in (cal.get("module", {}) or {}).items():
        s = set()
        for ds in obj.get("unavailable", []) or []:
            s.add(day_to_idx(ds))
        _MODULE_UNAVAIL[mod] = s

    # Module changeovers
    global _MODULE_CHANGEOVER
    _MODULE_CHANGEOVER = {m: {n:int(v) for n,v in d.items()} for m,d in (cfg.get("module_changeover_days", {}) or {}).items()}

    windows: List[TaskWindow] = []

    def _business_days_between(i0: int, i1: int) -> int:
        # inclusive range [i0, i1]
        cnt = 0
        for i in range(i0, i1 + 1):
            if (start_day + timedelta(days=i)).weekday() in _BUSINESS_DAYS:
                cnt += 1
        return cnt

    warnings = []
    for w in windows:
        days_in_window = _business_days_between(w.start_day_id, w.end_day_id)
        max_capacity = _STAFF_MAX_PER_DAY * days_in_window
        if w.workload > max_capacity:
            warnings.append(
                f"[WARN] {w.module}-P{w.process_id}-{w.task_letter}: "
                f"workload={w.workload} > max_cap={max_capacity} "
                f"(days={days_in_window}, max/day={_STAFF_MAX_PER_DAY})"
            )
    if warnings:
        print("\n".join(warnings))

    # ---- Modules ----
    global _MODULE_COMPANY, _MODULE_COUNTRY
    _MODULE_COMPANY, _MODULE_COUNTRY = {}, {}

    modules_cfg = cfg["modules"]
    for m in modules_cfg:
        mcode = str(m["code"]).strip()
        _MODULE_COMPANY[mcode] = str(m.get("company", "")).strip()
        _MODULE_COUNTRY[mcode] = str(m.get("country", "")).strip()

        module_start_idx = day_to_idx(m.get("start_date", cfg["start_day"]))

        for proc in m.get("processes", []):
            pid = int(proc["id"])
            p_end_idx = day_to_idx(proc["end_date"])

            # (Optional) per-process start override; otherwise inherit module start
            p_start_idx = module_start_idx
            if "start_date" in proc and proc["start_date"]:
                p_start_idx = max(p_start_idx, day_to_idx(proc["start_date"]))

            for t in proc.get("tasks", []):
                full_code = str(t["code"]).strip().upper()  # e.g. "S1-P2-A"
                parts = full_code.split("-")
                if len(parts) != 3 or not parts[1].startswith("P"):
                    raise ValueError(f"Bad task code '{full_code}' (expected 'Sx-Py-Z').")
                letter = parts[2]

                # NOTE: per your request, no per-task end_date; it inherits process end
                windows.append(TaskWindow(
                    module=mcode,
                    process_id=pid,
                    task_letter=letter,
                    start_day_id=p_start_idx,
                    end_day_id=p_end_idx,
                    workload=int(t["workload_days"])  # use workload_days from YAML
                ))

    # ---- Employees ----
    employees: List[Employee] = []
    eid = 1
    for e in cfg["employees"]:
        name = str(e["name"])
        skills = {str(k).strip().upper(): int(v) for k, v in (e.get("skills", {}) or {}).items()}

        # personal blackout
        unavailable_ids: Set[int] = set()
        for ds in (e.get("unavailable", []) or []):
            try:
                unavailable_ids.add(day_to_idx(str(ds)))
            except Exception:
                pass

        employees.append(Employee(eid, name, skills, unavailable_ids))
        eid += 1

    return start_day, days, windows, employees

def build_requirement_slots(windows: List[TaskWindow]) -> List[RequirementSlot]:
    reqs: List[RequirementSlot] = []
    rid = 1
    for w in windows:
        for _ in range(w.workload):
            reqs.append(RequirementSlot(
                id=rid,
                module=w.module,
                process_id=w.process_id,
                task_letter=w.task_letter,
                start_day_id=w.start_day_id,
                end_day_id=w.end_day_id
            ))
            rid += 1
    return reqs

# -------------------- Score/report helpers --------------------

def explain_and_write(solution: Schedule, start_day: date, out_path: str = "score_breakdown.txt") -> None:
    """Lightweight score breakdown + monthly cap check (post-solve)."""
    # Counts
    hard_counts = defaultdict(int)
    soft_counts = defaultdict(int)

    # Build quick maps
    emp_day_assign = defaultdict(list)   # (emp, day_id) -> [req ids]
    emp_assign_days = defaultdict(list)  # emp -> [day_id]
    task_pairs_continuity = 0

    for r in solution.reqs:
        if r.employee and r.day:
            emp_day_assign[(r.employee.name, r.day.id)].append(r.id)
            emp_assign_days[r.employee.name].append(r.day.id)

            # weekend check
            if r.day.d.weekday() not in _BUSINESS_DAYS:
                hard_counts["No weekend work"] += 1

            # personal unavailable
            if r.day.id in r.employee.unavailable_day_ids:
                hard_counts["Employee unavailable day"] += 1

            # org calendars
            m = r.module
            did = r.day.id
            comp = _MODULE_COMPANY.get(m, None)
            ctry = _MODULE_COUNTRY.get(m, None)
            if m in _MODULE_UNAVAIL and did in _MODULE_UNAVAIL[m]:
                hard_counts["Closed by module calendar"] += 1
            if comp and did in _COMPANY_UNAVAIL.get(comp, set()):
                hard_counts["Closed by company calendar"] += 1
            if ctry and did in _COUNTRY_UNAVAIL.get(ctry, set()):
                hard_counts["Closed by country calendar"] += 1

    # double booked same day
    for (emp, did), reqids in emp_day_assign.items():
        if len(reqids) > 1:
            hard_counts["Employee double-booked same day"] += (len(reqids) - 1)

    # balance workload (soft): each emp contributes C(n,2)
    total_pairs = 0
    for emp, days_list in emp_assign_days.items():
        n = len(days_list)
        total_pairs += n*(n-1)//2
    soft_counts["Balance workload across employees"] = total_pairs

    # continuity (soft): consecutive same task same worker — approximate by counting pairs
    # For brevity we skip reconstructing exact task equality here.

    # monthly cap check (report only)
    lines = []
    lines.append(f"SCORE BREAKDOWN (approx.)")
    lines.append(f"Hard counts:")
    for k,v in hard_counts.items():
        lines.append(f"  - {k}: {v}")
    lines.append(f"Soft counts:")
    for k,v in soft_counts.items():
        lines.append(f"  - {k}: {v}")

    if _MONTHLY_CAP_QUANTA is not None and _QUANTA_PER_DAY > 0:
        cap_days = _MONTHLY_CAP_QUANTA // _QUANTA_PER_DAY
        over_emp = []
        # Aggregate per (emp, yyyy-mm)
        per_emp_month = defaultdict(int)
        for emp, days_list in emp_assign_days.items():
            for did in days_list:
                y,m = _DAYID_YYYYMM.get(did,(0,0))
                per_emp_month[(emp,y,m)] += 1
        for (emp,y,m), cnt in per_emp_month.items():
            if cnt > cap_days:
                over_emp.append(f"  - {emp} {y}-{m:02d}: {cnt} days > cap {cap_days}")
        lines.append("")
        lines.append(f"Monthly cap report (global cap { _MONTHLY_CAP_QUANTA } quanta; ≈ {cap_days} days):")
        if over_emp:
            lines.extend(over_emp)
        else:
            lines.append("  (no overages)")

    with open(out_path, "w", encoding="utf-8") as f:
        f.write("\n".join(lines))

# -------------------- Public API --------------------

def _build_solver(best_limit: Optional[str] = None,
                  spent_seconds: Optional[int] = None,
                  unimproved_seconds: Optional[int] = None):
    term_kwargs = {}
    if best_limit is not None:
        term_kwargs["best_score_limit"] = best_limit           # e.g. "0hard/*soft"
    if spent_seconds is not None:
        term_kwargs["spent_limit"] = Duration(seconds=int(spent_seconds))
    if unimproved_seconds is not None:
        term_kwargs["unimproved_spent_limit"] = Duration(seconds=int(unimproved_seconds))

    solver_config = SolverConfig(
        solution_class=Schedule,
        entity_class_list=[RequirementSlot],
        score_director_factory_config=ScoreDirectorFactoryConfig(
            constraint_provider_function=define_constraints
        ),
        termination_config=TerminationConfig(**term_kwargs)
    )
    return SolverFactory.create(solver_config).build_solver()

def solve_from_config(cfg_path: str = "config_modules.yaml"):
    start_day, days, windows, employees = load_config_modules(cfg_path)
    reqs = build_requirement_slots(windows)
    problem = Schedule(days=days, employees=employees, reqs=reqs)

    # PASS 1: stop as soon as we get feasibility (hard == 0)
    t0 = time.time()
    solver1 = _build_solver(best_limit="0hard/*soft", spent_seconds=120)  # small safety cap
    feasible: Schedule = solver1.solve(problem)
    t1 = time.time()
    print(f"[Pass 1] feasible={feasible.score}  time={t1 - t0:.3f}s")

    # PASS 2: polish soft score for +5 minutes (stop early if no improvement for 60s)
    solver2 = _build_solver(spent_seconds=300, unimproved_seconds=60)
    t2 = time.time()
    final: Schedule = solver2.solve(feasible)
    t3 = time.time()
    print(f"[Pass 2] best={final.score}  time={t3 - t2:.3f}s  (total {t3 - t0:.3f}s)")

    def audit_staff_caps(solution, start_day):
        counts = defaultdict(int)
        for r in solution.reqs:
            if r.employee and r.day:
                key = (r.module, r.process_id, r.task_letter, r.day.id)
                counts[key] += 1
        bad = [(k, c) for k, c in counts.items()
               if c < _STAFF_MIN_PER_DAY or c > _STAFF_MAX_PER_DAY]
        bad.sort(key=lambda x: -x[1])
        print(f"[AUDIT] {len(bad)} cells violate min/max:")
        for (mod, p, t, did), c in bad[:20]:
            print(f"  {mod}-P{p}-{t} @ {(start_day + timedelta(days=did)).isoformat()} = {c} "
                  f"(min={_STAFF_MIN_PER_DAY}, max={_STAFF_MAX_PER_DAY})")

    audit_staff_caps(final, start_day)
    explain_and_write(final, start_day, out_path="score_breakdown.txt")
    return final, start_day

# -------------------- CLI (optional) --------------------

def main():
    solution, start_day = solve_from_config("config_modules.yaml")
    print(f"Best score: {solution.score}")
    print("Wrote score breakdown to score_breakdown.txt")
    # Minimal printout: count per (module, P, task, day)
    cell = defaultdict(list)
    for r in solution.reqs:
        if r.employee and r.day:
            cell[(r.module, r.process_id, r.task_letter, r.day.id)].append(r.employee.name)
    for k in sorted(cell.keys())[:15]:
        mod, p, t, did = k
        print(f"{mod}-P{p}-{t} @ {(start_day + timedelta(days=did)).isoformat()} -> {len(cell[k])} workers")

if __name__ == "__main__":
    main()
