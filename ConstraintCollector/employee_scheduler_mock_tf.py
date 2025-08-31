from __future__ import annotations
from dataclasses import dataclass, field
from typing import List, Optional, Dict, Tuple, Set, Annotated
from datetime import date, timedelta, datetime
import sys
import yaml
import time
# ---- Timefold imports (Python) ----
from timefold.solver.domain import (
    planning_entity, planning_solution,
    PlanningId, PlanningVariable,
    PlanningEntityCollectionProperty, ProblemFactCollectionProperty,
    PlanningScore, ValueRangeProvider
)
from timefold.solver import SolverFactory
from timefold.solver.config import (
    SolverConfig, TerminationConfig, ScoreDirectorFactoryConfig, Duration
)
from timefold.solver.score import (
    HardSoftScore, ConstraintFactory, Constraint, Joiners, constraint_provider, ConstraintCollectors
)

# -------------------- Domain --------------------

@dataclass(frozen=True)
class DaySlot:
    id: int
    d: date  # calendar date
    weekday: int  # Mon=0..Sun=6

@dataclass(frozen=True)
class Employee:
    id: int
    name: str

@planning_entity
@dataclass
class RequirementSlot:
    """
    One person-day of work for a specific task (task_id; country is attached to task).
    Planning variables: day, employee.
    """
    id: Annotated[int, PlanningId]
    task_id: str
    country: str
    # Planning vars:
    day: Annotated[Optional[DaySlot], PlanningVariable] = field(default=None)
    employee: Annotated[Optional[Employee], PlanningVariable] = field(default=None)

@planning_solution
@dataclass
class Schedule:
    days:      Annotated[List[DaySlot], ProblemFactCollectionProperty, ValueRangeProvider]
    employees: Annotated[List[Employee], ProblemFactCollectionProperty, ValueRangeProvider]
    reqs:      Annotated[List[RequirementSlot], PlanningEntityCollectionProperty]
    score:     Annotated[HardSoftScore, PlanningScore] = field(default=None)

# -------------------- Config knobs (filled from YAML) --------------------
_WEEKEND_WORK: bool = False
_STAFF_MIN: int = 2
_STAFF_MAX: int = 5

# country → limits and gaps
_COUNTRY_VISA_STAY_LIMIT: Dict[str, int] = {}
_COUNTRY_VISA_STAY_GAP: Dict[str, int] = {}
_COUNTRY_ANNUAL_LIMIT: Dict[str, int] = {}
_COUNTRY_ANNUAL_BREAK: Dict[str, int] = {}
# changeover[from][to] = required gap days
_COUNTRY_CHANGEOVER: Dict[str, Dict[str, int]] = {}

# -------------------- Constraints --------------------

def _is_weekend(day: DaySlot) -> bool:
    return day.weekday >= 5  # 5=Sat, 6=Sun

def _segments_from_days(day_ids: List[int], break_days: int) -> List[Tuple[int, int]]:
    """
    From a sorted list of day ids, make contiguous segments where consecutive
    assignments less than 'break_days' apart belong to the same segment.
    Each segment is (first_day_id, last_day_id); span is inclusive in calendar days.
    """
    if not day_ids:
        return []
    s = sorted(set(day_ids))
    segs: List[Tuple[int,int]] = []
    cur_start = s[0]
    prev = s[0]
    for d in s[1:]:
        if (d - prev) >= max(1, break_days):
            segs.append((cur_start, prev))
            cur_start = d
        prev = d
    segs.append((cur_start, prev))
    return segs

@constraint_provider
def define_constraints(cf: ConstraintFactory) -> List[Constraint]:
    return [
        require_employee_assigned(cf),
        require_day_assigned(cf),
        no_weekend_work(cf),
        employee_not_double_booked_same_day(cf),

        staffing_minmax_combined(cf),
        country_changeover_gap(cf),
        visa_presence_limit(cf),
        annual_presence_limit(cf),

        finish_asap(cf),
        minimize_task_makespan(cf),
    ]

def require_employee_assigned(cf: ConstraintFactory) -> Constraint:
    return (
        cf.for_each(RequirementSlot)
        .filter(lambda r: r.employee is None)
        .penalize(HardSoftScore.ONE_HARD)
        .as_constraint("Employee must be assigned")
    )

def require_day_assigned(cf: ConstraintFactory) -> Constraint:
    return (
        cf.for_each(RequirementSlot)
        .filter(lambda r: r.day is None)
        .penalize(HardSoftScore.ONE_HARD)
        .as_constraint("Day must be assigned")
    )

def no_weekend_work(cf: ConstraintFactory) -> Constraint:
    # If weekend work is disabled, penalize any assignment on Sat/Sun.
    def is_violation(r: RequirementSlot) -> bool:
        return (not _WEEKEND_WORK) and (r.day is not None) and _is_weekend(r.day)
    return (
        cf.for_each(RequirementSlot)
        .filter(is_violation)
        .penalize(HardSoftScore.ONE_HARD)
        .as_constraint("No weekend work")
    )

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

def staffing_minmax_combined(cf: ConstraintFactory) -> Constraint:
    """
    HARD: For each (task_id, day), enforce MIN and MAX headcount using a single group_by.
    """
    def penalty(cnt: int) -> int:
        if cnt < _STAFF_MIN:
            return (_STAFF_MIN - cnt)
        if cnt > _STAFF_MAX:
            return (cnt - _STAFF_MAX)
        return 0

    return (
        cf.for_each(RequirementSlot)
        .filter(lambda r: r.day is not None)
        .group_by(lambda r: (r.task_id, r.day.id), ConstraintCollectors.count())
        .filter(lambda key, cnt: cnt < _STAFF_MIN or cnt > _STAFF_MAX)
        .penalize(HardSoftScore.ONE_HARD, lambda key, cnt: penalty(cnt))
        .as_constraint("Daily staffing mini or max per task")
    )

def country_changeover_gap(cf: ConstraintFactory) -> Constraint:
    """
    HARD: For each employee, switching from country C1 to C2 requires at least
    `_COUNTRY_CHANGEOVER[C1][C2]` empty days BETWEEN the two assignments.
    i.e., if required gap = g, then b.day.id >= a.day.id + g + 1
    """
    def violates(a: RequirementSlot, b: RequirementSlot) -> bool:
        if a.employee is None or b.employee is None or a.day is None or b.day is None:
            return False
        if a.country == b.country:
            return False
        g = _COUNTRY_CHANGEOVER.get(a.country, {}).get(b.country, 0)
        return not (b.day.id >= a.day.id + g + 1)

    return (
        cf.for_each_unique_pair(
            RequirementSlot,
            Joiners.equal(lambda r: r.employee),
            Joiners.less_than(lambda r: r.day.id if r.day is not None else -1)
        )
        .filter(violates)
        .penalize(HardSoftScore.ONE_HARD)
        .as_constraint("Country changeover gap")
    )

def visa_presence_limit(cf: ConstraintFactory) -> Constraint:
    def key_emp_country(r: RequirementSlot):
        if r.employee is None or r.day is None:
            return None
        return (r.employee.id, r.country)

    def overstay_penalty(day_ids, ctry: str) -> int:
        if day_ids is None:
            return 0
        if len(day_ids) == 0:
            return 0
        stay_limit = _COUNTRY_VISA_STAY_LIMIT.get(ctry, None)
        gap = _COUNTRY_VISA_STAY_GAP.get(ctry, 0)
        if stay_limit is None:
            return 0
        pen = 0
        for a, b in _segments_from_days(sorted(day_ids), gap):
            span = (b - a) + 1
            if span > stay_limit:
                pen += (span - stay_limit)
        return int(pen)

    return (
        cf.for_each(RequirementSlot)
        .filter(lambda r: r.employee is not None and r.day is not None)
        .group_by(key_emp_country, ConstraintCollectors.to_list(lambda r: r.day.id))
        .filter(lambda key, day_list: key is not None)
        .penalize(HardSoftScore.ONE_HARD, lambda key, day_list: overstay_penalty(day_list, key[1]))
        .as_constraint("Visa stay limit")
    )


def annual_presence_limit(cf: ConstraintFactory) -> Constraint:
    def key_emp_country(r: RequirementSlot):
        if r.employee is None or r.day is None:
            return None
        return (r.employee.id, r.country)

    def annual_overflow(day_ids, ctry: str) -> int:
        if day_ids is None:
            return 0
        if len(day_ids) == 0:
            return 0
        limit = _COUNTRY_ANNUAL_LIMIT.get(ctry, None)
        br = _COUNTRY_ANNUAL_BREAK.get(ctry, 0)
        if limit is None:
            return 0
        total = 0
        for a, b in _segments_from_days(sorted(day_ids), br):
            total += (b - a) + 1
        return int(max(0, total - limit))

    return (
        cf.for_each(RequirementSlot)
        .filter(lambda r: r.employee is not None and r.day is not None)
        .group_by(key_emp_country, ConstraintCollectors.to_list(lambda r: r.day.id))
        .filter(lambda key, day_list: key is not None)
        .penalize(HardSoftScore.ONE_HARD, lambda key, day_list: annual_overflow(day_list, key[1]))
        .as_constraint("Annual presence limit")
    )


# ---------- Soft ----------
def finish_asap(cf: ConstraintFactory) -> Constraint:
    """
    SOFT: earlier days are cheaper (push assignments to the start of the horizon).
    Using a stronger soft weight helps the solver prefer packing earlier days,
    up to the max-per-day cap.
    """
    SOFT_WEIGHT = 10  # tune: bigger = stronger push to earlier days
    return (
        cf.for_each(RequirementSlot)
        .filter(lambda r: r.day is not None)
        .penalize(HardSoftScore.of_soft(SOFT_WEIGHT), lambda r: r.day.id)
        .as_constraint("Finish ASAP (earlier days cheaper)")
    )


def minimize_task_makespan(cf: ConstraintFactory) -> Constraint:
    """
    SOFT: compact each task’s used days by penalizing (max_day - min_day).
    """
    MAKESPAN_WEIGHT = 50  # tune as needed

    def span(day_ids):
        # Avoid Python truthiness on Java lists
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
        .group_by(lambda r: r.task_id,
                  ConstraintCollectors.to_list(lambda r: r.day.id))
        .penalize(HardSoftScore.of_soft(MAKESPAN_WEIGHT),
                  lambda tid, dlist: span(dlist))
        .as_constraint("Minimize task makespan")
    )


# -------------------- YAML loading & problem build --------------------

def _make_days(start_date: str, horizon_days: int) -> List[DaySlot]:
    Y, M, D = map(int, str(start_date).split("-"))
    start = date(Y, M, D)
    days: List[DaySlot] = []
    for i in range(horizon_days):
        d = start + timedelta(days=i)
        days.append(DaySlot(id=i, d=d, weekday=d.weekday()))
    return days

def load_problem_from_yaml(path: str) -> Tuple[date, List[DaySlot], List[Employee], List[RequirementSlot]]:
    with open(path, "r", encoding="utf-8") as f:
        cfg = yaml.safe_load(f)

    # Planning horizon
    start_date = cfg["planning"]["start_date"]
    horizon = int(cfg["planning"]["days"])
    days = _make_days(start_date, horizon)

    # Weekend work flag
    global _WEEKEND_WORK
    _WEEKEND_WORK = bool(cfg["planning"].get("weekend_work", False))

    # Staffing limits
    global _STAFF_MIN, _STAFF_MAX
    _STAFF_MIN = int(cfg["staffing_limits"]["min_per_task_per_day"])
    _STAFF_MAX = int(cfg["staffing_limits"]["max_per_task_per_day"])

    # Country rules
    global _COUNTRY_VISA_STAY_LIMIT, _COUNTRY_VISA_STAY_GAP, _COUNTRY_ANNUAL_LIMIT, _COUNTRY_ANNUAL_BREAK
    _COUNTRY_VISA_STAY_LIMIT = {}
    _COUNTRY_VISA_STAY_GAP = {}
    _COUNTRY_ANNUAL_LIMIT = {}
    _COUNTRY_ANNUAL_BREAK = {}
    for ctry, obj in (cfg.get("countries", {}) or {}).items():
        _COUNTRY_VISA_STAY_LIMIT[ctry] = int(obj["visa_stay_limit"])
        _COUNTRY_VISA_STAY_GAP[ctry] = int(obj["stay_gap_days"])
        _COUNTRY_ANNUAL_LIMIT[ctry] = int(obj["annual_limit"])
        _COUNTRY_ANNUAL_BREAK[ctry] = int(obj["annual_break_days"])

    # Changeover rules
    global _COUNTRY_CHANGEOVER
    _COUNTRY_CHANGEOVER = {src: {dst: int(g) for dst, g in (inner or {}).items()}
                           for src, inner in (cfg.get("country_changeover_days", {}) or {}).items()}

    # Employees (flat names)
    employees: List[Employee] = []
    for i, name in enumerate(cfg["employees"], start=1):
        employees.append(Employee(id=i, name=str(name)))

    # Build RequirementSlots: one per person-day of each task workload
    reqs: List[RequirementSlot] = []
    rid = 1
    for m in cfg["modules"]:
        for p in m["processes"]:
            for t in p["tasks"]:
                task_id = str(t["id"])
                country = str(t["country"])
                workload = int(t["workload"])
                for _ in range(workload):
                    reqs.append(RequirementSlot(
                        id=rid, task_id=task_id, country=country
                    ))
                    rid += 1

    # Return also the real start date for printing
    Y, M, D = map(int, str(start_date).split("-"))
    return date(Y, M, D), days, employees, reqs

# -------------------- Solve API --------------------

def _build_solver(best_limit: str | None = None,
                  spent_minutes: int | None = None,
                  unimproved_seconds: int | None = None):
    term_kwargs = {}
    if best_limit is not None:
        term_kwargs["best_score_limit"] = best_limit            # e.g. "0hard/*soft"
    if spent_minutes is not None:
        term_kwargs["spent_limit"] = Duration(minutes=spent_minutes)
    if unimproved_seconds is not None:
        term_kwargs["unimproved_spent_limit"] = Duration(seconds=unimproved_seconds)

    cfg = SolverConfig(
        solution_class=Schedule,
        entity_class_list=[RequirementSlot],
        score_director_factory_config=ScoreDirectorFactoryConfig(
            constraint_provider_function=define_constraints
        ),
        termination_config=TerminationConfig(**term_kwargs)
    )
    return SolverFactory.create(cfg).build_solver()

def solve_from_config(cfg_path: str = "config_mock.yaml"):
    start_day, days, employees, reqs = load_problem_from_yaml(cfg_path)
    problem = Schedule(days=days, employees=employees, reqs=reqs)

    # PASS 1 — stop as soon as we hit feasibility (hard == 0).
    t0 = time.time()
    solver1 = _build_solver(best_limit="0hard/*soft", spent_minutes=2)  # small guard cap
    feasible: Schedule = solver1.solve(problem)
    t1 = time.time()
    print(f"[Pass 1] feasible={feasible.score}  time={t1 - t0:.3f}s")

    # PASS 2 — keep feasibility, polish soft for up to 5 minutes,
    # stop earlier if no improvement for 60s.
    solver2 = _build_solver(spent_minutes=5, unimproved_seconds=60)
    t2 = time.time()
    final: Schedule = solver2.solve(feasible)   # start from pass-1 best
    t3 = time.time()
    print(f"[Pass 2] best={final.score}  time={t3 - t2:.3f}s  (total {t3 - t0:.3f}s)")

    return final, start_day

# -------------------- CLI --------------------

if __name__ == "__main__":
    cfg = sys.argv[1] if len(sys.argv) > 1 else "config_mock.yaml"
    solve_from_config(cfg)
