# employee_scheduler_overtime.py
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
    capacity_hours_per_day: int = 8
    overtime_hours_per_day: int = 0
    unavailable_day_ids: Set[int] = field(default_factory=set)  # optional personal time off


@dataclass(frozen=True)
class TaskWindow:
    """Task window within a module/process: allowed day range for that task."""
    module: str
    process_id: int
    task_letter: str
    start_day_id: int    # inclusive
    end_day_id: int      # inclusive
    workload_hours: int  # how many hour-slots must be scheduled

    def pcode(self) -> str: return f"P{self.process_id}-{self.task_letter}"
    def tcode(self) -> str: return f"{self.module}-P{self.process_id}-{self.task_letter}"


@planning_entity
@dataclass
class RequirementHour:
    """One person-hour of work for a specific (module, process, task)."""
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
    reqs:      Annotated[List[RequirementHour], PlanningEntityCollectionProperty]
    score:     Annotated[HardSoftScore, PlanningScore] = field(default=None)


# -------------------- Config knobs / globals --------------------

# Working week (you asked to remove weekend forbids; we only use this to align dates)
_BUSINESS_DAYS: Set[int] = {0, 1, 2, 3, 4}  # Mon..Fri by default (no hard constraint)

# Time quantum
_PER_DAY_HOURS: int = 8          # default daily capacity hint
_UNIT_HOURS: int = 8             # 1 workload unit = UNIT_HOURS

# Day id -> (year, month) for potential reporting
_DAYID_YYYYMM: Dict[int, Tuple[int, int]] = {}

# Optional staffing caps per (task, day) measured as distinct heads (employees) assigned that day
_STAFF_MIN_PER_DAY: Optional[int] = None
_STAFF_MAX_PER_DAY: Optional[int] = None


# -------------------- Constraints --------------------

@constraint_provider
def define_constraints(cf: ConstraintFactory) -> List[Constraint]:
    return [
        # HARD
        require_employee_assigned(cf),
        require_day_assigned(cf),
        day_within_window(cf),
        skill_must_exist(cf),
        employee_daily_capacity_hard(cf),   # <= base + overtime
        process_precedence_within_module(cf),

        # (Optional) HARD: distinct headcount per task/day min/max — uncomment if you want it enforced here
        # staffing_minmax_heads(cf),

        # SOFT
        finish_asap(cf),                    # encourage earlier scheduling
        avoid_overtime_soft(cf),            # prefer to stay within base hours
        workhorse_bias_soft(cf),            # slightly prefer higher-capacity employees
        prefer_continuity(cf),              # consecutive same task same worker
        # (Optional) softer balance — left out to keep runtime lean
    ]


# ---- HARD ----

def require_employee_assigned(cf: ConstraintFactory) -> Constraint:
    return (cf.for_each(RequirementHour)
            .filter(lambda r: r.employee is None)
            .penalize(HardSoftScore.ONE_HARD)
            .as_constraint("Employee must be assigned"))


def require_day_assigned(cf: ConstraintFactory) -> Constraint:
    return (cf.for_each(RequirementHour)
            .filter(lambda r: r.day is None)
            .penalize(HardSoftScore.ONE_HARD)
            .as_constraint("Day must be assigned"))


def day_within_window(cf: ConstraintFactory) -> Constraint:
    def outside(r: RequirementHour) -> bool:
        if r.day is None:
            return True
        return not (r.start_day_id <= r.day.id <= r.end_day_id)
    return (cf.for_each(RequirementHour)
            .filter(outside)
            .penalize(HardSoftScore.ONE_HARD)
            .as_constraint("Day outside task window (deadline or start)"))


def skill_must_exist(cf: ConstraintFactory) -> Constraint:
    def lacks(r: RequirementHour) -> bool:
        if r.employee is None: return True
        return r.employee.skills.get(r.pcode(), 0) < 1
    return (cf.for_each(RequirementHour)
            .filter(lacks)
            .penalize(HardSoftScore.ONE_HARD)
            .as_constraint("Missing skill for task"))


def employee_daily_capacity_hard(cf: ConstraintFactory) -> Constraint:
    """
    Per (employee, day): assigned hours must be <= (base + overtime).
    """
    def cap(emp: Employee) -> int:
        return max(0, int(emp.capacity_hours_per_day) + int(emp.overtime_hours_per_day))

    return (
        cf.for_each(RequirementHour)
        .filter(lambda r: r.employee is not None and r.day is not None)
        .group_by(lambda r: (r.employee, r.day.id), ConstraintCollectors.count())
        .filter(lambda key, cnt: cnt > cap(key[0]))
        .penalize(HardSoftScore.ONE_HARD, lambda key, cnt: cnt - cap(key[0]))
        .as_constraint("Employee daily hours exceed base+overtime (HARD)")
    )


def process_precedence_within_module(cf: ConstraintFactory) -> Constraint:
    """
    Enforce: for each module, any hour on process p+1 must be on a later day than any hour on p.
    Implementation: pair consecutive-process hours within the same module and penalize if day order is violated.
    """
    return (
        cf.for_each_unique_pair(
            RequirementHour,
            # same module
            Joiners.equal(lambda a: a.module, lambda b: b.module),
            # ensure a.process_id < b.process_id to make ordering deterministic
            Joiners.less_than(lambda a: a.process_id, lambda b: b.process_id)
        )
        .filter(lambda a, b:
            a.day is not None and b.day is not None
            and (a.process_id + 1 == b.process_id)     # consecutive processes only
            and not (b.day.id > a.day.id)              # violation if next process is not strictly later
        )
        .penalize(HardSoftScore.ONE_HARD)
        .as_constraint("Module process precedence (finish P before P+1)")
    )



def staffing_minmax_heads(cf: ConstraintFactory) -> Constraint:
    """
    OPTIONAL: enforce min/max distinct heads per (module,process,task,day).
    This counts DISTINCT EMPLOYEES (not hours). Enable by uncommenting in define_constraints().
    """
    if _STAFF_MIN_PER_DAY is None and _STAFF_MAX_PER_DAY is None:
        # Nothing to enforce
        return (
            cf.for_each(RequirementHour)
            .filter(lambda r: False)
            .penalize(HardSoftScore.ONE_HARD)
            .as_constraint("Disabled staffing min/max")
        )

    # Count distinct employees first
    heads_per_day = (
        cf.for_each(RequirementHour)
        .filter(lambda r: r.employee is not None and r.day is not None)
        .group_by(
            lambda r: (r.module, r.process_id, r.task_letter, r.day.id, r.employee),
            ConstraintCollectors.count()
        )
        .group_by(lambda key, _cnt: (key[0], key[1], key[2], key[3]),
                  ConstraintCollectors.count())  # number of distinct employees
    )

    def penalty(cnt: int) -> int:
        if _STAFF_MIN_PER_DAY is not None and cnt < _STAFF_MIN_PER_DAY:
            return _STAFF_MIN_PER_DAY - cnt
        if _STAFF_MAX_PER_DAY is not None and cnt > _STAFF_MAX_PER_DAY:
            return cnt - _STAFF_MAX_PER_DAY
        return 0

    return (
        heads_per_day
        .filter(lambda key, cnt: penalty(cnt) > 0)
        .penalize(HardSoftScore.ONE_HARD, lambda key, cnt: penalty(cnt))
        .as_constraint("Daily staffing min/max (distinct heads)")
    )


# ---- SOFT ----

def finish_asap(cf: ConstraintFactory) -> Constraint:
    def day_cost(r: RequirementHour) -> int:
        return r.day.id if r.day is not None else 1000
    return (
        cf.for_each(RequirementHour)
        .penalize(HardSoftScore.ONE_SOFT, lambda r: 3 * day_cost(r))
        .as_constraint("Finish ASAP")
    )


def avoid_overtime_soft(cf: ConstraintFactory) -> Constraint:
    """
    Per (employee, day), penalize hours above base (but not HARD if <= base+overtime).
    Incentivizes solver to prefer non-overtime, but allows it to meet deadlines.
    """
    def base_cap(emp: Employee) -> int:
        return max(0, int(emp.capacity_hours_per_day))

    def overtime_used(emp: Employee, cnt: int) -> int:
        extra = cnt - base_cap(emp)
        return extra if extra > 0 else 0

    return (
        cf.for_each(RequirementHour)
        .filter(lambda r: r.employee is not None and r.day is not None)
        .group_by(lambda r: (r.employee, r.day.id), ConstraintCollectors.count())
        .filter(lambda key, cnt: overtime_used(key[0], cnt) > 0)
        .penalize(HardSoftScore.ONE_SOFT, lambda key, cnt: overtime_used(key[0], cnt))
        .as_constraint("Avoid overtime (soft)")
    )


def workhorse_bias_soft(cf: ConstraintFactory) -> Constraint:
    """
    Light preference for assigning hours to higher-capacity employees.
    Implemented as a small reward per hour scaled by (capacity - base_day_hours).
    """
    return (
        cf.for_each(RequirementHour)
        .filter(lambda r: r.employee is not None and r.day is not None)
        .reward(HardSoftScore.ONE_SOFT,
                lambda r: max(0, int(r.employee.capacity_hours_per_day) - _PER_DAY_HOURS))
        .as_constraint("Prefer higher-capacity employees (workhorse bias)")
    )


def prefer_continuity(cf: ConstraintFactory) -> Constraint:
    """
    Reward same worker doing same task on consecutive days.
    """
    return (
        cf.for_each_unique_pair(
            RequirementHour,
            Joiners.equal(lambda r: r.employee),
            Joiners.equal(lambda r: r.module),
            Joiners.equal(lambda r: r.process_id),
            Joiners.equal(lambda r: r.task_letter),
            Joiners.less_than(lambda r: r.day.id if r.day is not None else -1)
        )
        .filter(lambda a, b:
            a.employee is not None and b.employee is not None and
            a.day is not None and b.day is not None and
            b.day.id == a.day.id + 1
        )
        .reward(HardSoftScore.ONE_SOFT)
        .as_constraint("Prefer continuity")
    )


# -------------------- YAML loading & problem build --------------------

@dataclass
class ModuleSpec:
    code: str
    start_day_id: int


def load_config_modules(path: str):
    """
    YAML (overtime version) — minimal fields used here:
      start_day: 2025-09-01
      horizon_days: 40
      business_days: ["Mon","Tue","Wed","Thu","Fri"]   # optional, no hard weekend ban
      quantum:
        per_day_hours: 8
        unit_hours: 8
      staffing_per_task_per_day: { min: 2, max: 8 }    # optional
      modules:
        - code: S1
          start_date: 2025-09-01
          processes:
            - id: 1
              end_date: 2025-09-10
              tasks:
                - code: S1-P1-A
                  workload_hours: 80           # or: workload_units: 10   (10 * unit_hours)
                ...
      employees:
        - name: AA
          skills: { P1-A: 3, P2-B: 2, ... }
          capacity_hours_per_day: 8
          overtime_hours_per_day: 2
          unavailable: ['2025-09-12']          # optional
    """
    with open(path, "r", encoding="utf-8") as f:
        cfg = yaml.safe_load(f)

    # Working week (used only to align start/end dates to business days if desired)
    global _BUSINESS_DAYS
    wd_map = {"Mon":0,"Tue":1,"Wed":2,"Thu":3,"Fri":4,"Sat":5,"Sun":6}
    _BUSINESS_DAYS = set(wd_map[d] for d in cfg.get("business_days", ["Mon","Tue","Wed","Thu","Fri"]))

    # Time quantum
    global _PER_DAY_HOURS, _UNIT_HOURS
    q = cfg.get("quantum", {}) or {}
    _PER_DAY_HOURS = int(q.get("per_day_hours", 8))
    _UNIT_HOURS = int(q.get("unit_hours", _PER_DAY_HOURS))

    # Optional staffing heads cap
    global _STAFF_MIN_PER_DAY, _STAFF_MAX_PER_DAY
    staff_cfg = cfg.get("staffing_per_task_per_day", {}) or {}
    _STAFF_MIN_PER_DAY = staff_cfg.get("min", None)
    _STAFF_MAX_PER_DAY = staff_cfg.get("max", None)
    if _STAFF_MIN_PER_DAY is not None:
        _STAFF_MIN_PER_DAY = int(_STAFF_MIN_PER_DAY)
    if _STAFF_MAX_PER_DAY is not None:
        _STAFF_MAX_PER_DAY = int(_STAFF_MAX_PER_DAY)

    # Horizon & calendar index
    start_day = datetime.strptime(str(cfg["start_day"]), "%Y-%m-%d").date()
    horizon_days = int(cfg.get("horizon_days", 30))
    days: List[DaySlot] = []
    for i in range(horizon_days):
        d = start_day + timedelta(days=i)
        days.append(DaySlot(i, d))
        _DAYID_YYYYMM[i] = (d.year, d.month)

    def align_to_business(d: date) -> date:
        # If you still want to snap to business days, do it here (no hard constraint otherwise)
        while d.weekday() not in _BUSINESS_DAYS:
            d += timedelta(days=1)
        return d

    def day_to_idx(datestr: str) -> int:
        d = datetime.strptime(str(datestr), "%Y-%m-%d").date()
        d = align_to_business(d)
        idx = (d - start_day).days
        return max(0, min(horizon_days - 1, idx))

    windows: List[TaskWindow] = []

    # ---- Modules ----
    modules_cfg = cfg["modules"]
    for m in modules_cfg:
        mcode = str(m["code"]).strip()
        module_start_idx = day_to_idx(m.get("start_date", cfg["start_day"]))
        for proc in m.get("processes", []):
            pid = int(proc["id"])
            p_end_idx = day_to_idx(proc["end_date"])
            p_start_idx = module_start_idx
            if "start_date" in proc and proc["start_date"]:
                p_start_idx = max(p_start_idx, day_to_idx(proc["start_date"]))
            for t in proc.get("tasks", []):
                full_code = str(t["code"]).strip().upper()  # e.g. "S1-P2-A"
                parts = full_code.split("-")
                if len(parts) != 3 or not parts[1].startswith("P"):
                    raise ValueError(f"Bad task code '{full_code}' (expected 'Sx-Py-Z').")
                letter = parts[2]

                wh = t.get("workload_hours", None)
                if wh is None:
                    units = int(t.get("workload_units", 0))
                    wh = int(units) * _UNIT_HOURS
                wh = int(wh)

                windows.append(TaskWindow(
                    module=mcode,
                    process_id=pid,
                    task_letter=letter,
                    start_day_id=p_start_idx,
                    end_day_id=p_end_idx,
                    workload_hours=wh
                ))

    # ---- Employees ----
    employees: List[Employee] = []
    eid = 1
    for e in cfg["employees"]:
        name = str(e["name"])
        skills = {str(k).strip().upper(): int(v) for k, v in (e.get("skills", {}) or {}).items()}
        base = int(e.get("capacity_hours_per_day", _PER_DAY_HOURS))
        ot = int(e.get("overtime_hours_per_day", 0))

        # optional personal blackout
        unavailable_ids: Set[int] = set()
        for ds in (e.get("unavailable", []) or []):
            try:
                unavailable_ids.add(day_to_idx(str(ds)))
            except Exception:
                pass

        employees.append(Employee(
            id=eid, name=name, skills=skills,
            capacity_hours_per_day=base, overtime_hours_per_day=ot,
            unavailable_day_ids=unavailable_ids
        ))
        eid += 1

    return start_day, days, windows, employees


def build_requirement_hours(windows: List[TaskWindow]) -> List[RequirementHour]:
    reqs: List[RequirementHour] = []
    rid = 1
    for w in windows:
        for _ in range(w.workload_hours):
            reqs.append(RequirementHour(
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
    """Lightweight score breakdown focused on hours + overtime."""
    # Aggregate per employee-day
    emp_day_hours = defaultdict(int)  # (emp, day_id) -> hours
    total_emp_hours = defaultdict(int)
    for r in solution.reqs:
        if r.employee and r.day:
            emp_day_hours[(r.employee.name, r.day.id)] += 1
            total_emp_hours[r.employee.name] += 1

    lines = []
    lines.append(f"SCORE: {solution.score}")
    lines.append("Top employee-day loads (sample):")
    for (emp, did), h in sorted(emp_day_hours.items(), key=lambda kv: -kv[1])[:20]:
        lines.append(f"  {emp} @ {(start_day + timedelta(days=did)).isoformat()}: {h}h")

    lines.append("")
    lines.append("Total hours per employee (sample):")
    for emp, h in list(sorted(total_emp_hours.items(), key=lambda kv: -kv[1]))[:20]:
        lines.append(f"  {emp}: {h}h")

    with open(out_path, "w", encoding="utf-8") as f:
        f.write("\n".join(lines))


# -------------------- Public API --------------------

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
        entity_class_list=[RequirementHour],
        score_director_factory_config=ScoreDirectorFactoryConfig(
            constraint_provider_function=define_constraints
        ),
        termination_config=TerminationConfig(**term_kwargs)
    )
    return SolverFactory.create(cfg).build_solver()

def solve_from_config(cfg_path: str = "config_modules.yaml"):
    # Build problem
    start_day, days, windows, employees = load_config_modules(cfg_path)
    reqs = build_requirement_hours(windows)
    problem = Schedule(days=days, employees=employees, reqs=reqs)

    # PASS 1 — stop as soon as we hit feasibility (hard == 0), with a small guard cap.
    t0 = time.time()
    solver1 = _build_solver(best_limit="0hard/*soft", spent_minutes=5)  # tweak minutes if you like
    feasible: Schedule = solver1.solve(problem)
    t1 = time.time()
    print(f"[Pass 1] feasible={feasible.score}  time={t1 - t0:.3f}s")

    # PASS 2 — keep feasibility, polish soft; stop earlier if unimproved for a while.
    solver2 = _build_solver(spent_minutes=5, unimproved_seconds=60)
    t2 = time.time()
    final: Schedule = solver2.solve(feasible)   # seed with pass-1 best
    t3 = time.time()
    print(f"[Pass 2] best={final.score}  time={t3 - t2:.3f}s  (total {t3 - t0:.3f}s)")

    explain_and_write(final, start_day, out_path="score_breakdown.txt")
    return final, start_day


# -------------------- CLI (optional) --------------------

def main():
    solution, start_day = solve_from_config("config_modules.yaml")
    print(f"Best score: {solution.score}")
    print("Wrote score breakdown to score_breakdown.txt")
    # Minimal visibility: people per (module, P, task, day) — counting distinct heads
    cell_heads: Dict[Tuple[str,int,str,int], Set[str]] = defaultdict(set)
    for r in solution.reqs:
        if r.employee and r.day:
            cell_heads[(r.module, r.process_id, r.task_letter, r.day.id)].add(r.employee.name)
    for k in sorted(cell_heads.keys())[:20]:
        mod, p, t, did = k
        print(f"{mod}-P{p}-{t} @ {(start_day + timedelta(days=did)).isoformat()} -> {len(cell_heads[k])} heads")

if __name__ == "__main__":
    main()