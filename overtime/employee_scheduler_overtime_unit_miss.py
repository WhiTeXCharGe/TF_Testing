# employee_scheduler_overtime_2.py
from __future__ import annotations
from dataclasses import dataclass, field
from datetime import date, datetime, timedelta
from typing import Annotated, Optional, Dict, List, Tuple
import time
import yaml
from collections import defaultdict

from timefold.solver.domain import (
    planning_entity, planning_solution,
    PlanningId, PlanningVariable,
    PlanningEntityCollectionProperty, ProblemFactCollectionProperty,
    PlanningScore, ValueRangeProvider
)
from timefold.solver import SolverFactory
from timefold.solver.config import SolverConfig, TerminationConfig, ScoreDirectorFactoryConfig, Duration
from timefold.solver.score import (
    HardSoftScore, ConstraintFactory, Constraint, Joiners,
    constraint_provider, ConstraintCollectors
)

# ======================= Domain =======================

@dataclass(frozen=True)
class DaySlot:
    id: int
    d: date

@dataclass(frozen=True)
class Employee:
    id: int
    name: str
    # key "P{1-4}-{A-D}" -> level 1..5 (>=1 eligible)
    skills: Dict[str, int]
    capacity_hours_per_day: int = 8
    overtime_hours_per_day: int = 0
    # Set >0 to enforce a minimum block if the employee works that day (disabled by default).
    min_block_hours: int = 0

@dataclass(frozen=True)
class TaskWindow:
    module: str
    process_id: int
    task_letter: str
    start_day_id: int   # inclusive
    end_day_id: int     # inclusive
    workload_units: int # how many units (each baseline = quantum.unit_hours)

    def pcode(self) -> str: return f"P{self.process_id}-{self.task_letter}"
    def tcode(self) -> str: return f"{self.module}-P{self.process_id}-{self.task_letter}"

@planning_entity
@dataclass
class RequirementUnit:
    """
    One workload UNIT; solver decides (employee, day, hours).
    'hours' is variable; total across a task must equal units*unit_hours.
    """
    id: Annotated[int, PlanningId]
    module: str
    process_id: int
    task_letter: str
    start_day_id: int
    end_day_id: int

    employee: Annotated[Optional[Employee], PlanningVariable] = None
    day:      Annotated[Optional[DaySlot],  PlanningVariable] = None
    hours:    Annotated[int, PlanningVariable(value_range_provider_refs=["hours_range"])] = 0

    def pcode(self) -> str: return f"P{self.process_id}-{self.task_letter}"
    def tcode(self) -> str: return f"{self.module}-P{self.process_id}-{self.task_letter}"

@planning_solution
@dataclass
class Schedule:
    # Problem facts + value ranges
    days:      Annotated[List[DaySlot],  ProblemFactCollectionProperty, ValueRangeProvider]
    employees: Annotated[List[Employee], ProblemFactCollectionProperty, ValueRangeProvider]

    # Planning entities
    units:     Annotated[List[RequirementUnit], PlanningEntityCollectionProperty]
    hours_lower: int = 6
    hours_upper: int = 12

    # Score
    score:     Annotated[HardSoftScore,  PlanningScore] = field(default=None)

    # ---- Value range provider for RequirementUnit.hours ----
    @ValueRangeProvider("hours_range")
    def hours_range(self) -> List[int]:
        return list(range(self.hours_lower, self.hours_upper + 1))

# ======================= Globals =======================

_UNIT_HOURS: int = 8  # from YAML quantum.unit_hours
_STAFF_MIN_PER_DAY: Optional[int] = None
_STAFF_MAX_PER_DAY: Optional[int] = None
# target workload IN HOURS per (module, process, task)
_TASK_WORKLOAD_HOURS: Dict[Tuple[str, int, str], int] = {}
_DAYID_YYYYMM: Dict[int, Tuple[int, int]] = {}
_TARGET_HOURS_PER_EMP: float = 0.0

# ======================= Constraints =======================

@constraint_provider
def define_constraints(cf: ConstraintFactory) -> List[Constraint]:
    return [
        # HARD
        require_employee_day_and_skill(cf),
        day_within_window(cf),
        employee_daily_cap_hard(cf),
        one_task_per_employee_per_day_hard(cf),
        staffing_minmax_heads_hard(cf),
        task_hours_exact_hard(cf),

        # SOFT
        avoid_overtime_soft(cf),
        finish_asap_soft(cf),
        continuity_soft(cf),
        fill_to_capacity_soft(cf),
        balance_total_hours_soft(cf),
    ]

def require_employee_day_and_skill(cf: ConstraintFactory) -> Constraint:
    return (
        cf.for_each(RequirementUnit)
        .filter(lambda u:
            u.hours > 0 and (
                u.employee is None or
                u.day is None or
                u.employee.skills.get(u.pcode(), 0) < 1
            )
        )
        .penalize(HardSoftScore.ONE_HARD)
        .as_constraint("Unit must have employee+day+eligible skill when hours>0")
    )

def day_within_window(cf: ConstraintFactory) -> Constraint:
    return (
        cf.for_each(RequirementUnit)
        .filter(lambda u: u.hours > 0 and (u.day is None or not (u.start_day_id <= u.day.id <= u.end_day_id)))
        .penalize(HardSoftScore.ONE_HARD)
        .as_constraint("Unit day outside window")
    )

def employee_daily_cap_hard(cf: ConstraintFactory) -> Constraint:
    return (
        cf.for_each(RequirementUnit)
        .filter(lambda u: u.employee is not None and u.day is not None and u.hours > 0)
        .group_by(lambda u: (u.employee, u.day.id), ConstraintCollectors.sum(lambda u: u.hours))
        .filter(lambda key, total: total > key[0].capacity_hours_per_day + key[0].overtime_hours_per_day)
        .penalize(HardSoftScore.ONE_HARD,
                  lambda key, total: total - (key[0].capacity_hours_per_day + key[0].overtime_hours_per_day))
        .as_constraint("Employee daily cap (HARD)")
    )

def one_task_per_employee_per_day_hard(cf: ConstraintFactory) -> Constraint:
    return (
        cf.for_each(RequirementUnit)
        .filter(lambda u: u.employee is not None and u.day is not None and u.hours > 0)
        .group_by(
            lambda u: (u.employee, u.day.id),
            ConstraintCollectors.count_distinct(lambda u: u.tcode())
        )
        .filter(lambda key, distinct_tasks: distinct_tasks > 1)
        .penalize(HardSoftScore.ONE_HARD, lambda key, distinct_tasks: distinct_tasks - 1)
        .as_constraint("One task per employee per day (HARD)")
    )

def staffing_minmax_heads_hard(cf: ConstraintFactory) -> Constraint:
    if _STAFF_MIN_PER_DAY is None and _STAFF_MAX_PER_DAY is None:
        return (cf.for_each(RequirementUnit)
                .filter(lambda _: False)
                .penalize(HardSoftScore.ONE_HARD)
                .as_constraint("Staffing min/max (disabled)"))
    return (
        cf.for_each(RequirementUnit)
        .filter(lambda u: u.employee is not None and u.day is not None and u.hours > 0)
        .group_by(
            lambda u: (u.module, u.process_id, u.task_letter, u.day.id),
            ConstraintCollectors.count_distinct(lambda u: u.employee)
        )
        .filter(lambda key, heads:
            (_STAFF_MIN_PER_DAY is not None and heads < _STAFF_MIN_PER_DAY) or
            (_STAFF_MAX_PER_DAY is not None and heads > _STAFF_MAX_PER_DAY)
        )
        .penalize(
            HardSoftScore.ONE_HARD,
            lambda key, heads:
                ((_STAFF_MIN_PER_DAY - heads) if (_STAFF_MIN_PER_DAY is not None and heads < _STAFF_MIN_PER_DAY) else 0) +
                ((heads - _STAFF_MAX_PER_DAY) if (_STAFF_MAX_PER_DAY is not None and heads > _STAFF_MAX_PER_DAY) else 0)
        )
        .as_constraint("Staffing min/max heads per task-day (HARD)")
    )

def task_hours_exact_hard(cf: ConstraintFactory) -> Constraint:
    """Sum(hours) == workload_units * unit_hours for each task code."""
    def diff(task_key, total) -> int:
        need = _TASK_WORKLOAD_HOURS.get(task_key, 0)
        return abs(need - (total or 0))
    return (
        cf.for_each(RequirementUnit)
        .group_by(lambda u: (u.module, u.process_id, u.task_letter),
                  ConstraintCollectors.sum(lambda u: u.hours))
        .filter(lambda key, total: diff(key, total) > 0)
        .penalize(HardSoftScore.ONE_HARD, lambda key, total: diff(key, total))
        .as_constraint("Task hours equality (HARD)")
    )

def avoid_overtime_soft(cf: ConstraintFactory) -> Constraint:
    return (
        cf.for_each(RequirementUnit)
        .filter(lambda u: u.employee is not None and u.day is not None and u.hours > 0)
        .group_by(lambda u: (u.employee, u.day.id), ConstraintCollectors.sum(lambda u: u.hours))
        .filter(lambda key, total: total > key[0].capacity_hours_per_day)
        .penalize(HardSoftScore.ONE_SOFT, lambda key, total: total - key[0].capacity_hours_per_day)
        .as_constraint("Avoid overtime (SOFT)")
    )

def finish_asap_soft(cf: ConstraintFactory) -> Constraint:
    return (
        cf.for_each(RequirementUnit)
        .filter(lambda u: u.day is not None and u.hours > 0)
        .penalize(HardSoftScore.ONE_SOFT, lambda u: u.day.id * u.hours)
        .as_constraint("Finish ASAP (SOFT)")
    )

def continuity_soft(cf: ConstraintFactory) -> Constraint:
    return (
        cf.for_each_unique_pair(
            RequirementUnit,
            Joiners.equal(lambda u: u.employee),
            Joiners.equal(lambda u: u.tcode()),
            Joiners.less_than(lambda u: u.day.id if u.day is not None else -1)
        )
        .filter(lambda a, b:
            a.employee is not None and b.employee is not None and
            a.day is not None and b.day is not None and
            a.hours > 0 and b.hours > 0 and
            b.day.id == a.day.id + 1
        )
        .reward(HardSoftScore.ONE_SOFT)
        .as_constraint("Continuity (SOFT)")
    )

def fill_to_capacity_soft(cf: ConstraintFactory) -> Constraint:
    """Pull each emp-day toward base capacity (8 unless employee has different base)."""
    return (
        cf.for_each(RequirementUnit)
        .filter(lambda u: u.employee is not None and u.day is not None and u.hours > 0)
        .group_by(lambda u: (u.employee, u.day.id), ConstraintCollectors.sum(lambda u: u.hours))
        .penalize(HardSoftScore.ONE_SOFT, lambda key, total: abs(total - key[0].capacity_hours_per_day))
        .as_constraint("Fill toward base capacity per day (SOFT)")
    )

def balance_total_hours_soft(cf: ConstraintFactory) -> Constraint:
    return (
        cf.for_each(RequirementUnit)
        .filter(lambda u: u.employee is not None and u.hours > 0)
        .group_by(lambda u: u.employee, ConstraintCollectors.sum(lambda u: u.hours))
        .penalize(HardSoftScore.ONE_SOFT,
                  lambda emp, total: int(abs(total - _TARGET_HOURS_PER_EMP)))
        .as_constraint("Balance total hours across employees (SOFT)")
    )

# ======================= YAML & builders =======================

def load_config_modules(path: str):
    with open(path, "r", encoding="utf-8") as f:
        cfg = yaml.safe_load(f)

    # globals
    global _UNIT_HOURS, _STAFF_MIN_PER_DAY, _STAFF_MAX_PER_DAY
    q = cfg.get("quantum", {}) or {}
    _UNIT_HOURS = int(q.get("unit_hours", 8))

    staff_cfg = cfg.get("staffing_per_task_per_day", {}) or {}
    _STAFF_MIN_PER_DAY = staff_cfg.get("min", None)
    _STAFF_MAX_PER_DAY = staff_cfg.get("max", None)
    if _STAFF_MIN_PER_DAY is not None: _STAFF_MIN_PER_DAY = int(_STAFF_MIN_PER_DAY)
    if _STAFF_MAX_PER_DAY is not None: _STAFF_MAX_PER_DAY = int(_STAFF_MAX_PER_DAY)

    start_day = datetime.strptime(str(cfg["start_day"]), "%Y-%m-%d").date()
    horizon_days = int(cfg.get("horizon_days", 30))
    days: List[DaySlot] = []
    for i in range(horizon_days):
        d = start_day + timedelta(days=i)
        days.append(DaySlot(i, d))
        _DAYID_YYYYMM[i] = (d.year, d.month)

    # task windows (use workload_units from YAML)
    windows: List[TaskWindow] = []
    for m in cfg["modules"]:
        mcode = str(m["code"]).strip()
        m_start_idx = (datetime.strptime(str(m.get("start_date", cfg["start_day"])), "%Y-%m-%d").date() - start_day).days
        for proc in m["processes"]:
            pid = int(proc["id"])
            p_end_idx = (datetime.strptime(str(proc["end_date"]), "%Y-%m-%d").date() - start_day).days
            for t in proc["tasks"]:
                full_code = str(t["code"]).strip().upper()
                letter = full_code.split("-")[2]
                units = int(t.get("workload_units"))
                windows.append(TaskWindow(
                    module=mcode, process_id=pid, task_letter=letter,
                    start_day_id=m_start_idx, end_day_id=p_end_idx,
                    workload_units=units
                ))

    # employees
    employees: List[Employee] = []
    eid = 1
    for e in cfg["employees"]:
        name = str(e["name"])
        skills = {str(k).strip().upper(): int(v) for k, v in (e.get("skills", {}) or {}).items()}
        base = int(e.get("capacity_hours_per_day", 8))
        ot = int(e.get("overtime_hours_per_day", 0))
        minb = int(e.get("min_block_hours", 0))
        employees.append(Employee(
            id=eid, name=name, skills=skills,
            capacity_hours_per_day=base, overtime_hours_per_day=ot,
            min_block_hours=minb
        ))
        eid += 1

    return start_day, days, windows, employees

def build_units(windows: List[TaskWindow]) -> List[RequirementUnit]:
    """
    Create EXACTLY workload_units entities per task window.
    Each unit contributes a variable 'hours' chosen from hours_range.
    """
    units: List[RequirementUnit] = []
    uid = 1
    for w in windows:
        for _ in range(max(0, w.workload_units)):
            units.append(RequirementUnit(
                id=uid,
                module=w.module,
                process_id=w.process_id,
                task_letter=w.task_letter,
                start_day_id=w.start_day_id,
                end_day_id=w.end_day_id
            ))
            uid += 1
    return units

# ======================= Reporting =======================

def explain(solution: Schedule, start_day: date) -> None:
    emp_day_hours = defaultdict(int)
    for u in solution.units:
        if u.employee and u.day and u.hours > 0:
            emp_day_hours[(u.employee.name, u.day.id)] += u.hours
    print("Top employee-day loads:")
    for (emp, did), h in sorted(emp_day_hours.items(), key=lambda kv: -kv[1])[:20]:
        print(f"  {emp} @ {(start_day + timedelta(days=did)).isoformat()}: {h}h")

# ======================= Solver =======================

def _build_solver(best_limit: str | None = None,
                  spent_minutes: int | None = None,
                  unimproved_seconds: int | None = None):
    term_kwargs = {}
    if best_limit is not None:
        term_kwargs["best_score_limit"] = best_limit
    if spent_minutes is not None:
        term_kwargs["spent_limit"] = Duration(minutes=spent_minutes)
    if unimproved_seconds is not None:
        term_kwargs["unimproved_spent_limit"] = Duration(seconds=unimproved_seconds)

    cfg = SolverConfig(
        solution_class=Schedule,
        entity_class_list=[RequirementUnit],
        score_director_factory_config=ScoreDirectorFactoryConfig(
            constraint_provider_function=define_constraints
        ),
        termination_config=TerminationConfig(**term_kwargs)
    )
    return SolverFactory.create(cfg).build_solver()

def solve_from_config(cfg_path="config_modules.yaml"):
    global _TASK_WORKLOAD_HOURS, _TARGET_HOURS_PER_EMP

    start_day, days, windows, employees = load_config_modules(cfg_path)

    # needed hours per task = units * _UNIT_HOURS (from YAML)
    _TASK_WORKLOAD_HOURS = {
        (w.module, w.process_id, w.task_letter): int(w.workload_units * _UNIT_HOURS)
        for w in windows
    }

    total_required_hours = sum(_TASK_WORKLOAD_HOURS.values())
    _TARGET_HOURS_PER_EMP = total_required_hours / max(1, len(employees))

    # entities: EXACTLY workload_units per task
    units = build_units(windows)

    # Build problem (allowed hours range lives on the solution method)
    problem = Schedule(
        days=days,
        employees=employees,
        units=units,
        hours_lower=6,
        hours_upper=14 
    )

    # Two-pass solve
    t0 = time.time()
    solver1 = _build_solver(best_limit="0hard/*soft", spent_minutes=5)
    feasible: Schedule = solver1.solve(problem)
    t1 = time.time()
    print(f"[Pass 1] feasible={feasible.score}  time={t1 - t0:.2f}s")

    solver2 = _build_solver(spent_minutes=5, unimproved_seconds=60)
    t2 = time.time()
    final: Schedule = solver2.solve(feasible)
    t3 = time.time()
    print(f"[Pass 2] best={final.score}  time={t3 - t2:.2f}s  (total {t3 - t0:.2f}s)")

    explain(final, start_day)
    return final, start_day

def main():
    solve_from_config("config_modules.yaml")

if __name__ == "__main__":
    main()
