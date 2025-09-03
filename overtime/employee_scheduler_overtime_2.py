# employee_scheduler_overtime_option_b.py
from __future__ import annotations
from dataclasses import dataclass, field
from datetime import date, datetime, timedelta
from typing import Annotated, Optional, Dict, List, Tuple, Set
import yaml
import time
from collections import defaultdict

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
    d: date

@dataclass(frozen=True)
class Employee:
    id: int
    name: str
    skills: Dict[str, int]
    capacity_hours_per_day: int = 8
    overtime_hours_per_day: int = 0
    min_block_hours: int = 8   # minimum hours if they work that day

@dataclass(frozen=True)
class TaskWindow:
    module: str
    process_id: int
    task_letter: str
    start_day_id: int
    end_day_id: int
    workload_hours: int

    def pcode(self) -> str: return f"P{self.process_id}-{self.task_letter}"
    def tcode(self) -> str: return f"{self.module}-P{self.process_id}-{self.task_letter}"

@planning_entity
@dataclass
class Assignment:
    id: Annotated[int, PlanningId]
    module: str
    process_id: int
    task_letter: str
    day: DaySlot

    employee: Annotated[Optional[Employee], PlanningVariable] = None
    # IMPORTANT: reference the value range provider "hours_range"
    hours: Annotated[int, PlanningVariable(value_range_provider_refs=["hours_range"])] = 0

    def pcode(self) -> str: return f"P{self.process_id}-{self.task_letter}"
    def tcode(self) -> str: return f"{self.module}-P{self.process_id}-{self.task_letter}"

@planning_solution
@dataclass
class Schedule:
    days:       Annotated[List[DaySlot],  ProblemFactCollectionProperty, ValueRangeProvider]
    employees:  Annotated[List[Employee], ProblemFactCollectionProperty, ValueRangeProvider]
    hours_range: Annotated[List[int], ProblemFactCollectionProperty, ValueRangeProvider(id="hours_range")]
    assigns:    Annotated[List[Assignment], PlanningEntityCollectionProperty]
    score:      Annotated[HardSoftScore,  PlanningScore] = field(default=None)

# -------------------- Globals --------------------
_DAYID_YYYYMM: Dict[int, Tuple[int, int]] = {}
_STAFF_MAX_PER_DAY: int = 8
_MAX_HOURS_PER_DAY: int = 14  # 8 base + up to 6 OT
# workload lookup: (module, process_id, task_letter) -> workload_hours
_TASK_WORKLOAD: Dict[Tuple[str, int, str], int] = {}

# -------------------- Constraints --------------------
@constraint_provider
def define_constraints(cf: ConstraintFactory) -> List[Constraint]:
    return [
        # HARD
        valid_if_hours(cf),
        employee_daily_cap(cf),
        min_block_if_touched_hard(cf),
        one_task_per_employee_day(cf),
        process_precedence(cf),
        task_underfill_hard(cf),

        # SOFT
        task_overfill_soft(cf),
        avoid_overtime_soft(cf),
        finish_asap_soft(cf),
        prefer_skill_soft(cf),
        continuity_soft(cf),
        minimize_empdays_soft(cf),
    ]

def valid_if_hours(cf: ConstraintFactory) -> Constraint:
    """If hours > 0 → must have employee and eligible skill (>=1)."""
    return (
        cf.for_each(Assignment)
        .filter(lambda a: a.hours > 0 and (a.employee is None or a.employee.skills.get(a.pcode(), 0) < 1))
        .penalize(HardSoftScore.ONE_HARD)
        .as_constraint("Assignment valid only if employee+skill")
    )

def employee_daily_cap(cf: ConstraintFactory) -> Constraint:
    """Per (employee, day): sum(hours) ≤ base + OT (HARD)."""
    return (
        cf.for_each(Assignment)
        .filter(lambda a: a.employee is not None and a.hours > 0)
        .group_by(lambda a: (a.employee, a.day.id),
                  ConstraintCollectors.sum(lambda a: a.hours))
        .filter(lambda key, total: total > key[0].capacity_hours_per_day + key[0].overtime_hours_per_day)
        .penalize(HardSoftScore.ONE_HARD,
                  lambda key, total: total - (key[0].capacity_hours_per_day + key[0].overtime_hours_per_day))
        .as_constraint("Employee daily cap (HARD)")
    )

def min_block_if_touched_hard(cf: ConstraintFactory) -> Constraint:
    """If an (employee, day) is used, enforce sum(hours) ≥ min_block_hours (default 8)."""
    def min_block(emp: Employee) -> int:
        return max(0, int(getattr(emp, "min_block_hours", 8)))
    return (
        cf.for_each(Assignment)
        .filter(lambda a: a.employee is not None and a.hours > 0)
        .group_by(lambda a: (a.employee, a.day.id),
                  ConstraintCollectors.sum(lambda a: a.hours))
        .filter(lambda key, total: 0 < total < min_block(key[0]))
        .penalize(HardSoftScore.ONE_HARD, lambda key, total: min_block(key[0]) - total)
        .as_constraint("Min 8h block if assigned (HARD)")
    )

def one_task_per_employee_day(cf: ConstraintFactory) -> Constraint:
    """
    HARD: For each (employee, day), they may work on at most ONE task code.
    """
    return (
        cf.for_each(Assignment)
        .filter(lambda a: a.employee is not None and a.hours > 0)
        .group_by(
            lambda a: (a.employee, a.day.id),
            ConstraintCollectors.count_distinct(lambda a: a.tcode())
        )
        .filter(lambda key, distinct_tasks: distinct_tasks > 1)
        .penalize(HardSoftScore.ONE_HARD, lambda key, distinct_tasks: distinct_tasks - 1)
        .as_constraint("One task per employee per day (HARD)")
    )


def process_precedence(cf: ConstraintFactory) -> Constraint:
    """
    HARD: For the same module, any work on process p+1 must be strictly AFTER all work on p.
    Penalize if there exists an assignment A on p+1 and B on p such that A.day.id <= B.day.id.
    """
    return (
        cf.for_each(Assignment)  # A: candidate on p+1
        .filter(lambda a: a.hours > 0)
        .if_exists(
            Assignment,  # B: candidate on p
            # same module
            Joiners.equal(lambda a: a.module, lambda b: b.module),
            # consecutive processes: A.process = B.process + 1
            Joiners.equal(lambda a: a.process_id, lambda b: b.process_id + 1),
            # out-of-order in time: A.day <= B.day  (must be strictly after)
            Joiners.less_than_or_equal(lambda a: a.day.id, lambda b: b.day.id),
            # ONLY JOINERS allowed here → use filtering joiner to require B has hours
            Joiners.filtering(lambda a, b: b.hours > 0),
        )
        .penalize(HardSoftScore.ONE_HARD)
        .as_constraint("Process precedence (p+1 strictly after p)")
    )

def task_underfill_hard(cf: ConstraintFactory) -> Constraint:
    """For each task, penalize missing hours (HARD)."""
    def missing(key, total) -> int:
        need = _TASK_WORKLOAD.get((key[0], key[1], key[2]), 0)
        return max(0, need - (total or 0))
    return (
        cf.for_each(Assignment)
        .group_by(lambda a: (a.module, a.process_id, a.task_letter),
                  ConstraintCollectors.sum(lambda a: a.hours))
        .filter(lambda key, total: missing(key, total) > 0)
        .penalize(HardSoftScore.ONE_HARD, lambda key, total: missing(key, total))
        .as_constraint("Task underfill (HARD)")
    )

def task_overfill_soft(cf: ConstraintFactory) -> Constraint:
    """Allow overfill but penalize extra hours (SOFT)."""
    def extra(key, total) -> int:
        need = _TASK_WORKLOAD.get((key[0], key[1], key[2]), 0)
        x = (total or 0) - need
        return x if x > 0 else 0
    return (
        cf.for_each(Assignment)
        .group_by(lambda a: (a.module, a.process_id, a.task_letter),
                  ConstraintCollectors.sum(lambda a: a.hours))
        .filter(lambda key, total: extra(key, total) > 0)
        .penalize(HardSoftScore.ONE_SOFT, lambda key, total: extra(key, total))
        .as_constraint("Task overfill (SOFT)")
    )

def avoid_overtime_soft(cf: ConstraintFactory) -> Constraint:
    """Penalty for hours above base per employee-day (SOFT)."""
    return (
        cf.for_each(Assignment)
        .filter(lambda a: a.employee is not None and a.hours > 0)
        .group_by(lambda a: (a.employee, a.day.id),
                  ConstraintCollectors.sum(lambda a: a.hours))
        .filter(lambda key, total: total > key[0].capacity_hours_per_day)
        .penalize(HardSoftScore.ONE_SOFT, lambda key, total: total - key[0].capacity_hours_per_day)
        .as_constraint("Avoid overtime (SOFT)")
    )

def finish_asap_soft(cf: ConstraintFactory) -> Constraint:
    """Penalize later days, scaled by hours (SOFT)."""
    return (
        cf.for_each(Assignment)
        .filter(lambda a: a.hours > 0 and a.day is not None)
        .penalize(HardSoftScore.ONE_SOFT, lambda a: a.day.id * a.hours)
        .as_constraint("Finish ASAP (SOFT)")
    )

def prefer_skill_soft(cf: ConstraintFactory) -> Constraint:
    """Small reward: higher skill × hours (SOFT)."""
    return (
        cf.for_each(Assignment)
        .filter(lambda a: a.employee is not None and a.hours > 0)
        .reward(HardSoftScore.ONE_SOFT, lambda a: a.employee.skills.get(a.pcode(), 1) * a.hours)
        .as_constraint("Prefer higher skill (SOFT)")
    )

def continuity_soft(cf: ConstraintFactory) -> Constraint:
    """Reward same employee continuing same task on consecutive days (SOFT)."""
    return (
        cf.for_each_unique_pair(
            Assignment,
            Joiners.equal(lambda a: a.employee),
            Joiners.equal(lambda a: a.tcode()),
            Joiners.less_than(lambda a: a.day.id)
        )
        .filter(lambda a, b:
            a.employee is not None and b.employee is not None and
            a.hours > 0 and b.hours > 0 and
            a.day is not None and b.day is not None and
            b.day.id == a.day.id + 1
        )
        .reward(HardSoftScore.ONE_SOFT)
        .as_constraint("Continuity (SOFT)")
    )

def minimize_empdays_soft(cf: ConstraintFactory) -> Constraint:
    """Small penalty per used (employee, day) to reduce heads (SOFT)."""
    return (
        cf.for_each(Assignment)
        .filter(lambda a: a.employee is not None and a.hours > 0)
        .group_by(lambda a: (a.employee, a.day.id), ConstraintCollectors.count())
        .penalize(HardSoftScore.ONE_SOFT, lambda key, cnt: 1)
        .as_constraint("Minimize employee-days used (SOFT)")
    )

# -------------------- YAML loading --------------------
def load_config_modules(path: str):
    with open(path, "r", encoding="utf-8") as f:
        cfg = yaml.safe_load(f)

    start_day = datetime.strptime(str(cfg["start_day"]), "%Y-%m-%d").date()
    horizon_days = int(cfg.get("horizon_days", 30))

    days: List[DaySlot] = []
    for i in range(horizon_days):
        d = start_day + timedelta(days=i)
        days.append(DaySlot(i, d))
        _DAYID_YYYYMM[i] = (d.year, d.month)

    # staffing caps -> how many assignment slots we pre-create per task-day
    global _STAFF_MAX_PER_DAY
    staff_cfg = cfg.get("staffing_per_task_per_day", {}) or {}
    _STAFF_MAX_PER_DAY = int(staff_cfg.get("max", 8))

    windows: List[TaskWindow] = []
    for m in cfg["modules"]:
        mcode = str(m["code"]).strip()
        m_start_idx = (datetime.strptime(str(m.get("start_date", cfg["start_day"])), "%Y-%m-%d").date() - start_day).days
        for proc in m["processes"]:
            pid = int(proc["id"])
            p_end_idx = (datetime.strptime(str(proc["end_date"]), "%Y-%m-%d").date() - start_day).days
            for t in proc["tasks"]:
                full_code = str(t["code"]).strip().upper()  # e.g., "S1-P2-A"
                parts = full_code.split("-")
                letter = parts[2]
                wh = int(t.get("workload_hours", 0))
                windows.append(TaskWindow(
                    module=mcode, process_id=pid, task_letter=letter,
                    start_day_id=m_start_idx, end_day_id=p_end_idx,
                    workload_hours=wh
                ))

    # employees
    employees: List[Employee] = []
    eid = 1
    for e in cfg["employees"]:
        name = str(e["name"])
        skills = {str(k).strip().upper(): int(v) for k, v in (e.get("skills", {}) or {}).items()}
        base = int(e.get("capacity_hours_per_day", 8))
        ot = int(e.get("overtime_hours_per_day", 0))
        minb = int(e.get("min_block_hours", 8))
        employees.append(Employee(
            id=eid, name=name, skills=skills,
            capacity_hours_per_day=base, overtime_hours_per_day=ot,
            min_block_hours=minb
        ))
        eid += 1

    return start_day, days, windows, employees

# -------------------- Assignment builder --------------------
def build_assignments(windows: List[TaskWindow], days: List[DaySlot]) -> List[Assignment]:
    """Pre-create up to _STAFF_MAX_PER_DAY assignment slots per (task, day) in its window."""
    assigns: List[Assignment] = []
    aid = 1
    for w in windows:
        for d in days:
            if w.start_day_id <= d.id <= w.end_day_id:
                for _ in range(_STAFF_MAX_PER_DAY):
                    assigns.append(Assignment(
                        id=aid, module=w.module, process_id=w.process_id, task_letter=w.task_letter, day=d
                    ))
                    aid += 1
    return assigns

# -------------------- Reporting --------------------
def explain(solution: Schedule, start_day: date):
    emp_day_hours = defaultdict(int)
    for a in solution.assigns:
        if a.employee and a.hours > 0:
            emp_day_hours[(a.employee.name, a.day.id)] += a.hours
    print("Top loads:")
    for (emp, did), h in sorted(emp_day_hours.items(), key=lambda kv: -kv[1])[:20]:
        print(f" {emp} @ {(start_day + timedelta(days=did)).isoformat()}: {h}h")

# -------------------- Solver --------------------
def _build_solver(best_limit: str | None = None, spent_minutes: int | None = None, unimproved_seconds: int | None = None):
    term_kwargs = {}
    if best_limit:        term_kwargs["best_score_limit"] = best_limit
    if spent_minutes:     term_kwargs["spent_limit"] = Duration(minutes=spent_minutes)
    if unimproved_seconds:term_kwargs["unimproved_spent_limit"] = Duration(seconds=unimproved_seconds)

    cfg = SolverConfig(
        solution_class=Schedule,
        entity_class_list=[Assignment],
        score_director_factory_config=ScoreDirectorFactoryConfig(
            constraint_provider_function=define_constraints
        ),
        termination_config=TerminationConfig(**term_kwargs)
    )
    return SolverFactory.create(cfg).build_solver()

def solve_from_config(cfg_path="config_modules.yaml"):
    global _TASK_WORKLOAD

    start_day, days, windows, employees = load_config_modules(cfg_path)

    # workload map for constraints
    _TASK_WORKLOAD = {(w.module, w.process_id, w.task_letter): int(w.workload_hours) for w in windows}

    assigns = build_assignments(windows, days)
    hours_range = list(range(_MAX_HOURS_PER_DAY + 1))  # 0..14 inclusive

    problem = Schedule(days=days, employees=employees, hours_range=hours_range, assigns=assigns)

    # PASS 1 — stop when feasible (0 hard)
    t0 = time.time()
    solver1 = _build_solver(best_limit="0hard/*soft", spent_minutes=2)
    feasible: Schedule = solver1.solve(problem)
    t1 = time.time()
    print(f"[Pass 1] feasible={feasible.score}  time={t1 - t0:.2f}s")

    # PASS 2 — polish soft
    solver2 = _build_solver(spent_minutes=5, unimproved_seconds=60)
    t2 = time.time()
    final: Schedule = solver2.solve(feasible)
    t3 = time.time()
    print(f"[Pass 2] best={final.score}  time={t3 - t2:.2f}s (total {t3 - t0:.2f}s)")

    explain(final, start_day)
    return final, start_day

def main():
    solve_from_config("config_modules.yaml")

if __name__ == "__main__":
    main()
