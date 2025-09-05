# employee_scheduler_overtime_2.py
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
    skills: Dict[str, int]
    capacity_hours_per_day: int = 8
    overtime_hours_per_day: int = 0
    min_block_hours: int = 0  # 0 disables

@dataclass(frozen=True)
class TaskWindow:
    module: str
    process_id: int
    task_letter: str
    start_day_id: int
    end_day_id: int
    workload_units: int  # 8h target per unit

    def pcode(self) -> str: return f"P{self.process_id}-{self.task_letter}"
    def tcode(self) -> str: return f"{self.module}-P{self.process_id}-{self.task_letter}"

@planning_entity
@dataclass
class CoreUnit:
    """One 6-hour base block that must be scheduled within the task window."""
    id: Annotated[int, PlanningId]
    module: str
    process_id: int
    task_letter: str
    start_day_id: int
    end_day_id: int

    employee: Annotated[Optional[Employee], PlanningVariable] = None
    day:      Annotated[Optional[DaySlot],  PlanningVariable] = None

    def hours(self) -> int: return 6
    def pcode(self) -> str: return f"P{self.process_id}-{self.task_letter}"
    def tcode(self) -> str: return f"{self.module}-P{self.process_id}-{self.task_letter}"

@planning_entity
@dataclass
class AddonHour:
    """One 1-hour token that floats to balance cores to the 8h average."""
    id: Annotated[int, PlanningId]
    module: str
    process_id: int
    task_letter: str
    start_day_id: int
    end_day_id: int

    employee: Annotated[Optional[Employee], PlanningVariable] = None
    day:      Annotated[Optional[DaySlot],  PlanningVariable] = None

    def hours(self) -> int: return 1
    def pcode(self) -> str: return f"P{self.process_id}-{self.task_letter}"
    def tcode(self) -> str: return f"{self.module}-P{self.process_id}-{self.task_letter}"

@planning_solution
@dataclass
class Schedule:
    days:      Annotated[List[DaySlot],  ProblemFactCollectionProperty, ValueRangeProvider]
    employees: Annotated[List[Employee], ProblemFactCollectionProperty, ValueRangeProvider]

    cores:   Annotated[List[CoreUnit],   PlanningEntityCollectionProperty]
    addons:  Annotated[List[AddonHour],  PlanningEntityCollectionProperty]

    score:   Annotated[HardSoftScore,    PlanningScore] = field(default=None)

# ======================= Globals =======================

_STAFF_MIN_PER_DAY: Optional[int] = None
_STAFF_MAX_PER_DAY: Optional[int] = None
_DAYID_YYYYMM: Dict[int, Tuple[int, int]] = {}
_TARGET_HOURS_PER_EMP: float = 0.0

# ======================= Constraints =======================

@constraint_provider
def define_constraints(cf: ConstraintFactory) -> List[Constraint]:
    cons = [
        # HARD
        require_employee_day_and_skill_core(cf),
        require_employee_day_and_skill_add(cf),
        day_within_window_core(cf),
        day_within_window_add(cf),
        employee_daily_cap_hard(cf),
        one_task_per_employee_per_day_hard(cf),
        staffing_minmax_heads_hard(cf),

        # SOFT
        avoid_overtime_soft(cf),
        fill_to_capacity_soft(cf),
        finish_asap_soft(cf),
        continuity_soft(cf),

        # addons stick to existing core on same task/day/employee (reduce extra heads)
        stick_addons_to_core_soft(cf),

        # balance total hours across employees
        balance_total_hours_soft(cf),
    ]
    return cons

# --- helpers to iterate both types uniformly ---
def _all_units_stream(cf: ConstraintFactory):
    # join CoreUnit and AddonHour via "if exists then include" trick:
    return cf.for_each(CoreUnit).join(
        cf.for_each(AddonHour),  # dummy join to enable reuse? We'll just write 2 streams where needed.
    )

def require_employee_day_and_skill_core(cf: ConstraintFactory) -> Constraint:
    return (
        cf.for_each(CoreUnit)
        .filter(lambda u:
            u.employee is None or u.day is None or
            u.employee.skills.get(u.pcode(), 0) < 1
        )
        .penalize(HardSoftScore.ONE_HARD)
        .as_constraint("Core must have employee+day+eligible skill")
    )

def require_employee_day_and_skill_add(cf: ConstraintFactory) -> Constraint:
    return (
        cf.for_each(AddonHour)
        .filter(lambda u:
            u.employee is None or u.day is None or
            u.employee.skills.get(u.pcode(), 0) < 1
        )
        .penalize(HardSoftScore.ONE_HARD)
        .as_constraint("Addon must have employee+day+eligible skill")
    )

def day_within_window_core(cf: ConstraintFactory) -> Constraint:
    return (
        cf.for_each(CoreUnit)
        .filter(lambda u: u.day is None or not (u.start_day_id <= u.day.id <= u.end_day_id))
        .penalize(HardSoftScore.ONE_HARD)
        .as_constraint("Core day outside window")
    )

def day_within_window_add(cf: ConstraintFactory) -> Constraint:
    return (
        cf.for_each(AddonHour)
        .filter(lambda u: u.day is None or not (u.start_day_id <= u.day.id <= u.end_day_id))
        .penalize(HardSoftScore.ONE_HARD)
        .as_constraint("Addon day outside window")
    )

def employee_daily_cap_hard(cf: ConstraintFactory) -> Constraint:
    # Sum(core.hours + addons.hours) per (employee, day) <= base + overtime
    return (
        cf.for_each(CoreUnit)
        .filter(lambda u: u.employee is not None and u.day is not None)
        .group_by(lambda u: (u.employee, u.day.id), ConstraintCollectors.sum(lambda u: u.hours()))
        .join(
            cf.for_each(AddonHour)
            .filter(lambda a: a.employee is not None and a.day is not None),
            Joiners.equal(lambda k, total: k[0], lambda a: a.employee),
            Joiners.equal(lambda k, total: k[1], lambda a: a.day.id),
            ConstraintCollectors.sum(lambda a: a.hours())
        )
        .penalize(
            HardSoftScore.ONE_HARD,
            lambda key_total_core, addon_sum:
                max(0, (key_total_core[1] + addon_sum) - (key_total_core[0][0].capacity_hours_per_day + key_total_core[0][0].overtime_hours_per_day))
        )
        .as_constraint("Employee daily cap (HARD)")
    )

def one_task_per_employee_per_day_hard(cf: ConstraintFactory) -> Constraint:
    # Count distinct tasks per (employee, day) across BOTH cores and addons
    core_task = (
        cf.for_each(CoreUnit)
        .filter(lambda u: u.employee is not None and u.day is not None)
        .group_by(lambda u: (u.employee, u.day.id), ConstraintCollectors.to_list(lambda u: u.tcode()))
    )
    add_task = (
        cf.for_each(AddonHour)
        .filter(lambda u: u.employee is not None and u.day is not None)
        .group_by(lambda u: (u.employee, u.day.id), ConstraintCollectors.to_list(lambda u: u.tcode()))
    )
    return (
        core_task.join(add_task,
            Joiners.equal(lambda k1, list1: k1, lambda k2, list2: k2),
            ConstraintCollectors.to_list(lambda k2, list2: list2))
        .penalize(
            HardSoftScore.ONE_HARD,
            lambda k_core_list, add_list_list:
                max(0, len(set((k_core_list[1] or []) + sum(add_list_list, []))) - 1)
        )
        .as_constraint("One task per employee per day (HARD)")
    )

def staffing_minmax_heads_hard(cf: ConstraintFactory) -> Constraint:
    # Heads on a task-day = distinct employees who did ANY core/addon there
    core_heads = (
        cf.for_each(CoreUnit)
        .filter(lambda u: u.employee is not None and u.day is not None)
        .group_by(lambda u: (u.module, u.process_id, u.task_letter, u.day.id),
                  ConstraintCollectors.to_set(lambda u: u.employee))
    )
    addon_heads = (
        cf.for_each(AddonHour)
        .filter(lambda u: u.employee is not None and u.day is not None)
        .group_by(lambda u: (u.module, u.process_id, u.task_letter, u.day.id),
                  ConstraintCollectors.to_set(lambda u: u.employee))
    )
    return (
        core_heads.join(addon_heads,
                        Joiners.equal(lambda k1, s1: k1, lambda k2, s2: k2),
                        ConstraintCollectors.to_set(lambda k2, s2: next(iter(s2), None)))
        .filter(lambda key_set_core, set_add:
            (_STAFF_MIN_PER_DAY is not None and len(set(key_set_core[1] | set_add)) < _STAFF_MIN_PER_DAY) or
            (_STAFF_MAX_PER_DAY is not None and len(set(key_set_core[1] | set_add)) > _STAFF_MAX_PER_DAY)
        )
        .penalize(
            HardSoftScore.ONE_HARD,
            lambda key_set_core, set_add:
                ((_STAFF_MIN_PER_DAY - len(set(key_set_core[1] | set_add)))
                    if (_STAFF_MIN_PER_DAY is not None and len(set(key_set_core[1] | set_add)) < _STAFF_MIN_PER_DAY) else 0) +
                ((len(set(key_set_core[1] | set_add)) - _STAFF_MAX_PER_DAY)
                    if (_STAFF_MAX_PER_DAY is not None and len(set(key_set_core[1] | set_add)) > _STAFF_MAX_PER_DAY) else 0)
        )
        .as_constraint("Staffing min/max heads per task-day (HARD)")
    )

def avoid_overtime_soft(cf: ConstraintFactory) -> Constraint:
    # Softly discourage going over base capacity (not counting OT allowance)
    core_sum = (
        cf.for_each(CoreUnit)
        .filter(lambda u: u.employee is not None and u.day is not None)
        .group_by(lambda u: (u.employee, u.day.id), ConstraintCollectors.sum(lambda u: u.hours()))
    )
    return (
        core_sum.join(
            cf.for_each(AddonHour)
            .filter(lambda a: a.employee is not None and a.day is not None)
            .group_by(lambda a: (a.employee, a.day.id), ConstraintCollectors.sum(lambda a: a.hours())),
            Joiners.equal(lambda k1, v1: k1, lambda k2, v2: k2)
        )
        .filter(lambda k_core_total, k_add_total:
            (k_core_total[1] + k_add_total[1]) > k_core_total[0][0].capacity_hours_per_day
        )
        .penalize(
            HardSoftScore.ONE_SOFT,
            lambda k_core_total, k_add_total:
                (k_core_total[1] + k_add_total[1]) - k_core_total[0][0].capacity_hours_per_day
        )
        .as_constraint("Avoid overtime (SOFT)")
    )

def fill_to_capacity_soft(cf: ConstraintFactory) -> Constraint:
    core_sum = (
        cf.for_each(CoreUnit)
        .filter(lambda u: u.employee is not None and u.day is not None)
        .group_by(lambda u: (u.employee, u.day.id), ConstraintCollectors.sum(lambda u: u.hours()))
    )
    add_sum = (
        cf.for_each(AddonHour)
        .filter(lambda a: a.employee is not None and a.day is not None)
        .group_by(lambda a: (a.employee, a.day.id), ConstraintCollectors.sum(lambda a: a.hours()))
    )
    return (
        core_sum.join(add_sum, Joiners.equal(lambda k1, v1: k1, lambda k2, v2: k2))
        .penalize(
            HardSoftScore.ONE_SOFT,
            lambda k_core_total, k_add_total:
                abs((k_core_total[1] + k_add_total[1]) - k_core_total[0][0].capacity_hours_per_day)
        )
        .as_constraint("Fill toward base capacity per day (SOFT)")
    )

def finish_asap_soft(cf: ConstraintFactory) -> Constraint:
    # Prefer earlier days for both entities
    c1 = (
        cf.for_each(CoreUnit)
        .filter(lambda u: u.day is not None)
        .penalize(HardSoftScore.ONE_SOFT, lambda u: u.day.id * u.hours())
    )
    c2 = (
        cf.for_each(AddonHour)
        .filter(lambda u: u.day is not None)
        .penalize(HardSoftScore.ONE_SOFT, lambda u: u.day.id * u.hours())
    )
    # Trick to return a single constraint: join them and penalize additively
    return (
        c1.join(c2, Joiners.less_than(lambda _: 0, lambda __: 1))
        .penalize(HardSoftScore.ONE_SOFT, lambda *_: 0)
        .as_constraint("Finish ASAP (SOFT)")
    )

def continuity_soft(cf: ConstraintFactory) -> Constraint:
    return (
        cf.for_each_unique_pair(
            CoreUnit,
            Joiners.equal(lambda u: u.employee),
            Joiners.equal(lambda u: u.tcode()),
            Joiners.less_than(lambda u: u.day.id if u.day is not None else -1)
        )
        .filter(lambda a, b:
            a.employee is not None and b.employee is not None and
            a.day is not None and b.day is not None and
            b.day.id == a.day.id + 1
        )
        .reward(HardSoftScore.ONE_SOFT)
        .as_constraint("Continuity on cores (SOFT)")
    )

def stick_addons_to_core_soft(cf: ConstraintFactory) -> Constraint:
    # Reward when an addon is assigned to same (emp, day, task) where that emp already has a core.
    return (
        cf.for_each(AddonHour)
        .filter(lambda a: a.employee is not None and a.day is not None)
        .join(
            cf.for_each(CoreUnit),
            Joiners.equal(lambda a: a.employee,      lambda c: c.employee),
            Joiners.equal(lambda a: a.day.id,        lambda c: c.day.id if c.day is not None else -1),
            Joiners.equal(lambda a: a.tcode(),       lambda c: c.tcode())
        )
        .reward(HardSoftScore.ONE_SOFT)
        .as_constraint("Stick addons to an existing core (SOFT)")
    )

def balance_total_hours_soft(cf: ConstraintFactory) -> Constraint:
    core_sum = (
        cf.for_each(CoreUnit)
        .filter(lambda u: u.employee is not None)
        .group_by(lambda u: u.employee, ConstraintCollectors.sum(lambda u: u.hours()))
    )
    add_sum = (
        cf.for_each(AddonHour)
        .filter(lambda u: u.employee is not None)
        .group_by(lambda u: u.employee, ConstraintCollectors.sum(lambda u: u.hours()))
    )
    return (
        core_sum.join(add_sum, Joiners.equal(lambda e, h: e, lambda e2, h2: e2))
        .penalize(HardSoftScore.ONE_SOFT,
                  lambda emp_total_core, emp_total_add: int(abs((emp_total_core[1] + emp_total_add[1]) - _TARGET_HOURS_PER_EMP)))
        .as_constraint("Balance total hours across employees (SOFT)")
    )

# ======================= YAML & builders =======================

def load_config_modules(path: str):
    with open(path, "r", encoding="utf-8") as f:
        cfg = yaml.safe_load(f)

    global _STAFF_MIN_PER_DAY, _STAFF_MAX_PER_DAY
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
                units = int(t.get("workload_units", 0))
                windows.append(TaskWindow(
                    module=mcode, process_id=pid, task_letter=letter,
                    start_day_id=m_start_idx, end_day_id=p_end_idx,
                    workload_units=units
                ))

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

def build_entities(windows: List[TaskWindow]) -> Tuple[List[CoreUnit], List[AddonHour]]:
    cores: List[CoreUnit] = []
    addons: List[AddonHour] = []
    cid = 1
    aid = 1
    for w in windows:
        # 1 core (6h) per workload unit
        for _ in range(max(0, w.workload_units)):
            cores.append(CoreUnit(
                id=cid, module=w.module, process_id=w.process_id, task_letter=w.task_letter,
                start_day_id=w.start_day_id, end_day_id=w.end_day_id
            ))
            cid += 1
        # 2 addons (1h) per workload unit -> averages a unit to 8h
        for _ in range(max(0, 2 * w.workload_units)):
            addons.append(AddonHour(
                id=aid, module=w.module, process_id=w.process_id, task_letter=w.task_letter,
                start_day_id=w.start_day_id, end_day_id=w.end_day_id
            ))
            aid += 1
    return cores, addons

# ======================= Reporting =======================

def explain(solution: Schedule, start_day: date) -> None:
    emp_day_hours = defaultdict(int)
    for u in solution.cores:
        if u.employee and u.day:
            emp_day_hours[(u.employee.name, u.day.id)] += u.hours()
    for u in solution.addons:
        if u.employee and u.day:
            emp_day_hours[(u.employee.name, u.day.id)] += u.hours()
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
        entity_class_list=[CoreUnit, AddonHour],
        score_director_factory_config=ScoreDirectorFactoryConfig(
            constraint_provider_function=define_constraints
        ),
        termination_config=TerminationConfig(**term_kwargs)
    )
    return SolverFactory.create(cfg).build_solver()

def solve_from_config(cfg_path="config_modules.yaml"):
    global _TARGET_HOURS_PER_EMP

    start_day, days, windows, employees = load_config_modules(cfg_path)

    # total required hours = 8 * sum(workload_units)
    total_required_hours = sum(w.workload_units for w in windows) * 8
    _TARGET_HOURS_PER_EMP = total_required_hours / max(1, len(employees))

    cores, addons = build_entities(windows)

    problem = Schedule(
        days=days,
        employees=employees,
        cores=cores,
        addons=addons,
    )

    t0 = time.time()
    solver1 = _build_solver(best_limit="0hard/*soft", spent_minutes=5)
    feasible: Schedule = solver1.solve(problem)
    t1 = time.time()
    print(f"[Pass 1] feasible={feasible.score}  time={t1 - t0:.2f}s")

    solver2 = _build_solver(spent_minutes=5, unimproved_seconds=30)
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
