from dataclasses import dataclass, field
from datetime import date, datetime, timedelta
from typing import Annotated, Dict, List, Tuple
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

@dataclass(frozen=True)
class TaskWindow:
    module: str
    factory: str
    process_id: int
    task_letter: str
    start_day_id: int
    end_day_id: int
    unit_hours: int           # 4 or 8
    workload_units: int       # number of tokens

    def pcode(self) -> str: return f"P{self.process_id}-{self.task_letter}"
    def tcode(self) -> str: return f"{self.module}-P{self.process_id}-{self.task_letter}"

@planning_entity
@dataclass
class UnitToken:
    """
    1 token = 1 workload unit.
    Solver assigns (employee, day, hours).
    Hours domain is globally provided, but enforced per-task by a HARD constraint.
    """
    id: Annotated[int, PlanningId]
    module: str
    factory: str
    process_id: int
    task_letter: str
    start_day_id: int
    end_day_id: int
    unit_hours: int  # copy from TaskWindow for constraints

    employee: Annotated[
        Employee,
        PlanningVariable(value_range_provider_refs=["employeeRange"])
    ]
    day: Annotated[
        DaySlot,
        PlanningVariable(value_range_provider_refs=["dayRange"])
    ]
    hours: Annotated[
        int,
        PlanningVariable(value_range_provider_refs=["hoursRange"])
    ] = 4  # initialize to a legal value for both cases

    def pcode(self) -> str: return f"P{self.process_id}-{self.task_letter}"
    def tcode(self) -> str: return f"{self.module}-P{self.process_id}-{self.task_letter}"

@planning_solution
@dataclass
class Schedule:
    # Problem facts + value ranges
    days: Annotated[
        List[DaySlot],
        ProblemFactCollectionProperty,
        ValueRangeProvider(id="dayRange")
    ]
    employees: Annotated[
        List[Employee],
        ProblemFactCollectionProperty,
        ValueRangeProvider(id="employeeRange")
    ]
    hours_options: Annotated[
        List[int],
        ProblemFactCollectionProperty,
        ValueRangeProvider(id="hoursRange")
    ]
    tokens: Annotated[List[UnitToken], PlanningEntityCollectionProperty]
    score: Annotated[HardSoftScore, PlanningScore] = field(default=None)

# -------------------- Globals --------------------

_DAILY_CAP: int = 12  # global capacity per employee-day
_TARGET_HOURS_PER_EMP: float = 0.0
_TASK_REQUIRED_HOURS: Dict[Tuple[str, int, str], int] = {}  # (module, process, letter) -> required total hours
_ALLOWED_HOURS_8U: List[int] = [4, 8, 10, 12]               # from YAML quantum.token_hours_options

# staffing min/max heads from YAML
_STAFF_MIN_PER_DAY: int | None = None
_STAFF_MAX_PER_DAY: int | None = None

# -------------------- Constraints --------------------

@constraint_provider
def define_constraints(cf: ConstraintFactory) -> List[Constraint]:
    return [
        require_assignment_and_skill(cf),
        enforce_hours_domain_per_task(cf),
        process_precedence_within_module(cf),
        within_window(cf),
        daily_capacity_hard(cf),
        single_factory_per_emp_day_hard(cf),
        staffing_minmax_heads_hard(cf),
        task_hours_equality_hard(cf),

        avoid_overtime_soft(cf),
        finish_earlier_soft(cf),
        balance_total_hours_soft(cf),
    ]

def _is_unassigned(emp: Employee) -> bool:
    return emp.id == 0  # dummy

def require_assignment_and_skill(cf: ConstraintFactory) -> Constraint:
    return (
        cf.for_each(UnitToken)
        .filter(lambda u:
            _is_unassigned(u.employee) or
            u.employee.skills.get(u.pcode(), 0) < 1 or
            (u.hours is None) or (u.hours < 1)
        )
        .penalize(HardSoftScore.ONE_HARD)
        .as_constraint("assigned+eligible-skill+min-hours")
    )

def enforce_hours_domain_per_task(cf: ConstraintFactory) -> Constraint:
    """
    unit_hours == 4 -> token.hours must be 4
    unit_hours == 8 -> token.hours must be in _ALLOWED_HOURS_8U
    """
    return (
        cf.for_each(UnitToken)
        .filter(lambda u:
            (u.unit_hours == 4 and u.hours != 4) or
            (u.unit_hours == 8 and u.hours not in _ALLOWED_HOURS_8U)
        )
        .penalize(HardSoftScore.ONE_HARD)
        .as_constraint("hours-domain-per-task")
    )

def process_precedence_within_module(cf: ConstraintFactory) -> Constraint:
    """
    For each module, any hour on process p+1 must be on a later day than any hour on p.
    No Joiners.less_than; normalize (a,b) so pa <= pb and only check consecutive.
    """
    def violates(a: UnitToken, b: UnitToken) -> bool:
        if a.day is None or b.day is None:
            return False
        if a.module != b.module:
            return False
        # normalize by process id
        if a.process_id > b.process_id:
            a, b = b, a
        if b.process_id != a.process_id + 1:
            return False
        return not (b.day.id > a.day.id)  # violation if not strictly later
    return (
        cf.for_each_unique_pair(UnitToken, Joiners.equal(lambda x: x.module, lambda y: y.module))
        .filter(violates)
        .penalize(HardSoftScore.ONE_HARD)
        .as_constraint("process-precedence-per-module")
    )

def within_window(cf: ConstraintFactory) -> Constraint:
    return (
        cf.for_each(UnitToken)
        .filter(lambda u:
            (u.day is None) or
            (u.day.id < 0) or
            (u.day.id < u.start_day_id) or
            (u.day.id > u.end_day_id)
        )
        .penalize(HardSoftScore.ONE_HARD)
        .as_constraint("within-window")
    )

def daily_capacity_hard(cf: ConstraintFactory) -> Constraint:
    # Sum of token.hours per (employee, day) ≤ global daily cap
    return (
        cf.for_each(UnitToken)
        .filter(lambda u: not _is_unassigned(u.employee) and u.day.id >= 0)
        .group_by(lambda u: (u.employee, u.day.id), ConstraintCollectors.sum(lambda u: u.hours))
        .filter(lambda key, total_h: total_h > _DAILY_CAP)
        .penalize(HardSoftScore.ONE_HARD, lambda key, total_h: total_h - _DAILY_CAP)
        .as_constraint("daily-cap-12h")
    )

def single_factory_per_emp_day_hard(cf: ConstraintFactory) -> Constraint:
    """
    An employee may work multiple tasks in a day only if they are all in the same factory.
    If factories per (emp, day) > 1, penalize the excess.
    """
    return (
        cf.for_each(UnitToken)
        .filter(lambda u: not _is_unassigned(u.employee) and u.day.id >= 0)
        .group_by(
            lambda u: (u.employee, u.day.id),
            ConstraintCollectors.count_distinct(lambda u: u.factory)
        )
        .filter(lambda key, distinct_factories: distinct_factories > 1)
        .penalize(HardSoftScore.ONE_HARD, lambda key, distinct_factories: distinct_factories - 1)
        .as_constraint("one-factory-per-emp-day")
    )

def staffing_minmax_heads_hard(cf: ConstraintFactory) -> Constraint:
    """
    Heads per (task, day) = distinct employees with hours>0.
    Enforce YAML staffing_per_task_per_day min/max.
    """
    if _STAFF_MIN_PER_DAY is None and _STAFF_MAX_PER_DAY is None:
        # disabled
        return (
            cf.for_each(UnitToken).filter(lambda _: False)
            .penalize(HardSoftScore.ONE_HARD)
            .as_constraint("staffing-minmax-disabled")
        )
    return (
        cf.for_each(UnitToken)
        .filter(lambda u: u.day.id >= 0 and u.employee.id != 0 and (u.hours or 0) > 0)
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
        .as_constraint("staffing-minmax-heads-per-task-day")
    )

def task_hours_equality_hard(cf: ConstraintFactory) -> Constraint:
    """
    Sum(hours of tokens) == workload_units * unit_hours, per task.
    """
    def required_hours(task_key) -> int:
        return _TASK_REQUIRED_HOURS.get(task_key, 0)

    return (
        cf.for_each(UnitToken)
        .group_by(lambda u: (u.module, u.process_id, u.task_letter),
                  ConstraintCollectors.sum(lambda u: u.hours))
        .filter(lambda key, total_h: total_h != required_hours(key))
        .penalize(HardSoftScore.ONE_HARD, lambda key, total_h: abs(total_h - required_hours(key)))
        .as_constraint("task-total-hours-equality")
    )

def avoid_overtime_soft(cf: ConstraintFactory) -> Constraint:
    # Softly penalize hours over 8 (base) even if within 12
    BASE = 8
    return (
        cf.for_each(UnitToken)
        .filter(lambda u: not _is_unassigned(u.employee) and u.day.id >= 0)
        .group_by(lambda u: (u.employee, u.day.id), ConstraintCollectors.sum(lambda u: u.hours))
        .filter(lambda key, total_h: total_h > BASE)
        .penalize(HardSoftScore.ONE_SOFT, lambda key, total_h: total_h - BASE)
        .as_constraint("avoid-overtime-over-8-soft")
    )

def finish_earlier_soft(cf: ConstraintFactory) -> Constraint:
    # Prefer earlier within the task window, weighted by distance from start
    WEIGHT = 1_000
    return (
        cf.for_each(UnitToken)
        .filter(lambda u: u.day is not None and u.day.id >= 0)
        .penalize(
            HardSoftScore.ONE_SOFT,
            lambda u: WEIGHT * max(0, (u.day.id - u.start_day_id)) * max(1, u.hours)
        )
        .as_constraint("finish-earlier-from-task-start-soft")
    )

def balance_total_hours_soft(cf: ConstraintFactory) -> Constraint:
    # Balance total hours across employees (ignore dummy)
    return (
        cf.for_each(UnitToken)
        .filter(lambda u: u.employee.id != 0)
        .group_by(lambda u: u.employee, ConstraintCollectors.sum(lambda u: u.hours))
        .penalize(HardSoftScore.ONE_SOFT,
                  lambda emp, total_h: int(abs(total_h - _TARGET_HOURS_PER_EMP)))
        .as_constraint("balance-total-hours-soft")
    )

# -------------------- YAML & builders --------------------

def load_config_modules(path: str):
    with open(path, "r", encoding="utf-8") as f:
        cfg = yaml.safe_load(f)

    global _STAFF_MIN_PER_DAY, _STAFF_MAX_PER_DAY, _ALLOWED_HOURS_8U

    # token hours domain for 8-hour units
    _ALLOWED_HOURS_8U = list((cfg.get("quantum") or {}).get("token_hours_options") or [4, 8, 10, 12])

    # staffing heads min/max
    staff_cfg = (cfg.get("staffing_per_task_per_day") or {})
    _STAFF_MIN_PER_DAY = int(staff_cfg["min"]) if "min" in staff_cfg else None
    _STAFF_MAX_PER_DAY = int(staff_cfg["max"]) if "max" in staff_cfg else None

    start_day = datetime.strptime(str(cfg["start_day"]), "%Y-%m-%d").date()
    horizon_days = int(cfg.get("horizon_days", 30))

    # Build days with a dummy first
    days: List[DaySlot] = [DaySlot(-1, start_day - timedelta(days=1))]  # dummy "unassigned"
    days += [DaySlot(i, start_day + timedelta(days=i)) for i in range(horizon_days)]

    # Task windows
    windows: List[TaskWindow] = []
    for m in cfg["modules"]:
        mcode = str(m["code"]).strip()
        mfactory = str(m.get("factory", "")).strip()
        m_start_idx = (datetime.strptime(str(m.get("start_date", cfg["start_day"])), "%Y-%m-%d").date() - start_day).days
        for proc in m["processes"]:
            pid = int(proc["id"])
            p_end_idx = (datetime.strptime(str(proc["end_date"]), "%Y-%m-%d").date() - start_day).days
            for t in proc["tasks"]:
                full_code = str(t["code"]).strip().upper()  # e.g., S1-P2-A
                parts = full_code.split("-")
                letter = parts[2] if len(parts) >= 3 else full_code[-1:]
                units = int(t.get("workload_units", 0))
                unit_hours = int(t.get("unit_hours", 8))
                windows.append(TaskWindow(
                    module=mcode, factory=mfactory, process_id=pid, task_letter=letter,
                    start_day_id=m_start_idx, end_day_id=p_end_idx,
                    unit_hours=unit_hours, workload_units=units
                ))

    # Employees with a dummy first
    employees: List[Employee] = [Employee(id=0, name="__UNASSIGNED__", skills={})]
    eid = 1
    for e in cfg["employees"]:
        name = str(e["name"])
        skills = {str(k).strip().upper(): int(v) for k, v in (e.get("skills", {}) or {}).items()}
        employees.append(Employee(id=eid, name=name, skills=skills))
        eid += 1

    # Hours range = union of allowed options + 4 (ensures feasibility)
    hours_options: List[int] = sorted(set([4] + _ALLOWED_HOURS_8U))

    return start_day, days, windows, employees, hours_options

def build_tokens(windows: List[TaskWindow], employees: List[Employee], days: List[DaySlot]) -> List[UnitToken]:
    # placeholders:
    dummy_emp = employees[0]           # id == 0
    dummy_day = days[0]                # id == -1
    tokens: List[UnitToken] = []
    tid = 1
    for w in windows:
        for _ in range(max(0, int(w.workload_units))):
            tokens.append(UnitToken(
                id=tid,
                module=w.module,
                factory=w.factory,
                process_id=w.process_id,
                task_letter=w.task_letter,
                start_day_id=w.start_day_id,
                end_day_id=w.end_day_id,
                unit_hours=w.unit_hours,
                employee=dummy_emp,   # start as UNASSIGNED
                day=dummy_day,        # start as dummy day
                hours=4 if w.unit_hours == 4 else _ALLOWED_HOURS_8U[0]
            ))
            tid += 1
    return tokens

# -------------------- Solver --------------------

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
        entity_class_list=[UnitToken],
        score_director_factory_config=ScoreDirectorFactoryConfig(
            constraint_provider_function=define_constraints
        ),
        termination_config=TerminationConfig(**term_kwargs)
    )
    return SolverFactory.create(cfg).build_solver()

def solve_from_config(cfg_path: str = "config_modules.yaml"):
    global _TARGET_HOURS_PER_EMP, _TASK_REQUIRED_HOURS

    start_day, days, windows, employees, hours_options = load_config_modules(cfg_path)

    # Required hours per task = units * unit_hours
    _TASK_REQUIRED_HOURS = {
        (w.module, w.process_id, w.task_letter): int(w.workload_units * w.unit_hours)
        for w in windows
    }
    total_required_hours = sum(_TASK_REQUIRED_HOURS.values())
    # exclude dummy from balancing target
    real_emp_count = max(1, len(employees) - 1)
    _TARGET_HOURS_PER_EMP = total_required_hours / real_emp_count

    tokens = build_tokens(windows, employees, days)

    problem = Schedule(
        days=days,
        employees=employees,
        hours_options=hours_options,
        tokens=tokens
    )

    t0 = time.time()

    # Pass 1: stop when feasible OR time limit
    solver1 = _build_solver(best_limit="0hard/*soft", spent_minutes=25)
    after_pass1: Schedule = solver1.solve(problem)
    t1 = time.time()
    print(f"[Pass 1] score={after_pass1.score}  time={t1 - t0:.2f}s")

    # Pass 2: improve soft with an unimproved limit
    solver2 = _build_solver(spent_minutes=5, unimproved_seconds=60)
    final: Schedule = solver2.solve(after_pass1)
    t2 = time.time()
    print(f"[Pass 2] score={final.score}  time={t2 - t1:.2f}s  (total {t2 - t0:.2f}s)")

    # Sanity: top employee-day loads (ignoring dummy)
    by_emp_day = defaultdict(int)
    for u in final.tokens:
        if not _is_unassigned(u.employee) and u.day.id >= 0:
            by_emp_day[(u.employee.name, u.day.id)] += int(u.hours)
    for (emp, did), h in sorted(by_emp_day.items(), key=lambda kv: -kv[1])[:10]:
        print(f"  {emp} @ day#{did}: {h}h")

    return final, start_day

def main():
    solve_from_config("config_modules.yaml")

if __name__ == "__main__":
    main()
