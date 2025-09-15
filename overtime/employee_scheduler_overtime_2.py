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
    HardMediumSoftScore,
    ConstraintFactory, Constraint, Joiners,
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
    ] = 4

    def pcode(self) -> str: return f"P{self.process_id}-{self.task_letter}"
    def tcode(self) -> str: return f"{self.module}-P{self.process_id}-{self.task_letter}"

@planning_solution
@dataclass
class Schedule:
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
    score: Annotated[HardMediumSoftScore, PlanningScore] = field(default=None)

# -------------------- Globals --------------------

_DAILY_CAP: int = 12
_TARGET_HOURS_PER_EMP: float = 0.0
_TASK_REQUIRED_HOURS: Dict[Tuple[str, int, str], int] = {}
_ALLOWED_HOURS_8U: List[int] = [4, 8, 10, 12]

_STAFF_MIN_PER_DAY: int | None = None
_STAFF_MAX_PER_DAY: int | None = None

# Staffing mode:
#   "max_only" -> enforce only the MAX heads bound
#   "minmax"   -> enforce both MIN and MAX bounds
_STAFFING_MODE: str = "minmax"
_STAFFING_HARD: bool = True
# -------------------- Constraints --------------------

@constraint_provider
def define_constraints(cf: ConstraintFactory) -> List[Constraint]:
    # choose staffing rule by mode+tier
    if _STAFFING_MODE == "max_only":
        staffing_rule = (staffing_max_heads_hard(cf) if _STAFFING_HARD else staffing_max_heads_medium(cf))
    else:  # "minmax"
        staffing_rule = (staffing_minmax_heads_hard(cf) if _STAFFING_HARD else staffing_minmax_heads_medium(cf))

    return [
        require_assignment_and_skill(cf),
        # enforce_hours_domain_per_task(cf),
        process_precedence_within_module(cf),
        within_window(cf),
        daily_capacity_hard(cf),
        single_factory_per_emp_day_hard(cf),
        task_hours_equality_hard(cf),

        staffing_rule,  # injected here

        avoid_overtime_soft(cf),
        finish_earlier_medium(cf),
        balance_total_hours_soft(cf),
    ]

def _is_unassigned(emp: Employee) -> bool:
    return emp.id == 0

def require_assignment_and_skill(cf: ConstraintFactory) -> Constraint:
    return (
        cf.for_each(UnitToken)
        .filter(lambda u:
            _is_unassigned(u.employee) or
            u.employee.skills.get(u.pcode(), 0) < 1 or
            (u.hours is None) or (u.hours < 1)
        )
        .penalize(HardMediumSoftScore.ONE_HARD)
        .as_constraint("assigned+eligible-skill+min-hours")
    )

def enforce_hours_domain_per_task(cf: ConstraintFactory) -> Constraint:
    return (
        cf.for_each(UnitToken)
        .filter(lambda u:
            (u.unit_hours == 4 and u.hours != 4) or
            (u.unit_hours == 8 and u.hours not in _ALLOWED_HOURS_8U)
        )
        .penalize(HardMediumSoftScore.ONE_HARD)
        .as_constraint("hours-domain-per-task")
    )

def process_precedence_within_module(cf: ConstraintFactory) -> Constraint:
    return (
        cf.for_each(UnitToken)
        .filter(lambda a: a.day is not None and a.day.id >= 0)
        .join(
            cf.for_each(UnitToken).filter(lambda b: b.day is not None and b.day.id >= 0),
            Joiners.equal(lambda a: a.module, lambda b: b.module),
            Joiners.equal(lambda a: a.process_id + 1, lambda b: b.process_id)
        )
        .filter(lambda a, b: not (b.day.id > a.day.id))
        .penalize(HardMediumSoftScore.ONE_HARD)
        .as_constraint("process-precedence-per-module")
    )

def within_window(cf: ConstraintFactory) -> Constraint:
    def distance_penalty(u: UnitToken) -> int:
        if (u.day is None) or (u.day.id < 0):
            return 100 * max(1, u.hours)
        if u.day.id < u.start_day_id:
            return (u.start_day_id - u.day.id) * max(1, u.hours)
        if u.day.id > u.end_day_id:
            return (u.day.id - u.end_day_id) * max(1, u.hours)
        return 0

    return (
        cf.for_each(UnitToken)
        .filter(lambda u: distance_penalty(u) > 0)
        .penalize(HardMediumSoftScore.ONE_HARD, lambda u: distance_penalty(u))
        .as_constraint("within-window-distance-weighted")
    )

def daily_capacity_hard(cf: ConstraintFactory) -> Constraint:
    return (
        cf.for_each(UnitToken)
        .filter(lambda u: not _is_unassigned(u.employee) and u.day.id >= 0)
        .group_by(lambda u: (u.employee, u.day.id), ConstraintCollectors.sum(lambda u: u.hours))
        .filter(lambda key, total_h: total_h > _DAILY_CAP)
        .penalize(HardMediumSoftScore.ONE_HARD, lambda key, total_h: total_h - _DAILY_CAP)
        .as_constraint("daily-cap-12h")
    )

def single_factory_per_emp_day_hard(cf: ConstraintFactory) -> Constraint:
    return (
        cf.for_each(UnitToken)
        .filter(lambda u: not _is_unassigned(u.employee) and u.day.id >= 0)
        .group_by(
            lambda u: (u.employee, u.day.id),
            ConstraintCollectors.count_distinct(lambda u: u.factory)
        )
        .filter(lambda key, distinct_factories: distinct_factories > 1)
        .penalize(HardMediumSoftScore.ONE_HARD, lambda key, distinct_factories: distinct_factories - 1)
        .as_constraint("one-factory-per-emp-day")
    )

def task_hours_equality_hard(cf: ConstraintFactory) -> Constraint:
    def required_hours(task_key) -> int:
        return _TASK_REQUIRED_HOURS.get(task_key, 0)

    return (
        cf.for_each(UnitToken)
        .group_by(lambda u: (u.module, u.process_id, u.task_letter),
                  ConstraintCollectors.sum(lambda u: u.hours))
        .filter(lambda key, total_h: total_h != required_hours(key))
        .penalize(HardMediumSoftScore.ONE_HARD, lambda key, total_h: abs(total_h - required_hours(key)))
        .as_constraint("task-total-hours-equality")
    )

# ---- Staffing constraints ----

def _count_heads(cf: ConstraintFactory):
    return (
        cf.for_each(UnitToken)
        .filter(lambda u: u.day.id >= 0 and u.employee.id != 0 and (u.hours or 0) > 0)
        .group_by(
            lambda u: (u.module, u.process_id, u.task_letter, u.day.id),
            ConstraintCollectors.count_distinct(lambda u: u.employee)
        )
    )

def staffing_max_heads_hard(cf: ConstraintFactory) -> Constraint:
    if _STAFF_MAX_PER_DAY is None:
        return (
            cf.for_each(UnitToken).filter(lambda _: False)
            .penalize(HardMediumSoftScore.ONE_HARD)
            .as_constraint("staffing-max-disabled")
        )
    return (
        _count_heads(cf)
        .filter(lambda key, heads: heads > _STAFF_MAX_PER_DAY)
        .penalize(HardMediumSoftScore.ONE_HARD, lambda key, heads: heads - _STAFF_MAX_PER_DAY)
        .as_constraint("staffing-max-heads-per-task-day")
    )

def staffing_max_heads_medium(cf: ConstraintFactory) -> Constraint:
    if _STAFF_MAX_PER_DAY is None:
        return (
            cf.for_each(UnitToken).filter(lambda _: False)
            .penalize(HardMediumSoftScore.ONE_MEDIUM)
            .as_constraint("staffing-max-medium-disabled")
        )
    return (
        _count_heads(cf)
        .filter(lambda key, heads: heads > _STAFF_MAX_PER_DAY)
        .penalize(HardMediumSoftScore.ONE_MEDIUM, lambda key, heads: heads - _STAFF_MAX_PER_DAY)
        .as_constraint("staffing-max-heads-per-task-day-medium")
    )

def staffing_minmax_heads_hard(cf: ConstraintFactory) -> Constraint:
    if _STAFF_MIN_PER_DAY is None and _STAFF_MAX_PER_DAY is None:
        return (
            cf.for_each(UnitToken).filter(lambda _: False)
            .penalize(HardMediumSoftScore.ONE_HARD)
            .as_constraint("staffing-minmax-disabled")
        )
    return (
        _count_heads(cf)
        .filter(lambda key, heads:
            (_STAFF_MIN_PER_DAY is not None and heads < _STAFF_MIN_PER_DAY) or
            (_STAFF_MAX_PER_DAY is not None and heads > _STAFF_MAX_PER_DAY)
        )
        .penalize(
            HardMediumSoftScore.ONE_HARD,
            lambda key, heads:
                ((_STAFF_MIN_PER_DAY - heads) if (_STAFF_MIN_PER_DAY is not None and heads < _STAFF_MIN_PER_DAY) else 0) +
                ((heads - _STAFF_MAX_PER_DAY) if (_STAFF_MAX_PER_DAY is not None and heads > _STAFF_MAX_PER_DAY) else 0)
        )
        .as_constraint("staffing-minmax-heads-per-task-day")
    )

def staffing_minmax_heads_medium(cf: ConstraintFactory) -> Constraint:
    if _STAFF_MIN_PER_DAY is None and _STAFF_MAX_PER_DAY is None:
        return (
            cf.for_each(UnitToken).filter(lambda _: False)
            .penalize(HardMediumSoftScore.ONE_MEDIUM)
            .as_constraint("staffing-minmax-medium-disabled")
        )
    return (
        _count_heads(cf)
        .filter(lambda key, heads:
            (_STAFF_MIN_PER_DAY is not None and heads < _STAFF_MIN_PER_DAY) or
            (_STAFF_MAX_PER_DAY is not None and heads > _STAFF_MAX_PER_DAY)
        )
        .penalize(
            HardMediumSoftScore.ONE_MEDIUM,
            lambda key, heads:
                ((_STAFF_MIN_PER_DAY - heads) if (_STAFF_MIN_PER_DAY is not None and heads < _STAFF_MIN_PER_DAY) else 0) +
                ((heads - _STAFF_MAX_PER_DAY) if (_STAFF_MAX_PER_DAY is not None and heads > _STAFF_MAX_PER_DAY) else 0)
        )
        .as_constraint("staffing-minmax-heads-per-task-day-medium")
    )

# ---- Softs ----

def finish_earlier_medium(cf: ConstraintFactory) -> Constraint:
    WEIGHT = 1 
    return (
        cf.for_each(UnitToken)
        .filter(lambda u: u.day is not None and u.day.id >= 0)
        .penalize(
            HardMediumSoftScore.ONE_MEDIUM,
            lambda u: WEIGHT * max(0, (u.day.id - u.start_day_id))
        )
        .as_constraint("finish-earlier-from-task-start-medium")
    )

def avoid_overtime_soft(cf: ConstraintFactory) -> Constraint:
    BASE = 8
    return (
        cf.for_each(UnitToken)
        .filter(lambda u: not _is_unassigned(u.employee) and u.day.id >= 0)
        .group_by(lambda u: (u.employee, u.day.id), ConstraintCollectors.sum(lambda u: u.hours))
        .filter(lambda key, total_h: total_h > BASE)
        .penalize(HardMediumSoftScore.ONE_SOFT, lambda key, total_h: total_h - BASE)
        .as_constraint("avoid-overtime-over-8-soft")
    )

def balance_total_hours_soft(cf: ConstraintFactory) -> Constraint:
    return (
        cf.for_each(UnitToken)
        .filter(lambda u: u.employee.id != 0)
        .group_by(lambda u: u.employee, ConstraintCollectors.sum(lambda u: u.hours))
        .penalize(HardMediumSoftScore.ONE_SOFT,
                  lambda emp, total_h: int(abs(total_h - _TARGET_HOURS_PER_EMP)))
        .as_constraint("balance-total-hours-soft")
    )

# -------------------- YAML & builders --------------------

def load_config_modules(path: str):
    with open(path, "r", encoding="utf-8") as f:
        cfg = yaml.safe_load(f)

    global _STAFF_MIN_PER_DAY, _STAFF_MAX_PER_DAY, _ALLOWED_HOURS_8U

    _ALLOWED_HOURS_8U = list((cfg.get("quantum") or {}).get("token_hours_options") or [4, 8, 10, 12])

    staff_cfg = (cfg.get("staffing_per_task_per_day") or {})
    _STAFF_MIN_PER_DAY = int(staff_cfg["min"]) if "min" in staff_cfg else None
    _STAFF_MAX_PER_DAY = int(staff_cfg["max"]) if "max" in staff_cfg else None

    start_day = datetime.strptime(str(cfg["start_day"]), "%Y-%m-%d").date()
    horizon_days = int(cfg.get("horizon_days", 30))

    days: List[DaySlot] = [DaySlot(-1, start_day - timedelta(days=1))]
    days += [DaySlot(i, start_day + timedelta(days=i)) for i in range(horizon_days)]

    windows: List[TaskWindow] = []
    for m in cfg["modules"]:
        mcode = str(m["code"]).strip()
        mfactory = str(m.get("factory", "")).strip()
        m_start_idx = (datetime.strptime(str(m.get("start_date", cfg["start_day"])), "%Y-%m-%d").date() - start_day).days
        for proc in m["processes"]:
            pid = int(proc["id"])
            p_end_idx = (datetime.strptime(str(proc["end_date"]), "%Y-%m-%d").date() - start_day).days
            for t in proc["tasks"]:
                full_code = str(t["code"]).strip().upper()
                parts = full_code.split("-")
                letter = parts[2] if len(parts) >= 3 else full_code[-1:]
                units = int(t.get("workload_units", 0))
                unit_hours = int(t.get("unit_hours", 8))
                windows.append(TaskWindow(
                    module=mcode, factory=mfactory, process_id=pid, task_letter=letter,
                    start_day_id=m_start_idx, end_day_id=p_end_idx,
                    unit_hours=unit_hours, workload_units=units
                ))

    employees: List[Employee] = [Employee(id=0, name="__UNASSIGNED__", skills={})]
    eid = 1
    for e in cfg["employees"]:
        name = str(e["name"])
        skills = {str(k).strip().upper(): int(v) for k, v in (e.get("skills", {}) or {}).items()}
        employees.append(Employee(id=eid, name=name, skills=skills))
        eid += 1

    hours_options: List[int] = sorted(set([4] + _ALLOWED_HOURS_8U))
    return start_day, days, windows, employees, hours_options

def build_tokens(windows: List[TaskWindow], employees: List[Employee], days: List[DaySlot]) -> List[UnitToken]:
    dummy_emp = employees[0]
    dummy_day = days[0]
    tokens: List[UnitToken] = []
    tid = 1
    for w in windows:
        total_required_hours = w.unit_hours * w.workload_units
        if w.unit_hours == 4:
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
                    employee=dummy_emp,
                    day=dummy_day,
                    hours=4
                ))
                tid += 1
        else:
            # greedy pack into allowed hour-chunks for speed
            allowed = sorted(_ALLOWED_HOURS_8U)  # e.g. [4, 8, 10, 12]
            remaining = int(total_required_hours)
            while remaining > 0:
                valid_options = [x for x in allowed if x <= remaining]
                if not valid_options:
                    raise ValueError(f"No valid hours option for remaining={remaining} in task {w.tcode()}")
                h = max(valid_options)
                tokens.append(UnitToken(
                    id=tid,
                    module=w.module,
                    factory=w.factory,
                    process_id=w.process_id,
                    task_letter=w.task_letter,
                    start_day_id=w.start_day_id,
                    end_day_id=w.end_day_id,
                    unit_hours=w.unit_hours,
                    employee=dummy_emp,
                    day=dummy_day,
                    hours=h
                ))
                tid += 1
                remaining -= h
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
    global _TARGET_HOURS_PER_EMP, _TASK_REQUIRED_HOURS, _STAFFING_HARD, _STAFFING_MODE

    start_day, days, windows, employees, hours_options = load_config_modules(cfg_path)

    _TASK_REQUIRED_HOURS = {
        (w.module, w.process_id, w.task_letter): int(w.workload_units * w.unit_hours)
        for w in windows
    }
    total_required_hours = sum(_TASK_REQUIRED_HOURS.values())
    real_emp_count = max(1, len(employees) - 1)
    _TARGET_HOURS_PER_EMP = total_required_hours / real_emp_count

    tokens = build_tokens(windows, employees, days)

    problem = Schedule(
        days=days,
        employees=employees,
        hours_options=hours_options,
        tokens=tokens
    )

    # knobs
    PASS1_MIN, PASS2_MIN, PASS3_MIN, PASS4_MIN = 15, 5, 10, 5
    UNIMPROVED_SEC = 60

    t0 = time.time()

    # ---------- Pass 1: ONLY MAX (HARD), aim 0 hard ----------
    _STAFFING_MODE = "max_only"
    _STAFFING_HARD = True
    solver1 = _build_solver(best_limit="0hard/*medium/*soft",
                            spent_minutes=PASS1_MIN,
                            unimproved_seconds=UNIMPROVED_SEC)
    after_pass1: Schedule = solver1.solve(problem)
    t1 = time.time()
    print(f"[Pass 1] score={after_pass1.score}  time={t1 - t0:.2f}s  (STAFFING=max_only HARD)")

    # ---------- Pass 2: ONLY MAX (HARD), polish soft ----------
    _STAFFING_MODE = "max_only"
    _STAFFING_HARD = True
    solver2 = _build_solver(spent_minutes=PASS2_MIN,
                            unimproved_seconds=UNIMPROVED_SEC)
    after_pass2: Schedule = solver2.solve(after_pass1)
    t2 = time.time()
    print(f"[Pass 2] score={after_pass2.score}  time={t2 - t1:.2f}s  (STAFFING=max_only HARD, soft polish)")

    # ---------- Pass 3: MIN+MAX (HARD), aim 0 hard ----------
    _STAFFING_MODE = "minmax"
    _STAFFING_HARD = True
    solver3 = _build_solver(best_limit="0hard/*medium/*soft",
                            spent_minutes=PASS3_MIN,
                            unimproved_seconds=UNIMPROVED_SEC)
    after_pass3: Schedule = solver3.solve(after_pass2)
    t3 = time.time()
    print(f"[Pass 3] score={after_pass3.score}  time={t3 - t2:.2f}s  (STAFFING=minmax HARD)")

    # ---------- Pass 4: MIN+MAX (HARD), polish soft ----------
    _STAFFING_MODE = "minmax"
    _STAFFING_HARD = True
    solver4 = _build_solver(spent_minutes=PASS4_MIN,
                            unimproved_seconds=UNIMPROVED_SEC)
    final: Schedule = solver4.solve(after_pass3)
    t4 = time.time()
    print(f"[Pass 4] score={final.score}  time={t4 - t3:.2f}s  (total {t4 - t0:.2f}s; STAFFING=minmax HARD, soft polish)")

    # ------- Hard-violation audit (debug prints) -------
    violations = []

    # within-window
    for u in final.tokens:
        if u.employee.id == 0 or u.day.id < 0:
            continue
        if not (u.start_day_id <= u.day.id <= u.end_day_id):
            violations.append(("WINDOW", u.tcode(), u.day.id, (u.start_day_id, u.end_day_id)))

    # hours-domain-per-task (kept off in constraints for speed, but still auditing)
    for u in final.tokens:
        if u.employee.id == 0 or u.day.id < 0:
            continue
        if (u.unit_hours == 4 and u.hours != 4) or (u.unit_hours == 8 and u.hours not in _ALLOWED_HOURS_8U):
            violations.append(("HOURS_DOMAIN", u.tcode(), u.hours, u.unit_hours))

    from collections import defaultdict as _dd
    by_emp_day_sum = _dd(int)
    for u in final.tokens:
        if u.employee.id == 0 or u.day.id < 0:
            continue
        by_emp_day_sum[(u.employee.name, u.day.id)] += int(u.hours)
    for (emp, did), total in by_emp_day_sum.items():
        if total > _DAILY_CAP:
            violations.append(("DAILY_CAP", emp, did, total))

    by_emp_day_factories = _dd(set)
    for u in final.tokens:
        if u.employee.id == 0 or u.day.id < 0:
            continue
        by_emp_day_factories[(u.employee.name, u.day.id)].add(u.factory)
    for (emp, did), facs in by_emp_day_factories.items():
        if len(facs) > 1:
            violations.append(("FACTORY_MIX", emp, did, sorted(facs)))

    heads = _dd(set)
    for u in final.tokens:
        if u.employee.id == 0 or u.day.id < 0 or (u.hours or 0) <= 0:
            continue
        heads[(u.module, u.process_id, u.task_letter, u.day.id)].add(u.employee.name)
    for key, people in heads.items():
        count = len(people)
        if (_STAFF_MIN_PER_DAY is not None and count < _STAFF_MIN_PER_DAY) or \
           (_STAFF_MAX_PER_DAY is not None and count > _STAFF_MAX_PER_DAY):
            violations.append(("STAFFING", key, count))

    hours_by_task = _dd(int)
    required = _TASK_REQUIRED_HOURS
    for u in final.tokens:
        if u.employee.id == 0 or u.day.id < 0:
            continue
        hours_by_task[(u.module, u.process_id, u.task_letter)] += int(u.hours)
    for key, total in hours_by_task.items():
        req = required.get(key, 0)
        if total != req:
            violations.append(("TASK_HOURS", key, total, req))

    print("=== HARD VIOLATIONS (first 20) ===")
    for v in violations[:20]:
        print(v)
    print(f"TOTAL HARD VIOLATIONS FOUND: {len(violations)}")

    # Return snapshots for exporter (pass1, pass2, pass3, final)
    return final, start_day, after_pass1, after_pass2, after_pass3

def main():
    solve_from_config("config_modules.yaml")

if __name__ == "__main__":
    main()
