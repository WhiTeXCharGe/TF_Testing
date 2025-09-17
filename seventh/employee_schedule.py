# employee_schedule.py
# Reads EnvConfig.yaml + Schedule.yaml, builds the problem entirely from them,
# solves with Timefold in 4 passes, prints hard-violation audits after each pass,
# and OVERWRITES Schedule.yaml with assignment_list (final pass).

from dataclasses import dataclass, field
from datetime import date, datetime, timedelta
from typing import Annotated, Dict, List, Optional, Tuple
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
from timefold.solver.score import (
    HardMediumSoftScore,
    ConstraintFactory, Constraint, Joiners,
    constraint_provider, ConstraintCollectors
)

from export_schedule import overwrite_schedule_with_assignments

# -------------------- Domain --------------------

@dataclass(frozen=True)
class DaySlot:
    id: int
    d: date

@dataclass(frozen=True)
class Employee:
    # internal numeric id for solver; wid for writing back to Schedule.yaml
    id: int
    wid: str
    name: str
    skills: Dict[str, int]

@dataclass(frozen=True)
class TaskWindow:
    module: str             # e.g., e1
    factory: str            # fab id from Schedule
    phase_id: str           # "p1".."p4"
    phase_num: int          # 1..4
    op_id: str              # "p#o#"
    start_day_id: int       # inclusive
    end_day_id: int         # inclusive
    allowed: List[int]      # allowed hours for this op (LIST, not tuple)
    min_heads: int
    max_heads: int
    workload_days: int      # from Schedule.yaml

@planning_entity
@dataclass
class UnitToken:
    """
    1 token = a chunk of workload hours for an op-task.
    Solver assigns (employee, day, hours).
    """
    id: Annotated[int, PlanningId]

    # Facts copied from TaskWindow (do not change during solving)
    module: str
    factory: str
    phase_id: str
    phase_num: int
    op_id: str
    start_day_id: int
    end_day_id: int
    allowed: List[int]      # LIST, not tuple

    # Planning variables
    employee: Annotated[Employee, PlanningVariable]
    day:      Annotated[DaySlot, PlanningVariable]
    hours:    Annotated[int, PlanningVariable]

@planning_solution
@dataclass
class Schedule:
    days: Annotated[List[DaySlot], ProblemFactCollectionProperty, ValueRangeProvider]
    employees: Annotated[List[Employee], ProblemFactCollectionProperty, ValueRangeProvider]
    hours_options: Annotated[List[int], ProblemFactCollectionProperty, ValueRangeProvider]
    tokens: Annotated[List[UnitToken], PlanningEntityCollectionProperty]
    score: Annotated[HardMediumSoftScore, PlanningScore] = field(default=None)

# -------------------- Globals (filled from YAML each run) --------------------

_DAILY_CAP: int = 12
_TASK_REQUIRED_HOURS: Dict[Tuple[str, str], int] = {}   # (module, op_id) -> required hours
_OP_MINMAX: Dict[str, Tuple[int, int]] = {}             # op_id -> (min, max)
_TARGET_HOURS_PER_EMP: float = 0.0

# staffing toggle for the 4-pass pipeline
_STAFFING_MODE: str = "minmax"   # "max_only" | "minmax"
_STAFFING_HARD: bool = True      # True: hard tier; False: medium tier

# -------------------- Constraints --------------------

def _is_unassigned(emp: Optional[Employee]) -> bool:
    return (emp is None) or (emp.id == 0)

def _skill_level(emp: Optional[Employee], op_id: str) -> int:
    if emp is None:
        return 0
    skills = emp.skills
    if skills is None:
        return 0
    return int(skills.get(op_id, 0))

@constraint_provider
def define_constraints(cf: ConstraintFactory) -> List[Constraint]:
    # choose staffing rule by mode+tier
    if _STAFFING_MODE == "max_only":
        staffing_rule = (staffing_max_heads_hard(cf) if _STAFFING_HARD else staffing_max_heads_medium(cf))
    else:  # "minmax"
        staffing_rule = (staffing_minmax_heads_hard(cf) if _STAFFING_HARD else staffing_minmax_heads_medium(cf))

    return [
        require_assignment_and_skill(cf),
        enforce_hours_domain_per_task(cf),
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

def require_assignment_and_skill(cf: ConstraintFactory) -> Constraint:
    return (
        cf.for_each(UnitToken)
          .filter(lambda u:
              _is_unassigned(u.employee) or
              (u.hours is None) or (u.hours < 1) or
              (_skill_level(u.employee, u.op_id) < 1)
          )
          .penalize(HardMediumSoftScore.ONE_HARD)
          .as_constraint("assigned+eligible-skill+min-hours")
    )

def enforce_hours_domain_per_task(cf: ConstraintFactory) -> Constraint:
    return (
        cf.for_each(UnitToken)
          .filter(lambda u: (u.hours is not None) and (u.allowed is not None) and (u.hours not in set(u.allowed)))
          .penalize(HardMediumSoftScore.ONE_HARD)
          .as_constraint("hours-domain-per-op")
    )

def process_precedence_within_module(cf: ConstraintFactory) -> Constraint:
    # within same module, no p(k+1) before p(k)
    return (
        cf.for_each(UnitToken)
          .filter(lambda a: a.day is not None and a.day.id >= 0)
          .join(
              cf.for_each(UnitToken).filter(lambda b: b.day is not None and b.day.id >= 0),
              Joiners.equal(lambda a: a.module, lambda b: b.module),
              Joiners.equal(lambda a: a.phase_num + 1, lambda b: b.phase_num)
          )
          .filter(lambda a, b: not (b.day.id > a.day.id))
          .penalize(HardMediumSoftScore.ONE_HARD)
          .as_constraint("process-precedence-per-module")
    )

def within_window(cf: ConstraintFactory) -> Constraint:
    def distance_penalty(u: UnitToken) -> int:
        h = u.hours if u.hours is not None else 1
        if (u.day is None) or (u.day.id < 0):
            return 100 * max(1, h)
        if u.day.id < u.start_day_id:
            return (u.start_day_id - u.day.id) * max(1, h)
        if u.day.id > u.end_day_id:
            return (u.day.id - u.end_day_id) * max(1, h)
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
          .filter(lambda u: (not _is_unassigned(u.employee)) and (u.day is not None) and (u.day.id >= 0))
          .group_by(
              lambda u: (u.employee.id, u.day.id),
              ConstraintCollectors.sum(lambda u: int(u.hours) if u.hours is not None else 0)
          )
          .filter(lambda key, total_h: total_h > _DAILY_CAP)
          .penalize(HardMediumSoftScore.ONE_HARD, lambda key, total_h: total_h - _DAILY_CAP)
          .as_constraint("daily-cap-12h")
    )

def single_factory_per_emp_day_hard(cf: ConstraintFactory) -> Constraint:
    return (
        cf.for_each(UnitToken)
          .filter(lambda u: (not _is_unassigned(u.employee)) and (u.day is not None) and (u.day.id >= 0))
          .group_by(
              lambda u: (u.employee.id, u.day.id),
              ConstraintCollectors.count_distinct(lambda u: u.factory)
          )
          .filter(lambda key, distinct_factories: distinct_factories > 1)
          .penalize(HardMediumSoftScore.ONE_HARD, lambda key, distinct_factories: distinct_factories - 1)
          .as_constraint("one-factory-per-emp-day")
    )

def task_hours_equality_hard(cf: ConstraintFactory) -> Constraint:
    def required_hours(key) -> int:
        return int(_TASK_REQUIRED_HOURS.get(key, 0))

    return (
        cf.for_each(UnitToken)
          .group_by(lambda u: (u.module, u.op_id),
                    ConstraintCollectors.sum(lambda u: int(u.hours) if u.hours is not None else 0))
          .filter(lambda key, total_h: total_h != required_hours(key))
          .penalize(HardMediumSoftScore.ONE_HARD, lambda key, total_h: abs(total_h - required_hours(key)))
          .as_constraint("task-total-hours-equality")
    )

# ---- Staffing constraints (mode/tier switchable) ----

def _count_heads(cf: ConstraintFactory):
    return (
        cf.for_each(UnitToken)
          .filter(lambda u: (u.day is not None and u.day.id >= 0) and (not _is_unassigned(u.employee)) and (u.hours is not None) and (u.hours > 0))
          .group_by(lambda u: (u.module, u.op_id, u.day.id),
                    ConstraintCollectors.count_distinct(lambda u: u.employee.id))
    )

def staffing_max_heads_hard(cf: ConstraintFactory) -> Constraint:
    return (
        _count_heads(cf)
          .filter(lambda key, heads: heads > _OP_MINMAX.get(key[1], (0, 10**9))[1])
          .penalize(HardMediumSoftScore.ONE_HARD,
                    lambda key, heads: heads - _OP_MINMAX.get(key[1], (0, 10**9))[1])
          .as_constraint("staffing-max-heads-per-task-day")
    )

def staffing_max_heads_medium(cf: ConstraintFactory) -> Constraint:
    return (
        _count_heads(cf)
          .filter(lambda key, heads: heads > _OP_MINMAX.get(key[1], (0, 10**9))[1])
          .penalize(HardMediumSoftScore.ONE_MEDIUM,
                    lambda key, heads: heads - _OP_MINMAX.get(key[1], (0, 10**9))[1])
          .as_constraint("staffing-max-heads-per-task-day-medium")
    )

def staffing_minmax_heads_hard(cf: ConstraintFactory) -> Constraint:
    def viol(key, heads):
        _, op_id, _ = key
        lo, hi = _OP_MINMAX.get(op_id, (0, 10**9))
        if heads < lo:
            return lo - heads
        if heads > hi:
            return heads - hi
        return 0
    return (
        _count_heads(cf)
          .filter(lambda key, heads: viol(key, heads) > 0)
          .penalize(HardMediumSoftScore.ONE_HARD, viol)
          .as_constraint("staffing-minmax-heads-per-task-day")
    )

def staffing_minmax_heads_medium(cf: ConstraintFactory) -> Constraint:
    def viol(key, heads):
        _, op_id, _ = key
        lo, hi = _OP_MINMAX.get(op_id, (0, 10**9))
        if heads < lo:
            return lo - heads
        if heads > hi:
            return heads - hi
        return 0
    return (
        _count_heads(cf)
          .filter(lambda key, heads: viol(key, heads) > 0)
          .penalize(HardMediumSoftScore.ONE_MEDIUM, viol)
          .as_constraint("staffing-minmax-heads-per-task-day-medium")
    )

# ---- Softs ----

def finish_earlier_medium(cf: ConstraintFactory) -> Constraint:
    WEIGHT = 1
    return (
        cf.for_each(UnitToken)
          .filter(lambda u: u.day is not None and u.day.id >= 0)
          .penalize(HardMediumSoftScore.ONE_MEDIUM,
                    lambda u: WEIGHT * max(0, (u.day.id - u.start_day_id)))
          .as_constraint("finish-earlier-from-task-start-medium")
    )

def avoid_overtime_soft(cf: ConstraintFactory) -> Constraint:
    BASE = 8
    return (
        cf.for_each(UnitToken)
          .filter(lambda u: (not _is_unassigned(u.employee)) and (u.day is not None) and (u.day.id >= 0))
          .group_by(lambda u: (u.employee.id, u.day.id),
                    ConstraintCollectors.sum(lambda u: int(u.hours) if u.hours is not None else 0))
          .filter(lambda key, total_h: total_h > BASE)
          .penalize(HardMediumSoftScore.ONE_SOFT, lambda key, total_h: total_h - BASE)
          .as_constraint("avoid-overtime-over-8-soft")
    )

def balance_total_hours_soft(cf: ConstraintFactory) -> Constraint:
    return (
        cf.for_each(UnitToken)
          .filter(lambda u: not _is_unassigned(u.employee))
          .group_by(lambda u: u.employee.id,
                    ConstraintCollectors.sum(lambda u: int(u.hours) if u.hours is not None else 0))
          .penalize(HardMediumSoftScore.ONE_SOFT,
                    lambda emp_id, total_h: int(abs(int(total_h) - int(_TARGET_HOURS_PER_EMP))))
          .as_constraint("balance-total-hours-soft")
    )

# -------------------- YAML & builders --------------------

def _phase_num_from_id(pid: str) -> int:
    try:
        return int(str(pid).strip().lower().replace("p", ""))
    except Exception:
        return 0

def _parse_env(env_path: str):
    with open(env_path, "r", encoding="utf-8") as f:
        env = yaml.safe_load(f)["environment"]

    # per-op settings from EnvConfig
    opdef: Dict[str, Dict] = {}
    hours_union = set()
    for ph in env["workflow_list"][0]["phase_list"]:
        for op in ph["operation_list"]:
            op_id = op["id"]          # e.g., p2o1
            hrs = list(op.get("work_hours", []) or [])
            if not hrs:
                hrs = [8]
            opdef[op_id] = {
                "phase_id": ph["id"],                 # "p1"
                "phase_num": _phase_num_from_id(ph["id"]),
                "allowed": list(sorted(hrs)),         # LIST, not tuple
                "min": int(op.get("min_worker_num", 0)),
                "max": int(op.get("max_worker_num", 10**9)),
            }
            hours_union.update(hrs)

    # employees
    employees: List[Employee] = [Employee(id=0, wid="__UNASSIGNED__", name="__UNASSIGNED__", skills={})]
    eid = 1
    for w in env["worker_list"]:
        wid = str(w.get("id"))  # "w1"
        name = w.get("name")
        skills = dict(w.get("skill_map") or {})
        employees.append(Employee(id=eid, wid=wid, name=name, skills=skills))
        eid += 1

    return opdef, employees, sorted(hours_union or {4, 8, 10, 12})

def _parse_schedule(env_opdef: Dict[str, Dict], sched_path: str):
    with open(sched_path, "r", encoding="utf-8") as f:
        s = yaml.safe_load(f)["schedule"]

    # horizon
    start_date = datetime.strptime(s["planrange"]["start_date"], "%Y/%m/%d").date()
    end_date   = datetime.strptime(s["planrange"]["end_date"], "%Y/%m/%d").date()
    horizon_days = (end_date - start_date).days + 1

    # days (include dummy day id -1 for unassigned)
    days: List[DaySlot] = [DaySlot(-1, start_date - timedelta(days=1))]
    days += [DaySlot(i, start_date + timedelta(days=i)) for i in range(horizon_days)]

    # windows from schedule
    windows: List[TaskWindow] = []
    required_hours: Dict[Tuple[str, str], int] = {}
    op_minmax: Dict[str, Tuple[int, int]] = {}

    for wf in s["workflow_task_list"]:
        module = wf["id"]            # e.g., e1
        fab = wf.get("fab")

        for ph_task in wf["phase_task_list"]:
            phase_id = ph_task["phase"]                      # "p1"
            phase_num = _phase_num_from_id(phase_id)
            p_start = datetime.strptime(ph_task["start_date"], "%Y/%m/%d").date()
            p_end   = datetime.strptime(ph_task["end_date"], "%Y/%m/%d").date()
            start_id = (p_start - start_date).days
            end_id   = (p_end   - start_date).days

            for ot in ph_task["operation_task_list"]:
                op_id = ot["operation"]                      # "p#o#"
                workload_days = int(ot["workload_days"])

                ocfg = env_opdef.get(op_id)
                if ocfg is None:
                    raise ValueError(f"operation {op_id} not present in EnvConfig.workflow_list")
                allowed = ocfg["allowed"]                    # LIST
                min_h = ocfg["min"]
                max_h = ocfg["max"]
                op_minmax[op_id] = (min_h, max_h)

                # required: workload_days × baseline (4 if only [4], else 8)
                baseline = 4 if (list(allowed) == [4]) else 8
                req_hours = workload_days * baseline
                required_hours[(module, op_id)] = required_hours.get((module, op_id), 0) + req_hours

                windows.append(TaskWindow(
                    module=module, factory=fab, phase_id=phase_id, phase_num=phase_num,
                    op_id=op_id, start_day_id=start_id, end_day_id=end_id,
                    allowed=list(allowed), min_heads=min_h, max_heads=max_h,
                    workload_days=workload_days
                ))

    return start_date, days, windows, required_hours, op_minmax

def _build_tokens(windows: List[TaskWindow]) -> List[UnitToken]:
    tokens: List[UnitToken] = []
    tid = 1
    for w in windows:
        baseline = 4 if list(w.allowed) == [4] else 8
        required_total = w.workload_days * baseline
        opts = sorted(set(w.allowed), reverse=True)
        remaining = int(required_total)
        if remaining <= 0:
            continue
        while remaining > 0:
            feasible = [h for h in opts if h <= remaining]
            h = feasible[0] if feasible else min(opts)
            tokens.append(UnitToken(
                id=tid,
                module=w.module, factory=w.factory,
                phase_id=w.phase_id, phase_num=w.phase_num,
                op_id=w.op_id,
                start_day_id=w.start_day_id, end_day_id=w.end_day_id,
                allowed=list(w.allowed),
                # dummy values (non-nullable planning vars)
                employee=Employee(0, "__UNASSIGNED__", "__UNASSIGNED__", {}),
                day=DaySlot(-1, date(1970, 1, 1)),
                hours=h
            ))
            tid += 1
            remaining -= h
    return tokens

# -------------------- Solver helpers --------------------

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

# -------------------- Auditing (print hard violations) --------------------

def _audit_hard(schedule: Schedule, check_min: bool) -> List[Tuple]:
    viols: List[Tuple] = []
    # within window
    for u in schedule.tokens:
        if u.employee.id == 0 or u.day.id < 0:
            continue
        if not (u.start_day_id <= u.day.id <= u.end_day_id):
            viols.append(("WINDOW", f"{u.module}-{u.op_id}", u.day.id, (u.start_day_id, u.end_day_id)))

    # hours domain
    for u in schedule.tokens:
        if u.employee.id == 0 or u.day.id < 0:
            continue
        if u.allowed is None or u.hours is None:
            continue
        if u.hours not in set(u.allowed):
            viols.append(("HOURS_DOMAIN", f"{u.module}-{u.op_id}", u.hours, list(u.allowed)))

    # daily cap
    by_emp_day_sum = defaultdict(int)
    for u in schedule.tokens:
        if u.employee.id == 0 or u.day.id < 0 or u.hours is None:
            continue
        by_emp_day_sum[(u.employee.name, u.day.id)] += int(u.hours)
    for (emp, did), total in by_emp_day_sum.items():
        if total > _DAILY_CAP:
            viols.append(("DAILY_CAP", emp, did, total))

    # factory mix
    by_emp_day_fabs = defaultdict(set)
    for u in schedule.tokens:
        if u.employee.id == 0 or u.day.id < 0:
            continue
        by_emp_day_fabs[(u.employee.name, u.day.id)].add(u.factory)
    for (emp, did), facs in by_emp_day_fabs.items():
        if len(facs) > 1:
            viols.append(("FACTORY_MIX", emp, did, sorted(facs)))

    # staffing heads per (module, op, day)
    heads = defaultdict(set)
    for u in schedule.tokens:
        if u.employee.id == 0 or u.day.id < 0 or (u.hours or 0) <= 0:
            continue
        heads[(u.module, u.op_id, u.day.id)].add(u.employee.name)
    for key, people in heads.items():
        _, op_id, _ = key
        lo, hi = _OP_MINMAX.get(op_id, (0, 10**9))
        cnt = len(people)
        if check_min and (cnt < lo):
            viols.append(("STAFF_MIN", key, cnt, lo))
        if cnt > hi:
            viols.append(("STAFF_MAX", key, cnt, hi))

    # task-hours equality
    hours_by_task = defaultdict(int)
    for u in schedule.tokens:
        if u.employee.id == 0 or u.day.id < 0 or u.hours is None:
            continue
        hours_by_task[(u.module, u.op_id)] += int(u.hours)
    for key, total in hours_by_task.items():
        req = _TASK_REQUIRED_HOURS.get(key, 0)
        if total != req:
            viols.append(("TASK_HOURS", key, total, req))

    return viols

def _print_audit(tag: str, schedule: Schedule, check_min: bool, elapsed_s: float | None = None):
    viols = _audit_hard(schedule, check_min=check_min)
    timing = f" | time: {elapsed_s:.2f}s" if elapsed_s is not None else ""
    print(f"\n=== {tag}: HARD VIOLATIONS ({len(viols)}){timing} ===")
    for v in viols[:20]:
        print(v)
    if len(viols) > 20:
        print(f"... ({len(viols) - 20} more)")

# -------------------- Build & 4-pass Solve --------------------

def solve_from_yaml(env_path: str = "EnvConfig.yaml", sched_path: str = "Schedule.yaml"):
    global _TASK_REQUIRED_HOURS, _OP_MINMAX, _TARGET_HOURS_PER_EMP, _STAFFING_MODE, _STAFFING_HARD

    env_opdef, employees, hours_union = _parse_env(env_path)
    start_day, days, windows, required_hours, op_minmax = _parse_schedule(env_opdef, sched_path)

    _TASK_REQUIRED_HOURS = dict(required_hours)
    _OP_MINMAX = dict(op_minmax)

    real_emp_count = max(1, len(employees) - 1)   # exclude dummy idx 0
    total_required_hours = sum(_TASK_REQUIRED_HOURS.values())
    _TARGET_HOURS_PER_EMP = total_required_hours / real_emp_count

    tokens = _build_tokens(windows)

    # seed planning vars with valid values (first of ranges)
    dummy_emp = employees[0]
    dummy_day = days[0]
    tokens = [UnitToken(
        id=t.id, module=t.module, factory=t.factory, phase_id=t.phase_id, phase_num=t.phase_num,
        op_id=t.op_id, start_day_id=t.start_day_id, end_day_id=t.end_day_id, allowed=t.allowed,
        employee=dummy_emp, day=dummy_day, hours=t.hours
    ) for t in tokens]

    base_problem = Schedule(
        days=days,
        employees=employees,
        hours_options=sorted(set([4] + hours_union)),
        tokens=tokens
    )

    # knobs
    PASS1_MIN, PASS2_MIN, PASS3_MIN, PASS4_MIN = 15, 5, 10, 5
    UNIMPROVED_SEC = 60

    t0 = time.time()

    # ---------- Pass 1 ----------
    _STAFFING_MODE = "max_only"; _STAFFING_HARD = True
    solver1 = _build_solver(best_limit="0hard/*medium/*soft",
                            spent_minutes=PASS1_MIN,
                            unimproved_seconds=UNIMPROVED_SEC)
    p1s = time.time()
    after1: Schedule = solver1.solve(base_problem)
    p1e = time.time()
    _print_audit("PASS 1 (max_only HARD, with best_score_limit)", after1, check_min=False, elapsed_s=(p1e - p1s))

    # ---------- Pass 2 ----------
    _STAFFING_MODE = "max_only"; _STAFFING_HARD = True
    solver2 = _build_solver(spent_minutes=PASS2_MIN, unimproved_seconds=UNIMPROVED_SEC)
    p2s = time.time()
    after2: Schedule = solver2.solve(after1)
    p2e = time.time()
    _print_audit("PASS 2 (max_only HARD, polish medium/soft)", after2, check_min=False, elapsed_s=(p2e - p2s))

    # ---------- Pass 3 ----------
    _STAFFING_MODE = "minmax"; _STAFFING_HARD = True
    solver3 = _build_solver(best_limit="0hard/*medium/*soft",
                            spent_minutes=PASS3_MIN,
                            unimproved_seconds=UNIMPROVED_SEC)
    p3s = time.time()
    after3: Schedule = solver3.solve(after2)
    p3e = time.time()
    _print_audit("PASS 3 (minmax HARD, with best_score_limit)", after3, check_min=True, elapsed_s=(p3e - p3s))

    # ---------- Pass 4 ----------
    _STAFFING_MODE = "minmax"; _STAFFING_HARD = True
    solver4 = _build_solver(spent_minutes=PASS4_MIN, unimproved_seconds=UNIMPROVED_SEC)
    p4s = time.time()
    final: Schedule = solver4.solve(after3)
    p4e = time.time()
    _print_audit("PASS 4 (minmax HARD, polish medium/soft)", final, check_min=True, elapsed_s=(p4e - p4s))

    print(f"\nTOTAL solving time: {p4e - t0:.2f}s "
          f"(P1 {p1e - p1s:.2f}s, P2 {p2e - p2s:.2f}s, P3 {p3e - p3s:.2f}s, P4 {p4e - p4s:.2f}s)")

    return final, start_day

def main():
    final, start_day = solve_from_yaml("EnvConfig.yaml", "Schedule.yaml")
    overwrite_schedule_with_assignments(final, start_day, "Schedule.yaml")

if __name__ == "__main__":
    main()
