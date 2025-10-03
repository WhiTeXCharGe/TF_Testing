# employee_schedule.py
# Pass 1 (Timefold): choose crew blocks inside each window:
#   (start_day, heads ∈ [min,max], days); HOURS ARE AUTO-DERIVED per op from EnvConfig.
#   - allow overfill (produced >= required), but forbid underfill
#   - cap overfill to at most one extra day (<= heads*hours)
#   - daily head capacity per op_id (can't exceed #qualified employees)
#   - soft priority: |hours-8| (highest), then smaller hours, then smaller heads, then fewer days, then earlier start
# Pass 2 (Timefold): assign one employee per seat (skill-eligible, daily cap 12h, one factory/day, soft balance)
# Writes results back through export_schedule.overwrite_schedule_with_assignments.

from dataclasses import dataclass, field
from datetime import date, datetime, timedelta
from typing import Annotated, Dict, List, Optional, Tuple
from collections import defaultdict
import yaml
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


# -------------------- Common problem facts --------------------

@dataclass(frozen=True)
class DaySlot:
    id: int
    d: date

@dataclass(frozen=True)
class Employee:
    id: int           # internal numeric id for solver
    wid: str          # "w#": used for writing back
    name: str
    skills: Dict[str, int]

@dataclass(frozen=True)
class TaskWindow:
    module: str
    factory: str
    phase_id: str
    phase_num: int
    op_id: str
    start_day_id: int       # inclusive
    end_day_id: int         # inclusive
    allowed: List[int]      # allowed hours for this op
    min_heads: int
    max_heads: int
    workload_days: int      # from Schedule.yaml


# -------------------- PASS 1: blocks (Timefold) --------------------

@planning_entity
@dataclass
class BlockDecision:
    """
    One decision for a single operation window:
      choose (start_day, heads, days); hours are derived from 'allowed'.
    Hard rules:
      - within window
      - hours in allowed (derived)
      - heads in [min,max]
      - daily head capacity by op_id cannot be exceeded
      - phase order across module
      - produced = heads*hours*days >= required_hours (no underfill)
      - produced - required_hours <= heads*hours (overfill ≤ one extra day)
    """
    id: Annotated[int, PlanningId]

    # Facts
    module: str
    factory: str
    phase_id: str
    phase_num: int
    op_id: str
    window_start: int
    window_end: int
    required_hours: int
    allowed: List[int]      # e.g. [8, 10, 12]
    min_heads: int
    max_heads: int

    # Planning variables (hours removed)
    start_day: Annotated[int, PlanningVariable(value_range_provider_refs=["vr_day_ids"])]
    heads:     Annotated[int, PlanningVariable(value_range_provider_refs=["vr_head_options"])]
    days:      Annotated[int, PlanningVariable(value_range_provider_refs=["vr_day_count_options"])]

    # Optional: seed/logging only (not used by constraints)
    seed_hours: int = 8


@planning_solution
@dataclass
class Pass1Plan:
    # value ranges for planning variables
    day_ids: Annotated[
        List[int],
        ProblemFactCollectionProperty,
        ValueRangeProvider(id="vr_day_ids")
    ]
    head_options: Annotated[
        List[int],
        ProblemFactCollectionProperty,
        ValueRangeProvider(id="vr_head_options")
    ]
    day_count_options: Annotated[
        List[int],
        ProblemFactCollectionProperty,
        ValueRangeProvider(id="vr_day_count_options")
    ]

    # problem facts (for capacity by day)
    day_slots: Annotated[
        List[DaySlot],
        ProblemFactCollectionProperty
    ]

    # entities
    blocks: Annotated[List[BlockDecision], PlanningEntityCollectionProperty]

    score: Annotated[HardMediumSoftScore, PlanningScore] = field(default=None)


# -------------------- PASS 2: seats & seat-days (Timefold assigns employees) --------------------

@dataclass(frozen=True)
class SeatDay:
    seat_key: str
    day: DaySlot
    hours: int
    factory: str

@planning_entity
@dataclass
class CrewSeat:
    """
    One fixed seat for the whole block; Timefold assigns exactly one employee to it.
    """
    id: Annotated[int, PlanningId]

    # Facts
    module: str
    factory: str
    phase_id: str
    phase_num: int
    op_id: str
    start_day_id: int
    days: int
    hours: int
    seat_index: int
    seat_key: str

    # Planning variable
    employee: Annotated[Employee, PlanningVariable]


@planning_solution
@dataclass
class Pass2Plan:
    # value ranges
    days: Annotated[List[DaySlot], ProblemFactCollectionProperty, ValueRangeProvider]
    employees: Annotated[List[Employee], ProblemFactCollectionProperty, ValueRangeProvider]

    # facts
    seat_days: Annotated[List[SeatDay], ProblemFactCollectionProperty]

    # entities
    seats: Annotated[List[CrewSeat], PlanningEntityCollectionProperty]

    score: Annotated[HardMediumSoftScore, PlanningScore] = field(default=None)


# -------------------- Globals --------------------

_DAILY_CAP: int = 12
_TARGET_HOURS_PER_EMP: float = 0.0

# capacity of qualified workers per op_id (computed from employees' skills)
_OP_CAPACITY_BY_OP: Dict[str, int] = {}

def _capacity_for_op(op_id: str) -> int:
    return _OP_CAPACITY_BY_OP.get(op_id, 999_999)


# -------------------- Utility --------------------

def _phase_num_from_id(pid: str) -> int:
    try:
        return int(str(pid).strip().lower().replace("p", ""))
    except Exception:
        return 0

def _skill_level(emp: Optional[Employee], op_id: str) -> int:
    if emp is None:
        return 0
    try:
        skills = getattr(emp, "skills", None)
        if skills is None:
            return 0
        v = skills.get(op_id, 0)
        return int(v) if v is not None else 0
    except Exception:
        return 0

def _is_unassigned(emp: Optional[Employee]) -> bool:
    if emp is None:
        return True
    try:
        return int(getattr(emp, "id", 0)) == 0
    except Exception:
        return True


# -------------------- YAML parsers --------------------

def _parse_env(env_path: str):
    with open(env_path, "r", encoding="utf-8") as f:
        env = yaml.safe_load(f)["environment"]

    opdef: Dict[str, Dict] = {}
    for ph in env["workflow_list"][0]["phase_list"]:
        for op in ph["operation_list"]:
            op_id = op["id"]
            hrs = list(op.get("work_hours", []) or [])
            if len(hrs) == 0:
                hrs = [8]
            opdef[op_id] = {
                "phase_id": ph["id"],
                "phase_num": _phase_num_from_id(ph["id"]),
                "allowed": list(sorted(int(x) for x in hrs)),
                "min": int(op.get("min_worker_num", 1)),
                "max": int(op.get("max_worker_num", 999999)),
            }

    # employees
    employees: List[Employee] = [Employee(id=0, wid="__UNASSIGNED__", name="__UNASSIGNED__", skills={})]
    eid = 1
    for w in env["worker_list"]:
        wid = str(w.get("id"))
        name = w.get("name")
        skills = dict(w.get("skill_map") or {})
        employees.append(Employee(id=eid, wid=wid, name=name, skills=skills))
        eid += 1

    return opdef, employees

def _parse_schedule(env_opdef: Dict[str, Dict], sched_path: str):
    with open(sched_path, "r", encoding="utf-8") as f:
        s = yaml.safe_load(f)["schedule"]

    start_date = datetime.strptime(s["plan_range"]["start_date"], "%Y/%m/%d").date()
    end_date   = datetime.strptime(s["plan_range"]["end_date"], "%Y/%m/%d").date()
    horizon_days = (end_date - start_date).days + 1

    days: List[DaySlot] = [DaySlot(i, start_date + timedelta(days=i)) for i in range(horizon_days)]

    windows: List[TaskWindow] = []
    required_hours: Dict[Tuple[str, str], int] = {}  # (module, op_id) -> baseline hours

    for wf in s["workflow_task_list"]:
        module = wf["id"]
        fab = wf.get("fab")
        for ph_task in wf["phase_task_list"]:
            phase_id = ph_task["phase"]
            phase_num = _phase_num_from_id(phase_id)
            p_start = datetime.strptime(ph_task["start_date"], "%Y/%m/%d").date()
            p_end   = datetime.strptime(ph_task["end_date"], "%Y/%m/%d").date()
            start_id = (p_start - start_date).days
            end_id   = (p_end   - start_date).days

            for ot in ph_task["operation_task_list"]:
                op_id = ot["operation"]
                workload_days = int(ot["workload_days"])

                ocfg = env_opdef.get(op_id)
                if ocfg is None:
                    raise ValueError(f"operation {op_id} not present in EnvConfig.workflow_list")
                allowed = ocfg["allowed"]
                min_h = ocfg["min"]; max_h = ocfg["max"]

                # required baseline: workload_days × baseline (4 if only [4], else 8)
                baseline = 4 if (list(allowed) == [4]) else 8
                req_hours = workload_days * baseline
                required_hours[(module, op_id)] = required_hours.get((module, op_id), 0) + req_hours

                windows.append(TaskWindow(
                    module=module, factory=fab, phase_id=phase_id, phase_num=phase_num,
                    op_id=op_id, start_day_id=start_id, end_day_id=end_id,
                    allowed=list(allowed), min_heads=min_h, max_heads=max_h,
                    workload_days=workload_days
                ))

    return start_date, days, windows, required_hours


# ---------- HOURS helpers (centralized; avoid list truthiness in JPy) ----------

def _allowed_list_from_block(b: BlockDecision) -> List[int]:
    """Return a non-empty list of allowed hours without using list truthiness."""
    al = getattr(b, "allowed", None)
    if al is None:
        return [8]
    return [int(x) for x in al] if len(al) > 0 else [8]

def _safe_allowed_list(b: BlockDecision) -> List[int]:
    return _allowed_list_from_block(b)

def _auto_hours(b: BlockDecision) -> int:
    """
    Deterministically choose hours from b.allowed based on current heads & days:
      - pick the smallest h in allowed such that heads*h*days >= required_hours
      - and (heads*h*days - required_hours) <= heads*h  (overfill ≤ one extra day)
    If none satisfy both, pick the least-bad:
      1) minimize missing hours if produced < required
      2) else minimize (overfill - heads*h)
    Ties: prefer |h-8| small, then smaller h.
    """
    allowed = sorted(_allowed_list_from_block(b))   # SAFE
    H = max(1, int(b.heads) if b.heads is not None else 1)
    D = max(1, int(b.days)  if b.days  is not None else 1)
    R = int(b.required_hours)

    feasible = []
    for h in allowed:
        prod = H * h * D
        over = prod - R
        if prod >= R and over <= H * h:
            feasible.append((h, prod, over))

    if len(feasible) > 0:
        feasible.sort(key=lambda t: (abs(t[0] - 8), t[0]))  # prefer near-8 then smaller
        return feasible[0][0]

    best_h = allowed[0]
    best_key = None
    for h in allowed:
        prod = H * h * D
        if prod < R:
            deficit = R - prod
            key = (0, deficit, abs(h - 8), h)
        else:
            over = prod - R
            extra = max(0, over - H * h)
            key = (1, extra, abs(h - 8), h)
        if (best_key is None) or (key < best_key):
            best_h = h
            best_key = key
    return best_h

def _produced(b: BlockDecision) -> int:
    return int(b.heads) * _auto_hours(b) * int(b.days)


# ---------- PASS 1 CONSTRAINTS ----------
def p1_within_window(cf: ConstraintFactory) -> Constraint:
    def within_window(b: BlockDecision) -> bool:
        if (b.days is None) or (b.start_day is None):
            return False
        sd = int(b.start_day)
        d  = int(b.days)
        ws = int(b.window_start)
        we = int(b.window_end)
        end = sd + d - 1
        return (sd >= ws) and (end <= we) and (d >= 1)

    return (
        cf.for_each(BlockDecision)
          .filter(lambda b: not within_window(b))
          .penalize(HardMediumSoftScore.ONE_HARD)
          .as_constraint("p1-within-window")
    )

def p1_days_not_exceed_window_len(cf: ConstraintFactory) -> Constraint:
    # Additional hard guardrail to shrink the search space:
    # force days <= (window_end - window_start + 1) for EACH block.
    return (
        cf.for_each(BlockDecision)
          .filter(lambda b: int(b.days) > (int(b.window_end) - int(b.window_start) + 1))
          .penalize(
              HardMediumSoftScore.ONE_HARD,
              lambda b: int(b.days) - (int(b.window_end) - int(b.window_start) + 1)
          )
          .as_constraint("p1-days-within-window-length")
    )

def p1_hours_value_in_allowed(cf: ConstraintFactory) -> Constraint:
    return (
        cf.for_each(BlockDecision)
          .filter(lambda b: _auto_hours(b) not in _safe_allowed_list(b))
          .penalize(HardMediumSoftScore.ONE_HARD)
          .as_constraint("p1-hours-in-allowed")
    )

def p1_heads_in_minmax(cf: ConstraintFactory) -> Constraint:
    return (
        cf.for_each(BlockDecision)
          .filter(lambda b: (b.heads is None) or (int(b.heads) < int(b.min_heads)) or (int(b.heads) > int(b.max_heads)))
          .penalize(HardMediumSoftScore.ONE_HARD)
          .as_constraint("p1-heads-in-minmax")
    )

def p1_no_underfill(cf: ConstraintFactory) -> Constraint:
    return (
        cf.for_each(BlockDecision)
          .filter(lambda b: _produced(b) < int(b.required_hours))
          .penalize(
              HardMediumSoftScore.ONE_HARD,
              lambda b: int(b.required_hours) - _produced(b)
          )
          .as_constraint("p1-no-underfill")
    )

def p1_overfill_at_most_one_day(cf: ConstraintFactory) -> Constraint:
    return (
        cf.for_each(BlockDecision)
          .filter(lambda b: (_produced(b) - int(b.required_hours)) > (int(b.heads) * _auto_hours(b)))
          .penalize(
              HardMediumSoftScore.ONE_HARD,
              lambda b: (_produced(b) - int(b.required_hours)) - (int(b.heads) * _auto_hours(b))
          )
          .as_constraint("p1-overfill-at-most-one-day")
    )

def p1_phase_order(cf: ConstraintFactory) -> Constraint:
    # strict: every p(k) must end before any p(k+1) starts (per module)
    return (
        cf.for_each(BlockDecision)
          .join(
              cf.for_each(BlockDecision),
              Joiners.equal(lambda a: a.module,    lambda b: b.module),
              Joiners.equal(lambda a: a.phase_num + 1, lambda b: b.phase_num),
          )
          .filter(lambda a, b: (int(a.start_day) + int(a.days) - 1) >= int(b.start_day))
          .penalize(
              HardMediumSoftScore.ONE_HARD,
              lambda a, b: (int(a.start_day) + int(a.days) - 1) - int(b.start_day) + 1
          )
          .as_constraint("p1-phase-order")
    )

def p1_daily_head_capacity(cf: ConstraintFactory) -> Constraint:
    """
    For each day and each op_id, the sum of heads of all blocks active on that day
    must not exceed the number of qualified employees for that op.
    """
    # For each day slot, join with blocks active on that day
    return (
        cf.for_each(DaySlot)
          .join(
              cf.for_each(BlockDecision),
              # Join on "block active on this day"
              Joiners.filtering(lambda d, b:
                  (int(b.start_day) <= int(d.id) <= (int(b.start_day) + int(b.days) - 1))
              )
          )
          # group by (day_id, op_id) and sum heads
          .group_by(
              lambda d, b: (int(d.id), b.op_id),
              ConstraintCollectors.sum(lambda d, b: int(b.heads))
          )
          # if demand > capacity(op), penalize the overflow as HARD
          .filter(lambda key, total_heads: total_heads > _capacity_for_op(key[1]))
          .penalize(
              HardMediumSoftScore.ONE_HARD,
              lambda key, total_heads: total_heads - _capacity_for_op(key[1])
          )
          .as_constraint("p1-daily-head-capacity-by-op")
    )

# Soft priorities (bigger multipliers = higher priority)
PREF_HOURS_WEIGHT   = 1000  # |hours-8| is highest priority
SMALLER_HOURS_W     = 100   # prefer smaller hours (after closeness to 8)
SMALLER_HEADS_W     = 10    # prefer smaller heads next
FEWER_DAYS_W        = 1     # then fewer days
EARLIER_START_W     = 1     # then earlier start (lowest)
STACK_PAIR_WEIGHT   = 2
def p1_med_penalize_stack_by_op(cf: ConstraintFactory) -> Constraint:
    """
    Medium penalty that discourages stacking many blocks of the same op_id on the same day.
    For each (day, op_id), let n = # of blocks active that day. We penalize C(n, 2) * STACK_PAIR_WEIGHT.
    This is convex in n and matches the example: n=3 -> 3 pairs -> 3*2 = 6 penalty.
    """
    return (
        cf.for_each(DaySlot)
          .join(
              cf.for_each(BlockDecision),
              # block active on this day
              Joiners.filtering(lambda d, b: int(b.start_day) <= int(d.id) <= (int(b.start_day) + int(b.days) - 1))
          )
          .group_by(
              lambda d, b: (int(d.id), b.op_id),
              ConstraintCollectors.count()
          )
          .filter(lambda key, cnt: cnt > 1)
          .penalize(
              HardMediumSoftScore.ONE_MEDIUM,
              # C(n,2) = n*(n-1)/2, then scale by STACK_PAIR_WEIGHT
              lambda key, cnt: STACK_PAIR_WEIGHT * (int(cnt) * (int(cnt) - 1) // 2)
          )
          .as_constraint("p1-med-penalize-stack-by-op")
    )

def p1_soft_prefer_hours_near_8(cf: ConstraintFactory) -> Constraint:
    return (
        cf.for_each(BlockDecision)
          .penalize(HardMediumSoftScore.ONE_SOFT, lambda b: PREF_HOURS_WEIGHT * abs(_auto_hours(b) - 8))
          .as_constraint("p1-soft-prefer-hours-near-8")
    )

def p1_soft_prefer_smaller_hours(cf: ConstraintFactory) -> Constraint:
    return (
        cf.for_each(BlockDecision)
          .penalize(HardMediumSoftScore.ONE_SOFT, lambda b: SMALLER_HOURS_W * _auto_hours(b))
          .as_constraint("p1-soft-prefer-smaller-hours")
    )

def p1_soft_minimize_heads(cf: ConstraintFactory) -> Constraint:
    return (
        cf.for_each(BlockDecision)
          .penalize(HardMediumSoftScore.ONE_SOFT, lambda b: SMALLER_HEADS_W * int(b.heads))
          .as_constraint("p1-soft-minimize-heads")
    )

def p1_soft_minimize_days(cf: ConstraintFactory) -> Constraint:
    return (
        cf.for_each(BlockDecision)
          .penalize(HardMediumSoftScore.ONE_SOFT, lambda b: FEWER_DAYS_W * int(b.days))
          .as_constraint("p1-soft-minimize-days")
    )

def p1_soft_prefer_earlier_start(cf: ConstraintFactory) -> Constraint:
    return (
        cf.for_each(BlockDecision)
          .penalize(HardMediumSoftScore.ONE_SOFT, lambda b: EARLIER_START_W * int(b.start_day))
          .as_constraint("p1-soft-prefer-earlier-start")
    )

@constraint_provider
def pass1_constraints(cf: ConstraintFactory) -> List[Constraint]:
    return [
        p1_within_window(cf),
        p1_days_not_exceed_window_len(cf),
        p1_hours_value_in_allowed(cf),
        p1_heads_in_minmax(cf),
        p1_no_underfill(cf),
        p1_overfill_at_most_one_day(cf),
        p1_phase_order(cf),
        p1_daily_head_capacity(cf),
        
        #medium
        p1_med_penalize_stack_by_op(cf),
        # Soft priority order:
        p1_soft_prefer_hours_near_8(cf),
        p1_soft_prefer_smaller_hours(cf),
        p1_soft_minimize_heads(cf),
        p1_soft_minimize_days(cf),
        p1_soft_prefer_earlier_start(cf),
    ]


# -------------------- PASS 2 constraints (employee assignment) --------------------

def p2_assigned_and_skill(cf: ConstraintFactory) -> Constraint:
    return (
        cf.for_each(CrewSeat)
          .filter(lambda s: _is_unassigned(s.employee) or _skill_level(s.employee, s.op_id) < 1)
          .penalize(HardMediumSoftScore.ONE_HARD)
          .as_constraint("p2-assigned+eligible-skill")
    )

def p2_one_factory_per_emp_day(cf: ConstraintFactory) -> Constraint:
    return (
        cf.for_each(SeatDay)
          .join(cf.for_each(CrewSeat), Joiners.equal(lambda sd: sd.seat_key, lambda cs: cs.seat_key))
          .filter(lambda sd, cs: not _is_unassigned(cs.employee))
          .group_by(lambda sd, cs: (cs.employee.id, sd.day.id),
                    ConstraintCollectors.count_distinct(lambda sd, cs: sd.factory))
          .filter(lambda key, fac_cnt: fac_cnt > 1)
          .penalize(HardMediumSoftScore.ONE_HARD, lambda key, fac_cnt: fac_cnt - 1)
          .as_constraint("p2-one-factory-per-emp-day")
    )

def p2_daily_cap_12h(cf: ConstraintFactory) -> Constraint:
    return (
        cf.for_each(SeatDay)
          .join(cf.for_each(CrewSeat), Joiners.equal(lambda sd: sd.seat_key, lambda cs: cs.seat_key))
          .filter(lambda sd, cs: not _is_unassigned(cs.employee))
          .group_by(lambda sd, cs: (cs.employee.id, sd.day.id),
                    ConstraintCollectors.sum(lambda sd, cs: int(sd.hours)))
          .filter(lambda key, tot: tot > _DAILY_CAP)
          .penalize(HardMediumSoftScore.ONE_HARD, lambda key, tot: tot - _DAILY_CAP)
          .as_constraint("p2-daily-cap-12h")
    )

def p2_soft_avoid_overtime_over8(cf: ConstraintFactory) -> Constraint:
    return (
        cf.for_each(SeatDay)
          .join(cf.for_each(CrewSeat), Joiners.equal(lambda sd: sd.seat_key, lambda cs: cs.seat_key))
          .filter(lambda sd, cs: not _is_unassigned(cs.employee))
          .group_by(lambda sd, cs: (cs.employee.id, sd.day.id),
                    ConstraintCollectors.sum(lambda sd, cs: int(sd.hours)))
          .filter(lambda key, tot: tot > 8)
          .penalize(HardMediumSoftScore.ONE_SOFT, lambda key, tot: tot - 8)
          .as_constraint("p2-soft-avoid-overtime-over-8")
    )

def p2_soft_balance_total_hours(cf: ConstraintFactory) -> Constraint:
    return (
        cf.for_each(SeatDay)
          .join(cf.for_each(CrewSeat), Joiners.equal(lambda sd: sd.seat_key, lambda cs: cs.seat_key))
          .filter(lambda sd, cs: not _is_unassigned(cs.employee))
          .group_by(lambda sd, cs: cs.employee.id,
                    ConstraintCollectors.sum(lambda sd, cs: int(sd.hours)))
          .penalize(HardMediumSoftScore.ONE_SOFT,
                    lambda emp_id, tot: int(abs(int(tot) - int(_TARGET_HOURS_PER_EMP))))
          .as_constraint("p2-soft-balance-total-hours")
    )

@constraint_provider
def pass2_constraints(cf: ConstraintFactory) -> List[Constraint]:
    return [
        p2_assigned_and_skill(cf),
        p2_one_factory_per_emp_day(cf),
        p2_daily_cap_12h(cf),
        p2_soft_avoid_overtime_over8(cf),
        p2_soft_balance_total_hours(cf),
    ]


# -------------------- Build seats + seat_days from blocks --------------------

def _expand_blocks_to_seats_and_days(blocks: List[BlockDecision], days: List[DaySlot]):
    seats: List[CrewSeat] = []
    seat_days: List[SeatDay] = []
    sid = 1
    day_by_id = {d.id: d for d in days}

    for b in blocks:
        hours = _auto_hours(b)      # hours derived from allowed + heads/days/required
        start = int(b.start_day)
        dcount = int(b.days)
        for sidx in range(int(b.heads)):
            seat_key = f"{b.module}|{b.op_id}|s{str(sidx).zfill(4)}|d{start}"
            seats.append(CrewSeat(
                id=sid,
                module=b.module, factory=b.factory,
                phase_id=b.phase_id, phase_num=b.phase_num,
                op_id=b.op_id,
                start_day_id=start,
                days=dcount, hours=hours,
                seat_index=sidx,
                seat_key=seat_key,
                employee=Employee(0, "__UNASSIGNED__", "__UNASSIGNED__", {})
            ))
            sid += 1
            for off in range(dcount):
                did = start + off
                if did in day_by_id:
                    seat_days.append(SeatDay(
                        seat_key=seat_key,
                        day=day_by_id[did],
                        hours=hours,
                        factory=b.factory
                    ))
    return seats, seat_days


# -------------------- Solvers --------------------

def _build_solver(solution_cls, entity_cls_list, constraint_fn,
                  best_limit: str | None = None,
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
        solution_class=solution_cls,
        entity_class_list=entity_cls_list,
        score_director_factory_config=ScoreDirectorFactoryConfig(
            constraint_provider_function=constraint_fn
        ),
        termination_config=TerminationConfig(**term_kwargs)
    )
    return SolverFactory.create(cfg).build_solver()


# -------------------- Build Pass 1 model & solve --------------------

def _build_pass1_and_solve(day_slots: List[DaySlot], windows: List[TaskWindow],
                           required_hours: Dict[Tuple[str, str], int]) -> List[BlockDecision]:
    max_heads_global = max((w.max_heads for w in windows), default=1)
    min_heads_global = min((w.min_heads for w in windows), default=1)
    max_window_len = max((w.end_day_id - w.start_day_id + 1 for w in windows), default=1)

    head_options = list(range(min_heads_global, max_heads_global + 1))
    day_ids = list(range(0, len(day_slots)))
    day_count_options = list(range(1, max_window_len + 1))  # global; per-block capped by hard constraint

    blocks: List[BlockDecision] = []
    bid = 1
    for w in windows:
        baseline = 4 if list(w.allowed) == [4] else 8
        req = w.workload_days * baseline

        # SAFE seed (no list truthiness)
        seed_hours = int(w.allowed[0]) if len(w.allowed) > 0 else baseline
        min_heads = max(w.min_heads, 1)
        max_days = w.end_day_id - w.start_day_id + 1

        safe_den = max(1, seed_hours * min_heads)
        seed_days = max(1, min((req + safe_den - 1) // safe_den, max_days))  # ceil-div

        blocks.append(BlockDecision(
            id=bid,
            module=w.module, factory=w.factory,
            phase_id=w.phase_id, phase_num=w.phase_num,
            op_id=w.op_id,
            window_start=w.start_day_id, window_end=w.end_day_id,
            required_hours=req,
            allowed=list(w.allowed),
            min_heads=w.min_heads, max_heads=w.max_heads,
            start_day=w.start_day_id,
            heads=min_heads,
            days=seed_days,
            seed_hours=seed_hours
        ))
        bid += 1

    pass1 = Pass1Plan(
        day_ids=day_ids,
        head_options=head_options,
        day_count_options=day_count_options,
        day_slots=day_slots,
        blocks=blocks
    )

    solver1 = _build_solver(Pass1Plan, [BlockDecision], pass1_constraints,
                            best_limit="0hard/*medium/*soft", spent_minutes=1, unimproved_seconds=60)
    t1s = time.time()
    solved: Pass1Plan = solver1.solve(pass1)
    t1e = time.time()
    print(f"PASS 1 done in {t1e - t1s:.2f}s | score={solved.score}")
    return solved.blocks


# -------------------- Driver --------------------

def solve_from_yaml(env_path: str = "EnvConfig.yaml", sched_path: str = "Schedule.yaml"):
    global _TARGET_HOURS_PER_EMP, _OP_CAPACITY_BY_OP

    env_opdef, employees = _parse_env(env_path)

    # Build op-level capacity from employee skills (how many are qualified for each op)
    cap: Dict[str, int] = {op_id: 0 for op_id in env_opdef.keys()}
    for e in employees:
        if e.id == 0:  # skip the __UNASSIGNED__ placeholder
            continue
        skills = e.skills or {}
        for op_id in cap.keys():
            try:
                if int(skills.get(op_id, 0) or 0) > 0:
                    cap[op_id] += 1
            except Exception:
                pass
    _OP_CAPACITY_BY_OP = cap

    start_day, day_slots, windows, required_hours = _parse_schedule(env_opdef, sched_path)

    # for Pass 2 balancing soft
    real_emp_count = max(1, len(employees) - 1)
    total_required_hours = sum(required_hours.values())
    _TARGET_HOURS_PER_EMP = total_required_hours / real_emp_count

    # ---- Pass 1 ----
    t0 = time.time()
    decided_blocks = _build_pass1_and_solve(day_slots, windows, required_hours)

    # ---- Expand to seats + seat_days ----
    seats, seat_days = _expand_blocks_to_seats_and_days(decided_blocks, day_slots)

    # ---- Pass 2 problem ----
    plan2 = Pass2Plan(
        days=day_slots,
        employees=employees,   # includes id=0 placeholder; hard constraint prevents leaving unassigned
        seat_days=seat_days,
        seats=seats
    )

    # ---- Solve Pass 2 ----
    solver2 = _build_solver(Pass2Plan, [CrewSeat], pass2_constraints,
                            best_limit="0hard/*medium/*soft", spent_minutes=1, unimproved_seconds=60)
    p2s = time.time()
    final: Pass2Plan = solver2.solve(plan2)
    p2e = time.time()
    print(f"PASS 2 done in {p2e - p2s:.2f}s | score={final.score}")
    print(f"TOTAL solving time: {p2e - t0:.2f}s")

    return final, start_day


def main():
    final, start_day = solve_from_yaml("EnvConfig.yaml", "Schedule.yaml")
    # export_schedule expects (seats + seat_days); it reconstructs per-day work_date_list.
    overwrite_schedule_with_assignments(final, start_day, "Schedule.yaml")


if __name__ == "__main__":
    main()
