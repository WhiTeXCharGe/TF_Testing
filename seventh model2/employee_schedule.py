from dataclasses import dataclass, field
from datetime import date, datetime, timedelta
from typing import Annotated, Dict, List, Optional, Tuple
from collections import defaultdict
import yaml
import math
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
      choose start_day, hours, heads, days such that heads*hours*days == required_hours
      and start_day..end within the window. Phase order gets enforced by pairwise joins.
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
    allowed: List[int]
    min_heads: int
    max_heads: int

    # Planning variables (value ranges are provided globally on the solution)
    start_day: Annotated[int, PlanningVariable]  # day index
    hours:     Annotated[int, PlanningVariable]  # choice from global union; allowed checked by hard constraint
    heads:     Annotated[int, PlanningVariable]  # 1..global_max
    days:      Annotated[int, PlanningVariable]  # 1..max_window_len

@planning_solution
@dataclass
class Pass1Plan:
    # value ranges for planning variables
    day_ids: Annotated[List[int], ProblemFactCollectionProperty, ValueRangeProvider]
    hour_options: Annotated[List[int], ProblemFactCollectionProperty, ValueRangeProvider]
    head_options: Annotated[List[int], ProblemFactCollectionProperty, ValueRangeProvider]
    day_count_options: Annotated[List[int], ProblemFactCollectionProperty, ValueRangeProvider]

    # entities
    blocks: Annotated[List[BlockDecision], PlanningEntityCollectionProperty]

    score: Annotated[HardMediumSoftScore, PlanningScore] = field(default=None)

# -------------------- PASS 2: seats & seat-days --------------------

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

# -------------------- Utility --------------------

def _phase_num_from_id(pid: str) -> int:
    try:
        return int(str(pid).strip().lower().replace("p", ""))
    except Exception:
        return 0

def _skill_level(emp: Optional[Employee], op_id: str) -> int:
    if emp is None:
        return 0
    skills = emp.skills
    if skills is None:
        return 0
    v = skills.get(op_id)
    try:
        return int(v) if v is not None else 0
    except Exception:
        return 0

def _is_unassigned(emp: Optional[Employee]) -> bool:
    return (emp is None) or (emp.id == 0)

# -------------------- YAML parsers --------------------

def _parse_env(env_path: str):
    with open(env_path, "r", encoding="utf-8") as f:
        env = yaml.safe_load(f)["environment"]

    opdef: Dict[str, Dict] = {}
    for ph in env["workflow_list"][0]["phase_list"]:
        for op in ph["operation_list"]:
            op_id = op["id"]
            hrs = list(op.get("work_hours", []) or [])
            if not hrs:
                hrs = [8]
            opdef[op_id] = {
                "phase_id": ph["id"],
                "phase_num": _phase_num_from_id(ph["id"]),
                "allowed": list(sorted(hrs)),
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

# -------------------- PASS 1 constraints --------------------

def _count_skilled_by_op(employees: List[Employee]) -> Dict[str, int]:
    cnt: Dict[str, int] = defaultdict(int)
    for e in employees:
        if e.id == 0:
            continue
        skills = e.skills or {}
        for op_id, lvl in skills.items():
            try:
                if int(lvl) >= 1:
                    cnt[op_id] += 1
            except Exception:
                continue
    return cnt

@constraint_provider
def pass1_constraints(cf: ConstraintFactory) -> List[Constraint]:
    def within_window(b: BlockDecision) -> bool:
        if b.days is None or b.start_day is None:
            return False
        end = int(b.start_day) + int(b.days) - 1
        return (b.start_day >= b.window_start) and (end <= b.window_end) and (b.days >= 1)

    return [
        # domain checks
        cf.for_each(BlockDecision)
          .filter(lambda b: not within_window(b))
          .penalize(HardMediumSoftScore.ONE_HARD)
          .as_constraint("within-window"),

        cf.for_each(BlockDecision)
          .filter(lambda b: (b.hours is None) or (b.allowed is None) or (int(b.hours) not in set(int(x) for x in b.allowed)))
          .penalize(HardMediumSoftScore.ONE_HARD)
          .as_constraint("hours-allowed"),

        cf.for_each(BlockDecision)
          .filter(lambda b: (b.heads is None) or (b.heads < int(b.min_heads)) or (b.heads > int(b.max_heads)))
          .penalize(HardMediumSoftScore.ONE_HARD)
          .as_constraint("heads-in-minmax"),

        # required equality
        cf.for_each(BlockDecision)
          .filter(lambda b: (int(b.heads) * int(b.hours) * int(b.days)) != int(b.required_hours))
          .penalize(HardMediumSoftScore.ONE_HARD, lambda b: abs(int(b.heads) * int(b.hours) * int(b.days) - int(b.required_hours)))
          .as_constraint("required-hours-equality"),

        # strict phase order: every p(k) must end before any p(k+1) starts (per module)
        cf.for_each(BlockDecision)
          .join(cf.for_each(BlockDecision),
                Joiners.equal(lambda a: a.module, lambda b: b.module),
                Joiners.equal(lambda a: a.phase_num + 1, lambda b: b.phase_num))
          .filter(lambda a, b: (a.start_day + a.days - 1) >= b.start_day)
          .penalize(HardMediumSoftScore.ONE_HARD,
                    lambda a, b: (a.start_day + a.days - 1) - b.start_day + 1)
          .as_constraint("phase-order"),

        # prefer 8h, fewer days/heads, earlier start (soft)
        cf.for_each(BlockDecision)
          .penalize(HardMediumSoftScore.ONE_SOFT, lambda b: abs(int(b.hours) - 8))
          .as_constraint("prefer-8h"),
        cf.for_each(BlockDecision)
          .penalize(HardMediumSoftScore.ONE_SOFT, lambda b: int(b.days))
          .as_constraint("minimize-days"),
        cf.for_each(BlockDecision)
          .penalize(HardMediumSoftScore.ONE_SOFT, lambda b: int(b.heads))
          .as_constraint("minimize-heads"),
        cf.for_each(BlockDecision)
          .penalize(HardMediumSoftScore.ONE_SOFT, lambda b: int(b.start_day))
          .as_constraint("prefer-earlier-start"),
    ]

# -------------------- PASS 2 constraints --------------------

@constraint_provider
def pass2_constraints(cf: ConstraintFactory) -> List[Constraint]:
    return [
        # must assign and be skill-eligible
        cf.for_each(CrewSeat)
          .filter(lambda s: _is_unassigned(s.employee) or _skill_level(s.employee, s.op_id) < 1)
          .penalize(HardMediumSoftScore.ONE_HARD)
          .as_constraint("assigned+eligible-skill"),

        # 1 factory per employee per day
        cf.for_each(SeatDay)
          .join(cf.for_each(CrewSeat), Joiners.equal(lambda sd: sd.seat_key, lambda cs: cs.seat_key))
          .filter(lambda sd, cs: not _is_unassigned(cs.employee))
          .group_by(lambda sd, cs: (cs.employee.id, sd.day.id),
                    ConstraintCollectors.count_distinct(lambda sd, cs: sd.factory))
          .filter(lambda key, fac_cnt: fac_cnt > 1)
          .penalize(HardMediumSoftScore.ONE_HARD, lambda key, fac_cnt: fac_cnt - 1)
          .as_constraint("one-factory-per-emp-day"),

        # 12h hard cap
        cf.for_each(SeatDay)
          .join(cf.for_each(CrewSeat), Joiners.equal(lambda sd: sd.seat_key, lambda cs: cs.seat_key))
          .filter(lambda sd, cs: not _is_unassigned(cs.employee))
          .group_by(lambda sd, cs: (cs.employee.id, sd.day.id),
                    ConstraintCollectors.sum(lambda sd, cs: int(sd.hours)))
          .filter(lambda key, tot: tot > _DAILY_CAP)
          .penalize(HardMediumSoftScore.ONE_HARD, lambda key, tot: tot - _DAILY_CAP)
          .as_constraint("daily-cap-12h"),

        # avoid overtime over 8 (soft)
        cf.for_each(SeatDay)
          .join(cf.for_each(CrewSeat), Joiners.equal(lambda sd: sd.seat_key, lambda cs: cs.seat_key))
          .filter(lambda sd, cs: not _is_unassigned(cs.employee))
          .group_by(lambda sd, cs: (cs.employee.id, sd.day.id),
                    ConstraintCollectors.sum(lambda sd, cs: int(sd.hours)))
          .filter(lambda key, tot: tot > 8)
          .penalize(HardMediumSoftScore.ONE_SOFT, lambda key, tot: tot - 8)
          .as_constraint("avoid-overtime-over-8-soft"),

        # balance total hours
        cf.for_each(SeatDay)
          .join(cf.for_each(CrewSeat), Joiners.equal(lambda sd: sd.seat_key, lambda cs: cs.seat_key))
          .filter(lambda sd, cs: not _is_unassigned(cs.employee))
          .group_by(lambda sd, cs: cs.employee.id,
                    ConstraintCollectors.sum(lambda sd, cs: int(sd.hours)))
          .penalize(HardMediumSoftScore.ONE_SOFT,
                    lambda emp_id, tot: int(abs(int(tot) - int(_TARGET_HOURS_PER_EMP))))
          .as_constraint("balance-total-hours-soft"),
    ]

# -------------------- Build seats + seat_days from blocks --------------------

def _expand_blocks_to_seats_and_days(blocks: List[BlockDecision], days: List[DaySlot]):
    seats: List[CrewSeat] = []
    seat_days: List[SeatDay] = []
    sid = 1
    day_by_id = {d.id: d for d in days}

    for b in blocks:
        hours = int(b.hours); start = int(b.start_day); dcount = int(b.days)
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

# -------------------- Driver --------------------

def solve_from_yaml(env_path: str = "EnvConfig.yaml", sched_path: str = "Schedule.yaml"):
    global _TARGET_HOURS_PER_EMP

    env_opdef, employees = _parse_env(env_path)
    start_day, day_slots, windows, required_hours = _parse_schedule(env_opdef, sched_path)

    real_emp_count = max(1, len(employees) - 1)
    total_required_hours = sum(required_hours.values())
    _TARGET_HOURS_PER_EMP = total_required_hours / real_emp_count

    # Pass 1 problem build
    hour_union = sorted({h for w in windows for h in (w.allowed or [8])} or [8])
    max_heads_global = max((w.max_heads for w in windows), default=1)
    max_window_len = max((w.end_day_id - w.start_day_id + 1 for w in windows), default=1)
    head_options = list(range(1, max_heads_global + 1))
    day_ids = list(range(0, len(day_slots)))
    day_count_options = list(range(1, max_window_len + 1))

    blocks: List[BlockDecision] = []
    bid = 1
    for w in windows:
        baseline = 4 if list(w.allowed) == [4] else 8
        req = w.workload_days * baseline
        blocks.append(BlockDecision(
            id=bid,
            module=w.module, factory=w.factory,
            phase_id=w.phase_id, phase_num=w.phase_num,
            op_id=w.op_id,
            window_start=w.start_day_id, window_end=w.end_day_id,
            required_hours=req,
            allowed=list(w.allowed), min_heads=w.min_heads, max_heads=w.max_heads,
            start_day=w.start_day_id,   # seed
            hours=baseline,             # seed
            heads=max(w.min_heads, 1),  # seed
            days=max(1, min(req // max(1, baseline * max(w.min_heads,1)), w.end_day_id - w.start_day_id + 1))
        ))
        bid += 1

    pass1 = Pass1Plan(
        day_ids=day_ids,
        hour_options=hour_union,
        head_options=head_options,
        day_count_options=day_count_options,
        blocks=blocks
    )

    # Solve Pass 1
    t0 = time.time()
    solver1 = _build_solver(Pass1Plan, [BlockDecision], pass1_constraints,
                            best_limit="0hard/*medium/*soft", spent_minutes=2, unimproved_seconds=30)
    p1s = time.time()
    after1: Pass1Plan = solver1.solve(pass1)
    p1e = time.time()
    print(f"PASS 1 done in {p1e - p1s:.2f}s | score={after1.score}")

    # Expand to seats + seat_days
    seats, seat_days = _expand_blocks_to_seats_and_days(after1.blocks, day_slots)

    # Pass 2 problem
    plan2 = Pass2Plan(
        days=day_slots,
        employees=employees,
        seat_days=seat_days,
        seats=seats
    )

    # Solve Pass 2
    solver2 = _build_solver(Pass2Plan, [CrewSeat], pass2_constraints,
                            best_limit="0hard/*medium/*soft", spent_minutes=2, unimproved_seconds=45)
    p2s = time.time()
    final: Pass2Plan = solver2.solve(plan2)
    p2e = time.time()
    print(f"PASS 2 done in {p2e - p2s:.2f}s | score={final.score}")
    print(f"TOTAL solving time: {p2e - t0:.2f}s (P1 {p1e - p1s:.2f}s, P2 {p2e - p2s:.2f}s)")

    return final, start_day

def main():
    final, start_day = solve_from_yaml("EnvConfig.yaml", "Schedule.yaml")
    overwrite_schedule_with_assignments(final, start_day, "Schedule.yaml")

if __name__ == "__main__":
    main()
