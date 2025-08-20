# employee_scheduler.py
from __future__ import annotations
from dataclasses import dataclass, field
from datetime import date, datetime, timedelta
from typing import Annotated, Optional, Dict, List, Tuple
import yaml

from timefold.solver.domain import (
    planning_entity, planning_solution,
    PlanningId, PlanningVariable,
    PlanningEntityCollectionProperty, ProblemFactCollectionProperty,
    PlanningScore, ValueRangeProvider
)
from timefold.solver import SolverFactory
from timefold.solver.config import SolverConfig, TerminationConfig, ScoreDirectorFactoryConfig, Duration
from timefold.solver.score import HardSoftScore, ConstraintFactory, Constraint, Joiners, constraint_provider

# -------------------- Domain --------------------

@dataclass(frozen=True)
class DaySlot:
    id: int
    d: date

@dataclass(frozen=True)
class SkillKey:
    process_id: int            # 1..N
    task: str                  # "A".."Z"
    def code(self) -> str:
        return f"P{self.process_id}-{self.task}"

@dataclass(frozen=True)
class Employee:
    id: int
    name: str
    # key: "P{process}-{task}" -> level 1..5 (if missing, assume level 1 for diversity scoring)
    skills: Dict[str, int]

@dataclass(frozen=True)
class TaskDayBucket:
    """Capacity bucket: at most ONE work unit can occupy this bucket.
       Create 'daily_max' buckets per (task_code, day)."""
    id: int
    task_code: str
    day: DaySlot

@planning_entity
@dataclass
class RequirementSlot:
    """One person-day of work for a specific (process, task)."""
    id: Annotated[int, PlanningId]
    skill: SkillKey
    deadline_day_id: int  # absolute day index (0-based) relative to START_DAY
    # Planning variables:
    employee: Annotated[Optional[Employee], PlanningVariable] = field(default=None)
    bucket:   Annotated[Optional[TaskDayBucket], PlanningVariable] = field(default=None)

@planning_solution
@dataclass
class Schedule:
    days:      Annotated[List[DaySlot], ProblemFactCollectionProperty, ValueRangeProvider]
    employees: Annotated[List[Employee], ProblemFactCollectionProperty, ValueRangeProvider]
    buckets:   Annotated[List[TaskDayBucket], ProblemFactCollectionProperty, ValueRangeProvider]
    reqs:      Annotated[List[RequirementSlot], PlanningEntityCollectionProperty]
    score:     Annotated[HardSoftScore, PlanningScore] = field(default=None)

# -------------------- Constraints --------------------

@constraint_provider
def define_constraints(cf: ConstraintFactory) -> List[Constraint]:
    cons = [
        require_employee_assigned(cf),           # HARD
        require_bucket_assigned(cf),             # HARD
        bucket_matches_task(cf),                 # HARD
        bucket_capacity_enforced(cf),            # HARD
        employee_not_double_booked_same_day(cf), # HARD
        respect_deadline(cf),                    # HARD
        process_precedence_chain(cf),            # HARD (P1 before P2, etc.)

        prefer_earlier_days(cf),                 # SOFT
        level_centered_around_3(cf),             # SOFT
        level_variety_same_task_day(cf),         # SOFT
    ]
    return cons

# ---- HARD ----

def require_employee_assigned(cf: ConstraintFactory) -> Constraint:
    return (cf.for_each(RequirementSlot)
            .filter(lambda r: r.employee is None)
            .penalize(HardSoftScore.ONE_HARD)
            .as_constraint("Employee must be assigned"))

def require_bucket_assigned(cf: ConstraintFactory) -> Constraint:
    return (cf.for_each(RequirementSlot)
            .filter(lambda r: r.bucket is None)
            .penalize(HardSoftScore.ONE_HARD)
            .as_constraint("Bucket (task-day) must be assigned"))

def bucket_matches_task(cf: ConstraintFactory) -> Constraint:
    return (cf.for_each(RequirementSlot)
            .filter(lambda r: r.bucket is not None and r.bucket.task_code != r.skill.code())
            .penalize(HardSoftScore.ONE_HARD)
            .as_constraint("Bucket task mismatch"))

def bucket_capacity_enforced(cf: ConstraintFactory) -> Constraint:
    return (cf.for_each_unique_pair(RequirementSlot,
                Joiners.equal(lambda r: r.bucket))
            .filter(lambda a, b: a.bucket is not None and b.bucket is not None)
            .penalize(HardSoftScore.ONE_HARD)
            .as_constraint("Bucket capacity exceeded"))

def employee_not_double_booked_same_day(cf: ConstraintFactory) -> Constraint:
    return (cf.for_each_unique_pair(RequirementSlot,
                Joiners.equal(lambda r: r.employee),
                Joiners.equal(lambda r: r.bucket.day if r.bucket is not None else None))
            .filter(lambda a, b:
                    a.employee is not None and b.employee is not None
                    and a.bucket is not None and b.bucket is not None
                    and a.bucket.day is not None and b.bucket.day is not None)
            .penalize(HardSoftScore.ONE_HARD)
            .as_constraint("Employee double-booked same day"))

def respect_deadline(cf: ConstraintFactory) -> Constraint:
    return (cf.for_each(RequirementSlot)
            .filter(lambda r: r.bucket is None or r.bucket.day.id > r.deadline_day_id)
            .penalize(HardSoftScore.ONE_HARD)
            .as_constraint("Deadline violated"))

def process_precedence_chain(cf: ConstraintFactory) -> Constraint:
    """
    Enforce process order by day: for any pair of work units where
    a.process_id < b.process_id, b must be scheduled on a STRICTLY later day than a.
    """
    def violates(a: RequirementSlot, b: RequirementSlot) -> bool:
        if a.bucket is None or b.bucket is None: return True
        if a.bucket.day is None or b.bucket.day is None: return True
        return b.bucket.day.id <= a.bucket.day.id
    return (
        cf.for_each_unique_pair(
            RequirementSlot,
            Joiners.less_than(lambda r: r.skill.process_id)  # snake_case in Python API
        )
        .filter(violates)
        .penalize(HardSoftScore.ONE_HARD)
        .as_constraint("Process precedence increasing by day")
    )

# ---- SOFT ----

def prefer_earlier_days(cf: ConstraintFactory) -> Constraint:
    def day_index(r: RequirementSlot) -> int:
        return r.bucket.day.id if (r.bucket is not None and r.bucket.day is not None) else 1000
    return (cf.for_each(RequirementSlot)
            .penalize(HardSoftScore.ONE_SOFT, day_index)
            .as_constraint("Prefer earlier days"))

def _level_for(r: RequirementSlot) -> int:
    if r.employee is None:
        return 3  # neutral
    lvl = r.employee.skills.get(r.skill.code(), None)
    return int(lvl) if isinstance(lvl, int) else 1  # default low

def level_centered_around_3(cf: ConstraintFactory) -> Constraint:
    return (cf.for_each(RequirementSlot)
            .penalize(HardSoftScore.ONE_SOFT, lambda r: max(0, abs(_level_for(r) - 3) // 2))
            .as_constraint("Levels near average 3 (gentle)"))

def level_variety_same_task_day(cf: ConstraintFactory) -> Constraint:
    def same_task_same_day(a: RequirementSlot, b: RequirementSlot) -> bool:
        return (a.bucket is not None and b.bucket is not None
                and a.bucket.day is not None and b.bucket.day is not None
                and a.bucket.day.id == b.bucket.day.id
                and a.skill.code() == b.skill.code())
    def closeness_penalty(a: RequirementSlot, b: RequirementSlot) -> int:
        la, lb = _level_for(a), _level_for(b)
        diff = abs(la - lb)
        base = max(0, 3 - min(3, diff))  # 3,2,1,0 for diff 0,1,2,>=3
        scaled = 2 * base
        if la == 3 and lb == 3:
            scaled += 4  # discourage 3-3 pairs
        return scaled
    return (cf.for_each_unique_pair(RequirementSlot)
            .filter(same_task_same_day)
            .penalize(HardSoftScore.ONE_SOFT, closeness_penalty)
            .as_constraint("Prefer diverse levels per task-day"))

# -------------------- YAML loading & problem build --------------------

@dataclass
class TaskSpec:
    code: str            # "P{process}-{task}"
    process_id: int
    task_letter: str
    workload: int        # total person-day units to schedule
    daily_max: int       # per-day capacity for this task (number of buckets per day)
    deadline_day_id: int # absolute day index (0-based) relative to START_DAY

def parse_skill_key(code: str) -> SkillKey:
    code = code.strip().upper()
    if not code.startswith("P") or "-" not in code:
        raise ValueError(f"Bad task code: {code} (expected like 'P2-D')")
    proc = int(code[1:code.index("-")])
    task = code[code.index("-")+1:]
    return SkillKey(proc, task)

def load_config(path: str):
    with open(path, "r", encoding="utf-8") as f:
        cfg = yaml.safe_load(f)

    start_day = datetime.strptime(str(cfg["start_day"]), "%Y-%m-%d").date()
    horizon_days = int(cfg.get("horizon_days", 14))

    days = [DaySlot(i, start_day + timedelta(days=i)) for i in range(horizon_days)]

    def day_to_index(datestr: str) -> int:
        d = datetime.strptime(str(datestr), "%Y-%m-%d").date()
        delta = (d - start_day).days
        return max(0, min(horizon_days - 1, delta))

    taskspecs: List[TaskSpec] = []
    for t in cfg["tasks"]:
        code = t["code"].strip().upper()
        sk = parse_skill_key(code)
        workload = int(t["workload"])
        daily_max = int(t.get("daily_max", workload))
        deadline_id = day_to_index(t["deadline"])
        taskspecs.append(TaskSpec(code=code, process_id=sk.process_id, task_letter=sk.task,
                                  workload=workload, daily_max=daily_max, deadline_day_id=deadline_id))

    emps: List[Employee] = []
    eid = 1
    for e in cfg["employees"]:
        name = str(e["name"])
        skills_dict = {}
        for k, v in (e.get("skills", {}) or {}).items():
            skills_dict[str(k).strip().upper()] = int(v)
        emps.append(Employee(eid, name, skills_dict)); eid += 1

    deadline_by_task = {t.code: t.deadline_day_id for t in taskspecs}
    return start_day, horizon_days, taskspecs, emps, days, deadline_by_task

def build_buckets(taskspecs: List[TaskSpec], days: List[DaySlot]) -> List[TaskDayBucket]:
    buckets: List[TaskDayBucket] = []
    bid = 1
    for t in taskspecs:
        for day in days:
            for _ in range(t.daily_max):
                buckets.append(TaskDayBucket(bid, t.code, day)); bid += 1
    return buckets

def build_requirement_slots(taskspecs: List[TaskSpec]) -> List[RequirementSlot]:
    reqs: List[RequirementSlot] = []
    rid = 1
    for t in taskspecs:
        sk = SkillKey(t.process_id, t.task_letter)
        for _ in range(t.workload):
            reqs.append(RequirementSlot(
                id=rid,
                skill=sk,
                deadline_day_id=t.deadline_day_id
            ))
            rid += 1
    return reqs

# -------------------- Public API --------------------

def solve_from_config(cfg_path: str = "config.yaml"):
    """Build the problem from YAML, solve, and return (solution, start_day, deadline_by_task)."""
    start_day, horizon_days, taskspecs, employees, days, deadline_by_task = load_config(cfg_path)
    buckets = build_buckets(taskspecs, days)
    reqs = build_requirement_slots(taskspecs)

    solver_config = SolverConfig(
        solution_class=Schedule,
        entity_class_list=[RequirementSlot],
        score_director_factory_config=ScoreDirectorFactoryConfig(
            constraint_provider_function=define_constraints
        ),
        termination_config=TerminationConfig(spent_limit=Duration(seconds=6))
    )
    solver = SolverFactory.create(solver_config).build_solver()
    problem = Schedule(days=days, employees=employees, buckets=buckets, reqs=reqs)
    solution: Schedule = solver.solve(problem)
    return solution, start_day, deadline_by_task

# -------------------- CLI (optional) --------------------

def main():
    solution, start_day, _ = solve_from_config("config.yaml")
    print(f"Best score: {solution.score}")

if __name__ == "__main__":
    main()
