# schedule.py  (calendar/Gantt version)
from __future__ import annotations
from dataclasses import dataclass, field
from datetime import date, timedelta
from typing import Annotated, Optional

from timefold.solver.domain import (
    planning_entity, planning_solution,
    PlanningId, PlanningVariable,
    PlanningEntityCollectionProperty, ProblemFactCollectionProperty,
    PlanningScore, ValueRangeProvider
)
from timefold.solver import SolverFactory
from timefold.solver.config import SolverConfig, TerminationConfig, ScoreDirectorFactoryConfig, Duration
from timefold.solver.score import HardSoftScore, ConstraintFactory, Constraint, Joiners, constraint_provider

# ---------- Domain ----------

@dataclass(frozen=True)
class DaySlot:
    id: int
    d: date

@dataclass(frozen=True)
class SkillKey:
    process_id: int  # 1..4
    task: str        # "A".."D"
    def code(self) -> str: return f"P{self.process_id}-{self.task}"

@dataclass(frozen=True)
class Employee:
    id: int
    name: str
    skills: dict[str, int]  # key: "P{process}-{task}" -> level 1..5

@planning_entity
@dataclass
class RequirementSlot:
    id: Annotated[int, PlanningId]
    skill: SkillKey
    required_level: int
    # Two variables to decide: WHO and WHEN (which calendar day)
    employee: Annotated[Optional[Employee], PlanningVariable] = field(default=None)
    day:      Annotated[Optional[DaySlot],  PlanningVariable] = field(default=None)

@planning_solution
@dataclass
class Schedule:
    days:      Annotated[list[DaySlot], ProblemFactCollectionProperty, ValueRangeProvider]
    employees: Annotated[list[Employee], ProblemFactCollectionProperty, ValueRangeProvider]
    reqs:      Annotated[list[RequirementSlot], PlanningEntityCollectionProperty]
    score:     Annotated[HardSoftScore, PlanningScore] = field(default=None)

# ---------- Constraints ----------

@constraint_provider
def define_constraints(cf: ConstraintFactory) -> list[Constraint]:
    return [
        employee_not_double_booked_same_day(cf),
        required_skill_and_level(cf),
        prefer_earlier_days(cf),
        minimize_overqualification(cf),
    ]

def employee_not_double_booked_same_day(cf: ConstraintFactory) -> Constraint:
    return (
        cf.for_each_unique_pair(
            RequirementSlot,
            Joiners.equal(lambda r: r.employee),
            Joiners.equal(lambda r: r.day)
        )
        .filter(lambda a, b: a.employee is not None and b.employee is not None
                    and a.day is not None and b.day is not None)
        .penalize(HardSoftScore.ONE_HARD)
        .as_constraint("Employee double-booked on same day")
    )

def required_skill_and_level(cf: ConstraintFactory) -> Constraint:
    def lacks(r: RequirementSlot) -> bool:
        if r.employee is None:
            return True
        return r.employee.skills.get(r.skill.code(), 0) < r.required_level

    return (
        cf.for_each(RequirementSlot)
        .filter(lacks)
        .penalize(HardSoftScore.ONE_HARD)
        .as_constraint("Missing required skill or level")
    )

def prefer_earlier_days(cf: ConstraintFactory) -> Constraint:
    # Softly nudge assignments toward earlier dates in the range.
    def day_index(r: RequirementSlot) -> int:
        return r.day.id if r.day is not None else 1000
    return (
        cf.for_each(RequirementSlot)
        .penalize(HardSoftScore.ONE_SOFT, day_index)
        .as_constraint("Prefer earlier days")
    )

def minimize_overqualification(cf: ConstraintFactory) -> Constraint:
    def overq(r: RequirementSlot) -> int:
        if r.employee is None:
            return 0
        return max(0, r.employee.skills.get(r.skill.code(), 0) - r.required_level)
    return (
        cf.for_each(RequirementSlot)
        .penalize(HardSoftScore.ONE_SOFT, overq)       # <-- changed
        .as_constraint("Avoid heavy overqualification")
    )

# ---------- Mock data & problem ----------

def make_days(start: date, horizon_days: int = 14) -> list[DaySlot]:
    return [DaySlot(i, start + timedelta(days=i)) for i in range(horizon_days)]

def make_employees() -> list[Employee]:
    names = ["Ann","Beth","Chad","Dai","Emi","Faye","Gus","Hiro","Ivy","Ken","Liu","Mori"]
    tasks = [(p, t) for p in range(1,5) for t in "ABCD"]
    import random
    emps = []
    eid = 1
    for name in names:
        skills = {}
        for (p,t) in random.sample(tasks, k=8):  # ~8 skills per person
            skills[f"P{p}-{t}"] = random.randint(2,5)
        skills.setdefault("P1-A", random.randint(4,5))
        skills.setdefault("P2-B", random.randint(3,5))
        emps.append(Employee(eid, name, skills)); eid += 1
    return emps

def make_requirements() -> list[RequirementSlot]:
    reqs = []
    rid = 1
    for p in range(1,5):
        for t in "ABCD":
            # Same pattern as before; change freely per your real demand.
            if t == "A": pattern = [(4,1), (1,2)]     # need 3 people
            elif t == "B": pattern = [(3,2)]          # 2 people
            elif t == "C": pattern = [(2,1), (5,1)]   # 2 people
            else: pattern = [(1,1), (3,1), (4,1)]     # 3 people
            for level, count in pattern:
                for _ in range(count):
                    reqs.append(RequirementSlot(rid, SkillKey(p,t), level))
                    rid += 1
    return reqs

def generate_problem() -> Schedule:
    start = date(2025, 9, 1)
    days = make_days(start, horizon_days=14)           # 2-week horizon
    employees = make_employees()
    reqs = make_requirements()
    return Schedule(days=days, employees=employees, reqs=reqs)

# ---------- Solve & Gantt print ----------

def print_gantt(solution: Schedule):
    # Build a map: (task_code, day) -> list of employee initials
    from collections import defaultdict
    cell = defaultdict(list)
    code_order = []
    for p in range(1,5):
        for t in "ABCD":
            code_order.append(f"P{p}-{t}")
    for r in solution.reqs:
        if r.employee is not None and r.day is not None:
            initials = "".join([w[0] for w in r.employee.name.split()])
            cell[(r.skill.code(), r.day.id)].append(initials)

    # Header
    day_labels = [d.d.strftime("%m-%d") for d in solution.days]
    print(" " * 9 + " | " + " | ".join(day_labels))
    print("-" * (11 + len(day_labels) * 6))

    # Rows
    for code in code_order:
        row = []
        for d in solution.days:
            val = ",".join(cell.get((code, d.id), []))
            row.append(val if val else " ")
        print(f"{code:<9} | " + " | ".join(f"{c:>3}" for c in row))

def main():
    solver_config = SolverConfig(
        solution_class=Schedule,
        entity_class_list=[RequirementSlot],
        score_director_factory_config=ScoreDirectorFactoryConfig(
            constraint_provider_function=define_constraints
        ),
        termination_config=TerminationConfig(spent_limit=Duration(seconds=5))
    )
    solver = SolverFactory.create(solver_config).build_solver()

    problem = generate_problem()
    solution: Schedule = solver.solve(problem)

    print(f"\nBest score: {solution.score}\n")
    print_gantt(solution)

if __name__ == "__main__":
    main()