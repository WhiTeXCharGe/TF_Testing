# employee_scheduler_by_capacity.py
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
    process_id: int      # 1..N
    task: str            # "A".."Z"
    def code(self) -> str:
        return f"P{self.process_id}-{self.task}"

@dataclass(frozen=True)
class Employee:
    id: int
    name: str
    # key: "P{process}-{task}" -> level 1..5 (used only for diversity; no hard min)
    skills: Dict[str, int]

@dataclass(frozen=True)
class TaskWindow:
    """The active window of a task (inclusive indices into the horizon)."""
    task_code: str               # e.g. "P1-A"
    process_id: int
    task_letter: str
    start_day_id: int            # inclusive
    end_day_id: int              # inclusive (deadline)
    daily_max: int               # team size (fixed across the entire window)

@planning_entity
@dataclass
class TeamSeat:
    """
    A fixed seat in the task's team.
    There are exactly 'daily_max' seats per task.
    The chosen employee occupies the seat for the entire window.
    """
    id: Annotated[int, PlanningId]
    task_code: str
    process_id: int
    task_letter: str
    start_day_id: int
    end_day_id: int
    employee: Annotated[Optional[Employee], PlanningVariable] = field(default=None)

@planning_solution
@dataclass
class Schedule:
    days:      Annotated[List[DaySlot], ProblemFactCollectionProperty, ValueRangeProvider]
    employees: Annotated[List[Employee], ProblemFactCollectionProperty, ValueRangeProvider]
    windows:   Annotated[List[TaskWindow], ProblemFactCollectionProperty, ValueRangeProvider]
    seats:     Annotated[List[TeamSeat],   PlanningEntityCollectionProperty]
    score:     Annotated[HardSoftScore, PlanningScore] = field(default=None)

# -------------------- Constraints --------------------

@constraint_provider
def define_constraints(cf: ConstraintFactory) -> List[Constraint]:
    return [
        require_employee_assigned(cf),             # HARD
        employee_not_double_booked_overlapping(cf),# HARD
        process_windows_are_sequential(cf),        # HARD (derived from YAML deadlines)

        prefer_earlier_processes(cf),              # SOFT (minor)
        level_centered_around_3(cf),               # SOFT (gentle)
        team_level_variety_per_task(cf),           # SOFT (strong)
    ]

# ---- HARD ----

def require_employee_assigned(cf: ConstraintFactory) -> Constraint:
    return (cf.for_each(TeamSeat)
            .filter(lambda s: s.employee is None)
            .penalize(HardSoftScore.ONE_HARD)
            .as_constraint("Seat must have an employee"))

def _ranges_overlap(a_start: int, a_end: int, b_start: int, b_end: int) -> bool:
    return not (a_end < b_start or b_end < a_start)

def employee_not_double_booked_overlapping(cf: ConstraintFactory) -> Constraint:
    # If the same employee is chosen in two seats whose windows overlap, forbid it.
    return (
        cf.for_each_unique_pair(
            TeamSeat,
            Joiners.equal(lambda s: s.employee)
        )
        .filter(lambda a, b:
            a.employee is not None and b.employee is not None and
            _ranges_overlap(a.start_day_id, a.end_day_id, b.start_day_id, b.end_day_id)
        )
        .penalize(HardSoftScore.ONE_HARD)
        .as_constraint("Employee double-booked on overlapping task windows")
    )

def process_windows_are_sequential(cf: ConstraintFactory) -> Constraint:
    """
    Enforce sequential processes: all seats in process p must end
    strictly before seats in process p+1 begin.
    (Windows are built from process deadlines in YAML.)
    """
    def violates(a: TeamSeat, b: TeamSeat) -> bool:
        # Compare different processes only, in order
        if a.process_id >= b.process_id:
            return False
        # require: a.end < b.start
        return not (a.end_day_id < b.start_day_id)
    return (
        cf.for_each_unique_pair(
            TeamSeat,
            Joiners.less_than(lambda s: s.process_id)
        )
        .filter(violates)
        .penalize(HardSoftScore.ONE_HARD)
        .as_constraint("Processes must not overlap")
    )

# ---- SOFT ----

def prefer_earlier_processes(cf: ConstraintFactory) -> Constraint:
    # Small nudge: earlier processes (lower process_id) slightly preferred
    return (
        cf.for_each(TeamSeat)
        .penalize(HardSoftScore.ONE_SOFT, lambda s: s.process_id - 1)
        .as_constraint("Prefer earlier processes (tiny)")
    )

def _level_for(seat: TeamSeat) -> int:
    if seat.employee is None:
        return 3
    lvl = seat.employee.skills.get(seat.task_code, None)
    return int(lvl) if isinstance(lvl, int) else 1

def level_centered_around_3(cf: ConstraintFactory) -> Constraint:
    # same gentle rule as before
    return (cf.for_each(TeamSeat)
            .penalize(HardSoftScore.ONE_SOFT, lambda s: max(0, abs(_level_for(s) - 3) // 2))
            .as_constraint("Levels near average 3 (gentle)"))

def team_level_variety_per_task(cf: ConstraintFactory) -> Constraint:
    """
    Within the same task_code (team), prefer diverse levels:
    pair penalty = 2*(3 - min(3, diff)), extra +4 if both are level 3.
    """
    def same_task(a: TeamSeat, b: TeamSeat) -> bool:
        return a.task_code == b.task_code
    def pair_penalty(a: TeamSeat, b: TeamSeat) -> int:
        la, lb = _level_for(a), _level_for(b)
        diff = abs(la - lb)
        base = max(0, 3 - min(3, diff))  # 3,2,1,0 for 0,1,2,>=3
        scaled = 2 * base
        if la == 3 and lb == 3:
            scaled += 4
        return scaled
    return (
        cf.for_each_unique_pair(TeamSeat)
        .filter(same_task)
        .penalize(HardSoftScore.ONE_SOFT, pair_penalty)
        .as_constraint("Prefer diverse levels per team")
    )

# -------------------- YAML & problem build --------------------

@dataclass
class TaskSpec:
    code: str         # "P{process}-{task}"
    daily_max: int    # team size

def parse_skill_key(code: str) -> SkillKey:
    code = code.strip().upper()
    if not code.startswith("P") or "-" not in code:
        raise ValueError(f"Bad task code: {code} (expected like 'P2-D')")
    proc = int(code[1:code.index("-")])
    task = code[code.index("-")+1:]
    return SkillKey(proc, task)

def load_config(path: str):
    """
    Supports the OLD YAML (no process_deadlines).
    - Derives each process's deadline as the MAX of its tasks' deadlines.
    - Uses optional 'precedence' to order processes; if not present, sorts process ids ascending.
    - Processes mentioned in precedence but WITHOUT tasks are ignored for window building.
    """
    with open(path, "r", encoding="utf-8") as f:
        cfg = yaml.safe_load(f)

    start_day = datetime.strptime(str(cfg["start_day"]), "%Y-%m-%d").date()
    horizon_days = int(cfg.get("horizon_days", 14))
    days = [DaySlot(i, start_day + timedelta(days=i)) for i in range(horizon_days)]

    def to_idx(datestr: str) -> int:
        d = datetime.strptime(str(datestr), "%Y-%m-%d").date()
        return max(0, min(horizon_days - 1, (d - start_day).days))

    # ---- read tasks (workload ignored; we only use daily_max and deadline) ----
    raw_tasks = cfg["tasks"]
    task_rows = []
    processes_present = set()
    for t in raw_tasks:
        code = t["code"].strip().upper()
        sk = parse_skill_key(code)
        processes_present.add(sk.process_id)
        daily_max = int(t.get("daily_max", 0))
        deadline_idx = to_idx(t["deadline"])
        task_rows.append({
            "code": code,
            "proc": sk.process_id,
            "letter": sk.task,
            "daily_max": daily_max,
            "deadline_idx": deadline_idx
        })

    # ---- derive per-process deadline as MAX of that process's task deadlines ----
    proc_deadline_idx: Dict[int, int] = {}
    for row in task_rows:
        p = row["proc"]
        proc_deadline_idx[p] = max(proc_deadline_idx.get(p, 0), row["deadline_idx"])

    # ---- decide process order (keep only processes that actually have tasks) ----
    if "precedence" in cfg and cfg["precedence"]:
        pairs = [(int(a), int(b)) for a, b in cfg["precedence"]]
        # Topo sort
        from collections import defaultdict, deque
        g = defaultdict(list)
        indeg = defaultdict(int)
        all_ps = set()  # from precedence
        for a, b in pairs:
            g[a].append(b)
            indeg[b] += 1
            all_ps.add(a); all_ps.add(b)
        q = deque(sorted([p for p in all_ps if indeg[p] == 0]))
        order = []
        while q:
            u = q.popleft()
            order.append(u)
            for v in g[u]:
                indeg[v] -= 1
                if indeg[v] == 0:
                    q.append(v)
        if not order:
            order = sorted(all_ps)

        # Keep only processes that have tasks:
        proc_ids_sorted = [p for p in order if p in processes_present]
        # Add any processes that have tasks but were not in precedence:
        for p in sorted(processes_present):
            if p not in proc_ids_sorted:
                proc_ids_sorted.append(p)
    else:
        proc_ids_sorted = sorted(processes_present)

    # ---- derive process starts: first starts at 0; each next starts after previous end
    proc_start_idx: Dict[int, int] = {}
    prev_end = -1
    for p in proc_ids_sorted:
        proc_start_idx[p] = prev_end + 1
        # if a process has no tasks (shouldn’t happen now), treat as zero-length
        prev_end = proc_deadline_idx.get(p, prev_end)

    # ---- employees ----
    emps: List[Employee] = []
    eid = 1
    for e in cfg["employees"]:
        name = str(e["name"])
        skills_dict = {}
        for k, v in (e.get("skills", {}) or {}).items():
            skills_dict[str(k).strip().upper()] = int(v)
        emps.append(Employee(eid, name, skills_dict)); eid += 1

    # ---- build windows & seats from daily_max only (ignore workload) ----
    windows: List[TaskWindow] = []
    seats: List[TeamSeat] = []
    sid = 1
    for row in task_rows:
        code, p, letter, daily_max, task_deadline = (
            row["code"], row["proc"], row["letter"], row["daily_max"], row["deadline_idx"]
        )
        # Window: from process start to min(task deadline, process deadline)
        start_idx = proc_start_idx[p]
        end_idx = min(task_deadline, proc_deadline_idx.get(p, task_deadline))
        windows.append(TaskWindow(task_code=code, process_id=p, task_letter=letter,
                                  start_day_id=start_idx, end_day_id=end_idx, daily_max=daily_max))
        for _ in range(daily_max):
            seats.append(TeamSeat(
                id=sid, task_code=code, process_id=p, task_letter=letter,
                start_day_id=start_idx, end_day_id=end_idx
            ))
            sid += 1

    # For Excel highlight
    deadline_by_task = {w.task_code: w.end_day_id for w in windows}

    return start_day, days, windows, seats, emps, deadline_by_task

# -------------------- Public API --------------------

def solve_from_config(cfg_path: str = "config_capacity.yaml"):
    start_day, days, windows, seats, employees, deadline_by_task = load_config(cfg_path)
    solver_config = SolverConfig(
        solution_class=Schedule,
        entity_class_list=[TeamSeat],
        score_director_factory_config=ScoreDirectorFactoryConfig(
            constraint_provider_function=define_constraints
        ),
        termination_config=TerminationConfig(spent_limit=Duration(seconds=60))
    )
    solver = SolverFactory.create(solver_config).build_solver()
    problem = Schedule(days=days, employees=employees, windows=windows, seats=seats)
    solution: Schedule = solver.solve(problem)
    return solution, start_day, deadline_by_task

def main():
    solution, start_day, _ = solve_from_config()
    print(f"Best score: {solution.score}")

if __name__ == "__main__":
    main()
