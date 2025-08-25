# employee_scheduler_fourth.py
from __future__ import annotations
from dataclasses import dataclass, field
from datetime import date, datetime, timedelta
from typing import Annotated, Optional, Dict, List, Tuple, Set   # <<< CHANGED (added Set)
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
class Employee:
    id: int
    name: str
    # key: "P{process}-{task}" -> level 1..5 (must exist to be eligible)
    skills: Dict[str, int]
    unavailable_day_ids: Set[int] = field(default_factory=set)  # <<< NEW (per-employee blackout by day.id)

@dataclass(frozen=True)
class TaskWindow:
    """Task window within a module/process: allowed day range for that task."""
    module: str
    process_id: int
    task_letter: str
    start_day_id: int    # inclusive
    end_day_id: int      # inclusive
    workload: int        # total person-days to schedule in this window

    def pcode(self) -> str:
        # The skill key used in employee.skills (no module prefix)
        return f"P{self.process_id}-{self.task_letter}"

    def tcode(self) -> str:
        # Full code including module (for exports): S1-P1-A
        return f"{self.module}-P{self.process_id}-{self.task_letter}"

@planning_entity
@dataclass
class RequirementSlot:
    """One person-day of work for a specific (module, process, task)."""
    id: Annotated[int, PlanningId]
    module: str
    process_id: int
    task_letter: str
    start_day_id: int
    end_day_id: int
    # Planning vars:
    employee: Annotated[Optional[Employee], PlanningVariable] = field(default=None)
    day:      Annotated[Optional[DaySlot],  PlanningVariable] = field(default=None)

    # helpers
    def pcode(self) -> str: return f"P{self.process_id}-{self.task_letter}"
    def tcode(self) -> str: return f"{self.module}-P{self.process_id}-{self.task_letter}"

@planning_solution
@dataclass
class Schedule:
    days:      Annotated[List[DaySlot], ProblemFactCollectionProperty, ValueRangeProvider]
    employees: Annotated[List[Employee], ProblemFactCollectionProperty, ValueRangeProvider]
    reqs:      Annotated[List[RequirementSlot], PlanningEntityCollectionProperty]
    score:     Annotated[HardSoftScore, PlanningScore] = field(default=None)

# -------------------- Config knobs / globals --------------------

_MAX_WORKER_PER_DAY: Optional[int] = None
_BUSINESS_DAYS = {0, 1, 2, 3, 4}  # <<< NEW (Mon–Fri only; 0=Mon, 6=Sun)

# -------------------- Constraints --------------------

@constraint_provider
def define_constraints(cf: ConstraintFactory) -> List[Constraint]:
    return [
        # HARD
        require_employee_assigned(cf),
        require_day_assigned(cf),
        day_within_window(cf),
        skill_must_exist(cf),
        employee_daily_capacity_limit(cf),   # uses maximum_worker
        employee_not_double_booked_same_day(cf),
        process_precedence_within_module(cf),

        employee_not_on_unavailable_day(cf),  # <<< NEW
        no_weekend_work(cf),                  # <<< NEW

        # SOFT — finish ASAP and human-friendly
        finish_asap(cf),                       # strong early-day bias
        gently_prefer_levels_near_3(cf),       # tiny nudge only; won’t block staffing
        prefer_continuity(cf),                 # <<< NEW
        prefer_senior_coverage(cf),            # <<< NEW
        deadline_risk(cf)                      # <<< NEW
    ]

# ---- HARD ----

def require_employee_assigned(cf: ConstraintFactory) -> Constraint:
    return (cf.for_each(RequirementSlot)
            .filter(lambda r: r.employee is None)
            .penalize(HardSoftScore.ONE_HARD)
            .as_constraint("Employee must be assigned"))

def require_day_assigned(cf: ConstraintFactory) -> Constraint:
    return (cf.for_each(RequirementSlot)
            .filter(lambda r: r.day is None)
            .penalize(HardSoftScore.ONE_HARD)
            .as_constraint("Day must be assigned"))

def day_within_window(cf: ConstraintFactory) -> Constraint:
    def outside(r: RequirementSlot) -> bool:
        if r.day is None:
            return True
        return not (r.start_day_id <= r.day.id <= r.end_day_id)
    return (cf.for_each(RequirementSlot)
            .filter(outside)
            .penalize(HardSoftScore.ONE_HARD)
            .as_constraint("Day outside task window (deadline or start)"))

def skill_must_exist(cf: ConstraintFactory) -> Constraint:
    # Require that assigned employee actually has the P?-? skill (>=1).
    def lacks(r: RequirementSlot) -> bool:
        if r.employee is None: return True
        return r.employee.skills.get(r.pcode(), 0) < 1
    return (cf.for_each(RequirementSlot)
            .filter(lacks)
            .penalize(HardSoftScore.ONE_HARD)
            .as_constraint("Missing skill for task"))

def employee_daily_capacity_limit(cf: ConstraintFactory) -> Constraint:
    cap = _MAX_WORKER_PER_DAY
    if cap is None or cap < 1:
        cap = 999999  # effectively unlimited

    return (
        cf.for_each_unique_pair(
            RequirementSlot,
            Joiners.equal(lambda r: r.employee),
            Joiners.equal(lambda r: r.day.id if r.day is not None else None)
        )
        .filter(lambda a, b:
            a.employee is not None and b.employee is not None and
            a.day is not None and b.day is not None
        )
        .penalize(HardSoftScore.ONE_HARD if cap == 1 else HardSoftScore.ZERO)
        .as_constraint("Per-employee daily cap")
    )

def employee_not_double_booked_same_day(cf: ConstraintFactory) -> Constraint:
    return (
        cf.for_each_unique_pair(
            RequirementSlot,
            Joiners.equal(lambda r: r.employee),
            Joiners.equal(lambda r: r.day.id if r.day is not None else None)
        )
        .filter(lambda a, b:
            a.employee is not None and b.employee is not None and
            a.day is not None and b.day is not None
        )
        .penalize(HardSoftScore.ONE_HARD)
        .as_constraint("Employee double-booked same day")
    )

def process_precedence_within_module(cf: ConstraintFactory) -> Constraint:
    """
    Inside the SAME module: any work unit of process p+1 must be scheduled strictly
    AFTER any work unit of process p.
    """
    def violates(a: RequirementSlot, b: RequirementSlot) -> bool:
        if a.day is None or b.day is None:
            return True  # force assignment
        return not (b.day.id > a.day.id)  # must be strictly later
    return (
        cf.for_each_unique_pair(
            RequirementSlot,
            Joiners.equal(lambda r: r.module),
            Joiners.less_than(lambda r: r.process_id)  # a.proc < b.proc in same module
        )
        .filter(violates)
        .penalize(HardSoftScore.ONE_HARD)
        .as_constraint("Module process precedence")
    )

# <<< NEW: HARD — per-employee blackout days (vacation/training/OOO)
def employee_not_on_unavailable_day(cf: ConstraintFactory) -> Constraint:
    return (
        cf.for_each(RequirementSlot)
        .filter(lambda r:
            r.employee is not None and r.day is not None and
            r.day.id in (r.employee.unavailable_day_ids or set())
        )
        .penalize(HardSoftScore.ONE_HARD)
        .as_constraint("Employee unavailable day")
    )

# <<< NEW: HARD — no work on weekends
def no_weekend_work(cf: ConstraintFactory) -> Constraint:
    return (
        cf.for_each(RequirementSlot)
        .filter(lambda r:
            r.day is not None and r.day.d.weekday() not in _BUSINESS_DAYS
        )
        .penalize(HardSoftScore.ONE_HARD)
        .as_constraint("No weekend work")
    )

# ---- SOFT ----

def finish_asap(cf: ConstraintFactory) -> Constraint:
    def day_cost(r: RequirementSlot) -> int:
        return r.day.id if r.day is not None else 1000
    return (
        cf.for_each(RequirementSlot)
        .penalize(HardSoftScore.ONE_SOFT, lambda r: 5 * day_cost(r))  # stronger weight
        .as_constraint("Finish ASAP (early-day preference)")
    )

def gently_prefer_levels_near_3(cf: ConstraintFactory) -> Constraint:
    # A tiny nudge only; won’t block assigning more people early.
    def lvl_cost(r: RequirementSlot) -> int:
        if r.employee is None: return 0
        lvl = r.employee.skills.get(r.pcode(), 1)
        return max(0, abs(lvl - 3) // 3)  # 0 or 1 only
    return (
        cf.for_each(RequirementSlot)
        .penalize(HardSoftScore.ONE_SOFT, lvl_cost)
        .as_constraint("Tiny level-centering")
    )

# <<< NEW: SOFT — prefer continuity (same employee on same task across adjacent days)
def prefer_continuity(cf: ConstraintFactory) -> Constraint:
    return (
        cf.for_each_unique_pair(
            RequirementSlot,
            Joiners.equal(lambda r: r.employee),
            Joiners.equal(lambda r: r.module),
            Joiners.equal(lambda r: r.process_id),
            Joiners.equal(lambda r: r.task_letter),
            Joiners.less_than(lambda r: r.day.id if r.day else -1)
        )
        .filter(lambda a, b:
            a.employee is not None and b.employee is not None and
            a.day is not None and b.day is not None and
            b.day.id == a.day.id + 1  # consecutive days
        )
        .reward(HardSoftScore.ONE_SOFT)
        .as_constraint("Prefer continuity (consecutive days same worker)")
    )

# <<< NEW: SOFT — prefer senior coverage when juniors present on same task/day
def prefer_senior_coverage(cf: ConstraintFactory) -> Constraint:
    # Approximates “coverage”: reward senior–junior pairs co-assigned to the same task/day.
    def lvl(emp: Employee, code: str) -> int:
        return emp.skills.get(code, 0)

    return (
        cf.for_each_unique_pair(
            RequirementSlot,
            Joiners.equal(lambda r: r.module),
            Joiners.equal(lambda r: r.process_id),
            Joiners.equal(lambda r: r.task_letter),
            Joiners.equal(lambda r: r.day.id if r.day else None)
        )
        .filter(lambda a, b:
            a.employee is not None and b.employee is not None and
            a.day is not None and b.day is not None
        )
        .filter(lambda a, b:
            # senior >=4, junior <=2 on the same pcode
            (lvl(a.employee, a.pcode()) >= 4 and lvl(b.employee, b.pcode()) <= 2) or
            (lvl(b.employee, b.pcode()) >= 4 and lvl(a.employee, a.pcode()) <= 2)
        )
        .reward(HardSoftScore.ONE_SOFT)
        .as_constraint("Prefer senior coverage when juniors present")
    )

# <<< NEW: SOFT — deadline risk ramp (stronger penalty as you get close to end_day_id)
def deadline_risk(cf: ConstraintFactory) -> Constraint:
    def cost(r: RequirementSlot) -> int:
        if r.day is None: return 0
        # distance to deadline in days
        dist = max(0, r.end_day_id - r.day.id)
        # escalate penalty near deadline (you can tune these)
        if dist <= 0:  # on the last allowed day
            return 10
        elif dist == 1:
            return 6
        elif dist == 2:
            return 3
        else:
            return 1
    return (
        cf.for_each(RequirementSlot)
        .penalize(HardSoftScore.ONE_SOFT, cost)
        .as_constraint("Deadline risk ramp")
    )

# -------------------- YAML loading & problem build --------------------

@dataclass
class ModuleSpec:
    code: str
    start_day_id: int

def load_config_modules(path: str):
    """
    YAML schema (4th model) — additions marked with (NEW):
      start_day: 2025-09-01
      horizon_days: 30
      maximum_worker: 1 | "NONE"
      modules: [...]
      employees:
        - name: AA
          skills: { P1-A: 3, P2-D: 4, ... }
          unavailable: [2025-09-03, 2025-09-10]     # (NEW) days AA cannot work
    """
    with open(path, "r", encoding="utf-8") as f:
        cfg = yaml.safe_load(f)

    # Horizon & calendar
    start_day = datetime.strptime(str(cfg["start_day"]), "%Y-%m-%d").date()
    horizon_days = int(cfg.get("horizon_days", 30))
    days = [DaySlot(i, start_day + timedelta(days=i)) for i in range(horizon_days)]

    # maximum_worker: "NONE" or int -> we keep the hard "no double-book same day" rule,
    # and use this as an additional cap if you ever allow >1/day later.
    global _MAX_WORKER_PER_DAY
    mw = cfg.get("maximum_worker", "NONE")
    if isinstance(mw, str) and mw.strip().upper() == "NONE":
        _MAX_WORKER_PER_DAY = 1
    else:
        try:
            _MAX_WORKER_PER_DAY = int(mw)
        except Exception:
            _MAX_WORKER_PER_DAY = 1

    def day_to_idx(datestr: str) -> int:
        d = datetime.strptime(str(datestr), "%Y-%m-%d").date()
        return max(0, min(horizon_days - 1, (d - start_day).days))

    windows: List[TaskWindow] = []

    # ---- Modules ----
    modules_cfg = cfg["modules"]
    for m in modules_cfg:
        mcode = str(m["code"]).strip()

        # Module start: default to file start_day if missing
        module_start_idx = day_to_idx(m.get("start_date", cfg["start_day"]))

        for proc in m.get("processes", []):
            pid = int(proc["id"])
            p_end_idx = day_to_idx(proc["end_date"])

            # Optional per-process start override; otherwise inherit module start
            p_start_idx = module_start_idx
            if "start_date" in proc and proc["start_date"]:
                p_start_idx = max(p_start_idx, day_to_idx(proc["start_date"]))

            for t in proc.get("tasks", []):
                full_code = str(t["code"]).strip().upper()  # e.g. "S1-P2-A"
                parts = full_code.split("-")
                if len(parts) != 3 or not parts[1].startswith("P"):
                    raise ValueError(f"Bad task code '{full_code}' (expected 'Sx-Py-Z').")

                letter = parts[2]
                t_end_idx = day_to_idx(t.get("end_date", proc["end_date"]))
                end_idx = min(p_end_idx, t_end_idx)  # never exceed process end

                windows.append(TaskWindow(
                    module=mcode,
                    process_id=pid,
                    task_letter=letter,
                    start_day_id=p_start_idx,
                    end_day_id=end_idx,
                    workload=int(t["workload"])
                ))

    # ---- Employees ----
    employees: List[Employee] = []
    eid = 1
    for e in cfg["employees"]:
        name = str(e["name"])
        skills = {str(k).strip().upper(): int(v) for k, v in (e.get("skills", {}) or {}).items()}

        # <<< NEW: parse optional per-employee blackout list: `unavailable: [YYYY-MM-DD, ...]`
        unavailable_ids: Set[int] = set()
        for ds in (e.get("unavailable", []) or []):
            try:
                unavailable_ids.add(day_to_idx(str(ds)))
            except Exception:
                pass

        employees.append(Employee(eid, name, skills, unavailable_ids))
        eid += 1

    return start_day, days, windows, employees

def build_requirement_slots(windows: List[TaskWindow]) -> List[RequirementSlot]:
    reqs: List[RequirementSlot] = []
    rid = 1
    for w in windows:
        for _ in range(w.workload):
            reqs.append(RequirementSlot(
                id=rid,
                module=w.module,
                process_id=w.process_id,
                task_letter=w.task_letter,
                start_day_id=w.start_day_id,
                end_day_id=w.end_day_id
            ))
            rid += 1
    return reqs

# -------------------- Public API --------------------

def solve_from_config(cfg_path: str = "config_modules.yaml"):
    start_day, days, windows, employees = load_config_modules(cfg_path)
    reqs = build_requirement_slots(windows)

    solver_config = SolverConfig(
        solution_class=Schedule,
        entity_class_list=[RequirementSlot],
        score_director_factory_config=ScoreDirectorFactoryConfig(
            constraint_provider_function=define_constraints
        ),
        termination_config=TerminationConfig(spent_limit=Duration(seconds=30))
    )
    solver = SolverFactory.create(solver_config).build_solver()
    problem = Schedule(days=days, employees=employees, reqs=reqs)
    solution: Schedule = solver.solve(problem)
    return solution, start_day

# -------------------- CLI (optional) --------------------

def main():
    solution, start_day = solve_from_config("config_modules.yaml")
    print(f"Best score: {solution.score}")
    # Minimal printout: count per (module, P, task, day)
    from collections import defaultdict
    cell = defaultdict(list)
    for r in solution.reqs:
        if r.employee and r.day:
            cell[(r.module, r.process_id, r.task_letter, r.day.id)].append(r.employee.name)
    # Show first few rows:
    for k in sorted(cell.keys())[:15]:
        mod, p, t, did = k
        print(f"{mod}-P{p}-{t} @ {(start_day + timedelta(days=did)).isoformat()} -> {len(cell[k])} workers")

if __name__ == "__main__":
    main()
