# load_schedule_yaml.py
from __future__ import annotations

from dataclasses import dataclass, field
from datetime import date, datetime
from pathlib import Path
from typing import List, Optional, Any, Dict

import sys
import yaml


# ---------- Helpers ----------

def parse_date(s: str | None) -> Optional[date]:
    """Accept 'YYYY/MM/DD' or 'YYYY-MM-DD'. Return None if missing/blank."""
    if not s:
        return None
    s = str(s).strip()
    for fmt in ("%Y/%m/%d", "%Y-%m-%d"):
        try:
            return datetime.strptime(s, fmt).date()
        except ValueError:
            pass
    raise ValueError(f"Unrecognized date format: {s!r} (expected YYYY/MM/DD or YYYY-MM-DD)")


def get(d: Dict[str, Any], key: str, default=None):
    """Tiny helper to read dict keys with a default."""
    return d.get(key, default)


# ---------- Data models ----------

@dataclass
class OperationTask:
    id: str
    name: str
    operation: str
    workload_days: int

    @staticmethod
    def from_dict(d: Dict[str, Any]) -> "OperationTask":
        return OperationTask(
            id=str(get(d, "id", "")),
            name=str(get(d, "name", "")),
            operation=str(get(d, "operation", "")),
            workload_days=int(get(d, "workload_days", 0)),
        )


@dataclass
class PhaseTask:
    id: str
    name: str
    phase: str
    start_date: Optional[date]
    end_date: Optional[date]
    operation_task_list: List[OperationTask] = field(default_factory=list)

    @staticmethod
    def from_dict(d: Dict[str, Any]) -> "PhaseTask":
        ops_raw = get(d, "operation_task_list", []) or []
        ops = [OperationTask.from_dict(x) for x in ops_raw]
        return PhaseTask(
            id=str(get(d, "id", "")),
            name=str(get(d, "name", "")),
            phase=str(get(d, "phase", "")),
            start_date=parse_date(get(d, "start_date")),
            end_date=parse_date(get(d, "end_date")),
            operation_task_list=ops,
        )

    def total_workload_days(self) -> int:
        return sum(o.workload_days for o in self.operation_task_list)


@dataclass
class WorkflowTask:
    id: str
    name: str
    workflow: str
    fab: str
    phase_task_list: List[PhaseTask] = field(default_factory=list)

    @staticmethod
    def from_dict(d: Dict[str, Any]) -> "WorkflowTask":
        phases_raw = get(d, "phase_task_list", []) or []
        phases = [PhaseTask.from_dict(x) for x in phases_raw]
        return WorkflowTask(
            id=str(get(d, "id", "")),
            name=str(get(d, "name", "")),
            workflow=str(get(d, "workflow", "")),
            fab=str(get(d, "fab", "")),
            phase_task_list=phases,
        )

    def total_workload_days(self) -> int:
        return sum(p.total_workload_days() for p in self.phase_task_list)


@dataclass
class PlanRange:
    start_date: Optional[date]
    end_date: Optional[date]

    @staticmethod
    def from_dict(d: Dict[str, Any]) -> "PlanRange":
        return PlanRange(
            start_date=parse_date(get(d, "start_date")),
            end_date=parse_date(get(d, "end_date")),
        )


@dataclass
class ScheduleRoot:
    planrange: PlanRange
    workflow_task_list: List[WorkflowTask]

    @staticmethod
    def from_dict(d: Dict[str, Any]) -> "ScheduleRoot":
        planrange = PlanRange.from_dict(get(d, "planrange", {}) or {})
        workflows_raw = get(d, "workflow_task_list", []) or []
        workflows = [WorkflowTask.from_dict(x) for x in workflows_raw]
        return ScheduleRoot(planrange=planrange, workflow_task_list=workflows)


# ---------- Loader ----------

def load_schedule_yaml(path: str | Path) -> ScheduleRoot:
    with open(path, "r", encoding="utf-8") as f:
        raw = yaml.safe_load(f) or {}
    schedule = get(raw, "schedule", {}) or {}
    return ScheduleRoot.from_dict(schedule)


# ---------- Pretty print ----------

def print_schedule(s: ScheduleRoot) -> None:
    pr = s.planrange
    print("=== SCHEDULE ===")
    print(f"Plan Range: {pr.start_date}  →  {pr.end_date}")
    print()

    if not s.workflow_task_list:
        print("(No workflow_task_list)")
        return

    for w in s.workflow_task_list:
        print(f"- Workflow Task: {w.id} | {w.name} | workflow={w.workflow} | fab={w.fab}")
        print(f"  Total workload (days): {w.total_workload_days()}")
        if not w.phase_task_list:
            print("  (No phases)")
            continue

        for p in w.phase_task_list:
            print(f"  • Phase: {p.id} | {p.name} | {p.phase}  "
                  f"[{p.start_date} → {p.end_date}]  "
                  f"total_workload_days={p.total_workload_days()}")
            if not p.operation_task_list:
                print("    (No operations)")
                continue

            for o in p.operation_task_list:
                print(f"      - Op: {o.id} | {o.name} | {o.operation} | workload_days={o.workload_days}")
        print()  # spacer between workflow tasks


# ---------- CLI ----------

def main(argv: List[str]) -> int:
    if len(argv) < 2:
        print("Usage: python load_schedule_yaml.py <path-to-yaml>")
        return 2
    path = argv[1]
    root = load_schedule_yaml(path)
    print_schedule(root)
    return 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv))
