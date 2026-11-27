"""
Workflow tasks (modules) of the Schedule (tool 2-4, step C): check a row of the seiban list, add the module of a white row,
delete the module of a pink row.

The functions work on the plain `schedule` dict (the part under `schedule:`) and change it in place: the caller passes a copy.
The caller puts weekly_planning/ on the import path (the tool script and the tests do).
"""
from __future__ import annotations

import re

from common.text import tool_code

MODULE_PREFIX = "e"                     # module ids are e1, e2 ... as in every Schedule made so far
_PROCESS_PROBLEM = re.compile(r"process (\d+) (?:start|end):")


# ----------------------------------------------------------------------------- checking a row

def delete_row_problems(row):
    """Why a pink row cannot be handled (only the seiban matters: the colour is what makes the row a deletion)."""
    return [] if row.code else [p for p in row.problems if "seiban" in p] or ["seiban is not a module code"]


def add_row_problems(row, template):
    """Why a white row cannot be added (empty list = valid): the seiban, the template and the dates of every process
    the template has.  Dates of a process the template does not have are ignored."""
    template_ok = template is not None and template.valid
    wanted = set(template.phases()) if template_ok else None
    problems = []
    for text in row.problems:
        found = _PROCESS_PROBLEM.match(text)
        if found and wanted is not None and int(found.group(1)) not in wanted:
            continue
        problems.append(text)
    if row.template is not None:
        if template is None:
            problems.append(f"there is no sheet for template {row.template} (工程テンプレート{row.template})")
        elif not template.valid:
            problems.append(f"template {row.template} cannot be used: " + ("; ".join(template.problems) or "it has no operations"))
    if template_ok:
        phases = template.phases()
        first_start = row.phases.get(phases[0], (None, None))[0]
        for p in phases:
            start, end = row.phases.get(p, (None, None))
            if not start:
                problems.append(f"process {p} has no start date")
            if not end:
                problems.append(f"process {p} has no end date")
            if start and end and end < start:
                problems.append(f"process {p}: end date {end} is before start date {start}")
            elif first_start and end and end < first_start:
                problems.append(f"process {p}: end date {end} is before the first process start date {first_start}")
    return problems


# ----------------------------------------------------------------------------- finding and numbering

def module_tasks(schedule, code):
    """The workflow tasks of the Schedule whose name holds this seiban."""
    return [t for t in schedule.get("workflow_task_list") or [] if tool_code(t.get("name")) == code]


def _used_ids(schedule):
    used = set()
    for t in schedule.get("workflow_task_list") or []:
        used.add(t.get("id"))
        for ph in t.get("phase_task_list") or []:
            used.add(ph.get("id"))
            used.update(op.get("id") for op in ph.get("operation_task_list") or [])
    used.update(m.get("id") for m in schedule.get("misc_task_list") or [])
    return used


def next_module_id(schedule):
    """The next free e<n>: above every e<number> of the workflow tasks, and no id of the Schedule starts with it."""
    used = _used_ids(schedule)
    pattern = re.compile(rf"^{MODULE_PREFIX}(\d+)$")
    n = max((int(m.group(1)) for i in used for m in [pattern.match(str(i))] if m), default=0) + 1
    while any(str(i) == f"{MODULE_PREFIX}{n}" or str(i).startswith(f"{MODULE_PREFIX}{n}p") for i in used):
        n += 1
    return f"{MODULE_PREFIX}{n}"


# ----------------------------------------------------------------------------- add

def _hours(value):
    return int(value) if float(value).is_integer() else value


def add_module(schedule, code, workflow_ref, fab_id, template, row):
    """Adds the workflow task of a valid white row and returns it.

    workflow_ref is the WorkflowRef of the template (ids and names of the phases and operations of the workflow).  The phase
    tasks and operation tasks get the short ids <module>p<P> and <module>p<P>o<K> (the solver cuts task ids at p<number>o<number>).
    recommends_worker_min / max are not written: the solver treats a missing value as no recommendation."""
    module_id = next_module_id(schedule)
    phase_tasks = []
    phases = template.phases()
    start = row.phases[phases[0]][0]
    for p in phases:
        _, end = row.phases[p]
        ops = []
        for o in template.ops_of(p):
            op = workflow_ref.ops[(p, o.op)]
            ops.append({"id": f"{module_id}p{p}o{o.op}", "name": op["name"], "description": None, "operation": op["id"],
                        "workload_hours": _hours(o.hours), "color_code": None})
        phase = workflow_ref.phases[p]
        phase_tasks.append({"id": f"{module_id}p{p}", "name": phase["name"], "description": None, "phase": phase["id"],
                            "start_date": start, "end_date": end, "operation_task_list": ops})
    task = {"id": module_id, "name": code, "description": None, "workflow": workflow_ref.workflow_id, "fab": fab_id,
            "phase_task_list": phase_tasks}
    schedule.setdefault("workflow_task_list", []).append(task)
    return task

# ----------------------------------------------------------------------------- delete

def delete_module(schedule, task):
    """Removes a workflow task and every assignment row of its operation tasks.
    Returns {"phases", "operations", "rows", "fixed_rows"} (fixed_rows: how many removed rows were Fixed)."""
    op_ids = {op.get("id") for ph in task.get("phase_task_list") or [] for op in ph.get("operation_task_list") or []}
    kept, rows, fixed = [], 0, 0
    for a in schedule.get("assignment_list") or []:
        if a.get("operation_task") in op_ids:
            rows += 1
            fixed += str(a.get("plan_flexibility") or "Flexible") == "Fixed"
        else:
            kept.append(a)
    if "assignment_list" in schedule:
        schedule["assignment_list"] = kept
    schedule["workflow_task_list"] = [t for t in schedule.get("workflow_task_list") or [] if t is not task]
    return {"phases": len(task.get("phase_task_list") or []), "operations": len(op_ids), "rows": rows, "fixed_rows": fixed}

