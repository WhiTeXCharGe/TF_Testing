"""
Workflows of the EnvConfig (tool 2-4, step B): find the workflow that a process template matches, or create a new one.

A template matches a workflow when both have the same operations: same process numbers, work numbers, work names and
minimum / maximum workers (ids, names and work_hours are not compared).  An existing workflow is never changed: a template
that no longer matches creates a new workflow and the old one stays.  The functions change the plain `environment` dict in
place (the caller passes a copy) and put "INFO: ..." / "WARN: ..." lines into `report`.

The caller puts weekly_planning/ on the import path (the tool script and the tests do).
"""
from __future__ import annotations

import re
from dataclasses import dataclass, field

from common.text import match_key

WORK_HOURS = [10]            # hours of a work day, as in every existing operation

_PHASE_NO = re.compile(r"p(\d+)", re.IGNORECASE)
_OP_NO = re.compile(r"o(\d+)", re.IGNORECASE)


@dataclass
class WorkflowRef:
    """What the Schedule needs to build the tasks of a module that uses a template."""
    workflow_id: str
    created: bool
    phases: dict = field(default_factory=dict)       # {P: {"id", "name"}}
    ops: dict = field(default_factory=dict)          # {(P, K): {"id", "name"}}
    new_ops: list = field(default_factory=list)      # [{"id", "name"}] of the operations that were just created


def _last_number(pattern, text):
    """The last number the pattern finds in an id ('wf1p2' -> 2 for p(\\d+)), as the solver reads phase numbers."""
    found = pattern.findall(str(text or ""))
    return int(found[-1]) if found else None


def workflow_signature(wf):
    """The operations of an EnvConfig workflow: {(process, work, work name key, min workers, max workers)}."""
    out = set()
    for ph in wf.get("phase_list") or []:
        p = _last_number(_PHASE_NO, ph.get("id"))
        for op in ph.get("operation_list") or []:
            out.add((p, _last_number(_OP_NO, op.get("id")), match_key(op.get("name")), op.get("min_worker_num"), op.get("max_worker_num")))
    return frozenset(out)


def template_signature(template):
    return frozenset((o.phase, o.op, match_key(o.name), o.min_workers, o.max_workers) for o in template.ops)


def _without_workers(signature):
    return frozenset(s[:3] for s in signature)


def _ref_of(wf, created, new_ops=()):
    ref = WorkflowRef(wf["id"], created, new_ops=list(new_ops))
    for ph in wf.get("phase_list") or []:
        p = _last_number(_PHASE_NO, ph.get("id"))
        ref.phases[p] = {"id": ph["id"], "name": ph.get("name")}
        for op in ph.get("operation_list") or []:
            ref.ops[(p, _last_number(_OP_NO, op.get("id")))] = {"id": op["id"], "name": op.get("name")}
    return ref


def next_workflow_number(env):
    """Highest wf<number> in the EnvConfig + 1 (ids without a number, such as the first workflow, are ignored)."""
    numbers = [int(m.group(1)) for wf in env.get("workflow_list") or [] for m in [re.fullmatch(r"wf(\d+)", str(wf.get("id", "")))] if m]
    return max(numbers, default=0) + 1


def create_workflow(env, template):
    """Adds a workflow built from the template; ids are wf<n>, wf<n>p<P>, wf<n>p<P>o<K>."""
    wf_id = f"wf{next_workflow_number(env)}"
    phases, new_ops = [], []
    for p in template.phases():
        ops = []
        for o in template.ops_of(p):
            op_id = f"{wf_id}p{p}o{o.op}"
            ops.append({"id": op_id, "name": o.name, "work_hours": list(WORK_HOURS),
                        "min_worker_num": o.min_workers, "max_worker_num": o.max_workers})
            new_ops.append({"id": op_id, "name": o.name})
        phases.append({"id": f"{wf_id}p{p}", "name": f"p{p}", "operation_list": ops})
    wf = {"id": wf_id, "name": wf_id, "phase_list": phases}
    env.setdefault("workflow_list", []).append(wf)
    return wf, new_ops


def ensure_workflow(env, template, report):
    """The WorkflowRef of the template: the existing workflow with the same operations, or a new one."""
    wanted = template_signature(template)
    workflows = env.get("workflow_list") or []
    for wf in workflows:
        if workflow_signature(wf) == wanted:
            report.append(f"INFO: template {template.number} uses the existing workflow {wf['id']}")
            return _ref_of(wf, created=False)
    hint = ""
    for wf in workflows:
        if _without_workers(workflow_signature(wf)) == _without_workers(wanted):
            hint = f"; workflow {wf['id']} has the same works but other minimum / maximum workers"
            break
    wf, new_ops = create_workflow(env, template)
    report.append(f"INFO: template {template.number}: new workflow {wf['id']} created (no existing workflow has the same operations{hint})")
    return _ref_of(wf, created=True, new_ops=new_ops)
