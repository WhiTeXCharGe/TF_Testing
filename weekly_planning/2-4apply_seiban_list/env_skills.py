"""
Skills of the workers for new operations (tool 2-4, step B).

When a workflow is created, its operations are new, so no worker has a skill for them.  A new operation gets, per worker, the
HIGHEST level the worker has for existing operations with the same work name (Mech, Elec, QC ...), together with the
worker_type_by_operation entry of that operation.  A worker with no operation of that name gets nothing.  The solver reads a level
above 0 as "can do" (higher = better) and a missing entry like 0, so copying a 0 changes nothing.

The functions change the plain `environment` dict in place (the caller passes a copy) and put "INFO: ..." / "WARN: ..." lines
into `report`.  The caller puts weekly_planning/ on the import path (the tool script and the tests do).
"""
from __future__ import annotations

from common.text import match_key


def _level(value):
    return value if isinstance(value, (int, float)) and not isinstance(value, bool) else 0


def add_skills(env, new_ops, report):
    """new_ops: [{"id", "name"}] of the operations that were just created (WorkflowRef.new_ops)."""
    ops_by_name = {}                        # work name key -> operation ids, in EnvConfig order
    for wf in env.get("workflow_list") or []:
        for ph in wf.get("phase_list") or []:
            for op in ph.get("operation_list") or []:
                ops_by_name.setdefault(match_key(op.get("name")), []).append(op["id"])

    created_now = {n["id"] for n in new_ops}
    for new in new_ops:
        sources = [i for i in ops_by_name.get(match_key(new["name"]), []) if i not in created_now]     # only operations that existed before
        if not sources:
            report.append(f"WARN: operation {new['id']} ({new['name']}): no existing operation has this work name, so nobody has the skill")
            continue
        able = 0
        for w in env.get("worker_list") or []:
            skills = w.get("skill_map") or {}
            have = [(_level(skills[i]), i) for i in sources if i in skills]
            if not have:
                continue
            best = max(level for level, _ in have)
            source = next(i for level, i in have if level == best)          # the first one in EnvConfig order on a tie
            w.setdefault("skill_map", {})[new["id"]] = best
            types = w.get("worker_type_by_operation") or {}
            if source in types:
                w.setdefault("worker_type_by_operation", {})[new["id"]] = types[source]
            able += best > 0
        if able:
            report.append(f"INFO: operation {new['id']} ({new['name']}): {able} workers can do it (highest level of the same work name)")
        else:
            report.append(f"WARN: operation {new['id']} ({new['name']}): no worker has a skill for it (no worker can do the work name)")
