# export_schedule.py
# Overwrites Schedule.yaml with assignment_list based on the solved plan.

import yaml
from collections import defaultdict
from datetime import timedelta

def overwrite_schedule_with_assignments(final, start_day, sched_path="Schedule.yaml"):
    with open(sched_path, "r", encoding="utf-8") as f:
        sched = yaml.safe_load(f)

    # Build (module_id, op_id) -> operation_task_id from Schedule.yaml
    op_task_id = {}
    for wf in sched["schedule"]["workflow_task_list"]:
        module = wf["id"]
        for ph in wf["phase_task_list"]:
            for ot in ph["operation_task_list"]:
                op_task_id[(module, ot["operation"])] = ot["id"]

    # Aggregate hours by (worker_wid, day_idx, module, op_id)
    per = defaultdict(int)
    for u in final.tokens:
        if u.employee is None or u.employee.id == 0:  # unassigned
            continue
        if u.day is None or u.day.id < 0:
            continue
        hrs = int(u.hours or 0)
        if hrs <= 0:
            continue
        wid = u.employee.wid  # "w#"
        key = (wid, u.day.id, u.module, u.op_id)
        per[key] += hrs

    # Bucket by (worker, module, op) -> {day_idx: hours}
    buckets = defaultdict(dict)
    for (wid, didx, mod, op_id), h in per.items():
        buckets[(wid, mod, op_id)][didx] = buckets[(wid, mod, op_id)].get(didx, 0) + h

    # Build assignment_list by folding consecutive days
    assignments = []
    for (wid, mod, op_id), daymap in buckets.items():
        didxs = sorted(daymap.keys())
        if not didxs:
            continue

        # split into runs of consecutive days
        run = [didxs[0]]
        runs = []
        for d in didxs[1:]:
            if d == run[-1] + 1:
                run.append(d)
            else:
                runs.append(run)
                run = [d]
        runs.append(run)

        for r in runs:
            start = start_day + timedelta(days=r[0])
            end   = start_day + timedelta(days=r[-1])
            work_list = [{"date": (start_day + timedelta(days=d)).strftime("%Y/%m/%d"),
                          "hour": int(daymap[d])} for d in r]
            task_id = op_task_id.get((mod, op_id))
            if not task_id:
                # skip if not found (shouldn't happen if YAMLs are consistent)
                continue
            assignments.append({
                "worker": wid,
                "operation_task": task_id,
                "start_date": start.strftime("%Y/%m/%d"),
                "end_date": end.strftime("%Y/%m/%d"),
                "work_date_lsit": work_list,          # keep your original field name
                "plan_flexibility": "Flexible"
            })

    # Overwrite Schedule.yaml
    sched["schedule"]["assignment_list"] = assignments
    with open(sched_path, "w", encoding="utf-8") as f:
        yaml.safe_dump(sched, f, sort_keys=False, allow_unicode=True)

    print(f"Overwrote {sched_path} with {len(assignments)} assignments.")
