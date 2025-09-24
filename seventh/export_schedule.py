# export_schedule.py
# Overwrites Schedule.yaml with assignment_list based on the solved plan.
# Compatible with both the old "tokens" model and the new "seats + seat_days" model.

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

    if hasattr(final, "seat_days") and hasattr(final, "seats"):
        # ---- New model: seats + seat_days ----
        # Map seat_key -> (employee, module, op)
        emp_by_seat = {}
        meta_by_seat = {}
        for s in final.seats:
            emp_by_seat[s.seat_key] = s.employee
            meta_by_seat[s.seat_key] = (s.module, s.op_id)
        # Fold per-day hours for each assigned seat
        for sd in final.seat_days:
            emp = emp_by_seat.get(sd.seat_key)
            if emp is None or getattr(emp, "id", 0) == 0:
                continue
            module, op_id = meta_by_seat.get(sd.seat_key, (None, None))
            if module is None or op_id is None:
                continue
            hrs = int(getattr(sd, "hours", 0) or 0)
            if hrs <= 0:
                continue
            didx = int(getattr(sd.day, "id", -1))
            if didx < 0:
                continue
            key = (emp.wid, didx, module, op_id)
            per[key] += hrs

    elif hasattr(final, "tokens"):
        # ---- Backward compatibility: old UnitToken model ----
        for u in final.tokens:
            if u.employee is None or u.employee.id == 0:
                continue
            if u.day is None or u.day.id < 0:
                continue
            hrs = int(u.hours or 0)
            if hrs <= 0:
                continue
            key = (u.employee.wid, u.day.id, u.module, u.op_id)
            per[key] += hrs
    else:
        raise RuntimeError("Unsupported solution structure: expected 'seats+seat_days' or 'tokens'.")

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
                "work_date_list": work_list,          # keep original field name
                "plan_flexibility": "Flexible"
            })

    # Overwrite Schedule.yaml
    sched["schedule"]["assignment_list"] = assignments
    with open(sched_path, "w", encoding="utf-8") as f:
        yaml.safe_dump(sched, f, sort_keys=False, allow_unicode=True)

    print(f"Overwrote {sched_path} with {len(assignments)} assignments.")
