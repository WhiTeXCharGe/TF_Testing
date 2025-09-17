
import yaml
from datetime import datetime

def parse_env(env_path="EnvConfig.yaml"):
    with open(env_path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f)

def parse_sched(sched_path="Schedule.yaml"):
    with open(sched_path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f)

def build_internal_from_env_sched(env: dict, sched: dict):
    """
    Returns a dict with everything the solver needs, built ONLY from the YAMLs:
      - start_day (date), horizon_days (int)
      - employees: list of {idx, id, name, company, is_manager, skills: {p#o#:level}}
      - modules: [{code, fab, processes: [{phase, start(date), end(date), operations: [
            {operation_task_id, op_id, allowed(list[int]), min_heads(int), max_heads(int), workload_days(int)}
        ]}]}]
      - hours_union: sorted union of all op allowed sets
      - required_hours: {(module, op_id): int} where int = workload_days * baseline (4 if [4], else 8)
      - op_minmax: {op_id: (min, max)} from EnvConfig
    """
    e = env["environment"]
    s = sched["schedule"]

    # Planning horizon from Schedule
    start_date = datetime.strptime(s["planrange"]["start_date"], "%Y/%m/%d").date()
    end_date   = datetime.strptime(s["planrange"]["end_date"], "%Y/%m/%d").date()
    horizon_days = (end_date - start_date).days + 1

    # Operations from EnvConfig
    opdef = {}
    hours_union = set()
    for ph in e["workflow_list"][0]["phase_list"]:
        for op in ph["operation_list"]:
            hrs = list(op.get("work_hours", []) or [])
            if not hrs:
                hrs = [8]
            opdef[op["id"]] = {
                "allowed": hrs,
                "min": int(op.get("min_worker_num", 0)),
                "max": int(op.get("max_worker_num", 10**9)),
                "phase": ph["id"]
            }
            hours_union.update(hrs)

    # Employees (skills keys are p#o# as-is)
    employees = []
    for i, w in enumerate(e["worker_list"]):
        employees.append({
            "idx": i + 1,
            "id": w.get("id"),
            "name": w.get("name"),
            "is_manager": bool(w.get("is_manager", False)),
            "company": w.get("worker_company"),
            "skills": dict(w.get("skill_map") or {})
        })

    # Modules + processes + operation tasks from Schedule
    modules = []
    required_hours = {}   # (module, op_id) -> hours
    op_minmax = {}        # op_id -> (min, max)
    for wf in s["workflow_task_list"]:
        mcode = wf["id"]
        module = {"code": mcode, "fab": wf.get("fab"), "processes": []}
        for ph_task in wf["phase_task_list"]:
            phase = ph_task["phase"]  # p1..p4
            p_start = datetime.strptime(ph_task["start_date"], "%Y/%m/%d").date()
            p_end   = datetime.strptime(ph_task["end_date"], "%Y/%m/%d").date()
            proc = {"phase": phase, "start": p_start, "end": p_end, "operations": []}
            for ot in ph_task["operation_task_list"]:
                op_id = ot["operation"]  # p#o#
                opcfg = opdef.get(op_id, {"allowed": [8], "min": 0, "max": 10**9})
                allowed = list(opcfg["allowed"])
                min_h = int(opcfg["min"])
                max_h = int(opcfg["max"])
                op_minmax[op_id] = (min_h, max_h)
                workload_days = int(ot["workload_days"])
                baseline = 4 if allowed == [4] else 8
                required_hours[(mcode, op_id)] = workload_days * baseline
                proc["operations"].append({
                    "operation_task_id": ot["id"],  # e1p1o2
                    "op_id": op_id,                 # p1o2
                    "allowed": allowed,
                    "min_heads": min_h,
                    "max_heads": max_h,
                    "workload_days": workload_days
                })
            module["processes"].append(proc)
        modules.append(module)

    cfg = {
        "start_day": start_date,
        "horizon_days": horizon_days,
        "employees": employees,
        "modules": modules,
        "hours_union": sorted(hours_union or {4,8,10,12}),
        "required_hours": required_hours,
        "op_minmax": op_minmax,
    }
    return cfg
