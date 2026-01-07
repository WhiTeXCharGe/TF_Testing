import pandas as pd
import yaml
from datetime import datetime
from pathlib import Path
from collections import defaultdict

# ============================================================
# Small helpers
# ============================================================

def _to_ymd(dt) -> str:
    """Convert pandas/datetime to 'YYYY/MM/DD' string."""
    if isinstance(dt, pd.Timestamp):
        return dt.strftime("%Y/%m/%d")
    if isinstance(dt, datetime):
        return dt.strftime("%Y/%m/%d")
    return str(dt)


# ============================================================
# SU_OTHER: workers + "historic" module tasks + sample assignments
# ============================================================

def parse_su_others(path: str, sheet_name: str = "予定表_2024"):
    xls = pd.ExcelFile(path)
    df = xls.parse(sheet_name, header=None)

    n_rows, n_cols = df.shape

    # -------- locate date header row & date columns --------
    date_row_idx = None
    for r in range(0, 5):
        row = df.iloc[r]
        if any(isinstance(v, (pd.Timestamp, datetime)) for v in row):
            date_row_idx = r
            break
    if date_row_idx is None:
        raise RuntimeError("Could not find row with date headers in SU_Others.")

    date_row = df.iloc[date_row_idx]
    date_cols = [c for c, v in enumerate(date_row) if isinstance(v, (pd.Timestamp, datetime))]
    if not date_cols:
        raise RuntimeError("Could not find any date columns in SU_Others.")

    first_date_col = min(date_cols)
    last_date_col = max(date_cols)

    # In your file: 企業名/姓名 row is 2 rows below date row
    worker_start_row = date_row_idx + 2

    # -------- worker_company_list & worker_list --------
    worker_company_map = {}   # company_name -> id
    worker_company_list = []
    worker_list = []
    worker_id_map = {}        # (company, name) -> worker_id

    def get_worker_company_id(company_name: str) -> str:
        company_name = str(company_name).strip()
        if company_name not in worker_company_map:
            cid = f"wc{len(worker_company_map) + 1}"
            worker_company_map[company_name] = cid
            worker_company_list.append({
                "id": cid,
                "name": company_name,
                "annual_overtime_limit": 360,
                "monthly_overtime_limit": 40,
                "unavailable_dates": [],
            })
        return worker_company_map[company_name]

    for r in range(worker_start_row, n_rows):
        company = df.iat[r, 0]
        name = df.iat[r, 1]

        # skip rows with no name
        if pd.isna(name):
            continue

        company_id = get_worker_company_id(company)
        wid = f"w{len(worker_list) + 1:03d}"

        worker_id_map[(str(company).strip(), str(name).strip())] = wid
        worker_list.append({
            "id": wid,
            "name": str(name).strip(),
            "worker_company": company_id,
            "is_manager": False,          # as requested
            "skill_map": {},              # unknown for now
            "fab_suitability_map": [],    # unknown for now
            "unavailable_dates": [],      # unknown for now
        })

    # -------- detect "historic" module codes (e.g. 530N02111A_TSMC12＿新規) --------
    # We treat long strings with "N0" and "A" in the date area as module names.
    hist_map_dates = defaultdict(set)

    for r in range(worker_start_row - 2, n_rows):
        for c in range(first_date_col, last_date_col + 1):
            v = df.iat[r, c]
            if isinstance(v, str):
                text = v.strip()
                if len(text) >= 15 and "N0" in text and "A" in text:
                    dt = date_row[c]
                    if isinstance(dt, (pd.Timestamp, datetime)):
                        hist_map_dates[text].add(dt)

    hist_tasks = []
    for idx, (name, dates) in enumerate(hist_map_dates.items(), start=1):
        sorted_dates = sorted(dates)
        start_dt = sorted_dates[0]
        end_dt = sorted_dates[-1]
        hist_tasks.append({
            "id": f"hist{idx}",
            "name": name,
            "workflow": "wf_hist",
            "fab": "f_tw",
            "phase_task_list": [
                {
                    "id": f"hist{idx}_p1",
                    "name": "Historic Onsite Work",
                    "phase": "hist_p1",
                    "start_date": _to_ymd(start_dt),
                    "end_date": _to_ymd(end_dt),
                    "operation_task_list": [
                        {
                            "id": f"hist{idx}_op",
                            "name": "Generic Historic Work",
                            "operation": "hist_op_generic",
                            "workload_days": len(sorted_dates),
                        }
                    ],
                }
            ],
        })

    # -------- sample assignment_list (attach workers to hist tasks) --------
    # This is deliberately approximate: it checks which worker rows contain
    # the module name string and uses those workers for that module.
    assignments = []
    hist_names = list(hist_map_dates.keys())
    name_to_hist = {name: f"hist{idx+1}" for idx, name in enumerate(hist_names)}

    for r in range(worker_start_row, n_rows):
        company = df.iat[r, 0]
        name = df.iat[r, 1]
        if pd.isna(name):
            continue

        worker_key = (str(company).strip(), str(name).strip())
        worker_id = worker_id_map.get(worker_key)
        if not worker_id:
            continue

        found_hist = None
        for c in range(first_date_col, last_date_col + 1):
            v = df.iat[r, c]
            if isinstance(v, str):
                for hist_name in hist_names:
                    if hist_name in v:
                        found_hist = hist_name
                        break
            if found_hist:
                break

        if not found_hist:
            continue

        hist_id = name_to_hist[found_hist]
        dates = sorted(hist_map_dates[found_hist])
        if not dates:
            continue

        work_date_list = [{"hour": 8, "date": _to_ymd(d)} for d in dates]
        assignments.append({
            "worker": worker_id,
            "operation_task": f"{hist_id}_op",
            "start_date": _to_ymd(dates[0]),
            "end_date": _to_ymd(dates[-1]),
            "work_date_list": work_date_list,
            "plan_flexibility": "Fixed",
        })

        # Ensure we generate at least 10 assignments, or one per hist task
        if len(assignments) >= max(10, len(hist_tasks)):
            break

    # -------- plan_range from all date columns --------
    all_dates = list(date_row[date_cols].dropna())
    all_dates.sort()
    plan_range = {
        "start_date": _to_ymd(all_dates[0]),
        "end_date": _to_ymd(all_dates[-1]),
    }

    return {
        "worker_company_list": worker_company_list,
        "worker_list": worker_list,
        "hist_tasks": hist_tasks,
        "assignments": assignments,
        "plan_range": plan_range,
    }


# ============================================================
# F22_Tool Schedule: operations + tool tasks
# ============================================================

def parse_tool_schedule(path: str, sheet_name: str = "F22_Tool Schedule"):
    xls = pd.ExcelFile(path)
    df = xls.parse(sheet_name, header=None)
    n_rows, n_cols = df.shape

    # -------- find operation header row (Power, Gas, Exh, …) --------
    op_row_idx = None
    for r in range(0, 10):
        row = df.iloc[r]
        non_empty = [v for v in row if isinstance(v, str) and v.strip()]
        if len(non_empty) >= 5:
            op_row_idx = r
            break

    if op_row_idx is None:
        raise RuntimeError("Could not find operation header row in F22_Tool Schedule.")

    op_row = df.iloc[op_row_idx]
    op_cols = [c for c, v in enumerate(op_row) if isinstance(v, str) and v.strip()]
    op_names = [str(op_row[c]).strip() for c in op_cols]

    # -------- operations for EnvConfig.workflow_list[wf_tool] --------
    operations = []
    op_id_map = {}
    for idx, (col, name) in enumerate(zip(op_cols, op_names), start=1):
        oid = f"f22op{idx}"
        op_id_map[col] = oid
        operations.append({
            "id": oid,
            "name": name,
            "work_hours": [8],
            "min_worker_num": 1,
            "max_worker_num": 4,
        })

    # -------- create tool tasks from rows below header --------
    tool_tasks = []
    all_dates = []

    for r in range(op_row_idx + 2, n_rows):
        location = df.iat[r, 0]
        tsmc_tool = df.iat[r, 2]
        screen_tool = df.iat[r, 3]

        if pd.isna(location) and pd.isna(tsmc_tool) and pd.isna(screen_tool):
            continue

        # Build readable name from SCREEN/TSMC tool
        name_parts = []
        for v in (screen_tool, tsmc_tool):
            if isinstance(v, str) and v.strip():
                name_parts.append(v.strip())
        if not name_parts:
            continue
        task_name = " / ".join(name_parts)

        op_task_list = []
        start_dates = []
        end_dates = []

        for col in op_cols:
            cell = df.iat[r, col]
            if pd.isna(cell):
                continue

            op_id = op_id_map[col]
            start_dt = None
            end_dt = None

            if isinstance(cell, pd.Timestamp):
                start_dt = end_dt = cell
            elif isinstance(cell, str):
                text = cell.strip()
                if "\n" in text:
                    # e.g. "2025/01/03\n>1/7"
                    first, second = text.split("\n", 1)
                    first = first.strip()
                    second = second.strip().lstrip(">")
                    dt1 = pd.to_datetime(first, errors="coerce")
                    if isinstance(dt1, pd.Timestamp):
                        start_dt = dt1
                        dt2 = pd.to_datetime(second, errors="coerce")
                        if isinstance(dt2, pd.Timestamp):
                            if dt2.year != dt1.year:
                                dt2 = dt2.replace(year=dt1.year)
                            end_dt = dt2
                        else:
                            end_dt = dt1
                else:
                    dt = pd.to_datetime(text, errors="coerce")
                    if isinstance(dt, pd.Timestamp):
                        start_dt = end_dt = dt

            if start_dt is None or end_dt is None:
                continue

            num_days = max(1, (end_dt.date() - start_dt.date()).days + 1)
            op_task_list.append({
                "id": f"f22_{len(tool_tasks)+1}_op_{op_id}",
                "name": op_row[col],
                "operation": op_id,
                "workload_days": int(num_days),
            })
            start_dates.append(start_dt)
            end_dates.append(end_dt)

        if not op_task_list:
            continue

        task_start = min(start_dates)
        task_end = max(end_dates)

        all_dates.append(task_start)
        all_dates.append(task_end)

        tool_tasks.append({
            "id": f"f22_{len(tool_tasks)+1}",
            "name": task_name,
            "workflow": "wf_tool",
            "fab": "f_tw",
            "phase_task_list": [
                {
                    "id": f"f22_{len(tool_tasks)+1}_p1",
                    "name": "Tool Setup",
                    "phase": "tool_p1",
                    "start_date": _to_ymd(task_start),
                    "end_date": _to_ymd(task_end),
                    "operation_task_list": op_task_list,
                }
            ],
        })

    return {
        "operations": operations,
        "tool_tasks": tool_tasks,
        "date_list": all_dates,
    }


# ============================================================
# Build EnvConfig + Schedule and dump YAML (wide output)
# ============================================================

def build_env_and_schedule(
    su_others_path: str,
    tool_schedule_path: str,
    envconfig_out: str = "EnvConfig_from_excel.yaml",
    schedule_out: str = "Schedule_from_excel.yaml",
):
    su_data = parse_su_others(su_others_path)
    tool_data = parse_tool_schedule(tool_schedule_path)

    # ---------- ENVIRONMENT ----------
    environment = {
        "workflow_list": [
            {
                "id": "wf_hist",
                "name": "Historic Onsite Work",
                "phase_list": [
                    {
                        "id": "hist_p1",
                        "name": "Historic Phase",
                        "operation_list": [
                            {
                                "id": "hist_op_generic",
                                "name": "Generic Onsite Work",
                                "work_hours": [8],
                                "min_worker_num": 1,
                                "max_worker_num": 4,
                            }
                        ],
                    }
                ],
            },
            {
                "id": "wf_tool",
                "name": "F22 Tool Schedule",
                "phase_list": [
                    {
                        "id": "tool_p1",
                        "name": "Tool Phase",
                        "operation_list": tool_data["operations"],
                    }
                ],
            },
        ],
        "fab_list": [
            {
                "id": "f_tw",
                "name": "Taiwan Fab (generic)",
                "region": "r_tw",
                "customer_company": "c_tsmc",
                "unavailable_dates": [],
            }
        ],
        "region_list": [
            {
                "id": "r_tw",
                "name": "Taiwan",
                "max_stay_on": 90,
                "max_annual_stay": 240,
                "stay_off_interval": 3,
                "unavailable_dates": [
                    {"weekly": {"weekdays": ["sat", "sun"]}},
                ],
            }
        ],
        "customer_company_list": [
            {
                "id": "c_tsmc",
                "name": "TSMC (generic)",
                "unavailable_dates": [],
            }
        ],
        "worker_company_list": su_data["worker_company_list"],
        "worker_list": su_data["worker_list"],
        "transite_day_map": [],
    }

    # ---------- SCHEDULE ----------
    # Merge plan ranges from SU_Others + F22
    all_dates = []
    all_dates.append(pd.to_datetime(su_data["plan_range"]["start_date"]))
    all_dates.append(pd.to_datetime(su_data["plan_range"]["end_date"]))
    all_dates.extend(tool_data["date_list"])
    all_dates = [d for d in all_dates if isinstance(d, pd.Timestamp)]
    all_dates.sort()

    if all_dates:
        plan_range = {
            "start_date": _to_ymd(all_dates[0]),
            "end_date": _to_ymd(all_dates[-1]),
        }
    else:
        plan_range = su_data["plan_range"]

    workflow_task_list = []
    workflow_task_list.extend(su_data["hist_tasks"])
    workflow_task_list.extend(tool_data["tool_tasks"])

    schedule = {
        "plan_range": plan_range,
        "workflow_task_list": workflow_task_list,
        "assignment_list": su_data["assignments"],
    }

    env_root = {"environment": environment}
    sch_root = {"schedule": schedule}

    # ---------- YAML dump with big line width and no anchors ----------
    class NoAliasDumper(yaml.SafeDumper):
        def ignore_aliases(self, data):
            return True

    with open(envconfig_out, "w", encoding="utf-8") as f:
        yaml.dump(
            env_root,
            f,
            Dumper=NoAliasDumper,
            sort_keys=False,
            allow_unicode=True,
            width=4096,   # make lines very wide so it doesn't wrap too aggressively
        )

    with open(schedule_out, "w", encoding="utf-8") as f:
        yaml.dump(
            sch_root,
            f,
            Dumper=NoAliasDumper,
            sort_keys=False,
            allow_unicode=True,
            width=4096,
        )

    return env_root, sch_root


# ============================================================
# Example CLI entrypoint
# ============================================================

if __name__ == "__main__":
    # Update these paths to where your Excel files actually are
    su_file = "20251201_2 SU_Others.xlsm"
    tool_file = "20260105 台湾出張者予定_2025latest.xlsx"

    su_path = Path(su_file)
    tool_path = Path(tool_file)

    if su_path.exists() and tool_path.exists():
        build_env_and_schedule(str(su_path), str(tool_path))
        print("EnvConfig_from_excel.yaml and Schedule_from_excel.yaml have been written.")
    else:
        print("Please update su_file and tool_file paths at the bottom of this script.")
