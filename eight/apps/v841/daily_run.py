#!/usr/bin/env python3
"""
daily_run.py

Incremental scheduler driver with "evaluate every X working days" and
cutoff = (last module start + 1 working day).

Behavior:
- Finds project root (directory containing pom.xml).
- Uses src/main/resource/EnvConfig.yaml and Schedule.yaml.
- Simulates over plan_range[start_date..end_date] but:
    * Only saves Schedule_YYYYMMDD.yaml (and runs solver)
      on:
        - the first day (baseline), and
        - days when new modules are actually added.
- For each evaluation day D:
    1) Reads current Schedule.yaml and finds last module start date.
    2) Sets in update_config:
         CUTOFF_DATE_STR      = (last_module_start + 1 working day)
         CURRENT_SIM_DAY_STR  = D
         MODULE_SEED_OFFSET   = eval_index (0,1,2,...)
    3) Runs update_schedule2.main() to:
         - set plan_flexibility (before cutoff = Fixed, after = Flexible)
         - maybe append new modules (starting after last module start)
         - update plan_range
    4) If (D == plan_start OR modules_added > 0):
         - Runs Maven Java solver (EmployeeSchedule).
         - Copies Schedule.yaml to:
               src/main/resource/schedule_outputs/Schedule_YYYYMMDD.yaml
- Evaluation days are spaced by EQ_EVAL_DAYS working days.
"""

import os
import sys
import shutil
import subprocess
import datetime as dt
from pathlib import Path

import yaml

# Relative paths from project root
RESOURCE_REL = Path("src") / "main" / "resource"
ENV_NAME = "EnvConfig.yaml"
SCHEDULE_NAME = "Schedule.yaml"
OUTPUT_DIR_NAME = "schedule_outputs"


def find_project_root(start: Path) -> Path:
    """Walk upwards until we find pom.xml."""
    for p in [start] + list(start.parents):
        if (p / "pom.xml").exists():
            return p
    raise SystemExit(f"Could not find pom.xml above {start}")


def parse_ymd(s: str) -> dt.date:
    return dt.datetime.strptime(s.strip().replace("-", "/"), "%Y/%m/%d").date()


def ymd(d: dt.date) -> str:
    return d.strftime("%Y/%m/%d")


def load_schedule(path: Path) -> dict:
    with path.open("r", encoding="utf-8") as f:
        return yaml.safe_load(f) or {}


def is_weekend(d: dt.date) -> bool:
    # Monday=0 ... Sunday=6
    return d.weekday() >= 5


def next_working_day(d: dt.date) -> dt.date:
    nd = d + dt.timedelta(days=1)
    while is_weekend(nd):
        nd += dt.timedelta(days=1)
    return nd


def advance_working_days(d: dt.date, n: int) -> dt.date:
    nd = d
    for _ in range(n):
        nd = next_working_day(nd)
    return nd


def main():
    # ---- Locate project + resource dir ----
    here = Path(__file__).resolve()
    project_root = find_project_root(here)
    resource_dir = project_root / RESOURCE_REL
    env_path = resource_dir / ENV_NAME
    sched_path = resource_dir / SCHEDULE_NAME
    out_dir = resource_dir / OUTPUT_DIR_NAME

    if not env_path.exists() or not sched_path.exists():
        raise SystemExit(
            f"Expected {env_path} and {sched_path} to exist. "
            f"Please check your project layout."
        )

    out_dir.mkdir(parents=True, exist_ok=True)

    # Make Python see update_config.py and update_schedule2.py in resource_dir
    sys.path.insert(0, str(resource_dir))
    os.chdir(resource_dir)

    import update_config as cfg
    import update_schedule2 as upd

    # ---------- Initial load to get plan_range ----------
    sched_root = load_schedule(sched_path)
    sched = sched_root.get("schedule") or sched_root
    plan_range = sched.get("plan_range") or {}

    if "start_date" not in plan_range or "end_date" not in plan_range:
        raise SystemExit("schedule.plan_range.start_date/end_date not found in Schedule.yaml")

    plan_start = parse_ymd(plan_range["start_date"])
    plan_end = parse_ymd(plan_range["end_date"])

    # Existing modules → find initial last start date
    modules0, last_start0, last_end0 = upd.collect_existing_modules(sched_root)

    # EQ_EVAL_DAYS (X working days per evaluation)
    step_days = max(1, int(getattr(cfg, "EQ_EVAL_DAYS", 1)))

    # First evaluation day:
    #   step_days = 1 → current = cutoff0
    #   step_days = 2 → current = cutoff0 + 1 working day, etc.
    if last_start0 is not None:
        cutoff0 = next_working_day(last_start0)
    else:
        # if no modules yet, treat plan_start as base and cutoff = plan_start
        cutoff0 = plan_start

    current = advance_working_days(cutoff0, step_days - 1)

    print(f"[INFO] Initial plan_range: {plan_start} .. {plan_end}")
    print("[INFO] Lookback is DISABLED (no trimming).")

    # Choose mvn executable (Windows vs others)
    mvn_exe = "mvn.cmd" if os.name == "nt" else "mvn"

    eval_index = 0  # increases only on evaluation steps (for MODULE_SEED_OFFSET)

    while True:
        if current > plan_end:
            print(f"[DONE] Reached plan_end {plan_end}, stop.")
            break

        if is_weekend(current):
            # This should rarely trigger because current advances in working days,
            # but keep it safe.
            print(f"[SKIP] {current} (weekend)")
            current += dt.timedelta(days=1)
            continue

        # ----- Load current schedule & compute cutoff from *current* last_start -----
        sched_before = load_schedule(sched_path)
        modules_before, last_start_before, last_end_before = upd.collect_existing_modules(sched_before)
        before_count = len(modules_before)

        if last_start_before is not None:
            cutoff = next_working_day(last_start_before)
        else:
            cutoff = plan_start  # no modules yet

        cfg.CUTOFF_DATE_STR = ymd(cutoff)
        cfg.CURRENT_SIM_DAY_STR = ymd(current)
        cfg.MODULE_SEED_OFFSET = eval_index

        print("\n==============================================")
        print(f"[DAY] {current}  | cutoff={cfg.CUTOFF_DATE_STR}  | modules_before={before_count}")
        print("==============================================")

        # 1) Update Schedule.yaml: mark Fixed/Flexible + maybe extend modules
        upd.main()

        # 2) Reload and inspect AFTER
        sched_after = load_schedule(sched_path)
        modules_after, last_start_after, last_end_after = upd.collect_existing_modules(sched_after)
        after_count = len(modules_after)

        # just for logging, read the current plan_range from the file
        sched2 = sched_after.get("schedule") or sched_after
        plan_range2 = sched2.get("plan_range") or {}

        modules_added = after_count - before_count
        print(f"[INFO] modules_after={after_count}, modules_added_today={modules_added}")
        print(f"[INFO] plan_range now: {plan_range2.get('start_date')} .. {plan_range2.get('end_date')}")

        # 3) Decide whether to run solver and snapshot
        assignment_list_now = sched2.get("assignm_list") or []
        run_solver = (current == plan_start) or (modules_added > 0) or (len(assignment_list_now) == 0)

        if run_solver:
            # ----------------------------------------------
            # NEW: Run Java solver directly (no Maven)
            # ----------------------------------------------
            
            java_cmd = [
                "java",
                "-Xms4g", "-Xmx8g",        # adjust depending on your RAM
                "-cp", "target/classes;target/dependency/*",   # Linux → ":" instead of ";"
                "com.yourorg.scheduler.RunEmployeeScheduleOnce",
                f"src/main/resource/{ENV_NAME}",
                f"src/main/resource/{SCHEDULE_NAME}",
            ]
            
            print(f"[RUN-JAVA] {' '.join(java_cmd)}  (cwd={project_root})")
            subprocess.run(java_cmd, cwd=project_root, check=True)
            
            # ----------------------------------------------
            # OLD MAVEN METHOD (KEEPED AS COMMENT)
            # ----------------------------------------------
"""
mvn_cmd = [
    mvn_exe,
    "-q",
    "-DskipTests",
    "exec:java",
    f"-Dexec.args=src/main/resource/{ENV_NAME} src/main/resource/{SCHEDULE_NAME}",
]
print(f"[RUN] {' '.join(mvn_cmd)}  (cwd={project_root})")
env = os.environ.copy()
env["MAVEN_OPTS"] = "-Xms4g -Xmx8g"
subprocess.run(mvn_cmd, cwd=project_root, check=True, env=env)
"""
# ----------------------------------------------

            out_name = f"Schedule_{current.strftime('%Y%m%d')}.yaml"
            out_path = out_dir / out_name
            shutil.copy2(sched_path, out_path)
            print(f"[OUT] Wrote {out_path.relative_to(project_root)}")
        else:
            print("[SKIP] No new modules today and not first day -> skip solver/snapshot.")

        # --- stop if EQ_NUM reached (AFTER we solved for this day) ---
        try:
            target_modules = int(cfg.EQ_NUM)
        except Exception:
            target_modules = None

        if target_modules is not None and after_count >= target_modules:
            print(f"[DONE] target modules reached: {after_count} / {target_modules}.")
            break

        # advance to the next evaluation day (every step_days working days)
        current = advance_working_days(current, step_days)
        eval_index += 1  # for MODULE_SEED_OFFSET

    print("\n[DONE] Daily run finished.")


if __name__ == "__main__":
    main()

