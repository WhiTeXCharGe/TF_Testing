#!/usr/bin/env python3
"""
daily_run.py

Incremental day-by-day scheduler driver.

Behavior:
- Finds project root (directory containing pom.xml).
- Uses src/main/resources/EnvConfig.yaml and Schedule.yaml.
- Each "simulation day":
    1) Sets CUTOFF_DATE_STR in update_config (in memory, no file rewrite).
    2) Runs update_schedule2.main() to:
         - mark old assignments as Fixed/Flexible
         - refresh plan_range.start_date = cutoff
         - extend modules if EQ_NUM not yet reached
    3) Runs Maven Java solver (EmployeeSchedule).
    4) Copies the updated Schedule.yaml to:
         src/main/resources/schedule_outputs/Schedule_YYYYMMDD.yaml
- Skips weekends (Saturday, Sunday).
"""

import os
import sys
import shutil
import subprocess
import datetime as dt
from pathlib import Path

import yaml


# ================== CONFIG ==================

# How many days of history to keep "Flexible" in the Java solver.
# Anything older than (current_day - MODULE_HISTORY_LOOKBACK_DAYS)
# will be marked Fixed by update_schedule2.
MODULE_HISTORY_LOOKBACK_DAYS = 5

# Relative paths from project root
RESOURCES_REL = Path("src") / "main" / "resources"
ENV_NAME = "EnvConfig.yaml"
SCHEDULE_NAME = "Schedule.yaml"
OUTPUT_DIR_NAME = "schedule_outputs"


# ================== HELPERS ==================

def find_project_root(start: Path) -> Path:
    """Walk upwards until we find pom.xml."""
    for p in [start] + list(start.parents):
        if (p / "pom.xml").exists():
            return p
    raise SystemExit("Could not find pom.xml above {}".format(start))


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


# ================== MAIN FLOW ==================

def main():
    here = Path(__file__).resolve()
    project_root = find_project_root(here)
    resources_dir = project_root / RESOURCES_REL
    env_path = resources_dir / ENV_NAME
    sched_path = resources_dir / SCHEDULE_NAME
    out_dir = resources_dir / OUTPUT_DIR_NAME

    if not env_path.exists() or not sched_path.exists():
        raise SystemExit(
            f"Expected {env_path} and {sched_path} to exist. "
            f"Please check your project layout."
        )

    out_dir.mkdir(parents=True, exist_ok=True)

    # Make Python see update_config.py and update_schedule2.py in resources_dir
    sys.path.insert(0, str(resources_dir))
    os.chdir(resources_dir)  # so their relative paths (EnvConfig.yaml, Schedule.yaml) work

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

    print(f"[INFO] Initial plan_range: {plan_start} .. {plan_end}")

    # ---------- (Optional) one-time module extension ----------
    # We let update_schedule2 ensure EQ_NUM modules exist and also
    # normalize plan_range.start_date to the cutoff (initially plan_start).
    cfg.CUTOFF_DATE_STR = ymd(plan_start)
    print(f"[INIT] Running update_schedule2 with cutoff={cfg.CUTOFF_DATE_STR}")
    upd.main()

    # Reload after extension to get final plan_range (may shift end_date)
    sched_root = load_schedule(sched_path)
    sched = sched_root.get("schedule") or sched_root
    plan_range = sched.get("plan_range") or {}
    plan_start = parse_ymd(plan_range["start_date"])
    plan_end = parse_ymd(plan_range["end_date"])

    print(f"[INFO] After extension, plan_range: {plan_start} .. {plan_end}")
    print(f"[INFO] Will run day-by-day over working days in this range.")
    print(f"[INFO] MODULE_HISTORY_LOOKBACK_DAYS = {MODULE_HISTORY_LOOKBACK_DAYS}")

    current = plan_start
    while current <= plan_end:
        if is_weekend(current):
            print(f"[SKIP] {current} (weekend)")
            current += dt.timedelta(days=1)
            continue

        # Compute cutoff date for Fixed / Flexible
        cutoff = current - dt.timedelta(days=MODULE_HISTORY_LOOKBACK_DAYS)
        if cutoff < plan_start:
            cutoff = plan_start

        cfg.CUTOFF_DATE_STR = ymd(cutoff)
        print("\n==============================================")
        print(f"[DAY] {current}  | cutoff={cfg.CUTOFF_DATE_STR}")
        print("==============================================")

        # 1) Update Schedule.yaml: mark Fixed/Flexible, maybe extend modules
        upd.main()

        # 2) Run Java solver via Maven
        mvn_cmd = [
            "mvn",
            "-q",
            "-DskipTests",
            "exec:java",
            # EmployeeSchedule.main(String[] args) reads:
            #   args[0] = EnvConfig path
            #   args[1] = Schedule path
            f"-Dexec.args=src/main/resources/{ENV_NAME} src/main/resources/{SCHEDULE_NAME}",
        ]
        print(f"[RUN] {' '.join(mvn_cmd)}  (cwd={project_root})")
        subprocess.run(mvn_cmd, cwd=project_root, check=True)

        # 3) Snapshot Schedule.yaml for this day
        out_name = f"Schedule_{current.strftime('%Y%m%d')}.yaml"
        out_path = out_dir / out_name
        shutil.copy2(sched_path, out_path)
        print(f"[OUT] Wrote {out_path.relative_to(project_root)}")

        current += dt.timedelta(days=1)

    print("\n[DONE] Daily run finished.")


if __name__ == "__main__":
    main()
