#!/usr/bin/env python3
"""
daily_run.py

Incremental day-by-day scheduler driver.

Behavior:
- Finds project root (directory containing pom.xml).
- Uses src/main/resource/EnvConfig.yaml and Schedule.yaml.
- First day:
    * Run update_schedule2 WITHOUT adding new modules
      (only plan_range + Fixed/Flexible updates).
    * Run Java solver.
    * Output Schedule_YYYYMMDD.yaml.
- Following days (working days only):
    * Set cutoff = max(plan_start, today - MODULE_HISTORY_LOOKBACK_DAYS).
    * Set cfg.CURRENT_SIM_DAY_STR = today (string).
    * Call update_schedule2.main() to:
         - mark assignments Fixed/Flexible
         - maybe add new modules (0 or more) for *this* day only
    * Reload Schedule.yaml and count modules.
    * If module count increased → run Java solver and export Schedule_YYYYMMDD.yaml.
    * If no new modules → skip solver & export (nothing “interesting” happened).
"""

import os
import sys
import shutil
import subprocess
import datetime as dt
from pathlib import Path

import yaml

# ================== CONFIG ==================

MODULE_HISTORY_LOOKBACK_DAYS = 5

RESOURCE_REL = Path("src") / "main" / "resource"
ENV_NAME = "EnvConfig.yaml"
SCHEDULE_NAME = "Schedule.yaml"
OUTPUT_DIR_NAME = "schedule_outputs"


# ================== HELPERS ==================

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
    return d.weekday() >= 5  # 5=Sat, 6=Sun


def count_modules_from_root(sched_root: dict) -> int:
    """Count e1, e2, ... in workflow_task_list."""
    sched = sched_root.get("schedule") or sched_root
    wf_list = sched.get("workflow_task_list") or []
    c = 0
    for mod in wf_list:
        mid = str(mod.get("id") or "")
        if mid.startswith("e") and mid[1:].isdigit():
            c += 1
    return c


# ================== MAIN FLOW ==================

def main():
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
    os.chdir(resource_dir)  # so update_* sees EnvConfig.yaml / Schedule.yaml

    import update_config as cfg
    import update_schedule2 as upd

    # ---------- Initial load to get plan_range + initial module count ----------
    sched_root = load_schedule(sched_path)
    sched = sched_root.get("schedule") or sched_root
    plan_range = sched.get("plan_range") or {}

    if "start_date" not in plan_range or "end_date" not in plan_range:
        raise SystemExit("schedule.plan_range.start_date/end_date not found in Schedule.yaml")

    plan_start = parse_ymd(plan_range["start_date"])
    plan_end = parse_ymd(plan_range["end_date"])
    initial_module_count = count_modules_from_root(sched_root)

    print(f"[INFO] Initial plan_range: {plan_start} .. {plan_end}")
    print(f"[INFO] Initial modules: {initial_module_count}")

    mvn_exe = "mvn.cmd" if os.name == "nt" else "mvn"

    current = plan_start
    prev_module_count = initial_module_count
    first_day_done = False

    while current <= plan_end:
        if is_weekend(current):
            print(f"[SKIP] {current} (weekend)")
            current += dt.timedelta(days=1)
            continue

        # cutoff: keep MODULE_HISTORY_LOOKBACK_DAYS flexible
        cutoff = current - dt.timedelta(days=MODULE_HISTORY_LOOKBACK_DAYS)
        if cutoff < plan_start:
            cutoff = plan_start

        print("\n==============================================")
        print(f"[DAY] {current}  | cutoff={ymd(cutoff)}")
        print("==============================================")

        # Tell update_* which logical day this is
        cfg.CUTOFF_DATE_STR = ymd(cutoff)
        cfg.CURRENT_SIM_DAY_STR = ymd(current)

        # First day: freeze EQ_NUM so NO new modules are created
        if not first_day_done:
            # reload current module count
            sched_root = load_schedule(sched_path)
            prev_module_count = count_modules_from_root(sched_root)

            orig_eq_num = cfg.EQ_NUM
            cfg.EQ_NUM = prev_module_count
            print(f"[INIT] First day: forcing EQ_NUM={cfg.EQ_NUM} (no new modules).")
            upd.main()
            cfg.EQ_NUM = orig_eq_num
        else:
            upd.main()

        # After update, reload and count modules
        sched_root = load_schedule(sched_path)
        new_module_count = count_modules_from_root(sched_root)
        print(f"[INFO] Modules: prev={prev_module_count}, now={new_module_count}")

        # Decide whether to run solver
        if not first_day_done:
            # Always run for the first day
            should_solve = True
        else:
            should_solve = new_module_count > prev_module_count

        if not should_solve:
            print("[SKIP] No new modules today → skip solver/export.")
            prev_module_count = new_module_count
            current += dt.timedelta(days=1)
            first_day_done = True
            continue

        # 2) Run Java solver via Maven
        mvn_cmd = [
            mvn_exe,
            "-q",
            "-DskipTests",
            "exec:java",
            f"-Dexec.args=src/main/resource/{ENV_NAME} src/main/resource/{SCHEDULE_NAME}",
        ]
        print(f"[RUN] {' '.join(mvn_cmd)}  (cwd={project_root})")
        subprocess.run(mvn_cmd, cwd=project_root, check=True)

        # 3) Snapshot Schedule.yaml for this day
        out_name = f"Schedule_{current.strftime('%Y%m%d')}.yaml"
        out_path = out_dir / out_name
        shutil.copy2(sched_path, out_path)
        print(f"[OUT] Wrote {out_path.relative_to(project_root)}")

        prev_module_count = new_module_count
        current += dt.timedelta(days=1)
        first_day_done = True

    print("\n[DONE] Daily run finished.")


if __name__ == "__main__":
    main()
