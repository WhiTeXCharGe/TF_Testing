"""The week folders of the programs the weekly user runs (plan_assist): Work/<YYYYMMDD>/ with input/, steps/, logs/ and the files of the next week.

No state is kept outside the week folders: the previous week is found by looking at them.
"""
from __future__ import annotations

import json
import re
from datetime import datetime
from pathlib import Path

WEEK_NAME = re.compile(r"^\d{8}$")
STATUS_FILE = "week_status.json"
RUN_INFO_FILE = "run_info.json"                 # written by tool 2-6 into steps/
FLEXIBLE_FILE = "FlexibleSchedule.yaml"         # written by part 2 into the week folder
ENV_FILE = "EnvConfig.yaml"                     # (the EnvConfig that the solver ran with)
KEY_FILES = (FLEXIBLE_FILE, ENV_FILE)           # what the next week needs


def week_name(day):
    return day.strftime("%Y%m%d")


def week_path(work_dir, name):
    return Path(work_dir) / name


def is_completed(path):
    """A week the next one can start from: it has FlexibleSchedule.yaml and EnvConfig.yaml (not empty)."""
    path = Path(path)
    return all((path / f).is_file() and (path / f).stat().st_size > 0 for f in KEY_FILES)


def previous_week(work_dir, name):
    """The newest completed week folder whose name is before `name` (None: this is the first week)."""
    work = Path(work_dir)
    if not work.is_dir():
        return None
    older = sorted((p for p in work.iterdir() if p.is_dir() and WEEK_NAME.match(p.name) and p.name < name), key=lambda p: p.name)
    return next((p for p in reversed(older) if is_completed(p)), None)


def read_status(path):
    try:
        return json.loads((Path(path) / STATUS_FILE).read_text(encoding="utf-8"))
    except (OSError, ValueError):
        return {}


def write_status(path, **fields):
    """Adds `fields` to week_status.json (kept keys stay) and stamps the time."""
    path = Path(path)
    path.mkdir(parents=True, exist_ok=True)
    status = read_status(path)
    status.update(fields)
    status["updated"] = datetime.now().isoformat(timespec="seconds")
    (path / STATUS_FILE).write_text(json.dumps(status, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    return status


def was_submitted(path):
    """The week was already sent to the solver."""
    path = Path(path)
    return (path / "steps" / RUN_INFO_FILE).is_file() or read_status(path).get("stage") in ("submitted", "finished")


def archive_old(path, stamp):
    """Renames steps/ and logs/ of an earlier run of the week (steps_old_<stamp>, logs_old_<stamp>); nothing is deleted."""
    moved = []
    for name in ("steps", "logs"):
        old = Path(path) / name
        if old.exists():
            target = Path(path) / f"{name}_old_{stamp}"
            old.rename(target)
            moved.append(target.name)
    return moved
