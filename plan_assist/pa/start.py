"""01_StartSchedule: tools 2-2-3, 2-4, 2-5 and 2-6 in one run, from two files of the user.

run_start() has no window and no command line: it takes the two files and the date, works in Work/<YYYYMMDD>/ and reports every line
through `log`.  The window (ui.py) and the command line (start_schedule.py) both call it.
"""
from __future__ import annotations

import json
import shutil
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path

from . import weeks
from .settings import SOLVER_KEYS, SettingsError
from .tools import ToolRunner

STEPS = ("2-2-3", "2-4", "2-5", "2-6")
EXCEL = (".xlsm", ".xlsx")


class StartError(Exception):
    """The run stops here; `step` names where."""

    def __init__(self, step, message):
        super().__init__(message)
        self.step = step
        self.message = message


@dataclass
class StartResult:
    ok: bool
    step: str                  # the step that stopped the run, or "done"
    message: str
    week_dir: Path = None
    run_id: str = None
    stage: str = None          # ready (03 files made, not sent) | submitted | failed


class Emitter:
    """Every line goes to the callback (window / terminal) and, once attached, to the log file with a time."""

    def __init__(self, callback=None):
        self.callback = callback or (lambda line: None)
        self.file = None

    def attach(self, path):
        Path(path).parent.mkdir(parents=True, exist_ok=True)
        self.file = open(path, "a", encoding="utf-8")

    def __call__(self, line):
        self.callback(line)
        if self.file:
            self.file.write(f"{datetime.now():%H:%M:%S} {line}\n")
            self.file.flush()

    def close(self):
        if self.file:
            self.file.close()
            self.file = None


def write_config(path, header, pairs):
    """A key = value settings file for a tool (only the settings that have a value)."""
    lines = [f"# {header}", "# written by plan_assist; this file is what the tool was run with"]
    lines += [f"{k} = {v}" for k, v in pairs if v not in (None, "")]
    Path(path).write_text("\n".join(lines) + "\n", encoding="utf-8")


def check_inputs(settings, su_others, seiban_list, submit, runner):
    """Everything that can be checked before the slow steps start; raises StartError."""
    for label, path in (("SU_Others", su_others), ("seiban list", seiban_list)):
        if not path or not Path(path).is_file():
            raise StartError("check", f"the {label} file was not found: {path or '(not chosen)'}")
        if Path(path).suffix.lower() not in EXCEL:
            raise StartError("check", f"the {label} file must be an Excel file ({', '.join(EXCEL)}): {Path(path).name}")
    if Path(su_others).resolve() == Path(seiban_list).resolve():
        raise StartError("check", "the SU_Others file and the seiban list file are the same file")
    missing = runner.missing(STEPS if submit else STEPS[:-1])
    if missing:
        raise StartError("check", "the weekly_planning scripts are not complete; not found: " + "; ".join(missing))
    if submit:
        mode = settings.text("mode", "online")
        if mode not in ("online", "local"):
            raise StartError("check", f"mode must be online or local (got {mode!r}); see {settings._where()}")
        url = settings.text("azure_url" if mode == "online" else "local_url", "http://localhost:3001" if mode == "local" else "")
        if not url or "<" in url:
            raise StartError("check", f"the address of the solver is not set: fill in {'azure_url' if mode == 'online' else 'local_url'} "
                                      f"in {settings._where()} (or use --no-submit)")


def run_start(settings, su_others, seiban_list, day, log=None, runner=None, submit=None, overwrite=False):
    """Runs the whole of part 1 for the week of `day` (a date). Never raises; the answer says what happened."""
    emit = Emitter(log)
    runner = runner or ToolRunner(settings.tools_dir)
    week = None
    try:
        submit = settings.flag("submit", True) if submit is None else submit
        check_inputs(settings, su_others, seiban_list, submit, runner)
        name = weeks.week_name(day)
        week = weeks.week_path(settings.work_dir, name)

        if week.exists() and weeks.was_submitted(week) and not overwrite:
            raise StartError("check", f"week {name} was already sent to the solver; run it again only if you mean it "
                                      f"(the old files are kept in steps_old_* / logs_old_*)")
        stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        moved = weeks.archive_old(week, stamp) if week.exists() else []
        steps_dir, input_dir = week / "steps", week / "input"
        for d in (steps_dir, input_dir, week / "logs"):
            d.mkdir(parents=True, exist_ok=True)
        emit.attach(week / "logs" / "start_log.txt")
        emit(f"Week {name}  ({week})")
        for key in settings.unknown:
            emit(f"  Note: the setting '{key}' in {settings.path.name if settings.path else 'the settings'} is not known and is ignored.")
        if moved:
            emit(f"  the files of the earlier run were kept as {', '.join(moved)}")
        weeks.write_status(week, stage="started", step="check", message="", run_id=None, started=datetime.now().isoformat(timespec="seconds"),
                           su_others=str(su_others), seiban_list=str(seiban_list))

        # --- the user's two files are copied; the originals are never touched
        su_copy = input_dir / f"01_SU_Others{Path(su_others).suffix.lower()}"
        sb_copy = input_dir / f"02_SeibanList{Path(seiban_list).suffix.lower()}"
        shutil.copy2(su_others, su_copy)
        shutil.copy2(seiban_list, sb_copy)

        # --- what last week left
        previous = weeks.previous_week(settings.work_dir, name)
        skill_file = None
        if previous:
            emit(f"Previous week: {previous.name}")
            prev_flex = previous / weeks.FLEXIBLE_FILE
            prev_env = previous / weeks.ENV_FILE
        else:
            emit("First week: there is no earlier week with FlexibleSchedule.yaml and EnvConfig.yaml.")
            prev_flex = steps_dir / "prev_FlexibleSchedule_none.yaml"
            prev_flex.write_text("{}\n", encoding="utf-8")
            start_env = settings.master_start_env()
            if start_env:
                prev_env = steps_dir / "prev_EnvConfig_start.yaml"
                shutil.copy2(start_env, prev_env)
                emit(f"  starting EnvConfig: {start_env.name}")
            else:
                prev_env = steps_dir / "prev_EnvConfig_none.yaml"
                prev_env.write_text("", encoding="utf-8")
                emit("  WARN: no EnvConfig_start.yaml in Master: the EnvConfig starts empty (regions, fabs and workflows are made "
                     "from the seiban list only).")
            skill_file = settings.master_skill_file()
            if skill_file:
                emit(f"  skill workbook: {skill_file.name}")
            else:
                emit("  WARN: no skill workbook in Master: workers get skills from their 担当職種 only.")
        weeks.write_status(week, previous_week=previous.name if previous else None, first_week=previous is None)

        def step(key, label, argv, expect):
            number = STEPS.index(key) + 1
            emit(f"== [{number}/{len(STEPS) if submit else len(STEPS) - 1}] {label}")
            weeks.write_status(week, step=key)
            result = runner.run(key, argv, lambda line: emit("   " + line))
            if result.rc != 0:
                raise StartError(key, f"{label} stopped: {result.last_error()}")
            for f in expect:
                if not Path(f).exists():
                    raise StartError(key, f"{label} finished but did not write {Path(f).name}")

        # --- 2-2-3
        argv = ["--prev-flexible", prev_flex, "--prev-env", prev_env, "--su-others", su_copy, "--seiban-list", sb_copy,
                "--out-dir", steps_dir, "--today", day.strftime("%Y/%m/%d"), "--report"]
        if skill_file:
            argv += ["--skill-file", skill_file]
        for opt, key in (("--su-sheets", "su_sheet"), ("--skill-sheet", "skill_sheet"), ("--seiban-sheet", "seiban_sheet"),
                         ("--default-skill-level", "default_skill_level")):
            if settings.has(key):
                argv += [opt, settings.text(key)]
        step("2-2-3", "2-2-3 prepare_schedule (what the planner changed, the EnvConfig and the Schedule)", argv,
             [steps_dir / "01_EnvConfig.yaml", steps_dir / "01_Schedule.yaml"])

        # --- 2-4
        cfg = steps_dir / "config_2-4.txt"
        write_config(cfg, "settings of tool 2-4 (apply_seiban_list)",
                     [("schedule", steps_dir / "01_Schedule.yaml"), ("env", steps_dir / "01_EnvConfig.yaml"),
                      ("seiban_list", sb_copy), ("out_dir", steps_dir), ("report", "true")])
        step("2-4", "2-4 apply_seiban_list (new and deleted seiban)", ["--config", cfg],
             [steps_dir / "02_Schedule.yaml", steps_dir / "02_EnvConfig.yaml"])

        # --- 2-5
        step("2-5", "2-5 set_plan_period (plan period, work before it becomes Fixed)",
             ["--input-schedule", steps_dir / "02_Schedule.yaml", "--input-env-config", steps_dir / "02_EnvConfig.yaml",
              "--output-schedule", steps_dir / "03_Schedule.yaml", "--output-env-config", steps_dir / "03_EnvConfig.yaml"],
             [steps_dir / "03_Schedule.yaml", steps_dir / "03_EnvConfig.yaml"])

        # --- 2-6
        run_id = None
        if submit:
            cfg = steps_dir / "config_2-6.txt"
            solver = {k: settings.text(k) for k in SOLVER_KEYS}
            solver["mode"] = solver["mode"] or "online"             # the same default the check above used
            write_config(cfg, "settings of tool 2-6 (submit_to_timefold)",
                         [("schedule", steps_dir / "03_Schedule.yaml"), ("env", steps_dir / "03_EnvConfig.yaml"),
                          ("out_dir", steps_dir)] + list(solver.items()))
            step("2-6", "2-6 submit_to_timefold (send to the solver)", ["--config", cfg], [steps_dir / weeks.RUN_INFO_FILE])
            try:
                run_id = json.loads((steps_dir / weeks.RUN_INFO_FILE).read_text(encoding="utf-8")).get("runId")
            except (OSError, ValueError):
                run_id = None
            message = f"Sent to the solver as run {run_id}. When it has finished, run 02_UpdateSchedule."
            stage = "submitted"
        else:
            message = "Prepared, not sent (submit is off). The solver input is steps/03_Schedule.yaml and steps/03_EnvConfig.yaml."
            stage = "ready"
        emit(message)
        weeks.write_status(week, stage=stage, step="done", message=message, run_id=run_id)
        return StartResult(True, "done", message, week, run_id, stage)
    except StartError as e:
        emit(f"STOPPED at {e.step}: {e.message}")
        if week is not None and week.exists() and e.step != "check":
            weeks.write_status(week, stage="failed", step=e.step, message=e.message)
        return StartResult(False, e.step, e.message, week, stage="failed")
    except (SettingsError, OSError) as e:
        emit(f"STOPPED: {e}")
        if week is not None and week.exists():
            weeks.write_status(week, stage="failed", step="error", message=str(e))
        return StartResult(False, "error", str(e), week, stage="failed")
    finally:
        emit.close()
