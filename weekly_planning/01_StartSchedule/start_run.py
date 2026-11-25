"""01_StartSchedule: tools 2-2-3, 2-4, 2-5 and 2-6 in one run, from two files of the user.

run_start() has no window and no command line: it takes the two files and the date, works in Work/<YYYYMMDD>/ and reports every line
through `log`.  The window (start_window.py) and the command line (start_schedule.py) both call it.
"""
from __future__ import annotations

import json
import shutil
import sys
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent))      # weekly_planning/ (the folder of common)
from common import week_folders as weeks   # noqa: E402
from common.file_source import FileSource, SourceError, is_link   # noqa: E402
from common.plan_settings import SOLVER_KEYS, SettingsError   # noqa: E402
from common.tool_runner import ToolRunner   # noqa: E402

STEPS = ("2-2-3", "2-4", "2-5", "2-6")
EXCEL = (".xlsm", ".xlsx")
LABEL_SU, LABEL_SEIBAN = "SU_Others", "製番リスト"          # how the two files are called in messages
STEP_TITLES = {
    "2-2-3": "2-2-3 計画の準備（担当者の変更を確認し、EnvConfigとScheduleを作成）",
    "2-4": "2-4 製番リストの反映（新規・削除の製番）",
    "2-5": "2-5 計画期間の設定（期間より前の作業をFixedにする）",
    "2-6": "2-6 ソルバーへ送信",
}
STEP_NAMES = {"check": "入力の確認", "download": "ファイルの取得", "error": "エラー"}      # where a run can stop besides the tools


def step_name(step):
    """What to call the place where a run stopped, for the user."""
    return STEP_NAMES.get(step) or STEP_TITLES.get(step, step).split("（")[0]


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


def check_inputs(settings, su_others, seiban_list, submit, runner, source):
    """Everything that can be checked before the slow steps start; raises StartError."""
    for label, path in ((LABEL_SU, su_others), (LABEL_SEIBAN, seiban_list)):
        try:
            source.check(path, label)
        except SourceError as e:
            raise StartError("check", str(e)) from None
        if not is_link(path) and Path(path).suffix.lower() not in EXCEL:
            raise StartError("check", f"{label}のファイルはExcelファイル（{' / '.join(EXCEL)}）を指定してください: {Path(path).name}")
    if is_link(su_others) or is_link(seiban_list):
        same = str(su_others).strip() == str(seiban_list).strip()
    else:
        same = Path(su_others).resolve() == Path(seiban_list).resolve()
    if same:
        raise StartError("check", f"{LABEL_SU}と{LABEL_SEIBAN}に同じファイルが指定されています。")
    missing = runner.missing(STEPS if submit else STEPS[:-1])
    if missing:
        raise StartError("check", "週次計画のスクリプトがそろっていません。見つからないもの: " + "; ".join(missing))
    if submit:
        mode = settings.text("mode", "online")
        if mode not in ("online", "local"):
            raise StartError("check", f"mode は online か local にしてください（現在: {mode!r}）。設定ファイル: {settings._where()}")
        url = settings.text("azure_url" if mode == "online" else "local_url", "http://localhost:3001" if mode == "local" else "")
        if not url or "<" in url:
            raise StartError("check", f"ソルバーのアドレスが設定されていません。設定ファイル（{settings._where()}）の "
                                      f"{'azure_url' if mode == 'online' else 'local_url'} を記入してください"
                                      f"（ソルバーへ送信しない設定にすることもできます）。")


def run_start(settings, su_others, seiban_list, day, log=None, runner=None, submit=None, overwrite=False, sign_in=None, source=None):
    """Runs the whole of part 1 for the week of `day` (a date). Never raises; the answer says what happened.
    su_others / seiban_list: a path on this PC or a SharePoint link. sign_in(url, code): how the SharePoint sign-in code is shown
    (default: a window); source: a replacement for the FileSource (tests)."""
    emit = Emitter(log)
    runner = runner or ToolRunner(settings.tools_dir)
    source = source or FileSource(settings.text("tenant_id"), settings.text("client_id"), sign_in=sign_in)
    week = None
    try:
        submit = settings.flag("submit", True) if submit is None else submit
        check_inputs(settings, su_others, seiban_list, submit, runner, source)
        name = weeks.week_name(day)
        week = weeks.week_path(settings.work_dir, name)

        if week.exists() and weeks.was_submitted(week) and not overwrite:
            raise StartError("check", f"{name} の週は既にソルバーへ送信済みです。もう一度実行するときは確認（--yes）が必要です。"
                                      f"古いファイルは steps_old_* / logs_old_* に残ります。")
        stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        moved = weeks.archive_old(week, stamp) if week.exists() else []
        steps_dir, input_dir = week / "steps", week / "input"
        for d in (steps_dir, input_dir, week / "logs"):
            d.mkdir(parents=True, exist_ok=True)
        emit.attach(week / "logs" / "start_log.txt")
        emit(f"週 {name}（{week}）")
        for key in settings.unknown:
            emit(f"  注意: 設定ファイル（{settings.path.name if settings.path else '設定'}）の項目 '{key}' は不明なので無視します。")
        if moved:
            emit(f"  前回の実行のファイルは {', '.join(moved)} として残しました。")
        weeks.write_status(week, stage="started", step="check", message="", run_id=None, started=datetime.now().isoformat(timespec="seconds"),
                           su_others=str(su_others), seiban_list=str(seiban_list))

        # --- the user's two files are copied (a SharePoint file is downloaded); the originals are never touched
        copies = {}
        for label, text, stem in ((LABEL_SU, su_others, "01_SU_Others"), (LABEL_SEIBAN, seiban_list, "02_SeibanList")):
            if is_link(text):
                emit(f"{label}をSharePointから取得しています ..." + (" （サインインを求められることがあります）" if not copies else ""))
            try:
                got = source.fetch(text, input_dir, stem, EXCEL)
            except SourceError as e:
                raise StartError("download", f"{label}: {e}") from None
            copies[label] = got.path
            if is_link(text):
                emit(f"  {label}: {got.name}")
        su_copy, sb_copy = copies[LABEL_SU], copies[LABEL_SEIBAN]

        # --- what last week left
        previous = weeks.previous_week(settings.work_dir, name)
        skill_file = None
        if previous:
            emit(f"前週: {previous.name}")
            carry_over = weeks.previous_week_inputs(previous)
            if carry_over is None:
                raise StartError("2-2-3", f"{previous.name} の週の引き継ぎファイルが見つかりません。")
            prev_flex, prev_env = carry_over
        else:
            emit("初回の週です: FlexibleSchedule.yaml と EnvConfig.yaml がそろった過去の週がありません。")
            prev_flex = steps_dir / "prev_FlexibleSchedule_none.yaml"
            prev_flex.write_text("{}\n", encoding="utf-8")
            start_env = settings.master_start_env()
            if start_env:
                prev_env = steps_dir / "prev_EnvConfig_start.yaml"
                shutil.copy2(start_env, prev_env)
                emit(f"  開始用のEnvConfig: {start_env.name}")
            else:
                prev_env = steps_dir / "prev_EnvConfig_none.yaml"
                prev_env.write_text("", encoding="utf-8")
                emit("  警告: MasterフォルダにEnvConfig_start.yamlがありません。EnvConfigは空から始まり、"
                     "地域・Fab・ワークフローは製番リストだけから作られます。")
            skill_file = settings.master_skill_file()
            if skill_file:
                emit(f"  スキルレベルのファイル: {skill_file.name}")
            else:
                emit("  警告: Masterフォルダにスキルレベルのファイルがありません。作業者のスキルは担当職種だけから決めます。")
        weeks.write_status(week, previous_week=previous.name if previous else None, first_week=previous is None)

        def step(key, label, argv, expect):
            number = STEPS.index(key) + 1
            emit(f"== [{number}/{len(STEPS) if submit else len(STEPS) - 1}] {STEP_TITLES[key]}")
            weeks.write_status(week, step=key)
            result = runner.run(key, argv, lambda line: emit("   " + line))
            if result.rc != 0:
                raise StartError(key, f"{STEP_TITLES[key]} が停止しました: {result.last_error()}")
            for f in expect:
                if not Path(f).exists():
                    raise StartError(key, f"{STEP_TITLES[key]} は終了しましたが {Path(f).name} が作られていません。")

        # --- 2-2-3
        argv = ["--prev-flexible", prev_flex, "--prev-env", prev_env, "--su-others", su_copy, "--seiban-list", sb_copy,
                "--out-dir", steps_dir, "--today", day.strftime("%Y/%m/%d")]
        if skill_file:
            argv += ["--skill-file", skill_file]
        for opt, key in (("--su-sheets", "su_sheet"), ("--skill-sheet", "skill_sheet"), ("--seiban-sheet", "seiban_sheet"),
                         ("--default-skill-level", "default_skill_level")):
            if settings.has(key):
                argv += [opt, settings.text(key)]
        step("2-2-3", "2-2-3", argv,
             [steps_dir / "01_EnvConfig.yaml", steps_dir / "01_Schedule.yaml"])

        # --- 2-4
        cfg = steps_dir / "config_2-4.txt"
        write_config(cfg, "settings of tool 2-4 (apply_seiban_list)",
                     [("schedule", steps_dir / "01_Schedule.yaml"), ("env", steps_dir / "01_EnvConfig.yaml"),
                      ("seiban_list", sb_copy), ("out_dir", steps_dir), ("report", "false")])
        step("2-4", "2-4", ["--config", cfg],
             [steps_dir / "02_Schedule.yaml", steps_dir / "02_EnvConfig.yaml"])

        # --- 2-5
        step("2-5", "2-5",
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
            step("2-6", "2-6", ["--config", cfg], [steps_dir / weeks.RUN_INFO_FILE])
            try:
                run_id = json.loads((steps_dir / weeks.RUN_INFO_FILE).read_text(encoding="utf-8")).get("runId")
            except (OSError, ValueError):
                run_id = None
            message = f"ソルバーへ送信しました（実行ID: {run_id}）。計算が終わったら 02_UpdateSchedule を実行してください。"
            stage = "submitted"
        else:
            message = "準備だけ行いました（ソルバーへは送信していません）。ソルバーへの入力は steps/03_Schedule.yaml と steps/03_EnvConfig.yaml です。"
            stage = "ready"
        emit(message)
        weeks.write_status(week, stage=stage, step="done", message=message, run_id=run_id)
        return StartResult(True, "done", message, week, run_id, stage)
    except StartError as e:
        emit(f"中断（{step_name(e.step)}）: {e.message}")
        if week is not None and week.exists() and e.step != "check":
            weeks.write_status(week, stage="failed", step=e.step, message=e.message)
        return StartResult(False, e.step, e.message, week, stage="failed")
    except (SettingsError, OSError) as e:
        emit(f"中断: {e}")
        if week is not None and week.exists():
            weeks.write_status(week, stage="failed", step="error", message=str(e))
        return StartResult(False, "error", str(e), week, stage="failed")
    finally:
        emit.close()