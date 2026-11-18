#!/usr/bin/env python3
"""最適化の計画期間を設定し、開始日前のassignmentを固定する。"""
from __future__ import annotations

import argparse
import os
import sys
from collections.abc import Sequence
from datetime import date, datetime, timedelta
from pathlib import Path
from typing import Any

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent))      # weekly_planning/ (the folder of common)
from common.dates import ymd   # noqa: E402
from common.yaml_io import dump_yaml, load_yaml   # noqa: E402


PLAN_FLEXIBILITIES = {"Flexible", "Reluctant", "Fixed"}
PLAN_START_DAYS = 7
PLAN_END_DAYS = 30


def _date(value: Any, location: str) -> date:
    """日付値を ``date`` に変換し、不正な場合は項目位置付きでエラーにする。"""
    normalized = ymd(value)
    if normalized is None:
        raise ValueError(f"{location}: invalid or missing date {value!r}")
    return datetime.strptime(normalized, "%Y/%m/%d").date()


def _mapping(value: Any, location: str) -> dict[Any, Any]:
    """Scheduleの値がマッピング形式であることを検証する。"""
    if not isinstance(value, dict):
        raise ValueError(f"{location}: expected a mapping")
    return value


def _list(value: Any, location: str) -> list[Any]:
    """Scheduleの値がリスト形式であることを検証する。"""
    if not isinstance(value, list):
        raise ValueError(f"{location}: expected a list")
    return value


def _validated_schedule(
    document: Any,
) -> tuple[
    dict[Any, Any],
    dict[Any, Any],
    bool,
    list[tuple[date, date]],
    list[dict[Any, Any]],
]:
    """Scheduleの構造と日付を検証し、ルート・Schedule・phase情報を返す。"""
    root = _mapping(document, "Schedule")
    wrapped = "schedule" in root
    schedule = _mapping(root["schedule"], "schedule") if wrapped else root

    required = ("plan_range", "workflow_task_list", "assignment_list")
    missing = [key for key in required if key not in schedule]
    if missing:
        raise ValueError(f"Schedule is missing required keys: {', '.join(missing)}")

    plan_range = _mapping(schedule["plan_range"], "plan_range")
    for key in ("start_date", "end_date"):
        if key not in plan_range:
            raise ValueError(f"plan_range is missing required key: {key}")

    phases = []
    workflows = _list(schedule["workflow_task_list"], "workflow_task_list")
    for workflow_index, workflow in enumerate(workflows):
        workflow = _mapping(workflow, f"workflow_task_list[{workflow_index}]")
        if "phase_task_list" not in workflow:
            raise ValueError(f"workflow_task_list[{workflow_index}] is missing phase_task_list")
        phase_list = _list(workflow["phase_task_list"],
                           f"workflow_task_list[{workflow_index}].phase_task_list")
        for phase_index, phase in enumerate(phase_list):
            location = f"workflow_task_list[{workflow_index}].phase_task_list[{phase_index}]"
            phase = _mapping(phase, location)
            start = _date(phase.get("start_date"), f"{location}.start_date")
            end = _date(phase.get("end_date"), f"{location}.end_date")
            if start > end:
                raise ValueError(f"{location}: start_date is after end_date")
            phases.append((start, end))

    assignments = _list(schedule["assignment_list"], "assignment_list")
    for assignment_index, assignment in enumerate(assignments):
        location = f"assignment_list[{assignment_index}]"
        assignment = _mapping(assignment, location)
        if "start_date" not in assignment or "end_date" not in assignment:
            raise ValueError(f"{location}: start_date and end_date are required")
        start = _date(assignment["start_date"], f"{location}.start_date")
        end = _date(assignment["end_date"], f"{location}.end_date")
        if start > end:
            raise ValueError(f"{location}: start_date is after end_date")

        if "plan_flexibility" in assignment:
            flexibility = assignment["plan_flexibility"]
            if not isinstance(flexibility, str) or flexibility not in PLAN_FLEXIBILITIES:
                raise ValueError(
                    f"{location}.plan_flexibility: expected one of "
                    f"{', '.join(sorted(PLAN_FLEXIBILITIES))}"
                )

        work_dates = _list(assignment.get("work_date_list"),
                           f"{location}.work_date_list")
        if not work_dates:
            raise ValueError(f"{location}.work_date_list: must not be empty")
        for work_index, work in enumerate(work_dates):
            work_location = f"{location}.work_date_list[{work_index}]"
            work = _mapping(work, work_location)
            work_date = _date(work.get("date"), f"{work_location}.date")
            if not start <= work_date <= end:
                raise ValueError(f"{work_location}.date: outside assignment start_date/end_date")

    return root, schedule, wrapped, phases, assignments


def _calculate_plan_period(
    phases: Sequence[tuple[date, date]],
    today: date,
    plan_start_override: date | None = None,
) -> tuple[date, date]:
    """phaseの日付または既定値を使って計画期間を算出する。"""
    if plan_start_override is not None:
        plan_start = plan_start_override
    elif phases:
        plan_start = max(min(start for start, _ in phases),
                         today + timedelta(days=PLAN_START_DAYS))
    else:
        plan_start = today + timedelta(days=PLAN_START_DAYS)

    if phases:
        plan_end = max(max(end for _, end in phases) + timedelta(days=PLAN_END_DAYS),
                       plan_start + timedelta(days=PLAN_END_DAYS))
    else:
        plan_end = plan_start + timedelta(days=PLAN_END_DAYS)
    return plan_start, plan_end


def _distinct_paths(paths: Sequence[str | Path]) -> None:
    """正規化後に同一ファイルを指すパスがあればエラーにする。"""
    resolved = [Path(path).resolve() for path in paths]
    keys = [os.path.normcase(str(path)) for path in resolved]
    if len(set(keys)) != len(keys):
        raise ValueError("input and output paths must refer to four distinct files")


def _set_plan_start(value: str) -> date:
    """CLIの計画開始日を ``date`` に変換する。"""
    normalized = ymd(value)
    if normalized is None:
        raise argparse.ArgumentTypeError("must be a valid date in YYYY/MM/DD format")
    return datetime.strptime(normalized, "%Y/%m/%d").date()


def set_plan_period(
    input_schedule: str | Path,
    input_env_config: str | Path,
    output_schedule: str | Path,
    output_env_config: str | Path,
    plan_start: date | None = None,
) -> None:
    """ScheduleとEnvConfigを読み込み、計画期間を設定して出力する。"""
    paths = (input_schedule, input_env_config, output_schedule, output_env_config)
    _distinct_paths(paths)

    document = load_yaml(input_schedule)
    root, schedule, wrapped, phases, assignments = _validated_schedule(document)
    env_config = Path(input_env_config).read_bytes()

    plan_start, plan_end = _calculate_plan_period(phases, date.today(), plan_start)
    schedule["plan_range"]["start_date"] = plan_start.strftime("%Y/%m/%d")
    schedule["plan_range"]["end_date"] = plan_end.strftime("%Y/%m/%d")
    for assignment in assignments:
        has_earlier_work = any(
            _date(work["date"], "assignment_list.work_date_list.date") < plan_start
            for work in assignment["work_date_list"]
        )
        if has_earlier_work:
            assignment["plan_flexibility"] = "Fixed"

    if wrapped:
        root["schedule"] = schedule
    dump_yaml(output_schedule, root)
    output_env = Path(output_env_config)
    output_env.parent.mkdir(parents=True, exist_ok=True)
    output_env.write_bytes(env_config)
    print(f"written {output_schedule} and {output_env_config}")


def main(argv: Sequence[str] | None = None) -> int:
    """CLI引数を解析し、計画期間を設定する。"""
    parser = argparse.ArgumentParser(description="Set the weekly planning optimization period.")
    parser.add_argument("--input-schedule", required=True, help="input Schedule YAML path")
    parser.add_argument("--input-env-config", required=True, help="input EnvConfig path")
    parser.add_argument("--output-schedule", required=True, help="output Schedule YAML path")
    parser.add_argument("--output-env-config", required=True, help="output EnvConfig path")
    parser.add_argument("--plan-start", type=_set_plan_start, help="override plan start date (YYYY/MM/DD)")
    args = parser.parse_args(argv)

    set_plan_period(
        args.input_schedule,
        args.input_env_config,
        args.output_schedule,
        args.output_env_config,
        args.plan_start,
    )
    return 0


if __name__ == "__main__":
    sys.exit(main())