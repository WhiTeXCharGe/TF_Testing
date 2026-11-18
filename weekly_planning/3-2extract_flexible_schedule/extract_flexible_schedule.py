#!/usr/bin/env python3
"""Extract the flexible portion of a solver Schedule for the next planning cycle."""
from __future__ import annotations

import argparse
import copy
import sys
from collections import defaultdict
from pathlib import Path

import yaml

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent))      # weekly_planning/ (the folder of common)
from common.cell_names import module_cell_name, module_name_info  # noqa: E402
from common.dates import ymd  # noqa: E402
from common.text import tool_code  # noqa: E402
from common.yaml_io import dump_yaml, load_yaml  # noqa: E402


def _schedule_body(doc, label, wrapper_key):
    """YAMLルートから指定キー配下、または直下のデータ本体を取得する。"""
    if not isinstance(doc, dict):
        raise ValueError(f"{label} root must be a mapping")
    wrapped = wrapper_key in doc
    body = doc[wrapper_key] if wrapped else doc
    if not isinstance(body, dict):
        raise ValueError(f"{label}.{wrapper_key} must be a mapping")
    return body, wrapped


def _required_list(schedule, key):
    """Schedule内の必須リストを取得し、型を検証する。"""
    value = schedule.get(key)
    if not isinstance(value, list):
        raise ValueError(f"Schedule.{key} must be a list")
    return value


def _optional_list(mapping, key, label):
    """省略可能なリストを取得し、存在する場合は型を検証する。"""
    value = mapping.get(key, [])
    if not isinstance(value, list):
        raise ValueError(f"{label}.{key} must be a list")
    return value


def _ensure_hashable(value, label):
    """辞書キーや集合要素に使える値であることを検証する。"""
    try:
        hash(value)
    except TypeError as exc:
        raise ValueError(f"{label} must be hashable") from exc
    return value


def _operations(task):
    """WorkflowTask内の全OperationTaskを平坦なリストで返す。"""
    return [
        operation
        for phase in (task.get("phase_task_list") or [])
        for operation in (phase.get("operation_task_list") or [])
    ]


def _set_id(task, label):
    """タスクのIDを検証して取得する。"""
    if not isinstance(task, dict):
        raise ValueError(f"{label} must be a mapping")
    if "id" not in task:
        raise ValueError(f"{label}.id is required")
    return _ensure_hashable(task["id"], f"{label}.id")


def _is_set(value):
    """値がNoneでも空文字列でもない設定済みの値か判定する。"""
    return value is not None and (not isinstance(value, str) or bool(value.strip()))


def _name_environment(env):
    """表示名生成に必要なEnvConfig項目だけを安全に取り出す。"""
    safe_env = {}
    for key in ("fab_list", "customer_company_list"):
        entries = env.get(key, [])
        if isinstance(entries, list):
            safe_env[key] = [entry for entry in entries if isinstance(entry, dict) and "id" in entry]
        else:
            safe_env[key] = []
    return safe_env


def _workflow_indexes(workflow_tasks):
    """OperationTask IDから所属WorkflowTaskのインデックスを引ける辞書を作る。"""
    operation_to_task = {}
    for task_index, task in enumerate(workflow_tasks):
        if not isinstance(task, dict):
            raise ValueError(f"Schedule.workflow_task_list[{task_index}] must be a mapping")
        phases = _optional_list(task, "phase_task_list", f"workflow_task_list[{task_index}]")
        for phase_index, phase in enumerate(phases):
            if not isinstance(phase, dict):
                raise ValueError(
                    f"workflow_task_list[{task_index}].phase_task_list[{phase_index}] must be a mapping"
                )
            operations = _optional_list(
                phase, "operation_task_list",
                f"workflow_task_list[{task_index}].phase_task_list[{phase_index}]",
            )
            for operation_index, operation in enumerate(operations):
                operation_id = _set_id(
                    operation,
                    f"workflow_task_list[{task_index}].phase_task_list[{phase_index}]"
                    f".operation_task_list[{operation_index}]",
                )
                if operation_id in operation_to_task:
                    raise ValueError(f"duplicate WorkflowTask OperationTask id: {operation_id!r}")
                operation_to_task[operation_id] = task_index
    return operation_to_task


def _validate_misc_tasks(misc_tasks):
    """MiscTaskのIDを検証し、参照判定用の集合を返す。"""
    misc_ids = set()
    for index, task in enumerate(misc_tasks):
        task_id = _set_id(task, f"Schedule.misc_task_list[{index}]")
        misc_ids.add(task_id)
    return misc_ids


def _assignment_work_dates(assignment, label):
    """Assignmentの作業日を読み取り、解釈できる日付だけを返す。"""
    work_dates = _optional_list(assignment, "work_date_list", label)
    dates = []
    for index, item in enumerate(work_dates):
        if not isinstance(item, dict):
            raise ValueError(f"{label}.work_date_list[{index}] must be a mapping")
        day = ymd(item.get("date"))
        if day:
            dates.append(day)
    return dates


def _extract_assignments(assignments, operation_to_task, misc_ids, workflow_tasks, env):
    """WorkflowTask向けAssignmentを抽出し、Trip判定日とMiscTask最終日も集計する。"""
    flexible_work_dates = defaultdict(set)
    misc_last_dates = {}
    selected = []
    safe_env = _name_environment(env)

    for index, assignment in enumerate(assignments):
        label = f"Schedule.assignment_list[{index}]"
        if not isinstance(assignment, dict):
            raise ValueError(f"{label} must be a mapping")
        operation_id = _ensure_hashable(assignment.get("operation_task"), f"{label}.operation_task")
        if operation_id in operation_to_task:
            task_index = operation_to_task[operation_id]
            task = workflow_tasks[task_index]
            dates = _assignment_work_dates(assignment, label)
            worker = assignment.get("worker")
            if assignment.get("plan_flexibility", "Flexible") == "Flexible" and worker is not None:
                _ensure_hashable(worker, f"{label}.worker")
                flexible_work_dates[worker].update(dates)

            row = copy.deepcopy(assignment)
            if not _is_set(row.get("cell_name")):
                description = row.get("description")
                if isinstance(description, str) and description.strip():
                    row["cell_name"] = description
                else:
                    code = tool_code(task.get("name")) or task.get("name") or task.get("id")
                    if code is None:
                        raise ValueError(
                            f"WorkflowTask for assignment {index} needs a name or id to generate cell_name"
                        )
                    row["cell_name"] = module_cell_name(str(code), module_name_info(task, safe_env))
            selected.append(row)
        elif operation_id in misc_ids:
            dates = _assignment_work_dates(assignment, label)
            if dates:
                last_date = max(dates)
                old_date = misc_last_dates.get(operation_id)
                if old_date is None or last_date > old_date:
                    misc_last_dates[operation_id] = last_date
        else:
            raise ValueError(
                f"{label}.operation_task references unknown OperationTask {operation_id!r}"
            )
    return selected, flexible_work_dates, misc_last_dates


def _is_flexible_trip(trip, item_index, worker, flexible_work_dates):
    """Tripを検証し、Trip自身がFlexibleかつ期間内にFlexible作業日があるか判定する。"""
    if not isinstance(trip, dict):
        raise ValueError(f"Schedule.trip_list[{item_index}] trip must be a mapping")
    start = ymd(trip.get("start_date"))
    end = ymd(trip.get("end_date"))
    if not start or not end:
        raise ValueError(f"Schedule.trip_list[{item_index}] has an invalid Trip date")
    if start > end:
        raise ValueError(f"Schedule.trip_list[{item_index}] Trip start_date is after end_date")
    return trip.get("plan_flexibility", "Flexible") == "Flexible" and any(
        start <= day <= end for day in flexible_work_dates.get(worker, ())
    )


def _extract_trips(trip_list, flexible_work_dates):
    """Flexible作業と期間が重なるTripを抽出し、入力時の形式を維持する。"""
    output = []
    for item_index, item in enumerate(trip_list):
        if not isinstance(item, dict) or item.get("worker") is None:
            raise ValueError(f"Schedule.trip_list[{item_index}] must have a worker")
        worker = _ensure_hashable(item["worker"], f"Schedule.trip_list[{item_index}].worker")

        if "trips" in item:
            if not isinstance(item["trips"], list):
                raise ValueError(f"Schedule.trip_list[{item_index}].trips must be a list")
            retained = [
                copy.deepcopy(trip) for trip in item["trips"]
                if _is_flexible_trip(trip, item_index, worker, flexible_work_dates)
            ]
            if retained:
                group = copy.deepcopy(item)
                group["trips"] = retained
                output.append(group)
        else:
            if "start_date" not in item:
                raise ValueError(f"Schedule.trip_list[{item_index}] must contain trips or start_date")
            if _is_flexible_trip(item, item_index, worker, flexible_work_dates):
                output.append(copy.deepcopy(item))
    return output


def _task_dates(task):
    """WorkflowTaskのPhaseTaskから最早開始日と最遅終了日を取得する。"""
    phases = task.get("phase_task_list") or []
    starts = []
    ends = []
    for phase in phases:
        if not isinstance(phase, dict):
            continue
        start = ymd(phase.get("start_date"))
        end = ymd(phase.get("end_date"))
        if start:
            starts.append(start)
        if end:
            ends.append(end)
    return (min(starts) if starts else None, max(ends) if ends else None)


def _color_candidates(workflow_tasks, misc_tasks, misc_last_dates):
    """補完に使える色を最終出現日順に並べ、色補完対象を特定する。"""
    latest = {}
    sequence = 0

    def add(color, day):
        """色の最終出現日と入力順を候補辞書へ反映する。"""
        nonlocal sequence
        if not _is_set(color) or day is None:
            return
        _ensure_hashable(color, "color_code")
        order = sequence
        sequence += 1
        previous = latest.get(color)
        if previous is None or day > previous[0] or (day == previous[0] and order < previous[1]):
            latest[color] = (day, order)

    target_tasks = []
    for task_index, task in enumerate(workflow_tasks):
        start, end = _task_dates(task)
        operations = _operations(task)
        missing_color = any(not _is_set(operation.get("color_code")) for operation in operations)
        if missing_color:
            target_tasks.append((start is None, start or "", task_index))
        for operation in operations:
            add(operation.get("color_code"), end)

    for misc_task in misc_tasks:
        if tool_code(misc_task.get("name")):
            add(misc_task.get("color_code"), misc_last_dates.get(misc_task["id"]))

    colors = sorted(
        latest,
        key=lambda color: (latest[color][0], latest[color][1], str(color)),
    )
    return colors, sorted(target_tasks)


def _fill_colors(workflow_tasks, misc_tasks, misc_last_dates):
    """未設定OperationTaskに色候補を順番に割り当てる。"""
    colors, targets = _color_candidates(workflow_tasks, misc_tasks, misc_last_dates)
    if not colors:
        return

    used = set()
    for _, _, task_index in targets:
        operations = _operations(workflow_tasks[task_index])
        existing = [operation.get("color_code") for operation in operations
                    if _is_set(operation.get("color_code"))]

        selected = next((color for color in colors if color not in used and color not in existing), None)
        if selected is None:
            used.clear()
            selected = next((color for color in colors if color not in existing), None)
        if selected is None:
            continue
        used.add(selected)
        for operation in operations:
            if not _is_set(operation.get("color_code")):
                operation["color_code"] = selected


def create_flexible_schedule(schedule_path: str | Path, env_path: str | Path,
                             output_path: str | Path) -> None:
    """ScheduleとEnvConfigを読み込み、FlexibleScheduleを抽出・補完して保存する。"""
    schedule_doc = load_yaml(schedule_path)
    env_doc = load_yaml(env_path)
    schedule, wrapped = _schedule_body(schedule_doc, "Schedule", "schedule")
    env, _ = _schedule_body(env_doc, "EnvConfig", "environment")

    if not isinstance(schedule.get("plan_range"), dict):
        raise ValueError("Schedule.plan_range must be a mapping")
    workflow_tasks = _required_list(schedule, "workflow_task_list")
    assignments = _required_list(schedule, "assignment_list")
    trip_list = _required_list(schedule, "trip_list")
    misc_tasks = _optional_list(schedule, "misc_task_list", "Schedule")

    operation_to_task = _workflow_indexes(workflow_tasks)
    misc_ids = _validate_misc_tasks(misc_tasks)
    retained_assignments, flexible_work_dates, misc_last_dates = _extract_assignments(
        assignments, operation_to_task, misc_ids, workflow_tasks, env
    )
    retained_trips = _extract_trips(trip_list, flexible_work_dates)

    output_tasks = copy.deepcopy(workflow_tasks)
    _fill_colors(output_tasks, misc_tasks, misc_last_dates)
    output_schedule = {
        "plan_range": copy.deepcopy(schedule["plan_range"]),
        "workflow_task_list": output_tasks,
        "assignment_list": retained_assignments,
        "trip_list": retained_trips,
    }
    output_doc = {"schedule": output_schedule} if wrapped else output_schedule
    dump_yaml(output_path, output_doc)


def main(argv=None):
    """CLI引数を解析し、FlexibleSchedule生成処理の終了コードを返す。"""
    parser = argparse.ArgumentParser(description="Extract a FlexibleSchedule from a solver Schedule.")
    parser.add_argument("--schedule", required=True, help="optimized Schedule YAML")
    parser.add_argument("--env", required=True, help="EnvConfig YAML")
    parser.add_argument("--output", required=True, help="output FlexibleSchedule YAML")
    args = parser.parse_args(argv)
    try:
        create_flexible_schedule(args.schedule, args.env, args.output)
    except (OSError, ValueError, TypeError, yaml.YAMLError) as exc:
        print(f"extract_flexible_schedule: {exc}", file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    sys.exit(main())