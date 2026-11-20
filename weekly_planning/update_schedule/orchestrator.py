from __future__ import annotations

import importlib.util
import json
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Callable, Literal, Protocol

from common import week_folders
from update_schedule.paths import ApplicationPaths
from update_schedule.run_state import (
    InvalidStateTransitionError,
    RunDirectory,
    RunStateError,
    StateUnknownError,
    UpdateStateStore,
    UpdateStatus,
    select_latest_run,
    validate_run_inputs,
)
from update_schedule.workbook_store import (
    FileSystemWorkbookStore,
    FileSystemWorkbookTarget,
    LockHandle,
    WorkbookStore,
)


class ReceiveResultLike(Protocol):
    status: str


class WriteResultLike(Protocol):
    output_path: Path
    warning_count: int
    conflict_count: int


@dataclass(frozen=True)
class ExistingTools:
    receive_result: Callable[..., ReceiveResultLike]
    create_flexible_schedule: Callable[..., None]
    write_su_others: Callable[..., WriteResultLike]


@dataclass(frozen=True)
class UpdateResult:
    status: Literal["completed", "not_finished", "already_applied"]
    run_directory: Path
    workbook_path: Path
    warning_count: int
    conflict_count: int
    warnings: tuple[str, ...]


class ToolLoadError(RunStateError):
    """Raised when an existing workflow script cannot be loaded."""


class PostPublishStateError(RunStateError):
    """Raised when publication succeeded but applied state could not be recorded."""


class RecoveryError(RunStateError):
    """Raised when processing failed and the workbook/state could not be safely restored."""


def _load_module(module_name: str, script_path: Path):
    if not script_path.is_file():
        raise ToolLoadError(f"Required workflow script not found: {script_path}")
    spec = importlib.util.spec_from_file_location(module_name, script_path)
    if spec is None or spec.loader is None:
        raise ToolLoadError(f"Cannot create a module loader for {script_path}")

    module = importlib.util.module_from_spec(spec)
    previous_module = sys.modules.get(module_name)
    sys.modules[module_name] = module
    try:
        spec.loader.exec_module(module)
    except Exception as error:
        if previous_module is None:
            sys.modules.pop(module_name, None)
        else:
            sys.modules[module_name] = previous_module
        raise ToolLoadError(f"Cannot load workflow script {script_path}: {error}") from error
    return module


def load_existing_tools(application_directory: Path) -> ExistingTools:
    """アプリ配置ディレクトリにある3-1〜3-3の既存関数APIを読み込む。"""
    application_directory = Path(application_directory)
    receive_module = _load_module(
        "_update_schedule_receive_from_timefold",
        application_directory
        / "3-1receive_from_timefold"
        / "receive_from_timefold.py",
    )
    extract_module = _load_module(
        "_update_schedule_extract_flexible_schedule",
        application_directory
        / "3-2extract_flexible_schedule"
        / "extract_flexible_schedule.py",
    )
    write_module = _load_module(
        "_update_schedule_write_su_others",
        application_directory / "3-3write_su_others" / "write_su_others.py",
    )
    for module, attribute, script in (
        (receive_module, "receive_result", "3-1"),
        (extract_module, "create_flexible_schedule", "3-2"),
        (write_module, "write_su_others", "3-3"),
    ):
        if not callable(getattr(module, attribute, None)):
            raise ToolLoadError(
                f"{script} script does not expose callable {attribute}: "
                f"{module.__file__}"
            )
    return ExistingTools(
        receive_result=receive_module.receive_result,
        create_flexible_schedule=extract_module.create_flexible_schedule,
        write_su_others=write_module.write_su_others,
    )


def _validate_run_id(run: RunDirectory) -> None:
    try:
        info = json.loads(run.run_info.read_text(encoding="utf-8"))
    except (OSError, UnicodeDecodeError, json.JSONDecodeError) as error:
        raise RunStateError(
            f"Cannot read valid run_info.json at {run.run_info}: {error}"
        ) from error
    run_id = info.get("runId") if isinstance(info, dict) else None
    if not isinstance(run_id, str) or not run_id.strip():
        raise RunStateError(
            f"run_info.json must contain a non-empty runId for 3-1: {run.run_info}"
        )


def _same_bytes(first: Path, second: Path) -> bool:
    with first.open("rb") as left, second.open("rb") as right:
        while True:
            left_chunk = left.read(1024 * 1024)
            right_chunk = right.read(1024 * 1024)
            if left_chunk != right_chunk:
                return False
            if not left_chunk:
                return True


def _require_output(path: Path, label: str) -> None:
    if not path.is_file():
        raise RunStateError(f"{label} was not generated: {path}")


def _restore_after_failure(
    store: WorkbookStore,
    lock: LockHandle,
    state: UpdateStateStore,
    original_error: Exception,
) -> None:
    try:
        store.release(lock, published=False)
    except Exception as restore_error:
        raise RecoveryError(
            f"{original_error}; failed to restore the workbook lock. "
            f"Inspect the selected workbook and its _SU_Others.xlsx lock file: "
            f"{restore_error}"
        ) from original_error

    try:
        current = state.read()
        if current is UpdateStatus.IN_PROGRESS:
            state.mark_ready_after_restore(workbook_restored=True)
        elif current is not UpdateStatus.READY:
            raise InvalidStateTransitionError(
                f"Restored workbook but state remains {current.value}"
            )
    except Exception as state_error:
        raise RecoveryError(
            f"{original_error}; the workbook was restored, but the run state "
            f"could not be made safely retryable: {state_error}"
        ) from original_error


def _release_without_update(
    store: WorkbookStore,
    lock: LockHandle,
    reason: str,
    original_error: Exception | None = None,
) -> None:
    try:
        store.release(lock, published=False)
    except Exception as release_error:
        detail = f"{reason}; failed to restore the workbook lock: {release_error}"
        if original_error is not None:
            detail = f"{original_error}; {detail}"
        raise RecoveryError(detail) from original_error or release_error


def run_update(
    paths: ApplicationPaths,
    selected_workbook: Path,
    *,
    store: WorkbookStore | None = None,
    tools: ExistingTools | None = None,
    on_stage: Callable[[str], None] | None = None,
) -> UpdateResult:
    """対象回をロックし、既存3-1〜3-3を実行してブックを公開する。"""
    paths.prepare()
    run = select_latest_run(paths.work_directory)
    validate_run_inputs(run, selected_workbook)
    _validate_run_id(run)
    store = store or FileSystemWorkbookStore()
    tools = tools or load_existing_tools(paths.application_directory)
    target = FileSystemWorkbookTarget(Path(selected_workbook))
    state = UpdateStateStore(run)

    lock = store.acquire(target)
    try:
        status = state.read()
    except Exception as error:
        _release_without_update(
            store, lock, "Run state is unknown; update was not started", error
        )
        raise

    if status is UpdateStatus.APPLIED:
        _release_without_update(
            store, lock, "Run was already applied; update was not started"
        )
        return UpdateResult(
            status="already_applied",
            run_directory=run.directory,
            workbook_path=target.path,
            warning_count=0,
            conflict_count=0,
            warnings=(),
        )
    if status is UpdateStatus.IN_PROGRESS:
        _release_without_update(
            store, lock, "Run is in_progress; automatic recovery is disabled"
        )
        raise StateUnknownError(
            f"Run is in_progress; automatic recovery is disabled: "
            f"{state.run.status_file}"
        )
    if status is not UpdateStatus.READY:
        _release_without_update(
            store, lock, f"Run state is {status.value}; update was not started"
        )
        raise StateUnknownError(
            f"Run has an unknown state; automatic recovery is disabled: "
            f"{state.run.status_file}"
        )

    release_attempted = False
    workbook_restored = False
    try:
        state.archive_intermediates()
        state.mark_in_progress()
        suffix = target.path.suffix.casefold()
        work_copy = run.steps / f"02_SU_Others{suffix}"
        store.download_locked(lock, work_copy)
        reference_copy = run.workbook_for_suffix(suffix)
        warnings: list[str] = []
        if not _same_bytes(reference_copy, work_copy):
            warnings.append(
                f"{reference_copy} と {work_copy} の内容が異なります。"
                "自動計画中にブックが編集された可能性があります。"
            )

        if on_stage is not None:
            on_stage("結果取得中")
        receive_result = tools.receive_result(
            run_info_path=run.run_info,
            output_dir=run.steps,
            config_path=paths.config_file,
            watch_seconds=0,
        )
        if receive_result.status == "not_finished":
            release_attempted = True
            store.release(lock, published=False)
            workbook_restored = True
            try:
                state.mark_ready_after_restore(workbook_restored=True)
            except Exception as error:
                raise RecoveryError(
                    f"Run is not finished and the workbook was restored, but "
                    f"the state could not be reset to ready: {error}"
                ) from error
            return UpdateResult(
                status="not_finished",
                run_directory=run.directory,
                workbook_path=target.path,
                warning_count=0,
                conflict_count=0,
                warnings=tuple(warnings),
            )
        if receive_result.status != "completed":
            raise RunStateError(
                f"3-1 returned an unsupported status: {receive_result.status!r}"
            )
        schedule_path = run.steps / "04_Schedule.yaml"
        env_path = run.steps / "04_EnvConfig.yaml"
        _require_output(schedule_path, "3-1 Schedule")
        _require_output(env_path, "3-1 EnvConfig")

        if on_stage is not None:
            on_stage("FlexibleSchedule生成中")
        flexible_schedule = run.steps / "FlexibleSchedule.yaml"
        tools.create_flexible_schedule(
            schedule_path=schedule_path,
            env_path=env_path,
            output_path=flexible_schedule,
        )
        _require_output(flexible_schedule, "3-2 FlexibleSchedule")

        if on_stage is not None:
            on_stage("Excel反映中")
        previous = week_folders.previous_week(paths.work_directory, run.directory.name)
        start_status = week_folders.read_status(run.directory)
        if "previous_week" in start_status:
            previous_name = start_status["previous_week"]
            if previous_name is None:
                previous = None
            elif (isinstance(previous_name, str) and len(previous_name) == 8
                  and previous_name.isdigit() and previous_name < run.directory.name):
                previous = paths.work_directory / previous_name
            else:
                raise RunStateError(f"Invalid previous_week in StartSchedule status: {previous_name!r}")
        previous_inputs = week_folders.previous_week_inputs(previous) if previous is not None else None
        if previous is not None and previous_inputs is None:
            raise RunStateError(f"Previous flexible schedule inputs are unavailable: {previous}")
        write_result = tools.write_su_others(
            schedule_path=flexible_schedule,
            env_path=env_path,
            su_others_path=work_copy,
            output_dir=run.steps,
            flexible_work_path=run.flexible_work,
            write_report=False,
            previous_flexible_path=previous_inputs[0] if previous_inputs is not None else None,
            previous_env_path=previous_inputs[1] if previous_inputs is not None else None,
        )
        expected_result = run.steps / f"02_SU_Others_rewritten{suffix}"
        if Path(write_result.output_path).resolve() != expected_result.resolve():
            raise RunStateError(
                f"3-3 returned an unexpected workbook path: "
                f"{write_result.output_path}; expected {expected_result}"
            )
        _require_output(expected_result, "3-3 rewritten workbook")
        if write_result.warning_count < 0 or write_result.conflict_count < 0:
            raise RunStateError("3-3 returned negative warning/conflict counts")

        store.publish(lock, expected_result)
    except Exception as error:
        if release_attempted:
            if workbook_restored:
                raise
            raise RecoveryError(
                f"{error}; workbook restoration was attempted but its outcome "
                "is unknown. Do not retry automatically."
            ) from error
        release_attempted = True
        _restore_after_failure(store, lock, state, error)
        raise

    try:
        state.mark_applied()
    except Exception as error:
        raise PostPublishStateError(
            f"Workbook was published to {target.path}, but applied state "
            f"could not be saved. Do not retry automatically; inspect "
            f"{state.run.status_file} and the locked workbook. Cause: {error}"
        ) from error

    try:
        store.release(lock, published=True)
    except Exception as error:
        warnings.append(
            f"反映は完了しましたが、ロックファイルを解除できませんでした。"
            f"対象: {target.path.with_name('_SU_Others.xlsx')}。原因: {error}"
        )

    return UpdateResult(
        status="completed",
        run_directory=run.directory,
        workbook_path=target.path,
        warning_count=write_result.warning_count,
        conflict_count=write_result.conflict_count,
        warnings=tuple(warnings),
    )
