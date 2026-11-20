from __future__ import annotations

import json
import os
import re
import shutil
import tempfile
import uuid
from dataclasses import dataclass
from datetime import datetime
from enum import Enum
from pathlib import Path
from typing import Any

from common.week_folders import RUN_INFO_FILE, UPDATE_STATUS_FILE


_DATE_FOLDER = re.compile(r"^\d{8}$")
_INTERMEDIATE_NAMES = (
    "02_SU_Others.xlsx",
    "02_SU_Others.xlsm",
    "04_Schedule.yaml",
    "04_EnvConfig.yaml",
    "FlexibleSchedule.yaml",
    "02_SU_Others_rewritten.xlsx",
    "02_SU_Others_rewritten.xlsm",
)


class RunStateError(RuntimeError):
    """Raised when a planning run cannot safely be selected or processed."""


class StateUnknownError(RunStateError):
    """Raised when persisted update state is missing in an unsafe context or invalid."""


class InvalidStateTransitionError(RunStateError):
    """Raised when a requested update-state transition is not permitted."""


class HistoryArchiveError(RunStateError):
    """Raised when intermediate files cannot be archived safely."""


class UpdateStatus(str, Enum):
    READY = "ready"
    IN_PROGRESS = "in_progress"
    APPLIED = "applied"


@dataclass(frozen=True)
class RunDirectory:
    directory: Path

    @property
    def steps(self) -> Path:
        return self.directory / "steps"

    @property
    def flexible_work(self) -> Path:
        """対象回のFlexibleWorkファイルのパスを返す。"""
        return self.steps / "FlexibleWork.yaml"

    @property
    def run_info(self) -> Path:
        """対象回のrun_info.jsonのパスを返す。"""
        return self.steps / RUN_INFO_FILE

    @property
    def env_config(self) -> Path:
        """対象回のEnvConfigファイルのパスを返す。"""
        return self.steps / "03_EnvConfig.yaml"

    @property
    def status_file(self) -> Path:
        """対象回の反映状態ファイルのパスを返す。"""
        return self.steps / UPDATE_STATUS_FILE

    def workbook_for_suffix(self, suffix: str) -> Path:
        """指定された拡張子に対応する自動計画開始時のブックコピーを返す。"""
        normalized_suffix = suffix.casefold()
        if normalized_suffix not in {".xlsx", ".xlsm"}:
            raise ValueError(f"Unsupported workbook extension: {suffix}")
        return self.directory / "input" / f"01_SU_Others{normalized_suffix}"


def _date_from_directory_name(name: str) -> datetime | None:
    """YYYYMMDD形式の名前を検証し、有効な日付ならdatetimeを返す。"""
    if not _DATE_FOLDER.fullmatch(name):
        return None
    try:
        parsed = datetime.strptime(name, "%Y%m%d")
    except ValueError:
        return None
    if f"{parsed.year:04d}{parsed.month:02d}{parsed.day:02d}" != name:
        return None
    return parsed


def select_latest_run(work_root: Path) -> RunDirectory:
    """Work直下からsteps/run_info.jsonを持つ最新の有効な日付フォルダを選ぶ。"""
    work_root = Path(work_root)
    if not work_root.is_dir():
        raise FileNotFoundError(f"Work directory not found: {work_root}")

    candidates: list[tuple[datetime, Path]] = []
    for entry in work_root.iterdir():
        if not entry.is_dir():
            continue
        run_date = _date_from_directory_name(entry.name)
        if run_date is not None and RunDirectory(entry).run_info.is_file():
            candidates.append((run_date, entry))

    if not candidates:
        raise RunStateError(
            f"No valid Work/YYYYMMDD run containing steps/run_info.json was found in {work_root}"
        )
    _, latest_directory = max(candidates, key=lambda candidate: candidate[0])
    return RunDirectory(latest_directory)


def _require_readable_file(path: Path, label: str) -> None:
    """必須ファイルが存在し、読み取り可能であることを確認する。"""
    if not path.is_file():
        raise RunStateError(f"Required {label} file not found: {path}")
    try:
        with path.open("rb") as input_file:
            input_file.read(1)
    except OSError as error:
        raise RunStateError(f"Cannot read {label} file {path}: {error}") from error


def validate_run_inputs(run: RunDirectory, selected_workbook: Path) -> None:
    """選択ブックと対象回の必須ファイル、形式、EnvConfig参照を検証する。"""
    selected_workbook = Path(selected_workbook)
    suffix = selected_workbook.suffix.casefold()
    if suffix not in {".xlsx", ".xlsm"}:
        raise RunStateError(
            f"Selected workbook must have an .xlsx or .xlsm extension: {selected_workbook}"
        )
    if not selected_workbook.is_absolute():
        raise RunStateError(
            f"Selected workbook path must be absolute: {selected_workbook}"
        )

    _require_readable_file(selected_workbook, "selected workbook")
    _require_readable_file(run.workbook_for_suffix(suffix), "initial workbook copy")
    _require_readable_file(run.flexible_work, "FlexibleWork")
    _require_readable_file(run.run_info, "run_info")
    _require_readable_file(run.env_config, "EnvConfig")

    try:
        info: Any = json.loads(run.run_info.read_text(encoding="utf-8"))
    except (OSError, UnicodeDecodeError, json.JSONDecodeError) as error:
        raise RunStateError(f"Cannot read valid run_info.json at {run.run_info}: {error}") from error
    if not isinstance(info, dict):
        raise RunStateError(f"run_info.json must contain a JSON object: {run.run_info}")

    env_file = info.get("envFile")
    if not isinstance(env_file, str) or not env_file.strip():
        raise RunStateError(
            f"run_info.json must specify envFile for {run.env_config}"
        )
    env_path = Path(env_file).expanduser()
    if not env_path.is_absolute():
        raise RunStateError(
            f"run_info.json envFile must be an absolute path: {env_file}"
        )
    if os.path.normcase(os.path.abspath(env_path)) != os.path.normcase(
        os.path.abspath(run.env_config)
    ):
        raise RunStateError(
            f"run_info.json envFile must refer to this run's EnvConfig: "
            f"expected {run.env_config}, got {env_path}"
        )


class UpdateStateStore:
    def __init__(self, run: RunDirectory) -> None:
        """状態を管理する対象回を設定する。"""
        self.run = run

    def read(self) -> UpdateStatus:
        """状態ファイルを検証して状態を返し、初回ならready相当を返す。"""
        path = self.run.status_file
        status_temps = self._status_temporary_files()
        if status_temps:
            names = ", ".join(item.name for item in status_temps)
            raise StateUnknownError(
                f"Uncommitted update_status temporary file(s) exist in "
                f"{self.run.steps}: {names}"
            )
        if path.is_symlink():
            raise StateUnknownError(
                f"update_status.json must not be a symbolic link: {path}"
            )
        if not path.exists():
            intermediates = self._intermediate_files()
            if intermediates:
                names = ", ".join(item.name for item in intermediates)
                raise StateUnknownError(
                    f"update_status.json is missing while intermediate files exist "
                    f"in {self.run.steps}: {names}"
                )
            return UpdateStatus.READY
        if not path.is_file():
            raise StateUnknownError(
                f"update_status.json exists but is not a regular file: {path}"
            )

        try:
            data: Any = json.loads(path.read_text(encoding="utf-8"))
        except (OSError, UnicodeDecodeError, json.JSONDecodeError) as error:
            raise StateUnknownError(
                f"Cannot read valid update_status.json at {path}: {error}"
            ) from error
        if not isinstance(data, dict) or not isinstance(data.get("status"), str):
            raise StateUnknownError(
                f"update_status.json has an unknown format: {path}"
            )
        try:
            return UpdateStatus(data["status"])
        except ValueError as error:
            raise StateUnknownError(
                f"update_status.json has an unknown status "
                f"{data['status']!r}: {path}"
            ) from error

    def mark_in_progress(self) -> None:
        """ready状態から処理中へ遷移し、状態を永続化する。"""
        current = self.read()
        if current is not UpdateStatus.READY:
            raise InvalidStateTransitionError(
                f"Cannot mark {current.value} run as in_progress"
            )
        self._write(UpdateStatus.IN_PROGRESS)

    def mark_ready_after_restore(self, *, workbook_restored: bool) -> None:
        """元ブックの復元を確認してから処理中状態をreadyへ戻す。"""
        if not workbook_restored:
            raise InvalidStateTransitionError(
                "Cannot mark the run ready because the original workbook was not restored"
            )
        current = self.read()
        if current is not UpdateStatus.IN_PROGRESS:
            raise InvalidStateTransitionError(
                f"Cannot mark {current.value} run as ready after restore"
            )
        self._write(UpdateStatus.READY)

    def mark_applied(self) -> None:
        """処理中状態を反映済みに遷移させ、状態を永続化する。"""
        current = self.read()
        if current is not UpdateStatus.IN_PROGRESS:
            raise InvalidStateTransitionError(
                f"Cannot mark {current.value} run as applied"
            )
        self._write(UpdateStatus.APPLIED)

    def archive_intermediates(self) -> Path | None:
        """ready状態で残っている02の途中成果物を一意なhistory先へ退避する。"""
        current = self.read()
        if current is not UpdateStatus.READY:
            raise InvalidStateTransitionError(
                f"Cannot archive intermediates for {current.value} run"
            )

        files = self._intermediate_files()
        if not files:
            return None

        history_directory = self.run.directory / "history"
        try:
            history_directory.mkdir(exist_ok=True)
            archive_directory = history_directory / (
                f"{datetime.now().strftime('%Y%m%dT%H%M%S%f')}_{uuid.uuid4().hex}"
            )
            archive_directory.mkdir()
        except OSError as error:
            raise HistoryArchiveError(
                f"Cannot create history archive under {history_directory}: {error}"
            ) from error

        try:
            for source in files:
                if source.is_symlink():
                    raise OSError(f"Refusing to archive a symbolic link: {source}")
                if not source.is_file():
                    raise OSError(
                        f"Refusing to archive a non-file intermediate: {source}"
                    )
                shutil.move(str(source), str(archive_directory / source.name))
        except OSError as error:
            raise HistoryArchiveError(
                f"Could not archive all intermediate files. Partial archive, if any, "
                f"is retained at {archive_directory}: {error}"
            ) from error
        return archive_directory

    def _write(self, status: UpdateStatus) -> None:
        """状態を同一フォルダの一時ファイル経由で安全に置き換えて保存する。"""
        temporary_path: Path | None = None
        try:
            with tempfile.NamedTemporaryFile(
                mode="w",
                encoding="utf-8",
                newline="\n",
                dir=self.run.steps,
                prefix=".update_status.",
                suffix=".tmp",
                delete=False,
            ) as output_file:
                temporary_path = Path(output_file.name)
                json.dump(
                    {"status": status.value},
                    output_file,
                    ensure_ascii=False,
                    separators=(",", ":"),
                )
                output_file.write("\n")
                output_file.flush()
                os.fsync(output_file.fileno())
            os.replace(temporary_path, self.run.status_file)
        except OSError as error:
            raise RunStateError(
                f"Cannot persist {status.value} state to {self.run.status_file}: {error}"
            ) from error
        finally:
            if temporary_path is not None:
                temporary_path.unlink(missing_ok=True)

    def _intermediate_files(self) -> list[Path]:
        """steps内にある既知の02途中成果物を名前順で返す。"""
        files = [
            self.run.steps / name
            for name in _INTERMEDIATE_NAMES
            if (self.run.steps / name).exists()
            or (self.run.steps / name).is_symlink()
        ]
        return sorted(files, key=lambda path: path.name.casefold())

    def _status_temporary_files(self) -> list[Path]:
        """状態書き込みの未完了を示す一時ファイルを名前順で返す。"""
        return sorted(
            self.run.steps.glob(".update_status.*.tmp"),
            key=lambda path: path.name.casefold(),
        )
