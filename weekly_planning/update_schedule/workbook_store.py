"""ブックのロック、取得、公開、解放を行うストアを定義する。"""

from __future__ import annotations

import os
import shutil
import tempfile
from dataclasses import dataclass
from pathlib import Path
from typing import Protocol, TypeAlias


_LOCKED_WORKBOOK_NAME = "_SU_Others.xlsx"


class WorkbookStoreError(RuntimeError):
    """ブックストアの操作を安全に完了できない場合に送出する例外。"""


@dataclass(frozen=True)
class FileSystemWorkbookTarget:
    """ローカルファイルシステム上の対象ブックを表す。"""

    path: Path

    def __post_init__(self) -> None:
        """対象が絶対パスのExcelブックで、ロック用の予約名でないことを確認する。"""
        path = Path(self.path)
        if not path.is_absolute():
            raise ValueError(f"Workbook target must be an absolute path: {path}")
        if path.suffix.casefold() not in {".xlsx", ".xlsm"}:
            raise ValueError(
                f"Workbook target must have an .xlsx or .xlsm extension: {path}"
            )
        if path.name.casefold() == _LOCKED_WORKBOOK_NAME.casefold():
            raise ValueError(f"Workbook target cannot use the reserved lock name: {path}")
        object.__setattr__(self, "path", path)


@dataclass(frozen=True)
class SharePointWorkbookTarget:
    """SharePoint上の対象ブックを特定する情報を保持する。"""

    site_url: str
    library_name: str
    item_path: str


WorkbookTarget: TypeAlias = FileSystemWorkbookTarget | SharePointWorkbookTarget


class LockHandle:
    """取得元のストア、対象ブックのパス、公開・解放状態を保持する。"""

    __slots__ = (
        "_store_token",
        "_target_path",
        "_locked_path",
        "_published",
        "_released",
    )

    def __init__(
        self,
        store_token: object,
        target_path: Path,
        locked_path: Path,
    ) -> None:
        """取得元のストアと、対象ブックの元名・ロック名のパスを記録する。"""
        self._store_token = store_token
        self._target_path = target_path
        self._locked_path = locked_path
        self._published = False
        self._released = False


class WorkbookStore(Protocol):
    """ブックストアが実装するロック操作のプロトコル。"""

    def acquire(self, target: WorkbookTarget) -> LockHandle:
        """対象ブックをロックし、その操作に使うハンドルを返す。"""
        ...

    def download_locked(self, lock: LockHandle, destination: Path) -> None:
        """ロック中のブックを指定先へコピーする。"""
        ...

    def publish(self, lock: LockHandle, result: Path) -> None:
        """処理結果を対象ブックとして公開する。"""
        ...

    def release(self, lock: LockHandle, *, published: bool) -> None:
        """公開結果を確定するか、元のブックを復元してロックを解放する。"""
        ...


def _move_no_replace(source: Path, destination: Path) -> None:
    """既存の移動先を置き換えずにファイルを移動する。"""
    if os.name == "nt":
        os.rename(source, destination)
        return

    os.link(source, destination)
    try:
        source.unlink()
    except OSError as error:
        try:
            destination.unlink()
        except OSError as cleanup_error:
            raise WorkbookStoreError(
                f"Could not finish moving {source} to {destination}; "
                f"both paths may remain ({cleanup_error})"
            ) from error
        raise


def _copy_to_temporary(source: Path, directory: Path) -> Path:
    """指定ディレクトリに同期済みの一時コピーを作成して返す。"""
    descriptor, temporary_name = tempfile.mkstemp(
        prefix=".SU_Others.",
        suffix=".tmp",
        dir=directory,
    )
    temporary_path = Path(temporary_name)
    try:
        with os.fdopen(descriptor, "wb") as temporary_file:
            with source.open("rb") as source_file:
                shutil.copyfileobj(source_file, temporary_file)
            temporary_file.flush()
            os.fsync(temporary_file.fileno())
    except BaseException:
        temporary_path.unlink(missing_ok=True)
        raise
    return temporary_path


class FileSystemWorkbookStore:
    """ローカルまたはSMB/UNC上のブックをリネームで簡易ロックする。"""

    def __init__(self) -> None:
        """このストアのロックハンドルを識別するトークンを作成する。"""
        self._token = object()

    def acquire(self, target: WorkbookTarget) -> LockHandle:
        """元ブックを既存ファイルを上書きせずロック名へ移動する。"""
        if not isinstance(target, FileSystemWorkbookTarget):
            raise TypeError(
                "FileSystemWorkbookStore requires a FileSystemWorkbookTarget"
            )

        source = target.path
        locked = source.with_name(_LOCKED_WORKBOOK_NAME)
        if not source.is_file():
            raise FileNotFoundError(f"Workbook not found: {source}")
        _move_no_replace(source, locked)
        return LockHandle(self._token, source, locked)

    def download_locked(self, lock: LockHandle, destination: Path) -> None:
        """ロック名のブックを指定先へコピーする。"""
        self._validate_lock(lock)
        self._require_locked_file(lock)

        destination = Path(destination)
        destination_key = os.path.normcase(os.path.abspath(destination))
        protected_paths = {
            os.path.normcase(os.path.abspath(lock._target_path)),
            os.path.normcase(os.path.abspath(lock._locked_path)),
        }
        if destination_key in protected_paths:
            raise ValueError(
                f"Download destination cannot be a server workbook path: {destination}"
            )
        if not destination.parent.is_dir():
            raise FileNotFoundError(
                f"Download destination directory not found: {destination.parent}"
            )

        temporary_path = _copy_to_temporary(
            lock._locked_path,
            destination.parent,
        )
        try:
            self._require_locked_file(lock)
            os.replace(temporary_path, destination)
        finally:
            temporary_path.unlink(missing_ok=True)

    def publish(self, lock: LockHandle, result: Path) -> None:
        """結果ファイルを既存ブックを上書きせずに公開する。"""
        self._validate_lock(lock)
        if lock._published:
            raise WorkbookStoreError("Workbook result has already been published")
        self._require_locked_file(lock)

        result = Path(result)
        if not result.is_file():
            raise FileNotFoundError(f"Workbook result not found: {result}")
        if lock._target_path.exists():
            raise FileExistsError(
                f"Refusing to overwrite an existing workbook: {lock._target_path}"
            )

        temporary_path = _copy_to_temporary(result, lock._target_path.parent)
        try:
            self._require_locked_file(lock)
            _move_no_replace(temporary_path, lock._target_path)
        finally:
            temporary_path.unlink(missing_ok=True)
        lock._published = True

    def release(self, lock: LockHandle, *, published: bool) -> None:
        """公開済みならロックファイルを削除し、未公開なら元ブックを復元する。"""
        self._validate_lock(lock)
        if published:
            if not lock._published:
                raise WorkbookStoreError(
                    "Cannot release as published before a successful publish"
                )
            if not lock._target_path.is_file():
                raise FileNotFoundError(
                    f"Published workbook not found: {lock._target_path}"
                )
            self._require_locked_file(lock)
            lock._locked_path.unlink()
        else:
            if lock._published:
                raise WorkbookStoreError(
                    "Cannot restore the original after publishing the result"
                )
            self._require_locked_file(lock)
            if lock._target_path.exists():
                raise FileExistsError(
                    "Refusing to restore over an existing workbook: "
                    f"{lock._target_path}"
                )
            _move_no_replace(lock._locked_path, lock._target_path)
        lock._released = True

    @staticmethod
    def _require_locked_file(lock: LockHandle) -> None:
        """ロック名のファイルが存在することを確認する。"""
        if not lock._locked_path.is_file():
            raise FileNotFoundError(
                f"Locked workbook not found: {lock._locked_path}"
            )

    def _validate_lock(self, lock: LockHandle) -> None:
        """ハンドルがこのストアで取得され、まだ解放されていないことを検証する。"""
        if not isinstance(lock, LockHandle) or lock._store_token is not self._token:
            raise TypeError("Lock handle was not acquired by this WorkbookStore")
        if lock._released:
            raise WorkbookStoreError("Lock handle has already been released")
