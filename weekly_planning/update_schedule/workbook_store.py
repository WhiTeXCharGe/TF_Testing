"""ブックのロック、取得、公開、解放を行うストアを定義する。"""

from __future__ import annotations

import os
import shutil
import tempfile
from dataclasses import dataclass
from pathlib import Path
from typing import Protocol, TypeAlias


_LOCKED_WORKBOOK_NAME = "_SU_Others.xlsx"
_SUPPORTED_WORKBOOK_SUFFIXES = {".xlsx", ".xlsm"}


def _validate_workbook_name(name: str) -> str:
    """SU_Othersのファイル名と拡張子を検証する。"""
    if not name or name in {".", ".."} or "/" in name or "\\" in name:
        raise ValueError(f"Invalid workbook file name: {name}")
    if Path(name).suffix.casefold() not in _SUPPORTED_WORKBOOK_SUFFIXES:
        raise ValueError(
            f"Workbook target must have an .xlsx or .xlsm extension: {name}"
        )
    if name.casefold() == _LOCKED_WORKBOOK_NAME.casefold():
        raise ValueError(f"Workbook target cannot use the reserved lock name: {name}")
    return name


class WorkbookStoreError(RuntimeError):
    """ブックストアの操作を安全に完了できない場合に送出する例外。"""


class PublicationOutcomeUnknown(WorkbookStoreError):
    """公開要求後にSharePoint上の結果を確定できない場合の例外。"""


@dataclass(frozen=True)
class FileSystemWorkbookTarget:
    """ローカルファイルシステム上の対象ブックを表す。"""

    path: Path

    def __post_init__(self) -> None:
        """対象が絶対パスのExcelブックで、ロック用の予約名でないことを確認する。"""
        path = Path(self.path)
        if not path.is_absolute():
            raise ValueError(f"Workbook target must be an absolute path: {path}")
        _validate_workbook_name(path.name)
        object.__setattr__(self, "path", path)


@dataclass(frozen=True)
class SharePointWorkbookTarget:
    """SharePoint上の対象ブックを共有リンクで特定する。"""

    sharing_link: str

    def __post_init__(self) -> None:
        sharing_link = str(self.sharing_link).strip()
        if not sharing_link:
            raise ValueError("SharePoint sharing link cannot be empty")
        if not sharing_link.casefold().startswith("https://"):
            raise ValueError(f"SharePoint sharing link must use HTTPS: {sharing_link}")
        object.__setattr__(self, "sharing_link", sharing_link)


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


@dataclass
class _SharePointLockHandle:
    """SharePoint Storeだけが解釈できるロック状態を保持する。"""

    store_token: object
    sharing_link: str
    original_url: str
    original_name: str
    parent_url: str
    locked_url: str
    published_url: str | None = None
    publication_unknown: bool = False
    released: bool = False


class FileOperationSharePointLike(Protocol):
    """SharePoint Storeが利用する既存Helperの操作契約。"""

    def get_file_url_from_sharing_link(self, sharing_link: str) -> str: ...

    def get_parent_folder_path(self, file_path: str) -> str: ...

    def download_file(self, file_path: str, local_path: str | os.PathLike[str]) -> None: ...

    def get_file_name(self, file_path: str) -> str: ...

    def rename_file(self, file_path: str, new_name: str) -> str: ...

    def upload_file(
        self,
        local_path: str | os.PathLike[str],
        parent_folder_path: str,
        file_name: str,
    ) -> str: ...

    def delete_file(self, file_path: str) -> None: ...


class WorkbookStore(Protocol):
    """ブックストアが実装するロック操作のプロトコル。"""

    def workbook_name(self, target: WorkbookTarget) -> str:
        """ロック前に対象ブック名を解決して返す。"""
        ...

    def acquire(self, target: WorkbookTarget) -> object:
        """対象ブックをロックし、その操作に使うハンドルを返す。"""
        ...

    def download_locked(self, lock: object, destination: Path) -> None:
        """ロック中のブックを指定先へコピーする。"""
        ...

    def publish(self, lock: object, result: Path) -> None:
        """処理結果を対象ブックとして公開する。"""
        ...

    def release(self, lock: object, *, published: bool) -> None:
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

    def workbook_name(self, target: WorkbookTarget) -> str:
        """ロック前に対象のファイル名を返す。"""
        if not isinstance(target, FileSystemWorkbookTarget):
            raise TypeError(
                "FileSystemWorkbookStore requires a FileSystemWorkbookTarget"
            )
        return target.path.name

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


@dataclass(frozen=True)
class _ResolvedSharePointWorkbook:
    """共有リンクから解決したSharePointアイテムの情報を保持する。"""

    original_url: str
    original_name: str
    parent_url: str


class SharePointWorkbookStore:
    """既存Helperを使ってSharePoint上のブックを簡易ロックする。"""

    def __init__(self, operation: FileOperationSharePointLike) -> None:
        """Graph操作を委譲するHelperとこのStoreの識別子を保持する。"""
        self._operation = operation
        self._token = object()
        self._resolved: dict[str, _ResolvedSharePointWorkbook] = {}

    def workbook_name(self, target: WorkbookTarget) -> str:
        """共有リンクをロック前に解決し、元のファイル名を返す。"""
        return self._resolve(target).original_name

    def acquire(self, target: WorkbookTarget) -> _SharePointLockHandle:
        """元アイテムを既存名を上書きせずロック名へ変更する。"""
        resolved = self._resolve(target)
        try:
            locked_url = self._operation.rename_file(
                resolved.original_url,
                _LOCKED_WORKBOOK_NAME,
            )
        except Exception as error:
            raise WorkbookStoreError(
                "SharePoint rename outcome must be checked manually; "
                f"link={target.sharing_link}, original={resolved.original_name}, "
                f"original_url={resolved.original_url}, "
                f"locked_name={_LOCKED_WORKBOOK_NAME}: {error}"
            ) from error
        return _SharePointLockHandle(
            store_token=self._token,
            sharing_link=target.sharing_link,
            original_url=resolved.original_url,
            original_name=resolved.original_name,
            parent_url=resolved.parent_url,
            locked_url=locked_url,
        )

    def download_locked(
        self,
        lock: object,
        destination: Path,
    ) -> None:
        """ロック中のSharePointアイテムをローカルへ取得する。"""
        handle = self._validate_lock(lock)
        self._operation.download_file(handle.locked_url, destination)

    def publish(self, lock: object, result: Path) -> None:
        """反映結果を元名の新しいSharePointアイテムとして公開する。"""
        handle = self._validate_lock(lock)
        if handle.published_url is not None:
            raise WorkbookStoreError("Workbook result has already been published")
        result = Path(result)
        if not result.is_file():
            raise FileNotFoundError(f"Workbook result not found: {result}")
        try:
            handle.published_url = self._operation.upload_file(
                result,
                handle.parent_url,
                handle.original_name,
            )
        except Exception as error:
            if isinstance(error, FileExistsError) or getattr(
                error, "status_code", None
            ) == 409:
                raise
            handle.publication_unknown = True
            raise PublicationOutcomeUnknown(
                "SharePoint publication outcome is unknown. Do not retry "
                f"automatically; inspect link={handle.sharing_link}, "
                f"original={handle.original_name}, "
                f"locked={handle.locked_url}. Cause: {error}"
            ) from error

    def release(self, lock: object, *, published: bool) -> None:
        """公開済みの旧アイテムを削除するか、元名へ復元する。"""
        handle = self._validate_lock(lock)
        if handle.publication_unknown:
            raise WorkbookStoreError(
                "Cannot release while SharePoint publication outcome is unknown"
            )
        if published:
            if handle.published_url is None:
                raise WorkbookStoreError(
                    "Cannot release as published before a successful publish"
                )
            self._operation.delete_file(handle.locked_url)
        else:
            if handle.published_url is not None:
                raise WorkbookStoreError(
                    "Cannot restore the original after publishing the result"
                )
            restored_url = self._operation.rename_file(
                handle.locked_url,
                handle.original_name,
            )
            if not restored_url:
                raise WorkbookStoreError(
                    "SharePoint did not return the restored workbook URL"
                )
        handle.released = True

    def _resolve(self, target: WorkbookTarget) -> _ResolvedSharePointWorkbook:
        """共有リンクの解決結果を同一Store内でキャッシュする。"""
        if not isinstance(target, SharePointWorkbookTarget):
            raise TypeError(
                "SharePointWorkbookStore requires a SharePointWorkbookTarget"
            )
        resolved = self._resolved.get(target.sharing_link)
        if resolved is not None:
            return resolved

        original_url = self._operation.get_file_url_from_sharing_link(
            target.sharing_link
        )
        original_name = _validate_workbook_name(
            self._operation.get_file_name(original_url)
        )
        parent_url = self._operation.get_parent_folder_path(original_url)
        resolved = _ResolvedSharePointWorkbook(
            original_url=original_url,
            original_name=original_name,
            parent_url=parent_url,
        )
        self._resolved[target.sharing_link] = resolved
        return resolved

    def _validate_lock(self, lock: object) -> _SharePointLockHandle:
        """ハンドルがこのStoreで取得され、未解放であることを確認する。"""
        if (
            not isinstance(lock, _SharePointLockHandle)
            or lock.store_token is not self._token
        ):
            raise TypeError("Lock handle was not acquired by this WorkbookStore")
        if lock.released:
            raise WorkbookStoreError("Lock handle has already been released")
        return lock
