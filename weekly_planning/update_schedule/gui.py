from __future__ import annotations

import queue
import threading
import tkinter as tk
import webbrowser
from dataclasses import dataclass
from pathlib import Path
from tkinter import filedialog, messagebox, ttk
from typing import Callable, Literal

from common import plan_settings
from update_schedule.orchestrator import UpdateResult, run_update
from update_schedule.paths import ApplicationPaths
from update_schedule.workbook_store import SharePointWorkbookStore


_SUPPORTED_SUFFIXES = {".xlsx", ".xlsm"}
_POLL_INTERVAL_MS = 100


class WorkbookInputError(ValueError):
    """Raised when the selected SU_Others workbook path is invalid."""


@dataclass
class _AuthenticationRequest:
    verification_uri: str
    user_code: str
    completed: threading.Event
    accepted: bool = False


@dataclass(frozen=True)
class _WorkerMessage:
    kind: Literal["stage", "authentication", "result", "error"]
    stage: str | None = None
    authentication: _AuthenticationRequest | None = None
    result: UpdateResult | None = None
    error: Exception | None = None


def _selected_path(path_text: str) -> Path | str:
    """入力文字列をローカルパスまたはSharePoint共有リンクとして返す。"""
    value = path_text.strip()
    if not value:
        raise WorkbookInputError("SU_Othersのパスを入力するか、ファイルを選択してください。")
    if value.casefold().startswith("https://"):
        return value

    path = Path(value).expanduser()
    if not path.is_absolute():
        raise WorkbookInputError(f"対象ブックには絶対パスを指定してください: {value}")
    if path.suffix.casefold() not in _SUPPORTED_SUFFIXES:
        raise WorkbookInputError(
            f"対象ブックは .xlsx または .xlsm を指定してください: {path}"
        )
    return path


def _is_unc_path(path: Path) -> bool:
    value = str(path)
    if value.startswith("\\\\?\\"):
        return value.casefold().startswith("\\\\?\\unc\\")
    return value.startswith("\\\\")


def _dialog_initial_directory(path_text: str) -> str | None:
    """入力済みの有効なローカルブックの親フォルダを返す。"""
    try:
        path = _selected_path(path_text)
    except WorkbookInputError:
        return None
    if isinstance(path, str):
        return None
    if not _is_unc_path(path) and path.is_file():
        return str(path.parent)
    return None


def _format_result(result: UpdateResult) -> str:
    """オーケストレーション結果を利用者向けの文面に変換する。"""
    if result.status == "already_applied":
        return (
            "この計画回は反映済みです。更新はありません。\n"
            f"対象: {result.workbook_path}"
        )
    if result.status == "not_finished":
        return (
            "計画結果はまだ完了していません。ブックは復元済みです。"
            "後から再実行できます。\n"
            f"対象: {result.workbook_path}"
        )
    if result.status != "completed":
        raise ValueError(f"Unsupported update result status: {result.status!r}")

    if result.warnings or result.warning_count or result.conflict_count:
        lines = [
            "警告ありで反映が完了しました。",
            f"警告: {result.warning_count}件、競合: {result.conflict_count}件",
            f"反映先: {result.workbook_path}",
        ]
        lines.extend(result.warnings)
        return "\n".join(lines)
    return f"反映が完了しました。\n反映先: {result.workbook_path}"


def _format_error(error: Exception, selected_path: str) -> str:
    """例外を画面表示し、共有先で確認すべき点を必要に応じて案内する。"""
    message = f"エラー: {error}"
    try:
        path = _selected_path(selected_path)
    except WorkbookInputError:
        return message
    if isinstance(path, str):
        if isinstance(error, PermissionError):
            message += "\nSharePointのサインイン状態と対象ライブラリの権限を確認してください。"
        return message
    if isinstance(error, PermissionError):
        message += (
            "\n対象ブックが開かれているか、アクセス権が不足している可能性があります。"
        )
    if _is_unc_path(path) and isinstance(error, OSError):
        message += (
            "\n共有先へのアクセスまたはリネームに失敗した可能性があります。"
            "リネーム失敗時は対象ブックを利用している全員にファイルを閉じてもらい、"
            "共有先のアクセス権も確認してください。"
        )
    return message


class UpdateScheduleApp:
    """Tkinter画面からSU_Othersへの計画結果反映を実行する。"""

    def __init__(
        self,
        root: tk.Tk,
        *,
        paths: ApplicationPaths | None = None,
        update_runner: Callable[..., UpdateResult] = run_update,
    ) -> None:
        self.root = root
        self.paths = paths or ApplicationPaths.current()
        self.update_runner = update_runner
        self._messages: queue.Queue[_WorkerMessage] = queue.Queue()
        self._running = False

        self.path_variable = tk.StringVar(master=root, value="")
        self.status_variable = tk.StringVar(master=root, value="SU_Othersを選択してください。")
        self.result_variable = tk.StringVar(master=root, value="")

        self.root.title("02_UpdateSchedule")
        self.root.geometry("800x300")
        self.root.minsize(620, 250)

        frame = ttk.Frame(root, padding=16)
        frame.grid(row=0, column=0, sticky="nsew")
        frame.columnconfigure(1, weight=1)
        root.columnconfigure(0, weight=1)
        root.rowconfigure(0, weight=1)

        ttk.Label(frame, text="SU_Others").grid(
            row=0, column=0, padx=(0, 8), sticky="w"
        )
        self.path_entry = ttk.Entry(frame, textvariable=self.path_variable)
        self.path_entry.grid(row=0, column=1, sticky="ew")
        self.browse_button = ttk.Button(
            frame, text="ファイル選択...", command=self._browse
        )
        self.browse_button.grid(row=0, column=2, padx=(8, 0))

        self.start_button = ttk.Button(
            frame, text="開始", command=self._start
        )
        self.start_button.grid(row=1, column=2, pady=(16, 8), sticky="e")
        ttk.Label(frame, textvariable=self.status_variable).grid(
            row=2, column=0, columnspan=3, pady=(12, 4), sticky="w"
        )
        ttk.Label(
            frame,
            textvariable=self.result_variable,
            justify="left",
            anchor="nw",
            wraplength=740,
        ).grid(row=3, column=0, columnspan=3, pady=(4, 0), sticky="nsew")
        frame.rowconfigure(3, weight=1)

        self.root.protocol("WM_DELETE_WINDOW", self._close)
        self.root.after(_POLL_INTERVAL_MS, self._poll_messages)

    def _browse(self) -> None:
        options = {
            "parent": self.root,
            "title": "SU_Othersを選択",
            "filetypes": [
                ("Excelブック", "*.xlsx *.xlsm"),
                ("すべてのファイル", "*.*"),
            ],
        }
        initial_directory = _dialog_initial_directory(self.path_variable.get())
        if initial_directory is not None:
            options["initialdir"] = initial_directory
        selected = filedialog.askopenfilename(**options)
        if selected:
            self.path_variable.set(selected)
            self.result_variable.set("")
            self.status_variable.set("開始ボタンを押してください。")

    def _start(self) -> None:
        if self._running:
            return
        selected_path = self.path_variable.get()
        try:
            workbook_path = _selected_path(selected_path)
        except WorkbookInputError as error:
            self.status_variable.set("入力を確認してください。")
            self.result_variable.set(str(error))
            return

        self._running = True
        self._set_inputs_enabled(False)
        self.start_button.configure(state="disabled")
        self.status_variable.set("入力を確認しています...")
        self.result_variable.set("")
        worker = threading.Thread(
            target=self._run_worker,
            args=(workbook_path,),
            name="UpdateScheduleWorker",
            daemon=False,
        )
        try:
            worker.start()
        except RuntimeError as error:
            self._running = False
            self._set_inputs_enabled(True)
            self.start_button.configure(state="normal")
            self.status_variable.set("処理を開始できませんでした。")
            self.result_variable.set(_format_error(error, selected_path))
            return

    def _run_worker(self, workbook_path: Path) -> None:
        try:
            arguments = {
                "on_stage": lambda stage: self._messages.put(
                    _WorkerMessage(kind="stage", stage=stage)
                )
            }
            if isinstance(workbook_path, str):
                arguments["store"] = self._create_sharepoint_store()
            result = self.update_runner(
                self.paths,
                workbook_path,
                **arguments,
            )
        except Exception as error:
            self._messages.put(_WorkerMessage(kind="error", error=error))
            return
        self._messages.put(_WorkerMessage(kind="result", result=result))

    def _create_sharepoint_store(self) -> SharePointWorkbookStore:
        """設定とGUI認証callbackを使ってSharePoint Storeを作る。"""
        settings = plan_settings.load(path=self.paths.config_file)
        tenant_id = settings.text("tenant_id")
        client_id = settings.text("client_id")
        if tenant_id is None or client_id is None:
            raise WorkbookInputError(
                "SharePointを利用するには tenant_id と client_id を設定してください。"
            )

        from update_schedule.sharepoint_auth import UpdateScheduleFileOperationSharePoint

        operation = UpdateScheduleFileOperationSharePoint(
            tenant_id,
            client_id,
            self._request_sign_in,
        )
        return SharePointWorkbookStore(operation)

    def _request_sign_in(self, verification_uri: str, user_code: str) -> bool:
        """workerの認証表示要求をGUIへ送り、利用者の応答を待つ。"""
        request = _AuthenticationRequest(
            verification_uri=verification_uri,
            user_code=user_code,
            completed=threading.Event(),
        )
        self._messages.put(
            _WorkerMessage(kind="authentication", authentication=request)
        )
        request.completed.wait()
        return request.accepted

    def _show_authentication_dialog(
        self,
        request: _AuthenticationRequest,
    ) -> None:
        """GUI threadでデバイスコード認証画面を表示する。"""
        dialog: tk.Toplevel | None = None
        try:
            dialog = tk.Toplevel(self.root)
            dialog.title("SharePointへのサインイン")
            dialog.transient(self.root)

            def finish(accepted: bool) -> None:
                request.accepted = accepted
                request.completed.set()
                dialog.destroy()

            frame = ttk.Frame(dialog, padding=16)
            frame.pack(fill="both", expand=True)
            ttk.Label(
                frame,
                text="コードをコピーしてブラウザーでサインインしてください。",
            ).pack(anchor="w")
            for label, value in (
                ("サインインURL:", request.verification_uri),
                ("サインインコード:", request.user_code),
            ):
                ttk.Label(frame, text=label).pack(anchor="w", pady=(10, 0))
                entry = ttk.Entry(frame, width=65)
                entry.insert(0, value)
                entry.configure(state="readonly")
                entry.pack(fill="x")

            buttons = ttk.Frame(frame)
            buttons.pack(pady=(12, 0))
            ttk.Button(buttons, text="OK", command=lambda: finish(True)).pack(
                side="left", padx=5
            )
            ttk.Button(buttons, text="キャンセル", command=lambda: finish(False)).pack(
                side="left", padx=5
            )
            dialog.protocol("WM_DELETE_WINDOW", lambda: finish(False))
            try:
                webbrowser.open(request.verification_uri)
            except (OSError, webbrowser.Error):
                pass
            dialog.wait_window()
        finally:
            request.completed.set()
            if dialog is not None and dialog.winfo_exists():
                dialog.destroy()

    def _poll_messages(self) -> None:
        while True:
            try:
                message = self._messages.get_nowait()
            except queue.Empty:
                break

            if message.kind == "stage":
                if message.stage is not None:
                    self.status_variable.set(message.stage)
            elif message.kind == "authentication":
                if message.authentication is not None:
                    self._show_authentication_dialog(message.authentication)
            elif message.kind == "result":
                if message.result is not None:
                    self.status_variable.set("処理が終了しました。")
                    self.result_variable.set(_format_result(message.result))
                else:
                    self.status_variable.set("処理結果を取得できませんでした。")
                    self.result_variable.set(
                        "処理結果がありません。Workと対象ブックの状態を確認してください。"
                    )
                self._finish()
            elif message.kind == "error":
                if message.error is not None:
                    self.status_variable.set("エラーが発生しました。")
                    self.result_variable.set(
                        _format_error(message.error, self.path_variable.get())
                    )
                else:
                    self.status_variable.set("エラーが発生しました。")
                    self.result_variable.set("エラーの詳細を取得できませんでした。")
                self._finish()

        self.root.after(_POLL_INTERVAL_MS, self._poll_messages)

    def _finish(self) -> None:
        self._running = False
        self._set_inputs_enabled(True)
        self.start_button.configure(state="normal")

    def _set_inputs_enabled(self, enabled: bool) -> None:
        state = "normal" if enabled else "disabled"
        self.path_entry.configure(state=state)
        self.browse_button.configure(state=state)

    def _close(self) -> None:
        if self._running:
            messagebox.showinfo(
                "処理中",
                "処理中は終了できません。",
                parent=self.root,
            )
            return
        self.root.destroy()


def main() -> None:
    """02_UpdateScheduleのTkinter画面を起動する。"""
    root = tk.Tk()
    UpdateScheduleApp(root)
    root.mainloop()


if __name__ == "__main__":
    main()
