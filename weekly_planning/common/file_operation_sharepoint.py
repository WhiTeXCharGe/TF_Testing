import base64
import os
import tempfile
import time
import tkinter as tk
import webbrowser
from pathlib import Path
from typing import Any
from urllib.parse import quote, unquote, urlsplit

import msal
import requests
from msal.exceptions import MsalError


GRAPH_BASE_URL = "https://graph.microsoft.com/v1.0"
GRAPH_SCOPES = ["Files.ReadWrite"]
REQUEST_TIMEOUT_SECONDS = 30
UPLOAD_CHUNK_SIZE = 10 * 1024 * 1024
RENAME_PATH_WAIT_TIMEOUT_SECONDS = 10.0
RENAME_PATH_POLL_INTERVAL_SECONDS = 0.5


class FileOperationSharePointError(Exception):
    """SharePoint file operation failure."""

    def __init__(self, message: str, *, status_code: int | None = None) -> None:
        super().__init__(message)
        self.status_code = status_code


class FileOperationSharePoint:
    def __init__(self, tenant_id: str, client_id: str) -> None:
        if not tenant_id or not client_id:
            raise ValueError("tenant_id、client_idは必須です。")

        self._drive_id: str | None = None
        self._msal_app = msal.PublicClientApplication(
            client_id=client_id,
            authority=f"https://login.microsoftonline.com/{tenant_id}",
        )
        self._drive_root_web_url: str | None = None
        self._item_id_cache: dict[str, str] = {}  # URLやパスからitemIDをキャッシュ

    def get_file_url_from_sharing_link(self, sharing_link: str) -> str:
        if not sharing_link:
            raise FileOperationSharePointError("共有リンクが空です。")

        encoded_link = base64.urlsafe_b64encode(
            sharing_link.encode("utf-8")
        ).decode("ascii").rstrip("=")
        share_id = f"u!{encoded_link}"
        endpoint = (
            f"{GRAPH_BASE_URL}/shares/{share_id}/driveItem"
            "?$select=id,name,webUrl,parentReference,file,folder"
        )
        response = self._request_graph("GET", endpoint)
        item = self._response_json(response, "共有リンクの情報を取得できませんでした。")
        self._require_drive(item, require_drive_id=True)
        self._require_file(item)

        # アイテムのIDをキャッシュ（後で直接使用可能）
        item_id = item.get("id")
        if item_id:
            self._item_id_cache[sharing_link] = item_id

        file_path = item.get("webUrl")
        if not isinstance(file_path, str) or not file_path:
            raise FileOperationSharePointError(
                "SharePointから通常のファイルパスが返されませんでした。"
            )
        
        # webUrlをキャッシュキーとしても保存
        if item_id and file_path:
            self._item_id_cache[file_path] = item_id
        
        return file_path

    def get_parent_folder_path(self, file_path: str) -> str:
        item = self._get_file_item(file_path)
        parent_reference = item.get("parentReference", {})
        parent_id = parent_reference.get("id")
        parent_drive_id = parent_reference.get("driveId")
        if not parent_id:
            raise FileOperationSharePointError(
                "対象ファイルの親フォルダーを特定できませんでした。"
            )
        if parent_drive_id and parent_drive_id != self._drive_id:
            raise FileOperationSharePointError(
                "対象ファイルは設定されたドキュメントライブラリにありません。"
            )

        parent = self._get_item_by_id(parent_id)
        if "folder" not in parent:
            raise FileOperationSharePointError(
                "対象ファイルの親アイテムがフォルダーではありません。"
            )

        parent_path = parent.get("webUrl")
        if not isinstance(parent_path, str) or not parent_path:
            raise FileOperationSharePointError(
                "親フォルダーのパスを取得できませんでした。"
            )
        return parent_path

    def download_file(self, file_path: str, local_path: str | os.PathLike[str]) -> None:
        item = self._get_file_item(file_path)
        item_id = self._required_item_id(item)
        destination = Path(local_path)

        if destination.exists():
            raise FileOperationSharePointError(
                "ダウンロード先にファイルが既に存在します。"
            )
        if not destination.parent.is_dir():
            raise FileOperationSharePointError(
                "ダウンロード先の親フォルダーが存在しません。"
            )

        endpoint = (
            f"{GRAPH_BASE_URL}/drives/{quote(self._drive_id, safe='')}"
            f"/items/{quote(item_id, safe='')}/content"
        )
        response = self._request_graph("GET", endpoint, stream=True)
        temporary_path: str | None = None
        try:
            descriptor, temporary_path = tempfile.mkstemp(
                prefix=f".{destination.name}.",
                suffix=".download",
                dir=str(destination.parent),
            )
            with os.fdopen(descriptor, "wb") as downloaded_file:
                for chunk in response.iter_content(chunk_size=1024 * 1024):
                    if chunk:
                        downloaded_file.write(chunk)
                downloaded_file.flush()
                os.fsync(downloaded_file.fileno())

            os.rename(temporary_path, destination)
            temporary_path = None
        except (OSError, requests.RequestException) as exc:
            raise FileOperationSharePointError(
                "SharePointファイルをローカルに保存できませんでした。"
            ) from exc
        finally:
            response.close()
            self._remove_temporary_file(temporary_path)

    def get_file_name(self, file_path: str) -> str:
        item = self._get_file_item(file_path)
        name = item.get("name")
        if not isinstance(name, str) or not name:
            raise FileOperationSharePointError(
                "SharePointからファイル名を取得できませんでした。"
            )
        return name

    def rename_file(self, file_path: str, new_name: str) -> str:
        if not new_name or "/" in new_name or "\\" in new_name:
            raise FileOperationSharePointError(
                "変更後のファイル名には、空文字やパス区切り文字を指定できません。"
            )

        item = self._get_file_item(file_path)
        item_id = self._required_item_id(item)
        endpoint = (
            f"{GRAPH_BASE_URL}/drives/{quote(self._drive_id, safe='')}"
            f"/items/{quote(item_id, safe='')}"
        )
        response = self._request_graph(
            "PATCH",
            endpoint,
            json_body={"name": new_name},
        )
        updated_item = self._response_json(
            response,
            "ファイル名を変更できませんでした。",
        )
        if updated_item.get("name") != new_name:
            raise FileOperationSharePointError(
                "SharePointから返されたファイル名が指定した名前と一致しません。"
            )

        updated_path = updated_item.get("webUrl")
        if not isinstance(updated_path, str) or not updated_path:
            raise FileOperationSharePointError(
                "ファイル名変更後のパスを取得できませんでした。"
            )

        self._item_id_cache[updated_path] = item_id
        deadline = time.monotonic() + RENAME_PATH_WAIT_TIMEOUT_SECONDS
        last_not_found: FileOperationSharePointError | None = None
        while True:
            remaining = deadline - time.monotonic()
            if remaining <= 0:
                message = (
                    "ファイル名変更後のファイルを"
                    f"{RENAME_PATH_WAIT_TIMEOUT_SECONDS:g}秒以内に"
                    "確認できませんでした。"
                )
                if last_not_found is not None:
                    raise FileOperationSharePointError(
                        f"{message}\n最後のエラー: {last_not_found}"
                    ) from last_not_found
                raise FileOperationSharePointError(message)

            try:
                renamed_item = self._get_item_by_path(
                    updated_path,
                    timeout_seconds=min(REQUEST_TIMEOUT_SECONDS, remaining),
                )
            except FileOperationSharePointError as exc:
                if exc.status_code != 404:
                    raise
                last_not_found = exc
            else:
                if renamed_item.get("id") != item_id:
                    raise FileOperationSharePointError(
                        "ファイル名変更後のパスが別のアイテムを指しています。"
                    )
                if renamed_item.get("name") == new_name:
                    return updated_path

            sleep_seconds = min(
                RENAME_PATH_POLL_INTERVAL_SECONDS,
                max(0.0, deadline - time.monotonic()),
            )
            if sleep_seconds:
                time.sleep(sleep_seconds)

    def upload_file(
        self,
        local_path: str | os.PathLike[str],
        parent_folder_path: str,
        file_name: str,
    ) -> str:
        if not file_name or file_name in {".", ".."} or "/" in file_name or "\\" in file_name:
            raise FileOperationSharePointError("アップロード先のファイル名が不正です。")

        source = Path(local_path)
        if not source.is_file():
            raise FileOperationSharePointError(
                "アップロード元のファイルが存在しません。"
            )

        parent = self._get_folder_item(parent_folder_path)
        parent_id = self._required_item_id(parent)
        try:
            source_size = source.stat().st_size
        except OSError as exc:
            raise FileOperationSharePointError(
                "アップロード元ファイルの情報を取得できませんでした。"
            ) from exc
        if source_size == 0:
            return self._create_empty_file(parent_id, file_name)
        return self._upload_with_session(source, parent_id, file_name, source_size)

    def delete_file(self, file_path: str) -> None:
        item = self._get_file_item(file_path)
        item_id = self._required_item_id(item)
        endpoint = (
            f"{GRAPH_BASE_URL}/drives/{quote(self._drive_id, safe='')}"
            f"/items/{quote(item_id, safe='')}"
        )
        self._request_graph("DELETE", endpoint)

    def _get_file_item(self, file_path: str) -> dict[str, Any]:
        item = self._get_item_by_path(file_path)
        self._require_file(item)
        return item

    def _get_folder_item(self, folder_path: str) -> dict[str, Any]:
        item = self._get_item_by_path(folder_path)
        if "folder" not in item:
            raise FileOperationSharePointError(
                "指定されたパスはフォルダーではありません。"
            )
        return item

    def _get_item_by_path(
        self,
        file_path: str,
        *,
        timeout_seconds: float = REQUEST_TIMEOUT_SECONDS,
    ) -> dict[str, Any]:
        # キャッシュにitemIDがあれば、直接そのIDを使用
        if file_path in self._item_id_cache:
            return self._get_item_by_id(
                self._item_id_cache[file_path],
                timeout_seconds=timeout_seconds,
            )
        
        root_web_url = self._get_drive_root_web_url(
            timeout_seconds=timeout_seconds,
        )
        relative_path = self._relative_path_from_url(file_path, root_web_url)
        encoded_path = quote(relative_path, safe="/")
        if encoded_path:
            item_path = f"/root:/{encoded_path}"
        else:
            item_path = "/root"

        endpoint = (
            f"{GRAPH_BASE_URL}/drives/{quote(self._drive_id, safe='')}{item_path}"
        )
        response = self._request_graph(
            "GET",
            endpoint,
            params={
                "$select": "id,name,webUrl,parentReference,file,folder",
            },
            timeout_seconds=timeout_seconds,
        )
        item = self._response_json(response, "指定されたパスのアイテムを取得できませんでした。")
        self._require_drive(item)
        return item

    def _get_item_by_id(
        self,
        item_id: str,
        *,
        timeout_seconds: float = REQUEST_TIMEOUT_SECONDS,
    ) -> dict[str, Any]:
        endpoint = (
            f"{GRAPH_BASE_URL}/drives/{quote(self._drive_id, safe='')}"
            f"/items/{quote(item_id, safe='')}"
        )
        response = self._request_graph(
            "GET",
            endpoint,
            params={"$select": "id,name,webUrl,parentReference,file,folder"},
            timeout_seconds=timeout_seconds,
        )
        item = self._response_json(response, "SharePointアイテムを取得できませんでした。")
        self._require_drive(item)
        return item

    def _get_drive_root_web_url(
        self,
        *,
        timeout_seconds: float = REQUEST_TIMEOUT_SECONDS,
    ) -> str:
        if self._drive_root_web_url is not None:
            return self._drive_root_web_url

        endpoint = (
            f"{GRAPH_BASE_URL}/drives/{quote(self._drive_id, safe='')}/root"
        )
        response = self._request_graph(
            "GET",
            endpoint,
            params={"$select": "id,name,webUrl,folder"},
            timeout_seconds=timeout_seconds,
        )
        root = self._response_json(
            response,
            "ドキュメントライブラリのルートを取得できませんでした。",
        )
        root_web_url = root.get("webUrl")
        if not isinstance(root_web_url, str) or not root_web_url:
            raise FileOperationSharePointError(
                "ドキュメントライブラリのルートパスが返されませんでした。"
            )
        self._drive_root_web_url = root_web_url
        return root_web_url

    @staticmethod
    def _relative_path_from_url(file_path: str, root_web_url: str) -> str:
        # ビューページのURL形式（/_layouts/15/Doc.aspx...）の場合、
        # クエリパラメータから相対パスを抽出
        if "/_layouts/15/Doc.aspx" in file_path:
            # クエリパラメータから 'file' パラメータを抽出
            file_parts_obj = urlsplit(file_path)
            query_params = {}
            if file_parts_obj.query:
                for param in file_parts_obj.query.split("&"):
                    if "=" in param:
                        key, value = param.split("=", 1)
                        query_params[key] = unquote(value)
            
            file_name = query_params.get("file")
            if file_name:
                return file_name
            else:
                raise FileOperationSharePointError(
                    "ビューページURLからファイル名を抽出できませんでした。"
                )
        
        try:
            file_parts = urlsplit(file_path)
            root_parts = urlsplit(root_web_url)
        except ValueError as exc:
            raise FileOperationSharePointError(
                "SharePointのパス形式が不正です。"
            ) from exc

        if (
            file_parts.scheme.lower() != "https"
            or file_parts.netloc.casefold() != root_parts.netloc.casefold()
            or root_parts.scheme.lower() != "https"
        ):
            raise FileOperationSharePointError(
                "指定されたパスは対象ドキュメントライブラリのURLではありません。"
            )

        root_segments = [
            segment for segment in unquote(root_parts.path).split("/") if segment
        ]
        item_segments = [
            segment for segment in unquote(file_parts.path).split("/") if segment
        ]
        if len(item_segments) < len(root_segments) or any(
            root_segment.casefold() != item_segment.casefold()
            for root_segment, item_segment in zip(
                root_segments,
                item_segments[: len(root_segments)],
            )
        ):
            raise FileOperationSharePointError(
                "指定されたパスは対象ドキュメントライブラリの外にあります。"
            )
        return "/".join(item_segments[len(root_segments) :])

    def _upload_with_session(
        self,
        source: Path,
        parent_id: str,
        file_name: str,
        total_size: int,
    ) -> str:
        encoded_name = quote(file_name, safe="")
        endpoint = (
            f"{GRAPH_BASE_URL}/drives/{quote(self._drive_id, safe='')}"
            f"/items/{quote(parent_id, safe='')}:/{encoded_name}:/createUploadSession"
        )
        response = self._request_graph(
            "POST",
            endpoint,
            json_body={
                "item": {
                    "@microsoft.graph.conflictBehavior": "fail",
                    "name": file_name,
                }
            },
        )
        upload_session = self._response_json(
            response,
            "アップロードセッションを開始できませんでした。",
        )
        upload_url = upload_session.get("uploadUrl")
        if not isinstance(upload_url, str) or not upload_url:
            raise FileOperationSharePointError(
                "SharePointからアップロード先が返されませんでした。"
            )

        offset = 0
        created_item: dict[str, Any] | None = None
        try:
            with source.open("rb") as source_file:
                while offset < total_size:
                    chunk = source_file.read(UPLOAD_CHUNK_SIZE)
                    if not chunk:
                        raise FileOperationSharePointError(
                            "アップロード元ファイルを最後まで読み取れませんでした。"
                        )

                    end = offset + len(chunk) - 1
                    try:
                        chunk_response = requests.put(
                            upload_url,
                            data=chunk,
                            headers={
                                "Content-Length": str(len(chunk)),
                                "Content-Range": f"bytes {offset}-{end}/{total_size}",
                            },
                            timeout=REQUEST_TIMEOUT_SECONDS,
                        )
                    except requests.RequestException as exc:
                        raise FileOperationSharePointError(
                            "SharePointへのファイル転送に失敗しました。"
                        ) from exc

                    try:
                        if not chunk_response.ok:
                            raise FileOperationSharePointError(
                                "SharePointへのファイル転送に失敗しました。"
                                f" HTTP {chunk_response.status_code}: "
                                f"{self._graph_error_message(chunk_response)}"
                            )

                        offset += len(chunk)
                        if offset == total_size:
                            created_item = self._response_json(
                                chunk_response,
                                "アップロード完了の応答を読み取れませんでした。",
                            )
                    finally:
                        chunk_response.close()

            if created_item is None:
                raise FileOperationSharePointError(
                    "アップロード完了の応答を取得できませんでした。"
                )
        except OSError as exc:
            raise FileOperationSharePointError(
                "アップロード元ファイルを読み取れませんでした。"
            ) from exc

        if created_item.get("name") != file_name or "file" not in created_item:
            raise FileOperationSharePointError(
                "アップロード結果が指定したファイルと一致しません。"
            )
        created_path = created_item.get("webUrl")
        if not isinstance(created_path, str) or not created_path:
            raise FileOperationSharePointError(
                "アップロード後のパスを取得できませんでした。"
            )
        return created_path

    def _create_empty_file(self, parent_id: str, file_name: str) -> str:
        endpoint = (
            f"{GRAPH_BASE_URL}/drives/{quote(self._drive_id, safe='')}"
            f"/items/{quote(parent_id, safe='')}/children"
        )
        response = self._request_graph(
            "POST",
            endpoint,
            json_body={
                "name": file_name,
                "file": {},
                "@microsoft.graph.conflictBehavior": "fail",
            },
        )
        item = self._response_json(response, "空のファイルを作成できませんでした。")
        if item.get("name") != file_name or "file" not in item:
            raise FileOperationSharePointError(
                "作成結果が指定した空ファイルと一致しません。"
            )
        created_path = item.get("webUrl")
        if not isinstance(created_path, str) or not created_path:
            raise FileOperationSharePointError(
                "作成したファイルのパスを取得できませんでした。"
            )
        return created_path

    def _request_graph(
        self,
        method: str,
        endpoint: str,
        *,
        params: dict[str, str] | None = None,
        json_body: dict[str, Any] | None = None,
        stream: bool = False,
        timeout_seconds: float = REQUEST_TIMEOUT_SECONDS,
    ) -> requests.Response:
        headers = {
            "Authorization": f"Bearer {self._get_access_token()}",
            "Accept": "application/json",
        }
        try:
            response = requests.request(
                method,
                endpoint,
                headers=headers,
                params=params,
                json=json_body,
                timeout=timeout_seconds,
                stream=stream,
            )
        except requests.RequestException as exc:
            raise FileOperationSharePointError(
                "Microsoft Graphへの通信に失敗しました。"
            ) from exc

        if not response.ok:
            error_message = self._graph_error_message(response)
            status_code = response.status_code
            response.close()
            raise FileOperationSharePointError(
                f"Microsoft Graphの要求に失敗しました。HTTP "
                f"{status_code}: {error_message}",
                status_code=status_code,
            )
        return response

    def _get_access_token(self) -> str:
        try:
            accounts = self._msal_app.get_accounts()
            if accounts:
                result = self._msal_app.acquire_token_silent(
                    scopes=GRAPH_SCOPES,
                    account=accounts[0],
                )
                if result and result.get("access_token"):
                    return result["access_token"]

            flow = self._msal_app.initiate_device_flow(scopes=GRAPH_SCOPES)
            user_code = flow.get("user_code")
            verification_uri = flow.get("verification_uri")
            if not user_code or not verification_uri:
                detail = flow.get("error_description") or flow.get("error") or str(flow)
                raise FileOperationSharePointError(
                    f"デバイスコードフローを開始できませんでした: {detail}"
                )

            if not self._show_sign_in_message(verification_uri, user_code):
                raise FileOperationSharePointError("サインインがキャンセルされました。")

            result = self._msal_app.acquire_token_by_device_flow(flow)
        except MsalError as exc:
            raise FileOperationSharePointError(
                f"サインイン処理に失敗しました: {exc}"
            ) from exc
        except requests.RequestException as exc:
            raise FileOperationSharePointError(
                "サインイン中にMicrosoft identity platformとの通信に失敗しました。"
            ) from exc

        access_token = result.get("access_token")
        if access_token:
            return access_token
        error = result.get("error", "unknown_error")
        description = result.get("error_description", "詳細情報はありません。")
        raise FileOperationSharePointError(
            f"アクセストークンを取得できませんでした: {error}: {description}"
        )

    @staticmethod
    def _show_sign_in_message(verification_uri: str, user_code: str) -> bool:
        window = None
        confirmed = False

        def finish(result: bool) -> None:
            nonlocal confirmed
            confirmed = result
            window.quit()

        try:
            window = tk.Tk()
            window.title("SharePointへのサインイン")
            frame = tk.Frame(window, padx=16, pady=16)
            frame.pack(fill="both", expand=True)
            instruction = tk.Label(
                frame,
                text="コードを選択して Ctrl+C でコピーし、ブラウザーでサインインしてください。",
            )
            instruction.pack(anchor="w")
            for label, value in (
                ("サインインURL:", verification_uri),
                ("サインインコード:", user_code),
            ):
                tk.Label(frame, text=label).pack(anchor="w", pady=(10, 0))
                entry = tk.Entry(frame, width=65)
                entry.insert(0, value)
                entry.configure(state="readonly")
                entry.pack(fill="x")
            tk.Label(
                frame,
                text="続行する場合は[OK]、中止する場合は[キャンセル]を選択してください。",
            ).pack(anchor="w", pady=(10, 0))
            buttons = tk.Frame(frame)
            buttons.pack(pady=(10, 0))
            tk.Button(buttons, text="OK", command=lambda: finish(True)).pack(
                side="left", padx=5
            )
            tk.Button(
                buttons, text="キャンセル", command=lambda: finish(False)
            ).pack(side="left", padx=5)
            window.protocol("WM_DELETE_WINDOW", lambda: finish(False))
            try:
                browser_opened = webbrowser.open(verification_uri)
            except (OSError, webbrowser.Error) as exc:
                instruction.configure(
                    text=f"ブラウザーを開けませんでした: {exc}\n"
                    "URLをコピーしてブラウザーで開いてください。"
                )
            else:
                if not browser_opened:
                    instruction.configure(
                        text="ブラウザーを開けませんでした。"
                        "URLをコピーしてブラウザーで開いてください。"
                    )
            window.mainloop()
        except tk.TclError as exc:
            raise FileOperationSharePointError(
                "サインイン画面を表示できませんでした。"
            ) from exc
        finally:
            if window is not None:
                window.destroy()
        return confirmed

    @staticmethod
    def _response_json(response: requests.Response, context: str) -> dict[str, Any]:
        try:
            body: Any = response.json()
        except ValueError as exc:
            raise FileOperationSharePointError(
                f"{context} Graphの応答がJSONではありません。"
            ) from exc
        if not isinstance(body, dict):
            raise FileOperationSharePointError(
                f"{context} Graphの応答形式が不正です。"
            )
        return body

    @staticmethod
    def _graph_error_message(response: requests.Response) -> str:
        try:
            body: Any = response.json()
        except ValueError:
            return response.text[:1000]
        if not isinstance(body, dict):
            return response.text[:1000]
        error = body.get("error", {})
        if not isinstance(error, dict):
            return str(error)
        code = error.get("code", "unknown")
        message = error.get("message", response.text[:1000])
        return f"{code}: {message}"

    def _require_drive(
        self,
        item: dict[str, Any],
        *,
        require_drive_id: bool = False,
    ) -> None:
        parent_reference = item.get("parentReference", {})
        if not isinstance(parent_reference, dict):
            raise FileOperationSharePointError(
                "SharePointアイテムのドライブ情報が不正です。"
            )
        item_drive_id = parent_reference.get("driveId")
        if require_drive_id and not item_drive_id:
            raise FileOperationSharePointError(
                "共有リンクの対象ドライブを特定できませんでした。"
            )
        
        if item_drive_id:
            if self._drive_id is None:
                self._drive_id = item_drive_id
            elif item_drive_id != self._drive_id:
                raise FileOperationSharePointError(
                    "対象アイテムは設定されたドキュメントライブラリの外にあります。"
                )

    @staticmethod
    def _require_file(item: dict[str, Any]) -> None:
        if "file" not in item or "folder" in item:
            raise FileOperationSharePointError(
                "指定されたパスはファイルではありません。"
            )

    @staticmethod
    def _required_item_id(item: dict[str, Any]) -> str:
        item_id = item.get("id")
        if not isinstance(item_id, str) or not item_id:
            raise FileOperationSharePointError(
                "SharePointからアイテムIDを取得できませんでした。"
            )
        return item_id

    @staticmethod
    def _remove_temporary_file(temporary_path: str | None) -> None:
        if temporary_path is None:
            return
        try:
            os.remove(temporary_path)
        except FileNotFoundError:
            return
        except OSError as exc:
            raise FileOperationSharePointError(
                "ダウンロード途中の一時ファイルを削除できませんでした。"
            ) from exc




# 週初めツールでの使い方サンプル
def sample():
    user_input_su_others_url = r""
    download_su_others_path = r"XXXX\\02_SU_Others.xlsx"
    result_su_others_path = r"XXXX\\result_SU_Others.xlsx"

    TENANT_ID = "a54b00a2-87e8-4e92-a76d-de53269b7ddd"
    CLIENT_ID = "18cd7185-631e-4b45-9668-aa8c0ad18469"

    file_operation = FileOperationSharePoint(
        tenant_id=TENANT_ID,
        client_id=CLIENT_ID
    )

    try:
        # 入力が共有リンクでもファイルURLでも動くための事前処理
        su_others_path = file_operation.get_file_url_from_sharing_link(user_input_su_others_url)

        # ファイルのロック
        parent_folder_path = file_operation.get_parent_folder_path(su_others_path)
        file_name = file_operation.get_file_name(su_others_path)

        locked_file_name = "_" + file_name
        locked_file_path = file_operation.rename_file(su_others_path, locked_file_name)
    except FileOperationSharePointError as e:
        # 誰かがSU_Othersを開いていてロックできない場合など
        print(f"ファイルのロックに失敗: {e}")
        return

    try:
        # ファイルのダウンロード
        file_operation.download_file(locked_file_path, download_su_others_path)
    except FileOperationSharePointError as e:
        # 理由不明だがファイルのダウンロードに失敗
        # ロックでリネーム済みなのでユーザーが手動で元に戻す必要がある場合がある
        print(f"ファイルのダウンロードに失敗: {e}")
        return


    # ここから3-1とかの処理

    # その他ツールの処理完了

    try:
        # 結果ファイルのアップロード
        file_operation.upload_file(result_su_others_path, parent_folder_path, file_name)
    except FileOperationSharePointError as e:
        # 理由不明だがファイルのアップロードに失敗
        # ロックでリネーム済みなのでユーザーが手動で元に戻す必要がある場合がある
        print(f"ファイルのアップロードに失敗: {e}")
        return

    try:
        # ファイルのロック解除
        file_operation.delete_file(locked_file_path)
    except FileOperationSharePointError as e:
        # 理由不明だがファイルのロック解除に失敗
        # SU_Others.xlsxと_SU_Others.xlsxが両方存在するため、手動で削除する必要がある場合がある
        print(f"ファイルのロック解除に失敗: {e}")
        return