"""SharePointの共有リンク解決と読み取りを確認する。"""

from __future__ import annotations

import sys
import tempfile
from pathlib import Path

from common import plan_settings
from common.file_operation_sharepoint import (
    FileOperationSharePoint,
    FileOperationSharePointError,
)
from update_schedule.paths import ApplicationPaths
from update_schedule.workbook_store import SharePointWorkbookTarget


def check_sharepoint_read_access(
    operation: FileOperationSharePoint,
    sharing_link: str,
) -> tuple[str, int]:
    """共有リンクを解決して一時ダウンロードし、ファイル名とサイズを返す。"""
    target = SharePointWorkbookTarget(sharing_link)
    file_url = operation.get_file_url_from_sharing_link(target.sharing_link)
    file_name = operation.get_file_name(file_url)

    with tempfile.TemporaryDirectory(prefix="sharepoint-check-") as directory:
        downloaded_file = Path(directory) / "downloaded_file"
        operation.download_file(file_url, downloaded_file)
        downloaded_size = downloaded_file.stat().st_size

    return file_name, downloaded_size


def main() -> int:
    """設定を読み込み、対話形式でSharePoint読み取り確認を実行する。"""
    try:
        paths = ApplicationPaths.current()
        settings = plan_settings.load(path=paths.config_file)
    except (plan_settings.SettingsError, OSError) as error:
        print(f"設定を読み込めませんでした: {error}", file=sys.stderr)
        return 1

    tenant_id = settings.text("tenant_id")
    client_id = settings.text("client_id")
    if tenant_id is None or client_id is None:
        print(
            "設定ファイルに tenant_id と client_id を設定してください。",
            file=sys.stderr,
        )
        return 1

    sharing_link = input("検証用ファイルのSharePoint共有リンクを入力してください: ")
    operation = FileOperationSharePoint(tenant_id, client_id)
    try:
        file_name, downloaded_size = check_sharepoint_read_access(
            operation,
            sharing_link,
        )
    except (FileOperationSharePointError, OSError, ValueError) as error:
        print(f"SharePoint読み取り確認に失敗しました: {error}", file=sys.stderr)
        return 1

    print("SharePoint読み取り確認に成功しました。")
    print(f"ファイル名: {file_name}")
    print(f"ダウンロードサイズ: {downloaded_size} bytes")
    print("一時ダウンロードファイルは削除済みです。")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
