"""02_UpdateSchedule用のSharePoint認証UI境界を定義する。"""

from __future__ import annotations

from collections.abc import Callable

from common.file_operation_sharepoint import FileOperationSharePoint


SignInPrompt = Callable[[str, str], bool]


class UpdateScheduleFileOperationSharePoint(FileOperationSharePoint):
    """既存Graph操作を維持し、サインイン表示だけGUIへ委譲する。"""

    def __init__(
        self,
        tenant_id: str,
        client_id: str,
        sign_in_prompt: SignInPrompt,
    ) -> None:
        """委任認証画面のcallbackを保持してHelperを初期化する。"""
        self._sign_in_prompt = sign_in_prompt
        super().__init__(tenant_id, client_id)

    def _show_sign_in_message(
        self,
        verification_uri: str,
        user_code: str,
    ) -> bool:
        """GUI側の認証promptへデバイスコード情報を渡す。"""
        return self._sign_in_prompt(verification_uri, user_code)
