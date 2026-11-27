# 02_UpdateSchedule 詳細設計

## 1. 目的と対象範囲

`02_UpdateSchedule.exe` は、自動計画の結果を取得し、指定されたSU_Othersブック（`.xlsx` または `.xlsm`）に反映する。利用者はファイルを選択して開始ボタンを押す。

運用フロー3-1〜3-3は既存実装を基本的に変更せずに呼び出す。02は、対象回の選択、保存先別のWorkbookStoreによるリネーム、コピー・比較、既存処理の実行、結果の公開、状態記録、通知を担当する。SU_Others以外の設定・Work・中間成果物・状態は、01と02を実行する同一PCの永続ローカル領域に保存し、他PCとは共有しない。3-1〜3-3の計画処理の再実装や、01の実装は対象外とする。

## 2. 前提と配置

- 01と02は同一PCで実行し、同じ `plan_assist.config.txt` の `work_dir` を使用する。01が `Work/YYYYMMDD` の日付フォルダを作成する。本書の `Work` はこの設定で解決されたフォルダを指す。アプリとWorkは永続化し、必要なWindowsユーザーが読み書きできる場所に配置する。`Program Files` など書き込みが制限される場所には配置しない。
- 01/02はカレントディレクトリやユーザーごとの異なるパスに依存せず、アプリ配置ディレクトリからWorkルートを解決する。01と02のexeを別々のディレクトリへ配置する運用、および別PCで実行する運用は対象外。
- SU_Othersの保存先はローカルファイル、共有ファイルサーバー（SMB/UNC）、またはSharePointとする。ローカルとSMB/UNCは`FileSystemWorkbookStore`、SharePointは`SharePointWorkbookStore`に分け、計画処理・状態遷移・GUIは保存先に依存させない。
- 02は共通の `plan_assist.config.txt` を `receive_result` に渡し、01と同じ接続モード・URLを使用する。3-1を単体実行する場合の設定は変更しない。実行単位の `run_info` と出力先は関数引数で渡す。
- `Work/YYYYMMDD` は01の自動計画開始日を表す。有効な日付名かつ `steps/run_info.json` があるフォルダのうち日付が最新のものを選択する。対象がない場合は終了する。02の実行日から算出しない。
- 同日の複数回の自動計画は01側で1つの対象回に整理されていることを前提とする。02はフォルダ内で複数回を選択しない。
- 01は `input/01_SU_Others.{xlsx,xlsm}` と、`steps/` に `FlexibleWork.yaml`、`run_info.json`、送信した `03_EnvConfig.yaml` を用意する。`run_info.json` の `envFile` は同じ実行回の `steps/03_EnvConfig.yaml` を指す。
- 01が選択したExcelと02の選択ファイルが異なるケースは想定せず、両者のパス一致検査は行わない。
- 01は、反映状態が `applied` の回の `steps/FlexibleSchedule.yaml` と `steps/04_EnvConfig.yaml` を次回の前週結果として利用する。

## 3. 入出力

### 3.1 利用者入力

| 項目 | 内容 |
|---|---|
| SU_Others | ローカル／SMBはファイル選択または入力する絶対パス。SharePointは入力・貼り付けする共有リンク |
| 開始 | ボタン押下で処理開始 |

対象の拡張子は `.xlsx` または `.xlsm` とし、大文字・小文字は区別しない。ブックの内容は従来どおりのSU_Others形式とするが、ファイル名はロック用の予約名 `_SU_Others.xlsx`（大文字・小文字を区別しない）を除き任意とする。ローカルパスとSMB/UNCパスは絶対パスを指定する。選択した保存先に対応するWorkbookStoreを使い、リネーム先は形式にかかわらず同じ保存先内の `_SU_Others.xlsx` とする。既存のリネーム先を上書きしない。この名前は簡易ロック専用であり、ブックの形式は変換しない。SharePointは共有リンクを入力し、`FileOperationSharePoint.get_file_url_from_sharing_link()` で対象URLを解決する。Web URLや共有リンクをローカルファイルパスとして扱わない。

### 3.2 対象フォルダ内のファイル

| ファイル | 入出力 | 用途 |
|---|---|---|
| `Work/YYYYMMDD/input/01_SU_Others.{xlsx,xlsm}` | ローカル入力 | 自動計画開始時点のSU_Othersコピー。結果反映開始時点のコピーとのバイナリ比較の基準 |
| `Work/YYYYMMDD/steps/FlexibleWork.yaml` | ローカル入力 | 前回のFlexible FI/FOの除去情報 |
| `Work/YYYYMMDD/steps/run_info.json` | ローカル入力 | 3-1の実行ID、接続先、EnvConfigパス |
| `Work/YYYYMMDD/steps/03_EnvConfig.yaml` | ローカル入力 | 01が送信したEnvConfig。`run_info.json` の `envFile` が指すファイル |
| `Work/YYYYMMDD/steps/02_SU_Others.{xlsx,xlsm}` | ローカル出力 | 結果反映開始時点のSU_Othersコピー |
| `Work/YYYYMMDD/steps/04_Schedule.yaml`, `steps/04_EnvConfig.yaml` | ローカル出力 | 3-1が取得する結果とEnvConfig |
| `Work/YYYYMMDD/steps/FlexibleSchedule.yaml` | ローカル出力 | 3-2が抽出した情報。反映済みの場合のみ次週の01が利用 |
| `Work/YYYYMMDD/steps/02_SU_Others_rewritten.{xlsx,xlsm}` | ローカル出力 | 3-3の反映結果 |
| `Work/YYYYMMDD/steps/update_status.json` | ローカル出力 | 02の反映状態。別PCとは共有しない |
| `Work/YYYYMMDD/history/` | ローカル出力 | 制御された失敗後、再実行前に退避する途中成果物 |

`01_SU_Others.{xlsx,xlsm}` は01_StartScheduleが自動計画を開始する時点で作成し、`02_SU_Others.{xlsx,xlsm}` は02_UpdateScheduleが結果反映を開始する時点で作成する。`{xlsx,xlsm}` は元ブックの形式に応じた拡張子を表す。作業用コピーと3-3の反映結果は元ブックと同じ形式とする。両コピーのバイナリ差分は、自動計画実行中にSU_Othersが変更された可能性を検出するために使う。

利用者向け資料の `SU_Others_01.xlsx` / `SU_Others_02.xlsx` とは名前が異なる。本設計では `01_SU_Others` / `02_SU_Others` に元ブックと同じ形式の拡張子を付け、利用者向け資料の整合は別途必要とする。

### 3.3 サーバー上のファイル

共有元ファイルの取得・ロック・結果公開・ロック解除はWorkbookStoreが担当する。ローカル／SMB/UNCはファイルシステム操作で、SharePointは `common.file_operation_sharepoint.FileOperationSharePoint` のAPIで行う。SharePointでは既存アイテムを同一ライブラリ内の `_SU_Others.xlsx` にリネームして保持し、結果を選択時の元ファイル名で新規アップロードする。公開後に旧アイテムを削除するため、既存アイテムの内容更新やアイテムID・メタデータの継承は行わない。正常反映時はロック名の旧ファイルを残さない。反映前コピーと生成物はローカルWorkに保持する。

## 4. 画面

- GUIはPython標準ライブラリのTkinterと `ttk` で実装する。GUI専用の追加依存は導入しない。
- SU_Others入力欄、ファイル選択ボタン、開始ボタン、処理状況、終了メッセージを表示する。入力欄は直接編集可能とし、ローカルパス、UNCパス、SharePoint共有リンクを入力・貼り付けできるようにする。ファイル選択ダイアログはローカル／SMBのブック選択に使い、SharePoint共有リンクは入力欄に指定する。
- ファイル選択ダイアログは `.xlsx` と `.xlsm` を対象とする。パス欄に有効なローカルファイルのパスがある場合は、その親フォルダを初期位置とする。UNCパスはダイアログを開く前に存在確認せず、未入力や無効なパスの場合と同様にOS標準の初期位置を使う。カレントディレクトリを前提としない。
- 開始時にGUIでローカル／SMBパスの絶対性と拡張子を確認する。SharePoint共有リンクはworker threadから呼び出す `run_update()` で解決・認証・拡張子確認する。ファイルの存在・読み取り可否もworker threadで検証し、不正な入力ではブックをロックせず、画面上に理由を表示する。
- 処理中は開始の再押下と通常のウィンドウ終了を無効にする。終了操作が要求された場合は「処理中は終了できません」と通知する。OSの強制終了や停電は防げない。
- 時間のかかる処理はGUIスレッドと別のworker threadで順次実行する。表示する進捗は「結果取得中」「FlexibleSchedule生成中」「Excel反映中」など処理段階のみとし、進捗率や細かなログは表示しない。workerからGUIを直接操作せず、`queue.Queue` で段階・認証要求・最終結果を通知し、Tkinterの `after()` を使ってGUIスレッドで画面を更新する。`update_schedule` 内の薄いサブクラスで `_show_sign_in_message()` をoverrideし、認証要求をQueueと応答用EventでGUIスレッドへ渡す。GUIスレッドは認証用 `Toplevel` を表示し、OK／キャンセル／ウィンドウ終了時に必ずEventを完了させる。`common/file_operation_sharepoint.py` とその `_get_access_token()`、Graph API処理は変更しない。
- 結果は、正常終了、警告付き正常終了、未完了、エラーを区別し、利用者向けの要約を画面に表示する。例外の詳細は復旧・問い合わせに必要な範囲に留める。利用者にコンソール操作は求めない。
- 比較不一致や3-3の `WARN` は完了画面に表示する。3-3の `CONFLICT` は例外として停止し、元ブックを復元して結果を公開しない。比較不一致では01/02コピーの場所を案内する。個別レポートは出力しない。
- 反映済み状態を検出した場合は、ロックを解除して「この計画回は反映済み。更新なし」と情報表示して正常終了する。
- WorkbookStoreでリネームできない場合は、対象ファイルが開かれているなどの可能性を表示して終了する。SMBでは全利用者にファイルを閉じてもらうよう案内する。原因を特定ユーザーの使用中と断定しない。

## 5. 処理手順

1. 利用者が指定したローカルパス、SMB/UNCパス、またはSharePoint共有リンクを検証する。対象のローカルWork/YYYYMMDDと必須入力を確認し、選択後は別の日付へ切り替えない。
2. WorkbookStoreを選び、対象ファイルを同じ保存先内の `_SU_Others.xlsx` にリネームしてロックを取得する。ローカルとSMB/UNCはFileSystemWorkbookStoreでファイルシステムのリネームを行う。SharePointは共有リンクを対象URLへ解決し、ファイル名・親フォルダーを取得してからAPIでリネームする。既存のロック名を上書きせず、失敗時は以降を実行しない。
3. ロック取得後にローカルWorkの対象回状態を再確認する。`applied` は再反映せずWorkbookStoreでロックを解除し、「反映済み、更新なし」と正常終了する。`in_progress` や状態不明は自動復旧・再実行せず、対象パスを表示して安全にロックを解放できる場合のみ解除する。
4. 制御された失敗後の再実行の場合、ローカル対象回の途中成果物を一意な `history` 配下へ退避する。01の入力、状態ファイル、他回のファイルは退避・上書きしない。退避失敗時は中止してWorkbookStoreに復旧を依頼する。
5. 反映状態を `in_progress` に更新する。WorkbookStoreでロック中の元Workbookを元の拡張子に対応する `02_SU_Others.{xlsx,xlsm}` に取得し、同じ形式の `01_SU_Others.{xlsx,xlsm}` と全バイト比較する。不一致は自動計画開始後に変更された可能性として警告し、処理は続行する。取得・比較エラーは中止する。
6. 3-1を1回呼び出し、完了済みなら確認なしで結果を取得する。未完了ならWorkbookStoreでロック解除し、後から再実行できる状態で終了する。
7. 3-2で `FlexibleSchedule.yaml` を生成し、3-3で入力コピーと同じ拡張子の `02_SU_Others_rewritten.{xlsx,xlsm}` を `steps/` に生成する。レポートは生成しない。既存処理のエラーは中止する。
8. WorkbookStoreにより反映結果を選択時の元のファイル名で公開する。既存の元名ファイルを上書きする操作は禁止する。ファイルシステム実装は同じ保存先に一時ファイルを書いてから配置する。SharePoint実装は `upload_file()` で同じ親フォルダーに元名の新しいアイテムを作成し、既存アイテムを上書きしない。ロック中の既存アイテムの内容は更新しない。
9. 公開成功後、ローカル状態を `applied` に記録し、WorkbookStoreでロック名の旧ファイルを削除してロックを解除する。SharePointでは新規公開したアイテムを残し、リネームしていた旧アイテムを削除する。解除失敗は反映済みの警告とし、再反映しない。

手順8の公開、手順9のローカル状態記録・ロック解除は一括確定できない。公開後に状態記録が失敗した場合、またはその間に強制終了した場合は状態不明として自動再実行せず、保存先上の元名ファイル／ロック名ファイルとローカル状態を確認する。`FileOperationSharePoint` はアップロードセッションの状態照会・中断を公開しないため、アップロード失敗・成否不明時はAPIへ自動再試行せず、元名／ロック名のアイテムとローカル状態を利用者が手動で確認する。元名や他者のファイルを上書きしない。

## 6. 既存処理の関数API

### 6.0 WorkbookStore

02の共通オーケストレーションはローカルWorkとローカル一時ファイルだけを扱い、SU_Othersの保存先操作は `update_schedule.workbook_store` のインターフェースに委譲する。`FileSystemWorkbookTarget` はローカルまたはSMB/UNCパス、`SharePointWorkbookTarget` は利用者が入力したSharePoint共有リンクを表す。SharePointの `tenant_id` と `client_id` は `common.plan_settings` の既知設定キーとして `config/plan_assist.config.txt` から渡し、client secretをexeに含めない。各Storeは保存先別の不透明な単回使用ハンドルを発行する。APIを使う側はハンドルの内部を参照せず、取得した同一のStoreに渡す。ファイルシステムの既存 `LockHandle` とSharePoint用privateハンドルは別型として、別Storeへの誤用を拒否する。

```python
class WorkbookStore(Protocol):
	def workbook_name(self, target: WorkbookTarget) -> str: ...
	def acquire(self, target: WorkbookTarget) -> WorkbookLockHandle: ...
	def download_locked(self, lock: WorkbookLockHandle, destination: Path) -> None: ...
	def publish(self, lock: WorkbookLockHandle, result: Path) -> None: ...
	def release(self, lock: WorkbookLockHandle, *, published: bool) -> None: ...
```

- `workbook_name` はロック前に対象のファイル名を返す読み取り専用操作とする。ファイルシステムは対象Pathのnameを返し、SharePointは共有リンクを解決して対象名を返す。解決済みURLは同じStore内で再利用する。02はこの名前から拡張子を判定し、対応するWorkの入力を検証してから `acquire` を呼ぶ。
- `acquire` は同じ保存先内の `_SU_Others.xlsx` へのリネームでロックを取得し、ロック名が既にある場合やリネームできない場合は失敗する。既存ファイルを上書きしない。
- `download_locked` はロック中の元WorkbookをローカルWorkへコピーし、保存先のロック状態は変更しない。
- `publish` はロック中Workbookの元名へ結果を公開する。元名が再出現している場合は上書きせず失敗する。ファイルシステムは同じ保存先に結果を配置し、SharePointは元名の新しいアイテムを作成する。
- `release(published=False)` は元名が空いている場合のみロック名から復元する。`release(published=True)` は公開済みの旧ロック名を削除する。SharePointではロック中の旧アイテムを削除し、新規公開したアイテムを残す。失敗時は残存ロックを維持して例外を通知する。
- `FileSystemWorkbookStore`: ローカルパスとSMB/UNCパスを同じファイルシステム操作で扱う。元ファイルを `_SU_Others.xlsx` にリネームし、ロック中コピーをローカルWorkへ取得、結果を元名に公開後、旧ファイルを削除して解除する。SMB/UNCのネットワーク切断や共有権限エラーは操作失敗として扱う。
- ファイルシステム上のリネームは、同じ手順に従う02同士の競合を防ぐための簡易ロックであり、OSのファイルロックではない。コピー・公開・解除時にロック名のファイルがなければ失敗する。ファイル識別情報は記録・照合せず、同名ファイルへの差し替え検知は対象外とする。処理中は手動操作や別ツールでロック名のファイルを変更しない。
- `SharePointWorkbookStore`: `common.file_operation_sharepoint.FileOperationSharePoint` を利用し、Entra IDの利用者委任認証とMicrosoft Graph APIで操作する。`workbook_name()` で共有リンクを `get_file_url_from_sharing_link()` により解決し、`get_file_name()` で元名を取得する。同じStore内で解決済みURLを再利用し、`get_parent_folder_path()` で親フォルダーを取得する。`rename_file()` で対象アイテムを `_SU_Others.xlsx` に変更し、`download_file()` でローカルWorkへ取得する。公開は `upload_file()` で元名の新しいアイテムを作成し、`delete_file()` で旧ロックアイテムを削除する。元アイテムの内容更新やアイテムIDの継承は行わない。公開前に通常の処理例外となった場合は `release(published=False)` から `rename_file()` で元名への復元を試みる。`tenant_id` と `client_id` は設定から渡し、client secretは含めない。
- リネームが失敗した場合はロック取得失敗とし、結果を強制上書きしない。`rename_file()` がPATCH後の確認でtimeoutして、変更済みか否かを確定できない場合は `acquire` の取得失敗とし、ロック状態不明として自動復元しない。対象の元名・ロック名・共有リンクを通知する。ロック取得済みで公開前の通常処理が失敗した場合のみ、元名が空いていることを確認して `release(published=False)` で戻す。公開後の状態不明時は自動で復旧・再公開しない。SharePointでは元名の新規アイテム公開後に旧ロックアイテムの削除が失敗すると、元名とロック名の両方が残る。
- SharePointのリネームは正式なチェックアウト機構ではない。同期クライアントや既存編集セッションとの排他保証は、対象テナント・ライブラリで確認する。強制チェックアウト／破棄は行わない。

02は既存処理を別CLIプロセスとして起動せず、関数APIをworker threadから順次呼び出す。計画ロジック、セル処理、YAML抽出規則は変更しない。3-1・3-3は既存スクリプト内に関数APIを追加し、CLI `main(argv)` と02の双方から同じ関数を呼び出す。3-2は既存の `create_flexible_schedule` を利用する。

処理本体は既存スクリプトに残し、`weekly_tools` への移動やロジックの複製は行わない。GUIはアプリの配置ディレクトリを基準に各スクリプトのパスを解決し、`importlib.util.spec_from_file_location` を使って読み込む。モジュールは `exec_module` の前に `sys.modules` へ登録する。作業ディレクトリには依存せず、ファイルが存在しない・読み込みに失敗した場合はエラーとして通知する。スクリプトのトップレベルではCLIを起動せず、CLI起動は `if __name__ == "__main__"` の場合に限る。以下の関数API契約を公開仕様とする。

### 6.1 3-1 結果取得

```python
receive_result(
	run_info_path: Path,
	output_dir: Path,
	config_path: Path,
	*,
	watch_seconds: int = 0,
) -> ReceiveResult

@dataclass(frozen=True)
class ReceiveResult:
	status: Literal["completed", "not_finished"]
```

- 既存設定ファイルの接続モード・URLを使い、設定内容は変更しない。`run_info_path` と `output_dir` は引数で明示し、設定ファイルにある別の実行情報や出力先を使わない。自動計画中のポーリングや対話ダウンロード確認は行わない。
- `output_dir` には `04_Schedule.yaml` と `04_EnvConfig.yaml` を出力する。`ReceiveResult` は状態だけを返し、02は既知の出力先で両ファイルの存在を確認する。未完了は通常結果として返し、02はExcelを復元して終了する。
- 設定不正、接続失敗、Timefoldの失敗・キャンセルは例外として02へ通知する。関数APIではCLI終了コードを公開契約にしない。
- `run_info.json` の `envFile` は同一PCのローカルWork内のEnvConfigを指す。EnvConfigを取得できない場合は成功扱いにしない。

### 6.2 3-2 Flexible情報抜粋

```python
create_flexible_schedule(
	schedule_path: str | Path,
	env_path: str | Path,
	output_path: str | Path,
) -> None
```

既存スクリプトの `create_flexible_schedule` をそのまま利用する。内容の抽出・補完規則、入力検証、出力形式は[3-2詳細設計](../3-2extract_flexible_schedule/詳細設計.md)に従う。例外は02へ伝播させ、部分成功として扱わない。

### 6.3 3-3 Excel反映

```python
write_su_others(
	schedule_path: Path,
	env_path: Path,
	su_others_path: Path,
	output_dir: Path,
	flexible_work_path: Path | None,
	*,
	sheet: str = "予定表_2026",
	from_date: date | None = None,
	write_report: bool = False,
	previous_flexible_path: Path | None = None,
	previous_env_path: Path | None = None,
) -> WriteResult

@dataclass(frozen=True)
class WriteResult:
	output_path: Path
	report_path: Path | None
	warning_count: int
	conflict_count: int
```

- 対象シートは既定の `予定表_2026`、反映開始日は指定がなければ既存どおり実行日。既定日より前のセルは3-3が変更しない。
- 02は01の状態記録にある前回のFlexibleScheduleとEnvConfigを渡す。状態記録がない場合は従来の前回選択規則を使う。初回は両方を `None` とする。3-3は前回FlexibleScheduleの作業者・日付と今回FlexibleWorkで削除対象を限定し、同じ製番コードの他セルは変更しない。前回入力が未指定または空なら削除対象なしとし、空セルへの新規配置だけを行う。全体のScheduleは3-3に渡さない。
- `WriteResult` は出力Workbookのパス、レポートパス（未出力時は `None`）、警告数、競合数を返す。02は `write_report=False` で呼び出す。3-3を単体実行する場合は従来どおり `--report` を指定できる。
- 配置先が占有済み、日付・作業者が配置不能、または新規配置が重複する場合は3-3が `WriteConflictError` を送出し、出力ブックを生成しない。02は元ブックを復元する。`WARN` だけなら処理を続行する。
- 致命的な入力・出力・検証エラーは例外として通知する。3-3は入力Workbookを上書きせず、新しい結果を出力する。
- 差分を強調したExcelの生成は対象外。比較用コピーと既存の差分確認マクロを使う。
- 既存3-3の説明は `.xlsm` 前提だが、出力名は入力拡張子を引き継ぐ。実際の `.xlsx` 動作は未検証のため結合テストで確認する。拡張子変更を形式変換として扱わない。

### 6.4 CLI互換性・起動形態

- 既存3-1・3-3のCLI引数は維持する。CLI `main(argv)` と公開関数APIは共通の処理関数を利用し、CLI側は従来の対話・終了コード・メッセージを維持する。3-2のCLIも既存関数を利用する。
- CLIは `main(argv)` で従来の終了コード・メッセージを維持する。3-3のCLIは `write_report` に `--report` の指定を渡し、従来の既定動作を変えない。
- 当面の実装・検証対象はexe化前とし、Python環境からGUIを起動できる状態を完成ゴールとする。PyInstallerの導入、exeの生成、配布物の作成はこの段階では行わない。Python実行時はGUI起動モジュールの配置ディレクトリをアプリ配置ディレクトリとする。
- GUIはアプリ配置ディレクトリから3-1〜3-3の既存スクリプトを読み込む。Python実行時はリポジトリ直下の `config/plan_assist.config.txt` を読み、`work_dir` の相対パスはその設定ファイルを基準に解決する。exe配布時も、01/02が共通の `config/plan_assist.config.txt` を読む。
- 将来exe化する場合はPyInstallerのonedir形式を採用する。01/02のexeと共通の `config/plan_assist.config.txt` を配置し、既存スクリプトと依存ライブラリを読み込み可能な配置で同梱する。配布対象環境で起動と共有フォルダ操作を検証する。
- `work_dir` は実行成果物・状態を保持する永続データであり、アプリ更新時に削除・初期化しない。配布物更新前にWorkをバックアップし、必要なWindowsユーザーの読み書き権限を設定する。
- GUIからの実行はworker threadで行い、3-1、3-2、3-3の各段階の開始時に段階表示を更新する。関数APIに進捗callbackや進捗率は設けない。worker完了後の状態・警告だけをキュー経由でGUIスレッドへ通知し、失敗例外は既存の復旧手順へ渡す。

## 7. 反映状態と再実行

状態は対象回の `steps/update_status.json` に保存する。状態更新は同一ディレクトリの一時ファイルへの完全書き込み後に置換し、不完全なJSONを正常状態として扱わない。状態ファイルが読めない・未知の値の場合は状態不明として終了する。

| 状態 | 意味 | 02再実行時 |
|---|---|---|
| `ready` | 反映前、またはExcelを正常に復元した制御された失敗 | 途中成果物を退避し再実行可能 |
| `in_progress` | ロック後の処理開始から公開後の状態確定前 | 自動再実行せず、状態とファイルを確認 |
| `applied` | 結果公開後に反映済みと記録 | 再反映しない。01はこの回を参照可能 |

状態ファイルがない初回は、対象回に02の途中成果物がないことを確認し `ready` 相当とする。状態なしで途中成果物が存在する場合は状態不明として終了する。01が `ready` ファイルをあらかじめ作成してもよい。

通常の未完了・例外で公開前に停止した場合は、ロックしたExcelを元の名前に戻せたときのみ `ready` にする。復元または状態更新に失敗した場合は自動再実行できる状態とせず、残存ファイルを通知する。`FlexibleSchedule.yaml` が生成済みでも `applied` でなければ01は使用しない。

## 8. エラーと復旧

| 条件 | 動作 |
|---|---|
| 必須ファイル不足、入力不正、認証／権限不足 | 理由と対象を表示して終了。取得済みロックは安全に解除できる場合のみ解除 |
| WorkbookStoreのリネーム失敗、ロック名が既存 | 以降を実行しない。既存ファイルを変更しない |
| SharePointリネーム後の確認timeout | ロック取得成否不明として自動復元・再試行をせず、元名・ロック名・共有リンクを通知 |
| 別の02が先にロック取得 | 後からロック取得できない処理は終了。追加の独自ロックファイルは作らない |
| 3-1未完了 | WorkbookStoreでロック解除し、後から再実行できることを通知 |
| 取得・抽出・Excel生成・公開失敗 | 公開前ならWorkbookStoreによる復元を試行し、エラーと対象を表示 |
| 結果公開前に元名のファイルが存在 | 上書きせず反映中止。旧ファイルと新規ファイルを残し場所を通知 |
| ロック解除／復元失敗 | ロック名のファイルが残っていれば維持し、バックエンド別の対象識別子と場所を表示 |
| 公開後の状態記録失敗・強制終了 | 反映の成否が不明として自動再実行せず、SharePoint/SMB上の元名ファイル・ロック名ファイルとローカル状態を確認 |
| 公開・状態記録後の旧ファイル削除失敗 | `applied` を維持し、残存するロック名ファイル／アイテムを通知 |

Workと状態は設定されたローカルフォルダに保存する。別PCでの運用は設計対象外とする。同一PCで別のWindowsアカウントを使う場合は、アプリ配置ディレクトリとWorkへの読み書きACLを設定する。SharePointのリネームは正式なチェックアウトではないため、利用者の既存編集セッションや同期遅延を排他できるとは限らない。両バックエンドのリネーム・上書き防止・復元動作を実環境で検証する。

## 9. テスト観点

1. **正常系**: `.xlsx` と `.xlsm` それぞれで、FileSystemWorkbookStoreはローカルとSMB/UNCで、SharePointWorkbookStoreは `FileOperationSharePoint` を通じて取得・リネーム・公開・解除が順に完了する。SharePointは公開結果が元名の新規アイテムとなり、成功時に旧ロックアイテムが削除されることを確認する。作業用コピーと反映結果の拡張子が元ブックと一致し、`.xlsm` のVBAが保持されることを確認する。同一PCのローカルWorkを01/02で共有し、`applied` 記録後、同日2回目は「反映済み、更新なし」で終了する。
2. **ロックと競合**: ローカル、SMB/UNC、SharePointでリネーム不可、既存ロック名、結果公開前に元名のファイル／アイテムが存在する場合、既存物を上書きせず終了する。ファイルシステムではロック名のファイルがなければコピー・公開・解除が失敗し、SharePointではAPIのリネーム・ダウンロード・アップロード・削除失敗を確認する。コピー不一致や3-3の `WARN` は通知して処理を続ける。3-3の `CONFLICT` は停止し、元ブックを復元して公開しない。
3. **処理失敗**: 3-1未完了・取得失敗・成果物不足、3-2/3-3失敗、コピー・書き込み失敗時に結果を公開せず、可能なら元ファイル名へ復元する。復元不能時は残存パスを通知する。
4. **再実行と異常終了**: `applied` は再反映せず、制御された失敗は成果物を退避して再実行できる。強制終了や公開後の状態不明では自動再実行せず、未反映の `FlexibleSchedule.yaml` を01が引き継がない。
5. **既存処理との結合**: 3-1〜3-3の関数APIで既存CLIと同じ計画結果が得られ、実 `.xlsx` の反映・FI/FO・書式保持を確認する。02の実行では個別レポートを出力せず、3-3単体の `--report` は引き続き動作する。
6. **GUI・配置**: worker thread実行中も画面が応答し、処理中の終了抑止、警告・エラー表示、およびPython環境での起動を確認する。カレントディレクトリに関係なく、01/02が同じ `work_dir` を使うこと、必要なWindowsユーザーが読み書きできること、アプリ更新時にWorkが保持されることを確認する。SharePointでは認証・サイト権限と同期クライアントによる競合も検証する。

## 10. 関連資料

- `doc/人員最適化ツール想定運用.txt` の3章
- `3-1receive_from_timefold/README.md`
- `3-2extract_flexible_schedule/詳細設計.md`
- `3-3write_su_others/README.md`