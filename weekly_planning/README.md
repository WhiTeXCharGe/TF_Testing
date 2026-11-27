# 週次計画ツール

週次計画の準備、ソルバーへの送信、計画結果の反映を行うツールです。

## 処理の流れ

| ステップ | ツール | 主な入力 | 主な出力 |
|---|---|---|---|
| 2-2・2-3 | [prepare_schedule](2-2-3prepare_schedule/)：計画を準備 | 前週の `FlexibleSchedule.yaml` と `04_EnvConfig.yaml`、SU_Others、製番リスト、スキルファイル | `FlexibleWork.yaml`、`01_EnvConfig.yaml`、`01_Schedule.yaml` |
| 2-4 | [apply_seiban_list](2-4apply_seiban_list/)：製番リストを反映 | `01_Schedule.yaml`、`01_EnvConfig.yaml`、製番リスト | `02_Schedule.yaml`、`02_EnvConfig.yaml` |
| 2-5 | [set_plan_period](2-5set_plan_period/)：計画期間を設定 | `02_Schedule.yaml`、`02_EnvConfig.yaml` | `03_Schedule.yaml`、`03_EnvConfig.yaml` |
| 2-6 | [submit_to_timefold](2-6submit_to_timefold/)：ソルバーへ送信 | `03_Schedule.yaml`、`03_EnvConfig.yaml` | `run_info.json`、`SolverParams.yaml` |
| 3-1 | [receive_from_timefold](3-1receive_from_timefold/)：計画結果を取得 | `run_info.json` | `04_Schedule.yaml`、`04_EnvConfig.yaml` |
| 3-2 | [extract_flexible_schedule](3-2extract_flexible_schedule/)：柔軟な計画を抽出 | `04_Schedule.yaml`、`04_EnvConfig.yaml` | `FlexibleSchedule.yaml` |
| 3-3 | [write_su_others](3-3write_su_others/)：計画結果を反映 | `FlexibleSchedule.yaml`、`04_EnvConfig.yaml`、SU_Othersのコピーなど | 更新済みのSU_Others |

`01_StartSchedule` の画面では、ステップ2-2・2-3から2-6までを実行し、自動的にソルバーへ送信します。
コマンドラインからの実行では、`--no-submit` または設定ファイルの `submit = false` により2-5までで停止できます。
`02_UpdateSchedule` はソルバーの完了後にステップ3-1から3-3を実行します。

## ファイルと設定

- 共通設定は `config/plan_assist.config.txt` に記載します。
- 初回は `Master` に `EnvConfig_start.yaml` と、名前に `skill` を含むスキルファイルを配置します。
- 週ごとの入力、処理結果、ログは `Work/<日付>/input`、`steps`、`logs` に保存されます。
- `01_StartSchedule` の実行ログは週フォルダの `logs/start_log.txt` にまとめられます。2-2・2-3と2-4の個別レポートは出力しません。
- 同じ週を再実行すると、以前の `steps` と `logs` は `steps_old_<日時>` と `logs_old_<日時>` に退避されます。

## Windows実行ファイルのビルド

Python 3.12、PyInstaller、`pyyaml`、`openpyxl`、`pandas`、`msal`、`requests` を用意し、リポジトリのルートで実行します。

```powershell
python build_exe.py --out <出力先フォルダ>
```

`<出力先フォルダ>\plan_assist` に両方の実行ファイルが作成されます。実行ファイル、`config`、`Master`、`Work` がトップレベルにあり、プログラムのコードと実行時ファイルは `weekly_planning` にまとめられます。配布・移動する際はフォルダ全体を一緒に扱ってください。

**注意:** 再ビルドすると出力先の `plan_assist` フォルダは削除され、作り直されます。設定や作業ファイルがある場合は事前にバックアップしてください。`--only 01` または `--only 02` で片方だけをビルドできます。

## Pythonでの実行

各ツールはPythonから個別に実行できます。共通コードや他のツールを参照するため、各フォルダを同じ階層に置いてください。
