# GanttChart エディター 調査まとめ

> 作成日: 2026-06-17  
> 対象: `web/GanttChart/` 配下の2025年度・2026年度両バージョン

---

## 目次

0. [ガントチャートエディターの概要・機能](#0-概要と機能)
1. [技術スタック・使用ツール](#1-技術スタック)
2. [2025年度版 vs 2026年度版の違い](#2-バージョン間の違い)
3. [要件に対して未実装の機能](#3-未実装機能)
4. [アプリの起動方法](#4-起動方法)
5. [コードの問題点・バグ](#5-コードの問題点)
6. [連携機能の実装方針（Webapp ↔ GanttChart）](#6-連携実装方針)
7. [その他の補足事項](#7-その他補足)

---

## 0. 概要と機能

### 何をするツールか

`GanttChart エディター`は、Timefold ソルバーが出力したスケジュール結果（Schedule.yaml）を手動で調整するためのデスクトップアプリです。スケジューラー Webapp でソルバーを回した後、最適化結果を人間が確認・微修正し、再度ソルバーに投入するための中間編集ツールとして位置づけられています。

### 現状の実装済み機能（2026年度版）

| 機能カテゴリ | 実装済み内容 |
|---|---|
| **ファイル操作** | EnvConfig.yaml + Schedule.yaml の読み込み（2回ダイアログ）、上書き保存、名前を付けて保存 |
| **ガントチャート表示** | ワーカービュー（行=作業者、バー=担当タスク）とデバイスビュー（行=装置フェーズ、バー=担当作業）の2モード |
| **バー操作** | バー全体の水平ドラッグ（開始日・終了日を同じ日数だけ移動） |
| **バー色分け** | 装置（workflow_task）ごとに20色パレットで色分け |
| **休日表示** | 祝日列を薄赤（`#ffe0e0`）で背景色付け |
| **デバイスビュー** | フェーズの折りたたみ/展開（`expandedPhaseTaskIds`） |
| **サイドパネル** | 選択バーの詳細表示、`plan_flexibility`（Flexible/Reluctant/Fixed）のドロップダウン編集、バー削除 |
| **タスク追加ダイアログ** | 作業者・操作タスク・開始日・終了日・1日あたり時間数・plan_flexibility を入力して新規アサイン追加 |
| **Undo/Redo** | 最大100回分の操作履歴（`undoStack` / `redoStack`） |
| **一括 plan_flexibility 変更** | 指定日付以前の全アサインメントの flexibility を一括設定 |
| **検索** | 検索クエリの State 管理（UIは SearchBar コンポーネントあり） |
| **表示期間指定** | ガント表示の開始日・終了日をユーザーが指定 |
| **制約チェック** | Rust側でworker不可日オーバー・バー重複・フェーズ日程超過を検知（`check_constraints` IPC） |
| **エラー表示** | ErrorDialog でエラーメッセージ表示 |
| **ログ出力** | Rust側 `write_log` IPC でデバッグログ記録 |

---

## 1. 技術スタック

### 2026年度版（現在の版）

| レイヤー | 技術 |
|---|---|
| **デスクトップフレームワーク** | [Tauri v2](https://tauri.app/) — WebView + Rust ネイティブ |
| **フロントエンド** | React 19 + TypeScript + Vite |
| **状態管理** | React Context + useReducer（Redux/Zustand なし） |
| **バックエンド（ネイティブ層）** | Rust（`serde_yaml` でYAML解析、`tauri::command` でIPCコマンド提供） |
| **IPC通信** | Tauri invoke API（`@tauri-apps/api` v2） |
| **CSSフレームワーク** | なし（インラインスタイルのみ） |
| **テスト** | Jest（ユニット）+ Cypress（コンポーネントテスト） |
| **パッケージ管理** | npm |
| **ビルド** | `npm run tauri:build` → Windows `.exe` / `.msi` 生成 |
| **注意: 未使用パッケージ** | `gantt-task-react ^0.3.9` がインストールされているが、ガント描画は自前実装（カスタムCSS/HTML） |

### 2025年度版（前バージョン）

| レイヤー | 技術 |
|---|---|
| **バックエンド** | Python + FastAPI（`app.py`） |
| **フロントエンド** | React（静的ファイルとしてFlask経由で配信） |
| **データモデル** | Python dataclass（`models.py`） |
| **YAML処理** | Python `yaml_loader.py` |
| **配布形式** | PyInstaller で `.exe` パッケージ化（`GanttEditor.exe`） |
| **起動スクリプト** | `main.py` → `gantt_editor_launcher.py` |

---

## 2. バージョン間の違い

| 観点 | 2025年度版 | 2026年度版 |
|---|---|---|
| **アーキテクチャ** | Python FastAPI サーバー + React SPA（ブラウザ） | Tauri デスクトップアプリ（Rust + React WebView） |
| **配布形式** | PyInstaller `.exe`（ポータブル） | Tauri ビルド `.exe` / `.msi` インストーラー |
| **バックエンド言語** | Python | Rust |
| **制約チェック** | Python側で実装 | Rust側で実装（`constraint_checker.rs`） |
| **状態管理** | Python `AppState` クラス（サーバーサイド） | React useReducer（クライアントサイド） |
| **Undo/Redo** | Python `undo_stack` / `redo_stack` | React state の `undoStack` / `redoStack` |
| **デバイスビュー** | `expand_state` dict でフェーズ展開管理 | `expandedPhaseTaskIds: Set<string>` |
| **モデル定義** | Python `dataclass` | TypeScript `interface` + Rust `struct` |
| **テスト** | 不明（テストファイル未確認） | Jest + Cypress コンポーネントテスト |
| **ガント描画** | React カスタム実装（詳細未確認） | 自前カスタムHTML/CSS実装（gantt-task-react は未使用） |
| **必要な環境** | Python ランタイム（またはexe） | Node.js + Rust/Cargo（開発時）、ビルド済みexe（本番） |

---

## 3. 未実装機能

> 根拠: `要求定義書.md`（2026年度版）および `外部設計書.md`

### 3-1. ガントチャート操作系

| 機能ID | 要件内容 | 現状 |
|---|---|---|
| F-03 | バーの**左右端ドラッグ**による開始日・終了日の個別変更 | **未実装**。現状はバー全体の移動（水平ドラッグ）のみ |
| F-04 | バーの**行間ドラッグ**による担当作業者変更（ワーカービュー） | **未実装** |
| F-05 | バーの重複をビジュアル（赤枠など）でハイライト表示 | 制約チェックは存在するが、バー上の視覚的ハイライトは未確認 |
| F-06 | 制約違反のバーをガント上で色変え表示 | **未実装**（SidePanelにviolation情報は表示されるがバー自体の色変更なし） |

### 3-2. サイドパネル系

| 機能ID | 要件内容 | 現状 |
|---|---|---|
| F-07 | `work_date_list[].hour`（日次作業時間）のサイドパネル編集UI | **未実装**。SidePanelはplan_flexibilityのみ編集可能 |
| F-08 | デバイスビュー専用サイドパネル — フェーズ日付編集・作業者追加/削除 | **未実装**。DeviceViewのサイドパネルは設計なし |

### 3-3. タスク追加ダイアログ系

| 機能ID | 要件内容 | 現状 |
|---|---|---|
| F-09 | 装置→フェーズ→オペレーション の**カスケードドロップダウン**選択 | **未実装**。現状は全operation_taskのフラットリスト |
| F-10 | 追加時の`workload_hours`尊重（時間数計算） | **実装不明**（追加ダイアログは`hoursPerDay`フィールドはあるが、`workload_hours`との連動なし） |
| F-11 | バリデーションエラーのインライン表示 | **未実装** |

### 3-4. ファイル操作系

| 機能ID | 要件内容 | 現状 |
|---|---|---|
| UI-IN-01 | 「1回の操作で2ファイル同時選択」（EnvConfig + Schedule） | **未実装**。現状は2回の順次ダイアログ（`FileButtons.tsx:11-29`） |

### 3-5. 連携機能系（最重要）

| 機能 | 要件内容 | 現状 |
|---|---|---|
| **Submit → Webapp** | GanttChart エディターで「提出」ボタン押下 → Scheduler Webapp の「New Run」ダイアログが開き、編集済みYAMLが自動入力 | **ボタン自体が存在しない**。Toolbarに未追加 |
| **Show Result → Gantt** | Scheduler Webapp の「Show Result」ボタン → GanttChart エディター起動・該当ラン結果YAMLを自動ロード | **未実装**。`RunLogPage.tsx` の `handleShowResult` は簡易ダイアログのみ（Tauriアプリ起動なし） |

---

## 4. 起動方法

### 2026年度版（開発モード）

```bash
# GanttChartソースコードディレクトリへ移動
cd "GanttChart/2026年度/ガントチャートエディターソースコード/今期ガントチャートエディターソースコード"

# 依存パッケージインストール（初回のみ）
npm install

# Tauri 開発サーバー起動（React + Rust バックエンド同時起動）
npm run tauri:dev
```

> `npm run dev` はフロントエンドのみ（Vite）。TauriのIPC（ファイルダイアログ等）を使うには必ず `tauri:dev` を使うこと。

### 2026年度版（本番ビルド）

```bash
npm run tauri:build
# → src-tauri/target/release/bundle/ 以下に .exe / .msi が生成される
```

### 2025年度版（Python版）

```bash
cd "GanttChart/2025年度/前期ガントチャートエディターソースコード"

# Pythonライブラリのインストール（初回のみ）
pip install fastapi uvicorn pyyaml

# 起動
python main.py
# → ブラウザで http://localhost:xxxx を開く
```

または、プレビルド済みの実行ファイルを使用:

```
GanttChart/2025年度/前期ガントチャートエディターソースコード/GanttEditor_1224/GanttEditor.exe
```

---

## 5. コードの問題点

### 5-1. 【最重要】YAMLスキーマの不一致

Timefold ソルバーが出力する YAML と、GanttChart 2026年度版の型定義が合っていない。

**Timefold 出力（実際のYAML）:**
```yaml
schedule:
  workflow_task_list:
    - id: e1p1o1
      workload_hours: 240    # ← 時間数（整数）
```

**GanttChart 2026年度版 Rust モデル（`schedule.rs:119`）:**
```rust
pub struct OperationTask {
    pub workload_days: u32,  // ← 日数として定義。"workload_hours"キーを読もうとしない
}
```

**GanttChart 2026年度版 TypeScript 型（`src/types/schedule.ts:25`）:**
```typescript
export interface OperationTask {
  workloadDays: number;  // ← 同様にworkload_daysを期待
}
```

**影響:** YAMLを読み込んだとき `workload_hours` フィールドはデシリアライズで無視され、`workloadDays` は `undefined` / 0 になる。タスク追加時の工数計算が機能しない。

**修正方針:**
- Rust: `workload_days` → `workload_hours` にフィールド名変更（または `#[serde(alias = "workload_hours")]` 追加）
- TypeScript: `workloadDays: number` → `workloadHours: number` に変更
- タスク追加ダイアログの計算ロジックも合わせて修正

---

### 5-2. 【重要】`schedule:` ルートキーラッパー

Timefold の出力YAMLは `schedule:` キーでラップされている:
```yaml
schedule:
  plan_range: ...
  workflow_task_list: ...
  assignment_list: ...
```

Rust側には `ScheduleWrapper` 構造体（`schedule.rs:6`）が定義されており、一見対応しているように見える。ただし保存時（`save_schedule_yaml`）に `ScheduleWrapper` に再ラップして書き出しているか確認が必要。ラップなしで保存すると次回ロード時にパースエラーになる。

---

### 5-3. 「Submit」ボタンが存在しない

**ユーザーの期待動作:** GanttChart で編集完了後に「Submit」（提出/送信）ボタンを押す → Scheduler Webapp の「New Run」ダイアログが開く

**実際のコード（`Toolbar.tsx`）:**
- 「ファイル読込」「上書保存」「名前を付けて保存」「タスク追加」ボタンのみ存在
- Submit ボタンは存在しない
- Timefold を直接実行するコードも存在しない（現バージョンでTimefoldを呼ぶIPCコマンドなし）

> 注: ユーザーが「Submit を押すとTimefoldが動くように見える」と言及しているが、コード上はそのような処理は確認できない。もし動いているとすれば別の経路（システム設定、以前のバージョンの残骸など）の可能性あり。

---

### 5-4. Webapp「Show Result」が GanttChart を起動しない

**`webapp/src/pages/RunLogPage.tsx` の `handleShowResult()`:**
```tsx
// 現状: ラン結果情報を表示するだけのダイアログを開く
setGanttDialog({ runId, outputDir });
```

実際にはシンプルな `Dialog` コンポーネントがラン ID とパスを表示するだけ。Tauri アプリ（GanttChart）の起動処理は一切ない。

---

### 5-5. ファイル選択UIが仕様と異なる

**仕様（外部設計書 UI-IN-01）:** 「1回の操作で2ファイル同時選択」  
**現状（`FileButtons.tsx:11-29`）:** ファイルダイアログを2回順次表示（1回目がEnvConfig、2回目がSchedule）

ユーザーが逆の順序で選ぶと誤ったデータがロードされる危険がある。

---

## 6. 連携実装方針

### 連携A: GanttChart「Submit」→ Webapp「New Run」ダイアログ

#### 全体フロー

```
[ユーザー] GanttChart で編集完了
    ↓ "Submit" ボタン押下
[GanttChart] 現在の Schedule.yaml を一時ファイル or 指定パスに保存
    ↓ Webapp に対して「New Run」を開くよう通知
[Webapp] 「New Run」ダイアログが開く（EnvConfig + Schedule が自動入力済み）
    ↓ ユーザーが確認して「Run」押下
[Timefold] ソルバー実行
```

#### 技術的な選択肢

**方法A（推奨）: ファイルパスをURLパラメータで渡す**

1. GanttChart（Tauri側）:
   - Submit ボタン押下 → 現在の Schedule.yaml を保存
   - `tauri-plugin-opener` または `shell::open()` でブラウザを起動:
     ```
     http://localhost:3000/newrun?env=<EnvConfigパス>&schedule=<Scheduleパス>
     ```
2. Webapp 側（React Router）:
   - `/newrun` ルートまたは `?env=&schedule=` パラメータを受け取る
   - `NewRunModal` を自動オープン、指定パスのYAMLをドロップゾーンに自動セット

**実装場所:**
- `GanttChart/src/components/Toolbar/FileButtons.tsx` に「Submit」ボタン追加
- `GanttChart/src/api/tauriCommands.ts` に `openBrowserToNewRun(envPath, schedPath)` 追加
- `GanttChart/src-tauri/src/commands/` に `open_browser_command` 追加
- `Webapp/src/App.tsx` にルート追加 or `RunLogPage.tsx` でURLパラメータ処理追加

**方法B: service 経由でHTTP通知**
1. Tauri から `web/service/` のローカルAPIにHTTP POST（ファイルパス付き）
2. service がWebappに WebSocket / SSE で通知
3. Webapp が New Run ダイアログを開く

---

### 連携B: Webapp「Show Result」→ GanttChart 起動

#### 全体フロー

```
[ユーザー] Webapp で「Show Result」ボタン押下
    ↓ 該当ランの EnvConfig + Schedule（ソルバー出力）のパスを特定
[Webapp] GanttChart Tauri アプリを起動（ファイルパスを引数で渡す）
    ↓
[GanttChart] 起動と同時に指定YAMLを自動ロード → ガント表示
```

#### 技術的な実装

**Webappから Tauri アプリを起動する方法:**
```javascript
// webapp 側（ブラウザ上のReact）
// Tauri はブラウザから直接起動できないため、service経由が必要

// 方法1: service の REST API を呼び、service が Tauri 実行ファイルを subprocess で起動
await fetch('http://localhost:<servicePort>/api/open-gantt', {
  method: 'POST',
  body: JSON.stringify({ envPath: '...', schedulePath: '...' })
});
```

**`web/service/` 側の実装（Node.js）:**
```javascript
// POST /api/open-gantt
const { spawn } = require('child_process');
app.post('/api/open-gantt', (req, res) => {
  const { envPath, schedulePath } = req.body;
  const ganttExePath = process.env.GANTT_EXE_PATH; // 設定ファイルから
  spawn(ganttExePath, ['--env', envPath, '--schedule', schedulePath]);
  res.json({ ok: true });
});
```

**GanttChart 側（Tauri）のCLI引数対応:**
- `src-tauri/src/main.rs` で起動引数（`--env`, `--schedule`）を受け取り、自動的にファイルロードする処理を追加

**実装場所まとめ:**

| 実装場所 | 変更内容 |
|---|---|
| `webapp/src/pages/RunLogPage.tsx` | `handleShowResult` から service の `/api/open-gantt` を呼ぶ |
| `web/service/` (Node.js) | `POST /api/open-gantt` エンドポイント追加、Tauri exe を subprocess 起動 |
| `GanttChart/src-tauri/src/main.rs` | CLI引数パース → 起動時に自動ファイルロードを `LOAD_FILES` アクションで発火 |
| `GanttChart/src/api/tauriCommands.ts` | 起動引数受け取り用 IPC コマンド（`get_startup_files` など）追加 |

---

## 7. その他補足

### 7-1. service レイヤーの現状

`web/service/` は Node.js サーバーで、独立した git リポジトリとデータフォルダを持つ。現時点でWebapp と GanttChart の仲介役として設計されていないが、上記連携実装において**その役割を担う最適な場所**。

### 7-2. Webapp の `ganttService.ts` と GanttChart の違い

- `webapp/src/services/ganttService.ts` はWebapp内でブラウザ上にガントを表示するためのサービス（デスクトップGanttエディターとは別物）
- Webappの `useGantt.ts` は public フォルダの YAML ファイルをフェッチしてプレビュー表示するだけ
- 実際の編集・保存機能はデスクトップ GanttChart エディターにしかない

### 7-3. 依存パッケージ `gantt-task-react` について

`package.json` に `gantt-task-react ^0.3.9` がインストールされているが、実際のガント描画にはこのライブラリを使っていない（自前実装）。不要なら削除可能。残す場合は意図を明記すること。

### 7-4. Cypress テストの設定

`window.__APP_CONTEXT__` に state/dispatch を公開（`AppContext.tsx:33-36`）しており、Cypress テストからコンテキストを直接操作できる設計になっている。ただしこれは `window.Cypress` が存在する場合のみ公開されるため本番では無効。

### 7-5. 日付フォーマットの正規化

Rust の `normalize_dates()` 関数（`schedule.rs:20`）が `YYYY/MM/DD` / `YYYY/M/D` を `YYYY-MM-DD` に変換する。Timefold が `2025/09/01` 形式で出力してもGanttChart側で正規化される。

### 7-6. 作業優先度の提案

| 優先度 | 対応内容 |
|---|---|
| 🔴 最高 | `workload_hours` スキーマ不一致の修正（Rust + TypeScript両方） |
| 🔴 最高 | `schedule:` ラッパーの保存時の動作確認・修正 |
| 🟠 高 | Submit ボタン追加 + Webapp New Run ダイアログとの連携 |
| 🟠 高 | Show Result → GanttChart 起動連携（service 経由） |
| 🟡 中 | バー左右端ドラッグ（開始日・終了日の独立変更） |
| 🟡 中 | タスク追加ダイアログのカスケードドロップダウン |
| 🟡 中 | サイドパネルの work_date_list 時間数編集UI |
| 🟢 低 | ファイル選択を2回→1回（同時2ファイル選択）に変更 |
| 🟢 低 | `gantt-task-react` パッケージの整理（使用するか削除するか決定） |

---

*本ドキュメントは 2026-06-17 時点のソースコード調査に基づく。Timefoldデータモデルのさらなる変更がある場合はスキーマ確認を再度行うこと。*
