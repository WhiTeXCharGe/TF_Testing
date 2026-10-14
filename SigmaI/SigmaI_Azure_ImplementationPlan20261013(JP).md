# Sigma-I Azure 実装計画（概要）

> 図中心の概要版です。詳細版（英語）: [SigmaI_Azure_DetailedDesign20261013.md](./SigmaI_Azure_DetailedDesign20261013.md)
> English: [SigmaI_Azure_ImplementationPlan20261013.md](./SigmaI_Azure_ImplementationPlan20261013.md)

**目的:** SchedulerWeb でソルバーのオプションを選んで実行する。Timefold で使っている Azure 構成（ACA + ACR + Blob + **同じ Batch アカウント**）をそのまま使い、Sigma-I には Batch アカウント内に **専用のプール（ノード群）** を用意する。

---

## 1. ソルバーのオプション

| # | SchedulerWeb のオプション | Stage 1 | Stage 2 | 実行場所 |
|---|---|---|---|---|
| ① | **Timefold** | Timefold | Timefold | Batch → **Timefold プール**（既存） |
| ② | **Sigma-I: D-Wave + SCIP** | D-Wave（量子アニーリング・クラウド） | SCIP | Batch → **Sigma-I プール**（新規） |
| ③ | **Sigma-I: SCIP + SCIP** | SCIP | SCIP | Batch → **Sigma-I プール**（新規） |

> Sigma-I の Stage 2 は常に SCIP（Sigma-I API の仕様で固定）。D-Wave を使えるのは Stage 1 のみ。

```mermaid
flowchart LR
    U(["ユーザーがオプション選択"]) --> O1["① Timefold"]
    U --> O2["② Sigma-I<br/>D-Wave + SCIP"]
    U --> O3["③ Sigma-I<br/>SCIP + SCIP"]

    O1 --> API["API Controller (ACA)<br/>POST /runSolver"]
    O2 --> API
    O3 --> API

    subgraph BA["Batch アカウント（現在と同じ）"]
        direction TB
        subgraph TP["Timefold プール（既存）"]
            TN1["ノード"]
            TN2["ノード"]
        end
        subgraph SP["Sigma-I プール（新規）"]
            SN1["ノード 1"]
            SN2["ノード 2"]
            SN3["ノード … N"]
        end
    end

    API -->|"engine = timefold"| TP
    API -->|"engine = sigmai<br/>stage1 = dwave"| SP
    API -->|"engine = sigmai<br/>stage1 = scip"| SP
    SP -.->|"②のみ: Stage 1 で呼び出し"| DW["D-Wave Leap（クラウド）"]
```

- **Batch アカウントは同じ**、**プールは別**。プールごとにノード・VM サイズ・起動処理が異なる。
- Sigma-I プールのノードは **複数台でもよい**。ノードが多いほど同時に実行できる数が増える。
- ②と③は **同じ Sigma-I ノード** を使う。ソルバーの違いはリクエストのパラメータだけ。

---

## 2. 全体像（ブロック図）

```mermaid
flowchart LR
    W["SchedulerWeb"] -->|"POST /runSolver<br/>yaml x2 + オプション"| ACA

    subgraph ACA["ACA — API Controller"]
        R["engine で振り分け"]
    end

    subgraph BATCH["Azure Batch アカウント（共通）"]
        direction TB
        subgraph TFP["Timefold プール（既存）"]
            TFN["ノード<br/>┌ Timefold コンテナ ┐<br/>1タスク = 起動 → 計算 → 終了"]
        end
        subgraph SGP["Sigma-I プール（新規）— ノード 1..N"]
            subgraph SGN["Sigma-I ノード"]
                direction TB
                ST["① StartTask<br/>（ノード起動時に1回）"]
                subgraph STACK["Sigma-I スタック（起動したまま）"]
                    SA["api :8000"]
                    SW["worker"]
                    SR["redis"]
                end
                BR["② Bridge タスク<br/>（実行ごとに1つ）"]
                ST -->|"docker compose up"| STACK
                BR -->|"localhost:8000"| SA
            end
        end
    end

    R -->|"timefold"| TFP
    R -->|"sigmai"| SGP

    BLOB[("Blob Storage<br/>input / output / status<br/>runs（履歴）")]
    ACR[("ACR<br/>イメージ")]
    KV[("Key Vault<br/>D-Wave トークン")]

    ACA <--> BLOB
    TFN <--> BLOB
    BR <--> BLOB
    ACR -.-> TFN
    ACR -.-> ST
    KV -.-> ST
```

---

## 3. SchedulerWeb が送る内容（オプション別）

3つとも **同じエンドポイント**: `POST /runSolver`（multipart/form-data）。

### ① Timefold — 現在と同じ + `engine`

```text
POST /runSolver
  env              = EnvConfig.yaml        (ファイル)
  sched            = Schedule.yaml         (ファイル)
  label            = "9-11月計画"
  engine           = timefold               ← 新規（省略時の既定値）
```

### ② Sigma-I — D-Wave（Stage 1）+ SCIP（Stage 2）

```text
POST /runSolver
  env              = EnvConfig.yaml        (ファイル)
  sched            = Schedule.yaml         (ファイル)
  label            = "9-11月計画 (D-Wave)"
  engine           = sigmai
  stage1Solver     = dwave
  stage2Solver     = scip
  stage1TimeoutSec = 600
  stage2TimeoutSec = 1800
  weightPreset     = standard              (任意)
```

### ③ Sigma-I — SCIP（Stage 1）+ SCIP（Stage 2）

```text
POST /runSolver
  env              = EnvConfig.yaml        (ファイル)
  sched            = Schedule.yaml         (ファイル)
  label            = "9-11月計画 (SCIP)"
  engine           = sigmai
  stage1Solver     = scip
  stage2Solver     = scip
  stage1TimeoutSec = 1800
  stage2TimeoutSec = 1800
  weightPreset     = standard              (任意)
```

### 各項目の行き先

```mermaid
flowchart LR
    subgraph SW["SchedulerWeb が送信"]
        f1["engine"]
        f2["stage1Solver / stage2Solver"]
        f3["stage1TimeoutSec / stage2TimeoutSec"]
        f4["weightPreset, disabledConstraints"]
        f5["env, sched（ファイル）"]
    end
    subgraph API["API Controller"]
        a1["プール / Batch ジョブを選択"]
        a2["runs/{runId}.json<br/>（マニフェスト + 履歴）"]
        a3["input/{runId}/sigmai_options.json"]
        a4["input/{runId}/*.yaml"]
    end
    subgraph SG["Bridge → Sigma-I POST /jobs"]
        s1["stage1_solver / stage2_solver"]
        s2["stage1_timeout / stage2_timeout"]
        s3["weight_preset, disabled_constraints"]
        s4["environment, schedule（ファイル）<br/>job_id = runId"]
    end
    f1 --> a1
    f1 --> a2
    f2 --> a2 & a3
    f3 --> a2 & a3
    f4 --> a3
    f5 --> a4
    a3 --> s1 & s2 & s3
    a4 --> s4
```

---

## 4. Sigma-I に専用プールが必要な理由

| | Timefold プール | Sigma-I プール |
|---|---|---|
| 実行するもの | コンテナ 1つ | コンテナ 3つ: **API + worker + Redis** |
| 呼び出し方 | コマンド（`entrypoint.sh`） | **HTTP API**（`POST /jobs` → 状態確認 → ダウンロード） |
| 起動コスト | 小（JVM 起動） | **大**: 起動のたびにコアロジック wheel を `pip install` |
| 最適な使い方 | 実行ごとに新規コンテナ | **起動したまま（ウォーム）** で複数回再利用 |
| ノード起動時 | — | **StartTask** が Sigma-I スタックを1回起動 |
| 実行ごと | Timefold コンテナタスク | **Bridge タスク**（小さなスクリプト） |
| シークレット | なし | `API_KEY`（ノード内で生成）、`DWAVE_API_TOKEN`（②のみ） |

---

## 5. Sigma-I ノード1台の中身

```mermaid
flowchart TB
    subgraph NODE["Sigma-I ノード（ノードごとに同じ構成）"]
        direction TB
        subgraph START["StartTask — ノード起動時に1回だけ"]
            S1["ACR からイメージ取得"] --> S2["Blob から wheel と<br/>compose を取得"] --> S3["Key Vault から<br/>D-Wave トークン取得"] --> S4["docker compose up -d<br/>(wheel を pip install)"] --> S5["/health = OK まで待機"]
        end
        subgraph STACK["Sigma-I スタック（起動したまま）"]
            A["api :8000"] --- WK["worker (SCIP / D-Wave)"] --- R["redis"]
        end
        subgraph TASKS["Bridge タスク — 実行ごとに1つ"]
            B["bridge スクリプト"]
        end
        START --> STACK
        B -->|"localhost:8000"| A
    end
    BLOB[("Blob")] -->|"入力 yaml + オプション"| B
    B -->|"状態 / 結果 / ログ"| BLOB
```

---

## 6. Bridge タスクの3つの役割

```mermaid
flowchart LR
    subgraph ONE["① 送信"]
        a1["Blob から<br/>EnvConfig + Schedule<br/>+ sigmai_options.json 取得"] --> a2["POST /jobs<br/>job_id = runId"]
    end
    subgraph TWO["② 監視"]
        b1["GET /jobs/{id}/status<br/>10秒ごと"] --> b2["status/{runId}.json 更新<br/>Running + 進捗"]
        b2 --> b1
    end
    subgraph THREE["③ 保存"]
        c1["GET /download<br/>GET /result"] --> c2["result_Schedule.yaml<br/>+ サマリ + ログを Blob へ"]
    end
    ONE --> TWO --> THREE
```

---

## 7. 全体シーケンス（オプション ② / ③）

```mermaid
sequenceDiagram
    autonumber
    participant W as SchedulerWeb
    participant API as API Controller (ACA)
    participant BLOB as Blob
    participant BQ as Batch (Sigma-I ジョブキュー)
    participant BR as Bridge タスク (ノード)
    participant SG as Sigma-I スタック (ノード)

    W->>API: POST /runSolver (engine=sigmai, stage1Solver, yaml x2)
    API->>BLOB: input/{runId}/ EnvConfig, Schedule, sigmai_options.json
    API->>BLOB: runs/{runId}.json (マニフェスト), status = Submitted
    API->>BQ: タスク {runId} 作成
    API-->>W: 202 {runId}

    Note over BQ: 全 Sigma-I ノードが使用中ならここで待機
    BQ->>BR: 空いているウォームノードでタスク開始
    BR->>SG: POST /jobs (job_id=runId, ソルバー, タイムアウト)
    SG-->>BR: 202 queued

    loop 10秒ごと
        BR->>SG: GET /jobs/{runId}/status
        BR->>BR: ライブ status.json 更新 (Running, 進捗)
        W->>API: GET /status/{runId}
        API-->>W: Running 45%
    end

    alt 成功
        BR->>SG: GET /download + /result
        BR->>BLOB: output/{runId}/result_Schedule.yaml + sigmai/*
        BR->>BLOB: status = Completed
    else 失敗
        BR->>BLOB: output/{runId}/sigmai/ error.log, optimization.log
        BR->>BLOB: status = Failed (errorType)
    end

    API->>BLOB: runs/{runId}.json に結果・時間・ノードを追記
    W->>API: GET /download/{runId}
    API-->>W: result_Schedule.yaml
```

---

## 8. 複数ノードとキュー

```mermaid
gantt
    title Sigma-I プール — ノード2台・1ノード1実行
    dateFormat HH:mm
    axisFormat %H:%M
    section ノード 1
    StartTask (起動 + pip install) :crit, n1s, 09:00, 8m
    実行 A (③ SCIP+SCIP)          :a, after n1s, 30m
    実行 C (② D-Wave+SCIP)        :c, after a, 20m
    section ノード 2
    StartTask (起動 + pip install) :crit, n2s, 09:05, 8m
    実行 B (③ SCIP+SCIP)          :b, after n2s, 30m
    section Batch キュー
    実行 C 待機                    :done, cw, 09:15, 23m
```

- 1ノード = 同時に1実行（`taskSlotsPerNode = 1`）。
- ノード数より実行が多い場合、余った実行は **Batch キューで待機**。SchedulerWeb では **Submitted（待ち #n）** と表示。
- 各ノードはスタックを **1回だけ起動** するので、同じノードでの2回目以降の実行はインストール不要。

---

## 9. 実行ステータス（SchedulerWeb の表示）

```mermaid
stateDiagram-v2
    [*] --> Submitted: API がタスク作成
    Submitted --> Submitted: Batch キューで待機 (queuePosition)
    Submitted --> Running: Bridge 開始
    Running --> Completed: Sigma-I 完了
    Running --> Failed: Sigma-I 失敗 / タイムアウト
    Submitted --> Cancelled: ユーザーがキャンセル
    Running --> Cancelled: ユーザーがキャンセル
    Completed --> [*]
    Failed --> [*]
    Cancelled --> [*]
```

| Sigma-I の状態 | → SchedulerWeb の状態 |
|---|---|
| *（Batch タスク待機中）* | `Submitted` + `queuePosition` |
| `queued`、`running`、`cancel_requested` | `Running` |
| `finished` | `Completed` |
| `failed` | `Failed`（+ `errorType`） |
| `canceled` | `Cancelled` |

Timefold と同じ5状態なので、UI の変更はわずか。

---

## 10. キャンセル

```mermaid
sequenceDiagram
    participant W as SchedulerWeb
    participant API as API Controller
    participant BQ as Batch
    participant BR as Bridge
    participant SG as Sigma-I
    participant BLOB as Blob

    W->>API: DELETE /run/{runId}
    API->>BQ: タスク終了 (terminate)
    alt キュー待機中
        BQ-->>API: 削除（未実行）
    else 実行中
        BQ->>BR: SIGTERM
        BR->>SG: DELETE /jobs/{runId}
        BR->>BR: status = Cancelled
    end
    API->>BLOB: runs/{runId}.json status = Cancelled
    API-->>W: 200
```

---

## 11. Blob Storage — 保存するもの

```mermaid
flowchart TB
    C[("Blob コンテナ")]
    C --> I["input/{runId}/<br/>EnvConfig.yaml · Schedule.yaml<br/><b>sigmai_options.json</b>（新規）"]
    C --> O["output/{runId}/<br/>result_Schedule.yaml<br/><b>sigmai/</b> result.json · result_summary.json · ログ（新規）"]
    C --> S["status/{runId}.json<br/>実行中の状態（+ engine, solver）"]
    C --> RN["<b>runs/{runId}.json</b>（新規）<br/>実行マニフェスト = 履歴"]
    C --> SU["<b>sigmai-setup/{version}/</b>（新規）<br/>wheel · compose · スクリプト"]
```

| プレフィックス | 内容 | 書き込み元 | 保存期間 |
|---|---|---|---|
| `input/{runId}/` | 入力 YAML + Sigma-I オプション | API | 30日 |
| `output/{runId}/` | 結果 YAML + Sigma-I サマリ / ログ | Batch タスク | 30日 |
| `status/{runId}.json` | 実行中のライブ状態 | API → タスク | 30日 |
| **`runs/{runId}.json`** | **誰が・いつ・どのソルバー・設定・バージョン・ノード・所要時間・最終結果** | API（作成 + 完了時に確定） | **長期（履歴）** |
| **`sigmai-setup/{ver}/`** | Sigma-I wheel、compose、StartTask / bridge スクリプト | 自チーム（デプロイ時） | バージョンごと |

`runs/{runId}.json`（簡単な例）:

```json
{
  "runId": "20261014_093000123",
  "label": "9-11月計画 (D-Wave)",
  "engine": "sigmai",
  "solver": { "stage1": "dwave", "stage2": "scip", "stage1TimeoutSec": 600, "stage2TimeoutSec": 1800, "weightPreset": "standard" },
  "versions": { "sigmaiImage": "0.1.0", "sigmaiWheel": "0.1.x", "apiController": "v2" },
  "submittedAt": "2026-10-14T00:30:00Z",
  "execution": { "pool": "pool-sigmai", "node": "tvm-…-1", "queuedSec": 95, "runSec": 1260 },
  "result": { "status": "Completed", "validationPassed": true, "finalScore": 123.4 }
}
```

履歴には **Blob インデックスタグ**（`engine`、`status`、`month`）を付ける。これで「10月の D-Wave 実行一覧」などをデータベースなしで検索できる。

---

## 12. Sigma-I ノードのライフサイクル

```mermaid
stateDiagram-v2
    [*] --> Zero: 実行なし
    Zero --> Booting: 実行が投入された
    Booting --> Ready: StartTask 成功 (約5〜15分・要計測)
    Ready --> Busy: Bridge タスク
    Busy --> Ready: タスク完了
    Ready --> Zero: 60分以上アイドル (オートスケール)
    Booting --> Failed: StartTask 失敗
```

オートスケール: ノード数 = 待機中 + 実行中の Sigma-I 実行数（上限 N）。各ノードは最後の実行から60分間ウォームを維持する。

---

## 13. 変更点

| コンポーネント | 変更内容 | 規模 |
|---|---|---|
| **SchedulerWeb** | オプション選択（①②③）、時間上限、待ち順表示、実行一覧・履歴にエンジン表示 | 小 |
| **API Controller** | `engine` 振り分け、オプション検証、`sigmai_options.json`、`runs/` マニフェスト（全エンジン）、待ち順、任意で `GET /runs` | 中 |
| **Blob** | 新プレフィックス `runs/`、`sigmai-setup/`、`output/{runId}/sigmai/`、インデックスタグ、ライフサイクルルール | 小 |
| **ACR** | Sigma-I API イメージ + `redis:7-alpine` のコピーを登録 | 極小 |
| **Key Vault** | `DWAVE_API_TOKEN`（オプション②） | 極小 |
| **Batch** | 新規 **Sigma-I プール** + ジョブ + StartTask + bridge スクリプト（同じ Batch アカウント） | **大** |
| **Timefold プール / コンテナ** | **変更なし** | — |

---

## 14. ロードマップ

```mermaid
flowchart LR
    P0["P0<br/>ローカル実行成功<br/>+ Sigma-i 確認"] --> P1["P1<br/>ACR + Blob<br/>+ Key Vault"] --> P2["P2<br/>Sigma-I プール<br/>StartTask"] --> P3["P3<br/>Bridge タスク<br/>手動テスト"] --> P4["P4<br/>API Controller<br/>振り分け + runs/"] --> P5["P5<br/>SchedulerWeb UI"] --> P6["P6<br/>複数ノード・<br/>オートスケール・D-Wave"]
```

⚠ **P0 の前提:** ローカルテスト `test-001` は `OPTIMIZATION_FAILED`（`validation_passed: false`）で終了した。先に Sigma-i と原因を解決する。入力やモデルの問題は Azure では解決しない。

まずは **Azure 上で動かすこと** を優先する。Batch のコスト最適化（Timefold とのノード共有、高度なキュー制御）は後で検討する。

---

## 15. 確認事項（Sigma-i / 社内 IT）

| # | 質問 | 確認先 |
|---|---|---|
| 1 | wheel を **イメージに事前インストール** して、毎回の pip を省略できるか？ | Sigma-i |
| 2 | イメージ + wheel を自社 Azure で動かしてよいか？ | Sigma-i |
| 3 | SCIP 用の推奨 VM サイズ（CPU / メモリ）は？ | Sigma-i |
| 4 | Batch ノードからのインターネット接続（PyPI、D-Wave）は可能か？ | 社内 IT |
| 5 | D-Wave Leap の契約 / トークン管理者 / 費用は？ | 社内 |
| 6 | Sigma-I ノードの最大台数（コスト上限）は？ 業務時間中はウォームを維持するか？ | 自チーム + 社内 |
| 7 | 実行履歴（`runs/`）の保存期間は？ | 自チーム + 社内 |
