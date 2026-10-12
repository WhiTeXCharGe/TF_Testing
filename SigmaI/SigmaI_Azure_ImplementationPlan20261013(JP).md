# Sigma-I Azure 実装計画（概要）

> 図中心の概要版です。詳細版（英語）: [SigmaI_Azure_DetailedDesign20261013.md](./SigmaI_Azure_DetailedDesign20261013.md)
> English: [SigmaI_Azure_ImplementationPlan20261013.md](./SigmaI_Azure_ImplementationPlan20261013.md)

**目的:** SchedulerWeb から **Timefold** と **Sigma-I（SCIP / D-Wave）** を選んで実行できるようにする。Timefold で使っている Azure 構成（ACA + ACR + Blob + Batch）をそのまま活用する。

---

## 1. 現在と目標

### 現在 — Timefold のみ

```mermaid
flowchart LR
    W["SchedulerWeb"] -->|"POST /runSolver<br/>EnvConfig + Schedule"| API["API Controller<br/>(ACA)"]
    API -->|"input/{runId}/*.yaml<br/>status/{runId}.json"| BLOB[("Blob Storage")]
    API -->|"タスク作成"| TFPOOL["Batch プール: Timefold"]
    TFPOOL -->|"1タスク = 1コンテナ<br/>起動 → 計算 → 終了"| BLOB
    ACR[("ACR<br/>timefold イメージ")] -.->|pull| TFPOOL
```

### 目標 — Timefold + Sigma-I

```mermaid
flowchart LR
    W["SchedulerWeb<br/>engine = timefold | sigmai"] -->|"POST /runSolver"| API["API Controller<br/>(ACA)"]
    API --> BLOB[("Blob Storage")]
    API -->|"engine = timefold"| TFPOOL["Batch プール: Timefold<br/>(変更なし)"]
    API -->|"engine = sigmai"| SPOOL["Batch プール: Sigma-I<br/>(新規・常駐ノード)"]
    TFPOOL --> BLOB
    SPOOL --> BLOB
    ACR[("ACR")] -.-> TFPOOL
    ACR -.-> SPOOL
    KV[("Key Vault<br/>DWAVE_API_TOKEN")] -.-> SPOOL
    SPOOL -.->|"Stage 1 = dwave の時のみ"| DWAVE["D-Wave Leap<br/>(クラウド)"]
```

---

## 2. Sigma-I で Batch 構成を変える理由

| | Timefold | Sigma-I |
|---|---|---|
| 実行するもの | コンテナ 1つ | コンテナ 3つ: **API + worker + Redis** |
| 呼び出し方 | コマンド（`entrypoint.sh`） | **HTTP API**（`POST /jobs` → 状態確認 → ダウンロード） |
| 起動コスト | 小（JVM 起動） | **大** — 起動のたびにコアロジック wheel を `pip install` |
| 最適な使い方 | 実行ごとに新規コンテナ | **起動したまま（ウォーム）** で複数回再利用 |
| ソルバー | Timefold | Stage 1: `scip` / `dwave`、Stage 2: `scip` のみ |
| シークレット | なし | `API_KEY`（ノード内）、`DWAVE_API_TOKEN`（D-Wave 利用時のみ） |

➡ **ポイント:** Sigma-I スタックは **ノードごとに1回だけ起動**（Batch *StartTask*）。各実行は小さな **Bridge タスク** が Sigma-I API を呼び出す。

---

## 3. Sigma-I Batch ノードの中身

```mermaid
flowchart TB
    subgraph NODE["Batch ノード (pool-sigmai)"]
        direction TB
        subgraph START["StartTask — ノード起動時に1回だけ"]
            S1["ACR からイメージ取得"] --> S2["Blob から wheel と<br/>compose を取得"] --> S3["Key Vault から<br/>DWAVE トークン取得"] --> S4["docker compose up -d<br/>(wheel を pip install)"] --> S5["/health = OK まで待機"]
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
    BLOB[("Blob")] -->|"入力 yaml"| B
    B -->|"状態 / 結果 / ログ"| BLOB
```

---

## 4. Bridge タスクの3つの役割

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

## 5. 全体シーケンス

```mermaid
sequenceDiagram
    autonumber
    participant W as SchedulerWeb
    participant API as API Controller (ACA)
    participant BLOB as Blob
    participant BQ as Batch ジョブキュー (job-sigmai)
    participant BR as Bridge タスク (ノード)
    participant SG as Sigma-I スタック (ノード)

    W->>API: POST /runSolver (engine=sigmai, yaml x2, オプション)
    API->>BLOB: input/{runId}/ EnvConfig, Schedule, sigmai_options.json
    API->>BLOB: status = Submitted
    API->>BQ: タスク {runId} 作成
    API-->>W: 202 {runId}

    Note over BQ: ノード使用中ならここで待機（キュー）
    BQ->>BR: ウォームノードでタスク開始
    BR->>SG: POST /jobs (job_id=runId, stage1/2 ソルバー, タイムアウト)
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

    W->>API: GET /download/{runId}
    API-->>W: result_Schedule.yaml
```

---

## 6. キュー — 同じノードで順番待ち

```mermaid
gantt
    title pool-sigmai — ノード1台・タスクスロット1
    dateFormat HH:mm
    axisFormat %H:%M
    section ノード
    StartTask (起動 + pip install)   :crit, s0, 09:00, 8m
    section 実行
    実行 A (SCIP)                    :a, after s0, 30m
    実行 B (Batch キューで待機)       :done, bq, 09:10, 28m
    実行 B (SCIP)                    :b, after a, 30m
    実行 C (Stage 1 = D-Wave)        :c, after b, 20m
    section ノード
    アイドル保持 → 0台へ縮小          :idle, after c, 30m
```

- Batch の **ジョブキュー** をそのままキューとして使う。`taskSlotsPerNode = 1` → 同時実行は1つ。
- 待機中の実行は SchedulerWeb で **Submitted（待ち #2）** と表示。
- スタックは **1回だけ起動** → 実行 B・C は **インストール時間なし**。

---

## 7. 実行ステータス（SchedulerWeb の表示）

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

Timefold と同じ5状態 → **UI 変更は最小限**。

---

## 8. キャンセル

```mermaid
sequenceDiagram
    participant W as SchedulerWeb
    participant API as API Controller
    participant BQ as Batch
    participant BR as Bridge
    participant SG as Sigma-I

    W->>API: DELETE /run/{runId}
    API->>BQ: タスク終了 (terminate)
    alt キュー待機中
        BQ-->>API: 削除（未実行）
    else 実行中
        BQ->>BR: SIGTERM
        BR->>SG: DELETE /jobs/{runId}
        BR->>BR: status = Cancelled
    end
    API-->>W: 200
```

---

## 9. ノードのライフサイクル（ウォーム維持）

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

---

## 10. コンポーネントごとの変更

| コンポーネント | 変更内容 | 規模 |
|---|---|---|
| **SchedulerWeb** | エンジン選択 + Sigma-I オプション + 待ち順表示 | 小 |
| **API Controller** | `engine` 項目、`job-sigmai` への振り分け、待ち順、`sigmai_options.json` 保存 | 中 |
| **Blob** | `sigmai-setup/` と `output/{runId}/sigmai/` を追加 | 極小 |
| **ACR** | Sigma-I API イメージ + `redis:7-alpine` を登録 | 極小 |
| **Key Vault** | `DWAVE_API_TOKEN` シークレット | 極小 |
| **Batch** | 新規 `pool-sigmai` + `job-sigmai` + StartTask + bridge スクリプト | **大** |
| **Timefold 側** | **変更なし** | — |

---

## 11. ロードマップ

```mermaid
flowchart LR
    P0["P0<br/>ローカル実行成功<br/>+ Sigma-i 確認"] --> P1["P1<br/>ACR + Blob<br/>+ Key Vault"] --> P2["P2<br/>pool-sigmai<br/>StartTask"] --> P3["P3<br/>Bridge タスク<br/>手動テスト"] --> P4["P4<br/>API Controller<br/>振り分け"] --> P5["P5<br/>SchedulerWeb UI"] --> P6["P6<br/>オートスケール<br/>+ 監視"]
```

⚠ **P0 の前提:** ローカルテスト `test-001` は `OPTIMIZATION_FAILED`（`validation_passed: false`）で終了。先に Sigma-i と原因を解決する — 入力やモデルの問題は Azure では解決しない。

---

## 12. 確認事項（Sigma-i / 社内 IT）

| # | 質問 | 確認先 |
|---|---|---|
| 1 | wheel を **イメージに事前インストール** して、毎回の pip を省略できるか？ | Sigma-i |
| 2 | イメージ + wheel を自社 Azure で動かしてよいか？ | Sigma-i |
| 3 | SCIP 用の推奨 VM サイズ（CPU / メモリ）は？ | Sigma-i |
| 4 | Batch ノードからのインターネット接続（PyPI、D-Wave）は可能か？ | 社内 IT |
| 5 | D-Wave Leap の契約 / トークン管理者 / 費用は？ | 社内 |
| 6 | 業務時間中は常時ウォームにするか、アイドル時間で停止のみか？ | 自チーム + 社内 |
