# Azure リソース・アクセス権要件 — Timefold Scheduler

本ドキュメントは、Timefold スケジューラを展開するために必要な Azure
リソースおよび RBAC ロール割り当てを定義します。申請経路上のいずれの
担当者でも理解できるよう、内容は技術仕様として記述しています。

---

## プロジェクト背景

Timefold スケジューラは、以下の要素で構成されます。

1. ユーザーのローカル PC で動作する React 製 Web アプリ。2 つの YAML
   ファイル (`EnvConfig.yaml` および `Schedule.yaml`) をアップロードし、
   ソルバが生成する結果 YAML をダウンロードします。
2. Azure Container Apps 上にホストされる HTTP API サービス。Web アプリ
   からのアップロードを受信し、Azure Blob Storage に保存して、結果
   ダウンロード用の署名付き URL を返します。
3. Docker コンテナとしてパッケージ化された Timefold ソルバ本体。Azure
   Batch によりオートスケール可能な計算ノード上で実行されます。ソルバは
   Blob から入力 YAML を読み込み、結果 YAML とステータスを Blob に書き
   出します。

アーキテクチャ全体図 (シーケンス図含む) は [`Azure.md`](../Azure.md) を
参照してください。

本リクエストは **Azure Batch 計算層を含むシステム全体** を対象とします。

---

## 構築が必要なリソース

ライフサイクル管理を簡潔にするため、全てのリソースを単一のリソース
グループ内に配置します。

### 1. リソースグループ
| 項目             | 値                                                            |
| ---------------- | ------------------------------------------------------------- |
| 名前             | `rg-timefold-prod` *(社内命名規則に従う場合は変更可)*         |
| リージョン        | `japaneast` *(低レイテンシの最寄りリージョン。変更可)*        |
| タグ              | `project=timefold`, `env=prod`                                |

### 2. ストレージアカウント + Blob コンテナー
| 項目                       | 値                                                                      |
| -------------------------- | ----------------------------------------------------------------------- |
| ストレージアカウント名       | `sttimefoldprod<suffix>` *(全世界でユニーク必須)*                       |
| SKU                        | `Standard_LRS` (ローカル冗長; 最安構成)                                 |
| 種類 (Kind)                | `StorageV2`                                                             |
| アクセスティア              | `Hot`                                                                   |
| パブリック Blob アクセス      | **無効**                                                                |
| 最小 TLS バージョン         | `TLS1_2`                                                                |
| コンテナー名                 | `timefold` (プライベート)                                               |
| ライフサイクルポリシー        | `timefold/` 配下で 90 日以上更新されていない Blob を自動削除            |

### 3. Azure Container Registry (ACR)
| 項目             | 値                                                                       |
| ---------------- | ------------------------------------------------------------------------ |
| 名前             | `acrtimefoldprod<suffix>` *(全世界でユニーク必須)*                       |
| SKU              | `Basic` (約 $5/月の定額。280 MB 程度のイメージに十分)                    |
| Admin ユーザー    | **無効** (AAD 認証のみ使用。共有キーは利用しない)                          |

### 4. Azure Container Apps 環境 + HTTP アプリ
| 項目                       | 値                                                                            |
| -------------------------- | ----------------------------------------------------------------------------- |
| 環境名                      | `cae-timefold-prod`                                                           |
| アプリ名                    | `ca-tf-api`                                                                   |
| 初期イメージ (プレースホルダー) | `mcr.microsoft.com/k8se/quickstart:latest` *(権限付与後にリビジョン更新で置き換え)* |
| ターゲットポート              | `8080`                                                                        |
| Ingress                    | External (パブリック HTTPS。TLS は Azure 管理)                                |
| 最小レプリカ数               | `0` (ゼロスケール)                                                            |
| 最大レプリカ数               | `2`                                                                           |
| CPU / メモリ                | 0.5 vCPU / 1 GiB                                                              |
| マネージド ID               | **System-assigned** (アプリ作成時に自動生成)                                    |
| 環境変数                     | `STORAGE_ACCOUNT=<ストレージアカウント名>`, `BLOB_CONTAINER=timefold`, `BATCH_ACCOUNT=<Batch アカウント名>`, `BATCH_POOL_ID=pool-timefold-prod` |
| ログ出力先                    | なし (本フェーズでは Log Analytics は不使用)                                   |

### 5. Azure Batch アカウント
| 項目                       | 値                                                                          |
| -------------------------- | --------------------------------------------------------------------------- |
| 名前                        | `batchtimefoldprod<suffix>` *(全世界でユニーク必須)*                        |
| 連携ストレージアカウント       | `sttimefoldprod<suffix>` (上記ストレージアカウント。resourceFiles / outputFiles の auto-storage に必要) |
| ID                         | **System-assigned** (Batch アカウント自身が ARM 操作に使用)                  |
| プール割り当てモード           | Batch service                                                              |
| パブリックネットワークアクセス  | 有効 (AAD 認証で制限)                                                       |

### 6. Azure Batch プール
| 項目                            | 値                                                                  |
| ------------------------------- | ------------------------------------------------------------------- |
| プール ID                        | `pool-timefold-prod`                                                |
| VM SKU                          | `Standard_F2s_v2` (2 vCPU / 4 GB; ソルバ向け計算最適化)              |
| ノード OS イメージ                | `microsoft-azure-batch / ubuntu-server-container 20-04-lts`         |
| Dedicated ノード目標数            | 0                                                                  |
| Low-priority ノード目標数         | 0 (保留中タスク数に応じて 0–3 でオートスケール)                       |
| 最大ノード数                      | 3 (コスト上限)                                                       |
| ノードあたりタスク数               | 1                                                                  |
| コンテナー構成                    | dockerCompatible; レジストリ `acrtimefoldprod<suffix>.azurecr.io`    |
| ID                              | **User-assigned** マネージド ID `mi-tf-pool` (#7 参照)               |
| オートスケール式                   | 保留中タスク 1 件につきノード 1 台; アイドル 10 分でノード解放        |

### 7. User-assigned マネージド ID (Batch プール用)
| 項目     | 値             |
| -------- | -------------- |
| 名前      | `mi-tf-pool`   |
| リージョン | リソースグループと同一 |

この ID は Batch プールに紐付けられます。各計算ノードがこの ID を用いて
ACR から Timefold コンテナイメージを pull し、Blob を読み書きします。

---

## サブスクリプションレベルのリソースプロバイダー登録

以下のプロバイダー登録が必要です (サブスクリプション毎に 1 回のみ、コストなし)。
各 `Microsoft.X` は、ある Azure サービスチームが所有する名前空間であり、
そのカテゴリのリソースを作成する前に当該プロバイダーを **登録 (Registered)**
状態にしておく必要があります。

| プロバイダー                       | 所管するリソース                                                       | 本プロジェクトでの用途                                                                  |
| --------------------------------- | --------------------------------------------------------------------- | --------------------------------------------------------------------------------------- |
| `Microsoft.Storage`               | ストレージアカウント、Blob コンテナー、キュー、テーブル、ファイル共有 | 全 YAML および `status.json` を保管する Blob ストレージ                                  |
| `Microsoft.Authorization`         | RBAC: ロール割り当て、ロール定義、ロック                              | `az role assignment` 全般に必須。未登録時は権限付与不可                                  |
| `Microsoft.App`                   | Azure Container Apps 環境 + HTTP アプリ                              | API Controller をホストするサーバーレス HTTP サービス                                    |
| `Microsoft.OperationalInsights`   | Log Analytics ワークスペース                                         | ログを無効化していても ACA の依存関係として登録必須                                       |
| `Microsoft.ContainerRegistry`     | Azure Container Registry (ACR)                                       | `api-controller` および `timefold` の Docker イメージを格納するプライベートレジストリ      |
| `Microsoft.ManagedIdentity`       | User-assigned マネージド ID                                          | Batch プールに紐付ける `mi-tf-pool` を作成するために必要                                 |
| `Microsoft.Batch`                 | Batch アカウント、プール、ジョブ、タスク                              | Timefold ソルバをコンテナタスクとして実行する計算層                                       |

登録状態の確認:
```bash
az provider list --query "[?contains(['Microsoft.Storage','Microsoft.Authorization','Microsoft.App','Microsoft.OperationalInsights','Microsoft.ContainerRegistry','Microsoft.ManagedIdentity','Microsoft.Batch'], namespace)].{name:namespace, state:registrationState}" -o table
```
`NotRegistered` と表示されたプロバイダーがあれば、以下で登録:
```bash
az provider register --namespace Microsoft.<X>
```

---

## vCPU クォータ申請 (Azure Batch)

個人および新規企業サブスクリプションでは、Batch の vCPU クォータが
**初期値 0** であるケースが多くあります。プールをスケールアップするため、
事前にクォータ引き上げが必要です。

| クォータ                                    | 申請値 | 根拠                                                              |
| ------------------------------------------ | ----- | ----------------------------------------------------------------- |
| Batch — Total Low-priority vCPUs           | `6`   | 最大ノード 3 台 × 2 vCPU/台 (Standard_F2s_v2 ファミリ)            |
| Batch — Dedicated cores per VM family (Fsv2) | `4` (任意) | 本番環境で dedicated (非 low-priority) VM が必要となった場合 |

申請経路: ポータル → **クォータ** → **コンピューティング** → サブスクリプ
ション + リージョンで絞り込み → 対象行を選択 → 引き上げ申請。通常の承認
所要時間は数分〜48 時間。

---

## 必要な RBAC ロール割り当て

### A. プロジェクトユーザーアカウント

アプリを展開・運用するユーザーアカウントに以下のロールを付与してください。

| ロール                                | スコープ                                | 用途                                                              |
| ------------------------------------ | -------------------------------------- | ----------------------------------------------------------------- |
| **Reader**                           | リソースグループ `rg-timefold-prod`     | ポータルおよび CLI でリソースを参照                                |
| **Storage Blob Data Contributor**    | ストレージアカウント `sttimefoldprod*`  | CLI からの Blob のアップロード・ダウンロード・確認 (デバッグ用)     |
| **AcrPush**                          | ACR `acrtimefoldprod*`                  | アプリケーションイメージのレジストリへの push                       |
| **Container Apps Contributor**       | ACA アプリ `ca-tf-api`                  | API の新リビジョン展開 (イメージ更新)                              |
| **Azure Batch Account Contributor**  | Batch アカウント `batchtimefoldprod*`   | テスト時の Batch タスク投入および終了                              |

`Owner` および `User Access Administrator` は**不要です**。RBAC 管理は
権限管理側に留まります。

### B. ACA アプリの System-assigned マネージド ID

ACA アプリ `ca-tf-api` を作成すると、System-assigned マネージド ID が
自動生成され、プリンシパル ID が付与されます。その ID に対して以下を
付与してください。

| ロール                                | スコープ                                | 用途                                                              |
| ------------------------------------ | -------------------------------------- | ----------------------------------------------------------------- |
| **Storage Blob Data Contributor**    | ストレージアカウント `sttimefoldprod*`  | API による入力・出力・ステータス Blob の読み書き                    |
| **AcrPull**                          | ACR `acrtimefoldprod*`                  | デプロイ時およびコールドスタート時のイメージ取得                    |
| **Azure Batch Account Contributor**  | Batch アカウント `batchtimefoldprod*`   | `POST /runSolver` 時の Batch タスク作成、`POST /cancel/{runId}` 時のタスク終了 |

### C. Batch プールの User-assigned マネージド ID (`mi-tf-pool`)

| ロール                                | スコープ                                | 用途                                                              |
| ------------------------------------ | -------------------------------------- | ----------------------------------------------------------------- |
| **Storage Blob Data Contributor**    | ストレージアカウント `sttimefoldprod*`  | 計算ノードによる入力 YAML の読み込みおよび出力 YAML / status.json の書き込み |
| **AcrPull**                          | ACR `acrtimefoldprod*`                  | 計算ノードによる Timefold コンテナイメージの pull                    |

**ロール割り当て合計: 9 件** (ユーザー 5 件 + ACA MI 3 件 + Pool MI 2 件)。

割り当て箇所:
- ACA アプリの MI プリンシパル ID — ポータル → ACA アプリ `ca-tf-api` →
  左サイドバー **Identity** → System assigned → オブジェクト (プリンシパル) ID
- プール MI プリンシパル ID — ポータル → マネージド ID `mi-tf-pool` →
  概要 → オブジェクト (プリンシパル) ID

---

## 想定コスト

| コンポーネント            | アイドル時月額       | 1 ソルブあたり (8 時間)                  | 備考                                                            |
| ------------------------ | ------------------- | --------------------------------------- | --------------------------------------------------------------- |
| リソースグループ           | $0                  | $0                                      | メタデータのみ                                                   |
| ストレージアカウント        | < $0.10             | わずか                                  | 数 MB の YAML。Hot ティアで約 $0.02/GB                           |
| ACR Basic                | $5 定額             | $0                                      | 10 GB のストレージ含む。イメージ pull はリージョン内無料           |
| ACA 環境 + アプリ          | $0                  | 約 $0.01                                 | 個人検証でゼロスケール動作を確認済み                              |
| Batch アカウント           | $0                  | $0                                      | サービス自体は無料                                              |
| Batch プール (ゼロスケール) | $0                  | $0                                      | オートスケール式によりアイドル時は 0 ノード                         |
| Batch 計算時間             | $0                  | 約 $0.30 (low-priority) 〜 $1.60 (standard) | `Standard_F2s_v2` での 8 時間ソルブあたり                       |
| User-assigned MI         | $0                  | $0                                      | 無料                                                            |
| **アイドル時合計**         | **約 $5/月**        | —                                       | 主に ACR Basic 分                                               |
| **1 実行あたり**           | —                   | **約 $0.30 〜 $1.60**                    | 主に VM 計算時間                                                 |

推奨予算アラート: サブスクリプションレベルで $30/月、50% / 90% / 100% の
3 段階通知。

---

## 提供後に必要な情報

アプリケーションコードを展開するため、以下の情報をご提供ください。

1. サブスクリプション ID (GUID)
2. リソースグループ名
3. ストレージアカウント名 (実際に作成された一意の名前)
4. コンテナー名 (`timefold` から変更がある場合)
5. ACR 名 + ログインサーバー (例: `acrtimefoldprodxyz` / `acrtimefoldprodxyz.azurecr.io`)
6. ACA 環境名 + ACA アプリ名
7. ACA アプリのパブリック URL (`https://...azurecontainerapps.io` 形式の完全な ingress エンドポイント)
8. Batch アカウント名 + URL (例: `batchtimefoldprodxyz` / `https://batchtimefoldprodxyz.<region>.batch.azure.com`)
9. Batch プール ID (`pool-timefold-prod`)
10. User-assigned MI 名 + リソース ID (`mi-tf-pool` および `/subscriptions/.../mi-tf-pool` 形式の完全リソース ID)
11. 使用リージョン
12. セクション A の 5 つのユーザーロールが全て割り当て済みである旨の確認
13. セクション B の 3 つの ACA MI ロールが割り当て済みである旨の確認
14. セクション C の 2 つのプール MI ロールが割り当て済みである旨の確認
15. Batch vCPU クォータの引き上げが完了している旨の確認

以上によりアプリケーション展開が可能となります。これ以降の権限管理側の
作業は発生しません。

---

## 提供後にプロジェクトユーザーが実施する作業

権限管理側の把握用に、提供後に実施される作業を以下に示します。

1. API Controller イメージのビルドと ACR への `docker push` (`AcrPush`
   ロールを使用)
2. Timefold ソルバイメージのビルドと ACR への `docker push`
3. `az containerapp update` による `ca-tf-api` への API イメージの展開
   (`Container Apps Contributor` ロールを使用)
4. Timefold イメージを参照する Batch タスクの投入 (`Azure Batch Account
   Contributor` ロールを使用)。初回はプールおよびイメージ pull の動作
   確認のため CLI から手動で実行
5. ローカル PC 上で `VITE_API_BASE_URL` を ACA アプリのパブリック URL
   に向けた状態で React Web アプリを起動
6. API 経由でのアップロード → ステータス → ダウンロードのエンドツー
   エンドフロー (実際の Batch タスクが作成される) の動作確認

これ以降の権限変更・追加プロビジョニング・管理者対応は不要です。

---

## 本リクエストの対象外

- **本番運用向けの強化項目** — API への Azure Entra ID 認証、カスタム
  ドメイン、プライベートエンドポイント、地理冗長ストレージなど。PoC
  段階では対象外。
- **CI/CD パイプライン** — 本フェーズでは手動 `docker push` および
  `az containerapp update` で十分。

---

## 参考資料

v1 アーキテクチャの全体像、シーケンス図、および設計判断の根拠は
[`Azure.md`](../Azure.md) を参照してください。同ドキュメントのセクション 8
「Auth and security」には全マネージド ID と必要ロールが記載されており、
本リクエストはその全てを対象としています。

本システムが使用する 4 つの Azure サービス (Blob、ACR、ACA、Batch) の
1 ページリファレンスおよび使用しないサービス一覧は
[`Azure-Products-Required.md`](./Azure-Products-Required.md) を参照して
ください。
