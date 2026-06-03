# Azure リソース・アクセス権要件 — Timefold Scheduler

本ドキュメントは、Timefold スケジューラの概念実証 (PoC) を展開するために
必要な Azure リソースおよび RBAC ロール割り当てを定義します。
申請経路上のいずれの担当者でも理解できるよう、内容は技術仕様として記述
しています。

---

## プロジェクト背景

Timefold スケジューラは、以下 2 つの要素で構成されます。

1. ユーザーのローカル PC で動作する React 製 Web アプリ。2 つの YAML
   ファイル (`EnvConfig.yaml` および `Schedule.yaml`) をアップロードし、
   ソルバが生成する結果 YAML をダウンロードします。
2. Azure Container Apps 上にホストされる HTTP API サービス。Web アプリ
   からのアップロードを受信し、Azure Blob Storage に保存して、結果
   ダウンロード用の署名付き URL を返します。

最終的な目標アーキテクチャ (詳細は [`Azure.md`](../Azure.md) を参照)
には、Timefold ソルバの計算層として Azure Batch も含まれます。
**本リクエストは API 層と Storage 層のみを対象とします。** Batch 計算層
については、API + Storage の動作が検証された後、別途リクエストを提出
します。

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
| 項目                      | 値                                                                          |
| ------------------------- | --------------------------------------------------------------------------- |
| 環境名                     | `cae-timefold-prod`                                                         |
| アプリ名                   | `ca-tf-api`                                                                 |
| 初期イメージ (プレースホルダー) | `mcr.microsoft.com/k8se/quickstart:latest` *(権限付与後にリビジョン更新で置き換え)* |
| ターゲットポート             | `8080`                                                                      |
| Ingress                   | External (パブリック HTTPS。TLS は Azure 管理)                              |
| 最小レプリカ数              | `0` (ゼロスケール)                                                          |
| 最大レプリカ数              | `2`                                                                         |
| CPU / メモリ               | 0.5 vCPU / 1 GiB                                                            |
| マネージド ID              | **System-assigned** (アプリ作成時に自動生成)                                  |
| 環境変数                    | `STORAGE_ACCOUNT=<ストレージアカウント名>`, `BLOB_CONTAINER=timefold`        |
| ログ出力先                   | なし (本フェーズでは Log Analytics は不使用)                                  |

---

## サブスクリプションレベルのリソースプロバイダー登録

以下のプロバイダー登録が必要です (サブスクリプション毎に 1 回のみ、コストなし):

- `Microsoft.Storage`
- `Microsoft.Authorization`
- `Microsoft.App`
- `Microsoft.OperationalInsights`
- `Microsoft.ContainerRegistry`
- `Microsoft.ManagedIdentity`

*(`Microsoft.Batch` は本フェーズでは不要です。Batch 計算層の追加リクエスト
時に別途申請します。)*

登録状態の確認:
```bash
az provider list --query "[?contains(['Microsoft.Storage','Microsoft.Authorization','Microsoft.App','Microsoft.OperationalInsights','Microsoft.ContainerRegistry','Microsoft.ManagedIdentity'], namespace)].{name:namespace, state:registrationState}" -o table
```
`NotRegistered` と表示されたプロバイダーがあれば、以下で登録:
```bash
az provider register --namespace Microsoft.<X>
```

---

## 必要な RBAC ロール割り当て

### A. プロジェクトユーザーアカウント

アプリを展開・運用するユーザーアカウントに以下のロールを付与してください。

| ロール                                | スコープ                              | 用途                                                         |
| ------------------------------------ | ------------------------------------ | ------------------------------------------------------------ |
| **Reader**                           | リソースグループ `rg-timefold-prod`   | ポータルおよび CLI でリソースを参照                            |
| **Storage Blob Data Contributor**    | ストレージアカウント `sttimefoldprod*` | CLI からの Blob のアップロード・ダウンロード・確認 (デバッグ用)  |
| **AcrPush**                          | ACR `acrtimefoldprod*`                | アプリケーションイメージのレジストリへの push                   |
| **Container Apps Contributor**       | ACA アプリ `ca-tf-api`                | API の新リビジョン展開 (イメージ更新)                          |

`Owner` および `User Access Administrator` は**不要です**。RBAC 管理は
権限管理側に留まります。

### B. ACA アプリの System-assigned マネージド ID

ACA アプリ `ca-tf-api` を作成すると、System-assigned マネージド ID が
自動生成され、プリンシパル ID が付与されます。その ID に対して以下を
付与してください。

| ロール                                | スコープ                              | 用途                                                         |
| ------------------------------------ | ------------------------------------ | ------------------------------------------------------------ |
| **Storage Blob Data Contributor**    | ストレージアカウント `sttimefoldprod*` | API による入力・出力・ステータス Blob の読み書き                |
| **AcrPull**                          | ACR `acrtimefoldprod*`                | デプロイ時およびコールドスタート時のイメージ取得               |

割り当て手順: ポータル → ACA アプリ `ca-tf-api` → 左サイドバー **Identity**
→ System assigned → オブジェクト (プリンシパル) ID をコピー → 該当リソースで
上記 2 つのロールを割り当て。

---

## 想定コスト

| コンポーネント       | 月額コスト    | 備考                                                              |
| ------------------- | ------------- | ----------------------------------------------------------------- |
| リソースグループ      | $0            | メタデータのみ                                                     |
| ストレージアカウント   | < $0.10       | 数 MB の YAML。Hot ティアで約 $0.02/GB                             |
| ACR Basic           | $5 定額       | 10 GB のストレージを含む                                            |
| ACA 環境             | $0            | アクティブなレプリカ単位の課金                                       |
| ACA アプリ (アイドル) | $0            | 個人検証環境でゼロスケール動作を確認済み                              |
| **アイドル時合計**    | **約 $5/月**  | 主に ACR Basic 分                                                  |
| デモ実行 1 回         | 約 $0.01      | ACA の数秒のアクティブ時間と Blob の小規模オペレーション             |

推奨予算アラート: サブスクリプションレベルで $20/月、50% / 90% / 100% の
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
8. 使用リージョン
9. セクション A の 4 つのユーザーロールが全て割り当て済みである旨の確認
10. セクション B の 2 つのマネージド ID ロールが割り当て済みである旨の確認

以上 10 項目によりアプリケーション展開が可能となります。本フェーズ
(API + Storage) において、これ以降の権限管理側の作業は発生しません。

---

## 提供後にプロジェクトユーザーが実施する作業

権限管理側の把握用に、提供後に実施される作業を以下に示します。

1. API Controller イメージおよび Timefold ソルバイメージのビルドと、ACR
   への `docker push` (`AcrPush` ロールを使用)
2. `az containerapp update` による `ca-tf-api` への API イメージの展開
   (`Container Apps Contributor` ロールを使用)
3. ローカル PC 上で `VITE_API_BASE_URL` を ACA アプリのパブリック URL に
   向けた状態で React Web アプリを起動
4. アップロード → ステータス → ダウンロードのエンドツーエンドフロー
   の動作確認

これ以降の権限変更・追加プロビジョニング・管理者対応は不要です。

---

## 本リクエストの対象外

- **Azure Batch** (Timefold ソルバ計算層) — 別途リクエストにて申請。
  追加で必要となる項目: Batch アカウント、コンピュートプール、選定 VM
  SKU の vCPU クォータ、`AcrPull` と `Storage Blob Data Contributor`
  を持つ User-assigned マネージド ID。
- **本番運用向けの強化項目** — API への Azure Entra ID 認証、カスタム
  ドメイン、プライベートエンドポイント、地理冗長ストレージなど。PoC
  段階では対象外。
- **CI/CD パイプライン** — 本フェーズでは手動 `docker push` で十分。

---

## 参考資料

v1 アーキテクチャの全体像、シーケンス図、および設計判断の根拠は
[`Azure.md`](../Azure.md) を参照してください。同ドキュメントのセクション 8
「Auth and security」には全マネージド ID と必要ロールが記載されており、
本リクエストはそのうち API + Storage 部分を対象としています。
