# 北村さんへの報告（Teamsチャット用）

Fill in the `【要記入】` spots before sending — get the values with:
```bash
source ~/azure-timefold-company-env.sh
echo "BATCH=$BATCH POOL_ID=$POOL_ID MI_NAME=$MI_NAME JOB_ID=$JOB_ID"
```

Everything below the line is copy-paste ready for Teams chat.

---

北村さん

お疲れ様です。本日のAzure作業(Timefoldソルバー構築)で、お昼頃から複数のリソースにアクセスできなくなっています。SYS様/Azure管理者側でご確認いただきたく、状況を共有します。

【発生している問題】

午前中は問題なく操作できていましたが、本日12時前後から以下が同時に発生するようになりました。

1. Container Registry(ACR: ASEQINSTALLATIONDEVRegistory)にアクセスできない
az acr login、docker push、az acr repository show-tagsが全て失敗します(新規イメージ・既存のTimefoldイメージ両方とも)。エラーは「Could not connect to the registry login server」。nslookupで確認したところ、このホスト名はPrivate Endpoint経由のプライベートIP(10.33.188.5)に解決されますが、こちらの端末からそのIPへ到達できていません。

2. Blob Storageにもアクセスできない
同じタイミングで、Storage Accountへのアクセス(az storage container list等)も同様に失敗するようになりました。ACRと同時に発生しているので、個別のリソースの問題ではなくネットワーク経路レベルの問題だと考えています。

3. Batch のPoolリサイズ・Job作成ができない
これはACR/Storageの問題より前、本日中盤から発生していました。az batch pool resize、az batch job createが「(AuthorizationFailure) This request is not authorized to perform this operation」で失敗します。付与されているロール(Azure Batch Account Contributor等)自体は変わっていないように見えるので、ロールとは別の制限がかかっている可能性があります。

4. Container Apps(API Controller)のデプロイも進められない
API ControllerのイメージをACRにpushする必要があるため、上記1のACR接続不可の影響でこちらも止まっています。ACRが直れば自動的に進められる想定です。

【原因の推測(未確定です)】

Activity Log上で、本日昼頃にポリシー(Audit系かDeploy系か未確認)の変更・適用と思われる記録がありました。もしDeployIfNotExists効果のポリシーであれば、自動修復によってACRやStorageのネットワーク設定が意図せず変更された可能性があります。社内VPN(Cato)側の変更の可能性も完全には否定できませんが、まずはこちらのAzureポリシーの変更が主因ではないかと考えています。

【お願いしたいこと】

1. 上記ポリシー変更の内容確認(意図した変更かどうか)
2. こちらの端末(Cato VPN経由)からACR/Storage/Batchに正常にアクセスできる状態に戻していただくこと。特にACRはPrivate Endpoint経由のみになっており、Cato VPNからの経路が現在通っていません。パブリックアクセスを許可する方針か、プライベート経路を正しく通す方針か、方針を確認の上対応いただきたいです
3. 可能であれば、Batch Poolを以下の要件でクリーンに作り直していただきたいです(本日何度か作成を試みた影響で途中の状態が残っている可能性があるため)

【Batch Pool再作成の要件】
- リソースグループ: AS-EQ-INSTALLATION-DEV-RG / リージョン: japaneast
- Batchアカウント: 【要記入】/ Pool ID: 【要記入】
- VMサイズ: Standard_F16s_v2、Dedicated(専用)2ノード、Low-priorityは使用しない
- OSイメージ: Publisher microsoft-dsvm / Offer ubuntu-hpc / Sku 2204、Node Agent SKU: batch.node.ubuntu 22.04(コンテナ対応の最新検証済みイメージを都度ご確認ください)
- コンテナ構成: 有効(Custom)。レジストリ aseqinstallationdevregistory-c4dkgaafg3cta9gp.azurecr.io、イメージ aseqinstallationdevregistory-c4dkgaafg3cta9gp.azurecr.io/timefold:v1
- Managed Identity: User-assigned 【要記入】(ACRへのAcrPull、StorageへのStorage Blob Data Contributorが付与済みのもの)
- タスクスロット数/ノード: 1
- Job: Poolにリンクしたものを1つ作成(Job ID: 【要記入】)
※ ネットワーク疎通(2)が直っていない状態でPoolだけ作り直しても、同じ理由でコンテナイメージのpullに失敗しますので、まずは2を優先していただきたいです

お手すきの際にご確認いただけますと助かります。よろしくお願いいたします。
