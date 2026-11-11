# 2-2-3 prepare_schedule 説明書（週次計画 ステップ 2-2 + 2-3）

英語版（オプション一覧）: [README.md](README.md)

運用フローの「2-2 Fixed とする製番・FI/FO の特定」と「2-3 Flexible 製番以外を misc タスクとした Schedule.yaml の生成」を 1 回の実行で行うツール。
特に **FI/FO をどう読み取り、補正し、前週の結果と比較するか** を第 4 章に詳しく書いた。

## 1. 何をするツールか
- 前週の自動計画結果（FlexibleSchedule.yaml）と、計画担当者が編集した後の SU_Others を比べる。
- 担当者が触っていない製番・FI/FO は、そのまま自動計画の対象（Flexible）に残す。
- 触られた製番・FI/FO と、それ以外のバーはすべて動かさないもの（misc / Fixed）として Schedule.yaml に入れる。
- ソルバーにそのまま渡せる `01_EnvConfig.yaml` と `01_Schedule.yaml`、および引き続き自動計画する対象のリスト `<日付>_FlexibleWork.yaml` を出力する。

## 2. 入力と出力
| 区分 | 内容 |
|---|---|
| 入力 | 前週の `FlexibleSchedule.yaml`（ツール 3-2 の出力） |
| 入力 | 前週の EnvConfig（workflow・fab・region・会社などの定義を引き継ぐ） |
| 入力 | SU_Others（進行中の作業バー、FI/FO、作業者一覧） |
| 入力（任意） | スキルレベルのファイル（スキルと Regular/Spot だけを読む） |
| 入力 | 製番情報ファイル（製番ごとの顧客・fab・region） |
| 出力 | `<日付>_FlexibleWork.yaml`（Flexible な製番のリストと Flexible な trip のリスト） |
| 出力 | `01_EnvConfig.yaml`、`01_Schedule.yaml` |
| 出力（`--report`） | 判断の理由と、解決できなかった項目のレポート |

## 3. 製番の判定（Fixed か Flexible か）
**製番は「作業」と「バーの文字」だけで判定する。FI/FO の変更は製番の判定に影響しない。**
FI/FO が動いていても、その中の製番の作業が動いていなければ、その製番は Flexible のままである。

| 判定 | 条件 | 結果 |
|---|---|---|
| 作業が変わった | SU_Others 上の（作業者, 日付）の集合が、前週の割り当てと一致しない | Fixed（misc タスク） |
| バーの文字が変わった | 作業者・日付は同じだが、バーの文字だけが書き換えられた（前週の割り当て行の `cell_name` と、そのバーの一番多い文字が違う。大文字小文字・全角半角・空白は無視） | Fixed（misc タスク。名前は新しい文字） |
| 完了 | 最終作業日が `--today` より前 | Fixed |
| 上のどれでもない | | Flexible |

- 時間数は比較しない。前週の FlexibleSchedule に無い製番は常に misc になる（新規製番は 2-4 で追加する）。
- `cell_name` は割り当て行ごとのバーの文字。ツール 3-2 が作り、3-3 がセルに書き、このツールが次週に比べる。`cell_name` が無い行は文字を確認しない。
- このツールは Flexible な割り当て行に `cell_name` を付け、`description` が空なら同じ文字を入れる（ソルバーは `description` を結果に残すため）。

## 4. FI/FO（trip）の扱い
### 4.1 読み取り
- 作業者の行のセルの文字が FI / FO なら印として読む。大文字小文字・全角・空白は無視。`F1`・`Fl`・`F0` などの打ち間違い、`FI(仮)` のような括弧書きも読む。
- `移動` は印として読まない（misc タスクのまま）。赤いセルの FI/FO も他と同じに読む。

### 4.2 基本の組み合わせ
作業者ごとに日付順に並べ、FI とその次の FO を 1 つの trip にする（FI の日から FO の日まで、両端を含む）。

### 4.3 組み合わせにならない FI/FO の補正
実データでは書き方が揃っていない。捨てずに、その作業者の「忙しい日」（セルに何か書かれている日。FI/FO のセルは含めない）を手がかりに補正する。
「連続した忙しい日」は間隔が `--max-gap-days`（既定 14 日）以内でつながる範囲。

| パターン | 補正 |
|---|---|
| FI のあと FO が無く、次が FI、または表の最後 | FI の後の連続した忙しい日の最終日を終わりにする。忙しい日が無ければ 1 日だけの trip |
| FO だけ（先頭、または FO のあと FO） | FO の前の連続した忙しい日の最初の日を始まりにする（前の trip より後）。先頭の FO で、忙しい日が表の開始日まで届くなら、表の開始日から始める |
| `FI | 作業A … | FO | 作業B … | FO` | 2 つ目の trip の始まりを **前の FO と同じ日付** にする（作業B の最初の日にはしない）。作業B が前の FO から `--chain-days`（既定 3 日、週末をまたげる）以内に始まる場合。離れていれば忙しい日の最初の日から始める |
| `FI | 作業A … | FI | 作業B … | FO` | 最初の trip の終わりを次の FI と同じ日付にする（同じく `--chain-days` 以内の場合） |
| 同じ印が 2 日以内に 2 つ（間に忙しい日が無い） | 1 つの印にまとめる（FI は先、FO は後） |
| FI と FO の間にタスクが無い、間隔が大きい | そのまま FI〜FO を 1 つの trip にする |

```
(a) FI→FI   10/20 FI … 作業 10/21〜11/18 … 11/25 FI … 作業 11/26〜12/23 … 12/30 FI（最後）
            → trip 10/20〜11/18、11/25〜12/23、12/30〜12/30
(b) FO だけ （忙しい日 05/02〜06/27）… 06/28 FO   → trip 05/02〜06/28
(c) 同じ日付 10/20 FI … 作業A 10/21〜10/29 … 10/30 FO … 作業B 10/31〜11/09 … 11/10 FO
            → trip 10/20〜10/30、trip 10/30〜11/10（2 つ目は前の FO の日から始まる）
```
trip の終わりと次の trip の始まりが同じ日付になることは、ソルバー側の検証でも許されている。補正した trip は推測であり、レポートに `INFERRED` として 1 件ずつ出る。`--no-fifo-repair` で補正をやめられる。

### 4.4 前週の trip との比較（Flexible に残すか、Fixed にするか）
比べるのは作業者・FI の日・FO の日の 3 つだけ（製番は見ない）。

| 前週の trip | 今の SU_Others | 結果 |
|---|---|---|
| Flexible | 同じ作業者に、同じ FI の日・FO の日の組がある | Flexible のまま。前週の trip をそのままコピーし、`FlexibleWork.yaml` の trips に出す |
| Flexible | 同じ組が無い（FI か FO が動いた、消えた） | Flexible から外す（レポートに `CHANGED flexible trip`）。シートの FI/FO は Fixed の trip になる |
| （前週に無い） | シートにある FI/FO の組 | Fixed の trip |

つまり `FlexibleWork.yaml` の trips は「前週のソルバー出力のうち、担当者が変更しなかった FI/FO」。trip が変更されても製番は Fixed にならない。

### 4.5 Fixed の trip の region
次の順で決める。(1) 前週に同じ trip があればその region。(2) FI と FO の間にある製番のバーの region（製番情報ファイルから。複数なら一番多いもの）。(3) バーのラベルに含まれる fab 名（3 文字以上で 1 つの region にだけ属するもの）。(4) 決まらなければ `r_other`（場所不明）。`--default-trip-region none` なら出力せずレポートに列挙する。
ソルバーは、同じ region の作業が中に無い Fixed trip を結果から消すことがある。週次運用では `allow_cut_fixed_trip: false` にする（ツール 2-6 の既定値）。

### 4.6 出力される trip
`01_Schedule.yaml` の `trip_list` に作業者ごとに出る。前週から継続した trip は `plan_flexibility: Flexible`、シートから作った trip は `Fixed`。

## 5. Schedule / EnvConfig の作り方
- Flexible な製番: workflow_task と作業割り当てを前週の FlexibleSchedule からそのままコピー（作業者 id だけ付け替え）。
- それ以外のバー: 同じ文字のセルを 1 つの misc タスクにまとめ、作業者・日付ごとの Fixed の割り当て（1 日 10 時間）にする。region は製番コードがあればその製番の region、無ければ `r_other`。灰色セルの個人予定は個人予定の misc、赤いセルは休み（unavailable）として EnvConfig に入れる。
- EnvConfig: workflow・fab・region・顧客・会社・移動日数は前週のものを引き継ぐ（既存の id は変えない）。新しい region / 顧客 / fab / 会社は末尾に追加する。`worker_list` は SU_Others から作り直す（前週にいた作業者は名前で対応付けて id を引き継ぐ）。スキルのファイルに居ない作業者は前週のスキルを引き継ぐ。

## 6. 実行方法
```
python prepare_schedule.py --prev-flexible <FlexibleSchedule.yaml> --prev-env <EnvConfig.yaml> --su-others <ファイル> --seiban-info <ファイル> --out-dir <フォルダ> --today YYYY/MM/DD [--skill-file <ファイル>] [--report]
```
Excel の読み込みに約 25 秒かかる。日本語のファイル名が文字化けする場合は `python -X utf8` で実行する。

## 7. レポートの見方
`MODULES:` Flexible のまま残った製番数と Fixed になった数 / `FIXED module …` Fixed の理由 / `TRIPS:` trip の内訳 / `CHANGED flexible trip` 動いた trip / `NOTE … INFERRED` 補正した trip / `WARN` 要確認。

## 8. 未決定・注意点
- 赤い FI/FO セルの意味、`移動` から trip を作るかどうかは未決定。
- 補正した FI/FO は推測。不要なら `--no-fifo-repair`。
- decoder7 の出力を前週の代わりに使うと、decoder7 が削ったセルのせいで一部の製番が「変更あり」になることがある。実運用（3-2 → 3-3 を通した後）では起こらない想定。
