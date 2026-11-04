# 2-2-3 prepare_schedule 説明書（週次計画 ステップ 2-2 + 2-3 の統合ツール）

フォルダ: `tools/weekly_planning/2-2-3prepare_schedule/`
英語版（オプション・ルール一覧）: [README.md](README.md)

このツールは、運用フロー「2-2 Fixedとする製番・FI/FOの特定」と「2-3 Flexible製番以外をmiscタスクとしたSchedule.yamlの生成」を1回の実行で行う。
特に **FI/FO をどう読み取り、どう補正し、前週の結果とどう比較するか** を第5章に詳しく書いた。

---

## 1. 何をするツールか（ひとことで）

- 前週の自動計画結果（FlexibleSchedule.yaml）と、計画担当者が編集した後の SU_Others.xlsm を比べる。
- **計画担当者が触っていない製番・FI/FO** は、そのまま自動計画の対象（Flexible）に残す。
- **触られた製番・FI/FO** と、それ以外のすべてのバーは「動かさないもの（Fixed / misc）」として Schedule.yaml に入れる。
- 自動計画（Timefold）にそのまま渡せる `01_EnvConfig.yaml` と `01_Schedule.yaml` を作る。

## 2. 入力と出力

| 区分 | ファイル | 内容 |
|---|---|---|
| 入力 | `(N-1)/FlexibleSchedule.yaml` | 前週の自動計画結果のうち、自動計画の対象だった製番・作業割り当て・trip（ツール 3-2 の出力）。Timefold の結果 `Schedule.yaml` でも読める |
| 入力 | `(N-1)/04_EnvConfig.yaml` | 前週の EnvConfig（workflow、fab、region、会社などの定義を引き継ぐ） |
| 入力 | `--su-others` | 計画担当者が更新する日付入り SU_Others（例 `20260726 SU_Others.xlsm`）。進行中の作業バー、FI/FO、作業者一覧 |
| 入力 | `--skill-file`（任意） | `SU_Others_skill level有配布禁止_.xlsm`。スキルレベルと Primary/Secondary（Regular/Spot）だけを読む |
| 入力 | `--seiban-info` | 初期データ追加情報.xlsx（製番シート）。製番ごとの顧客・fab・region |
| 出力 | `N/<today>_FlexibleWork.yaml` | 引き続き自動計画する **製番のリスト** と **trip（FI/FO）のリスト** |
| 出力 | `N/01_EnvConfig.yaml` | 次の工程（2-4 新規製番追加）に渡す EnvConfig |
| 出力 | `N/01_Schedule.yaml` | Flexible な製番だけが最適化対象、それ以外はすべて misc / Fixed の Schedule |
| 出力（`--report` 指定時） | `N/<today>_prepare_report.txt` | 判断の理由と、解決できなかった項目の全リスト |

`<today>` は `--today` で指定した日付（例 `20261106`）。

## 3. 全体の流れ

1. 前週の FlexibleSchedule.yaml を読む（製番ごとの割り当て、trip）。
2. SU_Others を読む（decoder7 と同じ読み方。バーの文字、色、FI/FO、作業者）。FI/FO は decoder7 が読み飛ばすので、このツールが自分で読む。
3. **2-2**: 製番ごとに「変更あり / 完了 / 変更なし」を判定する（第4章）。
4. **2-2**: trip ごとに「そのまま / 変更あり」を判定する（第5章）。
5. **2-3**: EnvConfig を作る（第7章）。
6. **2-3**: Flexible な製番の作業割り当てと Flexible な trip を前週からそのままコピーし、それ以外のバーを misc タスク、それ以外の FI/FO を Fixed の trip にして Schedule を作る（第6章）。

## 4. 製番の判定（Fixed か Flexible か）

**製番は「作業割り当て」だけで判定する。FI/FO（trip）の変更は製番の判定に一切影響しない。**
FI/FO が動いていても、その中の製番の作業が動いていなければ、その製番は Flexible のままである。

| 判定 | 条件 | 結果 |
|---|---|---|
| 変更あり | SU_Others 上の「その製番の（作業者, 日付）の集合」が、前週の割り当ての（作業者, 日付）の集合と一致しない（増えた・減った・日が動いた・作業者が変わった） | **Fixed**（以後 misc タスク） |
| 完了 | 前週の割り当ての最終作業日が `--today` より前 | **Fixed** |
| 変更なし | 上のどちらでもない | **Flexible**（`FlexibleWork.yaml` の modules に出す） |

- 製番は workflow_task の `name`（例 `530N03359A`）と、バーの文字に含まれる製番コードで対応付ける。
- 時間数（hour）は比較しない。日付と作業者だけ見る。
- 前週の FlexibleSchedule に無い製番は、常に misc タスクになる（新規製番の追加は 2-4 で計画担当者が行う）。
- decoder7 の例外: decoder7 は一部のセル（外れ値セル、データの少ない製番）を、製番名を付けた misc タスクに変換する。前週のファイルにそのような misc として既にあるセルは「変更」とみなさない（decoder7 の出力を前週データの代わりに使った時に誤検出しないため）。

## 5. FI/FO（trip）の扱い

### 5.1 FI/FO の読み取り

- 対象: `--su-sheets`（既定 `予定表_2026`）の作業者の行（企業名が空の行は無視）。
- セルの文字が次のいずれかなら FI / FO として読む。
  - 大文字小文字・全角・前後の空白は無視（`Fi`、`fo`、`ＦＩ`、` FO ` など）
  - 打ち間違い: `F1`、`Fl`、`F|`、`F!` は FI、`F0`（ゼロ）は FO
  - 末尾の括弧書きは無視（`FI(仮)` は FI、`FO（仮）` は FO）
- `移動` は FI/FO として読まない（第5.6節）。`F22`、`FC` など fab 名のようなものも読まない。
- 赤いセルの FI/FO も他と同じように読む（赤色の意味は未決定なので特別扱いしない）。
- 文字が FI/FO そのものでなかったセル（例 `FI(仮)`）は、レポートに「n cells with the text ... were read as FI」と出る。

### 5.2 基本の組み合わせ

作業者ごとに FI/FO を日付順に並べ、**FI と、その次に来る FO** を 1 つの trip にする。trip の期間は FI の日から FO の日まで（両端を含む）。

```
作業者A:  10/05 FI ─────────── 10/20 FO      → trip 10/05〜10/20
```

### 5.3 組み合わせにならない FI/FO の補正

実データには FI/FO の書き方の乱れが多い（2026-09-27 版で、先頭が FO の人が 97 人、FO の次が FO が 322 件、FI の次が FI が 103 件、最後が FI の人が 95 人）。
このツールは捨てずに、**シート上でその作業者が「忙しい日」** を手がかりに補正する。

- 「忙しい日」= その作業者のセルに何か文字が書かれている日（製番、その他の文字、個人予定のどれでもよい。FI/FO のセル自体は含めない）。
- 「連続した忙しい日」= 忙しい日同士の間隔が `--max-gap-days`（既定 **14 日**）以内でつながっている範囲。

| パターン | 補正のしかた |
|---|---|
| **FI のあと FO が無く、次が FI**（FI→FI） | 最初の FI から、次の FI の前までの連続した忙しい日の最終日を trip の終わりにする。忙しい日が無ければ FI の 1 日だけの trip |
| **最後の FI（FO が無いまま表が終わる）** | FI の後の連続した忙しい日の最終日を終わりにする。忙しい日が無ければ 1 日だけ。表の最終日まで続いていれば、その日が終わり |
| **FO だけ（先頭が FO、または FO のあと FO）** | FO の前の連続した忙しい日の最初の日を trip の始まりにする（前の FO/trip の終わりより後に限る）。忙しい日が無ければ FO の 1 日だけ。先頭の FO で、連続した忙しい日が表の開始日まで届く（開始日から 14 日以内）場合は、表の開始日から始まる trip とする（表が始まる前から出張中だったと見なす） |
| **ほぼ同じ日に同じ印が 2 つ**（FI と FI、FO と FO の間が 2 日以内） | 1 つの印にまとめる。FI は先のもの、FO は後のものを使う |
| **FI と FO の間にタスクが何も無い／間隔が大きい** | そのまま FI〜FO を 1 つの trip とする（補正しない） |
| 補正された trip の中で、忙しい日の途切れが 14 日を超える | そこで trip を打ち切る（補正のときだけ） |

補正の例（日付は仮）:

```
(a) FI→FI     10/20 FI … 忙しい日 10/21〜11/18 … 11/25 FI … 忙しい日 11/26〜12/23 … 12/30 FI（最後）
              → trip 10/20〜11/18、trip 11/25〜12/23、trip 12/30〜12/30

(b) FO だけ   ……（忙しい日 05/02〜06/27）… 06/28 FO
              → trip 05/02〜06/28

(c) 二重の印  05/01 FI、05/02 FI、……、05/22 FO、05/23 FO
              → trip 05/01〜05/23
```

- 補正した trip は **推測** である。レポートに `NOTE: FI/FO of <作業者>: INFERRED start/end ...` として 1 件ずつ出力される。
- 補正をやめたい場合は `--no-fifo-repair`（組み合わせにならない FI/FO は警告を出して捨てる）。補正の幅は `--max-gap-days` で変える。

### 5.4 前週の trip との比較（Flexible に残すか、Fixed にするか）

2-2 の trip の判定。比較に使うのは **作業者・FI の日・FO の日** の 3 つだけ（製番は見ない）。

| 前週の trip | 今の SU_Others | 結果 |
|---|---|---|
| Flexible（Timefold が出した trip） | 同じ作業者に、同じ FI の日・同じ FO の日の組がある | **Flexible のまま**。前週の trip をそのままコピー（region も前週のまま）。`FlexibleWork.yaml` の trips に出す |
| Flexible | 同じ組が無い（FI か FO が動いた、または消された） | **変更あり**。この trip は Flexible から外す（レポートに `CHANGED flexible trip` と出る）。シート上の FI/FO は Fixed の trip になる |
| （前週に無い） | シートにある FI/FO の組 | **Fixed の trip**（計画担当者が書いたもの） |

つまり `FlexibleWork.yaml` の trips は、**前週の Timefold の出力のうち、計画担当者が変更しなかった FI/FO** のリストである。
trip が変更されても、その中の製番は第4章のとおり作業が動いていなければ Flexible のまま（製番を Fixed にはしない）。

### 5.5 Fixed の trip の region（場所）の決め方

Fixed の trip には region が必要。次の順で決める。

1. 前週に同じ（作業者・FI・FO）の trip があれば、その region。
2. FI と FO の間（両端は含まない）に、その作業者の製番バーがあれば、その製番の region（初期データ追加情報の製番シートから）。複数あれば一番多いもの。
3. バーのラベルに fab 名（3 文字以上で、1 つの region にだけ属する名前）が含まれていれば、その fab の region。複数の region に当たる場合は決めない。
4. どれでも決まらなければ `r_other`（場所不明）。`--default-trip-region none` を指定すると、そのような trip は出力せず、レポートに列挙する。

`r_other` の trip は「出張中だが場所不明」を表す。EnvConfig の region_list に `r_other` が無ければ自動で追加される。
ただし Timefold は、trip の中に同じ region の作業が無いと Fixed trip を結果から消すことがある。週次運用では `SolverParams.yaml` に `allow_cut_fixed_trip: false` を設定することを推奨する。

### 5.6 `移動`

「移動」と書かれたセル（2026-09-27 版で 183 件）は FI/FO として扱わず、他のバーと同じく misc タスクにする。
183 件のうち 173 件は、一番近い FI/FO から 3 日より離れていた（出張の途中の移動と思われる）ため。**移動から trip を作る必要があるかは未決定**。

### 5.7 出力される trip

`01_Schedule.yaml` の `trip_list` に、作業者ごとに次の形で出る。

```yaml
trip_list:
- worker: w007
  trips:
  - start_date: 2026/10/19      # 前週から継続した Flexible trip
    end_date: 2026/10/23
    region: r1
    plan_flexibility: Flexible
    description: null
  - start_date: 2026/11/09      # シートの FI/FO から作った Fixed trip
    end_date: 2026/11/15
    region: r2
    plan_flexibility: Fixed
    description: null
```

## 6. Schedule.yaml の作り方（2-3）

- **Flexible な製番**: workflow_task と作業割り当てを前週の FlexibleSchedule からそのままコピー（作業者 id だけ今の EnvConfig の id に付け替える）。
- **それ以外のバー**: 同じ文字のセルを 1 つの misc タスクにまとめ、作業者・日付ごとに Fixed の割り当て（1 日 10 時間、色はセルの色）にする。
  misc タスクの region は、バーの文字に製番コードがあればその製番の region、無ければ `r_other`。
- **個人予定**（灰色セルで製番コードが無いもの）: `wf_personal_business` の misc タスクにする。
- **赤いセル**: その作業者の休み（unavailable date）として EnvConfig の worker_list に入れる（decoder7 と同じ）。
- **trip**: 第5章のとおり。Flexible な trip は前週のまま、それ以外は Fixed。

## 7. EnvConfig.yaml の作り方（2-3）

- workflow、fab、region、顧客、作業者の会社、移動日数（transite）は **前週の EnvConfig を引き継ぐ**。既存の id は変えない。
- 初期データ追加情報に新しい region / 顧客 / fab があれば、末尾に追加する（新しい id）。SU_Others にだけある会社も追加する。
- `worker_list` は SU_Others から作り直す。前週にいた作業者は名前（空白を除いた形）で対応付けて id を引き継ぎ、新しい作業者には次の空き番号（`w###`）を付ける。
  - 責任者フラグ、所属会社、休み: 日付入り SU_Others から
  - スキルレベル、Regular/Spot: スキルレベルのファイル（`--skill-file`）から。ファイルに居ない作業者は前週のスキルを引き継ぐ（レポートに一覧が出る）。それも無ければスキル無し
  - 前週の EnvConfig の workflow に無い作業（operation）のスキルは捨てる（レポートに出る）

## 8. 実行方法

英語版 README の Run を参照。例（PowerShell、このフォルダで実行。前週のファイルは `20261030\`、今週の出力先は `20261106\`）:

```powershell
python prepare_schedule.py --prev-flexible 20261030\FlexibleSchedule.yaml --prev-env 20261030\04_EnvConfig.yaml --su-others "..\..\decorder\20260726 SU_Others.xlsm" --skill-file "..\..\decorder\20260915 SU_Others_skill level有配布禁止_.xlsm" --seiban-info "..\..\decorder\初期データ追加情報 _20260930.rev1.xlsx" --out-dir 20261106 --today 2026/11/06 --report
```

- Excel の読み込みに約 25 秒かかる。
- 日本語のファイル名が文字化けする場合は `$env:PYTHONUTF8="1"; python -X utf8 prepare_schedule.py ...` で実行する。
- Timefold に渡すときは、出力を `Schedule.yaml` / `EnvConfig.yaml` の名前にし、2-4 → 2-5 を通してから実行する。入力が大きいので JVM のヒープは 12GB 程度が必要だった（6GB ではメモリ不足）。

## 9. レポートの見方

| 行の先頭 | 意味 |
|---|---|
| `MODULES:` | Flexible のまま残った製番数と、Fixed になった製番数 |
| `  FIXED module <製番>: ...` | Fixed にした理由（作業割り当てが変わった / 最終作業日が today より前） |
| `TRIPS:` | 変更のない Flexible trip 数、変更された Flexible trip 数、Fixed になった trip 数 |
| `  CHANGED flexible trip ...` | FI/FO が動いて Flexible から外れた前週の trip |
| `NOTE: FI/FO of <作業者>: INFERRED ...` | 補正（推測）した trip |
| `INFO: FI/FO of <作業者>: duplicate ...` | 二重の印を 1 つにまとめた |
| `NOTE trip ... region ... taken from a fab name` | region をバーのラベルの fab 名から決めた |
| `WARN: ...` | 要確認（region が決まらず捨てた trip、作業者が SU_Others に無い、など） |

## 10. 実データで試した結果

- 2026-09-27 版 SU_Others と、それから作った decoder7 の出力（前週の代わり）: 26 製番すべて Flexible のまま（Fixed 0）。trip は Fixed 2,271 件（うち補正 292 件、`r_other` 952 件）。
- 2026-07-26 版（日付入りファイル）+ スキルレベルファイルの組み合わせでも最後まで動作することを確認した。
- 未確認: 本物の FlexibleSchedule（ツール 3-2 の出力）を使った週次の通し運用。

## 11. 未決定・注意点

- 補正した FI/FO は推測。不要なら `--no-fifo-repair`、幅は `--max-gap-days`。
- 赤い FI/FO セルの意味、`移動` から trip を作るかどうかは未決定。
- decoder7 の出力を前週の代わりにすると、decoder7 が削ったセルのせいで一部の製番が「変更あり」になることがある。実運用では 3-3 が Excel に書いた結果と FlexibleSchedule が一致するので起こらない想定。
- 自動計画後の `allow_cut_fixed_trip` の設定（第5.5節）。
