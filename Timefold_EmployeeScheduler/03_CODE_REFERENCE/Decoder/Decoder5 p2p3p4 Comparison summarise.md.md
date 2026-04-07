# Decoder5 Summary
## Non-limit p3/p4 abnormal split cases

- Main target: summarize the abnormal split pattern found in the non-limit p3/p4 version
- Main issue: some modules become **p2 only 1 day**, while p3 and p4 become too large
- In this version, when QC-related timing is detected, the split can move to p3 while still leaving at least 1 day for p2
- Because p3/p4 are non-limit, the remaining worked days can continue flowing into late phases
- As a result, the phase balance can become very different from the planned shape

---

# Main behavior of non-limit p3/p4

- In non-limit p3/p4, once the split moves into phase 3, the remaining worked days can continue into p3 and p4 without hard upper bound
- This can preserve long real execution tails better
- However, it also makes some modules unbalanced
- Typical abnormal shape:
  - **p2 = 1 day**
  - **p3 and p4 become too long**
- This is the main abnormal pattern observed in this version

---

# Main abnormal groups

- The abnormal cases can be grouped into:
  - **1-A:** M/QC joined first
  - **1-B:** module looks like it starts from the middle
  - **1-C:** no QC worker
  - **1-D:** plan was cut by decoder logic
  - **1-E:** QC-only worker starts from the first date
- The largest focus is **1-A**, because this is the clearest case where p2 and p3 are separated unnaturally

---

# Case 1-A: M/QC joined first

- This is the biggest abnormal group
- In these modules, the first visible join is **M/QC**
- The result often looks like p3 started too early
- Because of that:
  - p2 is reduced to only 1 day
  - p3 and p4 take most of the worked timeline
- Example: **530900102A**
  - p2 = **1d**
  - p3 = **40d**
  - p4 = **39d**
- This is one of the clearest abnormal shapes in non-limit p3/p4 

---

# 1-A sub-patterns

- Inside 1-A, there are several patterns:
  - only **QC** appears later
  - **M/QC has a gap**
  - another **M/QC worker joins later**
  - no clear QC sign or M/QC gap
- Summary count in the note:
  - only QC appears later: **22 cases**
  - M/QC assignment has a gap: **5 cases**
  - another M/QC worker joins later: **5 cases**
  - no clear sign: **1 case**
- This means the issue is not only whether QC exists
- It is also related to how **M/QC timing** is interpreted in the phase split 

---

# Case 1-B: starts from the middle

- Some modules look like the visible work starts from the middle
- The earlier part is missing, so the first half of the actual flow cannot be seen
- In this type, p2 can still collapse to 1 day
- Then p3 and p4 occupy most of the remaining visible timeline
- Example: **830400201A**
  - p2 = **1d**
  - p3 = **22d**
  - p4 = **22d**
- This case shows that missing early context can also distort the split 

---

# Case 1-C / 1-D / 1-E

- **1-C: no QC worker**
  - abnormal split still appears even when no QC worker participates
  - example: **530L00935A**
- **1-D: plan was cut**
  - part of the planned timeline was removed by cutting logic
  - this can distort the remaining phase shape
  - example: **530N02888A**
- **1-E: really starts with QC**
  - QC-only worker joins from the first date
  - p3 then starts very early in the visible timeline
  - example: **530N02635A**
- These cases show that the abnormality is caused by both split timing and timeline condition 

---

# Why p2 becomes only 1 day

- The split is based on **compressed worked days**
- Once the p3 side starts too early, only the minimum p2 portion remains
- Then the remaining worked days continue into p3 and p4
- This makes the phase balance unstable in cases such as:
  - M/QC appears first
  - QC appears later
  - M/QC has a gap and rejoins
  - early part of module is missing
  - no QC exists and fallback behavior still creates late-heavy split
- As a result, p2 becomes too short compared with the whole task

---

# Main interpretation of the abnormality

- The core issue is not simply “QC exists”
- The core issue is **how to decide the p2/p3 separation point**
- In particular, modules with **M/QC at the beginning** can look like QC joined later-stage work too early
- This causes p3/p4 to start too soon on the compressed worked timeline
- That is why the main abnormal shape becomes:
  - short p2
  - long p3/p4

---

# Direction to prevent the abnormal split

- To reduce this abnormality, the p2/p3 separation should not rely on QC text alone
- The split should distinguish:
  - **QC without M**
  - and
  - **M/QC**
- Main idea:
  - **No only QC but have space on M/QC**
- This means that p2 and p3 should not be separated only because M/QC appears
- M/QC timing should be checked more carefully

---

# Suggested prevention approach

- Suggested order of interpretation:
  1. first check whether there is **QC without M**
  2. if it exists, use that join timing as the main signal for p3 start
  3. if there is no QC-without-M, then check **M/QC**
  4. when **M/QC has a gap**, use the rejoin point to separate p2 and p3
  5. otherwise use fallback logic
- This approach can better handle modules where:
  - M/QC joins first
  - QC appears later
  - M/QC leaves and rejoins
- It is expected to reduce p2 = 1 day abnormal cases

---

# Meaning of “No only QC but have space on M/QC”

- This phrase means:
  - not only looking for QC existence
  - but also checking whether **M/QC has a space / gap**
- If the module has pure QC participation, that can be treated as the stronger sign of p3
- If pure QC does not exist, M/QC can still help separate p2 and p3
- But in that case, the important point is not the first join
- The important point is whether there is a **gap and rejoin**
- This is the main direction to prevent 1-A type abnormality

---

# Example explanation for slide

- Abnormal pattern:
  - p2 becomes only 1 day
  - p3 and p4 dominate the module
- Typical type:
  - **1-A: M/QC joined first**
- Example module:
  - **530900102A**
- Interpretation:
  - p3 is separated too early in the visible worked timeline
- Prevention direction:
  - not only QC
  - also check whether M/QC has a gap before separating p2 and p3

---

# Summary of non-limit p3/p4

- Non-limit p3/p4 can preserve long real execution tails
- But it also creates the clearest abnormal shapes
- Main abnormal result:
  - **p2 collapses to 1 day**
  - **p3/p4 become too dominant**
- Main abnormal groups:
  - 1-A M/QC joined first
  - 1-B start from middle
  - 1-C no QC worker
  - 1-D plan cut
  - 1-E really starts with QC
- Main improvement direction:
  - separate **QC without M** from **M/QC**
  - use **M/QC gap / rejoin** to help separate p2 and p3
- This is the key idea to reduce the abnormal split in non-limit p3/p4

---

# Other version: 28-day limit p3+p4

- Another possible approach is the **28-day limit** version
- This version can also reduce part of the same abnormality
- Main effect:
  - p3+p4 growth is controlled
  - tail shape becomes easier to read
- But it creates another risk:
  - **p2 becomes too large**
  - p3/p4 become too standardized
- So this version can control the shape
- However, improving the p2/p3 split condition is still important for handling the root abnormal pattern 


---

# Decoder5結果まとめ
## non-limit p3/p4 における工程分割の異常パターン

- 対象は **non-limit p3/p4版** で見られた工程分割の異常傾向
- 主な問題は、**p2が1日だけになり、p3・p4が過大になる** モジュールがあること
- この版では、QCに関する参加タイミングを起点に p3 側へ分割が進みつつ、p2には最低1日を残す形になる
- さらに p3/p4 に上限がないため、残りの実績日が後工程側へ流れやすい
- その結果、計画上の工程バランスと大きく異なる形になる場合がある

---

# non-limit p3/p4 の基本的な傾向

- non-limit p3/p4 では、一度 p3 側に入ると、その後の実績日が p3・p4 に広く配分されやすい
- 実際の長い後半工程をそのまま反映しやすい点は利点である
- 一方で、工程の切れ目が早く判定されると、全体のバランスが崩れやすい
- 代表的な異常形は以下の通り
  - **p2 = 1日**
  - **p3 / p4 が長くなりすぎる**
- 今回の non-limit 版では、この形が最も目立つ異常であった

---

# 主な異常パターンの分類

- 異常ケースは主に以下に分類できる
  - **1-A：最初に M/QC が参加しているケース**
  - **1-B：途中から始まったように見えるケース**
  - **1-C：QC作業者が存在しないケース**
  - **1-D：Decoderのcut処理の影響を受けたケース**
  - **1-E：開始直後からQC専任作業者が入っているケース**
- この中でも、特に件数・影響ともに大きいのが **1-A** である

---

# 1-A：最初に M/QC が参加しているケース

- 最も多く見られた異常パターン
- このケースでは、最初に見える参加者が **M/QC** であることが多い
- その結果、p3 の開始が早すぎるような分割になりやすい
- その影響として
  - p2 が **1日まで縮む**
  - p3・p4 が実績期間の大部分を占める
- 例：**530900102A**
  - p2 = **1d**
  - p3 = **40d**
  - p4 = **39d**
- non-limit p3/p4 における代表的な異常形のひとつである

---

# 1-A の内訳

- 1-A の中にも、さらにいくつかのパターンがある
  - **後から pure QC が現れる**
  - **M/QC に空き期間がある**
  - **別の M/QC 作業者が後から参加する**
  - **QC や M/QC の明確な切れ目が見えない**
- 集計結果は以下の通り
  - 後から QC が現れる：**22件**
  - M/QC に gap がある：**5件**
  - 別の M/QC が後から参加：**5件**
  - 明確な兆候なし：**1件**
- つまり、問題は単に「QC がいるかどうか」ではなく、
  **M/QC の参加タイミングをどう工程分割に使うか** にある

---

# 1-B：途中から始まったように見えるケース

- 実績の見え方として、作業が途中から始まったように見えるケース
- 前半の流れが見えていないため、実際の工程開始位置を正しく捉えにくい
- このタイプでも p2 が 1 日まで縮むことがある
- その結果、残りの期間の多くが p3・p4 に割り当てられる
- 例：**830400201A**
  - p2 = **1d**
  - p3 = **22d**
  - p4 = **22d**
- 早期分割だけでなく、前半情報の欠落も異常の一因となる

---

# 1-C / 1-D / 1-E のケース

- **1-C：QC作業者がいないケース**
  - QCが存在しなくても異常な分割になる場合がある
  - 例：**530L00935A**
- **1-D：計画が cut されたケース**
  - Decoder の cut 処理により、予定範囲の一部が落ちている
  - その結果、残った期間だけを見ると工程形状が不自然になる
  - 例：**530N02888A**
- **1-E：開始時点からQC専任が入っているケース**
  - 開始直後から QC 専任作業者が参加しているため、p3 側への切り替えが早くなる
  - 例：**530N02635A**
- これらのケースから、異常は工程分割条件だけでなく、タイムライン自体の状態にも影響されることが分かる

---

# なぜ p2 が 1 日だけになるのか

- 分割は **compressed worked days** を基準に行われている
- そのため、p3 側への切り替え位置が少しでも早いと、p2 に残せる期間が最小限になる
- その後の実績日は p3・p4 側に流れていくため、後半工程が大きくなりやすい
- 特に以下のような条件で不安定になりやすい
  - 最初に M/QC が現れる
  - QC が後から現れる
  - M/QC に gap があって再参加する
  - 前半の実績が欠けている
  - QC がいないまま fallback 的に後工程寄りになる
- 結果として、p2 が全体に対して極端に短くなる

---

# 異常の本質的な見方

- 問題の本質は、単純に「QC がいる」ことではない
- 本質は、**どこを p2 / p3 の分割点として扱うか** にある
- 特に、冒頭から **M/QC** が入っているケースでは、
  後工程の開始として早く扱われすぎる場合がある
- その結果、compressed worked days 上で p3/p4 が早く始まり、
  典型的な
  - p2 が短い
  - p3/p4 が長い
 という異常形になる

---

# 異常を防ぐための考え方

- この異常を減らすためには、p2 / p3 の分割を **QC の文字だけ** で判断しないことが重要
- 少なくとも以下を分けて考える必要がある
  - **M を含まない QC**
  - **M/QC**
- 基本的な考え方は
  - **No only QC but have space on M/QC**
- つまり、M/QC が見えたからすぐ p3 に切るのではなく、
  **M/QC の出方や gap も含めて判断する** 必要がある

---

# 分割改善の方向性

- 分割判定の優先度としては、以下のような考え方が有効と考えられる
  1. まず **M を含まない QC** が存在するかを見る
  2. 存在する場合、その参加タイミングを p3 開始の主な候補とする
  3. 存在しない場合は **M/QC** を確認する
  4. **M/QC に gap がある場合** は、その再参加位置を p2 / p3 の分割候補とする
  5. それでも判断できない場合に fallback を使う
- この考え方により、
  - 最初に M/QC が入るケース
  - 後から QC が現れるケース
  - M/QC が抜けて再び入るケース
  に対応しやすくなる
- p2 = 1 日の異常ケースの削減が期待できる

---

# “No only QC but have space on M/QC” の意味

- この言葉が表しているのは、
  **QC の有無だけでなく、M/QC に空きがあるかも見る** という考え方である
- pure QC が存在する場合は、それを p3 のより強い兆候として扱える
- pure QC が存在しない場合でも、M/QC は分割のヒントにはなる
- ただし重要なのは「最初に入ったかどうか」ではなく、
  **gap があって再度入ったかどうか** である
- この考え方が、1-A 型の異常を抑える主な方向性になる

---

# スライド用の説明例

- 異常パターン：
  - p2 が 1 日だけになる
  - p3 / p4 が工程の大半を占める
- 代表例：
  - **1-A：最初に M/QC が参加しているケース**
- 例モジュール：
  - **530900102A**
- 解釈：
  - 実績上の工程分割位置が早すぎるため、p3 側に寄りすぎている
- 改善の方向：
  - QC だけで判断しない
  - M/QC に gap があるかも見て p2 / p3 を分ける

---

# non-limit p3/p4 のまとめ

- non-limit p3/p4 は、実際の長い後半工程を反映しやすい
- 一方で、工程分割の異常が最も見えやすい版でもある
- 主な異常結果は
  - **p2 が 1 日まで縮む**
  - **p3/p4 が過大になる**
- 主な異常グループは
  - 1-A 最初に M/QC が参加
  - 1-B 途中から始まったように見える
  - 1-C QC作業者なし
  - 1-D plan cut
  - 1-E 開始直後からQC専任が参加
- 主な改善方向は
  - **QC without M** と **M/QC** を分けて扱うこと
  - **M/QC の gap / 再参加** を p2 / p3 分割に活用すること
- これが non-limit p3/p4 の異常分割を減らすための重要な考え方である

---

# 参考：28-day limit p3+p4 版

- 別の対応案として、**28-day limit 版** も考えられる
- この版では p3+p4 の伸びを抑えやすく、後半工程の形が見やすくなる
- ただし、その代わりに
  - **p2 が大きくなりすぎる**
  - p3/p4 が均一化されすぎる
  といった別の偏りも出やすい
- そのため、形を抑える方法としては有効だが、
  根本的には p2 / p3 の分割条件そのものの見直しが重要である