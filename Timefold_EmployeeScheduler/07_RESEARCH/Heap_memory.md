## 1. What was the heap memory problem?

When we ran big scenarios (many modules → many CrewSeat rows), the JVM heap kept growing until:

- Timefold threw **`OutOfMemoryError: Java heap space`**, or
    
- GC was running all the time and the solve became extremely slow.
    

This happened **even though** the rest of the model (blocks, calendars, etc.) was reasonable. The main cause turned out to be two soft constraints:

- `softSameCompanyPairs`
    
- `softEncourageSkillVariety`
    

in their **old “pairwise join” form**.

---

## 2. Why did it happen? (old constraints)

### 2.1 Old pattern = “All pairs” inside each block

Both old constraints used this pattern:

`return f.forEach(CrewSeat.class)     .filter(a -> !isUnassigned(a.employee))     .join(f.forEach(CrewSeat.class),           Joiners.equal((CrewSeat a) -> a.blockId, (CrewSeat b) -> b.blockId)           // sometimes also same opId     )     .filter((a, b) -> !isUnassigned(b.employee) && a.id < b.id)     ...`

This means:

- For each block, Timefold builds **all pairs of seats in that block**  
    (or all pairs of seats with same block + same op).
    
- If a block has `N` seats, the number of pairs is roughly  
    **N × (N – 1) / 2 = O(N²)**.
    

With large cases (hundreds of modules, many heads per operation), `N` can be **thousands of seats per block**, so:

- Pairs per block = millions
    
- Across all blocks = **tens of millions of tuples**
    

Timefold has to:

- Keep these tuples in working memory,
    
- Track their matches for the incremental score calculation,
    
- Optionally keep constraint-match data (if enabled).
    

That’s exactly the kind of pattern that eats heap memory very quickly.

### 2.2 Both constraints had the same pattern

Both old constraints were quadratic:

1. **Same company pairs**
    
    - Builds all `(seat A, seat B)` in the same block
        
    - Filters by `company(A) == company(B)`
        
    - Rewards each pair
        
2. **Same skill level pairs**
    
    - Builds all `(seat A, seat B)` in the same block & op
        
    - Filters by `skill(A, op) == skill(B, op)`
        
    - Penalizes each pair
        

Even if each pair is “just a tuple”, millions of them in the score director results in:

- Huge heap usage
    
- GC pressure
    
- Eventually `OutOfMemoryError` in bigger test runs.
    

---

## 3. Why did we change the constraints?

Goal:

- Keep the **same logical meaning** (favor same company, penalize lack of skill variety),
    
- But avoid the **O(N²) pair explosion**.
    

Instead of iterating over every pair, we asked:

> “Can we derive the total number of pairs from a **count**, without materializing each pair?”

For example:

- If there are `count = 10` seats from the same company in a block,  
    the number of unordered pairs is **C(10,2) = 10×9/2 = 45**.
    

We don’t need to see each `(A, B)` individually — we can compute the total contribution from just the count.

So we rewrote both constraints to use **`groupBy + count`** (linear) instead of `forEach + join` (quadratic).

---

## 4. Why does the new version work better?

### 4.1 New `softSameCompanyPairs`

`Constraint softSameCompanyPairs(ConstraintFactory f) {     return f.forEach(CrewSeat.class)         .filter(s -> !isUnassigned(s.employee))         .filter(s -> !company(s.employee).isEmpty())         .groupBy(             // key: [blockId, company]             s -> Arrays.asList(s.blockId, company(s.employee)),             ConstraintCollectors.count()         )         .filter((key, count) -> count > 1)         .reward(HardMediumSoftScore.ONE_SOFT,             (key, count) -> COMPANY_PAIR_W * (count * (count - 1) / 2)         )         .asConstraint("soft-same-company-pairs"); }`

Now Timefold:

- Doesn’t build `(a, b)` pairs.
    
- Just groups seats by `(blockId, company)` and gets a **single row per group**:  
    `[block, company] -> count`.
    
- Computes the number of pairs as `count * (count - 1) / 2`.
    

**Complexity:**

- Seats per block: `N`
    
- Work done: **O(N)** (grouping + counting)
    
- Memory usage: number of groups is at most `(#blocks × #companies)`, which is tiny compared to `N²`.
    

### 4.2 New `softEncourageSkillVariety`

`Constraint softEncourageSkillVariety(ConstraintFactory f) {     return f.forEach(CrewSeat.class)         .filter(s -> !isUnassigned(s.employee))         .groupBy(             // key: [blockId, opId, skillLevel]             s -> Arrays.asList(s.blockId, s.opId, skill(s.employee, s.opId)),             ConstraintCollectors.count()         )         .filter((key, count) -> count > 1)         .penalize(HardMediumSoftScore.ONE_SOFT,             (key, count) -> SKILL_DIVERSITY_W * (count * (count - 1) / 2)         )         .asConstraint("soft-encourage-skill-variety"); }`

Same idea:

- Group by `(block, op, skillLevel)`,
    
- Count seats in each group,
    
- Convert `count` to “number of homogeneous pairs” `count * (count - 1) / 2`,
    
- Penalize large homogeneous groups.
    

Again this avoids generating all `(a, b)` pairs and uses only aggregated counts.

### 4.3 Effect on heap

With the new implementation:

- The solver no longer has to track **millions of pair tuples**.
    
- Working memory size stays close to **O(#seats)** instead of O(#seats²).
    
- GC pressure and peak heap usage drop a lot.
    
- Large runs that used to crash (or slow to a crawl) can now finish.
    

---

## 5. Short summary

- The heap memory issue came from two soft constraints that used **pairwise joins** on `CrewSeat`, generating **O(N²)** seat pairs per block.
    
- With many seats, this created **millions of tuples**, blowing up Timefold’s working memory and causing `OutOfMemoryError` / heavy GC.
    
- We rewrote both constraints using **`groupBy + count`**, and computed “pairs” mathematically as `count * (count - 1) / 2` instead of iterating over every pair.
    
- The new version keeps the same scheduling behavior (reward same-company groups, penalize lack of skill variety) but reduces complexity to **O(N)** and keeps heap usage under control.


## 1. 何が起きていたか（ヒープメモリ問題）

大きいケース（モジュール数・CrewSeat数が多いスケジュール）を解かせると、

- JVM がどんどんメモリを消費して  
    **`OutOfMemoryError: Java heap space`** が出たり
    
- GC が暴走して計算が極端に遅くなる
    

という現象が発生していました。

原因は、スコア計算の中でもとくにソフト制約：

- `softSameCompanyPairs`
    
- `softEncourageSkillVariety`
    

の **旧実装（ペアごとの join）** にありました。

---

## 2. なぜ発生していたのか（旧実装の問題）

### 2.1 「同じブロック内の全ペア」をつくっていた

旧実装はどちらも、こんな形でした：

`return f.forEach(CrewSeat.class)     .filter(a -> !isUnassigned(a.employee))     .join(f.forEach(CrewSeat.class),           Joiners.equal((CrewSeat a) -> a.blockId, (CrewSeat b) -> b.blockId)           // （opId も揃える場合もあり）     )     .filter((a, b) -> !isUnassigned(b.employee) && a.id < b.id)     ...`

つまり、

- ブロック内の **全ての `(Seat A, Seat B)` のペア** を作ってから、
    
- 「同じ会社か？」「同じスキルか？」をチェックしてスコアに反映
    

という作りになっていました。

ブロック内に `N` 個の CrewSeat があると、  
ペアの数はだいたい：

> `N × (N − 1) / 2`（O(N²)）

になります。

大規模ケースでは 1 ブロックあたりの Seat 数が **数千** になることもあり、  
そうするとブロックごとに **数百万ペア**、  
全体では **数千万のタプル** を Timefold が抱えることになります。

Timefold はこれらのタプルを

- ワーキングメモリに保持し、
    
- インクリメンタルスコアのために更新を追いかける
    

必要があるので、結果として **ヒープを食い尽くす** 形になっていました。

### 2.2 両方の制約が二乗オーダー

旧実装はどちらも二乗オーダーでした：

1. **`softSameCompanyPairs`（同一会社ペアのリワード）**
    
    - 同じ blockId の Seat 同士の全ペアを生成
        
    - `company(A) == company(B)` を満たすペアに報酬
        
2. **`softEncourageSkillVariety`（スキルが同じペアにペナルティ）**
    
    - 同じ blockId & opId の Seat 同士の全ペアを生成
        
    - `skill(A, op) == skill(B, op)` を満たすペアにペナルティ
        

どちらも **ペアをすべて列挙してから数える** 方式だったので、  
Seat 数が増えると急激にメモリ消費が跳ね上がっていました。

---

## 3. なぜ書き換えたのか

やりたいこと自体はシンプルで、

- 同じブロック・同じ会社の人数に応じて「ペア数分」リワードしたい
    
- 同じブロック・同じ工程・同じスキルレベルの人数に応じて「ペア数分」ペナルティしたい
    

だけです。

そこで、

> 「ペアを全列挙しなくても、**人数のカウントからペア数を計算できないか？**」

という発想に切り替えました。

人数 `count` がわかれば、同じグループ内の **組み合わせペア数** は：

> `count * (count - 1) / 2`

で計算できます。  
これを使うことで、

- `forEach + join + (a,b)` ではなく
    
- `groupBy + count` だけで、同じロジックを O(N) で表現する
    

ように変更しました。

---

## 4. 新しい実装で何が変わったか

### 4.1 新 `softSameCompanyPairs`

`Constraint softSameCompanyPairs(ConstraintFactory f) {     return f.forEach(CrewSeat.class)         .filter(s -> !isUnassigned(s.employee))         .filter(s -> !company(s.employee).isEmpty())         .groupBy(             // key: [blockId, company]             s -> Arrays.asList(s.blockId, company(s.employee)),             ConstraintCollectors.count()         )         .filter((key, count) -> count > 1)         .reward(HardMediumSoftScore.ONE_SOFT,             (key, count) -> COMPANY_PAIR_W * (count * (count - 1) / 2)         )         .asConstraint("soft-same-company-pairs"); }`

Timefold の立場から見ると：

- もう `(seat A, seat B)` の全ペアは作っていない
    
- `(blockId, company)` ごとに **一行だけ** まとめて、
    
    - そこに属する Seat 数 `count` を数える
        
- `count * (count - 1) / 2` で「ペア数」相当を計算してリワード
    

→ 作業量は

- もともと `O(N²)`（全ペア）だったのが
    
- 今は `O(N)`（Seat を 1 回なめて groupBy + count）
    

になっていて、  
メモリも **「グループ数」分** しか使いません  
（≈ ブロック数 × 会社数 程度で、Seat 数よりだいぶ少ない）。

### 4.2 新 `softEncourageSkillVariety`

`Constraint softEncourageSkillVariety(ConstraintFactory f) {     return f.forEach(CrewSeat.class)         .filter(s -> !isUnassigned(s.employee))         .groupBy(             // key: [blockId, opId, skillLevel]             s -> Arrays.asList(s.blockId, s.opId, skill(s.employee, s.opId)),             ConstraintCollectors.count()         )         .filter((key, count) -> count > 1)         .penalize(HardMediumSoftScore.ONE_SOFT,             (key, count) -> SKILL_DIVERSITY_W * (count * (count - 1) / 2)         )         .asConstraint("soft-encourage-skill-variety"); }`

こちらも同じ考え方で：

- `(block, op, skillLevel)` でグループを作り、
    
- 各グループの人数 `count` から
    
    - 「同じスキル同士のペア数」= `count * (count - 1) / 2` を計算し、
        
- その分だけペナルティをかける
    

という形に変更しています。

### 4.3 ヒープへの効果

新実装では：

- もはや **座席ペアのタプル（a,b）** を大量に持たない
    
- スコア計算で持つデータは
    
    - CrewSeat 自体
        
    - 少数の groupBy 結果（キー + count）のみ
        

なので、

- Timefold のワーキングメモリサイズは **O(#seats)** 近くに収まり、
    
- 大規模ケースでもヒープ使用量が安定
    
- `OutOfMemoryError` や GC 暴走が発生しにくくなりました。
    

---

## 5. まとめ

- ヒープメモリ問題の原因は、
    
    - `softSameCompanyPairs` / `softEncourageSkillVariety` が  
        **「ブロック内の CrewSeat 全ペア」を生成する二乗オーダーの join** だったこと。
        
- 大きなケースでは、ブロックごとに **数百万ペア** が発生し、  
    Timefold のワーキングメモリが膨れ上がって `OutOfMemoryError` や極端な遅さを引き起こしていた。
    
- 対策として、
    
    - `groupBy + count` で人数を集計し、
        
    - `count * (count - 1) / 2` でペア数を「数式」で表現するように変更。
        
- これにより、
    
    - 制約の意味（同一会社をまとめる・スキルの偏りにペナルティ）はそのまま、
        
    - 計算量は **O(N² → O(N))** に改善され、
        
    - ヒープ使用量と GC 負荷が大きく下がり、大規模の連続計画でも安定して解けるようになった