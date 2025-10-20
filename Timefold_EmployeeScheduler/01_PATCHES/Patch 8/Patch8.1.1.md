## 🎯 Overview

Patch 8.1.1 is a **luxury, two-pass scheduler**—same planning logic as 7.4.2, now **fully ported to Java (Timefold)** with a clean **Maven** layout.  
Pass 1 builds work **blocks** (start day, heads, days) while _auto-deriving hours_ from each operation’s allowed list; Pass 2 assigns **one employee per seat** with hard safety rules and soft balancing.

Goal: keep the proven Patch 7 behavior, but deliver **JVM performance, stronger typing, and enterprise toolchain** readiness.

---

## 🧭 Why this change

- Preserve the successful **2-pass design** from Patch 7.
    
- Move from Python to **Java + Timefold** for speed, reliability, and CI/CD.
    
- Keep inputs/outputs identical: **EnvConfig.yaml** + **Schedule.yaml** in, solved assignments out—so the existing Excel exporter continues to work.
    

---

## 🏗️ Architecture (2-Pass Planning)

### Pass 1 — “Create Blocks”

- Decision per task window: **startDay, heads, days**
    
- **Hours are auto-derived** from the operation’s allowed list to fulfill required workload with minimal overfill.
    
- Includes a **tiered hours ramp**: start with the smallest allowed hours; widen only if needed.
    
- Hard rules: window bounds, phase order, min/max heads, no underfill, limited overfill, daily capacity by op.
    
- Soft preferences: lean toward **8h/day**, **fewer heads**, **fewer days**, **earlier start**.
    

### Pass 2 — “Assign People”

- Expand blocks → **seats** and **seat-days**, then assign **one employee per seat**.
    
- Hard rules: eligible skill, **12h/day cap**, **one factory per person per day**, **≥1 manager per block**.
    
- Soft goals: encourage **same-company cohesion**, **skill variety**, keep block **avg skill near org average**, and **balance total hours** across employees.
    

---

## 🚀 Key Additions (vs. 7.4.2)

- **Full Java port** using Timefold constraint streams; no behavior change by design.
    
- **Hours ramp in Pass 1** kept and formalized in Java, with auto-hours selection that limits overfill to at most one extra day.
    
- **Domain types** with explicit IDs and value ranges: DaySlot, BlockDecision, CrewSeat, SeatDay, EmployeeFact.
    
- **Solver builders** with termination controls and optional polish when **0 hard** is reached.
    

---

## 📦 Project Layout (Maven)

- `apps/v812/src/main/java/com/yourorg/scheduler/EmployeeSchedule.java`  
    Core domain, parsers, Pass 1 & Pass 2 constraints, solver wiring, CLI.
    
- `apps/v812/src/main/java/com/yourorg/scheduler/ExportSchedule.java`  
    Writes assignments back to Schedule.yaml (compatible with your Excel exporter).
    
- `apps/v812/src/main/resources`  
    EnvConfig.yaml / Schedule.yaml samples for local runs.
    
- `pom.xml`  
    Timefold, SnakeYAML, build settings.
    

---

## 📊 Behavior Summary

|Item|Result|
|---|---|
|Inputs/Outputs|Unchanged (EnvConfig.yaml, Schedule.yaml)|
|Pass 1|Tiered hours ramp; auto-hours selection; strict hard rules|
|Pass 2|Skill, 12h cap, single-factory/day, ≥1 manager enforced|
|Scores|Target 0 hard, polish step for softer improvement|
|Excel|Existing exporter works without modification|

---

## 🔐 Compatibility & Migration

- Drop-in replacement for Patch 7 pipelines: same YAML contracts.
    
- Excel dashboard and breach sheets remain valid.
    
- No code changes required outside of switching the runtime to the **Maven Java app**.
    

---

## ⚠️ Risks & Mitigations

- JVM vs Python differences → covered by **identical constraint logic** and **score limits**.
    
- Overfill/underfill sensitivity → **auto-hours** and **at-most-one-extra-day** rule reduce oscillation.
    
- Capacity by operation → enforced as **hard** in Pass 1 to avoid infeasible Pass 2.
    

---

## 🔮 Next Steps

- Add Pass 1 “produced vs required” summary table to the Dashboard sheet.
    
- Optional: expose per-day OP capacity knobs from EnvConfig for finer throttling.
    
- Add a CLI flag to export intermediate Pass 1 ramp tiers for diagnostics.
    

---

## 🧩 What you get

- The **same 2-pass plan quality** you trust from Patch 7,
    
- Now with a **first-class Java/Timefold** engine,
    
- And a **dimensional**, analytics-ready output flow that plugs straight into your existing Excel dashboards.


## Patch 7.4.2 (Python) vs Patch 8.1.1 (Java/Timefold)

### Performance on the same dataset

|Stage|7.4.2 (Python)|8.1.1 (Java)|
|---|---|---|
|Pass 1 – reach 0 hard|30 min|2 min|
|Pass 1 – polish|30 min|20 min|
|Pass 2 – reach 0 hard|10 min|1 min|
|Pass 2 – polish|20 min|10 min|
|**Total**|**≈ 90 min**|**≈ 30–33 min***|

* Sum of components is ~33 min; we round to ~30 min to reflect typical wall-clock runs.

### Solution quality (soft objectives)

|Aspect|7.4.2 (Python)|8.1.1 (Java)|
|---|---|---|
|Workload balance across employees|Good|Very good (tighter variance after polish)|
|Prefer same group/company (cohesion)|Good|Very good (pair rewards converge faster)|
|Earlier start / fewer heads / fewer days|Good|Very good (priorities hold under ramping)|
|Block average skill near org average|Good|Very good (smaller residual gap)|

### Functional parity

|Dimension|7.4.2 (Python)|8.1.1 (Java)|
|---|---|---|
|Two-pass design (blocks → seats)|Yes|Yes (identical logic)|
|Inputs/outputs (EnvConfig.yaml / Schedule.yaml)|Same|Same|
|Excel exporter compatibility|Works|Works (no change)|
|Constraint set (hard + soft)|Same|Same (constraint streams)|

### Engineering differences

|Topic|7.4.2 (Python)|8.1.1 (Java)|
|---|---|---|
|Runtime/Toolchain|Python + custom solver|Java + Timefold, Maven project|
|Typing & models|Dynamic|Strongly typed entities (DaySlot, BlockDecision, CrewSeat, SeatDay, EmployeeFact)|
|Hours ramp in Pass 1|Implemented|Implemented and formalized; same behavior|
|Termination & polish|Supported|Supported, with clear best-score limits|

### Takeaways

- Java/Timefold keeps the exact 2-pass behavior but **cuts runtime by ~3×** on this dataset.
    
- Soft-score outcomes (workload balance, same-company cohesion, earlier starts) are **consistently better** after polish in 8.1.1.
    
- No migration friction: **same YAML contracts**, same Excel dashboards, just run the Maven app.


## Patch 7.4.2（Python） vs Patch 8.1.1（Java/Timefold）

### 同一データセットでの性能

|ステージ|7.4.2（Python）|8.1.1（Java）|
|---|---|---|
|Pass 1 – 0 hard 到達|30 min|2 min|
|Pass 1 – polish|30 min|20 min|
|Pass 2 – 0 hard 到達|10 min|1 min|
|Pass 2 – polish|20 min|10 min|
|**合計**|**≈ 90 min**|**≈ 30–33 min***|

* 各工程の合計は約 33 分ですが、実運用の壁時計時間では約 30 分に丸めています。

### 解の質（ソフト目的）

|観点|7.4.2（Python）|8.1.1（Java）|
|---|---|---|
|従業員間のワークロード平準化|良|とても良（polish 後の分散がより小さい）|
|同一グループ／同一会社の優先（結束）|良|とても良（ペア報酬の収束が速い）|
|早期開始／少人数／短日数の傾向|良|とても良（時間拡張中も優先度を維持）|
|ブロック平均スキルの組織平均への近さ|良|とても良（残差がより小さい）|

### 機能パリティ

|項目|7.4.2（Python）|8.1.1（Java）|
|---|---|---|
|2 パス設計（ブロック → シート）|あり|あり（同一ロジック）|
|入出力（EnvConfig.yaml / Schedule.yaml）|同じ|同じ|
|Excel エクスポータ互換性|動作|動作（変更不要）|
|制約集合（ハード＋ソフト）|同じ|同じ（Constraint Streams 実装）|

### エンジニアリング上の違い

|トピック|7.4.2（Python）|8.1.1（Java）|
|---|---|---|
|ランタイム／ツールチェーン|Python ＋ 独自ソルバ|Java ＋ Timefold、Maven プロジェクト|
|型付けとモデル|動的型付け|強い型付け（DaySlot、BlockDecision、CrewSeat、SeatDay、EmployeeFact）|
|Pass 1 の時間拡張（Hours ramp）|実装済み|実装・形式化（挙動は同じ）|
|終了条件と polish|対応|対応（明確なベストスコア閾値）|

### まとめ

- Java/Timefold 版は同一の 2 パス挙動を維持しつつ、このデータセットで **実行時間を約 3 倍短縮**。
    
- ソフトスコア（平準化・同社結束・早期開始など）は **polish 後に一貫して改善**。
    
- 乗り換えは容易：**YAML 契約はそのまま**、Excel ダッシュボードもそのまま、実行を Maven アプリに切り替えるだけ。