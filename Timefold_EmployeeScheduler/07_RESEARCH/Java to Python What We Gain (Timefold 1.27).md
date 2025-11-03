## 1) Why Java for this project

- **Full Constraint Streams API** (joins, collectors, multi-stage solve) — first-class in Java.
    
- **Entity-dependent value ranges** with **`ValueRangeFactory`** (clean, fast, debuggable).
    
- **Deterministic multi-stage solving** (warm starts, staged termination, best-score limit).
    
- **Performance**: JVM runs big horizons faster; better profilers/JFR for tuning.
    
- **Cleaner single-pass modeling**: you can encode feasibility in ranges instead of more constraints.
    

> Bottom line: Java lets us replace several “guard” constraints with **valid search spaces**, and then add a **compensator** loop to progress hours safely.

---

## 2) `ValueRangeFactory` — the killer feature Python doesn’t match cleanly

### A. Simple numeric ranges (no Streams surprises)

`import ai.timefold.solver.core.api.domain.valuerange.ValueRange;
`import ai.timefold.solver.core.api.domain.valuerange.ValueRangeFactory;  `
`@ValueRangeProvider(id = "vrStartWithinWindow")
`public ValueRange<Integer> startRange() {     return ValueRangeFactory.createIntValueRange(windowStart, windowEnd + 1); }  @ValueRangeProvider(id = "vrDaysWithinWindow") public ValueRange<Integer> dayCountRange() {     int maxLen = windowEnd - windowStart + 1;     return ValueRangeFactory.createIntValueRange(1, Math.max(1, maxLen) + 1); }`

### B. Entity-dependent **joined** ranges (filter at the source)

`// Example: restrict employees for this seat to those with skill >= 1 and available @ValueRangeProvider(id = "eligibleEmployeesForSeat") public ValueRange<EmployeeFact> eligibleEmployeesForSeat() {     List<EmployeeFact> list = employees.stream()         .filter(e -> e.id != 0)         .filter(e -> e.skills.getOrDefault(opId, 0) >= 1)         .filter(e -> isAvailable(e, estimatedStart(), estimatedDays(), factory))         .filter(e -> !needManager || e.isManager)         .toList();     return ValueRangeFactory.createListValueRange(list); }`

**Why this matters:**  
In Python you usually end up enforcing these as **constraints** (and/or hand-rolled lists), which still allows the solver to _try_ illegal moves and then penalize them. In Java with `ValueRangeFactory`, those illegal moves **don’t exist** in the search space → less thrashing, fewer constraints, faster convergence.

---

## 3) The **Compensator** pattern — progressive hour cap (Java loop)

**Goal:** Prefer 8h but still reach feasibility. We raise the allowed hours gradually and stop once **0 hard** is achieved.

``int GLOBAL_MAX_ALLOWED = 12; SinglePassPlan incumbent = initialPlan;  for (int HOUR_CAP = 8; HOUR_CAP <= GLOBAL_MAX_ALLOWED; HOUR_CAP++) {     final int cap = HOUR_CAP;      // autoHours(b) should read `cap` and clip choices (e.g., allowedUnderCap = allowed <= cap)     Solver<SinglePassPlan> quick = buildSolver(         SinglePassPlan.class,         new Class<?>[]{ BlockDecision.class, CrewSeat.class },         SinglePassConstraints.class,         "0hard/*/*",        // stop as soon as all hard constraints are satisfied         5,                  // minutes budget per cap (tune)         60                  // unimproved seconds (tune)     );      incumbent = quick.solve(incumbent);  // warm start each loop     if (incumbent.getScore().toString().startsWith("0hard")) break; }  // Optional polish: maximize medium/soft after feasibility Solver<SinglePassPlan> polish = buildSolver(     SinglePassPlan.class,     new Class<?>[]{ BlockDecision.class, CrewSeat.class },     SinglePassConstraints.class,     null, /* no best score limit */     20,   /* minutes */     300   /* unimproved seconds */ ); incumbent = polish.solve(incumbent);``

**Why this beats Python** for our use case:

- Clean **multi-stage** orchestration & warm-starts.
    
- Encapsulate **hours policy** (near-8 preference) inside the model; the loop only changes a single global cap.
    
- Keeps the model **single-pass** and feasible while honoring **transit**/**max-stay** rules.
    

---

## 4) From Python 2-Pass → Java Single-Pass (with fewer constraints)

### Old (Python, 2-pass)

1. Choose block timing & hours.
    
2. Try to assign people afterward.  
    → Many “oops” constraints needed: availability, factory/day uniqueness, transit, max-stay, etc.
    

### New (Java, single-pass)

- Model **both** timing and staffing as planning entities.
    
- Use **`ValueRangeFactory`** to **pre-exclude** illegal choices:
    
    - Start/days limited to window
        
    - Hours choices clipped by current compensator cap
        
    - Employee candidates filtered by skill, availability, manager requirement
        
- Keep only the **truly global** hard constraints (phase order, daily 12h sum, transit gap, region stay max).
    

**Result:** fewer constraints, smaller search space, better feasibility, simpler score graphs.

# 1) なぜ本プロジェクトは Java なのか

- **Constraint Streams API**（結合・コレクタ・マルチステージ解法）が **Java で第一級**。
    
- **`ValueRangeFactory`** による **エンティティ依存の値レンジ**（クリーン・高速・デバッグしやすい）。
    
- **決定的なマルチステージ解法**（ウォームスタート、段階的終了条件、ベストスコアしきい値）。
    
- **性能**：JVM は大きな計画地平で有利。プロファイラ/JFR によるチューニングも良好。
    
- **よりクリーンなシングルパス・モデリング**：制約で縛る代わりに、レンジで実現可能性をエンコードできる。
    

> 結論：Java なら複数の「ガード系制約」を **有効な探索空間** に置き換え、さらに **コンペンセータ** ループで安全に労働時間を段階的に進められる。

---

## 2) `ValueRangeFactory` — Python に綺麗に対応がないキラー機能

### A. シンプルな数値レンジ（Stream 由来の副作用なし）

`import ai.timefold.solver.core.api.domain.valuerange.ValueRange; import ai.timefold.solver.core.api.domain.valuerange.ValueRangeFactory;  @ValueRangeProvider(id = "vrStartWithinWindow") public ValueRange<Integer> startRange() {     // 包含レンジ: windowStart..windowEnd     return ValueRangeFactory.createIntValueRange(windowStart, windowEnd + 1); }  @ValueRangeProvider(id = "vrDaysWithinWindow") public ValueRange<Integer> dayCountRange() {     int maxLen = windowEnd - windowStart + 1;     return ValueRangeFactory.createIntValueRange(1, Math.max(1, maxLen) + 1); }`

### B. エンティティ依存の **結合済み** レンジ（発生源でフィルタ）

`// 例: この席に割り当て可能な従業員を「スキル>=1 かつ 出勤可」に限定 @ValueRangeProvider(id = "eligibleEmployeesForSeat") public ValueRange<EmployeeFact> eligibleEmployeesForSeat() {     List<EmployeeFact> list = employees.stream()         .filter(e -> e.id != 0)         .filter(e -> e.skills.getOrDefault(opId, 0) >= 1)         .filter(e -> isAvailable(e, estimatedStart(), estimatedDays(), factory))         .filter(e -> !needManager || e.isManager)         .toList();     return ValueRangeFactory.createListValueRange(list); }`

**なぜ重要か：**  
Python だと多くを **制約**（または手作りリスト）で表現するため、ソルバーは **不正な手** を一度試し、後からペナルティで戻す動きになりがち。Java の `ValueRangeFactory` なら不正手は **探索空間に存在しない** → 無駄が減り、制約が減り、収束が速い。

---

## 3) **コンペンセータ** パターン — 時間上限を段階的に上げる（Java ループ）

**狙い：** 原則 8h を優先しつつ、実現可能性(0 hard)に到達するまで許容上限を段階的に引き上げる。

`int GLOBAL_MAX_ALLOWED = 12; SinglePassPlan incumbent = initialPlan;  for (int HOUR_CAP = 8; HOUR_CAP <= GLOBAL_MAX_ALLOWED; HOUR_CAP++) {     final int cap = HOUR_CAP;      // autoHours(b) は cap を参照し、(allowed <= cap) にクリップする     Solver<SinglePassPlan> quick = buildSolver(         SinglePassPlan.class,         new Class<?>[]{ BlockDecision.class, CrewSeat.class },         SinglePassConstraints.class,         "0hard/*/*",         // ハードが0になったら即停止         5,                   // 各段の所要分（調整可）         60                   // 非改善秒（調整可）     );      incumbent = quick.solve(incumbent);  // 各段をウォームスタート     if (incumbent.getScore().toString().startsWith("0hard")) break; }  // 任意の仕上げ: 実現可能化後に medium/soft を最大化 Solver<SinglePassPlan> polish = buildSolver(     SinglePassPlan.class,     new Class<?>[]{ BlockDecision.class, CrewSeat.class },     SinglePassConstraints.class,     null,  /* ベストスコアしきい値なし */     20,    /* 分 */     300    /* 非改善秒 */ ); incumbent = polish.solve(incumbent);`

**本用途で Python より有利な点：**

- クリーンな **マルチステージ** オーケストレーションとウォームスタート。
    
- **8h 近傍優先** のポリシーはモデル内に保持し、ループは **グローバル上限 cap** だけを変更。
    
- **シングルパス** を維持しつつ、**地域間移動/最大滞在**（ビザ）制約も同時に守れる。
    

---

## 4) Python 2パス → Java シングルパス（制約を減らす）

### 旧（Python・2パス）

1. ブロックの開始/日数/時間を決める
    
2. その後に人員を割付  
    　→ 可用性・工場/日ユニーク・トランジット・最大滞在など **「後付けのOops制約」** が増える
    

### 新（Java・シングルパス）

- **時間計画と人員割当** をどちらも計画エンティティ化。
    
- **`ValueRangeFactory`** で不正手を **事前に排除**：
    
    - 開始/日数をウィンドウ内に限定
        
    - 時間は現在のコンペンセータ cap でクリップ
        
    - 候補従業員はスキル・可用性・管理者要件でフィルタ
        
- **本当にグローバルなハード制約**（フェーズ順序・日別12h合計・トランジット間隔・地域最大滞在）だけを残す。
    

**効果：** 制約が減る／探索空間が小さくなる／実現可能性が上がる／スコア推移が素直になる。