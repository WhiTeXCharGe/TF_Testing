# v822 summarize.md

このドキュメントは、v8.2.2（v822）版の **二段階スケジューラ** を全部説明します。

- `EmployeeSchedule.java`
    
    - ドメインモデル（Pass1 / Pass2）
        
    - カレンダー（休日判定）、パース、制約、ソルバー
        
    - **Pass1: SELECTIVE RAMP LOOP + Snapshot 出力**
        
- `ExportSchedule.java`
    
    - Pass2 の結果を `Schedule.yaml` / `ScheduleN.yaml` に書き戻し（v821 と同じ 4 引数版を想定）
        
- `pom.xml`（ここには出ていませんが、構成イメージは v831 とほぼ同じ）
    
    - Java 17, Timefold, SnakeYAML, exec-maven-plugin
        

v8.3.1 との違い：

- v8.2.2 も **二段階 (Pass1 / Pass2)** ですが、内容は少し軽いです
    
    - **Pass1** = Block（頭数・日数・開始日）だけを決める
        
    - **Pass2** = 各 Seat に「誰が座るか」だけを決める
        
- v8.2.2 の **Pass1 は “SELECTIVE RAMP LOOP”** 方式
    
    - 各 Block ごとに「許可される時間(Hours)の tier（8 ⇒ 8,10 ⇒ 8,10,12 …）」を徐々に増やす
        
    - Hard 制約に違反している Block だけ tier を上げる（良いブロックはそのまま固定）
        
- Pass1 の目的は「**Ramp の挙動や Block 配置を調査するためのデバッグ版**」
    
    - 各イテレーションごとに `Schedule1.yaml`, `Schedule2.yaml` … の **スナップショット** を出す
        
- 制約は v831 より少しシンプル
    
    - Pass1: 能力・窓・フェーズ順序・頭数・Under/Over fill・Stacking
        
    - Pass2: スキル、1 工場/日、12h/日、ブロック内マネージャ 1 人以上、個人 Off、Pinned など
        
    - **地域移動・滞在日数などの高度な制約はまだない**
        

---

## 1. ビルド & 実行方法

`EmployeeSchedule.main()` では、コマンドライン引数として

`args[0] = EnvConfig.yaml のパス args[1] = Schedule.yaml のパス`

を受け取り、

1. `solveFromYaml(envPath, schedPath)` を実行
    
2. 返ってきた `RunResult` から `finalPlan`（Pass2Plan）と `planStart` を受け取る
    
3. `ExportSchedule.overwriteScheduleWithAssignments(finalPlan, planStart, schedPath, envPath)` を呼び出して `Schedule.yaml` を上書き
    
4. `"Done."` と表示して終了
    

想定ワークフロー（v8.3 系と同じイメージ）：

1. **ビルド**
    
    `mvn -DskipTests clean package`
    
2. **実行**
    
    `mvn -q exec:java -D"exec.args=EnvConfig.yaml Schedule.yaml"`
    
3. 実行中、Pass1 のイテレーションごとに `Schedule1.yaml`, `Schedule2.yaml`, … の **スナップショット** が同じディレクトリに生成される
    
4. 最終的に、元の `Schedule.yaml` が **Pass2 の結果**で上書きされる
    

---

## 2. ドメインモデル（共通部分）

### 2.1 DaySlot

`public static class DaySlot {     @PlanningId public int id;     public LocalDate date; }`

- 計画期間内の **1 日** を表す
    
- `id` は 0,1,2,... の連番
    
- `date` は実カレンダー日付
    

### 2.2 EmployeeFact

`public static class EmployeeFact {     @PlanningId public int id;   // 0 = UNASSIGNED     public String wid;           // Worker ID     public String name;     public Map<String,Integer> skills;    // opId -> skill level     public boolean isManager;     public String workerCompany; }`

- 従業員の情報（スキル・所属会社・マネージャーかどうか）
    
- `id == 0` は **UNASSIGNED**（幽霊ワーカー）
    
- `skills.get(opId) >= 1` なら、その op を担当可能
    
- ヘルパー関数：
    
    - `isUnassigned(e)` / `skill(e, opId)` / `isManager(e)` / `company(e)`
        

### 2.3 TaskWindow

`public static class TaskWindow {     public String module;     public String factory;     public String phaseId;     public int phaseNum;     public String opId;     public int startDayId;     public int endDayId;     public List<Integer> allowed;   // allowed hours per day     public int minHeads;     public int maxHeads;     public int workloadDays; }`

- `Schedule.yaml` の `workflow_task_list` から作られる
    
- 1 つの `(module, phaseId, opId)` の「期間」と「必要作業量」の情報を持つ
    
- `workloadDays * baselineHours(4 or 8)` が基準総工数
    
- `allowed` は `EnvConfig` の `work_hours`（例: [8,10,12]）
    

---

## 3. カレンダー（休日・稼働日）とユーティリティ

### 3.1 Calendars

`static class Calendars {     Set<Integer> weekends;     Map<String, Set<Integer>> fabOff;     Map<String, Set<Integer>> regionOff;     Map<String, Set<Integer>> customerOff;     Map<String, Set<Integer>> workerCompanyOff;     Map<String, String> fabToRegion;     Map<String, String> fabToCustomer;     Map<String, Set<Integer>> workerOffByWid; }`

- Pass1 / Pass2 共通で使う「休日情報」の集約クラス
    

`buildCalendars(envPath, planStart, planEnd)`:

1. 期間内の **土日** を `weekends` に入れる
    
2. `fab_list` から：
    
    - fab ごとの `unavailable_dates` → `fabOff`
        
    - fab → region, customer のマッピング
        
3. `region_list` から：
    
    - region ごとの `unavailable_dates` → `regionOff`
        
4. `customer_company_list` から：
    
    - customer ごとの `unavailable_dates` → `customerOff`
        
5. `worker_company_list` から：
    
    - workerCompany ごとの `unavailable_dates` → `workerCompanyOff`
        
6. `worker_list` から：
    
    - 各 worker の `unavailable_dates` → `workerOffByWid`
        

### 3.2 稼働判定

`static boolean isWorkingDay(int dayId, String fabId) { ... } static int workingDaysCount(int startDay, int dayCount, String fabId) { ... }`

- `isWorkingDay`
    
    - 土日 → `false`
        
    - fab off / region off / customer off に入っていれば `false`
        
    - それ以外は `true`
        
- `workingDaysCount`
    
    - `[startDay .. startDay+dayCount-1]` のうち、`isWorkingDay` が `true` の日の数を数える
        
    - Pass1 の `produced()` / Pass2 の SeatDay 展開で使用
        
    - **非稼働日は** 工数にカウントされない
        

---

## 4. EnvConfig / Schedule のパース

### 4.1 ParsedEnv / OpDef

`static class OpDef {     String phaseId; int phaseNum;     List<Integer> allowed; int min; int max; } static class ParsedEnv {     Map<String,OpDef> opdef;     List<EmployeeFact> employees;     Map<String, EmployeeFact> byWid; }`

`parseEnv(envPath)`:

1. EnvConfig.yaml を読み込み、`environment` ノードを取得
    
2. `workflow_list.phase_list.operation_list` から `OpDef` を作る
    
    - `id` → opId (例: p1o1)
        
    - `work_hours` → allowed hours ([8,10,12] 等)
        
    - `min_worker_num` / `max_worker_num`
        
    - 所属フェーズ `phaseId`、`phaseNum`
        
3. `worker_list` から `EmployeeFact` を作る
    
    - まず UNASSIGNED (id=0) を追加
        
    - 各 worker ごとに `skills`, `is_manager`, `worker_company` を設定
        
    - `byWid` マップ（wid → EmployeeFact）も作成
        
4. `OP_CAPACITY`, `OP_AVG_SKILL` を計算
    
    - `OP_CAPACITY[opId]` = その op にスキル>0 を持つワーカー数
        
    - `OP_AVG_SKILL[opId]` = スキルレベルの平均（ソフト制約用）
        

### 4.2 ParsedSchedule / FixedAssign

`static class ParsedSchedule {     LocalDate planStart; LocalDate planEnd;     List<DaySlot> daySlots;     List<TaskWindow> windows;     Map<String,Integer> requiredByKey;   // module|opId -> baseline hours     List<FixedAssign> fixedRows;     Map<String,Integer> fixedHoursByKey; // module|opId -> fixed hours total }  static class FixedAssign {     String module; String opId; String wid;     int startDayId; int endDayId;     Map<Integer,Integer> hoursByDay; // dayId -> hours     String phaseId; int phaseNum; }`

`parseSchedule(schedPath, opdef)`:

1. `schedule.plan_range.start_date` / `end_date` から `planStart`, `planEnd` を取得
    
2. 全期間分の `DaySlot(id, date)` を作成
    
3. `workflow_task_list.phase_task_list.operation_task_list` を走査：
    
    - `TaskWindow` を作って `windows` に追加
        
    - `workload_days` と `OpDef.allowed` から基準時間を計算
        
        - allowed が [4] だけなら baseline=4、それ以外は baseline=8
            
    - `(module|opId)` ごとに `requiredByKey` に合計
        
4. `assignment_list` を走査して **固定行 (plan_flexibility == "fixed")** を抽出
    
    - `operation_task` から `module` と `opId` を推測
        
    - `worker`, `start_date`, `end_date`, `work_date_list` を読み
        
    - `hoursByDay`（dayId → hours）と `fixedHoursByKey[module|opId]` を更新
        
    - `FixedAssign` として `fixedRows` に追加
        

`fixedHoursByKey` は後で **Pass1 の requiredHours から固定分を差し引く**ために使う。

---

## 5. グローバル定数

`static final int DAILY_CAP = 12; static double TARGET_HOURS_PER_EMP = 0.0;  static final Map<String,Integer> OP_CAPACITY = new HashMap<>(); static final Map<String,Double>  OP_AVG_SKILL = new HashMap<>();`

- `DAILY_CAP` = 従業員 1 日 12h 上限（Pass2 hard 制約）
    
- `TARGET_HOURS_PER_EMP` = 全必要時間 / 人数 （ソフト制約 “TotalHours バランス” 用）
    
- `OP_CAPACITY` / `OP_AVG_SKILL` は `parseEnv` で計算済み
    

---

## 6. Pass 1 – BlockDecision と SELECTIVE RAMP LOOP

### 6.1 BlockDecision / Pass1Plan

`@PlanningEntity public static class BlockDecision {     @PlanningId public int id;     public String module, factory, phaseId, opId;     public int phaseNum;     public int windowStart, windowEnd;     public int requiredHours;     public List<Integer> allowed;     public int minHeads, maxHeads;      @PlanningVariable(valueRangeProviderRefs = "vrDayIds")     public Integer startDay;     @PlanningVariable(valueRangeProviderRefs = "vrHeadOptions")     public Integer heads;     @PlanningVariable(valueRangeProviderRefs = "vrDayCountOptions")     public Integer days;      public int seedHours = 8; }`

- Block 1 つに対して、以下を決める：
    
    - 開始日 `startDay`
        
    - 同時頭数 `heads`
        
    - 継続日数 `days`
        
- `requiredHours` は **すでに fixed を差し引いた必要柔軟時間**
    

`Pass1Plan`:

`@PlanningSolution public static class Pass1Plan {     @ValueRangeProvider(id = "vrDayIds")     @ProblemFactCollectionProperty     public List<Integer> dayIds;      @ValueRangeProvider(id = "vrHeadOptions")     @ProblemFactCollectionProperty     public List<Integer> headOptions;      @ValueRangeProvider(id = "vrDayCountOptions")     @ProblemFactCollectionProperty     public List<Integer> dayCountOptions;      @ProblemFactCollectionProperty     public List<DaySlot> daySlots;      @PlanningEntityCollectionProperty     public List<BlockDecision> blocks; }`

- dayIds / headOptions / dayCountOptions は全 Block 共通の ValueRange
    
- `blocks` が PlanningEntity のリスト
    

### 6.2 autoHours と produced

`static int autoHours(BlockDecision b) { ... } static int produced(BlockDecision b) { ... }`

- `autoHours(b)`
    
    - `allowed`（例えば [8,10,12]）から 1 つ `h` を選ぶロジック
        
    - まず「`produced >= requiredHours` かつ overfill が 1 ブロック日以内」の候補を探す
        
        - その中から |h-8| が小さいもの、さらに h が小さいものを優先
            
    - それが無い場合、
        
        - underfill と overfill の度合い、|h-8| などのタプルで比較してベストな h を選ぶ
            
- `produced(b)`
    
    - `H = heads` / `D = workingDaysCount(...)` / `h = autoHours(b)`
        
    - `H * h * D` を返す
        
    - 非稼働日は D に入らない
        

### 6.3 Pass1Constraints（v822 で使う制約）

Pass1 の制約は v831 とほぼ同じですが、対象は **BlockDecision だけ**（まだ座席や従業員は出てこない）。

主な hard 制約：

1. `withinWindow`
    
    - `startDay` / `days` が `[windowStart .. windowEnd]` に収まっているか
        
2. `daysWithinWindowLen`
    
    - `days <= windowEnd-windowStart+1` を超えないようにする
        
3. `hoursValueAllowed`
    
    - `autoHours(b)` が `allowed` リストに含まれているか
        
4. `headsInMinMax`
    
    - `minHeads <= heads <= maxHeads`
        
5. `noUnderfill`
    
    - `produced(b) >= requiredHours`（不足時間は hard ペナルティ）
        
6. `overfillAtMostOneDay`
    
    - overfill が 1 ブロック日 (H*h) を超えない
        
7. `phaseOrder`
    
    - 同じ module の phase N が終わる前に N+1 が始まらないようにする
        
8. `dailyHeadCapacityByOp`
    
    - 各 `(day, opId)` で、動いている Block の `heads` 合計 ≤ `OP_CAPACITY[opId]`
        
    - `isWorkingDay` を使うので **非稼働日は無視**
        
9. `penalizeStackByOp` (medium)
    
    - 同じ `(day, opId)` に複数 Block が重なると medium ペナルティ
        

主な soft 制約：

- `preferHoursNear8`
    
- `preferSmallerHours`（h が小さい方が少し好ましい）
    
- `minimizeHeads`（頭数を減らしたい）
    
- `minimizeDays`（日数を短くしたい）
    
- `preferEarlierStart`（早めに開始したい）
    

### 6.4 detectHardViolators と SELECTIVE RAMP LOOP

`solvePass1SelectiveRamp(...)` が v822 の特徴です。

流れ：

1. **ValueRange の準備**
    
    - `headOptions`：全 Window の minHeads..maxHeads の範囲
        
    - `dayIds`：全 DaySlot の id
        
    - `dayCountOptions`：1..最大ウィンドウ長
        
2. **perBlockTier の初期化**
    
    - 各 Block (TaskWindow) に tier=1 を与える
        
    - tier = 1 → 一番小さい allowed（例: [8]）だけ
        
    - tier = 2 → [8,10]、tier = 3 → [8,10,12] … のように増やす
        
3. **各 Block ごとの最大 tier を計算**
    
    - `perBlockMaxTier[id] = w.fullAllowedSorted().size()`
        
4. **ループ (iter=1..globalMaxLoops)**
    
    1. `seedBlocksForTier` で、現在の `perBlockTier` に応じた初期 Block を生成
        
        - `requiredHours = baseline*workloadDays - fixedHoursByKey`
            
        - tier = 1 のときは allowed=[最小時間] だけ
            
    2. その seed で Pass1 を Solve（30 分 / 60 秒 unimproved、bestScoreLimit=0hard）
        
    3. `writeScheduleSnapshot(iter, solved.blocks, snapshotCfg)` を呼んで、
        
        - Pass1 の Block を座席に展開
            
        - fixed から pinned seat も展開
            
        - Employee id=1 を flexible seat に全部割り当て
            
        - それを `Pass2Plan` として `Schedule{iter}.yaml` に書き出す（ExportSchedule 経由）
            
    4. ベストスコア更新ロジック
        
        - hard=0 の結果が出たら `best` を更新、Polish をかける可能性あり
            
    5. hard=0 なら：
        
        - もう一度 Polish（20 分 / 60 秒）して、その結果で return
            
    6. hard!=0 の場合：
        
        - `detectHardViolators(solved.blocks)` で、hard 制約に違反している Block の id を検出
            
        - 次の iteration のために：
            
            - **違反していない Block**
                
                - その Block の `autoHours(solvedB)` を 1 個だけ allowed として固定
                    
                - startDay / heads / days もそのままコピー
                    
                - → その Block は **tier を上げず、時間の選択も固定** される
                    
            - **違反している Block**
                
                - `perBlockTier[bid]` を最大全体の範囲で 1 つ上げる（tier++、ただし maxTier まで）
                    
                - 新しい tier（例: [8,10]）で seed を作り直す
                    
        - いずれの Block でも requiredHours=0 ならスキップ
            
    7. もし **どの Block も tier を上げられなくなった**ら（`anyTierChange==false`）
        
        - その時点の best or solved を返す（hardIsZero は hardZero(score)）
            

このループによって：

- 「簡単な Block」は早い段階で固定される（allowed が 1 つになり、位置も固定）
    
- 「難しい Block」だけが tier を増やして、8→10→12… と時間の組み合わせの自由度を増やしていく
    
- 各 iteration ごとに `ScheduleN.yaml` が出るため、**どの tier で何が起きているかを目視で確認できる**
    

---

## 7. Seat 展開と Pinned 座席

### 7.1 expandToSeats（Pass1 の Block → Flexible Seats）

`static Expanded expandToSeats(List<BlockDecision> blocks, List<DaySlot> days)`

- 各 Block に対して：
    
    - `hours = autoHours(b)`
        
    - `start = b.startDay`（null の場合は windowStart）
        
    - `dcount = b.days`（null の場合は 1）
        
    - `headCount = b.heads`（null の場合は 1）
        
- `headCount` 回ループして seat を作成：
    
    - `seatKey = module + "|" + opId + "|s" + 0000 + "|d" + start`
        
    - `CrewSeat` を作り、`employee` は UNASSIGNED で初期化
        
- 各 seat について日数分ループ：
    
    - 日ごとに `SeatDay(seatKey, thatDay, hours, factory)` を追加
        
    - 非稼働日は `isWorkingDay` でスキップ
        

### 7.2 expandPinnedSeats（FixedAssign → Pinned Seats）

`static Expanded expandPinnedSeats(ParsedSchedule sch, ParsedEnv env, List<TaskWindow> windows, List<DaySlot> days)`

- `TaskWindow` から補助マップを作る：
    
    - module → factory
        
    - module|opId → phaseId, phaseNum
        
- 各 `FixedAssign` について：
    
    - `seatKey = module + "|" + opId + "|PIN|" + wid + "|d" + startDayId`
        
    - `CrewSeat` を作成：
        
        - `id` は 1,000,000 以上（Flexible と区別）
            
        - `pinned = true`, `pinnedWid = wid`
            
        - `employee = env.byWid.get(wid)`（見つからない場合は UNASSIGNED）
            
        - `days` は `hoursByDay` の min/max dayId から計算
            
        - `hours` は `hoursByDay` の最大値（その行で一番長い時間）
            
    - `hoursByDay` を回して `SeatDay` を作成（非稼働日はスキップ）
        

### 7.3 Pass2 用の入力

`solveFromYaml` 内では：

`Expanded exFlex   = expandToSeats(p1.blocks, sch.daySlots); Expanded exPinned = expandPinnedSeats(sch, env, sch.windows, sch.daySlots);  List<CrewSeat> allSeats = new ArrayList<>(); allSeats.addAll(exPinned.seats); allSeats.addAll(exFlex.seats);  List<SeatDay> allSeatDays = new ArrayList<>(); allSeatDays.addAll(exPinned.seatDays); allSeatDays.addAll(exFlex.seatDays);`

- Pass2 は **pinned + flexible 全部** を対象に、`employee` を決める
    
- pinned seat は `respectPinnedAssignments` 制約で worker 固定
    

---

## 8. Pass 2 – CrewSeat / Pass2Plan / 制約

### 8.1 CrewSeat / SeatDay / Pass2Plan

`@PlanningEntity public static class CrewSeat {     @PlanningId public int id;     public String module, factory, phaseId, opId;     public int phaseNum;     public int startDayId, days, hours;     public int seatIndex;     public String seatKey;     public int blockId;      @PlanningVariable     public EmployeeFact employee;      public boolean pinned = false;     public String pinnedWid = null; }`

- Pass2 では **CrewSeat が唯一の PlanningEntity**
    
- 従業員は `EmployeeFact` の中から好きな人を選ぶ（v822 では per-seat 候補リストはまだ無し）
    

`public static class SeatDay {     public String seatKey;     public DaySlot day;     public int hours;     public String factory; }`

`@PlanningSolution public static class Pass2Plan {     @ValueRangeProvider @ProblemFactCollectionProperty     public List<DaySlot> days;      @ValueRangeProvider @ProblemFactCollectionProperty     public List<EmployeeFact> employees;      @ProblemFactCollectionProperty     public List<SeatDay> seatDays;      @PlanningEntityCollectionProperty     public List<CrewSeat> seats; }`

- `employees` は value range として使われる（全員共通）
    
- `seatDays` は「誰がどの日に何時間働くか」の集計に利用
    

### 8.2 Pass2Constraints（hard）

1. `assignedAndSkill`
    
    - `UNASSIGNED` または skill<1 の割当を hard ペナルティ  
        → できる限り「スキルありの人」に座らせる
        
2. `oneFactoryPerEmpDay`
    
    - `(employee, day)` ごとに工場数を数え、2 以上ならペナルティ  
        → 1 日 1 工場のみ
        
3. `dailyCap12h`
    
    - `(employee, day)` ごとの `SeatDay.hours` 合計 > 12 ならペナルティ
        
4. `atLeastOneManagerPerBlock`
    
    - `blockId` ごとに manager の人数を数え、0 ならペナルティ  
        → 各ブロックに最低 1 名の manager
        
5. `employeeAvailableOnSeatDays`
    
    - `CAL.workerOffByWid` を見て、個人 Off の日にアサインされたらペナルティ
        
    - 会社 Off (`workerCompanyOff`) は現状 hard にはしていない（コメントのみ）
        
6. `respectPinnedAssignments`
    
    - `pinned == true` の seat で、`employee.wid != pinnedWid` の場合ペナルティ  
        → fixed 行の worker は強制的に固定
        

### 8.3 Pass2Constraints（soft）

1. `softSameCompanyPairs`
    
    - 同じ blockId 内で同じ会社の組み合わせを reward  
        → 同じ会社メンバーをなるべく同じブロックに集める
        
2. `softEncourageSkillVariety`
    
    - 同じ blockId + opId 内で skill レベルが同じペアにペナルティ  
        → スキル構成に少しバラつきを持たせる
        
3. `softBalanceBlockAvgSkill`
    
    - blockId + opId ごとの平均スキルと `OP_AVG_SKILL[opId]` の差をペナルティ  
        → ブロックごとの平均スキルが全体平均に近づくようにする
        
4. `softBalanceTotalHours`
    
    - `(employee)` ごとの総工数と `TARGET_HOURS_PER_EMP` の差をペナルティ  
        → 全員の総工数が平均に近づくよう調整
        

---

## 9. ソルバー構成 & 実行

共通ソルバー構築：

`static <S> Solver<S> buildSolver(     Class<S> solutionClass,     Class<?>[] entityClasses,     Class<? extends ConstraintProvider> providerClass,     String bestScoreLimit,     Integer spentMinutes,     Integer unimprovedSeconds )`

- XML ではなく Java コードで Timefold を設定
    
- `TerminationConfig`：
    
    - `bestScoreLimit`（"0hard/*medium/*soft" など）
        
    - `spentLimit`（分）
        
    - `unimprovedSpentLimit`（秒）
        

### 9.1 Pass1 – solvePass1SelectiveRamp

前述の通り、Ramp ループ + Violator 検出 + tier 更新 + Snapshot 出力を実行。

### 9.2 Pass2 – solvePass2Once

`static Pass2Plan solvePass2Once(     List<DaySlot> days, List<EmployeeFact> employees,     List<CrewSeat> seats, List<SeatDay> seatDays )`

- まず通常 Solve（30 分 / 60 秒 / bestScoreLimit="0hard/*medium/*soft"）
    
- hard=0 なら、追加で Polish（20 分 / 60 秒, bestScoreLimit なし）
    
- 結果の `Pass2Plan` を返す
    

---

## 10. Public API – solveFromYaml / main

`public static RunResult solveFromYaml(String envPath, String schedPath) throws IOException`

1. `parseEnv(envPath)` → `ParsedEnv env`
    
2. `parseSchedule(schedPath, env.opdef)` → `ParsedSchedule sch`
    
3. `buildCalendars(envPath, sch.planStart, sch.planEnd)`
    
4. `TARGET_HOURS_PER_EMP` を計算
    
5. `SnapshotCfg` を設定（planStart, schedulePath, employees, env, sch, windows, daySlots, envPath）
    
6. `solvePass1SelectiveRamp(...)` を呼び出し（Pass1）
    
7. `expandToSeats` と `expandPinnedSeats` で Seat と SeatDay を構築
    
8. `p1.hardIsZero` なら Pass2 を実行、そうでなければ「診断用出力のみ」で Pass2 をスキップ
    
9. `RunResult` に `finalPlan` と `planStart` を詰めて返す
    

`public static void main(String[] args) throws Exception {     String envPath = args.length > 0 ? args[0] : "EnvConfig.yaml";     String schedPath = args.length > 1 ? args[1] : "Schedule.yaml";      RunResult rr = solveFromYaml(envPath, schedPath);     ExportSchedule.overwriteScheduleWithAssignments(rr.finalPlan, rr.planStart, schedPath, envPath);     System.out.println("Done."); }`

- **最終的な出力**：
    
    - 元の `Schedule.yaml` が Pass2 の結果で上書きされる
        
    - fixed 行はそのまま維持され、flexible 行だけが新規に生成される（ExportSchedule のロジックに依存）
        

---

## 11. v822 を上司・チームに説明するときのポイント

1. **入力と出力**
    
    - 入力：`EnvConfig.yaml` + `Schedule.yaml`
        
    - 出力：Pass1 の各イテレーションごとの `ScheduleN.yaml`（デバッグ用）と、最終的に更新された `Schedule.yaml`
        
2. **二段階構造**
    
    - **Pass1 (Block)**
        
        - 誰が働くかは考えない
            
        - 「いつ・何日・何人で」やるかだけを決める
            
        - 休日（fab/region/customer/weekend）を考慮した上で、必要工数を満たしつつ容量やフェーズ順序を守る
            
        - 難しい Block だけ hours の選択肢（tier）を増やす “SELECTIVE RAMP”
            
    - **Pass2 (Assignment)**
        
        - Pass1 の Block と fixed 行を Seat に展開
            
        - 各 Seat に対して誰を入れるかを Timefold が決める
            
        - スキル、12h/日、1 工場/日、マネージャ 1 人以上、個人 Off、Pinned を守る
            
        - 会社ペアやスキルバランス・総工数バランスは Soft で最適化
            
3. **v831 との違い（ざっくり）**
    
    - v822：
        
        - Pass1 は SELECTIVE RAMP + Snapshot にフォーカスしたデバッグ寄りバージョン
            
        - Pass2 は「全従業員から自由に選ぶ」シンプルなモデル（まだ座席ごとの候補絞り込みなし）
            
        - 地域移動や滞在日数などの制約は無い
            
    - v831：
        
        - Pass1 はもう少し完成形に近く、Pass2 も候補フィルタや地域制約などが追加されている
            
4. **Snapshot の意味**
    
    - ループごとに `ScheduleN.yaml` を書き出しているので、
        
        - 「どの tier のときにどんな Block 配置になっているか」
            
        - 「固定行を含めた座席構造がどう見えるか」  
            をレビュー用に共有しやすい
            
    - 実際の運用では、最終 `Schedule.yaml`（Pass2結果）を使う