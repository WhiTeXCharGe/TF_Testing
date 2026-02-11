## 1) Background: what Decoder2 did

Decoder2 builds **EnvConfig.yaml** + **Schedule.yaml** by combining 3 sources:

1. **新規製番リスト (planned master list)**
    
    - Reads each module/tool code and its planned phase dates:
        
        - 工程１開始可能日 / 工程2開始可能日 / 工程3開始可能日 / 工程4開始可能日 / 工程４終了予定日
            
    - Creates tasks (module → phases p1..p4) using those dates.
        
2. **SU_Others (actual daily schedule sheet)**
    
    - Reads daily cells per worker:
        
        - **Red cells** → unavailable dates
            
        - **Grey cells** → personal business fixed assignments
            
        - **Normal cells** → “worker is doing something” (module code / task text)
            
    - If a SU_Others normal cell matches a code from 新規製番リスト _and is inside the planned range_, it is treated as that module task.
        
    - If it matches code but is _outside planned range_ (or code not found), Decoder2 creates a **dummy “other”** task.
        
3. **スキル集計 (skill levels)**
    
    - Builds skill map from the skill Excel.
        
    - Also builds a second “skill inference” from SU_Others assignments.
        
    - Merges: `skill = max(skill_from_SU_Others, skill_from_skill_excel)`.
        

---

## 2) The core problem in Decoder2

Decoder2 assumes:

- 新規製番リスト dates = the real working window
    

But in reality:

- **新規製番リスト = “plan schedule” (often outdated or different data source)**
    
- **SU_Others = “actual plan / actual working span”**
    

So when the same module code appears in both files but with different date ranges, Decoder2 splits work into two different schedules:

### Example

- SU_Others shows module A actually worked: **09/01 – 11/10**
    
- 新規製番リスト planned module A: **07/01 – 09/30**
    

Decoder2 behavior:

- Creates planned tasks covering 07/01–09/30
    
- SU_Others work on 09/01–11/10 partially overlaps (09/01–09/30) but **10/01–11/10 becomes “other”**
    
- Result:
    
    - Extra “other” workload created
        
    - Real module workload becomes incomplete / mismatched
        
    - Assignment list can become too large or incorrectly duplicated
        
    - Solver gets wrong staffing pressure (more tasks than should exist)
        

---

## 3) Decoder3 fix: “percent + shifting (rescale) method”

Decoder3 changes the meaning of 新規製番リスト dates:

- It **does NOT directly use those dates as final task windows**
    
- It uses them to compute **phase proportions (percentages)** only
    
- Then it maps those proportions onto the **actual SU_Others span** for the same module code
    

### Concept

- 新規製番リスト defines: “phase1 is about 25% of the plan, phase2 25%, phase3 20%, phase4 30%” (example)
    
- SU_Others defines: “actual work happened from X to Y”
    
- Decoder3 creates shifted phases so that:
    
    - Phase proportions follow the plan ratio
        
    - But phase windows **exactly cover the actual span**
        
    - The module becomes **one consistent task**, not split into planned vs other
        

---

## 4) Decoder3 shifting algorithm (pseudocode)

### Step A: read planned phase dates & validate

Rules (same as Decoder2):

- phase1_start = 工程１開始可能日
    
- phase2_start = 工程2開始可能日
    
- phase3_start = 工程3開始可能日
    
- phase4_start = 工程4開始可能日
    
- phase4_end = 工程４終了予定日
    

Phase ends:

- phase1_end = phase2_start - 1
    
- phase2_end = phase3_start - 1
    
- phase3_end = phase4_start - 1
    
- phase4_end = 工程４終了予定日
    

Validation:

- All dates must exist and not be `"N/A"`
    
- Must satisfy ordering:
    
    - 工程１ ≤ 工程２ ≤ 工程３ ≤ 工程４ ≤ 工程４終了
        

If invalid → module is “cut” from 新規製番リスト list (treated as not-listed).

### Step B: compute phase percentages from plan

`plan_p1_days = days(phase1_start..phase1_end) plan_p2_days = days(phase2_start..phase2_end) plan_p3_days = days(phase3_start..phase3_end) plan_p4_days = days(phase4_start..phase4_end)  plan_total = plan_p1_days + plan_p2_days + plan_p3_days + plan_p4_days  ratio_p1 = plan_p1_days / plan_total ratio_p2 = plan_p2_days / plan_total ratio_p3 = plan_p3_days / plan_total ratio_p4 = plan_p4_days / plan_total`

### Step C: find actual span in SU_Others

For each module code:

- Scan SU_Others schedule sheet cells
    
- Collect all dates where this code appears (normal cells)
    
- Determine:
    
    - `actual_first_date = min(dates_found)`
        
    - `actual_last_date = max(dates_found)`
        

If code never appears in SU_Others:

- Decoder3 keeps the planned task dates as fallback (or treats it as planned-only task depending on your current design).
    

### Step D: allocate actual days by ratio (integer days)

We must convert ratios into real day counts that sum exactly to actual_total.

`actual_total_days = days(actual_first_date..actual_last_date)  raw_p1 = ratio_p1 * actual_total_days raw_p2 = ratio_p2 * actual_total_days raw_p3 = ratio_p3 * actual_total_days raw_p4 = ratio_p4 * actual_total_days  int_p1 = floor(raw_p1) int_p2 = floor(raw_p2) int_p3 = floor(raw_p3) int_p4 = floor(raw_p4)  remaining = actual_total_days - (int_p1+int_p2+int_p3+int_p4)  distribute remaining days to phases with largest (raw - floor(raw))`

### Step E: build shifted phase windows (contiguous)

`p1_start = actual_first_date p1_end   = p1_start + int_p1 - 1  p2_start = p1_end + 1 p2_end   = p2_start + int_p2 - 1  p3_start = p2_end + 1 p3_end   = p3_start + int_p3 - 1  p4_start = p3_end + 1 p4_end   = actual_last_date   # ensure exact cover`

Now module phases are “shifted” to match the real span.

---

## 5) How Decoder3 uses shifted phases (effects)

### Schedule generation

- Task dates for each module phase in Schedule.yaml use shifted windows.
    
- SU_Others daily assignments are mapped into the correct shifted phase (p1..p4), instead of being compared to the original planned windows.
    

### Dummy tasks (“other”)

- If SU_Others has a code/task that is not in valid 新規製番リスト list → becomes dummy “other”.
    
- This includes:
    
    - code not found in 新規製番リスト
        
    - code was found but got cut due to invalid date order / missing dates
        

### Skill map inference

Decoder3 changes the SU_Others skill inference pipeline:

- Decoder2 inferred skill based on “as-is date range”
    
- Decoder3 infers skill after shifting:
    
    - A worker who worked in the module’s shifted phase gets credited for that module/phase skill
        
- Merge remains:
    
    - `final_skill = max(skill_from_shifted_SU_Others, skill_from_skill_excel)`
        

### Workload counting

Decoder3 counts workloads after shifting:

- OptionB (real worker-days):
    
    - every worker assignment day counts `+1` for that module/phase
        
- OptionA (window-days):
    
    - phase duration = `end-start+1`
        

Config:

- If OptionA enabled → `workload_days = max(OptionA, OptionB)`
    
- If OptionA disabled → `workload_days = OptionB`
    

If OptionB is 0:

- log as warning/error (still output task)
    

---

## 6) TransformationLog contents (what it includes)

Decoder3 outputs a text log so you can audit the shifting result.

### A) Shifting results per module

For each module that exists in 新規製番リスト and also appears in SU_Others:

- Module code
    
- Planned date inputs from 新規製番リスト (phase starts + phase4 end)
    
- Computed planned phase lengths + ratios (%)
    
- SU_Others actual span:
    
    - first date + sample worker/task text
        
    - last date + sample worker/task text
        
- Shifted result:
    
    - shifted phase windows (p1..p4)
        
- Workload notes:
    
    - OptionA / OptionB results
        
    - warnings if OptionB == 0
        

### B) Cut-out section (invalid 新規製番リスト rows)

A list of module codes removed from planned list, with reasons:

- wrong date order
    
- missing date
    
- “N/A” present
    

### C) Dummy module list (exists in SU_Others but not in valid planned list)

A list of “tool code / task labels” that were converted to dummy “other” tasks, for quick review.

---

## 7) Summary of improvement

Decoder3 solves the main mismatch by making:

- **New規製番リスト dates = “phase ratios only”**
    
- **SU_Others dates = “final truth for actual span”**
    

So:

- the same module code no longer becomes “two tasks” (planned vs other)
    
- phase boundaries stay meaningful (based on planned proportions)
    
- Schedule.yaml becomes consistent with actual operations
    
- skill inference becomes more accurate because it follows the shifted phase mapping
    
- transformation log provides traceability for every shifted/cut/dummy decision