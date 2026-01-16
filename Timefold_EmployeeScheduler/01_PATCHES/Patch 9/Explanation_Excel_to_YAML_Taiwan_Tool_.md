# Excel → EnvConfig.yaml / Schedule.yaml 説明書

## 1. 全体イメージ

この仕組みでは、**2つの Excel ファイル**から情報を読み取り、2つの YAML を自動生成します。

- `20251201_2 SU_Others.xlsm`
    
    - シート：`予定表_2024` と（同じロジックで）`予定表_2025`
        
    - → **誰がどの会社か / 誰がマネージャーか / いつ出張 or 予定があるか / いつは休みか** を取り出す
        
- `20260105 台湾出張者予定_2025latest.xlsx`
    
    - シート：`F22_Tool Schedule` と `F20_Tool Schedule` 
        
    - → **各ツール（装置）の Phase1〜4 の予定期間** を取り出す
        

これらを組み合わせて、最終的に：

- `EnvConfig_from_excel.yaml`  
    → 人・会社・ワークフローの定義（環境情報）
    
- `Schedule_from_excel.yaml`  
    → 各装置の Phase1〜4 タスク + 各作業者の割当（スケジュール）
    

が作られます。

---

## 1.1 ざっくり対応表

| YAMLの場所                                                      | 主な中身                        | どのExcelから？                                  |
| ------------------------------------------------------------ | --------------------------- | ------------------------------------------- |
| `environment.worker_company_list`                            | 派遣元会社の一覧                    | SU_Others（予定表_2024 + 予定表_2025）の「企業名」        |
| `environment.worker_list`                                    | 作業者ID・氏名・所属会社・マネージャーフラグ・休み日 | SU_Others の「企業名」「姓名」「自由枠」「日付列（赤セル）」         |
| `environment.workflow_list[wf_tool]`                         | F22,F20ツール用の 大工程1〜4 定義      | 固定値（コード側で決め打ち、Excelには無い）                    |
| `environment.workflow_list[wf_other]`                        | その他作業用のワークフロー               | 固定値（Excelには無い）                              |
| `environment.fab_list / region_list / customer_company_list` | Fab名・地域・顧客など                | 固定値（テンプレ）                                   |
| `schedule.plan_range`                                        | スケジュール全体の開始日・終了日            | SU_Others の日付ヘッダ ＋ F22_Tool Schedule の日付    |
| `schedule.workflow_task_list`（F22,F20ツール）                    | 各装置の Phase1〜4 の期間           | F22_Tool Schedule の各行＋日付列                   |
| `schedule.workflow_task_list`（その他作業）                         | 作業                          | SU_Others のセル文字列（ツールコードを含まないもの）             |
| `schedule.assignment_list`（ツール関連）                            | 誰がどの装置のどの Phase に入るか        | SU_Others のセル文字列＋F22_Tool Schedule の装置コード一致 |
| `schedule.assignment_list`（その他作業）                            | ツール以外の作業割当                  | SU_Others のセル文字列（ツールコードなし）                  |

---

## 1.2 SU_Others（予定表_2024 / 2025）から取っている情報

### 1.2.1 企業名 → worker_company_list

- Excel：`SU_Others.xlsm`
    
    - シート `予定表_2024` と `予定表_2025`
        
    - **列A「企業名」**
        
- YAML：`environment.worker_company_list[]`
    

複数シートをまとめたイメージ：

- 列Aに出てくる会社名ごとに一つの会社ID（`wc1`, `wc2`, …）を作る
    
- 重複する会社名は **同じID** にまとめる
    
- YAMLでは例のような形になる：
    
    ```
    worker_company_list:   - id: wc1     
    name: ○○○○株式会社    
     annual_overtime_limit: 360   # 固定     
     monthly_overtime_limit: 40   # 固定     
     unavailable_dates: []        # 今は空
    ```
    

> 技術詳細は 2.1.1 SU_Others → worker_company_list を参照。

---

### 1.2.2 姓名 + 自由枠 + 日付けセル → worker_list

- Excel：`SU_Others.xlsm` (`予定表_2024` / `予定表_2025`)
    
    - **列B「姓名」** → 作業者の名前
        
    - **列E「自由枠」** → ここに「M/QC(責)」「M (責)」など「責」の文字があれば `is_manager: true`
        
    - 上部の **日付行（1〜数行のどこか）** → 予定の列（1/1, 1/2, …）
        
    - 各作業者行 × 日付列のセル：
        
        - **赤セル** → この日はその人は使えない（`unavailable_dates` に入る）
            
        - 赤以外で文字が入っているセル → 後で割当を作るために「何か作業している」として記録
            
- YAML：`environment.worker_list[]`
    

例（イメージ）：

```
worker_list:   - id: w001     
name: OO OO              # 列B     
worker_company: wc1          # 列Aから紐づけ     
is_manager: true             # 列Eに「責」が含まれる場合     
skill_map:                   # 今は全員フルスキルで仮置き       
	f22p1: 1       
	f22p2: 1       
	f22p3: 1       
	f22p4: 1       
	other_op: 1     
fab_suitability_map: []      # 今は空     
unavailable_dates:       
	- date: 2025/01/01         # 赤セルの日       
	- date: 2025/01/02
```

※ `予定表_2024` と `予定表_2025` は、同じロジックで **両方読んで合算するイメージ** です  
（現コードはデフォルトで `予定表_2024` を読む形ですが、運用としては同じ関数を `予定表_2025` に対しても呼び出してマージすることを想定）。

> 技術詳細は 2.1.2 SU_Others → worker_list & unavailable_dates を参照。

---

### 1.2.3 SU_Others のセル文字列 → 後の assignment_list / その他タスク

- Excel：`SU_Others.xlsm` の日付セル（**赤でない**セル）
    
    - 例：
        
        - `SG_830300179A＿Micron＿SG＿S/L新規`
            
        - `KST705wd->sd 530N01814A_852Z00771A F34_IRL_改造_I`
            
        - `MTG`, `OFF`, など
            
- 処理の考え方：
    
    1. セル文字列から、`530N02716A`, `830300179A` のような**ツールコード**を探す  
        （正規表現で抜き出し）
        
    2. そのコードが `F22_Tool Schedule` の装置コードと一致すれば → **F22ツールの作業**
        
    3. 一致しなければ → **「その他作業」ラベル**として扱う
        
- 後で `build_assignments` の中で、
    
    - ツールコード付き → `assignment_list`（Flexible）
        
    - コード無し → `wf_other` のダミータスク + `assignment_list`（Fixed）  
        を作る。
        

> 技術詳細は 2.3 build_assignments（割当生成） を参照。

---

## 1.3 F22_Tool Schedule から取っている情報

- Excel：`台湾出張者予定_2025latest.xlsx`
    
    - シート：`F22_Tool Schedule`
        

### 1.3.1 ツール行 → workflow_task_list（F22部分）

1. 行の基本情報
    
    - **列C**：TSMCツール名
        
    - **列D**：SCREENツール名  
        → どちらか（または両方）を `/` で繋いで `workflow_task_list[].name` に。
        
1. 装置コード
    
    - 例：`530N02716A`, `830300179A`, `852Z00771A`
        
    - SCREEN / TSMC ツール名の中から、正規表現で抽出して `module_code` として保持
        
    - YAMLには直接書かないが、SU_Othersと紐づけるために内部で使う
        
2. 各作業列（Power / Gas / Exh / …）の日付
    
    - ヘッダ行（上の方の1行）に `"Power"`, `"Gas"`, `"Exh"`, `"Through Running"` などが書かれている列を探す
        
    - その列のセルに書かれた日付・期間文字列（`2025/1/6`, `2025/1/6\n>1/8` など）を読み取り、
        
        - どの Phase（1〜4）に属するかをマップして、Phaseごとの日付リストを作る
            
3. Phase1〜4の期間調整
    
    - Phaseごとに「最初の日」と「最後の日」を決めたあと、
        
    - **必ず** `Phase1 → Phase2 → Phase3 → Phase4` の順番でつながるように、
        
        - Phase1 の終了日 = Phase2 の開始日 - 1日
            
        - Phase2 の終了日 = Phase3 の開始日 - 1日
            
        - Phase3 の終了日 = Phase4 の開始日 - 1日
            
        - Phase4 はそのまま
            
    - これで**前後の Phase がかぶらない綺麗な区切り**になる
        
4. YAML での形
    

```
workflow_task_list:   
	- id: f22_1     
	  name: "530N02716A / ADSTS8"  # SCREEN + TSMC ツール名     
	  workflow: wf_tool     
	  fab: f_tw     
	  phase_task_list:       
		  - id: f22_1_p1         
		    name: "Phase 1"         
		    phase: tool_p1         
		    start_date: 2025/01/01         
		    end_date:   2025/01/10         
		    operation_task_list:           
			    - id: f22_1_p1             
			      name: "Phase 1"             
			      operation: f22p1             
			      workload_days: 10       
			    - id: f22_1_p2         
			      ...       
			    - id: f22_1_p3       
			    - id: f22_1_p4
```


> 技術詳細は 2.2 parse_tool_schedule（F22_Tool Schedule 読み込み） を参照。

---

## 1.4 Schedule.plan_range はどこから来る？

`Schedule_from_excel.yaml` の

```
schedule:   
	plan_range:     
		start_date: ...     
		end_date:   ...
```


は、

1. SU_Others（予定表_2024 / 2025）の**日付ヘッダ**の最小/最大
    
2. F22_Tool Schedule の各作業列の日付（Power / Gas / Exh / …）
    

を全部集めて、その中の **最も早い日付**と**最も遅い日付**を取っています。

---

## 1.5 assignment_list はどう作っているか（考え方だけ）

### 1.5.1 ツールコードがあるセル → F22 の Flexible assignment

1. SU_Others の「赤くないセル」で文字列があるものを全部見る
    
2. そのセル文字列からツールコード（`530N02716A` など）を抜き出す
    
3. そのコードが F22_Tool Schedule 側の `module_code` と一致したら：
    
    - F22_Tool Schedule 上のその装置の Phase1〜4 の期間と照合して、
        
    - その日付が入っている Phase を見つける
        
    - 見つかった Phase ごとに、`plan_flexibility: Flexible` の assignment を作る
        

イメージ：

```
worker: w052   
operation_task: f22_8_p1   # 「f22_8 の Phase1」   
start_date: 2025/03/19   
end_date:   2025/03/26   
work_date_list:     
	- { hour: 12, date: 2025/03/19 }     
	  ...   
plan_flexibility: Flexible
  ```
  

### 1.5.2 ツールコードがないセル → その他作業（Fixed assignment）

- 装置コードが見つからない文字列（MTG, OFF, その他ラベル）は：
    
    1. ラベルごとに全日付を集めて、「その他タスク（misc_1, misc_2, ...）」として `workflow_task_list` に1行作る
        
    2. さらに「誰がそのラベルで働いていたか」ごとにまとめて、`plan_flexibility: Fixed` の assignment を作る
        

イメージ：

```
workflow_task_list:   
	- id: misc_1     
	  name: "MTG"     
	  workflow: wf_other     
	  fab: f_tw     
	  phase_task_list:       
	- id: misc_1_p1         
	  name: "Misc Phase"         
	  phase: other_p1         
	  start_date: 2025/02/01         
	  end_date:   2025/02/10         
	  operation_task_list:           
	- id: misc_1_p1             
	  name: "Misc Phase"             
	  operation: other_op             
	  workload_days: 10  
assignment_list:   
	- worker: w010     
	  operation_task: misc_1_p1     
	  start_date: 2025/02/02     
	  end_date:   2025/02/03     
	  work_date_list:       
		  - { hour: 8, date: 2025/02/02 }       
		  - { hour: 8, date: 2025/02/03 }     
	  plan_flexibility: Fixed
```


---

## 1.6 Excel に無いので「仮置き」している項目

以下は **Excelからは取っていない** 情報で、全部コード側の固定値です。

- `workflow_list` の中身（`wf_tool`, `wf_other`, `tool_p1..p4`, `other_p1`, `f22p1..4`, `other_op`）
    
- `fab_list` (`f_tw`), `region_list` (`r_tw`), `customer_company_list` (`c_tsmc`)
    
- `region_list[].max_stay_on`, `max_annual_stay`, `stay_off_interval`
    
- `worker_company_list[].annual_overtime_limit`, `monthly_overtime_limit`
    
- `worker_list[].skill_map`（全 Phase=1）
    
- assignment の `hour` 値
    
    - F22 = 12時間
        
    - その他 = 8時間
        
- `transite_day_map`（空配列）
    

このあたりを Excel 側に列を追加して持たせれば、将来的に「仮置き→実データ」に差し替えることができます。

---

# 2. 技術的な詳細

ここからは、実装イメージ（擬似コード）で説明します。  
実際の Python コードの変数名に近い名前を使っています。

---

## 2.1 `parse_su_others`（SU_Others 予定表を読み込む）

### 2.1.1 SU_Others → worker_company_list

```
function parse_su_others(path, sheet_name):     
	ws, df = load_sheet_as_df(path, sheet_name)     
	find date_row_idx = first row (0..4) that contains any datetime     
	date_row = df[date_row_idx]     
	date_cols = indices where date_row[c] is a datetime      
	
	worker_start_row = date_row_idx + 2      
	
	worker_company_map = {}   # company_name -> wcID     
	worker_company_list = []      
	
	for r in range(worker_start_row .. n_rows):         
		company = df[r, 0]   # 列A         
		name    = df[r, 1]   # 列B         
		if name is empty: continue          
		if company not in worker_company_map:             
			cid = "wc" + running_number             
			worker_company_map[company] = cid             
			append to worker_company_list:               
				{ id: cid, name: company, overtime limits = fixed }      
	return worker_company_list, ...
```


---

### 2.1.2 SU_Others → worker_list & unavailable_dates

```
for each worker row r:     
	company = df[r, 0]     
	name    = df[r, 1]     
	free_slot = df[r, 4]   # 列E「自由枠」      
	is_manager = (free_slot contains "責")      
	wid = "w" + 3-digit running number      
	unavailable_set = empty set     
	worker_date_map = global dict      
	for each date column c in date_cols:         
		date_val = date_row[c]         # ヘッダ行の日付         
		dt = to_timestamp(date_val)          
		cell = ws.cell(row=r+1, column=c+1)  # 色を見るために openpyxl を利用         
		if cell.fill is red:             
			add dt to unavailable_set             
			continue   # 赤セルは assignment に使わない          
		if cell has text:             
			worker_date_map[(wid, dt)] = that text      
			
	build skill_map = {f22p1:1, f22p2:1, f22p3:1, f22p4:1, other_op:1}      
	append to worker_list:         
		{           
			id: wid,           
			name: name,           
			worker_company: worker_company_map[company],           
			is_manager: is_manager,           
			skill_map: skill_map,           
			fab_suitability_map: [],           
			unavailable_dates: sorted(unavailable_set) formatted as "YYYY/MM/DD"         }
```

※ `予定表_2025` を扱う場合は、同じ処理を sheet_name=`"予定表_2025"` に対しても実行し、  
`worker_company_list`・`worker_list`・`worker_date_map` を **ID重複に注意しながらマージ**するイメージです。

---

### 2.1.3 SU_Others → 基本 plan_range

```
all_dates = all non-empty dates in date_row[date_cols] 
sort(all_dates)  

plan_range = {   
	start_date: min(all_dates),   
	end_date:   max(all_dates), 
}
```

---

## 2.2 `parse_tool_schedule`（F22_Tool Schedule 読み込み）

### 2.2.1 ヘッダ行と操作列の検出

```
df = load_sheet_as_df(tool_schedule_path, "F22_Tool Schedule")  
for r in first 30 rows:     
	row = df[r]     
	tmp_map = {}     
	for c, v in row:         
		if v is string:            
			key = OPS_NAME_TO_ID[_norm(v)]            
			if key exists:                
				tmp_map[c] = key   # v="Power" -> key="f22p1o1"     
	if len(tmp_map) >= 3:         
		header_row_idx = r         
		col_to_op_id = tmp_map         
		break
```

---

### 2.2.2 各ツール行 → module_code + Phase1〜4 の raw 日付

```
for r in range(header_row_idx+2 .. n_rows):     
	location   = df[r, 0]     
	tsmc_tool  = df[r, 2]     
	screen_tool= df[r, 3]      
	
	if location, tsmc_tool, screen_tool 全部空: continue      
	
	task_name = join non-empty of (screen_tool, tsmc_tool) with " / "      
	
	module_code = first tool code extracted from screen_tool or tsmc_tool      
	
	phase_dates = {1: [], 2: [], 3: [], 4: []}     
	
	row_dates   = []      
	
	for (c, op_id) in col_to_op_id:         
		cell = df[r, c]          
		if cell is datetime:             
			start_dt = end_dt = cell         
		else if cell is string:             
			if contains newline:                 
				parse first line as dt1, second line as dt2 (possibly month/day only)             
			else:                 
				parse as single date dt         
				
		else:             
			continue          
			
		phase_index = OP_PHASE_INDEX[op_id]  # 1..4         
		append start_dt and end_dt to phase_dates[phase_index]         
		append them also to row_dates and global all_dates
```

---

### 2.2.3 Phase1〜4 の期間を決定し「次Phaseの前日まで」にそろえる

```
row_start = min(row_dates) or None 
row_end   = max(row_dates) or None  
row_phase_meta = []  
for ph in 1..4:     
	if phase_dates[ph] not empty:         
		phase_start = min(phase_dates[ph])         
		phase_end   = max(phase_dates[ph])     
	else if row_start and row_end:         
		phase_start = row_start         
		phase_end   = row_end     
	else:         
		phase_start = phase_end = default("2025-01-01")      
	
	phase_id = f"{task_id}_p{ph}"     
	row_phase_meta.append({phase_index: ph, phase_id, start: phase_start, end: phase_end})  
	
sort row_phase_meta by phase_index  
for i in 0..2:     
	cur = row_phase_meta[i]     
	nxt = row_phase_meta[i+1]      
	new_end = nxt.start - 1 day     
	if new_end < cur.start:         
		new_end = cur.start      
	
	cur.end = new_end # 最後の phase の end はそのまま
```

---

### 2.2.4 workflow_task_list エントリ作成

```
phase_task_list = []  

for meta in row_phase_meta:     
	ph        = meta.phase_index     
	phase_id  = meta.phase_id     
	start_dt  = meta.start     
	end_dt    = meta.end     
	days      = (end_dt - start_dt) + 1      
	
	phase_task_list.append({         
		id: phase_id,         
		name: "Phase " + ph,         
		phase: "tool_p" + ph,         
		start_date: to_ymd(start_dt),         
		end_date:   to_ymd(end_dt),         
		operation_task_list: [             
			{               
				id: phase_id,               
				name: "Phase " + ph,               
				operation: "f22p" + ph,               
				workload_days: days             
			}         
		],     
	})      
	
	if module_code is not None:         
		module_to_phases[module_code].append({           
			phase_index: ph,           
			phase_id: phase_id,           
			start: start_dt,           
			end:   end_dt,         })  
	
tool_tasks.append({   
	id: task_id,   
	name: task_name,   
	workflow: "wf_tool",   
	fab: "f_tw",   
	phase_task_list: phase_task_list,   
	module_code: module_code,  # 後で YAML 出力前に削除 })
```

---

## 2.3 `build_assignments`（スケジュール割当生成）

### 2.3.1 SU_Others のセル文字列をツール or その他に分類

```
known_assign_map      = {}  # (wid, phase_id) -> [dates] 
misc_label_dates      = {}  # label -> set of dates 
misc_worker_label_dates = {}# (wid, label) -> [dates]  

for ((wid, dt), text) in worker_date_map:     
	code = extract_tool_code(text)   # 530N02716A 等      
	if code exists and code in module_to_phases:         
		for phase_meta in module_to_phases[code]:             
			if phase_meta.start <= dt <= phase_meta.end:                 
			known_assign_map[(wid, phase_meta.phase_id)].append(dt)     
	else:         
		label = text.strip()         
		misc_label_dates[label].add(dt)         
		misc_worker_label_dates[(wid, label)].append(dt)
```

---

### 2.3.2 F22 ツール割当（Flexible）

```
assignments = []  

for ((wid, phase_id), dates) in known_assign_map:     
	uniq_dates = sorted unique(dates)     
	if empty: continue      
	
	work_date_list = [{hour: 12, date: to_ymd(d)} for d in uniq_dates]      
	
	assignments.append({       
		worker: wid,       
		operation_task: phase_id,       
		start_date: to_ymd(first(uniq_dates)),       
		end_date:   to_ymd(last(uniq_dates)),       
		work_date_list: work_date_list,       
		plan_flexibility: "Flexible",     
	})
```

---

### 2.3.3 その他作業のダミータスク + Fixed 割当

```
misc_tasks = [] 
misc_label_to_phase = {} 
misc_counter = 1 
 
# 1) ラベルごとのタスク定義 
for (label, dates) in misc_label_dates:     
	start = min(dates)     
	end   = max(dates)     
	task_id  = "misc_" + misc_counter     
	phase_id = task_id + "_p1"     
	misc_counter++      
	
	misc_label_to_phase[label] = {phase_id, start, end}      
	
	workload_days = (end - start) + 1      
	
	misc_tasks.append({       
		id: task_id,       
		name: label,       
		workflow: "wf_other",       
		fab: "f_tw",       
		phase_task_list: [{         
			id: phase_id,         
			name: "Misc Phase",         
			phase: "other_p1",         
			start_date: to_ymd(start),         
			end_date:   to_ymd(end),         
			operation_task_list: [{           
				id: phase_id,           
				name: "Misc Phase",           
				operation: "other_op",           
				workload_days: workload_days,         
			}]       
		}]     
	})  
	
# 2) 各人ごとの Fixed assignment 
for ((wid, label), dates) in misc_worker_label_dates:     
	uniq_dates = sorted unique(dates)     
	if empty: continue      
	
	phase_meta = misc_label_to_phase[label]     
	phase_id   = phase_meta.phase_id      
	
	work_date_list = [{hour: 8, date: to_ymd(d)} for d in uniq_dates]      
	assignments.append({       
		worker: wid,       
		operation_task: phase_id,       
		start_date: to_ymd(first(uniq_dates)),       
		end_date:   to_ymd(last(uniq_dates)),       
		work_date_list: work_date_list,       
		plan_flexibility: "Fixed",     })
```

---

## 2.4 `build_env_and_schedule`（最終的な YAML をまとめる）

```
su_data   = parse_su_others(su_others_path, sheet="予定表_2024") 
# もし 2025 も使うなら: 
# su_data_2025 = parse_su_others(... sheet="予定表_2025") 
# → worker_company_list / worker_list / worker_date_map / plan_range をマージ  

tool_data = parse_tool_schedule(tool_schedule_path, sheet="F22_Tool Schedule")  

# ENVIRONMENT 
wf_tool_phases = [] for ph in 1..4:     
wf_tool_phases.append({       
	id: "tool_p" + ph,       
	name: "Phase " + ph,       
	operation_list: [{         
		id: "f22p" + ph,         
		name: "F22 Phase " + ph,         
		work_hours: [8],         
		min_worker_num: 1,         
		max_worker_num: 3,       
	}]     
})  

environment = {   
	workflow_list: [wf_tool, wf_other],   
	fab_list: [...],   
	region_list: [...],   
	customer_company_list: [...],   
	worker_company_list: su_data.worker_company_list,   
	worker_list: su_data.worker_list,   
	transite_day_map: [], 
}  

# SCHEDULE all_dates = [   
	su_data.plan_range.start_date,   
	su_data.plan_range.end_date,   
	*tool_data.date_list 
] 

plan_range = {   
	start_date: min(all_dates),   
	end_date:   max(all_dates), 
}  

tool_tasks_for_yaml = tool_data.tool_tasks with module_code removed  

assignments, misc_tasks = build_assignments(su_data, tool_data)  

schedule = {   
	plan_range: plan_range,   
	workflow_task_list: tool_tasks_for_yaml + misc_tasks,   
	assignment_list: assignments, 
}  

write_yaml("EnvConfig_from_excel.yaml", {environment}) 
write_yaml("Schedule_from_excel.yaml",  {schedule})
```

---

