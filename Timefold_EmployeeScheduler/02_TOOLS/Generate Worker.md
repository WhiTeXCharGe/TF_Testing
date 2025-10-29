# Tool: `generate_worker.py`

A small, deterministic generator that overwrites `environment.worker_list` in your `EnvConfig.yaml`. It’s designed for **quickly fabricating test workers** with realistic distributions (skills, levels, regions, availability, manager ratio) so you can stress-test your Timefold solver without hand-editing YAML.

---

## ✅ What it does

- Reads `EnvConfig.yaml` (and optionally `Schedule.yaml` for `plan_range`).
    
- Collects all operation IDs (e.g., `p1o1`, `p2o2`, …) from `workflow_list`.
    
- Creates `N` workers with:
    
    - IDs `w1..wN`, names in **AA, AB, AC, …** order.
        
    - Random **manager** flags matching a target percentage.
        
    - **Skill maps** in **inline flow YAML** style: `{p1o1: 3, p2o2: 1, ...}`.
        
    - **Balanced skill coverage** across operations (near-even distribution).
        
    - **Worker company** chosen randomly from `worker_company_list`.
        
    - **Region suitability** as an inline list: `[r1, r2]` (can be empty).
        
    - **Personal unavailable dates** (optional) within `plan_range`.
        
- Writes back to `EnvConfig.yaml` (preserving the outer `environment` wrapper if present).
    

---

## 🛠 Usage

`# Minimal python generate_worker.py --env EnvConfig.yaml  # Typical python generate_worker.py \   --env EnvConfig.yaml \   --schedule Schedule.yaml \   --num 100 \   --mgr 30 \   --skill-min 3 --skill-max 5 \   --levels 1,2,3,4,5 \   --level-weights 10,20,40,20,10 \   --region-cardinality "0:0.2,1:0.6,2:0.2" \   --unavail-emp 20 --unavail-max 3 \   --seed 42`

**Key flags**

- `--env` (required): path to `EnvConfig.yaml`.
    
- `--schedule` (optional): reads `schedule.plan_range` for date generation.
    
- `--num`: number of workers (default 80).
    
- `--mgr`: managers % (default 50.0). Randomly assigned across the population.
    
- `--skill-min`, `--skill-max`: per-employee distinct skill count.
    
- `--levels`: allowed levels (comma list).
    
- `--level-weights`: selection weights aligned to `--levels`.
    
- `--region-cardinality`: probabilities for how many regions an employee has (e.g., `"0:0.8,1:0.15,2:0.05"`).
    
- `--unavail-emp`, `--unavail-max`: % of employees with off-days and the per-employee max count.
    
- `--plan-start`, `--plan-end`: fallback date range if `--schedule` is not supplied.
    
- `--seed`: for reproducible results.
    

> All of these have sensible **defaults** at the top of the script so you can edit once and just run without flags.

---

## 📦 Output shape (inline/flow style)

Each worker is emitted like:

`- id: w37   name: AK   worker_company: c3   is_manager: false   skill_map: {p1o2: 3, p2o1: 2, p3o1: 4}   fab_suitability_map: [r1, r2]   unavailable_dates: [2025/11/02, 2025/12/05]`

Notes:

- **`skill_map` uses `{...}`** as requested.
    
- **`fab_suitability_map` and `unavailable_dates` use `[ ... ]`** inline lists.
    
- Companies are drawn from `environment.worker_company_list[].id`.
    
- Regions are drawn from `environment.region_list[].id`.
    

---

## ⚖️ Balancing logic (skills)

- The tool computes the total number of skill “slots” across all workers (based on per-employee `K ~ U(skill_min, skill_max)`).
    
- It then **distributes these slots nearly evenly** across all operations discovered under `workflow_list → phase_list → operation_list → id`.
    
- A shuffle + “bag” mechanism ensures each operation receives close to its target coverage before any operation is over- or under-represented.
    
- Levels are drawn **per skill** using the global `--levels` + `--level-weights`.
    

This keeps coverage reasonable (e.g., avoids `{p1o1}` ending up with 100 workers while `{p2o1}` has only 10).

---

## 👥 Managers

- Manager slots are **randomly sampled** among all employees to match `--mgr` percent (rounded).
    
- This avoids the “first N are all managers” pattern.
    

---

## 🌏 Regions (fab suitability)

- `--region-cardinality` controls how many regions an employee gets (0, 1, 2, …).
    
- For example: `"0:0.2,1:0.6,2:0.2"` means 20% none, 60% exactly one region, 20% exactly two regions.
    
- When `k > 0`, regions are random distinct choices from `region_list`.
    

---

## 📅 Unavailable dates (personal off)

- If `--schedule` is provided and contains `schedule.plan_range`, those dates are used as the pool.
    
- Otherwise, `--plan-start` / `--plan-end` are used.
    
- `--unavail-emp` controls **what percentage** of employees get off-days.
    
- `--unavail-max` controls **the maximum count** of off-days per such employee.
    
- Dates render inline: `[2025/11/01, 2025/12/19, 2025/12/28]`.
    

---

## 🔁 Determinism

- Pass `--seed <int>` to reproduce the exact same worker list (names, managers, skills, levels, regions, dates).
    
- Default seed is set at the top of the script; set to `None` for non-deterministic runs.
    

---

## 🧪 Quick recipes

**Generate 120 workers, 40% managers, skills 3–5, more senior bias**

`python generate_worker.py --env EnvConfig.yaml --num 120 --mgr 40 \   --skill-min 3 --skill-max 5 \   --levels 1,2,3,4,5 \   --level-weights 5,15,30,30,20 \   --schedule Schedule.yaml \   --seed 123`

**No personal off-days, but wider region spread**

`python generate_worker.py --env EnvConfig.yaml --num 90 \   --unavail-emp 0 --unavail-max 0 \   --region-cardinality "0:0.4,1:0.4,2:0.2"`

---

## ❗ Troubleshooting

- **“No operation ids found …”**  
    Your `workflow_list` is empty or operation `id` fields are missing. Fill `phase_list[].operation_list[].id` with IDs like `p1o1`, `p2o2`, etc.
    
- **Company IDs missing**  
    If `worker_company_list` is empty, the tool falls back to `["c2"]`. Define companies under `environment.worker_company_list`.
    
- **Min skills not honored**  
    This happens if duplicates sneak in before uniqueness. The loop now **tops up** from the remaining op pool until `K` unique skills is reached (or all ops are exhausted).
    
- **YAML not inline**  
    The script registers custom PyYAML representers for **FlowDict** and **FlowList**. Ensure you’re running this exact version.
    

---

## 📁 Where it writes

- Overwrites `environment.worker_list` (or `root.worker_list` if there’s no `environment` wrapper) **in place** at `--env`.
    

---

## ✍️ Naming & IDs

- IDs: `w1`, `w2`, …, `wN`
    
- Names: `AA, AB, AC, …, AZ, BA, BB, …` (i.e., spreadsheet-like sequence starting from two letters, not `AA, BB, CC`).
    

---

## 📌 Tips

- Keep `--skill-min` close to `--skill-max` if you want tighter per-employee variance.
    
- Adjust `--level-weights` to simulate workforce seniority.
    
- Use `--seed` during iteration so diffs are meaningful in Git.
    

---

If you want, I can also add a **`--company-weights`** flag (e.g., `"c2:0.7,c3:0.3"`) or enforce **at least one manager per op** coverage—just say the word.