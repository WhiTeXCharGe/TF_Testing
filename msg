Got it — both changes fit cleanly into the current Decoder3 structure. Below is a **drop-in patch** (minimal edits) that:

1. Adds **“distance-based cut-out”** using configurable `CUT_DISTANCE_DAYS`
2. Improves the **workload warning section** to include **module code** (not only phase_id)

---

## 1) Add new cut-out condition: “actual span too far from plan”

### A) Add config near the top

Put this under your existing CONFIG block:

```python
# If SU_Others actual span is completely outside the planned window
# by more than this many days, cut the module from 新規製番リスト
CUT_DISTANCE_DAYS = 40
```

### B) Add helper to compute the “gap”

Put this somewhere near helpers (anywhere before `build_env_and_schedule_decoder3` is fine):

```python
def _planned_actual_gap_days(planned_start, planned_end, actual_start, actual_end) -> int:
    """
    If actual overlaps planned -> gap=0.
    If actual is fully before planned -> gap = planned_start - actual_end
    If actual is fully after planned -> gap = actual_start - planned_end
    """
    if planned_start is None or planned_end is None or actual_start is None or actual_end is None:
        return 0
    if actual_end < planned_start:
        return int((planned_start - actual_end).days)
    if actual_start > planned_end:
        return int((actual_start - planned_end).days)
    return 0
```

### C) Apply distance cut *after* building `shifted_meta`, but *before* building tasks/assignments

Inside `build_env_and_schedule_decoder3`, right after:

```python
shifted_meta, code_to_shifted_phases_simple, _code_occ = build_shifted_meta(planned_meta, su_data if read_all_data else None)
```

add this block:

```python
    # 5.5) EXTRA CUT: actual span too far from planned window
    cut_due_to_distance = []
    if read_all_data and su_data is not None:
        for code, m in list(shifted_meta.items()):
            if not m.get("had_su_match"):
                continue

            plan = m["plan"]
            gap = _planned_actual_gap_days(
                plan["overall_start"], plan["overall_end"],
                m["actual_first"], m["actual_last"]
            )

            if gap > CUT_DISTANCE_DAYS:
                cut_due_to_distance.append((code, gap))

        if cut_due_to_distance:
            for code, gap in cut_due_to_distance:
                cut_rows.append((code, f"actual span is outside planned window by {gap} days (> {CUT_DISTANCE_DAYS})"))

                # remove it so it behaves like "not listed"
                planned_meta.pop(code, None)
                shifted_meta.pop(code, None)

            # recompute valid_code_set and valid_codes list
            valid_code_set = set(planned_meta.keys())
            task_meta["valid_codes"] = [c for c in task_meta["valid_codes"] if c in valid_code_set]
```

✅ Behavior with this:

* If the SU_Others span is **fully before** or **fully after** the planned span **and** the gap is `> CUT_DISTANCE_DAYS`, it is cut.
* If there is any overlap, or gap is small enough, it remains and shifting happens normally.
* Cut module occurrences in SU_Others become **dummy “other”**, because it’s no longer in `valid_code_set`.

---

## 2) TransformationLog: workload warning should include module code

### A) Build a `phase_id -> module_code` map

In `build_env_and_schedule_decoder3`, before the workload overwrite section, create a mapping.

Right after you build `tool_tasks_for_yaml` (or after `tool_tasks` is finalized), add:

```python
    # Map phase_id (eN_pX) -> module code (task name)
    phase_id_to_module_code = {}
    for t in tool_tasks_for_yaml:
        mod = t.get("name", "")
        for pt in t.get("phase_task_list", []):
            pid = pt.get("id")
            if pid:
                phase_id_to_module_code[pid] = mod
```

### B) Replace `workload_zero_phase_ids` with info tuples

In the workload overwrite section, replace:

```python
    workload_zero_phase_ids = []
```

with:

```python
    workload_zero_phase_info = []  # list of (phase_id, module_code)
```

Then inside the loop where you detect `b == 0`, replace:

```python
            if b == 0:
                workload_zero_phase_ids.append(phase_id)
```

with:

```python
            if b == 0:
                workload_zero_phase_info.append((phase_id, phase_id_to_module_code.get(phase_id, "")))
```

And when calling the log writer, replace:

```python
        workload_zero_phase_ids=sorted(set(workload_zero_phase_ids)),
```

with:

```python
        workload_zero_phase_info=sorted(set(workload_zero_phase_info)),
```

### C) Update `write_transformation_log` signature + output

Change the function signature:

```python
def write_transformation_log(
    out_path: str,
    cut_rows: list,
    shifted_meta: dict,
    worker_id_to_name: dict,
    workload_zero_phase_ids: list,
    dummy_tool_labels: dict,
):
```

to:

```python
def write_transformation_log(
    out_path: str,
    cut_rows: list,
    shifted_meta: dict,
    worker_id_to_name: dict,
    workload_zero_phase_info: list,  # [(phase_id, module_code)]
    dummy_tool_labels: dict,
):
```

Then update the “WORKLOAD WARNING” section to print both:

```python
    lines.append("---------------------- WORKLOAD WARNING (worker-days == 0) ----------------------")
    if not workload_zero_phase_info:
        lines.append("(none)")
    else:
        for pid, mod in workload_zero_phase_info:
            if mod:
                lines.append(f"- phase_id: {pid} / module: {mod} (no assigned worker-days in SU_Others after shifting)")
            else:
                lines.append(f"- phase_id: {pid} (no assigned worker-days in SU_Others after shifting)")
    lines.append("")
```

✅ Now warnings are searchable by both `eN_pX` and module code like `530N02704A`.

---

If you want one more quality-of-life improvement (no extra logic): in the CUT OUT section, I can group reasons by type and print counts at the top (makes Ctrl+F and reviewing easier), but I didn’t add it since you didn’t ask.
