You’re 100% right: the root cause is **this line in `build_assignments_v3()`**:

```py
if k in orig_map:
    text = orig_map[k]["old"]
code = extract_tool_code(text)
```

You “restore” the **old** text (which still contains the real tool code), so `extract_tool_code()` finds it again → the cell gets treated as a **real tool task (Flexible)** even though you already broke it to dummy in `worker_date_map`.

That also explains your workload bug: those “cut” cells get counted into the real module workload, and the dummy task workload/range becomes wrong.

Below is the exact fix (2 parts):

---

## ✅ Fix 1: cut cells must stay dummy in Schedule.yaml assignment_list

### Replace the NORMAL CELLS loop in `build_assignments_v3()` with this version

Find this part:

```py
# 1) NORMAL CELLS
for (wid, dt), text in worker_date_map.items():
    ...
    k = (wid, _to_ymd(dt))
    if k in orig_map:
        text = orig_map[k]["old"]
    code = extract_tool_code(text)
    ...
    label = text.strip() ...
```

### Replace it with this (important: **code from NEW, label from OLD**):

```py
# 1) NORMAL CELLS
for (wid, dt), raw_text in worker_date_map.items():
    if f_start is not None and f_end is not None and (dt < f_start or dt > f_end):
        continue

    # IMPORTANT:
    # - use internal_text (possibly broken by outlier-cut) for code extraction
    # - use display_text (original) for label and outputs
    internal_text = raw_text
    display_text = raw_text

    k = (wid, _to_ymd(dt))
    if k in orig_map:
        display_text = orig_map[k]["old"]     # original text for label/log/output
        internal_text = orig_map[k]["new"]    # broken text for matching (dummy behavior)

    code = extract_tool_code(internal_text)

    if code and (code in valid_code_set) and (code in shifted_code_to_phases):
        matched = False
        for phase_meta in shifted_code_to_phases[code]:
            ds = phase_meta.get("date_set")
            if ds is not None:
                if dt in ds:
                    phase_id = phase_meta["phase_id"]
                    known_assign_map[(wid, phase_id)].append(dt)
                    inferred_worker_phase[wid].add(phase_meta["operation"])
                    matched = True
                    break
            else:
                ps = phase_meta["start"]
                pe = phase_meta["end"]
                if ps is None or pe is None:
                    continue
                if ps <= dt <= pe:
                    phase_id = phase_meta["phase_id"]
                    known_assign_map[(wid, phase_id)].append(dt)
                    inferred_worker_phase[wid].add(phase_meta["operation"])
                    matched = True
                    break
        if matched:
            continue

    # otherwise: dummy "other" (Fixed)
    label = display_text.strip() if isinstance(display_text, str) else ""
    if not label:
        label = "other"
    misc_label_dates[label].add(dt)
    misc_worker_label_dates[(wid, label)].append(dt)

    # for log: tool-code-ish labels that are not in 新規製番リスト valid set
    # NOTE: use code extracted from DISPLAY (original) only for reporting, not matching
    code_disp = extract_tool_code(display_text)
    if code_disp and code_disp not in valid_code_set:
        dummy_tool_labels[code_disp].add(label)
```

### What this changes

* A cut cell’s `internal_text` contains broken code like `...A -> ...X` → `extract_tool_code()` returns `None`
* So it **cannot** map to tool phases anymore
* It becomes **misc “other”** and therefore **Fixed**, which is what you want for worker C/D (single-day / non-continuous).

This alone fixes the “WorkerB still becomes real task even after cut” issue.

---

## ✅ Fix 2: misc/personal workload_days must be worker-days (not window range)

Right now you do:

```py
workload_days = int((end - start).days) + 1
```

That’s exactly why your dummy task “TSMC_530N02046A” becomes 02–08 => 7 days, even if only 2 worker-days existed.

### Add worker-day counting for misc labels (before building misc_tasks)

Right before:

```py
# 4) MISC ("OTHER") TASKS + FIXED ASSIGNMENTS
misc_tasks = []
...
for label, dates in misc_label_dates.items():
```

Insert this:

```py
# worker-day counts for misc labels (unique wid+date)
misc_label_workerday = defaultdict(int)
tmp_label_wid_dates = defaultdict(set)  # label -> set((wid, dt))

for (wid, label), dates in misc_worker_label_dates.items():
    for d in set(dates):
        tmp_label_wid_dates[label].add((wid, d))

for label, s in tmp_label_wid_dates.items():
    misc_label_workerday[label] = len(s)
```

### Then change misc workload_days to use that count

Replace:

```py
workload_days = int((end - start).days) + 1
```

With:

```py
workload_days = int(misc_label_workerday.get(label, 0))
if workload_days <= 0:
    # fallback safety
    workload_days = int((end - start).days) + 1
```

---

### Do the same for personal tasks (recommended)

Before personal task creation loop, insert:

```py
pb_label_workerday = defaultdict(int)
tmp_pb_label_wid_dates = defaultdict(set)

for (wid, label), dates in personal_worker_label_dates.items():
    for d in set(dates):
        tmp_pb_label_wid_dates[label].add((wid, d))

for label, s in tmp_pb_label_wid_dates.items():
    pb_label_workerday[label] = len(s)
```

And replace personal workload:

```py
workload_days = int(pb_label_workerday.get(label, 0))
if workload_days <= 0:
    workload_days = int((end - start).days) + 1
```

---

## Result on your example

* A: 01–10 real tool → counts into module (10)
* B: 05–10 real tool → counts into module (6)
  ✅ flexible total = 16 worker-days (you wrote 15 with “10+5” but B from 05–10 is 6 days; if you meant 05–09 then it’s 5)
* C: only 08 → cut → dummy fixed (1)
* D: only 02 → cut → dummy fixed (1)

Dummy task range = **02–08** (still okay as the task window), but workload_days = **2** (correct worker-days).

And most important: C/D no longer inflate the real module workload.

---

If you paste just the `build_assignments_v3()` function into the chat (only that function), I can return a clean “final version” of the whole function with these changes already integrated so you can copy-paste safely.
