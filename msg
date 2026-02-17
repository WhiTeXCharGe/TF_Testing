Yep — your current logic is **module-level** (“cluster by all workers’ dates together”), so it will *keep* worker B’s one-off day for `task1` just because worker A has a long continuous run for the same tool code.

You want **employee-level** cutting:

* For each `(tool_code, worker)`:

  * Look only at that worker’s dates for that tool code.
  * If that worker’s dates are **not continuous enough** (ex: single isolated day), **cut those cells into dummy** (break tool code).
  * Do **not** let other workers “save” that cell.

Below is a clean patch that changes the clustering logic to be **per worker**, while keeping your existing “planned-window far cut” step.

---

## Fix: make outlier cut per employee (wid) instead of per module

### Replace the “cluster by dates” part in `cut_su_outlier_cells()`

Inside `cut_su_outlier_cells`, find this block (roughly):

```python
# Unique dates for clustering
dates = sorted(set([x[0] for x in occ]))
...
clusters = _cluster_by_date_gap(dates, gap_days=cluster_gap_days)
...
# Score each cluster by worker-days in that date set
date_to_count = defaultdict(int)
for dt, _wid, _text in occ:
    date_to_count[dt] += 1
...
# Rewrite occurrences not in kept dates
...
for dt, wid, text in occ:
    if dt in kept_dates:
        continue
    ...
```

…and replace that whole clustering section with **this per-worker version**:

```python
        # ------------------------------------------------------------
        # PER-WORKER OUTLIER CUT (your requested behavior)
        # For each (code, wid), cluster that worker's dates ONLY.
        # If the worker's best cluster is not "continuous enough" (run < 2),
        # cut ALL that worker's occurrences for this code into dummy.
        # Otherwise keep only the best cluster for that worker and cut the rest.
        # ------------------------------------------------------------

        occ_by_wid = defaultdict(list)  # wid -> list of (dt, text)
        for dt, wid, text in occ:
            occ_by_wid[wid].append((dt, text))

        broken = _break_tool_code(code)

        for wid, wid_occ in occ_by_wid.items():
            # unique sorted dates for THIS worker only
            wid_dates = sorted(set(dt for dt, _ in wid_occ))
            if len(wid_dates) <= 1:
                # single-day involvement => treat as outlier (cut it)
                for dt, text in wid_occ:
                    if not isinstance(text, str):
                        continue
                    old = text
                    new = old.replace(code, broken, 1)
                    if new == old:
                        continue
                    _remember_original_text(su_data, wid, dt, old, new)
                    worker_date_map[(wid, dt)] = new
                    corrections.append({
                        "wid": wid,
                        "date": _to_ymd(dt),
                        "code": code,
                        "text": old,
                        "reason": "per-worker cut: only 1 day for this worker on this module (treated as outlier)",
                    })
                continue

            # cluster THIS worker's dates
            wid_clusters = _cluster_by_date_gap(wid_dates, gap_days=cluster_gap_days)

            # pick "best cluster" for this worker:
            # prefer largest uniq-days, then earliest start
            wid_clusters.sort(key=lambda cl: (len(cl), -int(cl[0].timestamp())), reverse=True)
            best = wid_clusters[0]

            # longest consecutive run INSIDE best cluster
            best_run = _longest_consecutive_run(best)

            # If this worker does not have a consecutive run of >=2 days,
            # cut ALL occurrences for this worker for this code.
            if best_run < 2:
                for dt, text in wid_occ:
                    if not isinstance(text, str):
                        continue
                    old = text
                    new = old.replace(code, broken, 1)
                    if new == old:
                        continue
                    _remember_original_text(su_data, wid, dt, old, new)
                    worker_date_map[(wid, dt)] = new
                    corrections.append({
                        "wid": wid,
                        "date": _to_ymd(dt),
                        "code": code,
                        "text": old,
                        "reason": f"per-worker cut: no consecutive run (best_run={best_run}) for this worker on this module",
                    })
                continue

            # Otherwise keep ONLY dates in best cluster for this worker; cut the rest
            best_dates = set(best)
            for dt, text in wid_occ:
                if dt in best_dates:
                    continue
                if not isinstance(text, str):
                    continue
                old = text
                new = old.replace(code, broken, 1)
                if new == old:
                    continue
                _remember_original_text(su_data, wid, dt, old, new)
                worker_date_map[(wid, dt)] = new
                corrections.append({
                    "wid": wid,
                    "date": _to_ymd(dt),
                    "code": code,
                    "text": old,
                    "reason": f"per-worker cut: outside worker's main cluster (kept {len(best_dates)} days)",
                })

        # Done for this code
        continue
```

✅ What this does in your example:

* Worker A: `task1` on 01–09 → has consecutive runs → **kept**
* Worker B: `task1` only on 05 → only 1 day / no run → **cut into dummy**
* Worker B: other codes are irrelevant here; they do not protect the `task1` cell anymore

---

## Notes you should be aware of

* This change is intentionally **stricter**: any worker with only 1 day on a module will be cut.
* If you want “allow 1-day participation when it is near the worker’s main cluster” (some teams do this), tell me your exact rule (e.g., “allow single day if within ±2 days of the worker’s main cluster”) and I’ll adjust it.

If you paste your current `cut_su_outlier_cells()` and tell me exactly where you want this inserted (line region), I can return a tighter “diff-style” patch.
