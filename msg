You *can* delete it, but I **don’t recommend deleting the whole per-worker clustering block**, because then drag-fill noise comes back and your shifting/workload can explode again.

What you actually want is simpler:

* ✅ **Keep** *all* clusters that look “real work” (ex: cluster has ≥2 consecutive days, or at least length ≥2)
* ❌ **Cut** only tiny/noisy clusters (single day, or no consecutive run)

Right now your code does this (too aggressive):

* pick **best cluster only**
* cut everything else → reason = `"outside worker main cluster"`

That’s why a valid second cluster (even 2+ days) gets cut.

---

## Don’t delete — replace just the “keep only best cluster” rule

Find this part:

```python
# Otherwise keep ONLY dates in best cluster for this worker; cut the rest
best_dates = set(best)
for dt, text in wid_occ:
    if dt in best_dates:
        continue
    ...
    reason = f"per-worker cut: outside worker's main cluster (kept {len(best_dates)} days)"
```

### Replace it with this logic:

✅ Keep *any* cluster that has a consecutive run >= 2
❌ Cut clusters that don’t

```python
# Keep ALL "good" clusters; cut only "bad" clusters.
# A "good" cluster = has at least 2 consecutive days inside it.

def _cluster_has_run2(cluster_dates):
    return _longest_consecutive_run(cluster_dates) >= 2

# Decide which dates to keep for this worker+code
keep_dates = set()
for cl in wid_clusters:
    if _cluster_has_run2(cl):
        keep_dates.update(cl)

# If nothing qualifies, fall back to old behavior: cut all
if not keep_dates:
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
            "reason": "per-worker cut: no cluster has >=2 consecutive days (treated as outliers)",
        })
    continue

# Cut only dates NOT in keep_dates
for dt, text in wid_occ:
    if dt in keep_dates:
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
        "reason": "per-worker cut: cluster is too small / not consecutive (outlier)",
    })
```

### What this changes

* If Worker has **two real periods** like `05-10` and `20-25`, both stay **Flexible**
* Only “random crumbs” like `08` alone get turned into dummy **Fixed**
* The `"outside worker main cluster"` message mostly disappears (because we no longer force “one cluster only”)

---

## If you *really* want to delete something

Delete **only** the “cut rest outside best cluster” part, not the whole outlier system.

But the replacement above is the safe version that matches your requirement.

If you paste the exact block from your file around the “outside worker main cluster” section, I can point the exact start/end lines to replace (no extra changes).
