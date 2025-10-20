## 1) Skeletons

### Hard constraint (filter + penalize)

`return (     cf.for_each(Entity)       .filter(lambda e: violates(e))       .penalize(HardMediumSoftScore.ONE_HARD)       .as_constraint("hard-name") )`

### Hard with magnitude

`.penalize(     HardMediumSoftScore.ONE_HARD,     lambda e: int(magnitude(e))  # cast! )`

### Soft constraint

`.penalize(     HardMediumSoftScore.ONE_SOFT,     lambda e: int(cost(e)) ).as_constraint("soft-name")`

---

## 2) Joins (safe)

### Equality join

`.join(cf.for_each(Other), Joiners.equal(lambda a: a.key, lambda b: b.key))`

### Filtering join (range / activity)

`.join(     cf.for_each(Other),     Joiners.filtering(lambda a, b: int(b.start) <= int(a.day) <= int(b.end)) )`

---

## 3) `group_by` Basics (safe)

> Prefer multiple scalar keys; cast inside lambdas.

### Count via sum(1)

`.group_by(     lambda a, b: key1,     lambda a, b: key2,     ConstraintCollectors.sum(lambda a, b: 1) ) .filter(lambda k1, k2, n: int(n) > limit) .penalize(HardMediumSoftScore.ONE_HARD, lambda k1, k2, n: int(n) - limit)`

### Sum

`.group_by(     lambda a, b: key1,     lambda a, b: key2,     ConstraintCollectors.sum(lambda a, b: int(value(b))) ) .filter(lambda k1, k2, s: s > cap) .penalize(HardMediumSoftScore.ONE_HARD, lambda k1, k2, s: int(s - cap))`

### Max / Min

`.group_by(     lambda a, b: key1,     ConstraintCollectors.max(lambda a, b: int(expr1)),     ConstraintCollectors.min(lambda a, b: int(expr2)) ) .filter(lambda k, mx, mn: mx > mn) .penalize(HardMediumSoftScore.ONE_SOFT, lambda k, mx, mn: int(mx - mn))`

### Distinct count (safe)

`.group_by(     lambda a, b: key1,     lambda a, b: key2,     ConstraintCollectors.count_distinct(lambda a, b: distinct_key(b)) ) .filter(lambda k1, k2, c: int(c) > 1) .penalize(HardMediumSoftScore.ONE_HARD, lambda k1, k2, c: int(c) - 1)`

---

## 4) Common Patterns

### Window containment

`cf.for_each(Block)   .filter(lambda b: b.start is None or b.days is None                     or int(b.start) < int(b.win_start)                     or (int(b.start) + int(b.days) - 1) > int(b.win_end))   .penalize(HardMediumSoftScore.ONE_HARD)`

### Capacity per (day, key)

`cf.for_each(Day)   .join(cf.for_each(Block),         Joiners.filtering(lambda d, b: int(b.start) <= int(d.id) <= int(b.start) + int(b.days) - 1))   .group_by(lambda d, b: int(d.id),             lambda d, b: b.key,             ConstraintCollectors.sum(lambda d, b: int(b.amount)))   .filter(lambda day, key, total: total > cap_for(key))   .penalize(HardMediumSoftScore.ONE_HARD, lambda day, key, total: int(total - cap_for(key)))`

### Overtime over threshold

`.group_by(lambda sd, cs: (int(cs.employee.id), int(sd.day.id)),           ConstraintCollectors.sum(lambda sd, cs: int(sd.hours))) .filter(lambda key, tot: tot > 8) .penalize(HardMediumSoftScore.ONE_SOFT, lambda key, tot: int(tot - 8))`

### Balance around target

`.group_by(lambda sd, cs: int(cs.employee.id),           ConstraintCollectors.sum(lambda sd, cs: int(sd.hours))) .penalize(HardMediumSoftScore.ONE_SOFT, lambda emp, tot: int(abs(tot - TARGET)))`

---

## 5) Weighting Tips

- Use **large weights** for feasibility blockers; keep softs an order smaller.
    
- Prefer **linear penalties** first; only introduce quadratic terms if needed.
    
- Keep names stable: `"hard-…"`, `"med-…"`, `"soft-…"`, to aid diffing and logs.
    

---

## 6) Safety Checklist

-  All numeric lambdas cast with `int(...)`.
    
-  No tuple keys in `group_by` (unless verified).
    
-  Counts via `sum(lambda: 1)`.
    
-  Activity tests use `Joiners.filtering(...)`.
    
-  No cross-pass leakage (Pass 1 constraints do not reference employee/manager facts).