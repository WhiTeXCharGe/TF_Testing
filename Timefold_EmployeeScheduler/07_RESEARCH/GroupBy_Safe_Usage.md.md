## Golden Rules

1. **Match arity exactly**  
    Keys produced by `group_by` must match the parameters of subsequent `filter` / `penalize` lambdas.
    
2. **Prefer scalar keys**  
    Use multiple scalar keys (`k1, k2, …`) instead of a single tuple key, unless tuple keys are explicitly supported and tested.
    
3. **Count via `sum(lambda …: 1)`**  
    For counts, prefer `ConstraintCollectors.sum(lambda …: 1)` to avoid collector/arity surprises.
    
4. **Be explicit with types**  
    Cast inside lambdas (e.g., `int(...)`), avoid list/None truthiness.
    
5. **Keep grouping keys simple**  
    Put arithmetic and range checks in `filter` or collector lambdas, not inside key lambdas.
    
6. **Use direct joiners for activity tests**  
    For “active on day” logic, use `Joiners.filtering(...)` rather than encoding ranges into keys.
    

---

## Safe Patterns (Short Snippets)

### Count with two keys

`.group_by(     lambda a, b: int_key1,     lambda a, b: key2,     ConstraintCollectors.sum(lambda a, b: 1) ) .filter(lambda k1, k2, n: int(n) > 1)`

### Sum with two keys

`.group_by(     lambda a, b: int_key1,     lambda a, b: key2,     ConstraintCollectors.sum(lambda a, b: int(value)) ) .filter(lambda k1, k2, s: s > threshold)`

### Max/Min aggregation

`.group_by(     lambda a, b: key1,     lambda a, b: key2,     ConstraintCollectors.max(lambda a, b: int(expr1)),     ConstraintCollectors.min(lambda a, b: int(expr2)) ) .filter(lambda k1, k2, mx, mn: mx_condition(mx, mn))`

### Distinct count (safe form)

`.group_by(     lambda a, b: int_key1,     lambda a, b: int_key2,     ConstraintCollectors.count_distinct(lambda a, b: distinct_key) ) .filter(lambda k1, k2, c: int(c) > 1)`

> If `count_distinct` appears unstable in a specific version, replace with a manual distinct or separate constraints.

---

## Antipatterns (Avoid)

- **Tuple key grouping without guarantee**
    
    `.group_by(lambda a, b: (k1, k2), ConstraintCollectors.count())  # fragile`
    
- **Truthiness in lambdas**
    
    `.filter(lambda x: x.allowed and x.days)  # prefer explicit checks/casts`
    
- **Arithmetic inside key lambdas**  
    Keep keys simple; move math to filters/collectors.
    

---

## Quick Checklist

-  No tuple keys (unless verified).
    
-  All numeric values cast with `int(...)` in lambdas.
    
-  Counts implemented with `sum(lambda: 1)`.
    
-  Activity tests via `Joiners.filtering(...)`.
    
-  Keys simple; logic in filters/collectors.