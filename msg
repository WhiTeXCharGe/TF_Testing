You’re right — your last message is a different request.

Add it in the **worker creation / skill fill part**, not in the phase logic.

## What you want

In each worker:

```yaml
skill_map:
  p2: 1
  p3: 1
  p4: 1
worker_type_by_operation:
  p2: regular
  p3: regular
  p4: spot
```

Rules:

* default for `p2/p3/p4` = `regular`
* if worker company name is in a company list, then `p2/p3/p4 = spot`
* only for `p2/p3/p4`
* do not include `other_op` or `personal_business_op`

That fits well with your current `worker_list` structure, because workers are first created in `parse_su_others()` and later `skill_map` is filled in the environment build step. 

## 1. Add config at top

Add this near your other config values:

```python
SPOT_WORKER_COMPANY_NAMES = ["SCREEN SPE", "SCREEN CAS"]
```

---

## 2. Add the field when worker_list is created

In `parse_su_others()`, change the worker append block from:

```python
worker_list.append({
    "id": wid,
    "name": acc["name"],
    "worker_company": acc["worker_company"],
    "is_manager": acc["is_manager"],
    "role": worker_roles.get(wid, ""),
    "skill_map": {},
    "fab_suitability_map": [],
    "unavailable_dates": [{"date": d} for d in sorted(acc["unavailable_set"])],
})
```

to:

```python
worker_list.append({
    "id": wid,
    "name": acc["name"],
    "worker_company": acc["worker_company"],
    "is_manager": acc["is_manager"],
    "role": worker_roles.get(wid, ""),
    "skill_map": {},
    "worker_type_by_operation": {
        "p2": "regular",
        "p3": "regular",
        "p4": "regular",
    },
    "fab_suitability_map": [],
    "unavailable_dates": [{"date": d} for d in sorted(acc["unavailable_set"])],
})
```

That adds the new section for every worker from the start. 

---

## 3. After environment is built, mark spot workers by company

In your main worker skill fill section, add a company-name lookup and update `worker_type_by_operation`.

Use this block before or together with your skill fill:

```python
# worker_company id -> worker_company name
worker_company_name_by_id = {
    wc["id"]: wc["name"]
    for wc in environment.get("worker_company_list", [])
}

spot_company_set = {str(x).strip().upper() for x in SPOT_WORKER_COMPANY_NAMES}

for w in environment["worker_list"]:
    w.setdefault("worker_type_by_operation", {
        "p2": "regular",
        "p3": "regular",
        "p4": "regular",
    })

    company_id = w.get("worker_company")
    company_name = str(worker_company_name_by_id.get(company_id, "")).strip().upper()

    if company_name in spot_company_set:
        w["worker_type_by_operation"]["p2"] = "spot"
        w["worker_type_by_operation"]["p3"] = "spot"
        w["worker_type_by_operation"]["p4"] = "spot"
```

This works because the worker stores company as `worker_company` id, while the actual company names are in `worker_company_list`. 

---

## 4. Important note

This is only for:

* `p2`
* `p3`
* `p4`

So do **not** add:

* `other_op`
* `personal_business_op`

inside `worker_type_by_operation`.

---

## Result example

For normal company:

```yaml
worker_type_by_operation:
  p2: regular
  p3: regular
  p4: regular
```

For `SCREEN SPE` or `SCREEN CAS`:

```yaml
worker_type_by_operation:
  p2: spot
  p3: spot
  p4: spot
```

If you want, next I can give you one single paste-ready block for the exact place in your current file where the skill fill is happening.
