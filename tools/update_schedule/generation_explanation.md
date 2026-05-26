# config_20250526 — Parameter Rationale

**Base assumptions**
| Item | Value |
|---|---|
| Workers | 1,500 |
| Work period | Sep 1, 2025 → Mar 31, 2026 (~6.5 months) |
| Working days | ~139 days (Mon–Fri) |
| Target utilization | 70% |

---

## EQ_NUM = 365

Each module's total workload across all operations (normal template):

| Phase | Operations | Workload (days) |
|---|---|---|
| p1 | Heavy / Mech / Elec | 30 + 20 + 20 = 70 |
| p2 | Mech / Elec | 30 + 25 = 55 |
| p3 | QC | 15 |
| p4 | QC / Mech | 12 + 8 = 20 |
| **Total** | | **160 worker-days / module** |

With avg 2.5 workers per operation and 8 h/day:

```
160 days × 8 h × 2.5 workers = 3,200 worker-hours per module
```

Total supply and demand:

```
Supply : 1,500 × 139 × 8 = 1,668,000 wh
Target : 1,668,000 × 0.70 = 1,167,600 wh
Modules: 1,167,600 / 3,200 ≈ 365
```

---

## EQ_PER_DAYS = 3.7

One module spans 15+12+6+7 = **40 working days**.  
To finish all 365 modules within the 139-day window the last module must start by day 99 (139 − 40).

```
365 modules / 99 days ≈ 3.7 modules/day
Peak concurrent modules ≈ 3.7 × 40 ≈ 148
```

148 concurrent modules × ~6–7 active workers each ≈ **~950–1,000 workers busy at peak**, which is consistent with the 70% target.

---

## Affinity groups = 86

- Group size range: 2–5, average 3.5
- Target: ~20% of workers (300) carry at least one tag

```
300 memberships / 3.5 avg group size ≈ 86 groups
```

This gives meaningful relationship constraints without over-tagging the pool.

---

## Worker companies = 8

```
1,500 workers / 8 companies ≈ 187 workers per company
```

Enough companies to create real company-level scheduling tension, but each company large enough to have diverse skills and valid wct-affinity tags.
