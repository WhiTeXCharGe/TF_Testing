# Variants

Source: `sigmaI\2.Web APIソースコード一式\APIコンテナ（2026-07-15提供物と同一）\screenas-workerassign-api\20261013_01\inputs` - failing phases assumed: e10p3, e11p2, e11p3, e14p3, e16p2, e19p2, e1p2, e20p2, e20p3, e20p4, e24p2, e24p3, e26p4, e9p2, e9p3

| variant | rows in -> out | what it tests |
|---|---|---|
| `v0_baseline` | 5693 -> 5693 | Unchanged. Control: must reproduce the same 'no valid Y' phases as the failed run. |
| `v1_phase_fixed_to_flexible` | 5693 -> 5693 | Only the Fixed rows that belong to workflow phases (not misc) become Flexible. |
| `v2_no_phase_assignments` | 5693 -> 5424 | Delete all 269 assignment rows of workflow phases; keep the misc (other-work) rows. |
| `v3_no_misc_assignments` | 5693 -> 269 | Delete the misc (other-work) rows; keep the phase rows. |
| `v4_drop_failing_phase_rows` | 5693 -> 5576 | Delete only the assignment rows of the 15 failing phases. |
| `v6_flexible_only_blocked_phases` | 5693 -> 5693 | Fixed -> Flexible only in the 6 phases proven blocked (e10p3, e1p2, e20p3, e26p4, e9p2, e9p3); the other failing phases stay as they are. |
| `v5_no_assignments_at_all` | 5693 -> 0 | Delete every assignment row (plan from scratch). Sanity check: should solve. |
