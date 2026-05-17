# Employee Scheduler — Webapp Design Ideas

  

## Context

  

The solver already reads two config files at runtime:

- `EnvConfig.yaml` — employees, workflows, regions, FABs

- `Schedule.yaml` — tasks, assignments, plan range, cut-off date

  

The webapp holds these as the "current state" and lets users trigger re-solves with modifications.

  

---

  

## User Actions (Inputs to a New Solve)

  

### New Schedule Run

- Pick solve duration (quick / full)

- Add new job orders (製番):

  - Job ID, customer / FAB / region

  - Per-phase: skill required, man-hours, min/max workers, earliest start, target deadline

  - Default values pre-filled from EnvConfig workflow definitions

- Optionally attach to existing schedule or start fresh

  

### Edit Existing Task Dates

- Change earliest-start or deadline for a task window

- Shift a phase block's allowed date range

- Affects only flexible zone (after cut-off)

  

### Add New Task to Existing Schedule

- Insert a new workflow task into the current Schedule.yaml state

- Inherits operation definitions from EnvConfig

- Solver re-runs from current assignment state as warm start

  

### Lock / Unlock Assignments (Fixed Zone Control)

- Move the cut-off date forward or backward

- Mark specific seats as "pinned" — solver must not change them

- Mark specific seats as "free" — solver may reassign even before cut-off

  

### Update Plan Range

- Extend or shorten `plan_end` date

- Shift `plan_start` (re-solve only the new portion)

- Adjust flexible window independently of fixed window

  

### EnvConfig Changes (without full file replace)

- Add new employee:

  - ID, name, company, skills, region/customer preferences, manager flag

- Edit employee availability:

  - Add unavailable dates / vacation blocks

  - Monthly / annual overtime cap overrides

- Add new FAB or region (rare, but possible mid-project)

- Edit operation definitions (min/max workers, allowed hours list)

  

---

  

## Webapp State Model

  

```

[EnvConfig state]  +  [Schedule state]

        |                    |

        +--------------------+

                 |

         [Pending changes]   <-- user edits staged here

                 |

         [Trigger solve]

                 |

         [Solver job queue]  -- parallel runs allowed

                 |

         [Result store]      -- Azure Blob or local

```

  

- Changes are staged before solve — user can review diff before submitting

- Each solve run gets a job ID and timestamp

- Previous solve results are kept until explicitly deleted

  

---

  

## Additional Ideas

  

**Solve management**

- Run multiple solves in parallel with different condition sets (scenario comparison)

- Label each run (e.g. "conservative", "overtime allowed", "Alice unavailable")

- Side-by-side result comparison view

  

**Gantt / result view**

- Interactive Gantt after solve completes

- Drag to move a block's start date

- Drag an employee name to reassign a seat

- Delete a seat or task directly on the chart

- Constraint violation markers on the chart (red highlight with tooltip)

  

**Manual override**

- Force-assign a specific employee to a specific job+phase

- Becomes a pinned seat in the next solve

- System warns if the override creates a hard constraint violation before solving

  

**Notifications**

- Browser push or email when solve completes

- Alert if solve ends with unresolved hard violations

  

**Export**

- Download result as Excel (current ExportSchedule logic)

- Download modified Schedule.yaml (for local re-run or audit)

- Export Gantt as image (PNG / PDF)

  

**Audit / history**

- Show diff between two solve results

- Show what changed from previous plan (new assignments, moved blocks)

- Revert to a previous result

  

**Constraint visibility**

- Before solving: flag impossible constraints (e.g. deadline before earliest start)

- After solving: list all violated soft constraints with scores

- Highlight overloaded employees (overtime exceeded)

  

**EnvConfig live edit guard**

- Warn if adding an employee mid-schedule would make existing fixed assignments invalid

- Warn if removing a skill leaves a required operation with zero candidates

  

---

  

## Input Summary Table

  

| Action | Touches EnvConfig | Touches Schedule | Re-solve needed |

|---|---|---|---|

| New job order | no | yes | yes |

| Edit task dates | no | yes | yes |

| Add task | no | yes | yes |

| Pin / unpin assignment | no | yes | yes |

| Move cut-off date | no | yes | yes |

| Extend plan range | no | yes | yes |

| Add employee | yes | no | yes |

| Edit availability | yes | no | yes |

| Edit operation hours/workers | yes | no | yes |

| Manual Gantt edit (post-solve) | no | yes | optional |

  

---

  

## Notes

  

- Solver is stateless — each run reads config from scratch; staged changes just produce updated YAML before handing off

- Stage 1 (10 min) result can be shown early as a preview; Stage 2 (3 hr) refines it

- Warm-start from previous result makes re-solves after small edits much faster

- Azure Functions cold-start may add latency — consider keeping one warm instance