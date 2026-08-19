# GanttChartEditor — Multi-User Editing: Option Overview

**Goal:** let multiple people edit the same Gantt schedule.
**Constraint:** the Gantt timeline is a custom-built UI (not Excel, not a spreadsheet).

---

## Option A — Build Our Own Real-Time Sync

```mermaid
sequenceDiagram
    participant A as User A (App)
    participant S as Collab Server
    participant B as User B (App)
    A->>S: Join session
    B->>S: Join session
    S-->>B: Current live state
    A->>S: Edit (drag task)
    S-->>B: Live update (<1s)
    B->>S: Edit
    S-->>A: Live update (<1s)
    Note over A,B: Same custom UI, both editing at once
```

| Keeps our UI | Live edit | New infra | Est. time |
|---|---|---|---|
| ✅ | ✅ | Small custom server | 3–8 wks |

**Pro:** true live co-editing, our UI unchanged, no per-seat license.
**Con:** most engineering effort, we own the sync logic.

---

## Option B — Microsoft 365 / SharePoint (Excel Online / WOPI)

```mermaid
sequenceDiagram
    participant A as User A (Browser)
    participant W as SharePoint / WOPI
    participant E as Excel Online
    participant B as User B (Browser)
    A->>W: Open file
    W->>E: Load Excel Online
    B->>W: Open same file
    W->>E: Load Excel Online
    A->>E: Edit cell
    E-->>B: Live update
    Note over A,E: Editing happens in Excel's grid, not our Gantt UI
```

| Keeps our UI | Live edit | New infra | Est. time |
|---|---|---|---|
| ❌ | ✅ (Microsoft's engine) | SharePoint (+ WOPI host if custom) | Days (no integration) → 2–3+ mo (real integration) |

**Pro:** proven co-authoring, native Microsoft/Teams login, zero sync code to write.
**Con:** loses our custom timeline; becomes a spreadsheet; Microsoft 365 licensing.

---

## Option C — ONLYOFFICE Document Server

```mermaid
sequenceDiagram
    participant A as User A
    participant O as ONLYOFFICE Document Server
    participant B as User B
    A->>O: Open file
    B->>O: Open same file
    A->>O: Edit
    O-->>B: Live update
    Note over A,O: Self-hosted, but still ONLYOFFICE's own editor
```

| Keeps our UI | Live edit | New infra | Est. time |
|---|---|---|---|
| ❌ | ✅ (ONLYOFFICE's engine) | Self-hosted Document Server | 5–9 wks |

**Pro:** live co-editing without Microsoft licensing, self-hosted / data stays with us.
**Con:** same UI loss as Option B; another server to run; comparable cost to Option A for less UX fit.

---

## Option D — Shared-Folder File Sync (No Server)

```mermaid
sequenceDiagram
    participant A as User A (App)
    participant F as OneDrive / Network Folder
    participant B as User B (App)
    A->>F: Autosave YAML
    F-->>B: Synced (secs~min later)
    B->>B: File-watcher reloads
    B->>F: Autosave YAML
    F-->>A: Synced (secs~min later)
    Note over A,B: Delayed, real risk of overwrite if both save close together
```

| Keeps our UI | Live edit | New infra | Est. time |
|---|---|---|---|
| ✅ | ❌ | None | 3 days–2 wks |

**Pro:** zero new infrastructure/cost, fastest to build.
**Con:** not real-time, real risk of silently losing someone's changes.

---

## Option E — Check-Out / File-Lock Server *(senpai's idea)*

```mermaid
sequenceDiagram
    participant A as User A
    participant L as File Server (2 YAML + lock)
    participant B as User B
    A->>L: Open (request lock)
    L-->>A: Lock granted + YAML
    B->>L: Open (request lock)
    L-->>B: Locked by A — wait / read-only
    A->>A: Edits locally (as today)
    A->>L: Save (upload YAML, release lock)
    B->>L: Open (request lock)
    L-->>B: Lock granted + latest YAML
    Note over A,B: Works, but only one editor at a time — not simultaneous
```

| Keeps our UI | Live edit | New infra | Est. time |
|---|---|---|---|
| ✅ | ❌ (one editor at a time) | Small file+lock server | 3 days–2 wks |

**Pro:** cheapest server-based option, our UI unchanged, **zero data-loss risk** (only the lock-holder can write).
**Con:** does not show others' edits before save — no true "live" collaboration; can bottleneck if files stay open a long time.

---

## Compare All

| | A. Custom sync | B. Microsoft | C. ONLYOFFICE | D. Folder sync | E. Lock server |
|---|---|---|---|---|---|
| Our UI kept | ✅ | ❌ | ❌ | ✅ | ✅ |
| Live simultaneous edit | ✅ | ✅ | ✅ | ❌ | ❌ |
| Data-loss risk | Low | Low | Low | **High** | **None** |
| New infra | Small server | SharePoint | Self-hosted server | None | Small server |
| License cost | None | MS 365 seats | None (or Enterprise) | None | None |
| Est. time | 3–8 wks | Days → 2–3+ mo | 5–9 wks | 3 days–2 wks | 3 days–2 wks |

---

## Recommendation

```mermaid
flowchart LR
    Q1{Need true live\nsimultaneous editing?} -->|No, rare| E[Option E\ncheck-out lock]
    Q1 -->|Yes, common| Q2{Keep our custom\nGantt UI?}
    Q2 -->|Yes| A[Option A\ncustom sync]
    Q2 -->|OK to lose it| B[Option B / C\nMicrosoft or ONLYOFFICE]
```

**Suggested path:** ship **Option E** first — cheapest, safest, keeps the UI — then upgrade to **Option A** later if real usage shows people frequently need true simultaneous editing.
