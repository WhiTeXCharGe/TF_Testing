# Real-Time Multi-User Collaboration for GanttChartEditor

Full reference: every method considered, how each would actually work, pros/cons, and a rough time estimate. Nothing here has been implemented — this is the decision material.

---

## 1. Starting point

`GanttChartEditor` today is a single-user desktop tool: each person runs their own local Electron app, holding the schedule entirely in memory (React `useReducer`) and saving it manually to local YAML files (`EnvConfig.yaml`, `Schedule.yaml`). There is no shared server, no accounts, no database, and no real-time link between users — the only connection to the sibling `SchedulerWeb` app is a manual, one-time, single-use file handoff.

The Gantt timeline itself (worker/device rows, draggable task bars, drag-to-resize, unavailability ranges) is a **fully custom-built UI** (`WorkerTimelineGrid.tsx`, ~1200 lines of hand-written rendering and drag logic) — not Excel, not any Microsoft or ONLYOFFICE component, not a generic spreadsheet.

This is the fork every option below has to resolve one way or another: **Microsoft (WOPI/Graph) and ONLYOFFICE both give you real-time co-authoring, but only inside their own generic document/spreadsheet editor** — not inside our custom timeline. Keeping the custom UI means building our own sync layer, or accepting something less than full live simultaneous editing.

All time estimates below assume **one engineer already familiar with this codebase**, working roughly full-time on the feature. They are rough planning estimates, not commitments — actual time depends on team size, how much testing/hardening is wanted, and unknowns that only show up during implementation (this is normal for any estimate at this stage).

---

## 2. Options at a glance

| # | Option | Keeps custom Gantt UI? | Live simultaneous edit? | New infra | Est. build time |
|---|---|---|---|---|---|
| A | Build our own real-time sync | ✅ Yes | ✅ Yes | New small server (self-host or Azure) | ~3–4 wks MVP, ~6–8 wks production-ready |
| B | Microsoft 365 / SharePoint + Excel Online (WOPI/Graph) | ❌ No | ✅ Yes (Microsoft's engine) | SharePoint + Microsoft-hosted editor (or a custom WOPI host) | Days (no integration, just SharePoint) to 2–3+ months (real WOPI host) |
| C | ONLYOFFICE Document Server | ❌ No | ✅ Yes (ONLYOFFICE's engine) | Self-hosted Docker service | ~5–9 wks |
| D | Shared-folder file sync (no server) | ✅ Yes | ❌ No — seconds/minutes delay | None (reuse OneDrive/network folder) | ~3 days–2 wks |
| E | Exclusive check-out / file-lock server ("senpai's idea") | ✅ Yes | ❌ No — one editor at a time | Small file server with lock tracking | ~3 days (basic) – 2 wks (robust) |

---

## 3. Option A — Build our own real-time sync layer

Add a small collaboration server (Socket.IO/WebSocket) that every user's app connects to. Edits are broadcast live to everyone in the same session; the existing custom Gantt UI is untouched. Identity can optionally use Microsoft Entra ID (Azure AD) login so people sign in with their work/Teams account — without needing SharePoint or Excel Online underneath it.

### How it works

```
Client (reducer actions) → wrapped dispatch → Collab Server (room, in-memory) → broadcast → other Clients (apply same action to their own reducer)
```

- **Session/room**: a shared editing session = a room keyed by an ID. Creating one seeds the room from whoever's current in-memory schedule; joining sends the room's *current* (possibly unsaved) state so latecomers catch up to live edits, not just the last save.
- **Protocol**: the app's existing ~40 reducer action types (`ADD_ASSIGNMENT`, `UPDATE_ASSIGNMENT`, `UPDATE_PHASE_TASK`, etc.) are reused directly as the network protocol — each edit is broadcast as the same small action object the reducer already understands, so every client stays in sync by replaying the same actions through the same pure reducer function they already have.
- **Conflict model**: live broadcast, no locking — everyone sees everyone else's edits appear as they happen. Two people editing the exact same field at the exact same instant results in last-applied-wins (acceptable for this use case; a full conflict-free merge engine, CRDT-style, was considered and rejected as overkill for structured schedule data).
- **Identity**: Microsoft Entra ID (Azure AD) login via MSAL — satisfies "sign in with Microsoft/Teams account" without needing WOPI or Excel.
- **Persistence**: explicit Save still writes YAML, now via the collab server to Azure Blob Storage (or any shared storage); periodic server-side auto-checkpointing protects against data loss even before someone clicks Save.
- **Presence**: an avatar list of who's currently in the session; optionally, later, a lightweight "someone's editing this row" indicator (informational only, never blocking).

### Pros
- Keeps the polished, purpose-built Gantt timeline exactly as-is — no UX downgrade.
- Full control over collaboration behavior, tailored to how scheduling actually works.
- No per-seat Microsoft/ONLYOFFICE licensing.
- Can still offer Microsoft/Teams login for identity, independent of SharePoint.
- Reuses a pattern the team already runs (this project already hosts a small Azure service the same way, for the solver backend).

### Cons
- Real engineering effort — conflict handling, presence, reconnection, and a new server component all need to be designed and built from scratch; no vendor collaboration engine to lean on.
- We own bugs/edge cases in the sync logic long-term.
- Needs an always-on server somewhere — ongoing hosting cost/maintenance, however small.
- Registering an Entra ID app for Microsoft login needs a one-time tenant-admin action (not code, but an external dependency).

### Estimated time
- **MVP** (broadcast + presence + login, in-memory only, single test session): **~3–4 weeks**.
- **Production-ready** (+ Blob persistence/auto-checkpoint, reconnection handling, Electron desktop login flow): **~6–8 weeks total**.
- Optional polish (per-task "editing" indicators, shareable deep links): **+1–2 weeks**.

---

## 4. Option B — Microsoft 365 / SharePoint, via Excel Online (WOPI or Graph API)

Store the schedule as an actual Excel file in a SharePoint/OneDrive document library. People edit it either through **Excel Online embedded via the WOPI protocol** (the mechanism SharePoint itself uses to embed Office editors in any web app), or read/write it programmatically via the **Microsoft Graph Excel REST API**.

### How it works (WOPI variant)

```
App → WOPI Host (ours or SharePoint's) → Excel Online (Microsoft-hosted editor) → live co-authoring between all open sessions
```

Two very different sub-cases matter here:
1. **No custom integration at all**: just put the two files in a SharePoint library and let people open them with Excel Online directly from SharePoint. Zero engineering — but this abandons `GanttChartEditor` as a tool entirely; people are just editing a spreadsheet in SharePoint.
2. **Real integration** (embed Excel Online inside our own app, become a WOPI host): a substantial protocol implementation (WOPI has its own validation/compliance requirements), plus redesigning the schedule data model to fit spreadsheet rows/columns.

### Pros
- Real-time co-authoring is Microsoft's own proven engine — nothing to build.
- Native fit with company Microsoft/Teams accounts and SharePoint's existing permission model.
- Version history, comments, file management all come free from SharePoint.

### Cons
- **The custom Gantt timeline goes away.** Editing happens in Excel Online's grid, not the drag-and-drop worker/device timeline.
- Becoming a real WOPI host is a significant integration project in its own right.
- **Graph API alone (without embedding Excel Online) is not live collaboration** — it's request/response cell access; you'd be polling, not watching live edits.
- Task dependencies, worker assignments, drag-based date ranges don't map cleanly onto spreadsheet cells — real data-modeling rework either way.
- Ties core editing to Microsoft 365 licensing for every user.

### Estimated time
- **Plain SharePoint doc library, no custom integration**: days — but this is not really "our tool" anymore, just a business-process change.
- **Full custom WOPI host integrated into our app** + schedule→spreadsheet data remodel: **~2–3+ months**, and a meaningful share of that time is protocol/compliance work outside our own product logic.

---

## 5. Option C — ONLYOFFICE Document Server

Self-host ONLYOFFICE's Document Server (Docker container, open-source Community Edition or paid Enterprise Edition) and embed its editor for the schedule (stored as an .xlsx-style document), using ONLYOFFICE's own real-time co-editing engine.

### How it works

```
App → ONLYOFFICE Document Server (self-hosted, JWT-authenticated) → embedded editor iframe → live co-authoring between all open sessions
```

### Pros
- Real-time co-authoring engine included, like Option B, but **self-hosted** — no Microsoft 365 tenant or per-user Microsoft licensing required.
- Open, JWT-based API — can plug in our own identity/auth rather than requiring Microsoft accounts.
- Data stays on infrastructure we control.

### Cons
- **Same fundamental limitation as Option B** — the editing surface is ONLYOFFICE's generic spreadsheet/document editor, not our custom Gantt timeline. Its plugin/macro API can add buttons or automate cells, but cannot host a fully custom drag-and-drop rendering surface.
- Another infrastructure component to run and maintain — comparable operational weight to Option A's custom server, without any of the Gantt-specific UX benefit.
- Community (free) edition has real limits on concurrent connections; heavier use may need the paid Enterprise edition.
- Same schedule→spreadsheet data-modeling rework as Option B.

### Estimated time
- Self-host + embed + our own auth wiring: **~3–5 weeks**.
- Schedule → spreadsheet layout data remodel: **~2–4 weeks**.
- **Total: ~5–9 weeks** — comparable to or larger than Option A, for a worse UX fit.

---

## 6. Option D — Shared-folder file sync (no server at all)

Don't build a server or adopt any vendor. Point the existing Save/auto-save at a **shared folder** — a network drive, or simply a OneDrive/SharePoint-synced folder the company already has — used purely as passive file transport, not as an editor. Add: (1) autosave on every edit, debounced, to that shared file; (2) a file-watcher that notices when the file changed on disk (a teammate saved, and sync delivered it) and reloads it into the app.

### How it works

```
User A autosaves → shared folder (OneDrive/network) → sync propagates (secs–mins) → User B's file-watcher detects change → reloads
```

### Pros
- **Zero new infrastructure, zero new cost.** No server, no Azure resource, no Microsoft/ONLYOFFICE licensing beyond storage already available.
- Keeps the custom Gantt UI completely unchanged.
- Fastest to build — extends existing local save/load code rather than adding new systems.

### Cons
- **Not real-time.** Updates only appear after an autosave interval plus however long folder sync takes — typically seconds to over a minute, sometimes longer under sync backlogs. Does not meet "see others' edits before they're finished."
- No presence — no visibility into who else has the file open or what they're touching.
- Real collision risk: near-simultaneous autosaves can silently drop one person's changes (last-write-wins), and OneDrive itself may produce a "(conflicted copy)" file instead of merging, requiring manual resolution.
- Not a foundation for future features (permissions, audit history, per-field attribution) — more of a stopgap.

### Estimated time
- **~3 days – 2 weeks**, depending on how much conflict/edge-case handling (stale-file detection, conflicted-copy warnings) is wanted.

---

## 7. Option E — Exclusive check-out / file-lock server ("senpai's idea")

**The idea, as proposed:** a file server (reachable via a shared link) stores the two YAML files. A user opens `GanttChartEditor`, imports the YAML from the server, edits locally exactly like today, and "overwrite/save" pushes the update back to the server — but **while one user has the file open, nobody else can edit it** until they're done.

### Feasibility verdict: yes, this works, and it's a well-established, low-risk pattern

This is **pessimistic locking / check-out-check-in** — the same model classic SharePoint document check-out, Perforce, and most PLM/PDM file-vault tools have used for decades. It is simple, reliable, and has effectively zero risk of silently losing someone's edits (unlike Option D), because the server only ever accepts a save from whoever currently holds the lock.

### How it works

```
User A: open → server grants lock, sends YAML → User A edits locally (like today) → save → server stores YAML, releases lock
User B (while A has it open): open → server says "locked by User A since 10:32" → read-only / must wait
User B (after A saves/closes): open → server grants lock, sends latest YAML → edits...
```

### What's actually needed to build it
- A small file server (could directly extend the app's existing local Express server, just pointed at shared storage instead of local disk) exposing: fetch current files + lock status, acquire lock on open, release lock + upload new YAML on save/close.
- Client changes: on "Open shared file," check lock status first; show a clear "currently being edited by X since HH:MM" state if locked, instead of silently blocking.
- **Abandoned-lock handling is the one real design decision**: if User A's app crashes or loses network without releasing the lock, the file must not stay locked forever. Standard fix: a lease/heartbeat — the lock auto-expires after N minutes of inactivity (the app periodically "pings" the server while the file is open to renew the lock), or an admin/force-unlock action as a fallback.

### Pros
- Keeps the custom Gantt UI completely unchanged — this is just a gate in front of the existing open/save flow.
- **Zero conflict/data-loss risk** — unlike Options D or even A's last-write-wins model, only one person can ever be writing at a time, so nobody's edit is ever silently overwritten.
- Directly satisfies "share via a link, store on a file server" from the original idea.
- Cheapest server-based option to build — smaller than Option A by a wide margin, since there's no real-time broadcast, no presence system, no operation protocol, no undo-stack tagging.
- Can still add Microsoft/Teams login later for identity (who currently holds the lock) without changing the core design.

### Cons
- **Does not satisfy "see other users' edits before they save"** — the original ask. Only one person edits at a time; everyone else must wait or view a stale/read-only copy. This is a genuine, explicit trade-off against the original "live" requirement, not a partial version of it.
- No live presence beyond "who currently holds the lock."
- If people tend to leave files open for long stretches (e.g., all day), this can feel like a bottleneck in practice — worth checking against how the team actually works before committing to it.
- Still needs *some* small always-on server (though much simpler than Option A's).

### Estimated time
- **Basic version** (lock on open, release on save, no heartbeat): **~3–5 days**.
- **Robust version** (lease/heartbeat renewal, stale-lock recovery, clear "locked by" UI, "notify me when free"): **~1–2 weeks**.

---

## 8. Full comparison matrix

| Criterion | A. Custom sync | B. Microsoft WOPI/Graph | C. ONLYOFFICE | D. Shared-folder sync | E. Check-out/lock |
|---|---|---|---|---|---|
| Keeps custom Gantt UI | ✅ | ❌ | ❌ | ✅ | ✅ |
| True live simultaneous editing | ✅ | ✅ (in Excel Online) | ✅ (in ONLYOFFICE editor) | ❌ | ❌ (by design — one at a time) |
| Sees others' unsaved edits | ✅ | ✅ (within Excel Online) | ✅ (within ONLYOFFICE) | ❌ | ❌ |
| Data-loss / overwrite risk | Low (last-write-wins on exact-same-field only) | Low (Microsoft's engine handles merge) | Low (ONLYOFFICE's engine handles merge) | **Real risk** | **None** (locking prevents it) |
| New infrastructure | Small custom server | SharePoint (+ WOPI host if custom) | Self-hosted Document Server | None | Small custom server |
| Licensing dependency | None required | Microsoft 365 seats | None required (or Enterprise tier for scale) | None | None |
| Est. build time | ~3–8 wks | Days (no integration) to 2–3+ mo (real integration) | ~5–9 wks | ~3 days–2 wks | ~3 days–2 wks |

---

## 9. Recommendation

- If the custom Gantt timeline is worth preserving, **B and C are both out** — they require moving the actual editing surface into Microsoft's or ONLYOFFICE's own generic editor.
- Between the remaining three (A, D, E), the real decision is **how important true "live, before-save" visibility into others' edits actually is** in practice:
  - If it matters a lot (people frequently need to co-edit the same schedule at the same time and see it happen) → **Option A**.
  - If simultaneous editing is rare and "no data loss, one clear owner at a time" is good enough → **Option E** gets most of the practical benefit of A for a fraction of the cost, and is a very reasonable first step — can still be upgraded to A later without throwing the work away (the file-server piece of E is a subset of what A needs anyway).
  - **Option D** is only a good fit as a true stopgap when literally nothing can be built right now — its conflict risk is real and it doesn't meet the original "live" requirement either.
- A sensible phased path: **ship Option E first** (cheap, safe, immediately useful), then decide whether to invest in upgrading to **Option A** once real usage shows how often true simultaneous editing is actually needed.
