# GanttChartEditor Live Collab — Own-Action, Conflict-Aware Undo/Redo Design

**Companion to:** `GanttChartEditor_LiveCollabEdit_Design20260826.md` (original design), `GanttChartEditor_LiveCollabEdit_Plan20260826.md` (original 12-task plan), `GanttChartEditor_LiveCollabEdit_TestReport20260827.md` (test report), `GanttChartEditor_LiveCollabEdit_UXFeedbackDesign20260828.md` / `...Plan20260828.md` (round 2 UX feedback, already merged), `GanttChartEditor_LiveCollabEdit_ReliabilityDesign20260828.md` (separate, still-unimplemented action-ack design).

**Problem this closes:** today, "Undo" pops the most recent entry off a single shared-looking stack of full-document snapshots — but that stack silently includes *other participants'* synced edits alongside your own, in strict chronological arrival order. Concretely: if userB edits the document after userA's last edit, userA's next Undo click reverts **userB's** edit, not userA's own — with no indication this happened, and no way to undo your own older edit without first undoing everyone else's edits that landed after it. Separately, since undo works by jumping back to an old whole-document snapshot, undoing an older edit necessarily discards every edit (yours or anyone else's) that came after it — there is no way to undo just one specific change while keeping later, unrelated changes intact.

---

## 1. Goal

- Undo/Redo only ever act on **your own** edits — never silently revert someone else's.
- Undoing one of your own edits **keeps every other edit** made since (yours or anyone else's), as long as it didn't touch the same object.
- If your target edit's object **was** touched by someone else since, Undo/Redo is blocked outright: nothing changes, you're told why, and you'd need to try again later (no silent skip to an older edit, no partial application).
- Applies uniformly to every kind of edit — bar moves, colors, worker fields, unavailable-date ranges, bulk operations — not just bar drags.

## 2. Server: tag every action with its sender

`server/src/collab/collabSocket.ts` already generates a `participantId` per socket connection but never uses it beyond presence tracking. Two additions:

- `sessionStore.ts`'s `LoggedAction` gains `senderId: string`; `appendAction` takes and stores it.
- The `action` socket handler passes its own `participantId` through to `appendAction`, and includes `senderId` in both the broadcast to other participants and the `actions` array replayed to late joiners via `sync-init`.
- The server also tells a client its own `participantId` at join time (new field on the `sync-init` payload, or a dedicated field alongside it) — today nothing tells a client its own identity at all.

This is the only server-side change. The server remains "dumb" per its existing design comment (`sessionStore.ts`'s header) — it still just stores and orders actions; it never interprets them or computes anything about ownership or conflicts itself.

## 3. Client: per-action before/after patches replace the snapshot stacks

**Revised from the originally-approved shared-log/replay approach** (kept below in §7 as a noted future option) after finding two things while digging into the actual reducer: (1) `envConfig`-mutating actions (worker fields, unavailable dates) have **no undo support at all today** — `undoStack`/`redoStack` only ever store `ScheduleData`, so a replay engine would need a new mechanism to reconstruct `envConfig` too; (2) almost every action type already targets a specific object by a stable id (`operationTaskId`, `workerId`, ...) — only assignments are index-based. That means we don't need a shared history to replay at all: for each of *your own* edits we can just remember what the touched object's value was **before** and **after**, and undo becomes "if the object's current value still equals what I set it to, put the old value back" — a plain comparison against live state, no log-scanning, no replay.

`AppState` drops `undoStack`/`redoStack: ScheduleData[]` in favor of:

- **`myPendingUndo` / `myPendingRedo`: `UndoEntry[]`**, most-recent-first, where `UndoEntry` is `{ type: ActionType['type']; targets: { key: string; oldValue: unknown; newValue: unknown }[] }` — one entry per edit *you* made (never someone else's — inbound remote actions never get pushed here at all, which is simpler than filtering a mixed log after the fact). `targets` is almost always a single-item array; only the compound actions (§ below) have more than one.
- `canUndo`/`canRedo` become "does this list have entries" — same shape as today's `undoStack.length > 0` check.

`LOAD_FILES` / `SET_SESSION_BASELINE` reset both lists to empty, same as today's stack reset. The existing `MAX_UNDO_STACK = 100` cap still applies (entries are small before/after values now, not full documents, so this is even cheaper than before).

### Object identity

Every syncable action type gets a one-line **"what object(s) does this touch"** helper — the only per-action-type code this feature needs (no per-type reverse-apply logic):

| Action | Object key(s) |
|---|---|
| `UPDATE_ASSIGNMENT`, `DELETE_ASSIGNMENT` | the assignment's stable id (see below) |
| `UPDATE_PHASE_TASK` | `phaseTaskId` |
| `UPDATE_OPERATION_TASK`, `UPDATE_OPERATION_TASK_COLOR` | `operationTaskId` |
| `UPDATE_WORKFLOW_TASK_COLOR` | `workflowTaskId` |
| `UPDATE_WORKER_DEFINITION`, `UPDATE_WORKER_DESC_FIELD` | `workerId` |
| `DELETE_UNAVAILABLE_DATE`, `DELETE_UNAVAILABLE_RANGE`, `MOVE_UNAVAILABLE_DATE`, `RESIZE_UNAVAILABLE_RANGE`, `ADD_UNAVAILABLE_DATES` | `workerId` (+ date/range identifies which entry) |
| `BULK_UPDATE_FLEXIBILITY`, `ADD_WORKFLOW_TASKS`, `MERGE_DATA` | **every** object id it affected (a set, not a single key) |
| `ADD_ASSIGNMENT` | the new assignment's own stable id (assigned at creation, same as any other assignment) — so if someone else later edits or deletes that same new assignment, undoing the add is blocked, same as any other conflict |
| `UPDATE_PLAN_RANGE` | a fixed singleton key (e.g. `'planRange'`) — the plan's start/end date is document-level, not per-item, but still only conflicts with another `UPDATE_PLAN_RANGE`, not with unrelated edits elsewhere |

**Assignments need a stable id that doesn't exist today.** `Assignment` (`schedule.ts`) has no `id` field — `UPDATE_ASSIGNMENT`/`DELETE_ASSIGNMENT` reference one only by array index, which shifts whenever any assignment is added or removed. A hidden `_id` (client-generated UUID) is attached to every assignment when a file loads (`LOAD_FILES`) or a new one is created (`ADD_ASSIGNMENT`). This is purely an in-memory/session concern: `yamlService.ts`'s writer serializes only its own named fields (confirmed by reading it), so this extra field is silently dropped on save and never round-trips into the YAML file. The existing index-based editing (drag, delete) is completely unchanged — `_id` is consulted only by this feature's conflict check.

## 4. Undo / Redo mechanics

**When you make an edit** (any syncable action type), before applying it the dispatch wrapper looks up the current value at that action's object key(s) (the same key(s) from the §3 table), applies the edit as normal, then pushes `{ type, targets: [{ key, oldValue, newValue }] }` onto `myPendingUndo` and clears `myPendingRedo` — replacing the `pushUndo(state)` call at each of the ~14 mutating reducer cases with this one recording step, done once in the dispatch wrapper rather than scattered per case.

**Undo button click:**
1. Take the most recent entry in `myPendingUndo`. If empty, nothing to do (button is disabled anyway).
2. For every `target` in that entry, compare its `key`'s value in the **current live state** to `target.newValue`.
   - **All match (safe):** apply `target.oldValue` back at each `key`, through the same reducer case the original action used (so it applies and syncs exactly like a normal edit — no new sync mechanism). Move the entry from `myPendingUndo` to `myPendingRedo`.
   - **Any mismatch (blocked):** someone changed at least one of the touched objects since. Nothing is changed or sent. Dispatch `SET_ERROR` with a message naming the conflict (§5). `myPendingUndo` is untouched — clicking Undo again re-checks the same entry and shows the same result; nothing resolves this automatically.
3. Multi-target entries (bulk actions) are one unit: *any* mismatched target blocks the whole undo, matching the all-or-nothing default already agreed.

**Redo button click:** symmetric — take the top of `myPendingRedo`, compare current values against `target.oldValue` (what undo just set), and if all match, reapply `target.newValue` and move the entry back to `myPendingUndo`; otherwise block the same way.

Making any new edit after an undo clears `myPendingRedo`, same convention as today.

### Compound actions need a small revert vehicle

Most action types are naturally reversible by resending the same action type with old values (e.g. undo `UPDATE_WORKER_DEFINITION` by sending another `UPDATE_WORKER_DEFINITION` with the old text). Three don't fit that shape and need one new, internal-only action type each, used solely as the undo/redo vehicle (never dispatched by the UI directly):

- **`BULK_UPDATE_FLEXIBILITY`** → revert via a new `RESTORE_ASSIGNMENT_FIELDS: { assignmentId: string; updates: Partial<Assignment> }[]` (restores each affected assignment's own recorded old flexibility in one batch).
- **`ADD_WORKFLOW_TASKS`** → revert via a new `REMOVE_WORKFLOW_TASKS_BY_ID: string[]` (removes exactly the workflow task ids that were newly added — `ADD_WORKFLOW_TASKS` already dedupes against existing ids, so "newly added" is well-defined at apply time).
- **`MERGE_DATA`** → revert via a new `REVERT_MERGE: { workflowTaskIds: string[]; assignmentCount: number; envConfigAdditions: { [list: string]: string[] } }` (removes the specific items that were newly merged in — same "already deduped, so newly-added is known at apply time" logic, across both `schedule` and `envConfig`).

`ADD_ASSIGNMENT`/`DELETE_ASSIGNMENT`/`UPDATE_ASSIGNMENT` don't need a new action type — they revert via each other (add↔delete, update↔update-with-old-values) — see §3's assignment `_id`.

## 5. UX

Reuses the existing `SET_ERROR` → `ErrorDialog` pattern already used for the mid-session-load block and the disconnected-edit block (`uiText.ts`) — no new UI mechanism. Two new message keys:

- Undo blocked: "他の参加者がこの対象を変更したため、元に戻せません。" (Someone else changed this — can't undo.)
- Redo blocked: "他の参加者がこの対象を変更したため、やり直せません。" (Someone else changed this — can't redo.)

`UndoRedoButtons.tsx`'s `canUndo`/`canRedo` become `myPendingUndo.length > 0 && !isReadOnly` / `myPendingRedo.length > 0 && !isReadOnly` — same shape as today, just reading the new lists. The conflict check runs on click, not proactively on every render (matches "stop and tell," and avoids scanning the log continuously).

## 6. Testing

- **Server (Vitest):** `senderId` is stored and round-trips through the broadcast and the `sync-init` replay; a joining client receives its own `participantId`.
- **Client (Jest):** the "is every target's current value still what I set it to" check is a pure function, tested directly against constructed state + `UndoEntry` fixtures — including this design's originating scenario (userA: P1/P2 both on O1, P3 on O4, P4 on O5; userB: P1 on O6, P2 on O1) as a named test case, asserting userA can undo P4 and P3 freely, is blocked on P2 (O1, touched by userB after), and — once blocked — that a further Undo click doesn't cascade past it. Plus an `AppContext`-level integration test reproducing the same flow end-to-end through real dispatch, and per-type coverage for the three new compound revert actions.
- **Cross-client proof:** a two-socket Vitest integration test against the real collab server (this codebase already has that pattern in `collabSocket.test.ts`) — a genuine two-participant conflict can't be driven from a single Cypress browser context, same reasoning already documented in the companion reliability design.
- **Cypress:** extend or add a spec proving, from one client's point of view, that a safe undo/redo works and that a blocked one shows the error message without changing anything on screen.

## 7. Explicitly out of scope for this pass

- No visual indication *before* clicking Undo that it will be blocked (e.g. graying out preemptively, or naming who touched the object) — the button stays enabled whenever you have pending entries; you find out on click, per the "stop and tell" decision.
- No automatic retry, queueing, or "undo the next-safe one instead" fallback — a blocked entry simply blocks, matching the existing reliability design's "no silent retry" philosophy.
- No change to the action-delivery reliability problem described in the companion `ReliabilityDesign` doc (whether an edit reliably reaches the server at all) — this pass assumes today's best-effort delivery and only changes what Undo/Redo compute once actions have arrived.
- No persistent/cross-session identity — `senderId`/`participantId` remains per-connection, exactly as today's presence system already works; rejoining a session starts your pending-undo lists fresh, same as today's stack reset on join.
- **Deferred, not discarded: the shared action-log/replay approach originally designed above.** If a future need arises that per-action patches can't express well (e.g. true reordering/rebasing of concurrent edits, or undo semantics that need to see the *entire* session history rather than just your own edits), revisit that version — the server-side `senderId` tagging from §2 is required by both approaches, so nothing here is wasted if that upgrade happens later.
