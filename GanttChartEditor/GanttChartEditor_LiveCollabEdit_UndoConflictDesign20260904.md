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

## 3. Client: a shared action log replaces the snapshot stacks

`AppState` drops `undoStack`/`redoStack: ScheduleData[]` in favor of:

- **`actionLog: { senderId: string; type: ActionType['type']; payload: unknown }[]`** — every syncable action applied on this client, yours or remote, in the order applied. Appended in exactly one place (a single point in the reducer, or the dispatch wrapper) instead of the ~14 separate `undoStack: pushUndo(state)` call sites scattered across today's reducer cases — a simplification as a side effect of this change.
- **`myPendingUndo` / `myPendingRedo`** — small lists of indices into `actionLog` (or the entries themselves) that are *this client's own*, not yet undone / just undone, most-recent-first. These are what the Undo/Redo buttons actually walk — `canUndo`/`canRedo` become "does this list have entries," same shape as today's `undoStack.length > 0` check.

`LOAD_FILES` / `SET_SESSION_BASELINE` reset all three to empty, same as today's stack reset.

**`actionLog` is intentionally uncapped for the session's lifetime** — unlike today's snapshot stacks (`MAX_UNDO_STACK = 100`), replay correctness needs the true, complete history back to the session's baseline, so nothing can be evicted. This is affordable because each entry is a lightweight `{senderId, type, payload}` triple, not a full document copy — the old cap existed specifically because full-schedule snapshots are heavy; this log is not.

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

**Undo button click:**
1. Take the most recent entry in `myPendingUndo`. If empty, nothing to do (button is disabled anyway).
2. Compute that entry's object key(s).
3. Scan `actionLog` for any entry **after** it, from a **different** `senderId`, whose object key(s) overlap.
   - **No overlap found (safe):** recompute the document by starting from the session's baseline schedule (the same one already used for late-joiner `sync-init`) and replaying every remaining `actionLog` entry in order — skipping only the one being undone — through the same reducer already used for normal forward-apply (no bespoke reverse logic needed; this is the payoff of logging actions instead of snapshots: replay-minus-one naturally keeps every other edit, including ones that came after the undone one). Apply the result locally and forward it to the server exactly the way Undo already syncs today (`sendCollabAction('SET_SCHEDULE', resulting)` — no server/wire protocol change here; this only changes *how* the client computes what to send). Remove the entry from `myPendingUndo`, push it onto `myPendingRedo`.
   - **Overlap found (blocked):** nothing is changed or sent. Dispatch `SET_ERROR` with a message naming the conflict (§6). `myPendingUndo` is untouched — clicking Undo again re-checks the same entry and shows the same result, until whatever changed about the situation (nothing does so automatically; this is a dead end for that specific edit unless the user takes some other action).
4. **Bulk actions** (`BULK_UPDATE_FLEXIBILITY`, `ADD_WORKFLOW_TASKS`, `MERGE_DATA`) are one unit for this check: if *any* of the objects they touched were later touched by someone else, the whole undo is blocked — no partial undo of "the part that's still safe."

**Redo button click:** symmetric, but simpler — no replay needed since redo is inherently a forward step. Take the most recent entry in `myPendingRedo`, re-run the same overlap scan (using the *current*, post-undo log) against its object key(s); if safe, reapply its original `{type, payload}` directly onto the current live state via the normal dispatch path (same as making a fresh edit), push it back onto `myPendingUndo`; if blocked, same stop-and-tell behavior as Undo.

Making any new edit after an undo clears `myPendingRedo`, same convention as today.

## 5. UX

Reuses the existing `SET_ERROR` → `ErrorDialog` pattern already used for the mid-session-load block and the disconnected-edit block (`uiText.ts`) — no new UI mechanism. Two new message keys:

- Undo blocked: "他の参加者がこの対象を変更したため、元に戻せません。" (Someone else changed this — can't undo.)
- Redo blocked: "他の参加者がこの対象を変更したため、やり直せません。" (Someone else changed this — can't redo.)

`UndoRedoButtons.tsx`'s `canUndo`/`canRedo` become `myPendingUndo.length > 0 && !isReadOnly` / `myPendingRedo.length > 0 && !isReadOnly` — same shape as today, just reading the new lists. The conflict check runs on click, not proactively on every render (matches "stop and tell," and avoids scanning the log continuously).

## 6. Testing

- **Server (Vitest):** `senderId` is stored and round-trips through the broadcast and the `sync-init` replay; a joining client receives its own `participantId`.
- **Client (Jest):** the "does this action's target overlap with anything after it from someone else, and what does the document look like with it removed" logic is a pure function, tested directly against constructed logs — including this design's originating scenario (userA: P1/P2 both on O1, P3 on O4, P4 on O5; userB: P1 on O6, P2 on O1) as a named test case, asserting userA can undo P4 and P3 freely, is blocked on P2 (O1, touched by userB after), and — once blocked — that a further Undo click doesn't cascade past it. Plus an `AppContext`-level integration test reproducing the same flow end-to-end through real dispatch.
- **Cross-client proof:** a two-socket Vitest integration test against the real collab server (this codebase already has that pattern in `collabSocket.test.ts`) — a genuine two-participant conflict can't be driven from a single Cypress browser context, same reasoning already documented in the companion reliability design.
- **Cypress:** extend or add a spec proving, from one client's point of view, that a safe undo/redo works and that a blocked one shows the error message without changing anything on screen.

## 7. Explicitly out of scope for this pass

- No visual indication *before* clicking Undo that it will be blocked (e.g. graying out preemptively, or naming who touched the object) — the button stays enabled whenever you have pending entries; you find out on click, per the "stop and tell" decision.
- No automatic retry, queueing, or "undo the next-safe one instead" fallback — a blocked entry simply blocks, matching the existing reliability design's "no silent retry" philosophy.
- No change to the action-delivery reliability problem described in the companion `ReliabilityDesign` doc (whether an edit reliably reaches the server at all) — this pass assumes today's best-effort delivery and only changes what Undo/Redo compute once actions have arrived.
- No persistent/cross-session identity — `senderId`/`participantId` remains per-connection, exactly as today's presence system already works; rejoining a session starts your pending-undo lists fresh, same as today's stack reset on join.
