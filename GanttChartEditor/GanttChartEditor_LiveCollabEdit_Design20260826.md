# GanttChartEditor — Live Collaborative Editing (Option A) — Design

**Status:** approved for implementation planning
**Supersedes for editing purposes:** the read-only "Live View" broadcast (`viewBroadcastService.ts`, `ViewPage.tsx`, `ShareViewButton`/`ShareViewDialog`) described in the existing docs — this design folds that feature into one unified session flow instead of running two parallel systems.
**Builds on:** `GanttChartEditor_Collaboration_Presentation.md` and `GanttChartEditor_RealTime_Collaboration_FullReference.md` (Option A).

---

## 1. Goal

Let multiple people edit the same in-memory schedule at the same time, live, through the existing custom Gantt UI — no locking, no per-seat login. Any reducer-driven edit any participant makes is wrapped and sent to a small relay server, which logs it and broadcasts it to everyone else; a client who joins mid-session loads the session's original data plus every action applied so far and replays it locally to catch up.

## 2. What already exists (do not re-invent)

- `server/src/index.ts` (Express + Socket.IO, port 3010) already runs alongside the app, already has `/api/network-info` (LAN IP) and a small collab module at `server/src/collab/viewBroadcast.ts`.
- That module today does **snapshot broadcast only**: host pushes its full `{schedule, envConfig, currentView}` on every change; viewers get it read-only via `?view=1`, opened in a plain browser tab (confirmed working today, no Electron required for the viewer side).
- `electron/main.cts` spawns this server as a child process in packaged builds and currently kills it on `window-all-closed` → `app.quit()` → `before-quit` → `stopEmbeddedServer()`.

This design replaces the snapshot module with a session/action-log module, and changes the Electron shutdown wiring so the server survives the host closing their window.

## 3. Server — session + action log

Location: `server/src/collab/session.ts` (replaces `viewBroadcast.ts`; the file is still mounted the same way from `index.ts`).

**In-memory state:**
```ts
interface CollabSession {
  id: string;
  baseline: { schedule: ScheduleData; envConfig: EnvConfig; currentView: ViewMode };
  actions: { seq: number; type: string; payload: unknown; from: string }[];
  participants: Map<socketId, { name: string; role: 'edit' | 'view' }>;
  nextSeq: number;
  lastActivityAt: number;
}
const sessions = new Map<string, CollabSession>();
```

**HTTP:**
- `POST /api/collab/sessions` — body `{ schedule, envConfig, currentView }` → creates a session, returns `{ sessionId }`. The link itself is built client-side from the existing `network-info` LAN IP, same as today's `fetchShareableLink`.

**Socket.IO (new path, e.g. `/collab/socket.io`, same port 3010):**
- `join({ sessionId, name, role })` → registers the socket in a Socket.IO room named by `sessionId`; replies `sync-init({ baseline, actions, participants })`. If `sessionId` doesn't exist, replies `sync-init` with an error so the client can show "session not found / ended."
- `action({ type, payload })` (edit-role only) → server assigns `seq = ++nextSeq`, appends to `actions`, `socket.to(sessionId).emit('action', { seq, type, payload })`. No validation of the action's semantic correctness — the server is a dumb relay/log; the reducer (shared, pure, already tested) is the single source of truth for what an action does, on every client including the one that sent it (it already applied it locally before sending).
- `leave` / `disconnect` → removes the participant, broadcasts `presence(participants)`. **The session itself is never torn down by a participant leaving** — including whoever created it. There is no persistent "host" role after creation.
- Idle sweep: a `setInterval` drops any session with zero participants for >30 minutes, so a forgotten test session doesn't leak memory forever.

**No conflict resolution logic.** Actions are ordered by arrival at the single Node process (Socket.IO already serializes this), so "last applied wins" on true same-instant edits to the same field — the same trade-off the reference doc already accepted over building a CRDT.

## 4. Client — wrapping dispatch, not rewriting the editor

**`src/services/collabService.ts`** (replaces `viewBroadcastService.ts`):
- `createSession(schedule, envConfig, currentView): Promise<{ sessionId, link }>`
- `joinSession(sessionId, name, role, onSyncInit, onAction, onPresence, onStatusChange): () => void` (returns disconnect)
- `sendAction(type, payload)` — no-op if not connected or role is `view`
- `fetchShareableLink(sessionId, role)` — reuses the existing `network-info`-based LAN IP resolution

**Centralized dispatch wrapping in `AppContext.tsx`**, not threaded through every call site:
```
real dispatch (useReducer's) ──▶ reducer ──▶ new state
        ▲
        │ apply remote actions directly here (no re-forward)
        │
wrapped dispatch (exported via context, same name/shape as today)
        │
        └─▶ if session active, role=edit, and action.type ∈ SYNCABLE ⇒ collabService.sendAction(...)
```
Every existing component keeps calling `dispatch(...)` exactly as today — no per-component changes. The wrapping lives in one place: `AppContext.tsx`.

**Syncable action types** (schedule/envConfig-mutating; everyone must converge on these):
`SET_SCHEDULE, UPDATE_PLAN_RANGE, ADD_ASSIGNMENT, UPDATE_ASSIGNMENT, DELETE_ASSIGNMENT, UPDATE_PHASE_TASK, UPDATE_OPERATION_TASK, BULK_UPDATE_FLEXIBILITY, ADD_WORKFLOW_TASKS, MERGE_DATA, DELETE_UNAVAILABLE_DATE, DELETE_UNAVAILABLE_RANGE, MOVE_UNAVAILABLE_DATE, ADD_UNAVAILABLE_DATES, RESIZE_UNAVAILABLE_RANGE, UPDATE_OPERATION_TASK_COLOR, UPDATE_WORKFLOW_TASK_COLOR, UPDATE_WORKER_DEFINITION, UPDATE_WORKER_DESC_FIELD`

**Local-only** (UI state, per-user by nature — selection, filters, dialogs, which tab you're viewing, your own constraint-check run, your own save/dirty bookkeeping): everything else, e.g. `SWITCH_VIEW, SELECT_ASSIGNMENT, SELECT_UNAVAILABLE, TOGGLE_DEVICE, SET_*_FILTER, CLEAR_ALL_WORKER_FILTERS, OPEN_*/CLOSE_* dialogs, SET_ERROR, SAVE_PATHS, MARK_SAVED, SET_VIOLATIONS, SET_CONSTRAINT_CHECKING, SET_BACKEND_VIOLATIONS, TOGGLE_FLIGHT_STINTS, SELECT_ASSIGNMENT_AND_SCROLL, CLEAR_SCROLL_TO_ASSIGNMENT`.

**`LOAD_FILES` while in a session**: blocked in the UI ("Open File" disabled while joined) — loading a different file mid-session would silently replace shared data out from under other participants. Leaving the session re-enables it.

**Undo/Redo — the one real wrinkle:** `UNDO`/`REDO` are per-client snapshot-stack operations today (`state.undoStack`/`redoStack`, see `reducer.ts:89-109`), not really "actions" in the network sense — a late joiner's stack only contains what happened since *they* joined, so broadcasting the bare `UNDO` token would make different clients undo different things. Fix: the wrapper reads the resulting schedule right after applying `UNDO`/`REDO` locally, and forwards it as a `SET_SCHEDULE` sync action with that snapshot — so every participant converges on the same content, while each person keeps their own private undo history.

**Applying a remote `action` event:** call the *raw* `dispatch` (the one from `useReducer`, not the wrapped one) directly, so it never re-enters the "forward to server" path and can't echo back.

## 5. Join flow (both paths, one mechanism)

- **Link, opened in a plain browser tab** (works today for view-only, extended to edit): `http://<lan-ip>:5173/?session=<id>&role=edit|view`. `App.tsx`'s existing `isViewMode()` check becomes a `getSessionParams()` check; when present, it shows a small "enter your name" prompt, then renders the normal `GanttPage` (not a stripped-down page) with the session already joining in the background. `role=view` keeps today's `pointer-events:none` read-only rendering; `role=edit` is the fully interactive page.
- **Pasted into the app itself**: a "Join Session" dialog (replaces `ShareViewDialog`'s old read-only-only framing) with a text field — paste the full link or just the session id, pick a name, pick Edit/View, and the app parses and joins internally without navigating the Electron window's URL.
- **Starting a session**: the existing `ShareViewButton` becomes "Start Session" — calls `createSession` with current data, shows the link (both the browser-link form and the raw session id for in-app pasting), and switches this client into `role=edit` on its own session.
- **Presence**: a small bar (styled like today's `ViewPage` status strip) listing connected display names + role, shown in both `GanttPage` and browser joiners.

State additions to `AppState` (replacing the old `isSharingLiveView/liveViewShareLink/isShareViewDialogOpen/viewConnectionStatus/currentView-only SET_VIEW_STATE`): `session: { id, role, connectionStatus, participants } | null`, `isSessionDialogOpen`.

## 6. Process lifecycle — host leaving doesn't end the session

Per your decision: no outsourced/cloud server for v1 — it keeps running on whichever machine started it, self-hosted.

Change in `electron/main.cts`:
- Spawn the embedded server `detached: true` and `unref()` it.
- While a collab session this app started is active (or generally, once the server has ever been spawned), `window-all-closed` does **not** call `app.quit()` — instead the window hides and a tray icon appears ("GanttChartEditor — collaboration server running"). The host can reopen the window from the tray, or choose "Quit" from the tray menu to actually stop the server (equivalent to today's full quit).
- Plain window close (✕) while **no** session is active keeps today's behavior exactly (quits immediately) — this change only changes behavior when collaboration is actually in use.

This means: closing the editor window, or quitting the app outright, no longer kills the session as long as the host's PC stays on. It still cannot survive the host's machine sleeping/shutting down/losing network — that limitation is accepted for v1, matching the "fastest, localhost-first" path. If real usage later needs the session to survive a full machine shutdown, the fix is to point every client at a separately-run instance of the same server (`node dist/index.js` on a machine left on, or eventually a small cloud host) — no client-side protocol change needed, since the server was already designed to be a standalone connection target.

## 7. Explicitly out of scope for v1

- No persistence beyond process memory — a killed server process (via tray "Quit," or the host's PC going down) loses anything no participant already `Save`d to their own local file.
- No login/accounts — a typed display name only, for presence/attribution.
- No field-level locking or CRDT — last-applied-wins on true simultaneous same-field edits.
- No action-log compaction/checkpointing — fine for a single work session's worth of edits in memory; can be added later if sessions run for days.

## 8. Testing

- Single-PC: start a session in one window, join as edit and as view in two more browser tabs on `localhost`, confirm edits from any edit-role tab appear live in the others, confirm view-role tab stays read-only.
- Undo/redo: perform edits from two tabs, undo from one, confirm the other tab's content updates to match without touching the other tab's own undo stack.
- Host-leaves: start a session, join from a second tab, close the host's app window entirely (not just navigate away) — confirm the second tab keeps working and can still send/receive actions, confirming the tray/detach fix.
- Late joiner: after several edits from two participants, open a third tab and join — confirm it lands on the current state (baseline + replayed actions), not the original baseline.
