# GanttChartEditor Live Collab — Round 2 UX Feedback Design

**Companion to:** `GanttChartEditor_LiveCollabEdit_Design20260826.md` (original design), `GanttChartEditor_LiveCollabEdit_Plan20260826.md` (original 12-task plan), `GanttChartEditor_LiveCollabEdit_TestReport20260827.md` (test report), `GanttChartEditor_LiveCollabEdit_ReliabilityDesign20260828.md` (action-reliability design, separate in-flight work).

**Source:** six pieces of feedback from real usage, addressed together since they're all refinements to the same collab session UX, not independent subsystems.

---

## 1. Session name

**Data model:**
- Server: `SessionBaseline`/`CollabSession` (`server/src/collab/sessionStore.ts`) gains a `name: string`, set once at `createSession(name, baseline)` and never changed after.
- `POST /api/collab/sessions` (`server/src/routes/collab.ts`) accepts and requires `name` in the request body alongside the existing `schedule`/`envConfig`/`currentView`; rejects with 400 if missing/blank, same style as its existing validation.
- `sync-init`'s payload (`server/src/collab/collabSocket.ts`) includes the session's `name` so every joiner learns it, not just the creator.
- Client `SessionState` (`src/types/appState.ts`) gains `name: string`.

**UI:** `StartOrJoinPanel`'s "start" tab (`SessionDialog.tsx`) gets a required text field, "セッション名" (Session name), above or alongside the existing display-name field. The **Start** button is disabled until both the display name and the session name are non-blank (required, per the earlier decision — no auto-generated fallback).

## 2. Participant list on click

`MenuBar.tsx`'s presence indicator (currently a plain `<span>{UI.sessionParticipantsLabel(...)}</span>`) becomes a clickable trigger for a small dropdown, styled consistently with the existing File/Edit/etc. dropdowns (same absolute-positioned box pattern already in this file). Content: one row per `state.session.participants` entry, showing `name` and a localized role label — 編集者 (editor) for `role: 'edit'`, 閲覧者 (viewer) for `role: 'view'`. No new data — `participants` already carries this; this is a display-only addition. Toggled by click (matching every other menu in this bar); dismissed the same way the existing menus already dismiss (the bar's existing outside-click handler).

## 3. Join-dialog clarity

**New lightweight lookup endpoint:** `GET /api/collab/sessions/:id/name` (or equivalent — exact route decided at plan time) returning `{ ok: true, name }` or `{ ok: false }` for an unknown/expired session id. Deliberately minimal — name only, not the full baseline — so it's cheap to call speculatively as someone types/pastes.

**UI:** in `StartOrJoinPanel`'s "join" tab, once the link/ID field has a value that resolves to a real session (debounced lookup on change, or checked on blur — exact trigger decided at plan time), show a confirmation line above the display-name field: `「{name}」というセッションに参加します` (You are about to join the session "{name}"). If the lookup fails (bad id, session gone), show the existing-style error inline instead of a fabricated name. This directly addresses the reported confusion — the link/ID field's purpose becomes self-evident once it resolves to a named session, and the display-name field keeps its current, already-clear placeholder.

## 4. セッション情報 submenu

`MenuBar.tsx`'s 共同編集 menu, when `state.session` exists, currently shows one item (セッションを終了). It becomes two:
- **セッション情報** — dispatches `OPEN_SESSION_DIALOG` (same action already used to open the dialog pre-session), which now shows `ActiveSessionPanel` since `state.session` is truthy — this reuses the exact panel already built, just makes it reachable again after the initial start/join moment (today there is no way back into it once the dialog is first closed).
- **セッションを終了** — unchanged, `leaveCollabSession()`.

`ActiveSessionPanel` (`SessionDialog.tsx`) gains:
- The session name displayed at the top (from `state.session.name`, §1).
- The edit-link row (`fetchCollabLink(session.id, 'edit')`) now conditional on `state.session.role === 'edit'` — a viewer never sees it. Documented explicitly as a UX courtesy, not an access boundary: the underlying join mechanism has no authentication, so anyone holding the bare session id (or the view link, which contains it) could still choose `role=edit` themselves when joining. Hiding the edit link from viewers just avoids handing them something that implies they should use it.

## 5. Viewer parity — unify the editor and viewer shells

**This replaces the current dual-page architecture.** Today, `App.tsx` renders `<ViewPage />` for `role: 'view'` and `<EditorShell />` (via `AppContent`) for everything else — two separately-maintained UIs. `ViewPage.tsx` is deleted; both roles render through `EditorShell` (renamed `AppContent`'s role branch away — `AppContent` becomes unconditionally `<EditorShell />`, since `App.tsx`'s `SessionJoinGate` no longer needs a role branch either).

**The read-only boundary moves from a page-level `pointer-events: none` wrapper to per-interaction gating**, using one derived flag — `isReadOnly = state.session?.role === 'view'` — read via `useAppContext()` in each place that needs it, matching the existing pattern already used for session-gating (e.g. `Toolbar.tsx`'s `has`/`state.session` checks, `useKeyboardShortcuts.ts`'s `state.session` guard).

| Interaction | Viewer (role: view) |
|---|---|
| Scroll/pan the Gantt body | **Enabled** — it was only ever blocked by `ViewPage`'s blanket wrapper; the body itself is plain `overflow: auto`, nothing else to change |
| View toggle (worker/device), both filter panels | **Enabled** — already per-user local state, never synced; this was already true for editors, now the surface exists for viewers too |
| 制約チェック (constraint check) | **Enabled** — runs and shows results locally, mutates nothing shared |
| 出入国バー (flight stints) toggle | **Enabled** — local display toggle |
| Bar move/resize drag-start, unavailable-date drag | **Disabled** — the `onMouseDown` handlers in `WorkerTimelineGrid.tsx` (and the equivalent in the device view) become no-ops when `isReadOnly`; scoped to these specific handlers, not a page-wide block, since scrolling/filtering must keep working |
| Delete key (delete selected assignment) | **Disabled** |
| Undo / Redo | **Disabled** — a viewer has no local edit history of their own to step through |
| + バー配置, + 新規製番追加 | **Disabled** (button-level, same visual pattern as the existing session-gating on these buttons) |
| 計画管理ツールへ送信 | **Disabled** |
| Ctrl+S / Ctrl+O | **Disabled** — already effectively true for any joiner (no local file path exists to save to), now made consistent for the read-only case too |
| 共同編集 menu (セッション情報 / セッションを終了) | **Enabled** — a viewer is a participant like any other; §4 applies to them equally |

Exact prop-vs-context-read mechanics for threading `isReadOnly` into `WorkerTimelineGrid`/`DeviceViewGantt` (whichever already has `useAppContext()` access vs. needs a prop passed down) are a plan-time detail, not a design-level one — the design requirement is simply that these specific handlers, and no others, become inert.

## 6. Tab pre-selection

`OPEN_SESSION_DIALOG` gains a payload: `'start' | 'join'`. New `AppState` field `sessionDialogTab: 'start' | 'join'` (default `'start'`), set by the reducer on open. `MenuBar.tsx`'s two menu items dispatch `{ type: 'OPEN_SESSION_DIALOG', payload: 'start' }` and `{ type: 'OPEN_SESSION_DIALOG', payload: 'join' }` respectively. `StartOrJoinPanel` initializes its local tab state from `state.sessionDialogTab` instead of the current hardcoded `'start'`.

---

## Testing

- **Server (Vitest):** session-name required on create (400 without it); name round-trips through `sync-init`; the new name-lookup endpoint returns the right name for a real session and `ok:false` for an unknown one.
- **Client (Jest):** reducer cases for the new `sessionDialogTab` field and `SessionState.name`; `ActiveSessionPanel` hides the edit link for a view-role session and shows it for edit-role; the participant dropdown renders each participant's name and localized role.
- **Manual/Cypress:** since §5 is the one architectural change, a live check that a view-role join can scroll, toggle filters/flight-stints/constraint-check, and that drag-starting a bar, Undo/Redo, and the edit-only buttons are all inert — extending the existing Cypress suite alongside the reliability spec from the companion design doc, exact spec file(s) decided at plan time.

## Explicitly out of scope for this pass

- Session names are not unique or validated for collisions — two sessions can share a name; the id is still the real identifier, the name is purely a display label.
- No real access control is added anywhere in this pass (per §4's note) — the "no login" constraint from the original design is unchanged.
- No change to the sync protocol, conflict model, or the in-flight action-reliability work (companion doc) — this pass is UI/UX only, on top of the existing sync mechanism.
