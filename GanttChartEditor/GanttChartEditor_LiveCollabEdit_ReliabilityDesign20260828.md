# GanttChartEditor Live Collab — Action Reliability Design

**Companion to:** `GanttChartEditor_LiveCollabEdit_Design20260826.md` (original design), `GanttChartEditor_LiveCollabEdit_Plan20260826.md` (original 12-task plan), `GanttChartEditor_LiveCollabEdit_TestReport20260827.md` (test report).

**Problem this closes:** today, sending an edit to the collab server is pure fire-and-forget — `collabService.ts`'s `sendCollabAction()` is `socket?.emit('action', {type, payload})` with no acknowledgment. There is currently no way for a client to know whether a given edit was actually received and logged by the server, versus silently lost (socket not connected, connection dropped mid-send, or the server rejecting it for some reason). The local optimistic apply always happens regardless, so a lost edit looks identical to a successful one on the sender's own screen.

---

## 1. Server — acknowledge every action

`server/src/collab/collabSocket.ts`'s `action` handler currently has no response. Add a Socket.IO acknowledgment callback (the 3rd argument Socket.IO passes when the emitter provides one) and reply on every attempt:

- **Success:** `{ ok: true, seq }` — the action was appended to the session's log at that sequence number, matching what `appendAction` already returns.
- **Failure**, with a reason: the sender isn't in a validly-joined edit role for that session (not joined at all, or `role !== 'edit'`), or the session no longer exists (already idle-swept, or never existed). `{ ok: false, error: '<reason>' }`.

This is purely additive — the existing broadcast to other participants (`socket.to(sessionId).emit('action', ...)`) is unchanged; only the sender now gets a direct reply in addition to that broadcast.

## 2. Client — gate dispatch on connection, then wait for that ack

In `src/context/AppContext.tsx`'s wrapped `dispatch`, for a syncable action while the session is edit-role:

1. **Fast pre-check, no network round-trip needed:** if `state.session.connectionStatus !== 'connected'`, reject immediately — dispatch `SET_ERROR` with a clear message, and do **not** call `rawDispatch`. The edit never applies locally.
2. **Otherwise**, send the action to the server with an ack callback and a timeout (~5s — on a LAN, a healthy round-trip is milliseconds, so this only ever actually elapses when something is genuinely wrong). The edit is **not applied yet** — local state is unchanged while waiting.
3. **Ack arrives with `ok: true`** within the timeout → apply it now via `rawDispatch`, exactly as before this change.
4. **Ack arrives with `ok: false`, or the timeout elapses with no response** → do not apply it; dispatch `SET_ERROR` with a message reflecting which case it was.

No automatic retry or queueing (per the earlier decision) — a blocked edit is simply not applied; the person sees why, and decides whether to redo it once reconnected.

**Undo/redo:** these already get converted into a forwarded `SET_SCHEDULE` action inside the same wrapper (see the original design's §4) — they flow through this exact same gate with no special-casing needed. If the resulting state can't be confirmed placed on the server, the local undo/redo doesn't visibly happen either.

**Why this doesn't reintroduce drag lag:** confirmed by reading `WorkerTimelineGrid.tsx` — dragging a bar only updates a purely local, unsynced `dragPreview` React state on every `mousemove`; the actual `UPDATE_ASSIGNMENT` dispatch fires exactly once, on `mouseup` (drop). So this gate only adds a brief pause at the moment of release, not during the drag itself.

**Scope:** solo mode (no session) and view-role participants are completely unaffected — this gate only ever activates on the same `SYNCABLE_ACTION_TYPES` set, in the same place, that already only applies to an active edit-role session.

## 3. What the user sees

Reuses the existing `SET_ERROR` → `ErrorDialog` pattern already established for the mid-session file-load block (see the reliability report's bug #4/finding table), for consistency and to avoid introducing a second error-surfacing mechanism. Two distinct new message keys in `src/config/uiText.ts`:

- Not connected (pre-check failed): "接続が確認できないため、この操作は行えません。" (Can't perform this action — connection isn't confirmed.)
- Server didn't confirm it (ack failed or timed out): "サーバーがこの操作を記録できませんでした。もう一度お試しください。" (The server couldn't record this action — please try again.)

## 4. Testing

**Server (Vitest, `server/src/collab/collabSocket.test.ts`):** new cases — ack succeeds for a valid edit-role action; ack fails with a reason for a view-role sender; ack fails for a socket that never joined; ack fails for a session that no longer exists.

**Client (Jest, `src/__tests__/context/AppContext.test.tsx`):** dispatch waits for the ack before calling the underlying apply; rejects instantly (no server round-trip) when `connectionStatus !== 'connected'`; times out and shows the error when no ack ever arrives.

**Cypress — real, watchable browser proof, per your request:** a new spec, `cypress/e2e/05_collab_reliability.cy.ts`, following the existing suite's numbering and style (`01_empty_state.cy.ts` through `04_dialogs.cy.ts`). Two scenarios:
- **Connection-error path:** `cy.intercept()` the Socket.IO connection to simulate it dropping mid-session, then attempt an edit and assert it visibly does not apply and the "not connected" message appears.
- **Server-rejected path:** use `cy.request()` to call the collab API directly and end the session out from under the client (or otherwise make the server stop recognizing it), then attempt an edit and assert the "server couldn't record this" message appears instead, and the edit again does not apply.

A true two-real-browser-tabs scenario is out of scope for Cypress here (Cypress runs one browser context per spec) — the existing server-side Vitest integration tests already cover genuine two-client sync with real sockets; Cypress's job is proving what the *user* sees in the one client it drives, which is exactly what these reliability paths are about.

## 5. Cypress setup reference (for running these and the existing specs)

Cypress is already a project devDependency (`cypress: ^13.17.0`) with a working config and 4 existing specs — nothing new to install for the tool itself, just documenting how it's actually run since this feature adds to it:

- **Config:** `cypress.config.cjs` — `baseUrl: http://localhost:5173`, specs matched from `cypress/e2e/**/*.cy.ts`, screenshots on failure saved to `cypress/screenshots/` (video recording is off).
- **It does not start the dev server itself** — `baseUrl` only tells Cypress where to point the browser; nothing in the config spawns `npm run dev`. The dev server(s) must already be running before invoking Cypress:
  - Existing specs (`01`–`04`) only touch the client, so `npm run dev` (Vite alone, port 5173) is enough.
  - The new collab reliability spec also needs the collab API/socket server, so use `npm run dev:all` instead (starts both Vite on 5173 and the collab server on 3010 together, via `concurrently` — see `package.json`'s existing scripts).
- **Run scripts** (already defined in `package.json`, unchanged by this feature):
  - `npm run test:cypress` → `cypress open` — interactive runner, opens a real browser window, good for writing/debugging a spec and watching it click through live.
  - `npm run test:cypress:run` → `cypress run` — headless, prints pass/fail to the terminal, what you'd use in CI or a quick full-suite check.
- **First-time note:** Cypress downloads a separate browser binary via a postinstall step the first time its npm package is installed. If that download was ever blocked (e.g. by a proxy/firewall) and only the JS package landed, `cypress open`/`run` will fail immediately with a clear "binary not found" message — the fix is `npx cypress install` to fetch it explicitly. Not expected to be an issue here since Cypress is already committed as a working devDependency with existing passing specs, but worth knowing if a fresh clone ever hits it.
- **Practical sequence to run everything, including the new spec, once implemented:**
  1. Terminal 1: `npm run dev:all` (from the project root) — leave running.
  2. Terminal 2: `npm run test:cypress` (interactive, to watch it) or `npm run test:cypress:run` (headless, for a quick pass/fail).

## 6. Explicitly out of scope for this pass

- No automatic retry/queueing of a blocked action (per the earlier decision) — purely flag-and-stop.
- No change to the underlying sync protocol's ordering guarantees or conflict model (still last-applied-wins, per the original design) — this only adds confirmation of *delivery*, not new conflict resolution.
- No true multi-browser-tab Cypress orchestration — covered by existing Vitest server integration tests instead, as explained in §4.
